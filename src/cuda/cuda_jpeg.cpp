#include <thread>
#include <mutex>
#include <condition_variable>

#include <nvjpeg.h>

#include "saga.h"
#include "tensor.h"
#include "context.h"

#include "cuda_common.h"
#include "cuda_tensor.h"
#include "cuda_kernels.h"

#define JPEG_THREADS 8

namespace saga {

struct CudaJpeg : public CudaOperation,
                  public std::enable_shared_from_this<CudaJpeg> {
    const std::shared_ptr<CudaTensor> y_;
    const std::shared_ptr<CudaTensorStorageDoubleBuffered> s_;
    const Loader loader_;
    const int batch_size_;

    std::unique_ptr<nvjpegImage_t[]> output_images_[2];

    cudaStream_t stream_;
    cudaEvent_t event_;

    nvjpegHandle_t handle_;
    nvjpegJpegState_t jpeg_handle_;

    std::thread threads_[JPEG_THREADS];
    std::mutex work_mutex_;
    std::condition_variable_any work_cond_;
    std::condition_variable_any complete_cond_;

    int decode_;
    int completed_;
    long current_batch_;

    CudaJpeg(CudaProgram &p, std::shared_ptr<CudaTensorStorageDoubleBuffered> s,
             std::shared_ptr<CudaTensor> y, Loader loader, int batch_size)
      : CudaOperation("jpegdec")
      , y_(y)
      , s_(s)
      , loader_(loader)
      , batch_size_(batch_size)
    {
        output_images_[0] = std::make_unique<nvjpegImage_t[]>(batch_size_);
        output_images_[1] = std::make_unique<nvjpegImage_t[]>(batch_size_);

        decode_ = batch_size_;
        completed_ = batch_size_;

        chkCuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
        chkCuda(cudaEventCreate(&event_));

        nvjpegCreateSimple(&handle_);
        nvjpegJpegStateCreate(handle_, &jpeg_handle_);

        nvjpegDecodeBatchedInitialize(handle_, jpeg_handle_, batch_size_,
                                      JPEG_THREADS, NVJPEG_OUTPUT_RGBI);

        const int max_rank = 8;
        int dimsA[max_rank];
        int stridesA[max_rank];
        int rank;
        cudnnDataType_t data_type;

        chkCUDNN(cudnnGetTensorNdDescriptor(y_->m_desc, max_rank, &data_type,
                                            &rank, dimsA, stridesA));

        for(int i = 0; i < 2; i++) {
            uint8_t *ymem = (uint8_t *)s->data(i);

            for(int n = 0; n < batch_size_; n++) {
                for(int c = 0; c < 3; c++) {
                    output_images_[i][n].channel[c] =
                        ymem + n * stridesA[0] + c * stridesA[1];
                    output_images_[i][n].pitch[c] = stridesA[2];
                }
            }
        }

        for(int i = 0; i < JPEG_THREADS; i++) {
            threads_[i] = std::thread(&CudaJpeg::worker, this, i);
        }
    }

    ~CudaJpeg()
    {
        work_mutex_.lock();
        decode_ = -1;
        work_cond_.notify_all();
        work_mutex_.unlock();

        for(int i = 0; i < JPEG_THREADS; i++) {
            threads_[i].join();
        }

        nvjpegJpegStateDestroy(jpeg_handle_);
        nvjpegDestroy(handle_);
    }

    const char *exec(CudaProgram &p, long batch) override
    {
        int buffer_wr = s_->flip();

        work_mutex_.lock();
        current_batch_ = batch;
        completed_ = 0;
        decode_ = 0;
        work_cond_.notify_all();

        while(completed_ != batch_size_) {
            complete_cond_.wait(work_mutex_);
        }

        work_mutex_.unlock();

        nvjpegDecodeBatchedPhaseTwo(handle_, jpeg_handle_, stream_);

        nvjpegDecodeBatchedPhaseThree(handle_, jpeg_handle_,
                                      &output_images_[buffer_wr][0], stream_);
        chkCuda(cudaEventRecord(event_, stream_));
        return NULL;
    }

    void worker(int id)
    {
        uint8_t jpeg_buffer[65536];

        work_mutex_.lock();
        while(1) {
            if(decode_ == batch_size_) {
                work_cond_.wait(work_mutex_);
                continue;
            }

            if(decode_ == -1)
                break;

            const int n = decode_++;

            work_mutex_.unlock();

            size_t l =
                loader_(current_batch_, n, jpeg_buffer, sizeof(jpeg_buffer));

            if(l > 0) {
                nvjpegDecodeBatchedPhaseOne(handle_, jpeg_handle_, jpeg_buffer,
                                            l, n, id, stream_);
            }

            work_mutex_.lock();
            completed_++;
            complete_cond_.notify_one();
        }
        work_mutex_.unlock();
    }

    std::vector<std::shared_ptr<CudaTensor>> getInputs() const override
    {
        return {};
    }

    std::vector<std::shared_ptr<CudaTensor>> getOutputs() const override
    {
        return {y_};
    }

    std::shared_ptr<CudaOperation> getSyncOp() override;
};

struct CudaJpegSync : public CudaOperation {
    const std::shared_ptr<CudaJpeg> j_;

    CudaJpegSync(std::shared_ptr<CudaJpeg> j)
      : CudaOperation("jpegdecoder.wait"), j_(j)
    {
    }

    const char *exec(CudaProgram &p, long batch)
    {
        chkCuda(cudaStreamWaitEvent(p.m_ctx->m_stream, j_->event_, 0));
        return NULL;
    }

    std::vector<std::shared_ptr<CudaTensor>> getInputs() const override
    {
        return {};
    }

    std::vector<std::shared_ptr<CudaTensor>> getOutputs() const override
    {
        return {j_->y_};
    }
};

std::shared_ptr<CudaOperation>
CudaJpeg::getSyncOp()
{
    return std::make_shared<CudaJpegSync>(shared_from_this());
}

static const char *
jpegdecoder_setup(CudaProgram &p, const Node &n, bool training)
{
    auto yh = n.outputs_.get("y");
    auto j = p.m_load_map[yh];

    if(!j) {
        // Lower into a double buffered tensor
        auto dims = yh->dims_.n(p.m_batch_size);

        auto fmt = CUDNN_TENSOR_NHWC;
        auto s = std::make_shared<CudaTensorStorageDoubleBuffered>(
            yh->data_type_, dims, fmt, p.m_ctx);
        auto y = std::make_shared<CudaTensor>(s, dims, fmt);

        p.m_tensors[yh] = y;
        j = std::make_shared<CudaJpeg>(p, s, y, n.loader_, p.m_batch_size);
        p.m_load_map[yh] = j;
    }

    if(training)
        p.fwd(j->getSyncOp());
    else
        p.infer(j->getSyncOp());
    return NULL;
}

REGISTER_CUDA_OP("jpegdecoder", jpegdecoder_setup);

};  // namespace saga
