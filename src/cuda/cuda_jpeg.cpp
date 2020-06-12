#include <nvjpeg.h>

#include "saga.h"
#include "tensor.h"
#include "context.h"

#include "cuda_common.h"
#include "cuda_tensor.h"
#include "cuda_kernels.h"


namespace saga {


//------------------------------------------------------------------------
struct CudaJpeg : public CudaOperation {

  const std::shared_ptr<CudaContext> ctx_;
  const std::shared_ptr<CudaTensor> y_;
  const std::shared_ptr<Tensor> x_;
  const std::unique_ptr<nvjpegImage_t[]> output_images_;

  nvjpegHandle_t handle_;
  nvjpegJpegState_t jpeg_handle_;


  CudaJpeg(CudaProgram &p, const Node &n)
    : ctx_(p.ctx_)
    , y_(p.lower_tensor_batch(n.outputs_.get("y")))
    , x_(n.inputs_.get("x"))
    , output_images_(std::make_unique<nvjpegImage_t[]>(y_->dims_[0]))
  {
    const size_t batch_size = x_->dims_[0];

    nvjpegCreateSimple(&handle_);
    nvjpegJpegStateCreate(handle_, &jpeg_handle_);

    nvjpegDecodeBatchedInitialize(handle_, jpeg_handle_,
                                  batch_size, 8, NVJPEG_OUTPUT_RGB);

    const int max_rank = 8;
    int dimsA[max_rank];
    int stridesA[max_rank];
    int rank;
    cudnnDataType_t data_type;

    chkCUDNN(cudnnGetTensorNdDescriptor(y_->desc_, max_rank, &data_type,
                                        &rank, dimsA, stridesA));

    uint8_t *ymem = (uint8_t *)y_->deviceMem();

    for(size_t n = 0; n < batch_size; n++) {
      for(size_t c = 0; c < 3; c++) {
        output_images_[n].channel[c] = ymem + n * stridesA[0] + c * stridesA[1];
        output_images_[n].pitch[c] = stridesA[2];
      }
    }
  }

  ~CudaJpeg()
  {
    nvjpegJpegStateDestroy(jpeg_handle_);
    nvjpegDestroy(handle_);
  }

  void print() const {
    printf("JPEG decoder\n");
  }

  void exec(CudaProgram &p) {

    const int batch_size = x_->dims_[0];

    const uint8_t *data[batch_size];
    size_t lengths[batch_size];

    auto ta = x_->access();

    for(int n = 0; n < batch_size; n++) {
      const uint8_t *jpeg = (const uint8_t *)ta->getAddr(Dims{n});
      size_t len = *(uint32_t *)jpeg;
      data[n] = jpeg + sizeof(uint32_t);
      lengths[n] = len;
    }

    nvjpegDecodeBatched(handle_, jpeg_handle_,
                        data, lengths, &output_images_[0], ctx_->stream_);
  }

};


static void
jpegdecoder_infer(CudaProgram &p, const Node &n)
{
  p.infer(std::make_shared<CudaJpeg>(p, n));
}


static void
jpegdecoder_train(CudaProgram &p, const Node &n)
{
  p.train(std::make_shared<CudaJpeg>(p, n));
}


REGISTER_CUDA_OP("jpegdecoder", jpegdecoder_infer, jpegdecoder_train);


};


