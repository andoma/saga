#include <math.h>
#include <assert.h>

#include <limits>
#include <sstream>
#include <algorithm>

#include "common.h"

namespace saga {



// https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
static double
generateGaussianNoise(void)
{
  static __thread double z1;
  static __thread bool generate;
  static const double epsilon = std::numeric_limits<double>::min();

  generate = !generate;

  if (!generate)
    return z1;

  double u1, u2;
  do {
    u1 = drand48();
    u2 = drand48();
  } while(u1 <= epsilon);

  double z0;
  z0 = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
  z1 = sqrt(-2.0 * log(u1)) * sin(2.0 * M_PI * u2);
  return z0;
}



TensorDescriptor::TensorDescriptor(cudnnDataType_t data_type,
                                   cudnnTensorFormat_t format,
                                   unsigned int n,
                                   unsigned int c,
                                   unsigned int h,
                                   unsigned int w)
  : Size(n, c, h, w)
  , data_type_(data_type)
  , format_(format)
{
  chkCUDNN(cudnnCreateTensorDescriptor(&desc_));
  chkCUDNN(cudnnSetTensor4dDescriptor(desc_, format, data_type, n, c, h, w));
}


TensorDescriptor::~TensorDescriptor()
{
  chkCUDNN(cudnnDestroyTensorDescriptor(desc_));
}



Tensor::Tensor(const TensorDescriptor &td)
  : TensorDescriptor(td)
  , device_mem_(NULL)
{
  chkCUDNN(cudnnGetTensorSizeInBytes(desc(), &bytes_));
  chkCuda(cudaMalloc(&device_mem_, bytes_));
  chkCuda(cudaMemset(device_mem_, 0, bytes_));
};

Tensor::~Tensor()
{
  chkCuda(cudaFree(device_mem_));
}


void Tensor::save(float *data) const
{
  cudaMemcpy((void *)data, device_mem_,
             bytes_, cudaMemcpyDeviceToHost);
}


std::vector<unsigned int> Tensor::prediction() const
{
  float values[n * c];

  std::vector<unsigned int> r;
  r.reserve(n);

  cudaMemcpy((void *)values, device_mem_, bytes_, cudaMemcpyDeviceToHost);

  for(unsigned int i = 0; i < n; i++) {
    const float *v = values + i * c;
    unsigned int label = 0;
    for(unsigned int j = 1; j < c; j++) {
      if(v[j] > v[label]) {
        label = j;
      }
    }
    r.push_back(label);
  }
  return r;
}


float Tensor::loss(const unsigned int *labels) const
{
  float values[n * c];
  cudaMemcpy((void *)values, device_mem_, bytes_, cudaMemcpyDeviceToHost);

  float loss_sum = 0;
  for(unsigned int i = 0; i < n; i++) {
    const float *v = values + i * c;
    loss_sum += -log(v[labels[i]]);
  }
  return loss_sum / n;
}


void Tensor::load(const float *data)
{
  cudaMemcpy(device_mem_, (const void *)data,
             bytes_, cudaMemcpyHostToDevice);
}


void Tensor::load(const std::vector<float> &data)
{
  load(&data[0]);
}

void Tensor::load(__restrict__ const uint8_t *data)
{
  const size_t num_elements = elements();
  float floats[num_elements];
  for(size_t i = 0; i < num_elements; i++)
    floats[i] = data[i] / 255.0f;
  load(floats);
}

void Tensor::load(__restrict__ const uint8_t **data)
{
  const size_t num_elements = elements();
  float floats[num_elements];
  const size_t chw = c * h * w;
  float *dst = &floats[0];
  for(size_t i = 0; i < n; i++) {
    const uint8_t *src = data[i];
    for(size_t j = 0; j < chw; j++)
      *dst++ = src[j] / 255.0f;
  }
  load(floats);
}


void Tensor::load(const void *data, size_t size)
{
  assert(size == bytes_);
  cudaMemcpy(device_mem_, data, bytes_, cudaMemcpyHostToDevice);
}


void Tensor::fill(float value)
{
  assert(dataType() == CUDNN_DATA_FLOAT);

  if(value == 0) {
    cudaMemset(device_mem_, 0, bytes_);
    return;
  }

  load(std::vector<float>(elements(), value));
}


void Tensor::randomize(float sigma)
{
  assert(dataType() == CUDNN_DATA_FLOAT);

  if(sigma == 0) {
    fill(0);
    return;
  }

  std::vector<float> values(elements());
  chkCuda(cudaMalloc(&device_mem_, bytes_));

  for(size_t i = 0; i < values.size(); i++) {
    values[i] = generateGaussianNoise() * sigma;
  }
  load(values);
}


std::string Size::name() const {
  std::stringstream ss;
  ss << "[" << n << ", " << c << ", " << w << ", " << h << "]";
  return ss.str();
}

std::string TensorDescriptor::name() const {
  std::stringstream ss;
  ss << (format_ == CUDNN_TENSOR_NCHW ? "NCHW" :
         format_ == CUDNN_TENSOR_NHWC ? "NHWC" :
         "????") << Size::name();
  return ss.str();
}


static void
tensor_print_raw(const char *prefix, const float *p,
                 int in, int ic, int ih, int iw,
                 bool intensity)
{
  const int dim_size = 4;
  for(int n = 0; n < in; n++) {
    if(in > dim_size * 2 && n == dim_size) n = in - dim_size;

    for(int c = 0; c < ic; c++) {
      if(ic > dim_size * 2 && c == dim_size) c = ic - dim_size;

      printf("%10s: N%-2dC%-2d", prefix, n, c);

      for(int y = 0; y < ih; y++) {
        if(ih > dim_size * 2 && y == dim_size) {
          y = ih - dim_size;
          printf("%10s: ...\n", prefix);
        }

        if(y) {
          printf("%10s:        ", "");
        }

        for(int x = 0; x < iw; x++) {
          if(iw > dim_size * 2 && x == dim_size) {
            x = iw - dim_size;
            printf(" ... ");
          }

          float v = p[x + y * iw + c * ih * iw + n * ic * ih * iw];
          if(intensity) {
            v = fabs(v);
            if(v < 0.25) {
              printf(" ");
            } else if(v < 0.5) {
              //              printf("%lc", 0x2591);
              printf(".");
            } else if(v < 0.75) {
              //              printf("%lc", 0x2592);
              printf("x");
            } else if(v < 2) {
              //              printf("%lc", 0x2593);
              printf("X");
            } else {
              //              printf("%lc", 0x2588);
              printf("#");
            }

          } else {
            printf("%s%2.6f ", v < 0 ? "" : " ", v);
          }
        }
        printf("\n");
      }
    }
  }
}


void Tensor::dump(const char *prefix, bool intensity) const {
  float *hostmem = (float *)malloc(bytes_);

  cudaMemcpy((void *)hostmem, device_mem_,
             bytes_, cudaMemcpyDeviceToHost);
  tensor_print_raw(prefix, hostmem, n, c, h, w, intensity);
  free(hostmem);

}

void Tensor::check() const {
  float *hostmem = (float *)malloc(bytes_);

  cudaMemcpy((void *)hostmem, device_mem_,
             bytes_, cudaMemcpyDeviceToHost);

  for(size_t i = 0; i < elements(); i++) {
    if(isnan(hostmem[i])) {
      fprintf(stderr, "Tensor %s has NAN at %zd\n",
              name().c_str(), i);
      abort();
    }
  }
  free(hostmem);
}



float Tensor::peak() const {

  float *hostmem = (float *)malloc(bytes_);

  cudaMemcpy((void *)hostmem, device_mem_,
             bytes_, cudaMemcpyDeviceToHost);

  float r = 0;
  for(size_t i = 0; i < elements(); i++) {
    r = std::max(r, fabs(hostmem[i]));
  }
  free(hostmem);
  return r;
}

Tensor& Tensor::operator=(const Tensor& src) {
  abort();
  assert(src.bytes_ == bytes_);
#if 0
  chkCuda(cudaMemcpy(deviceMem(), src.deviceMem(),
                     bytes_, cudaMemcpyDeviceToDevice));
#endif

  return *this;

}



void Tensor::upload(void){
  fprintf(stderr, "Please implement %s\n", __FUNCTION__);
  abort();

}


}

