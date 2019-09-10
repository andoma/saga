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



Tensor::Tensor(const TensorDescriptor &td, bool host)
  : TensorDescriptor(td)
  , device_mem_(NULL)
  , host_mem_(NULL)
{
  int dump;
  cudnnDataType_t datatype;
  cudnnGetTensor4dDescriptor(desc(), &datatype,
                             &dump, &dump, &dump, &dump,
                             &ns_, &cs_, &hs_, &ws_);

  chkCUDNN(cudnnGetTensorSizeInBytes(desc(), &bytes_));
  if(host) {
    host_mem_ = malloc(bytes_);
  } else {
    chkCuda(cudaMalloc(&device_mem_, bytes_));
    chkCuda(cudaMemset(device_mem_, 0, bytes_));
  }
};

Tensor::~Tensor()
{
  if(device_mem_)
    chkCuda(cudaFree(device_mem_));
  free(host_mem_);
}


void Tensor::save(float *data) const
{
  assert(device_mem_ != NULL);
  cudaMemcpy((void *)data, device_mem_,
             bytes_, cudaMemcpyDeviceToHost);
}


std::vector<unsigned int> Tensor::prediction() const
{
  float values[n * c];

  std::vector<unsigned int> r;
  r.reserve(n);

  assert(device_mem_ != NULL);
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
  assert(device_mem_ != NULL);
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
  assert(device_mem_ != NULL);
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
  assert(device_mem_ != NULL);
  cudaMemcpy(device_mem_, data, bytes_, cudaMemcpyHostToDevice);
}


void Tensor::fill(float value)
{
  assert(dataType() == CUDNN_DATA_FLOAT);

  if(value == 0) {
    assert(device_mem_ != NULL);
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

      printf("%10s: N%-2dC%-3d", prefix, n, c);

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

  assert(device_mem_ != NULL);
  cudaMemcpy((void *)hostmem, device_mem_,
             bytes_, cudaMemcpyDeviceToHost);
  tensor_print_raw(prefix, hostmem, n, c, h, w, intensity);
  free(hostmem);

}

void Tensor::check() const {
  float *hostmem = (float *)malloc(bytes_);

  assert(device_mem_ != NULL);
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


Tensor::Stats Tensor::stats() const {

  Tensor host(*this, true);
  host = *this;

  const float *x = (const float *)host.hostMem();

  float max = -INFINITY;
  float min = INFINITY;

  double sum = 0;
  for(size_t i = 0; i < elements(); i++) {
    float v = x[i];
    max = std::max(max, v);
    min = std::min(min, v);
    sum += v;
  }

  const double mean = sum / elements();

  double sum2 = 0;
  for(size_t i = 0; i < elements(); i++) {
    double v = x[i] - mean;
    sum2 += v * v;
  }

  Stats s;
  s.min = min;
  s.max = max;
  s.mean = mean;
  s.stddev = sqrt(sum2 / elements());
  return s;
}

Tensor& Tensor::operator=(const Tensor& src) {
  assert(src.bytes_ == bytes_);

  if(src.device_mem_ != NULL && host_mem_ != NULL) {
    chkCuda(cudaMemcpy(host_mem_, src.device_mem_,
                       bytes_, cudaMemcpyDeviceToHost));
  } else {
    abort();
  }

  return *this;

}


}

