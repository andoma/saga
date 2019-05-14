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

Tensor::Tensor(cudnnDataType_t data_type, const Size &s)
  : data_type_(data_type)
  , size_(s)
  , device_mem_(NULL)
{
  chkCUDNN(cudnnCreateTensorDescriptor(&desc_));
  chkCUDNN(cudnnSetTensor4dDescriptor(desc_, CUDNN_TENSOR_NCHW,
                                      data_type, s.n, s.c, s.h, s.w));

  chkCUDNN(cudnnGetTensorSizeInBytes(desc_, &bytes_));
  chkCuda(cudaMalloc(&device_mem_, bytes_));
  chkCuda(cudaMemset(device_mem_, 0, bytes_));
};

Tensor::~Tensor()
{
  chkCUDNN(cudnnDestroyTensorDescriptor(desc_));
}


void Tensor::save(float *data)
{
  const size_t bytes = size().elements() * sizeof(float);
  cudaMemcpy((void *)data, device_mem_,
             bytes, cudaMemcpyDeviceToHost);

  for(size_t i = 0; i < size().elements(); i++)
    assert(!isnan(data[i]));
}


void Tensor::load(const float *data)
{
  const size_t bytes = size().elements() * sizeof(float);

  for(size_t i = 0; i < size().elements(); i++)
    assert(!isnan(data[i]));

  cudaMemcpy(device_mem_, (const void *)data,
             bytes, cudaMemcpyHostToDevice);
}


void Tensor::load(const std::vector<float> &data)
{
  load(&data[0]);
}

void Tensor::load(__restrict__ const uint8_t *data)
{
  const size_t elements = size().elements();
  float floats[elements];
  for(size_t i = 0; i < elements; i++)
    floats[i] = data[i] / 255.0f;
  load(floats);
}

void Tensor::load(__restrict__ const uint8_t **data)
{
  const size_t elements = size().elements();
  float floats[elements];
  const size_t chw = size().c * size().h * size().w;
  float *dst = &floats[0];
  for(size_t i = 0; i < size().n; i++) {
    const uint8_t *src = data[i];
    for(size_t j = 0; j < chw; j++)
      *dst++ = src[j] / 255.0f;
  }
  load(floats);
}



void Tensor::load(const TensorValues &v)
{
  assert(v.size() == size());

  const size_t bytes = v.size().elements() * sizeof(float);
  cudaMemcpy(device_mem_, v.data(), bytes, cudaMemcpyHostToDevice);
}


void Tensor::fill(float value)
{
  assert(data_type_ == CUDNN_DATA_FLOAT);

  const size_t bytes = size_.elements() * sizeof(float);

  if(value == 0) {
    cudaMemset(device_mem_, 0, bytes);
    return;
  }

  load(std::vector<float>(size().elements(), value));
}


void Tensor::randomize(float sigma)
{
  assert(data_type_ == CUDNN_DATA_FLOAT);

  if(sigma == 0) {
    fill(0);
    return;
  }

  std::vector<float> values(size().elements());

  for(size_t i = 0; i < values.size(); i++) {
    values[i] = generateGaussianNoise() * sigma;
  }
  load(values);
}


void Tensor::loadOrRandomize(InitData id, const std::string &name, float sigma)
{
  auto it = id.find(name);
  if(it != id.end()) {
    load(it->second);
  } else {
    randomize(sigma);
  }
}


std::string Size::name() const {
  std::stringstream ss;
  ss << "[" << n << ", " << c << ", " << w << ", " << h << "]";
  return ss.str();

}


std::string Tensor::name() const {
  return size().name();
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
            printf("%s%2.3f ", v < 0 ? "" : " ", v);
          }
        }
        printf("\n");
      }
    }
  }
}


void Tensor::dump(const char *prefix, bool intensity) const {
  const size_t bytes = size().elements() * sizeof(float);

  float *hostmem = (float *)malloc(bytes);

  cudaMemcpy((void *)hostmem, device_mem_,
             bytes, cudaMemcpyDeviceToHost);

  for(size_t i = 0; i < size().elements(); i++)
    assert(!isnan(hostmem[i]));

  tensor_print_raw(prefix, hostmem,
                   size().n,
                   size().c,
                   size().h,
                   size().w, intensity);
  free(hostmem);

}

void Tensor::check() const {

  const size_t bytes = size().elements() * sizeof(float);

  float *hostmem = (float *)malloc(bytes);

  cudaMemcpy((void *)hostmem, device_mem_,
             bytes, cudaMemcpyDeviceToHost);

  for(size_t i = 0; i < size().elements(); i++) {
    if(isnan(hostmem[i])) {
      fprintf(stderr, "Tensor %s has NAN at %zd\n",
              name().c_str(), i);
      abort();
    }
  }
  free(hostmem);
}



float Tensor::peak() const {

  const size_t bytes = size().elements() * sizeof(float);

  float *hostmem = (float *)malloc(bytes);

  cudaMemcpy((void *)hostmem, device_mem_,
             bytes, cudaMemcpyDeviceToHost);

  float r = 0;
  for(size_t i = 0; i < size().elements(); i++) {
    r = std::max(r, fabs(hostmem[i]));
  }
  free(hostmem);
  return r;
}



}
