#include <stdio.h>
#include <stdint.h>
#include "cuda_kernels.hpp"

namespace saga {

//------------------------------------------------------------------------
// Category Classifier
//------------------------------------------------------------------------

template <typename T, typename L>
__global__ static void
catclassifier_fwd(int n, const T *x, L *y, unsigned int channels)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n)
        return;

    x += channels * i;

    L label = 0;
    T best = x[0];

    for(unsigned int j = 1; j < channels; j++) {
        if(x[j] > best) {
            best = x[j];
            label = j;
        }
    }
    y[i] = label;
}

template <typename T, typename L>
__global__ static void
catclassifier_bwd(int n, const T *x, T *dx, const L *y, const L *dy,
                  float *loss, unsigned int channels, float scale)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n)
        return;

    x += channels * i;
    dx += channels * i;

    // Softmax
    const double max = x[y[i]];
    double sum = 0;
    for(unsigned int j = 0; j < channels; j++) {
        sum += exp((double)x[j] - max);
    }
    const double offset = max + log(sum);

    L label = dy[i];
    for(unsigned int j = 0; j < channels; j++) {
        const double p = exp((double)x[j] - offset);

        if(label == j) {
            dx[j] = (p - 1.0f) * scale;
            loss[i] = -log(p);
        } else {
            dx[j] = p * scale;
        }
    }
}

void
catclassifier_fwd_float_i32(int n, const float *x, int32_t *y, unsigned int c,
                            cudaStream_t stream)
{
    catclassifier_fwd<<<(n + 255) / 256, 256, 0, stream>>>(n, x, y, c);
}

void
catclassifier_bwd_float_i32(int n, const float *x, float *dx, const int32_t *y,
                            const int32_t *dy, float *loss, unsigned int c,
                            float scale, cudaStream_t stream)
{
    catclassifier_bwd<<<(n + 255) / 256, 256, 0, stream>>>(n, x, dx, y, dy,
                                                           loss, c, scale);
}

void
catclassifier_fwd_half_i32(int n, const __half *x, int32_t *y, unsigned int c,
                           cudaStream_t stream)
{
    catclassifier_fwd<<<(n + 255) / 256, 256, 0, stream>>>(n, x, y, c);
}

void
catclassifier_bwd_half_i32(int n, const __half *x, __half *dx, const int32_t *y,
                           const int32_t *dy, float *loss, unsigned int c,
                           float scale, cudaStream_t stream)
{
    catclassifier_bwd<<<(n + 255) / 256, 256, 0, stream>>>(n, x, dx, y, dy,
                                                           loss, c, scale);
}

//------------------------------------------------------------------------
// MSE
//------------------------------------------------------------------------

template <typename T, typename L>
__global__ static void
mse_bwd(int n, const T *x, T *dx, const L *dy, float *loss,
        unsigned int channels, float scale)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n)
        return;

    x += channels * i;
    dy += channels * i;
    dx += channels * i;

    float sum = 0;
    for(unsigned int j = 0; j < channels; j++) {
        float yy = dy[j];
        float xx = x[j];
        float d = xx - yy;
        sum += d * d;

        if(d < -1.0f)
            d = -1.0f;
        else if(d > 1.0f)
            d = 1.0f;

        dx[j] = d * scale;
    }
    loss[i] = sum / channels;
}

void
mse_bwd_half_float(int n, const __half *x, __half *dx, const float *dy,
                   float *loss, unsigned int c, float scale,
                   cudaStream_t stream)
{
    mse_bwd<<<(n + 255) / 256, 256, 0, stream>>>(n, x, dx, dy, loss, c, scale);
}

//------------------------------------------------------------------------
// Datatype conversion
//------------------------------------------------------------------------

template <typename S, typename D>
__global__ static void
convert(int n, const S *src, D *dst, float scale)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n)
        return;

    dst[i] = (float)src[i] * scale;
}

void
convert_u8_float(const void *src, void *dst, int elements, float scale,
                 cudaStream_t stream)
{
    convert<<<(elements + 255) / 256, 256, 0, stream>>>(
        elements, (const uint8_t *)src, (float *)dst, scale);
}

void
convert_u8_half(const void *src, void *dst, int elements, float scale,
                cudaStream_t stream)
{
    convert<<<(elements + 255) / 256, 256, 0, stream>>>(
        elements, (const uint8_t *)src, (__half *)dst, scale);
}

void
convert_i16_half(const void *src, void *dst, int elements, float scale,
                 cudaStream_t stream)
{
    convert<<<(elements + 255) / 256, 256, 0, stream>>>(
        elements, (const int16_t *)src, (__half *)dst, scale);
}

void
convert_float_half(const void *src, void *dst, int elements, float scale,
                   cudaStream_t stream)
{
    convert<<<(elements + 255) / 256, 256, 0, stream>>>(
        elements, (const float *)src, (__half *)dst, scale);
}

void
convert_half_float(const void *src, void *dst, int elements, float scale,
                   cudaStream_t stream)
{
    convert<<<(elements + 255) / 256, 256, 0, stream>>>(
        elements, (const __half *)src, (float *)dst, scale);
}

//------------------------------------------------------------------------
// Adam weight update
//------------------------------------------------------------------------

#define ADAM_EPSILON 1e-8
#define ADAM_B1      0.9
#define ADAM_B2      0.999

__global__ static void
adam_kernel(int n, float *weights, const float *dweights, float *t, float b1t,
            float b2t, float lr)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n)
        return;

    t += i * 2;  // mt and vt are stored next to each other in RAM

    const float b1 = ADAM_B1;
    const float b2 = ADAM_B2;
    const float e = ADAM_EPSILON;
    const float dw = dweights[i];

    const float m = t[0] = b1 * t[0] + (1.0f - b1) * dw;
    const float v = t[1] = b2 * t[1] + (1.0f - b2) * dw * dw;
    const float m_hat = m * b1t;
    const float v_hat = v * b2t;

    weights[i] -= lr * m_hat / (sqrtf(v_hat) + e);
}

__global__ static void
adam_kernel_mp(int n, float alpha, __half *weights, const __half *dweights,
               float *t, float b1t, float b2t, float lr, int *range)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n)
        return;

    const uint16_t u16 = ((const uint16_t *)dweights)[i];
    if((u16 & 0x7800) == 0x7800) {
        atomicOr(range, 1);
        if((u16 & 0x7c00) == 0x7c00) {
            if(u16 & 0x3ff) {
                // NaN
                atomicOr(range, 2);
            } else {
                // +-Inf
            }
            return;
        }
    }

    t += i * 3;

    const float dw = (float)dweights[i] * alpha;

    const float b1 = ADAM_B1;
    const float b2 = ADAM_B2;
    const float e = ADAM_EPSILON;

    const float m = t[0] = b1 * t[0] + (1.0f - b1) * dw;
    const float v = t[1] = b2 * t[1] + (1.0f - b2) * dw * dw;
    const float m_hat = m * b1t;
    const float v_hat = v * b2t;

    const float w = t[2] - lr * m_hat / (sqrtf(v_hat) + e);
    t[2] = w;
    weights[i] = w;
}

void
adam_float(int n, float *weights, const float *dweights, float *t, float b1t,
           float b2t, float lr, cudaStream_t stream)
{
    adam_kernel<<<(n + 255) / 256, 256, 0, stream>>>(n, weights, dweights, t,
                                                     b1t, b2t, lr);
}

void
adam_mixed(int n, float alpha, __half *weights, const __half *dweights,
           float *t, float b1t, float b2t, float lr, int *range,
           cudaStream_t stream)
{
    adam_kernel_mp<<<(n + 255) / 256, 256, 0, stream>>>(
        n, alpha, weights, dweights, t, b1t, b2t, lr, range);
}

__inline__ __device__ float4
warpReduceStats(float4 val)
{
    for(int offset = warpSize / 2; offset > 0; offset /= 2) {
        const float x = __shfl_down_sync(0xffffffff, val.x, offset);
        const float y = __shfl_down_sync(0xffffffff, val.y, offset);
        const float z = __shfl_down_sync(0xffffffff, val.z, offset);
        const float w = __shfl_down_sync(0xffffffff, val.w, offset);

        val.x = min(val.x, x);
        val.y = max(val.y, y);
        val.z += z;
        val.w += w;
    }
    return val;
}

__inline__ __device__ float4
blockReduceStats(float4 val)
{
    static __shared__ float4 shared[32];
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;
    val = warpReduceStats(val);
    if(lane == 0)
        shared[wid] = val;

    __syncthreads();

    val = (threadIdx.x < blockDim.x / warpSize)
              ? shared[lane]
              : float4{0.0f, 0.0f, 0.0f, 0.0f};
    if(wid == 0)
        val = warpReduceStats(val);
    return val;
}

__device__ __forceinline__ static float
atomicMinFloat(float *addr, float value)
{
    float old;
    old = (value >= 0)
              ? __int_as_float(atomicMin((int *)addr, __float_as_int(value)))
              : __uint_as_float(
                    atomicMax((unsigned int *)addr, __float_as_uint(value)));
    return old;
}

__device__ __forceinline__ static float
atomicMaxFloat(float *addr, float value)
{
    float old;
    old = (value >= 0)
              ? __int_as_float(atomicMax((int *)addr, __float_as_int(value)))
              : __uint_as_float(
                    atomicMin((unsigned int *)addr, __float_as_uint(value)));
    return old;
}

template <typename T>
__global__ static void
deviceReduceStats(const T *in, float *out, int N)
{
    float min = INFINITY;
    float max = -INFINITY;
    float sum = 0;
    float sumsum = 0;

    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
        i += blockDim.x * gridDim.x) {
        float v = in[i];
        min = fminf(min, v);
        max = fmaxf(max, v);
        sum += v;
        sumsum += v * v;
    }

    float4 re4{sum, sum, sum, sumsum};

    re4 = blockReduceStats(re4);
    if(threadIdx.x == 0) {
        atomicMinFloat(out + 0, re4.x);
        atomicMaxFloat(out + 1, re4.y);
        atomicAdd(out + 2, re4.z);
        atomicAdd(out + 3, re4.w);
    }
}

__global__ static void
compute_mean_stddev(float *v, float elements)
{
    const float sum = v[2];
    const float sumsum = v[3];
    const float mean = sum / elements;
    const float var = (sumsum - sum * sum / elements) / elements;

    v[2] = mean;
    v[3] = var;
}

void
tensor_stats_float(int n, const float *src, float *output, cudaStream_t stream)
{
    cudaMemsetAsync(output, 0, sizeof(float) * 4, stream);
    deviceReduceStats<<<(n + 255) / 256, 256, 0, stream>>>(src, output, n);
    compute_mean_stddev<<<1, 1, 0, stream>>>(output, n);
}

void
tensor_stats_half(int n, const __half *src, float *output, cudaStream_t stream)
{
    cudaMemsetAsync(output, 0, sizeof(float) * 4, stream);
    deviceReduceStats<<<(n + 255) / 256, 256, 0, stream>>>(src, output, n);
    compute_mean_stddev<<<1, 1, 0, stream>>>(output, n);
}

//------------------------------------------------------------------------
// Leaky RELU
//------------------------------------------------------------------------

template <typename T>
__global__ static void
leaky_relu(int n, T *dst, const T *src, float alpha)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n)
        return;
    float v = src[i];
    dst[i] = v > 0 ? v : v * alpha;
}

void
leaky_relu_float(int elements, float *y, const float *x, float alpha,
                 cudaStream_t stream)
{
    leaky_relu<<<(elements + 255) / 256, 256, 0, stream>>>(elements, y, x,
                                                           alpha);
}

//------------------------------------------------------------------------
// Find non-finite values
//------------------------------------------------------------------------

template <typename T>
__global__ static void
find_non_finite_1d(int n_x, const T *src, uint32_t *dst, bool catch_inf)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if(x >= n_x)
        return;
    const float v = src[x];
    if(!isfinite(v)) {
        if(isnan(v) || catch_inf)
            *dst = 1;
    }
}

template <typename T>
__global__ static void
find_non_finite_2d(int n_x, int n_y, const T *src, uint32_t *dst, int stride,
                   bool catch_inf)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x >= n_x || y >= n_y)
        return;

    int o = x + stride * y;
    const float v = src[o];
    if(!isfinite(v)) {
        if(isnan(v) || catch_inf)
            *dst = 1;
    }
}

void
find_non_finite_float_1d(int x, const float *src, uint32_t *dst, bool catch_inf,
                         cudaStream_t stream)
{
    find_non_finite_1d<<<(x + 255) / 256, 256, 0, stream>>>(x, src, dst,
                                                            catch_inf);
}

void
find_non_finite_float_2d(int width, int height, int stride, const float *src,
                         uint32_t *dst, bool catch_inf, cudaStream_t stream)
{
    dim3 dimBlock(16, 16, 1);
    dim3 dimGrid;

    dimGrid.x = (width + dimBlock.x - 1) / dimBlock.x;
    dimGrid.y = (height + dimBlock.y - 1) / dimBlock.y;

    find_non_finite_2d<<<dimGrid, dimBlock, 0, stream>>>(
        width, height, src, dst, stride, catch_inf);
}

void
find_non_finite_half_1d(int x, const __half *src, uint32_t *dst, bool catch_inf,
                        cudaStream_t stream)
{
    find_non_finite_1d<<<(x + 255) / 256, 256, 0, stream>>>(x, src, dst,
                                                            catch_inf);
}

void
find_non_finite_half_2d(int width, int height, int stride, const __half *src,
                        uint32_t *dst, bool catch_inf, cudaStream_t stream)
{
    dim3 dimBlock(16, 16, 1);
    dim3 dimGrid;

    dimGrid.x = (width + dimBlock.x - 1) / dimBlock.x;
    dimGrid.y = (height + dimBlock.y - 1) / dimBlock.y;

    find_non_finite_2d<<<dimGrid, dimBlock, 0, stream>>>(
        width, height, src, dst, stride, catch_inf);
}

};  // namespace saga
