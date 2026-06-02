#include <cuda_runtime.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define CUDA_CHECK(expr)                                                       \
  do {                                                                         \
    cudaError_t err__ = (expr);                                                \
    if (err__ != cudaSuccess) {                                                \
      fprintf(stderr, "%s failed: %s\n", #expr, cudaGetErrorString(err__));    \
      return 2;                                                                \
    }                                                                          \
  } while (0)

static inline int index_for_i(int i, int mask) {
  return ((i * 13) ^ (i >> 3)) & mask;
}

__device__ __host__ static inline int device_index_for_i(int i, int mask) {
  return ((i * 13) ^ (i >> 3)) & mask;
}

__device__ __host__ static inline float dout_for_i(int i) {
  return 1.0f + 0.125f * (float)(i & 7);
}

__device__ __attribute__((noinline)) void
repeated_load(float *ac, float *bias, float *out, int i, int mask) {
  int j = device_index_for_i(i, mask);
  int k = (j + 17) & mask;
  float a0 = ac[j];
  float b = bias[j];
  float a1 = ac[j];
  float c = ac[k];
  float a2 = ac[j];
  out[i] = a0 * a1 + 0.5f * a2 * b + 0.125f * a0 + 0.25f * c;
}

typedef void (*repeated_fn)(float *, float *, float *, int, int);
extern __device__ void __enzyme_autodiff(repeated_fn, ...);
extern __device__ int enzyme_dup;
extern __device__ int enzyme_const;

__global__ void init_arrays(float *ac, float *bias, float *dac, float *out,
                            float *dout, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;

  ac[i] = 0.75f + 0.001f * (float)(i & 2047);
  bias[i] = 0.5f + 0.0003f * (float)((i * 7) & 1023);
  dac[i] = 0.0f;
  out[i] = 0.0f;
  dout[i] = dout_for_i(i);
}

__global__ void reset_results(float *dac, float *out, float *dout, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;

  dac[i] = 0.0f;
  out[i] = 0.0f;
  dout[i] = dout_for_i(i);
}

__global__ void enzyme_grad(float *ac, float *bias, float *dac, float *out,
                            float *dout, int n, int mask) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;

  __enzyme_autodiff(repeated_load, enzyme_dup, ac, dac, enzyme_const, bias,
                    enzyme_dup, out, dout, enzyme_const, i, enzyme_const, mask);
}

static int verify_result(float *ac, float *bias, float *dac, float *out, int n,
                         int bins, int mask) {
  float *hac = (float *)malloc((size_t)n * sizeof(float));
  float *hbias = (float *)malloc((size_t)n * sizeof(float));
  float *hdac = (float *)malloc((size_t)n * sizeof(float));
  float *hout = (float *)malloc((size_t)n * sizeof(float));
  float *expected = (float *)calloc((size_t)n, sizeof(float));
  if (!hac || !hbias || !hdac || !hout || !expected) {
    fprintf(stderr, "host allocation failed\n");
    free(hac);
    free(hbias);
    free(hdac);
    free(hout);
    free(expected);
    return 3;
  }

  CUDA_CHECK(
      cudaMemcpy(hac, ac, (size_t)n * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(hbias, bias, (size_t)n * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(
      cudaMemcpy(hdac, dac, (size_t)n * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(
      cudaMemcpy(hout, out, (size_t)n * sizeof(float), cudaMemcpyDeviceToHost));

  for (int i = 0; i < n; ++i) {
    int j = index_for_i(i, mask);
    int k = (j + 17) & mask;
    float a = hac[j];
    float b = hbias[j];
    float c = hac[k];
    float d = dout_for_i(i);
    float expected_out = a * a + 0.5f * a * b + 0.125f * a + 0.25f * c;
    if (fabsf(hout[i] - expected_out) > 2.0e-5f) {
      fprintf(stderr, "output mismatch at %d: j=%d k=%d out=%g expected=%g\n",
              i, j, k, hout[i], expected_out);
      free(hac);
      free(hbias);
      free(hdac);
      free(hout);
      free(expected);
      return 4;
    }
    expected[j] += d * (2.0f * a + 0.5f * b + 0.125f);
    expected[k] += d * 0.25f;
  }

  for (int i = 0; i < n; ++i) {
    float tolerance = fmaxf(5.0e-2f, fabsf(expected[i]) * 5.0e-4f);
    if (fabsf(hdac[i] - expected[i]) > tolerance) {
      fprintf(stderr,
              "gradient mismatch at %d: dac=%g expected=%g tolerance=%g "
              "bins=%d\n",
              i, hdac[i], expected[i], tolerance, bins);
      free(hac);
      free(hbias);
      free(hdac);
      free(hout);
      free(expected);
      return 4;
    }
  }

  free(hac);
  free(hbias);
  free(hdac);
  free(hout);
  free(expected);
  return 0;
}

int main(int argc, char **argv) {
  int n = argc > 1 ? atoi(argv[1]) : (1 << 20);
  int reps = argc > 2 ? atoi(argv[2]) : 1;
  int bins = argc > 3 ? atoi(argv[3]) : 1024;
  if (bins <= 0 || (bins & (bins - 1)) != 0 || bins > n) {
    fprintf(stderr, "bins must be a power of two between 1 and n\n");
    return 6;
  }
  int mask = bins - 1;
  int threads = 256;
  int blocks = (n + threads - 1) / threads;

  float *ac = nullptr;
  float *bias = nullptr;
  float *dac = nullptr;
  float *out = nullptr;
  float *dout = nullptr;
  CUDA_CHECK(cudaMalloc(&ac, (size_t)n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&bias, (size_t)n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dac, (size_t)n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&out, (size_t)n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dout, (size_t)n * sizeof(float)));

  init_arrays<<<blocks, threads>>>(ac, bias, dac, out, dout, n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  for (int r = 0; r < reps; ++r) {
    reset_results<<<blocks, threads>>>(dac, out, dout, n);
    CUDA_CHECK(cudaGetLastError());
    enzyme_grad<<<blocks, threads>>>(ac, bias, dac, out, dout, n, mask);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  int verify = verify_result(ac, bias, dac, out, n, bins, mask);
  if (verify != 0)
    return verify;

  printf("n=%d bins=%d reps=%d cuda repeated-load gradients verified\n", n,
         bins, reps);

  CUDA_CHECK(cudaFree(ac));
  CUDA_CHECK(cudaFree(bias));
  CUDA_CHECK(cudaFree(dac));
  CUDA_CHECK(cudaFree(out));
  CUDA_CHECK(cudaFree(dout));
  return 0;
}
