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

__device__ __attribute__((noinline)) void repeated_load(float *ac, float *out,
                                                        int i, int mask) {
  int j = i & mask;
  float a = ac[j];
  float aa = a * a;
  float b = ac[j];
  float bb = b * b;
  out[i] = aa + bb;
}

typedef void (*repeated_fn)(float *, float *, int, int);
extern __device__ void __enzyme_autodiff(repeated_fn, int, float *, float *,
                                         int, float *, float *, int, int, int,
                                         int);
extern __device__ int enzyme_dup;
extern __device__ int enzyme_const;

__global__ void init_arrays(float *ac, float *dac, float *out, float *dout,
                            int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;

  ac[i] = 1.0f + 0.001f * (float)(i & 1023);
  dac[i] = 0.0f;
  out[i] = 0.0f;
  dout[i] = 1.0f;
}

__global__ void reset_gradients(float *dac, float *out, float *dout, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;

  dac[i] = 0.0f;
  out[i] = 0.0f;
  dout[i] = 1.0f;
}

__global__ void enzyme_grad(float *ac, float *dac, float *out, float *dout,
                            int n, int mask) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;

  __enzyme_autodiff(repeated_load, enzyme_dup, ac, dac, enzyme_dup, out, dout,
                    enzyme_const, i, enzyme_const, mask);
}

__global__ void manual_two_atomic_grad(float *ac, float *dac, float *out,
                                       float *dout, int n, int mask) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;

  int j = i & mask;
  float a = ac[j];
  float aa = a * a;
  float b = ac[j];
  float bb = b * b;
  out[i] = aa + bb;
  float part = 2.0f * a * dout[i];
  atomicAdd(&dac[j], part);
  atomicAdd(&dac[j], part);
}

__global__ void manual_folded_atomic_grad(float *ac, float *dac, float *out,
                                          float *dout, int n, int mask) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;

  int j = i & mask;
  float a = ac[j];
  float aa = a * a;
  float b = ac[j];
  float bb = b * b;
  out[i] = aa + bb;
  float total = (2.0f * a + 2.0f * b) * dout[i];
  atomicAdd(&dac[j], total);
}

static int verify_result(const char *label, float *ac, float *dac, float *out,
                         int n, int bins, int mask) {
  float *hac = (float *)malloc((size_t)n * sizeof(float));
  float *hdac = (float *)malloc((size_t)n * sizeof(float));
  float *hout = (float *)malloc((size_t)n * sizeof(float));
  int *counts = (int *)calloc((size_t)bins, sizeof(int));
  if (!hac || !hdac || !hout || !counts) {
    fprintf(stderr, "host allocation failed\n");
    free(hac);
    free(hdac);
    free(hout);
    free(counts);
    return 3;
  }

  CUDA_CHECK(
      cudaMemcpy(hac, ac, (size_t)n * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(
      cudaMemcpy(hdac, dac, (size_t)n * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(
      cudaMemcpy(hout, out, (size_t)n * sizeof(float), cudaMemcpyDeviceToHost));

  for (int i = 0; i < n; ++i) {
    counts[i & mask] += 1;
    float expected_out = 2.0f * hac[i & mask] * hac[i & mask];
    if (fabsf(hout[i] - expected_out) > 2.0e-5f) {
      fprintf(stderr,
              "%s output mismatch at %d: ac=%g out=%g expected_out=%g\n", label,
              i, hac[i & mask], hout[i], expected_out);
      free(hac);
      free(hdac);
      free(hout);
      free(counts);
      return 4;
    }
  }

  for (int i = 0; i < n; ++i) {
    float expected_dac = i < bins ? 4.0f * hac[i] * (float)counts[i] : 0.0f;
    float tolerance = fmaxf(2.0e-3f, fabsf(expected_dac) * 5.0e-4f);
    if (fabsf(hdac[i] - expected_dac) > tolerance) {
      fprintf(stderr,
              "%s gradient mismatch at %d: ac=%g count=%d dac=%g "
              "expected_dac=%g tolerance=%g\n",
              label, i, hac[i], i < bins ? counts[i] : 0, hdac[i], expected_dac,
              tolerance);
      free(hac);
      free(hdac);
      free(hout);
      free(counts);
      return 4;
    }
  }

  free(hac);
  free(hdac);
  free(hout);
  free(counts);
  return 0;
}

#define DEFINE_TIME_KERNEL(timer_name, kernel_name)                            \
  static int timer_name(float *ac, float *dac, float *out, float *dout, int n, \
                        int mask, int blocks, int threads, int reps,           \
                        float *ms) {                                           \
    cudaEvent_t start;                                                         \
    cudaEvent_t stop;                                                          \
    CUDA_CHECK(cudaEventCreate(&start));                                       \
    CUDA_CHECK(cudaEventCreate(&stop));                                        \
    reset_gradients<<<blocks, threads>>>(dac, out, dout, n);                   \
    CUDA_CHECK(cudaGetLastError());                                            \
    CUDA_CHECK(cudaEventRecord(start));                                        \
    for (int i = 0; i < reps; ++i)                                             \
      kernel_name<<<blocks, threads>>>(ac, dac, out, dout, n, mask);           \
    CUDA_CHECK(cudaGetLastError());                                            \
    CUDA_CHECK(cudaEventRecord(stop));                                         \
    CUDA_CHECK(cudaEventSynchronize(stop));                                    \
    CUDA_CHECK(cudaEventElapsedTime(ms, start, stop));                         \
    *ms /= (float)reps;                                                        \
    CUDA_CHECK(cudaEventDestroy(start));                                       \
    CUDA_CHECK(cudaEventDestroy(stop));                                        \
    return 0;                                                                  \
  }

DEFINE_TIME_KERNEL(time_two_atomic, manual_two_atomic_grad)
DEFINE_TIME_KERNEL(time_folded_atomic, manual_folded_atomic_grad)
DEFINE_TIME_KERNEL(time_enzyme, enzyme_grad)

int main(int argc, char **argv) {
  int n = argc > 1 ? atoi(argv[1]) : (1 << 22);
  int reps = argc > 2 ? atoi(argv[2]) : 50;
  int bins = argc > 3 ? atoi(argv[3]) : 1;
  if (bins <= 0 || (bins & (bins - 1)) != 0 || bins > n) {
    fprintf(stderr, "bins must be a power of two between 1 and n\n");
    return 6;
  }
  int mask = bins - 1;
  int threads = 256;
  int blocks = (n + threads - 1) / threads;

  float *ac = nullptr;
  float *dac = nullptr;
  float *out = nullptr;
  float *dout = nullptr;
  CUDA_CHECK(cudaMalloc(&ac, (size_t)n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dac, (size_t)n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&out, (size_t)n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dout, (size_t)n * sizeof(float)));

  init_arrays<<<blocks, threads>>>(ac, dac, out, dout, n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  enzyme_grad<<<blocks, threads>>>(ac, dac, out, dout, n, mask);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  int verify = verify_result("enzyme", ac, dac, out, n, bins, mask);
  if (verify != 0)
    return verify;

  reset_gradients<<<blocks, threads>>>(dac, out, dout, n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  manual_folded_atomic_grad<<<blocks, threads>>>(ac, dac, out, dout, n, mask);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  verify = verify_result("manual_folded", ac, dac, out, n, bins, mask);
  if (verify != 0)
    return verify;

  float two_atomic_ms = 0.0f;
  float folded_ms = 0.0f;
  float enzyme_ms = 0.0f;
  int timing = time_two_atomic(ac, dac, out, dout, n, mask, blocks, threads,
                               reps, &two_atomic_ms);
  if (timing != 0)
    return timing;
  timing = time_folded_atomic(ac, dac, out, dout, n, mask, blocks, threads,
                              reps, &folded_ms);
  if (timing != 0)
    return timing;
  timing = time_enzyme(ac, dac, out, dout, n, mask, blocks, threads, reps,
                       &enzyme_ms);
  if (timing != 0)
    return timing;

  printf("n=%d bins=%d reps=%d two_atomic_ms=%.6f folded_ms=%.6f "
         "enzyme_ms=%.6f "
         "manual_speedup=%.3fx enzyme_speedup=%.3fx\n",
         n, bins, reps, two_atomic_ms, folded_ms, enzyme_ms,
         two_atomic_ms / folded_ms, two_atomic_ms / enzyme_ms);
  if (folded_ms >= two_atomic_ms || enzyme_ms >= two_atomic_ms) {
    fprintf(stderr,
            "expected folded and Enzyme paths to be faster than two atomics\n");
    return 5;
  }

  CUDA_CHECK(cudaFree(ac));
  CUDA_CHECK(cudaFree(dac));
  CUDA_CHECK(cudaFree(out));
  CUDA_CHECK(cudaFree(dout));
  return 0;
}
