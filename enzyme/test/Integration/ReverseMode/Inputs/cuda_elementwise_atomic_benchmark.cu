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

__device__ __attribute__((noinline)) void square_atomic(float *x, float *y,
                                                        int i) {
  y[i] = x[i] * x[i];
}

__device__ __attribute__((annotate("enzyme_elementwise_read")))
__attribute__((noinline)) void
square_elementwise(float *x, float *y, int i) {
  y[i] = x[i] * x[i];
}

typedef void (*square_fn)(float *, float *, int);
extern __device__ void __enzyme_autodiff(square_fn, int, float *, float *, int,
                                         float *, float *, int, int);
extern __device__ int enzyme_dup;
extern __device__ int enzyme_const;

__global__ void init_arrays(float *x, float *dx, float *y, float *dy, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;

  x[i] = 1.0f + 0.001f * (float)(i & 1023);
  dx[i] = 0.0f;
  y[i] = 0.0f;
  dy[i] = 1.0f;
}

__global__ void reset_gradients(float *dx, float *y, float *dy, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;

  dx[i] = 0.0f;
  y[i] = 0.0f;
  dy[i] = 1.0f;
}

__global__ void square_grad_atomic(float *x, float *dx, float *y, float *dy,
                                   int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;

  __enzyme_autodiff(square_atomic, enzyme_dup, x, dx, enzyme_dup, y, dy,
                    enzyme_const, i);
}

__global__ void square_grad_elementwise(float *x, float *dx, float *y,
                                        float *dy, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;

  __enzyme_autodiff(square_elementwise, enzyme_dup, x, dx, enzyme_dup, y, dy,
                    enzyme_const, i);
}

static int verify_result(const char *label, float *x, float *dx, float *y,
                         int n) {
  float *hx = (float *)malloc((size_t)n * sizeof(float));
  float *hdx = (float *)malloc((size_t)n * sizeof(float));
  float *hy = (float *)malloc((size_t)n * sizeof(float));
  if (!hx || !hdx || !hy) {
    fprintf(stderr, "host allocation failed\n");
    free(hx);
    free(hdx);
    free(hy);
    return 3;
  }

  CUDA_CHECK(
      cudaMemcpy(hx, x, (size_t)n * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(
      cudaMemcpy(hdx, dx, (size_t)n * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(
      cudaMemcpy(hy, y, (size_t)n * sizeof(float), cudaMemcpyDeviceToHost));

  for (int i = 0; i < n; ++i) {
    float expected_y = hx[i] * hx[i];
    float expected_dx = 2.0f * hx[i];
    if (fabsf(hy[i] - expected_y) > 2.0e-5f ||
        fabsf(hdx[i] - expected_dx) > 2.0e-5f) {
      fprintf(stderr,
              "%s mismatch at %d: x=%g y=%g expected_y=%g dx=%g "
              "expected_dx=%g\n",
              label, i, hx[i], hy[i], expected_y, hdx[i], expected_dx);
      free(hx);
      free(hdx);
      free(hy);
      return 4;
    }
  }

  free(hx);
  free(hdx);
  free(hy);
  return 0;
}

static int time_atomic(float *x, float *dx, float *y, float *dy, int n,
                       int blocks, int threads, int reps, float *ms) {
  cudaEvent_t start;
  cudaEvent_t stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  reset_gradients<<<blocks, threads>>>(dx, y, dy, n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < reps; ++i)
    square_grad_atomic<<<blocks, threads>>>(x, dx, y, dy, n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  CUDA_CHECK(cudaEventElapsedTime(ms, start, stop));
  *ms /= (float)reps;
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return 0;
}

static int time_elementwise(float *x, float *dx, float *y, float *dy, int n,
                            int blocks, int threads, int reps, float *ms) {
  cudaEvent_t start;
  cudaEvent_t stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  reset_gradients<<<blocks, threads>>>(dx, y, dy, n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < reps; ++i)
    square_grad_elementwise<<<blocks, threads>>>(x, dx, y, dy, n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  CUDA_CHECK(cudaEventElapsedTime(ms, start, stop));
  *ms /= (float)reps;
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return 0;
}

int main(int argc, char **argv) {
  int n = argc > 1 ? atoi(argv[1]) : (1 << 22);
  int reps = argc > 2 ? atoi(argv[2]) : 50;
  int threads = 256;
  int blocks = (n + threads - 1) / threads;

  float *x = nullptr;
  float *dx = nullptr;
  float *y = nullptr;
  float *dy = nullptr;
  CUDA_CHECK(cudaMalloc(&x, (size_t)n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dx, (size_t)n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&y, (size_t)n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dy, (size_t)n * sizeof(float)));

  init_arrays<<<blocks, threads>>>(x, dx, y, dy, n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  square_grad_atomic<<<blocks, threads>>>(x, dx, y, dy, n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  int verify = verify_result("atomic", x, dx, y, n);
  if (verify != 0)
    return verify;

  reset_gradients<<<blocks, threads>>>(dx, y, dy, n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  square_grad_elementwise<<<blocks, threads>>>(x, dx, y, dy, n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  verify = verify_result("elementwise", x, dx, y, n);
  if (verify != 0)
    return verify;

  float atomic_ms = 0.0f;
  float elementwise_ms = 0.0f;
  int timing = time_atomic(x, dx, y, dy, n, blocks, threads, reps, &atomic_ms);
  if (timing != 0)
    return timing;
  timing =
      time_elementwise(x, dx, y, dy, n, blocks, threads, reps, &elementwise_ms);
  if (timing != 0)
    return timing;

  printf("n=%d reps=%d atomic_ms=%.6f elementwise_ms=%.6f speedup=%.3fx\n", n,
         reps, atomic_ms, elementwise_ms, atomic_ms / elementwise_ms);
  if (elementwise_ms >= atomic_ms) {
    fprintf(stderr,
            "expected elementwise path to be faster than atomic path\n");
    return 5;
  }

  CUDA_CHECK(cudaFree(x));
  CUDA_CHECK(cudaFree(dx));
  CUDA_CHECK(cudaFree(y));
  CUDA_CHECK(cudaFree(dy));
  return 0;
}
