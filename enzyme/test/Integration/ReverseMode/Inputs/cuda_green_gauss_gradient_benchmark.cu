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

__device__ __host__ static inline float phi_for_cell(int i) {
  return 0.75f + 0.0005f * (float)(i & 1023) +
         0.0002f * (float)((i * 17) & 2047);
}

__device__ __host__ static inline float dgradx_for_cell(int i) {
  return 0.5f + 0.003f * (float)((i * 13) & 255);
}

__device__ __host__ static inline float dgrady_for_cell(int i) {
  return -0.25f + 0.002f * (float)((i * 29) & 255);
}

__device__ __host__ static inline void
green_gauss_neighbors(int i, int nx, int ny, int *east, int *west, int *north,
                      int *south) {
  int xmask = nx - 1;
  int ymask = ny - 1;
  int x = i & xmask;
  int y = i / nx;

  *east = y * nx + ((x + 1) & xmask);
  *west = y * nx + ((x - 1) & xmask);
  *north = ((y + 1) & ymask) * nx + x;
  *south = ((y - 1) & ymask) * nx + x;
}

__device__ __attribute__((noinline)) void
green_gauss_cell(float *phi, float *gradx, float *grady, int i, int nx, int ny,
                 float inv_dx, float inv_dy) {
  int east;
  int west;
  int north;
  int south;
  green_gauss_neighbors(i, nx, ny, &east, &west, &north, &south);

  float phi_center_east = phi[i];
  float phi_east = phi[east];
  float face_east = 0.5f * (phi_center_east + phi_east);

  float phi_center_west = phi[i];
  float phi_west = phi[west];
  float face_west = 0.5f * (phi_center_west + phi_west);

  float phi_center_north = phi[i];
  float phi_north = phi[north];
  float face_north = 0.5f * (phi_center_north + phi_north);

  float phi_center_south = phi[i];
  float phi_south = phi[south];
  float face_south = 0.5f * (phi_center_south + phi_south);

  gradx[i] = (face_east - face_west) * inv_dx;
  grady[i] = (face_north - face_south) * inv_dy;
}

typedef void (*green_gauss_fn)(float *, float *, float *, int, int, int, float,
                               float);
extern __device__ void __enzyme_autodiff(green_gauss_fn, int, float *, float *,
                                         int, float *, float *, int, float *,
                                         float *, int, int, int, int, int, int,
                                         int, float, int, float);
extern __device__ int enzyme_dup;
extern __device__ int enzyme_const;

__global__ void init_fields(float *phi, float *dphi, float *gradx, float *grady,
                            float *dgradx, float *dgrady, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;

  phi[i] = phi_for_cell(i);
  dphi[i] = 0.0f;
  gradx[i] = 0.0f;
  grady[i] = 0.0f;
  dgradx[i] = dgradx_for_cell(i);
  dgrady[i] = dgrady_for_cell(i);
}

__global__ void reset_results(float *dphi, float *gradx, float *grady,
                              float *dgradx, float *dgrady, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;

  dphi[i] = 0.0f;
  gradx[i] = 0.0f;
  grady[i] = 0.0f;
  dgradx[i] = dgradx_for_cell(i);
  dgrady[i] = dgrady_for_cell(i);
}

__global__ void enzyme_green_gauss_grad(float *phi, float *dphi, float *gradx,
                                        float *grady, float *dgradx,
                                        float *dgrady, int n, int nx, int ny,
                                        float inv_dx, float inv_dy) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;

  __enzyme_autodiff(green_gauss_cell, enzyme_dup, phi, dphi, enzyme_dup, gradx,
                    dgradx, enzyme_dup, grady, dgrady, enzyme_const, i,
                    enzyme_const, nx, enzyme_const, ny, enzyme_const, inv_dx,
                    enzyme_const, inv_dy);
}

__global__ void manual_green_gauss_grad(float *phi, float *dphi, float *gradx,
                                        float *grady, float *dgradx,
                                        float *dgrady, int n, int nx, int ny,
                                        float inv_dx, float inv_dy) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;

  int east;
  int west;
  int north;
  int south;
  green_gauss_neighbors(i, nx, ny, &east, &west, &north, &south);

  float face_east = 0.5f * (phi[i] + phi[east]);
  float face_west = 0.5f * (phi[i] + phi[west]);
  float face_north = 0.5f * (phi[i] + phi[north]);
  float face_south = 0.5f * (phi[i] + phi[south]);

  gradx[i] = (face_east - face_west) * inv_dx;
  grady[i] = (face_north - face_south) * inv_dy;

  float dgradx_i = dgradx[i];
  float dgrady_i = dgrady[i];
  dgradx[i] = 0.0f;
  dgrady[i] = 0.0f;

  float dface_east = dgradx_i * inv_dx;
  float dface_west = -dgradx_i * inv_dx;
  float dface_north = dgrady_i * inv_dy;
  float dface_south = -dgrady_i * inv_dy;

  atomicAdd(&dphi[east], 0.5f * dface_east);
  atomicAdd(&dphi[west], 0.5f * dface_west);
  atomicAdd(&dphi[north], 0.5f * dface_north);
  atomicAdd(&dphi[south], 0.5f * dface_south);
}

static int verify_result(const char *label, float *phi, float *dphi,
                         float *gradx, float *grady, int nx, int ny,
                         float inv_dx, float inv_dy) {
  int n = nx * ny;
  float *hphi = (float *)malloc((size_t)n * sizeof(float));
  float *hdphi = (float *)malloc((size_t)n * sizeof(float));
  float *hgradx = (float *)malloc((size_t)n * sizeof(float));
  float *hgrady = (float *)malloc((size_t)n * sizeof(float));
  float *expected_dphi = (float *)calloc((size_t)n, sizeof(float));
  if (!hphi || !hdphi || !hgradx || !hgrady || !expected_dphi) {
    fprintf(stderr, "host allocation failed\n");
    free(hphi);
    free(hdphi);
    free(hgradx);
    free(hgrady);
    free(expected_dphi);
    return 3;
  }

  CUDA_CHECK(
      cudaMemcpy(hphi, phi, (size_t)n * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(hdphi, dphi, (size_t)n * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(hgradx, gradx, (size_t)n * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(hgrady, grady, (size_t)n * sizeof(float),
                        cudaMemcpyDeviceToHost));

  for (int i = 0; i < n; ++i) {
    int east;
    int west;
    int north;
    int south;
    green_gauss_neighbors(i, nx, ny, &east, &west, &north, &south);

    float face_east = 0.5f * (hphi[i] + hphi[east]);
    float face_west = 0.5f * (hphi[i] + hphi[west]);
    float face_north = 0.5f * (hphi[i] + hphi[north]);
    float face_south = 0.5f * (hphi[i] + hphi[south]);
    float expected_gradx = (face_east - face_west) * inv_dx;
    float expected_grady = (face_north - face_south) * inv_dy;
    if (fabsf(hgradx[i] - expected_gradx) > 2.0e-5f ||
        fabsf(hgrady[i] - expected_grady) > 2.0e-5f) {
      fprintf(stderr,
              "%s gradient mismatch at %d: grad=(%g,%g) expected=(%g,%g)\n",
              label, i, hgradx[i], hgrady[i], expected_gradx, expected_grady);
      free(hphi);
      free(hdphi);
      free(hgradx);
      free(hgrady);
      free(expected_dphi);
      return 4;
    }

    float dface_east = dgradx_for_cell(i) * inv_dx;
    float dface_west = -dgradx_for_cell(i) * inv_dx;
    float dface_north = dgrady_for_cell(i) * inv_dy;
    float dface_south = -dgrady_for_cell(i) * inv_dy;
    expected_dphi[i] +=
        0.5f * (dface_east + dface_west + dface_north + dface_south);
    expected_dphi[east] += 0.5f * dface_east;
    expected_dphi[west] += 0.5f * dface_west;
    expected_dphi[north] += 0.5f * dface_north;
    expected_dphi[south] += 0.5f * dface_south;
  }

  for (int i = 0; i < n; ++i) {
    float tolerance = fmaxf(2.0e-3f, fabsf(expected_dphi[i]) * 5.0e-4f);
    if (fabsf(hdphi[i] - expected_dphi[i]) > tolerance) {
      fprintf(stderr,
              "%s adjoint mismatch at %d: dphi=%g expected=%g tolerance=%g\n",
              label, i, hdphi[i], expected_dphi[i], tolerance);
      free(hphi);
      free(hdphi);
      free(hgradx);
      free(hgrady);
      free(expected_dphi);
      return 4;
    }
  }

  free(hphi);
  free(hdphi);
  free(hgradx);
  free(hgrady);
  free(expected_dphi);
  return 0;
}

#define DEFINE_TIME_KERNEL(timer_name, kernel_name)                            \
  static int timer_name(float *phi, float *dphi, float *gradx, float *grady,   \
                        float *dgradx, float *dgrady, int n, int nx, int ny,   \
                        float inv_dx, float inv_dy, int blocks, int threads,   \
                        int reps, float *ms) {                                 \
    cudaEvent_t start;                                                         \
    cudaEvent_t stop;                                                          \
    CUDA_CHECK(cudaEventCreate(&start));                                       \
    CUDA_CHECK(cudaEventCreate(&stop));                                        \
    reset_results<<<blocks, threads>>>(dphi, gradx, grady, dgradx, dgrady, n); \
    CUDA_CHECK(cudaGetLastError());                                            \
    CUDA_CHECK(cudaEventRecord(start));                                        \
    for (int r = 0; r < reps; ++r)                                             \
      kernel_name<<<blocks, threads>>>(phi, dphi, gradx, grady, dgradx,        \
                                       dgrady, n, nx, ny, inv_dx, inv_dy);     \
    CUDA_CHECK(cudaGetLastError());                                            \
    CUDA_CHECK(cudaEventRecord(stop));                                         \
    CUDA_CHECK(cudaEventSynchronize(stop));                                    \
    CUDA_CHECK(cudaEventElapsedTime(ms, start, stop));                         \
    *ms /= (float)reps;                                                        \
    CUDA_CHECK(cudaEventDestroy(start));                                       \
    CUDA_CHECK(cudaEventDestroy(stop));                                        \
    return 0;                                                                  \
  }

DEFINE_TIME_KERNEL(time_manual, manual_green_gauss_grad)
DEFINE_TIME_KERNEL(time_enzyme, enzyme_green_gauss_grad)

static int is_power_of_two(int value) {
  return value > 0 && (value & (value - 1)) == 0;
}

int main(int argc, char **argv) {
  int nx = argc > 1 ? atoi(argv[1]) : 2048;
  int ny = argc > 2 ? atoi(argv[2]) : 2048;
  int reps = argc > 3 ? atoi(argv[3]) : 50;
  if (!is_power_of_two(nx) || !is_power_of_two(ny)) {
    fprintf(stderr, "nx and ny must be powers of two\n");
    return 6;
  }
  if (nx > 65536 || ny > 65536 || nx > 2147483647 / ny) {
    fprintf(stderr, "grid dimensions are too large\n");
    return 6;
  }

  int n = nx * ny;
  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  float inv_dx = 1.0f;
  float inv_dy = 1.0f;

  float *phi = nullptr;
  float *dphi = nullptr;
  float *gradx = nullptr;
  float *grady = nullptr;
  float *dgradx = nullptr;
  float *dgrady = nullptr;
  CUDA_CHECK(cudaMalloc(&phi, (size_t)n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dphi, (size_t)n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&gradx, (size_t)n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&grady, (size_t)n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dgradx, (size_t)n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dgrady, (size_t)n * sizeof(float)));

  init_fields<<<blocks, threads>>>(phi, dphi, gradx, grady, dgradx, dgrady, n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  enzyme_green_gauss_grad<<<blocks, threads>>>(
      phi, dphi, gradx, grady, dgradx, dgrady, n, nx, ny, inv_dx, inv_dy);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  int verify =
      verify_result("enzyme", phi, dphi, gradx, grady, nx, ny, inv_dx, inv_dy);
  if (verify != 0)
    return verify;

  reset_results<<<blocks, threads>>>(dphi, gradx, grady, dgradx, dgrady, n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  manual_green_gauss_grad<<<blocks, threads>>>(
      phi, dphi, gradx, grady, dgradx, dgrady, n, nx, ny, inv_dx, inv_dy);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  verify =
      verify_result("manual", phi, dphi, gradx, grady, nx, ny, inv_dx, inv_dy);
  if (verify != 0)
    return verify;

  float manual_ms = 0.0f;
  float enzyme_ms = 0.0f;
  int timing = time_manual(phi, dphi, gradx, grady, dgradx, dgrady, n, nx, ny,
                           inv_dx, inv_dy, blocks, threads, reps, &manual_ms);
  if (timing != 0)
    return timing;
  timing = time_enzyme(phi, dphi, gradx, grady, dgradx, dgrady, n, nx, ny,
                       inv_dx, inv_dy, blocks, threads, reps, &enzyme_ms);
  if (timing != 0)
    return timing;

  printf("nx=%d ny=%d cells=%d reps=%d manual_ms=%.6f enzyme_ms=%.6f "
         "manual_over_enzyme=%.3fx\n",
         nx, ny, n, reps, manual_ms, enzyme_ms, manual_ms / enzyme_ms);

  CUDA_CHECK(cudaFree(phi));
  CUDA_CHECK(cudaFree(dphi));
  CUDA_CHECK(cudaFree(gradx));
  CUDA_CHECK(cudaFree(grady));
  CUDA_CHECK(cudaFree(dgradx));
  CUDA_CHECK(cudaFree(dgrady));
  return 0;
}
