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

__device__ __host__ static inline float u_for_cell(int i) {
  return 0.35f + 0.0007f * (float)(i & 1023) +
         0.0001f * (float)((i * 17) & 511);
}

__device__ __host__ static inline float v_for_cell(int i) {
  return -0.15f + 0.0005f * (float)((i * 11) & 1023) +
         0.0002f * (float)((i * 23) & 255);
}

__device__ __host__ static inline float p_for_cell(int i) {
  return 1.0f + 0.0013f * (float)((i * 7) & 2047);
}

__device__ __host__ static inline float nu_for_cell(int i) {
  return 0.02f + 0.00001f * (float)((i * 5) & 255);
}

__device__ __host__ static inline float dresu_for_cell(int i) {
  return 0.5f + 0.002f * (float)((i * 13) & 255);
}

__device__ __host__ static inline float dresv_for_cell(int i) {
  return -0.25f + 0.0015f * (float)((i * 29) & 255);
}

__device__ __host__ static inline void cell_neighbors(int i, int nx, int ny,
                                                      int *east, int *west,
                                                      int *north, int *south) {
  int xmask = nx - 1;
  int ymask = ny - 1;
  int x = i & xmask;
  int y = i / nx;

  *east = y * nx + ((x + 1) & xmask);
  *west = y * nx + ((x - 1) & xmask);
  *north = ((y + 1) & ymask) * nx + x;
  *south = ((y - 1) & ymask) * nx + x;
}

__device__ void momentum_residual_cell(float *u, float *v, float *p, float *nu,
                                       float *resu, float *resv, int i, int nx,
                                       int ny, float inv_dx, float inv_dy) {
  int east;
  int west;
  int north;
  int south;
  cell_neighbors(i, nx, ny, &east, &west, &north, &south);

  float u_c_e = u[i];
  float u_e = u[east];
  float u_face_e = 0.5f * (u_c_e + u_e);

  float u_c_w = u[i];
  float u_w = u[west];
  float u_face_w = 0.5f * (u_w + u_c_w);

  float u_c_n = u[i];
  float u_n = u[north];
  float u_face_n = 0.5f * (u_c_n + u_n);

  float u_c_s = u[i];
  float u_s = u[south];
  float u_face_s = 0.5f * (u_s + u_c_s);

  float v_c_e = v[i];
  float v_e = v[east];
  float v_face_e = 0.5f * (v_c_e + v_e);

  float v_c_w = v[i];
  float v_w = v[west];
  float v_face_w = 0.5f * (v_w + v_c_w);

  float v_c_n = v[i];
  float v_n = v[north];
  float v_face_n = 0.5f * (v_c_n + v_n);

  float v_c_s = v[i];
  float v_s = v[south];
  float v_face_s = 0.5f * (v_s + v_c_s);

  float nu_c_e = nu[i];
  float nu_e = nu[east];
  float nu_face_e = 0.5f * (nu_c_e + nu_e);

  float nu_c_w = nu[i];
  float nu_w = nu[west];
  float nu_face_w = 0.5f * (nu_w + nu_c_w);

  float nu_c_n = nu[i];
  float nu_n = nu[north];
  float nu_face_n = 0.5f * (nu_c_n + nu_n);

  float nu_c_s = nu[i];
  float nu_s = nu[south];
  float nu_face_s = 0.5f * (nu_s + nu_c_s);

  float inv_dx2 = inv_dx * inv_dx;
  float inv_dy2 = inv_dy * inv_dy;

  float conv_u = (u_face_e * u_face_e - u_face_w * u_face_w) * inv_dx +
                 (v_face_n * u_face_n - v_face_s * u_face_s) * inv_dy;
  float conv_v = (u_face_e * v_face_e - u_face_w * v_face_w) * inv_dx +
                 (v_face_n * v_face_n - v_face_s * v_face_s) * inv_dy;

  float dpdx = 0.5f * (p[east] - p[west]) * inv_dx;
  float dpdy = 0.5f * (p[north] - p[south]) * inv_dy;

  float diff_u =
      (nu_face_e * (u_e - u_c_e) - nu_face_w * (u_c_w - u_w)) * inv_dx2 +
      (nu_face_n * (u_n - u_c_n) - nu_face_s * (u_c_s - u_s)) * inv_dy2;
  float diff_v =
      (nu_face_e * (v_e - v_c_e) - nu_face_w * (v_c_w - v_w)) * inv_dx2 +
      (nu_face_n * (v_n - v_c_n) - nu_face_s * (v_c_s - v_s)) * inv_dy2;

  resu[i] = conv_u + dpdx - diff_u;
  resv[i] = conv_v + dpdy - diff_v;
}

typedef void (*momentum_fn)(float *, float *, float *, float *, float *,
                            float *, int, int, int, float, float);
extern __device__ void __enzyme_autodiff(momentum_fn, int, float *, float *,
                                         int, float *, float *, int, float *,
                                         float *, int, float *, float *, int,
                                         float *, float *, int, float *,
                                         float *, int, int, int, int, int, int,
                                         int, float, int, float);
extern __device__ int enzyme_dup;
extern __device__ int enzyme_const;

__global__ void init_primal(float *u, float *v, float *p, float *nu, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;

  u[i] = u_for_cell(i);
  v[i] = v_for_cell(i);
  p[i] = p_for_cell(i);
  nu[i] = nu_for_cell(i);
}

__global__ void reset_results(float *du, float *dv, float *dp, float *dnu,
                              float *resu, float *resv, float *dresu,
                              float *dresv, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;

  du[i] = 0.0f;
  dv[i] = 0.0f;
  dp[i] = 0.0f;
  dnu[i] = 0.0f;
  resu[i] = 0.0f;
  resv[i] = 0.0f;
  dresu[i] = dresu_for_cell(i);
  dresv[i] = dresv_for_cell(i);
}

__global__ void enzyme_momentum_grad(float *u, float *du, float *v, float *dv,
                                     float *p, float *dp, float *nu, float *dnu,
                                     float *resu, float *resv, float *dresu,
                                     float *dresv, int n, int nx, int ny,
                                     float inv_dx, float inv_dy) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;

  __enzyme_autodiff(momentum_residual_cell, enzyme_dup, u, du, enzyme_dup, v,
                    dv, enzyme_dup, p, dp, enzyme_dup, nu, dnu, enzyme_dup,
                    resu, dresu, enzyme_dup, resv, dresv, enzyme_const, i,
                    enzyme_const, nx, enzyme_const, ny, enzyme_const, inv_dx,
                    enzyme_const, inv_dy);
}

__device__ static inline void
add_face_average_adjoint(float face_adj, float *left_adj, float *right_adj) {
  float half_adj = 0.5f * face_adj;
  *left_adj += half_adj;
  *right_adj += half_adj;
}

__global__ void manual_momentum_grad(float *u, float *du, float *v, float *dv,
                                     float *p, float *dp, float *nu, float *dnu,
                                     float *resu, float *resv, float *dresu,
                                     float *dresv, int n, int nx, int ny,
                                     float inv_dx, float inv_dy) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;

  int east;
  int west;
  int north;
  int south;
  cell_neighbors(i, nx, ny, &east, &west, &north, &south);

  float u_c = u[i];
  float u_e = u[east];
  float u_w = u[west];
  float u_n = u[north];
  float u_s = u[south];

  float v_c = v[i];
  float v_e = v[east];
  float v_w = v[west];
  float v_n = v[north];
  float v_s = v[south];

  float nu_c = nu[i];
  float nu_e = nu[east];
  float nu_w = nu[west];
  float nu_n = nu[north];
  float nu_s = nu[south];

  float u_face_e = 0.5f * (u_c + u_e);
  float u_face_w = 0.5f * (u_w + u_c);
  float u_face_n = 0.5f * (u_c + u_n);
  float u_face_s = 0.5f * (u_s + u_c);

  float v_face_e = 0.5f * (v_c + v_e);
  float v_face_w = 0.5f * (v_w + v_c);
  float v_face_n = 0.5f * (v_c + v_n);
  float v_face_s = 0.5f * (v_s + v_c);

  float nu_face_e = 0.5f * (nu_c + nu_e);
  float nu_face_w = 0.5f * (nu_w + nu_c);
  float nu_face_n = 0.5f * (nu_c + nu_n);
  float nu_face_s = 0.5f * (nu_s + nu_c);

  float inv_dx2 = inv_dx * inv_dx;
  float inv_dy2 = inv_dy * inv_dy;

  float conv_u = (u_face_e * u_face_e - u_face_w * u_face_w) * inv_dx +
                 (v_face_n * u_face_n - v_face_s * u_face_s) * inv_dy;
  float conv_v = (u_face_e * v_face_e - u_face_w * v_face_w) * inv_dx +
                 (v_face_n * v_face_n - v_face_s * v_face_s) * inv_dy;

  float dpdx = 0.5f * (p[east] - p[west]) * inv_dx;
  float dpdy = 0.5f * (p[north] - p[south]) * inv_dy;

  float diff_u = (nu_face_e * (u_e - u_c) - nu_face_w * (u_c - u_w)) * inv_dx2 +
                 (nu_face_n * (u_n - u_c) - nu_face_s * (u_c - u_s)) * inv_dy2;
  float diff_v = (nu_face_e * (v_e - v_c) - nu_face_w * (v_c - v_w)) * inv_dx2 +
                 (nu_face_n * (v_n - v_c) - nu_face_s * (v_c - v_s)) * inv_dy2;

  resu[i] = conv_u + dpdx - diff_u;
  resv[i] = conv_v + dpdy - diff_v;

  float ru_adj = dresu[i];
  float rv_adj = dresv[i];
  dresu[i] = 0.0f;
  dresv[i] = 0.0f;

  float u_c_adj = 0.0f;
  float u_e_adj = 0.0f;
  float u_w_adj = 0.0f;
  float u_n_adj = 0.0f;
  float u_s_adj = 0.0f;

  float v_c_adj = 0.0f;
  float v_e_adj = 0.0f;
  float v_w_adj = 0.0f;
  float v_n_adj = 0.0f;
  float v_s_adj = 0.0f;

  float p_e_adj = 0.5f * inv_dx * ru_adj;
  float p_w_adj = -0.5f * inv_dx * ru_adj;
  float p_n_adj = 0.5f * inv_dy * rv_adj;
  float p_s_adj = -0.5f * inv_dy * rv_adj;

  float nu_c_adj = 0.0f;
  float nu_e_adj = 0.0f;
  float nu_w_adj = 0.0f;
  float nu_n_adj = 0.0f;
  float nu_s_adj = 0.0f;

  float u_face_e_adj =
      ru_adj * inv_dx * (2.0f * u_face_e) + rv_adj * inv_dx * v_face_e;
  float u_face_w_adj =
      -ru_adj * inv_dx * (2.0f * u_face_w) - rv_adj * inv_dx * v_face_w;
  float u_face_n_adj = ru_adj * inv_dy * v_face_n;
  float u_face_s_adj = -ru_adj * inv_dy * v_face_s;

  float v_face_e_adj = rv_adj * inv_dx * u_face_e;
  float v_face_w_adj = -rv_adj * inv_dx * u_face_w;
  float v_face_n_adj =
      ru_adj * inv_dy * u_face_n + rv_adj * inv_dy * (2.0f * v_face_n);
  float v_face_s_adj =
      -ru_adj * inv_dy * u_face_s - rv_adj * inv_dy * (2.0f * v_face_s);

  float diff_u_adj = -ru_adj;
  float diff_v_adj = -rv_adj;

  float nu_face_e_adj =
      diff_u_adj * inv_dx2 * (u_e - u_c) + diff_v_adj * inv_dx2 * (v_e - v_c);
  float nu_face_w_adj =
      -diff_u_adj * inv_dx2 * (u_c - u_w) - diff_v_adj * inv_dx2 * (v_c - v_w);
  float nu_face_n_adj =
      diff_u_adj * inv_dy2 * (u_n - u_c) + diff_v_adj * inv_dy2 * (v_n - v_c);
  float nu_face_s_adj =
      -diff_u_adj * inv_dy2 * (u_c - u_s) - diff_v_adj * inv_dy2 * (v_c - v_s);

  u_e_adj += diff_u_adj * inv_dx2 * nu_face_e;
  u_c_adj -= diff_u_adj * inv_dx2 * nu_face_e;
  u_c_adj -= diff_u_adj * inv_dx2 * nu_face_w;
  u_w_adj += diff_u_adj * inv_dx2 * nu_face_w;
  u_n_adj += diff_u_adj * inv_dy2 * nu_face_n;
  u_c_adj -= diff_u_adj * inv_dy2 * nu_face_n;
  u_c_adj -= diff_u_adj * inv_dy2 * nu_face_s;
  u_s_adj += diff_u_adj * inv_dy2 * nu_face_s;

  v_e_adj += diff_v_adj * inv_dx2 * nu_face_e;
  v_c_adj -= diff_v_adj * inv_dx2 * nu_face_e;
  v_c_adj -= diff_v_adj * inv_dx2 * nu_face_w;
  v_w_adj += diff_v_adj * inv_dx2 * nu_face_w;
  v_n_adj += diff_v_adj * inv_dy2 * nu_face_n;
  v_c_adj -= diff_v_adj * inv_dy2 * nu_face_n;
  v_c_adj -= diff_v_adj * inv_dy2 * nu_face_s;
  v_s_adj += diff_v_adj * inv_dy2 * nu_face_s;

  add_face_average_adjoint(u_face_e_adj, &u_c_adj, &u_e_adj);
  add_face_average_adjoint(u_face_w_adj, &u_w_adj, &u_c_adj);
  add_face_average_adjoint(u_face_n_adj, &u_c_adj, &u_n_adj);
  add_face_average_adjoint(u_face_s_adj, &u_s_adj, &u_c_adj);

  add_face_average_adjoint(v_face_e_adj, &v_c_adj, &v_e_adj);
  add_face_average_adjoint(v_face_w_adj, &v_w_adj, &v_c_adj);
  add_face_average_adjoint(v_face_n_adj, &v_c_adj, &v_n_adj);
  add_face_average_adjoint(v_face_s_adj, &v_s_adj, &v_c_adj);

  add_face_average_adjoint(nu_face_e_adj, &nu_c_adj, &nu_e_adj);
  add_face_average_adjoint(nu_face_w_adj, &nu_w_adj, &nu_c_adj);
  add_face_average_adjoint(nu_face_n_adj, &nu_c_adj, &nu_n_adj);
  add_face_average_adjoint(nu_face_s_adj, &nu_s_adj, &nu_c_adj);

  atomicAdd(&du[i], u_c_adj);
  atomicAdd(&du[east], u_e_adj);
  atomicAdd(&du[west], u_w_adj);
  atomicAdd(&du[north], u_n_adj);
  atomicAdd(&du[south], u_s_adj);

  atomicAdd(&dv[i], v_c_adj);
  atomicAdd(&dv[east], v_e_adj);
  atomicAdd(&dv[west], v_w_adj);
  atomicAdd(&dv[north], v_n_adj);
  atomicAdd(&dv[south], v_s_adj);

  atomicAdd(&dp[east], p_e_adj);
  atomicAdd(&dp[west], p_w_adj);
  atomicAdd(&dp[north], p_n_adj);
  atomicAdd(&dp[south], p_s_adj);

  atomicAdd(&dnu[i], nu_c_adj);
  atomicAdd(&dnu[east], nu_e_adj);
  atomicAdd(&dnu[west], nu_w_adj);
  atomicAdd(&dnu[north], nu_n_adj);
  atomicAdd(&dnu[south], nu_s_adj);
}

static int compare_device_arrays(const char *label, float *actual,
                                 float *expected, int n) {
  float *hactual = (float *)malloc((size_t)n * sizeof(float));
  float *hexpected = (float *)malloc((size_t)n * sizeof(float));
  if (!hactual || !hexpected) {
    fprintf(stderr, "host allocation failed while checking %s\n", label);
    free(hactual);
    free(hexpected);
    return 3;
  }

  CUDA_CHECK(cudaMemcpy(hactual, actual, (size_t)n * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(hexpected, expected, (size_t)n * sizeof(float),
                        cudaMemcpyDeviceToHost));

  for (int i = 0; i < n; ++i) {
    float tolerance = fmaxf(3.0e-3f, fabsf(hexpected[i]) * 8.0e-4f);
    if (fabsf(hactual[i] - hexpected[i]) > tolerance) {
      fprintf(stderr, "%s mismatch at %d: actual=%g expected=%g tolerance=%g\n",
              label, i, hactual[i], hexpected[i], tolerance);
      free(hactual);
      free(hexpected);
      return 4;
    }
  }

  free(hactual);
  free(hexpected);
  return 0;
}

#define CHECK_COMPARE(label, actual, expected, n)                              \
  do {                                                                         \
    int cmp__ = compare_device_arrays(label, actual, expected, n);             \
    if (cmp__ != 0)                                                            \
      return cmp__;                                                            \
  } while (0)

static int verify_against_manual(float *du, float *dv, float *dp, float *dnu,
                                 float *resu, float *resv, float *du_ref,
                                 float *dv_ref, float *dp_ref, float *dnu_ref,
                                 float *resu_ref, float *resv_ref, float *dresu,
                                 float *dresv, float *dresu_ref,
                                 float *dresv_ref, int n) {
  CHECK_COMPARE("resu", resu, resu_ref, n);
  CHECK_COMPARE("resv", resv, resv_ref, n);
  CHECK_COMPARE("du", du, du_ref, n);
  CHECK_COMPARE("dv", dv, dv_ref, n);
  CHECK_COMPARE("dp", dp, dp_ref, n);
  CHECK_COMPARE("dnu", dnu, dnu_ref, n);
  CHECK_COMPARE("dresu", dresu, dresu_ref, n);
  CHECK_COMPARE("dresv", dresv, dresv_ref, n);
  return 0;
}

static int time_reset(float *du, float *dv, float *dp, float *dnu, float *resu,
                      float *resv, float *dresu, float *dresv, int n,
                      int blocks, int threads, int reps, float *ms) {
  cudaEvent_t start;
  cudaEvent_t stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int r = 0; r < reps; ++r)
    reset_results<<<blocks, threads>>>(du, dv, dp, dnu, resu, resv, dresu,
                                       dresv, n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  CUDA_CHECK(cudaEventElapsedTime(ms, start, stop));
  *ms /= (float)reps;
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return 0;
}

#define DEFINE_TIME_KERNEL(timer_name, kernel_name)                            \
  static int timer_name(float *u, float *du, float *v, float *dv, float *p,    \
                        float *dp, float *nu, float *dnu, float *resu,         \
                        float *resv, float *dresu, float *dresv, int n,        \
                        int nx, int ny, float inv_dx, float inv_dy,            \
                        int blocks, int threads, int reps, float *ms) {        \
    cudaEvent_t start;                                                         \
    cudaEvent_t stop;                                                          \
    CUDA_CHECK(cudaEventCreate(&start));                                       \
    CUDA_CHECK(cudaEventCreate(&stop));                                        \
    CUDA_CHECK(cudaEventRecord(start));                                        \
    for (int r = 0; r < reps; ++r) {                                           \
      reset_results<<<blocks, threads>>>(du, dv, dp, dnu, resu, resv, dresu,   \
                                         dresv, n);                            \
      kernel_name<<<blocks, threads>>>(u, du, v, dv, p, dp, nu, dnu, resu,     \
                                       resv, dresu, dresv, n, nx, ny, inv_dx,  \
                                       inv_dy);                                \
    }                                                                          \
    CUDA_CHECK(cudaGetLastError());                                            \
    CUDA_CHECK(cudaEventRecord(stop));                                         \
    CUDA_CHECK(cudaEventSynchronize(stop));                                    \
    CUDA_CHECK(cudaEventElapsedTime(ms, start, stop));                         \
    *ms /= (float)reps;                                                        \
    CUDA_CHECK(cudaEventDestroy(start));                                       \
    CUDA_CHECK(cudaEventDestroy(stop));                                        \
    return 0;                                                                  \
  }

DEFINE_TIME_KERNEL(time_manual, manual_momentum_grad)
DEFINE_TIME_KERNEL(time_enzyme, enzyme_momentum_grad)

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

  float *u = nullptr;
  float *v = nullptr;
  float *p = nullptr;
  float *nu = nullptr;
  CUDA_CHECK(cudaMalloc(&u, (size_t)n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&v, (size_t)n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&p, (size_t)n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&nu, (size_t)n * sizeof(float)));

  float *du = nullptr;
  float *dv = nullptr;
  float *dp = nullptr;
  float *dnu = nullptr;
  float *resu = nullptr;
  float *resv = nullptr;
  float *dresu = nullptr;
  float *dresv = nullptr;
  CUDA_CHECK(cudaMalloc(&du, (size_t)n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dv, (size_t)n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dp, (size_t)n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dnu, (size_t)n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&resu, (size_t)n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&resv, (size_t)n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dresu, (size_t)n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dresv, (size_t)n * sizeof(float)));

  float *du_ref = nullptr;
  float *dv_ref = nullptr;
  float *dp_ref = nullptr;
  float *dnu_ref = nullptr;
  float *resu_ref = nullptr;
  float *resv_ref = nullptr;
  float *dresu_ref = nullptr;
  float *dresv_ref = nullptr;
  CUDA_CHECK(cudaMalloc(&du_ref, (size_t)n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dv_ref, (size_t)n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dp_ref, (size_t)n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dnu_ref, (size_t)n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&resu_ref, (size_t)n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&resv_ref, (size_t)n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dresu_ref, (size_t)n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dresv_ref, (size_t)n * sizeof(float)));

  init_primal<<<blocks, threads>>>(u, v, p, nu, n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  reset_results<<<blocks, threads>>>(du, dv, dp, dnu, resu, resv, dresu, dresv,
                                     n);
  reset_results<<<blocks, threads>>>(du_ref, dv_ref, dp_ref, dnu_ref, resu_ref,
                                     resv_ref, dresu_ref, dresv_ref, n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  enzyme_momentum_grad<<<blocks, threads>>>(u, du, v, dv, p, dp, nu, dnu, resu,
                                            resv, dresu, dresv, n, nx, ny,
                                            inv_dx, inv_dy);
  manual_momentum_grad<<<blocks, threads>>>(
      u, du_ref, v, dv_ref, p, dp_ref, nu, dnu_ref, resu_ref, resv_ref,
      dresu_ref, dresv_ref, n, nx, ny, inv_dx, inv_dy);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  int verify = verify_against_manual(
      du, dv, dp, dnu, resu, resv, du_ref, dv_ref, dp_ref, dnu_ref, resu_ref,
      resv_ref, dresu, dresv, dresu_ref, dresv_ref, n);
  if (verify != 0)
    return verify;

  float reset_ms = 0.0f;
  float manual_ms = 0.0f;
  float enzyme_ms = 0.0f;
  int timing = time_reset(du, dv, dp, dnu, resu, resv, dresu, dresv, n, blocks,
                          threads, reps, &reset_ms);
  if (timing != 0)
    return timing;
  timing =
      time_manual(u, du, v, dv, p, dp, nu, dnu, resu, resv, dresu, dresv, n, nx,
                  ny, inv_dx, inv_dy, blocks, threads, reps, &manual_ms);
  if (timing != 0)
    return timing;
  timing =
      time_enzyme(u, du, v, dv, p, dp, nu, dnu, resu, resv, dresu, dresv, n, nx,
                  ny, inv_dx, inv_dy, blocks, threads, reps, &enzyme_ms);
  if (timing != 0)
    return timing;

  float manual_net_ms = manual_ms - reset_ms;
  float enzyme_net_ms = enzyme_ms - reset_ms;
  printf("nx=%d ny=%d cells=%d reps=%d reset_ms=%.6f manual_ms=%.6f "
         "enzyme_ms=%.6f manual_net_ms=%.6f enzyme_net_ms=%.6f "
         "manual_net_over_enzyme_net=%.3fx\n",
         nx, ny, n, reps, reset_ms, manual_ms, enzyme_ms, manual_net_ms,
         enzyme_net_ms, manual_net_ms / enzyme_net_ms);

  CUDA_CHECK(cudaFree(u));
  CUDA_CHECK(cudaFree(v));
  CUDA_CHECK(cudaFree(p));
  CUDA_CHECK(cudaFree(nu));
  CUDA_CHECK(cudaFree(du));
  CUDA_CHECK(cudaFree(dv));
  CUDA_CHECK(cudaFree(dp));
  CUDA_CHECK(cudaFree(dnu));
  CUDA_CHECK(cudaFree(resu));
  CUDA_CHECK(cudaFree(resv));
  CUDA_CHECK(cudaFree(dresu));
  CUDA_CHECK(cudaFree(dresv));
  CUDA_CHECK(cudaFree(du_ref));
  CUDA_CHECK(cudaFree(dv_ref));
  CUDA_CHECK(cudaFree(dp_ref));
  CUDA_CHECK(cudaFree(dnu_ref));
  CUDA_CHECK(cudaFree(resu_ref));
  CUDA_CHECK(cudaFree(resv_ref));
  CUDA_CHECK(cudaFree(dresu_ref));
  CUDA_CHECK(cudaFree(dresv_ref));
  return 0;
}
