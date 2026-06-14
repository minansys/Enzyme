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

__device__ __host__ static inline float w_for_cell(int i) {
  return 0.08f + 0.0004f * (float)((i * 19) & 1023) -
         0.00015f * (float)((i * 31) & 255);
}

__device__ __host__ static inline float p_for_cell(int i) {
  return 1.0f + 0.0013f * (float)((i * 7) & 2047);
}

__device__ __host__ static inline float nu_for_cell(int i) {
  return 0.02f + 0.00001f * (float)((i * 5) & 255);
}

__device__ __host__ static inline float dresu_for_face(int i) {
  return 0.5f + 0.002f * (float)((i * 13) & 255);
}

__device__ __host__ static inline float dresv_for_face(int i) {
  return -0.25f + 0.0015f * (float)((i * 29) & 255);
}

__device__ __host__ static inline float dresw_for_face(int i) {
  return 0.125f - 0.001f * (float)((i * 37) & 255);
}

__device__ __host__ static inline int east_cell(int i, int nx) {
  int xmask = nx - 1;
  return (i & ~xmask) + ((i + 1) & xmask);
}

__device__ __host__ static inline int north_cell(int i, int nx, int ny) {
  int xmask = nx - 1;
  int ymask = ny - 1;
  int x = i & xmask;
  int y = (i / nx) & ymask;
  int slab = i - y * nx - x;
  return slab + ((y + 1) & ymask) * nx + x;
}

__device__ __host__ static inline int top_cell(int i, int nx, int ny, int nz) {
  int xmask = nx - 1;
  int ymask = ny - 1;
  int zmask = nz - 1;
  int x = i & xmask;
  int y = (i / nx) & ymask;
  int z = i / (nx * ny);
  return (((z + 1) & zmask) * ny + y) * nx + x;
}

__device__ void momentum_face3_direct(float *u, float *v, float *w, float *p,
                                      float *nu, float *resu, float *resv,
                                      float *resw, int face, int n, int nx,
                                      int ny, int nz, float inv_dx,
                                      float inv_dy, float inv_dz) {
  if (face < n) {
    int left = face;
    int right = east_cell(left, nx);
    resu[face] =
        ((0.5f * (u[left] + u[right])) * (0.5f * (u[left] + u[right])) +
         0.5f * (p[left] + p[right]) -
         0.5f * (nu[left] + nu[right]) * (u[right] - u[left])) *
        inv_dx;
    resv[face] =
        ((0.5f * (u[left] + u[right])) * (0.5f * (v[left] + v[right])) -
         0.5f * (nu[left] + nu[right]) * (v[right] - v[left])) *
        inv_dx;
    resw[face] =
        ((0.5f * (u[left] + u[right])) * (0.5f * (w[left] + w[right])) -
         0.5f * (nu[left] + nu[right]) * (w[right] - w[left])) *
        inv_dx;
  } else if (face < 2 * n) {
    int lower = face - n;
    int upper = north_cell(lower, nx, ny);
    resu[face] =
        ((0.5f * (v[lower] + v[upper])) * (0.5f * (u[lower] + u[upper])) -
         0.5f * (nu[lower] + nu[upper]) * (u[upper] - u[lower])) *
        inv_dy;
    resv[face] =
        ((0.5f * (v[lower] + v[upper])) * (0.5f * (v[lower] + v[upper])) +
         0.5f * (p[lower] + p[upper]) -
         0.5f * (nu[lower] + nu[upper]) * (v[upper] - v[lower])) *
        inv_dy;
    resw[face] =
        ((0.5f * (v[lower] + v[upper])) * (0.5f * (w[lower] + w[upper])) -
         0.5f * (nu[lower] + nu[upper]) * (w[upper] - w[lower])) *
        inv_dy;
  } else {
    int back = face - 2 * n;
    int front = top_cell(back, nx, ny, nz);
    resu[face] =
        ((0.5f * (w[back] + w[front])) * (0.5f * (u[back] + u[front])) -
         0.5f * (nu[back] + nu[front]) * (u[front] - u[back])) *
        inv_dz;
    resv[face] =
        ((0.5f * (w[back] + w[front])) * (0.5f * (v[back] + v[front])) -
         0.5f * (nu[back] + nu[front]) * (v[front] - v[back])) *
        inv_dz;
    resw[face] =
        ((0.5f * (w[back] + w[front])) * (0.5f * (w[back] + w[front])) +
         0.5f * (p[back] + p[front]) -
         0.5f * (nu[back] + nu[front]) * (w[front] - w[back])) *
        inv_dz;
  }
}

__device__ void momentum_face3_cached(float *u, float *v, float *w, float *p,
                                      float *nu, float *resu, float *resv,
                                      float *resw, int face, int n, int nx,
                                      int ny, int nz, float inv_dx,
                                      float inv_dy, float inv_dz) {
  int dir;
  int left;
  int right;
  float scale;
  if (face < n) {
    dir = 0;
    left = face;
    right = east_cell(left, nx);
    scale = inv_dx;
  } else if (face < 2 * n) {
    dir = 1;
    left = face - n;
    right = north_cell(left, nx, ny);
    scale = inv_dy;
  } else {
    dir = 2;
    left = face - 2 * n;
    right = top_cell(left, nx, ny, nz);
    scale = inv_dz;
  }

  float ul = u[left];
  float ur = u[right];
  float vl = v[left];
  float vr = v[right];
  float wl = w[left];
  float wr = w[right];
  float pl = p[left];
  float pr = p[right];
  float nul = nu[left];
  float nur = nu[right];
  float uf = 0.5f * (ul + ur);
  float vf = 0.5f * (vl + vr);
  float wf = 0.5f * (wl + wr);
  float nuf = 0.5f * (nul + nur);

  if (dir == 0) {
    resu[face] = (uf * uf + 0.5f * (pl + pr) - nuf * (ur - ul)) * scale;
    resv[face] = (uf * vf - nuf * (vr - vl)) * scale;
    resw[face] = (uf * wf - nuf * (wr - wl)) * scale;
  } else if (dir == 1) {
    resu[face] = (vf * uf - nuf * (ur - ul)) * scale;
    resv[face] = (vf * vf + 0.5f * (pl + pr) - nuf * (vr - vl)) * scale;
    resw[face] = (vf * wf - nuf * (wr - wl)) * scale;
  } else {
    resu[face] = (wf * uf - nuf * (ur - ul)) * scale;
    resv[face] = (wf * vf - nuf * (vr - vl)) * scale;
    resw[face] = (wf * wf + 0.5f * (pl + pr) - nuf * (wr - wl)) * scale;
  }
}

typedef void (*face3_fn)(float *, float *, float *, float *, float *, float *,
                         float *, float *, int, int, int, int, int, float,
                         float, float);
extern __device__ void
__enzyme_autodiff(face3_fn, int, float *, float *, int, float *, float *, int,
                  float *, float *, int, float *, float *, int, float *,
                  float *, int, float *, float *, int, float *, float *, int,
                  float *, float *, int, int, int, int, int, int, int, int, int,
                  int, int, float, int, float, int, float);
extern __device__ int enzyme_dup;
extern __device__ int enzyme_const;

__global__ void init_primal(float *u, float *v, float *w, float *p, float *nu,
                            int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;

  u[i] = u_for_cell(i);
  v[i] = v_for_cell(i);
  w[i] = w_for_cell(i);
  p[i] = p_for_cell(i);
  nu[i] = nu_for_cell(i);
}

__global__ void reset_cells(float *du, float *dv, float *dw, float *dp,
                            float *dnu, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;

  du[i] = 0.0f;
  dv[i] = 0.0f;
  dw[i] = 0.0f;
  dp[i] = 0.0f;
  dnu[i] = 0.0f;
}

__global__ void reset_faces(float *resu, float *resv, float *resw, float *dresu,
                            float *dresv, float *dresw, int nf) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nf)
    return;

  resu[i] = 0.0f;
  resv[i] = 0.0f;
  resw[i] = 0.0f;
  dresu[i] = dresu_for_face(i);
  dresv[i] = dresv_for_face(i);
  dresw[i] = dresw_for_face(i);
}

__global__ void enzyme_face3_direct_grad(
    float *u, float *du, float *v, float *dv, float *w, float *dw, float *p,
    float *dp, float *nu, float *dnu, float *resu, float *resv, float *resw,
    float *dresu, float *dresv, float *dresw, int nf, int n, int nx, int ny,
    int nz, float inv_dx, float inv_dy, float inv_dz) {
  int face = blockIdx.x * blockDim.x + threadIdx.x;
  if (face >= nf)
    return;

  __enzyme_autodiff(
      momentum_face3_direct, enzyme_dup, u, du, enzyme_dup, v, dv, enzyme_dup,
      w, dw, enzyme_dup, p, dp, enzyme_dup, nu, dnu, enzyme_dup, resu, dresu,
      enzyme_dup, resv, dresv, enzyme_dup, resw, dresw, enzyme_const, face,
      enzyme_const, n, enzyme_const, nx, enzyme_const, ny, enzyme_const, nz,
      enzyme_const, inv_dx, enzyme_const, inv_dy, enzyme_const, inv_dz);
}

__global__ void enzyme_face3_cached_grad(
    float *u, float *du, float *v, float *dv, float *w, float *dw, float *p,
    float *dp, float *nu, float *dnu, float *resu, float *resv, float *resw,
    float *dresu, float *dresv, float *dresw, int nf, int n, int nx, int ny,
    int nz, float inv_dx, float inv_dy, float inv_dz) {
  int face = blockIdx.x * blockDim.x + threadIdx.x;
  if (face >= nf)
    return;

  __enzyme_autodiff(
      momentum_face3_cached, enzyme_dup, u, du, enzyme_dup, v, dv, enzyme_dup,
      w, dw, enzyme_dup, p, dp, enzyme_dup, nu, dnu, enzyme_dup, resu, dresu,
      enzyme_dup, resv, dresv, enzyme_dup, resw, dresw, enzyme_const, face,
      enzyme_const, n, enzyme_const, nx, enzyme_const, ny, enzyme_const, nz,
      enzyme_const, inv_dx, enzyme_const, inv_dy, enzyme_const, inv_dz);
}

__device__ static inline void
add_face_average_adjoint(float face_adj, float *left_adj, float *right_adj) {
  float half_adj = 0.5f * face_adj;
  *left_adj += half_adj;
  *right_adj += half_adj;
}

__global__ void manual_face3_grad(float *u, float *du, float *v, float *dv,
                                  float *w, float *dw, float *p, float *dp,
                                  float *nu, float *dnu, float *resu,
                                  float *resv, float *resw, float *dresu,
                                  float *dresv, float *dresw, int nf, int n,
                                  int nx, int ny, int nz, float inv_dx,
                                  float inv_dy, float inv_dz) {
  int face = blockIdx.x * blockDim.x + threadIdx.x;
  if (face >= nf)
    return;

  int dir;
  int left;
  int right;
  float scale;
  if (face < n) {
    dir = 0;
    left = face;
    right = east_cell(left, nx);
    scale = inv_dx;
  } else if (face < 2 * n) {
    dir = 1;
    left = face - n;
    right = north_cell(left, nx, ny);
    scale = inv_dy;
  } else {
    dir = 2;
    left = face - 2 * n;
    right = top_cell(left, nx, ny, nz);
    scale = inv_dz;
  }

  float ul = u[left];
  float ur = u[right];
  float vl = v[left];
  float vr = v[right];
  float wl = w[left];
  float wr = w[right];
  float pl = p[left];
  float pr = p[right];
  float nul = nu[left];
  float nur = nu[right];
  float uf = 0.5f * (ul + ur);
  float vf = 0.5f * (vl + vr);
  float wf = 0.5f * (wl + wr);
  float nuf = 0.5f * (nul + nur);

  float fu_raw;
  float fv_raw;
  float fw_raw;
  if (dir == 0) {
    fu_raw = uf * uf + 0.5f * (pl + pr) - nuf * (ur - ul);
    fv_raw = uf * vf - nuf * (vr - vl);
    fw_raw = uf * wf - nuf * (wr - wl);
  } else if (dir == 1) {
    fu_raw = vf * uf - nuf * (ur - ul);
    fv_raw = vf * vf + 0.5f * (pl + pr) - nuf * (vr - vl);
    fw_raw = vf * wf - nuf * (wr - wl);
  } else {
    fu_raw = wf * uf - nuf * (ur - ul);
    fv_raw = wf * vf - nuf * (vr - vl);
    fw_raw = wf * wf + 0.5f * (pl + pr) - nuf * (wr - wl);
  }
  resu[face] = fu_raw * scale;
  resv[face] = fv_raw * scale;
  resw[face] = fw_raw * scale;

  float fu_adj = dresu[face] * scale;
  float fv_adj = dresv[face] * scale;
  float fw_adj = dresw[face] * scale;
  dresu[face] = 0.0f;
  dresv[face] = 0.0f;
  dresw[face] = 0.0f;

  float ul_adj = 0.0f;
  float ur_adj = 0.0f;
  float vl_adj = 0.0f;
  float vr_adj = 0.0f;
  float wl_adj = 0.0f;
  float wr_adj = 0.0f;
  float pl_adj = 0.0f;
  float pr_adj = 0.0f;
  float nul_adj = 0.0f;
  float nur_adj = 0.0f;
  float uf_adj = 0.0f;
  float vf_adj = 0.0f;
  float wf_adj = 0.0f;
  float nuf_adj = 0.0f;

  if (dir == 0) {
    uf_adj += 2.0f * uf * fu_adj;
    pl_adj += 0.5f * fu_adj;
    pr_adj += 0.5f * fu_adj;
    nuf_adj -= (ur - ul) * fu_adj;
    ur_adj -= nuf * fu_adj;
    ul_adj += nuf * fu_adj;

    uf_adj += vf * fv_adj;
    vf_adj += uf * fv_adj;
    nuf_adj -= (vr - vl) * fv_adj;
    vr_adj -= nuf * fv_adj;
    vl_adj += nuf * fv_adj;

    uf_adj += wf * fw_adj;
    wf_adj += uf * fw_adj;
    nuf_adj -= (wr - wl) * fw_adj;
    wr_adj -= nuf * fw_adj;
    wl_adj += nuf * fw_adj;
  } else if (dir == 1) {
    vf_adj += uf * fu_adj;
    uf_adj += vf * fu_adj;
    nuf_adj -= (ur - ul) * fu_adj;
    ur_adj -= nuf * fu_adj;
    ul_adj += nuf * fu_adj;

    vf_adj += 2.0f * vf * fv_adj;
    pl_adj += 0.5f * fv_adj;
    pr_adj += 0.5f * fv_adj;
    nuf_adj -= (vr - vl) * fv_adj;
    vr_adj -= nuf * fv_adj;
    vl_adj += nuf * fv_adj;

    vf_adj += wf * fw_adj;
    wf_adj += vf * fw_adj;
    nuf_adj -= (wr - wl) * fw_adj;
    wr_adj -= nuf * fw_adj;
    wl_adj += nuf * fw_adj;
  } else {
    wf_adj += uf * fu_adj;
    uf_adj += wf * fu_adj;
    nuf_adj -= (ur - ul) * fu_adj;
    ur_adj -= nuf * fu_adj;
    ul_adj += nuf * fu_adj;

    wf_adj += vf * fv_adj;
    vf_adj += wf * fv_adj;
    nuf_adj -= (vr - vl) * fv_adj;
    vr_adj -= nuf * fv_adj;
    vl_adj += nuf * fv_adj;

    wf_adj += 2.0f * wf * fw_adj;
    pl_adj += 0.5f * fw_adj;
    pr_adj += 0.5f * fw_adj;
    nuf_adj -= (wr - wl) * fw_adj;
    wr_adj -= nuf * fw_adj;
    wl_adj += nuf * fw_adj;
  }

  add_face_average_adjoint(uf_adj, &ul_adj, &ur_adj);
  add_face_average_adjoint(vf_adj, &vl_adj, &vr_adj);
  add_face_average_adjoint(wf_adj, &wl_adj, &wr_adj);
  add_face_average_adjoint(nuf_adj, &nul_adj, &nur_adj);

  atomicAdd(&du[left], ul_adj);
  atomicAdd(&du[right], ur_adj);
  atomicAdd(&dv[left], vl_adj);
  atomicAdd(&dv[right], vr_adj);
  atomicAdd(&dw[left], wl_adj);
  atomicAdd(&dw[right], wr_adj);
  atomicAdd(&dp[left], pl_adj);
  atomicAdd(&dp[right], pr_adj);
  atomicAdd(&dnu[left], nul_adj);
  atomicAdd(&dnu[right], nur_adj);
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
    float tolerance = fmaxf(7.5e-2f, fabsf(hexpected[i]) * 3.0e-3f);
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

static int verify_against_manual(const char *label, float *du, float *dv,
                                 float *dw, float *dp, float *dnu, float *resu,
                                 float *resv, float *resw, float *du_ref,
                                 float *dv_ref, float *dw_ref, float *dp_ref,
                                 float *dnu_ref, float *resu_ref,
                                 float *resv_ref, float *resw_ref, int n,
                                 int nf) {
  char name[64];
  snprintf(name, sizeof(name), "%s resu", label);
  CHECK_COMPARE(name, resu, resu_ref, nf);
  snprintf(name, sizeof(name), "%s resv", label);
  CHECK_COMPARE(name, resv, resv_ref, nf);
  snprintf(name, sizeof(name), "%s resw", label);
  CHECK_COMPARE(name, resw, resw_ref, nf);
  snprintf(name, sizeof(name), "%s du", label);
  CHECK_COMPARE(name, du, du_ref, n);
  snprintf(name, sizeof(name), "%s dv", label);
  CHECK_COMPARE(name, dv, dv_ref, n);
  snprintf(name, sizeof(name), "%s dw", label);
  CHECK_COMPARE(name, dw, dw_ref, n);
  snprintf(name, sizeof(name), "%s dp", label);
  CHECK_COMPARE(name, dp, dp_ref, n);
  snprintf(name, sizeof(name), "%s dnu", label);
  CHECK_COMPARE(name, dnu, dnu_ref, n);
  return 0;
}

static int reset_all(float *du, float *dv, float *dw, float *dp, float *dnu,
                     float *resu, float *resv, float *resw, float *dresu,
                     float *dresv, float *dresw, int n, int nf, int cell_blocks,
                     int face_blocks, int threads) {
  reset_cells<<<cell_blocks, threads>>>(du, dv, dw, dp, dnu, n);
  reset_faces<<<face_blocks, threads>>>(resu, resv, resw, dresu, dresv, dresw,
                                        nf);
  CUDA_CHECK(cudaGetLastError());
  return 0;
}

static int time_reset(float *du, float *dv, float *dw, float *dp, float *dnu,
                      float *resu, float *resv, float *resw, float *dresu,
                      float *dresv, float *dresw, int n, int nf,
                      int cell_blocks, int face_blocks, int threads, int reps,
                      float *ms) {
  cudaEvent_t start;
  cudaEvent_t stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int r = 0; r < reps; ++r)
    reset_all(du, dv, dw, dp, dnu, resu, resv, resw, dresu, dresv, dresw, n, nf,
              cell_blocks, face_blocks, threads);
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  CUDA_CHECK(cudaEventElapsedTime(ms, start, stop));
  *ms /= (float)reps;
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return 0;
}

#define DEFINE_TIME_KERNEL(timer_name, kernel_name)                            \
  static int timer_name(                                                       \
      float *u, float *du, float *v, float *dv, float *w, float *dw, float *p, \
      float *dp, float *nu, float *dnu, float *resu, float *resv, float *resw, \
      float *dresu, float *dresv, float *dresw, int nf, int n, int nx, int ny, \
      int nz, float inv_dx, float inv_dy, float inv_dz, int cell_blocks,       \
      int face_blocks, int threads, int reps, float *ms) {                     \
    cudaEvent_t start;                                                         \
    cudaEvent_t stop;                                                          \
    CUDA_CHECK(cudaEventCreate(&start));                                       \
    CUDA_CHECK(cudaEventCreate(&stop));                                        \
    CUDA_CHECK(cudaEventRecord(start));                                        \
    for (int r = 0; r < reps; ++r) {                                           \
      reset_all(du, dv, dw, dp, dnu, resu, resv, resw, dresu, dresv, dresw, n, \
                nf, cell_blocks, face_blocks, threads);                        \
      kernel_name<<<face_blocks, threads>>>(                                   \
          u, du, v, dv, w, dw, p, dp, nu, dnu, resu, resv, resw, dresu, dresv, \
          dresw, nf, n, nx, ny, nz, inv_dx, inv_dy, inv_dz);                   \
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

DEFINE_TIME_KERNEL(time_manual, manual_face3_grad)
DEFINE_TIME_KERNEL(time_enzyme_direct, enzyme_face3_direct_grad)
DEFINE_TIME_KERNEL(time_enzyme_cached, enzyme_face3_cached_grad)

static int is_power_of_two(int value) {
  return value > 0 && (value & (value - 1)) == 0;
}

static void free_all(float **ptrs, int count) {
  for (int i = 0; i < count; ++i)
    cudaFree(ptrs[i]);
}

int main(int argc, char **argv) {
  int nx = argc > 1 ? atoi(argv[1]) : 128;
  int ny = argc > 2 ? atoi(argv[2]) : 128;
  int nz = argc > 3 ? atoi(argv[3]) : 128;
  int reps = argc > 4 ? atoi(argv[4]) : 50;
  if (!is_power_of_two(nx) || !is_power_of_two(ny) || !is_power_of_two(nz)) {
    fprintf(stderr, "nx, ny, and nz must be powers of two\n");
    return 6;
  }
  if (nx > 4096 || ny > 4096 || nz > 4096 || nx > 2147483647 / ny ||
      nx * ny > 2147483647 / nz) {
    fprintf(stderr, "grid dimensions are too large\n");
    return 6;
  }

  int n = nx * ny * nz;
  int nf = 3 * n;
  int threads = 256;
  int cell_blocks = (n + threads - 1) / threads;
  int face_blocks = (nf + threads - 1) / threads;
  float inv_dx = 1.0f;
  float inv_dy = 1.0f;
  float inv_dz = 1.0f;

  float *u = nullptr, *v = nullptr, *w = nullptr, *p = nullptr, *nu = nullptr;
  float *du = nullptr, *dv = nullptr, *dw = nullptr, *dp = nullptr;
  float *dnu = nullptr, *resu = nullptr, *resv = nullptr, *resw = nullptr;
  float *dresu = nullptr, *dresv = nullptr, *dresw = nullptr;
  float *du_ref = nullptr, *dv_ref = nullptr, *dw_ref = nullptr;
  float *dp_ref = nullptr, *dnu_ref = nullptr, *resu_ref = nullptr;
  float *resv_ref = nullptr, *resw_ref = nullptr, *dresu_ref = nullptr;
  float *dresv_ref = nullptr, *dresw_ref = nullptr;

  CUDA_CHECK(cudaMalloc(&u, (size_t)n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&v, (size_t)n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&w, (size_t)n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&p, (size_t)n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&nu, (size_t)n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&du, (size_t)n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dv, (size_t)n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dw, (size_t)n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dp, (size_t)n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dnu, (size_t)n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&resu, (size_t)nf * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&resv, (size_t)nf * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&resw, (size_t)nf * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dresu, (size_t)nf * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dresv, (size_t)nf * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dresw, (size_t)nf * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&du_ref, (size_t)n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dv_ref, (size_t)n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dw_ref, (size_t)n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dp_ref, (size_t)n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dnu_ref, (size_t)n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&resu_ref, (size_t)nf * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&resv_ref, (size_t)nf * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&resw_ref, (size_t)nf * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dresu_ref, (size_t)nf * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dresv_ref, (size_t)nf * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dresw_ref, (size_t)nf * sizeof(float)));

  init_primal<<<cell_blocks, threads>>>(u, v, w, p, nu, n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  reset_all(du, dv, dw, dp, dnu, resu, resv, resw, dresu, dresv, dresw, n, nf,
            cell_blocks, face_blocks, threads);
  reset_all(du_ref, dv_ref, dw_ref, dp_ref, dnu_ref, resu_ref, resv_ref,
            resw_ref, dresu_ref, dresv_ref, dresw_ref, n, nf, cell_blocks,
            face_blocks, threads);
  CUDA_CHECK(cudaDeviceSynchronize());
  enzyme_face3_direct_grad<<<face_blocks, threads>>>(
      u, du, v, dv, w, dw, p, dp, nu, dnu, resu, resv, resw, dresu, dresv,
      dresw, nf, n, nx, ny, nz, inv_dx, inv_dy, inv_dz);
  manual_face3_grad<<<face_blocks, threads>>>(
      u, du_ref, v, dv_ref, w, dw_ref, p, dp_ref, nu, dnu_ref, resu_ref,
      resv_ref, resw_ref, dresu_ref, dresv_ref, dresw_ref, nf, n, nx, ny, nz,
      inv_dx, inv_dy, inv_dz);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  int verify = verify_against_manual(
      "direct", du, dv, dw, dp, dnu, resu, resv, resw, du_ref, dv_ref, dw_ref,
      dp_ref, dnu_ref, resu_ref, resv_ref, resw_ref, n, nf);
  if (verify != 0)
    return verify;

  reset_all(du, dv, dw, dp, dnu, resu, resv, resw, dresu, dresv, dresw, n, nf,
            cell_blocks, face_blocks, threads);
  reset_all(du_ref, dv_ref, dw_ref, dp_ref, dnu_ref, resu_ref, resv_ref,
            resw_ref, dresu_ref, dresv_ref, dresw_ref, n, nf, cell_blocks,
            face_blocks, threads);
  CUDA_CHECK(cudaDeviceSynchronize());
  enzyme_face3_cached_grad<<<face_blocks, threads>>>(
      u, du, v, dv, w, dw, p, dp, nu, dnu, resu, resv, resw, dresu, dresv,
      dresw, nf, n, nx, ny, nz, inv_dx, inv_dy, inv_dz);
  manual_face3_grad<<<face_blocks, threads>>>(
      u, du_ref, v, dv_ref, w, dw_ref, p, dp_ref, nu, dnu_ref, resu_ref,
      resv_ref, resw_ref, dresu_ref, dresv_ref, dresw_ref, nf, n, nx, ny, nz,
      inv_dx, inv_dy, inv_dz);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  verify = verify_against_manual("cached", du, dv, dw, dp, dnu, resu, resv,
                                 resw, du_ref, dv_ref, dw_ref, dp_ref, dnu_ref,
                                 resu_ref, resv_ref, resw_ref, n, nf);
  if (verify != 0)
    return verify;

  float reset_ms = 0.0f;
  float manual_ms = 0.0f;
  float enzyme_direct_ms = 0.0f;
  float enzyme_cached_ms = 0.0f;
  int timing =
      time_reset(du, dv, dw, dp, dnu, resu, resv, resw, dresu, dresv, dresw, n,
                 nf, cell_blocks, face_blocks, threads, reps, &reset_ms);
  if (timing != 0)
    return timing;
  timing =
      time_manual(u, du, v, dv, w, dw, p, dp, nu, dnu, resu, resv, resw, dresu,
                  dresv, dresw, nf, n, nx, ny, nz, inv_dx, inv_dy, inv_dz,
                  cell_blocks, face_blocks, threads, reps, &manual_ms);
  if (timing != 0)
    return timing;
  timing = time_enzyme_direct(u, du, v, dv, w, dw, p, dp, nu, dnu, resu, resv,
                              resw, dresu, dresv, dresw, nf, n, nx, ny, nz,
                              inv_dx, inv_dy, inv_dz, cell_blocks, face_blocks,
                              threads, reps, &enzyme_direct_ms);
  if (timing != 0)
    return timing;
  timing = time_enzyme_cached(u, du, v, dv, w, dw, p, dp, nu, dnu, resu, resv,
                              resw, dresu, dresv, dresw, nf, n, nx, ny, nz,
                              inv_dx, inv_dy, inv_dz, cell_blocks, face_blocks,
                              threads, reps, &enzyme_cached_ms);
  if (timing != 0)
    return timing;

  float manual_net = manual_ms - reset_ms;
  float enzyme_direct_net = enzyme_direct_ms - reset_ms;
  float enzyme_cached_net = enzyme_cached_ms - reset_ms;
  printf("nx=%d ny=%d nz=%d cells=%d faces=%d reps=%d reset_ms=%.6f "
         "manual_ms=%.6f enzyme_direct_ms=%.6f enzyme_cached_ms=%.6f "
         "manual_net_ms=%.6f enzyme_direct_net_ms=%.6f "
         "enzyme_cached_net_ms=%.6f manual_over_direct=%.3fx "
         "manual_over_cached=%.3fx\n",
         nx, ny, nz, n, nf, reps, reset_ms, manual_ms, enzyme_direct_ms,
         enzyme_cached_ms, manual_net, enzyme_direct_net, enzyme_cached_net,
         manual_net / enzyme_direct_net, manual_net / enzyme_cached_net);

  float *ptrs[] = {u,         v,         w,        p,        nu,       du,
                   dv,        dw,        dp,       dnu,      resu,     resv,
                   resw,      dresu,     dresv,    dresw,    du_ref,   dv_ref,
                   dw_ref,    dp_ref,    dnu_ref,  resu_ref, resv_ref, resw_ref,
                   dresu_ref, dresv_ref, dresw_ref};
  free_all(ptrs, (int)(sizeof(ptrs) / sizeof(ptrs[0])));
  return 0;
}
