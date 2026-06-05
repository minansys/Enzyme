#include <cuda_runtime.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// 3D unstructured finite-volume face-loop repro. This mirrors the shape of
// industrial CFD residual assembly: Green-Gauss gradients are accumulated to
// cells from faces, then a face loop reconstructs states and atomically adds
// momentum residual contributions to owner/neighbor cells.

#define CUDA_CHECK(expr)                                                       \
  do {                                                                         \
    cudaError_t err__ = (expr);                                                \
    if (err__ != cudaSuccess) {                                                \
      fprintf(stderr, "%s failed: %s\n", #expr, cudaGetErrorString(err__));    \
      return 2;                                                                \
    }                                                                          \
  } while (0)

__device__ __host__ static inline unsigned hash_u32(unsigned x) {
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  x ^= x >> 16;
  return x;
}

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

__device__ __host__ static inline float invvol_for_cell(int i) {
  return 0.8f + 0.0002f * (float)((i * 3) & 511);
}

__device__ __host__ static inline float dres_for_cell(int dim, int i) {
  float base = dim == 0 ? 0.5f : (dim == 1 ? -0.25f : 0.125f);
  int scale = dim == 0 ? 13 : (dim == 1 ? 29 : 37);
  return base + 0.0015f * (float)((i * scale) & 255);
}

__global__ void init_primal(float **vel, float *p, float *nu, float *invvol,
                            int ncell) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= ncell)
    return;

  vel[0][i] = u_for_cell(i);
  vel[1][i] = v_for_cell(i);
  vel[2][i] = w_for_cell(i);
  p[i] = p_for_cell(i);
  nu[i] = nu_for_cell(i);
  invvol[i] = invvol_for_cell(i);
}

__global__ void init_unstructured_faces(int *owner, int *neighbor, float *snx,
                                        float *sny, float *snz, float *area,
                                        float *inv_dist, int nface, int ncell) {
  int face = blockIdx.x * blockDim.x + threadIdx.x;
  if (face >= nface)
    return;

  unsigned h0 = hash_u32((unsigned)face);
  unsigned h1 = hash_u32(h0 + 0x9e3779b9u);
  int left = (int)(h0 % (unsigned)ncell);
  int jump = 1 + (int)(h1 % (unsigned)(ncell - 1));
  int right = left + jump;
  if (right >= ncell)
    right -= ncell;

  float fx = (float)((h0 >> 0) & 255) * (1.0f / 255.0f);
  float fy = (float)((h0 >> 8) & 255) * (1.0f / 255.0f);
  float fz = (float)((h0 >> 16) & 255) * (1.0f / 255.0f);
  float nx = 2.0f * fx - 1.0f;
  float ny = 2.0f * fy - 1.0f;
  float nz = 2.0f * fz - 1.0f;
  float len = sqrtf(nx * nx + ny * ny + nz * nz) + 1.0e-6f;

  owner[face] = left;
  neighbor[face] = right;
  snx[face] = nx / len;
  sny[face] = ny / len;
  snz[face] = nz / len;
  area[face] = 0.75f + 0.5f * (float)((h1 >> 4) & 255) * (1.0f / 255.0f);
  inv_dist[face] = 0.6f + 0.4f * (float)((h1 >> 12) & 255) * (1.0f / 255.0f);
}

__global__ void zero_rows(float **rows, int nrow, int ncell) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = nrow * ncell;
  if (idx >= total)
    return;
  int row = idx / ncell;
  int cell = idx - row * ncell;
  rows[row][cell] = 0.0f;
}

__global__ void zero_scalar(float *x, int ncell) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= ncell)
    return;
  x[i] = 0.0f;
}

__global__ void reset_residual(float **res, int ncell) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = 3 * ncell;
  if (idx >= total)
    return;
  int row = idx / ncell;
  int cell = idx - row * ncell;
  res[row][cell] = 0.0f;
}

__global__ void reset_residual_and_seed(float **res, float **dres, int ncell) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = 3 * ncell;
  if (idx >= total)
    return;
  int row = idx / ncell;
  int cell = idx - row * ncell;
  res[row][cell] = 0.0f;
  dres[row][cell] = dres_for_cell(row, cell);
}

__global__ void green_gauss_gradients(float **vel, float *p, float **gradv,
                                      float **gradp, int *owner, int *neighbor,
                                      float *snx, float *sny, float *snz,
                                      float *area, float *invvol, int nface) {
  int face = blockIdx.x * blockDim.x + threadIdx.x;
  if (face >= nface)
    return;

  int left = owner[face];
  int right = neighbor[face];
  float sx = snx[face] * area[face];
  float sy = sny[face] * area[face];
  float sz = snz[face] * area[face];
  float il = invvol[left];
  float ir = invvol[right];

  float uf = 0.5f * (vel[0][left] + vel[0][right]);
  float vf = 0.5f * (vel[1][left] + vel[1][right]);
  float wf = 0.5f * (vel[2][left] + vel[2][right]);
  float pf = 0.5f * (p[left] + p[right]);

  atomicAdd(&gradv[0][left], uf * sx * il);
  atomicAdd(&gradv[1][left], uf * sy * il);
  atomicAdd(&gradv[2][left], uf * sz * il);
  atomicAdd(&gradv[3][left], vf * sx * il);
  atomicAdd(&gradv[4][left], vf * sy * il);
  atomicAdd(&gradv[5][left], vf * sz * il);
  atomicAdd(&gradv[6][left], wf * sx * il);
  atomicAdd(&gradv[7][left], wf * sy * il);
  atomicAdd(&gradv[8][left], wf * sz * il);
  atomicAdd(&gradp[0][left], pf * sx * il);
  atomicAdd(&gradp[1][left], pf * sy * il);
  atomicAdd(&gradp[2][left], pf * sz * il);

  atomicAdd(&gradv[0][right], -uf * sx * ir);
  atomicAdd(&gradv[1][right], -uf * sy * ir);
  atomicAdd(&gradv[2][right], -uf * sz * ir);
  atomicAdd(&gradv[3][right], -vf * sx * ir);
  atomicAdd(&gradv[4][right], -vf * sy * ir);
  atomicAdd(&gradv[5][right], -vf * sz * ir);
  atomicAdd(&gradv[6][right], -wf * sx * ir);
  atomicAdd(&gradv[7][right], -wf * sy * ir);
  atomicAdd(&gradv[8][right], -wf * sz * ir);
  atomicAdd(&gradp[0][right], -pf * sx * ir);
  atomicAdd(&gradp[1][right], -pf * sy * ir);
  atomicAdd(&gradp[2][right], -pf * sz * ir);
}

__device__ void momentum_fvm3_direct(float **vel, float **gradv, float **gradp,
                                     float *p, float *nu, float **res,
                                     int *owner, int *neighbor, float *snx,
                                     float *sny, float *snz, float *area,
                                     float *inv_dist, float *invvol, int face) {
  int left = owner[face];
  int right = neighbor[face];
  float dx = 0.5f * snx[face] / inv_dist[face];
  float dy = 0.5f * sny[face] / inv_dist[face];
  float dz = 0.5f * snz[face] / inv_dist[face];

  float ulf = vel[0][left] + gradv[0][left] * dx + gradv[1][left] * dy +
              gradv[2][left] * dz;
  float urf = vel[0][right] - gradv[0][right] * dx - gradv[1][right] * dy -
              gradv[2][right] * dz;
  float vlf = vel[1][left] + gradv[3][left] * dx + gradv[4][left] * dy +
              gradv[5][left] * dz;
  float vrf = vel[1][right] - gradv[3][right] * dx - gradv[4][right] * dy -
              gradv[5][right] * dz;
  float wlf = vel[2][left] + gradv[6][left] * dx + gradv[7][left] * dy +
              gradv[8][left] * dz;
  float wrf = vel[2][right] - gradv[6][right] * dx - gradv[7][right] * dy -
              gradv[8][right] * dz;
  float plf =
      p[left] + gradp[0][left] * dx + gradp[1][left] * dy + gradp[2][left] * dz;
  float prf = p[right] - gradp[0][right] * dx - gradp[1][right] * dy -
              gradp[2][right] * dz;

  float uf = 0.5f * (ulf + urf);
  float vf = 0.5f * (vlf + vrf);
  float wf = 0.5f * (wlf + wrf);
  float pf = 0.5f * (plf + prf);
  float un = uf * snx[face] + vf * sny[face] + wf * snz[face];
  float nuf = 0.5f * (nu[left] + nu[right]);

  float gudn = 0.5f * (gradv[0][left] + gradv[0][right]) * snx[face] +
               0.5f * (gradv[1][left] + gradv[1][right]) * sny[face] +
               0.5f * (gradv[2][left] + gradv[2][right]) * snz[face];
  float gvdn = 0.5f * (gradv[3][left] + gradv[3][right]) * snx[face] +
               0.5f * (gradv[4][left] + gradv[4][right]) * sny[face] +
               0.5f * (gradv[5][left] + gradv[5][right]) * snz[face];
  float gwdn = 0.5f * (gradv[6][left] + gradv[6][right]) * snx[face] +
               0.5f * (gradv[7][left] + gradv[7][right]) * sny[face] +
               0.5f * (gradv[8][left] + gradv[8][right]) * snz[face];

  float fluxu = area[face] * (un * uf + pf * snx[face] - nuf * gudn);
  float fluxv = area[face] * (un * vf + pf * sny[face] - nuf * gvdn);
  float fluxw = area[face] * (un * wf + pf * snz[face] - nuf * gwdn);

  atomicAdd(&res[0][left], fluxu * invvol[left]);
  atomicAdd(&res[1][left], fluxv * invvol[left]);
  atomicAdd(&res[2][left], fluxw * invvol[left]);
  atomicAdd(&res[0][right], -fluxu * invvol[right]);
  atomicAdd(&res[1][right], -fluxv * invvol[right]);
  atomicAdd(&res[2][right], -fluxw * invvol[right]);
}

__device__ void momentum_fvm3_cached(float **vel, float **gradv, float **gradp,
                                     float *p, float *nu, float **res,
                                     int *owner, int *neighbor, float *snx,
                                     float *sny, float *snz, float *area,
                                     float *inv_dist, float *invvol, int face) {
  float *u = vel[0];
  float *v = vel[1];
  float *w = vel[2];
  float *resu = res[0];
  float *resv = res[1];
  float *resw = res[2];
  float *gu0 = gradv[0];
  float *gu1 = gradv[1];
  float *gu2 = gradv[2];
  float *gv0 = gradv[3];
  float *gv1 = gradv[4];
  float *gv2 = gradv[5];
  float *gw0 = gradv[6];
  float *gw1 = gradv[7];
  float *gw2 = gradv[8];
  float *gp0 = gradp[0];
  float *gp1 = gradp[1];
  float *gp2 = gradp[2];
  int left = owner[face];
  int right = neighbor[face];
  float nx = snx[face];
  float ny = sny[face];
  float nz = snz[face];
  float a = area[face];
  float id = inv_dist[face];
  float dx = 0.5f * nx / id;
  float dy = 0.5f * ny / id;
  float dz = 0.5f * nz / id;
  float il = invvol[left];
  float ir = invvol[right];

  float ulf = u[left] + gu0[left] * dx + gu1[left] * dy + gu2[left] * dz;
  float urf = u[right] - gu0[right] * dx - gu1[right] * dy - gu2[right] * dz;
  float vlf = v[left] + gv0[left] * dx + gv1[left] * dy + gv2[left] * dz;
  float vrf = v[right] - gv0[right] * dx - gv1[right] * dy - gv2[right] * dz;
  float wlf = w[left] + gw0[left] * dx + gw1[left] * dy + gw2[left] * dz;
  float wrf = w[right] - gw0[right] * dx - gw1[right] * dy - gw2[right] * dz;
  float plf = p[left] + gp0[left] * dx + gp1[left] * dy + gp2[left] * dz;
  float prf = p[right] - gp0[right] * dx - gp1[right] * dy - gp2[right] * dz;
  float uf = 0.5f * (ulf + urf);
  float vf = 0.5f * (vlf + vrf);
  float wf = 0.5f * (wlf + wrf);
  float pf = 0.5f * (plf + prf);
  float un = uf * nx + vf * ny + wf * nz;
  float nuf = 0.5f * (nu[left] + nu[right]);
  float gudn = 0.5f * (gu0[left] + gu0[right]) * nx +
               0.5f * (gu1[left] + gu1[right]) * ny +
               0.5f * (gu2[left] + gu2[right]) * nz;
  float gvdn = 0.5f * (gv0[left] + gv0[right]) * nx +
               0.5f * (gv1[left] + gv1[right]) * ny +
               0.5f * (gv2[left] + gv2[right]) * nz;
  float gwdn = 0.5f * (gw0[left] + gw0[right]) * nx +
               0.5f * (gw1[left] + gw1[right]) * ny +
               0.5f * (gw2[left] + gw2[right]) * nz;
  float fluxu = a * (un * uf + pf * nx - nuf * gudn);
  float fluxv = a * (un * vf + pf * ny - nuf * gvdn);
  float fluxw = a * (un * wf + pf * nz - nuf * gwdn);

  atomicAdd(&resu[left], fluxu * il);
  atomicAdd(&resv[left], fluxv * il);
  atomicAdd(&resw[left], fluxw * il);
  atomicAdd(&resu[right], -fluxu * ir);
  atomicAdd(&resv[right], -fluxv * ir);
  atomicAdd(&resw[right], -fluxw * ir);
}

typedef void (*fvm3_fn)(float **, float **, float **, float *, float *,
                        float **, int *, int *, float *, float *, float *,
                        float *, float *, float *, int);
extern __device__ void __enzyme_autodiff(fvm3_fn, ...);
extern __device__ int enzyme_dup;
extern __device__ int enzyme_const;

__global__ void forward_fvm3_cached(float **vel, float **gradv, float **gradp,
                                    float *p, float *nu, float **res,
                                    int *owner, int *neighbor, float *snx,
                                    float *sny, float *snz, float *area,
                                    float *inv_dist, float *invvol, int nface) {
  int face = blockIdx.x * blockDim.x + threadIdx.x;
  if (face >= nface)
    return;
  momentum_fvm3_cached(vel, gradv, gradp, p, nu, res, owner, neighbor, snx, sny,
                       snz, area, inv_dist, invvol, face);
}

__global__ void enzyme_fvm3_direct_grad(
    float **vel, float **dvel, float **gradv, float **dgradv, float **gradp,
    float **dgradp, float *p, float *dp, float *nu, float *dnu, float **res,
    float **dres, int *owner, int *neighbor, float *snx, float *sny, float *snz,
    float *area, float *inv_dist, float *invvol, int nface) {
  int face = blockIdx.x * blockDim.x + threadIdx.x;
  if (face >= nface)
    return;

  __enzyme_autodiff(
      momentum_fvm3_direct, enzyme_dup, vel, dvel, enzyme_dup, gradv, dgradv,
      enzyme_dup, gradp, dgradp, enzyme_dup, p, dp, enzyme_dup, nu, dnu,
      enzyme_dup, res, dres, enzyme_const, owner, enzyme_const, neighbor,
      enzyme_const, snx, enzyme_const, sny, enzyme_const, snz, enzyme_const,
      area, enzyme_const, inv_dist, enzyme_const, invvol, enzyme_const, face);
}

__global__ void enzyme_fvm3_cached_grad(
    float **vel, float **dvel, float **gradv, float **dgradv, float **gradp,
    float **dgradp, float *p, float *dp, float *nu, float *dnu, float **res,
    float **dres, int *owner, int *neighbor, float *snx, float *sny, float *snz,
    float *area, float *inv_dist, float *invvol, int nface) {
  int face = blockIdx.x * blockDim.x + threadIdx.x;
  if (face >= nface)
    return;

  __enzyme_autodiff(
      momentum_fvm3_cached, enzyme_dup, vel, dvel, enzyme_dup, gradv, dgradv,
      enzyme_dup, gradp, dgradp, enzyme_dup, p, dp, enzyme_dup, nu, dnu,
      enzyme_dup, res, dres, enzyme_const, owner, enzyme_const, neighbor,
      enzyme_const, snx, enzyme_const, sny, enzyme_const, snz, enzyme_const,
      area, enzyme_const, inv_dist, enzyme_const, invvol, enzyme_const, face);
}

__global__ void manual_fvm3_grad(float **vel, float **dvel, float **gradv,
                                 float **dgradv, float **gradp, float **dgradp,
                                 float *p, float *dp, float *nu, float *dnu,
                                 float **res, float **dres, int *owner,
                                 int *neighbor, float *snx, float *sny,
                                 float *snz, float *area, float *inv_dist,
                                 float *invvol, int nface) {
  int face = blockIdx.x * blockDim.x + threadIdx.x;
  if (face >= nface)
    return;

  float *u = vel[0];
  float *v = vel[1];
  float *w = vel[2];
  float *du = dvel[0];
  float *dv = dvel[1];
  float *dw = dvel[2];
  float *resu = res[0];
  float *resv = res[1];
  float *resw = res[2];
  float *dresu = dres[0];
  float *dresv = dres[1];
  float *dresw = dres[2];
  float *gu0 = gradv[0];
  float *gu1 = gradv[1];
  float *gu2 = gradv[2];
  float *gv0 = gradv[3];
  float *gv1 = gradv[4];
  float *gv2 = gradv[5];
  float *gw0 = gradv[6];
  float *gw1 = gradv[7];
  float *gw2 = gradv[8];
  float *dgu0 = dgradv[0];
  float *dgu1 = dgradv[1];
  float *dgu2 = dgradv[2];
  float *dgv0 = dgradv[3];
  float *dgv1 = dgradv[4];
  float *dgv2 = dgradv[5];
  float *dgw0 = dgradv[6];
  float *dgw1 = dgradv[7];
  float *dgw2 = dgradv[8];
  float *gp0 = gradp[0];
  float *gp1 = gradp[1];
  float *gp2 = gradp[2];
  float *dgp0 = dgradp[0];
  float *dgp1 = dgradp[1];
  float *dgp2 = dgradp[2];

  int left = owner[face];
  int right = neighbor[face];
  float nx = snx[face];
  float ny = sny[face];
  float nz = snz[face];
  float a = area[face];
  float id = inv_dist[face];
  float dx = 0.5f * nx / id;
  float dy = 0.5f * ny / id;
  float dz = 0.5f * nz / id;
  float il = invvol[left];
  float ir = invvol[right];

  float ulf = u[left] + gu0[left] * dx + gu1[left] * dy + gu2[left] * dz;
  float urf = u[right] - gu0[right] * dx - gu1[right] * dy - gu2[right] * dz;
  float vlf = v[left] + gv0[left] * dx + gv1[left] * dy + gv2[left] * dz;
  float vrf = v[right] - gv0[right] * dx - gv1[right] * dy - gv2[right] * dz;
  float wlf = w[left] + gw0[left] * dx + gw1[left] * dy + gw2[left] * dz;
  float wrf = w[right] - gw0[right] * dx - gw1[right] * dy - gw2[right] * dz;
  float plf = p[left] + gp0[left] * dx + gp1[left] * dy + gp2[left] * dz;
  float prf = p[right] - gp0[right] * dx - gp1[right] * dy - gp2[right] * dz;
  float uf = 0.5f * (ulf + urf);
  float vf = 0.5f * (vlf + vrf);
  float wf = 0.5f * (wlf + wrf);
  float pf = 0.5f * (plf + prf);
  float un = uf * nx + vf * ny + wf * nz;
  float nuf = 0.5f * (nu[left] + nu[right]);
  float gudn = 0.5f * (gu0[left] + gu0[right]) * nx +
               0.5f * (gu1[left] + gu1[right]) * ny +
               0.5f * (gu2[left] + gu2[right]) * nz;
  float gvdn = 0.5f * (gv0[left] + gv0[right]) * nx +
               0.5f * (gv1[left] + gv1[right]) * ny +
               0.5f * (gv2[left] + gv2[right]) * nz;
  float gwdn = 0.5f * (gw0[left] + gw0[right]) * nx +
               0.5f * (gw1[left] + gw1[right]) * ny +
               0.5f * (gw2[left] + gw2[right]) * nz;
  float fluxu = a * (un * uf + pf * nx - nuf * gudn);
  float fluxv = a * (un * vf + pf * ny - nuf * gvdn);
  float fluxw = a * (un * wf + pf * nz - nuf * gwdn);

  atomicAdd(&resu[left], fluxu * il);
  atomicAdd(&resv[left], fluxv * il);
  atomicAdd(&resw[left], fluxw * il);
  atomicAdd(&resu[right], -fluxu * ir);
  atomicAdd(&resv[right], -fluxv * ir);
  atomicAdd(&resw[right], -fluxw * ir);

  float fluxu_adj = dresu[left] * il - dresu[right] * ir;
  float fluxv_adj = dresv[left] * il - dresv[right] * ir;
  float fluxw_adj = dresw[left] * il - dresw[right] * ir;

  float uf_adj = 0.0f;
  float vf_adj = 0.0f;
  float wf_adj = 0.0f;
  float pf_adj = 0.0f;
  float un_adj = 0.0f;
  float nuf_adj = 0.0f;
  float gudn_adj = 0.0f;
  float gvdn_adj = 0.0f;
  float gwdn_adj = 0.0f;

  float expr_adj = a * fluxu_adj;
  un_adj += uf * expr_adj;
  uf_adj += un * expr_adj;
  pf_adj += nx * expr_adj;
  nuf_adj -= gudn * expr_adj;
  gudn_adj -= nuf * expr_adj;

  expr_adj = a * fluxv_adj;
  un_adj += vf * expr_adj;
  vf_adj += un * expr_adj;
  pf_adj += ny * expr_adj;
  nuf_adj -= gvdn * expr_adj;
  gvdn_adj -= nuf * expr_adj;

  expr_adj = a * fluxw_adj;
  un_adj += wf * expr_adj;
  wf_adj += un * expr_adj;
  pf_adj += nz * expr_adj;
  nuf_adj -= gwdn * expr_adj;
  gwdn_adj -= nuf * expr_adj;

  uf_adj += nx * un_adj;
  vf_adj += ny * un_adj;
  wf_adj += nz * un_adj;

  atomicAdd(&dnu[left], 0.5f * nuf_adj);
  atomicAdd(&dnu[right], 0.5f * nuf_adj);

  atomicAdd(&dgu0[left], 0.5f * nx * gudn_adj);
  atomicAdd(&dgu0[right], 0.5f * nx * gudn_adj);
  atomicAdd(&dgu1[left], 0.5f * ny * gudn_adj);
  atomicAdd(&dgu1[right], 0.5f * ny * gudn_adj);
  atomicAdd(&dgu2[left], 0.5f * nz * gudn_adj);
  atomicAdd(&dgu2[right], 0.5f * nz * gudn_adj);
  atomicAdd(&dgv0[left], 0.5f * nx * gvdn_adj);
  atomicAdd(&dgv0[right], 0.5f * nx * gvdn_adj);
  atomicAdd(&dgv1[left], 0.5f * ny * gvdn_adj);
  atomicAdd(&dgv1[right], 0.5f * ny * gvdn_adj);
  atomicAdd(&dgv2[left], 0.5f * nz * gvdn_adj);
  atomicAdd(&dgv2[right], 0.5f * nz * gvdn_adj);
  atomicAdd(&dgw0[left], 0.5f * nx * gwdn_adj);
  atomicAdd(&dgw0[right], 0.5f * nx * gwdn_adj);
  atomicAdd(&dgw1[left], 0.5f * ny * gwdn_adj);
  atomicAdd(&dgw1[right], 0.5f * ny * gwdn_adj);
  atomicAdd(&dgw2[left], 0.5f * nz * gwdn_adj);
  atomicAdd(&dgw2[right], 0.5f * nz * gwdn_adj);

  float ulf_adj = 0.5f * uf_adj;
  float urf_adj = 0.5f * uf_adj;
  float vlf_adj = 0.5f * vf_adj;
  float vrf_adj = 0.5f * vf_adj;
  float wlf_adj = 0.5f * wf_adj;
  float wrf_adj = 0.5f * wf_adj;
  float plf_adj = 0.5f * pf_adj;
  float prf_adj = 0.5f * pf_adj;

  atomicAdd(&du[left], ulf_adj);
  atomicAdd(&dgu0[left], dx * ulf_adj);
  atomicAdd(&dgu1[left], dy * ulf_adj);
  atomicAdd(&dgu2[left], dz * ulf_adj);
  atomicAdd(&du[right], urf_adj);
  atomicAdd(&dgu0[right], -dx * urf_adj);
  atomicAdd(&dgu1[right], -dy * urf_adj);
  atomicAdd(&dgu2[right], -dz * urf_adj);

  atomicAdd(&dv[left], vlf_adj);
  atomicAdd(&dgv0[left], dx * vlf_adj);
  atomicAdd(&dgv1[left], dy * vlf_adj);
  atomicAdd(&dgv2[left], dz * vlf_adj);
  atomicAdd(&dv[right], vrf_adj);
  atomicAdd(&dgv0[right], -dx * vrf_adj);
  atomicAdd(&dgv1[right], -dy * vrf_adj);
  atomicAdd(&dgv2[right], -dz * vrf_adj);

  atomicAdd(&dw[left], wlf_adj);
  atomicAdd(&dgw0[left], dx * wlf_adj);
  atomicAdd(&dgw1[left], dy * wlf_adj);
  atomicAdd(&dgw2[left], dz * wlf_adj);
  atomicAdd(&dw[right], wrf_adj);
  atomicAdd(&dgw0[right], -dx * wrf_adj);
  atomicAdd(&dgw1[right], -dy * wrf_adj);
  atomicAdd(&dgw2[right], -dz * wrf_adj);

  atomicAdd(&dp[left], plf_adj);
  atomicAdd(&dgp0[left], dx * plf_adj);
  atomicAdd(&dgp1[left], dy * plf_adj);
  atomicAdd(&dgp2[left], dz * plf_adj);
  atomicAdd(&dp[right], prf_adj);
  atomicAdd(&dgp0[right], -dx * prf_adj);
  atomicAdd(&dgp1[right], -dy * prf_adj);
  atomicAdd(&dgp2[right], -dz * prf_adj);
}

static int make_rows(float ***rows, float **host_rows, int count) {
  CUDA_CHECK(cudaMalloc(rows, (size_t)count * sizeof(float *)));
  CUDA_CHECK(cudaMemcpy(*rows, host_rows, (size_t)count * sizeof(float *),
                        cudaMemcpyHostToDevice));
  return 0;
}

static int alloc_rows(float **rows, int count, int ncell) {
  for (int i = 0; i < count; ++i)
    CUDA_CHECK(cudaMalloc(&rows[i], (size_t)ncell * sizeof(float)));
  return 0;
}

static void free_rows(float **rows, int count) {
  for (int i = 0; i < count; ++i)
    cudaFree(rows[i]);
}

static int reset_gradients(float **gradv, float **gradp, int ncell, int blocks,
                           int threads) {
  int rows9 = (9 * ncell + threads - 1) / threads;
  int rows3 = (3 * ncell + threads - 1) / threads;
  zero_rows<<<rows9, threads>>>(gradv, 9, ncell);
  zero_rows<<<rows3, threads>>>(gradp, 3, ncell);
  CUDA_CHECK(cudaGetLastError());
  return 0;
}

static int reset_for_forward(float **res, int ncell, int threads) {
  int rows3 = (3 * ncell + threads - 1) / threads;
  reset_residual<<<rows3, threads>>>(res, ncell);
  CUDA_CHECK(cudaGetLastError());
  return 0;
}

static int reset_for_ad(float **dvel, float *dp, float *dnu, float **dgradv,
                        float **dgradp, float **res, float **dres, int ncell,
                        int threads) {
  int rows9 = (9 * ncell + threads - 1) / threads;
  int rows3 = (3 * ncell + threads - 1) / threads;
  int cells = (ncell + threads - 1) / threads;
  zero_rows<<<rows3, threads>>>(dvel, 3, ncell);
  zero_rows<<<rows9, threads>>>(dgradv, 9, ncell);
  zero_rows<<<rows3, threads>>>(dgradp, 3, ncell);
  zero_scalar<<<cells, threads>>>(dp, ncell);
  zero_scalar<<<cells, threads>>>(dnu, ncell);
  reset_residual_and_seed<<<rows3, threads>>>(res, dres, ncell);
  CUDA_CHECK(cudaGetLastError());
  return 0;
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
    float tolerance = fmaxf(1.0f, fabsf(hexpected[i]) * 2.5e-2f);
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

static int verify_rows(const char *prefix, float **actual, float **expected,
                       int rows, int ncell) {
  char label[96];
  for (int r = 0; r < rows; ++r) {
    snprintf(label, sizeof(label), "%s row %d", prefix, r);
    CHECK_COMPARE(label, actual[r], expected[r], ncell);
  }
  return 0;
}

static int verify_against_manual(const char *label, float **dvel, float *dp,
                                 float *dnu, float **dgradv, float **dgradp,
                                 float **res, float **dvel_ref, float *dp_ref,
                                 float *dnu_ref, float **dgradv_ref,
                                 float **dgradp_ref, float **res_ref,
                                 int ncell) {
  char name[96];
  snprintf(name, sizeof(name), "%s res", label);
  int err = verify_rows(name, res, res_ref, 3, ncell);
  if (err != 0)
    return err;
  snprintf(name, sizeof(name), "%s dvel", label);
  err = verify_rows(name, dvel, dvel_ref, 3, ncell);
  if (err != 0)
    return err;
  snprintf(name, sizeof(name), "%s dgradv", label);
  err = verify_rows(name, dgradv, dgradv_ref, 9, ncell);
  if (err != 0)
    return err;
  snprintf(name, sizeof(name), "%s dgradp", label);
  err = verify_rows(name, dgradp, dgradp_ref, 3, ncell);
  if (err != 0)
    return err;
  snprintf(name, sizeof(name), "%s dp", label);
  CHECK_COMPARE(name, dp, dp_ref, ncell);
  snprintf(name, sizeof(name), "%s dnu", label);
  CHECK_COMPARE(name, dnu, dnu_ref, ncell);
  return 0;
}

static int time_forward(float **vel, float **gradv, float **gradp, float *p,
                        float *nu, float **res, int *owner, int *neighbor,
                        float *snx, float *sny, float *snz, float *area,
                        float *inv_dist, float *invvol, int ncell, int nface,
                        int face_blocks, int threads, int reps, float *ms) {
  cudaEvent_t start;
  cudaEvent_t stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int r = 0; r < reps; ++r) {
    reset_for_forward(res, ncell, threads);
    forward_fvm3_cached<<<face_blocks, threads>>>(
        vel, gradv, gradp, p, nu, res, owner, neighbor, snx, sny, snz, area,
        inv_dist, invvol, nface);
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  CUDA_CHECK(cudaEventElapsedTime(ms, start, stop));
  *ms /= (float)reps;
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return 0;
}

#define DEFINE_TIME_AD(timer_name, kernel_name)                                \
  static int timer_name(                                                       \
      float **vel, float **dvel, float **gradv, float **dgradv, float **gradp, \
      float **dgradp, float *p, float *dp, float *nu, float *dnu, float **res, \
      float **dres, int *owner, int *neighbor, float *snx, float *sny,         \
      float *snz, float *area, float *inv_dist, float *invvol, int ncell,      \
      int nface, int face_blocks, int threads, int reps, float *ms) {          \
    cudaEvent_t start;                                                         \
    cudaEvent_t stop;                                                          \
    CUDA_CHECK(cudaEventCreate(&start));                                       \
    CUDA_CHECK(cudaEventCreate(&stop));                                        \
    CUDA_CHECK(cudaEventRecord(start));                                        \
    for (int r = 0; r < reps; ++r) {                                           \
      reset_for_ad(dvel, dp, dnu, dgradv, dgradp, res, dres, ncell, threads);  \
      kernel_name<<<face_blocks, threads>>>(                                   \
          vel, dvel, gradv, dgradv, gradp, dgradp, p, dp, nu, dnu, res, dres,  \
          owner, neighbor, snx, sny, snz, area, inv_dist, invvol, nface);      \
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

DEFINE_TIME_AD(time_manual, manual_fvm3_grad)
DEFINE_TIME_AD(time_enzyme_direct, enzyme_fvm3_direct_grad)
DEFINE_TIME_AD(time_enzyme_cached, enzyme_fvm3_cached_grad)

int main(int argc, char **argv) {
  int ncell = argc > 1 ? atoi(argv[1]) : (1 << 20);
  int nface = argc > 2 ? atoi(argv[2]) : 4 * ncell;
  int reps = argc > 3 ? atoi(argv[3]) : 50;
  if (ncell < 2 || nface < 1) {
    fprintf(stderr, "ncell must be >= 2 and nface must be >= 1\n");
    return 6;
  }

  int threads = 256;
  int cell_blocks = (ncell + threads - 1) / threads;
  int face_blocks = (nface + threads - 1) / threads;

  float *vel_rows[3] = {};
  float *dvel_rows[3] = {};
  float *dvel_ref_rows[3] = {};
  float *res_rows[3] = {};
  float *dres_rows[3] = {};
  float *res_ref_rows[3] = {};
  float *dres_ref_rows[3] = {};
  float *gradv_rows[9] = {};
  float *dgradv_rows[9] = {};
  float *dgradv_ref_rows[9] = {};
  float *gradp_rows[3] = {};
  float *dgradp_rows[3] = {};
  float *dgradp_ref_rows[3] = {};

  float **vel = nullptr, **dvel = nullptr, **dvel_ref = nullptr;
  float **res = nullptr, **dres = nullptr, **res_ref = nullptr;
  float **dres_ref = nullptr, **gradv = nullptr, **dgradv = nullptr;
  float **dgradv_ref = nullptr, **gradp = nullptr, **dgradp = nullptr;
  float **dgradp_ref = nullptr;
  float *p = nullptr, *dp = nullptr, *dp_ref = nullptr;
  float *nu = nullptr, *dnu = nullptr, *dnu_ref = nullptr;
  float *invvol = nullptr;
  float *snx = nullptr, *sny = nullptr, *snz = nullptr, *area = nullptr;
  float *inv_dist = nullptr;
  int *owner = nullptr, *neighbor = nullptr;

  int err = alloc_rows(vel_rows, 3, ncell);
  if (err != 0)
    return err;
  err = alloc_rows(dvel_rows, 3, ncell);
  if (err != 0)
    return err;
  err = alloc_rows(dvel_ref_rows, 3, ncell);
  if (err != 0)
    return err;
  err = alloc_rows(res_rows, 3, ncell);
  if (err != 0)
    return err;
  err = alloc_rows(dres_rows, 3, ncell);
  if (err != 0)
    return err;
  err = alloc_rows(res_ref_rows, 3, ncell);
  if (err != 0)
    return err;
  err = alloc_rows(dres_ref_rows, 3, ncell);
  if (err != 0)
    return err;
  err = alloc_rows(gradv_rows, 9, ncell);
  if (err != 0)
    return err;
  err = alloc_rows(dgradv_rows, 9, ncell);
  if (err != 0)
    return err;
  err = alloc_rows(dgradv_ref_rows, 9, ncell);
  if (err != 0)
    return err;
  err = alloc_rows(gradp_rows, 3, ncell);
  if (err != 0)
    return err;
  err = alloc_rows(dgradp_rows, 3, ncell);
  if (err != 0)
    return err;
  err = alloc_rows(dgradp_ref_rows, 3, ncell);
  if (err != 0)
    return err;

  CUDA_CHECK(cudaMalloc(&p, (size_t)ncell * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dp, (size_t)ncell * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dp_ref, (size_t)ncell * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&nu, (size_t)ncell * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dnu, (size_t)ncell * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dnu_ref, (size_t)ncell * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&invvol, (size_t)ncell * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&owner, (size_t)nface * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&neighbor, (size_t)nface * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&snx, (size_t)nface * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&sny, (size_t)nface * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&snz, (size_t)nface * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&area, (size_t)nface * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&inv_dist, (size_t)nface * sizeof(float)));

  make_rows(&vel, vel_rows, 3);
  make_rows(&dvel, dvel_rows, 3);
  make_rows(&dvel_ref, dvel_ref_rows, 3);
  make_rows(&res, res_rows, 3);
  make_rows(&dres, dres_rows, 3);
  make_rows(&res_ref, res_ref_rows, 3);
  make_rows(&dres_ref, dres_ref_rows, 3);
  make_rows(&gradv, gradv_rows, 9);
  make_rows(&dgradv, dgradv_rows, 9);
  make_rows(&dgradv_ref, dgradv_ref_rows, 9);
  make_rows(&gradp, gradp_rows, 3);
  make_rows(&dgradp, dgradp_rows, 3);
  make_rows(&dgradp_ref, dgradp_ref_rows, 3);

  init_primal<<<cell_blocks, threads>>>(vel, p, nu, invvol, ncell);
  init_unstructured_faces<<<face_blocks, threads>>>(
      owner, neighbor, snx, sny, snz, area, inv_dist, nface, ncell);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  reset_gradients(gradv, gradp, ncell, cell_blocks, threads);
  green_gauss_gradients<<<face_blocks, threads>>>(vel, p, gradv, gradp, owner,
                                                  neighbor, snx, sny, snz, area,
                                                  invvol, nface);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  reset_for_ad(dvel, dp, dnu, dgradv, dgradp, res, dres, ncell, threads);
  reset_for_ad(dvel_ref, dp_ref, dnu_ref, dgradv_ref, dgradp_ref, res_ref,
               dres_ref, ncell, threads);
  CUDA_CHECK(cudaDeviceSynchronize());
  enzyme_fvm3_direct_grad<<<face_blocks, threads>>>(
      vel, dvel, gradv, dgradv, gradp, dgradp, p, dp, nu, dnu, res, dres, owner,
      neighbor, snx, sny, snz, area, inv_dist, invvol, nface);
  manual_fvm3_grad<<<face_blocks, threads>>>(
      vel, dvel_ref, gradv, dgradv_ref, gradp, dgradp_ref, p, dp_ref, nu,
      dnu_ref, res_ref, dres_ref, owner, neighbor, snx, sny, snz, area,
      inv_dist, invvol, nface);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  int verify = verify_against_manual("direct", dvel_rows, dp, dnu, dgradv_rows,
                                     dgradp_rows, res_rows, dvel_ref_rows,
                                     dp_ref, dnu_ref, dgradv_ref_rows,
                                     dgradp_ref_rows, res_ref_rows, ncell);
  if (verify != 0)
    return verify;

  reset_for_ad(dvel, dp, dnu, dgradv, dgradp, res, dres, ncell, threads);
  reset_for_ad(dvel_ref, dp_ref, dnu_ref, dgradv_ref, dgradp_ref, res_ref,
               dres_ref, ncell, threads);
  CUDA_CHECK(cudaDeviceSynchronize());
  enzyme_fvm3_cached_grad<<<face_blocks, threads>>>(
      vel, dvel, gradv, dgradv, gradp, dgradp, p, dp, nu, dnu, res, dres, owner,
      neighbor, snx, sny, snz, area, inv_dist, invvol, nface);
  manual_fvm3_grad<<<face_blocks, threads>>>(
      vel, dvel_ref, gradv, dgradv_ref, gradp, dgradp_ref, p, dp_ref, nu,
      dnu_ref, res_ref, dres_ref, owner, neighbor, snx, sny, snz, area,
      inv_dist, invvol, nface);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  verify = verify_against_manual("cached", dvel_rows, dp, dnu, dgradv_rows,
                                 dgradp_rows, res_rows, dvel_ref_rows, dp_ref,
                                 dnu_ref, dgradv_ref_rows, dgradp_ref_rows,
                                 res_ref_rows, ncell);
  if (verify != 0)
    return verify;

  float forward_ms = 0.0f;
  float manual_ms = 0.0f;
  float enzyme_direct_ms = 0.0f;
  float enzyme_cached_ms = 0.0f;
  int timing = time_forward(vel, gradv, gradp, p, nu, res, owner, neighbor, snx,
                            sny, snz, area, inv_dist, invvol, ncell, nface,
                            face_blocks, threads, reps, &forward_ms);
  if (timing != 0)
    return timing;
  timing =
      time_manual(vel, dvel, gradv, dgradv, gradp, dgradp, p, dp, nu, dnu, res,
                  dres, owner, neighbor, snx, sny, snz, area, inv_dist, invvol,
                  ncell, nface, face_blocks, threads, reps, &manual_ms);
  if (timing != 0)
    return timing;
  timing = time_enzyme_direct(vel, dvel, gradv, dgradv, gradp, dgradp, p, dp,
                              nu, dnu, res, dres, owner, neighbor, snx, sny,
                              snz, area, inv_dist, invvol, ncell, nface,
                              face_blocks, threads, reps, &enzyme_direct_ms);
  if (timing != 0)
    return timing;
  timing = time_enzyme_cached(vel, dvel, gradv, dgradv, gradp, dgradp, p, dp,
                              nu, dnu, res, dres, owner, neighbor, snx, sny,
                              snz, area, inv_dist, invvol, ncell, nface,
                              face_blocks, threads, reps, &enzyme_cached_ms);
  if (timing != 0)
    return timing;

  printf("cells=%d faces=%d reps=%d forward_ms=%.6f manual_ms=%.6f "
         "enzyme_direct_ms=%.6f enzyme_cached_ms=%.6f "
         "enzyme_direct_over_forward=%.3fx enzyme_cached_over_forward=%.3fx "
         "enzyme_direct_over_manual=%.3fx enzyme_cached_over_manual=%.3fx\n",
         ncell, nface, reps, forward_ms, manual_ms, enzyme_direct_ms,
         enzyme_cached_ms, enzyme_direct_ms / forward_ms,
         enzyme_cached_ms / forward_ms, enzyme_direct_ms / manual_ms,
         enzyme_cached_ms / manual_ms);

  free_rows(vel_rows, 3);
  free_rows(dvel_rows, 3);
  free_rows(dvel_ref_rows, 3);
  free_rows(res_rows, 3);
  free_rows(dres_rows, 3);
  free_rows(res_ref_rows, 3);
  free_rows(dres_ref_rows, 3);
  free_rows(gradv_rows, 9);
  free_rows(dgradv_rows, 9);
  free_rows(dgradv_ref_rows, 9);
  free_rows(gradp_rows, 3);
  free_rows(dgradp_rows, 3);
  free_rows(dgradp_ref_rows, 3);
  cudaFree(vel);
  cudaFree(dvel);
  cudaFree(dvel_ref);
  cudaFree(res);
  cudaFree(dres);
  cudaFree(res_ref);
  cudaFree(dres_ref);
  cudaFree(gradv);
  cudaFree(dgradv);
  cudaFree(dgradv_ref);
  cudaFree(gradp);
  cudaFree(dgradp);
  cudaFree(dgradp_ref);
  cudaFree(p);
  cudaFree(dp);
  cudaFree(dp_ref);
  cudaFree(nu);
  cudaFree(dnu);
  cudaFree(dnu_ref);
  cudaFree(invvol);
  cudaFree(owner);
  cudaFree(neighbor);
  cudaFree(snx);
  cudaFree(sny);
  cudaFree(snz);
  cudaFree(area);
  cudaFree(inv_dist);
  return 0;
}
