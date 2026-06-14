#include <cuda_runtime.h>

#include <float.h>
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

__device__ __host__ static inline float perturb_for_slot(int slot, int i) {
  unsigned h = hash_u32((unsigned)i ^ (0x9e3779b9u * (unsigned)(slot + 1)));
  float unit = (float)(h & 2047u) * (1.0f / 1023.5f) - 1.0f;
  float scale = 0.25f + 0.03f * (float)(slot % 7);
  return scale * unit;
}

__device__ __host__ static inline float fold_x_for_cell(int i) {
  return 0.7f + 0.0009f * (float)(i & 1023) - 0.0002f * (float)((i * 13) & 255);
}

__device__ __host__ static inline float fold_dout_for_slot(int i) {
  return -0.35f + 0.0011f * (float)((i * 19) & 511);
}

template <typename T> struct Buffer {
  T *buffer;
  int size;

  __host__ __device__ Buffer() : buffer(nullptr), size(0) {}
  __host__ __device__ Buffer(T *buffer, int size)
      : buffer(buffer), size(size) {}

  __device__ T &operator()(int index) { return buffer[index]; }
  __device__ const T &operator()(int index) const { return buffer[index]; }
};

typedef Buffer<float> BufferView;

struct RowBufferView {
  BufferView *rows;
  int nrow;

  __host__ __device__ RowBufferView() : rows(nullptr), nrow(0) {}
  __host__ __device__ RowBufferView(BufferView *rows, int nrow)
      : rows(rows), nrow(nrow) {}

  __device__ float &operator()(int row, int cell) { return rows[row](cell); }
  __device__ const float &operator()(int row, int cell) const {
    return rows[row](cell);
  }
  __device__ float *row(int row) const { return rows[row].buffer; }
};

struct UnstructuredGridView {
  Buffer<int> owner;
  Buffer<int> neighbor;
  BufferView snx;
  BufferView sny;
  BufferView snz;
  BufferView area;
  BufferView inv_dist;
  BufferView invvol;
  int ncell;
  int nface;

  __host__ __device__ UnstructuredGridView()
      : owner(), neighbor(), snx(), sny(), snz(), area(), inv_dist(), invvol(),
        ncell(0), nface(0) {}

  __host__ __device__ UnstructuredGridView(Buffer<int> owner,
                                           Buffer<int> neighbor, BufferView snx,
                                           BufferView sny, BufferView snz,
                                           BufferView area, BufferView inv_dist,
                                           BufferView invvol, int ncell,
                                           int nface)
      : owner(owner), neighbor(neighbor), snx(snx), sny(sny), snz(snz),
        area(area), inv_dist(inv_dist), invvol(invvol), ncell(ncell),
        nface(nface) {}
};

struct MomentumFieldView {
  RowBufferView vel;
  RowBufferView gradv;
  RowBufferView gradp;
  BufferView p;
  BufferView nu;
  RowBufferView res;

  __host__ __device__ MomentumFieldView()
      : vel(), gradv(), gradp(), p(), nu(), res() {}

  __host__ __device__ MomentumFieldView(RowBufferView vel, RowBufferView gradv,
                                        RowBufferView gradp, BufferView p,
                                        BufferView nu, RowBufferView res)
      : vel(vel), gradv(gradv), gradp(gradp), p(p), nu(nu), res(res) {}
};

struct InitPrimalCellOp {
  RowBufferView vel;
  BufferView p;
  BufferView nu;
  BufferView invvol;

  __host__ __device__ InitPrimalCellOp() : vel(), p(), nu(), invvol() {}
  __host__ __device__ InitPrimalCellOp(RowBufferView vel, BufferView p,
                                       BufferView nu, BufferView invvol)
      : vel(vel), p(p), nu(nu), invvol(invvol) {}

  __device__ void operator()(int cell) {
    vel(0, cell) = u_for_cell(cell);
    vel(1, cell) = v_for_cell(cell);
    vel(2, cell) = w_for_cell(cell);
    p(cell) = p_for_cell(cell);
    nu(cell) = nu_for_cell(cell);
    invvol(cell) = invvol_for_cell(cell);
  }
};

struct GreenGaussGradientFaceOp {
  RowBufferView vel;
  BufferView p;
  RowBufferView gradv;
  RowBufferView gradp;

  __host__ __device__ GreenGaussGradientFaceOp()
      : vel(), p(), gradv(), gradp() {}
  __host__ __device__ GreenGaussGradientFaceOp(RowBufferView vel, BufferView p,
                                               RowBufferView gradv,
                                               RowBufferView gradp)
      : vel(vel), p(p), gradv(gradv), gradp(gradp) {}

  __device__ void operator()(int face, const UnstructuredGridView &grid) {
    int left = grid.owner(face);
    int right = grid.neighbor(face);
    float sx = grid.snx(face) * grid.area(face);
    float sy = grid.sny(face) * grid.area(face);
    float sz = grid.snz(face) * grid.area(face);
    float il = grid.invvol(left);
    float ir = grid.invvol(right);

    float uf = 0.5f * (vel(0, left) + vel(0, right));
    float vf = 0.5f * (vel(1, left) + vel(1, right));
    float wf = 0.5f * (vel(2, left) + vel(2, right));
    float pf = 0.5f * (p(left) + p(right));

    atomicAdd(&gradv(0, left), uf * sx * il);
    atomicAdd(&gradv(1, left), uf * sy * il);
    atomicAdd(&gradv(2, left), uf * sz * il);
    atomicAdd(&gradv(3, left), vf * sx * il);
    atomicAdd(&gradv(4, left), vf * sy * il);
    atomicAdd(&gradv(5, left), vf * sz * il);
    atomicAdd(&gradv(6, left), wf * sx * il);
    atomicAdd(&gradv(7, left), wf * sy * il);
    atomicAdd(&gradv(8, left), wf * sz * il);
    atomicAdd(&gradp(0, left), pf * sx * il);
    atomicAdd(&gradp(1, left), pf * sy * il);
    atomicAdd(&gradp(2, left), pf * sz * il);

    atomicAdd(&gradv(0, right), -uf * sx * ir);
    atomicAdd(&gradv(1, right), -uf * sy * ir);
    atomicAdd(&gradv(2, right), -uf * sz * ir);
    atomicAdd(&gradv(3, right), -vf * sx * ir);
    atomicAdd(&gradv(4, right), -vf * sy * ir);
    atomicAdd(&gradv(5, right), -vf * sz * ir);
    atomicAdd(&gradv(6, right), -wf * sx * ir);
    atomicAdd(&gradv(7, right), -wf * sy * ir);
    atomicAdd(&gradv(8, right), -wf * sz * ir);
    atomicAdd(&gradp(0, right), -pf * sx * ir);
    atomicAdd(&gradp(1, right), -pf * sy * ir);
    atomicAdd(&gradp(2, right), -pf * sz * ir);
  }
};

struct MomentumFaceDirectOp {
  MomentumFieldView field;

  __host__ __device__ MomentumFaceDirectOp() : field() {}
  __host__ __device__ MomentumFaceDirectOp(MomentumFieldView field)
      : field(field) {}

  __device__ void operator()(int face, const UnstructuredGridView &grid) {
    int left = grid.owner(face);
    int right = grid.neighbor(face);
    float dx = 0.5f * grid.snx(face) / grid.inv_dist(face);
    float dy = 0.5f * grid.sny(face) / grid.inv_dist(face);
    float dz = 0.5f * grid.snz(face) / grid.inv_dist(face);

    float ulf = field.vel(0, left) + field.gradv(0, left) * dx +
                field.gradv(1, left) * dy + field.gradv(2, left) * dz;
    float urf = field.vel(0, right) - field.gradv(0, right) * dx -
                field.gradv(1, right) * dy - field.gradv(2, right) * dz;
    float vlf = field.vel(1, left) + field.gradv(3, left) * dx +
                field.gradv(4, left) * dy + field.gradv(5, left) * dz;
    float vrf = field.vel(1, right) - field.gradv(3, right) * dx -
                field.gradv(4, right) * dy - field.gradv(5, right) * dz;
    float wlf = field.vel(2, left) + field.gradv(6, left) * dx +
                field.gradv(7, left) * dy + field.gradv(8, left) * dz;
    float wrf = field.vel(2, right) - field.gradv(6, right) * dx -
                field.gradv(7, right) * dy - field.gradv(8, right) * dz;
    float plf = field.p(left) + field.gradp(0, left) * dx +
                field.gradp(1, left) * dy + field.gradp(2, left) * dz;
    float prf = field.p(right) - field.gradp(0, right) * dx -
                field.gradp(1, right) * dy - field.gradp(2, right) * dz;

    float uf = 0.5f * (ulf + urf);
    float vf = 0.5f * (vlf + vrf);
    float wf = 0.5f * (wlf + wrf);
    float pf = 0.5f * (plf + prf);
    float un = uf * grid.snx(face) + vf * grid.sny(face) + wf * grid.snz(face);
    float nuf = 0.5f * (field.nu(left) + field.nu(right));

    float gudn =
        0.5f * (field.gradv(0, left) + field.gradv(0, right)) * grid.snx(face) +
        0.5f * (field.gradv(1, left) + field.gradv(1, right)) * grid.sny(face) +
        0.5f * (field.gradv(2, left) + field.gradv(2, right)) * grid.snz(face);
    float gvdn =
        0.5f * (field.gradv(3, left) + field.gradv(3, right)) * grid.snx(face) +
        0.5f * (field.gradv(4, left) + field.gradv(4, right)) * grid.sny(face) +
        0.5f * (field.gradv(5, left) + field.gradv(5, right)) * grid.snz(face);
    float gwdn =
        0.5f * (field.gradv(6, left) + field.gradv(6, right)) * grid.snx(face) +
        0.5f * (field.gradv(7, left) + field.gradv(7, right)) * grid.sny(face) +
        0.5f * (field.gradv(8, left) + field.gradv(8, right)) * grid.snz(face);

    float fluxu =
        grid.area(face) * (un * uf + pf * grid.snx(face) - nuf * gudn);
    float fluxv =
        grid.area(face) * (un * vf + pf * grid.sny(face) - nuf * gvdn);
    float fluxw =
        grid.area(face) * (un * wf + pf * grid.snz(face) - nuf * gwdn);

    atomicAdd(&field.res(0, left), fluxu * grid.invvol(left));
    atomicAdd(&field.res(1, left), fluxv * grid.invvol(left));
    atomicAdd(&field.res(2, left), fluxw * grid.invvol(left));
    atomicAdd(&field.res(0, right), -fluxu * grid.invvol(right));
    atomicAdd(&field.res(1, right), -fluxv * grid.invvol(right));
    atomicAdd(&field.res(2, right), -fluxw * grid.invvol(right));
  }
};

struct MomentumFaceCachedOp {
  MomentumFieldView field;

  __host__ __device__ MomentumFaceCachedOp() : field() {}
  __host__ __device__ MomentumFaceCachedOp(MomentumFieldView field)
      : field(field) {}

  __device__ void operator()(int face, const UnstructuredGridView &grid) {
    float *u = field.vel.row(0);
    float *v = field.vel.row(1);
    float *w = field.vel.row(2);
    float *resu = field.res.row(0);
    float *resv = field.res.row(1);
    float *resw = field.res.row(2);
    float *gu0 = field.gradv.row(0);
    float *gu1 = field.gradv.row(1);
    float *gu2 = field.gradv.row(2);
    float *gv0 = field.gradv.row(3);
    float *gv1 = field.gradv.row(4);
    float *gv2 = field.gradv.row(5);
    float *gw0 = field.gradv.row(6);
    float *gw1 = field.gradv.row(7);
    float *gw2 = field.gradv.row(8);
    float *gp0 = field.gradp.row(0);
    float *gp1 = field.gradp.row(1);
    float *gp2 = field.gradp.row(2);
    int left = grid.owner(face);
    int right = grid.neighbor(face);
    float nx = grid.snx(face);
    float ny = grid.sny(face);
    float nz = grid.snz(face);
    float a = grid.area(face);
    float id = grid.inv_dist(face);
    float dx = 0.5f * nx / id;
    float dy = 0.5f * ny / id;
    float dz = 0.5f * nz / id;
    float il = grid.invvol(left);
    float ir = grid.invvol(right);

    float ulf = u[left] + gu0[left] * dx + gu1[left] * dy + gu2[left] * dz;
    float urf = u[right] - gu0[right] * dx - gu1[right] * dy - gu2[right] * dz;
    float vlf = v[left] + gv0[left] * dx + gv1[left] * dy + gv2[left] * dz;
    float vrf = v[right] - gv0[right] * dx - gv1[right] * dy - gv2[right] * dz;
    float wlf = w[left] + gw0[left] * dx + gw1[left] * dy + gw2[left] * dz;
    float wrf = w[right] - gw0[right] * dx - gw1[right] * dy - gw2[right] * dz;
    float plf =
        field.p(left) + gp0[left] * dx + gp1[left] * dy + gp2[left] * dz;
    float prf =
        field.p(right) - gp0[right] * dx - gp1[right] * dy - gp2[right] * dz;
    float uf = 0.5f * (ulf + urf);
    float vf = 0.5f * (vlf + vrf);
    float wf = 0.5f * (wlf + wrf);
    float pf = 0.5f * (plf + prf);
    float un = uf * nx + vf * ny + wf * nz;
    float nuf = 0.5f * (field.nu(left) + field.nu(right));
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
};

template <typename return_type, typename... T>
__device__ return_type __enzyme_autodiff(void *, T...);

extern __device__ int enzyme_dup;
extern __device__ int enzyme_const;

template <class Operation>
__device__ void cell_loop_cuda_device(int size, void *op_ptr) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= size)
    return;

  Operation *operation = static_cast<Operation *>(op_ptr);
  (*operation)(idx);
}

template <class Operation, class ConstData>
__device__ void face_loop_cuda_device(int size, void *op_ptr,
                                      const void *const_ptr) {
  int face = threadIdx.x + blockDim.x * blockIdx.x;
  if (face >= size)
    return;

  Operation *operation = static_cast<Operation *>(op_ptr);
  const ConstData *const_data = static_cast<const ConstData *>(const_ptr);
  (*operation)(face, *const_data);
}

template <class Operation>
__global__ void launch_cell_loop_forward(int size, Operation *operation_ptr) {
  cell_loop_cuda_device<Operation>(size, operation_ptr);
}

template <class Operation, class ConstData>
__global__ void launch_face_loop_forward(int size, Operation *operation_ptr,
                                         const ConstData *const_ptr) {
  face_loop_cuda_device<Operation, ConstData>(size, operation_ptr, const_ptr);
}

template <class Operation, class ConstData>
__global__ void launch_face_loop_reverse(int size, Operation *operation_ptr,
                                         Operation *d_operation_ptr,
                                         const ConstData *const_ptr) {
  __enzyme_autodiff<void>((void *)face_loop_cuda_device<Operation, ConstData>,
                          enzyme_const, size, enzyme_dup, (void *)operation_ptr,
                          (void *)d_operation_ptr, enzyme_const,
                          (void *)const_ptr);
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

__device__ void foldable_atomic_source(float *x, float *out, int cell) {
  float x0 = x[cell];
  out[2 * cell] = x0 * x0;

  // This second load intentionally remains a real load in unoptimized input IR:
  // the intervening store may alias x without noalias information.
  float x1 = x[cell];
  out[2 * cell + 1] = x1 * x1;
}

typedef void (*atomic_fold_fn)(float *, float *, int);
extern __device__ void __enzyme_autodiff(atomic_fold_fn, ...);

__global__ void init_atomic_fold(float *x, float *dx, float *dx_ref, float *out,
                                 float *out_ref, float *dout, float *dout_ref,
                                 int ncell) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < ncell) {
    x[idx] = fold_x_for_cell(idx);
    dx[idx] = 0.0f;
    dx_ref[idx] = 0.0f;
  }
  if (idx < 2 * ncell) {
    out[idx] = 0.0f;
    out_ref[idx] = 0.0f;
    float seed = fold_dout_for_slot(idx);
    dout[idx] = seed;
    dout_ref[idx] = seed;
  }
}

__global__ void enzyme_atomic_fold_grad(float *x, float *dx, float *out,
                                        float *dout, int ncell) {
  int cell = blockIdx.x * blockDim.x + threadIdx.x;
  if (cell >= ncell)
    return;

  __enzyme_autodiff(foldable_atomic_source, enzyme_dup, x, dx, enzyme_dup, out,
                    dout, enzyme_const, cell);
}

__global__ void manual_atomic_fold_grad(float *x, float *dx, float *out,
                                        float *dout, int ncell) {
  int cell = blockIdx.x * blockDim.x + threadIdx.x;
  if (cell >= ncell)
    return;

  float value = x[cell];
  out[2 * cell] = value * value;
  out[2 * cell + 1] = value * value;
  atomicAdd(&dx[cell], 2.0f * value * (dout[2 * cell] + dout[2 * cell + 1]));
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

__global__ void perturb_inputs(float **vel, float *p, float *nu, float **gradv,
                               float **gradp, int ncell, float eps) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = 17 * ncell;
  if (idx >= total)
    return;

  int slot = idx / ncell;
  int cell = idx - slot * ncell;
  float delta = eps * perturb_for_slot(slot, cell);
  if (slot < 3) {
    vel[slot][cell] += delta;
  } else if (slot == 3) {
    p[cell] += delta;
  } else if (slot == 4) {
    nu[cell] += delta;
  } else if (slot < 14) {
    gradv[slot - 5][cell] += delta;
  } else {
    gradp[slot - 14][cell] += delta;
  }
}

__global__ void objective_dot_residual(float **res, float **dres,
                                       double *partials, int ncell) {
  extern __shared__ double cache[];
  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = 3 * ncell;
  double sum = 0.0;

  if (idx < total) {
    int row = idx / ncell;
    int cell = idx - row * ncell;
    sum = (double)res[row][cell] * (double)dres[row][cell];
  }
  cache[tid] = sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride)
      cache[tid] += cache[tid + stride];
    __syncthreads();
  }

  if (tid == 0)
    partials[blockIdx.x] = cache[0];
}

__global__ void adjoint_dot_perturb(float **dvel, float *dp, float *dnu,
                                    float **dgradv, float **dgradp,
                                    double *partials, int ncell) {
  extern __shared__ double cache[];
  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = 17 * ncell;
  double sum = 0.0;

  if (idx < total) {
    int slot = idx / ncell;
    int cell = idx - slot * ncell;
    float adj = 0.0f;
    if (slot < 3) {
      adj = dvel[slot][cell];
    } else if (slot == 3) {
      adj = dp[cell];
    } else if (slot == 4) {
      adj = dnu[cell];
    } else if (slot < 14) {
      adj = dgradv[slot - 5][cell];
    } else {
      adj = dgradp[slot - 14][cell];
    }
    sum = (double)adj * (double)perturb_for_slot(slot, cell);
  }
  cache[tid] = sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride)
      cache[tid] += cache[tid + stride];
    __syncthreads();
  }

  if (tid == 0)
    partials[blockIdx.x] = cache[0];
}

static int make_rows(float ***rows, float **host_rows, int count) {
  CUDA_CHECK(cudaMalloc(rows, (size_t)count * sizeof(float *)));
  CUDA_CHECK(cudaMemcpy(*rows, host_rows, (size_t)count * sizeof(float *),
                        cudaMemcpyHostToDevice));
  return 0;
}

static int make_row_views(BufferView **views, float **host_rows, int count,
                          int ncell) {
  BufferView *host_views =
      (BufferView *)malloc((size_t)count * sizeof(BufferView));
  if (!host_views) {
    fprintf(stderr, "host allocation failed while creating row views\n");
    return 3;
  }
  for (int i = 0; i < count; ++i)
    host_views[i] = BufferView(host_rows[i], ncell);

  CUDA_CHECK(cudaMalloc(views, (size_t)count * sizeof(BufferView)));
  CUDA_CHECK(cudaMemcpy(*views, host_views, (size_t)count * sizeof(BufferView),
                        cudaMemcpyHostToDevice));
  free(host_views);
  return 0;
}

template <class T> static int copy_device_object(T **device, const T &host) {
  CUDA_CHECK(cudaMalloc(device, sizeof(T)));
  CUDA_CHECK(cudaMemcpy(*device, &host, sizeof(T), cudaMemcpyHostToDevice));
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

typedef struct {
  double max_abs;
  double max_rel;
  double sum_sq;
  long long count;
  int worst_row;
  int worst_index;
  float worst_actual;
  float worst_expected;
  char worst_label[96];
} ErrorSummary;

static void init_error_summary(ErrorSummary *summary) {
  summary->max_abs = 0.0;
  summary->max_rel = 0.0;
  summary->sum_sq = 0.0;
  summary->count = 0;
  summary->worst_row = -1;
  summary->worst_index = -1;
  summary->worst_actual = 0.0f;
  summary->worst_expected = 0.0f;
  summary->worst_label[0] = '\0';
}

static double error_rmse(const ErrorSummary *summary) {
  if (summary->count == 0)
    return 0.0;
  return sqrt(summary->sum_sq / (double)summary->count);
}

static void print_error_summary(const char *label,
                                const ErrorSummary *summary) {
  printf("check=%s max_abs=%.6e max_rel=%.6e rmse=%.6e worst=%s[%d,%d] "
         "actual=%.8e expected=%.8e\n",
         label, summary->max_abs, summary->max_rel, error_rmse(summary),
         summary->worst_label[0] ? summary->worst_label : "none",
         summary->worst_row, summary->worst_index, summary->worst_actual,
         summary->worst_expected);
}

static int compare_device_arrays(const char *label, int row, float *actual,
                                 float *expected, int n, double abs_tol,
                                 double rel_tol, ErrorSummary *summary) {
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

  int bad = 0;
  for (int i = 0; i < n; ++i) {
    double diff = fabs((double)hactual[i] - (double)hexpected[i]);
    double denom = fmax(1.0e-12, fabs((double)hexpected[i]));
    double rel = diff / denom;
    double tolerance = abs_tol + rel_tol * fabs((double)hexpected[i]);
    summary->sum_sq += diff * diff;
    summary->count += 1;
    if (diff > summary->max_abs) {
      summary->max_abs = diff;
      summary->max_rel = rel;
      summary->worst_row = row;
      summary->worst_index = i;
      summary->worst_actual = hactual[i];
      summary->worst_expected = hexpected[i];
      snprintf(summary->worst_label, sizeof(summary->worst_label), "%s", label);
    }
    if (diff > tolerance) {
      if (bad == 0) {
        fprintf(stderr,
                "%s mismatch row=%d index=%d actual=%.8e expected=%.8e "
                "abs=%.6e rel=%.6e tolerance=%.6e\n",
                label, row, i, hactual[i], hexpected[i], diff, rel, tolerance);
      }
      bad += 1;
    }
  }

  free(hactual);
  free(hexpected);
  if (bad != 0) {
    fprintf(stderr, "%s had %d values outside tolerance\n", label, bad);
    return 4;
  }
  return 0;
}

static int verify_rows(const char *prefix, float **actual, float **expected,
                       int rows, int ncell, double abs_tol, double rel_tol,
                       ErrorSummary *summary) {
  char label[96];
  for (int r = 0; r < rows; ++r) {
    snprintf(label, sizeof(label), "%s row %d", prefix, r);
    int err = compare_device_arrays(label, r, actual[r], expected[r], ncell,
                                    abs_tol, rel_tol, summary);
    if (err != 0)
      return err;
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
  ErrorSummary summary;
  init_error_summary(&summary);
  double abs_tol = 3.0e-2;
  double rel_tol = 3.0e-2;

  snprintf(name, sizeof(name), "%s res", label);
  int err =
      verify_rows(name, res, res_ref, 3, ncell, abs_tol, rel_tol, &summary);
  if (err != 0)
    return err;
  snprintf(name, sizeof(name), "%s dvel", label);
  err = verify_rows(name, dvel, dvel_ref, 3, ncell, abs_tol, rel_tol, &summary);
  if (err != 0)
    return err;
  snprintf(name, sizeof(name), "%s dgradv", label);
  err = verify_rows(name, dgradv, dgradv_ref, 9, ncell, abs_tol, rel_tol,
                    &summary);
  if (err != 0)
    return err;
  snprintf(name, sizeof(name), "%s dgradp", label);
  err = verify_rows(name, dgradp, dgradp_ref, 3, ncell, abs_tol, rel_tol,
                    &summary);
  if (err != 0)
    return err;
  snprintf(name, sizeof(name), "%s dp", label);
  err = compare_device_arrays(name, 0, dp, dp_ref, ncell, abs_tol, rel_tol,
                              &summary);
  if (err != 0)
    return err;
  snprintf(name, sizeof(name), "%s dnu", label);
  err = compare_device_arrays(name, 0, dnu, dnu_ref, ncell, abs_tol, rel_tol,
                              &summary);
  if (err != 0)
    return err;
  print_error_summary(label, &summary);
  return 0;
}

static int verify_forward_match(const char *label, float **actual,
                                float **expected, int ncell) {
  ErrorSummary summary;
  init_error_summary(&summary);
  int err =
      verify_rows(label, actual, expected, 3, ncell, 3.0e-3, 1.0e-3, &summary);
  if (err != 0)
    return err;
  print_error_summary(label, &summary);
  return 0;
}

static int verify_atomic_fold_case(int ncell, int threads) {
  float *x = nullptr;
  float *dx = nullptr;
  float *dx_ref = nullptr;
  float *out = nullptr;
  float *out_ref = nullptr;
  float *dout = nullptr;
  float *dout_ref = nullptr;
  int blocks = (ncell + threads - 1) / threads;
  int out_blocks = (2 * ncell + threads - 1) / threads;

  CUDA_CHECK(cudaMalloc(&x, (size_t)ncell * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dx, (size_t)ncell * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dx_ref, (size_t)ncell * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&out, (size_t)2 * ncell * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&out_ref, (size_t)2 * ncell * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dout, (size_t)2 * ncell * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dout_ref, (size_t)2 * ncell * sizeof(float)));

  init_atomic_fold<<<out_blocks, threads>>>(x, dx, dx_ref, out, out_ref, dout,
                                            dout_ref, ncell);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  enzyme_atomic_fold_grad<<<blocks, threads>>>(x, dx, out, dout, ncell);
  manual_atomic_fold_grad<<<blocks, threads>>>(x, dx_ref, out_ref, dout_ref,
                                               ncell);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  ErrorSummary summary;
  init_error_summary(&summary);
  int err = compare_device_arrays("atomic_fold out", 0, out, out_ref, 2 * ncell,
                                  1.0e-5, 1.0e-5, &summary);
  if (err == 0)
    err = compare_device_arrays("atomic_fold dx", 0, dx, dx_ref, ncell, 1.0e-5,
                                1.0e-5, &summary);
  if (err == 0)
    print_error_summary("atomic_fold", &summary);

  cudaFree(x);
  cudaFree(dx);
  cudaFree(dx_ref);
  cudaFree(out);
  cudaFree(out_ref);
  cudaFree(dout);
  cudaFree(dout_ref);
  return err;
}

static int sum_partials(double *partials, int blocks, double *sum) {
  double *host = (double *)malloc((size_t)blocks * sizeof(double));
  if (!host) {
    fprintf(stderr, "host allocation failed while reducing partial sums\n");
    return 3;
  }
  CUDA_CHECK(cudaMemcpy(host, partials, (size_t)blocks * sizeof(double),
                        cudaMemcpyDeviceToHost));
  double total = 0.0;
  for (int i = 0; i < blocks; ++i)
    total += host[i];
  free(host);
  *sum = total;
  return 0;
}

static int residual_objective(MomentumFaceCachedOp *operation,
                              const UnstructuredGridView *grid, float **res,
                              float **dres, int ncell, int nface,
                              int face_blocks, int threads, double *partials,
                              double *objective) {
  int obj_blocks = (3 * ncell + threads - 1) / threads;
  reset_for_forward(res, ncell, threads);
  launch_face_loop_forward<MomentumFaceCachedOp, UnstructuredGridView>
      <<<face_blocks, threads>>>(nface, operation, grid);
  CUDA_CHECK(cudaGetLastError());
  objective_dot_residual<<<obj_blocks, threads, threads * sizeof(double)>>>(
      res, dres, partials, ncell);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  return sum_partials(partials, obj_blocks, objective);
}

static int adjoint_directional_dot(float **dvel, float *dp, float *dnu,
                                   float **dgradv, float **dgradp, int ncell,
                                   int threads, double *partials, double *dot) {
  int dot_blocks = (17 * ncell + threads - 1) / threads;
  adjoint_dot_perturb<<<dot_blocks, threads, threads * sizeof(double)>>>(
      dvel, dp, dnu, dgradv, dgradp, partials, ncell);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  return sum_partials(partials, dot_blocks, dot);
}

static int finite_difference_check(float **vel, float **gradv, float **gradp,
                                   float *p, float *nu, float **dvel, float *dp,
                                   float *dnu, float **dgradv, float **dgradp,
                                   float **res, float **dres,
                                   MomentumFaceCachedOp *operation,
                                   const UnstructuredGridView *grid, int ncell,
                                   int nface, int face_blocks, int threads) {
  int max_blocks = (17 * ncell + threads - 1) / threads;
  int perturb_blocks = max_blocks;
  int seed_blocks = (3 * ncell + threads - 1) / threads;
  double *partials = nullptr;
  CUDA_CHECK(cudaMalloc(&partials, (size_t)max_blocks * sizeof(double)));

  reset_residual_and_seed<<<seed_blocks, threads>>>(res, dres, ncell);
  CUDA_CHECK(cudaGetLastError());

  double adjoint_dot = 0.0;
  int err = adjoint_directional_dot(dvel, dp, dnu, dgradv, dgradp, ncell,
                                    threads, partials, &adjoint_dot);
  if (err != 0) {
    cudaFree(partials);
    return err;
  }

  const float eps = 1.0e-3f;
  double objective_plus = 0.0;
  double objective_minus = 0.0;

  perturb_inputs<<<perturb_blocks, threads>>>(vel, p, nu, gradv, gradp, ncell,
                                              eps);
  CUDA_CHECK(cudaGetLastError());
  err = residual_objective(operation, grid, res, dres, ncell, nface,
                           face_blocks, threads, partials, &objective_plus);
  if (err != 0) {
    cudaFree(partials);
    return err;
  }

  perturb_inputs<<<perturb_blocks, threads>>>(vel, p, nu, gradv, gradp, ncell,
                                              -2.0f * eps);
  CUDA_CHECK(cudaGetLastError());
  err = residual_objective(operation, grid, res, dres, ncell, nface,
                           face_blocks, threads, partials, &objective_minus);
  if (err != 0) {
    cudaFree(partials);
    return err;
  }

  perturb_inputs<<<perturb_blocks, threads>>>(vel, p, nu, gradv, gradp, ncell,
                                              eps);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaFree(partials);

  double finite_diff = (objective_plus - objective_minus) / (2.0 * eps);
  double abs_err = fabs(finite_diff - adjoint_dot);
  double rel_err = abs_err / fmax(1.0, fabs(finite_diff));
  printf("check=finite_difference objective_plus=%.12e "
         "objective_minus=%.12e finite_diff=%.12e adjoint_dot=%.12e "
         "abs=%.6e rel=%.6e\n",
         objective_plus, objective_minus, finite_diff, adjoint_dot, abs_err,
         rel_err);

  double tolerance = 2.0e-1 + 5.0e-2 * fabs(finite_diff);
  if (abs_err > tolerance) {
    fprintf(stderr, "finite-difference check failed: abs=%.6e tolerance=%.6e\n",
            abs_err, tolerance);
    return 5;
  }
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

typedef int (*ad_timer_fn)(float **, float **, float **, float **, float **,
                           float **, float *, float *, float *, float *,
                           float **, float **, int *, int *, float *, float *,
                           float *, float *, float *, float *, int, int, int,
                           int, int, float *);

static int best_ad_time(ad_timer_fn timer, float **vel, float **dvel,
                        float **gradv, float **dgradv, float **gradp,
                        float **dgradp, float *p, float *dp, float *nu,
                        float *dnu, float **res, float **dres, int *owner,
                        int *neighbor, float *snx, float *sny, float *snz,
                        float *area, float *inv_dist, float *invvol, int ncell,
                        int nface, int face_blocks, int threads, int reps,
                        int rounds, float *best, float *mean) {
  *best = FLT_MAX;
  *mean = 0.0f;
  for (int r = 0; r < rounds; ++r) {
    float sample = 0.0f;
    int err = timer(vel, dvel, gradv, dgradv, gradp, dgradp, p, dp, nu, dnu,
                    res, dres, owner, neighbor, snx, sny, snz, area, inv_dist,
                    invvol, ncell, nface, face_blocks, threads, reps, &sample);
    if (err != 0)
      return err;
    if (sample < *best)
      *best = sample;
    *mean += sample;
  }
  *mean /= (float)rounds;
  return 0;
}

template <class Operation>
static int time_struct_forward(Operation *operation,
                               const UnstructuredGridView *grid, float **res,
                               int ncell, int nface, int face_blocks,
                               int threads, int reps, float *ms) {
  cudaEvent_t start;
  cudaEvent_t stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int r = 0; r < reps; ++r) {
    reset_for_forward(res, ncell, threads);
    launch_face_loop_forward<Operation, UnstructuredGridView>
        <<<face_blocks, threads>>>(nface, operation, grid);
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

template <class Operation>
static int time_struct_ad(Operation *operation, Operation *d_operation,
                          const UnstructuredGridView *grid, float **dvel,
                          float *dp, float *dnu, float **dgradv, float **dgradp,
                          float **res, float **dres, int ncell, int nface,
                          int face_blocks, int threads, int reps, float *ms) {
  cudaEvent_t start;
  cudaEvent_t stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int r = 0; r < reps; ++r) {
    reset_for_ad(dvel, dp, dnu, dgradv, dgradp, res, dres, ncell, threads);
    launch_face_loop_reverse<Operation, UnstructuredGridView>
        <<<face_blocks, threads>>>(nface, operation, d_operation, grid);
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

template <class Operation>
static int best_struct_forward_time(Operation *operation,
                                    const UnstructuredGridView *grid,
                                    float **res, int ncell, int nface,
                                    int face_blocks, int threads, int reps,
                                    int rounds, float *best, float *mean) {
  *best = FLT_MAX;
  *mean = 0.0f;
  for (int r = 0; r < rounds; ++r) {
    float sample = 0.0f;
    int err = time_struct_forward(operation, grid, res, ncell, nface,
                                  face_blocks, threads, reps, &sample);
    if (err != 0)
      return err;
    if (sample < *best)
      *best = sample;
    *mean += sample;
  }
  *mean /= (float)rounds;
  return 0;
}

template <class Operation>
static int best_struct_ad_time(Operation *operation, Operation *d_operation,
                               const UnstructuredGridView *grid, float **dvel,
                               float *dp, float *dnu, float **dgradv,
                               float **dgradp, float **res, float **dres,
                               int ncell, int nface, int face_blocks,
                               int threads, int reps, int rounds, float *best,
                               float *mean) {
  *best = FLT_MAX;
  *mean = 0.0f;
  for (int r = 0; r < rounds; ++r) {
    float sample = 0.0f;
    int err = time_struct_ad(operation, d_operation, grid, dvel, dp, dnu,
                             dgradv, dgradp, res, dres, ncell, nface,
                             face_blocks, threads, reps, &sample);
    if (err != 0)
      return err;
    if (sample < *best)
      *best = sample;
    *mean += sample;
  }
  *mean /= (float)rounds;
  return 0;
}

int main(int argc, char **argv) {
  int ncell = argc > 1 ? atoi(argv[1]) : (1 << 20);
  int nface = argc > 2 ? atoi(argv[2]) : 4 * ncell;
  int reps = argc > 3 ? atoi(argv[3]) : 50;
  int rounds = argc > 4 ? atoi(argv[4]) : 5;
  if (ncell < 2 || nface < 1) {
    fprintf(stderr, "ncell must be >= 2 and nface must be >= 1\n");
    return 6;
  }
  if (reps < 1 || rounds < 1) {
    fprintf(stderr, "reps and rounds must both be >= 1\n");
    return 6;
  }

  int threads = 256;
  int cell_blocks = (ncell + threads - 1) / threads;
  int face_blocks = (nface + threads - 1) / threads;

  int atomic_verify = verify_atomic_fold_case(ncell, threads);
  if (atomic_verify != 0)
    return atomic_verify;

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
  BufferView *vel_view = nullptr, *dvel_view = nullptr, *res_view = nullptr;
  BufferView *dres_view = nullptr, *res_ref_view = nullptr;
  BufferView *gradv_view = nullptr, *dgradv_view = nullptr;
  BufferView *gradp_view = nullptr, *dgradp_view = nullptr;
  float *p = nullptr, *dp = nullptr, *dp_ref = nullptr;
  float *nu = nullptr, *dnu = nullptr, *dnu_ref = nullptr;
  float *invvol = nullptr;
  float *snx = nullptr, *sny = nullptr, *snz = nullptr, *area = nullptr;
  float *inv_dist = nullptr;
  int *owner = nullptr, *neighbor = nullptr;
  UnstructuredGridView *grid_view = nullptr;
  InitPrimalCellOp *init_op = nullptr;
  GreenGaussGradientFaceOp *green_gauss_op = nullptr;
  MomentumFaceDirectOp *direct_op = nullptr, *d_direct_op = nullptr;
  MomentumFaceCachedOp *cached_op = nullptr, *d_cached_op = nullptr;
  MomentumFaceCachedOp *cached_ref_op = nullptr;

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

  err = make_rows(&vel, vel_rows, 3);
  if (err != 0)
    return err;
  err = make_rows(&dvel, dvel_rows, 3);
  if (err != 0)
    return err;
  err = make_rows(&dvel_ref, dvel_ref_rows, 3);
  if (err != 0)
    return err;
  err = make_rows(&res, res_rows, 3);
  if (err != 0)
    return err;
  err = make_rows(&dres, dres_rows, 3);
  if (err != 0)
    return err;
  err = make_rows(&res_ref, res_ref_rows, 3);
  if (err != 0)
    return err;
  err = make_rows(&dres_ref, dres_ref_rows, 3);
  if (err != 0)
    return err;
  err = make_rows(&gradv, gradv_rows, 9);
  if (err != 0)
    return err;
  err = make_rows(&dgradv, dgradv_rows, 9);
  if (err != 0)
    return err;
  err = make_rows(&dgradv_ref, dgradv_ref_rows, 9);
  if (err != 0)
    return err;
  err = make_rows(&gradp, gradp_rows, 3);
  if (err != 0)
    return err;
  err = make_rows(&dgradp, dgradp_rows, 3);
  if (err != 0)
    return err;
  err = make_rows(&dgradp_ref, dgradp_ref_rows, 3);
  if (err != 0)
    return err;

  err = make_row_views(&vel_view, vel_rows, 3, ncell);
  if (err != 0)
    return err;
  err = make_row_views(&dvel_view, dvel_rows, 3, ncell);
  if (err != 0)
    return err;
  err = make_row_views(&res_view, res_rows, 3, ncell);
  if (err != 0)
    return err;
  err = make_row_views(&dres_view, dres_rows, 3, ncell);
  if (err != 0)
    return err;
  err = make_row_views(&res_ref_view, res_ref_rows, 3, ncell);
  if (err != 0)
    return err;
  err = make_row_views(&gradv_view, gradv_rows, 9, ncell);
  if (err != 0)
    return err;
  err = make_row_views(&dgradv_view, dgradv_rows, 9, ncell);
  if (err != 0)
    return err;
  err = make_row_views(&gradp_view, gradp_rows, 3, ncell);
  if (err != 0)
    return err;
  err = make_row_views(&dgradp_view, dgradp_rows, 3, ncell);
  if (err != 0)
    return err;

  UnstructuredGridView host_grid(
      Buffer<int>(owner, nface), Buffer<int>(neighbor, nface),
      BufferView(snx, nface), BufferView(sny, nface), BufferView(snz, nface),
      BufferView(area, nface), BufferView(inv_dist, nface),
      BufferView(invvol, ncell), ncell, nface);
  err = copy_device_object(&grid_view, host_grid);
  if (err != 0)
    return err;

  InitPrimalCellOp host_init(RowBufferView(vel_view, 3), BufferView(p, ncell),
                             BufferView(nu, ncell), BufferView(invvol, ncell));
  err = copy_device_object(&init_op, host_init);
  if (err != 0)
    return err;

  GreenGaussGradientFaceOp host_green_gauss(
      RowBufferView(vel_view, 3), BufferView(p, ncell),
      RowBufferView(gradv_view, 9), RowBufferView(gradp_view, 3));
  err = copy_device_object(&green_gauss_op, host_green_gauss);
  if (err != 0)
    return err;

  MomentumFieldView host_field(
      RowBufferView(vel_view, 3), RowBufferView(gradv_view, 9),
      RowBufferView(gradp_view, 3), BufferView(p, ncell), BufferView(nu, ncell),
      RowBufferView(res_view, 3));
  MomentumFieldView host_field_ref(
      RowBufferView(vel_view, 3), RowBufferView(gradv_view, 9),
      RowBufferView(gradp_view, 3), BufferView(p, ncell), BufferView(nu, ncell),
      RowBufferView(res_ref_view, 3));
  MomentumFieldView host_dfield(
      RowBufferView(dvel_view, 3), RowBufferView(dgradv_view, 9),
      RowBufferView(dgradp_view, 3), BufferView(dp, ncell),
      BufferView(dnu, ncell), RowBufferView(dres_view, 3));

  err = copy_device_object(&direct_op, MomentumFaceDirectOp(host_field));
  if (err != 0)
    return err;
  err = copy_device_object(&d_direct_op, MomentumFaceDirectOp(host_dfield));
  if (err != 0)
    return err;
  err = copy_device_object(&cached_op, MomentumFaceCachedOp(host_field));
  if (err != 0)
    return err;
  err = copy_device_object(&d_cached_op, MomentumFaceCachedOp(host_dfield));
  if (err != 0)
    return err;
  err =
      copy_device_object(&cached_ref_op, MomentumFaceCachedOp(host_field_ref));
  if (err != 0)
    return err;

  launch_cell_loop_forward<InitPrimalCellOp>
      <<<cell_blocks, threads>>>(ncell, init_op);
  init_unstructured_faces<<<face_blocks, threads>>>(
      owner, neighbor, snx, sny, snz, area, inv_dist, nface, ncell);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  reset_gradients(gradv, gradp, ncell, cell_blocks, threads);
  launch_face_loop_forward<GreenGaussGradientFaceOp, UnstructuredGridView>
      <<<face_blocks, threads>>>(nface, green_gauss_op, grid_view);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  reset_for_forward(res, ncell, threads);
  reset_for_forward(res_ref, ncell, threads);
  launch_face_loop_forward<MomentumFaceDirectOp, UnstructuredGridView>
      <<<face_blocks, threads>>>(nface, direct_op, grid_view);
  launch_face_loop_forward<MomentumFaceCachedOp, UnstructuredGridView>
      <<<face_blocks, threads>>>(nface, cached_ref_op, grid_view);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  int verify = verify_forward_match("forward_struct_direct_vs_struct_cached",
                                    res_rows, res_ref_rows, ncell);
  if (verify != 0)
    return verify;

  reset_for_ad(dvel, dp, dnu, dgradv, dgradp, res, dres, ncell, threads);
  reset_for_ad(dvel_ref, dp_ref, dnu_ref, dgradv_ref, dgradp_ref, res_ref,
               dres_ref, ncell, threads);
  CUDA_CHECK(cudaDeviceSynchronize());
  launch_face_loop_reverse<MomentumFaceDirectOp, UnstructuredGridView>
      <<<face_blocks, threads>>>(nface, direct_op, d_direct_op, grid_view);
  manual_fvm3_grad<<<face_blocks, threads>>>(
      vel, dvel_ref, gradv, dgradv_ref, gradp, dgradp_ref, p, dp_ref, nu,
      dnu_ref, res_ref, dres_ref, owner, neighbor, snx, sny, snz, area,
      inv_dist, invvol, nface);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  verify = verify_against_manual("direct", dvel_rows, dp, dnu, dgradv_rows,
                                 dgradp_rows, res_rows, dvel_ref_rows, dp_ref,
                                 dnu_ref, dgradv_ref_rows, dgradp_ref_rows,
                                 res_ref_rows, ncell);
  if (verify != 0)
    return verify;

  reset_for_ad(dvel, dp, dnu, dgradv, dgradp, res, dres, ncell, threads);
  reset_for_ad(dvel_ref, dp_ref, dnu_ref, dgradv_ref, dgradp_ref, res_ref,
               dres_ref, ncell, threads);
  CUDA_CHECK(cudaDeviceSynchronize());
  launch_face_loop_reverse<MomentumFaceCachedOp, UnstructuredGridView>
      <<<face_blocks, threads>>>(nface, cached_op, d_cached_op, grid_view);
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

  verify = finite_difference_check(vel, gradv, gradp, p, nu, dvel_ref, dp_ref,
                                   dnu_ref, dgradv_ref, dgradp_ref, res_ref,
                                   dres_ref, cached_ref_op, grid_view, ncell,
                                   nface, face_blocks, threads);
  if (verify != 0)
    return verify;

  launch_cell_loop_forward<InitPrimalCellOp>
      <<<cell_blocks, threads>>>(ncell, init_op);
  reset_gradients(gradv, gradp, ncell, cell_blocks, threads);
  launch_face_loop_forward<GreenGaussGradientFaceOp, UnstructuredGridView>
      <<<face_blocks, threads>>>(nface, green_gauss_op, grid_view);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  float forward_direct_best_ms = 0.0f;
  float forward_direct_mean_ms = 0.0f;
  float forward_cached_best_ms = 0.0f;
  float forward_cached_mean_ms = 0.0f;
  float manual_best_ms = 0.0f;
  float manual_mean_ms = 0.0f;
  float enzyme_direct_best_ms = 0.0f;
  float enzyme_direct_mean_ms = 0.0f;
  float enzyme_cached_best_ms = 0.0f;
  float enzyme_cached_mean_ms = 0.0f;
  int timing = best_struct_forward_time(
      direct_op, grid_view, res, ncell, nface, face_blocks, threads, reps,
      rounds, &forward_direct_best_ms, &forward_direct_mean_ms);
  if (timing != 0)
    return timing;
  timing = best_struct_forward_time(
      cached_op, grid_view, res, ncell, nface, face_blocks, threads, reps,
      rounds, &forward_cached_best_ms, &forward_cached_mean_ms);
  if (timing != 0)
    return timing;
  timing =
      best_ad_time(time_manual, vel, dvel, gradv, dgradv, gradp, dgradp, p, dp,
                   nu, dnu, res, dres, owner, neighbor, snx, sny, snz, area,
                   inv_dist, invvol, ncell, nface, face_blocks, threads, reps,
                   rounds, &manual_best_ms, &manual_mean_ms);
  if (timing != 0)
    return timing;
  timing = best_struct_ad_time(direct_op, d_direct_op, grid_view, dvel, dp, dnu,
                               dgradv, dgradp, res, dres, ncell, nface,
                               face_blocks, threads, reps, rounds,
                               &enzyme_direct_best_ms, &enzyme_direct_mean_ms);
  if (timing != 0)
    return timing;
  timing = best_struct_ad_time(cached_op, d_cached_op, grid_view, dvel, dp, dnu,
                               dgradv, dgradp, res, dres, ncell, nface,
                               face_blocks, threads, reps, rounds,
                               &enzyme_cached_best_ms, &enzyme_cached_mean_ms);
  if (timing != 0)
    return timing;

  printf("cells=%d faces=%d reps=%d rounds=%d "
         "forward_direct_best_ms=%.6f forward_direct_mean_ms=%.6f "
         "forward_cached_best_ms=%.6f forward_cached_mean_ms=%.6f "
         "manual_best_ms=%.6f manual_mean_ms=%.6f "
         "enzyme_direct_best_ms=%.6f enzyme_direct_mean_ms=%.6f "
         "enzyme_cached_best_ms=%.6f enzyme_cached_mean_ms=%.6f "
         "enzyme_direct_over_forward_cached=%.3fx "
         "enzyme_cached_over_forward_cached=%.3fx "
         "enzyme_direct_over_manual=%.3fx enzyme_cached_over_manual=%.3fx "
         "enzyme_direct_over_cached=%.3fx\n",
         ncell, nface, reps, rounds, forward_direct_best_ms,
         forward_direct_mean_ms, forward_cached_best_ms, forward_cached_mean_ms,
         manual_best_ms, manual_mean_ms, enzyme_direct_best_ms,
         enzyme_direct_mean_ms, enzyme_cached_best_ms, enzyme_cached_mean_ms,
         enzyme_direct_best_ms / forward_cached_best_ms,
         enzyme_cached_best_ms / forward_cached_best_ms,
         enzyme_direct_best_ms / manual_best_ms,
         enzyme_cached_best_ms / manual_best_ms,
         enzyme_direct_best_ms / enzyme_cached_best_ms);

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
  cudaFree(vel_view);
  cudaFree(dvel_view);
  cudaFree(res_view);
  cudaFree(dres_view);
  cudaFree(res_ref_view);
  cudaFree(gradv_view);
  cudaFree(dgradv_view);
  cudaFree(gradp_view);
  cudaFree(dgradp_view);
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
  cudaFree(grid_view);
  cudaFree(init_op);
  cudaFree(green_gauss_op);
  cudaFree(direct_op);
  cudaFree(d_direct_op);
  cudaFree(cached_op);
  cudaFree(d_cached_op);
  cudaFree(cached_ref_op);
  return 0;
}
