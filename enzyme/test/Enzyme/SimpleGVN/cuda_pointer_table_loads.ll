; RUN: if [ %llvmver -lt 17 ]; then %opt < %s %newLoadEnzyme -opaque-pointers -enzyme-enable-cuda-repeated-loads=1 -passes="simple-gvn" -S | FileCheck %s --check-prefix=BASE; fi
; RUN: if [ %llvmver -ge 17 ]; then %opt < %s %newLoadEnzyme -enzyme-enable-cuda-repeated-loads=1 -passes="simple-gvn" -S | FileCheck %s --check-prefix=BASE; fi
; RUN: if [ %llvmver -lt 17 ]; then %opt < %s %newLoadEnzyme -opaque-pointers -enzyme-enable-cuda-pointer-table-loads=1 -passes="simple-gvn" -S | FileCheck %s --check-prefix=BASE; fi
; RUN: if [ %llvmver -ge 17 ]; then %opt < %s %newLoadEnzyme -enzyme-enable-cuda-pointer-table-loads=1 -passes="simple-gvn" -S | FileCheck %s --check-prefix=BASE; fi
; RUN: if [ %llvmver -lt 17 ]; then %opt < %s %newLoadEnzyme -opaque-pointers -enzyme-enable-cuda-repeated-loads=1 -enzyme-enable-cuda-pointer-table-loads=1 -passes="simple-gvn" -S | FileCheck %s --check-prefix=PTRTABLE; fi
; RUN: if [ %llvmver -ge 17 ]; then %opt < %s %newLoadEnzyme -enzyme-enable-cuda-repeated-loads=1 -enzyme-enable-cuda-pointer-table-loads=1 -passes="simple-gvn" -S | FileCheck %s --check-prefix=PTRTABLE; fi

target triple = "nvptx64-nvidia-cuda"

define void @cache_row_pointer_across_row_atomic(ptr readonly %res, i64 %left,
                                                 i64 %right, float %a,
                                                 float %b) {
entry:
  %slot0 = getelementptr inbounds ptr, ptr %res, i64 0
  %row0 = load ptr, ptr %slot0, align 8
  %leftp = getelementptr inbounds float, ptr %row0, i64 %left
  atomicrmw fadd ptr %leftp, float %a seq_cst, align 4
  %slot1 = getelementptr inbounds ptr, ptr %res, i64 0
  %row1 = load ptr, ptr %slot1, align 8
  %rightp = getelementptr inbounds float, ptr %row1, i64 %right
  atomicrmw fadd ptr %rightp, float %b seq_cst, align 4
  ret void
}

define float @cache_readonly_array_across_row_atomic(ptr readonly %res,
                                                     ptr readonly %invvol,
                                                     i64 %cell, float %a) {
entry:
  %slot = getelementptr inbounds ptr, ptr %res, i64 0
  %row = load ptr, ptr %slot, align 8
  %volp0 = getelementptr inbounds float, ptr %invvol, i64 %cell
  %v0 = load float, ptr %volp0, align 4
  %resp = getelementptr inbounds float, ptr %row, i64 %cell
  atomicrmw fadd ptr %resp, float %a seq_cst, align 4
  %volp1 = getelementptr inbounds float, ptr %invvol, i64 %cell
  %v1 = load float, ptr %volp1, align 4
  %sum = fadd float %v0, %v1
  ret float %sum
}

define void @keep_direct_table_clobber(ptr %res, i64 %left, i64 %right,
                                       float %a, float %b) {
entry:
  %slot0 = getelementptr inbounds ptr, ptr %res, i64 0
  %row0 = load ptr, ptr %slot0, align 8
  %leftp = getelementptr inbounds float, ptr %row0, i64 %left
  atomicrmw fadd ptr %leftp, float %a seq_cst, align 4
  store ptr null, ptr %slot0, align 8
  %slot1 = getelementptr inbounds ptr, ptr %res, i64 0
  %row1 = load ptr, ptr %slot1, align 8
  %rightp = getelementptr inbounds float, ptr %row1, i64 %right
  atomicrmw fadd ptr %rightp, float %b seq_cst, align 4
  ret void
}

define void @keep_unknown_clobber(ptr readonly %res, ptr %maybe_alias, i64 %left,
                                  i64 %right, float %a, float %b) {
entry:
  %slot0 = getelementptr inbounds ptr, ptr %res, i64 0
  %row0 = load ptr, ptr %slot0, align 8
  %leftp = getelementptr inbounds float, ptr %row0, i64 %left
  atomicrmw fadd ptr %leftp, float %a seq_cst, align 4
  store float 0.000000e+00, ptr %maybe_alias, align 4
  %slot1 = getelementptr inbounds ptr, ptr %res, i64 0
  %row1 = load ptr, ptr %slot1, align 8
  %rightp = getelementptr inbounds float, ptr %row1, i64 %right
  atomicrmw fadd ptr %rightp, float %b seq_cst, align 4
  ret void
}

define void @keep_nonreadonly_table(ptr %res, i64 %left, i64 %right, float %a,
                                    float %b) {
entry:
  %slot0 = getelementptr inbounds ptr, ptr %res, i64 0
  %row0 = load ptr, ptr %slot0, align 8
  %leftp = getelementptr inbounds float, ptr %row0, i64 %left
  atomicrmw fadd ptr %leftp, float %a seq_cst, align 4
  %slot1 = getelementptr inbounds ptr, ptr %res, i64 0
  %row1 = load ptr, ptr %slot1, align 8
  %rightp = getelementptr inbounds float, ptr %row1, i64 %right
  atomicrmw fadd ptr %rightp, float %b seq_cst, align 4
  ret void
}

; BASE-LABEL: define void @cache_row_pointer_across_row_atomic(
; BASE: %row0 = load ptr, ptr %slot0, align 8
; BASE: atomicrmw fadd ptr %leftp, float %a seq_cst
; BASE: %row1 = load ptr, ptr %slot1, align 8
; BASE: atomicrmw fadd ptr %rightp, float %b seq_cst
; BASE: ret void

; PTRTABLE-LABEL: define void @cache_row_pointer_across_row_atomic(
; PTRTABLE: %row0 = load ptr, ptr %slot0, align 8
; PTRTABLE: atomicrmw fadd ptr %leftp, float %a seq_cst
; PTRTABLE-NOT: %row1 = load ptr
; PTRTABLE: %rightp = getelementptr inbounds float, ptr %row0, i64 %right
; PTRTABLE: atomicrmw fadd ptr %rightp, float %b seq_cst
; PTRTABLE: ret void

; BASE-LABEL: define float @cache_readonly_array_across_row_atomic(
; BASE: %v0 = load float, ptr %volp0, align 4
; BASE: atomicrmw fadd ptr %resp, float %a seq_cst
; BASE: %v1 = load float, ptr %volp1, align 4
; BASE: %sum = fadd float %v0, %v1
; BASE: ret float %sum

; PTRTABLE-LABEL: define float @cache_readonly_array_across_row_atomic(
; PTRTABLE: %v0 = load float, ptr %volp0, align 4
; PTRTABLE: atomicrmw fadd ptr %resp, float %a seq_cst
; PTRTABLE-NOT: %v1 = load float
; PTRTABLE: %sum = fadd float %v0, %v0
; PTRTABLE: ret float %sum

; PTRTABLE-LABEL: define void @keep_direct_table_clobber(
; PTRTABLE: %row0 = load ptr, ptr %slot0, align 8
; PTRTABLE: atomicrmw fadd ptr %leftp, float %a seq_cst
; PTRTABLE: store ptr null, ptr %slot0, align 8
; PTRTABLE: %row1 = load ptr, ptr %slot1, align 8
; PTRTABLE: atomicrmw fadd ptr %rightp, float %b seq_cst
; PTRTABLE: ret void

; PTRTABLE-LABEL: define void @keep_unknown_clobber(
; PTRTABLE: %row0 = load ptr, ptr %slot0, align 8
; PTRTABLE: atomicrmw fadd ptr %leftp, float %a seq_cst
; PTRTABLE: store float 0.000000e+00, ptr %maybe_alias, align 4
; PTRTABLE: %row1 = load ptr, ptr %slot1, align 8
; PTRTABLE: atomicrmw fadd ptr %rightp, float %b seq_cst
; PTRTABLE: ret void

; PTRTABLE-LABEL: define void @keep_nonreadonly_table(
; PTRTABLE: %row0 = load ptr, ptr %slot0, align 8
; PTRTABLE: atomicrmw fadd ptr %leftp, float %a seq_cst
; PTRTABLE: %row1 = load ptr, ptr %slot1, align 8
; PTRTABLE: atomicrmw fadd ptr %rightp, float %b seq_cst
; PTRTABLE: ret void
