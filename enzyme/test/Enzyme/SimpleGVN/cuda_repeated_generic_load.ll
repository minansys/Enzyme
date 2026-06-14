; RUN: if [ %llvmver -lt 17 ]; then %opt < %s %newLoadEnzyme -opaque-pointers -passes="simple-gvn" -S | FileCheck %s --check-prefix=OFF; fi
; RUN: if [ %llvmver -ge 17 ]; then %opt < %s %newLoadEnzyme -passes="simple-gvn" -S | FileCheck %s --check-prefix=OFF; fi
; RUN: if [ %llvmver -lt 17 ]; then %opt < %s %newLoadEnzyme -opaque-pointers -enzyme-enable-cuda-repeated-loads=1 -passes="simple-gvn" -S | FileCheck %s --check-prefix=ON; fi
; RUN: if [ %llvmver -ge 17 ]; then %opt < %s %newLoadEnzyme -enzyme-enable-cuda-repeated-loads=1 -passes="simple-gvn" -S | FileCheck %s --check-prefix=ON; fi

target triple = "nvptx64-nvidia-cuda"

define float @same_generic_load(ptr %ac, i64 %i) {
entry:
  %p0 = getelementptr inbounds float, ptr %ac, i64 %i
  %a = load float, ptr %p0, align 4
  %aa = fmul float %a, %a
  %p1 = getelementptr inbounds float, ptr %ac, i64 %i
  %b = load float, ptr %p1, align 4
  %sum = fadd float %aa, %b
  ret float %sum
}

define float @same_generic_row_load(ptr %u, i64 %dim, i64 %left) {
entry:
  %row.slot0 = getelementptr inbounds ptr, ptr %u, i64 %dim
  %row0 = load ptr, ptr %row.slot0, align 8
  %left.p0 = getelementptr inbounds float, ptr %row0, i64 %left
  %ul0 = load float, ptr %left.p0, align 4
  %mul = fmul float %ul0, 2.000000e+00
  %row.slot1 = getelementptr inbounds ptr, ptr %u, i64 %dim
  %row1 = load ptr, ptr %row.slot1, align 8
  %left.p1 = getelementptr inbounds float, ptr %row1, i64 %left
  %ul1 = load float, ptr %left.p1, align 4
  %sum = fadd float %mul, %ul1
  ret float %sum
}

define float @same_generic_row_load_clobber(ptr %u, i64 %dim, i64 %left) {
entry:
  %row.slot0 = getelementptr inbounds ptr, ptr %u, i64 %dim
  %row0 = load ptr, ptr %row.slot0, align 8
  %left.p0 = getelementptr inbounds float, ptr %row0, i64 %left
  %ul0 = load float, ptr %left.p0, align 4
  store ptr null, ptr %row.slot0, align 8
  %row.slot1 = getelementptr inbounds ptr, ptr %u, i64 %dim
  %row1 = load ptr, ptr %row.slot1, align 8
  %left.p1 = getelementptr inbounds float, ptr %row1, i64 %left
  %ul1 = load float, ptr %left.p1, align 4
  %sum = fadd float %ul0, %ul1
  ret float %sum
}

; OFF-LABEL: define float @same_generic_load(
; OFF: %a = load float, ptr %p0, align 4
; OFF: %b = load float, ptr %p1, align 4
; OFF: %sum = fadd float %aa, %b
; OFF: ret float %sum

; ON-LABEL: define float @same_generic_load(
; ON: %a = load float, ptr %p0, align 4
; ON-NOT: %b = load float
; ON: %sum = fadd float %aa, %a
; ON: ret float %sum

; OFF-LABEL: define float @same_generic_row_load(
; OFF: %row0 = load ptr, ptr %row.slot0, align 8
; OFF: %ul0 = load float, ptr %left.p0, align 4
; OFF: %row1 = load ptr, ptr %row.slot1, align 8
; OFF: %ul1 = load float, ptr %left.p1, align 4
; OFF: %sum = fadd float %mul, %ul1
; OFF: ret float %sum

; ON-LABEL: define float @same_generic_row_load(
; ON: %row0 = load ptr, ptr %row.slot0, align 8
; ON: %ul0 = load float, ptr %left.p0, align 4
; ON-NOT: load
; ON: %sum = fadd float %mul, %ul0
; ON: ret float %sum

; ON-LABEL: define float @same_generic_row_load_clobber(
; ON: %row0 = load ptr, ptr %row.slot0, align 8
; ON: %ul0 = load float, ptr %left.p0, align 4
; ON: store ptr null, ptr %row.slot0, align 8
; ON: %row1 = load ptr, ptr %row.slot1, align 8
; ON: %ul1 = load float, ptr %left.p1, align 4
; ON: %sum = fadd float %ul0, %ul1
; ON: ret float %sum
