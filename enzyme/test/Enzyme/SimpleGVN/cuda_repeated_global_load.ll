; RUN: if [ %llvmver -lt 17 ]; then %opt < %s %newLoadEnzyme -opaque-pointers -passes="simple-gvn" -S | FileCheck %s --check-prefix=OFF; fi
; RUN: if [ %llvmver -ge 17 ]; then %opt < %s %newLoadEnzyme -passes="simple-gvn" -S | FileCheck %s --check-prefix=OFF; fi
; RUN: if [ %llvmver -lt 17 ]; then %opt < %s %newLoadEnzyme -opaque-pointers -enzyme-enable-cuda-repeated-loads=1 -passes="simple-gvn" -S | FileCheck %s --check-prefix=ON; fi
; RUN: if [ %llvmver -ge 17 ]; then %opt < %s %newLoadEnzyme -enzyme-enable-cuda-repeated-loads=1 -passes="simple-gvn" -S | FileCheck %s --check-prefix=ON; fi

target triple = "nvptx64-nvidia-cuda"

define float @same_global_load(ptr addrspace(1) %ac, i64 %i) {
entry:
  %p0 = getelementptr inbounds float, ptr addrspace(1) %ac, i64 %i
  %a = load float, ptr addrspace(1) %p0, align 4
  %aa = fmul float %a, %a
  %p1 = getelementptr inbounds float, ptr addrspace(1) %ac, i64 %i
  %b = load float, ptr addrspace(1) %p1, align 4
  %sum = fadd float %aa, %b
  ret float %sum
}

define float @same_global_load_noalias_store(ptr addrspace(1) noalias %ac,
                                             ptr addrspace(1) noalias %out,
                                             i64 %i) {
entry:
  %p0 = getelementptr inbounds float, ptr addrspace(1) %ac, i64 %i
  %a = load float, ptr addrspace(1) %p0, align 4
  %op = getelementptr inbounds float, ptr addrspace(1) %out, i64 %i
  store float 0.000000e+00, ptr addrspace(1) %op, align 4
  %p1 = getelementptr inbounds float, ptr addrspace(1) %ac, i64 %i
  %b = load float, ptr addrspace(1) %p1, align 4
  %sum = fadd float %a, %b
  ret float %sum
}

define float @same_global_load_clobber(ptr addrspace(1) %ac, i64 %i) {
entry:
  %p = getelementptr inbounds float, ptr addrspace(1) %ac, i64 %i
  %a = load float, ptr addrspace(1) %p, align 4
  store float 1.000000e+00, ptr addrspace(1) %p, align 4
  %b = load float, ptr addrspace(1) %p, align 4
  %sum = fadd float %a, %b
  ret float %sum
}

declare void @unknown()

define float @same_global_load_call(ptr addrspace(1) %ac, i64 %i) {
entry:
  %p = getelementptr inbounds float, ptr addrspace(1) %ac, i64 %i
  %a = load float, ptr addrspace(1) %p, align 4
  call void @unknown()
  %b = load float, ptr addrspace(1) %p, align 4
  %sum = fadd float %a, %b
  ret float %sum
}

; OFF-LABEL: define float @same_global_load(
; OFF: %a = load float, ptr addrspace(1) %p0, align 4
; OFF: %b = load float, ptr addrspace(1) %p1, align 4
; OFF: %sum = fadd float %aa, %b
; OFF: ret float %sum

; ON-LABEL: define float @same_global_load(
; ON: %a = load float, ptr addrspace(1) %p0, align 4
; ON-NOT: %b = load float
; ON: %sum = fadd float %aa, %a
; ON: ret float %sum

; ON-LABEL: define float @same_global_load_noalias_store(
; ON: %a = load float, ptr addrspace(1) %p0, align 4
; ON: store float 0.000000e+00, ptr addrspace(1) %op, align 4
; ON-NOT: %b = load float
; ON: %sum = fadd float %a, %a
; ON: ret float %sum

; ON-LABEL: define float @same_global_load_clobber(
; ON: %a = load float, ptr addrspace(1) %p, align 4
; ON: store float 1.000000e+00, ptr addrspace(1) %p, align 4
; ON: %b = load float, ptr addrspace(1) %p, align 4
; ON: %sum = fadd float %a, %b
; ON: ret float %sum

; ON-LABEL: define float @same_global_load_call(
; ON: %a = load float, ptr addrspace(1) %p, align 4
; ON: call void @unknown()
; ON: %b = load float, ptr addrspace(1) %p, align 4
; ON: %sum = fadd float %a, %b
; ON: ret float %sum
