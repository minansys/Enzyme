; RUN: if [ %llvmver -lt 17 ]; then %opt < %s %newLoadEnzyme -opaque-pointers -enzyme-enable-cuda-shadow-atomic-cache=0 -passes="simple-gvn" -S | FileCheck %s --check-prefix=OFF; fi
; RUN: if [ %llvmver -ge 17 ]; then %opt < %s %newLoadEnzyme -enzyme-enable-cuda-shadow-atomic-cache=0 -passes="simple-gvn" -S | FileCheck %s --check-prefix=OFF; fi
; RUN: if [ %llvmver -lt 17 ]; then %opt < %s %newLoadEnzyme -opaque-pointers -enzyme-enable-cuda-repeated-loads=1 -enzyme-enable-cuda-shadow-atomic-cache=0 -passes="simple-gvn" -S | FileCheck %s --check-prefix=ON; fi
; RUN: if [ %llvmver -ge 17 ]; then %opt < %s %newLoadEnzyme -enzyme-enable-cuda-repeated-loads=1 -enzyme-enable-cuda-shadow-atomic-cache=0 -passes="simple-gvn" -S | FileCheck %s --check-prefix=ON; fi

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

define float @same_global_load_single_pred(ptr addrspace(1) %ac, i64 %i) {
entry:
  %p0 = getelementptr inbounds float, ptr addrspace(1) %ac, i64 %i
  %a = load float, ptr addrspace(1) %p0, align 4
  br label %cont

cont:
  %p1 = getelementptr inbounds float, ptr addrspace(1) %ac, i64 %i
  %b = load float, ptr addrspace(1) %p1, align 4
  %sum = fadd float %a, %b
  ret float %sum
}

define float @same_global_load_diamond(ptr addrspace(1) %ac, i64 %i,
                                       i1 %cond) {
entry:
  %p0 = getelementptr inbounds float, ptr addrspace(1) %ac, i64 %i
  %a = load float, ptr addrspace(1) %p0, align 4
  br i1 %cond, label %then, label %else

then:
  br label %cont

else:
  br label %cont

cont:
  %p1 = getelementptr inbounds float, ptr addrspace(1) %ac, i64 %i
  %b = load float, ptr addrspace(1) %p1, align 4
  %sum = fadd float %a, %b
  ret float %sum
}

define float @keep_global_load_diamond_clobber(ptr addrspace(1) %ac, i64 %i,
                                               i1 %cond) {
entry:
  %p0 = getelementptr inbounds float, ptr addrspace(1) %ac, i64 %i
  %a = load float, ptr addrspace(1) %p0, align 4
  br i1 %cond, label %then, label %else

then:
  store float 1.000000e+00, ptr addrspace(1) %p0, align 4
  br label %cont

else:
  br label %cont

cont:
  %p1 = getelementptr inbounds float, ptr addrspace(1) %ac, i64 %i
  %b = load float, ptr addrspace(1) %p1, align 4
  %sum = fadd float %a, %b
  ret float %sum
}

define float @same_global_load_pred_phi(ptr addrspace(1) %ac, i64 %i,
                                        i1 %cond) {
entry:
  br i1 %cond, label %then, label %else

then:
  %pt = getelementptr inbounds float, ptr addrspace(1) %ac, i64 %i
  %a = load float, ptr addrspace(1) %pt, align 4
  br label %cont

else:
  %pe = getelementptr inbounds float, ptr addrspace(1) %ac, i64 %i
  %b = load float, ptr addrspace(1) %pe, align 4
  br label %cont

cont:
  %p1 = getelementptr inbounds float, ptr addrspace(1) %ac, i64 %i
  %c = load float, ptr addrspace(1) %p1, align 4
  %sum = fadd float %c, 1.000000e+00
  ret float %sum
}

define float @keep_global_load_pred_phi_clobber(ptr addrspace(1) %ac, i64 %i,
                                                i1 %cond) {
entry:
  br i1 %cond, label %then, label %else

then:
  %pt = getelementptr inbounds float, ptr addrspace(1) %ac, i64 %i
  %a = load float, ptr addrspace(1) %pt, align 4
  store float 1.000000e+00, ptr addrspace(1) %pt, align 4
  br label %cont

else:
  %pe = getelementptr inbounds float, ptr addrspace(1) %ac, i64 %i
  %b = load float, ptr addrspace(1) %pe, align 4
  br label %cont

cont:
  %p1 = getelementptr inbounds float, ptr addrspace(1) %ac, i64 %i
  %c = load float, ptr addrspace(1) %p1, align 4
  %sum = fadd float %c, 1.000000e+00
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

; ON-LABEL: define float @same_global_load_single_pred(
; ON: %a = load float, ptr addrspace(1) %p0, align 4
; ON: br label %cont
; ON: cont:
; ON-NOT: %b = load float
; ON: %sum = fadd float %a, %a
; ON: ret float %sum

; ON-LABEL: define float @same_global_load_diamond(
; ON: %a = load float, ptr addrspace(1) %p0, align 4
; ON: cont:
; ON-NOT: %b = load float
; ON: %sum = fadd float %a, %a
; ON: ret float %sum

; ON-LABEL: define float @keep_global_load_diamond_clobber(
; ON: %a = load float, ptr addrspace(1) %p0, align 4
; ON: store float 1.000000e+00, ptr addrspace(1) %p0, align 4
; ON: cont:
; ON: %b = load float, ptr addrspace(1) %p1, align 4
; ON: %sum = fadd float %a, %b
; ON: ret float %sum

; ON-LABEL: define float @same_global_load_pred_phi(
; ON: then:
; ON: %a = load float, ptr addrspace(1) %pt, align 4
; ON: else:
; ON: %b = load float, ptr addrspace(1) %pe, align 4
; ON: cont:
; ON: [[PHI:%.*]] = phi float
; ON-NOT: %c = load float
; ON: %sum = fadd float [[PHI]], 1.000000e+00
; ON: ret float %sum

; ON-LABEL: define float @keep_global_load_pred_phi_clobber(
; ON: then:
; ON: %a = load float, ptr addrspace(1) %pt, align 4
; ON: store float 1.000000e+00, ptr addrspace(1) %pt, align 4
; ON: cont:
; ON: %c = load float, ptr addrspace(1) %p1, align 4
; ON: %sum = fadd float %c, 1.000000e+00
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
