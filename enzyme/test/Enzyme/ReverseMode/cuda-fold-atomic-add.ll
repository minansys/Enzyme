; RUN: if [ %llvmver -lt 17 ]; then %opt < %s %newLoadEnzyme -opaque-pointers -enzyme-preopt=false -enzyme-detect-readthrow=0 -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -S | FileCheck %s --check-prefix=OFF; fi
; RUN: if [ %llvmver -ge 17 ]; then %opt < %s %newLoadEnzyme -enzyme-preopt=false -enzyme-detect-readthrow=0 -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -S | FileCheck %s --check-prefix=OFF; fi
; RUN: if [ %llvmver -lt 17 ]; then %opt < %s %newLoadEnzyme -opaque-pointers -enzyme-enable-cuda-repeated-loads=1 -enzyme-preopt=false -enzyme-detect-readthrow=0 -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -S | FileCheck %s --check-prefix=ON; fi
; RUN: if [ %llvmver -ge 17 ]; then %opt < %s %newLoadEnzyme -enzyme-enable-cuda-repeated-loads=1 -enzyme-preopt=false -enzyme-detect-readthrow=0 -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -S | FileCheck %s --check-prefix=ON; fi

; ModuleID = 'cuda-fold-atomic-add.ll'
source_filename = "cuda-fold-atomic-add.ll"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16-v32:32-v64:64:64-v128:128:128-n16:32:64-ni:10:11:12:13"
target triple = "nvptx64-nvidia-cuda"

define void @repeated(ptr addrspace(1) %ac, ptr addrspace(1) %out, i64 %i) {
entry:
  %p0 = getelementptr inbounds float, ptr addrspace(1) %ac, i64 %i
  %a = load float, ptr addrspace(1) %p0, align 4
  %aa = fmul float %a, %a
  %p1 = getelementptr inbounds float, ptr addrspace(1) %ac, i64 %i
  %b = load float, ptr addrspace(1) %p1, align 4
  %bb = fmul float %b, %b
  %sum = fadd float %aa, %bb
  %op = getelementptr inbounds float, ptr addrspace(1) %out, i64 %i
  store float %sum, ptr addrspace(1) %op, align 4
  ret void
}

define void @test(ptr addrspace(1) %ac, ptr addrspace(1) %dac,
                  ptr addrspace(1) %out, ptr addrspace(1) %dout, i64 %i) {
entry:
  call void (ptr, ...) @__enzyme_autodiff(ptr @repeated,
                                          ptr addrspace(1) %ac,
                                          ptr addrspace(1) %dac,
                                          ptr addrspace(1) %out,
                                          ptr addrspace(1) %dout,
                                          i64 %i)
  ret void
}

declare void @__enzyme_autodiff(ptr, ...)

; OFF-LABEL: define internal void @differepeated(
; OFF: %a = load float, ptr addrspace(1) %p0
; OFF: %sum = fadd float %aa, %aa
; OFF: store float 0.000000e+00
; OFF: atomicrmw fadd {{.*}} monotonic
; OFF: atomicrmw fadd {{.*}} monotonic
; OFF-NOT: atomicrmw
; OFF: ret void

; ON-LABEL: define internal void @differepeated(
; ON: %a = load float, ptr addrspace(1) %p0
; ON-NOT: %b = load float
; ON: %sum = fadd float %aa, %aa
; ON: store float 0.000000e+00
; ON: fadd fast float
; ON: atomicrmw fadd {{.*}} monotonic{{.*}} !alias.scope !{{[0-9]+}}, !noalias !{{[0-9]+}}, !enzyme_shadow_atomic !{{[0-9]+}}
; ON-NOT: atomicrmw
; ON: ret void
