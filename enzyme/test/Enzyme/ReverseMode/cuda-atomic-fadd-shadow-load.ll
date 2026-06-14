; RUN: if [ %llvmver -lt 17 ]; then %opt < %s %newLoadEnzyme -opaque-pointers -enzyme-preopt=false -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -S | FileCheck %s; fi
; RUN: if [ %llvmver -ge 17 ]; then %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -S | FileCheck %s; fi

target triple = "nvptx64-nvidia-cuda"

define void @atomic_residual_add(ptr %res, ptr %x) {
entry:
  %v = load float, ptr %x, align 4
  atomicrmw fadd ptr %res, float %v seq_cst, align 4
  ret void
}

define void @test(ptr %res, ptr %dres, ptr %x, ptr %dx) {
entry:
  call void (ptr, ...) @__enzyme_autodiff(ptr @atomic_residual_add,
                                          metadata !"enzyme_dup", ptr %res, ptr %dres,
                                          metadata !"enzyme_dup", ptr %x, ptr %dx)
  ret void
}

declare void @__enzyme_autodiff(ptr, ...)

; CHECK-LABEL: define internal void @diffeatomic_residual_add(
; CHECK-NOT: load atomic float
; CHECK: load float, ptr %"res'"
; CHECK-NOT: load atomic float
; CHECK: ret void
