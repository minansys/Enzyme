; RUN: if [ %llvmver -lt 17 ]; then %opt < %s %newLoadEnzyme -opaque-pointers -enzyme-preopt=false -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -S | FileCheck %s --check-prefix=BASE; fi
; RUN: if [ %llvmver -ge 17 ]; then %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -S | FileCheck %s --check-prefix=BASE; fi
; RUN: if [ %llvmver -lt 17 ]; then %opt < %s %newLoadEnzyme -opaque-pointers -enzyme-noalias=1 -enzyme-preopt=false -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -S | FileCheck %s --check-prefix=NOALIAS; fi
; RUN: if [ %llvmver -ge 17 ]; then %opt < %s %newLoadEnzyme -enzyme-noalias=1 -enzyme-preopt=false -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -S | FileCheck %s --check-prefix=NOALIAS; fi

define void @caller(ptr %x, ptr %dx, ptr %y, ptr %dy) {
entry:
  call void (ptr, ...) @__enzyme_autodiff(ptr @pair_product,
                                          metadata !"enzyme_dup", ptr %x, ptr %dx,
                                          metadata !"enzyme_dup", ptr %y, ptr %dy)
  ret void
}

define void @pair_product(ptr %x, ptr %y) {
entry:
  %xv = load double, ptr %x, align 8
  %yv = load double, ptr %y, align 8
  %prod = fmul double %xv, %yv
  store double %prod, ptr %y, align 8
  ret void
}

declare void @__enzyme_autodiff(ptr, ...)

; BASE-LABEL: define internal void @diffepair_product(
; BASE-NOT: noalias
; BASE-SAME: ptr {{.*}}%x, ptr {{.*}}%"x'", ptr {{.*}}%y, ptr {{.*}}%"y'")

; NOALIAS-LABEL: define internal void @diffepair_product(
; NOALIAS-SAME: ptr noalias {{.*}}%x
; NOALIAS-SAME: ptr noalias {{.*}}%"x'"
; NOALIAS-SAME: ptr noalias {{.*}}%y
; NOALIAS-SAME: ptr noalias {{.*}}%"y'")
