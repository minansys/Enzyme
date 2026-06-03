; RUN: if [ %llvmver -lt 17 ]; then %opt < %s %newLoadEnzyme -opaque-pointers -passes="simple-gvn" -S | FileCheck %s --check-prefix=OFF; fi
; RUN: if [ %llvmver -ge 17 ]; then %opt < %s %newLoadEnzyme -passes="simple-gvn" -S | FileCheck %s --check-prefix=OFF; fi
; RUN: if [ %llvmver -lt 17 ]; then %opt < %s %newLoadEnzyme -opaque-pointers -enzyme-enable-cuda-repeated-loads=1 -passes="simple-gvn" -S | FileCheck %s --check-prefix=ON; fi
; RUN: if [ %llvmver -ge 17 ]; then %opt < %s %newLoadEnzyme -enzyme-enable-cuda-repeated-loads=1 -passes="simple-gvn" -S | FileCheck %s --check-prefix=ON; fi

target triple = "nvptx64-nvidia-cuda"

define void @fold_float_noalias_interleaved(ptr noalias %du, ptr noalias %dv,
                                            i64 %i, float %a, float %b,
                                            float %c) {
entry:
  %p0 = getelementptr inbounds float, ptr %du, i64 %i
  atomicrmw fadd ptr %p0, float %a monotonic, !enzyme_shadow_atomic !5
  %q = getelementptr inbounds float, ptr %dv, i64 %i
  atomicrmw fadd ptr %q, float %b monotonic, !enzyme_shadow_atomic !5
  %p1 = getelementptr inbounds float, ptr %du, i64 %i
  atomicrmw fadd ptr %p1, float %c monotonic, !enzyme_shadow_atomic !5
  ret void
}

define void @fold_double_noalias_interleaved(ptr noalias %du, ptr noalias %dv,
                                             i64 %i, double %a, double %b,
                                             double %c) {
entry:
  %p0 = getelementptr inbounds double, ptr %du, i64 %i
  atomicrmw fadd ptr %p0, double %a monotonic, !enzyme_shadow_atomic !5
  %q = getelementptr inbounds double, ptr %dv, i64 %i
  atomicrmw fadd ptr %q, double %b monotonic, !enzyme_shadow_atomic !5
  %p1 = getelementptr inbounds double, ptr %du, i64 %i
  atomicrmw fadd ptr %p1, double %c monotonic, !enzyme_shadow_atomic !5
  ret void
}

define void @fold_scoped_noalias_interleaved(ptr %du, ptr %dv, i64 %i,
                                             float %a, float %b, float %c) {
entry:
  %p0 = getelementptr inbounds float, ptr %du, i64 %i
  atomicrmw fadd ptr %p0, float %a monotonic, align 4, !enzyme_shadow_atomic !5, !alias.scope !0, !noalias !3
  %q = getelementptr inbounds float, ptr %dv, i64 %i
  atomicrmw fadd ptr %q, float %b monotonic, align 4, !enzyme_shadow_atomic !5, !alias.scope !3, !noalias !0
  %p1 = getelementptr inbounds float, ptr %du, i64 %i
  atomicrmw fadd ptr %p1, float %c monotonic, align 4, !enzyme_shadow_atomic !5, !alias.scope !0, !noalias !3
  ret void
}

define void @keep_float_may_alias_interleaved(ptr %du, ptr %dv, i64 %i,
                                              float %a, float %b, float %c) {
entry:
  %p0 = getelementptr inbounds float, ptr %du, i64 %i
  atomicrmw fadd ptr %p0, float %a monotonic, !enzyme_shadow_atomic !5
  %q = getelementptr inbounds float, ptr %dv, i64 %i
  atomicrmw fadd ptr %q, float %b monotonic, !enzyme_shadow_atomic !5
  %p1 = getelementptr inbounds float, ptr %du, i64 %i
  atomicrmw fadd ptr %p1, float %c monotonic, !enzyme_shadow_atomic !5
  ret void
}

define void @keep_float_store_clobber(ptr noalias %du, i64 %i, float %a,
                                      float %b) {
entry:
  %p0 = getelementptr inbounds float, ptr %du, i64 %i
  atomicrmw fadd ptr %p0, float %a monotonic, !enzyme_shadow_atomic !5
  store float 0.000000e+00, ptr %p0, align 4
  %p1 = getelementptr inbounds float, ptr %du, i64 %i
  atomicrmw fadd ptr %p1, float %b monotonic, !enzyme_shadow_atomic !5
  ret void
}

define void @keep_float_seq_cst(ptr noalias %du, i64 %i, float %a, float %b) {
entry:
  %p0 = getelementptr inbounds float, ptr %du, i64 %i
  atomicrmw fadd ptr %p0, float %a seq_cst, !enzyme_shadow_atomic !5
  %p1 = getelementptr inbounds float, ptr %du, i64 %i
  atomicrmw fadd ptr %p1, float %b seq_cst, !enzyme_shadow_atomic !5
  ret void
}

define void @keep_untagged_float_noalias(ptr noalias %du, i64 %i, float %a,
                                         float %b) {
entry:
  %p0 = getelementptr inbounds float, ptr %du, i64 %i
  atomicrmw fadd ptr %p0, float %a monotonic
  %p1 = getelementptr inbounds float, ptr %du, i64 %i
  atomicrmw fadd ptr %p1, float %b monotonic
  ret void
}

!0 = !{!1}
!1 = distinct !{!1, !2, !"du"}
!2 = distinct !{!2, !"shadow-domain"}
!3 = !{!4}
!4 = distinct !{!4, !2, !"dv"}
!5 = !{}

; OFF-LABEL: define void @fold_float_noalias_interleaved(
; OFF: atomicrmw fadd ptr %p0, float %a monotonic
; OFF: atomicrmw fadd ptr %q, float %b monotonic
; OFF: atomicrmw fadd ptr %p1, float %c monotonic
; OFF: ret void

; ON-LABEL: define void @fold_float_noalias_interleaved(
; ON-NOT: atomicrmw
; ON: atomicrmw fadd ptr %q, float %b monotonic
; ON: [[FOLD:%.*]] = fadd float %a, %c
; ON: atomicrmw fadd ptr %p1, float [[FOLD]] monotonic
; ON-NOT: atomicrmw
; ON: ret void

; OFF-LABEL: define void @fold_double_noalias_interleaved(
; OFF: atomicrmw fadd ptr %p0, double %a monotonic
; OFF: atomicrmw fadd ptr %q, double %b monotonic
; OFF: atomicrmw fadd ptr %p1, double %c monotonic
; OFF: ret void

; ON-LABEL: define void @fold_double_noalias_interleaved(
; ON-NOT: atomicrmw
; ON: atomicrmw fadd ptr %q, double %b monotonic
; ON: [[DFOLD:%.*]] = fadd double %a, %c
; ON: atomicrmw fadd ptr %p1, double [[DFOLD]] monotonic
; ON-NOT: atomicrmw
; ON: ret void

; OFF-LABEL: define void @fold_scoped_noalias_interleaved(
; OFF: atomicrmw fadd ptr %p0, float %a monotonic
; OFF: atomicrmw fadd ptr %q, float %b monotonic
; OFF: atomicrmw fadd ptr %p1, float %c monotonic
; OFF: ret void

; ON-LABEL: define void @fold_scoped_noalias_interleaved(
; ON-NOT: atomicrmw
; ON: atomicrmw fadd ptr %q, float %b monotonic
; ON: [[MFOLD:%.*]] = fadd float %a, %c
; ON: atomicrmw fadd ptr %p1, float [[MFOLD]] monotonic
; ON-NOT: atomicrmw
; ON: ret void

; ON-LABEL: define void @keep_float_may_alias_interleaved(
; ON: atomicrmw fadd ptr %p0, float %a monotonic
; ON: atomicrmw fadd ptr %q, float %b monotonic
; ON: atomicrmw fadd ptr %p1, float %c monotonic
; ON: ret void

; ON-LABEL: define void @keep_float_store_clobber(
; ON: atomicrmw fadd ptr %p0, float %a monotonic
; ON: store float 0.000000e+00, ptr %p0, align 4
; ON: atomicrmw fadd ptr %p1, float %b monotonic
; ON: ret void

; ON-LABEL: define void @keep_float_seq_cst(
; ON: atomicrmw fadd ptr %p0, float %a seq_cst
; ON: atomicrmw fadd ptr %p1, float %b seq_cst
; ON: ret void

; ON-LABEL: define void @keep_untagged_float_noalias(
; ON: atomicrmw fadd ptr %p0, float %a monotonic
; ON: atomicrmw fadd ptr %p1, float %b monotonic
; ON: ret void
