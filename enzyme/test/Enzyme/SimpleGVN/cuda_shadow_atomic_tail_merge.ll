; RUN: if [ %llvmver -lt 17 ]; then %opt < %s %newLoadEnzyme -opaque-pointers -enzyme-enable-cuda-repeated-loads=1 -passes="simple-gvn" -S | FileCheck %s --check-prefix=OFF; fi
; RUN: if [ %llvmver -ge 17 ]; then %opt < %s %newLoadEnzyme -enzyme-enable-cuda-repeated-loads=1 -passes="simple-gvn" -S | FileCheck %s --check-prefix=OFF; fi
; RUN: if [ %llvmver -lt 17 ]; then %opt < %s %newLoadEnzyme -opaque-pointers -enzyme-enable-cuda-repeated-loads=1 -enzyme-enable-cuda-atomic-tail-merge=1 -passes="simple-gvn" -S | FileCheck %s --check-prefix=MERGE; fi
; RUN: if [ %llvmver -ge 17 ]; then %opt < %s %newLoadEnzyme -enzyme-enable-cuda-repeated-loads=1 -enzyme-enable-cuda-atomic-tail-merge=1 -passes="simple-gvn" -S | FileCheck %s --check-prefix=MERGE; fi

target triple = "nvptx64-nvidia-cuda"

define void @merge_shadow_atomic_tails(ptr noalias %du, ptr noalias %dv,
                                       ptr noalias %scratch, i1 %cond,
                                       i64 %i, float %a, float %b,
                                       float %c, float %d) {
entry:
  br i1 %cond, label %left, label %right

left:
  %p0 = getelementptr inbounds float, ptr %du, i64 %i
  atomicrmw fadd ptr %p0, float %a monotonic, !enzyme_shadow_atomic !0
  store float %c, ptr %scratch, align 4
  %q0 = getelementptr inbounds float, ptr %dv, i64 %i
  atomicrmw fadd ptr %q0, float %b monotonic, !enzyme_shadow_atomic !0
  br label %done

right:
  %q1 = getelementptr inbounds float, ptr %dv, i64 %i
  atomicrmw fadd ptr %q1, float %d monotonic, !enzyme_shadow_atomic !0
  store float %b, ptr %scratch, align 4
  %p1 = getelementptr inbounds float, ptr %du, i64 %i
  atomicrmw fadd ptr %p1, float %c monotonic, !enzyme_shadow_atomic !0
  br label %done

done:
  ret void
}

define void @keep_may_alias_tail(ptr %du, i1 %cond, i64 %i, float %a,
                                 float %b) {
entry:
  br i1 %cond, label %left, label %right

left:
  %p0 = getelementptr inbounds float, ptr %du, i64 %i
  atomicrmw fadd ptr %p0, float %a monotonic, !enzyme_shadow_atomic !0
  store float 0.000000e+00, ptr %p0, align 4
  %p1 = getelementptr inbounds float, ptr %du, i64 %i
  atomicrmw fadd ptr %p1, float %b monotonic, !enzyme_shadow_atomic !0
  br label %done

right:
  %p2 = getelementptr inbounds float, ptr %du, i64 %i
  atomicrmw fadd ptr %p2, float %b monotonic, !enzyme_shadow_atomic !0
  store float 0.000000e+00, ptr %p2, align 4
  %p3 = getelementptr inbounds float, ptr %du, i64 %i
  atomicrmw fadd ptr %p3, float %a monotonic, !enzyme_shadow_atomic !0
  br label %done

done:
  ret void
}

!0 = !{}

; OFF-LABEL: define void @merge_shadow_atomic_tails(
; OFF: left:
; OFF: atomicrmw fadd ptr %p0, float %a monotonic
; OFF: store float %c, ptr %scratch, align 4
; OFF: atomicrmw fadd ptr %q0, float %b monotonic
; OFF: right:
; OFF: atomicrmw fadd ptr %q1, float %d monotonic
; OFF: store float %b, ptr %scratch, align 4
; OFF: atomicrmw fadd ptr %p1, float %c monotonic
; OFF: done:
; OFF: ret void

; MERGE-LABEL: define void @merge_shadow_atomic_tails(
; MERGE: left:
; MERGE-NOT: atomicrmw
; MERGE: br label %enzyme.atomic.merge
; MERGE: right:
; MERGE-NOT: atomicrmw
; MERGE: br label %enzyme.atomic.merge
; MERGE: enzyme.atomic.merge:
; MERGE: [[PTR0:%.*]] = phi ptr
; MERGE: [[VAL0:%.*]] = phi float
; MERGE: [[PTR1:%.*]] = phi ptr
; MERGE: [[VAL1:%.*]] = phi float
; MERGE: atomicrmw fadd ptr [[PTR0]], float [[VAL0]] monotonic
; MERGE: atomicrmw fadd ptr [[PTR1]], float [[VAL1]] monotonic
; MERGE-NOT: atomicrmw
; MERGE: done:
; MERGE: ret void

; MERGE-LABEL: define void @keep_may_alias_tail(
; MERGE: left:
; MERGE: atomicrmw fadd ptr %p0, float %a monotonic
; MERGE: store float 0.000000e+00, ptr %p0, align 4
; MERGE: atomicrmw fadd ptr %p1, float %b monotonic
; MERGE: right:
; MERGE: atomicrmw fadd ptr %p2, float %b monotonic
; MERGE: store float 0.000000e+00, ptr %p2, align 4
; MERGE: atomicrmw fadd ptr %p3, float %a monotonic
; MERGE: done:
; MERGE: ret void
