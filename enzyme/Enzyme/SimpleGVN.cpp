//=- SimpleGVN.cpp - GVN-like load forwarding optimization ============//
//
//                             Enzyme Project
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// If using this code in an academic setting, please cite the following:
// @incollection{enzymeNeurips,
// title = {Instead of Rewriting Foreign Code for Machine Learning,
//          Automatically Synthesize Fast Gradients},
// author = {Moses, William S. and Churavy, Valentin},
// booktitle = {Advances in Neural Information Processing Systems 33},
// year = {2020},
// note = {To appear in},
// }
//
//===----------------------------------------------------------------------===//
//
// This file contains a GVN-like optimization pass that forwards loads from
// noalias/nocapture arguments to their corresponding stores, with support
// for offsets and type conversions.
// It also performs conservative repeated-load forwarding for NVPTX global
// memory loads when no intervening instruction can clobber the loaded address.
//
// This pass addresses the limitation of LLVM's built-in GVN pass which has
// a small limit on the number of instructions/memory offsets it analyzes
// via its use of the memdep analysis.
//
// Algorithm:
// 1. Identify function arguments with noalias and nocapture attributes
// 2. Verify all uses are exclusively loads, stores, or GEP instructions
// 3. For each load from such an argument:
//    a. Find all stores to the argument with constant offsets
//    b. Find a dominating store that covers the load's memory range
//    c. Check that no aliasing store exists between the store and load
//    d. If safe, replace the load with the stored value, performing
//       type conversion or extraction as needed
//
// Example transformation:
//   define i32 @foo(i32* noalias nocapture %ptr) {
//     store i32 42, i32* %ptr
//     %v = load i32, i32* %ptr
//     ret i32 %v
//   }
// becomes:
//   define i32 @foo(i32* noalias nocapture %ptr) {
//     store i32 42, i32* %ptr
//     ret i32 42
//   }
//
//===----------------------------------------------------------------------===//
#include <llvm/Config/llvm-config.h>

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#if LLVM_VERSION_MAJOR >= 17
#include "llvm/TargetParser/Triple.h"
#else
#include "llvm/ADT/Triple.h"
#endif

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Dominators.h"

#include "llvm/IR/CFG.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GetElementPtrTypeIterator.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/Value.h"

#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/MemoryLocation.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/ValueTracking.h"

#include "llvm/IR/LegacyPassManager.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include "llvm/Transforms/Utils/Local.h"

#include "SimpleGVN.h"
#include "Utils.h"

using namespace llvm;

#ifdef DEBUG_TYPE
#undef DEBUG_TYPE
#endif
#define DEBUG_TYPE "simple-gvn"

llvm::cl::opt<bool> EnzymeEnableCudaRepeatedLoads(
    "enzyme-enable-cuda-repeated-loads", cl::init(false), cl::Hidden,
    cl::desc("Enable CUDA repeated global load forwarding and adjoint atomic "
             "folding"));

llvm::cl::opt<bool> EnzymeEnableCudaAtomicTailMerge(
    "enzyme-enable-cuda-atomic-tail-merge", cl::init(false), cl::Hidden,
    cl::desc("Enable CUDA Enzyme shadow atomic tail merging"));

llvm::cl::opt<bool> EnzymeEnableCudaPointerTableLoads(
    "enzyme-enable-cuda-pointer-table-loads", cl::init(false), cl::Hidden,
    cl::desc("Assume CUDA pointer-table row writes are disjoint from readonly "
             "argument/global storage for repeated-load forwarding"));

namespace {

// Extract a value with potential type conversion
Value *extractValue(IRBuilder<> &Builder, Value *StoredVal, Type *LoadType,
                    const DataLayout &DL, APInt LoadOffset, APInt StoreOffset,
                    uint64_t LoadSize) {
  Type *StoreType = StoredVal->getType();
  uint64_t StoreSize = DL.getTypeStoreSize(StoreType);

  // Calculate relative offset
  int64_t RelativeOffset = (LoadOffset - StoreOffset).getSExtValue();

  // Check if the load is completely within the stored value
  if (RelativeOffset < 0 || (uint64_t)RelativeOffset + LoadSize > StoreSize) {
    return nullptr;
  }

  // If types match and offsets are the same, return directly
  if (RelativeOffset == 0 && LoadType == StoreType) {
    return StoredVal;
  }

  if (RelativeOffset == 0 && isa<PointerType>(LoadType) &&
      isa<PointerType>(StoreType)) {
    auto FromAS = cast<PointerType>(StoreType)->getAddressSpace();
    auto ToAS = cast<PointerType>(LoadType)->getAddressSpace();
    if (FromAS == 10 && ToAS == 0) {
      Module *mod = Builder.GetInsertBlock()->getModule();
      Function *F = mod->getFunction("julia.pointer_from_objref");
      if (!F) {
        Type *T_jlvalue = StructType::get(mod->getContext());
        Type *T_prjlvalue = PointerType::get(T_jlvalue, 11); // Derived is 11
        Type *T_pjlvalue = PointerType::get(T_jlvalue, 0);
        FunctionType *FTy = FunctionType::get(T_pjlvalue, {T_prjlvalue}, false);
        F = Function::Create(FTy, Function::ExternalLinkage,
                             "julia.pointer_from_objref", mod);
      }

      Type *T_jlvalue = StructType::get(mod->getContext());
      Value *TrackedVal = Builder.CreateAddrSpaceCast(
          StoredVal, PointerType::get(T_jlvalue, 11));
      Value *RawPtr = Builder.CreateCall(F, {TrackedVal});
      return Builder.CreatePointerCast(RawPtr, LoadType);
    }
    return Builder.CreatePointerCast(StoredVal, LoadType);
  }

  if (RelativeOffset == 0 && StoreSize >= LoadSize &&
      StoreType->isAggregateType()) {
    auto first = Builder.CreateExtractValue(StoredVal, 0);
    auto res = extractValue(Builder, first, LoadType, DL, LoadOffset,
                            StoreOffset, LoadSize);
    if (res) {
      return res;
    } else {
      if (auto I = dyn_cast<Instruction>(first))
        I->eraseFromParent();
    }
  }

  // Handle extraction with offset or type mismatch
  // First, bitcast to an integer type if needed
  if (!StoreType->isIntegerTy()) {
    IntegerType *IntTy = Builder.getIntNTy(StoreSize * 8);
    if (!CastInst::castIsValid(Instruction::BitCast, StoredVal->getType(),
                               IntTy)) {
      return nullptr;
    }
    StoredVal = Builder.CreateBitCast(StoredVal, IntTy);
  }

  // Extract the relevant bits if there's an offset
  if (RelativeOffset > 0) {
    uint64_t ShiftBits = RelativeOffset * 8;
    StoredVal = Builder.CreateLShr(StoredVal, ShiftBits);
  }

  // Truncate or extend to the load size if needed
  IntegerType *LoadIntTy = Builder.getIntNTy(LoadSize * 8);
  if (StoredVal->getType() != LoadIntTy) {
    unsigned StoredWidth = StoredVal->getType()->getIntegerBitWidth();
    unsigned LoadWidth = LoadIntTy->getIntegerBitWidth();
    if (StoredWidth > LoadWidth) {
      StoredVal = Builder.CreateTrunc(StoredVal, LoadIntTy);
    } else if (StoredWidth < LoadWidth) {
      StoredVal = Builder.CreateZExt(StoredVal, LoadIntTy);
    }
  }

  // Bitcast to the final type if needed
  if (LoadIntTy != LoadType) {
    if (LoadType->isPointerTy()) {
      if (cast<PointerType>(LoadType)->getAddressSpace() == 10) {
        Type *EmptyStructTy = StructType::get(LoadType->getContext());
        Type *PtrAddrSpace0 = PointerType::get(EmptyStructTy, 0);
        Value *Ptr0 = Builder.CreateIntToPtr(StoredVal, PtrAddrSpace0);
        StoredVal = Builder.CreateAddrSpaceCast(Ptr0, LoadType);
      } else {
        StoredVal = Builder.CreateIntToPtr(StoredVal, LoadType);
      }
    } else {
      if (!CastInst::castIsValid(Instruction::BitCast, StoredVal->getType(),
                                 LoadType)) {
        return nullptr;
      }
      StoredVal = Builder.CreateBitCast(StoredVal, LoadType);
    }
  }

  return StoredVal;
}

// Helper to check if a source instruction dominates and completely covers a
// target instruction's memory access
// For stores: checks if store covers a load
// For loads: checks if load covers another load
static bool dominatesAndCovers(Instruction *Source, Instruction *Target,
                               const APInt &SourceOffset,
                               const APInt &TargetOffset, uint64_t TargetSize,
                               const DataLayout &DL, DominatorTree &DT) {
  if (!DT.dominates(Source, Target))
    return false;

  // Get the size of the source memory access
  uint64_t SourceSize;
  if (auto *SI = dyn_cast<StoreInst>(Source)) {
    SourceSize = DL.getTypeStoreSize(SI->getValueOperand()->getType());
  } else if (auto *LI = dyn_cast<LoadInst>(Source)) {
    SourceSize = DL.getTypeStoreSize(LI->getType());
  } else {
    return false;
  }

  int64_t RelOffset = (TargetOffset - SourceOffset).getSExtValue();
  return RelOffset >= 0 && (uint64_t)RelOffset + TargetSize <= SourceSize;
}

// Helper to check if two memory ranges alias
// Range1: [Offset1, Offset1 + Size1)
// Range2: [Offset2, Offset2 + Size2)
static bool memoryRangesAlias(const APInt &Offset1, uint64_t Size1,
                              const APInt &Offset2, uint64_t Size2) {
  // Check if range2 ends before range1 begins
  if ((Offset2 + Size2).sle(Offset1))
    return false;

  // Check if range1 ends before range2 begins
  if ((Offset1 + Size1).sle(Offset2))
    return false;

  // Otherwise, they may alias
  return true;
}

static bool isNVPTXModule(const Module *M) {
  if (!M)
    return false;
  Triple::ArchType Arch = Triple(M->getTargetTriple()).getArch();
  return Arch == Triple::nvptx || Arch == Triple::nvptx64;
}

static bool isRepeatedGlobalLoadCandidate(LoadInst *LI) {
  if (!LI->isSimple())
    return false;

  // CUDA C++ device pointers commonly reach LLVM as generic addrspace(0)
  // pointers even when they refer to global memory. The forwarding below is
  // guarded by AA clobber checks, so it is also safe for generic pointers.
  unsigned AS = LI->getPointerAddressSpace();
  return AS == 0 || AS == 1;
}

static Value *getPointerBase(Value *V) {
  SmallPtrSet<Value *, 8> Seen;
  while (V && Seen.insert(V).second) {
    V = V->stripPointerCasts();
    if (auto *GEP = dyn_cast<GEPOperator>(V)) {
      V = GEP->getPointerOperand();
      continue;
    }
    return V;
  }
  return V;
}

static bool isReadonlyArgOrGlobalBase(Value *Base) {
  Base = getPointerBase(Base);
  if (auto *Arg = dyn_cast_or_null<Argument>(Base))
    return Arg->hasAttribute(Attribute::ReadOnly);
  if (auto *GV = dyn_cast_or_null<GlobalVariable>(Base))
    return GV->isConstant();
  return false;
}

static bool isReadonlyArgOrGlobalLoad(LoadInst *LI) {
  return isRepeatedGlobalLoadCandidate(LI) &&
         isReadonlyArgOrGlobalBase(LI->getPointerOperand());
}

static bool isPointerTableEntryLoad(LoadInst *LI) {
  return LI->getType()->isPointerTy() && isReadonlyArgOrGlobalLoad(LI);
}

static Value *getMemoryAccessPointer(Instruction *I) {
  if (auto *SI = dyn_cast<StoreInst>(I))
    return SI->getPointerOperand();
  if (auto *RMW = dyn_cast<AtomicRMWInst>(I))
    return RMW->getPointerOperand();
  if (auto *CXI = dyn_cast<AtomicCmpXchgInst>(I))
    return CXI->getPointerOperand();
  return nullptr;
}

static bool isCudaPointerTableRowAccess(Instruction *I) {
  Value *AccessPtr = getMemoryAccessPointer(I);
  if (!AccessPtr)
    return false;

  auto *RowLoad = dyn_cast_or_null<LoadInst>(getPointerBase(AccessPtr));
  return RowLoad && isPointerTableEntryLoad(RowLoad);
}

static bool canIgnorePointerTableRowClobber(Instruction *MaybeClobber,
                                            LoadInst *LI) {
  return EnzymeEnableCudaPointerTableLoads && isReadonlyArgOrGlobalLoad(LI) &&
         isCudaPointerTableRowAccess(MaybeClobber);
}

static bool sameLoadPointer(Value *LHS, Value *RHS) {
  if (LHS == RHS)
    return true;

  LHS = LHS->stripPointerCasts();
  RHS = RHS->stripPointerCasts();
  if (LHS == RHS)
    return true;

  auto *LHSGEP = dyn_cast<GetElementPtrInst>(LHS);
  auto *RHSGEP = dyn_cast<GetElementPtrInst>(RHS);
  return LHSGEP && RHSGEP && LHSGEP->isIdenticalToWhenDefined(RHSGEP);
}

static bool canForwardRepeatedLoad(LoadInst *Prev, LoadInst *LI,
                                   AAResults &AA) {
  if (Prev->getType() != LI->getType() ||
      Prev->getPointerAddressSpace() != LI->getPointerAddressSpace())
    return false;
  if (sameLoadPointer(Prev->getPointerOperand(), LI->getPointerOperand()))
    return true;
  return AA.isMustAlias(MemoryLocation::get(Prev), MemoryLocation::get(LI));
}

static bool mayClobberLoad(Instruction *MaybeClobber, LoadInst *LI,
                           AAResults &AA) {
  if (!MaybeClobber->mayWriteToMemory())
    return false;
  if (canIgnorePointerTableRowClobber(MaybeClobber, LI))
    return false;
  return isModSet(AA.getModRefInfo(MaybeClobber, MemoryLocation::get(LI)));
}

static void invalidateAvailableLoads(Instruction *I,
                                     SmallVectorImpl<LoadInst *> &Available,
                                     AAResults &AA) {
  if (Available.empty())
    return;

  if (I->mayWriteToMemory()) {
    for (auto It = Available.begin(); It != Available.end();) {
      if (mayClobberLoad(I, *It, AA))
        It = Available.erase(It);
      else
        ++It;
    }
    return;
  }

  if (I->mayHaveSideEffects())
    Available.clear();
}

static bool simplifyRepeatedGlobalLoadsImpl(Function &F, AAResults &AA) {
  if (!EnzymeEnableCudaRepeatedLoads || !isNVPTXModule(F.getParent()))
    return false;

  bool Changed = false;
  for (BasicBlock &BB : F) {
    SmallVector<LoadInst *, 8> AvailableLoads;
    for (auto It = BB.begin(), End = BB.end(); It != End;) {
      Instruction *I = &*It++;

      if (auto *LI = dyn_cast<LoadInst>(I)) {
        if (isRepeatedGlobalLoadCandidate(LI)) {
          LoadInst *ForwardFrom = nullptr;
          for (LoadInst *Prev : AvailableLoads) {
            if (canForwardRepeatedLoad(Prev, LI, AA)) {
              ForwardFrom = Prev;
              break;
            }
          }

          if (ForwardFrom) {
            LLVM_DEBUG(dbgs() << "SimpleGVN: Forwarding repeated global load\n"
                              << "  Load: " << *ForwardFrom << "\n"
                              << "  Redundant load: " << *LI << "\n");
            LI->replaceAllUsesWith(ForwardFrom);
            LI->eraseFromParent();
            Changed = true;
            continue;
          }

          AvailableLoads.push_back(LI);
          continue;
        }
      }

      invalidateAvailableLoads(I, AvailableLoads, AA);
    }
  }

  return Changed;
}

static bool sameAtomicUpdatePointer(Value *LHS, Value *RHS) {
  if (LHS == RHS)
    return true;

  LHS = LHS->stripPointerCasts();
  RHS = RHS->stripPointerCasts();
  if (LHS == RHS)
    return true;

  auto *LHSI = dyn_cast<Instruction>(LHS);
  auto *RHSI = dyn_cast<Instruction>(RHS);
  return LHSI && RHSI && LHSI->isIdenticalToWhenDefined(RHSI);
}

static bool isCudaAtomicFAddCandidate(AtomicRMWInst *RMW) {
  return RMW->getMetadata("enzyme_shadow_atomic") &&
         RMW->getOperation() == AtomicRMWInst::FAdd && !RMW->isVolatile() &&
         RMW->use_empty() && RMW->getOrdering() == AtomicOrdering::Monotonic &&
         RMW->getValOperand()->getType()->isFloatingPointTy();
}

static bool sameAtomicLocation(AtomicRMWInst *Prev, AtomicRMWInst *RMW,
                               AAResults &AA) {
  if (sameAtomicUpdatePointer(Prev->getPointerOperand(),
                              RMW->getPointerOperand()))
    return true;
  return AA.isMustAlias(MemoryLocation::get(Prev), MemoryLocation::get(RMW));
}

static bool canCoalesceAtomicFAdd(AtomicRMWInst *Prev, AtomicRMWInst *RMW,
                                  AAResults &AA) {
  if (!isCudaAtomicFAddCandidate(Prev) || !isCudaAtomicFAddCandidate(RMW))
    return false;
  if (Prev->getOrdering() != RMW->getOrdering() ||
      Prev->getSyncScopeID() != RMW->getSyncScopeID() ||
      Prev->getAlign() != RMW->getAlign())
    return false;
  return sameAtomicLocation(Prev, RMW, AA);
}

static bool instructionMayAccessLocation(Instruction *I,
                                         const MemoryLocation &Loc,
                                         AAResults &AA) {
  if (!I->mayReadOrWriteMemory())
    return false;
  return !isNoModRef(AA.getModRefInfo(I, Loc));
}

static void invalidatePendingAtomicFAdds(
    Instruction *I,
    SmallVectorImpl<std::pair<AtomicRMWInst *, MemoryLocation>> &Pending,
    AAResults &AA) {
  if (Pending.empty())
    return;

  if (I->mayHaveSideEffects() && !isa<AtomicRMWInst>(I)) {
    Pending.clear();
    return;
  }

  if (!I->mayReadOrWriteMemory())
    return;

  for (auto It = Pending.begin(); It != Pending.end();) {
    if (instructionMayAccessLocation(I, It->second, AA))
      It = Pending.erase(It);
    else
      ++It;
  }
}

static bool coalesceRepeatedCudaAtomicFAddsImpl(Function &F, AAResults &AA) {
  if (!EnzymeEnableCudaRepeatedLoads || !isNVPTXModule(F.getParent()))
    return false;

  bool Changed = false;
  for (BasicBlock &BB : F) {
    SmallVector<std::pair<AtomicRMWInst *, MemoryLocation>, 8> Pending;
    for (auto It = BB.begin(), End = BB.end(); It != End;) {
      Instruction *I = &*It++;
      auto *RMW = dyn_cast<AtomicRMWInst>(I);
      if (!RMW || !isCudaAtomicFAddCandidate(RMW)) {
        invalidatePendingAtomicFAdds(I, Pending, AA);
        continue;
      }

      AtomicRMWInst *FoldWith = nullptr;
      unsigned FoldIndex = 0;
      for (auto I = 0u; I < Pending.size(); ++I) {
        if (canCoalesceAtomicFAdd(Pending[I].first, RMW, AA)) {
          FoldWith = Pending[I].first;
          FoldIndex = I;
          break;
        }
      }

      if (!FoldWith) {
        invalidatePendingAtomicFAdds(RMW, Pending, AA);
        Pending.push_back({RMW, MemoryLocation::get(RMW)});
        continue;
      }

      IRBuilder<> Builder(RMW);
      Value *Folded =
          Builder.CreateFAdd(FoldWith->getValOperand(), RMW->getValOperand());
      auto *NewRMW = Builder.CreateAtomicRMW(
          RMW->getOperation(), RMW->getPointerOperand(), Folded,
          RMW->getAlign(), RMW->getOrdering(), RMW->getSyncScopeID());
      NewRMW->copyMetadata(*RMW);
      NewRMW->setDebugLoc(RMW->getDebugLoc());

      FoldWith->eraseFromParent();
      RMW->eraseFromParent();
      Pending.erase(Pending.begin() + FoldIndex);

      invalidatePendingAtomicFAdds(NewRMW, Pending, AA);
      Pending.push_back({NewRMW, MemoryLocation::get(NewRMW)});
      Changed = true;
    }
  }
  return Changed;
}

struct AtomicTailGroup {
  BasicBlock *Succ;
  SmallVector<BasicBlock *, 4> Preds;
  SmallVector<SmallVector<AtomicRMWInst *, 8>, 4> Atomics;
};

static bool sameTailMergeAtomicShape(AtomicRMWInst *LHS, AtomicRMWInst *RHS) {
  return LHS->getOperation() == RHS->getOperation() &&
         LHS->getOrdering() == RHS->getOrdering() &&
         LHS->getSyncScopeID() == RHS->getSyncScopeID() &&
         LHS->getAlign() == RHS->getAlign() &&
         LHS->getPointerOperand()->getType() ==
             RHS->getPointerOperand()->getType() &&
         LHS->getValOperand()->getType() == RHS->getValOperand()->getType();
}

static bool sameTailMergeShape(ArrayRef<AtomicRMWInst *> LHS,
                               ArrayRef<AtomicRMWInst *> RHS) {
  if (LHS.size() != RHS.size())
    return false;
  for (auto I = 0u; I < LHS.size(); ++I)
    if (!sameTailMergeAtomicShape(LHS[I], RHS[I]))
      return false;
  return true;
}

static bool
canMoveAtomicAfterInstructions(AtomicRMWInst *RMW,
                               ArrayRef<Instruction *> LaterNonAtomicInsts,
                               AAResults &AA) {
  MemoryLocation Loc = MemoryLocation::get(RMW);
  for (Instruction *I : LaterNonAtomicInsts) {
    if (isa<DbgInfoIntrinsic>(I))
      continue;
    if (isa<AtomicRMWInst>(I) || isa<AtomicCmpXchgInst>(I) || isa<FenceInst>(I))
      return false;
    if (I->mayHaveSideEffects() && !I->mayReadOrWriteMemory())
      return false;
    if (instructionMayAccessLocation(I, Loc, AA))
      return false;
  }
  return true;
}

static bool
collectMovableCudaShadowAtomics(BasicBlock *BB, AAResults &AA,
                                SmallVectorImpl<AtomicRMWInst *> &RMWs) {
  auto *BI = dyn_cast<BranchInst>(BB->getTerminator());
  if (!BI || !BI->isUnconditional())
    return false;

  SmallVector<Instruction *, 16> LaterNonAtomicInsts;
  for (Instruction &I : llvm::reverse(*BB)) {
    if (&I == BB->getTerminator() || isa<DbgInfoIntrinsic>(&I))
      continue;

    if (auto *RMW = dyn_cast<AtomicRMWInst>(&I);
        RMW && isCudaAtomicFAddCandidate(RMW)) {
      if (!canMoveAtomicAfterInstructions(RMW, LaterNonAtomicInsts, AA))
        return false;
      RMWs.push_back(RMW);
      continue;
    }

    LaterNonAtomicInsts.push_back(&I);
  }

  std::reverse(RMWs.begin(), RMWs.end());
  return !RMWs.empty();
}

static bool
compatibleTailMergeGroup(const SmallVectorImpl<AtomicRMWInst *> &Base,
                         const SmallVectorImpl<AtomicRMWInst *> &RMWs) {
  return sameTailMergeShape(Base, RMWs);
}

static bool findCudaShadowAtomicTailGroup(BasicBlock *Succ, AAResults &AA,
                                          AtomicTailGroup &Group) {
  if (!EnzymeEnableCudaAtomicTailMerge || pred_empty(Succ) ||
      isa<PHINode>(Succ->begin()))
    return false;

  SmallVector<BasicBlock *, 8> Preds(predecessors(Succ));
  SmallVector<SmallVector<AtomicRMWInst *, 8>, 8> PredAtomics;
  for (BasicBlock *Pred : Preds) {
    SmallVector<AtomicRMWInst *, 8> RMWs;
    if (collectMovableCudaShadowAtomics(Pred, AA, RMWs))
      PredAtomics.push_back(std::move(RMWs));
    else
      PredAtomics.push_back({});
  }

  for (auto I = 0u; I < Preds.size(); ++I) {
    if (PredAtomics[I].empty())
      continue;

    SmallVector<unsigned, 4> Matching;
    Matching.push_back(I);
    for (auto J = I + 1; J < Preds.size(); ++J) {
      if (!PredAtomics[J].empty() &&
          compatibleTailMergeGroup(PredAtomics[I], PredAtomics[J]))
        Matching.push_back(J);
    }

    if (Matching.size() < 2)
      continue;

    Group.Succ = Succ;
    for (unsigned Idx : Matching) {
      Group.Preds.push_back(Preds[Idx]);
      Group.Atomics.push_back(std::move(PredAtomics[Idx]));
    }
    return true;
  }

  return false;
}

static void copyCommonAtomicMetadata(AtomicRMWInst *NewRMW,
                                     ArrayRef<AtomicRMWInst *> OldRMWs) {
  NewRMW->setMetadata("enzyme_shadow_atomic",
                      MDNode::get(NewRMW->getContext(), {}));
  for (unsigned Kind : {LLVMContext::MD_tbaa, LLVMContext::MD_tbaa_struct}) {
    MDNode *MD = OldRMWs.front()->getMetadata(Kind);
    bool AllSame = true;
    for (AtomicRMWInst *RMW : OldRMWs.drop_front()) {
      if (RMW->getMetadata(Kind) != MD) {
        AllSame = false;
        break;
      }
    }
    if (AllSame)
      NewRMW->setMetadata(Kind, MD);
  }
}

static bool applyCudaShadowAtomicTailMerge(const AtomicTailGroup &Group) {
  LLVMContext &Ctx = Group.Succ->getContext();
  BasicBlock *MergeBB = BasicBlock::Create(Ctx, "enzyme.atomic.merge",
                                           Group.Succ->getParent(), Group.Succ);
  BranchInst::Create(Group.Succ, MergeBB);

  for (BasicBlock *Pred : Group.Preds) {
    auto *BI = cast<BranchInst>(Pred->getTerminator());
    BI->setSuccessor(0, MergeBB);
  }

  IRBuilder<> Builder(MergeBB->getTerminator());
  unsigned NumAtomics = Group.Atomics.front().size();
  SmallVector<PHINode *, 8> PtrPhis;
  SmallVector<PHINode *, 8> ValPhis;
  SmallVector<SmallVector<AtomicRMWInst *, 4>, 8> OldRMWGroups;

  for (unsigned I = 0; I < NumAtomics; ++I) {
    AtomicRMWInst *Template = Group.Atomics.front()[I];
    auto *PtrPhi = PHINode::Create(Template->getPointerOperand()->getType(),
                                   Group.Preds.size(), "atomic.ptr",
                                   MergeBB->getTerminator());
    auto *ValPhi = PHINode::Create(Template->getValOperand()->getType(),
                                   Group.Preds.size(), "atomic.val",
                                   MergeBB->getTerminator());

    SmallVector<AtomicRMWInst *, 4> OldRMWs;
    for (auto PredIdx = 0u; PredIdx < Group.Preds.size(); ++PredIdx) {
      AtomicRMWInst *RMW = Group.Atomics[PredIdx][I];
      PtrPhi->addIncoming(RMW->getPointerOperand(), Group.Preds[PredIdx]);
      ValPhi->addIncoming(RMW->getValOperand(), Group.Preds[PredIdx]);
      OldRMWs.push_back(RMW);
    }

    PtrPhis.push_back(PtrPhi);
    ValPhis.push_back(ValPhi);
    OldRMWGroups.push_back(std::move(OldRMWs));
  }

  for (unsigned I = 0; I < NumAtomics; ++I) {
    AtomicRMWInst *Template = Group.Atomics.front()[I];
    auto *NewRMW = Builder.CreateAtomicRMW(
        Template->getOperation(), PtrPhis[I], ValPhis[I], Template->getAlign(),
        Template->getOrdering(), Template->getSyncScopeID());
    copyCommonAtomicMetadata(NewRMW, OldRMWGroups[I]);
  }

  for (const auto &RMWs : Group.Atomics)
    for (AtomicRMWInst *RMW : RMWs)
      RMW->eraseFromParent();

  return true;
}

static bool mergeCudaShadowAtomicFAddTailsImpl(Function &F, AAResults &AA) {
  if (!EnzymeEnableCudaRepeatedLoads || !EnzymeEnableCudaAtomicTailMerge ||
      !isNVPTXModule(F.getParent()))
    return false;

  bool Changed = false;
  SmallVector<BasicBlock *, 8> Blocks;
  for (BasicBlock &BB : F)
    Blocks.push_back(&BB);

  for (BasicBlock *Succ : Blocks) {
    AtomicTailGroup Group;
    if (findCudaShadowAtomicTailGroup(Succ, AA, Group))
      Changed |= applyCudaShadowAtomicTailMerge(Group);
  }

  return Changed;
}

// Collect memory operations (loads, stores) and calls for a given pointer value
// Returns false if the value has uses that prevent optimization
// Nocapture calls are only rejected (causing failure) if Calls is empty on
// entry If Calls is non-empty on entry, nocapture calls are collected
static bool
collectMemoryOps(Value *Arg, const DataLayout &DL,
                 SmallVectorImpl<std::pair<StoreInst *, APInt>> &Stores,
                 SmallVectorImpl<std::pair<LoadInst *, APInt>> &Loads,
                 SmallVectorImpl<std::pair<CallInst *, APInt>> &Calls) {
  // WorkList tracks (Value*, Offset from Arg)
  SmallVector<std::pair<Value *, APInt>, 16> ToProcess;
  SmallPtrSet<Value *, 16> Visited;

  APInt ZeroOffset(DL.getIndexTypeSizeInBits(Arg->getType()), 0);
  ToProcess.push_back({Arg, ZeroOffset});

  while (!ToProcess.empty()) {
    auto [V, CurrentOffset] = ToProcess.pop_back_val();

    // Skip if already visited
    if (!Visited.insert(V).second)
      continue;

    for (Use &U : V->uses()) {
      User *Usr = U.getUser();
      if (auto *LI = dyn_cast<LoadInst>(Usr)) {
        Loads.push_back({LI, CurrentOffset});
      } else if (auto *SI = dyn_cast<StoreInst>(Usr)) {
        // Check if this is a store TO the pointer (not storing the pointer
        // value)
        if (SI->getPointerOperand() == V) {
          Stores.push_back({SI, CurrentOffset});
        } else {
          // Pointer value is being stored somewhere - reject this argument
          return false;
        }
      } else if (auto *GEP = dyn_cast<GetElementPtrInst>(Usr)) {
        // Compute the offset for this GEP
        APInt GEPOffset(DL.getIndexTypeSizeInBits(GEP->getType()), 0);
        if (!GEP->accumulateConstantOffset(DL, GEPOffset)) {
          // Cannot compute constant offset - reject this argument
          return false;
        }

        APInt NewOffset = CurrentOffset + GEPOffset;
        ToProcess.push_back({GEP, NewOffset});
      } else if (auto *CI = dyn_cast<CastInst>(Usr)) {
        // Casts don't change offset
        ToProcess.push_back({CI, CurrentOffset});
      } else if (auto *Call = dyn_cast<CallInst>(Usr)) {
        // Get the argument index from the Use
        unsigned ArgIdx = U.getOperandNo();
        if (isNoCapture(Call, ArgIdx)) {
          Calls.push_back({Call, CurrentOffset});
        } else {
          // Call that may capture - reject this argument
          return false;
        }
      } else {
        // Unknown use - reject this argument
        return false;
      }
    }
  }

  return true;
}

// Main optimization function
bool simplifyGVN(Function &F, DominatorTree &DT, const DataLayout &DL) {
  bool Changed = false;

  // Find noalias arguments
  SmallVector<Value *, 4> CandidateArgs;
  for (Argument &Arg : F.args()) {
    if (Arg.getType()->isPointerTy() && Arg.hasNoAliasAttr()) {
      CandidateArgs.push_back(&Arg);
    }
  }

  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      if (isa<AllocaInst>(&I)) {
        CandidateArgs.push_back(&I);
      }
    }
  }

  if (CandidateArgs.empty())
    return false;

  // For each candidate argument, collect stores and loads with their offsets
  for (Value *Arg : CandidateArgs) {
    // Collect all stores and loads to this argument with offsets
    SmallVector<std::pair<StoreInst *, APInt>, 8> Stores;
    SmallVector<std::pair<LoadInst *, APInt>, 8> Loads;
    SmallVector<std::pair<CallInst *, APInt>, 8> Calls;

    // First pass: strict collection (no nocapture calls) for store-load
    // forwarding (pass empty Calls to reject nocapture calls)
    if (!collectMemoryOps(Arg, DL, Stores, Loads, Calls)) {
      // Argument has uses that prevent optimization
      continue;
    }

    APInt ZeroOffset(DL.getIndexTypeSizeInBits(Arg->getType()), 0);

    // Try to forward {stores, previous loads} to loads using simplified
    // algorithm
    for (auto &[LI, LoadOffset] : Loads) {
      uint64_t LoadSize = DL.getTypeStoreSize(LI->getType());

      // Step 1: Find all stores that may alias with this load
      SmallVector<std::tuple<Instruction *, APInt, uint64_t>, 8> AliasingStores;
      for (auto &[SI, StoreOffset] : Stores) {
        uint64_t StoreSize =
            DL.getTypeStoreSize(SI->getValueOperand()->getType());
        if (memoryRangesAlias(LoadOffset, LoadSize, StoreOffset, StoreSize)) {
          AliasingStores.push_back({SI, StoreOffset, StoreSize});
        }
      }

      // Assume the call can touch any memory, so just set it to directly
      // overlap.
      for (auto &[CI, CallOffset] : Calls) {
        AliasingStores.push_back({CI, LoadOffset, LoadSize});
      }

      // Step 2: Filter to dominating + covering stores
      // Tuple of instruction storing, offset in the instruction, and the
      // equivalent value.
      SmallVector<std::tuple<Instruction *, APInt, Value *>, 8>
          DominatingCoveringStores;
      for (auto &[I, StoreOffset, StoreSize] : AliasingStores) {
        if (auto SI = dyn_cast<StoreInst>(I))
          if (dominatesAndCovers(SI, LI, StoreOffset, LoadOffset, LoadSize, DL,
                                 DT)) {
            DominatingCoveringStores.push_back(
                {SI, StoreOffset, SI->getValueOperand()});
          }
      }

      // Step 3: If only one aliasing store and it's dominating+covering,
      // forward
      if (AliasingStores.size() == 1 && DominatingCoveringStores.size() == 1) {
        Instruction *SI = std::get<0>(DominatingCoveringStores[0]);
        APInt StoreOffset = std::get<1>(DominatingCoveringStores[0]);

        IRBuilder<> Builder(LI);
        Value *StoredVal = std::get<2>(DominatingCoveringStores[0]);
        Value *ExtractedVal =
            extractValue(Builder, StoredVal, LI->getType(), DL, LoadOffset,
                         StoreOffset, LoadSize);

        if (ExtractedVal) {
          LLVM_DEBUG(dbgs() << "SimpleGVN: Forwarding (single alias)\n"
                            << "  Store: " << *SI << "\n"
                            << "  Load:  " << *LI << "\n");
          LI->replaceAllUsesWith(ExtractedVal);
          LI->eraseFromParent();
          LI = nullptr;
          Changed = true;
        }
        continue;
      }

      for (auto &[LI2, LoadOffset2] : Loads) {
        if (!LI2 || LI2 == LI)
          continue;
        if (dominatesAndCovers(LI2, LI, LoadOffset2, LoadOffset, LoadSize, DL,
                               DT)) {
          DominatingCoveringStores.emplace_back(LI2, LoadOffset2, LI2);
        }
      }

      // Step 4: If no dominating+covering stores, bail
      if (DominatingCoveringStores.empty()) {
        continue;
      }

      // Step 5: Build map of last store in each block before LI
      DenseMap<BasicBlock *, std::tuple<Instruction *, APInt, uint64_t>>
          LastStoreInBlockBeforeLI;
      for (auto &[SI, StoreOffset, Size] : AliasingStores) {
        BasicBlock *BB = SI->getParent();
        if (BB == LI->getParent()) {
          // Only consider stores before LI in the same block
          if (SI->comesBefore(LI)) {
            auto &Entry = LastStoreInBlockBeforeLI[BB];
            if (!std::get<0>(Entry) || std::get<0>(Entry)->comesBefore(SI)) {
              Entry = {SI, StoreOffset, Size};
            }
          }
        } else {
          // For other blocks, take the last store in the block
          auto &Entry = LastStoreInBlockBeforeLI[BB];
          if (!std::get<0>(Entry) || std::get<0>(Entry)->comesBefore(SI)) {
            Entry = {SI, StoreOffset, Size};
          }
        }
      }

      // Step 6: Check if LI's parent block has a dominating+covering store
      BasicBlock *LIBlock = LI->getParent();
      auto It = LastStoreInBlockBeforeLI.find(LIBlock);
      if (It != LastStoreInBlockBeforeLI.end()) {
        Instruction *SI = std::get<0>(It->second);

        for (auto &&[DCS, StoreOffset, StoredVal] : DominatingCoveringStores) {
          if (SI == DCS ||
              (DCS->getParent() == LI->getParent() && SI->comesBefore(DCS))) {

            IRBuilder<> Builder(LI);
            Value *ExtractedVal =
                extractValue(Builder, StoredVal, LI->getType(), DL, LoadOffset,
                             StoreOffset, LoadSize);

            if (ExtractedVal) {
              LLVM_DEBUG(dbgs() << "SimpleGVN: Forwarding (same block)\n"
                                << "  Store: " << *DCS << "\n"
                                << "  Load:  " << *LI << "\n");
              LI->replaceAllUsesWith(ExtractedVal);
              LI->eraseFromParent();
              LI = nullptr;
              Changed = true;
              break;
            }
          }
        }
        continue;
      } else {
        for (auto &&[DCS, StoreOffset, StoredVal] : DominatingCoveringStores) {
          if (DCS->getParent() == LI->getParent()) {

            IRBuilder<> Builder(LI);
            Value *ExtractedVal =
                extractValue(Builder, StoredVal, LI->getType(), DL, LoadOffset,
                             StoreOffset, LoadSize);

            if (ExtractedVal) {
              LLVM_DEBUG(dbgs() << "SimpleGVN: Forwarding (same block)\n"
                                << "  Store: " << *DCS << "\n"
                                << "  Load:  " << *LI << "\n");
              LI->replaceAllUsesWith(ExtractedVal);
              LI->eraseFromParent();
              LI = nullptr;
              Changed = true;
              break;
            }
          }
        }
        if (LI == nullptr) {
          continue;
        }
      }

      // Step 7: BFS backwards from LI's parent block
      SmallPtrSet<BasicBlock *, 32> Visited;
      SmallVector<BasicBlock *, 16> Worklist;
      StoreInst *Candidate = nullptr;
      APInt CandidateOffset = ZeroOffset;

      // Start with predecessors of LI's block
      for (BasicBlock *Pred : predecessors(LIBlock)) {
        if (Visited.insert(Pred).second)
          Worklist.push_back(Pred);
      }

      while (!Worklist.empty()) {
        BasicBlock *BB = Worklist.pop_back_val();

        auto It = LastStoreInBlockBeforeLI.find(BB);
        if (It != LastStoreInBlockBeforeLI.end()) {
          StoreInst *SI = dyn_cast<StoreInst>(std::get<0>(It->second));
          APInt StoreOffset = std::get<1>(It->second);

          if (!SI || !dominatesAndCovers(SI, LI, StoreOffset, LoadOffset,
                                         LoadSize, DL, DT)) {
            // Non-dominating+covering store on path, bail
            Candidate = nullptr;
            break;
          }

          // Found dominating+covering store
          if (!Candidate) {
            Candidate = SI;
            CandidateOffset = StoreOffset;
          } else if (Candidate != SI) {
            // Multiple different candidates, bail
            Candidate = nullptr;
            break;
          }
        }

        // Continue BFS
        for (BasicBlock *Pred : predecessors(BB)) {
          if (Visited.insert(Pred).second)
            Worklist.push_back(Pred);
        }
      }

      // Step 8: If unique candidate found, forward
      if (Candidate) {
        IRBuilder<> Builder(LI);
        Value *StoredVal = Candidate->getValueOperand();
        Value *ExtractedVal =
            extractValue(Builder, StoredVal, LI->getType(), DL, LoadOffset,
                         CandidateOffset, LoadSize);

        if (ExtractedVal) {
          LLVM_DEBUG(dbgs() << "SimpleGVN: Forwarding (BFS candidate)\n"
                            << "  Store: " << *Candidate << "\n"
                            << "  Load:  " << *LI << "\n");
          LI->replaceAllUsesWith(ExtractedVal);
          LI->eraseFromParent();
          LI = nullptr;
          Changed = true;
        }
      }
    }
  }
  return Changed;
}

class SimpleGVN final : public FunctionPass {
public:
  static char ID;
  SimpleGVN() : FunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<DominatorTreeWrapperPass>();
    if (EnzymeEnableCudaRepeatedLoads)
      AU.addRequired<AAResultsWrapperPass>();
  }

  bool runOnFunction(Function &F) override {
    auto &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();
    const DataLayout &DL = F.getParent()->getDataLayout();
    bool Changed = false;
    if (EnzymeEnableCudaRepeatedLoads) {
      auto &AA = getAnalysis<AAResultsWrapperPass>().getAAResults();
      Changed |= simplifyRepeatedGlobalLoadsImpl(F, AA);
      Changed |= coalesceRepeatedCudaAtomicFAddsImpl(F, AA);
      Changed |= mergeCudaShadowAtomicFAddTailsImpl(F, AA);
    }
    Changed |= simplifyGVN(F, DT, DL);
    return Changed;
  }
};

} // namespace

bool simplifyRepeatedGlobalLoads(Function &F, AAResults &AA) {
  return simplifyRepeatedGlobalLoadsImpl(F, AA);
}

bool coalesceRepeatedCudaAtomicFAdds(Function &F, AAResults &AA) {
  return coalesceRepeatedCudaAtomicFAddsImpl(F, AA);
}

bool mergeCudaShadowAtomicFAddTails(Function &F, AAResults &AA) {
  return mergeCudaShadowAtomicFAddTailsImpl(F, AA);
}

FunctionPass *createSimpleGVNPass() { return new SimpleGVN(); }

extern "C" void LLVMAddSimpleGVNPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createSimpleGVNPass());
}

char SimpleGVN::ID = 0;

static RegisterPass<SimpleGVN> X("simple-gvn",
                                 "GVN-like load forwarding optimization");

SimpleGVNNewPM::Result SimpleGVNNewPM::run(Function &F,
                                           FunctionAnalysisManager &FAM) {
  bool Changed = false;
  const DataLayout &DL = F.getParent()->getDataLayout();
  if (EnzymeEnableCudaRepeatedLoads) {
    auto &AA = FAM.getResult<AAManager>(F);
    Changed |= simplifyRepeatedGlobalLoadsImpl(F, AA);
    Changed |= coalesceRepeatedCudaAtomicFAddsImpl(F, AA);
    Changed |= mergeCudaShadowAtomicFAddTailsImpl(F, AA);
  }
  Changed |= simplifyGVN(F, FAM.getResult<DominatorTreeAnalysis>(F), DL);
  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}

llvm::AnalysisKey SimpleGVNNewPM::Key;
