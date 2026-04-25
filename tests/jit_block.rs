//! Block JIT tests — verify correctness of compiled loops and branches.

#![cfg(feature = "jit")]

use fusevm::{ChunkBuilder, JitCompiler, Op};

#[test]
fn block_jit_sum_loop() {
    // sum = 0; i = 0; while (i < 100) { sum += i; i++ } → sum = 4950
    let mut b = ChunkBuilder::new();
    b.emit(Op::PushFrame, 1);
    b.emit(Op::LoadInt(0), 1);
    b.emit(Op::SetSlot(0), 1); // sum = 0
    b.emit(Op::LoadInt(0), 1);
    b.emit(Op::SetSlot(1), 1); // i = 0
    // ip=5: loop body
    b.emit(Op::GetSlot(0), 1);
    b.emit(Op::GetSlot(1), 1);
    b.emit(Op::Add, 1);
    b.emit(Op::SetSlot(0), 1); // sum += i
    b.emit(Op::PreIncSlotVoid(1), 1); // i++
    b.emit(Op::SlotLtIntJumpIfFalse(1, 100, 12), 1);
    b.emit(Op::Jump(5), 1);
    // ip=12: exit
    b.emit(Op::GetSlot(0), 1);
    let chunk = b.build();

    let jit = JitCompiler::new();
    assert!(jit.is_block_eligible(&chunk));

    let mut slots = vec![0i64; 4];
    let result = jit.try_run_block(&chunk, &mut slots).unwrap();
    assert_eq!(result, 4950);
}

#[test]
fn block_jit_accum_sum_loop() {
    // AccumSumLoop(0, 1, 1000) → sum = 499500
    let mut b = ChunkBuilder::new();
    b.emit(Op::PushFrame, 1);
    b.emit(Op::LoadInt(0), 1);
    b.emit(Op::SetSlot(0), 1);
    b.emit(Op::LoadInt(0), 1);
    b.emit(Op::SetSlot(1), 1);
    b.emit(Op::AccumSumLoop(0, 1, 1000), 1);
    b.emit(Op::GetSlot(0), 1);
    let chunk = b.build();

    let jit = JitCompiler::new();
    assert!(jit.is_block_eligible(&chunk));

    let mut slots = vec![0i64; 4];
    let result = jit.try_run_block(&chunk, &mut slots).unwrap();
    assert_eq!(result, 499500);
}

#[test]
fn block_jit_conditional() {
    // if (1) { slot[0] = 42 } else { slot[0] = 99 } → 42
    let mut b = ChunkBuilder::new();
    b.emit(Op::PushFrame, 1);
    b.emit(Op::LoadInt(1), 1);        // condition = true
    b.emit(Op::JumpIfFalse(6), 1);    // if false, goto else
    // then:
    b.emit(Op::LoadInt(42), 1);
    b.emit(Op::SetSlot(0), 1);
    b.emit(Op::Jump(8), 1);           // skip else
    // ip=6: else
    b.emit(Op::LoadInt(99), 1);
    b.emit(Op::SetSlot(0), 1);
    // ip=8: after
    b.emit(Op::GetSlot(0), 1);
    let chunk = b.build();

    let jit = JitCompiler::new();
    assert!(jit.is_block_eligible(&chunk));

    let mut slots = vec![0i64; 4];
    let result = jit.try_run_block(&chunk, &mut slots).unwrap();
    assert_eq!(result, 42);
}

#[test]
fn block_jit_conditional_false() {
    // if (0) { slot[0] = 42 } else { slot[0] = 99 } → 99
    let mut b = ChunkBuilder::new();
    b.emit(Op::PushFrame, 1);
    b.emit(Op::LoadInt(0), 1);        // condition = false
    b.emit(Op::JumpIfFalse(6), 1);
    b.emit(Op::LoadInt(42), 1);
    b.emit(Op::SetSlot(0), 1);
    b.emit(Op::Jump(8), 1);
    // ip=6:
    b.emit(Op::LoadInt(99), 1);
    b.emit(Op::SetSlot(0), 1);
    // ip=8:
    b.emit(Op::GetSlot(0), 1);
    let chunk = b.build();

    let jit = JitCompiler::new();
    let mut slots = vec![0i64; 4];
    let result = jit.try_run_block(&chunk, &mut slots).unwrap();
    assert_eq!(result, 99);
}

#[test]
fn block_jit_fused_backedge() {
    // i = 0; sum = 0; loop { sum += i; if (++i >= 50) break } → sum = 1225
    let mut b = ChunkBuilder::new();
    b.emit(Op::PushFrame, 1);
    b.emit(Op::LoadInt(0), 1);
    b.emit(Op::SetSlot(0), 1); // sum = 0
    b.emit(Op::LoadInt(0), 1);
    b.emit(Op::SetSlot(1), 1); // i = 0
    // ip=5: body
    b.emit(Op::AddAssignSlotVoid(0, 1), 1); // sum += i
    b.emit(Op::SlotIncLtIntJumpBack(1, 50, 5), 1); // i++; if i < 50 goto 5
    // exit
    b.emit(Op::GetSlot(0), 1);
    let chunk = b.build();

    let jit = JitCompiler::new();
    assert!(jit.is_block_eligible(&chunk));

    let mut slots = vec![0i64; 4];
    let result = jit.try_run_block(&chunk, &mut slots).unwrap();
    assert_eq!(result, 1225);
}

#[test]
fn block_jit_ineligible_with_print() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(42), 1);
    b.emit(Op::Print(1), 1); // Print is not block-eligible
    let chunk = b.build();

    let jit = JitCompiler::new();
    assert!(!jit.is_block_eligible(&chunk));
}

#[test]
fn block_jit_slots_written_back() {
    // Verify slots are modified in-place after the JIT runs
    let mut b = ChunkBuilder::new();
    b.emit(Op::PushFrame, 1);
    b.emit(Op::LoadInt(0), 1);
    b.emit(Op::SetSlot(0), 1);
    b.emit(Op::LoadInt(0), 1);
    b.emit(Op::SetSlot(1), 1);
    b.emit(Op::AccumSumLoop(0, 1, 10), 1);
    b.emit(Op::GetSlot(0), 1);
    let chunk = b.build();

    let jit = JitCompiler::new();
    let mut slots = vec![0i64; 4];
    let _ = jit.try_run_block(&chunk, &mut slots);
    assert_eq!(slots[0], 45); // sum 0..10
    assert_eq!(slots[1], 10); // i after loop
}
