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
    let result = jit.try_run_block_eager(&chunk, &mut slots).unwrap();
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
    let result = jit.try_run_block_eager(&chunk, &mut slots).unwrap();
    assert_eq!(result, 499500);
}

#[test]
fn block_jit_conditional() {
    // if (1) { slot[0] = 42 } else { slot[0] = 99 } → 42
    let mut b = ChunkBuilder::new();
    b.emit(Op::PushFrame, 1);
    b.emit(Op::LoadInt(1), 1); // condition = true
    b.emit(Op::JumpIfFalse(6), 1); // if false, goto else
                                   // then:
    b.emit(Op::LoadInt(42), 1);
    b.emit(Op::SetSlot(0), 1);
    b.emit(Op::Jump(8), 1); // skip else
                            // ip=6: else
    b.emit(Op::LoadInt(99), 1);
    b.emit(Op::SetSlot(0), 1);
    // ip=8: after
    b.emit(Op::GetSlot(0), 1);
    let chunk = b.build();

    let jit = JitCompiler::new();
    assert!(jit.is_block_eligible(&chunk));

    let mut slots = vec![0i64; 4];
    let result = jit.try_run_block_eager(&chunk, &mut slots).unwrap();
    assert_eq!(result, 42);
}

#[test]
fn block_jit_conditional_false() {
    // if (0) { slot[0] = 42 } else { slot[0] = 99 } → 99
    let mut b = ChunkBuilder::new();
    b.emit(Op::PushFrame, 1);
    b.emit(Op::LoadInt(0), 1); // condition = false
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
    let result = jit.try_run_block_eager(&chunk, &mut slots).unwrap();
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
    let result = jit.try_run_block_eager(&chunk, &mut slots).unwrap();
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
fn partial_jit_finds_eligible_region() {
    // A chunk with mixed eligible/ineligible ops:
    // [PushFrame, LoadInt(0), SetSlot(0), LoadInt(0), SetSlot(1),  // eligible: ip 0..5
    //  AccumSumLoop, GetSlot(0),                                   // eligible: continues
    //  Print(1),                                                    // INELIGIBLE: ip 7
    //  GetSlot(0)]                                                  // eligible (size 1, too small)
    let mut b = ChunkBuilder::new();
    b.emit(Op::PushFrame, 1);
    b.emit(Op::LoadInt(0), 1);
    b.emit(Op::SetSlot(0), 1);
    b.emit(Op::LoadInt(0), 1);
    b.emit(Op::SetSlot(1), 1);
    b.emit(Op::AccumSumLoop(0, 1, 100), 1);
    b.emit(Op::GetSlot(0), 1);
    b.emit(Op::Print(1), 1); // ineligible
    b.emit(Op::GetSlot(0), 1);
    let chunk = b.build();

    let jit = JitCompiler::new();
    assert!(!jit.is_block_eligible(&chunk));
    let region = jit
        .find_jit_region(&chunk)
        .expect("should find eligible region");
    assert_eq!(region, (0, 7));
}

#[test]
fn partial_jit_compiles_extracted_region() {
    // Same as above — extract the eligible region and JIT-compile it.
    let mut b = ChunkBuilder::new();
    b.emit(Op::PushFrame, 1);
    b.emit(Op::LoadInt(0), 1);
    b.emit(Op::SetSlot(0), 1);
    b.emit(Op::LoadInt(0), 1);
    b.emit(Op::SetSlot(1), 1);
    b.emit(Op::AccumSumLoop(0, 1, 100), 1);
    b.emit(Op::GetSlot(0), 1);
    b.emit(Op::Print(1), 1);
    let chunk = b.build();

    let jit = JitCompiler::new();
    let (start, end) = jit.find_jit_region(&chunk).unwrap();
    let sub_chunk = jit.extract_region(&chunk, start, end);

    assert!(jit.is_block_eligible(&sub_chunk));
    let mut slots = vec![0i64; 4];
    let result = jit.try_run_block_eager(&sub_chunk, &mut slots).unwrap();
    assert_eq!(result, 4950); // sum 0..100
}

#[test]
fn partial_jit_rebases_jumps() {
    // Region with internal jumps — verify they're rebased to local indices.
    let mut b = ChunkBuilder::new();
    b.emit(Op::PushFrame, 1);
    b.emit(Op::LoadInt(0), 1);
    b.emit(Op::SetSlot(0), 1);
    b.emit(Op::LoadInt(1), 1); // condition
    b.emit(Op::JumpIfFalse(7), 1); // ip=4, target ip=7
    b.emit(Op::LoadInt(42), 1);
    b.emit(Op::SetSlot(0), 1);
    // ip=7
    b.emit(Op::GetSlot(0), 1);
    let chunk = b.build();

    let jit = JitCompiler::new();
    let (start, end) = jit.find_jit_region(&chunk).unwrap();
    let sub_chunk = jit.extract_region(&chunk, start, end);

    // Find the JumpIfFalse in sub_chunk and verify target was rebased
    for op in &sub_chunk.ops {
        if let Op::JumpIfFalse(t) = op {
            assert_eq!(*t, 7 - start);
        }
    }
    let mut slots = vec![0i64; 4];
    let result = jit.try_run_block_eager(&sub_chunk, &mut slots).unwrap();
    assert_eq!(result, 42);
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
    let _ = jit.try_run_block_eager(&chunk, &mut slots);
    assert_eq!(slots[0], 45); // sum 0..10
    assert_eq!(slots[1], 10); // i after loop
}
