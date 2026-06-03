//! Block JIT tests — verify correctness of compiled loops and branches.

#![cfg(feature = "jit")]

use fusevm::{ChunkBuilder, JitCompiler, Op};

#[test]
fn block_jit_awk_sin_float_slot_matches_libm() {
    use fusevm::SlotKind;
    // slot0 = sin(slot0) where slot0 starts as f64 0.5 → sin(0.5).
    let mut b = ChunkBuilder::new();
    b.emit(Op::GetSlot(0), 1);
    b.emit(Op::AwkSin, 1);
    b.emit(Op::Dup, 1);
    b.emit(Op::SetSlot(0), 1);
    b.emit(Op::Pop, 1);
    let chunk = b.build();

    let jit = JitCompiler::new();
    assert!(
        jit.is_block_eligible(&chunk),
        "AwkSin must be block-eligible"
    );

    let kinds = [SlotKind::Float];
    let mut slots = vec![0.5f64.to_bits() as i64; 1];
    let _ = jit
        .try_run_block_eager_kinded(&chunk, &mut slots, &kinds)
        .expect("AwkSin float-slot chunk must compile");
    assert_eq!(
        f64::from_bits(slots[0] as u64),
        0.5f64.sin(),
        "block JIT sin libcall must match libm"
    );
}

#[test]
fn block_jit_awk_atan2_float_slots_match_libm() {
    use fusevm::SlotKind;
    // slot0 = atan2(slot0, slot1); awk pushes y then x, so GetSlot(0)=y first,
    // GetSlot(1)=x second (x on top), matching Op::AwkAtan2's pop order.
    let mut b = ChunkBuilder::new();
    b.emit(Op::GetSlot(0), 1);
    b.emit(Op::GetSlot(1), 1);
    b.emit(Op::AwkAtan2, 1);
    b.emit(Op::Dup, 1);
    b.emit(Op::SetSlot(0), 1);
    b.emit(Op::Pop, 1);
    let chunk = b.build();

    let jit = JitCompiler::new();
    assert!(jit.is_block_eligible(&chunk));

    let kinds = [SlotKind::Float, SlotKind::Float];
    let mut slots = vec![1.0f64.to_bits() as i64, 2.0f64.to_bits() as i64];
    let _ = jit
        .try_run_block_eager_kinded(&chunk, &mut slots, &kinds)
        .expect("AwkAtan2 float-slot chunk must compile");
    assert_eq!(
        f64::from_bits(slots[0] as u64),
        1.0f64.atan2(2.0),
        "block JIT atan2 libcall must match libm (y=1, x=2)"
    );
}

#[test]
fn block_jit_typed_returns_exact_float_result() {
    use fusevm::{BlockNum, SlotKind};
    // Chunk whose RESULT (top of stack at return) is a float:
    //   slot0 (f64) * 1.5 + 2.0, left on the operand stack.
    let mut b = ChunkBuilder::new();
    b.emit(Op::GetSlot(0), 1);
    b.emit(Op::LoadFloat(1.5), 1);
    b.emit(Op::Mul, 1);
    b.emit(Op::LoadFloat(2.0), 1);
    b.emit(Op::Add, 1);
    let chunk = b.build();

    let jit = JitCompiler::new();
    assert!(jit.is_block_eligible(&chunk));

    let kinds = [SlotKind::Float];
    let mut slots = vec![4.0f64.to_bits() as i64; 1];
    let out = jit
        .try_run_block_eager_typed_kinded(&chunk, &mut slots, &kinds)
        .expect("float-result chunk must compile");
    match out {
        BlockNum::Float(v) => assert_eq!(v, 4.0 * 1.5 + 2.0, "exact float result preserved"),
        BlockNum::Int(n) => panic!("expected Float, got Int({n})"),
    }

    // The plain i64 entry point must still truncate the same result.
    let mut slots2 = vec![4.0f64.to_bits() as i64; 1];
    let truncated = jit
        .try_run_block_eager_kinded(&chunk, &mut slots2, &kinds)
        .expect("compile");
    assert_eq!(
        truncated,
        (4.0 * 1.5 + 2.0) as i64,
        "i64 entry truncates float"
    );
}

#[test]
fn block_jit_awk_div_mod_float_slots_compute() {
    use fusevm::SlotKind;
    // slot0 = OP(slot0, slot1). awk div/mod pop divisor (top) then dividend,
    // so GetSlot(0)=dividend first, GetSlot(1)=divisor second (divisor on top).
    let run = |op: Op| -> f64 {
        let mut b = ChunkBuilder::new();
        b.emit(Op::GetSlot(0), 1);
        b.emit(Op::GetSlot(1), 1);
        b.emit(op.clone(), 1);
        b.emit(Op::Dup, 1);
        b.emit(Op::SetSlot(0), 1);
        b.emit(Op::Pop, 1);
        let chunk = b.build();

        let jit = JitCompiler::new();
        assert!(
            jit.is_block_eligible(&chunk),
            "{op:?} must be block-eligible"
        );

        let kinds = [SlotKind::Float, SlotKind::Float];
        let mut slots = vec![17.0f64.to_bits() as i64, 5.0f64.to_bits() as i64];
        jit.try_run_block_eager_kinded(&chunk, &mut slots, &kinds)
            .unwrap_or_else(|| panic!("{op:?} float-slot chunk must compile"));
        f64::from_bits(slots[0] as u64)
    };

    assert_eq!(run(Op::AwkDivJit), 17.0 / 5.0, "17 / 5");
    assert_eq!(run(Op::AwkModJit), 17.0 % 5.0, "17 % 5");
}

#[test]
fn block_jit_awk_div_mod_nonzero_no_trap() {
    use fusevm::SlotKind;
    // A nonzero divisor must NOT set the trap channel: a subsequent VM run on a
    // div/mod chunk would observe no error (verified end-to-end via awkrs). Here
    // we only assert the compiled block returns the correct quotient without the
    // guarded early-exit firing (result is well-defined, not the sentinel).
    let mut b = ChunkBuilder::new();
    b.emit(Op::GetSlot(0), 1);
    b.emit(Op::GetSlot(1), 1);
    b.emit(Op::AwkDivJit, 1);
    b.emit(Op::Dup, 1);
    b.emit(Op::SetSlot(0), 1);
    b.emit(Op::Pop, 1);
    let chunk = b.build();

    let jit = JitCompiler::new();
    let kinds = [SlotKind::Float, SlotKind::Float];
    let mut slots = vec![10.0f64.to_bits() as i64, 4.0f64.to_bits() as i64];
    jit.try_run_block_eager_kinded(&chunk, &mut slots, &kinds)
        .expect("div chunk must compile");
    assert_eq!(f64::from_bits(slots[0] as u64), 2.5);
}

#[test]
fn block_jit_awk_and_or_xor_float_slots_match_scalar() {
    use fusevm::SlotKind;
    // slot0 = OP(slot0, slot1) with Float slots 12.0 and 10.0.
    // and(12,10)=8, or(12,10)=14, xor(12,10)=6 — pushed Int, stored as f64.
    let run = |op: Op| -> f64 {
        let mut b = ChunkBuilder::new();
        b.emit(Op::GetSlot(0), 1);
        b.emit(Op::GetSlot(1), 1);
        b.emit(op.clone(), 1);
        b.emit(Op::Dup, 1);
        b.emit(Op::SetSlot(0), 1);
        b.emit(Op::Pop, 1);
        let chunk = b.build();

        let jit = JitCompiler::new();
        assert!(
            jit.is_block_eligible(&chunk),
            "{op:?} must be block-eligible"
        );

        let kinds = [SlotKind::Float, SlotKind::Float];
        let mut slots = vec![12.0f64.to_bits() as i64, 10.0f64.to_bits() as i64];
        jit.try_run_block_eager_kinded(&chunk, &mut slots, &kinds)
            .unwrap_or_else(|| panic!("{op:?} float-slot chunk must compile"));
        f64::from_bits(slots[0] as u64)
    };

    assert_eq!(run(Op::AwkAnd(2)), 8.0, "and(12,10)");
    assert_eq!(run(Op::AwkOr(2)), 14.0, "or(12,10)");
    assert_eq!(run(Op::AwkXor(2)), 6.0, "xor(12,10)");
}

#[test]
fn block_jit_awk_and_saturates_like_awkrs() {
    use fusevm::SlotKind;
    // num_to_u64 = `n.trunc() as i64` saturates: a huge f64 → i64::MAX, and
    // and(huge, huge) = i64::MAX & i64::MAX = i64::MAX → that as f64. Verifies
    // the JIT uses `fcvt_to_sint_sat` (no trap on out-of-range).
    let mut b = ChunkBuilder::new();
    b.emit(Op::GetSlot(0), 1);
    b.emit(Op::GetSlot(0), 1);
    b.emit(Op::AwkAnd(2), 1);
    b.emit(Op::Dup, 1);
    b.emit(Op::SetSlot(0), 1);
    b.emit(Op::Pop, 1);
    let chunk = b.build();

    let jit = JitCompiler::new();
    let kinds = [SlotKind::Float];
    let mut slots = vec![1e30f64.to_bits() as i64];
    jit.try_run_block_eager_kinded(&chunk, &mut slots, &kinds)
        .expect("saturating and() must compile");
    // Rust reference: ((1e30_f64.trunc() as i64) & (1e30_f64.trunc() as i64)) as f64
    let want = ((1e30f64.trunc() as i64) & (1e30f64.trunc() as i64)) as f64;
    assert_eq!(f64::from_bits(slots[0] as u64), want);
}

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
fn block_jit_ternary_merge_carries_stack() {
    // slot[0] = (cond ? 42 : 99); return slot[0]
    // The branch result is left on the operand stack and consumed AFTER the
    // control-flow merge — a value live across a basic-block boundary. This
    // exercises operand-stack values carried as Cranelift block params (both via
    // an explicit Jump and via fallthrough into the merge block).
    let build_chunk = |cond: i64| {
        let mut b = ChunkBuilder::new();
        b.emit(Op::PushFrame, 1);
        b.emit(Op::LoadInt(cond), 1); // ip1: condition
        b.emit(Op::JumpIfFalse(5), 1); // ip2: if false goto else (ip5)
        b.emit(Op::LoadInt(42), 1); // ip3: then value (left on stack)
        b.emit(Op::Jump(6), 1); // ip4: goto merge (ip6)
        b.emit(Op::LoadInt(99), 1); // ip5: else value (falls through to merge)
        b.emit(Op::SetSlot(0), 1); // ip6: merge — consume stack value
        b.emit(Op::GetSlot(0), 1); // ip7
        b.build()
    };

    let jit = JitCompiler::new();

    let chunk_t = build_chunk(1);
    assert!(jit.is_block_eligible(&chunk_t));
    let mut slots = vec![0i64; 4];
    let result = jit
        .try_run_block_eager(&chunk_t, &mut slots)
        .expect("ternary-merge chunk must block-JIT compile (true)");
    assert_eq!(result, 42, "cond=true must yield 42");

    let chunk_f = build_chunk(0);
    let mut slots = vec![0i64; 4];
    let result = jit
        .try_run_block_eager(&chunk_f, &mut slots)
        .expect("ternary-merge chunk must block-JIT compile (false)");
    assert_eq!(result, 99, "cond=false must yield 99");
}

#[test]
fn block_jit_ternary_as_return_jumps_to_end() {
    // return (cond ? 42 : 99) — the ternary result is the function's return value,
    // so the then-branch does a value-carrying Jump to the segment END (ops.len()),
    // and the else-branch falls through to it. Both merge at the implicit end block
    // with the value still on the operand stack. This pins cross-block stack carry
    // through a jump-to-end merge (the shape strykelang emits for ternary-bodied subs).
    let build_chunk = |cond: i64| {
        let mut b = ChunkBuilder::new();
        b.emit(Op::PushFrame, 1); // ip0
        b.emit(Op::LoadInt(cond), 1); // ip1
        b.emit(Op::JumpIfFalse(5), 1); // ip2: false -> else (ip5)
        b.emit(Op::LoadInt(42), 1); // ip3: then value
        b.emit(Op::Jump(6), 1); // ip4: -> end (ops.len() == 6)
        b.emit(Op::LoadInt(99), 1); // ip5: else value (falls through to end)
        b.build()
    };

    let jit = JitCompiler::new();

    let chunk_t = build_chunk(1);
    assert!(jit.is_block_eligible(&chunk_t));
    let mut slots = vec![0i64; 4];
    let result = jit
        .try_run_block_eager(&chunk_t, &mut slots)
        .expect("ternary-as-return chunk must block-JIT compile (true)");
    assert_eq!(result, 42);

    let chunk_f = build_chunk(0);
    let mut slots = vec![0i64; 4];
    let result = jit
        .try_run_block_eager(&chunk_f, &mut slots)
        .expect("ternary-as-return chunk must block-JIT compile (false)");
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

/// A unique block-eligible sum loop (limit picks the op_hash so the per-thread
/// block cache entry doesn't collide with other tests on the same thread).
fn unique_sum_loop(limit: i32) -> fusevm::Chunk {
    let mut b = ChunkBuilder::new();
    b.emit(Op::PushFrame, 1);
    b.emit(Op::LoadInt(0), 1);
    b.emit(Op::SetSlot(0), 1);
    b.emit(Op::LoadInt(0), 1);
    b.emit(Op::SetSlot(1), 1);
    b.emit(Op::GetSlot(0), 1);
    b.emit(Op::GetSlot(1), 1);
    b.emit(Op::Add, 1);
    b.emit(Op::SetSlot(0), 1);
    b.emit(Op::PreIncSlotVoid(1), 1);
    b.emit(Op::SlotLtIntJumpIfFalse(1, limit, 12), 1);
    b.emit(Op::Jump(5), 1);
    b.emit(Op::GetSlot(0), 1);
    b.build()
}

#[test]
fn block_threshold_is_configurable() {
    use fusevm::TraceJitConfig;
    let jit = JitCompiler::new();

    // Lower the block warmup to 1: the chunk must stay interpreted (None) on
    // the first call and compile (Some) on the second.
    jit.set_config(TraceJitConfig {
        block_threshold: 1,
        ..TraceJitConfig::defaults()
    });
    let chunk = unique_sum_loop(37);
    let mut slots = vec![0i64; 4];
    assert_eq!(
        jit.try_run_block(&chunk, &mut slots),
        None,
        "first call is below threshold 1"
    );
    assert_eq!(
        jit.try_run_block(&chunk, &mut slots),
        Some(666),
        "second call should compile and run with block_threshold=1"
    );

    // With an explicitly higher threshold, a different chunk must still be None
    // on its second call — proving the knob (not the compiled default) drives
    // tier selection. Uses an explicit value so the test is independent of
    // whatever the shipped default `block_threshold` happens to be.
    jit.set_config(TraceJitConfig {
        block_threshold: 5,
        ..TraceJitConfig::defaults()
    });
    let chunk2 = unique_sum_loop(38);
    let mut slots2 = vec![0i64; 4];
    assert_eq!(jit.try_run_block(&chunk2, &mut slots2), None);
    assert_eq!(jit.try_run_block(&chunk2, &mut slots2), None);
}

#[test]
fn block_jit_awk_int_truncates_float() {
    // slot0 = int(3.7) → 3.0 ; slot1 = int(-2.9) → -2.0 ; return slot0
    let mut b = ChunkBuilder::new();
    b.emit(Op::PushFrame, 1);
    b.emit(Op::LoadFloat(3.7), 1);
    b.emit(Op::AwkInt, 1);
    b.emit(Op::SetSlot(0), 1);
    b.emit(Op::LoadFloat(-2.9), 1);
    b.emit(Op::AwkInt, 1);
    b.emit(Op::SetSlot(1), 1);
    b.emit(Op::GetSlot(0), 1);
    let chunk = b.build();

    let jit = JitCompiler::new();
    assert!(
        jit.is_block_eligible(&chunk),
        "AwkInt must be block-eligible"
    );

    let mut slots = vec![0i64; 4];
    let result = jit.try_run_block_eager(&chunk, &mut slots).unwrap();
    // int() yields an integral value; the block JIT returns it as a plain i64.
    assert_eq!(result, 3, "int(3.7) == 3");
    assert_eq!(slots[1], -2, "int(-2.9) == -2 (toward zero)");
}

#[test]
fn block_jit_awk_int_in_loop_matches_scalar() {
    // s = 0; for (i = 0; i < 10; i++) s += int(i + 0.9); → s = 0+1+..+9 = 45
    let mut b = ChunkBuilder::new();
    b.emit(Op::PushFrame, 1);
    b.emit(Op::LoadFloat(0.0), 1);
    b.emit(Op::SetSlot(0), 1); // s = 0.0
    b.emit(Op::LoadFloat(0.0), 1);
    b.emit(Op::SetSlot(1), 1); // i = 0.0
                               // ip=5: body
    b.emit(Op::GetSlot(0), 1);
    b.emit(Op::GetSlot(1), 1);
    b.emit(Op::LoadFloat(0.9), 1);
    b.emit(Op::Add, 1);
    b.emit(Op::AwkInt, 1); // int(i + 0.9) == i
    b.emit(Op::Add, 1);
    b.emit(Op::SetSlot(0), 1); // s += int(i + 0.9)
    b.emit(Op::GetSlot(1), 1);
    b.emit(Op::LoadFloat(1.0), 1);
    b.emit(Op::Add, 1);
    b.emit(Op::SetSlot(1), 1); // i += 1
    b.emit(Op::GetSlot(1), 1);
    b.emit(Op::LoadFloat(10.0), 1);
    b.emit(Op::NumLt, 1);
    b.emit(Op::JumpIfTrue(5), 1);
    b.emit(Op::GetSlot(0), 1);
    let chunk = b.build();

    let jit = JitCompiler::new();
    assert!(jit.is_block_eligible(&chunk));

    let mut slots = vec![0i64; 4];
    let result = jit.try_run_block_eager(&chunk, &mut slots).unwrap();
    assert_eq!(result, 45);
}
