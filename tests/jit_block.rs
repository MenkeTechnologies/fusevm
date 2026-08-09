//! Block JIT tests — verify correctness of compiled loops and branches.

#![cfg(feature = "jit")]

use fusevm::{ChunkBuilder, JitCompiler, Op, VMResult, Value, VM};

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

/// `Op::AwkInt` is host-dispatched (`VM::run` -> `AwkHost::int`), so no tier
/// but the interpreter can know what it evaluates to. This test used to assert
/// `is_block_eligible(&chunk)` — i.e. it pinned the block tier's right to
/// lower it — and only checked the two operands where every candidate answer
/// happens to agree. On the pre-change library the same chunk measured:
///
/// ```text
///   is_block_eligible          = true
///   interpreter (default host) = Int(3)     slots=[Int(3), Int(-2)]
///   interpreter (f64 host)     = Float(3.0) slots=[Float(3.0), Float(-2.0)]
///   block try_run_block_eager  = Some(3)    slots=[3, -2]
///   aot                        = Int(3)
/// ```
///
/// The block tier answered `3` for a chunk whose value under a registered host
/// is `Float(3.0)` — a divergence the old assertions could not see. The chunk
/// and both value expectations are kept; the eligibility assertion is inverted
/// and the two host readings are now checked directly.
#[test]
fn block_jit_declines_awk_int_because_it_is_host_dispatched() {
    // slot0 = int(3.7) ; slot1 = int(-2.9) ; return slot0
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
        !jit.is_block_eligible(&chunk),
        "AwkInt is host-dispatched; no native lowering is right for every AwkHost"
    );
    let mut slots = vec![0i64; 4];
    assert_eq!(
        jit.try_run_block_eager(&chunk, &mut slots),
        None,
        "declining hands the chunk to the interpreter, the only host-aware tier"
    );

    // The interpreter's answers — the ones the op is actually specified to
    // have. Truncation toward zero, exactly as before.
    let mut vm = VM::new(chunk.clone());
    assert!(
        matches!(vm.run(), VMResult::Ok(Value::Int(3))),
        "int(3.7) == 3"
    );
    let s = &vm.frames.last().unwrap().slots;
    assert_eq!(s[0], Value::Int(3), "int(3.7) == 3");
    assert_eq!(s[1], Value::Int(-2), "int(-2.9) == -2 (toward zero)");

    // And with a host that models awk numbers as f64, the same chunk is a
    // Float throughout. This is the reading the native lowering destroyed.
    struct FloatIntHost;
    impl fusevm::AwkHost for FloatIntHost {
        fn int(&mut self, x: &Value) -> Value {
            Value::Float(x.to_float().trunc())
        }
    }
    let mut vm = VM::new(chunk);
    vm.set_awk_host(Box::new(FloatIntHost));
    assert!(matches!(vm.run(), VMResult::Ok(Value::Float(f)) if f == 3.0));
    let s = &vm.frames.last().unwrap().slots;
    assert_eq!(s[0], Value::Float(3.0));
    assert_eq!(s[1], Value::Float(-2.0));
}

#[test]
fn block_jit_awk_mkbool_returns_one_or_zero() {
    use fusevm::SlotKind;
    // slot0 = mkbool(3.7)  → 1.0
    // slot1 = mkbool(0.0)  → 0.0
    // slot2 = mkbool(-1.5) → 1.0 (nonzero)
    let mut b = ChunkBuilder::new();
    b.emit(Op::PushFrame, 1);
    b.emit(Op::LoadFloat(3.7), 1);
    b.emit(Op::AwkMkbool, 1);
    b.emit(Op::SetSlot(0), 1);
    b.emit(Op::LoadFloat(0.0), 1);
    b.emit(Op::AwkMkbool, 1);
    b.emit(Op::SetSlot(1), 1);
    b.emit(Op::LoadFloat(-1.5), 1);
    b.emit(Op::AwkMkbool, 1);
    b.emit(Op::SetSlot(2), 1);
    b.emit(Op::GetSlot(0), 1);
    let chunk = b.build();

    let jit = JitCompiler::new();
    assert!(
        jit.is_block_eligible(&chunk),
        "AwkMkbool must be block-eligible"
    );

    let kinds = [
        SlotKind::Float,
        SlotKind::Float,
        SlotKind::Float,
        SlotKind::Float,
    ];
    let mut slots = vec![0i64; 4];
    jit.try_run_block_eager_kinded(&chunk, &mut slots, &kinds)
        .expect("block JIT must compile");
    assert_eq!(f64::from_bits(slots[0] as u64), 1.0, "mkbool(3.7) == 1.0");
    assert_eq!(f64::from_bits(slots[1] as u64), 0.0, "mkbool(0.0) == 0.0");
    assert_eq!(
        f64::from_bits(slots[2] as u64),
        1.0,
        "mkbool(-1.5) == 1.0 (nonzero)"
    );
}

#[test]
fn block_jit_awk_int_in_loop_matches_scalar() {
    use fusevm::{BlockNum, SlotKind};
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
    // An `Op::AwkInt` anywhere in the chunk makes the whole chunk ineligible,
    // because the op is host-dispatched. On the pre-change library this chunk
    // measured `is_block_eligible = true` and `try_run_block_eager = Some(45)`
    // — an i64 — where the interpreter (under either host) and the AOT tier
    // both answered `Float(45.0)`. The `45` the old assertion pinned came out
    // of `try_run_block_eager`'s documented float→i64 truncation, so it could
    // not distinguish the two.
    assert!(!jit.is_block_eligible(&chunk));
    let mut slots = vec![0i64; 4];
    assert_eq!(jit.try_run_block_eager(&chunk, &mut slots), None);

    // The value the loop actually computes, on the tier that owns the op.
    let mut vm = VM::new(chunk.clone());
    assert!(matches!(vm.run(), VMResult::Ok(Value::Float(f)) if f == 45.0));

    // Replacing the host-dispatched `int()` with the pure `Op::TruncFloat` —
    // the op a frontend should emit when it wants awk's `int()` in native code
    // — restores block-JIT eligibility with no host in the picture.
    let mut b = ChunkBuilder::new();
    for op in chunk.ops.iter() {
        b.emit(
            if matches!(op, Op::AwkInt) {
                Op::TruncFloat
            } else {
                op.clone()
            },
            1,
        );
    }
    let pure = b.build();
    assert!(
        jit.is_block_eligible(&pure),
        "Op::TruncFloat is host-free and must stay block-eligible"
    );
    // The slots hold f64s, so the caller must say so — `try_run_block_eager`
    // (every slot `Int`) now declines rather than truncating them, which is
    // the second half of this fix.
    let kinds = [
        SlotKind::Float,
        SlotKind::Float,
        SlotKind::Float,
        SlotKind::Float,
    ];
    let mut slots = vec![0i64; 4];
    assert_eq!(
        jit.try_run_block_eager(&pure, &mut slots),
        None,
        "float slots declared as Int must decline, not silently truncate"
    );
    let mut slots = vec![0.0f64.to_bits() as i64; 4];
    match jit.try_run_block_eager_typed_kinded(&pure, &mut slots, &kinds) {
        Some(BlockNum::Float(v)) => assert_eq!(v, 45.0),
        other => panic!("expected Float(45.0), got {other:?}"),
    }
    let mut vm = VM::new(pure);
    assert!(matches!(vm.run(), VMResult::Ok(Value::Float(f)) if f == 45.0));
}

// ── Block JIT tests for AwkSqrtJit / AwkLogJit / AwkLshiftJit / AwkRshiftJit /
// AwkComplJit (the 5 ops added in 0.13.6 at interpreter tier, lowered to native
// in 0.13.7). Each builds a 1- or 2-arg chunk, runs through
// try_run_block_eager_kinded with SlotKind::Float, and checks the result bit
// pattern. Negative-path tests cover the warn libcall (sqrt/log) and the trap
// libcall (lshift/rshift/compl) — for the trap variants we read take_awk_div_trap
// to confirm the JIT recorded the right code without going through the VM.

#[test]
fn block_jit_awk_sqrt_jit_positive_matches_libm() {
    use fusevm::SlotKind;
    let mut b = ChunkBuilder::new();
    b.emit(Op::GetSlot(0), 1);
    b.emit(Op::AwkSqrtJit, 1);
    b.emit(Op::Dup, 1);
    b.emit(Op::SetSlot(0), 1);
    b.emit(Op::Pop, 1);
    let chunk = b.build();

    let jit = JitCompiler::new();
    assert!(jit.is_block_eligible(&chunk));
    let kinds = [SlotKind::Float];
    let mut slots = vec![16.0f64.to_bits() as i64];
    jit.try_run_block_eager_kinded(&chunk, &mut slots, &kinds)
        .expect("AwkSqrtJit chunk must compile");
    assert_eq!(f64::from_bits(slots[0] as u64), 4.0);
}

#[test]
fn block_jit_awk_sqrt_jit_negative_yields_nan() {
    use fusevm::SlotKind;
    let mut b = ChunkBuilder::new();
    b.emit(Op::GetSlot(0), 1);
    b.emit(Op::AwkSqrtJit, 1);
    b.emit(Op::Dup, 1);
    b.emit(Op::SetSlot(0), 1);
    b.emit(Op::Pop, 1);
    let chunk = b.build();

    let jit = JitCompiler::new();
    let kinds = [SlotKind::Float];
    let mut slots = vec![(-1.0f64).to_bits() as i64];
    jit.try_run_block_eager_kinded(&chunk, &mut slots, &kinds)
        .expect("AwkSqrtJit chunk must compile");
    // Warn libcall printed to stderr; result is NaN.
    assert!(f64::from_bits(slots[0] as u64).is_nan());
}

#[test]
fn block_jit_awk_log_jit_e_yields_one() {
    use fusevm::SlotKind;
    let mut b = ChunkBuilder::new();
    b.emit(Op::GetSlot(0), 1);
    b.emit(Op::AwkLogJit, 1);
    b.emit(Op::Dup, 1);
    b.emit(Op::SetSlot(0), 1);
    b.emit(Op::Pop, 1);
    let chunk = b.build();

    let jit = JitCompiler::new();
    let kinds = [SlotKind::Float];
    let mut slots = vec![std::f64::consts::E.to_bits() as i64];
    jit.try_run_block_eager_kinded(&chunk, &mut slots, &kinds)
        .expect("AwkLogJit chunk must compile");
    assert!((f64::from_bits(slots[0] as u64) - 1.0).abs() < 1e-10);
}

#[test]
fn block_jit_awk_lshift_jit_computes_shift() {
    use fusevm::SlotKind;
    // lshift(1, 4) == 16. awkrs pushes a then n.
    let mut b = ChunkBuilder::new();
    b.emit(Op::GetSlot(0), 1);
    b.emit(Op::GetSlot(1), 1);
    b.emit(Op::AwkLshiftJit, 1);
    b.emit(Op::Dup, 1);
    b.emit(Op::SetSlot(0), 1);
    b.emit(Op::Pop, 1);
    let chunk = b.build();

    let jit = JitCompiler::new();
    let kinds = [SlotKind::Float, SlotKind::Float];
    let mut slots = vec![1.0f64.to_bits() as i64, 4.0f64.to_bits() as i64];
    jit.try_run_block_eager_kinded(&chunk, &mut slots, &kinds)
        .expect("AwkLshiftJit chunk must compile");
    assert_eq!(f64::from_bits(slots[0] as u64), 16.0);
}

#[test]
fn block_jit_awk_rshift_jit_computes_shift() {
    use fusevm::SlotKind;
    let mut b = ChunkBuilder::new();
    b.emit(Op::GetSlot(0), 1);
    b.emit(Op::GetSlot(1), 1);
    b.emit(Op::AwkRshiftJit, 1);
    b.emit(Op::Dup, 1);
    b.emit(Op::SetSlot(0), 1);
    b.emit(Op::Pop, 1);
    let chunk = b.build();

    let jit = JitCompiler::new();
    let kinds = [SlotKind::Float, SlotKind::Float];
    let mut slots = vec![16.0f64.to_bits() as i64, 2.0f64.to_bits() as i64];
    jit.try_run_block_eager_kinded(&chunk, &mut slots, &kinds)
        .expect("AwkRshiftJit chunk must compile");
    assert_eq!(f64::from_bits(slots[0] as u64), 4.0);
}

#[test]
fn block_jit_awk_compl_jit_negates_bits() {
    use fusevm::SlotKind;
    // compl(15) == !15_i64 == -16.
    let mut b = ChunkBuilder::new();
    b.emit(Op::GetSlot(0), 1);
    b.emit(Op::AwkComplJit, 1);
    b.emit(Op::Dup, 1);
    b.emit(Op::SetSlot(0), 1);
    b.emit(Op::Pop, 1);
    let chunk = b.build();

    let jit = JitCompiler::new();
    let kinds = [SlotKind::Float];
    let mut slots = vec![15.0f64.to_bits() as i64];
    jit.try_run_block_eager_kinded(&chunk, &mut slots, &kinds)
        .expect("AwkComplJit chunk must compile");
    assert_eq!(f64::from_bits(slots[0] as u64), -16.0);
}

// ── Block JIT tests for AwkGetFieldNum (the host-hook libcall variant added
// in 0.13.9 to lower awk's `$N` field-read with constant N). Tests both the
// "hook installed → returns hook's value" path and the "no hook → returns 0.0"
// path. Each test uses a fresh thread to keep the thread-local hook isolated.

extern "C" fn fake_field_hook(idx: i64) -> f64 {
    // Pretend $1 = 10.0, $2 = 20.0, $3 = 30.0, ... so the test can verify
    // both the libcall dispatch AND that the right field index is passed.
    (idx as f64) * 10.0
}

#[test]
fn block_jit_awk_get_field_num_calls_installed_hook() {
    use fusevm::{set_awk_field_num_hook, SlotKind};
    // slot0 = $3 (== 30.0 via the fake hook).
    let mut b = ChunkBuilder::new();
    b.emit(Op::AwkGetFieldNum(3), 1);
    b.emit(Op::Dup, 1);
    b.emit(Op::SetSlot(0), 1);
    b.emit(Op::Pop, 1);
    let chunk = b.build();

    // Run inside a fresh thread so the thread-local hook doesn't leak.
    std::thread::spawn(move || {
        set_awk_field_num_hook(Some(fake_field_hook));
        let jit = JitCompiler::new();
        assert!(jit.is_block_eligible(&chunk));
        let kinds = [SlotKind::Float];
        let mut slots = vec![0i64];
        jit.try_run_block_eager_kinded(&chunk, &mut slots, &kinds)
            .expect("AwkGetFieldNum chunk must compile");
        assert_eq!(f64::from_bits(slots[0] as u64), 30.0);
        set_awk_field_num_hook(None);
    })
    .join()
    .unwrap();
}

#[test]
fn block_jit_awk_get_field_num_no_hook_returns_zero() {
    use fusevm::SlotKind;
    let mut b = ChunkBuilder::new();
    b.emit(Op::AwkGetFieldNum(7), 1);
    b.emit(Op::Dup, 1);
    b.emit(Op::SetSlot(0), 1);
    b.emit(Op::Pop, 1);
    let chunk = b.build();

    // Fresh thread: no hook is installed in this thread, so the libcall
    // returns 0.0 (awk's missing-field default).
    std::thread::spawn(move || {
        let jit = JitCompiler::new();
        let kinds = [SlotKind::Float];
        let mut slots = vec![999i64];
        jit.try_run_block_eager_kinded(&chunk, &mut slots, &kinds)
            .expect("AwkGetFieldNum chunk must compile");
        assert_eq!(f64::from_bits(slots[0] as u64), 0.0);
    })
    .join()
    .unwrap();
}

#[test]
fn block_jit_awk_get_field_num_sum_loop() {
    use fusevm::{set_awk_field_num_hook, SlotKind};
    // slot0 = $1 + $2 + $3 + $4 + $5 — the canonical `{sum += $N}` pattern
    // expressed as straight-line code (one chunk, multiple field reads).
    // Expected: 10 + 20 + 30 + 40 + 50 == 150.
    let mut b = ChunkBuilder::new();
    b.emit(Op::AwkGetFieldNum(1), 1);
    b.emit(Op::AwkGetFieldNum(2), 1);
    b.emit(Op::Add, 1);
    b.emit(Op::AwkGetFieldNum(3), 1);
    b.emit(Op::Add, 1);
    b.emit(Op::AwkGetFieldNum(4), 1);
    b.emit(Op::Add, 1);
    b.emit(Op::AwkGetFieldNum(5), 1);
    b.emit(Op::Add, 1);
    b.emit(Op::Dup, 1);
    b.emit(Op::SetSlot(0), 1);
    b.emit(Op::Pop, 1);
    let chunk = b.build();

    std::thread::spawn(move || {
        set_awk_field_num_hook(Some(fake_field_hook));
        let jit = JitCompiler::new();
        let kinds = [SlotKind::Float];
        let mut slots = vec![0i64];
        jit.try_run_block_eager_kinded(&chunk, &mut slots, &kinds)
            .expect("multi-field chunk must compile");
        assert_eq!(f64::from_bits(slots[0] as u64), 150.0);
        set_awk_field_num_hook(None);
    })
    .join()
    .unwrap();
}

#[test]
fn block_jit_negate_neg_zero_float_kind_preserved() {
    use fusevm::BlockNum;
    // (- -0.0) → +0.0 as a FLOAT through the block tier. Regression: an
    // integral-valued LoadFloat constant (±0.0 included) was lowered as
    // JitTy::Int, collapsing the result to Int(0).
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadFloat(-0.0), 1);
    b.emit(Op::Negate, 1);
    let chunk = b.build();

    let jit = JitCompiler::new();
    let mut slots: Vec<i64> = vec![];
    match jit
        .try_run_block_eager_typed_kinded(&chunk, &mut slots, &[])
        .expect("LoadFloat+Negate chunk must block-JIT compile")
    {
        BlockNum::Float(f) => assert_eq!(
            f.to_bits(),
            0.0f64.to_bits(),
            "-(-0.0) must be +0.0, got {f:?}"
        ),
        BlockNum::Int(n) => panic!("float kind collapsed to Int({n})"),
    }
}

#[test]
fn block_jit_sub_neg_zero_float_kind_preserved() {
    use fusevm::BlockNum;
    // (- -0.0 0) → -0.0 as a FLOAT through the block tier.
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadFloat(-0.0), 1);
    b.emit(Op::LoadInt(0), 1);
    b.emit(Op::Sub, 1);
    let chunk = b.build();

    let jit = JitCompiler::new();
    let mut slots: Vec<i64> = vec![];
    match jit
        .try_run_block_eager_typed_kinded(&chunk, &mut slots, &[])
        .expect("LoadFloat+LoadInt+Sub chunk must block-JIT compile")
    {
        BlockNum::Float(f) => assert_eq!(
            f.to_bits(),
            (-0.0f64).to_bits(),
            "-0.0 - 0 must be -0.0, got {f:?}"
        ),
        BlockNum::Int(n) => panic!("float kind collapsed to Int({n})"),
    }
}

// ── Interpreter/native tier agreement ─────────────────────────────────────
//
// The interpreter is the specification: it is what runs before a chunk goes
// hot, so any op the native tiers lower differently produces an answer that
// depends on JIT warmup state. These pin the three defects a two-tier
// differential sweep (every arithmetic/comparison/conversion op against zero,
// negative zero, NaN, +/-inf, i64::MIN and i64::MAX) turned up:
//
//   1. `Op::Div` lowered as integer `sdiv` — `1 / 2` answered `Int(0)` where
//      the interpreter answers `Float(0.5)` — and by zero it TRAPPED, killing
//      the process, where the interpreter answers `Undef`.
//   2. `Op::Pow` lowered as an integer power — `2 ** -1` answered `Int(0)`
//      where the interpreter answers `Float(0.5)`.
//   3. `Op::LogNot` on a float panicked Cranelift's aarch64 backend at compile
//      time ("not implemented").

/// Run `ops` with the block tier forced off, then forced on, and require the
/// two to agree exactly — value AND numeric type.
fn assert_tiers_agree(label: &str, ops: &[Op]) {
    use fusevm::{TraceJitConfig, VMResult, VM};

    let run = |threshold: u32| -> VMResult {
        let jit = JitCompiler::new();
        let mut cfg = TraceJitConfig::defaults();
        cfg.block_threshold = threshold;
        jit.set_config(cfg);
        let mut b = ChunkBuilder::new();
        for op in ops {
            b.emit(op.clone(), 1);
        }
        let mut vm = VM::new(b.build());
        vm.enable_tracing_jit();
        vm.run()
    };

    // u32::MAX never reaches the threshold, so the interpreter owns the run;
    // 0 compiles on the very first call.
    let interp = run(u32::MAX);
    let native = run(0);
    let (a, b) = match (&interp, &native) {
        (VMResult::Ok(a), VMResult::Ok(b)) => (a, b),
        (a, b) => panic!("{label}: unexpected results interp={a:?} block={b:?}"),
    };
    if same_value(a, b) {
        return;
    }
    panic!("{label}: interpreter and block tier disagree (interp={a:?} block={b:?})");
}

/// Whether two results are the same answer.
///
/// Exact equality, plus exactly two documented equivalences — no others, so a
/// real value difference can never slip through:
///
///  * Two NaNs. `f64::NAN != f64::NAN`, but both tiers producing NaN IS
///    agreement.
///  * `Bool(b)` against `Int(0|1)`. The block tier's return channel is numeric
///    (`BlockNum`), so a chunk whose last value is a predicate comes back as
///    the integer the interpreter's `Bool` coerces to. This is a
///    representation difference at the chunk boundary only — the coercions
///    (`to_int`, `to_str`, `is_truthy`) all agree — and it applies to every
///    comparison op, not just the ones under test here.
fn same_value(a: &fusevm::Value, b: &fusevm::Value) -> bool {
    use fusevm::Value;
    match (a, b) {
        (Value::Float(x), Value::Float(y)) if x.is_nan() && y.is_nan() => true,
        (Value::Bool(x), Value::Int(y)) | (Value::Int(y), Value::Bool(x)) => i64::from(*x) == *y,
        _ => a == b,
    }
}

/// Run `ops` in the interpreter and return the produced value.
fn interp_value(ops: &[Op]) -> fusevm::Value {
    use fusevm::{VMResult, VM};
    let mut b = ChunkBuilder::new();
    for op in ops {
        b.emit(op.clone(), 1);
    }
    match VM::new(b.build()).run() {
        VMResult::Ok(v) => v,
        other => panic!("expected a value, got {other:?}"),
    }
}

#[test]
fn div_agrees_across_tiers() {
    use fusevm::Value;
    // Inexact integer division is a FLOAT — an `sdiv` lowering answers 0 here.
    assert_tiers_agree("1 / 2", &[Op::LoadInt(1), Op::LoadInt(2), Op::Div]);
    assert_tiers_agree("7 / 2", &[Op::LoadInt(7), Op::LoadInt(2), Op::Div]);
    assert_tiers_agree("-7 / 2", &[Op::LoadInt(-7), Op::LoadInt(2), Op::Div]);
    // Exact division is still a float, not an int.
    assert_tiers_agree("20 / 5", &[Op::LoadInt(20), Op::LoadInt(5), Op::Div]);
    // The i64 division overflow case, which `sdiv` traps on.
    assert_tiers_agree(
        "i64::MIN / -1",
        &[Op::LoadInt(i64::MIN), Op::LoadInt(-1), Op::Div],
    );
    // Zero divisors: the interpreter answers Undef and native code has no
    // Undef, so these must run interpreted rather than trap or answer inf.
    assert_tiers_agree("1 / 0", &[Op::LoadInt(1), Op::LoadInt(0), Op::Div]);
    assert_tiers_agree("0 / 0", &[Op::LoadInt(0), Op::LoadInt(0), Op::Div]);
    assert_tiers_agree("1 / 0.0", &[Op::LoadInt(1), Op::LoadFloat(0.0), Op::Div]);
    // -0.0 is zero for this op: the interpreter tests `b.to_float() == 0.0`.
    assert_tiers_agree("1 / -0.0", &[Op::LoadInt(1), Op::LoadFloat(-0.0), Op::Div]);

    // And pin the actual contract, so "they agree" can't become "they agree on
    // the wrong answer".
    assert_eq!(
        interp_value(&[Op::LoadInt(1), Op::LoadInt(2), Op::Div]),
        Value::Float(0.5),
        "Op::Div is always-float"
    );
    assert_eq!(
        interp_value(&[Op::LoadInt(1), Op::LoadInt(0), Op::Div]),
        Value::Undef,
        "Op::Div by zero is Undef"
    );
}

#[test]
fn mod_agrees_across_tiers() {
    // `srem` traps on a zero divisor and on i64::MIN % -1; the interpreter
    // answers 0 for both.
    assert_tiers_agree("1 % 0", &[Op::LoadInt(1), Op::LoadInt(0), Op::Mod]);
    assert_tiers_agree("0 % 0", &[Op::LoadInt(0), Op::LoadInt(0), Op::Mod]);
    assert_tiers_agree(
        "i64::MIN % -1",
        &[Op::LoadInt(i64::MIN), Op::LoadInt(-1), Op::Mod],
    );
    assert_tiers_agree("7 % 3", &[Op::LoadInt(7), Op::LoadInt(3), Op::Mod]);
    assert_tiers_agree("7 % 2.5", &[Op::LoadInt(7), Op::LoadFloat(2.5), Op::Mod]);
    assert_tiers_agree("7 % 0.0", &[Op::LoadInt(7), Op::LoadFloat(0.0), Op::Mod]);
}

#[test]
fn mod_by_minus_one_does_not_panic_the_interpreter() {
    use fusevm::Value;
    // Rust's `%` panics on `i64::MIN % -1` — an overflow check that runs in
    // release too — so this used to abort the whole VM. The answer is 0.
    assert_eq!(
        interp_value(&[Op::LoadInt(i64::MIN), Op::LoadInt(-1), Op::Mod]),
        Value::Int(0)
    );
}

#[test]
fn pow_agrees_across_tiers() {
    use fusevm::Value;
    assert_tiers_agree("2 ** 10", &[Op::LoadInt(2), Op::LoadInt(10), Op::Pow]);
    // A negative exponent is where an integer power silently answered 0.
    assert_tiers_agree("2 ** -1", &[Op::LoadInt(2), Op::LoadInt(-1), Op::Pow]);
    assert_tiers_agree("0 ** -1", &[Op::LoadInt(0), Op::LoadInt(-1), Op::Pow]);
    assert_tiers_agree("2.5 ** 2", &[Op::LoadFloat(2.5), Op::LoadInt(2), Op::Pow]);

    assert_eq!(
        interp_value(&[Op::LoadInt(2), Op::LoadInt(10), Op::Pow]),
        Value::Float(1024.0),
        "Op::Pow is always-float"
    );
}

#[test]
fn lognot_on_a_float_compiles_and_agrees() {
    // This used to panic Cranelift's aarch64 backend with "not implemented"
    // while compiling, so the process died rather than answering anything.
    assert_tiers_agree("!0.0", &[Op::LoadFloat(0.0), Op::LogNot]);
    assert_tiers_agree("!-0.0", &[Op::LoadFloat(-0.0), Op::LogNot]);
    assert_tiers_agree("!1.0", &[Op::LoadFloat(1.0), Op::LogNot]);
    assert_tiers_agree("!nan", &[Op::LoadFloat(f64::NAN), Op::LogNot]);
    assert_tiers_agree("!inf", &[Op::LoadFloat(f64::INFINITY), Op::LogNot]);
}
