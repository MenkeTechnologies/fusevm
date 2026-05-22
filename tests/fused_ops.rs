//! Tests for fused superinstructions and slot-indexed fast paths.
//!
//! These ops are the VM's performance secret sauce. Frontends emit them
//! to bypass multi-op interpreter overhead in hot loops. Each fused op
//! self-contains a tight loop or stack-free slot update.

use fusevm::{ChunkBuilder, Op, VMResult, Value, VM};

fn build_and_run(mk: impl FnOnce(&mut ChunkBuilder)) -> VMResult {
    let mut b = ChunkBuilder::new();
    mk(&mut b);
    VM::new(b.build()).run()
}

// ── PreIncSlot / PreIncSlotVoid ────────────────────────────────────────────

#[test]
fn pre_inc_slot_increments_and_pushes_new_value() {
    let r = build_and_run(|b| {
        b.emit(Op::PushFrame, 1);
        b.emit(Op::LoadInt(10), 1);
        b.emit(Op::SetSlot(0), 1);
        b.emit(Op::PreIncSlot(0), 1);
    });
    assert!(matches!(r, VMResult::Ok(Value::Int(11))), "got {:?}", r);
}

#[test]
fn pre_inc_slot_void_updates_slot_without_pushing() {
    // Two PreIncSlotVoid then GetSlot proves the slot was updated and
    // nothing extra landed on the value stack.
    let r = build_and_run(|b| {
        b.emit(Op::PushFrame, 1);
        b.emit(Op::LoadInt(0), 1);
        b.emit(Op::SetSlot(0), 1);
        b.emit(Op::PreIncSlotVoid(0), 1);
        b.emit(Op::PreIncSlotVoid(0), 1);
        b.emit(Op::GetSlot(0), 1);
    });
    assert!(matches!(r, VMResult::Ok(Value::Int(2))), "got {:?}", r);
}

// ── AddAssignSlotVoid ──────────────────────────────────────────────────────

#[test]
fn add_assign_slot_void_accumulates_into_destination() {
    let r = build_and_run(|b| {
        b.emit(Op::PushFrame, 1);
        b.emit(Op::LoadInt(40), 1);
        b.emit(Op::SetSlot(0), 1);
        b.emit(Op::LoadInt(2), 1);
        b.emit(Op::SetSlot(1), 1);
        b.emit(Op::AddAssignSlotVoid(0, 1), 1); // slot0 += slot1
        b.emit(Op::GetSlot(0), 1);
    });
    assert!(matches!(r, VMResult::Ok(Value::Int(42))), "got {:?}", r);
}

// ── AccumSumLoop ───────────────────────────────────────────────────────────

#[test]
fn accum_sum_loop_zero_iters_when_already_at_limit() {
    let r = build_and_run(|b| {
        b.emit(Op::PushFrame, 1);
        b.emit(Op::LoadInt(0), 1);
        b.emit(Op::SetSlot(0), 1); // sum = 0
        b.emit(Op::LoadInt(5), 1);
        b.emit(Op::SetSlot(1), 1); // i = 5 (already at limit)
        b.emit(Op::AccumSumLoop(0, 1, 5), 1);
        b.emit(Op::GetSlot(0), 1);
    });
    assert!(matches!(r, VMResult::Ok(Value::Int(0))), "got {:?}", r);
}

#[test]
fn accum_sum_loop_small_range() {
    // sum i for i in [0..10) = 45
    let r = build_and_run(|b| {
        b.emit(Op::PushFrame, 1);
        b.emit(Op::LoadInt(0), 1);
        b.emit(Op::SetSlot(0), 1);
        b.emit(Op::LoadInt(0), 1);
        b.emit(Op::SetSlot(1), 1);
        b.emit(Op::AccumSumLoop(0, 1, 10), 1);
        b.emit(Op::GetSlot(0), 1);
    });
    assert!(matches!(r, VMResult::Ok(Value::Int(45))), "got {:?}", r);
}

#[test]
fn accum_sum_loop_advances_loop_var_to_limit() {
    // Verify i_slot ends at `limit` after the loop completes.
    let r = build_and_run(|b| {
        b.emit(Op::PushFrame, 1);
        b.emit(Op::LoadInt(0), 1);
        b.emit(Op::SetSlot(0), 1);
        b.emit(Op::LoadInt(0), 1);
        b.emit(Op::SetSlot(1), 1);
        b.emit(Op::AccumSumLoop(0, 1, 7), 1);
        b.emit(Op::GetSlot(1), 1);
    });
    assert!(matches!(r, VMResult::Ok(Value::Int(7))), "got {:?}", r);
}

// ── PushIntRangeLoop ───────────────────────────────────────────────────────

#[test]
fn push_int_range_loop_fills_array_with_range() {
    // Build [0,1,2,3,4] via the fused op.
    let r = build_and_run(|b| {
        b.emit(Op::PushFrame, 1);
        let arr = b.add_name("arr");
        b.emit(Op::DeclareArray(arr), 1);
        b.emit(Op::LoadInt(0), 1);
        b.emit(Op::SetSlot(0), 1); // i = 0
        b.emit(Op::PushIntRangeLoop(arr, 0, 5), 1);
        b.emit(Op::ArrayLen(arr), 1);
    });
    assert!(matches!(r, VMResult::Ok(Value::Int(5))), "got {:?}", r);
}

#[test]
fn push_int_range_loop_with_zero_iters_does_not_grow() {
    let r = build_and_run(|b| {
        b.emit(Op::PushFrame, 1);
        let arr = b.add_name("arr");
        b.emit(Op::DeclareArray(arr), 1);
        b.emit(Op::LoadInt(10), 1);
        b.emit(Op::SetSlot(0), 1); // i = 10 already past limit
        b.emit(Op::PushIntRangeLoop(arr, 0, 5), 1);
        b.emit(Op::ArrayLen(arr), 1);
    });
    assert!(matches!(r, VMResult::Ok(Value::Int(0))), "got {:?}", r);
}

// ── ConcatConstLoop ────────────────────────────────────────────────────────

#[test]
fn concat_const_loop_repeats_constant() {
    // s = ""; for i in 0..4 { s .= "x" } → "xxxx"
    let r = build_and_run(|b| {
        b.emit(Op::PushFrame, 1);
        let c = b.add_constant(Value::str("x"));
        let empty = b.add_constant(Value::str(""));
        b.emit(Op::LoadConst(empty), 1);
        b.emit(Op::SetSlot(0), 1);
        b.emit(Op::LoadInt(0), 1);
        b.emit(Op::SetSlot(1), 1);
        b.emit(Op::ConcatConstLoop(c, 0, 1, 4), 1);
        b.emit(Op::GetSlot(0), 1);
    });
    match r {
        VMResult::Ok(v) => assert_eq!(v.to_str(), "xxxx"),
        other => panic!("got {:?}", other),
    }
}

#[test]
fn concat_const_loop_preserves_existing_prefix() {
    let r = build_and_run(|b| {
        b.emit(Op::PushFrame, 1);
        let prefix = b.add_constant(Value::str("hi:"));
        let glue = b.add_constant(Value::str("-"));
        b.emit(Op::LoadConst(prefix), 1);
        b.emit(Op::SetSlot(0), 1);
        b.emit(Op::LoadInt(0), 1);
        b.emit(Op::SetSlot(1), 1);
        b.emit(Op::ConcatConstLoop(glue, 0, 1, 3), 1);
        b.emit(Op::GetSlot(0), 1);
    });
    match r {
        VMResult::Ok(v) => assert_eq!(v.to_str(), "hi:---"),
        other => panic!("got {:?}", other),
    }
}

// ── SlotLtIntJumpIfFalse / SlotIncLtIntJumpBack ────────────────────────────

#[test]
fn slot_lt_int_jump_if_false_takes_branch_when_ge_limit() {
    // Layout:
    //   0: PushFrame
    //   1: LoadInt(10)
    //   2: SetSlot(0)         ; i = 10
    //   3: SlotLtIntJumpIfFalse(0, 5, 5)  ; if !(i < 5) jump to ip 5
    //   4: LoadInt(999)       ; skipped
    //   5: LoadInt(42)        ; landed here
    let mut b = ChunkBuilder::new();
    b.emit(Op::PushFrame, 1);
    b.emit(Op::LoadInt(10), 1);
    b.emit(Op::SetSlot(0), 1);
    b.emit(Op::SlotLtIntJumpIfFalse(0, 5, 5), 1);
    b.emit(Op::LoadInt(999), 1);
    b.emit(Op::LoadInt(42), 1);
    match VM::new(b.build()).run() {
        VMResult::Ok(Value::Int(42)) => {}
        other => panic!("got {:?}", other),
    }
}

#[test]
fn slot_lt_int_jump_if_false_falls_through_when_lt_limit() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::PushFrame, 1);
    b.emit(Op::LoadInt(2), 1);
    b.emit(Op::SetSlot(0), 1);
    b.emit(Op::SlotLtIntJumpIfFalse(0, 5, 6), 1); // ip 3
    b.emit(Op::LoadInt(111), 1);                   // ip 4 — executed
    b.emit(Op::LoadInt(222), 1);                   // ip 5 — also executed
    // ip 6: end
    match VM::new(b.build()).run() {
        VMResult::Ok(Value::Int(222)) => {}
        other => panic!("got {:?}", other),
    }
}

#[test]
fn slot_inc_lt_int_jump_back_runs_loop_body_n_times() {
    // Compile: i = 0; do { sum += 1; i++ } while (i < 4);
    //   0: PushFrame
    //   1: LoadInt(0); SetSlot(0)          ; sum = 0  -> ops 1,2
    //   3: LoadInt(0); SetSlot(1)          ; i = 0    -> ops 3,4
    //   5: LoadInt(1); LoadInt = sum + 1 — easier: AddAssignSlotVoid needs two slots.
    //      Instead: increment sum directly with PreIncSlotVoid(0).
    //   5: PreIncSlotVoid(0)                ; sum++   (body)
    //   6: SlotIncLtIntJumpBack(1, 4, 5)    ; ++i; if (i < 4) goto 5
    //   7: GetSlot(0)                        ; result
    let mut b = ChunkBuilder::new();
    b.emit(Op::PushFrame, 1);
    b.emit(Op::LoadInt(0), 1);
    b.emit(Op::SetSlot(0), 1);
    b.emit(Op::LoadInt(0), 1);
    b.emit(Op::SetSlot(1), 1);
    let body = b.current_pos();
    b.emit(Op::PreIncSlotVoid(0), 1);
    b.emit(Op::SlotIncLtIntJumpBack(1, 4, body), 1);
    b.emit(Op::GetSlot(0), 1);
    match VM::new(b.build()).run() {
        VMResult::Ok(Value::Int(4)) => {}
        other => panic!("got {:?}", other),
    }
}

// ── SlotArrayGet / SlotArraySet (slot-local arrays) ─────────────────────────

#[test]
fn slot_array_get_returns_undef_when_slot_is_not_array() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::PushFrame, 1);
    b.emit(Op::LoadInt(7), 1);
    b.emit(Op::SetSlot(0), 1); // slot 0 = Int, not Array
    b.emit(Op::LoadInt(0), 1); // index
    b.emit(Op::SlotArrayGet(0), 1);
    match VM::new(b.build()).run() {
        VMResult::Ok(Value::Undef) => {}
        other => panic!("got {:?}", other),
    }
}

#[test]
fn slot_array_set_then_get_round_trip() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::PushFrame, 1);
    // Initialize slot 0 with an empty array via MakeArray(0) + SetSlot.
    b.emit(Op::MakeArray(0), 1);
    b.emit(Op::SetSlot(0), 1);
    // a[3] = 99  → push 99, push 3, SlotArraySet
    b.emit(Op::LoadInt(99), 1);
    b.emit(Op::LoadInt(3), 1);
    b.emit(Op::SlotArraySet(0), 1);
    // Read it back: push index 3, SlotArrayGet
    b.emit(Op::LoadInt(3), 1);
    b.emit(Op::SlotArrayGet(0), 1);
    match VM::new(b.build()).run() {
        VMResult::Ok(Value::Int(99)) => {}
        other => panic!("got {:?}", other),
    }
}

#[test]
fn slot_array_get_out_of_bounds_returns_undef() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::PushFrame, 1);
    b.emit(Op::MakeArray(0), 1);
    b.emit(Op::SetSlot(0), 1);
    b.emit(Op::LoadInt(100), 1);
    b.emit(Op::SlotArrayGet(0), 1);
    match VM::new(b.build()).run() {
        VMResult::Ok(Value::Undef) => {}
        other => panic!("got {:?}", other),
    }
}

// ── Status ops ──────────────────────────────────────────────────────────────

#[test]
fn set_status_then_get_status_round_trip() {
    let r = build_and_run(|b| {
        b.emit(Op::LoadInt(7), 1);
        b.emit(Op::SetStatus, 1);
        b.emit(Op::GetStatus, 1);
    });
    match r {
        VMResult::Ok(Value::Status(7)) => {}
        other => panic!("got {:?}", other),
    }
}

#[test]
fn get_status_defaults_to_zero() {
    let r = build_and_run(|b| {
        b.emit(Op::GetStatus, 1);
    });
    match r {
        VMResult::Ok(Value::Status(0)) => {}
        other => panic!("got {:?}", other),
    }
}