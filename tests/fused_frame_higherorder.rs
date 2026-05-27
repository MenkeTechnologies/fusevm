//! Coverage for fused superinstructions, scope frame ops, higher-order
//! block stubs, and `CallBuiltin` dispatch.

use fusevm::{ChunkBuilder, Op, VMResult, Value, VM};

fn run(b: ChunkBuilder) -> Value {
    match VM::new(b.build()).run() {
        VMResult::Ok(v) => v,
        VMResult::Halted => Value::Undef,
        VMResult::Error(e) => panic!("unexpected VM error: {e}"),
    }
}

fn i(v: Value) -> i64 {
    match v {
        Value::Int(n) => n,
        other => panic!("expected Int, got {:?}", other),
    }
}

// ══════════════════════════════════════════════════════════════════════════
// Fused: PreIncSlot / PreIncSlotVoid
// ══════════════════════════════════════════════════════════════════════════

#[test]
fn preincslot_increments_and_pushes_new_value() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(5), 1);
    b.emit(Op::SetSlot(0), 1);
    b.emit(Op::PreIncSlot(0), 1);
    assert_eq!(i(run(b)), 6);
}

#[test]
fn preincslot_on_unset_slot_treats_as_zero_then_increments() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::PreIncSlot(0), 1);
    assert_eq!(i(run(b)), 1);
}

#[test]
fn preincslot_followed_by_getslot_observes_increment() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(9), 1);
    b.emit(Op::SetSlot(0), 1);
    b.emit(Op::PreIncSlot(0), 1);
    b.emit(Op::Pop, 1);
    b.emit(Op::GetSlot(0), 1);
    assert_eq!(i(run(b)), 10);
}

#[test]
fn preincslotvoid_increments_without_pushing() {
    // Push sentinel; PreIncSlotVoid doesn't push so sentinel remains on top.
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(0), 1);
    b.emit(Op::SetSlot(0), 1);
    b.emit(Op::LoadInt(777), 1);
    b.emit(Op::PreIncSlotVoid(0), 1);
    // Stack top still 777; slot 0 == 1.
    assert_eq!(i(run(b)), 777);
}

#[test]
fn preincslotvoid_updates_slot_value() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(40), 1);
    b.emit(Op::SetSlot(2), 1);
    b.emit(Op::PreIncSlotVoid(2), 1);
    b.emit(Op::PreIncSlotVoid(2), 1);
    b.emit(Op::GetSlot(2), 1);
    assert_eq!(i(run(b)), 42);
}

// ══════════════════════════════════════════════════════════════════════════
// Fused: SlotLtIntJumpIfFalse
// ══════════════════════════════════════════════════════════════════════════

#[test]
fn slotltintjumpiffalse_jumps_when_slot_geq_limit() {
    // Layout:
    //   0: SetSlot(0) ← put 10 in slot
    //   1: SlotLtIntJumpIfFalse(slot=0, limit=5, target=4)
    //   2: LoadInt(99)  ← should be skipped
    //   3: Halt? no — fall to 4: LoadInt(77)
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(10), 1);
    b.emit(Op::SetSlot(0), 1);
    b.emit(Op::SlotLtIntJumpIfFalse(0, 5, 5), 1);
    b.emit(Op::LoadInt(99), 1);
    b.emit(Op::LoadInt(88), 1);
    b.emit(Op::LoadInt(77), 1);
    // After dispatch: jumped past 99 and 88, landed on 77.
    assert_eq!(i(run(b)), 77);
}

#[test]
fn slotltintjumpiffalse_falls_through_when_slot_lt_limit() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(2), 1);
    b.emit(Op::SetSlot(0), 1);
    b.emit(Op::SlotLtIntJumpIfFalse(0, 5, 99), 1);
    b.emit(Op::LoadInt(7), 1);
    assert_eq!(i(run(b)), 7);
}

#[test]
fn slotltintjumpiffalse_unset_slot_is_zero_so_falls_through() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SlotLtIntJumpIfFalse(0, 1, 99), 1);
    b.emit(Op::LoadInt(42), 1);
    assert_eq!(i(run(b)), 42);
}

// ══════════════════════════════════════════════════════════════════════════
// Fused: SlotIncLtIntJumpBack
// ══════════════════════════════════════════════════════════════════════════

#[test]
fn slotincltintjumpback_does_finite_loop() {
    // slot 0 = 0; each iteration increments slot 0 then jumps back to start
    // while slot < 3. After loop: slot == 3.
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(0), 1);
    b.emit(Op::SetSlot(0), 1);
    let loop_start = b.emit(Op::Nop, 1);
    b.emit(Op::SlotIncLtIntJumpBack(0, 3, loop_start), 1);
    b.emit(Op::GetSlot(0), 1);
    assert_eq!(i(run(b)), 3);
}

#[test]
fn slotincltintjumpback_exits_when_reaching_limit() {
    // slot starts at limit-1; one increment makes it equal limit and exits.
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(2), 1);
    b.emit(Op::SetSlot(0), 1);
    let after = b.emit(Op::Nop, 1);
    b.emit(Op::SlotIncLtIntJumpBack(0, 3, after), 1);
    b.emit(Op::GetSlot(0), 1);
    assert_eq!(i(run(b)), 3);
}

// ══════════════════════════════════════════════════════════════════════════
// Fused: AccumSumLoop / AddAssignSlotVoid
// ══════════════════════════════════════════════════════════════════════════

#[test]
fn accumsumloop_sums_range_from_i_to_limit_minus_one() {
    // sum slot starts at 0; i slot starts at 1; sum += i for i in [1,5);
    //   1 + 2 + 3 + 4 = 10
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(0), 1);
    b.emit(Op::SetSlot(0), 1);
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::SetSlot(1), 1);
    b.emit(Op::AccumSumLoop(0, 1, 5), 1);
    b.emit(Op::GetSlot(0), 1);
    assert_eq!(i(run(b)), 10);
}

#[test]
fn accumsumloop_i_geq_limit_does_nothing() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(100), 1);
    b.emit(Op::SetSlot(0), 1);
    b.emit(Op::LoadInt(5), 1);
    b.emit(Op::SetSlot(1), 1);
    b.emit(Op::AccumSumLoop(0, 1, 5), 1);
    b.emit(Op::GetSlot(0), 1);
    assert_eq!(i(run(b)), 100);
}

#[test]
fn addassignslotvoid_does_a_plus_equals_b_in_place() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(10), 1);
    b.emit(Op::SetSlot(0), 1);
    b.emit(Op::LoadInt(3), 1);
    b.emit(Op::SetSlot(1), 1);
    b.emit(Op::AddAssignSlotVoid(0, 1), 1);
    b.emit(Op::GetSlot(0), 1);
    assert_eq!(i(run(b)), 13);
    // and slot 1 unchanged
}

#[test]
fn addassignslotvoid_does_not_push_to_stack() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(5), 1);
    b.emit(Op::SetSlot(0), 1);
    b.emit(Op::LoadInt(7), 1);
    b.emit(Op::SetSlot(1), 1);
    b.emit(Op::LoadInt(999), 1); // sentinel
    b.emit(Op::AddAssignSlotVoid(0, 1), 1);
    assert_eq!(i(run(b)), 999);
}

// ══════════════════════════════════════════════════════════════════════════
// Fused: ConcatConstLoop
// ══════════════════════════════════════════════════════════════════════════

#[test]
fn concatconstloop_appends_constant_n_times() {
    let mut b = ChunkBuilder::new();
    let c = b.add_constant(Value::str("ab"));
    b.emit(Op::LoadConst(c), 1);
    b.emit(Op::SetSlot(0), 1); // s_slot holds initial string "ab"
    let initial = b.add_constant(Value::str(""));
    b.emit(Op::LoadConst(initial), 1);
    b.emit(Op::SetSlot(0), 1); // reset s_slot to ""
    b.emit(Op::LoadInt(0), 1);
    b.emit(Op::SetSlot(1), 1); // i_slot = 0
    b.emit(Op::ConcatConstLoop(c, 0, 1, 3), 1);
    b.emit(Op::GetSlot(0), 1);
    match run(b) {
        Value::Str(s) => assert_eq!(s.as_str(), "ababab"),
        other => panic!("expected Str, got {:?}", other),
    }
}

#[test]
fn concatconstloop_no_iterations_keeps_initial_string() {
    let mut b = ChunkBuilder::new();
    let init = b.add_constant(Value::str("start"));
    b.emit(Op::LoadConst(init), 1);
    b.emit(Op::SetSlot(0), 1);
    b.emit(Op::LoadInt(5), 1);
    b.emit(Op::SetSlot(1), 1); // i == 5, limit == 5 → no iterations
    let c = b.add_constant(Value::str("X"));
    b.emit(Op::ConcatConstLoop(c, 0, 1, 5), 1);
    b.emit(Op::GetSlot(0), 1);
    match run(b) {
        Value::Str(s) => assert_eq!(s.as_str(), "start"),
        other => panic!("expected Str, got {:?}", other),
    }
}

// ══════════════════════════════════════════════════════════════════════════
// Fused: PushIntRangeLoop
// ══════════════════════════════════════════════════════════════════════════

#[test]
fn pushintrangeloop_extends_array_with_range_values() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::DeclareArray(0), 1);
    b.emit(Op::LoadInt(0), 1);
    b.emit(Op::SetSlot(0), 1); // i_slot = 0
    b.emit(Op::PushIntRangeLoop(0, 0, 4), 1);
    b.emit(Op::ArrayLen(0), 1);
    assert_eq!(i(run(b)), 4);
}

#[test]
fn pushintrangeloop_with_nonempty_initial_appends() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::DeclareArray(0), 1);
    b.emit(Op::LoadInt(99), 1);
    b.emit(Op::ArrayPush(0), 1);
    b.emit(Op::LoadInt(0), 1);
    b.emit(Op::SetSlot(0), 1);
    b.emit(Op::PushIntRangeLoop(0, 0, 3), 1);
    b.emit(Op::ArrayLen(0), 1);
    assert_eq!(i(run(b)), 4);
}

// ══════════════════════════════════════════════════════════════════════════
// PushFrame / PopFrame — scope discards top-of-stack on exit
// ══════════════════════════════════════════════════════════════════════════

#[test]
fn pushframe_popframe_discards_intermediate_stack_values() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(7), 1);
    b.emit(Op::PushFrame, 1);
    b.emit(Op::LoadInt(99), 1);
    b.emit(Op::LoadInt(99), 1);
    b.emit(Op::PopFrame, 1);
    // After PopFrame, stack should be truncated back to height-1 (just the 7).
    assert_eq!(i(run(b)), 7);
}

#[test]
fn popframe_without_pushframe_does_not_panic() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(5), 1);
    b.emit(Op::PopFrame, 1);
    // Should not crash; behavior with no active frame is implementation-defined
    // but must be safe (no underflow / panic).
    let _ = run(b);
}

#[test]
fn nested_push_pop_frames() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::PushFrame, 1);
    b.emit(Op::LoadInt(2), 1);
    b.emit(Op::PushFrame, 1);
    b.emit(Op::LoadInt(3), 1);
    b.emit(Op::PopFrame, 1); // discards 3
    b.emit(Op::PopFrame, 1); // discards 2
    assert_eq!(i(run(b)), 1);
}

// ══════════════════════════════════════════════════════════════════════════
// Higher-order block stubs — currently no-ops, must not crash
// ══════════════════════════════════════════════════════════════════════════

#[test]
fn mapblock_is_currently_a_noop_in_vm() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(42), 1);
    b.emit(Op::MapBlock(0), 1);
    assert_eq!(i(run(b)), 42);
}

#[test]
fn grepblock_is_currently_a_noop_in_vm() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(7), 1);
    b.emit(Op::GrepBlock(0), 1);
    assert_eq!(i(run(b)), 7);
}

#[test]
fn sortblock_is_currently_a_noop_in_vm() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(9), 1);
    b.emit(Op::SortBlock(0), 1);
    assert_eq!(i(run(b)), 9);
}

#[test]
fn sortdefault_is_currently_a_noop_in_vm() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(11), 1);
    b.emit(Op::SortDefault, 1);
    assert_eq!(i(run(b)), 11);
}

#[test]
fn foreachblock_is_currently_a_noop_in_vm() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(33), 1);
    b.emit(Op::ForEachBlock(0), 1);
    assert_eq!(i(run(b)), 33);
}

// ══════════════════════════════════════════════════════════════════════════
// CallBuiltin — registered handler is invoked; unregistered id is silent
// ══════════════════════════════════════════════════════════════════════════

#[test]
fn callbuiltin_invokes_registered_handler_and_pushes_result() {
    fn h(_vm: &mut VM, argc: u8) -> Value {
        Value::Int(argc as i64 * 2)
    }
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(7, 4), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(7, h);
    match vm.run() {
        VMResult::Ok(Value::Int(8)) => {}
        other => panic!("expected Int(8), got {:?}", other),
    }
}

#[test]
fn callbuiltin_unregistered_does_not_panic() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(123), 1);
    b.emit(Op::CallBuiltin(254, 0), 1);
    // No builtin registered at id 254 → silent no-op, stack still has 123.
    assert_eq!(i(run(b)), 123);
}

#[test]
fn callbuiltin_handler_can_pop_arguments_from_stack() {
    fn sum(vm: &mut VM, argc: u8) -> Value {
        let mut s = 0;
        for _ in 0..argc {
            s += vm.pop().to_int();
        }
        Value::Int(s)
    }
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(10), 1);
    b.emit(Op::LoadInt(20), 1);
    b.emit(Op::LoadInt(30), 1);
    b.emit(Op::CallBuiltin(5, 3), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(5, sum);
    match vm.run() {
        VMResult::Ok(Value::Int(60)) => {}
        other => panic!("expected Int(60), got {:?}", other),
    }
}
