use fusevm::{ChunkBuilder, Op, VMResult, Value, VM};

fn run(b: ChunkBuilder) -> Value {
    match VM::new(b.build()).run() {
        VMResult::Ok(v) => v,
        other => panic!("unexpected result: {:?}", other),
    }
}

// ── Basic slot ops ──

#[test]
fn set_get_slot_round_trip() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::PushFrame, 1);
    b.emit(Op::LoadInt(42), 1);
    b.emit(Op::SetSlot(0), 1);
    b.emit(Op::GetSlot(0), 1);
    assert_eq!(run(b), Value::Int(42));
}

#[test]
fn get_unset_slot_is_undef() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::PushFrame, 1);
    b.emit(Op::GetSlot(5), 1);
    assert!(matches!(run(b), Value::Undef));
}

#[test]
fn set_slot_resizes() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::PushFrame, 1);
    b.emit(Op::LoadInt(7), 1);
    b.emit(Op::SetSlot(10), 1);
    b.emit(Op::GetSlot(10), 1);
    assert_eq!(run(b), Value::Int(7));
}

#[test]
fn set_slot_overwrites() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::PushFrame, 1);
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::SetSlot(0), 1);
    b.emit(Op::LoadInt(99), 1);
    b.emit(Op::SetSlot(0), 1);
    b.emit(Op::GetSlot(0), 1);
    assert_eq!(run(b), Value::Int(99));
}

#[test]
fn multiple_slots_independent() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::PushFrame, 1);
    b.emit(Op::LoadInt(10), 1);
    b.emit(Op::SetSlot(0), 1);
    b.emit(Op::LoadInt(20), 1);
    b.emit(Op::SetSlot(1), 1);
    b.emit(Op::LoadInt(30), 1);
    b.emit(Op::SetSlot(2), 1);
    b.emit(Op::GetSlot(0), 1);
    b.emit(Op::GetSlot(2), 1);
    b.emit(Op::Add, 1);
    assert_eq!(run(b), Value::Int(40));
}

#[test]
fn slot_holds_string() {
    let mut b = ChunkBuilder::new();
    let k = b.add_constant(Value::str("hello"));
    b.emit(Op::PushFrame, 1);
    b.emit(Op::LoadConst(k), 1);
    b.emit(Op::SetSlot(0), 1);
    b.emit(Op::GetSlot(0), 1);
    match run(b) {
        Value::Str(s) => assert_eq!(&*s, "hello"),
        other => panic!("expected Str, got {:?}", other),
    }
}

#[test]
fn slot_holds_array() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::PushFrame, 1);
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::LoadInt(2), 1);
    b.emit(Op::LoadInt(3), 1);
    b.emit(Op::MakeArray(3), 1);
    b.emit(Op::SetSlot(0), 1);
    b.emit(Op::GetSlot(0), 1);
    match run(b) {
        Value::Array(a) => assert_eq!(a.len(), 3),
        other => panic!("expected array, got {:?}", other),
    }
}

// ── SlotArrayGet / SlotArraySet ──

#[test]
fn slot_array_get_basic() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::PushFrame, 1);
    b.emit(Op::LoadInt(10), 1);
    b.emit(Op::LoadInt(20), 1);
    b.emit(Op::LoadInt(30), 1);
    b.emit(Op::MakeArray(3), 1);
    b.emit(Op::SetSlot(0), 1);
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::SlotArrayGet(0), 1);
    assert_eq!(run(b), Value::Int(20));
}

#[test]
fn slot_array_get_oob_undef() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::PushFrame, 1);
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::MakeArray(1), 1);
    b.emit(Op::SetSlot(0), 1);
    b.emit(Op::LoadInt(99), 1);
    b.emit(Op::SlotArrayGet(0), 1);
    assert!(matches!(run(b), Value::Undef));
}

#[test]
fn slot_array_get_non_array_undef() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::PushFrame, 1);
    b.emit(Op::LoadInt(123), 1);
    b.emit(Op::SetSlot(0), 1);
    b.emit(Op::LoadInt(0), 1);
    b.emit(Op::SlotArrayGet(0), 1);
    assert!(matches!(run(b), Value::Undef));
}

#[test]
fn slot_array_set_basic() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::PushFrame, 1);
    b.emit(Op::LoadInt(10), 1);
    b.emit(Op::LoadInt(20), 1);
    b.emit(Op::LoadInt(30), 1);
    b.emit(Op::MakeArray(3), 1);
    b.emit(Op::SetSlot(0), 1);
    b.emit(Op::LoadInt(99), 1); // value
    b.emit(Op::LoadInt(1), 1); // index
    b.emit(Op::SlotArraySet(0), 1);
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::SlotArrayGet(0), 1);
    assert_eq!(run(b), Value::Int(99));
}

#[test]
fn slot_array_set_grows() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::PushFrame, 1);
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::MakeArray(1), 1);
    b.emit(Op::SetSlot(0), 1);
    b.emit(Op::LoadInt(42), 1);
    b.emit(Op::LoadInt(5), 1);
    b.emit(Op::SlotArraySet(0), 1);
    b.emit(Op::LoadInt(5), 1);
    b.emit(Op::SlotArrayGet(0), 1);
    assert_eq!(run(b), Value::Int(42));
}

#[test]
fn slot_array_set_grows_intermediate_undef() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::PushFrame, 1);
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::MakeArray(1), 1);
    b.emit(Op::SetSlot(0), 1);
    b.emit(Op::LoadInt(42), 1);
    b.emit(Op::LoadInt(3), 1);
    b.emit(Op::SlotArraySet(0), 1);
    b.emit(Op::LoadInt(2), 1);
    b.emit(Op::SlotArrayGet(0), 1);
    assert!(matches!(run(b), Value::Undef));
}

// ── PreIncSlot / PreIncSlotVoid ──

#[test]
fn pre_inc_slot_pushes_new_value() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::PushFrame, 1);
    b.emit(Op::LoadInt(0), 1);
    b.emit(Op::SetSlot(0), 1);
    b.emit(Op::PreIncSlot(0), 1);
    assert_eq!(run(b), Value::Int(1));
}

#[test]
fn pre_inc_slot_persists_value() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::PushFrame, 1);
    b.emit(Op::LoadInt(5), 1);
    b.emit(Op::SetSlot(0), 1);
    b.emit(Op::PreIncSlot(0), 1);
    b.emit(Op::Pop, 1);
    b.emit(Op::GetSlot(0), 1);
    assert_eq!(run(b), Value::Int(6));
}

#[test]
fn pre_inc_slot_undef_starts_at_one() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::PushFrame, 1);
    b.emit(Op::PreIncSlot(0), 1);
    assert_eq!(run(b), Value::Int(1));
}

#[test]
fn pre_inc_slot_void_no_stack_change() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::PushFrame, 1);
    b.emit(Op::LoadInt(7), 1);
    b.emit(Op::SetSlot(0), 1);
    b.emit(Op::PreIncSlotVoid(0), 1);
    b.emit(Op::GetSlot(0), 1);
    assert_eq!(run(b), Value::Int(8));
}

#[test]
fn pre_inc_slot_chained() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::PushFrame, 1);
    b.emit(Op::LoadInt(0), 1);
    b.emit(Op::SetSlot(0), 1);
    for _ in 0..10 {
        b.emit(Op::PreIncSlotVoid(0), 1);
    }
    b.emit(Op::GetSlot(0), 1);
    assert_eq!(run(b), Value::Int(10));
}

// ── AddAssignSlotVoid ──

#[test]
fn add_assign_slot_void_basic() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::PushFrame, 1);
    b.emit(Op::LoadInt(10), 1);
    b.emit(Op::SetSlot(0), 1);
    b.emit(Op::LoadInt(7), 1);
    b.emit(Op::SetSlot(1), 1);
    b.emit(Op::AddAssignSlotVoid(0, 1), 1);
    b.emit(Op::GetSlot(0), 1);
    assert_eq!(run(b), Value::Int(17));
}

#[test]
fn add_assign_slot_void_self() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::PushFrame, 1);
    b.emit(Op::LoadInt(5), 1);
    b.emit(Op::SetSlot(0), 1);
    b.emit(Op::AddAssignSlotVoid(0, 0), 1);
    b.emit(Op::GetSlot(0), 1);
    assert_eq!(run(b), Value::Int(10));
}

#[test]
fn add_assign_slot_void_doesnt_touch_other() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::PushFrame, 1);
    b.emit(Op::LoadInt(10), 1);
    b.emit(Op::SetSlot(0), 1);
    b.emit(Op::LoadInt(7), 1);
    b.emit(Op::SetSlot(1), 1);
    b.emit(Op::AddAssignSlotVoid(0, 1), 1);
    b.emit(Op::GetSlot(1), 1);
    assert_eq!(run(b), Value::Int(7));
}

// ── AccumSumLoop ──

#[test]
fn accum_sum_loop_basic_10() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::PushFrame, 1);
    b.emit(Op::LoadInt(0), 1);
    b.emit(Op::SetSlot(0), 1);
    b.emit(Op::LoadInt(0), 1);
    b.emit(Op::SetSlot(1), 1);
    b.emit(Op::AccumSumLoop(0, 1, 10), 1);
    b.emit(Op::GetSlot(0), 1);
    assert_eq!(run(b), Value::Int(45));
}

#[test]
fn accum_sum_loop_zero_iterations() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::PushFrame, 1);
    b.emit(Op::LoadInt(0), 1);
    b.emit(Op::SetSlot(0), 1);
    b.emit(Op::LoadInt(10), 1);
    b.emit(Op::SetSlot(1), 1);
    b.emit(Op::AccumSumLoop(0, 1, 5), 1);
    b.emit(Op::GetSlot(0), 1);
    assert_eq!(run(b), Value::Int(0));
}

#[test]
fn accum_sum_loop_index_advances() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::PushFrame, 1);
    b.emit(Op::LoadInt(0), 1);
    b.emit(Op::SetSlot(0), 1);
    b.emit(Op::LoadInt(0), 1);
    b.emit(Op::SetSlot(1), 1);
    b.emit(Op::AccumSumLoop(0, 1, 5), 1);
    b.emit(Op::GetSlot(1), 1);
    assert_eq!(run(b), Value::Int(5));
}

#[test]
fn accum_sum_loop_starts_from_existing() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::PushFrame, 1);
    b.emit(Op::LoadInt(100), 1);
    b.emit(Op::SetSlot(0), 1);
    b.emit(Op::LoadInt(0), 1);
    b.emit(Op::SetSlot(1), 1);
    b.emit(Op::AccumSumLoop(0, 1, 5), 1);
    b.emit(Op::GetSlot(0), 1);
    assert_eq!(run(b), Value::Int(110)); // 100 + 0+1+2+3+4
}

// ── ConcatConstLoop ──

#[test]
fn concat_const_loop_basic() {
    let mut b = ChunkBuilder::new();
    let c = b.add_constant(Value::str("ab"));
    b.emit(Op::PushFrame, 1);
    let init = b.add_constant(Value::str(""));
    b.emit(Op::LoadConst(init), 1);
    b.emit(Op::SetSlot(0), 1);
    b.emit(Op::LoadInt(0), 1);
    b.emit(Op::SetSlot(1), 1);
    b.emit(Op::ConcatConstLoop(c, 0, 1, 3), 1);
    b.emit(Op::GetSlot(0), 1);
    match run(b) {
        Value::Str(s) => assert_eq!(&*s, "ababab"),
        other => panic!("expected Str, got {:?}", other),
    }
}

#[test]
fn concat_const_loop_zero_iterations() {
    let mut b = ChunkBuilder::new();
    let c = b.add_constant(Value::str("x"));
    b.emit(Op::PushFrame, 1);
    let init = b.add_constant(Value::str("seed"));
    b.emit(Op::LoadConst(init), 1);
    b.emit(Op::SetSlot(0), 1);
    b.emit(Op::LoadInt(5), 1);
    b.emit(Op::SetSlot(1), 1);
    b.emit(Op::ConcatConstLoop(c, 0, 1, 2), 1);
    b.emit(Op::GetSlot(0), 1);
    match run(b) {
        Value::Str(s) => assert_eq!(&*s, "seed"),
        other => panic!("expected Str, got {:?}", other),
    }
}

#[test]
fn concat_const_loop_preserves_seed() {
    let mut b = ChunkBuilder::new();
    let c = b.add_constant(Value::str("!"));
    b.emit(Op::PushFrame, 1);
    let init = b.add_constant(Value::str("hi"));
    b.emit(Op::LoadConst(init), 1);
    b.emit(Op::SetSlot(0), 1);
    b.emit(Op::LoadInt(0), 1);
    b.emit(Op::SetSlot(1), 1);
    b.emit(Op::ConcatConstLoop(c, 0, 1, 3), 1);
    b.emit(Op::GetSlot(0), 1);
    match run(b) {
        Value::Str(s) => assert_eq!(&*s, "hi!!!"),
        other => panic!("expected Str, got {:?}", other),
    }
}

// ── PushIntRangeLoop ──

#[test]
fn push_int_range_loop_basic() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::PushFrame, 1);
    b.emit(Op::LoadInt(0), 1);
    b.emit(Op::SetSlot(0), 1);
    let empty = b.add_constant(Value::Array(vec![]));
    b.emit(Op::LoadConst(empty), 1);
    b.emit(Op::DeclareVar(0), 1);
    b.emit(Op::PushIntRangeLoop(0, 0, 5), 1);
    b.emit(Op::GetVar(0), 1);
    match run(b) {
        Value::Array(v) => {
            assert_eq!(v.len(), 5);
            assert_eq!(v[0], Value::Int(0));
            assert_eq!(v[4], Value::Int(4));
        }
        other => panic!("expected array, got {:?}", other),
    }
}

#[test]
fn push_int_range_loop_index_advances() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::PushFrame, 1);
    b.emit(Op::LoadInt(0), 1);
    b.emit(Op::SetSlot(0), 1);
    let empty = b.add_constant(Value::Array(vec![]));
    b.emit(Op::LoadConst(empty), 1);
    b.emit(Op::DeclareVar(0), 1);
    b.emit(Op::PushIntRangeLoop(0, 0, 3), 1);
    b.emit(Op::GetSlot(0), 1);
    assert_eq!(run(b), Value::Int(3));
}

#[test]
fn push_int_range_loop_non_array_var_creates_new() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::PushFrame, 1);
    b.emit(Op::LoadInt(0), 1);
    b.emit(Op::SetSlot(0), 1);
    b.emit(Op::LoadInt(999), 1);
    b.emit(Op::DeclareVar(0), 1);
    b.emit(Op::PushIntRangeLoop(0, 0, 2), 1);
    b.emit(Op::GetVar(0), 1);
    match run(b) {
        Value::Array(v) => assert_eq!(v.len(), 2),
        other => panic!("expected array, got {:?}", other),
    }
}

#[test]
fn push_int_range_loop_appends_to_existing() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::PushFrame, 1);
    b.emit(Op::LoadInt(10), 1);
    b.emit(Op::SetSlot(0), 1);
    let pre = b.add_constant(Value::Array(vec![Value::Int(7)]));
    b.emit(Op::LoadConst(pre), 1);
    b.emit(Op::DeclareVar(0), 1);
    b.emit(Op::PushIntRangeLoop(0, 0, 13), 1);
    b.emit(Op::GetVar(0), 1);
    match run(b) {
        Value::Array(v) => {
            assert_eq!(v.len(), 4); // 7 + (10, 11, 12)
            assert_eq!(v[0], Value::Int(7));
            assert_eq!(v[1], Value::Int(10));
            assert_eq!(v[3], Value::Int(12));
        }
        other => panic!("expected array, got {:?}", other),
    }
}

// ── SlotLtIntJumpIfFalse / SlotIncLtIntJumpBack ──

#[test]
fn slot_lt_int_jump_if_false_taken() {
    // Jumps past the LoadInt(99) when slot value >= 10
    let mut b = ChunkBuilder::new();
    b.emit(Op::PushFrame, 1);
    b.emit(Op::LoadInt(20), 1);
    b.emit(Op::SetSlot(0), 1);
    b.emit(Op::SlotLtIntJumpIfFalse(0, 10, 5), 1); // target = ip 5
    b.emit(Op::LoadInt(99), 1);
    b.emit(Op::Jump(6), 1);
    b.emit(Op::LoadInt(42), 1);
    assert_eq!(run(b), Value::Int(42));
}

#[test]
fn slot_lt_int_jump_if_false_not_taken() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::PushFrame, 1);
    b.emit(Op::LoadInt(3), 1);
    b.emit(Op::SetSlot(0), 1);
    b.emit(Op::SlotLtIntJumpIfFalse(0, 10, 999), 1);
    b.emit(Op::LoadInt(7), 1);
    assert_eq!(run(b), Value::Int(7));
}

#[test]
fn slot_inc_lt_int_jump_back_loop() {
    // i = 0; while ++i < 5 { goto back }; result = i
    let mut b = ChunkBuilder::new();
    b.emit(Op::PushFrame, 1);
    b.emit(Op::LoadInt(0), 1);
    b.emit(Op::SetSlot(0), 1);
    // SlotIncLtIntJumpBack itself is at ip 3; jump back to ip 3 to re-loop
    b.emit(Op::SlotIncLtIntJumpBack(0, 5, 3), 1);
    b.emit(Op::GetSlot(0), 1);
    assert_eq!(run(b), Value::Int(5));
}

#[test]
fn slot_inc_lt_int_jump_back_immediate_exit() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::PushFrame, 1);
    b.emit(Op::LoadInt(100), 1);
    b.emit(Op::SetSlot(0), 1);
    b.emit(Op::SlotIncLtIntJumpBack(0, 5, 2), 1);
    b.emit(Op::GetSlot(0), 1);
    assert_eq!(run(b), Value::Int(101));
}

// ── Frame isolation ──

#[test]
fn slot_scoped_to_frame() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::PushFrame, 1);
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::SetSlot(0), 1);
    b.emit(Op::PushFrame, 1);
    b.emit(Op::LoadInt(99), 1);
    b.emit(Op::SetSlot(0), 1);
    b.emit(Op::PopFrame, 1);
    b.emit(Op::GetSlot(0), 1);
    assert_eq!(run(b), Value::Int(1));
}

#[test]
fn slot_inner_frame_independent() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::PushFrame, 1);
    b.emit(Op::LoadInt(7), 1);
    b.emit(Op::SetSlot(0), 1);
    b.emit(Op::PushFrame, 1);
    b.emit(Op::GetSlot(0), 1); // inner slot 0 is Undef
    match run(b) {
        Value::Undef => {}
        other => panic!("expected Undef, got {:?}", other),
    }
}

// ── Higher-order block ops are documented stubs ──

#[test]
fn map_block_stub_is_noop() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(42), 1);
    b.emit(Op::MapBlock(0), 1);
    assert_eq!(run(b), Value::Int(42));
}

#[test]
fn grep_block_stub_is_noop() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(7), 1);
    b.emit(Op::GrepBlock(0), 1);
    assert_eq!(run(b), Value::Int(7));
}

#[test]
fn sort_block_stub_is_noop() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(3), 1);
    b.emit(Op::SortBlock(0), 1);
    assert_eq!(run(b), Value::Int(3));
}

#[test]
fn sort_default_stub_is_noop() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(2), 1);
    b.emit(Op::SortDefault, 1);
    assert_eq!(run(b), Value::Int(2));
}

#[test]
fn for_each_block_stub_is_noop() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(11), 1);
    b.emit(Op::ForEachBlock(0), 1);
    assert_eq!(run(b), Value::Int(11));
}
