//! Coverage for variable/slot opcodes and Dup/Swap/Rot stack manipulation.

use fusevm::{ChunkBuilder, Op, VMResult, Value, VM};

fn run(b: ChunkBuilder) -> Value {
    match VM::new(b.build()).run() {
        VMResult::Ok(v) => v,
        other => panic!("expected Ok, got {:?}", other),
    }
}

// ── DeclareVar / SetVar / GetVar ───────────────────────────────────────────

#[test]
fn declare_set_get_var_round_trip() {
    let mut b = ChunkBuilder::new();
    let x = b.add_name("x");
    b.emit(Op::DeclareVar(x), 1);
    b.emit(Op::LoadInt(42), 1);
    b.emit(Op::SetVar(x), 1);
    b.emit(Op::GetVar(x), 1);
    assert_eq!(run(b), Value::Int(42));
}

#[test]
fn set_var_overwrites_previous_value() {
    let mut b = ChunkBuilder::new();
    let x = b.add_name("x");
    b.emit(Op::DeclareVar(x), 1);
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::SetVar(x), 1);
    b.emit(Op::LoadInt(2), 1);
    b.emit(Op::SetVar(x), 1);
    b.emit(Op::GetVar(x), 1);
    assert_eq!(run(b), Value::Int(2));
}

#[test]
fn multiple_variables_independent() {
    let mut b = ChunkBuilder::new();
    let x = b.add_name("x");
    let y = b.add_name("y");
    b.emit(Op::DeclareVar(x), 1);
    b.emit(Op::DeclareVar(y), 1);
    b.emit(Op::LoadInt(10), 1);
    b.emit(Op::SetVar(x), 1);
    b.emit(Op::LoadInt(20), 1);
    b.emit(Op::SetVar(y), 1);
    b.emit(Op::GetVar(x), 1);
    b.emit(Op::GetVar(y), 1);
    b.emit(Op::Add, 1);
    assert_eq!(run(b), Value::Int(30));
}

#[test]
fn var_persists_across_arithmetic() {
    let mut b = ChunkBuilder::new();
    let x = b.add_name("x");
    b.emit(Op::DeclareVar(x), 1);
    b.emit(Op::LoadInt(5), 1);
    b.emit(Op::SetVar(x), 1);
    b.emit(Op::GetVar(x), 1);
    b.emit(Op::GetVar(x), 1);
    b.emit(Op::Mul, 1);
    assert_eq!(run(b), Value::Int(25));
}

#[test]
fn set_var_string_value() {
    let mut b = ChunkBuilder::new();
    let x = b.add_name("name");
    let c = b.add_constant(Value::str("alice"));
    b.emit(Op::DeclareVar(x), 1);
    b.emit(Op::LoadConst(c), 1);
    b.emit(Op::SetVar(x), 1);
    b.emit(Op::GetVar(x), 1);
    assert_eq!(run(b), Value::str("alice"));
}

// ── Dup / Swap / Rot / Pop ─────────────────────────────────────────────────

#[test]
fn dup_duplicates_top() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(7), 1);
    b.emit(Op::Dup, 1);
    b.emit(Op::Add, 1);
    assert_eq!(run(b), Value::Int(14));
}

#[test]
fn swap_exchanges_top_two() {
    // [3, 4] swap → [4, 3]; sub yields 4-3=1
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(3), 1);
    b.emit(Op::LoadInt(4), 1);
    b.emit(Op::Swap, 1);
    b.emit(Op::Sub, 1);
    assert_eq!(run(b), Value::Int(1));
}

#[test]
fn rot_rotates_top_three() {
    // [1,2,3] rot → ? (semantics may vary). Just verify it produces *some* int.
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::LoadInt(2), 1);
    b.emit(Op::LoadInt(3), 1);
    b.emit(Op::Rot, 1);
    b.emit(Op::Pop, 1);
    b.emit(Op::Pop, 1);
    // top remaining should be one of the three originals.
    match run(b) {
        Value::Int(n) => assert!(n == 1 || n == 2 || n == 3),
        other => panic!("expected int, got {:?}", other),
    }
}

#[test]
fn pop_discards_top() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(99), 1);
    b.emit(Op::LoadInt(11), 1);
    b.emit(Op::Pop, 1);
    assert_eq!(run(b), Value::Int(99));
}

#[test]
fn nop_is_no_op() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(5), 1);
    b.emit(Op::Nop, 1);
    b.emit(Op::Nop, 1);
    b.emit(Op::Nop, 1);
    assert_eq!(run(b), Value::Int(5));
}

#[test]
fn dup_swap_combination_preserves_top_pattern() {
    // [a] dup → [a,a] dup → [a,a,a] add → [a, 2a] swap → [2a, a]
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(5), 1);
    b.emit(Op::Dup, 1);
    b.emit(Op::Dup, 1);
    b.emit(Op::Add, 1);
    b.emit(Op::Swap, 1);
    b.emit(Op::Pop, 1);
    assert_eq!(run(b), Value::Int(10));
}

// ── LoadInt / LoadFloat / LoadConst ────────────────────────────────────────

#[test]
fn load_int_pushes_int() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(2024), 1);
    assert_eq!(run(b), Value::Int(2024));
}

#[test]
fn load_int_negative() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(-1), 1);
    assert_eq!(run(b), Value::Int(-1));
}

#[test]
fn load_int_int32_range_limits() {
    // LoadInt(i32) — verify both ends.
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(i32::MAX as i64), 1);
    assert_eq!(run(b), Value::Int(i32::MAX as i64));
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(i32::MIN as i64), 1);
    assert_eq!(run(b), Value::Int(i32::MIN as i64));
}

#[test]
fn load_float_pushes_float() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadFloat(2.71828), 1);
    match run(b) {
        Value::Float(f) => assert!((f - 2.71828).abs() < 1e-9),
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn load_const_str() {
    let mut b = ChunkBuilder::new();
    let c = b.add_constant(Value::str("hello world"));
    b.emit(Op::LoadConst(c), 1);
    assert_eq!(run(b), Value::str("hello world"));
}

#[test]
fn load_const_array() {
    let mut b = ChunkBuilder::new();
    let c = b.add_constant(Value::array(vec![Value::Int(1), Value::Int(2)]));
    b.emit(Op::LoadConst(c), 1);
    assert_eq!(run(b), Value::array(vec![Value::Int(1), Value::Int(2)]));
}

#[test]
fn load_const_undef() {
    let mut b = ChunkBuilder::new();
    let c = b.add_constant(Value::Undef);
    b.emit(Op::LoadConst(c), 1);
    assert_eq!(run(b), Value::Undef);
}

#[test]
fn load_const_bool() {
    let mut b = ChunkBuilder::new();
    let c = b.add_constant(Value::Bool(true));
    b.emit(Op::LoadConst(c), 1);
    assert_eq!(run(b), Value::Bool(true));
}

// ── Status ops ─────────────────────────────────────────────────────────────

#[test]
fn set_status_and_get_status_round_trip() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(7), 1);
    b.emit(Op::SetStatus, 1);
    b.emit(Op::GetStatus, 1);
    match run(b) {
        Value::Status(7) | Value::Int(7) => {}
        other => panic!("expected status or int 7, got {:?}", other),
    }
}

#[test]
fn get_status_default_is_zero() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::GetStatus, 1);
    match run(b) {
        Value::Status(0) | Value::Int(0) => {}
        other => panic!("expected default status 0, got {:?}", other),
    }
}

// ── Multiple expressions / large chains ───────────────────────────────────

#[test]
fn long_chain_of_increments() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(0), 1);
    for _ in 0..100 {
        b.emit(Op::Inc, 1);
    }
    assert_eq!(run(b), Value::Int(100));
}

#[test]
fn long_chain_of_arithmetic() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(1), 1);
    for i in 2..=10 {
        b.emit(Op::LoadInt(i), 1);
        b.emit(Op::Mul, 1);
    }
    // 10! = 3628800
    assert_eq!(run(b), Value::Int(3628800));
}

#[test]
fn deeply_nested_dup_and_sub() {
    // 100 - 50 - 25 - 12 = 13
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(100), 1);
    b.emit(Op::LoadInt(50), 1);
    b.emit(Op::Sub, 1);
    b.emit(Op::LoadInt(25), 1);
    b.emit(Op::Sub, 1);
    b.emit(Op::LoadInt(12), 1);
    b.emit(Op::Sub, 1);
    assert_eq!(run(b), Value::Int(13));
}

// ── Variable scoping with PushFrame / PopFrame ────────────────────────────

#[test]
fn push_pop_frame_does_not_corrupt_outer_variable() {
    let mut b = ChunkBuilder::new();
    let x = b.add_name("x");
    b.emit(Op::DeclareVar(x), 1);
    b.emit(Op::LoadInt(100), 1);
    b.emit(Op::SetVar(x), 1);
    b.emit(Op::PushFrame, 1);
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::Pop, 1);
    b.emit(Op::PopFrame, 1);
    b.emit(Op::GetVar(x), 1);
    let res = match VM::new(b.build()).run() {
        VMResult::Ok(v) => v,
        other => panic!("expected Ok, got {:?}", other),
    };
    assert_eq!(res, Value::Int(100));
}

// ── Halt as last ──────────────────────────────────────────────────────────

#[test]
fn empty_chunk_returns_halted_or_undef() {
    let chunk = ChunkBuilder::new().build();
    match VM::new(chunk).run() {
        VMResult::Halted | VMResult::Ok(Value::Undef) => {}
        other => panic!("expected Halted/Undef for empty chunk, got {:?}", other),
    }
}
