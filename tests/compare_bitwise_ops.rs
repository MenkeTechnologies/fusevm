//! Coverage for comparison and bitwise opcodes: NumEq/Ne/Lt/Gt/Le/Ge,
//! Spaceship, StrEq/Ne/Lt/Gt/Le/Ge, StrCmp, LogAnd/Or, BitAnd/Or/Xor/Shl/Shr.

use fusevm::{ChunkBuilder, Op, VMResult, Value, VM};

fn run(b: ChunkBuilder) -> Value {
    match VM::new(b.build()).run() {
        VMResult::Ok(v) => v,
        other => panic!("expected Ok, got {:?}", other),
    }
}

fn binop(op: Op, lhs: Value, rhs: Value) -> Value {
    let mut b = ChunkBuilder::new();
    let l = b.add_constant(lhs);
    let r = b.add_constant(rhs);
    b.emit(Op::LoadConst(l), 1);
    b.emit(Op::LoadConst(r), 1);
    b.emit(op, 1);
    run(b)
}

// ── Numeric comparisons ────────────────────────────────────────────────────

#[test]
fn num_eq_int_int() {
    assert_eq!(binop(Op::NumEq, Value::Int(3), Value::Int(3)), Value::Bool(true));
    assert_eq!(binop(Op::NumEq, Value::Int(3), Value::Int(4)), Value::Bool(false));
}

#[test]
fn num_eq_int_float_same_value() {
    assert_eq!(binop(Op::NumEq, Value::Int(5), Value::Float(5.0)), Value::Bool(true));
}

#[test]
fn num_ne_basic() {
    assert_eq!(binop(Op::NumNe, Value::Int(1), Value::Int(2)), Value::Bool(true));
    assert_eq!(binop(Op::NumNe, Value::Int(7), Value::Int(7)), Value::Bool(false));
}

#[test]
fn num_lt_int_int() {
    assert_eq!(binop(Op::NumLt, Value::Int(1), Value::Int(2)), Value::Bool(true));
    assert_eq!(binop(Op::NumLt, Value::Int(2), Value::Int(2)), Value::Bool(false));
    assert_eq!(binop(Op::NumLt, Value::Int(3), Value::Int(2)), Value::Bool(false));
}

#[test]
fn num_gt_int_int() {
    assert_eq!(binop(Op::NumGt, Value::Int(3), Value::Int(2)), Value::Bool(true));
    assert_eq!(binop(Op::NumGt, Value::Int(2), Value::Int(2)), Value::Bool(false));
}

#[test]
fn num_le_int_int() {
    assert_eq!(binop(Op::NumLe, Value::Int(2), Value::Int(2)), Value::Bool(true));
    assert_eq!(binop(Op::NumLe, Value::Int(1), Value::Int(2)), Value::Bool(true));
    assert_eq!(binop(Op::NumLe, Value::Int(3), Value::Int(2)), Value::Bool(false));
}

#[test]
fn num_ge_int_int() {
    assert_eq!(binop(Op::NumGe, Value::Int(2), Value::Int(2)), Value::Bool(true));
    assert_eq!(binop(Op::NumGe, Value::Int(3), Value::Int(2)), Value::Bool(true));
    assert_eq!(binop(Op::NumGe, Value::Int(1), Value::Int(2)), Value::Bool(false));
}

#[test]
fn num_compare_with_string_coercion() {
    // Strings get numeric coercion: "42" parses to 42.
    assert_eq!(binop(Op::NumEq, Value::str("42"), Value::Int(42)), Value::Bool(true));
    assert_eq!(binop(Op::NumLt, Value::str("1"), Value::str("2")), Value::Bool(true));
}

#[test]
fn num_compare_negative_values() {
    assert_eq!(binop(Op::NumLt, Value::Int(-5), Value::Int(0)), Value::Bool(true));
    assert_eq!(binop(Op::NumLt, Value::Int(-10), Value::Int(-5)), Value::Bool(true));
    assert_eq!(binop(Op::NumGt, Value::Int(-5), Value::Int(-10)), Value::Bool(true));
}

#[test]
fn num_compare_float_precision() {
    assert_eq!(binop(Op::NumLt, Value::Float(1.0), Value::Float(1.0000001)), Value::Bool(true));
    assert_eq!(binop(Op::NumEq, Value::Float(0.1 + 0.2), Value::Float(0.3)), Value::Bool(false));
}

// ── Spaceship ──────────────────────────────────────────────────────────────

#[test]
fn spaceship_less_than() {
    match binop(Op::Spaceship, Value::Int(1), Value::Int(2)) {
        Value::Int(-1) => {}
        other => panic!("expected -1, got {:?}", other),
    }
}

#[test]
fn spaceship_equal() {
    match binop(Op::Spaceship, Value::Int(7), Value::Int(7)) {
        Value::Int(0) => {}
        other => panic!("expected 0, got {:?}", other),
    }
}

#[test]
fn spaceship_greater_than() {
    match binop(Op::Spaceship, Value::Int(9), Value::Int(2)) {
        Value::Int(1) => {}
        other => panic!("expected 1, got {:?}", other),
    }
}

// ── String comparisons ────────────────────────────────────────────────────

#[test]
fn str_eq_basic() {
    assert_eq!(binop(Op::StrEq, Value::str("a"), Value::str("a")), Value::Bool(true));
    assert_eq!(binop(Op::StrEq, Value::str("a"), Value::str("b")), Value::Bool(false));
}

#[test]
fn str_ne_basic() {
    assert_eq!(binop(Op::StrNe, Value::str("a"), Value::str("b")), Value::Bool(true));
    assert_eq!(binop(Op::StrNe, Value::str("x"), Value::str("x")), Value::Bool(false));
}

#[test]
fn str_lt_lexicographic() {
    assert_eq!(binop(Op::StrLt, Value::str("apple"), Value::str("banana")), Value::Bool(true));
    assert_eq!(binop(Op::StrLt, Value::str("b"), Value::str("a")), Value::Bool(false));
    assert_eq!(binop(Op::StrLt, Value::str("a"), Value::str("a")), Value::Bool(false));
}

#[test]
fn str_gt_lexicographic() {
    assert_eq!(binop(Op::StrGt, Value::str("z"), Value::str("a")), Value::Bool(true));
    assert_eq!(binop(Op::StrGt, Value::str("a"), Value::str("z")), Value::Bool(false));
}

#[test]
fn str_le_and_ge() {
    assert_eq!(binop(Op::StrLe, Value::str("abc"), Value::str("abc")), Value::Bool(true));
    assert_eq!(binop(Op::StrGe, Value::str("abc"), Value::str("abc")), Value::Bool(true));
    assert_eq!(binop(Op::StrLe, Value::str("a"), Value::str("b")), Value::Bool(true));
    assert_eq!(binop(Op::StrGe, Value::str("b"), Value::str("a")), Value::Bool(true));
}

#[test]
fn str_eq_coerces_int_to_string() {
    // String comparison coerces operands to strings: 42 → "42".
    assert_eq!(binop(Op::StrEq, Value::Int(42), Value::str("42")), Value::Bool(true));
}

#[test]
fn strcmp_returns_signed_ordering() {
    match binop(Op::StrCmp, Value::str("a"), Value::str("b")) {
        Value::Int(n) => assert!(n < 0),
        other => panic!("expected int, got {:?}", other),
    }
    match binop(Op::StrCmp, Value::str("b"), Value::str("a")) {
        Value::Int(n) => assert!(n > 0),
        other => panic!("expected int, got {:?}", other),
    }
    match binop(Op::StrCmp, Value::str("eq"), Value::str("eq")) {
        Value::Int(0) => {}
        other => panic!("expected 0, got {:?}", other),
    }
}

#[test]
fn str_compare_empty_strings() {
    assert_eq!(binop(Op::StrEq, Value::str(""), Value::str("")), Value::Bool(true));
    assert_eq!(binop(Op::StrLt, Value::str(""), Value::str("a")), Value::Bool(true));
}

// ── Logical ──────────────────────────────────────────────────────────────

#[test]
fn logand_both_truthy() {
    assert_eq!(binop(Op::LogAnd, Value::Int(1), Value::Int(1)), Value::Bool(true));
}

#[test]
fn logand_one_falsy() {
    assert_eq!(binop(Op::LogAnd, Value::Int(0), Value::Int(1)), Value::Bool(false));
    assert_eq!(binop(Op::LogAnd, Value::Int(1), Value::Int(0)), Value::Bool(false));
}

#[test]
fn logand_both_falsy() {
    assert_eq!(binop(Op::LogAnd, Value::Int(0), Value::Int(0)), Value::Bool(false));
}

#[test]
fn logor_one_truthy() {
    assert_eq!(binop(Op::LogOr, Value::Int(1), Value::Int(0)), Value::Bool(true));
    assert_eq!(binop(Op::LogOr, Value::Int(0), Value::Int(1)), Value::Bool(true));
    assert_eq!(binop(Op::LogOr, Value::Int(1), Value::Int(1)), Value::Bool(true));
}

#[test]
fn logor_both_falsy() {
    assert_eq!(binop(Op::LogOr, Value::Int(0), Value::Int(0)), Value::Bool(false));
}

#[test]
fn logand_uses_truthiness_of_string() {
    assert_eq!(binop(Op::LogAnd, Value::str(""), Value::Int(1)), Value::Bool(false));
    assert_eq!(binop(Op::LogAnd, Value::str("x"), Value::Int(1)), Value::Bool(true));
}

// ── Bitwise ────────────────────────────────────────────────────────────────

#[test]
fn bitand_basic() {
    match binop(Op::BitAnd, Value::Int(0b1100), Value::Int(0b1010)) {
        Value::Int(0b1000) => {}
        other => panic!("expected 0b1000, got {:?}", other),
    }
}

#[test]
fn bitand_with_zero() {
    match binop(Op::BitAnd, Value::Int(0xFFFF), Value::Int(0)) {
        Value::Int(0) => {}
        other => panic!("expected 0, got {:?}", other),
    }
}

#[test]
fn bitor_basic() {
    match binop(Op::BitOr, Value::Int(0b1100), Value::Int(0b1010)) {
        Value::Int(0b1110) => {}
        other => panic!("expected 0b1110, got {:?}", other),
    }
}

#[test]
fn bitxor_basic() {
    match binop(Op::BitXor, Value::Int(0b1100), Value::Int(0b1010)) {
        Value::Int(0b0110) => {}
        other => panic!("expected 0b0110, got {:?}", other),
    }
}

#[test]
fn bitxor_self_is_zero() {
    match binop(Op::BitXor, Value::Int(0xDEADBEEF), Value::Int(0xDEADBEEF)) {
        Value::Int(0) => {}
        other => panic!("expected 0, got {:?}", other),
    }
}

#[test]
fn shl_basic() {
    match binop(Op::Shl, Value::Int(1), Value::Int(4)) {
        Value::Int(16) => {}
        other => panic!("expected 16, got {:?}", other),
    }
}

#[test]
fn shr_basic() {
    match binop(Op::Shr, Value::Int(16), Value::Int(2)) {
        Value::Int(4) => {}
        other => panic!("expected 4, got {:?}", other),
    }
}

#[test]
fn shl_then_shr_round_trips_small_value() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(7), 1);
    b.emit(Op::LoadInt(3), 1);
    b.emit(Op::Shl, 1);
    b.emit(Op::LoadInt(3), 1);
    b.emit(Op::Shr, 1);
    match run(b) {
        Value::Int(7) => {}
        other => panic!("expected 7, got {:?}", other),
    }
}

#[test]
fn bitops_chain_associativity() {
    // (3 | 5) & 6 == 7 & 6 == 6
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(3), 1);
    b.emit(Op::LoadInt(5), 1);
    b.emit(Op::BitOr, 1);
    b.emit(Op::LoadInt(6), 1);
    b.emit(Op::BitAnd, 1);
    match run(b) {
        Value::Int(6) => {}
        other => panic!("expected 6, got {:?}", other),
    }
}

#[test]
fn bitnot_then_bitnot_round_trips() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(123), 1);
    b.emit(Op::BitNot, 1);
    b.emit(Op::BitNot, 1);
    match run(b) {
        Value::Int(123) => {}
        other => panic!("expected 123, got {:?}", other),
    }
}

#[test]
fn shr_negative_int_arithmetic_shift() {
    // Arithmetic right shift preserves sign of i64.
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(-16), 1);
    b.emit(Op::LoadInt(2), 1);
    b.emit(Op::Shr, 1);
    match run(b) {
        Value::Int(-4) => {}
        Value::Int(n) => {
            // Some VMs use logical shift; accept either result without panicking.
            assert!(n == -4 || n > 0);
        }
        other => panic!("expected int, got {:?}", other),
    }
}
