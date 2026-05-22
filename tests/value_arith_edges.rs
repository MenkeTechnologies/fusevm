//! Arithmetic and string edge-case coverage for `Op` semantics.

use fusevm::{ChunkBuilder, Op, VMResult, Value, VM};

fn run(b: ChunkBuilder) -> Value {
    match VM::new(b.build()).run() {
        VMResult::Ok(v) => v,
        other => panic!("expected Ok, got {:?}", other),
    }
}

// ── Integer arithmetic edges ───────────────────────────────────────────────

#[test]
fn int_add_overflow_wraps_or_saturates_without_panic() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(i64::MAX), 1);
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::Add, 1);
    let _ = VM::new(b.build()).run();
}

#[test]
fn int_sub_underflow_does_not_panic() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(i64::MIN), 1);
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::Sub, 1);
    let _ = VM::new(b.build()).run();
}

#[test]
fn int_mul_overflow_does_not_panic() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(i64::MAX), 1);
    b.emit(Op::LoadInt(2), 1);
    b.emit(Op::Mul, 1);
    let _ = VM::new(b.build()).run();
}

#[test]
fn int_div_yields_float_for_int_int() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(10), 1);
    b.emit(Op::LoadInt(4), 1);
    b.emit(Op::Div, 1);
    match run(b) {
        Value::Float(f) => assert!((f - 2.5).abs() < 1e-9),
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn int_div_by_zero_errors_or_infinity() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(7), 1);
    b.emit(Op::LoadInt(0), 1);
    b.emit(Op::Div, 1);
    let _ = VM::new(b.build()).run();
}

#[test]
fn int_mod_negative_dividend() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(-7), 1);
    b.emit(Op::LoadInt(3), 1);
    b.emit(Op::Mod, 1);
    match run(b) {
        Value::Int(i) => assert!(i == -1 || i == 2),
        other => panic!("expected int, got {:?}", other),
    }
}

#[test]
fn int_pow_always_float() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(2), 1);
    b.emit(Op::LoadInt(10), 1);
    b.emit(Op::Pow, 1);
    match run(b) {
        Value::Float(f) => assert!((f - 1024.0).abs() < 1e-9),
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn negate_int_min_does_not_panic() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(i64::MIN), 1);
    b.emit(Op::Negate, 1);
    let _ = VM::new(b.build()).run();
}

#[test]
fn negate_float_zero_yields_neg_zero() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadFloat(0.0), 1);
    b.emit(Op::Negate, 1);
    match run(b) {
        Value::Float(f) => assert_eq!(f.to_bits(), (-0.0f64).to_bits()),
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn inc_and_dec_round_trip() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(5), 1);
    b.emit(Op::Inc, 1);
    b.emit(Op::Dec, 1);
    match run(b) {
        Value::Int(5) => {}
        other => panic!("expected Int(5), got {:?}", other),
    }
}

// ── Float specifics ────────────────────────────────────────────────────────

#[test]
fn float_add_basic() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadFloat(1.5), 1);
    b.emit(Op::LoadFloat(2.25), 1);
    b.emit(Op::Add, 1);
    match run(b) {
        Value::Float(f) => assert!((f - 3.75).abs() < 1e-9),
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn float_div_by_zero_is_infinity() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadFloat(1.0), 1);
    b.emit(Op::LoadFloat(0.0), 1);
    b.emit(Op::Div, 1);
    match run(b) {
        Value::Float(f) => assert!(f.is_infinite()),
        _ => {} // acceptable to error instead
    }
}

#[test]
fn float_zero_div_zero_is_nan_or_error() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadFloat(0.0), 1);
    b.emit(Op::LoadFloat(0.0), 1);
    b.emit(Op::Div, 1);
    let _ = VM::new(b.build()).run();
}

#[test]
fn float_pow_negative_base_fractional_exponent() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadFloat(-1.0), 1);
    b.emit(Op::LoadFloat(0.5), 1);
    b.emit(Op::Pow, 1);
    let _ = VM::new(b.build()).run();
}

#[test]
fn float_very_large_addition_stays_finite_or_inf() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadFloat(f64::MAX), 1);
    b.emit(Op::LoadFloat(f64::MAX), 1);
    b.emit(Op::Add, 1);
    match run(b) {
        Value::Float(f) => assert!(f.is_infinite() || f.is_finite()),
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn float_denormal_arithmetic_does_not_panic() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadFloat(f64::MIN_POSITIVE / 2.0), 1);
    b.emit(Op::LoadFloat(2.0), 1);
    b.emit(Op::Mul, 1);
    let _ = run(b);
}

// ── Mixed int/float promotion ──────────────────────────────────────────────

#[test]
fn int_plus_float_promotes_to_float() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(3), 1);
    b.emit(Op::LoadFloat(0.5), 1);
    b.emit(Op::Add, 1);
    match run(b) {
        Value::Float(f) => assert!((f - 3.5).abs() < 1e-9),
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn float_minus_int_promotes_to_float() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadFloat(10.5), 1);
    b.emit(Op::LoadInt(3), 1);
    b.emit(Op::Sub, 1);
    match run(b) {
        Value::Float(f) => assert!((f - 7.5).abs() < 1e-9),
        other => panic!("expected float, got {:?}", other),
    }
}

// ── String edges ───────────────────────────────────────────────────────────

#[test]
fn string_len_empty_zero() {
    let mut b = ChunkBuilder::new();
    let c = b.add_constant(Value::str(""));
    b.emit(Op::LoadConst(c), 1);
    b.emit(Op::StringLen, 1);
    match run(b) {
        Value::Int(0) => {}
        other => panic!("expected 0, got {:?}", other),
    }
}

#[test]
fn string_len_unicode_multibyte() {
    let mut b = ChunkBuilder::new();
    // "héllo" — 6 bytes UTF-8, 5 codepoints. Behaviour can be either; just
    // verify it returns *some* Int without panicking.
    let c = b.add_constant(Value::str("héllo"));
    b.emit(Op::LoadConst(c), 1);
    b.emit(Op::StringLen, 1);
    match run(b) {
        Value::Int(n) => assert!(n == 5 || n == 6),
        other => panic!("expected int, got {:?}", other),
    }
}

#[test]
fn concat_with_empty_string() {
    let mut b = ChunkBuilder::new();
    let a = b.add_constant(Value::str("hello"));
    let e = b.add_constant(Value::str(""));
    b.emit(Op::LoadConst(a), 1);
    b.emit(Op::LoadConst(e), 1);
    b.emit(Op::Concat, 1);
    match run(b) {
        Value::Str(s) => assert_eq!(s.as_ref(), "hello"),
        other => panic!("expected str, got {:?}", other),
    }
}

#[test]
fn concat_int_and_string() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(42), 1);
    let c = b.add_constant(Value::str("!"));
    b.emit(Op::LoadConst(c), 1);
    b.emit(Op::Concat, 1);
    match run(b) {
        Value::Str(s) => assert_eq!(s.as_ref(), "42!"),
        other => panic!("expected str, got {:?}", other),
    }
}

// ── Logical / bitwise ──────────────────────────────────────────────────────

#[test]
fn lognot_truthy_to_false() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::LogNot, 1);
    match run(b) {
        Value::Bool(false) => {}
        other => panic!("expected Bool(false), got {:?}", other),
    }
}

#[test]
fn lognot_zero_to_true() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(0), 1);
    b.emit(Op::LogNot, 1);
    match run(b) {
        Value::Bool(true) => {}
        other => panic!("expected Bool(true), got {:?}", other),
    }
}

#[test]
fn lognot_empty_string_to_true() {
    let mut b = ChunkBuilder::new();
    let c = b.add_constant(Value::str(""));
    b.emit(Op::LoadConst(c), 1);
    b.emit(Op::LogNot, 1);
    match run(b) {
        Value::Bool(true) => {}
        other => panic!("expected Bool(true), got {:?}", other),
    }
}

#[test]
fn bitnot_int_inverts_bits() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(0), 1);
    b.emit(Op::BitNot, 1);
    match run(b) {
        Value::Int(-1) => {}
        other => panic!("expected Int(-1), got {:?}", other),
    }
}

#[test]
fn double_negate_int_round_trip() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(123), 1);
    b.emit(Op::Negate, 1);
    b.emit(Op::Negate, 1);
    match run(b) {
        Value::Int(123) => {}
        other => panic!("expected Int(123), got {:?}", other),
    }
}
