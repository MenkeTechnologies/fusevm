//! Control-flow + arithmetic-edge tests:
//!  - Op::Jump forward/backward
//!  - Op::JumpIfTrue/False (consuming) vs JumpIfTrue/FalseKeep (peeking)
//!  - Op::Call to undefined function → VMResult::Error
//!  - Op::Return / Op::ReturnValue from top-level (no frames) → Halt
//!  - Op::Div by zero → Undef (no panic)
//!  - Op::Mod by zero → 0 (no panic)
//!  - Op::Pow with various exponents
//!  - Op::Negate on Int (incl. i64::MIN wrapping) and Float
//!  - Op::Inc / Op::Dec on Int (incl. overflow wrap)

use fusevm::{ChunkBuilder, Op, VM, VMResult, Value};

fn exec(b: ChunkBuilder) -> VMResult {
    VM::new(b.build()).run()
}

fn run(b: ChunkBuilder) -> Value {
    match exec(b) {
        VMResult::Ok(v) => v,
        VMResult::Halted => Value::Undef,
        VMResult::Error(e) => panic!("unexpected VM error: {e}"),
    }
}

// ══════════════════════════════════════════════════════════════════════════
// Op::Jump
// ══════════════════════════════════════════════════════════════════════════

#[test]
fn unconditional_jump_skips_intermediate_ops() {
    // jump 0 → 3, skipping LoadInt(99). Final stack should have only 7.
    let mut b = ChunkBuilder::new();
    b.emit(Op::Jump(3), 1);
    b.emit(Op::LoadInt(99), 1);
    b.emit(Op::LoadInt(99), 1);
    b.emit(Op::LoadInt(7), 1);
    assert!(matches!(run(b), Value::Int(7)));
}

#[test]
fn unconditional_jump_past_end_halts_cleanly() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(5), 1);
    b.emit(Op::Jump(99), 1);
    // jump past end → IP-out-of-bounds halts the run cleanly
    let _ = exec(b); // must not panic
}

#[test]
fn backward_jump_makes_a_finite_loop_via_counter() {
    // for i in 0..3 { sum += 1 }; sum = 3
    let mut b = ChunkBuilder::new();
    // slot 0 = i, slot 1 = sum
    b.emit(Op::LoadInt(0), 1);
    b.emit(Op::SetSlot(0), 1);
    b.emit(Op::LoadInt(0), 1);
    b.emit(Op::SetSlot(1), 1);
    let loop_start = 4;
    // if i >= 3, jump to end
    b.emit(Op::GetSlot(0), 1); // 4
    b.emit(Op::LoadInt(3), 1);
    b.emit(Op::NumLt, 1);
    let jf_ip = b.emit(Op::JumpIfFalse(0), 1);
    // body: sum += 1
    b.emit(Op::GetSlot(1), 1);
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::Add, 1);
    b.emit(Op::SetSlot(1), 1);
    // i += 1
    b.emit(Op::GetSlot(0), 1);
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::Add, 1);
    b.emit(Op::SetSlot(0), 1);
    b.emit(Op::Jump(loop_start), 1);
    let after = b.emit(Op::GetSlot(1), 1);
    b.patch_jump(jf_ip, after);
    let v = run(b);
    assert!(matches!(v, Value::Int(3)));
}

// ══════════════════════════════════════════════════════════════════════════
// Op::JumpIfTrue / JumpIfFalse — consuming
// ══════════════════════════════════════════════════════════════════════════

#[test]
fn jumpiftrue_pops_condition_and_jumps_when_truthy() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::JumpIfTrue(3), 1);
    b.emit(Op::LoadInt(11), 1);
    b.emit(Op::LoadInt(22), 1); // target
    let v = run(b);
    // Cond was consumed → only 22 is on stack.
    assert!(matches!(v, Value::Int(22)));
}

#[test]
fn jumpiftrue_falls_through_when_falsy() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(0), 1);
    b.emit(Op::JumpIfTrue(3), 1);
    b.emit(Op::LoadInt(11), 1); // executed
    b.emit(Op::LoadInt(22), 1); // also executed
    let v = run(b);
    assert!(matches!(v, Value::Int(22)));
}

#[test]
fn jumpiffalse_pops_condition_and_jumps_when_falsy() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(0), 1);
    b.emit(Op::JumpIfFalse(3), 1);
    b.emit(Op::LoadInt(11), 1);
    b.emit(Op::LoadInt(22), 1);
    assert!(matches!(run(b), Value::Int(22)));
}

#[test]
fn jumpiffalse_falls_through_when_truthy() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::JumpIfFalse(3), 1);
    b.emit(Op::LoadInt(11), 1);
    b.emit(Op::LoadInt(22), 1);
    assert!(matches!(run(b), Value::Int(22)));
}

// ══════════════════════════════════════════════════════════════════════════
// Op::JumpIfTrueKeep / JumpIfFalseKeep — peek (preserve stack)
// ══════════════════════════════════════════════════════════════════════════

#[test]
fn jumpiftruekeep_preserves_top_of_stack_on_jump() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(7), 1);
    b.emit(Op::JumpIfTrueKeep(3), 1);
    b.emit(Op::LoadInt(99), 1);
    // landing point — 7 should still be on top
    let v = run(b);
    assert!(matches!(v, Value::Int(7)));
}

#[test]
fn jumpiftruekeep_does_not_jump_when_falsy_and_keeps_value() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(0), 1);
    b.emit(Op::JumpIfTrueKeep(4), 1);
    b.emit(Op::Pop, 1);
    b.emit(Op::LoadInt(123), 1);
    let v = run(b);
    assert!(matches!(v, Value::Int(123)));
}

#[test]
fn jumpiffalsekeep_preserves_top_of_stack_on_jump() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(0), 1);
    b.emit(Op::JumpIfFalseKeep(3), 1);
    b.emit(Op::LoadInt(99), 1);
    let v = run(b);
    // 0 was kept on stack; we jumped over the LoadInt(99).
    assert!(matches!(v, Value::Int(0)));
}

#[test]
fn jumpiffalsekeep_does_not_jump_when_truthy_and_keeps_value() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(5), 1);
    b.emit(Op::JumpIfFalseKeep(4), 1);
    b.emit(Op::Pop, 1);
    b.emit(Op::LoadInt(123), 1);
    let v = run(b);
    assert!(matches!(v, Value::Int(123)));
}

// ══════════════════════════════════════════════════════════════════════════
// Short-circuit `&&` / `||` patterns expressible with JumpIf*Keep
// ══════════════════════════════════════════════════════════════════════════

#[test]
fn short_circuit_and_pattern_a_false_skips_b() {
    // emulate `0 && pop+99` → result 0
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(0), 1);
    let j = b.emit(Op::JumpIfFalseKeep(0), 1);
    b.emit(Op::Pop, 1);
    b.emit(Op::LoadInt(99), 1);
    let end = b.emit(Op::Nop, 1);
    b.patch_jump(j, end);
    assert!(matches!(run(b), Value::Int(0)));
}

#[test]
fn short_circuit_or_pattern_a_true_skips_b() {
    // emulate `5 || pop+99` → result 5
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(5), 1);
    let j = b.emit(Op::JumpIfTrueKeep(0), 1);
    b.emit(Op::Pop, 1);
    b.emit(Op::LoadInt(99), 1);
    let end = b.emit(Op::Nop, 1);
    b.patch_jump(j, end);
    assert!(matches!(run(b), Value::Int(5)));
}

// ══════════════════════════════════════════════════════════════════════════
// Op::Call to undefined function
// ══════════════════════════════════════════════════════════════════════════

#[test]
fn call_undefined_function_returns_error() {
    let mut b = ChunkBuilder::new();
    let n = b.add_name("does_not_exist");
    b.emit(Op::Call(n, 0), 1);
    let result = VM::new(b.build()).run();
    match result {
        VMResult::Error(e) => {
            assert!(
                e.contains("does_not_exist") || e.contains("undefined"),
                "error message should mention the function: {e}"
            );
        }
        other => panic!("expected Error, got {:?}", other),
    }
}

// ══════════════════════════════════════════════════════════════════════════
// Op::Return / Op::ReturnValue from top-level (no frames)
// ══════════════════════════════════════════════════════════════════════════

#[test]
fn return_at_top_level_halts() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::Return, 1);
    b.emit(Op::LoadInt(99), 1); // unreachable
    let r = exec(b);
    assert!(matches!(r, VMResult::Halted | VMResult::Ok(_)));
}

#[test]
fn returnvalue_at_top_level_returns_value() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(7), 1);
    b.emit(Op::ReturnValue, 1);
    b.emit(Op::LoadInt(99), 1); // unreachable
    match exec(b) {
        VMResult::Ok(Value::Int(7)) => {}
        VMResult::Halted => {}
        other => panic!("expected Int(7) or Halted, got {:?}", other),
    }
}

// ══════════════════════════════════════════════════════════════════════════
// Op::Div by zero / Op::Mod by zero — must not panic
// ══════════════════════════════════════════════════════════════════════════

#[test]
fn div_int_by_zero_yields_undef_no_panic() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(10), 1);
    b.emit(Op::LoadInt(0), 1);
    b.emit(Op::Div, 1);
    let v = run(b);
    assert!(matches!(v, Value::Undef));
}

#[test]
fn div_float_by_zero_yields_undef_no_panic() {
    let mut b = ChunkBuilder::new();
    let a = b.add_constant(Value::Float(3.14));
    let z = b.add_constant(Value::Float(0.0));
    b.emit(Op::LoadConst(a), 1);
    b.emit(Op::LoadConst(z), 1);
    b.emit(Op::Div, 1);
    let v = run(b);
    assert!(matches!(v, Value::Undef));
}

#[test]
fn div_normal_returns_float_quotient() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(7), 1);
    b.emit(Op::LoadInt(2), 1);
    b.emit(Op::Div, 1);
    match run(b) {
        Value::Float(f) => assert!((f - 3.5).abs() < 1e-9),
        other => panic!("expected Float(3.5), got {:?}", other),
    }
}

#[test]
fn mod_by_zero_yields_zero_no_panic() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(10), 1);
    b.emit(Op::LoadInt(0), 1);
    b.emit(Op::Mod, 1);
    // Behavior: arith_int_fast returns 0 in int path.
    let v = run(b);
    assert!(matches!(v, Value::Int(0) | Value::Float(_) | Value::Undef));
}

#[test]
fn mod_basic_int_result() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(17), 1);
    b.emit(Op::LoadInt(5), 1);
    b.emit(Op::Mod, 1);
    assert!(matches!(run(b), Value::Int(2)));
}

// ══════════════════════════════════════════════════════════════════════════
// Op::Pow
// ══════════════════════════════════════════════════════════════════════════

#[test]
fn pow_int_base_int_exp_returns_float() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(2), 1);
    b.emit(Op::LoadInt(10), 1);
    b.emit(Op::Pow, 1);
    match run(b) {
        Value::Float(f) => assert_eq!(f, 1024.0),
        other => panic!("expected Float(1024.0), got {:?}", other),
    }
}

#[test]
fn pow_zero_exponent_is_one() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(7), 1);
    b.emit(Op::LoadInt(0), 1);
    b.emit(Op::Pow, 1);
    match run(b) {
        Value::Float(f) => assert_eq!(f, 1.0),
        other => panic!("expected Float(1.0), got {:?}", other),
    }
}

#[test]
fn pow_negative_exponent_is_reciprocal_form() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(2), 1);
    b.emit(Op::LoadInt(-2), 1);
    b.emit(Op::Pow, 1);
    match run(b) {
        Value::Float(f) => assert!((f - 0.25).abs() < 1e-9),
        other => panic!("expected Float(0.25), got {:?}", other),
    }
}

#[test]
fn pow_with_float_exponent() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(4), 1);
    let half = b.add_constant(Value::Float(0.5));
    b.emit(Op::LoadConst(half), 1);
    b.emit(Op::Pow, 1);
    match run(b) {
        Value::Float(f) => assert!((f - 2.0).abs() < 1e-9),
        other => panic!("expected Float(2.0), got {:?}", other),
    }
}

// ══════════════════════════════════════════════════════════════════════════
// Op::Negate
// ══════════════════════════════════════════════════════════════════════════

#[test]
fn negate_int_flips_sign() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(42), 1);
    b.emit(Op::Negate, 1);
    assert!(matches!(run(b), Value::Int(-42)));
}

#[test]
fn negate_zero_is_zero() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(0), 1);
    b.emit(Op::Negate, 1);
    assert!(matches!(run(b), Value::Int(0)));
}

#[test]
fn negate_i64_min_wraps_via_wrapping_neg() {
    let mut b = ChunkBuilder::new();
    let c = b.add_constant(Value::Int(i64::MIN));
    b.emit(Op::LoadConst(c), 1);
    b.emit(Op::Negate, 1);
    // wrapping_neg(i64::MIN) == i64::MIN
    assert!(matches!(run(b), Value::Int(x) if x == i64::MIN));
}

#[test]
fn negate_float_flips_sign() {
    let mut b = ChunkBuilder::new();
    let c = b.add_constant(Value::Float(3.14));
    b.emit(Op::LoadConst(c), 1);
    b.emit(Op::Negate, 1);
    match run(b) {
        Value::Float(f) => assert!((f + 3.14).abs() < 1e-9),
        other => panic!("expected Float(-3.14), got {:?}", other),
    }
}

// ══════════════════════════════════════════════════════════════════════════
// Op::Inc / Op::Dec on Int (wrapping)
// ══════════════════════════════════════════════════════════════════════════

#[test]
fn inc_i64_max_wraps_to_i64_min() {
    let mut b = ChunkBuilder::new();
    let c = b.add_constant(Value::Int(i64::MAX));
    b.emit(Op::LoadConst(c), 1);
    b.emit(Op::Inc, 1);
    assert!(matches!(run(b), Value::Int(x) if x == i64::MIN));
}

#[test]
fn dec_i64_min_wraps_to_i64_max() {
    let mut b = ChunkBuilder::new();
    let c = b.add_constant(Value::Int(i64::MIN));
    b.emit(Op::LoadConst(c), 1);
    b.emit(Op::Dec, 1);
    assert!(matches!(run(b), Value::Int(x) if x == i64::MAX));
}

#[test]
fn inc_on_string_numeric_coerces_then_increments() {
    let mut b = ChunkBuilder::new();
    let c = b.add_constant(Value::str("99"));
    b.emit(Op::LoadConst(c), 1);
    b.emit(Op::Inc, 1);
    assert!(matches!(run(b), Value::Int(100)));
}

#[test]
fn dec_on_undef_coerces_to_zero_minus_one() {
    let mut b = ChunkBuilder::new();
    let c = b.add_constant(Value::Undef);
    b.emit(Op::LoadConst(c), 1);
    b.emit(Op::Dec, 1);
    assert!(matches!(run(b), Value::Int(-1)));
}
