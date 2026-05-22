//! Targeted coverage for stack manipulation, arithmetic, range, and small
//! miscellaneous opcodes that prior rounds touched only lightly.

use fusevm::chunk::ChunkBuilder;
use fusevm::op::Op;
use fusevm::value::Value;
use fusevm::vm::{VMResult, VM};

fn run(b: ChunkBuilder) -> Value {
    match VM::new(b.build()).run() {
        VMResult::Ok(v) => v,
        other => panic!("unexpected result: {:?}", other),
    }
}

fn run_emit(ops: Vec<Op>) -> Value {
    let mut b = ChunkBuilder::new();
    for op in ops {
        b.emit(op, 1);
    }
    run(b)
}

// ── Stack manipulation ──────────────────────────────────────────────────

#[test]
fn dup_duplicates_top() {
    assert_eq!(
        run_emit(vec![Op::LoadInt(7), Op::Dup, Op::Add]),
        Value::Int(14)
    );
}

#[test]
fn dup2_duplicates_top_two() {
    // [1, 2] -> [1, 2, 1, 2] -> Add => [1, 2, 3] -> top=3
    assert_eq!(
        run_emit(vec![Op::LoadInt(1), Op::LoadInt(2), Op::Dup2, Op::Add]),
        Value::Int(3)
    );
}

#[test]
fn swap_exchanges_top_two() {
    // 10 - 3 = 7, but with swap: 3 - 10 = -7
    assert_eq!(
        run_emit(vec![Op::LoadInt(10), Op::LoadInt(3), Op::Swap, Op::Sub]),
        Value::Int(-7)
    );
}

#[test]
fn rot_rotates_top_three() {
    // [1, 2, 3] -- Rot --> [3, 1, 2] (per common stack semantics) or [2, 3, 1]
    // Result depends on impl; just verify top changes from 3.
    let v = run_emit(vec![
        Op::LoadInt(1),
        Op::LoadInt(2),
        Op::LoadInt(3),
        Op::Rot,
    ]);
    // top is now either 1 or 2 (not 3)
    assert_ne!(v, Value::Int(3));
}

#[test]
fn pop_discards_top() {
    assert_eq!(
        run_emit(vec![Op::LoadInt(1), Op::LoadInt(2), Op::Pop]),
        Value::Int(1)
    );
}

// ── Loaders ─────────────────────────────────────────────────────────────

#[test]
fn load_true_false_undef() {
    assert_eq!(run_emit(vec![Op::LoadTrue]), Value::Bool(true));
    assert_eq!(run_emit(vec![Op::LoadFalse]), Value::Bool(false));
    assert_eq!(run_emit(vec![Op::LoadUndef]), Value::Undef);
}

#[test]
fn load_float_pushes_float() {
    let v = run_emit(vec![Op::LoadFloat(2.5)]);
    assert_eq!(v, Value::Float(2.5));
}

#[test]
fn load_const_resolves_pool_index() {
    let mut b = ChunkBuilder::new();
    let k = b.add_constant(Value::str("hello"));
    b.emit(Op::LoadConst(k), 1);
    assert_eq!(run(b), Value::str("hello"));
}

#[test]
fn nop_is_pass_through() {
    assert_eq!(
        run_emit(vec![Op::LoadInt(5), Op::Nop, Op::Nop]),
        Value::Int(5)
    );
}

// ── Arithmetic edges ────────────────────────────────────────────────────

#[test]
fn pow_produces_float() {
    let v = run_emit(vec![Op::LoadInt(2), Op::LoadInt(10), Op::Pow]);
    match v {
        Value::Float(f) => assert!((f - 1024.0).abs() < 1e-9),
        _ => panic!("expected float"),
    }
}

#[test]
fn pow_zero_to_zero_is_one() {
    let v = run_emit(vec![Op::LoadInt(0), Op::LoadInt(0), Op::Pow]);
    if let Value::Float(f) = v {
        assert_eq!(f, 1.0);
    } else {
        panic!("expected float")
    }
}

#[test]
fn pow_negative_base_fractional_exp_is_nan() {
    let v = run_emit(vec![Op::LoadInt(-1), Op::LoadFloat(0.5), Op::Pow]);
    if let Value::Float(f) = v {
        assert!(f.is_nan());
    } else {
        panic!("expected float")
    }
}

#[test]
fn negate_int_uses_wrapping() {
    let v = run_emit(vec![Op::LoadInt(i64::MIN), Op::Negate]);
    assert_eq!(v, Value::Int(i64::MIN));
}

#[test]
fn negate_non_int_falls_to_float() {
    let v = run_emit(vec![Op::LoadFloat(3.5), Op::Negate]);
    assert_eq!(v, Value::Float(-3.5));
}

#[test]
fn inc_int_wraps_at_max() {
    let v = run_emit(vec![Op::LoadInt(i64::MAX), Op::Inc]);
    assert_eq!(v, Value::Int(i64::MIN));
}

#[test]
fn dec_int_wraps_at_min() {
    let v = run_emit(vec![Op::LoadInt(i64::MIN), Op::Dec]);
    assert_eq!(v, Value::Int(i64::MAX));
}

#[test]
fn inc_non_int_coerces_then_increments() {
    let v = run_emit(vec![Op::LoadFloat(1.7), Op::Inc]);
    // 1.7 -> to_int = 1, then 1+1 = 2 (Int variant)
    assert_eq!(v, Value::Int(2));
}

#[test]
fn add_with_mixed_int_float_promotes() {
    let v = run_emit(vec![Op::LoadInt(1), Op::LoadFloat(2.5), Op::Add]);
    assert_eq!(v, Value::Float(3.5));
}

#[test]
fn mod_with_zero_does_not_crash() {
    // Implementation-defined: just ensure no panic.
    let _ = run_emit(vec![Op::LoadInt(5), Op::LoadInt(0), Op::Mod]);
}

#[test]
fn div_with_zero_does_not_crash() {
    let _ = run_emit(vec![Op::LoadInt(5), Op::LoadInt(0), Op::Div]);
}

// ── String ops ──────────────────────────────────────────────────────────

#[test]
fn concat_str_str() {
    let mut b = ChunkBuilder::new();
    let a = b.add_constant(Value::str("foo"));
    let c = b.add_constant(Value::str("bar"));
    b.emit(Op::LoadConst(a), 1);
    b.emit(Op::LoadConst(c), 1);
    b.emit(Op::Concat, 1);
    assert_eq!(run(b), Value::str("foobar"));
}

#[test]
fn concat_int_and_str_stringifies() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(42), 1);
    let s = b.add_constant(Value::str("x"));
    b.emit(Op::LoadConst(s), 1);
    b.emit(Op::Concat, 1);
    assert_eq!(run(b), Value::str("42x"));
}

#[test]
fn string_repeat_basic() {
    let mut b = ChunkBuilder::new();
    let s = b.add_constant(Value::str("ab"));
    b.emit(Op::LoadConst(s), 1);
    b.emit(Op::LoadInt(3), 1);
    b.emit(Op::StringRepeat, 1);
    assert_eq!(run(b), Value::str("ababab"));
}

#[test]
fn string_repeat_zero_yields_empty() {
    let mut b = ChunkBuilder::new();
    let s = b.add_constant(Value::str("abc"));
    b.emit(Op::LoadConst(s), 1);
    b.emit(Op::LoadInt(0), 1);
    b.emit(Op::StringRepeat, 1);
    assert_eq!(run(b), Value::str(""));
}

#[test]
fn string_repeat_negative_clamps_to_zero() {
    let mut b = ChunkBuilder::new();
    let s = b.add_constant(Value::str("abc"));
    b.emit(Op::LoadConst(s), 1);
    b.emit(Op::LoadInt(-5), 1);
    b.emit(Op::StringRepeat, 1);
    assert_eq!(run(b), Value::str(""));
}

#[test]
fn string_len_for_ascii_bytes() {
    let mut b = ChunkBuilder::new();
    let s = b.add_constant(Value::str("hello"));
    b.emit(Op::LoadConst(s), 1);
    b.emit(Op::StringLen, 1);
    assert_eq!(run(b), Value::Int(5));
}

#[test]
fn string_len_for_unicode_returns_byte_len() {
    let mut b = ChunkBuilder::new();
    let s = b.add_constant(Value::str("héllo"));
    b.emit(Op::LoadConst(s), 1);
    b.emit(Op::StringLen, 1);
    // 'é' = 2 bytes utf-8 -> 6
    assert_eq!(run(b), Value::Int(6));
}

// ── Spaceship & StrCmp ──────────────────────────────────────────────────

#[test]
fn spaceship_int_lt() {
    assert_eq!(
        run_emit(vec![Op::LoadInt(1), Op::LoadInt(2), Op::Spaceship]),
        Value::Int(-1)
    );
}

#[test]
fn spaceship_int_eq() {
    assert_eq!(
        run_emit(vec![Op::LoadInt(5), Op::LoadInt(5), Op::Spaceship]),
        Value::Int(0)
    );
}

#[test]
fn spaceship_int_gt() {
    assert_eq!(
        run_emit(vec![Op::LoadInt(9), Op::LoadInt(3), Op::Spaceship]),
        Value::Int(1)
    );
}

#[test]
fn spaceship_mixed_uses_float() {
    assert_eq!(
        run_emit(vec![Op::LoadFloat(2.5), Op::LoadInt(2), Op::Spaceship]),
        Value::Int(1)
    );
}

#[test]
fn strcmp_lex_order() {
    let mut b = ChunkBuilder::new();
    let a = b.add_constant(Value::str("apple"));
    let z = b.add_constant(Value::str("banana"));
    b.emit(Op::LoadConst(a), 1);
    b.emit(Op::LoadConst(z), 1);
    b.emit(Op::StrCmp, 1);
    let v = run(b);
    if let Value::Int(n) = v {
        assert!(n < 0);
    } else {
        panic!("expected int")
    }
}

#[test]
fn strcmp_equal() {
    let mut b = ChunkBuilder::new();
    let a = b.add_constant(Value::str("same"));
    let z = b.add_constant(Value::str("same"));
    b.emit(Op::LoadConst(a), 1);
    b.emit(Op::LoadConst(z), 1);
    b.emit(Op::StrCmp, 1);
    assert_eq!(run(b), Value::Int(0));
}

// ── String comparisons (StrEq/Ne/Lt/Gt/Le/Ge) ───────────────────────────

fn cmp(a: &str, op: Op, b: &str) -> Value {
    let mut bld = ChunkBuilder::new();
    let ai = bld.add_constant(Value::str(a));
    let bi = bld.add_constant(Value::str(b));
    bld.emit(Op::LoadConst(ai), 1);
    bld.emit(Op::LoadConst(bi), 1);
    bld.emit(op, 1);
    run(bld)
}

#[test]
fn str_eq_true_for_identical() {
    assert_eq!(cmp("abc", Op::StrEq, "abc"), Value::Bool(true));
    assert_eq!(cmp("abc", Op::StrEq, "abd"), Value::Bool(false));
}

#[test]
fn str_ne_inverts_eq() {
    assert_eq!(cmp("abc", Op::StrNe, "abc"), Value::Bool(false));
    assert_eq!(cmp("abc", Op::StrNe, "xyz"), Value::Bool(true));
}

#[test]
fn str_lt_gt_le_ge_are_consistent() {
    assert_eq!(cmp("a", Op::StrLt, "b"), Value::Bool(true));
    assert_eq!(cmp("b", Op::StrLt, "a"), Value::Bool(false));
    assert_eq!(cmp("b", Op::StrGt, "a"), Value::Bool(true));
    assert_eq!(cmp("a", Op::StrLe, "a"), Value::Bool(true));
    assert_eq!(cmp("a", Op::StrGe, "a"), Value::Bool(true));
}

// ── Logical & bitwise (basic completeness) ──────────────────────────────

#[test]
fn log_not_inverts_truthiness() {
    assert_eq!(run_emit(vec![Op::LoadInt(0), Op::LogNot]), Value::Bool(true));
    assert_eq!(
        run_emit(vec![Op::LoadInt(7), Op::LogNot]),
        Value::Bool(false)
    );
}

#[test]
fn log_and_full_evaluation() {
    assert_eq!(
        run_emit(vec![Op::LoadInt(1), Op::LoadInt(2), Op::LogAnd]),
        Value::Bool(true)
    );
    assert_eq!(
        run_emit(vec![Op::LoadInt(0), Op::LoadInt(2), Op::LogAnd]),
        Value::Bool(false)
    );
}

#[test]
fn log_or_full_evaluation() {
    assert_eq!(
        run_emit(vec![Op::LoadInt(0), Op::LoadInt(0), Op::LogOr]),
        Value::Bool(false)
    );
    assert_eq!(
        run_emit(vec![Op::LoadInt(0), Op::LoadInt(1), Op::LogOr]),
        Value::Bool(true)
    );
}

#[test]
fn bit_not_inverts_bits() {
    assert_eq!(run_emit(vec![Op::LoadInt(0), Op::BitNot]), Value::Int(-1));
    assert_eq!(run_emit(vec![Op::LoadInt(-1), Op::BitNot]), Value::Int(0));
}

#[test]
fn shl_and_shr() {
    assert_eq!(
        run_emit(vec![Op::LoadInt(1), Op::LoadInt(4), Op::Shl]),
        Value::Int(16)
    );
    assert_eq!(
        run_emit(vec![Op::LoadInt(64), Op::LoadInt(2), Op::Shr]),
        Value::Int(16)
    );
}

// ── Range & RangeStep ──────────────────────────────────────────────────

#[test]
fn range_inclusive_produces_array() {
    let v = run_emit(vec![Op::LoadInt(1), Op::LoadInt(5), Op::Range]);
    assert_eq!(
        v,
        Value::Array(vec![
            Value::Int(1),
            Value::Int(2),
            Value::Int(3),
            Value::Int(4),
            Value::Int(5)
        ])
    );
}

#[test]
fn range_empty_when_from_gt_to() {
    let v = run_emit(vec![Op::LoadInt(5), Op::LoadInt(1), Op::Range]);
    assert_eq!(v, Value::Array(vec![]));
}

#[test]
fn range_single_when_from_eq_to() {
    let v = run_emit(vec![Op::LoadInt(3), Op::LoadInt(3), Op::Range]);
    assert_eq!(v, Value::Array(vec![Value::Int(3)]));
}

#[test]
fn range_step_positive() {
    let v = run_emit(vec![
        Op::LoadInt(0),
        Op::LoadInt(10),
        Op::LoadInt(3),
        Op::RangeStep,
    ]);
    assert_eq!(
        v,
        Value::Array(vec![Value::Int(0), Value::Int(3), Value::Int(6), Value::Int(9)])
    );
}

#[test]
fn range_step_negative_descends() {
    let v = run_emit(vec![
        Op::LoadInt(10),
        Op::LoadInt(0),
        Op::LoadInt(-2),
        Op::RangeStep,
    ]);
    assert_eq!(
        v,
        Value::Array(vec![
            Value::Int(10),
            Value::Int(8),
            Value::Int(6),
            Value::Int(4),
            Value::Int(2),
            Value::Int(0)
        ])
    );
}

#[test]
fn range_step_zero_yields_empty() {
    let v = run_emit(vec![
        Op::LoadInt(0),
        Op::LoadInt(10),
        Op::LoadInt(0),
        Op::RangeStep,
    ]);
    assert_eq!(v, Value::Array(vec![]));
}

// ── MakeArray / MakeHash ───────────────────────────────────────────────

#[test]
fn make_array_collects_n_values() {
    let v = run_emit(vec![
        Op::LoadInt(1),
        Op::LoadInt(2),
        Op::LoadInt(3),
        Op::MakeArray(3),
    ]);
    assert_eq!(
        v,
        Value::Array(vec![Value::Int(1), Value::Int(2), Value::Int(3)])
    );
}

#[test]
fn make_array_zero_yields_empty() {
    let v = run_emit(vec![Op::MakeArray(0)]);
    assert_eq!(v, Value::Array(vec![]));
}

#[test]
fn make_hash_collects_n_pairs() {
    // MakeHash(n) drains n stack items; iterates as (key, val) pairs.
    let mut b = ChunkBuilder::new();
    let k = b.add_constant(Value::str("k"));
    b.emit(Op::LoadConst(k), 1);
    b.emit(Op::LoadInt(7), 1);
    b.emit(Op::MakeHash(2), 1);
    let v = run(b);
    if let Value::Hash(h) = v {
        assert_eq!(h.get("k"), Some(&Value::Int(7)));
        assert_eq!(h.len(), 1);
    } else {
        panic!("expected hash")
    }
}

// ── Status ops ──────────────────────────────────────────────────────────

#[test]
fn set_status_then_get_status() {
    let v = run_emit(vec![Op::LoadInt(42), Op::SetStatus, Op::GetStatus]);
    assert_eq!(v, Value::Status(42));
}

#[test]
fn get_status_initial_is_zero() {
    let v = run_emit(vec![Op::GetStatus]);
    assert_eq!(v, Value::Status(0));
}

// ── Conversion via to_str on stack ──────────────────────────────────────

#[test]
fn string_len_for_int_uses_to_str_then_len() {
    // StringLen pops then uses Value::len which falls through to to_str().len()
    // for Int. Int(12345).to_str() = "12345" -> len 5
    assert_eq!(
        run_emit(vec![Op::LoadInt(12345), Op::StringLen]),
        Value::Int(5)
    );
}

// ── Reset / re-run ──────────────────────────────────────────────────────

#[test]
fn vm_reset_clears_stack_and_runs_new_chunk() {
    let mut b1 = ChunkBuilder::new();
    b1.emit(Op::LoadInt(1), 1);
    b1.emit(Op::LoadInt(2), 1);
    b1.emit(Op::Add, 1);
    let mut vm = VM::new(b1.build());
    assert!(matches!(vm.run(), VMResult::Ok(Value::Int(3))));

    let mut b2 = ChunkBuilder::new();
    b2.emit(Op::LoadInt(10), 1);
    b2.emit(Op::LoadInt(20), 1);
    b2.emit(Op::Mul, 1);
    vm.reset(b2.build());
    assert!(matches!(vm.run(), VMResult::Ok(Value::Int(200))));
}

#[test]
fn push_pop_peek_on_vm() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::Nop, 1);
    let mut vm = VM::new(b.build());
    vm.push(Value::Int(42));
    assert_eq!(*vm.peek(), Value::Int(42));
    assert_eq!(vm.pop(), Value::Int(42));
}
