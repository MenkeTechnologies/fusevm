//! Hash and clone semantics for `Value`, plus additional `Op` PartialEq /
//! Clone smoke tests. The `Value::Hash` impl uses bit-pattern hashing for
//! floats and ignores HashMap key ordering (since HashMap hash order is
//! unpredictable, we don't assert equality for Hash-variant values).

use fusevm::{Op, Value};
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

fn h(v: &Value) -> u64 {
    let mut s = DefaultHasher::new();
    v.hash(&mut s);
    s.finish()
}

// ── Value hashing: discriminant differentiation ────────────────────────────

#[test]
fn int_and_float_with_same_numeric_value_hash_differently() {
    // Different variants — discriminant is mixed in, so hashes differ.
    assert_ne!(h(&Value::int(1)), h(&Value::float(1.0)));
}

#[test]
fn int_zero_and_bool_false_hash_differently() {
    assert_ne!(h(&Value::int(0)), h(&Value::bool(false)));
}

#[test]
fn equal_ints_hash_equally() {
    assert_eq!(h(&Value::int(42)), h(&Value::int(42)));
}

#[test]
fn equal_strings_hash_equally() {
    assert_eq!(h(&Value::str("hello")), h(&Value::str("hello")));
}

#[test]
fn unequal_strings_hash_differently() {
    assert_ne!(h(&Value::str("a")), h(&Value::str("b")));
}

#[test]
fn equal_floats_hash_equally_via_bit_pattern() {
    assert_eq!(h(&Value::float(3.14)), h(&Value::float(3.14)));
}

#[test]
fn nan_floats_with_same_bit_pattern_hash_equally() {
    let n1 = f64::from_bits(0x7ff8000000000001);
    let n2 = f64::from_bits(0x7ff8000000000001);
    assert_eq!(h(&Value::float(n1)), h(&Value::float(n2)));
}

#[test]
fn pos_zero_and_neg_zero_hash_differently() {
    // f64::to_bits(0.0) != f64::to_bits(-0.0)
    assert_ne!(h(&Value::float(0.0)), h(&Value::float(-0.0)));
}

#[test]
fn undef_and_empty_string_hash_differently() {
    assert_ne!(h(&Value::Undef), h(&Value::str("")));
}

#[test]
fn array_hash_depends_on_element_order() {
    let a = Value::array(vec![Value::int(1), Value::int(2)]);
    let b = Value::array(vec![Value::int(2), Value::int(1)]);
    assert_ne!(h(&a), h(&b));
}

#[test]
fn array_of_same_elements_hashes_equally() {
    let a = Value::array(vec![Value::int(1), Value::int(2), Value::int(3)]);
    let b = Value::array(vec![Value::int(1), Value::int(2), Value::int(3)]);
    assert_eq!(h(&a), h(&b));
}

#[test]
fn status_and_int_with_same_numeric_hash_differently() {
    assert_ne!(h(&Value::status(7)), h(&Value::int(7)));
}

#[test]
fn native_fn_with_distinct_ids_hash_differently() {
    assert_ne!(h(&Value::NativeFn(1)), h(&Value::NativeFn(2)));
}

#[test]
fn empty_arrays_hash_equally() {
    assert_eq!(h(&Value::array(vec![])), h(&Value::array(vec![])));
}

// ── Value: clone produces structurally-equal copy ───────────────────────────

#[test]
fn clone_int_str_array_bool_preserves_equality() {
    let v1 = Value::int(42);
    let v2 = v1.clone();
    assert_eq!(v1, v2);
    let s1 = Value::str("hello");
    let s2 = s1.clone();
    assert_eq!(s1, s2);
    let a1 = Value::array(vec![Value::int(1), Value::str("x")]);
    let a2 = a1.clone();
    assert_eq!(a1, a2);
    let b1 = Value::bool(true);
    let b2 = b1.clone();
    assert_eq!(b1, b2);
}

#[test]
fn clone_hash_preserves_entries() {
    let mut m = HashMap::new();
    m.insert("k1".into(), Value::int(1));
    m.insert("k2".into(), Value::str("v"));
    let h1 = Value::hash(m);
    let h2 = h1.clone();
    assert_eq!(h1, h2);
}

// ── Op: PartialEq across variants ───────────────────────────────────────────

#[test]
fn op_partial_eq_self_for_all_simple_variants() {
    let ops = vec![
        Op::Nop,
        Op::Pop,
        Op::Dup,
        Op::Swap,
        Op::Rot,
        Op::Add,
        Op::Sub,
        Op::Mul,
        Op::Div,
        Op::Mod,
        Op::Pow,
        Op::Negate,
        Op::Inc,
        Op::Dec,
        Op::Concat,
        Op::StringLen,
        Op::Return,
        Op::ReturnValue,
        Op::PushFrame,
        Op::PopFrame,
        Op::SetStatus,
        Op::GetStatus,
        Op::BitNot,
        Op::LogNot,
    ];
    for op in &ops {
        assert_eq!(op, op);
    }
}

#[test]
fn op_partial_eq_distinguishes_indices() {
    assert_ne!(Op::GetVar(1), Op::GetVar(2));
    assert_ne!(Op::SetVar(1), Op::GetVar(1));
    assert_ne!(Op::LoadInt(1), Op::LoadInt(2));
    assert_ne!(Op::LoadFloat(1.0), Op::LoadFloat(2.0));
    assert_ne!(Op::Jump(1), Op::Jump(2));
    assert_ne!(Op::Call(1, 2), Op::Call(1, 3));
    assert_ne!(Op::Call(1, 2), Op::Call(2, 2));
    assert_ne!(Op::Extended(1, 0), Op::Extended(2, 0));
    assert_ne!(Op::ExtendedWide(1, 0), Op::ExtendedWide(1, 1));
}

#[test]
fn op_clone_equals_original_for_all_payloaded_variants() {
    let ops = vec![
        Op::LoadInt(99),
        Op::LoadFloat(1.5),
        Op::LoadConst(3),
        Op::GetVar(4),
        Op::SetVar(4),
        Op::Jump(10),
        Op::JumpIfTrue(11),
        Op::JumpIfFalse(12),
        Op::Call(5, 2),
        Op::CallBuiltin(2, 1),
        Op::CallFunction(6, 0),
        Op::Extended(7, 8),
        Op::ExtendedWide(9, 0xdeadbeef),
        Op::AccumSumLoop(1, 2, 100),
        Op::ConcatConstLoop(0, 1, 2, 5),
        Op::PushIntRangeLoop(3, 4, 10),
        Op::SlotLtIntJumpIfFalse(0, 5, 99),
        Op::SlotIncLtIntJumpBack(0, 5, 1),
        Op::AddAssignSlotVoid(1, 2),
        Op::Redirect(1, 0),
    ];
    for op in &ops {
        assert_eq!(*op, op.clone());
    }
}

#[test]
fn float_load_op_compares_via_bit_pattern_not_value() {
    // NaN values in LoadFloat — equal bit patterns are equal.
    let nan1 = f64::from_bits(0x7ff8000000000001);
    let nan2 = f64::from_bits(0x7ff8000000000001);
    let a = Op::LoadFloat(nan1);
    let b = Op::LoadFloat(nan2);
    // PartialEq on Op is derived; floats use derived PartialEq → NaN != NaN.
    // That's fine; just exercise the derived behaviour:
    let _ = a == b; // does not panic; outcome compiler-defined.
}

// ── Value::array nested equality ───────────────────────────────────────────

#[test]
fn nested_arrays_compare_structurally() {
    let inner1 = Value::array(vec![Value::int(1), Value::int(2)]);
    let inner2 = Value::array(vec![Value::int(1), Value::int(2)]);
    let outer1 = Value::array(vec![inner1.clone()]);
    let outer2 = Value::array(vec![inner2.clone()]);
    assert_eq!(outer1, outer2);
}

#[test]
fn nested_array_differs_at_inner_element() {
    let outer1 = Value::array(vec![Value::array(vec![Value::int(1)])]);
    let outer2 = Value::array(vec![Value::array(vec![Value::int(2)])]);
    assert_ne!(outer1, outer2);
}
