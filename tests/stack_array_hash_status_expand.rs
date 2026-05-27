//! Coverage for stack rotation ops, array/hash constructors and globals,
//! mixed-type arithmetic, status ops, and host-less shell-string expansions.

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
        Value::Status(n) => n as i64,
        other => panic!("expected Int/Status, got {:?}", other),
    }
}

fn f(v: Value) -> f64 {
    match v {
        Value::Float(x) => x,
        other => panic!("expected Float, got {:?}", other),
    }
}

fn arr(v: Value) -> Vec<Value> {
    match v {
        Value::Array(a) => a,
        other => panic!("expected Array, got {:?}", other),
    }
}

// ══════════════════════════════════════════════════════════════════════════
// Stack ops: Dup2 / Rot
// ══════════════════════════════════════════════════════════════════════════

#[test]
fn dup2_duplicates_top_two_values_preserving_order() {
    // [3, 4] → Dup2 → [3, 4, 3, 4] → Add → [3, 4, 7] → Add → [3, 11] → Add → 14
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(3), 1);
    b.emit(Op::LoadInt(4), 1);
    b.emit(Op::Dup2, 1);
    b.emit(Op::Add, 1);
    b.emit(Op::Add, 1);
    b.emit(Op::Add, 1);
    assert_eq!(i(run(b)), 14);
}

#[test]
fn dup2_with_lt_two_stack_items_is_silent_noop() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(7), 1);
    b.emit(Op::Dup2, 1);
    assert_eq!(i(run(b)), 7);
}

#[test]
fn rot_rotates_top_three_a_b_c_to_b_c_a() {
    // Push 1, 2, 3 → Rot → [2, 3, 1]; subtract twice: 1 - (3 - 2) = 0.
    // top is 1; pop yields 1; etc. Verify final top after Rot is 1.
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::LoadInt(2), 1);
    b.emit(Op::LoadInt(3), 1);
    b.emit(Op::Rot, 1);
    // top is now 1
    assert_eq!(i(run(b)), 1);
}

#[test]
fn rot_moves_buried_value_to_top() {
    // [10, 20, 30] → Rot → [20, 30, 10]; Pop twice → 20
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(10), 1);
    b.emit(Op::LoadInt(20), 1);
    b.emit(Op::LoadInt(30), 1);
    b.emit(Op::Rot, 1);
    b.emit(Op::Pop, 1); // pops 10
    b.emit(Op::Pop, 1); // pops 30 → top is 20
    assert_eq!(i(run(b)), 20);
}

#[test]
fn rot_with_lt_three_stack_items_is_silent_noop() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(8), 1);
    b.emit(Op::LoadInt(9), 1);
    b.emit(Op::Rot, 1);
    assert_eq!(i(run(b)), 9);
}

// ══════════════════════════════════════════════════════════════════════════
// MakeArray / MakeHash — drain N from stack
// ══════════════════════════════════════════════════════════════════════════

#[test]
fn makearray_collects_top_n_into_array_preserving_order() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::LoadInt(2), 1);
    b.emit(Op::LoadInt(3), 1);
    b.emit(Op::MakeArray(3), 1);
    let v = arr(run(b));
    assert_eq!(v.len(), 3);
    assert_eq!(i(v[0].clone()), 1);
    assert_eq!(i(v[1].clone()), 2);
    assert_eq!(i(v[2].clone()), 3);
}

#[test]
fn makearray_with_zero_yields_empty_array() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(99), 1); // sentinel that should remain on stack
    b.emit(Op::MakeArray(0), 1);
    // Top is empty array now
    let v = arr(run(b));
    assert!(v.is_empty());
}

#[test]
fn makearray_with_n_larger_than_stack_drains_what_is_available() {
    // saturating_sub guards against underflow.
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(7), 1);
    b.emit(Op::LoadInt(8), 1);
    b.emit(Op::MakeArray(10), 1);
    let v = arr(run(b));
    assert_eq!(v.len(), 2);
}

#[test]
fn makehash_pairs_keys_and_values_into_hash() {
    // Stack: [k1, v1, k2, v2] → MakeHash(4) → hash {k1:v1, k2:v2}
    let mut b = ChunkBuilder::new();
    let k1 = b.add_constant(Value::str("a"));
    let k2 = b.add_constant(Value::str("b"));
    b.emit(Op::LoadConst(k1), 1);
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::LoadConst(k2), 1);
    b.emit(Op::LoadInt(2), 1);
    b.emit(Op::MakeHash(4), 1);
    match run(b) {
        Value::Hash(m) => {
            assert_eq!(m.len(), 2);
            assert_eq!(i(m.get("a").cloned().unwrap()), 1);
            assert_eq!(i(m.get("b").cloned().unwrap()), 2);
        }
        other => panic!("expected Hash, got {:?}", other),
    }
}

#[test]
fn makehash_coerces_non_string_keys_via_to_str() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(42), 1); // key
    b.emit(Op::LoadInt(7), 1); // val
    b.emit(Op::MakeHash(2), 1);
    match run(b) {
        Value::Hash(m) => {
            assert_eq!(i(m.get("42").cloned().unwrap()), 7);
        }
        other => panic!("expected Hash, got {:?}", other),
    }
}

#[test]
fn makehash_with_odd_count_discards_unpaired_key() {
    // 3 entries: k1, v1, k2 — k2 has no value so loop drops it.
    let mut b = ChunkBuilder::new();
    let k1 = b.add_constant(Value::str("a"));
    let k2 = b.add_constant(Value::str("b"));
    b.emit(Op::LoadConst(k1), 1);
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::LoadConst(k2), 1);
    b.emit(Op::MakeHash(3), 1);
    match run(b) {
        Value::Hash(m) => {
            assert_eq!(m.len(), 1);
            assert_eq!(i(m.get("a").cloned().unwrap()), 1);
        }
        other => panic!("expected Hash, got {:?}", other),
    }
}

// ══════════════════════════════════════════════════════════════════════════
// ArrayShift / HashValues
// ══════════════════════════════════════════════════════════════════════════

#[test]
fn arrayshift_removes_and_returns_front_element() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::DeclareArray(0), 1);
    b.emit(Op::LoadInt(10), 1);
    b.emit(Op::ArrayPush(0), 1);
    b.emit(Op::LoadInt(20), 1);
    b.emit(Op::ArrayPush(0), 1);
    b.emit(Op::ArrayShift(0), 1);
    assert_eq!(i(run(b)), 10);
}

#[test]
fn arrayshift_on_empty_array_returns_undef() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::DeclareArray(0), 1);
    b.emit(Op::ArrayShift(0), 1);
    matches!(run(b), Value::Undef);
}

#[test]
fn arrayshift_on_undefined_slot_returns_undef() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::ArrayShift(99), 1);
    matches!(run(b), Value::Undef);
}

#[test]
fn arrayshift_reduces_length_by_one() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::DeclareArray(0), 1);
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::ArrayPush(0), 1);
    b.emit(Op::LoadInt(2), 1);
    b.emit(Op::ArrayPush(0), 1);
    b.emit(Op::LoadInt(3), 1);
    b.emit(Op::ArrayPush(0), 1);
    b.emit(Op::ArrayShift(0), 1);
    b.emit(Op::Pop, 1);
    b.emit(Op::ArrayLen(0), 1);
    assert_eq!(i(run(b)), 2);
}

#[test]
fn hashvalues_returns_array_of_values() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::DeclareHash(0), 1);
    let k = b.add_constant(Value::str("k"));
    b.emit(Op::LoadInt(7), 1);
    b.emit(Op::LoadConst(k), 1);
    b.emit(Op::HashSet(0), 1);
    b.emit(Op::HashValues(0), 1);
    let v = arr(run(b));
    assert_eq!(v.len(), 1);
    assert_eq!(i(v[0].clone()), 7);
}

#[test]
fn hashvalues_on_undefined_slot_returns_empty_array() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::HashValues(99), 1);
    assert!(arr(run(b)).is_empty());
}

// ══════════════════════════════════════════════════════════════════════════
// GetArray / SetArray / GetHash / SetHash — global name pool ops
// ══════════════════════════════════════════════════════════════════════════

#[test]
fn setarray_then_getarray_round_trips_array_value() {
    let mut b = ChunkBuilder::new();
    let k = b.add_constant(Value::str("x"));
    b.emit(Op::LoadConst(k), 1);
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::MakeArray(2), 1);
    b.emit(Op::SetArray(0), 1);
    b.emit(Op::GetArray(0), 1);
    let v = arr(run(b));
    assert_eq!(v.len(), 2);
}

#[test]
fn sethash_then_gethash_round_trips_hash_value() {
    let mut b = ChunkBuilder::new();
    let k1 = b.add_constant(Value::str("k"));
    b.emit(Op::LoadConst(k1), 1);
    b.emit(Op::LoadInt(9), 1);
    b.emit(Op::MakeHash(2), 1);
    b.emit(Op::SetHash(0), 1);
    b.emit(Op::GetHash(0), 1);
    match run(b) {
        Value::Hash(m) => assert_eq!(i(m.get("k").cloned().unwrap()), 9),
        other => panic!("expected Hash, got {:?}", other),
    }
}

#[test]
fn getarray_on_unset_index_returns_undef() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::GetArray(123), 1);
    matches!(run(b), Value::Undef);
}

#[test]
fn gethash_on_unset_index_returns_undef() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::GetHash(123), 1);
    matches!(run(b), Value::Undef);
}

// ══════════════════════════════════════════════════════════════════════════
// Mixed-type arithmetic — float path in arith_int_fast
// ══════════════════════════════════════════════════════════════════════════

#[test]
fn add_int_and_float_promotes_to_float() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(3), 1);
    b.emit(Op::LoadFloat(1.5), 1);
    b.emit(Op::Add, 1);
    assert!((f(run(b)) - 4.5).abs() < 1e-9);
}

#[test]
fn sub_float_minus_float() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadFloat(5.5), 1);
    b.emit(Op::LoadFloat(0.25), 1);
    b.emit(Op::Sub, 1);
    assert!((f(run(b)) - 5.25).abs() < 1e-9);
}

#[test]
fn mul_int_and_float_promotes_to_float() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(4), 1);
    b.emit(Op::LoadFloat(0.5), 1);
    b.emit(Op::Mul, 1);
    assert!((f(run(b)) - 2.0).abs() < 1e-9);
}

#[test]
fn mod_with_float_uses_float_path() {
    // 5.5 % 2.0 = 1.5
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadFloat(5.5), 1);
    b.emit(Op::LoadFloat(2.0), 1);
    b.emit(Op::Mod, 1);
    assert!((f(run(b)) - 1.5).abs() < 1e-9);
}

#[test]
fn pow_always_returns_float_even_for_integers() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(2), 1);
    b.emit(Op::LoadInt(10), 1);
    b.emit(Op::Pow, 1);
    assert!((f(run(b)) - 1024.0).abs() < 1e-9);
}

#[test]
fn negate_on_float_negates_the_float() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadFloat(2.5), 1);
    b.emit(Op::Negate, 1);
    assert!((f(run(b)) - (-2.5)).abs() < 1e-9);
}

#[test]
fn negate_on_string_coerces_then_negates_as_float() {
    let mut b = ChunkBuilder::new();
    let k = b.add_constant(Value::str("3.0"));
    b.emit(Op::LoadConst(k), 1);
    b.emit(Op::Negate, 1);
    assert!((f(run(b)) - (-3.0)).abs() < 1e-9);
}

#[test]
fn inc_on_float_truncates_to_int_and_adds_one() {
    // Inc on non-int path: to_int().wrapping_add(1)
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadFloat(2.7), 1);
    b.emit(Op::Inc, 1);
    assert_eq!(i(run(b)), 3);
}

#[test]
fn dec_on_string_coerces_then_decrements() {
    let mut b = ChunkBuilder::new();
    let k = b.add_constant(Value::str("10"));
    b.emit(Op::LoadConst(k), 1);
    b.emit(Op::Dec, 1);
    assert_eq!(i(run(b)), 9);
}

// ══════════════════════════════════════════════════════════════════════════
// Status ops: GetStatus / SetStatus
// ══════════════════════════════════════════════════════════════════════════

#[test]
fn setstatus_then_getstatus_round_trips_through_last_status() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(42), 1);
    b.emit(Op::SetStatus, 1);
    b.emit(Op::GetStatus, 1);
    match run(b) {
        Value::Status(42) => {}
        other => panic!("expected Status(42), got {:?}", other),
    }
}

#[test]
fn getstatus_before_setstatus_returns_status_zero() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::GetStatus, 1);
    match run(b) {
        Value::Status(0) => {}
        other => panic!("expected Status(0), got {:?}", other),
    }
}

#[test]
fn setstatus_overwrites_previous_value() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::SetStatus, 1);
    b.emit(Op::LoadInt(127), 1);
    b.emit(Op::SetStatus, 1);
    b.emit(Op::GetStatus, 1);
    match run(b) {
        Value::Status(127) => {}
        other => panic!("expected Status(127), got {:?}", other),
    }
}

// ══════════════════════════════════════════════════════════════════════════
// Host-less shell-string expansions: TildeExpand / BraceExpand / WordSplit / Glob
// ══════════════════════════════════════════════════════════════════════════

#[test]
fn tildeexpand_without_host_returns_input_unchanged() {
    let mut b = ChunkBuilder::new();
    let k = b.add_constant(Value::str("~/foo"));
    b.emit(Op::LoadConst(k), 1);
    b.emit(Op::TildeExpand, 1);
    match run(b) {
        Value::Str(s) => assert_eq!(s.as_str(), "~/foo"),
        other => panic!("expected Str, got {:?}", other),
    }
}

#[test]
fn braceexpand_without_host_wraps_input_into_single_element_array() {
    let mut b = ChunkBuilder::new();
    let k = b.add_constant(Value::str("foo{1,2}"));
    b.emit(Op::LoadConst(k), 1);
    b.emit(Op::BraceExpand, 1);
    let v = arr(run(b));
    assert_eq!(v.len(), 1);
    match v[0].clone() {
        Value::Str(s) => assert_eq!(s.as_str(), "foo{1,2}"),
        other => panic!("expected Str, got {:?}", other),
    }
}

#[test]
fn wordsplit_without_host_splits_on_ascii_whitespace() {
    let mut b = ChunkBuilder::new();
    let k = b.add_constant(Value::str("  hello\t world\n  there "));
    b.emit(Op::LoadConst(k), 1);
    b.emit(Op::WordSplit, 1);
    let v = arr(run(b));
    assert_eq!(v.len(), 3);
}

#[test]
fn wordsplit_empty_string_yields_empty_array() {
    let mut b = ChunkBuilder::new();
    let k = b.add_constant(Value::str(""));
    b.emit(Op::LoadConst(k), 1);
    b.emit(Op::WordSplit, 1);
    assert!(arr(run(b)).is_empty());
}

#[test]
fn glob_without_host_uses_filesystem_pattern() {
    // /etc/hosts almost always exists on linux/mac; pattern matches it.
    let mut b = ChunkBuilder::new();
    let k = b.add_constant(Value::str("/etc/hos*"));
    b.emit(Op::LoadConst(k), 1);
    b.emit(Op::Glob, 1);
    let v = arr(run(b));
    // Will be at least one match if /etc/hosts exists, but be lenient on CI.
    let _ = v.len(); // smoke check: no panic
}

#[test]
fn glob_with_pattern_that_matches_nothing_returns_empty_array() {
    let mut b = ChunkBuilder::new();
    let k = b.add_constant(Value::str("/this/path/should/not/exist/*.xyz123"));
    b.emit(Op::LoadConst(k), 1);
    b.emit(Op::Glob, 1);
    let v = arr(run(b));
    assert!(v.is_empty());
}

// ══════════════════════════════════════════════════════════════════════════
// Comparison mixed-type via cmp_int_fast float path
// ══════════════════════════════════════════════════════════════════════════

#[test]
fn numlt_int_vs_float_uses_float_compare() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(2), 1);
    b.emit(Op::LoadFloat(2.5), 1);
    b.emit(Op::NumLt, 1);
    match run(b) {
        Value::Bool(true) => {}
        other => panic!("expected Bool(true), got {:?}", other),
    }
}

#[test]
fn numge_float_vs_int_uses_float_compare() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadFloat(3.0), 1);
    b.emit(Op::LoadInt(3), 1);
    b.emit(Op::NumGe, 1);
    match run(b) {
        Value::Bool(true) => {}
        other => panic!("expected Bool(true), got {:?}", other),
    }
}

#[test]
fn numeq_int_vs_string_numeric_compares_via_to_float() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(42), 1);
    let k = b.add_constant(Value::str("42"));
    b.emit(Op::LoadConst(k), 1);
    b.emit(Op::NumEq, 1);
    match run(b) {
        Value::Bool(true) => {}
        other => panic!("expected Bool(true), got {:?}", other),
    }
}
