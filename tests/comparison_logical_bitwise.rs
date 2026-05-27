//! Exhaustive comparison + logical/bitwise op coverage.
//! - NumEq/Ne/Lt/Gt/Le/Ge across int+int, int+float, str+int (coerced)
//! - Spaceship: -1/0/1 incl. mixed-type fallback
//! - StrEq/Ne/Lt/Gt/Le/Ge and StrCmp on as_str_cow representations
//! - LogAnd/LogOr/LogNot evaluate both sides (not short-circuit)
//! - BitAnd/Or/Xor/Not, Shl/Shr — coercion via to_int, large-shift mask 63

use fusevm::{ChunkBuilder, Op, VMResult, Value, VM};

fn run(b: ChunkBuilder) -> Value {
    match VM::new(b.build()).run() {
        VMResult::Ok(v) => v,
        VMResult::Halted => Value::Undef,
        VMResult::Error(e) => panic!("unexpected VM error: {e}"),
    }
}

fn two_int(a: i64, b: i64, op: Op) -> Value {
    let mut bld = ChunkBuilder::new();
    let ca = bld.add_constant(Value::Int(a));
    let cb = bld.add_constant(Value::Int(b));
    bld.emit(Op::LoadConst(ca), 1);
    bld.emit(Op::LoadConst(cb), 1);
    bld.emit(op, 1);
    run(bld)
}

fn two_val(a: Value, b: Value, op: Op) -> Value {
    let mut bld = ChunkBuilder::new();
    let ca = bld.add_constant(a);
    let cb = bld.add_constant(b);
    bld.emit(Op::LoadConst(ca), 1);
    bld.emit(Op::LoadConst(cb), 1);
    bld.emit(op, 1);
    run(bld)
}

fn one_val(a: Value, op: Op) -> Value {
    let mut bld = ChunkBuilder::new();
    let ca = bld.add_constant(a);
    bld.emit(Op::LoadConst(ca), 1);
    bld.emit(op, 1);
    run(bld)
}

fn b(v: Value) -> bool {
    match v {
        Value::Bool(b) => b,
        Value::Undef => false,
        other => panic!("expected Bool, got {:?}", other),
    }
}

fn i(v: Value) -> i64 {
    match v {
        Value::Int(n) => n,
        other => panic!("expected Int, got {:?}", other),
    }
}

// ══════════════════════════════════════════════════════════════════════════
// Numeric comparison
// ══════════════════════════════════════════════════════════════════════════

#[test]
fn numeq_int_int() {
    assert!(b(two_int(3, 3, Op::NumEq)));
    assert!(!b(two_int(3, 4, Op::NumEq)));
}

#[test]
fn numne_int_int() {
    assert!(!b(two_int(3, 3, Op::NumNe)));
    assert!(b(two_int(3, 4, Op::NumNe)));
}

#[test]
fn numlt_lt_le_gt_ge_int_int() {
    assert!(b(two_int(1, 2, Op::NumLt)));
    assert!(!b(two_int(2, 2, Op::NumLt)));
    assert!(!b(two_int(3, 2, Op::NumLt)));

    assert!(b(two_int(1, 2, Op::NumLe)));
    assert!(b(two_int(2, 2, Op::NumLe)));
    assert!(!b(two_int(3, 2, Op::NumLe)));

    assert!(!b(two_int(1, 2, Op::NumGt)));
    assert!(!b(two_int(2, 2, Op::NumGt)));
    assert!(b(two_int(3, 2, Op::NumGt)));

    assert!(!b(two_int(1, 2, Op::NumGe)));
    assert!(b(two_int(2, 2, Op::NumGe)));
    assert!(b(two_int(3, 2, Op::NumGe)));
}

#[test]
fn numcmp_with_float_compares_numerically() {
    assert!(b(two_val(Value::Int(2), Value::Float(2.0), Op::NumEq)));
    assert!(b(two_val(Value::Float(2.5), Value::Int(3), Op::NumLt)));
    assert!(b(two_val(Value::Float(3.5), Value::Int(3), Op::NumGt)));
}

#[test]
fn numcmp_with_numeric_string_coerces() {
    // "5" coerces to 5; numeric compare with Int.
    assert!(b(two_val(Value::str("5"), Value::Int(5), Op::NumEq)));
    assert!(b(two_val(Value::str("4"), Value::Int(5), Op::NumLt)));
}

#[test]
fn numcmp_non_numeric_string_coerces_to_zero() {
    assert!(b(two_val(Value::str("abc"), Value::Int(0), Op::NumEq)));
}

// ══════════════════════════════════════════════════════════════════════════
// Spaceship
// ══════════════════════════════════════════════════════════════════════════

#[test]
fn spaceship_int_int_returns_negative_zero_positive() {
    assert_eq!(i(two_int(1, 2, Op::Spaceship)), -1);
    assert_eq!(i(two_int(2, 2, Op::Spaceship)), 0);
    assert_eq!(i(two_int(3, 2, Op::Spaceship)), 1);
}

#[test]
fn spaceship_mixed_types_uses_float_compare() {
    assert_eq!(
        i(two_val(Value::Float(1.5), Value::Int(2), Op::Spaceship)),
        -1
    );
    assert_eq!(
        i(two_val(Value::Float(2.0), Value::Int(2), Op::Spaceship)),
        0
    );
    assert_eq!(
        i(two_val(Value::Float(2.5), Value::Int(2), Op::Spaceship)),
        1
    );
}

#[test]
fn spaceship_extreme_ints() {
    assert_eq!(i(two_int(i64::MIN, i64::MAX, Op::Spaceship)), -1);
    assert_eq!(i(two_int(i64::MAX, i64::MIN, Op::Spaceship)), 1);
}

// ══════════════════════════════════════════════════════════════════════════
// String comparison via as_str_cow
// ══════════════════════════════════════════════════════════════════════════

#[test]
fn streq_uses_string_repr_so_int_vs_str_can_match() {
    // Int(42).as_str_cow() == "42"; Str("42") == "42".
    assert!(b(two_val(Value::Int(42), Value::str("42"), Op::StrEq)));
    assert!(b(two_val(Value::str("42"), Value::Int(42), Op::StrEq)));
}

#[test]
fn strne_distinguishes_different_reprs() {
    assert!(b(two_val(Value::str("a"), Value::str("b"), Op::StrNe)));
    assert!(!b(two_val(Value::str("a"), Value::str("a"), Op::StrNe)));
}

#[test]
fn strlt_le_gt_ge_lexicographic() {
    assert!(b(two_val(
        Value::str("apple"),
        Value::str("banana"),
        Op::StrLt
    )));
    assert!(b(two_val(
        Value::str("apple"),
        Value::str("apple"),
        Op::StrLe
    )));
    assert!(b(two_val(
        Value::str("banana"),
        Value::str("apple"),
        Op::StrGt
    )));
    assert!(b(two_val(
        Value::str("banana"),
        Value::str("banana"),
        Op::StrGe
    )));
    assert!(!b(two_val(
        Value::str("apple"),
        Value::str("apple"),
        Op::StrLt
    )));
}

#[test]
fn strcmp_returns_negative_zero_positive() {
    assert_eq!(i(two_val(Value::str("a"), Value::str("b"), Op::StrCmp)), -1);
    assert_eq!(i(two_val(Value::str("a"), Value::str("a"), Op::StrCmp)), 0);
    assert_eq!(i(two_val(Value::str("b"), Value::str("a"), Op::StrCmp)), 1);
}

#[test]
fn strcmp_uses_bool_repr_true_is_one_false_is_empty() {
    // Bool(true).as_str_cow() == "1", Bool(false) == ""
    assert_eq!(
        i(two_val(Value::Bool(true), Value::str("1"), Op::StrCmp)),
        0
    );
    assert_eq!(
        i(two_val(Value::Bool(false), Value::str(""), Op::StrCmp)),
        0
    );
}

#[test]
fn streq_undef_equals_empty_string() {
    // Undef.as_str_cow() == ""
    assert!(b(two_val(Value::Undef, Value::str(""), Op::StrEq)));
}

#[test]
fn streq_array_space_joined() {
    let arr = Value::Array(vec![Value::Int(1), Value::Int(2), Value::Int(3)]);
    assert!(b(two_val(arr, Value::str("1 2 3"), Op::StrEq)));
}

// ══════════════════════════════════════════════════════════════════════════
// Logical ops (non-short-circuit; both sides evaluated)
// ══════════════════════════════════════════════════════════════════════════

#[test]
fn logand_both_truthy_is_true() {
    assert!(b(two_int(1, 2, Op::LogAnd)));
    assert!(!b(two_int(0, 2, Op::LogAnd)));
    assert!(!b(two_int(1, 0, Op::LogAnd)));
    assert!(!b(two_int(0, 0, Op::LogAnd)));
}

#[test]
fn logor_at_least_one_truthy_is_true() {
    assert!(b(two_int(1, 2, Op::LogOr)));
    assert!(b(two_int(0, 2, Op::LogOr)));
    assert!(b(two_int(1, 0, Op::LogOr)));
    assert!(!b(two_int(0, 0, Op::LogOr)));
}

#[test]
fn lognot_inverts_truthiness() {
    assert!(b(one_val(Value::Int(0), Op::LogNot)));
    assert!(!b(one_val(Value::Int(1), Op::LogNot)));
    assert!(b(one_val(Value::Undef, Op::LogNot)));
    assert!(b(one_val(Value::str(""), Op::LogNot)));
    assert!(b(one_val(Value::str("0"), Op::LogNot)));
    assert!(!b(one_val(Value::str("nonempty"), Op::LogNot)));
}

#[test]
fn logical_with_strings_uses_shell_truthiness() {
    // Empty/"0" are falsy; everything else (incl. "false") is truthy.
    assert!(b(two_val(Value::str("false"), Value::str("x"), Op::LogAnd)));
    assert!(!b(two_val(Value::str("0"), Value::str("x"), Op::LogAnd)));
    assert!(!b(two_val(Value::str(""), Value::str("x"), Op::LogAnd)));
}

// ══════════════════════════════════════════════════════════════════════════
// Bitwise ops
// ══════════════════════════════════════════════════════════════════════════

#[test]
fn bitand_or_xor_basic() {
    assert_eq!(i(two_int(0b1100, 0b1010, Op::BitAnd)), 0b1000);
    assert_eq!(i(two_int(0b1100, 0b1010, Op::BitOr)), 0b1110);
    assert_eq!(i(two_int(0b1100, 0b1010, Op::BitXor)), 0b0110);
}

#[test]
fn bitnot_complement() {
    assert_eq!(i(one_val(Value::Int(0), Op::BitNot)), -1);
    assert_eq!(i(one_val(Value::Int(-1), Op::BitNot)), 0);
    assert_eq!(i(one_val(Value::Int(5), Op::BitNot)), !5i64);
}

#[test]
fn shl_shr_basic() {
    assert_eq!(i(two_int(1, 0, Op::Shl)), 1);
    assert_eq!(i(two_int(1, 1, Op::Shl)), 2);
    assert_eq!(i(two_int(1, 4, Op::Shl)), 16);
    assert_eq!(i(two_int(64, 2, Op::Shr)), 16);
    assert_eq!(i(two_int(0, 5, Op::Shr)), 0);
}

#[test]
fn shl_amount_is_masked_to_63() {
    // 64 & 63 == 0, so this is a no-op shift.
    assert_eq!(i(two_int(7, 64, Op::Shl)), 7);
}

#[test]
fn shr_amount_is_masked_to_63() {
    assert_eq!(i(two_int(7, 64, Op::Shr)), 7);
}

#[test]
fn bitops_coerce_via_to_int() {
    // String "12" coerces to 12; Float 5.7 coerces to 5.
    assert_eq!(
        i(two_val(Value::str("12"), Value::Int(10), Op::BitAnd)),
        12 & 10
    );
    assert_eq!(
        i(two_val(Value::Float(5.7), Value::Int(3), Op::BitAnd)),
        5 & 3
    );
    // Undef coerces to 0; bool true coerces to 1.
    assert_eq!(i(two_val(Value::Undef, Value::Int(15), Op::BitAnd)), 0);
    assert_eq!(i(two_val(Value::Bool(true), Value::Int(1), Op::BitOr)), 1);
}

// ══════════════════════════════════════════════════════════════════════════
// Stack ops: Dup, Swap, Pop, Rot
// ══════════════════════════════════════════════════════════════════════════

#[test]
fn dup_duplicates_top_so_add_doubles_it() {
    let mut bld = ChunkBuilder::new();
    bld.emit(Op::LoadInt(7), 1);
    bld.emit(Op::Dup, 1);
    bld.emit(Op::Add, 1);
    assert_eq!(i(run(bld)), 14);
}

#[test]
fn pop_discards_top_so_only_second_remains() {
    let mut bld = ChunkBuilder::new();
    bld.emit(Op::LoadInt(99), 1);
    bld.emit(Op::LoadInt(5), 1);
    bld.emit(Op::Pop, 1);
    assert_eq!(i(run(bld)), 99);
}

#[test]
fn swap_then_sub_uses_swapped_order() {
    // push 10, push 3, swap → stack [3, 10] (top); Sub: 3 - 10 = -7
    let mut bld = ChunkBuilder::new();
    bld.emit(Op::LoadInt(10), 1);
    bld.emit(Op::LoadInt(3), 1);
    bld.emit(Op::Swap, 1);
    bld.emit(Op::Sub, 1);
    assert_eq!(i(run(bld)), -7);
}

// ══════════════════════════════════════════════════════════════════════════
// Boolean load ops
// ══════════════════════════════════════════════════════════════════════════

#[test]
fn loadtrue_pushes_true() {
    let mut bld = ChunkBuilder::new();
    bld.emit(Op::LoadTrue, 1);
    assert!(matches!(run(bld), Value::Bool(true)));
}

#[test]
fn loadfalse_pushes_false() {
    let mut bld = ChunkBuilder::new();
    bld.emit(Op::LoadFalse, 1);
    assert!(matches!(run(bld), Value::Bool(false)));
}

#[test]
fn loadundef_pushes_undef() {
    let mut bld = ChunkBuilder::new();
    bld.emit(Op::LoadUndef, 1);
    assert!(matches!(run(bld), Value::Undef));
}
