//! Coverage for collection ops: variables, slots, arrays (incl. slot-array),
//! hashes, MakeArray, Range/RangeStep, Concat, StringRepeat/StringLen.

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

fn s(v: Value) -> String {
    match v {
        Value::Str(s) => s.as_str().to_string(),
        other => panic!("expected Str, got {:?}", other),
    }
}

// ══════════════════════════════════════════════════════════════════════════
// Vars (GetVar/SetVar/DeclareVar)
// ══════════════════════════════════════════════════════════════════════════

#[test]
fn setvar_then_getvar_round_trips() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(42), 1);
    b.emit(Op::SetVar(0), 1);
    b.emit(Op::GetVar(0), 1);
    assert_eq!(i(run(b)), 42);
}

#[test]
fn getvar_unset_index_returns_undef() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::GetVar(99), 1);
    assert!(matches!(run(b), Value::Undef));
}

#[test]
fn declarevar_initializes_with_popped_value() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(7), 1);
    b.emit(Op::DeclareVar(3), 1);
    b.emit(Op::GetVar(3), 1);
    assert_eq!(i(run(b)), 7);
}

// ══════════════════════════════════════════════════════════════════════════
// Slots (GetSlot/SetSlot)
// ══════════════════════════════════════════════════════════════════════════

#[test]
fn setslot_then_getslot_round_trips() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(99), 1);
    b.emit(Op::SetSlot(0), 1);
    b.emit(Op::GetSlot(0), 1);
    assert_eq!(i(run(b)), 99);
}

#[test]
fn getslot_unset_index_returns_undef() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::GetSlot(50), 1);
    assert!(matches!(run(b), Value::Undef));
}

#[test]
fn multiple_slots_are_independent() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(11), 1);
    b.emit(Op::SetSlot(0), 1);
    b.emit(Op::LoadInt(22), 1);
    b.emit(Op::SetSlot(1), 1);
    b.emit(Op::GetSlot(0), 1);
    b.emit(Op::GetSlot(1), 1);
    b.emit(Op::Add, 1);
    assert_eq!(i(run(b)), 33);
}

// ══════════════════════════════════════════════════════════════════════════
// SlotArrayGet / SlotArraySet
// ══════════════════════════════════════════════════════════════════════════

#[test]
fn slotarrayset_then_slotarrayget_round_trips() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(0), 1);
    b.emit(Op::LoadInt(0), 1);
    b.emit(Op::LoadInt(0), 1);
    b.emit(Op::MakeArray(3), 1);
    b.emit(Op::SetSlot(0), 1);
    // arr[1] = 77
    b.emit(Op::LoadInt(77), 1);
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::SlotArraySet(0), 1);
    // load arr[1]
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::SlotArrayGet(0), 1);
    assert_eq!(i(run(b)), 77);
}

#[test]
fn slotarrayget_oob_returns_undef() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::MakeArray(1), 1);
    b.emit(Op::SetSlot(0), 1);
    b.emit(Op::LoadInt(50), 1);
    b.emit(Op::SlotArrayGet(0), 1);
    assert!(matches!(run(b), Value::Undef));
}

#[test]
fn slotarrayset_oob_grows_array() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(0), 1);
    b.emit(Op::MakeArray(1), 1);
    b.emit(Op::SetSlot(0), 1);
    // arr[5] = 99
    b.emit(Op::LoadInt(99), 1);
    b.emit(Op::LoadInt(5), 1);
    b.emit(Op::SlotArraySet(0), 1);
    b.emit(Op::LoadInt(5), 1);
    b.emit(Op::SlotArrayGet(0), 1);
    assert_eq!(i(run(b)), 99);
}

#[test]
fn slotarrayget_on_non_array_returns_undef() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(42), 1);
    b.emit(Op::SetSlot(0), 1);
    b.emit(Op::LoadInt(0), 1);
    b.emit(Op::SlotArrayGet(0), 1);
    assert!(matches!(run(b), Value::Undef));
}

// ══════════════════════════════════════════════════════════════════════════
// Global arrays (DeclareArray/ArrayPush/Pop/Len/Get/Set)
// ══════════════════════════════════════════════════════════════════════════

#[test]
fn arraypush_then_arraylen() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::DeclareArray(0), 1);
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::ArrayPush(0), 1);
    b.emit(Op::LoadInt(2), 1);
    b.emit(Op::ArrayPush(0), 1);
    b.emit(Op::LoadInt(3), 1);
    b.emit(Op::ArrayPush(0), 1);
    b.emit(Op::ArrayLen(0), 1);
    assert_eq!(i(run(b)), 3);
}

#[test]
fn arraypush_then_arraypop_returns_last() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::DeclareArray(0), 1);
    b.emit(Op::LoadInt(11), 1);
    b.emit(Op::ArrayPush(0), 1);
    b.emit(Op::LoadInt(22), 1);
    b.emit(Op::ArrayPush(0), 1);
    b.emit(Op::ArrayPop(0), 1);
    assert_eq!(i(run(b)), 22);
}

#[test]
fn arraypop_empty_returns_undef() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::DeclareArray(0), 1);
    b.emit(Op::ArrayPop(0), 1);
    assert!(matches!(run(b), Value::Undef));
}

#[test]
fn arraypop_undefined_slot_returns_undef() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::ArrayPop(99), 1);
    assert!(matches!(run(b), Value::Undef));
}

#[test]
fn arraylen_unset_var_returns_zero() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::ArrayLen(99), 1);
    assert_eq!(i(run(b)), 0);
}

#[test]
fn arrayset_at_index_then_arrayget_returns_value() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::DeclareArray(0), 1);
    b.emit(Op::LoadInt(55), 1);
    b.emit(Op::LoadInt(0), 1);
    b.emit(Op::ArraySet(0), 1);
    b.emit(Op::LoadInt(0), 1);
    b.emit(Op::ArrayGet(0), 1);
    assert_eq!(i(run(b)), 55);
}

#[test]
fn arrayget_oob_returns_undef() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::DeclareArray(0), 1);
    b.emit(Op::LoadInt(7), 1);
    b.emit(Op::ArrayGet(0), 1);
    assert!(matches!(run(b), Value::Undef));
}

#[test]
fn arrayset_oob_grows_array_with_undef_holes() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::DeclareArray(0), 1);
    // arr[3] = 9
    b.emit(Op::LoadInt(9), 1);
    b.emit(Op::LoadInt(3), 1);
    b.emit(Op::ArraySet(0), 1);
    b.emit(Op::ArrayLen(0), 1);
    assert_eq!(i(run(b)), 4);
}

// ══════════════════════════════════════════════════════════════════════════
// Hashes
// ══════════════════════════════════════════════════════════════════════════

#[test]
fn hashset_then_hashget_round_trips() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::DeclareHash(0), 1);
    // h["key"] = 100
    b.emit(Op::LoadInt(100), 1);
    let k = b.add_constant(Value::str("key"));
    b.emit(Op::LoadConst(k), 1);
    b.emit(Op::HashSet(0), 1);
    // load h["key"]
    let k2 = b.add_constant(Value::str("key"));
    b.emit(Op::LoadConst(k2), 1);
    b.emit(Op::HashGet(0), 1);
    assert_eq!(i(run(b)), 100);
}

#[test]
fn hashget_missing_key_returns_undef() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::DeclareHash(0), 1);
    let k = b.add_constant(Value::str("nope"));
    b.emit(Op::LoadConst(k), 1);
    b.emit(Op::HashGet(0), 1);
    assert!(matches!(run(b), Value::Undef));
}

#[test]
fn hashexists_returns_correct_bool() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::DeclareHash(0), 1);
    b.emit(Op::LoadInt(1), 1);
    let k = b.add_constant(Value::str("k"));
    b.emit(Op::LoadConst(k), 1);
    b.emit(Op::HashSet(0), 1);
    let k2 = b.add_constant(Value::str("k"));
    b.emit(Op::LoadConst(k2), 1);
    b.emit(Op::HashExists(0), 1);
    assert!(matches!(run(b), Value::Bool(true)));
}

#[test]
fn hashexists_unknown_key_returns_false() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::DeclareHash(0), 1);
    let k = b.add_constant(Value::str("nope"));
    b.emit(Op::LoadConst(k), 1);
    b.emit(Op::HashExists(0), 1);
    assert!(matches!(run(b), Value::Bool(false)));
}

#[test]
fn hashdelete_removes_and_returns_value_then_exists_false() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::DeclareHash(0), 1);
    b.emit(Op::LoadInt(50), 1);
    let k = b.add_constant(Value::str("x"));
    b.emit(Op::LoadConst(k), 1);
    b.emit(Op::HashSet(0), 1);
    let k2 = b.add_constant(Value::str("x"));
    b.emit(Op::LoadConst(k2), 1);
    b.emit(Op::HashDelete(0), 1);
    // After delete, exists should be false.
    b.emit(Op::Pop, 1);
    let k3 = b.add_constant(Value::str("x"));
    b.emit(Op::LoadConst(k3), 1);
    b.emit(Op::HashExists(0), 1);
    assert!(matches!(run(b), Value::Bool(false)));
}

#[test]
fn hashkeys_returns_array_of_string_keys() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::DeclareHash(0), 1);
    for (i, k) in ["a", "b", "c"].iter().enumerate() {
        b.emit(Op::LoadInt(i as i64), 1);
        let c = b.add_constant(Value::str(*k));
        b.emit(Op::LoadConst(c), 1);
        b.emit(Op::HashSet(0), 1);
    }
    b.emit(Op::HashKeys(0), 1);
    let v = run(b);
    match v {
        Value::Array(items) => {
            assert_eq!(items.len(), 3);
            let mut got: Vec<String> = items.into_iter().map(|v| v.to_str()).collect();
            got.sort();
            assert_eq!(got, vec!["a", "b", "c"]);
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn hashset_with_non_string_key_coerces_via_to_str() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::DeclareHash(0), 1);
    // h[42] = "value"
    let v = b.add_constant(Value::str("value"));
    b.emit(Op::LoadConst(v), 1);
    b.emit(Op::LoadInt(42), 1);
    b.emit(Op::HashSet(0), 1);
    // lookup with "42"
    let k = b.add_constant(Value::str("42"));
    b.emit(Op::LoadConst(k), 1);
    b.emit(Op::HashGet(0), 1);
    assert_eq!(s(run(b)), "value");
}

// ══════════════════════════════════════════════════════════════════════════
// Range / RangeStep
// ══════════════════════════════════════════════════════════════════════════

#[test]
fn range_inclusive_ascending() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::LoadInt(5), 1);
    b.emit(Op::Range, 1);
    let v = run(b);
    match v {
        Value::Array(arr) => {
            let vals: Vec<i64> = arr.into_iter().map(|v| v.to_int()).collect();
            assert_eq!(vals, vec![1, 2, 3, 4, 5]);
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn range_from_equals_to_yields_single_element() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(7), 1);
    b.emit(Op::LoadInt(7), 1);
    b.emit(Op::Range, 1);
    let v = run(b);
    assert!(matches!(&v, Value::Array(arr) if arr.len() == 1 && arr[0] == Value::Int(7)));
}

#[test]
fn range_from_greater_than_to_yields_empty() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(5), 1);
    b.emit(Op::LoadInt(2), 1);
    b.emit(Op::Range, 1);
    let v = run(b);
    assert!(matches!(&v, Value::Array(a) if a.is_empty()));
}

#[test]
fn rangestep_positive_step() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(0), 1);
    b.emit(Op::LoadInt(10), 1);
    b.emit(Op::LoadInt(2), 1);
    b.emit(Op::RangeStep, 1);
    let v = run(b);
    match v {
        Value::Array(arr) => {
            let vals: Vec<i64> = arr.into_iter().map(|v| v.to_int()).collect();
            assert_eq!(vals, vec![0, 2, 4, 6, 8, 10]);
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn rangestep_negative_step_descends() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(10), 1);
    b.emit(Op::LoadInt(0), 1);
    b.emit(Op::LoadInt(-3), 1);
    b.emit(Op::RangeStep, 1);
    let v = run(b);
    match v {
        Value::Array(arr) => {
            let vals: Vec<i64> = arr.into_iter().map(|v| v.to_int()).collect();
            assert_eq!(vals, vec![10, 7, 4, 1]);
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn rangestep_zero_step_yields_empty_array_no_panic() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(0), 1);
    b.emit(Op::LoadInt(10), 1);
    b.emit(Op::LoadInt(0), 1);
    b.emit(Op::RangeStep, 1);
    let v = run(b);
    assert!(matches!(&v, Value::Array(a) if a.is_empty()));
}

#[test]
fn rangestep_step_overshoots_endpoint() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(0), 1);
    b.emit(Op::LoadInt(10), 1);
    b.emit(Op::LoadInt(7), 1);
    b.emit(Op::RangeStep, 1);
    let v = run(b);
    match v {
        Value::Array(arr) => {
            let vals: Vec<i64> = arr.into_iter().map(|v| v.to_int()).collect();
            assert_eq!(vals, vec![0, 7]);
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

// ══════════════════════════════════════════════════════════════════════════
// Concat
// ══════════════════════════════════════════════════════════════════════════

#[test]
fn concat_two_strings() {
    let mut b = ChunkBuilder::new();
    let a = b.add_constant(Value::str("hello, "));
    let c = b.add_constant(Value::str("world"));
    b.emit(Op::LoadConst(a), 1);
    b.emit(Op::LoadConst(c), 1);
    b.emit(Op::Concat, 1);
    assert_eq!(s(run(b)), "hello, world");
}

#[test]
fn concat_int_and_string_via_as_str_cow() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(42), 1);
    let suf = b.add_constant(Value::str(" bottles"));
    b.emit(Op::LoadConst(suf), 1);
    b.emit(Op::Concat, 1);
    assert_eq!(s(run(b)), "42 bottles");
}

#[test]
fn concat_with_undef_yields_empty_for_undef_side() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadUndef, 1);
    let x = b.add_constant(Value::str("x"));
    b.emit(Op::LoadConst(x), 1);
    b.emit(Op::Concat, 1);
    assert_eq!(s(run(b)), "x");
}

#[test]
fn concat_array_renders_space_joined() {
    let mut b = ChunkBuilder::new();
    let a = b.add_constant(Value::Array(vec![Value::Int(1), Value::Int(2)]));
    b.emit(Op::LoadConst(a), 1);
    let suf = b.add_constant(Value::str("!"));
    b.emit(Op::LoadConst(suf), 1);
    b.emit(Op::Concat, 1);
    assert_eq!(s(run(b)), "1 2!");
}

// ══════════════════════════════════════════════════════════════════════════
// StringRepeat / StringLen
// ══════════════════════════════════════════════════════════════════════════

#[test]
fn stringrepeat_basic() {
    let mut b = ChunkBuilder::new();
    let a = b.add_constant(Value::str("ab"));
    b.emit(Op::LoadConst(a), 1);
    b.emit(Op::LoadInt(3), 1);
    b.emit(Op::StringRepeat, 1);
    assert_eq!(s(run(b)), "ababab");
}

#[test]
fn stringrepeat_zero_yields_empty() {
    let mut b = ChunkBuilder::new();
    let a = b.add_constant(Value::str("x"));
    b.emit(Op::LoadConst(a), 1);
    b.emit(Op::LoadInt(0), 1);
    b.emit(Op::StringRepeat, 1);
    assert_eq!(s(run(b)), "");
}

#[test]
fn stringlen_returns_byte_length() {
    let mut b = ChunkBuilder::new();
    let a = b.add_constant(Value::str("hello"));
    b.emit(Op::LoadConst(a), 1);
    b.emit(Op::StringLen, 1);
    assert_eq!(i(run(b)), 5);
}

#[test]
fn stringlen_of_empty_is_zero() {
    let mut b = ChunkBuilder::new();
    let a = b.add_constant(Value::str(""));
    b.emit(Op::LoadConst(a), 1);
    b.emit(Op::StringLen, 1);
    assert_eq!(i(run(b)), 0);
}

#[test]
fn stringlen_multibyte_returns_byte_count_not_char_count() {
    let mut b = ChunkBuilder::new();
    let s_val = "héllo";
    let a = b.add_constant(Value::str(s_val));
    b.emit(Op::LoadConst(a), 1);
    b.emit(Op::StringLen, 1);
    assert_eq!(i(run(b)) as usize, s_val.len()); // 6 bytes
}
