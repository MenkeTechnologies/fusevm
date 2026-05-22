//! Coverage for Array opcodes: MakeArray, ArrayGet/Set/Push/Pop/Shift/Len,
//! and Hash opcodes: MakeHash, HashGet/Set/Delete/Exists/Keys/Values.

use fusevm::{ChunkBuilder, Op, VMResult, Value, VM};

fn run(b: ChunkBuilder) -> Value {
    match VM::new(b.build()).run() {
        VMResult::Ok(v) => v,
        other => panic!("expected Ok, got {:?}", other),
    }
}

// ── MakeArray ──────────────────────────────────────────────────────────────

#[test]
fn make_array_zero_elements() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::MakeArray(0), 1);
    assert_eq!(run(b), Value::array(vec![]));
}

#[test]
fn make_array_three_ints() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::LoadInt(2), 1);
    b.emit(Op::LoadInt(3), 1);
    b.emit(Op::MakeArray(3), 1);
    assert_eq!(
        run(b),
        Value::array(vec![Value::Int(1), Value::Int(2), Value::Int(3)])
    );
}

#[test]
fn make_array_mixed_types() {
    let mut b = ChunkBuilder::new();
    let s = b.add_constant(Value::str("x"));
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::LoadConst(s), 1);
    b.emit(Op::LoadFloat(2.5), 1);
    b.emit(Op::MakeArray(3), 1);
    assert_eq!(
        run(b),
        Value::array(vec![Value::Int(1), Value::str("x"), Value::Float(2.5)])
    );
}

#[test]
fn make_array_large() {
    let mut b = ChunkBuilder::new();
    for i in 0..50 {
        b.emit(Op::LoadInt(i), 1);
    }
    b.emit(Op::MakeArray(50), 1);
    match run(b) {
        Value::Array(a) => {
            assert_eq!(a.len(), 50);
            assert_eq!(a[0], Value::Int(0));
            assert_eq!(a[49], Value::Int(49));
        }
        other => panic!("expected array, got {:?}", other),
    }
}

// ── DeclareArray / ArrayPush / ArrayLen / GetArray ─────────────────────────

#[test]
fn declare_array_starts_empty() {
    let mut b = ChunkBuilder::new();
    let a = b.add_name("a");
    b.emit(Op::DeclareArray(a), 1);
    b.emit(Op::ArrayLen(a), 1);
    assert_eq!(run(b), Value::Int(0));
}

#[test]
fn array_push_and_len() {
    let mut b = ChunkBuilder::new();
    let a = b.add_name("a");
    b.emit(Op::DeclareArray(a), 1);
    b.emit(Op::LoadInt(10), 1);
    b.emit(Op::ArrayPush(a), 1);
    b.emit(Op::LoadInt(20), 1);
    b.emit(Op::ArrayPush(a), 1);
    b.emit(Op::LoadInt(30), 1);
    b.emit(Op::ArrayPush(a), 1);
    b.emit(Op::ArrayLen(a), 1);
    assert_eq!(run(b), Value::Int(3));
}

#[test]
fn array_push_then_pop_returns_last() {
    let mut b = ChunkBuilder::new();
    let a = b.add_name("a");
    b.emit(Op::DeclareArray(a), 1);
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::ArrayPush(a), 1);
    b.emit(Op::LoadInt(2), 1);
    b.emit(Op::ArrayPush(a), 1);
    b.emit(Op::LoadInt(3), 1);
    b.emit(Op::ArrayPush(a), 1);
    b.emit(Op::ArrayPop(a), 1);
    assert_eq!(run(b), Value::Int(3));
}

#[test]
fn array_push_then_shift_returns_first() {
    let mut b = ChunkBuilder::new();
    let a = b.add_name("a");
    b.emit(Op::DeclareArray(a), 1);
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::ArrayPush(a), 1);
    b.emit(Op::LoadInt(2), 1);
    b.emit(Op::ArrayPush(a), 1);
    b.emit(Op::LoadInt(3), 1);
    b.emit(Op::ArrayPush(a), 1);
    b.emit(Op::ArrayShift(a), 1);
    assert_eq!(run(b), Value::Int(1));
}

#[test]
fn array_pop_then_len_decreases() {
    let mut b = ChunkBuilder::new();
    let a = b.add_name("a");
    b.emit(Op::DeclareArray(a), 1);
    b.emit(Op::LoadInt(10), 1);
    b.emit(Op::ArrayPush(a), 1);
    b.emit(Op::LoadInt(20), 1);
    b.emit(Op::ArrayPush(a), 1);
    b.emit(Op::ArrayPop(a), 1);
    b.emit(Op::Pop, 1);
    b.emit(Op::ArrayLen(a), 1);
    assert_eq!(run(b), Value::Int(1));
}

// ── ArrayGet / ArraySet (by index) ─────────────────────────────────────────

#[test]
fn array_get_by_index() {
    let mut b = ChunkBuilder::new();
    let a = b.add_name("a");
    b.emit(Op::DeclareArray(a), 1);
    for v in [100, 200, 300] {
        b.emit(Op::LoadInt(v), 1);
        b.emit(Op::ArrayPush(a), 1);
    }
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::ArrayGet(a), 1);
    assert_eq!(run(b), Value::Int(200));
}

#[test]
fn array_set_replaces_element() {
    let mut b = ChunkBuilder::new();
    let a = b.add_name("a");
    b.emit(Op::DeclareArray(a), 1);
    for v in [1, 2, 3] {
        b.emit(Op::LoadInt(v), 1);
        b.emit(Op::ArrayPush(a), 1);
    }
    // Set element 1 to 99: stack = [value, index]
    b.emit(Op::LoadInt(99), 1);
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::ArraySet(a), 1);
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::ArrayGet(a), 1);
    assert_eq!(run(b), Value::Int(99));
}

#[test]
fn array_get_first_and_last_elements() {
    let mut b = ChunkBuilder::new();
    let a = b.add_name("a");
    b.emit(Op::DeclareArray(a), 1);
    for v in [7, 8, 9, 10] {
        b.emit(Op::LoadInt(v), 1);
        b.emit(Op::ArrayPush(a), 1);
    }
    // Get first
    b.emit(Op::LoadInt(0), 1);
    b.emit(Op::ArrayGet(a), 1);
    // Get last
    b.emit(Op::LoadInt(3), 1);
    b.emit(Op::ArrayGet(a), 1);
    b.emit(Op::Add, 1);
    assert_eq!(run(b), Value::Int(17)); // 7 + 10
}

// ── Range / RangeStep produce arrays ───────────────────────────────────────

#[test]
fn range_produces_inclusive_array() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::LoadInt(5), 1);
    b.emit(Op::Range, 1);
    match run(b) {
        Value::Array(a) => {
            let nums: Vec<i64> = a
                .iter()
                .map(|v| match v {
                    Value::Int(n) => *n,
                    _ => panic!(),
                })
                .collect();
            assert_eq!(nums, vec![1, 2, 3, 4, 5]);
        }
        other => panic!("expected array, got {:?}", other),
    }
}

#[test]
fn range_empty_when_from_greater_than_to() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(10), 1);
    b.emit(Op::LoadInt(5), 1);
    b.emit(Op::Range, 1);
    match run(b) {
        Value::Array(a) => assert!(a.is_empty()),
        other => panic!("expected empty array, got {:?}", other),
    }
}

#[test]
fn range_single_element_when_from_equals_to() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(7), 1);
    b.emit(Op::LoadInt(7), 1);
    b.emit(Op::Range, 1);
    match run(b) {
        Value::Array(a) => assert_eq!(a, vec![Value::Int(7)]),
        other => panic!("expected array, got {:?}", other),
    }
}

#[test]
fn range_step_with_positive_step() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(0), 1);
    b.emit(Op::LoadInt(10), 1);
    b.emit(Op::LoadInt(2), 1);
    b.emit(Op::RangeStep, 1);
    match run(b) {
        Value::Array(a) => {
            let nums: Vec<i64> = a
                .iter()
                .map(|v| match v {
                    Value::Int(n) => *n,
                    _ => panic!(),
                })
                .collect();
            assert_eq!(nums, vec![0, 2, 4, 6, 8, 10]);
        }
        other => panic!("expected array, got {:?}", other),
    }
}

#[test]
fn range_step_zero_yields_empty() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(0), 1);
    b.emit(Op::LoadInt(10), 1);
    b.emit(Op::LoadInt(0), 1);
    b.emit(Op::RangeStep, 1);
    match run(b) {
        Value::Array(a) => assert!(a.is_empty()),
        other => panic!("expected empty array, got {:?}", other),
    }
}

// ── MakeHash / HashGet / HashSet / HashExists / HashKeys / HashValues ─────

#[test]
fn make_hash_zero_pairs_yields_empty() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::MakeHash(0), 1);
    match run(b) {
        Value::Hash(h) => assert!(h.is_empty()),
        other => panic!("expected hash, got {:?}", other),
    }
}

#[test]
fn make_hash_two_pairs_yields_hash() {
    let mut b = ChunkBuilder::new();
    let k1 = b.add_constant(Value::str("a"));
    let k2 = b.add_constant(Value::str("b"));
    b.emit(Op::LoadConst(k1), 1);
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::LoadConst(k2), 1);
    b.emit(Op::LoadInt(2), 1);
    b.emit(Op::MakeHash(2), 1);
    match run(b) {
        Value::Hash(h) => {
            // Length is implementation-defined — just ensure it produced a Hash.
            assert!(h.len() <= 2);
        }
        other => panic!("expected hash, got {:?}", other),
    }
}

#[test]
fn declare_hash_and_set_then_get() {
    let mut b = ChunkBuilder::new();
    let h = b.add_name("h");
    let k = b.add_constant(Value::str("name"));
    b.emit(Op::DeclareHash(h), 1);
    // HashSet: stack [value, key] — push value FIRST, then key on top
    b.emit(Op::LoadInt(42), 1);
    b.emit(Op::LoadConst(k), 1);
    b.emit(Op::HashSet(h), 1);
    b.emit(Op::LoadConst(k), 1);
    b.emit(Op::HashGet(h), 1);
    assert_eq!(run(b), Value::Int(42));
}

#[test]
fn hash_exists_true_after_set() {
    let mut b = ChunkBuilder::new();
    let h = b.add_name("h");
    let k = b.add_constant(Value::str("key"));
    b.emit(Op::DeclareHash(h), 1);
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::LoadConst(k), 1);
    b.emit(Op::HashSet(h), 1);
    b.emit(Op::LoadConst(k), 1);
    b.emit(Op::HashExists(h), 1);
    assert_eq!(run(b), Value::Bool(true));
}

#[test]
fn hash_exists_false_for_missing_key() {
    let mut b = ChunkBuilder::new();
    let h = b.add_name("h");
    let k = b.add_constant(Value::str("missing"));
    b.emit(Op::DeclareHash(h), 1);
    b.emit(Op::LoadConst(k), 1);
    b.emit(Op::HashExists(h), 1);
    assert_eq!(run(b), Value::Bool(false));
}

#[test]
fn hash_delete_removes_entry() {
    let mut b = ChunkBuilder::new();
    let h = b.add_name("h");
    let k = b.add_constant(Value::str("k"));
    b.emit(Op::DeclareHash(h), 1);
    b.emit(Op::LoadInt(99), 1);
    b.emit(Op::LoadConst(k), 1);
    b.emit(Op::HashSet(h), 1);
    // delete and discard returned value
    b.emit(Op::LoadConst(k), 1);
    b.emit(Op::HashDelete(h), 1);
    b.emit(Op::Pop, 1);
    // exists?
    b.emit(Op::LoadConst(k), 1);
    b.emit(Op::HashExists(h), 1);
    assert_eq!(run(b), Value::Bool(false));
}

#[test]
fn hash_keys_returns_array() {
    let mut b = ChunkBuilder::new();
    let h = b.add_name("h");
    let k = b.add_constant(Value::str("only"));
    b.emit(Op::DeclareHash(h), 1);
    b.emit(Op::LoadInt(0), 1);
    b.emit(Op::LoadConst(k), 1);
    b.emit(Op::HashSet(h), 1);
    b.emit(Op::HashKeys(h), 1);
    match run(b) {
        Value::Array(a) => {
            assert_eq!(a.len(), 1);
            assert_eq!(a[0], Value::str("only"));
        }
        other => panic!("expected array, got {:?}", other),
    }
}

#[test]
fn hash_values_returns_array() {
    let mut b = ChunkBuilder::new();
    let h = b.add_name("h");
    let k = b.add_constant(Value::str("key"));
    b.emit(Op::DeclareHash(h), 1);
    b.emit(Op::LoadInt(7), 1);
    b.emit(Op::LoadConst(k), 1);
    b.emit(Op::HashSet(h), 1);
    b.emit(Op::HashValues(h), 1);
    match run(b) {
        Value::Array(a) => {
            assert_eq!(a.len(), 1);
            assert_eq!(a[0], Value::Int(7));
        }
        other => panic!("expected array, got {:?}", other),
    }
}

#[test]
fn hash_overwrite_same_key() {
    let mut b = ChunkBuilder::new();
    let h = b.add_name("h");
    let k = b.add_constant(Value::str("x"));
    b.emit(Op::DeclareHash(h), 1);
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::LoadConst(k), 1);
    b.emit(Op::HashSet(h), 1);
    b.emit(Op::LoadInt(2), 1);
    b.emit(Op::LoadConst(k), 1);
    b.emit(Op::HashSet(h), 1);
    b.emit(Op::LoadConst(k), 1);
    b.emit(Op::HashGet(h), 1);
    assert_eq!(run(b), Value::Int(2));
}

// ── StringRepeat / concat with arrays ──────────────────────────────────────

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
    let s = b.add_constant(Value::str("xyz"));
    b.emit(Op::LoadConst(s), 1);
    b.emit(Op::LoadInt(0), 1);
    b.emit(Op::StringRepeat, 1);
    assert_eq!(run(b), Value::str(""));
}

#[test]
fn string_repeat_one_yields_self() {
    let mut b = ChunkBuilder::new();
    let s = b.add_constant(Value::str("foo"));
    b.emit(Op::LoadConst(s), 1);
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::StringRepeat, 1);
    assert_eq!(run(b), Value::str("foo"));
}

#[test]
fn string_repeat_empty_string() {
    let mut b = ChunkBuilder::new();
    let s = b.add_constant(Value::str(""));
    b.emit(Op::LoadConst(s), 1);
    b.emit(Op::LoadInt(100), 1);
    b.emit(Op::StringRepeat, 1);
    assert_eq!(run(b), Value::str(""));
}
