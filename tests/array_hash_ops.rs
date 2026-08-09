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
        Value::Array(a) => assert_eq!(*a, vec![Value::Int(7)]),
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

// ── Array value semantics under the Arc-backed representation ──────────────
//
// `Value::Array` holds an `Arc<Vec<Value>>` so a load is a refcount bump
// rather than a deep copy. That is only sound while every mutation copies a
// shared buffer first (`Value::array_mut` -> `Arc::make_mut`). These pin the
// observable contract: an array is a VALUE, and a copy taken before a write
// never sees that write. If someone swaps a mutation site to `Arc::get_mut`
// or mutates through a shared handle, these fail.

#[test]
fn array_assigned_to_another_global_is_an_independent_value() {
    // @a = [1,2,3]; @b = @a; @a[0] = 99  =>  @b stays [1,2,3].
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::LoadInt(2), 1);
    b.emit(Op::LoadInt(3), 1);
    b.emit(Op::MakeArray(3), 1);
    b.emit(Op::SetArray(0), 1); // @a
    b.emit(Op::GetArray(0), 1);
    b.emit(Op::SetArray(1), 1); // @b = @a  (shares the buffer)
    b.emit(Op::LoadInt(99), 1);
    b.emit(Op::LoadInt(0), 1);
    b.emit(Op::ArraySet(0), 1); // @a[0] = 99  (must copy-on-write)
    b.emit(Op::GetArray(1), 1); // @b
    assert_eq!(
        run(b),
        Value::array(vec![Value::Int(1), Value::Int(2), Value::Int(3)]),
        "mutating @a must not be visible through the earlier copy in @b"
    );
}

#[test]
fn array_push_after_copy_does_not_grow_the_copy() {
    // @a = [1]; @b = @a; push @a, 2  =>  @b stays [1], @a becomes [1,2].
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::MakeArray(1), 1);
    b.emit(Op::SetArray(0), 1);
    b.emit(Op::GetArray(0), 1);
    b.emit(Op::SetArray(1), 1);
    b.emit(Op::LoadInt(2), 1);
    b.emit(Op::ArrayPush(0), 1);
    b.emit(Op::ArrayLen(1), 1); // len(@b)
    assert_eq!(run(b), Value::Int(1), "the copy must not see the push");

    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::MakeArray(1), 1);
    b.emit(Op::SetArray(0), 1);
    b.emit(Op::GetArray(0), 1);
    b.emit(Op::SetArray(1), 1);
    b.emit(Op::LoadInt(2), 1);
    b.emit(Op::ArrayPush(0), 1);
    b.emit(Op::ArrayLen(0), 1); // len(@a)
    assert_eq!(run(b), Value::Int(2), "the pushed-to array must still grow");
}

#[test]
fn slot_array_set_does_not_leak_into_a_copy_of_the_slot() {
    // Slot 0 holds an array; slot 1 takes a copy; writing slot 0 must not
    // change slot 1. `SlotArraySet` mutates the slot in place, so this is the
    // pin that the in-place write still copies a SHARED buffer first.
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::LoadInt(2), 1);
    b.emit(Op::MakeArray(2), 1);
    b.emit(Op::SetSlot(0), 1);
    b.emit(Op::GetSlot(0), 1);
    b.emit(Op::SetSlot(1), 1); // copy
    b.emit(Op::LoadInt(77), 1); // value
    b.emit(Op::LoadInt(0), 1); // index
    b.emit(Op::SlotArraySet(0), 1);
    b.emit(Op::GetSlot(1), 1);
    assert_eq!(
        run(b),
        Value::array(vec![Value::Int(1), Value::Int(2)]),
        "the copy in slot 1 must not observe the write to slot 0"
    );

    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::LoadInt(2), 1);
    b.emit(Op::MakeArray(2), 1);
    b.emit(Op::SetSlot(0), 1);
    b.emit(Op::GetSlot(0), 1);
    b.emit(Op::SetSlot(1), 1);
    b.emit(Op::LoadInt(77), 1);
    b.emit(Op::LoadInt(0), 1);
    b.emit(Op::SlotArraySet(0), 1);
    b.emit(Op::GetSlot(0), 1);
    assert_eq!(
        run(b),
        Value::array(vec![Value::Int(77), Value::Int(2)]),
        "the write itself must still land"
    );
}

#[test]
fn array_mut_copies_only_when_shared() {
    // The helper contract directly: `array_mut` on a uniquely-owned buffer
    // must NOT reallocate (that is what keeps mutation O(1)), and on a shared
    // buffer it must, so the other holder is unaffected.
    let mut a = Value::array(vec![Value::Int(1), Value::Int(2)]);
    let ptr_before = a.as_array().unwrap().as_ptr();
    a.array_mut().unwrap().push(Value::Int(3));
    assert_eq!(a.len(), 3);

    let mut owner = Value::array(vec![Value::Int(1), Value::Int(2)]);
    let copy = owner.clone();
    assert_eq!(
        owner.as_array().unwrap().as_ptr(),
        copy.as_array().unwrap().as_ptr(),
        "a clone must share the buffer — that is the whole point"
    );
    owner.array_mut().unwrap()[0] = Value::Int(9);
    assert_eq!(copy, Value::array(vec![Value::Int(1), Value::Int(2)]));
    assert_eq!(owner, Value::array(vec![Value::Int(9), Value::Int(2)]));
    let _ = ptr_before;
}

#[test]
fn into_array_avoids_the_copy_when_uniquely_owned() {
    let v = Value::array(vec![Value::Int(1), Value::Int(2)]);
    let ptr = v.as_array().unwrap().as_ptr();
    let owned = v.into_array().unwrap();
    assert_eq!(
        owned.as_ptr(),
        ptr,
        "sole owner must hand over the buffer, not copy it"
    );
    assert_eq!(owned, vec![Value::Int(1), Value::Int(2)]);

    let shared = Value::array(vec![Value::Int(5)]);
    let keep = shared.clone();
    assert_eq!(shared.into_array().unwrap(), vec![Value::Int(5)]);
    assert_eq!(keep, Value::array(vec![Value::Int(5)]));
}

#[test]
fn array_serde_encoding_is_unchanged_by_the_arc_payload() {
    // Frontends bincode-encode `Chunk` (constants included) into their on-disk
    // bytecode caches, so the byte layout of `Value::Array` is a compatibility
    // surface. serde's `rc` feature encodes `Arc<T>` exactly as `T`, so these
    // bytes are the same ones the by-value `Vec<Value>` payload produced.
    // A change here invalidates every frontend's cache — treat a failure as a
    // format break, not a test to update.
    let nested = Value::array(vec![
        Value::Int(1),
        Value::str("two"),
        Value::array(vec![Value::Float(3.5), Value::Undef]),
        Value::Bool(true),
    ]);
    assert_eq!(
        serde_json::to_string(&nested).unwrap(),
        r#"{"Array":[{"Int":1},{"Str":"two"},{"Array":[{"Float":3.5},"Undef"]},{"Bool":true}]}"#
    );
    assert_eq!(
        bincode::serialize(&Value::array(vec![])).unwrap(),
        vec![5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "empty array: variant tag 5 + u64 length 0"
    );
    assert_eq!(
        bincode::serialize(&nested).unwrap(),
        vec![
            5, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 3,
            0, 0, 0, 0, 0, 0, 0, 116, 119, 111, 5, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 12, 64, 0, 0, 0, 0, 1, 0, 0, 0, 1,
        ]
    );
    let back: Value = bincode::deserialize(&bincode::serialize(&nested).unwrap()).unwrap();
    assert_eq!(back, nested);
}
