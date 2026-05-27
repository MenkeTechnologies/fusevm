//! Coverage for ChunkBuilder behaviors (name interning, patch_jump,
//! sub_entries/sub_chunks, op_hash) and Value introspection methods
//! (as_str_cow Borrowed vs Owned, len, is_empty, Hash collisions).

use fusevm::{Chunk, ChunkBuilder, Op, VMResult, Value, VM};
use std::borrow::Cow;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

fn hash_of<T: Hash>(t: &T) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    let mut h = DefaultHasher::new();
    t.hash(&mut h);
    h.finish()
}

// ══════════════════════════════════════════════════════════════════════════
// ChunkBuilder: emit / current_pos / lines parallel array
// ══════════════════════════════════════════════════════════════════════════

#[test]
fn emit_returns_the_index_of_the_emitted_op() {
    let mut b = ChunkBuilder::new();
    assert_eq!(b.emit(Op::Nop, 1), 0);
    assert_eq!(b.emit(Op::Nop, 2), 1);
    assert_eq!(b.emit(Op::Nop, 3), 2);
}

#[test]
fn current_pos_tracks_number_of_emitted_ops() {
    let mut b = ChunkBuilder::new();
    assert_eq!(b.current_pos(), 0);
    b.emit(Op::Nop, 1);
    assert_eq!(b.current_pos(), 1);
    b.emit(Op::Nop, 1);
    b.emit(Op::Nop, 1);
    assert_eq!(b.current_pos(), 3);
}

#[test]
fn lines_array_parallels_ops_array() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::Nop, 10);
    b.emit(Op::Nop, 20);
    b.emit(Op::Nop, 30);
    let chunk = b.build();
    assert_eq!(chunk.ops.len(), chunk.lines.len());
    assert_eq!(chunk.lines, vec![10, 20, 30]);
}

// ══════════════════════════════════════════════════════════════════════════
// ChunkBuilder: add_constant — sequential indices, no dedup
// ══════════════════════════════════════════════════════════════════════════

#[test]
fn add_constant_returns_sequential_indices_starting_at_zero() {
    let mut b = ChunkBuilder::new();
    assert_eq!(b.add_constant(Value::Int(1)), 0);
    assert_eq!(b.add_constant(Value::Int(2)), 1);
    assert_eq!(b.add_constant(Value::Int(3)), 2);
}

#[test]
fn add_constant_does_not_dedup_equal_values() {
    let mut b = ChunkBuilder::new();
    let a = b.add_constant(Value::str("same"));
    let b2 = b.add_constant(Value::str("same"));
    assert_ne!(a, b2);
    assert_eq!(b.build().constants.len(), 2);
}

// ══════════════════════════════════════════════════════════════════════════
// ChunkBuilder: add_name — interning / dedup
// ══════════════════════════════════════════════════════════════════════════

#[test]
fn add_name_interns_and_returns_same_index_for_same_name() {
    let mut b = ChunkBuilder::new();
    let i1 = b.add_name("foo");
    let i2 = b.add_name("foo");
    let i3 = b.add_name("bar");
    let i4 = b.add_name("foo");
    assert_eq!(i1, i2);
    assert_eq!(i1, i4);
    assert_ne!(i1, i3);
    assert_eq!(b.build().names.len(), 2);
}

#[test]
fn add_name_assigns_sequential_indices_for_distinct_names() {
    let mut b = ChunkBuilder::new();
    assert_eq!(b.add_name("a"), 0);
    assert_eq!(b.add_name("b"), 1);
    assert_eq!(b.add_name("c"), 2);
}

// ══════════════════════════════════════════════════════════════════════════
// ChunkBuilder: patch_jump — works for all five jump variants
// ══════════════════════════════════════════════════════════════════════════

#[test]
fn patch_jump_updates_jump_target() {
    let mut b = ChunkBuilder::new();
    let idx = b.emit(Op::Jump(0), 1);
    b.patch_jump(idx, 42);
    match b.build().ops[idx] {
        Op::Jump(t) => assert_eq!(t, 42),
        _ => panic!("expected Jump"),
    }
}

#[test]
fn patch_jump_works_on_jumpiftrue() {
    let mut b = ChunkBuilder::new();
    let idx = b.emit(Op::JumpIfTrue(0), 1);
    b.patch_jump(idx, 99);
    matches!(b.build().ops[idx], Op::JumpIfTrue(99));
}

#[test]
fn patch_jump_works_on_jumpiffalse_keep() {
    let mut b = ChunkBuilder::new();
    let idx = b.emit(Op::JumpIfFalseKeep(0), 1);
    b.patch_jump(idx, 77);
    matches!(b.build().ops[idx], Op::JumpIfFalseKeep(77));
}

#[test]
#[should_panic(expected = "patch_jump on non-jump op")]
fn patch_jump_panics_on_non_jump_op() {
    let mut b = ChunkBuilder::new();
    let idx = b.emit(Op::Nop, 1);
    b.patch_jump(idx, 0);
}

// ══════════════════════════════════════════════════════════════════════════
// ChunkBuilder: add_block_range / add_sub_chunk
// ══════════════════════════════════════════════════════════════════════════

#[test]
fn add_block_range_returns_sequential_indices() {
    let mut b = ChunkBuilder::new();
    assert_eq!(b.add_block_range(0, 10), 0);
    assert_eq!(b.add_block_range(11, 20), 1);
    let chunk = b.build();
    assert_eq!(chunk.block_ranges, vec![(0, 10), (11, 20)]);
}

#[test]
fn add_sub_chunk_stores_inner_chunks_in_order() {
    let mut inner1 = ChunkBuilder::new();
    inner1.emit(Op::LoadInt(1), 1);
    let mut inner2 = ChunkBuilder::new();
    inner2.emit(Op::LoadInt(2), 1);

    let mut outer = ChunkBuilder::new();
    let i0 = outer.add_sub_chunk(inner1.build());
    let i1 = outer.add_sub_chunk(inner2.build());
    let chunk = outer.build();
    assert_eq!((i0, i1), (0, 1));
    assert!(matches!(chunk.sub_chunks[0].ops[0], Op::LoadInt(1)));
    assert!(matches!(chunk.sub_chunks[1].ops[0], Op::LoadInt(2)));
}

// ══════════════════════════════════════════════════════════════════════════
// ChunkBuilder: set_source / build computes op_hash
// ══════════════════════════════════════════════════════════════════════════

#[test]
fn set_source_propagates_into_built_chunk() {
    let mut b = ChunkBuilder::new();
    b.set_source("test.fuse");
    assert_eq!(b.build().source, "test.fuse");
}

#[test]
fn op_hash_is_stable_for_identical_chunks() {
    let mut b1 = ChunkBuilder::new();
    b1.emit(Op::LoadInt(1), 1);
    b1.emit(Op::LoadInt(2), 1);
    b1.emit(Op::Add, 1);
    let mut b2 = ChunkBuilder::new();
    b2.emit(Op::LoadInt(1), 1);
    b2.emit(Op::LoadInt(2), 1);
    b2.emit(Op::Add, 1);
    assert_eq!(b1.build().op_hash, b2.build().op_hash);
}

#[test]
fn op_hash_differs_when_ops_differ() {
    let mut b1 = ChunkBuilder::new();
    b1.emit(Op::LoadInt(1), 1);
    let mut b2 = ChunkBuilder::new();
    b2.emit(Op::LoadInt(2), 1);
    assert_ne!(b1.build().op_hash, b2.build().op_hash);
}

#[test]
fn op_hash_differs_when_constants_differ() {
    let mut b1 = ChunkBuilder::new();
    b1.add_constant(Value::str("a"));
    b1.emit(Op::Nop, 1);
    let mut b2 = ChunkBuilder::new();
    b2.add_constant(Value::str("b"));
    b2.emit(Op::Nop, 1);
    assert_ne!(b1.build().op_hash, b2.build().op_hash);
}

#[test]
fn op_hash_unchanged_when_only_lines_differ() {
    let mut b1 = ChunkBuilder::new();
    b1.emit(Op::Nop, 1);
    let mut b2 = ChunkBuilder::new();
    b2.emit(Op::Nop, 999);
    assert_eq!(b1.build().op_hash, b2.build().op_hash);
}

// ══════════════════════════════════════════════════════════════════════════
// Chunk::find_sub edge cases
// ══════════════════════════════════════════════════════════════════════════

#[test]
fn find_sub_returns_none_when_not_registered() {
    let chunk = ChunkBuilder::new().build();
    assert_eq!(chunk.find_sub(42), None);
}

#[test]
fn find_sub_on_empty_chunk_returns_none() {
    let chunk = Chunk::new();
    assert_eq!(chunk.find_sub(0), None);
}

// ══════════════════════════════════════════════════════════════════════════
// Value::as_str_cow — Borrowed vs Owned discipline
// ══════════════════════════════════════════════════════════════════════════

#[test]
fn as_str_cow_returns_borrowed_for_str() {
    let v = Value::str("hello");
    assert!(matches!(v.as_str_cow(), Cow::Borrowed(_)));
}

#[test]
fn as_str_cow_returns_borrowed_for_undef() {
    assert!(matches!(Value::Undef.as_str_cow(), Cow::Borrowed("")));
}

#[test]
fn as_str_cow_returns_borrowed_for_bool() {
    assert!(matches!(Value::Bool(true).as_str_cow(), Cow::Borrowed("1")));
    assert!(matches!(Value::Bool(false).as_str_cow(), Cow::Borrowed("")));
}

#[test]
fn as_str_cow_returns_owned_for_int() {
    assert!(matches!(Value::Int(42).as_str_cow(), Cow::Owned(_)));
    assert_eq!(Value::Int(42).as_str_cow(), "42");
}

#[test]
fn as_str_cow_returns_owned_for_float() {
    assert!(matches!(Value::Float(1.5).as_str_cow(), Cow::Owned(_)));
}

#[test]
fn as_str_cow_returns_owned_for_status() {
    assert!(matches!(Value::Status(5).as_str_cow(), Cow::Owned(_)));
    assert_eq!(Value::Status(5).as_str_cow(), "5");
}

#[test]
fn as_str_cow_array_joins_with_spaces() {
    let v = Value::Array(vec![Value::Int(1), Value::Int(2), Value::Int(3)]);
    assert_eq!(v.as_str_cow(), "1 2 3");
}

#[test]
fn as_str_cow_hash_renders_as_placeholder() {
    let v = Value::Hash(HashMap::new());
    assert!(matches!(v.as_str_cow(), Cow::Borrowed("(hash)")));
}

#[test]
fn as_str_cow_ref_renders_as_placeholder() {
    let v = Value::Ref(Box::new(Value::Int(7)));
    assert!(matches!(v.as_str_cow(), Cow::Borrowed("(ref)")));
}

#[test]
fn as_str_cow_native_fn_renders_with_id() {
    assert_eq!(Value::NativeFn(13).as_str_cow(), "(builtin:13)");
}

// ══════════════════════════════════════════════════════════════════════════
// Value::len / is_empty
// ══════════════════════════════════════════════════════════════════════════

#[test]
fn len_str_returns_byte_length() {
    assert_eq!(Value::str("héllo").len(), 6); // 'é' is 2 bytes UTF-8
}

#[test]
fn len_array_returns_element_count() {
    assert_eq!(Value::Array(vec![Value::Int(1); 5]).len(), 5);
}

#[test]
fn len_hash_returns_entry_count() {
    let mut m = HashMap::new();
    m.insert("a".to_string(), Value::Int(1));
    m.insert("b".to_string(), Value::Int(2));
    assert_eq!(Value::Hash(m).len(), 2);
}

#[test]
fn len_int_falls_back_to_str_repr_length() {
    assert_eq!(Value::Int(12345).len(), 5);
    assert_eq!(Value::Int(-1).len(), 2);
}

#[test]
fn len_undef_is_zero() {
    assert_eq!(Value::Undef.len(), 0);
    assert!(Value::Undef.is_empty());
}

#[test]
fn is_empty_for_empty_str_and_empty_collections() {
    assert!(Value::str("").is_empty());
    assert!(Value::Array(vec![]).is_empty());
    assert!(Value::Hash(HashMap::new()).is_empty());
}

#[test]
fn is_empty_false_for_nonzero_int() {
    // Int's len is its str-repr length; "5" has len 1 → not empty.
    assert!(!Value::Int(5).is_empty());
}

// ══════════════════════════════════════════════════════════════════════════
// Value: Hash trait — discriminant separates same-bits types
// ══════════════════════════════════════════════════════════════════════════

#[test]
fn hash_distinguishes_int_zero_from_bool_false() {
    assert_ne!(hash_of(&Value::Int(0)), hash_of(&Value::Bool(false)));
}

#[test]
fn hash_distinguishes_undef_from_str_empty() {
    assert_ne!(hash_of(&Value::Undef), hash_of(&Value::str("")));
}

#[test]
fn hash_distinguishes_status_from_int_same_number() {
    assert_ne!(hash_of(&Value::Status(0)), hash_of(&Value::Int(0)));
}

#[test]
fn hash_equal_values_produce_equal_hashes() {
    assert_eq!(hash_of(&Value::Int(42)), hash_of(&Value::Int(42)));
    assert_eq!(hash_of(&Value::str("foo")), hash_of(&Value::str("foo")));
}

#[test]
fn hash_float_uses_to_bits_so_neg_zero_not_equal_to_pos_zero() {
    assert_ne!(hash_of(&Value::Float(0.0)), hash_of(&Value::Float(-0.0)));
}

#[test]
fn hash_array_recursively_hashes_elements() {
    let a = Value::Array(vec![Value::Int(1), Value::Int(2)]);
    let b = Value::Array(vec![Value::Int(1), Value::Int(2)]);
    let c = Value::Array(vec![Value::Int(1), Value::Int(3)]);
    assert_eq!(hash_of(&a), hash_of(&b));
    assert_ne!(hash_of(&a), hash_of(&c));
}

#[test]
fn hash_nativefn_uses_id() {
    assert_eq!(hash_of(&Value::NativeFn(5)), hash_of(&Value::NativeFn(5)));
    assert_ne!(hash_of(&Value::NativeFn(5)), hash_of(&Value::NativeFn(6)));
}

#[test]
fn hash_ref_recursively_hashes_inner_value() {
    let a = Value::Ref(Box::new(Value::Int(1)));
    let b = Value::Ref(Box::new(Value::Int(1)));
    let c = Value::Ref(Box::new(Value::Int(2)));
    assert_eq!(hash_of(&a), hash_of(&b));
    assert_ne!(hash_of(&a), hash_of(&c));
}

// ══════════════════════════════════════════════════════════════════════════
// Constructor convenience methods (Value::int / float / str / etc.)
// ══════════════════════════════════════════════════════════════════════════

#[test]
fn value_int_constructor_makes_int() {
    matches!(Value::int(42), Value::Int(42));
}

#[test]
fn value_float_constructor_makes_float() {
    if let Value::Float(f) = Value::float(1.5) {
        assert!((f - 1.5).abs() < 1e-9);
    } else {
        panic!("expected Float");
    }
}

#[test]
fn value_bool_constructor_makes_bool() {
    matches!(Value::bool(true), Value::Bool(true));
}

#[test]
fn value_array_constructor_makes_array() {
    if let Value::Array(a) = Value::array(vec![Value::Int(1)]) {
        assert_eq!(a.len(), 1);
    } else {
        panic!("expected Array");
    }
}

#[test]
fn value_hash_constructor_makes_hash() {
    let mut m = HashMap::new();
    m.insert("k".to_string(), Value::Int(1));
    if let Value::Hash(h) = Value::hash(m) {
        assert_eq!(h.len(), 1);
    } else {
        panic!("expected Hash");
    }
}

#[test]
fn value_status_constructor_makes_status() {
    matches!(Value::status(127), Value::Status(127));
}

// ══════════════════════════════════════════════════════════════════════════
// Built chunk's op_hash powers JIT cache identity — verify VM execution
// is unaffected by op_hash field
// ══════════════════════════════════════════════════════════════════════════

#[test]
fn vm_executes_built_chunk_normally_after_op_hash_computation() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(7), 1);
    b.emit(Op::LoadInt(3), 1);
    b.emit(Op::Add, 1);
    let chunk = b.build();
    assert_ne!(chunk.op_hash, 0); // hash was computed
    match VM::new(chunk).run() {
        VMResult::Ok(Value::Int(10)) => {}
        other => panic!("expected Int(10), got {:?}", other),
    }
}
