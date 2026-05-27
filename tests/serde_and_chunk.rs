#![allow(clippy::approx_constant)]
//! Serde round-trip and Chunk/ChunkBuilder edge-case coverage.

use fusevm::{Chunk, ChunkBuilder, Op, Value};
use std::collections::HashMap;

// ── Value serde round-trip ─────────────────────────────────────────────────

fn rt(v: &Value) -> Value {
    let json = serde_json::to_string(v).expect("serialize");
    serde_json::from_str(&json).expect("deserialize")
}

#[test]
fn value_serde_undef() {
    assert_eq!(rt(&Value::Undef), Value::Undef);
}

#[test]
fn value_serde_bool() {
    assert_eq!(rt(&Value::Bool(true)), Value::Bool(true));
    assert_eq!(rt(&Value::Bool(false)), Value::Bool(false));
}

#[test]
fn value_serde_int_extremes() {
    assert_eq!(rt(&Value::Int(0)), Value::Int(0));
    assert_eq!(rt(&Value::Int(i64::MAX)), Value::Int(i64::MAX));
    assert_eq!(rt(&Value::Int(i64::MIN)), Value::Int(i64::MIN));
    assert_eq!(rt(&Value::Int(-1)), Value::Int(-1));
}

#[test]
fn value_serde_float_normal() {
    assert_eq!(rt(&Value::Float(3.14159)), Value::Float(3.14159));
    assert_eq!(rt(&Value::Float(0.0)), Value::Float(0.0));
}

#[test]
fn value_serde_str_with_special_chars() {
    let v = Value::str("hello\nworld\t\"quoted\"\\back");
    assert_eq!(rt(&v), v);
}

#[test]
fn value_serde_empty_str() {
    assert_eq!(rt(&Value::str("")), Value::str(""));
}

#[test]
fn value_serde_unicode_str() {
    let v = Value::str("héllo 🌍 日本語");
    assert_eq!(rt(&v), v);
}

#[test]
fn value_serde_status() {
    assert_eq!(rt(&Value::Status(127)), Value::Status(127));
    assert_eq!(rt(&Value::Status(-1)), Value::Status(-1));
}

#[test]
fn value_serde_native_fn() {
    assert_eq!(rt(&Value::NativeFn(99)), Value::NativeFn(99));
}

#[test]
fn value_serde_array_of_mixed() {
    let v = Value::array(vec![
        Value::Int(1),
        Value::str("x"),
        Value::Bool(true),
        Value::Undef,
        Value::Float(2.5),
    ]);
    assert_eq!(rt(&v), v);
}

#[test]
fn value_serde_nested_array() {
    let inner = Value::array(vec![Value::Int(1), Value::Int(2)]);
    let outer = Value::array(vec![inner.clone(), inner.clone(), Value::str("end")]);
    assert_eq!(rt(&outer), outer);
}

#[test]
fn value_serde_hash() {
    let mut m = HashMap::new();
    m.insert("name".into(), Value::str("alice"));
    m.insert("age".into(), Value::Int(30));
    let v = Value::hash(m);
    assert_eq!(rt(&v), v);
}

#[test]
fn value_serde_empty_collections() {
    assert_eq!(rt(&Value::array(vec![])), Value::array(vec![]));
    assert_eq!(
        rt(&Value::hash(HashMap::new())),
        Value::hash(HashMap::new())
    );
}

#[test]
fn value_serde_ref_round_trip() {
    let v = Value::Ref(Box::new(Value::Int(7)));
    assert_eq!(rt(&v), v);
}

#[test]
fn value_serde_deeply_nested_ref() {
    let v = Value::Ref(Box::new(Value::Ref(Box::new(Value::Ref(Box::new(
        Value::str("deep"),
    ))))));
    assert_eq!(rt(&v), v);
}

// ── Chunk serde ────────────────────────────────────────────────────────────

#[test]
fn chunk_serde_empty() {
    let c = Chunk::new();
    let json = serde_json::to_string(&c).expect("serialize");
    let c2: Chunk = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(c.ops.len(), c2.ops.len());
    assert!(c2.ops.is_empty());
}

#[test]
fn chunk_serde_with_ops_and_constants() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(42), 1);
    let c = b.add_constant(Value::str("hello"));
    b.emit(Op::LoadConst(c), 2);
    b.emit(Op::Concat, 3);
    let chunk = b.build();

    let json = serde_json::to_string(&chunk).expect("serialize");
    let chunk2: Chunk = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(chunk.ops, chunk2.ops);
    assert_eq!(chunk.constants, chunk2.constants);
    assert_eq!(chunk.lines, chunk2.lines);
}

#[test]
fn chunk_serde_op_hash_is_skipped_but_recomputable() {
    // op_hash has #[serde(skip)] — deserialized chunk has op_hash=0.
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(1), 1);
    let chunk = b.build();
    assert_ne!(chunk.op_hash, 0);

    let json = serde_json::to_string(&chunk).expect("serialize");
    let chunk2: Chunk = serde_json::from_str(&json).expect("deserialize");
    // After round-trip, op_hash is reset to default (0).
    assert_eq!(chunk2.op_hash, 0);
    // But ops are preserved.
    assert_eq!(chunk.ops, chunk2.ops);
}

#[test]
fn chunk_serde_preserves_sub_chunks_recursively() {
    let inner = {
        let mut ib = ChunkBuilder::new();
        ib.emit(Op::LoadInt(99), 1);
        ib.build()
    };
    let mut b = ChunkBuilder::new();
    b.add_sub_chunk(inner);
    let chunk = b.build();

    let json = serde_json::to_string(&chunk).expect("serialize");
    let chunk2: Chunk = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(chunk2.sub_chunks.len(), 1);
    assert_eq!(chunk2.sub_chunks[0].ops, vec![Op::LoadInt(99)]);
}

#[test]
fn chunk_serde_preserves_sub_entries_and_block_ranges() {
    let mut b = ChunkBuilder::new();
    let n = b.add_name("main");
    b.add_sub_entry(n, 5);
    b.add_block_range(0, 10);
    let chunk = b.build();

    let json = serde_json::to_string(&chunk).expect("serialize");
    let chunk2: Chunk = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(chunk2.sub_entries, vec![(0u16, 5usize)]);
    assert_eq!(chunk2.block_ranges, vec![(0usize, 10usize)]);
    assert_eq!(chunk2.names, vec!["main".to_string()]);
}

#[test]
fn chunk_serde_preserves_source() {
    let mut b = ChunkBuilder::new();
    b.set_source("path/to/file.fuse");
    let chunk = b.build();
    let json = serde_json::to_string(&chunk).expect("serialize");
    let chunk2: Chunk = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(chunk2.source, "path/to/file.fuse");
}

// ── ChunkBuilder behaviour ─────────────────────────────────────────────────

#[test]
fn add_name_dedupes() {
    let mut b = ChunkBuilder::new();
    let a1 = b.add_name("var");
    let a2 = b.add_name("var");
    let other = b.add_name("other");
    assert_eq!(a1, a2);
    assert_ne!(a1, other);
    let chunk = b.build();
    assert_eq!(chunk.names.len(), 2);
}

#[test]
fn add_constant_does_not_dedupe() {
    // Constants are NOT deduplicated (documented behaviour).
    let mut b = ChunkBuilder::new();
    let i1 = b.add_constant(Value::Int(7));
    let i2 = b.add_constant(Value::Int(7));
    assert_ne!(i1, i2);
    let chunk = b.build();
    assert_eq!(chunk.constants.len(), 2);
}

#[test]
fn current_pos_reports_next_op_index() {
    let mut b = ChunkBuilder::new();
    assert_eq!(b.current_pos(), 0);
    b.emit(Op::Nop, 1);
    assert_eq!(b.current_pos(), 1);
    b.emit(Op::Nop, 1);
    b.emit(Op::Nop, 1);
    assert_eq!(b.current_pos(), 3);
}

#[test]
fn emit_returns_position_of_emitted_op() {
    let mut b = ChunkBuilder::new();
    assert_eq!(b.emit(Op::Nop, 1), 0);
    assert_eq!(b.emit(Op::Nop, 1), 1);
    assert_eq!(b.emit(Op::Nop, 1), 2);
}

#[test]
fn patch_jump_modifies_jump_target() {
    let mut b = ChunkBuilder::new();
    let jmp = b.emit(Op::Jump(0), 1);
    b.emit(Op::Nop, 1);
    b.emit(Op::Nop, 1);
    let target = b.current_pos();
    b.patch_jump(jmp, target);
    let chunk = b.build();
    assert_eq!(chunk.ops[jmp], Op::Jump(target));
}

#[test]
fn patch_jump_works_on_conditional_variants() {
    let mut b = ChunkBuilder::new();
    let jt = b.emit(Op::JumpIfTrue(0), 1);
    let jf = b.emit(Op::JumpIfFalse(0), 1);
    let jtk = b.emit(Op::JumpIfTrueKeep(0), 1);
    let jfk = b.emit(Op::JumpIfFalseKeep(0), 1);
    b.patch_jump(jt, 100);
    b.patch_jump(jf, 101);
    b.patch_jump(jtk, 102);
    b.patch_jump(jfk, 103);
    let chunk = b.build();
    assert_eq!(chunk.ops[jt], Op::JumpIfTrue(100));
    assert_eq!(chunk.ops[jf], Op::JumpIfFalse(101));
    assert_eq!(chunk.ops[jtk], Op::JumpIfTrueKeep(102));
    assert_eq!(chunk.ops[jfk], Op::JumpIfFalseKeep(103));
}

#[test]
#[should_panic(expected = "patch_jump on non-jump op")]
fn patch_jump_panics_on_non_jump_op() {
    let mut b = ChunkBuilder::new();
    let idx = b.emit(Op::Nop, 1);
    b.patch_jump(idx, 42);
}

#[test]
fn build_records_op_count_and_lines_in_parallel() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(1), 10);
    b.emit(Op::LoadInt(2), 20);
    b.emit(Op::Add, 30);
    let chunk = b.build();
    assert_eq!(chunk.ops.len(), chunk.lines.len());
    assert_eq!(chunk.lines, vec![10u32, 20, 30]);
}

#[test]
fn op_hash_differs_for_different_ops() {
    let a = {
        let mut b = ChunkBuilder::new();
        b.emit(Op::LoadInt(1), 1);
        b.build()
    };
    let b = {
        let mut b = ChunkBuilder::new();
        b.emit(Op::LoadInt(2), 1);
        b.build()
    };
    assert_ne!(a.op_hash, b.op_hash);
}

#[test]
fn op_hash_differs_for_different_constants() {
    let a = {
        let mut b = ChunkBuilder::new();
        b.add_constant(Value::str("alpha"));
        b.build()
    };
    let b = {
        let mut b = ChunkBuilder::new();
        b.add_constant(Value::str("beta"));
        b.build()
    };
    assert_ne!(a.op_hash, b.op_hash);
}

#[test]
fn empty_chunk_op_hash_is_stable() {
    let a = ChunkBuilder::new().build();
    let b = ChunkBuilder::new().build();
    assert_eq!(a.op_hash, b.op_hash);
}

#[test]
fn add_block_range_returns_indices_in_order() {
    let mut b = ChunkBuilder::new();
    let i0 = b.add_block_range(0, 5);
    let i1 = b.add_block_range(6, 10);
    assert_eq!((i0, i1), (0, 1));
    let chunk = b.build();
    assert_eq!(chunk.block_ranges, vec![(0usize, 5), (6, 10)]);
}

#[test]
fn find_sub_returns_none_for_unknown_name() {
    let chunk = ChunkBuilder::new().build();
    assert_eq!(chunk.find_sub(0), None);
    assert_eq!(chunk.find_sub(99), None);
}

#[test]
fn chunk_default_is_empty() {
    let c = Chunk::default();
    assert!(c.ops.is_empty());
    assert!(c.constants.is_empty());
    assert!(c.names.is_empty());
    assert!(c.lines.is_empty());
    assert!(c.sub_entries.is_empty());
    assert!(c.block_ranges.is_empty());
    assert!(c.sub_chunks.is_empty());
    assert_eq!(c.source, "");
    assert_eq!(c.op_hash, 0);
}

#[test]
fn chunk_clone_preserves_all_fields() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(42), 1);
    b.add_constant(Value::str("k"));
    b.add_name("foo");
    b.set_source("x.fuse");
    let chunk = b.build();
    let clone = chunk.clone();
    assert_eq!(chunk.ops, clone.ops);
    assert_eq!(chunk.constants, clone.constants);
    assert_eq!(chunk.names, clone.names);
    assert_eq!(chunk.lines, clone.lines);
    assert_eq!(chunk.source, clone.source);
    assert_eq!(chunk.op_hash, clone.op_hash);
}
