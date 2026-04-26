//! Tests for ChunkBuilder — the API frontends use to emit bytecode.
//!
//! These tests verify the chunk-building primitives in isolation, without
//! actually running the VM. Useful for catching bugs in the bytecode emission
//! infrastructure that would be masked by VM-level tests.

use fusevm::{Chunk, ChunkBuilder, Op, Value};

#[test]
fn empty_builder_produces_empty_chunk() {
    let chunk = ChunkBuilder::new().build();
    assert!(chunk.ops.is_empty());
    assert!(chunk.constants.is_empty());
    assert!(chunk.names.is_empty());
}

#[test]
fn emit_returns_op_index() {
    let mut b = ChunkBuilder::new();
    let i0 = b.emit(Op::LoadInt(1), 1);
    let i1 = b.emit(Op::LoadInt(2), 1);
    let i2 = b.emit(Op::Add, 1);
    assert_eq!(i0, 0);
    assert_eq!(i1, 1);
    assert_eq!(i2, 2);
}

#[test]
fn line_numbers_recorded_per_op() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(1), 10);
    b.emit(Op::LoadInt(2), 20);
    b.emit(Op::Add, 30);
    let chunk = b.build();
    assert_eq!(chunk.lines, vec![10, 20, 30]);
}

#[test]
fn name_pool_dedupes_repeated_names() {
    let mut b = ChunkBuilder::new();
    let a1 = b.add_name("foo");
    let a2 = b.add_name("foo");
    let a3 = b.add_name("bar");
    let a4 = b.add_name("foo");
    assert_eq!(a1, a2, "same name should return same index");
    assert_eq!(a1, a4);
    assert_ne!(a1, a3, "different names should have different indices");
    assert_eq!(b.build().names.len(), 2);
}

#[test]
fn constant_pool_does_not_dedupe() {
    // Constants are NOT deduplicated — two identical constants get separate slots.
    // (This is intentional; frontends can emit constants without tracking dups.)
    let mut b = ChunkBuilder::new();
    let i1 = b.add_constant(Value::Int(42));
    let i2 = b.add_constant(Value::Int(42));
    assert_ne!(i1, i2);
    assert_eq!(b.build().constants.len(), 2);
}

#[test]
fn current_pos_reflects_op_count() {
    let mut b = ChunkBuilder::new();
    assert_eq!(b.current_pos(), 0);
    b.emit(Op::Nop, 1);
    assert_eq!(b.current_pos(), 1);
    b.emit(Op::Nop, 1);
    b.emit(Op::Nop, 1);
    assert_eq!(b.current_pos(), 3);
}

#[test]
fn patch_jump_updates_target() {
    let mut b = ChunkBuilder::new();
    let jump_idx = b.emit(Op::Jump(0), 1);
    b.emit(Op::Nop, 1);
    b.emit(Op::Nop, 1);
    let target = b.current_pos();
    b.patch_jump(jump_idx, target);
    let chunk = b.build();
    match chunk.ops[jump_idx] {
        Op::Jump(t) => assert_eq!(t, target),
        _ => panic!("expected Jump op"),
    }
}

#[test]
fn patch_jump_works_on_all_jump_variants() {
    let mut b = ChunkBuilder::new();
    let j1 = b.emit(Op::JumpIfTrue(0), 1);
    let j2 = b.emit(Op::JumpIfFalse(0), 1);
    let j3 = b.emit(Op::JumpIfTrueKeep(0), 1);
    let j4 = b.emit(Op::JumpIfFalseKeep(0), 1);
    b.patch_jump(j1, 100);
    b.patch_jump(j2, 200);
    b.patch_jump(j3, 300);
    b.patch_jump(j4, 400);
    let chunk = b.build();
    assert!(matches!(chunk.ops[j1], Op::JumpIfTrue(100)));
    assert!(matches!(chunk.ops[j2], Op::JumpIfFalse(200)));
    assert!(matches!(chunk.ops[j3], Op::JumpIfTrueKeep(300)));
    assert!(matches!(chunk.ops[j4], Op::JumpIfFalseKeep(400)));
}

#[test]
#[should_panic(expected = "patch_jump on non-jump op")]
fn patch_jump_panics_on_non_jump_op() {
    let mut b = ChunkBuilder::new();
    let idx = b.emit(Op::LoadInt(1), 1);
    b.patch_jump(idx, 99);
}

#[test]
fn sub_entry_lookup_finds_function() {
    let mut b = ChunkBuilder::new();
    let foo = b.add_name("foo");
    b.add_sub_entry(foo, 42);
    let chunk = b.build();
    assert_eq!(chunk.find_sub(foo), Some(42));
    assert_eq!(chunk.find_sub(999), None);
}

#[test]
fn block_range_assigned_sequential_indices() {
    let mut b = ChunkBuilder::new();
    let i1 = b.add_block_range(10, 20);
    let i2 = b.add_block_range(30, 40);
    assert_eq!(i1, 0);
    assert_eq!(i2, 1);
    let chunk = b.build();
    assert_eq!(chunk.block_ranges, vec![(10, 20), (30, 40)]);
}

#[test]
fn source_filename_persists() {
    let mut b = ChunkBuilder::new();
    b.set_source("test.fuse");
    let chunk = b.build();
    assert_eq!(chunk.source, "test.fuse");
}

#[test]
fn op_hash_identical_for_equal_chunks() {
    // Two chunks with identical ops + constants should have the same op_hash
    // (allowing the JIT cache to deduplicate compilation work).
    let chunk_a = {
        let mut b = ChunkBuilder::new();
        b.emit(Op::LoadInt(1), 1);
        b.emit(Op::LoadInt(2), 1);
        b.emit(Op::Add, 1);
        b.build()
    };
    let chunk_b = {
        let mut b = ChunkBuilder::new();
        b.emit(Op::LoadInt(1), 99); // different line — shouldn't affect hash
        b.emit(Op::LoadInt(2), 99);
        b.emit(Op::Add, 99);
        b.build()
    };
    assert_eq!(chunk_a.op_hash, chunk_b.op_hash);
}

#[test]
fn op_hash_differs_for_different_ops() {
    let chunk_a = {
        let mut b = ChunkBuilder::new();
        b.emit(Op::LoadInt(1), 1);
        b.emit(Op::LoadInt(2), 1);
        b.emit(Op::Add, 1);
        b.build()
    };
    let chunk_b = {
        let mut b = ChunkBuilder::new();
        b.emit(Op::LoadInt(1), 1);
        b.emit(Op::LoadInt(2), 1);
        b.emit(Op::Sub, 1); // different op
        b.build()
    };
    assert_ne!(chunk_a.op_hash, chunk_b.op_hash);
}

#[test]
fn op_hash_differs_for_different_constants() {
    let chunk_a = {
        let mut b = ChunkBuilder::new();
        let c = b.add_constant(Value::str("hello"));
        b.emit(Op::LoadConst(c), 1);
        b.build()
    };
    let chunk_b = {
        let mut b = ChunkBuilder::new();
        let c = b.add_constant(Value::str("world"));
        b.emit(Op::LoadConst(c), 1);
        b.build()
    };
    assert_ne!(chunk_a.op_hash, chunk_b.op_hash);
}

#[test]
fn op_size_under_24_bytes() {
    // Op should stay small for cache-friendly dispatch — verify the bound.
    assert!(
        std::mem::size_of::<Op>() <= 24,
        "Op size: {} bytes",
        std::mem::size_of::<Op>()
    );
}

#[test]
fn default_chunk_is_empty() {
    let c: Chunk = Default::default();
    assert!(c.ops.is_empty());
    assert_eq!(c.op_hash, 0);
}
