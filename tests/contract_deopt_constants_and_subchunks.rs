//! Contract tests for previously-uncovered fusevm surfaces:
//!   - DeoptInfo size constants: MAX_DEOPT_FRAMES=4, MAX_DEOPT_SLOTS_PER_FRAME=16,
//!     MAX_DEOPT_STACK=32. The trace-JIT ABI is defined by these; any change
//!     ripples into the deopt frame buffer layout that the trace fn writes.
//!   - STACK_KIND_INT=0 / STACK_KIND_FLOAT=1: tags interpreted by the VM when
//!     materializing abstract-stack entries after a side-exit. The numeric
//!     values are load-bearing — flipping them would silently corrupt
//!     materialized float values into int bit-patterns.
//!   - `Chunk::new()` produces a fully-empty chunk with empty source.
//!   - `ChunkBuilder::add_sub_chunk` assigns sequential u16 indices starting
//!     from 0; the returned index is the value passed to Op::CmdSubst /
//!     Op::ProcessSubIn/Out / Op::TrapSet.
//!   - `ChunkBuilder.emit` keeps `lines` parallel to `ops` (one push each).
//!   - `ChunkBuilder::set_source` propagates the source name into the built
//!     chunk for error messages.
//!   - `Op::Pop`, `Op::Dup`, `Op::Swap`, `Op::Nop` round-trip via serde
//!     (covers the basic stack ops not in the fused-superinstruction set).
//!   - `DeoptFrame::zeroed().slots` is fully zero-initialized to the MAX
//!     length (so trace fn writes hit valid memory).
//!
//! Earlier rounds pinned:
//!   - Op serde roundtrip for the 8 FUSED superinstructions (not the basic
//!     stack ops Pop/Dup/Swap/Nop)
//!   - DeoptInfo::zeroed all-zero counts (not the size CONSTANTS)
//!   - ChunkBuilder add_constant sequential + add_name dedup
//!   - chunk_default_source_empty (not via Chunk::new directly with set_source)

use fusevm::chunk::{Chunk, ChunkBuilder};
use fusevm::jit::{
    DeoptFrame, MAX_DEOPT_FRAMES, MAX_DEOPT_SLOTS_PER_FRAME, MAX_DEOPT_STACK, STACK_KIND_FLOAT,
    STACK_KIND_INT,
};
use fusevm::op::Op;

/// MAX_DEOPT_FRAMES, MAX_DEOPT_SLOTS_PER_FRAME, MAX_DEOPT_STACK are the
/// trace-JIT ABI sizes. Pinning their exact values guards against silent
/// buffer-size changes that the trace fn (raw extern "C") writes through.
#[test]
fn test_deopt_size_constants_match_jit_abi_contract() {
    assert_eq!(
        MAX_DEOPT_FRAMES, 4,
        "MAX_DEOPT_FRAMES must be 4 (JIT ABI contract); got {MAX_DEOPT_FRAMES}"
    );
    assert_eq!(
        MAX_DEOPT_SLOTS_PER_FRAME, 16,
        "MAX_DEOPT_SLOTS_PER_FRAME must be 16; got {MAX_DEOPT_SLOTS_PER_FRAME}"
    );
    assert_eq!(
        MAX_DEOPT_STACK, 32,
        "MAX_DEOPT_STACK must be 32; got {MAX_DEOPT_STACK}"
    );
}

/// STACK_KIND_INT=0, STACK_KIND_FLOAT=1. The numeric values are load-bearing
/// because the VM dispatches on them when materializing abstract-stack entries
/// after a side-exit.
#[test]
fn test_stack_kind_tag_numeric_values_are_load_bearing() {
    assert_eq!(STACK_KIND_INT, 0, "STACK_KIND_INT must be 0 (tag dispatch)");
    assert_eq!(
        STACK_KIND_FLOAT, 1,
        "STACK_KIND_FLOAT must be 1 (tag dispatch)"
    );
}

/// `Chunk::new()` produces an empty chunk identical to `Chunk::default()`.
/// Pin: ops/constants/names/lines/sub_entries/block_ranges/sub_chunks empty,
/// source empty, op_hash 0 (computed only at `build()`).
#[test]
fn test_chunk_new_yields_fully_empty_default_state() {
    let c = Chunk::new();
    assert!(c.ops.is_empty(), "Chunk::new ops must be empty");
    assert!(c.constants.is_empty(), "Chunk::new constants must be empty");
    assert!(c.names.is_empty(), "Chunk::new names must be empty");
    assert!(c.lines.is_empty(), "Chunk::new lines must be empty");
    assert!(
        c.sub_entries.is_empty(),
        "Chunk::new sub_entries must be empty"
    );
    assert!(
        c.block_ranges.is_empty(),
        "Chunk::new block_ranges must be empty"
    );
    assert!(
        c.sub_chunks.is_empty(),
        "Chunk::new sub_chunks must be empty"
    );
    assert_eq!(c.source, "", "Chunk::new source must be empty string");
    assert_eq!(c.op_hash, 0, "Chunk::new op_hash must be 0 (not yet built)");
}

/// `ChunkBuilder::add_sub_chunk` returns sequential u16 indices starting at 0.
/// These indices are what Op::CmdSubst / Op::ProcessSubIn / Op::ProcessSubOut /
/// Op::TrapSet carry inline.
#[test]
fn test_add_sub_chunk_returns_sequential_u16_indices_from_zero() {
    let mut b = ChunkBuilder::new();
    let i0 = b.add_sub_chunk(ChunkBuilder::new().build());
    let i1 = b.add_sub_chunk(ChunkBuilder::new().build());
    let i2 = b.add_sub_chunk(ChunkBuilder::new().build());
    assert_eq!(i0, 0, "first sub_chunk index must be 0");
    assert_eq!(i1, 1, "second sub_chunk index must be 1");
    assert_eq!(i2, 2, "third sub_chunk index must be 2");
    let c = b.build();
    assert_eq!(
        c.sub_chunks.len(),
        3,
        "sub_chunks must contain all 3 entries"
    );
}

/// `ChunkBuilder.emit` keeps `lines` parallel to `ops` — one push per emit
/// call, indices align.
#[test]
fn test_emit_keeps_lines_parallel_to_ops_index_aligned() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::Nop, 10);
    b.emit(Op::Pop, 20);
    b.emit(Op::Dup, 30);
    let c = b.build();
    assert_eq!(c.ops.len(), 3);
    assert_eq!(
        c.lines,
        vec![10, 20, 30],
        "lines must be parallel to ops in emit order"
    );
}

/// `ChunkBuilder::set_source` propagates the source name into the built chunk.
#[test]
fn test_set_source_propagates_into_built_chunk_source_field() {
    let mut b = ChunkBuilder::new();
    b.set_source("test_script.fv");
    let c = b.build();
    assert_eq!(
        c.source, "test_script.fv",
        "set_source must propagate into Chunk.source; got {:?}",
        c.source
    );
}

/// Basic stack Ops (Pop, Dup, Swap, Nop, Rot, Dup2) must round-trip via serde.
/// Pins the serialization-stability promise for the most-used non-fused ops.
#[test]
fn test_basic_stack_ops_roundtrip_via_serde() {
    let ops = vec![Op::Nop, Op::Pop, Op::Dup, Op::Dup2, Op::Swap, Op::Rot];
    for op in ops {
        let json = serde_json::to_string(&op).expect("serialize");
        let back: Op = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(op, back, "basic stack op {op:?} must roundtrip via serde");
    }
}

/// `DeoptFrame::zeroed().slots` is a fully zero-initialized fixed array of
/// MAX_DEOPT_SLOTS_PER_FRAME entries. Pins the buffer-size contract: the
/// trace fn writes through this buffer by raw pointer.
#[test]
fn test_deopt_frame_zeroed_slots_array_is_max_length_all_zero() {
    let f = DeoptFrame::zeroed();
    assert_eq!(
        f.slots.len(),
        MAX_DEOPT_SLOTS_PER_FRAME,
        "DeoptFrame slots length must match MAX_DEOPT_SLOTS_PER_FRAME"
    );
    assert!(
        f.slots.iter().all(|&v| v == 0),
        "all slots must be zero-initialized in zeroed DeoptFrame"
    );
}
