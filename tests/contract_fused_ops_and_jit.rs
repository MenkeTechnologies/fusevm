//! Contract tests for previously-uncovered fusevm surfaces.
//!
//! Targets:
//! - Op serde roundtrip for ALL 8 fused superinstructions
//! - Empty Chunk: VM::new + VM::run produces VMResult::Halted (no panic)
//! - VM::new allocates exactly chunk.names.len() globals (not max(1, len))
//! - ChunkBuilder add_constant returns sequential indices (NOT deduped)
//! - ChunkBuilder add_name DEDUPES (round-tripped from the docstring claim)
//! - DeoptInfo::zeroed has frame_count=0, stack_count=0, resume_ip=0
//! - DeoptFrame::zeroed has slot_count=0 and zero return_ip
//! - TraceJitConfig::defaults values are all positive (sanity)

use fusevm::chunk::{Chunk, ChunkBuilder};
use fusevm::jit::{DeoptFrame, DeoptInfo, TraceJitConfig};
use fusevm::op::Op;
use fusevm::value::Value;
use fusevm::vm::{VMResult, VM};

#[test]
fn test_op_serde_roundtrip_for_all_eight_fused_superinstructions() {
    let fused_ops = vec![
        Op::PreIncSlot(7),
        Op::SlotLtIntJumpIfFalse(2, 100, 42),
        Op::SlotIncLtIntJumpBack(1, 50, 10),
        Op::AccumSumLoop(0, 1, 100),
        Op::ConcatConstLoop(3, 0, 1, 50),
        Op::PushIntRangeLoop(4, 0, 25),
        Op::AddAssignSlotVoid(2, 3),
        Op::PreIncSlotVoid(5),
    ];
    for op in fused_ops {
        let json = serde_json::to_string(&op).expect("serialize");
        let back: Op = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(op, back, "fused op {op:?} must roundtrip via serde");
    }
}

#[test]
fn test_vm_run_on_empty_chunk_returns_halted() {
    let chunk = ChunkBuilder::new().build();
    let mut vm = VM::new(chunk);
    let result = vm.run();
    assert!(
        matches!(result, VMResult::Halted),
        "empty chunk must return Halted (no panic, no Error); got {result:?}"
    );
}

#[test]
fn test_vm_new_allocates_globals_equal_to_names_length() {
    // Build a chunk with exactly 3 distinct names.
    let mut b = ChunkBuilder::new();
    b.add_name("a");
    b.add_name("b");
    b.add_name("c");
    let chunk = b.build();
    let names_len = chunk.names.len();
    let vm = VM::new(chunk);
    assert_eq!(
        vm.globals.len(),
        names_len,
        "VM::new must allocate exactly chunk.names.len() globals"
    );
    assert_eq!(
        vm.globals.len(),
        3,
        "expected 3 names from 3 add_name calls"
    );
}

#[test]
fn test_chunk_builder_add_constant_returns_sequential_indices() {
    let mut b = ChunkBuilder::new();
    let i0 = b.add_constant(Value::Int(42));
    let i1 = b.add_constant(Value::Int(42)); // same value
    let i2 = b.add_constant(Value::Int(99));
    assert_eq!(i0, 0, "first constant gets index 0");
    assert_eq!(
        i1, 1,
        "constants are NOT deduped; second Int(42) gets a fresh index"
    );
    assert_eq!(i2, 2, "third constant gets sequential index 2");
}

#[test]
fn test_chunk_builder_add_name_dedupes_repeated_names() {
    let mut b = ChunkBuilder::new();
    let i0 = b.add_name("x");
    let i1 = b.add_name("x"); // duplicate
    let i2 = b.add_name("y");
    let i3 = b.add_name("x"); // duplicate again
    assert_eq!(
        i0, i1,
        "second add_name(\"x\") must return same index as first"
    );
    assert_eq!(
        i0, i3,
        "third add_name(\"x\") must return same index as first"
    );
    assert_ne!(i0, i2, "different name must get a different index");
}

#[test]
fn test_deopt_info_zeroed_has_all_counts_zero() {
    let d = DeoptInfo::zeroed();
    assert_eq!(d.resume_ip, 0, "zeroed DeoptInfo resume_ip must be 0");
    assert_eq!(d.frame_count, 0, "zeroed DeoptInfo frame_count must be 0");
    assert_eq!(d.stack_count, 0, "zeroed DeoptInfo stack_count must be 0");
}

#[test]
fn test_deopt_frame_zeroed_has_zero_slot_count_and_return_ip() {
    let f = DeoptFrame::zeroed();
    assert_eq!(f.return_ip, 0, "zeroed DeoptFrame return_ip must be 0");
    assert_eq!(f.slot_count, 0, "zeroed DeoptFrame slot_count must be 0");
}

#[test]
fn test_trace_jit_config_defaults_are_positive_and_sane() {
    let c = TraceJitConfig::defaults();
    assert!(
        c.trace_threshold > 0,
        "trace_threshold must be positive; got {}",
        c.trace_threshold
    );
    assert!(
        c.max_side_exits > 0,
        "max_side_exits must be positive; got {}",
        c.max_side_exits
    );
    assert!(
        c.max_inline_recursion > 0,
        "max_inline_recursion must be positive; got {}",
        c.max_inline_recursion
    );
    assert!(
        c.max_trace_chain > 0,
        "max_trace_chain must be positive; got {}",
        c.max_trace_chain
    );
    assert!(
        c.max_trace_len > 0,
        "max_trace_len must be positive; got {}",
        c.max_trace_len
    );
    // Default must equal defaults() (impl Default uses defaults() body).
    let d: TraceJitConfig = TraceJitConfig::default();
    assert_eq!(d.trace_threshold, c.trace_threshold);
    assert_eq!(d.max_trace_len, c.max_trace_len);
}

/// Smoke check that the public API contract for a built chunk
/// (Chunk::find_sub) returns None for an unregistered name.
#[test]
fn test_chunk_find_sub_returns_none_for_unregistered_name() {
    let chunk: Chunk = ChunkBuilder::new().build();
    let nope = chunk.find_sub(9999);
    assert!(
        nope.is_none(),
        "find_sub on an empty chunk must return None for any name_idx"
    );
}
