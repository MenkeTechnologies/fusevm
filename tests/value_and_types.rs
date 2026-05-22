//! Additional unit tests for public types: Value coercions/truthiness on
//! all variants, ChunkBuilder constant/name dedup behaviour, JIT type
//! defaults (no `jit` feature needed — these structs are always exported),
//! and Op serde round-trips for fused superinstructions.

use fusevm::jit::{
    DeoptFrame, DeoptInfo, SlotKind, TraceJitConfig, TraceMetadata, MAX_DEOPT_FRAMES,
    MAX_DEOPT_SLOTS_PER_FRAME, MAX_DEOPT_STACK, STACK_KIND_FLOAT, STACK_KIND_INT,
};
use fusevm::{ChunkBuilder, Op, Value};
use std::collections::HashMap;

// ── Value: is_truthy edge cases ─────────────────────────────────────────────

#[test]
fn value_truthy_string_zero_is_false() {
    // Bash semantics: literal "0" is falsy.
    assert!(!Value::str("0").is_truthy());
}

#[test]
fn value_truthy_non_empty_non_zero_string_is_true() {
    assert!(Value::str("hello").is_truthy());
    assert!(Value::str("00").is_truthy()); // not exactly "0"
    assert!(Value::str(" ").is_truthy());
}

#[test]
fn value_truthy_status_zero_is_true_nonzero_is_false() {
    // Shell exit status: 0 = success = true.
    assert!(Value::status(0).is_truthy());
    assert!(!Value::status(1).is_truthy());
    assert!(!Value::status(127).is_truthy());
}

#[test]
fn value_truthy_undef_is_false() {
    assert!(!Value::Undef.is_truthy());
}

#[test]
fn value_truthy_native_fn_is_true() {
    assert!(Value::NativeFn(0).is_truthy());
    assert!(Value::NativeFn(42).is_truthy());
}

#[test]
fn value_truthy_float_zero_and_neg_zero_are_false() {
    assert!(!Value::float(0.0).is_truthy());
    assert!(!Value::float(-0.0).is_truthy());
    assert!(Value::float(0.1).is_truthy());
}

// ── Value::to_int / to_float on non-standard variants ───────────────────────

#[test]
fn to_int_array_returns_length() {
    let arr = Value::array(vec![Value::int(1), Value::int(2), Value::int(3)]);
    assert_eq!(arr.to_int(), 3);
}

#[test]
fn to_int_hash_returns_zero() {
    let mut m = HashMap::new();
    m.insert("k".into(), Value::int(99));
    assert_eq!(Value::hash(m).to_int(), 0);
}

#[test]
fn to_int_undef_native_fn_ref_default_zero() {
    assert_eq!(Value::Undef.to_int(), 0);
    assert_eq!(Value::NativeFn(7).to_int(), 0);
}

#[test]
fn to_int_status_passes_through_as_i64() {
    assert_eq!(Value::status(127).to_int(), 127);
    assert_eq!(Value::status(-1).to_int(), -1);
}

#[test]
fn to_float_bool_true_is_one_false_is_zero() {
    assert_eq!(Value::bool(true).to_float(), 1.0);
    assert_eq!(Value::bool(false).to_float(), 0.0);
}

#[test]
fn to_float_unparseable_string_is_zero() {
    assert_eq!(Value::str("not a number").to_float(), 0.0);
}

#[test]
fn to_float_status_passes_through() {
    assert_eq!(Value::status(42).to_float(), 42.0);
}

// ── Value::len / is_empty on all variants ───────────────────────────────────

#[test]
fn len_of_str_uses_byte_length() {
    assert_eq!(Value::str("hello").len(), 5);
    assert_eq!(Value::str("").len(), 0);
}

#[test]
fn len_of_array_uses_element_count() {
    assert_eq!(Value::array(vec![Value::int(1); 10]).len(), 10);
}

#[test]
fn len_of_hash_uses_entry_count() {
    let mut m = HashMap::new();
    m.insert("a".into(), Value::int(1));
    m.insert("b".into(), Value::int(2));
    assert_eq!(Value::hash(m).len(), 2);
}

#[test]
fn len_of_int_uses_string_length() {
    // Int doesn't have a natural len; falls back to to_str().len()
    assert_eq!(Value::int(12345).len(), 5);
    assert_eq!(Value::int(-7).len(), 2); // "-7"
}

#[test]
fn is_empty_consistent_with_len() {
    assert!(Value::str("").is_empty());
    assert!(Value::array(vec![]).is_empty());
    assert!(Value::hash(HashMap::new()).is_empty());
    assert!(!Value::int(0).is_empty()); // "0" has length 1
    assert!(!Value::str("a").is_empty());
}

// ── Value::as_str_cow Cow variants ──────────────────────────────────────────

#[test]
fn as_str_cow_borrows_for_undef_and_bool() {
    use std::borrow::Cow;
    assert!(matches!(Value::Undef.as_str_cow(), Cow::Borrowed("")));
    assert!(matches!(Value::bool(true).as_str_cow(), Cow::Borrowed("1")));
    assert!(matches!(Value::bool(false).as_str_cow(), Cow::Borrowed("")));
}

#[test]
fn as_str_cow_borrows_for_hash_marker() {
    use std::borrow::Cow;
    let h = Value::hash(HashMap::new());
    assert!(matches!(h.as_str_cow(), Cow::Borrowed("(hash)")));
}

#[test]
fn as_str_cow_for_native_fn_includes_id() {
    let s = Value::NativeFn(123).to_str();
    assert!(s.contains("123"));
    assert!(s.starts_with("("));
}

#[test]
fn as_str_cow_for_array_joins_with_space() {
    let arr = Value::array(vec![Value::int(1), Value::int(2), Value::int(3)]);
    assert_eq!(arr.to_str(), "1 2 3");
}

// ── Value default and equality ──────────────────────────────────────────────

#[test]
fn value_default_is_undef() {
    let v: Value = Default::default();
    assert!(matches!(v, Value::Undef));
}

#[test]
fn value_eq_across_int_and_float_is_false() {
    // PartialEq derived → Int(1) != Float(1.0) at the type level
    assert_ne!(Value::int(1), Value::float(1.0));
    assert_eq!(Value::int(1), Value::int(1));
    assert_eq!(Value::float(1.5), Value::float(1.5));
}

// ── ChunkBuilder constant pool: no dedup, name pool dedup ───────────────────

#[test]
fn constant_pool_does_not_dedupe_identical_values() {
    let mut b = ChunkBuilder::new();
    let a = b.add_constant(Value::int(7));
    let c = b.add_constant(Value::int(7));
    assert_ne!(a, c, "constants should not be deduplicated");
}

#[test]
fn name_pool_dedupes_repeated_names() {
    let mut b = ChunkBuilder::new();
    let a = b.add_name("foo");
    let c = b.add_name("foo");
    assert_eq!(a, c, "name pool should reuse existing entries");
}

#[test]
fn name_pool_indices_are_distinct_for_distinct_names() {
    let mut b = ChunkBuilder::new();
    let a = b.add_name("alpha");
    let c = b.add_name("beta");
    assert_ne!(a, c);
}

#[test]
fn current_pos_grows_with_each_emit() {
    let mut b = ChunkBuilder::new();
    assert_eq!(b.current_pos(), 0);
    b.emit(Op::Nop, 1);
    assert_eq!(b.current_pos(), 1);
    b.emit(Op::LoadInt(0), 1);
    assert_eq!(b.current_pos(), 2);
}

#[test]
fn emit_returns_the_index_of_inserted_op() {
    let mut b = ChunkBuilder::new();
    assert_eq!(b.emit(Op::Nop, 1), 0);
    assert_eq!(b.emit(Op::Nop, 1), 1);
    assert_eq!(b.emit(Op::Nop, 1), 2);
}

// ── Op serde round-trips for fused ops ──────────────────────────────────────

fn serde_round_trip(op: Op) -> Op {
    let json = serde_json::to_string(&op).expect("serialize");
    serde_json::from_str(&json).expect("deserialize")
}

#[test]
fn serde_roundtrip_accum_sum_loop() {
    let op = Op::AccumSumLoop(1, 2, 100);
    assert_eq!(serde_round_trip(op.clone()), op);
}

#[test]
fn serde_roundtrip_concat_const_loop() {
    let op = Op::ConcatConstLoop(7, 1, 2, 3);
    assert_eq!(serde_round_trip(op.clone()), op);
}

#[test]
fn serde_roundtrip_push_int_range_loop() {
    let op = Op::PushIntRangeLoop(4, 5, 10);
    assert_eq!(serde_round_trip(op.clone()), op);
}

#[test]
fn serde_roundtrip_slot_lt_int_jump_if_false() {
    let op = Op::SlotLtIntJumpIfFalse(0, 5, 99);
    assert_eq!(serde_round_trip(op.clone()), op);
}

#[test]
fn serde_roundtrip_slot_inc_lt_int_jump_back() {
    let op = Op::SlotIncLtIntJumpBack(2, 10, 3);
    assert_eq!(serde_round_trip(op.clone()), op);
}

#[test]
fn serde_roundtrip_extended_wide() {
    let op = Op::ExtendedWide(13, 0xDEADBEEF);
    assert_eq!(serde_round_trip(op.clone()), op);
}

#[test]
fn serde_roundtrip_call_and_callbuiltin() {
    let a = Op::Call(7, 3);
    let b = Op::CallBuiltin(2, 1);
    assert_eq!(serde_round_trip(a.clone()), a);
    assert_eq!(serde_round_trip(b.clone()), b);
}

// ── JIT public type defaults (compile w/ or w/o jit feature) ────────────────

#[test]
fn trace_jit_config_defaults_match_constants() {
    let c = TraceJitConfig::default();
    let d = TraceJitConfig::defaults();
    assert_eq!(c.trace_threshold, d.trace_threshold);
    assert_eq!(c.max_side_exits, d.max_side_exits);
    assert_eq!(c.max_inline_recursion, d.max_inline_recursion);
    assert_eq!(c.max_trace_chain, d.max_trace_chain);
    assert_eq!(c.max_trace_len, d.max_trace_len);
}

#[test]
fn trace_jit_config_defaults_are_documented_values() {
    let d = TraceJitConfig::defaults();
    assert_eq!(d.trace_threshold, 50);
    assert_eq!(d.max_side_exits, 50);
    assert_eq!(d.max_inline_recursion, 4);
    assert_eq!(d.max_trace_chain, 4);
    assert_eq!(d.max_trace_len, 256);
}

#[test]
fn deopt_frame_zeroed_is_all_zeros() {
    let f = DeoptFrame::zeroed();
    assert_eq!(f.return_ip, 0);
    assert_eq!(f.slot_count, 0);
    assert!(f.slots.iter().all(|&s| s == 0));
}

#[test]
fn deopt_constants_have_expected_values() {
    assert_eq!(MAX_DEOPT_FRAMES, 4);
    assert_eq!(MAX_DEOPT_SLOTS_PER_FRAME, 16);
    assert_eq!(MAX_DEOPT_STACK, 32);
    assert_eq!(STACK_KIND_INT, 0);
    assert_eq!(STACK_KIND_FLOAT, 1);
    assert_ne!(STACK_KIND_INT, STACK_KIND_FLOAT);
}

#[test]
fn slot_kind_eq_and_distinct() {
    assert_eq!(SlotKind::Int, SlotKind::Int);
    assert_ne!(SlotKind::Int, SlotKind::Float);
}

#[test]
fn slot_kind_serde_roundtrip() {
    let j = serde_json::to_string(&SlotKind::Float).unwrap();
    let back: SlotKind = serde_json::from_str(&j).unwrap();
    assert_eq!(back, SlotKind::Float);
}

#[test]
fn trace_metadata_serde_roundtrip() {
    let md = TraceMetadata {
        chunk_op_hash: 0xdeadbeef,
        anchor_ip: 7,
        fallthrough_ip: 12,
        ops: vec![Op::LoadInt(1), Op::LoadInt(2), Op::Add],
        recorded_ips: vec![3, 4, 5],
        slot_kinds_at_anchor: vec![SlotKind::Int, SlotKind::Float],
    };
    let j = serde_json::to_string(&md).expect("ser");
    let back: TraceMetadata = serde_json::from_str(&j).expect("de");
    assert_eq!(back.chunk_op_hash, md.chunk_op_hash);
    assert_eq!(back.anchor_ip, md.anchor_ip);
    assert_eq!(back.fallthrough_ip, md.fallthrough_ip);
    assert_eq!(back.ops, md.ops);
    assert_eq!(back.recorded_ips, md.recorded_ips);
    assert_eq!(back.slot_kinds_at_anchor, md.slot_kinds_at_anchor);
}

#[test]
fn deopt_info_zero_init_via_struct_literal() {
    // DeoptInfo is repr(C) and Copy; ensure we can construct one and inspect.
    let info = DeoptInfo {
        resume_ip: 0,
        frame_count: 0,
        stack_count: 0,
        frames: [DeoptFrame::zeroed(); MAX_DEOPT_FRAMES],
        stack_buf: [0; MAX_DEOPT_STACK],
        stack_kinds: [STACK_KIND_INT; MAX_DEOPT_STACK],
    };
    assert_eq!(info.frame_count, 0);
    assert_eq!(info.stack_count, 0);
    assert_eq!(info.stack_buf.len(), MAX_DEOPT_STACK);
}
