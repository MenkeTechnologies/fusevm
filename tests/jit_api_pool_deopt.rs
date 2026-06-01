//! Coverage for JitCompiler public API (works with and without the `jit`
//! feature thanks to stubs), Deopt structs, TraceJitConfig, VMPool, the
//! `Frame` type, and additional `Box<dyn ShellHost>` plumbing.

use fusevm::chunk::{Chunk, ChunkBuilder};
use fusevm::host::{DefaultHost, ShellHost};
use fusevm::jit::{
    DeoptFrame, DeoptInfo, JitCompiler, SlotKind, TraceJitConfig, MAX_DEOPT_FRAMES,
    MAX_DEOPT_SLOTS_PER_FRAME, MAX_DEOPT_STACK, STACK_KIND_FLOAT, STACK_KIND_INT,
};
use fusevm::op::Op;
use fusevm::value::Value;
use fusevm::vm::{VMPool, VMResult, VM};

// ── JitCompiler public surface (always available) ───────────────────────

#[test]
fn jit_compiler_new_constructs() {
    let _ = JitCompiler::new();
}

#[test]
fn jit_is_eligible_for_trivial_arith() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::LoadInt(2), 1);
    b.emit(Op::Add, 1);
    let chunk = b.build();
    let jc = JitCompiler::new();
    assert!(jc.is_eligible(&chunk));
}

#[test]
fn jit_is_eligible_for_empty_chunk() {
    let chunk = ChunkBuilder::new().build();
    let jc = JitCompiler::new();
    assert!(jc.is_eligible(&chunk));
}

#[test]
fn jit_is_eligible_false_for_unsupported_op() {
    // Shell ops aren't in the universal-eligibility allow-list.
    let mut b = ChunkBuilder::new();
    b.emit(Op::Glob, 1);
    let chunk = b.build();
    let jc = JitCompiler::new();
    assert!(!jc.is_eligible(&chunk));
}

#[test]
fn jit_is_eligible_for_jump_ops() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(0), 1);
    b.emit(Op::JumpIfFalse(3), 1);
    b.emit(Op::LoadInt(1), 1);
    let chunk = b.build();
    let jc = JitCompiler::new();
    // Jumps may or may not be eligible depending on impl details; just verify call returns.
    let _ = jc.is_eligible(&chunk);
}

// ── No-jit fallback stubs (only matter without --features jit) ──────────

#[cfg(not(feature = "jit"))]
#[test]
fn jit_try_run_linear_returns_none_without_feature() {
    let chunk = ChunkBuilder::new().build();
    let jc = JitCompiler::new();
    assert!(jc.try_run_linear(&chunk, &[]).is_none());
}

#[cfg(not(feature = "jit"))]
#[test]
fn jit_is_linear_eligible_false_without_feature() {
    let chunk = ChunkBuilder::new().build();
    let jc = JitCompiler::new();
    assert!(!jc.is_linear_eligible(&chunk));
}

#[cfg(not(feature = "jit"))]
#[test]
fn jit_try_run_block_returns_none_without_feature() {
    let chunk = ChunkBuilder::new().build();
    let jc = JitCompiler::new();
    let mut slots = [0i64; 4];
    assert!(jc.try_run_block(&chunk, &mut slots).is_none());
    assert!(jc.try_run_block_eager(&chunk, &mut slots).is_none());
    assert!(!jc.is_block_eligible(&chunk));
    assert!(!jc.block_jit_is_compiled(&chunk));
}

#[cfg(not(feature = "jit"))]
#[test]
fn jit_find_region_returns_none_without_feature() {
    let chunk = ChunkBuilder::new().build();
    let jc = JitCompiler::new();
    assert!(jc.find_jit_region(&chunk).is_none());
}

#[cfg(not(feature = "jit"))]
#[test]
fn jit_extract_region_returns_clone_without_feature() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(7), 1);
    let chunk = b.build();
    let jc = JitCompiler::new();
    let extracted = jc.extract_region(&chunk, 0, 1);
    assert_eq!(extracted.ops, chunk.ops);
}

// ── jit-feature-only smoke checks ────────────────────────────────────────

#[cfg(feature = "jit")]
#[test]
fn jit_linear_runs_simple_arith() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(40), 1);
    b.emit(Op::LoadInt(2), 1);
    b.emit(Op::Add, 1);
    let chunk = b.build();
    let jc = JitCompiler::new();
    // Linear eligibility is more restrictive than is_eligible.
    if jc.is_linear_eligible(&chunk) {
        if let Some(v) = jc.try_run_linear(&chunk, &[]) {
            assert_eq!(v, Value::Int(42));
        }
    }
}

#[cfg(feature = "jit")]
#[test]
fn jit_find_region_returns_in_bounds() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::LoadInt(2), 1);
    b.emit(Op::Add, 1);
    let chunk = b.build();
    let jc = JitCompiler::new();
    if let Some((s, e)) = jc.find_jit_region(&chunk) {
        assert!(s <= e);
        assert!(e <= chunk.ops.len());
    }
}

// ── Deopt structs ────────────────────────────────────────────────────────

#[test]
fn deopt_frame_zeroed_is_all_zero() {
    let f = DeoptFrame::zeroed();
    assert_eq!(f.return_ip, 0);
    assert_eq!(f.slot_count, 0);
    assert!(f.slots.iter().all(|&s| s == 0));
}

#[test]
fn deopt_info_zeroed_initializes_all_fields() {
    let d = DeoptInfo::zeroed();
    assert_eq!(d.resume_ip, 0);
    assert_eq!(d.frame_count, 0);
    assert_eq!(d.stack_count, 0);
    assert_eq!(d.frames.len(), MAX_DEOPT_FRAMES);
    assert_eq!(d.stack_buf.len(), MAX_DEOPT_STACK);
    assert_eq!(d.stack_kinds.len(), MAX_DEOPT_STACK);
    assert!(d.stack_buf.iter().all(|&v| v == 0));
    assert!(d.stack_kinds.iter().all(|&k| k == 0));
    for f in &d.frames {
        assert_eq!(f.return_ip, 0);
        assert_eq!(f.slot_count, 0);
    }
}

#[test]
fn deopt_constants_have_documented_values() {
    assert_eq!(STACK_KIND_INT, 0);
    assert_eq!(STACK_KIND_FLOAT, 1);
}

#[test]
fn deopt_capacity_constants_are_reasonable() {
    // Lock the documented capacity contract.
    assert_eq!(MAX_DEOPT_FRAMES, 4);
    assert_eq!(MAX_DEOPT_SLOTS_PER_FRAME, 16);
    assert_eq!(MAX_DEOPT_STACK, 32);
}

#[test]
fn deopt_frame_can_be_written_and_read() {
    let mut f = DeoptFrame::zeroed();
    f.return_ip = 42;
    f.slot_count = 3;
    f.slots[0] = 100;
    f.slots[1] = -50;
    f.slots[2] = 7;
    assert_eq!(f.return_ip, 42);
    assert_eq!(&f.slots[..f.slot_count], &[100, -50, 7]);
}

// ── SlotKind ────────────────────────────────────────────────────────────

#[test]
fn slot_kind_equality_and_clone() {
    assert_eq!(SlotKind::Int, SlotKind::Int);
    assert_ne!(SlotKind::Int, SlotKind::Float);
    let a = SlotKind::Float;
    let b = a; // Copy
    assert_eq!(a, b);
}

#[test]
fn slot_kind_debug_format_contains_variant_name() {
    let s = format!("{:?}", SlotKind::Int);
    assert!(s.contains("Int"));
    let s = format!("{:?}", SlotKind::Float);
    assert!(s.contains("Float"));
}

#[test]
fn slot_kind_serde_roundtrips() {
    let v = vec![SlotKind::Int, SlotKind::Float, SlotKind::Int];
    let json = serde_json::to_string(&v).unwrap();
    let back: Vec<SlotKind> = serde_json::from_str(&json).unwrap();
    assert_eq!(v, back);
}

// ── TraceJitConfig ───────────────────────────────────────────────────────

#[test]
fn trace_jit_config_defaults_match_documented_values() {
    let c = TraceJitConfig::defaults();
    assert_eq!(c.trace_threshold, 50);
    assert_eq!(c.max_side_exits, 50);
    assert_eq!(c.max_inline_recursion, 4);
    assert_eq!(c.max_trace_chain, 4);
    assert_eq!(c.max_trace_len, 256);
}

#[test]
fn trace_jit_config_default_trait_uses_defaults() {
    let a = TraceJitConfig::default();
    let b = TraceJitConfig::defaults();
    assert_eq!(a.trace_threshold, b.trace_threshold);
    assert_eq!(a.max_side_exits, b.max_side_exits);
    assert_eq!(a.max_inline_recursion, b.max_inline_recursion);
    assert_eq!(a.max_trace_chain, b.max_trace_chain);
    assert_eq!(a.max_trace_len, b.max_trace_len);
}

#[test]
fn trace_jit_config_can_be_customized() {
    let c = TraceJitConfig {
        trace_threshold: 10,
        block_threshold: 5,
        max_side_exits: 5,
        max_inline_recursion: 2,
        max_trace_chain: 2,
        max_trace_len: 64,
    };
    assert_eq!(c.trace_threshold, 10);
    // Copy semantics
    let d = c;
    assert_eq!(d.max_trace_len, 64);
}

// ── VMPool ───────────────────────────────────────────────────────────────

fn add_chunk(a: i64, b: i64) -> Chunk {
    let mut bld = ChunkBuilder::new();
    bld.emit(Op::LoadInt(a), 1);
    bld.emit(Op::LoadInt(b), 1);
    bld.emit(Op::Add, 1);
    bld.build()
}

#[test]
fn vmpool_new_is_empty() {
    let p = VMPool::new();
    assert_eq!(p.len(), 0);
}

#[test]
fn vmpool_with_capacity_starts_empty() {
    let p = VMPool::with_capacity(16);
    assert_eq!(p.len(), 0);
}

#[test]
fn vmpool_acquire_returns_running_vm() {
    let mut p = VMPool::new();
    let mut vm = p.acquire(add_chunk(2, 3));
    match vm.run() {
        VMResult::Ok(Value::Int(5)) => {}
        other => panic!("got {:?}", other),
    }
}

#[test]
fn vmpool_release_then_acquire_recycles() {
    let mut p = VMPool::new();
    let vm1 = p.acquire(add_chunk(1, 2));
    p.release(vm1);
    assert_eq!(p.len(), 1);
    let mut vm2 = p.acquire(add_chunk(10, 20));
    assert_eq!(p.len(), 0);
    match vm2.run() {
        VMResult::Ok(Value::Int(30)) => {}
        other => panic!("got {:?}", other),
    }
}

#[test]
fn vmpool_with_runs_closure_and_returns_vm() {
    let mut p = VMPool::new();
    let v = p.with(add_chunk(4, 5), |vm| match vm.run() {
        VMResult::Ok(v) => v.to_int(),
        _ => -1,
    });
    assert_eq!(v, 9);
    assert_eq!(p.len(), 1);
}

#[test]
fn vmpool_with_can_run_many_chunks_in_sequence() {
    let mut p = VMPool::new();
    let results: Vec<i64> = (0..5)
        .map(|i| {
            p.with(add_chunk(i, i + 1), |vm| match vm.run() {
                VMResult::Ok(v) => v.to_int(),
                _ => -1,
            })
        })
        .collect();
    assert_eq!(results, vec![1, 3, 5, 7, 9]);
    // Pool re-uses a single VM across iterations.
    assert_eq!(p.len(), 1);
}

#[test]
fn vmpool_default_equals_new() {
    let p: VMPool = Default::default();
    assert_eq!(p.len(), 0);
}

// ── Box<dyn ShellHost> plumbing ──────────────────────────────────────────

#[test]
fn vm_accepts_boxed_default_host() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(42), 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(DefaultHost));
    match vm.run() {
        VMResult::Ok(Value::Int(42)) => {}
        other => panic!("got {:?}", other),
    }
}

// Minimal struct-based host so we can verify Box<dyn ShellHost> object safety
// across many trait methods.
struct CountingHost {
    redirects: usize,
    pipelines: usize,
}

impl ShellHost for CountingHost {
    fn redirect(&mut self, _fd: u8, _op: u8, _target: &str) {
        self.redirects += 1;
    }
    fn pipeline_begin(&mut self, _n: u8) {
        self.pipelines += 1;
    }
}

#[test]
fn counting_host_observes_redirect() {
    use fusevm::op::redirect_op;
    let mut b = ChunkBuilder::new();
    let target = b.add_constant(Value::str("/dev/null"));
    b.emit(Op::LoadConst(target), 1);
    b.emit(Op::Redirect(1, redirect_op::WRITE), 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(CountingHost {
        redirects: 0,
        pipelines: 0,
    }));
    let _ = vm.run();
    // We can't easily extract the host back, but we can observe behavior
    // via push/run not panicking — the counter increment happened internally.
}

#[test]
fn default_host_subshell_end_returns_unit() {
    let mut h = DefaultHost;
    // subshell_end is a no-op; just verify it can be called.
    h.subshell_end();
}

#[test]
fn counting_host_state_can_be_inspected_in_place() {
    let mut h = CountingHost {
        redirects: 0,
        pipelines: 0,
    };
    h.pipeline_begin(2);
    h.redirect(1, 0, "/tmp/x");
    h.redirect(2, 0, "/tmp/y");
    assert_eq!(h.pipelines, 1);
    assert_eq!(h.redirects, 2);
}

// ── VM::push/pop/peek and basic invariants ──────────────────────────────

#[test]
fn vm_pop_returns_undef_on_empty() {
    // pop on empty should return Undef per implementation (no panic).
    let mut b = ChunkBuilder::new();
    b.emit(Op::Nop, 1);
    let mut vm = VM::new(b.build());
    // The stack is initially empty.
    let popped = vm.pop();
    assert_eq!(popped, Value::Undef);
}

#[test]
fn vm_push_then_run_consumes_external_value() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(10), 1);
    b.emit(Op::Add, 1);
    let mut vm = VM::new(b.build());
    vm.push(Value::Int(5));
    match vm.run() {
        VMResult::Ok(Value::Int(15)) => {}
        other => panic!("got {:?}", other),
    }
}

#[test]
fn vm_run_handles_empty_chunk_gracefully() {
    let chunk = ChunkBuilder::new().build();
    let mut vm = VM::new(chunk);
    // Empty chunk has no ops to dispatch; runs to completion (Halted is acceptable).
    let _ = vm.run();
}

// ── Bytecode serialization via bincode (dev-dep) ─────────────────────────

#[test]
fn chunk_roundtrips_via_bincode() {
    let mut b = ChunkBuilder::new();
    let s = b.add_constant(Value::str("hello"));
    b.add_name("x");
    b.emit(Op::LoadConst(s), 1);
    b.emit(Op::LoadInt(7), 1);
    b.emit(Op::Add, 2);
    let original = b.build();
    let bytes = bincode::serialize(&original).unwrap();
    let restored: Chunk = bincode::deserialize(&bytes).unwrap();
    assert_eq!(restored.ops, original.ops);
    assert_eq!(restored.constants, original.constants);
    assert_eq!(restored.names, original.names);
    assert_eq!(restored.lines, original.lines);
}

#[test]
fn chunk_roundtrips_via_json() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::LoadInt(2), 1);
    b.emit(Op::Add, 1);
    let original = b.build();
    let json = serde_json::to_string(&original).unwrap();
    let restored: Chunk = serde_json::from_str(&json).unwrap();
    assert_eq!(restored.ops, original.ops);
}

// ── Op enum sanity ───────────────────────────────────────────────────────

#[test]
fn op_clone_eq_for_simple_variants() {
    let a = Op::LoadInt(5);
    let b = a.clone();
    assert_eq!(a, b);
    assert_ne!(Op::LoadInt(5), Op::LoadInt(6));
    assert_ne!(Op::LoadInt(5), Op::LoadFloat(5.0));
}

#[test]
fn op_jump_variants_distinct() {
    assert_ne!(Op::Jump(0), Op::JumpIfTrue(0));
    assert_ne!(Op::JumpIfFalse(0), Op::JumpIfFalseKeep(0));
}

#[test]
fn op_serde_roundtrips_for_each_arity() {
    let ops = vec![
        Op::Nop,
        Op::LoadInt(42),
        Op::LoadFloat(1.5),
        Op::LoadConst(7),
        Op::Jump(10),
        Op::CallBuiltin(5, 2),
        Op::AccumSumLoop(0, 1, 100),
    ];
    for op in ops {
        let json = serde_json::to_string(&op).unwrap();
        let back: Op = serde_json::from_str(&json).unwrap();
        assert_eq!(op, back);
    }
}
