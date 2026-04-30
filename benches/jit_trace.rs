//! Tracing JIT speedup benchmarks.
//!
//! Compares interpreter dispatch against the tracing JIT for hot loops.
//! Run with `cargo bench --features jit --bench jit_trace`.
//!
//! The chunks are built with the same shape used in the trace integration
//! tests. Each benchmark runs the loop with N iterations large enough to
//! cross the trace recording threshold, then measures the steady-state
//! cost (trace cache primed, hot path runs in native code).

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use fusevm::{ChunkBuilder, JitCompiler, Op, TraceJitConfig, Value, VM};

fn build_counter_loop(limit: i64) -> (fusevm::Chunk, usize) {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(0), 1);
    b.emit(Op::SetSlot(0), 1);
    let anchor = b.current_pos();
    b.emit(Op::PreIncSlotVoid(0), 1);
    b.emit(Op::GetSlot(0), 1);
    b.emit(Op::LoadInt(limit), 1);
    b.emit(Op::NumLt, 1);
    let jmp = b.emit(Op::JumpIfTrue(0), 1);
    b.patch_jump(jmp, anchor);
    b.emit(Op::GetSlot(0), 1);
    (b.build(), anchor)
}

fn build_loop_with_branch(limit: i64) -> (fusevm::Chunk, usize) {
    // for i in 1..=limit { if i > 0 { /* always */ } }
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(0), 1);
    b.emit(Op::SetSlot(0), 1);
    let anchor = b.current_pos();
    b.emit(Op::PreIncSlotVoid(0), 1);
    b.emit(Op::GetSlot(0), 1);
    b.emit(Op::LoadInt(0), 1);
    b.emit(Op::NumGt, 1);
    let jif = b.emit(Op::JumpIfFalse(0), 1);
    b.emit(Op::Nop, 1); // then-arm
    let after_if = b.current_pos();
    b.patch_jump(jif, after_if);
    b.emit(Op::GetSlot(0), 1);
    b.emit(Op::LoadInt(limit), 1);
    b.emit(Op::NumLt, 1);
    let jmp = b.emit(Op::JumpIfTrue(0), 1);
    b.patch_jump(jmp, anchor);
    b.emit(Op::GetSlot(0), 1);
    (b.build(), anchor)
}

fn ensure_slots(vm: &mut VM, n: usize) {
    let frame = vm.frames.last_mut().unwrap();
    while frame.slots.len() < n {
        frame.slots.push(Value::Int(0));
    }
}

fn bench_counter_loop(c: &mut Criterion) {
    let mut group = c.benchmark_group("counter_loop");

    for &n in &[1_000i64, 10_000, 100_000] {
        let (chunk, _) = build_counter_loop(n);

        // Interpreter baseline.
        group.bench_with_input(BenchmarkId::new("interp", n), &n, |b, _| {
            b.iter(|| {
                let mut vm = VM::new(chunk.clone());
                ensure_slots(&mut vm, 1);
                let _ = black_box(vm.run());
            });
        });

        // Block JIT — direct fn-ptr invocation through `try_run_block`.
        // Warm cache once via `try_run_block_eager`. The bench measures
        // steady-state native execution; chunk's loop runs entirely in
        // compiled code with no VM dispatch on the hot path.
        group.bench_with_input(BenchmarkId::new("block_jit", n), &n, |b, _| {
            let jit = JitCompiler::new();
            if !jit.is_block_eligible(&chunk) {
                return;
            }
            let mut warm = vec![0i64; 4];
            let _ = jit.try_run_block_eager(&chunk, &mut warm);
            b.iter(|| {
                let mut slots = vec![0i64; 4];
                let r = jit.try_run_block(black_box(&chunk), &mut slots);
                black_box(r);
            });
        });

        // Tracing JIT — dispatched through VM at backward branches.
        // Warm up the cache once. Steady-state run goes: VM startup +
        // a few interpreter ops + native trace + VM teardown.
        group.bench_with_input(BenchmarkId::new("trace_jit", n), &n, |b, _| {
            let cfg = TraceJitConfig {
                trace_threshold: 5,
                ..TraceJitConfig::defaults()
            };
            {
                let warm = JitCompiler::new();
                warm.set_config(cfg);
                let mut vm = VM::new(chunk.clone());
                vm.enable_tracing_jit();
                ensure_slots(&mut vm, 1);
                let _ = vm.run();
            }
            b.iter(|| {
                let mut vm = VM::new(chunk.clone());
                vm.enable_tracing_jit();
                ensure_slots(&mut vm, 1);
                let _ = black_box(vm.run());
            });
        });
    }
    group.finish();
}

fn bench_loop_with_branch(c: &mut Criterion) {
    let mut group = c.benchmark_group("loop_with_branch");

    for &n in &[1_000i64, 10_000, 100_000] {
        let (chunk, _) = build_loop_with_branch(n);

        group.bench_with_input(BenchmarkId::new("interp", n), &n, |b, _| {
            b.iter(|| {
                let mut vm = VM::new(chunk.clone());
                ensure_slots(&mut vm, 1);
                let _ = black_box(vm.run());
            });
        });

        // Block JIT — works on the same chunk via direct fn-ptr.
        group.bench_with_input(BenchmarkId::new("block_jit", n), &n, |b, _| {
            let jit = JitCompiler::new();
            if !jit.is_block_eligible(&chunk) {
                return;
            }
            let mut warm = vec![0i64; 4];
            let _ = jit.try_run_block_eager(&chunk, &mut warm);
            b.iter(|| {
                let mut slots = vec![0i64; 4];
                let r = jit.try_run_block(black_box(&chunk), &mut slots);
                black_box(r);
            });
        });

        group.bench_with_input(BenchmarkId::new("trace_jit", n), &n, |b, _| {
            let cfg = TraceJitConfig {
                trace_threshold: 5,
                ..TraceJitConfig::defaults()
            };
            {
                let warm = JitCompiler::new();
                warm.set_config(cfg);
                let mut vm = VM::new(chunk.clone());
                vm.enable_tracing_jit();
                ensure_slots(&mut vm, 1);
                let _ = vm.run();
            }
            b.iter(|| {
                let mut vm = VM::new(chunk.clone());
                vm.enable_tracing_jit();
                ensure_slots(&mut vm, 1);
                let _ = black_box(vm.run());
            });
        });
    }
    group.finish();
}

fn build_simple_expr_chunk() -> fusevm::Chunk {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(40), 1);
    b.emit(Op::LoadInt(2), 1);
    b.emit(Op::Add, 1);
    b.build()
}

fn bench_vm_pool_vs_fresh(c: &mut Criterion) {
    let mut group = c.benchmark_group("vm_pool");
    let chunk = build_simple_expr_chunk();

    // Fresh VM per call — measures VM::new() overhead.
    group.bench_function("fresh_vm_per_call", |b| {
        b.iter(|| {
            let mut vm = VM::new(chunk.clone());
            let r = vm.run();
            black_box(r);
        });
    });

    // Pooled VM — measures the steady-state cost after the first acquire
    // has populated the pool.
    group.bench_function("pooled_vm_per_call", |b| {
        let mut pool = fusevm::VMPool::new();
        // Warm the pool with one acquire/release cycle.
        let v = pool.acquire(chunk.clone());
        pool.release(v);
        b.iter(|| {
            let mut vm = pool.acquire(chunk.clone());
            let r = vm.run();
            black_box(r);
            pool.release(vm);
        });
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_counter_loop,
    bench_loop_with_branch,
    bench_vm_pool_vs_fresh,
);
criterion_main!(benches);
