//! Find the crossover point where JIT beats interpreter.
//!
//! Run: cargo bench --features jit --bench jit_crossover

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use fusevm::{ChunkBuilder, JitCompiler, Op, VM};

fn build_int_add_chain(n: usize) -> fusevm::Chunk {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(0), 1);
    for _ in 0..n {
        b.emit(Op::LoadInt(1), 1);
        b.emit(Op::Add, 1);
    }
    b.build()
}

fn build_bitwise_chain(n: usize) -> fusevm::Chunk {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(0x_DEAD_BEEF), 1);
    for _ in 0..n {
        b.emit(Op::LoadInt(0xFF), 1);
        b.emit(Op::BitAnd, 1);
        b.emit(Op::LoadInt(3), 1);
        b.emit(Op::Shl, 1);
        b.emit(Op::LoadInt(0x_1234), 1);
        b.emit(Op::BitXor, 1);
    }
    b.build()
}

fn bench_int_add_crossover(c: &mut Criterion) {
    let jit = JitCompiler::new();
    let mut group = c.benchmark_group("int_add_crossover");

    for &n in &[10, 50, 100, 500, 1000, 5000, 10000] {
        let chunk = build_int_add_chain(n);
        // warm JIT cache
        let _ = jit.try_run_linear(&chunk, &[]);

        group.bench_with_input(BenchmarkId::new("interp", n), &n, |b, _| {
            b.iter(|| {
                let mut vm = VM::new(chunk.clone());
                let r = vm.run();
                black_box(r);
            })
        });
        group.bench_with_input(BenchmarkId::new("jit", n), &n, |b, _| {
            b.iter(|| {
                let r = jit.try_run_linear(black_box(&chunk), &[]);
                black_box(r);
            })
        });
    }
    group.finish();
}

fn bench_bitwise_crossover(c: &mut Criterion) {
    let jit = JitCompiler::new();
    let mut group = c.benchmark_group("bitwise_crossover");

    for &n in &[10, 50, 100, 500, 1000, 5000] {
        let chunk = build_bitwise_chain(n);
        let _ = jit.try_run_linear(&chunk, &[]);

        group.bench_with_input(BenchmarkId::new("interp", n), &n, |b, _| {
            b.iter(|| {
                let mut vm = VM::new(chunk.clone());
                let r = vm.run();
                black_box(r);
            })
        });
        group.bench_with_input(BenchmarkId::new("jit", n), &n, |b, _| {
            b.iter(|| {
                let r = jit.try_run_linear(black_box(&chunk), &[]);
                black_box(r);
            })
        });
    }
    group.finish();
}

/// Isolate JIT overhead: just the cache lookup, no execution
fn bench_jit_overhead(c: &mut Criterion) {
    let jit = JitCompiler::new();
    let mut group = c.benchmark_group("jit_overhead");

    // Tiny chunk: 3 ops
    let tiny = {
        let mut b = ChunkBuilder::new();
        b.emit(Op::LoadInt(1), 1);
        b.emit(Op::LoadInt(2), 1);
        b.emit(Op::Add, 1);
        b.build()
    };
    let _ = jit.try_run_linear(&tiny, &[]);

    group.bench_function("cache_lookup_3ops", |b| {
        b.iter(|| {
            let r = jit.try_run_linear(black_box(&tiny), &[]);
            black_box(r);
        })
    });

    // Medium chunk: 100 ops
    let medium = build_int_add_chain(50);
    let _ = jit.try_run_linear(&medium, &[]);

    group.bench_function("cache_lookup_100ops", |b| {
        b.iter(|| {
            let r = jit.try_run_linear(black_box(&medium), &[]);
            black_box(r);
        })
    });

    // Large chunk: 2000 ops
    let large = build_int_add_chain(1000);
    let _ = jit.try_run_linear(&large, &[]);

    group.bench_function("cache_lookup_2000ops", |b| {
        b.iter(|| {
            let r = jit.try_run_linear(black_box(&large), &[]);
            black_box(r);
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_jit_overhead,
    bench_int_add_crossover,
    bench_bitwise_crossover,
);
criterion_main!(benches);
