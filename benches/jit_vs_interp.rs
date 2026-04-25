//! Interpreter vs Cranelift JIT vs native Rust — same workloads.
//!
//! Run: cargo bench --features jit --bench jit_vs_interp

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use fusevm::{ChunkBuilder, JitCompiler, Op, VM};

fn build_chunk(ops: &[(Op, u32)]) -> fusevm::Chunk {
    let mut b = ChunkBuilder::new();
    for (op, line) in ops {
        b.emit(op.clone(), *line);
    }
    b.build()
}

// ── Workload: chain of 1000 integer adds ──

fn arith_chain_ops() -> Vec<(Op, u32)> {
    let mut ops = Vec::with_capacity(2001);
    ops.push((Op::LoadInt(1), 1));
    for _ in 0..1000 {
        ops.push((Op::LoadInt(1), 1));
        ops.push((Op::Add, 1));
    }
    ops
}

fn native_arith_chain() -> i64 {
    let mut v: i64 = 1;
    for _ in 0..1000 {
        v = v.wrapping_add(1);
    }
    v
}

// ── Workload: mixed arithmetic (add/sub/mul) ──

fn mixed_arith_ops() -> Vec<(Op, u32)> {
    let mut ops = Vec::with_capacity(601);
    ops.push((Op::LoadInt(1), 1));
    for _ in 0..100 {
        ops.push((Op::LoadInt(3), 1));
        ops.push((Op::Mul, 1));
        ops.push((Op::LoadInt(7), 1));
        ops.push((Op::Add, 1));
        ops.push((Op::LoadInt(2), 1));
        ops.push((Op::Sub, 1));
    }
    ops
}

fn native_mixed_arith() -> i64 {
    let mut v: i64 = 1;
    for _ in 0..100 {
        v = v.wrapping_mul(3);
        v = v.wrapping_add(7);
        v = v.wrapping_sub(2);
    }
    v
}

// ── Workload: bitwise ops ──

fn bitwise_ops() -> Vec<(Op, u32)> {
    let mut ops = Vec::with_capacity(601);
    ops.push((Op::LoadInt(0x_DEAD_BEEF), 1));
    for _ in 0..200 {
        ops.push((Op::LoadInt(0xFF), 1));
        ops.push((Op::BitAnd, 1));
        ops.push((Op::LoadInt(3), 1));
        ops.push((Op::Shl, 1));
        ops.push((Op::LoadInt(0x_1234), 1));
        ops.push((Op::BitXor, 1));
    }
    ops
}

fn native_bitwise() -> i64 {
    let mut v: i64 = 0x_DEAD_BEEF;
    for _ in 0..200 {
        v &= 0xFF;
        v <<= 3;
        v ^= 0x_1234;
    }
    v
}

// ── Workload: float arithmetic ──

fn float_arith_ops() -> Vec<(Op, u32)> {
    let mut ops = Vec::with_capacity(601);
    ops.push((Op::LoadFloat(1.0), 1));
    for _ in 0..200 {
        ops.push((Op::LoadFloat(0.1), 1));
        ops.push((Op::Add, 1));
        ops.push((Op::LoadFloat(2.0), 1));
        ops.push((Op::Mul, 1));
    }
    ops
}

fn native_float_arith() -> f64 {
    let mut v: f64 = 1.0;
    for _ in 0..200 {
        v += 0.1;
        v *= 2.0;
    }
    v
}

// ── Workload: comparisons ──

fn comparison_ops() -> Vec<(Op, u32)> {
    let mut ops = Vec::with_capacity(601);
    ops.push((Op::LoadInt(0), 1));
    for i in 0..500 {
        ops.push((Op::LoadInt(i), 1));
        ops.push((Op::NumLt, 1));
    }
    ops
}

fn native_comparison() -> i64 {
    let mut v: i64 = 0;
    for i in 0..500 {
        v = if v < i { 1 } else { 0 };
    }
    v
}

// ── Benchmark runner ──

fn bench_workload(
    c: &mut Criterion,
    name: &str,
    ops: &[(Op, u32)],
    native_fn: fn() -> i64,
) {
    let chunk = build_chunk(ops);
    let jit = JitCompiler::new();
    let eligible = jit.is_linear_eligible(&chunk);

    let mut group = c.benchmark_group(name);

    group.bench_function("interpreter", |b| {
        b.iter(|| {
            let mut vm = VM::new(chunk.clone());
            let result = vm.run();
            black_box(result);
        })
    });

    if eligible {
        // warm the JIT cache
        let _ = jit.try_run_linear(&chunk, &[]);
        group.bench_function("jit_cached", |b| {
            b.iter(|| {
                let result = jit.try_run_linear(black_box(&chunk), &[]);
                black_box(result);
            })
        });
    }

    group.bench_function("native_rust", |b| {
        b.iter(|| {
            let result = native_fn();
            black_box(result);
        })
    });

    group.finish();
}

fn bench_float_workload(c: &mut Criterion) {
    let ops = float_arith_ops();
    let chunk = build_chunk(&ops);
    let jit = JitCompiler::new();
    let eligible = jit.is_linear_eligible(&chunk);

    let mut group = c.benchmark_group("float_arith_200");

    group.bench_function("interpreter", |b| {
        b.iter(|| {
            let mut vm = VM::new(chunk.clone());
            let result = vm.run();
            black_box(result);
        })
    });

    if eligible {
        let _ = jit.try_run_linear(&chunk, &[]);
        group.bench_function("jit_cached", |b| {
            b.iter(|| {
                let result = jit.try_run_linear(black_box(&chunk), &[]);
                black_box(result);
            })
        });
    }

    group.bench_function("native_rust", |b| {
        b.iter(|| {
            let result = native_float_arith();
            black_box(result);
        })
    });

    group.finish();
}

fn bench_int_add(c: &mut Criterion) {
    bench_workload(c, "int_add_1k", &arith_chain_ops(), native_arith_chain);
}

fn bench_mixed(c: &mut Criterion) {
    bench_workload(c, "mixed_arith_100", &mixed_arith_ops(), native_mixed_arith);
}

fn bench_bitwise(c: &mut Criterion) {
    bench_workload(c, "bitwise_200", &bitwise_ops(), native_bitwise);
}

fn bench_cmp(c: &mut Criterion) {
    bench_workload(c, "comparison_500", &comparison_ops(), native_comparison);
}

criterion_group!(
    benches,
    bench_int_add,
    bench_mixed,
    bench_bitwise,
    bench_float_workload,
    bench_cmp,
);
criterion_main!(benches);
