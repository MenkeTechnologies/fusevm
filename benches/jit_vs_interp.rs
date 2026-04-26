//! Interpreter vs Cranelift JIT vs native Rust — same workloads.
//!
//! All workloads use slot-based inputs so the JIT can't constant-fold.
//! This gives honest apples-to-apples numbers.
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

// ── Workload: slot-based arithmetic (not constant-foldable) ──
// slot[0] = input, result = slot[0] + 1 + 1 + 1 ... (N times)

fn slot_add_ops(n: usize) -> Vec<(Op, u32)> {
    let mut ops = Vec::with_capacity(2 * n + 2);
    ops.push((Op::GetSlot(0), 1)); // load input from slot
    for _ in 0..n {
        ops.push((Op::LoadInt(1), 1));
        ops.push((Op::Add, 1));
    }
    ops
}

fn native_slot_add(input: i64, n: usize) -> i64 {
    let mut v = input;
    for _ in 0..n {
        v = v.wrapping_add(1);
    }
    v
}

// ── Workload: slot-based mixed arithmetic ──

fn slot_mixed_arith_ops(n: usize) -> Vec<(Op, u32)> {
    let mut ops = Vec::with_capacity(6 * n + 2);
    ops.push((Op::GetSlot(0), 1));
    for _ in 0..n {
        ops.push((Op::LoadInt(3), 1));
        ops.push((Op::Mul, 1));
        ops.push((Op::LoadInt(7), 1));
        ops.push((Op::Add, 1));
        ops.push((Op::LoadInt(2), 1));
        ops.push((Op::Sub, 1));
    }
    ops
}

fn native_slot_mixed(input: i64, n: usize) -> i64 {
    let mut v = input;
    for _ in 0..n {
        v = v.wrapping_mul(3);
        v = v.wrapping_add(7);
        v = v.wrapping_sub(2);
    }
    v
}

// ── Workload: slot-based bitwise ──

fn slot_bitwise_ops(n: usize) -> Vec<(Op, u32)> {
    let mut ops = Vec::with_capacity(6 * n + 2);
    ops.push((Op::GetSlot(0), 1));
    for _ in 0..n {
        ops.push((Op::LoadInt(0xFF), 1));
        ops.push((Op::BitAnd, 1));
        ops.push((Op::LoadInt(3), 1));
        ops.push((Op::Shl, 1));
        ops.push((Op::LoadInt(0x_1234), 1));
        ops.push((Op::BitXor, 1));
    }
    ops
}

fn native_slot_bitwise(input: i64, n: usize) -> i64 {
    let mut v = input;
    for _ in 0..n {
        v &= 0xFF;
        v <<= 3;
        v ^= 0x_1234;
    }
    v
}

// ── Workload: slot-based float ──

fn slot_float_ops(n: usize) -> Vec<(Op, u32)> {
    let mut ops = Vec::with_capacity(4 * n + 2);
    ops.push((Op::GetSlot(0), 1)); // int slot, promoted to float by Add
    for _ in 0..n {
        ops.push((Op::LoadFloat(0.1), 1));
        ops.push((Op::Add, 1));
        ops.push((Op::LoadFloat(1.001), 1));
        ops.push((Op::Mul, 1));
    }
    ops
}

fn native_slot_float(input: f64, n: usize) -> f64 {
    let mut v = input;
    for _ in 0..n {
        v += 0.1;
        v *= 1.001;
    }
    v
}

// ── Benchmark runner ──

fn bench_slot_workload(
    c: &mut Criterion,
    name: &str,
    ops: &[(Op, u32)],
    slots: &[i64],
    native_fn: fn(i64, usize) -> i64,
    n: usize,
) {
    let chunk = build_chunk(ops);
    let jit = JitCompiler::new();
    let eligible = jit.is_linear_eligible(&chunk);

    let mut group = c.benchmark_group(name);

    group.bench_function("interpreter", |b| {
        b.iter(|| {
            let mut vm = VM::new(chunk.clone());
            // set up frame with slots
            vm.frames.push(fusevm::Frame {
                return_ip: 0,
                stack_base: 0,
                slots: slots.iter().map(|&v| fusevm::Value::Int(v)).collect(),
            });
            let result = vm.run();
            black_box(result);
        })
    });

    if eligible {
        let _ = jit.try_run_linear(&chunk, slots);
        group.bench_function("jit_cached", |b| {
            b.iter(|| {
                let result = jit.try_run_linear(black_box(&chunk), black_box(slots));
                black_box(result);
            })
        });
    }

    group.bench_function("native_rust", |b| {
        let input = slots[0];
        b.iter(|| {
            let result = native_fn(black_box(input), n);
            black_box(result);
        })
    });

    group.finish();
}

fn bench_add_100(c: &mut Criterion) {
    let ops = slot_add_ops(100);
    let slots = [42i64];
    bench_slot_workload(c, "slot_add_100", &ops, &slots, native_slot_add, 100);
}

fn bench_add_1000(c: &mut Criterion) {
    let ops = slot_add_ops(1000);
    let slots = [42i64];
    bench_slot_workload(c, "slot_add_1000", &ops, &slots, native_slot_add, 1000);
}

fn bench_mixed_100(c: &mut Criterion) {
    let ops = slot_mixed_arith_ops(100);
    let slots = [42i64];
    bench_slot_workload(c, "slot_mixed_100", &ops, &slots, native_slot_mixed, 100);
}

fn bench_bitwise_200(c: &mut Criterion) {
    let ops = slot_bitwise_ops(200);
    let slots = [0x_DEAD_BEEFi64];
    bench_slot_workload(
        c,
        "slot_bitwise_200",
        &ops,
        &slots,
        native_slot_bitwise,
        200,
    );
}

fn bench_float_200(c: &mut Criterion) {
    let ops = slot_float_ops(200);
    let chunk = build_chunk(&ops);
    let jit = JitCompiler::new();
    let eligible = jit.is_linear_eligible(&chunk);
    let slots = [1i64]; // will be promoted to float

    let mut group = c.benchmark_group("slot_float_200");

    group.bench_function("interpreter", |b| {
        b.iter(|| {
            let mut vm = VM::new(chunk.clone());
            vm.frames.push(fusevm::Frame {
                return_ip: 0,
                stack_base: 0,
                slots: slots.iter().map(|&v| fusevm::Value::Int(v)).collect(),
            });
            let result = vm.run();
            black_box(result);
        })
    });

    if eligible {
        let _ = jit.try_run_linear(&chunk, &slots);
        group.bench_function("jit_cached", |b| {
            b.iter(|| {
                let result = jit.try_run_linear(black_box(&chunk), black_box(&slots));
                black_box(result);
            })
        });
    }

    group.bench_function("native_rust", |b| {
        b.iter(|| {
            let result = native_slot_float(black_box(1.0), 200);
            black_box(result);
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_add_100,
    bench_add_1000,
    bench_mixed_100,
    bench_bitwise_200,
    bench_float_200,
);
criterion_main!(benches);
