//! Interpreter vs native AOT vs native Rust — the same numeric loop.
//!
//! This validates the closed-world AOT native path ([`fusevm::aot`]): a hot
//! integer loop whose body and counter live in slots. The native path lowers it
//! to a register-only loop (no per-op dispatch, no per-op shim calls), so it
//! should land far closer to hand-written Rust than the interpreter.
//!
//! The loop chunk is tiny (~20 ops) but runs many iterations, so the one-time
//! Cranelift compile folded into `run_chunk_native` is negligible next to the
//! execution it's measured against — this is the real "compile once, run a hot
//! program" story, not a microbenchmark of a single op.
//!
//! Run: cargo bench --features aot --bench aot_native

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use fusevm::{aot, Chunk, ChunkBuilder, Op, VMResult, Value, VM};

/// `sum = 0; i = 0; while i < n { sum += i*3 - 1; i += 1 } return sum`
/// — all in integer slots, so the native path holds it entirely in registers.
fn loop_chunk(n: i64) -> Chunk {
    let mut b = ChunkBuilder::new();
    // slot 0 = sum, slot 1 = i
    b.emit(Op::LoadInt(0), 1);
    b.emit(Op::SetSlot(0), 1);
    b.emit(Op::LoadInt(0), 1);
    b.emit(Op::SetSlot(1), 1);
    let top = b.current_pos();
    b.emit(Op::GetSlot(1), 1);
    b.emit(Op::LoadInt(n), 1);
    b.emit(Op::NumLt, 1);
    let exit = b.emit(Op::JumpIfFalse(0), 1);
    // sum += i*3 - 1
    b.emit(Op::GetSlot(0), 1); // sum
    b.emit(Op::GetSlot(1), 1); // i
    b.emit(Op::LoadInt(3), 1);
    b.emit(Op::Mul, 1); // i*3
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::Sub, 1); // i*3 - 1
    b.emit(Op::Add, 1); // sum + (i*3 - 1)
    b.emit(Op::SetSlot(0), 1);
    // i += 1
    b.emit(Op::GetSlot(1), 1);
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::Add, 1);
    b.emit(Op::SetSlot(1), 1);
    b.emit(Op::Jump(top), 1);
    let end = b.current_pos();
    b.patch_jump(exit, end);
    b.emit(Op::GetSlot(0), 1);
    b.build()
}

/// The same computation in hand-written Rust (wrapping, to match the VM). The
/// `black_box` on the accumulator keeps LLVM from collapsing this arithmetic
/// series into a closed-form formula — Cranelift does no such scalar-evolution,
/// so this stays an honest "real scalar loop" ceiling to compare against.
fn native_baseline(n: i64) -> i64 {
    let mut sum = 0i64;
    let mut i = 0i64;
    while i < n {
        sum = black_box(sum.wrapping_add(i.wrapping_mul(3).wrapping_sub(1)));
        i = i.wrapping_add(1);
    }
    sum
}

fn bench(c: &mut Criterion) {
    let n = 200_000i64;
    let chunk = loop_chunk(n);

    // Sanity: all three contenders must agree before we time them.
    let expect = native_baseline(n);
    let interp = {
        let mut vm = VM::new(chunk.clone());
        vm.run()
    };
    let native = aot::run_chunk_native(&chunk, |_| {}).expect("native compile/run");
    assert!(
        matches!(interp, VMResult::Ok(Value::Int(v)) if v == expect),
        "interpreter result mismatch"
    );
    assert!(
        matches!(native, VMResult::Ok(Value::Int(v)) if v == expect),
        "native AOT result mismatch"
    );

    let mut g = c.benchmark_group("numeric_loop_200k");
    g.bench_function("interpreter", |b| {
        b.iter(|| {
            let mut vm = VM::new(chunk.clone());
            black_box(vm.run())
        })
    });
    g.bench_function("aot_native", |b| {
        b.iter(|| black_box(aot::run_chunk_native(black_box(&chunk), |_| {}).unwrap()))
    });
    g.bench_function("native_rust", |b| {
        b.iter(|| black_box(native_baseline(black_box(n))))
    });
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
