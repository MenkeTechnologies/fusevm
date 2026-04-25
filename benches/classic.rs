//! Classic VM benchmark algorithms — produces numbers comparable to other VMs.
//!
//! Each benchmark reports iterations/sec. Compare against published numbers from
//! LuaJIT, CPython, Ruby YARV, wasm3, etc.
//!
//! Run: cargo bench --bench classic

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use fusevm::{ChunkBuilder, Op, VMResult, Value, VM};

fn run_chunk(chunk: &fusevm::Chunk) -> Value {
    let mut vm = VM::new(chunk.clone());
    match vm.run() {
        VMResult::Ok(val) => val,
        _ => Value::Undef,
    }
}

// ── Fibonacci iterative: fib(35) = 9227465 ──

fn fib_iterative_chunk(n: i64) -> fusevm::Chunk {
    let mut b = ChunkBuilder::new();
    let fib_name = b.add_name("fib");

    // main: call fib(n)
    b.emit(Op::LoadInt(n), 1);
    b.emit(Op::Call(fib_name, 1), 1);
    let jump_end = b.emit(Op::Jump(0), 1);

    // fib(n): iterative
    let fib_ip = b.current_pos();
    b.add_sub_entry(fib_name, fib_ip);
    // slots: 0=a, 1=b, 2=i, 3=temp, arg is on stack
    b.emit(Op::SetSlot(2), 2);        // i = n (from stack arg)
    b.emit(Op::LoadInt(0), 2);
    b.emit(Op::SetSlot(0), 2);        // a = 0
    b.emit(Op::LoadInt(1), 2);
    b.emit(Op::SetSlot(1), 2);        // b = 1
    // loop: while i > 0
    let loop_start = b.current_pos();
    b.emit(Op::GetSlot(2), 2);        // push i
    b.emit(Op::LoadInt(0), 2);
    b.emit(Op::NumLe, 2);             // i <= 0?
    let exit_jump = b.emit(Op::JumpIfTrue(0), 2);
    // temp = a + b
    b.emit(Op::GetSlot(0), 2);
    b.emit(Op::GetSlot(1), 2);
    b.emit(Op::Add, 2);
    b.emit(Op::SetSlot(3), 2);        // temp = a+b
    // a = b
    b.emit(Op::GetSlot(1), 2);
    b.emit(Op::SetSlot(0), 2);
    // b = temp
    b.emit(Op::GetSlot(3), 2);
    b.emit(Op::SetSlot(1), 2);
    // i--
    b.emit(Op::GetSlot(2), 2);
    b.emit(Op::Dec, 2);
    b.emit(Op::SetSlot(2), 2);
    b.emit(Op::Jump(loop_start), 2);
    // exit
    let exit_ip = b.current_pos();
    b.patch_jump(exit_jump, exit_ip);
    b.emit(Op::GetSlot(0), 2);        // return a
    b.emit(Op::ReturnValue, 2);

    let end_ip = b.current_pos();
    b.patch_jump(jump_end, end_ip);
    b.build()
}

fn bench_fib_35(c: &mut Criterion) {
    let chunk = fib_iterative_chunk(35);
    c.bench_function("fib_iterative_35", |b| {
        b.iter(|| {
            let val = run_chunk(black_box(&chunk));
            black_box(val);
        })
    });
}

// ── Fibonacci recursive: fib(20) = 6765 ──

fn fib_recursive_chunk(n: i64) -> fusevm::Chunk {
    let mut b = ChunkBuilder::new();
    let fib_name = b.add_name("fib");

    b.emit(Op::LoadInt(n), 1);
    b.emit(Op::Call(fib_name, 1), 1);
    let jump_end = b.emit(Op::Jump(0), 1);

    // fib(n): if n <= 1 return n; else return fib(n-1) + fib(n-2)
    let fib_ip = b.current_pos();
    b.add_sub_entry(fib_name, fib_ip);
    b.emit(Op::SetSlot(0), 2);          // slot0 = n
    b.emit(Op::GetSlot(0), 2);
    b.emit(Op::LoadInt(1), 2);
    b.emit(Op::NumLe, 2);               // n <= 1?
    let base_jump = b.emit(Op::JumpIfTrue(0), 2);
    // recursive case
    b.emit(Op::GetSlot(0), 2);
    b.emit(Op::Dec, 2);                 // n-1
    b.emit(Op::Call(fib_name, 1), 2);   // fib(n-1)
    b.emit(Op::GetSlot(0), 2);
    b.emit(Op::LoadInt(2), 2);
    b.emit(Op::Sub, 2);                 // n-2
    b.emit(Op::Call(fib_name, 1), 2);   // fib(n-2)
    b.emit(Op::Add, 2);                 // fib(n-1) + fib(n-2)
    b.emit(Op::ReturnValue, 2);
    // base case
    let base_ip = b.current_pos();
    b.patch_jump(base_jump, base_ip);
    b.emit(Op::GetSlot(0), 2);
    b.emit(Op::ReturnValue, 2);

    let end_ip = b.current_pos();
    b.patch_jump(jump_end, end_ip);
    b.build()
}

fn bench_fib_recursive_20(c: &mut Criterion) {
    let chunk = fib_recursive_chunk(20);
    c.bench_function("fib_recursive_20", |b| {
        b.iter(|| {
            let val = run_chunk(black_box(&chunk));
            black_box(val);
        })
    });
}

// ── Sum 1..N using fused vs unfused ──

fn sum_fused_chunk(n: i32) -> fusevm::Chunk {
    let mut b = ChunkBuilder::new();
    b.emit(Op::PushFrame, 1);
    b.emit(Op::LoadInt(0), 1);
    b.emit(Op::SetSlot(0), 1);
    b.emit(Op::LoadInt(0), 1);
    b.emit(Op::SetSlot(1), 1);
    b.emit(Op::AccumSumLoop(0, 1, n), 1);
    b.emit(Op::GetSlot(0), 1);
    b.build()
}

fn sum_unfused_chunk(n: i64) -> fusevm::Chunk {
    let mut b = ChunkBuilder::new();
    b.emit(Op::PushFrame, 1);
    b.emit(Op::LoadInt(0), 1);
    b.emit(Op::SetSlot(0), 1);
    b.emit(Op::LoadInt(0), 1);
    b.emit(Op::SetSlot(1), 1);
    // loop body at ip=5
    b.emit(Op::GetSlot(0), 1);
    b.emit(Op::GetSlot(1), 1);
    b.emit(Op::Add, 1);
    b.emit(Op::SetSlot(0), 1);
    b.emit(Op::PreIncSlotVoid(1), 1);
    b.emit(Op::SlotLtIntJumpIfFalse(1, n as i32, 12), 1);
    b.emit(Op::Jump(5), 1);
    // ip=12
    b.emit(Op::GetSlot(0), 1);
    b.build()
}

fn bench_sum_1m_fused(c: &mut Criterion) {
    let chunk = sum_fused_chunk(1_000_000);
    c.bench_function("sum_1M_fused", |b| {
        b.iter(|| {
            let val = run_chunk(black_box(&chunk));
            black_box(val);
        })
    });
}

fn bench_sum_1m_unfused(c: &mut Criterion) {
    let chunk = sum_unfused_chunk(1_000_000);
    c.bench_function("sum_1M_unfused", |b| {
        b.iter(|| {
            let val = run_chunk(black_box(&chunk));
            black_box(val);
        })
    });
}

fn bench_sum_1m_native(c: &mut Criterion) {
    c.bench_function("sum_1M_native_rust", |b| {
        b.iter(|| {
            let mut sum: i64 = 0;
            for i in 0i64..1_000_000 {
                sum += i;
            }
            black_box(sum);
        })
    });
}

// ── Nested loop: matrix-style i*j accumulator ──

fn nested_loop_chunk(outer: i32, inner: i32) -> fusevm::Chunk {
    let mut b = ChunkBuilder::new();
    b.emit(Op::PushFrame, 1);
    // slot 0 = sum, slot 1 = i, slot 2 = j
    b.emit(Op::LoadInt(0), 1);
    b.emit(Op::SetSlot(0), 1);       // sum = 0
    b.emit(Op::LoadInt(0), 1);
    b.emit(Op::SetSlot(1), 1);       // i = 0
    // ip=5: outer loop
    let outer_start = b.current_pos();
    b.emit(Op::LoadInt(0), 1);
    b.emit(Op::SetSlot(2), 1);       // j = 0
    // ip=7: inner loop
    let inner_start = b.current_pos();
    b.emit(Op::GetSlot(0), 1);       // sum
    b.emit(Op::GetSlot(1), 1);       // i
    b.emit(Op::GetSlot(2), 1);       // j
    b.emit(Op::Add, 1);              // i + j
    b.emit(Op::Add, 1);              // sum + (i+j)
    b.emit(Op::SetSlot(0), 1);       // sum = ...
    b.emit(Op::SlotIncLtIntJumpBack(2, inner, inner_start), 1);
    // inner done
    b.emit(Op::SlotIncLtIntJumpBack(1, outer, outer_start), 1);
    // done
    b.emit(Op::GetSlot(0), 1);
    b.build()
}

fn bench_nested_100x100(c: &mut Criterion) {
    let chunk = nested_loop_chunk(100, 100);
    c.bench_function("nested_loop_100x100", |b| {
        b.iter(|| {
            let val = run_chunk(black_box(&chunk));
            black_box(val);
        })
    });
}

fn bench_nested_100x100_native(c: &mut Criterion) {
    c.bench_function("nested_loop_100x100_native", |b| {
        b.iter(|| {
            let mut sum: i64 = 0;
            for i in 0i64..100 {
                for j in 0i64..100 {
                    sum += i + j;
                }
            }
            black_box(sum);
        })
    });
}

// ── Ackermann (shallow): ack(3, 4) = 125 ──

fn ackermann_chunk() -> fusevm::Chunk {
    let mut b = ChunkBuilder::new();
    let ack_name = b.add_name("ack");

    // main: ack(3, 4)
    b.emit(Op::LoadInt(3), 1);
    b.emit(Op::LoadInt(4), 1);
    b.emit(Op::Call(ack_name, 2), 1);
    let jump_end = b.emit(Op::Jump(0), 1);

    // ack(m, n): slots 0=m, 1=n
    let ack_ip = b.current_pos();
    b.add_sub_entry(ack_name, ack_ip);
    b.emit(Op::SetSlot(1), 2);       // n (second arg, popped first)
    b.emit(Op::SetSlot(0), 2);       // m (first arg)

    // if m == 0: return n + 1
    b.emit(Op::GetSlot(0), 2);
    b.emit(Op::LoadInt(0), 2);
    b.emit(Op::NumEq, 2);
    let m0_jump = b.emit(Op::JumpIfTrue(0), 2);

    // if n == 0: return ack(m-1, 1)
    b.emit(Op::GetSlot(1), 2);
    b.emit(Op::LoadInt(0), 2);
    b.emit(Op::NumEq, 2);
    let n0_jump = b.emit(Op::JumpIfTrue(0), 2);

    // else: return ack(m-1, ack(m, n-1))
    b.emit(Op::GetSlot(0), 2);       // m
    b.emit(Op::GetSlot(1), 2);       // n
    b.emit(Op::Dec, 2);              // n-1
    b.emit(Op::Call(ack_name, 2), 2); // ack(m, n-1)
    // now stack has: result of inner ack
    b.emit(Op::GetSlot(0), 2);       // m
    b.emit(Op::Dec, 2);              // m-1
    b.emit(Op::Swap, 2);             // swap so args are (m-1, result)
    b.emit(Op::Call(ack_name, 2), 2); // ack(m-1, ack(m, n-1))
    b.emit(Op::ReturnValue, 2);

    // n == 0 case
    let n0_ip = b.current_pos();
    b.patch_jump(n0_jump, n0_ip);
    b.emit(Op::GetSlot(0), 2);
    b.emit(Op::Dec, 2);              // m-1
    b.emit(Op::LoadInt(1), 2);       // 1
    b.emit(Op::Call(ack_name, 2), 2); // ack(m-1, 1)
    b.emit(Op::ReturnValue, 2);

    // m == 0 case
    let m0_ip = b.current_pos();
    b.patch_jump(m0_jump, m0_ip);
    b.emit(Op::GetSlot(1), 2);
    b.emit(Op::Inc, 2);              // n + 1
    b.emit(Op::ReturnValue, 2);

    let end_ip = b.current_pos();
    b.patch_jump(jump_end, end_ip);
    b.build()
}

fn bench_ackermann_3_4(c: &mut Criterion) {
    let chunk = ackermann_chunk();
    c.bench_function("ackermann_3_4", |b| {
        b.iter(|| {
            let val = run_chunk(black_box(&chunk));
            black_box(val);
        })
    });
}

// ── Dispatch throughput: Nop loop ──

fn bench_dispatch_1m(c: &mut Criterion) {
    let mut b = ChunkBuilder::new();
    for _ in 0..1_000_000 {
        b.emit(Op::Nop, 1);
    }
    b.emit(Op::LoadInt(0), 1);
    let chunk = b.build();
    c.bench_function("dispatch_nop_1M", |b| {
        b.iter(|| {
            let val = run_chunk(black_box(&chunk));
            black_box(val);
        })
    });
}

// ── String building ──

fn bench_string_build_10k(c: &mut Criterion) {
    let mut builder = ChunkBuilder::new();
    let const_idx = builder.add_constant(Value::str("x"));
    let mut b2 = ChunkBuilder::new();
    let ci = b2.add_constant(Value::str("x"));
    b2.emit(Op::PushFrame, 1);
    b2.emit(Op::LoadConst(ci), 1);
    b2.emit(Op::SetSlot(0), 1);
    b2.emit(Op::LoadInt(0), 1);
    b2.emit(Op::SetSlot(1), 1);
    b2.emit(Op::ConcatConstLoop(ci, 0, 1, 10_000), 1);
    b2.emit(Op::GetSlot(0), 1);
    let chunk = b2.build();
    c.bench_function("string_build_10k", |b| {
        b.iter(|| {
            let val = run_chunk(black_box(&chunk));
            black_box(val);
        })
    });
}

fn bench_string_build_10k_native(c: &mut Criterion) {
    c.bench_function("string_build_10k_native", |b| {
        b.iter(|| {
            let mut s = String::with_capacity(10_001);
            s.push('x');
            for _ in 0..10_000 {
                s.push('x');
            }
            black_box(s);
        })
    });
}

criterion_group!(
    benches,
    bench_fib_35,
    bench_fib_recursive_20,
    bench_sum_1m_fused,
    bench_sum_1m_unfused,
    bench_sum_1m_native,
    bench_nested_100x100,
    bench_nested_100x100_native,
    bench_ackermann_3_4,
    bench_dispatch_1m,
    bench_string_build_10k,
    bench_string_build_10k_native,
);
criterion_main!(benches);
