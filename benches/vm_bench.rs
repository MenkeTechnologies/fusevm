//! fusevm benchmarks — tracks performance of the dispatch loop, fused ops,
//! string/array/hash operations, and JIT eligibility analysis.
//!
//! Run:   cargo bench
//! Compare against baseline:  cargo bench -- --save-baseline <name>
//! HTML report:  target/criterion/report/index.html

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use fusevm::{ChunkBuilder, Op, VMResult, Value, VM};

// ── Helpers ──

fn run_ops(ops: &[(Op, u32)]) -> Value {
    let mut b = ChunkBuilder::new();
    for (op, line) in ops {
        b.emit(op.clone(), *line);
    }
    let mut vm = VM::new(b.build());
    match vm.run() {
        VMResult::Ok(val) => val,
        _ => Value::Undef,
    }
}

// ── Arithmetic ──

fn bench_int_add_1k(c: &mut Criterion) {
    let mut ops: Vec<(Op, u32)> = Vec::with_capacity(2001);
    ops.push((Op::LoadInt(0), 1));
    for _ in 0..1000 {
        ops.push((Op::LoadInt(1), 1));
        ops.push((Op::Add, 1));
    }
    c.bench_function("int_add_1k", |b| {
        b.iter(|| {
            let val = run_ops(black_box(&ops));
            black_box(val);
        })
    });
}

fn bench_float_add_1k(c: &mut Criterion) {
    let mut ops: Vec<(Op, u32)> = Vec::with_capacity(2001);
    ops.push((Op::LoadFloat(0.0), 1));
    for _ in 0..1000 {
        ops.push((Op::LoadFloat(1.5), 1));
        ops.push((Op::Add, 1));
    }
    c.bench_function("float_add_1k", |b| {
        b.iter(|| {
            let val = run_ops(black_box(&ops));
            black_box(val);
        })
    });
}

fn bench_mixed_arith(c: &mut Criterion) {
    let mut ops: Vec<(Op, u32)> = Vec::with_capacity(501);
    ops.push((Op::LoadInt(1), 1));
    for _ in 0..100 {
        ops.push((Op::LoadInt(3), 1));
        ops.push((Op::Mul, 1));
        ops.push((Op::LoadInt(7), 1));
        ops.push((Op::Add, 1));
        ops.push((Op::LoadInt(2), 1));
        ops.push((Op::Sub, 1));
    }
    c.bench_function("mixed_arith_100", |b| {
        b.iter(|| {
            let val = run_ops(black_box(&ops));
            black_box(val);
        })
    });
}

// ── Fused superinstructions ──

fn bench_fused_accum_sum(c: &mut Criterion) {
    // sum = 0; for i in 0..10_000 { sum += i }
    let ops = vec![
        (Op::PushFrame, 1),
        (Op::LoadInt(0), 1),
        (Op::SetSlot(0), 1),
        (Op::LoadInt(0), 1),
        (Op::SetSlot(1), 1),
        (Op::AccumSumLoop(0, 1, 10_000), 1),
        (Op::GetSlot(0), 1),
    ];
    c.bench_function("fused_accum_sum_10k", |b| {
        b.iter(|| {
            let val = run_ops(black_box(&ops));
            black_box(val);
        })
    });
}

fn bench_fused_slot_inc_loop(c: &mut Criterion) {
    // i = 0; while (i < 10_000) { i++ }
    let ops = vec![
        (Op::PushFrame, 1),
        (Op::LoadInt(0), 1),
        (Op::SetSlot(0), 1),
        // ip=3: loop body (nop)
        (Op::Nop, 1),
        // ip=4: backedge
        (Op::SlotIncLtIntJumpBack(0, 10_000, 3), 1),
        (Op::GetSlot(0), 1),
    ];
    c.bench_function("fused_slot_inc_10k", |b| {
        b.iter(|| {
            let val = run_ops(black_box(&ops));
            black_box(val);
        })
    });
}

fn bench_unfused_sum_loop(c: &mut Criterion) {
    // Same as fused_accum_sum but using individual ops to show the speedup
    let ops = vec![
        (Op::PushFrame, 1),
        (Op::LoadInt(0), 1),
        (Op::SetSlot(0), 1), // sum = 0
        (Op::LoadInt(0), 1),
        (Op::SetSlot(1), 1), // i = 0
        // ip=5: loop body
        (Op::GetSlot(0), 1),                          // push sum
        (Op::GetSlot(1), 1),                          // push i
        (Op::Add, 1),                                 // sum + i
        (Op::SetSlot(0), 1),                          // sum = result
        (Op::PreIncSlotVoid(1), 1),                   // i++
        (Op::SlotLtIntJumpIfFalse(1, 10_000, 12), 1), // if i < 10000 skip exit
        (Op::Jump(5), 1),                             // loop back
        // ip=12: exit
        (Op::GetSlot(0), 1),
    ];
    c.bench_function("unfused_sum_loop_10k", |b| {
        b.iter(|| {
            let val = run_ops(black_box(&ops));
            black_box(val);
        })
    });
}

// ── String operations ──

fn bench_concat_1k(c: &mut Criterion) {
    let mut b = ChunkBuilder::new();
    let const_idx = b.add_constant(Value::str("x"));
    let mut ops: Vec<(Op, u32)> = Vec::with_capacity(2001);
    ops.push((Op::LoadConst(const_idx), 1));
    for _ in 0..999 {
        ops.push((Op::LoadConst(const_idx), 1));
        ops.push((Op::Concat, 1));
    }
    c.bench_function("concat_1k", |bencher| {
        bencher.iter(|| {
            let val = run_ops(black_box(&ops));
            black_box(val);
        })
    });
}

fn bench_fused_concat_const_loop(c: &mut Criterion) {
    let mut b = ChunkBuilder::new();
    let const_idx = b.add_constant(Value::str("x"));
    let ops = vec![
        (Op::PushFrame, 1),
        (Op::LoadConst(const_idx), 1),
        (Op::SetSlot(0), 1), // s = "x"
        (Op::LoadInt(0), 1),
        (Op::SetSlot(1), 1), // i = 0
        (Op::ConcatConstLoop(const_idx, 0, 1, 1000), 1),
        (Op::GetSlot(0), 1),
    ];
    c.bench_function("fused_concat_const_1k", |bencher| {
        bencher.iter(|| {
            let val = run_ops(black_box(&ops));
            black_box(val);
        })
    });
}

fn bench_str_compare(c: &mut Criterion) {
    let mut b = ChunkBuilder::new();
    let a_idx = b.add_constant(Value::str("hello world"));
    let b_idx = b.add_constant(Value::str("hello world"));
    let mut ops: Vec<(Op, u32)> = Vec::with_capacity(3001);
    for _ in 0..1000 {
        ops.push((Op::LoadConst(a_idx), 1));
        ops.push((Op::LoadConst(b_idx), 1));
        ops.push((Op::StrEq, 1));
        ops.push((Op::Pop, 1));
    }
    // leave one result on stack
    ops.push((Op::LoadConst(a_idx), 1));
    ops.push((Op::LoadConst(b_idx), 1));
    ops.push((Op::StrEq, 1));
    c.bench_function("str_eq_1k", |bencher| {
        bencher.iter(|| {
            let val = run_ops(black_box(&ops));
            black_box(val);
        })
    });
}

// ── Array operations ──

fn bench_array_push_pop_1k(c: &mut Criterion) {
    let mut b = ChunkBuilder::new();
    let arr_name = b.add_name("arr");
    let mut ops: Vec<(Op, u32)> = Vec::with_capacity(4003);
    ops.push((Op::DeclareArray(arr_name), 1));
    // push 1000 elements
    for i in 0..1000 {
        ops.push((Op::LoadInt(i), 1));
        ops.push((Op::ArrayPush(arr_name), 1));
    }
    // pop all
    for _ in 0..1000 {
        ops.push((Op::ArrayPop(arr_name), 1));
        ops.push((Op::Pop, 1));
    }
    ops.push((Op::ArrayLen(arr_name), 1));
    c.bench_function("array_push_pop_1k", |bencher| {
        bencher.iter(|| {
            let val = run_ops(black_box(&ops));
            black_box(val);
        })
    });
}

fn bench_array_index_1k(c: &mut Criterion) {
    let mut b = ChunkBuilder::new();
    let arr_name = b.add_name("arr");
    let mut ops: Vec<(Op, u32)> = Vec::with_capacity(5003);
    ops.push((Op::DeclareArray(arr_name), 1));
    // build array of 1000 elements
    for i in 0..1000 {
        ops.push((Op::LoadInt(i), 1));
        ops.push((Op::ArrayPush(arr_name), 1));
    }
    // random-access read 1000 times
    for i in 0..1000 {
        ops.push((Op::LoadInt(i % 1000), 1));
        ops.push((Op::ArrayGet(arr_name), 1));
        ops.push((Op::Pop, 1));
    }
    ops.push((Op::ArrayLen(arr_name), 1));
    c.bench_function("array_index_1k", |bencher| {
        bencher.iter(|| {
            let val = run_ops(black_box(&ops));
            black_box(val);
        })
    });
}

fn bench_fused_push_int_range(c: &mut Criterion) {
    let mut b = ChunkBuilder::new();
    let arr_name = b.add_name("arr");
    let ops = vec![
        (Op::DeclareArray(arr_name), 1),
        (Op::PushFrame, 1),
        (Op::LoadInt(0), 1),
        (Op::SetSlot(0), 1),
        (Op::PushIntRangeLoop(arr_name, 0, 10_000), 1),
        (Op::ArrayLen(arr_name), 1),
    ];
    c.bench_function("fused_push_range_10k", |bencher| {
        bencher.iter(|| {
            let val = run_ops(black_box(&ops));
            black_box(val);
        })
    });
}

fn bench_range_10k(c: &mut Criterion) {
    let ops = vec![(Op::LoadInt(0), 1), (Op::LoadInt(9999), 1), (Op::Range, 1)];
    c.bench_function("range_10k", |bencher| {
        bencher.iter(|| {
            let val = run_ops(black_box(&ops));
            black_box(val);
        })
    });
}

// ── Hash operations ──

fn bench_hash_set_get_1k(c: &mut Criterion) {
    let mut builder = ChunkBuilder::new();
    let hash_name = builder.add_name("h");
    let mut const_keys = Vec::new();
    for i in 0..100 {
        const_keys.push(builder.add_constant(Value::str(format!("key_{i}"))));
    }
    let mut ops: Vec<(Op, u32)> = Vec::with_capacity(1003);
    ops.push((Op::DeclareHash(hash_name), 1));
    // insert 100 keys
    for (i, &k) in const_keys.iter().enumerate() {
        ops.push((Op::LoadInt(i as i64), 1)); // value
        ops.push((Op::LoadConst(k), 1)); // key
        ops.push((Op::HashSet(hash_name), 1));
    }
    // lookup each key 10 times = 1000 lookups
    for _ in 0..10 {
        for &k in &const_keys {
            ops.push((Op::LoadConst(k), 1));
            ops.push((Op::HashGet(hash_name), 1));
            ops.push((Op::Pop, 1));
        }
    }
    ops.push((Op::LoadInt(0), 1));
    c.bench_function("hash_set_get_1k", |bencher| {
        bencher.iter(|| {
            let val = run_ops(black_box(&ops));
            black_box(val);
        })
    });
}

// ── Stack operations ──

fn bench_dup_swap_rot_1k(c: &mut Criterion) {
    let mut ops: Vec<(Op, u32)> = Vec::with_capacity(4004);
    ops.push((Op::LoadInt(1), 1));
    ops.push((Op::LoadInt(2), 1));
    ops.push((Op::LoadInt(3), 1));
    for _ in 0..1000 {
        ops.push((Op::Rot, 1));
        ops.push((Op::Swap, 1));
        ops.push((Op::Dup, 1));
        ops.push((Op::Pop, 1));
    }
    c.bench_function("stack_dup_swap_rot_1k", |bencher| {
        bencher.iter(|| {
            let val = run_ops(black_box(&ops));
            black_box(val);
        })
    });
}

// ── Slot variables ──

fn bench_slot_read_write_10k(c: &mut Criterion) {
    let mut ops: Vec<(Op, u32)> = Vec::with_capacity(40006);
    ops.push((Op::PushFrame, 1));
    ops.push((Op::LoadInt(0), 1));
    ops.push((Op::SetSlot(0), 1));
    ops.push((Op::LoadInt(0), 1));
    ops.push((Op::SetSlot(1), 1));
    for _ in 0..10_000 {
        ops.push((Op::GetSlot(0), 1));
        ops.push((Op::GetSlot(1), 1));
        ops.push((Op::Add, 1));
        ops.push((Op::SetSlot(0), 1));
    }
    ops.push((Op::GetSlot(0), 1));
    c.bench_function("slot_read_write_10k", |bencher| {
        bencher.iter(|| {
            let val = run_ops(black_box(&ops));
            black_box(val);
        })
    });
}

// ── Comparison ──

fn bench_int_cmp_1k(c: &mut Criterion) {
    let mut ops: Vec<(Op, u32)> = Vec::with_capacity(4001);
    for i in 0..1000 {
        ops.push((Op::LoadInt(i), 1));
        ops.push((Op::LoadInt(i + 1), 1));
        ops.push((Op::NumLt, 1));
        ops.push((Op::Pop, 1));
    }
    ops.push((Op::LoadInt(1), 1));
    c.bench_function("int_cmp_1k", |bencher| {
        bencher.iter(|| {
            let val = run_ops(black_box(&ops));
            black_box(val);
        })
    });
}

// ── Control flow ──

fn bench_jump_loop_10k(c: &mut Criterion) {
    // Manual counted loop with jumps: i=0; while(i<10000) { i++ }
    let ops = vec![
        (Op::PushFrame, 1),
        (Op::LoadInt(0), 1),
        (Op::SetSlot(0), 1),
        // ip=3: body
        (Op::PreIncSlotVoid(0), 1),
        (Op::SlotLtIntJumpIfFalse(0, 10_000, 6), 1),
        (Op::Jump(3), 1),
        // ip=6: exit
        (Op::GetSlot(0), 1),
    ];
    c.bench_function("jump_loop_10k", |bencher| {
        bencher.iter(|| {
            let val = run_ops(black_box(&ops));
            black_box(val);
        })
    });
}

// ── Function calls ──

fn bench_function_call_1k(c: &mut Criterion) {
    let mut b = ChunkBuilder::new();
    let inc_name = b.add_name("inc");
    let mut ops: Vec<(Op, u32)> = Vec::with_capacity(2005);
    ops.push((Op::LoadInt(0), 1));
    // 1000 calls to inc function
    for _ in 0..1000 {
        ops.push((Op::Call(inc_name, 1), 1));
    }
    ops.push((Op::Jump(0), 1)); // placeholder, will be patched
                                // inc function body at ip = 1002
    let func_ip = ops.len();
    ops.push((Op::Inc, 2));
    ops.push((Op::ReturnValue, 2));

    // Build manually to set up sub_entries and patch jump
    let mut builder = ChunkBuilder::new();
    let name = builder.add_name("inc");
    for (i, (op, line)) in ops.iter().enumerate() {
        if i == 1001 {
            // patch the jump to skip past function body
            builder.emit(Op::Jump(func_ip + 2), *line);
        } else {
            builder.emit(op.clone(), *line);
        }
    }
    builder.add_sub_entry(name, func_ip);
    let chunk = builder.build();

    c.bench_function("function_call_1k", |bencher| {
        bencher.iter(|| {
            let mut vm = VM::new(chunk.clone());
            let result = vm.run();
            black_box(result);
        })
    });
}

// ── LoadConst ──

fn bench_load_const_int_1k(c: &mut Criterion) {
    let mut b = ChunkBuilder::new();
    let idx = b.add_constant(Value::Int(42));
    let mut ops: Vec<(Op, u32)> = Vec::with_capacity(2001);
    for _ in 0..1000 {
        ops.push((Op::LoadConst(idx), 1));
        ops.push((Op::Pop, 1));
    }
    ops.push((Op::LoadConst(idx), 1));
    c.bench_function("load_const_int_1k", |bencher| {
        bencher.iter(|| {
            let val = run_ops(black_box(&ops));
            black_box(val);
        })
    });
}

fn bench_load_const_str_1k(c: &mut Criterion) {
    let mut b = ChunkBuilder::new();
    let idx = b.add_constant(Value::str("hello world"));
    let mut ops: Vec<(Op, u32)> = Vec::with_capacity(2001);
    for _ in 0..1000 {
        ops.push((Op::LoadConst(idx), 1));
        ops.push((Op::Pop, 1));
    }
    ops.push((Op::LoadConst(idx), 1));
    c.bench_function("load_const_str_1k", |bencher| {
        bencher.iter(|| {
            let val = run_ops(black_box(&ops));
            black_box(val);
        })
    });
}

// ── MakeHash / MakeArray ──

fn bench_make_hash_100(c: &mut Criterion) {
    let mut ops: Vec<(Op, u32)> = Vec::with_capacity(201);
    for i in 0..100 {
        ops.push((Op::LoadInt(i), 1)); // value
        ops.push((Op::LoadInt(i * 100), 1)); // key (coerced to str)
    }
    ops.push((Op::MakeHash(200), 1));
    c.bench_function("make_hash_100", |bencher| {
        bencher.iter(|| {
            let val = run_ops(black_box(&ops));
            black_box(val);
        })
    });
}

// ── Global variable access ──

fn bench_global_var_read_write_1k(c: &mut Criterion) {
    let mut builder = ChunkBuilder::new();
    let x = builder.add_name("x");
    let mut ops: Vec<(Op, u32)> = Vec::with_capacity(4003);
    ops.push((Op::LoadInt(0), 1));
    ops.push((Op::SetVar(x), 1));
    for _ in 0..1000 {
        ops.push((Op::GetVar(x), 1));
        ops.push((Op::LoadInt(1), 1));
        ops.push((Op::Add, 1));
        ops.push((Op::SetVar(x), 1));
    }
    ops.push((Op::GetVar(x), 1));
    c.bench_function("global_var_rw_1k", |bencher| {
        bencher.iter(|| {
            let val = run_ops(black_box(&ops));
            black_box(val);
        })
    });
}

criterion_group!(
    benches,
    // arithmetic
    bench_int_add_1k,
    bench_float_add_1k,
    bench_mixed_arith,
    // fused vs unfused
    bench_fused_accum_sum,
    bench_unfused_sum_loop,
    bench_fused_slot_inc_loop,
    bench_fused_concat_const_loop,
    bench_fused_push_int_range,
    // strings
    bench_concat_1k,
    bench_str_compare,
    // arrays
    bench_array_push_pop_1k,
    bench_array_index_1k,
    bench_range_10k,
    // hashes
    bench_hash_set_get_1k,
    bench_make_hash_100,
    // stack
    bench_dup_swap_rot_1k,
    // slots + globals
    bench_slot_read_write_10k,
    bench_global_var_read_write_1k,
    // comparisons
    bench_int_cmp_1k,
    // control flow
    bench_jump_loop_10k,
    // functions
    bench_function_call_1k,
    // constants
    bench_load_const_int_1k,
    bench_load_const_str_1k,
);
criterion_main!(benches);
