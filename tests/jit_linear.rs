#![allow(clippy::approx_constant)]
#![cfg(feature = "jit")]

use fusevm::{ChunkBuilder, JitCompiler, Op, Value};

fn jit_run(ops: &[(Op, u32)]) -> Option<Value> {
    let mut b = ChunkBuilder::new();
    for (op, line) in ops {
        b.emit(op.clone(), *line);
    }
    let chunk = b.build();
    let jit = JitCompiler::new();
    jit.try_run_linear(&chunk, &[])
}

fn jit_expect_int(ops: &[(Op, u32)], expected: i64) {
    match jit_run(ops) {
        Some(Value::Int(n)) => assert_eq!(n, expected, "expected {expected}, got {n}"),
        other => panic!("expected Some(Int({expected})), got {other:?}"),
    }
}

fn jit_expect_float(ops: &[(Op, u32)], expected: f64) {
    match jit_run(ops) {
        Some(Value::Float(f)) => {
            assert!((f - expected).abs() < 1e-10, "expected {expected}, got {f}");
        }
        other => panic!("expected Some(Float({expected})), got {other:?}"),
    }
}

// ── Arithmetic ──

#[test]
fn jit_add() {
    jit_expect_int(
        &[(Op::LoadInt(40), 1), (Op::LoadInt(2), 1), (Op::Add, 1)],
        42,
    );
}

#[test]
fn jit_sub() {
    jit_expect_int(
        &[(Op::LoadInt(50), 1), (Op::LoadInt(8), 1), (Op::Sub, 1)],
        42,
    );
}

#[test]
fn jit_mul() {
    jit_expect_int(
        &[(Op::LoadInt(6), 1), (Op::LoadInt(7), 1), (Op::Mul, 1)],
        42,
    );
}

#[test]
fn jit_negate() {
    jit_expect_int(&[(Op::LoadInt(42), 1), (Op::Negate, 1)], -42);
}

#[test]
fn jit_inc_dec() {
    jit_expect_int(&[(Op::LoadInt(41), 1), (Op::Inc, 1)], 42);
    jit_expect_int(&[(Op::LoadInt(43), 1), (Op::Dec, 1)], 42);
}

#[test]
fn jit_pow() {
    jit_expect_int(
        &[(Op::LoadInt(2), 1), (Op::LoadInt(10), 1), (Op::Pow, 1)],
        1024,
    );
}

#[test]
fn jit_float_add() {
    jit_expect_float(
        &[
            (Op::LoadFloat(1.5), 1),
            (Op::LoadFloat(2.5), 1),
            (Op::Add, 1),
        ],
        4.0,
    );
}

#[test]
fn jit_mixed_int_float() {
    jit_expect_float(
        &[(Op::LoadInt(1), 1), (Op::LoadFloat(2.5), 1), (Op::Add, 1)],
        3.5,
    );
}

#[test]
fn jit_chained_arithmetic() {
    // (2 + 3) * 4 = 20
    jit_expect_int(
        &[
            (Op::LoadInt(2), 1),
            (Op::LoadInt(3), 1),
            (Op::Add, 1),
            (Op::LoadInt(4), 1),
            (Op::Mul, 1),
        ],
        20,
    );
}

// ── Comparisons ──

#[test]
fn jit_numeric_comparisons() {
    jit_expect_int(
        &[(Op::LoadInt(1), 1), (Op::LoadInt(2), 1), (Op::NumLt, 1)],
        1,
    );
    jit_expect_int(
        &[(Op::LoadInt(2), 1), (Op::LoadInt(1), 1), (Op::NumLt, 1)],
        0,
    );
    jit_expect_int(
        &[(Op::LoadInt(5), 1), (Op::LoadInt(5), 1), (Op::NumEq, 1)],
        1,
    );
    jit_expect_int(
        &[(Op::LoadInt(5), 1), (Op::LoadInt(3), 1), (Op::NumGt, 1)],
        1,
    );
    jit_expect_int(
        &[(Op::LoadInt(5), 1), (Op::LoadInt(5), 1), (Op::NumLe, 1)],
        1,
    );
    jit_expect_int(
        &[(Op::LoadInt(5), 1), (Op::LoadInt(5), 1), (Op::NumGe, 1)],
        1,
    );
    jit_expect_int(
        &[(Op::LoadInt(5), 1), (Op::LoadInt(5), 1), (Op::NumNe, 1)],
        0,
    );
}

#[test]
fn jit_spaceship() {
    jit_expect_int(
        &[(Op::LoadInt(1), 1), (Op::LoadInt(2), 1), (Op::Spaceship, 1)],
        -1,
    );
    jit_expect_int(
        &[(Op::LoadInt(5), 1), (Op::LoadInt(5), 1), (Op::Spaceship, 1)],
        0,
    );
    jit_expect_int(
        &[(Op::LoadInt(9), 1), (Op::LoadInt(3), 1), (Op::Spaceship, 1)],
        1,
    );
}

// ── Bitwise ──

#[test]
fn jit_bitwise() {
    jit_expect_int(
        &[
            (Op::LoadInt(0xFF), 1),
            (Op::LoadInt(0x0F), 1),
            (Op::BitAnd, 1),
        ],
        0x0F,
    );
    jit_expect_int(
        &[
            (Op::LoadInt(0xF0), 1),
            (Op::LoadInt(0x0F), 1),
            (Op::BitOr, 1),
        ],
        0xFF,
    );
    jit_expect_int(
        &[
            (Op::LoadInt(0xFF), 1),
            (Op::LoadInt(0xFF), 1),
            (Op::BitXor, 1),
        ],
        0,
    );
    jit_expect_int(
        &[(Op::LoadInt(1), 1), (Op::LoadInt(8), 1), (Op::Shl, 1)],
        256,
    );
    jit_expect_int(
        &[(Op::LoadInt(256), 1), (Op::LoadInt(4), 1), (Op::Shr, 1)],
        16,
    );
}

// ── Stack ops ──

#[test]
fn jit_dup() {
    jit_expect_int(&[(Op::LoadInt(21), 1), (Op::Dup, 1), (Op::Add, 1)], 42);
}

#[test]
fn jit_swap() {
    jit_expect_int(
        &[
            (Op::LoadInt(10), 1),
            (Op::LoadInt(3), 1),
            (Op::Swap, 1),
            (Op::Sub, 1),
        ],
        -7,
    );
}

// ── LogNot ──

#[test]
fn jit_lognot() {
    jit_expect_int(&[(Op::LoadInt(0), 1), (Op::LogNot, 1)], 1);
    jit_expect_int(&[(Op::LoadInt(42), 1), (Op::LogNot, 1)], 0);
}

// ── Slot ops ──

#[test]
fn jit_slot_get_set() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::GetSlot(0), 1);
    b.emit(Op::GetSlot(1), 1);
    b.emit(Op::Add, 1);
    let chunk = b.build();
    let jit = JitCompiler::new();
    let slots: Vec<i64> = vec![100, 200];
    match jit.try_run_linear(&chunk, &slots) {
        Some(Value::Int(300)) => {}
        other => panic!("expected Int(300), got {other:?}"),
    }
}

#[test]
fn jit_pre_inc_slot() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::PreIncSlot(0), 1);
    let chunk = b.build();
    let jit = JitCompiler::new();
    let slots: Vec<i64> = vec![41];
    match jit.try_run_linear(&chunk, &slots) {
        Some(Value::Int(42)) => {}
        other => panic!("expected Int(42), got {other:?}"),
    }
}

// ── Constants ──

#[test]
fn jit_load_const_int() {
    let mut b = ChunkBuilder::new();
    let c = b.add_constant(Value::Int(42));
    b.emit(Op::LoadConst(c), 1);
    let chunk = b.build();
    let jit = JitCompiler::new();
    match jit.try_run_linear(&chunk, &[]) {
        Some(Value::Int(42)) => {}
        other => panic!("expected Int(42), got {other:?}"),
    }
}

#[test]
fn jit_load_const_float() {
    let mut b = ChunkBuilder::new();
    let c = b.add_constant(Value::Float(3.14));
    b.emit(Op::LoadConst(c), 1);
    let chunk = b.build();
    let jit = JitCompiler::new();
    match jit.try_run_linear(&chunk, &[]) {
        Some(Value::Float(f)) => assert!((f - 3.14).abs() < 1e-10),
        other => panic!("expected Float(3.14), got {other:?}"),
    }
}

// ── Eligibility ──

#[test]
fn jit_ineligible_string_ops() {
    // Concat is not JIT-eligible in linear mode
    let result = jit_run(&[(Op::LoadInt(1), 1), (Op::LoadInt(2), 1), (Op::Concat, 1)]);
    assert!(result.is_none());
}

#[test]
fn jit_ineligible_exec() {
    let result = jit_run(&[(Op::Exec(0), 1)]);
    assert!(result.is_none());
}

// ── Cache hit (run twice, second should be cached) ──

#[test]
fn jit_cache_hit() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(21), 1);
    b.emit(Op::LoadInt(21), 1);
    b.emit(Op::Add, 1);
    let chunk = b.build();
    let jit = JitCompiler::new();
    let first = jit.try_run_linear(&chunk, &[]);
    let second = jit.try_run_linear(&chunk, &[]);
    assert_eq!(format!("{first:?}"), format!("{second:?}"),);
}

// ── Complex expression ──

#[test]
fn jit_complex_expression() {
    // ((10 + 20) * 3 - 5) = 85
    jit_expect_int(
        &[
            (Op::LoadInt(10), 1),
            (Op::LoadInt(20), 1),
            (Op::Add, 1),
            (Op::LoadInt(3), 1),
            (Op::Mul, 1),
            (Op::LoadInt(5), 1),
            (Op::Sub, 1),
        ],
        85,
    );
}

#[test]
fn jit_bool_constants() {
    jit_expect_int(&[(Op::LoadTrue, 1)], 1);
    jit_expect_int(&[(Op::LoadFalse, 1)], 0);
}

#[test]
fn jit_float_spaceship() {
    jit_expect_int(
        &[
            (Op::LoadFloat(1.5), 1),
            (Op::LoadFloat(2.5), 1),
            (Op::Spaceship, 1),
        ],
        -1,
    );
}

#[test]
fn jit_float_negate() {
    jit_expect_float(&[(Op::LoadFloat(3.14), 1), (Op::Negate, 1)], -3.14);
}

#[test]
fn jit_mod_int() {
    // 17 % 5 = 2 — but only if constant non-zero divisor
    jit_expect_int(
        &[(Op::LoadInt(17), 1), (Op::LoadInt(5), 1), (Op::Mod, 1)],
        2,
    );
}

// ── Float-kind preservation (zero-valued and integral-valued float constants) ──
//
// The interpreter keeps `Value::Float` for float operands regardless of the
// value; the JIT must agree. Regression tests for the elisprs bug where an
// integral-valued `LoadFloat` constant (including ±0.0) was lowered as
// `JitTy::Int`, so `LoadFloat(-0.0); Negate` came back as `Int(0)` from the
// JIT tier while the first (interpreted) evaluation returned `Float(0.0)`.
// Bit-exact assertions so ±0.0 sign is pinned too.

fn jit_expect_float_bits(ops: &[(Op, u32)], expected: f64) {
    match jit_run(ops) {
        Some(Value::Float(f)) => assert_eq!(
            f.to_bits(),
            expected.to_bits(),
            "expected Float({expected:?}), got Float({f:?})"
        ),
        other => panic!("expected Some(Float({expected:?})), got {other:?}"),
    }
}

#[test]
fn jit_negate_neg_zero_float_kind_preserved() {
    // (- -0.0) → +0.0, Float — never Int(0).
    jit_expect_float_bits(&[(Op::LoadFloat(-0.0), 1), (Op::Negate, 1)], 0.0);
}

#[test]
fn jit_negate_pos_zero_float_kind_preserved() {
    // (- 0.0) → -0.0, Float — never Int(0).
    jit_expect_float_bits(&[(Op::LoadFloat(0.0), 1), (Op::Negate, 1)], -0.0);
}

#[test]
fn jit_sub_neg_zero_float_kind_preserved() {
    // (- -0.0 0) → -0.0, Float (Int operand promotes, kind stays Float).
    jit_expect_float_bits(
        &[(Op::LoadFloat(-0.0), 1), (Op::LoadInt(0), 1), (Op::Sub, 1)],
        -0.0,
    );
}

#[test]
fn jit_negate_integral_float_kind_preserved() {
    // (- 2.0) → -2.0, Float — an integral-valued float is still a float.
    jit_expect_float_bits(&[(Op::LoadFloat(2.0), 1), (Op::Negate, 1)], -2.0);
}

#[test]
fn jit_add_float_zero_result_kind_preserved() {
    // 1.5 + -1.5 → +0.0, Float.
    jit_expect_float_bits(
        &[
            (Op::LoadFloat(1.5), 1),
            (Op::LoadFloat(-1.5), 1),
            (Op::Add, 1),
        ],
        0.0,
    );
}

#[test]
fn jit_mul_float_zero_result_kind_preserved() {
    // 0.0 * 5.0 → +0.0, Float.
    jit_expect_float_bits(
        &[
            (Op::LoadFloat(0.0), 1),
            (Op::LoadFloat(5.0), 1),
            (Op::Mul, 1),
        ],
        0.0,
    );
}

#[test]
fn jit_div_float_zero_result_kind_preserved() {
    // 0.0 / 2.0 → +0.0, Float (not the exact-int-division fast path).
    jit_expect_float_bits(
        &[
            (Op::LoadFloat(0.0), 1),
            (Op::LoadFloat(2.0), 1),
            (Op::Div, 1),
        ],
        0.0,
    );
}

#[test]
fn jit_mod_float_zero_result_kind_preserved() {
    // 4.0 % 2.0 → +0.0, Float (interpreter uses the float path for Floats).
    jit_expect_float_bits(
        &[
            (Op::LoadFloat(4.0), 1),
            (Op::LoadFloat(2.0), 1),
            (Op::Mod, 1),
        ],
        0.0,
    );
}

#[test]
fn jit_pow_float_zero_result_kind_preserved() {
    // 0.0 ** 3.0 → +0.0, Float (interpreter Pow is always float).
    jit_expect_float_bits(
        &[
            (Op::LoadFloat(0.0), 1),
            (Op::LoadFloat(3.0), 1),
            (Op::Pow, 1),
        ],
        0.0,
    );
}
