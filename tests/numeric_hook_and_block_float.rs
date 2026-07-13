//! Strict numeric mode (`VM::set_numeric_hook`) and the block-JIT float result.
//!
//! Both exist for frontends whose language is not awk: elisp signals on a
//! non-numeric operand and promotes integer overflow to a bignum, where the
//! default policy coerces (`"a"` → `0.0`) and wraps. The tests below pin:
//!
//! 1. a chunk whose result is a float still returns a float once the block-JIT
//!    cache is warm (it used to truncate to an integer from the second run on);
//! 2. the default policy still coerces and wraps — zshrs/awkrs/stryke semantics
//!    are untouched by the hook's existence;
//! 3. with a hook installed, a non-numeric operand and an overflowing integer
//!    op both reach the host, *including* after the JIT has compiled the chunk,
//!    which is the case native code would otherwise silently get wrong.

#![cfg(feature = "jit")]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use fusevm::{Chunk, ChunkBuilder, NumOp, Op, VMResult, Value, VM};

/// Mirrors how elisprs drives fusevm: a fresh VM per chunk, tracing JIT on,
/// every chunk run on the same thread (so the JIT caches are shared).
fn run(chunk: Chunk, hook: Option<fusevm::NumericHook>) -> Result<Value, String> {
    let mut vm = VM::new(chunk);
    vm.enable_tracing_jit();
    if let Some(h) = hook {
        vm.set_numeric_hook(h);
    }
    match vm.run() {
        VMResult::Ok(v) => Ok(v),
        VMResult::Halted => Ok(vm.stack.last().cloned().unwrap_or(Value::Undef)),
        VMResult::Error(e) => Err(e),
    }
}

fn float_chunk(f: f64) -> Chunk {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadFloat(f), 0);
    b.build()
}

/// Push a single value the cheapest way the loader allows.
fn push_value(bd: &mut ChunkBuilder, v: Value) {
    match v {
        Value::Int(n) => bd.emit(Op::LoadInt(n), 0),
        Value::Float(f) => bd.emit(Op::LoadFloat(f), 0),
        other => {
            let idx = bd.add_constant(other);
            bd.emit(Op::LoadConst(idx), 0)
        }
    };
}

/// `OP a` as a one-constant chunk (unary).
fn unop_chunk(a: Value, op: Op) -> Chunk {
    let mut bd = ChunkBuilder::new();
    push_value(&mut bd, a);
    bd.emit(op, 0);
    bd.build()
}

/// `a OP b` as a two-constant chunk.
fn binop_chunk(a: Value, b: Value, op: Op) -> Chunk {
    let mut bd = ChunkBuilder::new();
    match a {
        Value::Int(n) => bd.emit(Op::LoadInt(n), 0),
        Value::Float(f) => bd.emit(Op::LoadFloat(f), 0),
        other => {
            let idx = bd.add_constant(other);
            bd.emit(Op::LoadConst(idx), 0)
        }
    };
    match b {
        Value::Int(n) => bd.emit(Op::LoadInt(n), 0),
        Value::Float(f) => bd.emit(Op::LoadFloat(f), 0),
        other => {
            let idx = bd.add_constant(other);
            bd.emit(Op::LoadConst(idx), 0)
        }
    };
    bd.emit(op, 0);
    bd.build()
}

/// A float chunk must keep returning a float no matter how many times it runs.
///
/// Regression: `VM::run` decoded the block tier's `i64` return register as
/// `Value::Int` unconditionally, so once the block cache warmed up (run 2), a
/// chunk loading `2.5` returned `Int(2)`. elisprs hit this as `(eval 2.5 t)`
/// evaluating to `2` the second time the same form was evaluated.
#[test]
fn block_jit_preserves_a_float_chunk_result_across_runs() {
    for i in 1..=12 {
        let v = run(float_chunk(2.5), None).expect("float chunk runs");
        assert_eq!(v, Value::Float(2.5), "run {i} lost the float");
    }
    // A different float, to prove the cache is keyed and decoded per chunk.
    assert_eq!(run(float_chunk(-0.5), None).unwrap(), Value::Float(-0.5));
}

/// No hook installed → the awk/shell contract is exactly as before: a string
/// operand coerces through `to_float`, and integer overflow wraps.
#[test]
fn default_policy_still_coerces_and_wraps() {
    let v = run(
        binop_chunk(Value::Int(1), Value::str("a".to_string()), Op::Add),
        None,
    )
    .expect("coercing add of a string does not error");
    assert_eq!(v, Value::Float(1.0));

    let v = run(
        binop_chunk(Value::Int(i64::MAX), Value::Int(1), Op::Add),
        None,
    )
    .expect("coercing add does not error on overflow");
    assert_eq!(v, Value::Int(i64::MIN), "overflow must wrap, not trap");
}

/// With a hook installed, a non-numeric operand is the host's decision.
#[test]
fn strict_mode_delegates_a_non_numeric_operand() {
    let hook: fusevm::NumericHook = Arc::new(|_op, a, b| {
        let bad = if matches!(a, Value::Int(_) | Value::Float(_)) {
            b
        } else {
            a
        };
        Err(format!("wrong-type-argument: number-or-marker-p {bad:?}"))
    });
    let err = run(
        binop_chunk(Value::Int(1), Value::str("a".to_string()), Op::Add),
        Some(hook),
    )
    .expect_err("strict add of a string must signal");
    assert!(
        err.starts_with("wrong-type-argument"),
        "hook's error must propagate verbatim, got: {err}"
    );
}

/// With a hook installed, integer overflow reaches the host so it can widen —
/// and it keeps reaching the host after the block JIT has compiled the chunk,
/// which is the case the overflow-checked lowering exists for. Without the
/// checked lowering this test fails from the warmup threshold on (native `iadd`
/// wraps silently and never calls back).
#[test]
fn strict_mode_delegates_integer_overflow_even_once_jit_compiled() {
    let calls = Arc::new(AtomicUsize::new(0));
    let seen = calls.clone();
    let hook: fusevm::NumericHook = Arc::new(move |op, a, b| {
        seen.fetch_add(1, Ordering::Relaxed);
        assert_eq!(op, NumOp::Add);
        assert_eq!((a, b), (&Value::Int(i64::MAX), &Value::Int(1)));
        // Stand in for a bignum: the host returns a value fusevm never could.
        Ok(Value::str("BIGNUM".to_string()))
    });

    // Well past the block-JIT warmup threshold (10), so the later iterations run
    // as native code.
    for i in 1..=25 {
        let v = run(
            binop_chunk(Value::Int(i64::MAX), Value::Int(1), Op::Add),
            Some(hook.clone()),
        )
        .expect("overflow is delegated, not an error");
        assert_eq!(
            v,
            Value::str("BIGNUM".to_string()),
            "run {i}: overflow escaped the hook (wrapped in native code?)"
        );
    }
    assert_eq!(
        calls.load(Ordering::Relaxed),
        25,
        "every run must delegate, including the JIT-compiled ones"
    );
}

/// Multiplication and negation overflow the same way, and a non-overflowing
/// strict chunk still returns the plain fixnum result (the checked lowering must
/// not change results, only catch the cases i64 cannot represent).
#[test]
fn strict_mode_checked_ops_are_exact_when_they_fit() {
    let hook: fusevm::NumericHook = Arc::new(|_, _, _| Ok(Value::str("BIG".to_string())));

    for i in 1..=15 {
        let v = run(
            binop_chunk(Value::Int(6), Value::Int(7), Op::Mul),
            Some(hook.clone()),
        )
        .unwrap();
        assert_eq!(v, Value::Int(42), "run {i}: checked mul changed the result");
    }
    let v = run(
        binop_chunk(Value::Int(i64::MAX), Value::Int(2), Op::Mul),
        Some(hook.clone()),
    )
    .unwrap();
    assert_eq!(
        v,
        Value::str("BIG".to_string()),
        "mul overflow must delegate"
    );
}

/// A host with tagged fixnums (Emacs: 62-bit) must see results that still fit an
/// `i64` but leave its fixnum range — `most-positive-fixnum + 1` is a bignum in
/// Emacs even though 2^61 fits an i64 twice over. The bounds check rides in the
/// same accumulator as the overflow bit, so it must survive JIT compilation too.
#[test]
fn strict_mode_delegates_results_outside_a_narrowed_fixnum_range() {
    const MOST_POSITIVE_FIXNUM: i64 = 2_305_843_009_213_693_951; // 2^61 - 1
    const MOST_NEGATIVE_FIXNUM: i64 = -2_305_843_009_213_693_952;

    let calls = Arc::new(AtomicUsize::new(0));
    let seen = calls.clone();
    let hook: fusevm::NumericHook = Arc::new(move |_op, _a, _b| {
        seen.fetch_add(1, Ordering::Relaxed);
        Ok(Value::str("BIGNUM".to_string()))
    });

    let run_ranged = |chunk: Chunk| -> Value {
        let mut vm = VM::new(chunk);
        vm.enable_tracing_jit();
        vm.set_numeric_hook(hook.clone());
        vm.set_fixnum_range(MOST_NEGATIVE_FIXNUM, MOST_POSITIVE_FIXNUM);
        match vm.run() {
            VMResult::Ok(v) => v,
            VMResult::Halted => vm.stack.last().cloned().unwrap_or(Value::Undef),
            VMResult::Error(e) => panic!("vm error: {e}"),
        }
    };

    // Past the block-JIT threshold, so the tail of this loop is native code.
    for i in 1..=25 {
        let v = run_ranged(binop_chunk(
            Value::Int(MOST_POSITIVE_FIXNUM),
            Value::Int(1),
            Op::Add,
        ));
        assert_eq!(
            v,
            Value::str("BIGNUM".to_string()),
            "run {i}: 2^61 escaped as a fixnum"
        );
    }
    assert_eq!(calls.load(Ordering::Relaxed), 25);

    // In range: still an exact native fixnum, no delegation.
    let v = run_ranged(binop_chunk(
        Value::Int(MOST_POSITIVE_FIXNUM - 1),
        Value::Int(1),
        Op::Add,
    ));
    assert_eq!(v, Value::Int(MOST_POSITIVE_FIXNUM));
    assert_eq!(
        calls.load(Ordering::Relaxed),
        25,
        "in-range must not delegate"
    );
}

/// A counting hook that stands in for a bignum. Returns the number of times it
/// was called alongside the reusable hook.
fn counting_bignum_hook() -> (Arc<AtomicUsize>, fusevm::NumericHook) {
    let calls = Arc::new(AtomicUsize::new(0));
    let seen = calls.clone();
    let hook: fusevm::NumericHook = Arc::new(move |_op, _a, _b| {
        seen.fetch_add(1, Ordering::Relaxed);
        Ok(Value::str("BIGNUM".to_string()))
    });
    (calls, hook)
}

/// `Sub` has its own opcode and its own checked lowering (`i64::checked_sub`),
/// so overflow delegation must be proved for it independently of `Add` — and it
/// must survive JIT compilation, which is the case the overflow accumulator in
/// the block tier exists for. `i64::MIN - 1` is the only interesting operand.
#[test]
fn strict_mode_delegates_subtraction_overflow_across_jit() {
    let (calls, hook) = counting_bignum_hook();
    for i in 1..=25 {
        let v = run(
            binop_chunk(Value::Int(i64::MIN), Value::Int(1), Op::Sub),
            Some(hook.clone()),
        )
        .expect("sub overflow is delegated, not an error");
        assert_eq!(v, Value::str("BIGNUM".to_string()), "run {i}: sub overflow wrapped");
    }
    assert_eq!(calls.load(Ordering::Relaxed), 25, "every sub overflow must delegate");
}

/// Unary negate is the fourth block-eligible op. `-i64::MIN` overflows (`i64`
/// has no positive `MIN`), so strict mode must hand it to the host on every
/// run, JIT-compiled ones included. An in-range negation stays an exact fixnum
/// and never touches the hook.
#[test]
fn strict_mode_delegates_negate_overflow_but_not_in_range() {
    let (calls, hook) = counting_bignum_hook();

    for i in 1..=25 {
        let v = run(unop_chunk(Value::Int(i64::MIN), Op::Negate), Some(hook.clone()))
            .expect("negate overflow is delegated");
        assert_eq!(v, Value::str("BIGNUM".to_string()), "run {i}: -i64::MIN wrapped");
    }
    assert_eq!(calls.load(Ordering::Relaxed), 25, "every -i64::MIN must delegate");

    // A representable negation must not delegate — the checked lowering only
    // catches what i64 cannot hold, it never changes an in-range result.
    for i in 1..=15 {
        let v = run(unop_chunk(Value::Int(42), Op::Negate), Some(hook.clone())).unwrap();
        assert_eq!(v, Value::Int(-42), "run {i}: in-range negate changed");
    }
    assert_eq!(calls.load(Ordering::Relaxed), 25, "in-range negate must not delegate");
}

/// A non-number operand is not negatable in a strict language, so it reaches the
/// host (elisp: `(wrong-type-argument number-or-marker-p "a")`).
#[test]
fn strict_mode_delegates_negate_of_a_non_number() {
    let err = run(
        unop_chunk(Value::str("a".to_string()), Op::Negate),
        Some(Arc::new(|op, a, b| {
            assert_eq!(op, NumOp::Neg);
            assert_eq!((a, b), (&Value::str("a".to_string()), &Value::Undef));
            Err("wrong-type-argument".to_string())
        })),
    )
    .expect_err("negating a string must signal in strict mode");
    assert_eq!(err, "wrong-type-argument");
}

/// The lower fixnum bound is enforced by the same accumulator as the upper one.
/// `most-negative-fixnum - 1` fits an `i64` but leaves Emacs's 62-bit range, so
/// it must delegate; one step inside the range stays an exact native fixnum.
#[test]
fn strict_mode_delegates_below_the_narrowed_fixnum_range() {
    const MOST_POSITIVE_FIXNUM: i64 = 2_305_843_009_213_693_951; // 2^61 - 1
    const MOST_NEGATIVE_FIXNUM: i64 = -2_305_843_009_213_693_952; // -2^61

    let (calls, hook) = counting_bignum_hook();
    let run_ranged = |chunk: Chunk| -> Value {
        let mut vm = VM::new(chunk);
        vm.enable_tracing_jit();
        vm.set_numeric_hook(hook.clone());
        vm.set_fixnum_range(MOST_NEGATIVE_FIXNUM, MOST_POSITIVE_FIXNUM);
        match vm.run() {
            VMResult::Ok(v) => v,
            VMResult::Halted => vm.stack.last().cloned().unwrap_or(Value::Undef),
            VMResult::Error(e) => panic!("vm error: {e}"),
        }
    };

    for i in 1..=25 {
        let v = run_ranged(binop_chunk(
            Value::Int(MOST_NEGATIVE_FIXNUM),
            Value::Int(1),
            Op::Sub,
        ));
        assert_eq!(v, Value::str("BIGNUM".to_string()), "run {i}: -2^61-1 escaped as fixnum");
    }
    assert_eq!(calls.load(Ordering::Relaxed), 25);

    let v = run_ranged(binop_chunk(
        Value::Int(MOST_NEGATIVE_FIXNUM + 1),
        Value::Int(1),
        Op::Sub,
    ));
    assert_eq!(v, Value::Int(MOST_NEGATIVE_FIXNUM));
    assert_eq!(calls.load(Ordering::Relaxed), 25, "in-range must not delegate");
}

/// Comparison can never overflow, so `cmp_int_fast` delegates only a non-numeric
/// operand — and two fixnums compare natively without ever touching the hook.
#[test]
fn strict_mode_comparison_delegates_only_non_numbers() {
    // Non-numeric operand → the host decides the ordering (elisp signals).
    let err = run(
        binop_chunk(Value::Int(1), Value::str("a".to_string()), Op::NumLt),
        Some(Arc::new(|op, _a, _b| {
            assert_eq!(op, NumOp::Lt);
            Err("wrong-type-argument".to_string())
        })),
    )
    .expect_err("comparing against a string must signal in strict mode");
    assert_eq!(err, "wrong-type-argument");

    // Two fixnums: native comparison, hook untouched even after JIT warmup.
    let (calls, hook) = counting_bignum_hook();
    for i in 1..=25 {
        let v = run(
            binop_chunk(Value::Int(3), Value::Int(7), Op::NumLt),
            Some(hook.clone()),
        )
        .unwrap();
        assert_eq!(v, Value::Bool(true), "run {i}: 3 < 7 wrong");
    }
    assert_eq!(calls.load(Ordering::Relaxed), 0, "numeric comparison must not delegate");
}

/// `Div` and `Pow` are float-native with no overflow case, so strict mode
/// delegates only a non-numeric operand; a numeric divide stays an exact float
/// and never reaches the host.
#[test]
fn strict_mode_div_and_pow_delegate_only_non_numbers() {
    for op in [Op::Div, Op::Pow] {
        let name = format!("{op:?}");
        let (calls, hook) = counting_bignum_hook();
        let v = run(
            binop_chunk(Value::Int(2), Value::str("a".to_string()), op),
            Some(hook.clone()),
        )
        .expect("delegated op returns the host value");
        assert_eq!(v, Value::str("BIGNUM".to_string()), "{name} of a string must delegate");
        assert_eq!(calls.load(Ordering::Relaxed), 1);
    }

    // Numeric divide is exact in f64 — no delegation.
    let (calls, hook) = counting_bignum_hook();
    let v = run(binop_chunk(Value::Int(7), Value::Int(2), Op::Div), Some(hook)).unwrap();
    assert_eq!(v, Value::Float(3.5));
    assert_eq!(calls.load(Ordering::Relaxed), 0, "numeric divide must not delegate");
}

/// The hook is for integer overflow and non-numbers only: mixed int/float and
/// float/float arithmetic is exact in `f64` and must stay on the fast path, even
/// with a hook installed and the JIT warm. A hook that panics proves it is never
/// consulted for float operands.
#[test]
fn strict_mode_never_delegates_exact_float_arithmetic() {
    let hook: fusevm::NumericHook =
        Arc::new(|op, a, b| panic!("float arithmetic delegated: {op:?} {a:?} {b:?}"));

    for i in 1..=25 {
        // i64::MAX as a float can't overflow f64, and one operand is a float, so
        // this is the mixed path — never the checked-int path.
        let v = run(
            binop_chunk(Value::Int(i64::MAX), Value::Float(2.0), Op::Add),
            Some(hook.clone()),
        )
        .expect("mixed float add stays native");
        assert_eq!(v, Value::Float(i64::MAX as f64 + 2.0), "run {i}");
    }
}
