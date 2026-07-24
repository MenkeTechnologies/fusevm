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
        assert_eq!(
            v,
            Value::str("BIGNUM".to_string()),
            "run {i}: sub overflow wrapped"
        );
    }
    assert_eq!(
        calls.load(Ordering::Relaxed),
        25,
        "every sub overflow must delegate"
    );
}

/// Unary negate is the fourth block-eligible op. `-i64::MIN` overflows (`i64`
/// has no positive `MIN`), so strict mode must hand it to the host on every
/// run, JIT-compiled ones included. An in-range negation stays an exact fixnum
/// and never touches the hook.
#[test]
fn strict_mode_delegates_negate_overflow_but_not_in_range() {
    let (calls, hook) = counting_bignum_hook();

    for i in 1..=25 {
        let v = run(
            unop_chunk(Value::Int(i64::MIN), Op::Negate),
            Some(hook.clone()),
        )
        .expect("negate overflow is delegated");
        assert_eq!(
            v,
            Value::str("BIGNUM".to_string()),
            "run {i}: -i64::MIN wrapped"
        );
    }
    assert_eq!(
        calls.load(Ordering::Relaxed),
        25,
        "every -i64::MIN must delegate"
    );

    // A representable negation must not delegate — the checked lowering only
    // catches what i64 cannot hold, it never changes an in-range result.
    for i in 1..=15 {
        let v = run(unop_chunk(Value::Int(42), Op::Negate), Some(hook.clone())).unwrap();
        assert_eq!(v, Value::Int(-42), "run {i}: in-range negate changed");
    }
    assert_eq!(
        calls.load(Ordering::Relaxed),
        25,
        "in-range negate must not delegate"
    );
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
        assert_eq!(
            v,
            Value::str("BIGNUM".to_string()),
            "run {i}: -2^61-1 escaped as fixnum"
        );
    }
    assert_eq!(calls.load(Ordering::Relaxed), 25);

    let v = run_ranged(binop_chunk(
        Value::Int(MOST_NEGATIVE_FIXNUM + 1),
        Value::Int(1),
        Op::Sub,
    ));
    assert_eq!(v, Value::Int(MOST_NEGATIVE_FIXNUM));
    assert_eq!(
        calls.load(Ordering::Relaxed),
        25,
        "in-range must not delegate"
    );
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
    assert_eq!(
        calls.load(Ordering::Relaxed),
        0,
        "numeric comparison must not delegate"
    );
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
        assert_eq!(
            v,
            Value::str("BIGNUM".to_string()),
            "{name} of a string must delegate"
        );
        assert_eq!(calls.load(Ordering::Relaxed), 1);
    }

    // Numeric divide is exact in f64 — no delegation.
    let (calls, hook) = counting_bignum_hook();
    let v = run(
        binop_chunk(Value::Int(7), Value::Int(2), Op::Div),
        Some(hook),
    )
    .unwrap();
    assert_eq!(v, Value::Float(3.5));
    assert_eq!(
        calls.load(Ordering::Relaxed),
        0,
        "numeric divide must not delegate"
    );
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

/// Build a strict-mode accumulator loop, the shape a compiled frontend emits for
/// `for i in range(limit): acc += step`:
///
/// ```text
///   LoadInt(0);    SetSlot(0)          // i = 0
///   LoadInt(start); SetSlot(1)         // acc = start
/// anchor:
///   GetSlot(1); LoadInt(step); Add; SetSlot(1)
///   GetSlot(0); LoadInt(1);    Add; SetSlot(0)
///   GetSlot(0); LoadInt(limit); NumLt; JumpIfTrue(anchor)
///   GetSlot(1)                         // result = acc
/// ```
///
/// Only ops a strict VM keeps JIT-eligible are used (no `Inc`/`PreIncSlotVoid`),
/// so the loop really does reach the trace tier. Returns (chunk, anchor_ip).
fn strict_accum_loop(start: i64, step: i64, limit: i64) -> (Chunk, usize) {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(0), 1);
    b.emit(Op::SetSlot(0), 1);
    b.emit(Op::LoadInt(start), 1);
    b.emit(Op::SetSlot(1), 1);
    let anchor = b.current_pos();
    b.emit(Op::GetSlot(1), 1);
    b.emit(Op::LoadInt(step), 1);
    b.emit(Op::Add, 1);
    b.emit(Op::SetSlot(1), 1);
    b.emit(Op::GetSlot(0), 1);
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::Add, 1);
    b.emit(Op::SetSlot(0), 1);
    b.emit(Op::GetSlot(0), 1);
    b.emit(Op::LoadInt(limit), 1);
    b.emit(Op::NumLt, 1);
    let jmp = b.emit(Op::JumpIfTrue(0), 1);
    b.patch_jump(jmp, anchor);
    b.emit(Op::GetSlot(1), 1);
    (b.build(), anchor)
}

fn run_with_slots(chunk: Chunk, hook: fusevm::NumericHook, slots: usize) -> Value {
    let mut vm = VM::new(chunk);
    vm.enable_tracing_jit();
    vm.set_numeric_hook(hook);
    {
        let frame = vm.frames.last_mut().unwrap();
        while frame.slots.len() < slots {
            frame.slots.push(Value::Int(0));
        }
    }
    match vm.run() {
        VMResult::Ok(v) => v,
        VMResult::Halted => vm.stack.last().cloned().unwrap_or(Value::Undef),
        VMResult::Error(e) => panic!("vm error: {e}"),
    }
}

/// Integer overflow inside a *compiled trace* must reach the hook, exactly as it
/// does in the interpreter and in the block tier.
///
/// This is the loop tier, not the straight-line one the tests above cover: the
/// accumulator overflows only after the loop is hot, so the overflowing `Add`
/// executes in native trace code. That code wrapped silently — a frontend with
/// bignums (pythonrs: `sum(i*i*i)`) got an `i64`-wrapped answer with no
/// diagnostic, while the same program run below the trace threshold was
/// correct. The trace now carries the block tier's overflow accumulator and
/// bails to the interpreter, where the hook runs.
#[test]
fn strict_mode_delegates_overflow_inside_a_compiled_trace() {
    let (calls, hook) = counting_bignum_hook();
    // 400 iterations: past the trace threshold, and the accumulator crosses
    // i64::MAX at iteration ~301 — long after the loop is running natively.
    let step = 1_000_000_000_000_000;
    let start = i64::MAX - step * 300;
    let (chunk, _anchor) = strict_accum_loop(start, step, 400);

    let v = run_with_slots(chunk, hook, 2);
    assert_eq!(
        v,
        Value::str("BIGNUM".to_string()),
        "overflow in a hot loop must delegate to the hook, not wrap to an i64"
    );
    assert!(
        calls.load(Ordering::Relaxed) > 0,
        "the hook was never consulted — the trace wrapped silently"
    );
}

/// The same loop below the overflow point must stay entirely native: exact
/// `i64` arithmetic, hook never consulted. Guards against "fix overflow by
/// delegating everything", which would erase the JIT's reason to exist.
#[test]
fn strict_mode_hot_loop_without_overflow_never_delegates() {
    let hook: fusevm::NumericHook =
        Arc::new(|op, a, b| panic!("exact i64 arithmetic delegated: {op:?} {a:?} {b:?}"));
    let (chunk, _anchor) = strict_accum_loop(0, 7, 400);
    let v = run_with_slots(chunk, hook, 2);
    assert_eq!(v, Value::Int(7 * 400));
}

/// `Op::Mod` by a nonzero constant `|k| >= 2` is JIT-eligible in strict mode:
/// `checked_rem` cannot fail for such a divisor, so native `srem` and the
/// interpreter agree exactly and the hook is never needed. A loop full of them
/// must produce the interpreter's answer without a single delegation.
#[test]
fn strict_mode_compiles_modulo_by_a_constant_divisor() {
    let hook: fusevm::NumericHook =
        Arc::new(|op, a, b| panic!("constant-divisor modulo delegated: {op:?} {a:?} {b:?}"));

    // acc += i % 7, over 400 iterations.
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(0), 1);
    b.emit(Op::SetSlot(0), 1);
    b.emit(Op::LoadInt(0), 1);
    b.emit(Op::SetSlot(1), 1);
    let anchor = b.current_pos();
    b.emit(Op::GetSlot(1), 1);
    b.emit(Op::GetSlot(0), 1);
    b.emit(Op::LoadInt(7), 1);
    b.emit(Op::Mod, 1);
    b.emit(Op::Add, 1);
    b.emit(Op::SetSlot(1), 1);
    b.emit(Op::GetSlot(0), 1);
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::Add, 1);
    b.emit(Op::SetSlot(0), 1);
    b.emit(Op::GetSlot(0), 1);
    b.emit(Op::LoadInt(400), 1);
    b.emit(Op::NumLt, 1);
    let jmp = b.emit(Op::JumpIfTrue(0), 1);
    b.patch_jump(jmp, anchor);
    b.emit(Op::GetSlot(1), 1);
    let _ = anchor;

    let expected: i64 = (0..400).map(|i: i64| i % 7).sum();
    let v = run_with_slots(b.build(), hook, 2);
    assert_eq!(v, Value::Int(expected));
}

/// `Op::MulModFloor` in a hot loop under a strict VM: the products overflow `i64`
/// every iteration, and the whole point of the op is that this needs neither a
/// bignum nor an overflow bail. The hook panics if consulted.
#[test]
fn strict_mode_mulmod_never_delegates_and_stays_exact() {
    let hook: fusevm::NumericHook =
        Arc::new(|op, a, b| panic!("exact mulmod_floor delegated: {op:?} {a:?} {b:?}"));

    // acc = (acc + (i * 6364136223846793005) % 1000000007) over 400 iterations.
    const MUL: i64 = 6_364_136_223_846_793_005;
    const K: i64 = 1_000_000_007;
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(0), 1);
    b.emit(Op::SetSlot(0), 1);
    b.emit(Op::LoadInt(0), 1);
    b.emit(Op::SetSlot(1), 1);
    let anchor = b.current_pos();
    b.emit(Op::GetSlot(1), 1);
    b.emit(Op::GetSlot(0), 1);
    b.emit(Op::LoadInt(MUL), 1);
    b.emit(Op::LoadInt(K), 1);
    b.emit(Op::MulModFloor, 1);
    b.emit(Op::Add, 1);
    b.emit(Op::SetSlot(1), 1);
    b.emit(Op::GetSlot(0), 1);
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::Add, 1);
    b.emit(Op::SetSlot(0), 1);
    b.emit(Op::GetSlot(0), 1);
    b.emit(Op::LoadInt(400), 1);
    b.emit(Op::NumLt, 1);
    let jmp = b.emit(Op::JumpIfTrue(0), 1);
    b.patch_jump(jmp, anchor);
    b.emit(Op::GetSlot(1), 1);

    // Floored remainder, exactly as the op documents.
    let expected: i64 = (0..400i64)
        .map(|i| fusevm::floor_rem_i128(i as i128 * MUL as i128, K))
        .sum();
    let v = run_with_slots(b.build(), hook, 2);
    assert_eq!(v, Value::Int(expected));
}

/// A non-integer operand must take the unfused `Mul`-then-`Mod` path so the hook
/// sees the same two ops it would have seen without the fusion — that is what
/// keeps a frontend's bignum/`__mul__` semantics intact when the fusion fires on
/// a value that turns out not to be an integer.
#[test]
fn mulmod_with_a_non_integer_operand_replays_the_unfused_ops() {
    let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
    let log = seen.clone();
    let hook: fusevm::NumericHook = Arc::new(move |op, _a, _b| {
        log.lock().unwrap().push(op);
        Ok(Value::Int(7))
    });

    let mut b = ChunkBuilder::new();
    let c = b.add_constant(Value::str("x"));
    b.emit(Op::LoadConst(c), 1); // a string: not a native number
    b.emit(Op::LoadInt(3), 1);
    b.emit(Op::LoadInt(5), 1);
    b.emit(Op::MulModFloor, 1);

    let v = run(b.build(), Some(hook)).expect("delegated mulmod_floor runs");
    // Mul delegated (string operand) -> 7; then 7 % 5 is native = 2.
    assert_eq!(v, Value::Int(2));
    assert_eq!(
        *seen.lock().unwrap(),
        vec![NumOp::Mul],
        "the fusion must delegate the Mul exactly once, then reduce natively"
    );
}
