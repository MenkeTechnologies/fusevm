//! Strict numeric mode must not answer from a rounded `f64`.
//!
//! fusevm's native comparison and arithmetic arms take a shortcut when both
//! operands are `Int` or `Float`: convert both to `f64` and compute there. For
//! `Float`/`Float` that is exact by construction, and for `Int`/`Float` it is
//! exact *while the integer is one an `f64` can hold*. Past `2^53` the
//! conversion rounds, and the shortcut then answers about the rounded value
//! rather than the one the program has.
//!
//! In the coercing (awk/shell) policy that is the defined behaviour. In strict
//! numeric mode it is a silent wrong answer, because the host — the only party
//! that can represent the integer exactly — never sees the operands:
//!
//! ```text
//! 3**34 == (3**34).to_f     MRI: false     fusevm native shortcut: true
//! ```
//!
//! `3**34` is 16_677_181_699_666_569; `(3**34) as f64` is
//! 16_677_181_699_666_568. The two are different integers and the comparison
//! is `false`, but both sides collapse onto the same `f64`.
//!
//! What these pin:
//!
//! 1. a strict host is handed the operands whenever an integer operand is
//!    outside the exactly-representable range, for every comparison op and
//!    every arithmetic op that routes through the native shortcut;
//! 2. the coercing policy is untouched — no hook, no delegation, and the same
//!    `f64` answer as before;
//! 3. inside the range the shortcut still runs, so strict mode keeps its fast
//!    path for ordinary mixed arithmetic;
//! 4. the guarantee holds under the JIT tiers too, not just the interpreter.

#![cfg(feature = "jit")]

use std::sync::{Arc, Mutex};

use fusevm::{ChunkBuilder, NumOp, Op, VMResult, Value, VM};

/// `3**34` — the smallest power of three above `2^53`, and the case rubylang
/// measured. Its `f64` image is one less than the integer itself.
const P34: i64 = 16_677_181_699_666_569;
/// `(3**34) as f64`, written out so the test does not depend on the cast it is
/// testing.
const P34_F: f64 = 16_677_181_699_666_568.0;

/// Every `(op, a, b)` the VM delegated.
type Log = Arc<Mutex<Vec<(NumOp, Value, Value)>>>;

/// A host that answers exactly: it compares an `Int` against a `Float` by
/// value, the way a bignum-capable frontend does, and marks any arithmetic it
/// is handed with a sentinel so the test can tell delegation from the native
/// shortcut.
fn exact_host(log: &Log) -> fusevm::NumericHook {
    let log = Arc::clone(log);
    Arc::new(move |op, a: &Value, b: &Value| {
        log.lock()
            .expect("log")
            .push((op, a.clone(), b.clone()));
        use NumOp::*;
        // Exact Int-vs-Float ordering: compare in the integer domain by
        // splitting the float into its floor and its fractional part, so no
        // rounding of the integer ever happens.
        let ord = match (a, b) {
            (Value::Int(x), Value::Float(y)) => exact_cmp_int_float(*x, *y),
            (Value::Float(x), Value::Int(y)) => exact_cmp_int_float(*y, *x).map(|o| o.reverse()),
            _ => None,
        };
        if let (Some(ord), Lt | Gt | Le | Ge | Eq | Ne) = (ord, op) {
            use std::cmp::Ordering::*;
            return Ok(Value::Bool(match op {
                Lt => ord == Less,
                Gt => ord == Greater,
                Le => ord != Greater,
                Ge => ord != Less,
                Eq => ord == Equal,
                Ne => ord != Equal,
                _ => unreachable!(),
            }));
        }
        // Arithmetic: a sentinel, so reaching the host is observable in the
        // program's result and not only in the log.
        Ok(Value::str("HOST".to_string()))
    })
}

/// Compare integer `x` against float `y` without rounding `x`.
fn exact_cmp_int_float(x: i64, y: f64) -> Option<std::cmp::Ordering> {
    if y.is_nan() {
        return None;
    }
    if y >= 9_223_372_036_854_775_808.0 {
        return Some(std::cmp::Ordering::Less);
    }
    if y < -9_223_372_036_854_775_808.0 {
        return Some(std::cmp::Ordering::Greater);
    }
    let floor = y.floor();
    let yi = floor as i64;
    Some(x.cmp(&yi).then(if y > floor {
        std::cmp::Ordering::Less
    } else {
        std::cmp::Ordering::Equal
    }))
}

/// Build `a <op> b` as a two-operand chunk.
fn chunk_for(a: Value, b: Value, op: Op) -> fusevm::Chunk {
    let mut b_ = ChunkBuilder::new();
    for v in [a, b] {
        match v {
            Value::Int(n) => {
                b_.emit(Op::LoadInt(n), 1);
            }
            other => {
                let k = b_.add_constant(other);
                b_.emit(Op::LoadConst(k), 1);
            }
        }
    }
    b_.emit(op, 1);
    b_.build()
}

/// Run `a <op> b` in strict mode; return the result and the delegation log.
fn run_strict(a: Value, b: Value, op: Op, tracing: bool) -> (Value, Vec<(NumOp, Value, Value)>) {
    let log: Log = Arc::new(Mutex::new(Vec::new()));
    let mut vm = VM::new(chunk_for(a, b, op));
    vm.set_numeric_hook(exact_host(&log));
    if tracing {
        vm.enable_tracing_jit();
    }
    let out = match vm.run() {
        VMResult::Ok(v) => v,
        VMResult::Halted => vm.stack.last().cloned().unwrap_or(Value::Undef),
        VMResult::Error(e) => panic!("vm error: {e}"),
    };
    let seen = log.lock().expect("log").clone();
    (out, seen)
}

/// Run `a <op> b` with no hook — the coercing policy.
fn run_coercing(a: Value, b: Value, op: Op) -> Value {
    let mut vm = VM::new(chunk_for(a, b, op));
    match vm.run() {
        VMResult::Ok(v) => v,
        VMResult::Halted => vm.stack.last().cloned().unwrap_or(Value::Undef),
        VMResult::Error(e) => panic!("vm error: {e}"),
    }
}

// ── 1. the comparison arm ────────────────────────────────────────────────────

#[test]
fn a_strict_host_decides_an_int_float_comparison_it_alone_can_answer() {
    // The measured case: `3**34 == (3**34).to_f` is false, and is true if the
    // integer is rounded into an f64 first.
    for tracing in [false, true] {
        let (out, seen) = run_strict(
            Value::Int(P34),
            Value::Float(P34_F),
            Op::NumEq,
            tracing,
        );
        assert_eq!(
            seen.len(),
            1,
            "tracing={tracing}: the host must be handed the operands, got {seen:?}"
        );
        assert_eq!(seen[0].0, NumOp::Eq);
        assert_eq!(
            out,
            Value::Bool(false),
            "tracing={tracing}: {P34} != {P34_F}, and only the host can tell"
        );
    }
}

#[test]
fn every_comparison_op_delegates_past_the_exact_range() {
    // `<= < >= > == !=` all take the same shortcut, so all six must decline it.
    let cases = [
        (Op::NumEq, NumOp::Eq, false),
        (Op::NumNe, NumOp::Ne, true),
        (Op::NumLt, NumOp::Lt, false),
        (Op::NumGt, NumOp::Gt, true),
        (Op::NumLe, NumOp::Le, false),
        (Op::NumGe, NumOp::Ge, true),
    ];
    for (op, numop, expected) in cases {
        let (out, seen) = run_strict(Value::Int(P34), Value::Float(P34_F), op, false);
        assert_eq!(seen.len(), 1, "{numop:?}: not delegated, got {seen:?}");
        assert_eq!(seen[0].0, numop, "{numop:?}: wrong op reached the host");
        assert_eq!(
            out,
            Value::Bool(expected),
            "{numop:?}: {P34} vs {P34_F} answered from the rounded value"
        );
    }
}

#[test]
fn the_float_operand_may_be_on_either_side() {
    // The arm is symmetric, and so is the defect.
    let (out, seen) = run_strict(Value::Float(P34_F), Value::Int(P34), Op::NumEq, false);
    assert_eq!(seen.len(), 1, "Float-first must delegate too: {seen:?}");
    assert_eq!(out, Value::Bool(false));
}

// ── 2. the arithmetic arm ────────────────────────────────────────────────────

#[test]
fn arithmetic_on_an_inexact_integer_reaches_the_host() {
    // Same false premise, same fix: a sum computed on a rounded operand is as
    // silent as a comparison answered on one.
    for (op, numop) in [
        (Op::Add, NumOp::Add),
        (Op::Sub, NumOp::Sub),
        (Op::Mul, NumOp::Mul),
        (Op::Mod, NumOp::Mod),
    ] {
        let (out, seen) = run_strict(Value::Int(P34), Value::Float(2.0), op, false);
        assert_eq!(seen.len(), 1, "{numop:?}: not delegated, got {seen:?}");
        assert_eq!(seen[0].0, numop);
        assert_eq!(
            out,
            Value::str("HOST".to_string()),
            "{numop:?}: answered natively from a rounded operand"
        );
    }
}

// ── 3. the range boundary ────────────────────────────────────────────────────

#[test]
fn the_shortcut_still_runs_while_the_integer_is_exact() {
    // 2^53 is the last integer f64 holds exactly along with all its
    // predecessors, so it and everything below stay on the fast path.
    let exact = (1i64 << 53) - 1;
    let (out, seen) = run_strict(Value::Int(exact), Value::Float(exact as f64), Op::NumEq, false);
    assert!(
        seen.is_empty(),
        "an exactly-representable integer must not cost a host call: {seen:?}"
    );
    assert_eq!(out, Value::Bool(true));

    // Float/Float never rounds an operand and never delegates, at any size.
    let (out, seen) = run_strict(Value::Float(1e300), Value::Float(1e300), Op::NumEq, false);
    assert!(seen.is_empty(), "float/float must stay native: {seen:?}");
    assert_eq!(out, Value::Bool(true));
}

#[test]
fn i64_min_does_not_panic_the_range_check() {
    // `i64::MIN.abs()` overflows; the check must use an unsigned magnitude.
    // (-2^63 is exactly representable, so delegating here is conservative, not
    // required — what matters is that asking the question is safe.)
    let (_out, seen) = run_strict(Value::Int(i64::MIN), Value::Float(0.0), Op::NumLt, false);
    assert_eq!(seen.len(), 1, "i64::MIN must be handled, not panicked on");
}

// ── 4. the native tiers must not answer past the interpreter ────────────────

/// `slot0 <op> slot1`, straight-line and so block-JIT eligible.
fn slot_chunk(op: Op) -> fusevm::Chunk {
    let mut b = ChunkBuilder::new();
    b.emit(Op::GetSlot(0), 1);
    b.emit(Op::GetSlot(1), 1);
    b.emit(op, 1);
    b.build()
}

/// Run `slot0 <op> slot1` `iters` times with the block/tracing JIT on, and
/// return how many of those runs reached the host plus every distinct result.
fn drive_jit(op: Op, a: Value, b: Value, iters: usize) -> (usize, Vec<String>) {
    let calls = Arc::new(Mutex::new(0usize));
    let mut distinct: Vec<String> = Vec::new();
    for _ in 0..iters {
        let c = Arc::clone(&calls);
        let mut vm = VM::new(slot_chunk(op.clone()));
        vm.set_numeric_hook(Arc::new(move |_op, _a: &Value, _b: &Value| {
            *c.lock().expect("count") += 1;
            Ok(Value::str("HOST"))
        }));
        vm.enable_tracing_jit();
        vm.set_slot(0, a.clone());
        vm.set_slot(1, b.clone());
        let out = match vm.run() {
            VMResult::Ok(v) => v,
            VMResult::Halted => vm.stack.last().cloned().unwrap_or(Value::Undef),
            VMResult::Error(e) => panic!("vm error: {e}"),
        };
        let s = format!("{out:?}");
        if !distinct.contains(&s) {
            distinct.push(s);
        }
    }
    let n = *calls.lock().expect("count");
    (n, distinct)
}

#[test]
fn a_compiled_block_cannot_outvote_the_host() {
    // The block JIT lowers a mixed Int/Float pair with `fcvt_from_sint`, which
    // rounds the same way the interpreter's `to_float` did. Before this was
    // declined the host was called on the first run and never again: the
    // chunk warmed up, compiled, and answered natively from then on —
    // `host_calls=1 over 30`, with the compiled answer differing from the
    // interpreted one. The tier must not be able to overrule the host.
    let iters = 30;
    for (op, label) in [(Op::Add, "Add"), (Op::Mul, "Mul"), (Op::Sub, "Sub")] {
        let (calls, distinct) = drive_jit(op, Value::Int(P34), Value::Float(2.0), iters);
        assert_eq!(
            calls, iters,
            "{label}: the host must be reached on every run, not only before the \
             block compiled — got {calls}/{iters}, results {distinct:?}"
        );
        assert_eq!(
            distinct.len(),
            1,
            "{label}: the answer changed once native code took over: {distinct:?}"
        );
    }
}

#[test]
fn warming_up_does_not_change_an_in_range_answer() {
    // The decline is on the compile-time kinds, so an all-int chunk keeps its
    // native code. This pins that the fast path still exists: no host call at
    // all, before or after warmup.
    let (calls, distinct) = drive_jit(Op::Add, Value::Int(2), Value::Int(3), 30);
    assert_eq!(calls, 0, "an int/int chunk must never delegate: {distinct:?}");
    assert_eq!(distinct, vec!["Int(5)".to_string()]);

    // Float/float likewise never rounds an operand and stays compiled.
    let (calls, distinct) = drive_jit(Op::Add, Value::Float(1.5), Value::Float(2.5), 30);
    assert_eq!(calls, 0, "a float/float chunk must never delegate: {distinct:?}");
    assert_eq!(distinct, vec!["Float(4.0)".to_string()]);
}

// ── 5. the coercing policy is untouched ──────────────────────────────────────

#[test]
fn without_a_hook_the_answer_is_the_f64_one_it_always_was() {
    // awk/shell semantics are defined in terms of doubles: `3**34` and its f64
    // image *are* the same number there. No hook, no delegation, no change.
    assert_eq!(
        run_coercing(Value::Int(P34), Value::Float(P34_F), Op::NumEq),
        Value::Bool(true),
        "the coercing policy must keep answering in f64"
    );
    assert_eq!(
        run_coercing(Value::Int(P34), Value::Float(2.0), Op::Add),
        Value::Float(P34 as f64 + 2.0),
        "the coercing policy must keep computing in f64"
    );
}
