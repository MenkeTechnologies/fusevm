//! `Chunk::nan_result_hook`: in strict numeric mode, a native float
//! `Add`/`Sub`/`Mul` whose result is NaN goes to the numeric hook.
//!
//! tclsh reports `domain error: argument not in valid range` for `inf - inf`,
//! where IEEE-754 (and every other fusevm frontend) answers NaN. The tests pin
//! that the opt-in reaches the hook in the interpreter and from inside JIT'd
//! code, and that without the flag nothing changes.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use fusevm::{Chunk, ChunkBuilder, NumOp, Op, VMResult, Value, VM};

const DOMAIN: &str = "domain error: argument not in valid range";

/// A hook that refuses a NaN result the way tclrs does and counts its calls.
/// Anything else reaching it is a test failure: these chunks hold only floats.
fn domain_hook() -> (Arc<AtomicUsize>, fusevm::NumericHook) {
    let calls = Arc::new(AtomicUsize::new(0));
    let seen = calls.clone();
    let hook: fusevm::NumericHook = Arc::new(move |op, a, b| {
        seen.fetch_add(1, Ordering::Relaxed);
        let (Value::Float(x), Value::Float(y)) = (a, b) else {
            panic!("non-float operands delegated: {op:?} {a:?} {b:?}");
        };
        let r = match op {
            NumOp::Add => x + y,
            NumOp::Sub => x - y,
            NumOp::Mul => x * y,
            other => panic!("unexpected op delegated: {other:?}"),
        };
        assert!(r.is_nan(), "a non-NaN result was delegated: {op:?} {x} {y}");
        Err(DOMAIN.to_string())
    });
    (calls, hook)
}

/// `(1e308 * 10.0) OP (1e308 * 10.0)`: both sides overflow to `inf` from
/// finite constants, so the chunk stays JIT-eligible (a non-finite
/// `LoadFloat` would make every tier decline it).
fn inf_op_inf(op: Op, flag: bool) -> Chunk {
    let mut b = ChunkBuilder::new();
    for _ in 0..2 {
        b.emit(Op::LoadFloat(1e308), 0);
        b.emit(Op::LoadFloat(10.0), 0);
        b.emit(Op::Mul, 0);
    }
    b.emit(op, 0);
    b.set_nan_result_hook(flag);
    b.build()
}

fn run(chunk: Chunk, hook: Option<fusevm::NumericHook>, jit: bool) -> Result<Value, String> {
    let mut vm = VM::new(chunk);
    // Only the jit-gated tests pass `true`.
    #[cfg(feature = "jit")]
    if jit {
        vm.enable_tracing_jit();
    }
    #[cfg(not(feature = "jit"))]
    let _ = jit;
    if let Some(h) = hook {
        vm.set_numeric_hook(h);
    }
    match vm.run() {
        VMResult::Ok(v) => Ok(v),
        VMResult::Halted => Ok(vm.stack.last().cloned().unwrap_or(Value::Undef)),
        VMResult::Error(e) => Err(e),
    }
}

fn is_nan(v: &Value) -> bool {
    matches!(v, Value::Float(f) if f.is_nan())
}

#[test]
fn interpreter_hands_a_nan_result_to_the_hook() {
    // inf - inf, and inf + -inf via Sub of a negated side is the same case; Mul
    // gets its own NaN shape below.
    let (calls, hook) = domain_hook();
    assert_eq!(
        run(inf_op_inf(Op::Sub, true), Some(hook.clone()), false),
        Err(DOMAIN.to_string())
    );
    // inf * 0.0
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadFloat(1e308), 0);
    b.emit(Op::LoadFloat(10.0), 0);
    b.emit(Op::Mul, 0);
    b.emit(Op::LoadFloat(0.0), 0);
    b.emit(Op::Mul, 0);
    b.set_nan_result_hook(true);
    assert_eq!(run(b.build(), Some(hook), false), Err(DOMAIN.to_string()));
    assert_eq!(calls.load(Ordering::Relaxed), 2);
}

#[test]
fn without_the_flag_a_nan_result_is_answered_as_before() {
    let (calls, hook) = domain_hook();
    let v = run(inf_op_inf(Op::Sub, false), Some(hook), false).expect("no delegation");
    assert!(is_nan(&v), "expected NaN, got {v:?}");
    assert_eq!(calls.load(Ordering::Relaxed), 0);
}

#[test]
fn the_flag_without_a_hook_changes_nothing() {
    // Coercing mode has no hook to hand anything to.
    let v = run(inf_op_inf(Op::Sub, true), None, false).expect("no hook, no error");
    assert!(is_nan(&v), "expected NaN, got {v:?}");
}

#[test]
fn a_finite_or_infinite_result_never_reaches_the_hook() {
    // `inf + inf` is `inf`, not NaN; `1.5 * 4.0` is plain. Neither delegates.
    let (calls, hook) = domain_hook();
    assert_eq!(
        run(inf_op_inf(Op::Add, true), Some(hook.clone()), false),
        Ok(Value::Float(f64::INFINITY))
    );
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadFloat(1.5), 0);
    b.emit(Op::LoadFloat(4.0), 0);
    b.emit(Op::Mul, 0);
    b.set_nan_result_hook(true);
    assert_eq!(run(b.build(), Some(hook), false), Ok(Value::Float(6.0)));
    assert_eq!(calls.load(Ordering::Relaxed), 0);
}

/// Straight-line chunk, run well past the block-JIT warmup: every run must
/// still reach the hook, including the ones that execute native code.
#[cfg(feature = "jit")]
#[test]
fn block_tier_bails_to_the_hook_on_a_nan_result() {
    let (calls, hook) = domain_hook();
    let chunk = inf_op_inf(Op::Sub, true);
    for i in 1..=25 {
        assert_eq!(
            run(chunk.clone(), Some(hook.clone()), true),
            Err(DOMAIN.to_string()),
            "run {i}: NaN escaped the hook"
        );
    }
    assert_eq!(calls.load(Ordering::Relaxed), 25);
    assert!(
        fusevm::JitCompiler::new().block_jit_is_compiled(&chunk),
        "the chunk must reach the block tier, or this test proves nothing"
    );
}

/// The block tier without the flag: still compiled, still answers NaN, and the
/// flagged chunk's code (same ops, same hash) is not reused for it.
#[cfg(feature = "jit")]
#[test]
fn block_tier_without_the_flag_answers_nan() {
    let (calls, hook) = domain_hook();
    // Warm the flagged variant first so a policy-blind cache would hand its
    // bailing code to the unflagged run.
    for _ in 0..25 {
        let _ = run(inf_op_inf(Op::Mul, true), Some(hook.clone()), true);
    }
    let before = calls.load(Ordering::Relaxed);
    let chunk = inf_op_inf(Op::Mul, false);
    // `inf * inf` is inf; use Sub for a NaN in the unflagged chunk.
    let nan_chunk = inf_op_inf(Op::Sub, false);
    for _ in 0..25 {
        assert_eq!(
            run(chunk.clone(), Some(hook.clone()), true),
            Ok(Value::Float(f64::INFINITY))
        );
        let v = run(nan_chunk.clone(), Some(hook.clone()), true).expect("no delegation");
        assert!(is_nan(&v), "expected NaN, got {v:?}");
    }
    assert_eq!(calls.load(Ordering::Relaxed), before);
}

/// A hot float loop whose NaN appears only after the trace is installed:
///
/// ```text
///   i = 0; x = 1.0; y = 0.0
/// anchor:
///   x = x * 2.0          // inf from iteration 1024 on
///   y = x - x            // 0.0, then NaN once x is inf
///   i = i + 1
///   if i < 1100 goto anchor
///   y
/// ```
#[cfg(feature = "jit")]
fn doubling_loop(flag: bool) -> (Chunk, usize) {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(0), 1);
    b.emit(Op::SetSlot(0), 1);
    b.emit(Op::LoadFloat(1.0), 1);
    b.emit(Op::SetSlot(1), 1);
    b.emit(Op::LoadFloat(0.0), 1);
    b.emit(Op::SetSlot(2), 1);
    let anchor = b.current_pos();
    b.emit(Op::GetSlot(1), 1);
    b.emit(Op::LoadFloat(2.0), 1);
    b.emit(Op::Mul, 1);
    b.emit(Op::SetSlot(1), 1);
    b.emit(Op::GetSlot(1), 1);
    b.emit(Op::GetSlot(1), 1);
    b.emit(Op::Sub, 1);
    b.emit(Op::SetSlot(2), 1);
    b.emit(Op::GetSlot(0), 1);
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::Add, 1);
    b.emit(Op::SetSlot(0), 1);
    b.emit(Op::GetSlot(0), 1);
    b.emit(Op::LoadInt(1100), 1);
    b.emit(Op::NumLt, 1);
    let jmp = b.emit(Op::JumpIfTrue(0), 1);
    b.patch_jump(jmp, anchor);
    b.emit(Op::GetSlot(2), 1);
    b.set_nan_result_hook(flag);
    (b.build(), anchor)
}

#[cfg(feature = "jit")]
fn run_loop(chunk: Chunk, hook: fusevm::NumericHook) -> Result<Value, String> {
    let mut vm = VM::new(chunk);
    vm.enable_tracing_jit();
    vm.set_numeric_hook(hook);
    {
        let frame = vm.frames.last_mut().unwrap();
        while frame.slots.len() < 3 {
            frame.slots.push(Value::Float(0.0));
        }
    }
    match vm.run() {
        VMResult::Ok(v) => Ok(v),
        VMResult::Halted => Ok(vm.stack.last().cloned().unwrap_or(Value::Undef)),
        VMResult::Error(e) => Err(e),
    }
}

#[cfg(feature = "jit")]
#[test]
fn trace_tier_bails_to_the_hook_on_a_nan_result() {
    // Unflagged first: the loop must reach a compiled trace (or the flagged
    // half proves nothing) and answer NaN without consulting the hook.
    let (calls, hook) = domain_hook();
    let (plain, anchor) = doubling_loop(false);
    let probe = plain.clone();
    let v = run_loop(plain, hook.clone()).expect("unflagged loop must not delegate");
    assert!(is_nan(&v), "expected NaN, got {v:?}");
    assert_eq!(calls.load(Ordering::Relaxed), 0);
    assert!(
        fusevm::JitCompiler::new().trace_is_compiled(&probe, anchor),
        "this loop shape must reach a compiled trace"
    );

    // Flagged: the NaN appears ~1000 iterations after the trace is hot, inside
    // native code, and must still be refused by the hook.
    let (flagged, _) = doubling_loop(true);
    assert_eq!(run_loop(flagged, hook), Err(DOMAIN.to_string()));
    assert_eq!(calls.load(Ordering::Relaxed), 1);
}

/// The flag survives the bincode round trip an AOT executable embeds its chunk
/// through, and it separates the op hash (and so every JIT cache key) from the
/// same ops without it, while an unflagged chunk keeps the hash it always had.
#[test]
fn the_flag_round_trips_and_keys_the_op_hash() {
    let flagged = inf_op_inf(Op::Sub, true);
    let plain = inf_op_inf(Op::Sub, false);
    assert_ne!(flagged.op_hash, plain.op_hash);

    let mut b = ChunkBuilder::new();
    for _ in 0..2 {
        b.emit(Op::LoadFloat(1e308), 0);
        b.emit(Op::LoadFloat(10.0), 0);
        b.emit(Op::Mul, 0);
    }
    b.emit(Op::Sub, 0);
    assert_eq!(b.build().op_hash, plain.op_hash, "unflagged hash changed");

    let bytes = bincode::serialize(&flagged).expect("serialize");
    let back: Chunk = bincode::deserialize(&bytes).expect("deserialize");
    assert!(back.nan_result_hook);
}
