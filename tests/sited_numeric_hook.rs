//! `VM::set_sited_numeric_hook` — a numeric hook told *where* the arithmetic was.
//!
//! Some frontends need to word one refusal differently from another for
//! arithmetic that is identical as arithmetic. Tcl is the case: `incr x` and
//! `expr {$x + 1}` both lower to `GetVar` / `LoadInt(1)` / `Add`, and on a
//! non-numeric `x` the reference interpreter says `expected integer but got
//! "abc"` for the first and `cannot use non-numeric string "abc" as left
//! operand of "+"` for the second. Nothing about the op or the operands
//! separates them; the site does.
//!
//! What these pin:
//!
//! 1. the site reaches the hook, and is the op index of the operation itself;
//! 2. two chunks that share an op vector are still distinct, so a table keyed
//!    by site answers for the right one — the same guarantee `UndefRead::chunk`
//!    carries, and for the same reason;
//! 3. the site is the *interpreter's* index even for an operation a native tier
//!    started, which is the part that would silently answer wrongly if the
//!    deopt path reported its own idea of where it was;
//! 4. installing only the sited hook still puts the VM in strict numeric mode —
//!    without that, native code would wrap an overflow and the hook would never
//!    be reached.

#![cfg(feature = "jit")]

use std::sync::{Arc, Mutex};

use fusevm::{ChunkBuilder, NumOp, NumericCall, Op, VMResult, Value, VM};

/// Every delegated operation, as `(chunk, ip, op)`.
type Log = Arc<Mutex<Vec<(u64, usize, NumOp)>>>;

fn recording(log: &Log) -> fusevm::SitedNumericHook {
    let log = Arc::clone(log);
    Arc::new(move |call: NumericCall<'_>| {
        log.lock()
            .expect("log")
            .push((call.chunk, call.ip, call.op));
        Ok(Value::Int(0))
    })
}

#[test]
fn the_hook_is_told_the_op_index_of_the_operation() {
    // 0: LoadConst "abc"   1: LoadInt 1   2: Add
    let mut b = ChunkBuilder::new();
    let s = b.add_constant(Value::str("abc".to_string()));
    b.emit(Op::LoadConst(s), 1);
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::Add, 1);
    let chunk = b.build();

    let log: Log = Arc::new(Mutex::new(Vec::new()));
    let mut vm = VM::new(chunk);
    vm.set_sited_numeric_hook(recording(&log));
    vm.run();

    let seen = log.lock().expect("log").clone();
    assert_eq!(seen.len(), 1, "one delegated op: {seen:?}");
    assert_eq!(seen[0].1, 2, "the Add is at index 2, not the chunk's end");
    assert_eq!(seen[0].2, NumOp::Add);
}

#[test]
fn two_chunks_sharing_an_op_vector_are_still_distinct() {
    let log: Log = Arc::new(Mutex::new(Vec::new()));

    // Same ops, different name pools — which is exactly the pair `Chunk::op_hash`
    // cannot separate, because it keys the JIT's native-code cache where a name
    // is only an index.
    for name in ["x", "y"] {
        let mut b = ChunkBuilder::new();
        let n = b.add_name(name);
        let s = b.add_constant(Value::str("abc".to_string()));
        b.emit(Op::LoadConst(s), 1);
        b.emit(Op::SetVar(n), 1);
        b.emit(Op::GetVar(n), 1);
        b.emit(Op::LoadInt(1), 1);
        b.emit(Op::Add, 1);
        let chunk = b.build();

        let mut vm = VM::new(chunk);
        vm.set_sited_numeric_hook(recording(&log));
        vm.run();
    }

    let seen = log.lock().expect("log").clone();
    assert_eq!(seen.len(), 2, "one per chunk: {seen:?}");
    assert_eq!(seen[0].1, seen[1].1, "both delegate at the same op index");
    assert_ne!(
        seen[0].0, seen[1].0,
        "and the chunk identity separates them: {seen:?}"
    );
}

/// Builds `acc = start; while (i < iters) { acc += 1; i += 1 }` and returns the
/// chunk, the accumulator's `Add` index, and the loop's trace anchor — which is
/// the *target* of the closing backward branch, not the condition's index.
fn overflow_loop(start: i64, iters: i64) -> (fusevm::Chunk, usize, usize) {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(0), 1);
    b.emit(Op::SetSlot(0), 1);
    b.emit(Op::LoadInt(start), 1);
    b.emit(Op::SetSlot(1), 1);
    let enter = b.emit(Op::Jump(usize::MAX), 1);

    let body = b.current_pos();
    b.emit(Op::GetSlot(1), 1);
    b.emit(Op::LoadInt(1), 1);
    let add_at = b.current_pos();
    b.emit(Op::Add, 1);
    b.emit(Op::SetSlot(1), 1);
    b.emit(Op::GetSlot(0), 1);
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::Add, 1);
    b.emit(Op::SetSlot(0), 1);

    let cond = b.current_pos();
    b.patch_jump(enter, cond);
    b.emit(Op::GetSlot(0), 1);
    b.emit(Op::LoadInt(iters), 1);
    b.emit(Op::NumLt, 1);
    b.emit(Op::JumpIfTrue(body), 1);
    b.emit(Op::GetSlot(0), 1);
    (b.build(), add_at, body)
}

/// An operation a *native tier* started reports the interpreter's index.
///
/// This is the case that would answer wrongly in silence: a trace runs the
/// arithmetic natively, traps on overflow, is discarded whole, and the
/// interpreter re-runs the operation. A frontend keyed by site needs the index
/// it would have seen had the trace never existed.
///
/// Two parts, because a single loop cannot show both. The first proves this loop
/// shape really is traced — otherwise the second would be a test of the
/// interpreter wearing a JIT's name. The second puts the overflow far enough in
/// that the trace is long installed when it happens, and substitutes a value
/// that cannot overflow again, so exactly one delegation happens and its index
/// is unambiguous.
#[test]
fn a_deopted_operation_reports_the_interpreters_index() {
    // Part one: the same shape, nowhere near the ceiling.
    let (safe, _, anchor) = overflow_loop(0, 2_000);
    let probe = safe.clone();
    let mut vm = VM::new(safe);
    vm.enable_tracing_jit();
    vm.set_sited_numeric_hook(recording(&Arc::new(Mutex::new(Vec::new()))));
    if let VMResult::Error(e) = vm.run() {
        panic!("vm error: {e}");
    }
    assert!(
        fusevm::JitCompiler::new().trace_is_compiled(&probe, anchor),
        "this loop shape must reach a compiled trace, or part two proves nothing"
    );

    // Part two: overflow 1_500 iterations in, long after the trace is installed.
    let (chunk, add_at, _) = overflow_loop(i64::MAX - 1_500, 2_000);
    let log: Log = Arc::new(Mutex::new(Vec::new()));
    let sink = Arc::clone(&log);
    let mut vm = VM::new(chunk);
    vm.enable_tracing_jit();
    vm.set_sited_numeric_hook(Arc::new(move |call: NumericCall<'_>| {
        sink.lock()
            .expect("log")
            .push((call.chunk, call.ip, call.op));
        // Cannot overflow again, so the delegation happens exactly once.
        Ok(Value::Int(0))
    }));
    if let VMResult::Error(e) = vm.run() {
        panic!("vm error: {e}");
    }

    let seen = log.lock().expect("log").clone();
    assert_eq!(seen.len(), 1, "one overflow, so one delegation: {seen:?}");
    assert_eq!(seen[0].2, NumOp::Add);
    assert_eq!(
        seen[0].1, add_at,
        "the delegated op reports the interpreter's index for the \
         accumulator's Add, not a native tier's"
    );
}

/// The sited hook alone must arm strict mode. If it did not, the checked
/// arithmetic and the JIT's strictness gates would both stay off and an
/// overflow would wrap in native code without the host ever hearing about it.
#[test]
fn the_sited_hook_alone_is_strict_numeric_mode() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(i64::MAX), 1);
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::Add, 1);
    let chunk = b.build();

    let log: Log = Arc::new(Mutex::new(Vec::new()));
    let mut vm = VM::new(chunk);
    assert!(!vm.is_strict_numeric());
    vm.set_sited_numeric_hook(recording(&log));
    assert!(
        vm.is_strict_numeric(),
        "the sited hook is a strict-mode hook"
    );
    vm.run();

    assert_eq!(
        log.lock().expect("log").len(),
        1,
        "i64::MAX + 1 must delegate rather than wrap"
    );
}
