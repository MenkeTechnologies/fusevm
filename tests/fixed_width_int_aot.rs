//! Fixed-width integer semantics under the AOT closed-world compiler.
//!
//! `tests/fixed_width_int_semantics.rs` pins the two recipes in the interpreter
//! and the tracing/block JIT. The AOT tier is a separate compiler
//! (`src/aot.rs`) with its own lowering of integer arithmetic, so it needs its
//! own pins — a width policy that holds interpreted and breaks when a frontend
//! runs `--aot` is worse than one that never worked.
//!
//! The two recipes come out differently here, and the difference is the reason
//! this file exists:
//!
//! - **the bytecode sign-extend agrees across tiers.** `Op::Shl` / `Op::Shr`
//!   are ordinary ops the AOT compiler lowers to `ishl` / `sshr`
//!   (`src/aot.rs:3319`), so a narrowed result is narrowed identically in
//!   native code.
//! - **`VM::set_fixnum_range` does not reach the AOT tier at all.** The range
//!   is VM state (`src/vm.rs:172`), the AOT compiler reads only `Chunk`, and
//!   its overflow-checked integer path tests for `i64` overflow alone
//!   (`src/aot.rs:2436-2467`) — there is no counterpart to the bounds check the
//!   tracing JIT folds into its accumulator (`src/jit.rs:2088-2096`). A result
//!   that leaves the range but fits an `i64` therefore stays native and never
//!   reaches the hook.

#![cfg(feature = "aot")]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use fusevm::{Chunk, ChunkBuilder, Op, VMResult, Value, VM};

/// Run `chunk` through the AOT compiler and through the interpreter, returning
/// `(native, interpreted)`. `configure` installs the same numeric policy on both.
fn both_tiers(chunk: &Chunk, configure: impl Fn(&mut VM) + Copy + 'static) -> (Value, Value) {
    let native = match fusevm::aot::run_chunk_native(chunk, configure).expect("native run") {
        VMResult::Ok(v) => v,
        other => panic!("native: unexpected {other:?}"),
    };
    let mut vm = VM::new(chunk.clone());
    configure(&mut vm);
    let interpreted = match vm.run() {
        VMResult::Ok(v) => v,
        VMResult::Halted => vm.stack.last().cloned().unwrap_or(Value::Undef),
        VMResult::Error(e) => panic!("interp error: {e}"),
    };
    (native, interpreted)
}

/// `a OP b` narrowed to `bits` by the two-op sign-extend.
fn wrapped(a: i64, b: i64, op: Op, bits: u32) -> Chunk {
    let mut bd = ChunkBuilder::new();
    bd.emit(Op::LoadInt(a), 0);
    bd.emit(Op::LoadInt(b), 0);
    bd.emit(op, 0);
    let sh = (64 - bits) as i64;
    bd.emit(Op::LoadInt(sh), 0);
    bd.emit(Op::Shl, 0);
    bd.emit(Op::LoadInt(sh), 0);
    bd.emit(Op::Shr, 0);
    bd.build()
}

/// The recommended recipe is AOT-safe: native and interpreted agree, and both
/// agree with Java/Kotlin/Go/Groovy.
#[test]
fn bytecode_sign_extend_agrees_between_aot_and_interpreter() {
    let cases: &[(u32, i64, Op, i64, i64)] = &[
        (32, i32::MAX as i64, Op::Add, 1, i32::MIN as i64),
        (32, i32::MIN as i64, Op::Sub, 1, i32::MAX as i64),
        (32, 1_000_000, Op::Mul, 1_000_000, -727_379_968),
        (8, 127, Op::Add, 1, -128),
        (16, 32767, Op::Add, 1, -32768),
        (32, 2_000_000, Op::Add, 3_000_000, 5_000_000),
    ];
    for (bits, a, op, b, want) in cases {
        let (bits, a, b, want) = (*bits, *a, *b, *want);
        let chunk = wrapped(a, b, op.clone(), bits);
        let (native, interp) = both_tiers(&chunk, |_vm| {});
        assert_eq!(
            native, interp,
            "{bits}-bit {a} {op:?} {b}: AOT and interpreter must agree"
        );
        assert_eq!(
            native,
            Value::Int(want),
            "{bits}-bit {a} {op:?} {b}: must match the reference language"
        );
    }
}

/// The unsigned mask is AOT-safe for the same reason.
#[test]
fn bitand_mask_agrees_between_aot_and_interpreter() {
    let mut bd = ChunkBuilder::new();
    bd.emit(Op::LoadInt(0), 0);
    bd.emit(Op::LoadInt(1), 0);
    bd.emit(Op::Sub, 0);
    bd.emit(Op::LoadInt(0xFF), 0);
    bd.emit(Op::BitAnd, 0);
    let chunk = bd.build();
    let (native, interp) = both_tiers(&chunk, |_vm| {});
    assert_eq!(native, interp);
    assert_eq!(native, Value::Int(255), "Go: var u8 uint8 = 0; u8-- == 255");
}

/// **Known divergence, recorded so nobody discovers it in a shipped binary.**
///
/// This test asserts what the VM does *today*, not what it should do.
/// `VM::set_fixnum_range` is honored by the interpreter (`src/vm.rs:1444`) and
/// by the tracing JIT (`src/jit.rs:2088`), and ignored by the AOT compiler,
/// which has no access to it: the range lives on the `VM` and `src/aot.rs`
/// lowers from the `Chunk`. So the same chunk, same hook, same range answers
/// `-2147483648` interpreted and `2147483648` compiled, with the hook never
/// consulted in the native run.
///
/// Consequences, in order of who is exposed:
///
/// - `elisprs` is the only frontend that calls `set_fixnum_range`
///   (`elisprs/src/host.rs:4865`) *and* it ships `elisp --aot`
///   (`elisprs/src/aot.rs:40`). Its AOT runtime installs neither the numeric
///   hook nor the range (`elisprs/src/aot_runtime.rs:29-48`), so an
///   AOT-compiled elisp program silently loses bignum promotion.
/// - any frontend that adopts recipe 2 for fixed-width integers inherits the
///   same divergence the moment it AOT-compiles.
///
/// Recipe 1 (the bytecode sign-extend, pinned above) has no such exposure,
/// which is the main reason it is the recommended one.
///
/// If the AOT compiler is ever taught the range — the natural shape is a
/// `Chunk` field alongside `int_overflow_deopt` (`src/chunk.rs:41`), folded
/// into the existing overflow-deopt branch at `src/aot.rs:2436` the way
/// `src/jit.rs:2088` folds it into the accumulator — this test will fail, and
/// the right response is to rewrite it as an equality assertion.
#[test]
fn known_divergence_aot_ignores_the_fixnum_range() {
    let mut bd = ChunkBuilder::new();
    bd.emit(Op::LoadInt(i32::MAX as i64), 0);
    bd.emit(Op::LoadInt(1), 0);
    bd.emit(Op::Add, 0);
    bd.set_int_overflow_deopt(true);
    let chunk = bd.build();

    let calls = Arc::new(AtomicUsize::new(0));
    let native_calls = calls.clone();
    let native = match fusevm::aot::run_chunk_native(&chunk, move |vm: &mut VM| {
        let c = native_calls.clone();
        vm.set_numeric_hook(Arc::new(move |_op, a, b| {
            c.fetch_add(1, Ordering::Relaxed);
            let n = a.to_int().wrapping_add(b.to_int());
            Ok(Value::Int((n << 32) >> 32))
        }));
        vm.set_fixnum_range(i32::MIN as i64, i32::MAX as i64);
    })
    .expect("native run")
    {
        VMResult::Ok(v) => v,
        other => panic!("native: unexpected {other:?}"),
    };

    assert_eq!(
        native,
        Value::Int(2_147_483_648),
        "recorded behavior: AOT keeps the i64 result"
    );
    assert_eq!(
        calls.load(Ordering::Relaxed),
        0,
        "recorded behavior: AOT never consults the hook for an in-i64 result"
    );

    // The interpreter, same chunk and same policy, narrows.
    let interp_calls = Arc::new(AtomicUsize::new(0));
    let c = interp_calls.clone();
    let mut vm = VM::new(chunk);
    vm.set_numeric_hook(Arc::new(move |_op, a, b| {
        c.fetch_add(1, Ordering::Relaxed);
        let n = a.to_int().wrapping_add(b.to_int());
        Ok(Value::Int((n << 32) >> 32))
    }));
    vm.set_fixnum_range(i32::MIN as i64, i32::MAX as i64);
    let interp = match vm.run() {
        VMResult::Ok(v) => v,
        other => panic!("interp: unexpected {other:?}"),
    };
    assert_eq!(interp, Value::Int(i32::MIN as i64));
    assert_eq!(interp_calls.load(Ordering::Relaxed), 1);

    assert_ne!(
        native, interp,
        "if these now agree, AOT learned the range — rewrite this test as assert_eq"
    );
}
