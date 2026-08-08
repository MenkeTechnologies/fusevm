//! Fixed-width integer semantics for frontends whose integers are narrower than
//! fusevm's `Value::Int(i64)`.
//!
//! Java's `int`, Kotlin's `Int`, Go's `int8`/`int32` and Groovy's `Integer` all
//! wrap below 64 bits: `2147483647 + 1` is `-2147483648`, not `2147483648`.
//! fusevm carries no width tag on `Value::Int` and these tests are not asking
//! for one — they pin the two ways a frontend gets the right answer with the VM
//! exactly as it is, and pin the seam where each stops working.
//!
//! **Recipe 1 — a two-op bytecode sign-extend, for a frontend with static
//! types.** `Op::Shl(64-w)` then `Op::Shr(64-w)` truncates a value to `w` bits
//! and sign-extends it, which is the whole of two's-complement narrowing. It
//! needs no hook, no strict mode and no VM-global setting; it is emitted *per
//! site*, so a language with both `int` and `long` simply emits it after the
//! `int` operations. Both ops lower to native code in every tier
//! (`src/jit.rs:2649`, `src/aot.rs:3319`), so a wrapped loop keeps its trace.
//! This is what `javars` already ships (`javars/src/compiler.rs:1892`,
//! `emit_wrap32`); the tests below pin that it is exact and tier-stable.
//!
//! **Recipe 2 — [`VM::set_fixnum_range`] plus a [`fusevm::NumericHook`], for a
//! frontend without static types.** The range narrows what counts as a native
//! integer, so any result outside the frontend's width leaves the fast path and
//! the hook returns the narrowed answer. Built for Emacs's 62-bit tagged
//! fixnums, where the host *widens* an out-of-range result to a bignum; a
//! fixed-width frontend uses the identical seam in the opposite direction.
//! Costs a hook call per overflow and, being VM-global, needs a
//! [`fusevm::SitedNumericHook`] to tell an `int` site from a `long` one.
//!
//! What these tests are protecting:
//!
//! 1. the sign-extend recipe reproduces Java/Kotlin/Go/Groovy exactly, stays on
//!    the native path, and survives JIT compilation;
//! 2. the range+hook recipe reproduces the same matrix, in the interpreter *and*
//!    after the JIT has compiled the chunk — native code folds the same bounds
//!    check into its overflow accumulator, so a warmed loop must not start
//!    answering `2147483648`;
//! 3. in-range arithmetic never reaches the hook, so recipe 2 costs nothing on
//!    the common path;
//! 4. a [`fusevm::SitedNumericHook`] separates two widths *inside one program*,
//!    which a VM-global range alone cannot express;
//! 5. the default policy (no hook, no range) still wraps at i64, so every
//!    frontend that never opts in is untouched;
//! 6. `Op::Shl`/`Op::Shr` themselves are i64-wide and bypass recipe 2 entirely —
//!    the seam a fixed-width frontend must not walk into.

#![cfg(feature = "jit")]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use fusevm::{Chunk, ChunkBuilder, NumOp, Op, VMResult, Value, VM};

/// Sign-extend the low `bits` of `n` — the arithmetic every fixed-width
/// frontend needs and the only thing its hook body has to do.
///
/// `bits == 64` shifts by zero and is the identity, so one helper covers
/// `byte`/`short`/`int`/`long` without a special case.
fn narrow(n: i64, bits: u32) -> i64 {
    let sh = 64 - bits;
    (n << sh) >> sh
}

/// The inclusive `[lo, hi]` a `bits`-wide two's-complement integer spans — what
/// a frontend passes to [`VM::set_fixnum_range`].
///
/// Note this is *not* `(narrow(i64::MIN, bits), narrow(i64::MAX, bits))`:
/// truncating `i64::MAX` to 32 bits gives `-1`, which would make the range empty
/// and send every single operation to the hook. Build the bounds from the width.
fn width_range(bits: u32) -> (i64, i64) {
    if bits >= 64 {
        return (i64::MIN, i64::MAX);
    }
    let hi = (1i64 << (bits - 1)) - 1;
    (-(hi + 1), hi)
}

/// The recipe under test: a [`fusevm::NumericHook`] that reproduces `bits`-wide
/// two's-complement arithmetic.
///
/// The VM hands it the operands *before* any wrapping, so the body computes in
/// `i64` — which cannot itself overflow for `bits <= 32` — and truncates. The
/// counter proves how often the slow path was taken.
fn width_hook(bits: u32, calls: Arc<AtomicUsize>) -> fusevm::NumericHook {
    Arc::new(move |op, a, b| {
        calls.fetch_add(1, Ordering::Relaxed);
        let (x, y) = (a.to_int(), b.to_int());
        let wide = match op {
            NumOp::Add => x.wrapping_add(y),
            NumOp::Sub => x.wrapping_sub(y),
            NumOp::Mul => x.wrapping_mul(y),
            NumOp::Neg => x.wrapping_neg(),
            other => return Err(format!("width hook: unexpected {other:?}")),
        };
        Ok(Value::Int(narrow(wide, bits)))
    })
}

/// `a OP b` as a two-constant chunk.
fn binop_chunk(a: i64, b: i64, op: Op) -> Chunk {
    let mut bd = ChunkBuilder::new();
    bd.emit(Op::LoadInt(a), 0);
    bd.emit(Op::LoadInt(b), 0);
    bd.emit(op, 0);
    bd.build()
}

/// Run `chunk` on a VM configured for `bits`-wide integers, tracing JIT on.
fn run_width(chunk: Chunk, bits: u32, hook: fusevm::NumericHook) -> Value {
    let mut vm = VM::new(chunk);
    vm.enable_tracing_jit();
    vm.set_numeric_hook(hook);
    let (lo, hi) = width_range(bits);
    vm.set_fixnum_range(lo, hi);
    match vm.run() {
        VMResult::Ok(v) => v,
        VMResult::Halted => vm.stack.last().cloned().unwrap_or(Value::Undef),
        VMResult::Error(e) => panic!("vm error: {e}"),
    }
}

/// Recipe 1: emit `a OP b` followed by a `bits`-wide sign-extend.
///
/// The wrap is `Shl (64-bits)` then `Shr (64-bits)` — `Op::Shr` is arithmetic
/// (`src/vm.rs:2288`), so the high bit of the narrowed value is smeared back
/// across the top, which is exactly two's-complement sign extension.
fn wrapped_binop_chunk(a: i64, b: i64, op: Op, bits: u32) -> Chunk {
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

/// Run a plain chunk with the tracing JIT on and no numeric policy at all.
fn run_plain(chunk: Chunk) -> Value {
    let mut vm = VM::new(chunk);
    vm.enable_tracing_jit();
    match vm.run() {
        VMResult::Ok(v) => v,
        VMResult::Halted => vm.stack.last().cloned().unwrap_or(Value::Undef),
        VMResult::Error(e) => panic!("vm error: {e}"),
    }
}

/// The recommended answer: a two-op bytecode sign-extend reproduces every wrap
/// the five frontends reported as missing, with no hook, no strict mode and no
/// VM-wide setting — so `int` and `long` can coexist in one chunk by
/// construction.
///
/// Expected values are the reference languages' own: `java`, `kotlinc` and
/// `groovy` all print `-2147483648` for `Integer.MAX_VALUE + 1` and
/// `-727379968` for `1000000 * 1000000`.
#[test]
fn bytecode_sign_extend_reproduces_fixed_width_wrapping() {
    const I32_MAX: i64 = i32::MAX as i64;
    const I32_MIN: i64 = i32::MIN as i64;

    let cases: &[(u32, i64, Op, i64, i64)] = &[
        (32, I32_MAX, Op::Add, 1, I32_MIN),
        (32, I32_MIN, Op::Sub, 1, I32_MAX),
        (32, 1_000_000, Op::Mul, 1_000_000, -727_379_968),
        // Groovy's other reported case: 9993973 * -490 == -602079474.
        (32, 9_993_973, Op::Mul, -490, -602_079_474),
        (8, 127, Op::Add, 1, -128),
        (8, -128, Op::Sub, 1, 127),
        (16, 32767, Op::Add, 1, -32768),
        (16, 1000, Op::Mul, 1000, 16960),
        // In-range values must pass through the wrap untouched.
        (32, 2_000_000, Op::Add, 3_000_000, 5_000_000),
        (8, 3, Op::Mul, 4, 12),
    ];

    for (bits, a, op, b, want) in cases {
        let (bits, a, b, want) = (*bits, *a, *b, *want);
        let got = run_plain(wrapped_binop_chunk(a, b, op.clone(), bits));
        assert_eq!(
            got,
            Value::Int(want),
            "{bits}-bit {a} {op:?} {b} via Shl/Shr must match the reference language"
        );
    }
}

/// The sign-extend must keep answering the same thing once the chunk is native
/// code. `Op::Shl`/`Op::Shr` are ordinary lowered ops, so this is really a pin
/// that the JIT's shift semantics match the interpreter's `& 63` masking.
#[test]
fn bytecode_sign_extend_survives_jit_compilation() {
    for i in 1..=40 {
        let got = run_plain(wrapped_binop_chunk(i32::MAX as i64, 1, Op::Add, 32));
        assert_eq!(
            got,
            Value::Int(i32::MIN as i64),
            "run {i}: native code disagreed with the interpreter's sign-extend"
        );
    }
}

/// Go's unsigned widths narrow with a mask instead of a sign-extend: `uint8`
/// spans `0..=255`, so `Op::BitAnd` against `0xFF` is the whole operation.
/// `var u8 uint8 = 0; u8--` is `255` in Go — the case go-rs's BUGS.md reports.
#[test]
fn bitand_mask_reproduces_unsigned_fixed_width_wrapping() {
    let masked = |a: i64, op: Op, b: i64, mask: i64| -> Value {
        let mut bd = ChunkBuilder::new();
        bd.emit(Op::LoadInt(a), 0);
        bd.emit(Op::LoadInt(b), 0);
        bd.emit(op, 0);
        bd.emit(Op::LoadInt(mask), 0);
        bd.emit(Op::BitAnd, 0);
        run_plain(bd.build())
    };

    // uint8: 0 - 1 == 255, 255 + 1 == 0.
    assert_eq!(masked(0, Op::Sub, 1, 0xFF), Value::Int(255));
    assert_eq!(masked(255, Op::Add, 1, 0xFF), Value::Int(0));
    // uint16 and uint32.
    assert_eq!(masked(0, Op::Sub, 1, 0xFFFF), Value::Int(65535));
    assert_eq!(
        masked(0, Op::Sub, 1, 0xFFFF_FFFF),
        Value::Int(4_294_967_295)
    );
    // In range: untouched.
    assert_eq!(masked(200, Op::Add, 7, 0xFF), Value::Int(207));
}

/// Recipe 1 is per-site, so a chunk holding an `int` operation and a `long`
/// operation needs no way to tell the VM about either — the `int` one carries a
/// wrap and the `long` one does not. This is the property recipe 2 has to spend
/// a [`fusevm::SitedNumericHook`] to recover.
#[test]
fn sign_extend_lets_int_and_long_coexist_without_any_vm_setting() {
    let mut bd = ChunkBuilder::new();
    // `long`: 2147483647 + 1, exact.
    bd.emit(Op::LoadInt(i32::MAX as i64), 0);
    bd.emit(Op::LoadInt(1), 0);
    bd.emit(Op::Add, 0);
    // `int`: the same arithmetic, wrapped.
    bd.emit(Op::LoadInt(i32::MAX as i64), 0);
    bd.emit(Op::LoadInt(1), 0);
    bd.emit(Op::Add, 0);
    bd.emit(Op::LoadInt(32), 0);
    bd.emit(Op::Shl, 0);
    bd.emit(Op::LoadInt(32), 0);
    bd.emit(Op::Shr, 0);
    // The run's result is the top of stack (the `int` site); the `long` site's
    // value stays below it.
    let mut vm = VM::new(bd.build());
    vm.enable_tracing_jit();
    let int_result = match vm.run() {
        VMResult::Ok(v) => v,
        VMResult::Halted => vm.stack.pop().unwrap_or(Value::Undef),
        VMResult::Error(e) => panic!("vm error: {e}"),
    };
    assert_eq!(
        int_result,
        Value::Int(i32::MIN as i64),
        "the int site must wrap to 32 bits"
    );
    assert_eq!(
        vm.stack.last(),
        Some(&Value::Int(2_147_483_648)),
        "the long site, in the same chunk, must stay exact"
    );
}

/// Recipe 2's matrix: every wrap the five frontends reported as missing,
/// answered with `set_fixnum_range` + a hook that truncates.
///
/// Expected values are the reference languages' own, not fusevm's:
/// `Integer.MAX_VALUE + 1 == Integer.MIN_VALUE` and `1000000 * 1000000 ==
/// -727379968` are what `java`, `kotlinc` and `groovy` print.
#[test]
fn narrowed_range_plus_hook_reproduces_fixed_width_wrapping() {
    const I32_MAX: i64 = i32::MAX as i64;
    const I32_MIN: i64 = i32::MIN as i64;

    // (bits, a, op, b, expected) — the reference language's answer.
    let cases: &[(u32, i64, Op, i64, i64)] = &[
        // Kotlin/Java `Int`: MAX_VALUE + 1 wraps to MIN_VALUE.
        (32, I32_MAX, Op::Add, 1, I32_MIN),
        // ...and MIN_VALUE - 1 wraps back to MAX_VALUE.
        (32, I32_MIN, Op::Sub, 1, I32_MAX),
        // Groovy `Integer`: 1000000 * 1000000 == -727379968.
        (32, 1_000_000, Op::Mul, 1_000_000, -727_379_968),
        // Go `int8`: 127 + 1 == -128.
        (8, 127, Op::Add, 1, -128),
        (8, -128, Op::Sub, 1, 127),
        // Go `int16`.
        (16, 32767, Op::Add, 1, -32768),
        // Java `short`-width multiply that overflows several times over.
        (16, 1000, Op::Mul, 1000, 16960),
    ];

    for (bits, a, op, b, want) in cases {
        let (bits, a, b, want) = (*bits, *a, *b, *want);
        let calls = Arc::new(AtomicUsize::new(0));
        let hook = width_hook(bits, calls.clone());
        let got = run_width(binop_chunk(a, b, op.clone()), bits, hook);
        assert_eq!(
            got,
            Value::Int(want),
            "{bits}-bit {a} {op:?} {b}: fusevm must agree with the reference language"
        );
        assert_eq!(
            calls.load(Ordering::Relaxed),
            1,
            "{bits}-bit {a} {op:?} {b}: exactly one delegation"
        );
    }
}

/// The bounds check rides in the JIT's overflow accumulator, so a chunk that has
/// been compiled must keep wrapping. Without this, a hot loop would silently
/// switch from the Java answer to the i64 one once native code took over — the
/// worst shape of bug, because the first iterations are right.
#[test]
fn fixed_width_wrapping_survives_jit_compilation() {
    let calls = Arc::new(AtomicUsize::new(0));
    let hook = width_hook(32, calls.clone());

    // Well past the block-JIT warmup threshold.
    for i in 1..=40 {
        let got = run_width(binop_chunk(i32::MAX as i64, 1, Op::Add), 32, hook.clone());
        assert_eq!(
            got,
            Value::Int(i32::MIN as i64),
            "run {i}: native code let the i64 result escape"
        );
    }
    assert_eq!(
        calls.load(Ordering::Relaxed),
        40,
        "every run must delegate, warm or cold"
    );
}

/// The width policy must be free when nothing overflows: an in-range add stays
/// on the native path and never reaches the hook. This is what makes the recipe
/// usable in a hot loop rather than just correct.
#[test]
fn in_range_arithmetic_never_reaches_the_width_hook() {
    let calls = Arc::new(AtomicUsize::new(0));
    let hook = width_hook(32, calls.clone());

    for _ in 0..40 {
        let got = run_width(binop_chunk(2_000_000, 3_000_000, Op::Add), 32, hook.clone());
        assert_eq!(got, Value::Int(5_000_000));
    }
    assert_eq!(
        calls.load(Ordering::Relaxed),
        0,
        "in-range 32-bit arithmetic must stay native"
    );
}

/// Two widths in one program.
///
/// `set_fixnum_range` is a property of the VM, not of an operation, so a
/// language with both `int` and `long` cannot express its policy with the range
/// alone: narrowing to 32 bits sends every wide-but-legal `long` result to the
/// hook too. A [`fusevm::SitedNumericHook`] closes that — it is told the op
/// index, and the frontend's own type checker already knows the static width of
/// each arithmetic site. The narrow range becomes a *filter* ("this result left
/// 32 bits, ask the frontend") and the site decides what that means.
///
/// The chunk below is `(2147483647 + 1)` at one site and `(2147483647 + 1)` at
/// another, identical as arithmetic. The first is `int` and must wrap; the
/// second is `long` and must not.
#[test]
fn sited_hook_separates_int_and_long_sites_in_one_chunk() {
    let mut bd = ChunkBuilder::new();
    // Site A (`int`): 2147483647 + 1 → must wrap to -2147483648.
    bd.emit(Op::LoadInt(i32::MAX as i64), 0);
    bd.emit(Op::LoadInt(1), 0);
    let int_site = bd.emit(Op::Add, 0);
    bd.emit(Op::Pop, 0);
    // Site B (`long`): the same arithmetic, exact.
    bd.emit(Op::LoadInt(i32::MAX as i64), 0);
    bd.emit(Op::LoadInt(1), 0);
    let long_site = bd.emit(Op::Add, 0);
    let chunk = bd.build();

    // What a frontend's lowering records: the static width of each site.
    let widths: std::collections::HashMap<usize, u32> =
        [(int_site, 32), (long_site, 64)].into_iter().collect();

    let hook: fusevm::SitedNumericHook = Arc::new(move |call| {
        let bits = *widths.get(&call.ip).unwrap_or(&64);
        let (x, y) = (call.a.to_int(), call.b.to_int());
        let wide = match call.op {
            NumOp::Add => x.wrapping_add(y),
            other => return Err(format!("unexpected {other:?}")),
        };
        Ok(Value::Int(narrow(wide, bits)))
    });

    let mut vm = VM::new(chunk);
    vm.enable_tracing_jit();
    vm.set_sited_numeric_hook(hook);
    // Narrowed to the *narrowest* width in the program, so both sites are seen.
    vm.set_fixnum_range(i32::MIN as i64, i32::MAX as i64);

    let got = match vm.run() {
        VMResult::Ok(v) => v,
        VMResult::Halted => vm.stack.last().cloned().unwrap_or(Value::Undef),
        VMResult::Error(e) => panic!("vm error: {e}"),
    };
    // The value left on the stack is site B's — the `long` one, unwrapped.
    assert_eq!(
        got,
        Value::Int(2_147_483_648),
        "the long site must keep the exact i64 result"
    );

    // And site A, run alone, wraps.
    let mut bd = ChunkBuilder::new();
    bd.emit(Op::LoadInt(i32::MAX as i64), 0);
    bd.emit(Op::LoadInt(1), 0);
    let only_site = bd.emit(Op::Add, 0);
    assert_eq!(only_site, int_site, "site index must match the first chunk");
    let widths2: std::collections::HashMap<usize, u32> = [(int_site, 32)].into_iter().collect();
    let hook2: fusevm::SitedNumericHook = Arc::new(move |call| {
        let bits = *widths2.get(&call.ip).unwrap_or(&64);
        Ok(Value::Int(narrow(
            call.a.to_int().wrapping_add(call.b.to_int()),
            bits,
        )))
    });
    let mut vm = VM::new(bd.build());
    vm.set_sited_numeric_hook(hook2);
    vm.set_fixnum_range(i32::MIN as i64, i32::MAX as i64);
    let got = match vm.run() {
        VMResult::Ok(v) => v,
        VMResult::Halted => vm.stack.last().cloned().unwrap_or(Value::Undef),
        VMResult::Error(e) => panic!("vm error: {e}"),
    };
    assert_eq!(got, Value::Int(i32::MIN as i64), "the int site must wrap");
}

/// A frontend that opts into nothing must be byte-identical to before: no hook,
/// no range, `i64::MAX + 1` still wraps at 64 bits and nothing is consulted.
/// This is the blast-radius pin for the other sixteen frontends.
#[test]
fn default_policy_still_wraps_at_i64_with_no_hook() {
    let mut vm = VM::new(binop_chunk(i64::MAX, 1, Op::Add));
    vm.enable_tracing_jit();
    let got = match vm.run() {
        VMResult::Ok(v) => v,
        VMResult::Halted => vm.stack.last().cloned().unwrap_or(Value::Undef),
        VMResult::Error(e) => panic!("vm error: {e}"),
    };
    assert_eq!(got, Value::Int(i64::MIN), "default policy must wrap at i64");

    // Java/Kotlin `long` is exactly this default — no opt-in needed.
    let mut vm = VM::new(binop_chunk(4_000_000_000, 4_000_000_000, Op::Mul));
    let got = match vm.run() {
        VMResult::Ok(v) => v,
        VMResult::Halted => vm.stack.last().cloned().unwrap_or(Value::Undef),
        VMResult::Error(e) => panic!("vm error: {e}"),
    };
    assert_eq!(
        got,
        Value::Int(4_000_000_000_i64.wrapping_mul(4_000_000_000)),
        "64-bit wrapping is the default and matches Java `long`"
    );
}

/// The limit of the recipe, pinned so nobody discovers it in production.
///
/// `Op::Shl` / `Op::Shr` do not route through `arith_int_fast`
/// (`src/vm.rs:2280`), so neither the fixnum range nor the hook is consulted:
/// the shift count is masked at 63 and the result keeps all 64 bits. Java's
/// `int << n` masks the count at 31 and truncates to 32. A fixed-width frontend
/// must therefore lower its shifts through its own builtin, not `Op::Shl`.
#[test]
fn shifts_bypass_the_width_policy_and_must_be_lowered_by_the_frontend() {
    let calls = Arc::new(AtomicUsize::new(0));
    let hook = width_hook(32, calls.clone());

    // Java: `1 << 31 == -2147483648`. fusevm's Shl gives the i64 answer.
    let got = run_width(binop_chunk(1, 31, Op::Shl), 32, hook.clone());
    assert_eq!(
        got,
        Value::Int(2_147_483_648),
        "Shl is i64-wide; the frontend, not the VM, owns int-width shifts"
    );
    assert_eq!(
        calls.load(Ordering::Relaxed),
        0,
        "Shl never consults the numeric hook"
    );

    // Java: `1 << 32 == 1` (count masked at 31). fusevm masks at 63.
    let got = run_width(binop_chunk(1, 32, Op::Shl), 32, hook);
    assert_eq!(got, Value::Int(4_294_967_296));
}
