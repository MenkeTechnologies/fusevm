//! Exhaustive interpreter-vs-native differential matrix.
//!
//! fusevm has no external reference implementation — its parity contract is
//! internal: for one `Chunk`, the interpreter, the linear/block JIT, and the
//! AOT compiler must produce the *same* `Value`, variant included. A tier
//! divergence is a silent wrong answer in every frontend at once.
//!
//! `tests/jit_fuzz.rs` random-walks JIT-friendly loops; this file is the
//! complementary axis — every native-lowerable op crossed with the operand
//! edges that break naive lowerings: `0`, `-1`, `i64::MIN`, `i64::MAX`,
//! `-0.0`, non-integral floats, and shift counts at/over the word width.
//!
//! The comparison is on the `Value` *variant*, not on numeric equality:
//! `Int(0)` where the interpreter answers `Float(0.5)` is exactly the class of
//! bug this file exists to catch, and a float-tolerant comparator would let it
//! through.

#![cfg(feature = "jit")]

use fusevm::{Chunk, ChunkBuilder, JitCompiler, Op, VMResult, Value, VM};

/// One operand, as the op sequence that pushes it.
#[derive(Clone, Debug)]
struct Operand {
    label: &'static str,
    push: Vec<Op>,
}

fn int(label: &'static str, n: i64) -> Operand {
    Operand {
        label,
        push: vec![Op::LoadInt(n)],
    }
}

fn float(label: &'static str, f: f64) -> Operand {
    Operand {
        label,
        push: vec![Op::LoadFloat(f)],
    }
}

/// Integer operand edges. These are the values that separate a correct
/// lowering from a plausible one.
fn int_operands() -> Vec<Operand> {
    vec![
        int("0", 0),
        int("1", 1),
        int("2", 2),
        int("-1", -1),
        int("-2", -2),
        int("3", 3),
        int("7", 7),
        int("63", 63),
        int("64", 64),
        int("65", 65),
        int("i64::MIN", i64::MIN),
        int("i64::MAX", i64::MAX),
        // Past 2^53 an `i64` no longer round-trips through `f64`. Any lowering
        // that reaches a float register — or any interpreter path that does —
        // starts answering a *different integer*.
        int("2^53+1", (1i64 << 53) + 1),
        int("i64::MAX-1", i64::MAX - 1),
        int("i64::MIN+1", i64::MIN + 1),
    ]
}

/// Float operand edges, including the signed zero that a naive
/// `f as i64` round-trip collapses.
fn float_operands() -> Vec<Operand> {
    vec![
        float("0.0", 0.0),
        float("-0.0", -0.0),
        float("1.0", 1.0),
        float("-1.0", -1.0),
        float("0.5", 0.5),
        float("-2.5", -2.5),
        float("3.0", 3.0),
        float("1e18", 1e18),
        float("-1e18", -1e18),
        // Past `i64::MAX` as an f64. This is the edge that separates
        // Cranelift's `fcvt_to_sint` (traps — an illegal instruction in
        // JIT-compiled code) from `fcvt_to_sint_sat` (saturates, which is what
        // Rust's `f as i64` and so `Value::to_int` do). `1e18` is inside the
        // range and cannot tell them apart.
        float("1e30", 1e30),
        float("-1e30", -1e30),
    ]
}

/// Booleans, and the comparisons that produce them. `Value::Bool` is not a
/// native number — the interpreter coerces it through `to_float`, so `true + 1`
/// is `Float(2.0)`, not `Int(2)` — which makes it the operand kind most likely
/// to be folded into an integer by a native tier.
fn bool_operands() -> Vec<Operand> {
    vec![
        Operand {
            label: "true",
            push: vec![Op::LoadTrue],
        },
        Operand {
            label: "false",
            push: vec![Op::LoadFalse],
        },
        Operand {
            label: "(1<2)",
            push: vec![Op::LoadInt(1), Op::LoadInt(2), Op::NumLt],
        },
        Operand {
            label: "(1>2)",
            push: vec![Op::LoadInt(1), Op::LoadInt(2), Op::NumGt],
        },
        Operand {
            label: "!0",
            push: vec![Op::LoadInt(0), Op::LogNot],
        },
    ]
}

fn mixed_operands() -> Vec<Operand> {
    let mut v = int_operands();
    v.extend(float_operands());
    v
}

/// Every operand kind the linear tier can see, booleans included.
fn all_operands() -> Vec<Operand> {
    let mut v = mixed_operands();
    v.extend(bool_operands());
    v
}

fn chunk_for(a: &Operand, b: Option<&Operand>, op: &Op) -> Chunk {
    let mut bd = ChunkBuilder::new();
    for o in &a.push {
        bd.emit(o.clone(), 1);
    }
    if let Some(b) = b {
        for o in &b.push {
            bd.emit(o.clone(), 1);
        }
    }
    bd.emit(op.clone(), 1);
    bd.build()
}

/// Interpret `chunk` and return the value left on top, or `None` for an error.
fn interp(chunk: &Chunk) -> Option<Value> {
    let mut vm = VM::new(chunk.clone());
    match vm.run() {
        VMResult::Ok(v) => Some(v),
        VMResult::Halted => vm.stack.last().cloned(),
        VMResult::Error(_) => None,
    }
}

/// Bit-exact `Value` comparison. `Float` compares on raw bits so `-0.0` and
/// `NaN` are distinguished — the JIT is required to reproduce them exactly.
fn same(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Float(x), Value::Float(y)) => {
            x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan())
        }
        _ => a == b,
    }
}

fn describe(v: &Option<Value>) -> String {
    match v {
        None => "<error>".to_string(),
        Some(Value::Float(f)) => format!("Float({f:?} bits={:#x})", f.to_bits()),
        Some(other) => format!("{other:?}"),
    }
}

/// Cross `ops` with the operand matrix and report every case where the linear
/// JIT accepted the chunk but disagreed with the interpreter.
fn diff_binary(ops: &[Op], operands: &[Operand]) -> Vec<String> {
    let jit = JitCompiler::new();
    let mut diffs = Vec::new();
    for op in ops {
        for a in operands {
            for b in operands {
                let chunk = chunk_for(a, Some(b), op);
                let Some(native) = jit.try_run_linear(&chunk, &[]) else {
                    continue; // declined — not a divergence
                };
                let expected = interp(&chunk);
                if !expected.as_ref().is_some_and(|e| same(e, &native)) {
                    diffs.push(format!(
                        "{:?}({}, {}): interp={} native={}",
                        op,
                        a.label,
                        b.label,
                        describe(&expected),
                        describe(&Some(native))
                    ));
                }
            }
        }
    }
    diffs
}

fn diff_unary(ops: &[Op], operands: &[Operand]) -> Vec<String> {
    let jit = JitCompiler::new();
    let mut diffs = Vec::new();
    for op in ops {
        for a in operands {
            let chunk = chunk_for(a, None, op);
            let Some(native) = jit.try_run_linear(&chunk, &[]) else {
                continue;
            };
            let expected = interp(&chunk);
            if !expected.as_ref().is_some_and(|e| same(e, &native)) {
                diffs.push(format!(
                    "{:?}({}): interp={} native={}",
                    op,
                    a.label,
                    describe(&expected),
                    describe(&Some(native))
                ));
            }
        }
    }
    diffs
}

fn assert_no_diffs(what: &str, diffs: Vec<String>) {
    if !diffs.is_empty() {
        for d in &diffs {
            eprintln!("TIER DIVERGENCE {what}: {d}");
        }
        panic!("{} interpreter/JIT divergence(s) in {what}", diffs.len());
    }
}

#[test]
fn linear_jit_matches_interpreter_on_arithmetic() {
    assert_no_diffs(
        "arithmetic",
        diff_binary(
            &[Op::Add, Op::Sub, Op::Mul, Op::Div, Op::Mod, Op::Pow],
            &mixed_operands(),
        ),
    );
}

#[test]
fn linear_jit_matches_interpreter_on_bitwise_and_shifts() {
    assert_no_diffs(
        "bitwise",
        diff_binary(
            &[Op::BitAnd, Op::BitOr, Op::BitXor, Op::Shl, Op::Shr],
            &mixed_operands(),
        ),
    );
}

#[test]
fn linear_jit_matches_interpreter_on_comparisons() {
    assert_no_diffs(
        "comparison",
        diff_binary(
            &[
                Op::NumEq,
                Op::NumNe,
                Op::NumLt,
                Op::NumGt,
                Op::NumLe,
                Op::NumGe,
                Op::Spaceship,
            ],
            &mixed_operands(),
        ),
    );
}

#[test]
fn linear_jit_matches_interpreter_on_int_intrinsics() {
    assert_no_diffs(
        "int intrinsics",
        diff_binary(&[Op::GcdInt, Op::LcmInt], &int_operands()),
    );
}

#[test]
fn linear_jit_matches_interpreter_on_unary_ops() {
    assert_no_diffs(
        "unary",
        diff_unary(
            &[
                Op::Negate,
                Op::Inc,
                Op::Dec,
                Op::BitNot,
                Op::LogNot,
                Op::AbsInt,
                Op::TruncInt,
                Op::RubyTruthy,
            ],
            &mixed_operands(),
        ),
    );
}

/// The boolean axis: a `Value::Bool` reaching a native op, and a `Value::Bool`
/// coming back out of one. Both directions were wrong before the linear tier
/// grew a boolean cell kind — `Op::NumLt` handed back `Int(1)` where the
/// interpreter and the AOT tier both answer `Bool(true)`, and `true + 1`
/// folded to `Int(2)` where both answer `Float(2.0)`.
#[test]
fn linear_jit_matches_interpreter_on_boolean_operands() {
    let mut diffs = diff_binary(
        &[
            Op::Add,
            Op::Sub,
            Op::Mul,
            Op::Div,
            Op::Mod,
            Op::Pow,
            Op::BitAnd,
            Op::BitOr,
            Op::BitXor,
            Op::Shl,
            Op::Shr,
            Op::NumEq,
            Op::NumLt,
            Op::Spaceship,
            Op::GcdInt,
            Op::LcmInt,
        ],
        &all_operands(),
    );
    diffs.extend(diff_unary(
        &[
            Op::Negate,
            Op::Inc,
            Op::Dec,
            Op::BitNot,
            Op::LogNot,
            Op::AbsInt,
            Op::TruncInt,
            Op::SqrtFloat,
            Op::AwkInt,
            Op::AwkMkbool,
        ],
        &all_operands(),
    ));
    assert_no_diffs("boolean operands", diffs);
}

#[test]
fn linear_jit_matches_interpreter_on_float_intrinsics() {
    assert_no_diffs(
        "float intrinsics",
        diff_unary(
            &[
                Op::SqrtFloat,
                Op::AbsFloat,
                Op::TruncFloat,
                Op::RoundFloat,
                Op::SinFloat,
                Op::CosFloat,
                Op::ExpFloat,
                Op::LogFloat,
                Op::Log2Float,
                Op::Log10Float,
            ],
            &mixed_operands(),
        ),
    );
}

// ── AOT tier ──
//
// `src/aot.rs` is a *separate* compiler from `src/jit.rs` with its own kind
// lattice and its own lowering of every op, so linear-tier agreement says
// nothing about it. `Op::GcdInt` is the proof: the linear tier saturated the
// unrepresentable `gcd(0, i64::MIN)` magnitude at `i64::MAX` while the
// interpreter and the AOT compiler both wrapped it to a *negative* gcd.

/// Cross `ops` with the operand matrix through the AOT compiler.
#[cfg(feature = "aot")]
fn diff_aot(ops: &[Op], operands: &[Operand], binary: bool) -> Vec<String> {
    let mut diffs = Vec::new();
    for op in ops {
        for a in operands {
            let bs: Vec<Option<&Operand>> = if binary {
                operands.iter().map(Some).collect()
            } else {
                vec![None]
            };
            for b in bs {
                let chunk = chunk_for(a, b, op);
                let native = match fusevm::aot::run_chunk_native(&chunk, |_| {}) {
                    Ok(VMResult::Ok(v)) => Some(v),
                    Ok(VMResult::Error(_)) | Err(_) => None,
                    Ok(VMResult::Halted) => continue,
                };
                let expected = interp(&chunk);
                let agree = match (&expected, &native) {
                    (Some(e), Some(n)) => same(e, n),
                    (None, None) => true,
                    _ => false,
                };
                if !agree {
                    diffs.push(format!(
                        "{:?}({}{}): interp={} aot={}",
                        op,
                        a.label,
                        b.map(|b| format!(", {}", b.label)).unwrap_or_default(),
                        describe(&expected),
                        describe(&native)
                    ));
                }
            }
        }
    }
    diffs
}

#[cfg(feature = "aot")]
#[test]
fn aot_matches_interpreter_on_int_intrinsics() {
    assert_no_diffs(
        "aot int intrinsics",
        diff_aot(&[Op::GcdInt, Op::LcmInt], &int_operands(), true),
    );
}

#[cfg(feature = "aot")]
#[test]
fn aot_matches_interpreter_on_unary_ops() {
    assert_no_diffs(
        "aot unary",
        diff_aot(
            &[
                Op::Negate,
                Op::Inc,
                Op::Dec,
                Op::BitNot,
                Op::LogNot,
                Op::AbsInt,
                Op::TruncInt,
            ],
            &all_operands(),
            false,
        ),
    );
}

#[cfg(feature = "aot")]
#[test]
fn aot_matches_interpreter_on_arithmetic_and_bitwise() {
    assert_no_diffs(
        "aot arithmetic",
        diff_aot(
            &[
                Op::Add,
                Op::Sub,
                Op::Mul,
                Op::Div,
                Op::Mod,
                Op::Pow,
                Op::BitAnd,
                Op::BitOr,
                Op::BitXor,
                Op::Shl,
                Op::Shr,
            ],
            &all_operands(),
            true,
        ),
    );
}

#[cfg(feature = "aot")]
#[test]
fn aot_matches_interpreter_on_comparisons() {
    assert_no_diffs(
        "aot comparison",
        diff_aot(
            &[
                Op::NumEq,
                Op::NumNe,
                Op::NumLt,
                Op::NumGt,
                Op::NumLe,
                Op::NumGe,
                Op::Spaceship,
            ],
            &all_operands(),
            true,
        ),
    );
}

// ── Block tier ──
//
// The third compiler. `src/jit.rs`'s block path (`build_block_function`) has
// its own slot model and its own `Op::SetSlot`/`Op::GetSlot` lowering, and it
// *warms up*: `block_threshold` defaults to 1, so a chunk runs on the
// interpreter once and in native code forever after. A block/interpreter
// disagreement is therefore not a static wrong answer but a `Value` that
// changes on the second call of the same `Chunk` on the same `VM` —
// `VM::run` boxes the block result directly (`src/vm.rs`, `BlockNum::Int(n)
// => Value::Int(n)`), so nothing downstream re-derives the kind.
//
// The block return channel is numeric (`BlockNum`), so an interpreter `Bool`
// is compared against `Int(0|1)` here. That leniency is NOT a judgement that
// the kind does not matter — see `block_jit_preserves_boolean_result_kind`
// below, which asserts the strict contract and is `#[ignore]`d because the
// block tier does not meet it yet.

/// Run `chunk` through the block JIT eagerly and describe the result.
fn block(chunk: &Chunk) -> Option<fusevm::BlockNum> {
    let jit = JitCompiler::new();
    let mut slots = vec![0i64; 8];
    let kinds = vec![fusevm::SlotKind::Int; 8];
    jit.try_run_block_eager_typed_kinded(chunk, &mut slots, &kinds)
}

/// A block-tier chunk needs a frame so the slot ops are legal.
fn block_chunk_for(a: &Operand, b: Option<&Operand>, op: &Op) -> Chunk {
    let mut bd = ChunkBuilder::new();
    bd.emit(Op::PushFrame, 1);
    for o in &a.push {
        bd.emit(o.clone(), 1);
    }
    if let Some(b) = b {
        for o in &b.push {
            bd.emit(o.clone(), 1);
        }
    }
    bd.emit(op.clone(), 1);
    bd.build()
}

fn describe_block(v: &Option<fusevm::BlockNum>) -> String {
    match v {
        None => "<declined>".to_string(),
        Some(fusevm::BlockNum::Int(n)) => format!("Int({n})"),
        Some(fusevm::BlockNum::Float(f)) => format!("Float({f:?} bits={:#x})", f.to_bits()),
    }
}

/// Declining is always correct — the caller falls back to the interpreter.
/// Answering differently is not.
///
/// There is deliberately no `Bool`-answers-`Int` arm. The block tier has no
/// boolean kind, so it must *decline* every chunk whose result is a
/// `Value::Bool` rather than flatten it to `Int(0|1)` — see
/// `bool_is_consumed_in_place` in `src/jit.rs`. Accepting `Bool(true)` against
/// `BlockNum::Int(1)` here is what let that flattening go unnoticed.
fn block_agrees(expected: &Option<Value>, got: &Option<fusevm::BlockNum>) -> bool {
    match (expected, got) {
        (_, None) => true,
        (Some(Value::Int(a)), Some(fusevm::BlockNum::Int(b))) => a == b,
        (Some(Value::Float(a)), Some(fusevm::BlockNum::Float(b))) => {
            a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
        }
        _ => false,
    }
}

fn diff_block(ops: &[Op], operands: &[Operand], binary: bool) -> Vec<String> {
    let mut diffs = Vec::new();
    for op in ops {
        for a in operands {
            let bs: Vec<Option<&Operand>> = if binary {
                operands.iter().map(Some).collect()
            } else {
                vec![None]
            };
            for b in bs {
                let chunk = block_chunk_for(a, b, op);
                let got = block(&chunk);
                let expected = interp(&chunk_for(a, b, op));
                if !block_agrees(&expected, &got) {
                    diffs.push(format!(
                        "{:?}({}{}): interp={} block={}",
                        op,
                        a.label,
                        b.map(|b| format!(", {}", b.label)).unwrap_or_default(),
                        describe(&expected),
                        describe_block(&got)
                    ));
                }
            }
        }
    }
    diffs
}

#[test]
fn block_jit_matches_interpreter_on_unary_ops() {
    assert_no_diffs(
        "block unary",
        diff_block(
            &[
                Op::Negate,
                Op::Inc,
                Op::Dec,
                Op::BitNot,
                Op::LogNot,
                Op::AbsInt,
                Op::TruncInt,
                Op::AbsFloat,
                Op::CeilFloat,
                Op::FloorFloat,
                Op::TruncFloat,
                Op::RoundFloat,
                Op::SqrtFloat,
            ],
            &mixed_operands(),
            false,
        ),
    );
}

#[test]
fn block_jit_matches_interpreter_on_transcendentals() {
    assert_no_diffs(
        "block transcendentals",
        diff_block(
            &[
                Op::SinFloat,
                Op::CosFloat,
                Op::TanFloat,
                Op::AsinFloat,
                Op::AcosFloat,
                Op::AtanFloat,
                Op::SinhFloat,
                Op::CoshFloat,
                Op::TanhFloat,
                Op::ExpFloat,
                Op::LogFloat,
                Op::Log2Float,
                Op::Log10Float,
                Op::AwkSin,
                Op::AwkCos,
                Op::AwkExp,
                Op::AwkMkbool,
            ],
            &mixed_operands(),
            false,
        ),
    );
}

#[test]
fn block_jit_matches_interpreter_on_arithmetic_and_bitwise() {
    assert_no_diffs(
        "block arithmetic",
        diff_block(
            &[
                Op::Add,
                Op::Sub,
                Op::Mul,
                Op::Div,
                Op::Mod,
                Op::Pow,
                Op::PowFloat,
                Op::BitAnd,
                Op::BitOr,
                Op::BitXor,
                Op::Shl,
                Op::Shr,
            ],
            &mixed_operands(),
            true,
        ),
    );
}

#[test]
fn block_jit_matches_interpreter_on_comparisons_and_int_intrinsics() {
    assert_no_diffs(
        "block comparison",
        diff_block(
            &[
                Op::NumEq,
                Op::NumNe,
                Op::NumLt,
                Op::NumGt,
                Op::NumLe,
                Op::NumGe,
                Op::Spaceship,
                Op::GcdInt,
                Op::LcmInt,
                Op::Atan2Float,
                Op::AwkAtan2,
            ],
            &mixed_operands(),
            true,
        ),
    );
}

// ── `Op::AwkInt` is host-dispatched and must never be lowered natively ──
//
// `VM::run` sends it to `AwkHost::int`, an overridable trait method whose
// implementations disagree on the result *variant*: the default
// (`awk_host::awk_int`) answers `Int(3)` for `int(3.7)`, while a frontend that
// models awk numbers as `f64` answers `Float(3.0)`. Native code sees neither,
// so no lowering is right for every host.

/// An `AwkHost` that models awk numbers as `f64` — `int()` is always a Float.
struct FloatIntHost;
impl fusevm::AwkHost for FloatIntHost {
    fn int(&mut self, x: &Value) -> Value {
        Value::Float(x.to_float().trunc())
    }
}

fn awk_int_chunk(push: Op, salt: i64) -> Chunk {
    let mut b = ChunkBuilder::new();
    b.emit(Op::PushFrame, 1);
    // Unique padding: the block cache is keyed on `chunk.op_hash`, so two
    // cases built from the same ops share a cache entry and the second one
    // never gets a cold (interpreter) run.
    b.emit(Op::LoadInt(salt), 1);
    b.emit(Op::Pop, 1);
    b.emit(push, 1);
    b.emit(Op::AwkInt, 1);
    b.build()
}

/// The regression this pins: the answer must not depend on how many times the
/// chunk has run. Before the fix, `int(3.7)` was `Int(3)` on the first call
/// and `Float(3.0)` on every call after, because the block JIT compiled a
/// kind-preserving `trunc` behind the interpreter's back.
#[test]
fn awk_int_answer_is_stable_across_block_warmup() {
    let mut salt = 7_000_000;
    for push in [
        Op::LoadFloat(3.7),
        Op::LoadFloat(-2.5),
        Op::LoadFloat(0.5),
        Op::LoadInt(2),
        Op::LoadInt((1i64 << 53) + 1),
        Op::LoadInt(i64::MAX - 1),
        Op::LoadInt(i64::MIN + 1),
    ] {
        for use_host in [false, true] {
            salt += 1;
            let chunk = awk_int_chunk(push.clone(), salt);
            let mut answers = Vec::new();
            for _ in 0..4 {
                let mut vm = VM::new(chunk.clone());
                if use_host {
                    vm.set_awk_host(Box::new(FloatIntHost));
                }
                vm.enable_tracing_jit();
                answers.push(describe(&match vm.run() {
                    VMResult::Ok(v) => Some(v),
                    _ => None,
                }));
            }
            assert!(
                answers.windows(2).all(|w| w[0] == w[1]),
                "AwkInt({push:?}) host={use_host} changed answer across block warmup: {answers:?}"
            );
        }
    }
}

/// And the answer must be the *host's*, since the host is what the op
/// dispatches to. With a host that models awk numbers as `f64`, `int()` is a
/// `Float` for every operand — including integer ones, where the old native
/// lowering returned the operand unchanged as an `Int`.
#[test]
fn awk_int_honours_a_registered_host_for_every_operand() {
    let mut salt = 7_500_000;
    for push in [
        Op::LoadFloat(3.7),
        Op::LoadFloat(-2.5),
        Op::LoadInt(0),
        Op::LoadInt(2),
        Op::LoadInt(-2),
        Op::LoadInt((1i64 << 53) + 1),
        Op::LoadInt(i64::MAX),
    ] {
        salt += 1;
        let chunk = awk_int_chunk(push.clone(), salt);
        for call in 1..=3 {
            let mut vm = VM::new(chunk.clone());
            vm.set_awk_host(Box::new(FloatIntHost));
            vm.enable_tracing_jit();
            match vm.run() {
                VMResult::Ok(Value::Float(_)) => {}
                other => panic!(
                    "AwkInt({push:?}) call #{call}: host returns Float for every operand, \
                     got {other:?}"
                ),
            }
        }
    }
}

/// The value error that survived past `2^53` even when the variant matched:
/// the default host truncates *through an `f64`*, native code never left the
/// integer register, and the two answered different integers.
#[test]
fn awk_int_matches_the_default_host_past_2_pow_53() {
    let mut salt = 7_900_000;
    for (n, want) in [
        ((1i64 << 53) + 1, 9_007_199_254_740_992i64),
        (i64::MAX - 1, i64::MAX),
        (i64::MIN + 1, i64::MIN),
    ] {
        salt += 1;
        let chunk = awk_int_chunk(Op::LoadInt(n), salt);
        for call in 1..=3 {
            let mut vm = VM::new(chunk.clone());
            vm.enable_tracing_jit();
            match vm.run() {
                VMResult::Ok(Value::Int(got)) => assert_eq!(
                    got, want,
                    "int({n}) call #{call}: default host answers {want}"
                ),
                other => panic!("int({n}) call #{call}: expected Int({want}), got {other:?}"),
            }
        }
    }
}

// ── Traps: native code must not execute an illegal instruction where the
//    interpreter returns a value ──

/// `Op::Inc`/`Op::Dec` on a float operand. The interpreter is
/// `Value::Int(v.to_int().wrapping_add(1))` and `Value::to_int` saturates
/// (Rust `f as i64`), so `Inc(1e30)` is `Int(i64::MIN)`. The block tier used
/// Cranelift's *trapping* `fcvt_to_sint` and died with SIGILL instead — and
/// nothing guarded it, because `is_block_eligible_op` rejects `Inc`/`Dec`
/// only under `strict_numeric()`, which is not the default.
#[test]
fn block_jit_inc_dec_saturate_instead_of_trapping_on_huge_floats() {
    for (op, f, want) in [
        (Op::Inc, 1e30f64, i64::MIN),
        (Op::Inc, -1e30f64, i64::MIN + 1),
        (Op::Dec, 1e30f64, i64::MAX - 1),
        (Op::Dec, -1e30f64, i64::MAX),
    ] {
        let mut b = ChunkBuilder::new();
        b.emit(Op::PushFrame, 1);
        b.emit(Op::LoadFloat(f), 1);
        b.emit(op.clone(), 1);
        let chunk = b.build();
        let expected = interp(&chunk);
        assert_eq!(
            expected,
            Some(Value::Int(want)),
            "interpreter reference for {op:?}({f})"
        );
        // Reaching this line at all is half the assertion: before the fix the
        // process died here with SIGILL and the test binary never reported.
        match block(&chunk) {
            None => {}
            Some(fusevm::BlockNum::Int(got)) => assert_eq!(
                got, want,
                "{op:?}({f}): block must saturate like Value::to_int"
            ),
            other => panic!("{op:?}({f}): expected Int({want}), got {other:?}"),
        }
    }
}

/// `Op::SetSlot` of a float into a slot the caller declared `SlotKind::Int`.
/// Memory-backed and register-promoted slots are both raw `i64`, so the value
/// cannot survive; the block tier truncated it through the trapping
/// conversion. `x = 3.5` into a slot that entered the chunk holding an integer
/// read back as `Int(3)` once warm, and `x = 1e30` was a SIGILL.
#[test]
fn block_jit_declines_storing_a_float_into_an_int_kinded_slot() {
    for f in [3.5f64, -2.5, 1e30, -1e30, 0.5] {
        let mut b = ChunkBuilder::new();
        b.emit(Op::PushFrame, 1);
        b.emit(Op::LoadFloat(f), 1);
        b.emit(Op::SetSlot(0), 1);
        b.emit(Op::GetSlot(0), 1);
        let chunk = b.build();
        assert_eq!(
            block(&chunk),
            None,
            "storing {f} into an Int-kinded slot has no correct i64 lowering; \
             the block tier must decline so the interpreter runs it"
        );
    }
}

/// End-to-end through `VM::run`: the same shape a frontend actually emits for
/// `x = 3.5` when `x` entered the chunk holding an integer. The float must
/// survive every call, not just the cold one.
#[test]
fn float_assignment_to_an_int_slot_survives_block_warmup() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadFloat(3.5), 1);
    b.emit(Op::SetSlot(0), 1);
    b.emit(Op::GetSlot(0), 1);
    let chunk = b.build();
    for call in 1..=4 {
        let mut vm = VM::new(chunk.clone());
        vm.frames.last_mut().unwrap().slots.push(Value::Int(0));
        vm.enable_tracing_jit();
        match vm.run() {
            VMResult::Ok(Value::Float(got)) => assert_eq!(got, 3.5, "call #{call}"),
            other => panic!("call #{call}: expected Float(3.5), got {other:?}"),
        }
    }
}

// ── The block/tracing lattice has no boolean kind, so it must decline ──
//
// `JitTy` is Int-or-Float and `BlockNum` is Int-or-Float, so a `Value::Bool`
// has nowhere to live. `VM::run` boxes `BlockNum::Int(n)` as `Value::Int(n)`,
// so every boolean-valued block-eligible chunk changed variant on its second
// call: `1 < 2` was `Bool(true)` cold and `Int(1)` warm, and `Bool(false)`
// stringifies as `""` where `Int(0)` stringifies as `"0"`. A `Bool` *operand*
// was the same gap in the other direction — `true + 1` is `Float(2.0)` on the
// interpreter (booleans coerce through `to_float`) and was `Int(2)` in the
// block tier.
//
// The linear tier solved this with a `Cell::ConstB`/`Cell::DynB`, and the AOT
// tier with a `Kind::Bool`. Widening this lattice would touch every `JitTy`
// match plus the public `BlockNum`, so instead the tier now *declines* any
// chunk where a boolean outlives the op that produced it
// (`bool_is_consumed_in_place` in `src/jit.rs`) and the interpreter runs it.
// Declining is always a correct answer; flattening to `Int` was not.
#[test]
fn block_jit_preserves_boolean_result_kind() {
    let mut diffs = Vec::new();
    for (label, ops) in [
        ("1 < 2", vec![Op::LoadInt(1), Op::LoadInt(2), Op::NumLt]),
        ("1 > 2", vec![Op::LoadInt(1), Op::LoadInt(2), Op::NumGt]),
        ("1 == 1", vec![Op::LoadInt(1), Op::LoadInt(1), Op::NumEq]),
        ("LoadTrue", vec![Op::LoadTrue]),
        ("LoadFalse", vec![Op::LoadFalse]),
        ("!0", vec![Op::LoadInt(0), Op::LogNot]),
        ("true + 1", vec![Op::LoadTrue, Op::LoadInt(1), Op::Add]),
    ] {
        let mut b = ChunkBuilder::new();
        for o in &ops {
            b.emit(o.clone(), 1);
        }
        let chunk = b.build();
        let mut answers = Vec::new();
        for _ in 0..3 {
            let mut vm = VM::new(chunk.clone());
            vm.enable_tracing_jit();
            answers.push(describe(&match vm.run() {
                VMResult::Ok(v) => Some(v),
                _ => None,
            }));
        }
        if !answers.windows(2).all(|w| w[0] == w[1]) {
            diffs.push(format!("{label}: {answers:?}"));
        }
    }
    assert_no_diffs("block boolean kind", diffs);
}

// ── Tracing tier ──
//
// The fourth compiler, and the one this file could not see at all until now:
// every harness above drives `try_run_linear` / `try_run_block_*` /
// `run_chunk_native`, and a trace anchors only on a **conditional backward
// branch**. The whole corpus above is straight-line, so no chunk in it ever
// closed a trace — the tracing tier was scoring agreement it had never been
// asked a single question about.
//
// It was not agreeing. With the op under test inside a hot loop, 252 of the
// 589 compiling combinations disagreed with the interpreter, and the failures
// were not confined to the variant: a boolean stored into a float-kinded slot
// came back as a *bit pattern*, so `0 + true` answered `Float(5e-324)` (the
// bits of `1`) and `0 - true` answered `NaN` (the bits of `-1`) where the
// interpreter answers `Float(1.0)` / `Float(-1.0)`.

/// Wrap `a OP b` in a hot do-while loop so the op is recorded into a trace,
/// and hand the computed value back through a slot.
///
/// ```text
///   LoadInt(salt); Pop            // unique op_hash per case
///   LoadInt(0); SetSlot(0)
/// anchor:
///   PreIncSlotVoid(0)             // loop counter
///   <push a> <push b> OP SetSlot(1)
///   GetSlot(0) LoadInt(200) NumLt JumpIfTrue(anchor)
///   GetSlot(1)                    // the value under test
/// ```
fn trace_chunk_for(a: &Operand, b: Option<&Operand>, op: &Op, salt: i64) -> (Chunk, usize) {
    let mut bd = ChunkBuilder::new();
    bd.emit(Op::LoadInt(salt), 1);
    bd.emit(Op::Pop, 1);
    bd.emit(Op::LoadInt(0), 1);
    bd.emit(Op::SetSlot(0), 1);
    let anchor = bd.current_pos();
    bd.emit(Op::PreIncSlotVoid(0), 1);
    for o in &a.push {
        bd.emit(o.clone(), 1);
    }
    if let Some(b) = b {
        for o in &b.push {
            bd.emit(o.clone(), 1);
        }
    }
    bd.emit(op.clone(), 1);
    bd.emit(Op::SetSlot(1), 1);
    bd.emit(Op::GetSlot(0), 1);
    bd.emit(Op::LoadInt(200), 1);
    bd.emit(Op::NumLt, 1);
    let jmp = bd.emit(Op::JumpIfTrue(0), 1);
    bd.patch_jump(jmp, anchor);
    bd.emit(Op::GetSlot(1), 1);
    (bd.build(), anchor)
}

/// Run `chunk` with the given slot count, optionally with the tracing JIT on.
fn run_with_slots(chunk: &Chunk, tracing: bool) -> Option<Value> {
    let mut vm = VM::new(chunk.clone());
    if tracing {
        vm.enable_tracing_jit();
    }
    let frame = vm.frames.last_mut().unwrap();
    while frame.slots.len() < 4 {
        frame.slots.push(Value::Int(0));
    }
    match vm.run() {
        VMResult::Ok(v) => Some(v),
        VMResult::Halted => vm.stack.last().cloned(),
        VMResult::Error(_) => None,
    }
}

/// Cross `ops` with `operands` through the tracing tier.
///
/// Returns `(diffs, compiled)`. A case whose trace never compiled is SKIPPED,
/// not scored — but `compiled` is reported so a caller can refuse to pass on an
/// empty corpus. The reference is always the interpreter running the *same*
/// chunk with tracing off, never the traced answer.
fn diff_trace(ops: &[Op], operands: &[Operand], binary: bool, salt0: i64) -> (Vec<String>, usize) {
    let jit = JitCompiler::new();
    let mut diffs = Vec::new();
    let mut compiled = 0usize;
    let mut salt = salt0;
    for op in ops {
        for a in operands {
            let bs: Vec<Option<&Operand>> = if binary {
                operands.iter().map(Some).collect()
            } else {
                vec![None]
            };
            for b in bs {
                salt += 1;
                let (chunk, anchor) = trace_chunk_for(a, b, op, salt);
                let traced = run_with_slots(&chunk, true);
                if !jit.trace_is_compiled(&chunk, anchor) {
                    continue; // never reached the tier — not evidence of agreement
                }
                compiled += 1;
                let expected = run_with_slots(&chunk, false);
                let agree = match (&expected, &traced) {
                    (Some(e), Some(t)) => same(e, t),
                    (None, None) => true,
                    _ => false,
                };
                if !agree {
                    diffs.push(format!(
                        "{:?}({}{}): interp={} trace={}",
                        op,
                        a.label,
                        b.map(|b| format!(", {}", b.label)).unwrap_or_default(),
                        describe(&expected),
                        describe(&traced)
                    ));
                }
            }
        }
    }
    (diffs, compiled)
}

fn assert_trace_clean(what: &str, (diffs, compiled): (Vec<String>, usize)) {
    assert!(
        compiled > 0,
        "{what}: no case in this corpus ever compiled a trace, so a clean run \
         proves nothing. A trace anchors only on a conditional backward branch \
         — check the loop shape before trusting this test."
    );
    assert_no_diffs(what, diffs);
}

#[test]
fn tracing_jit_matches_interpreter_on_arithmetic_and_bitwise() {
    assert_trace_clean(
        "trace arithmetic",
        diff_trace(
            &[
                Op::Add,
                Op::Sub,
                Op::Mul,
                Op::Div,
                Op::Mod,
                Op::Pow,
                Op::BitAnd,
                Op::BitOr,
                Op::BitXor,
                Op::Shl,
                Op::Shr,
            ],
            &all_operands(),
            true,
            1_000_000,
        ),
    );
}

#[test]
fn tracing_jit_matches_interpreter_on_comparisons_and_int_intrinsics() {
    assert_trace_clean(
        "trace comparison",
        diff_trace(
            &[
                Op::NumEq,
                Op::NumNe,
                Op::NumLt,
                Op::NumGt,
                Op::NumLe,
                Op::NumGe,
                Op::Spaceship,
                Op::GcdInt,
                Op::LcmInt,
            ],
            &all_operands(),
            true,
            2_000_000,
        ),
    );
}

#[test]
fn tracing_jit_matches_interpreter_on_unary_ops() {
    assert_trace_clean(
        "trace unary",
        diff_trace(
            &[
                Op::Negate,
                Op::Inc,
                Op::Dec,
                Op::BitNot,
                Op::LogNot,
                Op::AbsInt,
                Op::TruncInt,
                Op::AbsFloat,
                Op::CeilFloat,
                Op::FloorFloat,
                Op::TruncFloat,
                Op::RoundFloat,
                Op::SqrtFloat,
            ],
            &all_operands(),
            false,
            3_000_000,
        ),
    );
}
