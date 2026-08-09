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
