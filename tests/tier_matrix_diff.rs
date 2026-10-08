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

/// Non-finite floats. `Op::LoadFloat` of a non-finite constant declines the
/// linear tier outright, so these are *computed* in-chunk — the only way NaN
/// and the infinities reach a native op. They separate saturating from
/// trapping float->int conversions (`fcvt_to_sint_sat(NaN)` is `0`, as is
/// Rust's `NaN as i64`) and ordered from unordered float compares.
fn nonfinite_operands() -> Vec<Operand> {
    let inf = vec![Op::LoadFloat(1e300), Op::LoadFloat(1e300), Op::Mul];
    let mut neg_inf = inf.clone();
    neg_inf.push(Op::Negate);
    let mut nan = inf.clone();
    nan.extend(inf.clone());
    nan.push(Op::Sub);
    vec![
        Operand {
            label: "inf",
            push: inf,
        },
        Operand {
            label: "-inf",
            push: neg_inf,
        },
        Operand {
            label: "NaN",
            push: nan,
        },
    ]
}

fn mixed_operands() -> Vec<Operand> {
    let mut v = int_operands();
    v.extend(float_operands());
    v
}

/// Every numeric operand kind, non-finite floats included.
fn numeric_operands() -> Vec<Operand> {
    let mut v = mixed_operands();
    v.extend(nonfinite_operands());
    v
}

/// Two operands pushed back to back, for ops that consume more than two
/// stack values: `diff_binary(op, pairs × operands)` drives a ternary op
/// through the same harness as every binary one. The label is leaked because
/// `Operand` labels are `&'static str`; test-binary lifetime, bounded count.
fn pair(a: &Operand, b: &Operand) -> Operand {
    let mut push = a.push.clone();
    push.extend(b.push.iter().cloned());
    Operand {
        label: Box::leak(format!("{}, {}", a.label, b.label).into_boxed_str()),
        push,
    }
}

/// Every operand kind the linear tier can see, booleans included.
fn all_operands() -> Vec<Operand> {
    let mut v = mixed_operands();
    v.extend(bool_operands());
    v
}

fn chunk_for(a: &Operand, b: Option<&Operand>, op: &Op) -> Chunk {
    let operands: Vec<&Operand> = std::iter::once(a).chain(b).collect();
    chunk_of(&operands, |_| vec![op.clone()])
}

/// Push each operand in order, then append the ops `tail` builds — the general
/// form of `chunk_for`, for sequences of more than one op or more than two
/// operands. `tail` is handed the ip it starts at: operand pushes vary in
/// length, and a jump in the tail needs an absolute target.
fn chunk_of(operands: &[&Operand], tail: impl FnOnce(usize) -> Vec<Op>) -> Chunk {
    let mut bd = ChunkBuilder::new();
    let mut base = 0;
    for o in operands.iter().flat_map(|a| &a.push) {
        bd.emit(o.clone(), 1);
        base += 1;
    }
    for o in tail(base) {
        bd.emit(o, 1);
    }
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
                if let Some(d) = linear_diff(&jit, &chunk, &[]) {
                    diffs.push(format!("{:?}({}, {}): {d}", op, a.label, b.label));
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
            if let Some(d) = linear_diff(&jit, &chunk, &[]) {
                diffs.push(format!("{:?}({}): {d}", op, a.label));
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

/// The rest of the linear tier's always-float unary lowerings — every op
/// `emit_data_op` routes through `pop_as_f64` — plus the int intrinsics that
/// coerce a float operand through a saturating conversion, all crossed with
/// every operand kind: booleans, and the computed non-finite floats that
/// `LoadFloat` alone can never produce.
#[test]
fn linear_jit_matches_interpreter_on_remaining_unary_ops() {
    let mut operands = numeric_operands();
    operands.extend(bool_operands());
    assert_no_diffs(
        "remaining unary",
        diff_unary(
            &[
                Op::CeilFloat,
                Op::FloorFloat,
                Op::TanFloat,
                Op::AsinFloat,
                Op::AcosFloat,
                Op::AtanFloat,
                Op::SinhFloat,
                Op::CoshFloat,
                Op::TanhFloat,
                Op::AwkMkbool,
                Op::AwkSin,
                Op::AwkCos,
                Op::AwkExp,
                // Already in the matrix above, but never against NaN/±inf.
                Op::Negate,
                Op::Inc,
                Op::Dec,
                Op::LogNot,
                Op::AbsInt,
                Op::TruncInt,
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
            &operands,
        ),
    );
}

/// Binary float intrinsics, the int intrinsics against float operands (the
/// interpreter coerces through `Value::to_int`), and the binary ops already
/// covered above re-run against NaN/±inf.
#[test]
fn linear_jit_matches_interpreter_on_remaining_binary_ops() {
    let mut operands = numeric_operands();
    operands.extend(bool_operands());
    let mut diffs = diff_binary(
        &[
            Op::PowFloat,
            Op::Atan2Float,
            Op::AwkAtan2,
            Op::GcdInt,
            Op::LcmInt,
        ],
        &operands,
    );
    // The already-covered ops only need the crosses that involve a
    // non-finite operand; the rest of their matrix runs above.
    let covered = [
        Op::Add,
        Op::Sub,
        Op::Mul,
        Op::Div,
        Op::Mod,
        Op::Pow,
        Op::NumEq,
        Op::NumNe,
        Op::NumLt,
        Op::NumGt,
        Op::NumLe,
        Op::NumGe,
        Op::Spaceship,
    ];
    let nonfinite = nonfinite_operands();
    for a in &nonfinite {
        diffs.extend(diff_binary_with(&covered, a, &operands));
    }
    for a in &operands {
        diffs.extend(diff_binary_with(&covered, a, &nonfinite));
    }
    assert_no_diffs("remaining binary", diffs);
}

/// `Op::MulModFloor` (3 operands) and `Op::MulAddModFloor` (4), driven through
/// `diff_binary` by pairing operands. The interpreter takes the fused i128
/// path only when every operand is a `Value::Int`; anything else replays the
/// unfused `Mul`/`Add`/`Mod` with their float coercions.
#[test]
fn linear_jit_matches_interpreter_on_fused_mod_ops() {
    // A cut of the operand edges: every JIT compile costs milliseconds in a
    // debug build and a full 4-operand cross is ~10^5 chunks. Kept: the
    // floor-vs-truncate signs, the i128-only products, a zero divisor (both
    // sides answer 0), and one float and one bool operand per position.
    let mut scalars = mixed_operands();
    scalars.extend(bool_operands());
    scalars.retain(|o| {
        [
            "0", "1", "-1", "-2", "7", "i64::MIN", "i64::MAX", "2^53+1", "-2.5", "true",
        ]
        .contains(&o.label)
    });
    let pairs: Vec<Operand> = scalars
        .iter()
        .flat_map(|a| scalars.iter().map(move |b| pair(a, b)))
        .collect();
    let mut diffs = Vec::new();
    for p in &pairs {
        diffs.extend(diff_binary_with(&[Op::MulModFloor], p, &scalars));
    }
    // `a*b + c` — pair a pair with a third operand, keep `k` last.
    let triples: Vec<Operand> = pairs
        .iter()
        .step_by(3)
        .flat_map(|ab| scalars.iter().map(move |c| pair(ab, c)))
        .collect();
    for t in &triples {
        diffs.extend(diff_binary_with(&[Op::MulAddModFloor], t, &scalars));
    }
    assert_no_diffs("fused mod", diffs);
}

/// `diff_binary` with the left operand fixed — keeps a ternary/quaternary
/// matrix at `|pairs| × |divisors|` instead of `|pairs| × |pairs|`.
fn diff_binary_with(ops: &[Op], a: &Operand, bs: &[Operand]) -> Vec<String> {
    let jit = JitCompiler::new();
    let mut diffs = Vec::new();
    for op in ops {
        for b in bs {
            let chunk = chunk_for(a, Some(b), op);
            if let Some(d) = linear_diff(&jit, &chunk, &[]) {
                diffs.push(format!("{:?}({}, {}): {d}", op, a.label, b.label));
            }
        }
    }
    diffs
}

/// Run `chunk` on the linear tier with `slots` and on the interpreter with the
/// same slot values (as `Value::Int`, the only kind a linear slot holds), and
/// describe the disagreement, if any. A declined chunk is not a divergence.
fn linear_diff(jit: &JitCompiler, chunk: &Chunk, slots: &[i64]) -> Option<String> {
    let native = jit.try_run_linear(chunk, slots)?;
    let mut vm = VM::new(chunk.clone());
    vm.frames
        .last_mut()
        .unwrap()
        .slots
        .extend(slots.iter().map(|&n| Value::Int(n)));
    let expected = match vm.run() {
        VMResult::Ok(v) => Some(v),
        VMResult::Halted => vm.stack.last().cloned(),
        VMResult::Error(_) => None,
    };
    if expected.as_ref().is_some_and(|e| same(e, &native)) {
        return None;
    }
    Some(format!(
        "interp={} native={}",
        describe(&expected),
        describe(&Some(native))
    ))
}

/// Multi-op chunks: what single-op chunks cannot reach. The result kind of a
/// linear chunk is decided by `simulate_one_op` and the native value by
/// `emit_data_op`; the two must agree on every *intermediate* kind, or the
/// return coercion in `compile_linear` silently converts. Stack shuffles move
/// booleans and floats past each other, and slot ops read and write the raw
/// `i64` frame.
#[test]
fn linear_jit_matches_interpreter_on_op_sequences() {
    use Op::*;
    let cases: Vec<(&str, Vec<Op>)> = vec![
        // Kind changes mid-chunk.
        ("Inc(2.5) + 0.5", vec![LoadFloat(2.5), Inc, LoadFloat(0.5), Add]),
        ("TruncInt(-2.5) / 2", vec![LoadFloat(-2.5), TruncInt, LoadInt(2), Div]),
        ("AbsInt(-0.5) - 0.0", vec![LoadFloat(-0.5), AbsInt, LoadFloat(0.0), Sub]),
        ("(1 < 2) == (2 < 1)", vec![LoadInt(1), LoadInt(2), NumLt, LoadInt(2), LoadInt(1), NumLt, NumEq]),
        ("!(1 < 2)", vec![LoadInt(1), LoadInt(2), NumLt, LogNot]),
        ("!!0.0", vec![LoadFloat(0.0), LogNot, LogNot]),
        ("(1 <=> 2) * 2.5", vec![LoadInt(1), LoadInt(2), Spaceship, LoadFloat(2.5), Mul]),
        ("sqrt(4) % 3", vec![LoadInt(4), SqrtFloat, LoadInt(3), Mod]),
        ("mkbool(0) + 1", vec![LoadInt(0), AwkMkbool, LoadInt(1), Add]),
        ("-(0.0 * -1)", vec![LoadFloat(0.0), LoadInt(-1), Mul, Negate]),
        ("TruncInt(NaN) & 1", vec![
            LoadFloat(1e300), LoadFloat(1e300), Mul, Dup, Sub, TruncInt, LoadInt(1), BitAnd,
        ]),
        ("Inc(-inf) ^ -1", vec![
            LoadFloat(1e300), LoadFloat(1e300), Mul, Negate, Inc, LoadInt(-1), BitXor,
        ]),
        // Stack shuffles across kinds.
        ("Dup(true)", vec![LoadTrue, Dup, Pop]),
        ("Dup(1.5) + ", vec![LoadFloat(1.5), Dup, Add]),
        ("Swap(1, true)", vec![LoadInt(1), LoadTrue, Swap, Pop]),
        ("Swap(true, 1)", vec![LoadTrue, LoadInt(1), Swap, Pop]),
        ("Swap(1, 2.5)", vec![LoadInt(1), LoadFloat(2.5), Swap, Sub]),
        ("Swap(1, 2) -", vec![LoadInt(10), LoadInt(3), Swap, Sub]),
        ("Swap(1.0, 2.0) -", vec![LoadFloat(10.0), LoadFloat(3.0), Swap, Sub]),
        ("Rot(1, 2, 3)", vec![LoadInt(1), LoadInt(2), LoadInt(3), Rot, Sub, Sub]),
        ("Rot(1, 2, true)", vec![LoadInt(1), LoadInt(2), LoadTrue, Rot, Pop, Pop]),
        ("Rot(true, 1, 2)", vec![LoadTrue, LoadInt(1), LoadInt(2), Rot, Pop, Pop]),
        ("Rot(1.0, 2.0, 3.0)", vec![LoadFloat(1.0), LoadFloat(2.0), LoadFloat(3.0), Rot, Sub, Sub]),
        ("Pop(true)", vec![LoadInt(7), LoadTrue, Pop]),
        // Slot round-trips through the raw `i64` frame.
        ("set/get int", vec![LoadInt(-5), SetSlot(0), GetSlot(0)]),
        ("set/get bool", vec![LoadTrue, SetSlot(0), GetSlot(0)]),
        ("set/get float", vec![LoadFloat(2.5), SetSlot(0), GetSlot(0)]),
        ("set (1<2)", vec![LoadInt(1), LoadInt(2), NumLt, SetSlot(0), GetSlot(0)]),
        ("get + 0.5", vec![GetSlot(1), LoadFloat(0.5), Add]),
        ("++slot", vec![LoadInt(41), SetSlot(0), PreIncSlot(0)]),
        ("--slot", vec![LoadInt(41), SetSlot(0), PreDecSlot(0)]),
        ("slot++", vec![LoadInt(41), SetSlot(0), PostIncSlot(0), GetSlot(0), Add]),
        ("slot--", vec![LoadInt(41), SetSlot(0), PostDecSlot(0), GetSlot(0), Add]),
        ("++slot void", vec![LoadInt(-1), SetSlot(2), PreIncSlotVoid(2), GetSlot(2)]),
        ("slot += slot", vec![
            LoadInt(40), SetSlot(0), LoadInt(2), SetSlot(1), AddAssignSlotVoid(0, 1), GetSlot(0),
        ]),
        ("slot / 4", vec![LoadInt(6), SetSlot(3), GetSlot(3), LoadInt(4), Div]),
        ("slot as divisor", vec![LoadInt(6), SetSlot(3), LoadInt(1), GetSlot(3), Div]),
        ("slot % 3", vec![LoadInt(-7), SetSlot(0), GetSlot(0), LoadInt(3), Mod]),
        ("gcd(slot, 12)", vec![LoadInt(-18), SetSlot(0), GetSlot(0), LoadInt(12), GcdInt]),
        ("TruncInt(2.5) -> slot", vec![LoadFloat(2.5), TruncInt, SetSlot(0), GetSlot(0)]),
    ];
    let jit = JitCompiler::new();
    let mut diffs = Vec::new();
    for (label, ops) in cases {
        let mut bd = ChunkBuilder::new();
        for o in ops {
            bd.emit(o, 1);
        }
        let chunk = bd.build();
        if let Some(d) = linear_diff(&jit, &chunk, &[0; 4]) {
            diffs.push(format!("{label}: {d}"));
        }
    }
    assert_no_diffs("op sequences", diffs);
}

/// `try_run_linear` takes `slots: &[i64]` and native code addresses slots by
/// raw offset. A chunk reaching past the slice used to read out of bounds
/// (through a null pointer for `&[]`), and a chunk that wrote a slot stored
/// through the shared borrow — the `&[0; 4]` above lives in read-only memory,
/// so that was a SIGBUS. Short slices must decline; writes must not reach the
/// caller's slice.
#[test]
fn linear_jit_slot_access_stays_inside_the_callers_slice() {
    use Op::*;
    let jit = JitCompiler::new();
    let build = |ops: Vec<Op>| {
        let mut bd = ChunkBuilder::new();
        for o in ops {
            bd.emit(o, 1);
        }
        bd.build()
    };
    let read3 = build(vec![GetSlot(3), LoadInt(1), Add]);
    assert_eq!(jit.try_run_linear(&read3, &[]), None);
    assert_eq!(jit.try_run_linear(&read3, &[5, 6, 7]), None);
    assert_eq!(jit.try_run_linear(&read3, &[5, 6, 7, 8]), Some(Value::Int(9)));

    let slots = vec![41i64, 0];
    let bump = build(vec![PreIncSlot(0), PostIncSlot(0), Add]);
    assert_eq!(jit.try_run_linear(&bump, &slots), Some(Value::Int(84)));
    assert_eq!(slots, [41, 0], "a shared slot slice must not be written");
    let add_assign = build(vec![AddAssignSlotVoid(0, 1), GetSlot(0)]);
    assert_eq!(jit.try_run_linear(&add_assign, &[1]), None);
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
                let label = format!(
                    "{:?}({}{})",
                    op,
                    a.label,
                    b.map(|b| format!(", {}", b.label)).unwrap_or_default()
                );
                diffs.extend(aot_diff(&label, &chunk_for(a, b, op)));
            }
        }
    }
    diffs
}

/// Run one chunk through the AOT compiler and the interpreter, and describe
/// the disagreement, if any. A run that halts with nothing to compare is not
/// a divergence.
#[cfg(feature = "aot")]
fn aot_diff(label: &str, chunk: &Chunk) -> Option<String> {
    let native = match fusevm::aot::run_chunk_native(chunk, |_| {}) {
        Ok(VMResult::Ok(v)) => Some(v),
        Ok(VMResult::Error(_)) | Err(_) => None,
        Ok(VMResult::Halted) => return None,
    };
    let expected = interp(chunk);
    let agree = match (&expected, &native) {
        (Some(e), Some(n)) => same(e, n),
        (None, None) => true,
        _ => false,
    };
    (!agree).then(|| {
        format!(
            "{label}: interp={} aot={}",
            describe(&expected),
            describe(&native)
        )
    })
}

/// Cross every `ops` with every `arity`-tuple of `operands`, the tuple pushed
/// in order and then the ops `tail` builds from `(start ip, op)`. Reaches the
/// specialized lowerings a lone op cannot: the AOT plan gives a chunk whose
/// result is a `Bool` to the threaded path, so a comparison is only lowered
/// natively when something consumes it.
#[cfg(feature = "aot")]
fn diff_aot_seq(
    ops: &[Op],
    operands: &[Operand],
    arity: u32,
    tail: impl Fn(usize, &Op) -> Vec<Op>,
) -> Vec<String> {
    let n = operands.len();
    let mut diffs = Vec::new();
    for op in ops {
        for mut i in 0..n.pow(arity) {
            let mut tuple = Vec::new();
            for _ in 0..arity {
                tuple.push(&operands[i % n]);
                i /= n;
            }
            let labels: Vec<&str> = tuple.iter().map(|o| o.label).collect();
            let label = format!("{:?}[{}]", op, labels.join(", "));
            diffs.extend(aot_diff(&label, &chunk_of(&tuple, |base| tail(base, op))));
        }
    }
    diffs
}

/// `$?` operands. `Value::Status` is the AOT lattice's fourth scalar kind: an
/// `i64` register that float-promotes in arithmetic and is truthy when *zero*.
fn status_operands() -> Vec<Operand> {
    let status = |label, code| Operand {
        label,
        push: vec![Op::LoadInt(code), Op::SetStatus, Op::GetStatus],
    };
    vec![
        Operand {
            label: "$?",
            push: vec![Op::GetStatus],
        },
        status("$?=3", 3),
        status("$?=-2", -2),
    ]
}

/// A cross-kind cut of the operand matrix, for crossing three or four operands
/// or two ops, where the full matrix is too large to compile every tuple of.
fn edge_operands() -> Vec<Operand> {
    let mut v = vec![
        int("0", 0),
        int("1", 1),
        int("-1", -1),
        int("3", 3),
        int("64", 64),
        int("i64::MIN", i64::MIN),
        int("i64::MAX", i64::MAX),
        int("2^53+1", (1i64 << 53) + 1),
        float("0.0", 0.0),
        float("-0.0", -0.0),
        float("0.5", 0.5),
        float("-2.5", -2.5),
        float("1e30", 1e30),
    ];
    v.extend(bool_operands().into_iter().take(2));
    v.push(status_operands().swap_remove(1));
    v
}

/// The smallest cut that still holds every kind and both i64 extremes.
fn kind_operands() -> Vec<Operand> {
    vec![
        int("-1", -1),
        int("7", 7),
        int("i64::MIN", i64::MIN),
        int("i64::MAX", i64::MAX),
        float("-0.0", -0.0),
        float("0.5", 0.5),
        Operand {
            label: "true",
            push: vec![Op::LoadTrue],
        },
        status_operands().swap_remove(1),
    ]
}

/// `cond`'s truthiness, as an `Int` the native plan can return: `1` when
/// `jump` (one of the `JumpIf*` family) does not take its branch, `0` when it
/// does. The `*Keep` variants leave the condition on the stack on both arms,
/// so each arm pops it first.
fn branch_on(base: usize, cond: Vec<Op>, jump: fn(usize) -> Op) -> Vec<Op> {
    let at = base + cond.len();
    let keep = matches!(jump(0), Op::JumpIfTrueKeep(_) | Op::JumpIfFalseKeep(_));
    let mut v = cond;
    if keep {
        // at: J(at+4)  at+1: Pop  at+2: LoadInt 1  at+3: Jump(at+6)
        // at+4: Pop    at+5: LoadInt 0             at+6: end
        v.extend([
            jump(at + 4),
            Op::Pop,
            Op::LoadInt(1),
            Op::Jump(at + 6),
            Op::Pop,
            Op::LoadInt(0),
        ]);
    } else {
        // at: J(at+3)  at+1: LoadInt 1  at+2: Jump(at+4)  at+3: LoadInt 0
        v.extend([jump(at + 3), Op::LoadInt(1), Op::Jump(at + 4), Op::LoadInt(0)]);
    }
    v
}

#[cfg(feature = "aot")]
#[test]
fn aot_matches_interpreter_on_float_intrinsics() {
    let mut operands = all_operands();
    operands.extend(status_operands());
    assert_no_diffs(
        "aot float intrinsics",
        diff_aot(
            &[
                Op::AbsFloat,
                Op::SqrtFloat,
                Op::CeilFloat,
                Op::FloorFloat,
                Op::TruncFloat,
                Op::RoundFloat,
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
                Op::AwkSqrtJit,
                Op::AwkLogJit,
                Op::AwkComplJit,
                Op::AwkInt,
                Op::AwkSqrt,
                Op::AwkSin,
                Op::AwkCos,
                Op::AwkExp,
                Op::AwkLog,
            ],
            &operands,
            false,
        ),
    );
}

#[cfg(feature = "aot")]
#[test]
fn aot_matches_interpreter_on_binary_float_math() {
    assert_no_diffs(
        "aot binary float math",
        diff_aot(
            &[
                Op::PowFloat,
                Op::Atan2Float,
                Op::AwkDiv,
                Op::AwkMod,
                Op::AwkDivJit,
                Op::AwkModJit,
                Op::AwkLshiftJit,
                Op::AwkRshiftJit,
                Op::AwkAtan2,
            ],
            &edge_operands(),
            true,
        ),
    );
}

/// The integer intrinsics and unary ops on every operand kind, `$?` included —
/// the existing matrices feed them integers (`GcdInt`/`LcmInt`) or omit `$?`.
#[cfg(feature = "aot")]
#[test]
fn aot_matches_interpreter_on_int_intrinsics_of_every_kind() {
    let mut diffs = diff_aot(&[Op::GcdInt, Op::LcmInt], &edge_operands(), true);
    diffs.extend(diff_aot(
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
        &status_operands(),
        false,
    ));
    assert_no_diffs("aot int intrinsics of every kind", diffs);
}

/// The fused modular super-ops. The interpreter takes the fused i128 path only
/// when every operand is a `Value::Int`, and otherwise replays the unfused
/// `Mul`/`Add`/`Mod` — so a `Bool` or `Float` operand changes the result kind.
#[cfg(feature = "aot")]
#[test]
fn aot_matches_interpreter_on_fused_mod_ops() {
    let operands = kind_operands();
    let mut diffs = diff_aot_seq(&[Op::MulModFloor], &operands, 3, |_, op| vec![op.clone()]);
    diffs.extend(diff_aot_seq(&[Op::MulAddModFloor], &operands, 4, |_, op| {
        vec![op.clone()]
    }));
    assert_no_diffs("aot fused mod ops", diffs);
}

/// Comparisons and logical ops lowered natively: a `Bool` result would send
/// the chunk to the threaded path, so each is consumed by a branch.
#[cfg(feature = "aot")]
#[test]
fn aot_matches_interpreter_on_branches_on_comparisons() {
    let jumps: [fn(usize) -> Op; 4] = [
        Op::JumpIfTrue,
        Op::JumpIfFalse,
        Op::JumpIfTrueKeep,
        Op::JumpIfFalseKeep,
    ];
    let mut diffs = diff_aot_seq(
        &[
            Op::NumEq,
            Op::NumNe,
            Op::NumLt,
            Op::NumGt,
            Op::NumLe,
            Op::NumGe,
            Op::LogAnd,
            Op::LogOr,
        ],
        &edge_operands(),
        2,
        |base, op| branch_on(base, vec![op.clone()], Op::JumpIfFalse),
    );
    // Bare truthiness of every kind, through each branch op, and through the
    // `LogNot` / `RubyTruthy` that produce a `Bool` from it.
    let mut operands = all_operands();
    operands.extend(status_operands());
    for jump in jumps {
        diffs.extend(diff_aot_seq(&[Op::Nop, Op::LogNot, Op::RubyTruthy], &operands, 1, |base, op| {
            branch_on(base, vec![op.clone()], jump)
        }));
    }
    assert_no_diffs("aot branches on comparisons", diffs);
}

/// Two ops in sequence, `(a op1 b) op2 c`: the kind one op produces is the
/// operand kind the next op is lowered for, which no single-op chunk reaches.
#[cfg(feature = "aot")]
#[test]
fn aot_matches_interpreter_on_kind_propagation() {
    let firsts = [Op::Sub, Op::Div, Op::Mod, Op::Spaceship];
    let mut diffs = Vec::new();
    for first in &firsts {
        diffs.extend(diff_aot_seq(
            &[Op::Mul, Op::Mod, Op::Shl, Op::TruncInt],
            &kind_operands(),
            3,
            |_, second| match second {
                // Unary: applied to `b op1 c`, with `a` dropped from beneath it.
                Op::TruncInt => vec![first.clone(), Op::Swap, Op::Pop, second.clone()],
                _ => vec![Op::Rot, Op::Rot, first.clone(), Op::Swap, second.clone()],
            },
        ));
    }
    assert_no_diffs("aot kind propagation", diffs);
}

/// Values round-tripped through slots and globals before the op: the slot and
/// global registers are typed by the kind stored to them.
#[cfg(feature = "aot")]
#[test]
fn aot_matches_interpreter_on_slot_and_global_operands() {
    let ops = [Op::Add, Op::Sub, Op::Mul, Op::Div, Op::Mod, Op::NumLt, Op::BitXor];
    let mut diffs = diff_aot_seq(&ops, &edge_operands(), 2, |base, op| {
        let cond = vec![
            Op::SetSlot(1),
            Op::SetSlot(0),
            Op::GetSlot(0),
            Op::GetSlot(1),
            op.clone(),
        ];
        match op {
            Op::NumLt => branch_on(base, cond, Op::JumpIfFalse),
            _ => cond,
        }
    });
    diffs.extend(diff_aot_seq(&ops, &edge_operands(), 2, |base, op| {
        let cond = vec![
            Op::SetVar(1),
            Op::DeclareVar(0),
            Op::GetVar(0),
            Op::GetVar(1),
            op.clone(),
        ];
        match op {
            Op::NumLt => branch_on(base, cond, Op::JumpIfFalse),
            _ => cond,
        }
    }));
    assert_no_diffs("aot slot and global operands", diffs);
}

/// The slot read-modify-write super-ops, from a slot seeded with each operand
/// kind. Seeds stay inside `i64` with room to spare: the interpreter's slot
/// super-ops (`src/vm.rs`, `Op::PreIncSlot` …) use plain `+`/`-`, which is an
/// overflow *panic* in a debug build and a wrap in a release one, so an
/// extreme seed tests the build profile rather than the tiers. The same bound
/// keeps `AccumSumLoop`'s iteration count small.
#[cfg(feature = "aot")]
#[test]
fn aot_matches_interpreter_on_slot_super_ops() {
    let mut operands: Vec<Operand> = mixed_operands()
        .into_iter()
        .filter(|o| match o.push[..] {
            [Op::LoadInt(n)] => n.unsigned_abs() < 1 << 60,
            [Op::LoadFloat(f)] => f.abs() < 1e6,
            _ => true,
        })
        .collect();
    operands.extend(bool_operands());
    operands.extend(status_operands());
    let mut diffs = diff_aot_seq(
        &[
            Op::PreIncSlot(0),
            Op::PreDecSlot(0),
            Op::PostIncSlot(0),
            Op::PostDecSlot(0),
        ],
        &operands,
        1,
        |_, op| vec![Op::SetSlot(0), op.clone(), Op::GetSlot(0), Op::Add],
    );
    diffs.extend(diff_aot_seq(&[Op::PreIncSlotVoid(0)], &operands, 1, |_, op| {
        vec![Op::SetSlot(0), op.clone(), Op::GetSlot(0)]
    }));
    diffs.extend(diff_aot_seq(&[Op::AddAssignSlotVoid(0, 1)], &operands, 2, |_, op| {
        vec![Op::SetSlot(1), Op::SetSlot(0), op.clone(), Op::GetSlot(0)]
    }));
    // `while i < 5 { sum += i; i += 1 }` from the seeded `sum` and `i`.
    diffs.extend(diff_aot_seq(&[Op::AccumSumLoop(0, 1, 5)], &operands, 2, |_, op| {
        vec![Op::SetSlot(1), Op::SetSlot(0), op.clone(), Op::GetSlot(0), Op::GetSlot(1), Op::Add]
    }));
    assert_no_diffs("aot slot super-ops", diffs);
}

/// A slot accumulator updated in a native loop: `acc = a; repeat 4 { acc = acc
/// op b }`, closed both by the fused `SlotIncLtIntJumpBack` and by a plain
/// compare-and-`JumpIfTrue` back-edge.
#[cfg(feature = "aot")]
#[test]
fn aot_matches_interpreter_on_loops() {
    let ops = [Op::Add, Op::Sub, Op::Mul, Op::Div, Op::Mod, Op::Pow];
    let mut diffs = diff_aot_seq(&ops, &edge_operands(), 2, |base, op| {
        // base: SetSlot(2) SetSlot(0) LoadInt 0 SetSlot(1)
        // L=base+4: GetSlot(0) GetSlot(2) op SetSlot(0) SlotIncLtIntJumpBack(1,4,L)
        let l = base + 4;
        vec![
            Op::SetSlot(2),
            Op::SetSlot(0),
            Op::LoadInt(0),
            Op::SetSlot(1),
            Op::GetSlot(0),
            Op::GetSlot(2),
            op.clone(),
            Op::SetSlot(0),
            Op::SlotIncLtIntJumpBack(1, 4, l),
            Op::GetSlot(0),
        ]
    });
    diffs.extend(diff_aot_seq(&ops, &edge_operands(), 2, |base, op| {
        let l = base + 4;
        vec![
            Op::SetSlot(2),
            Op::SetSlot(0),
            Op::LoadInt(0),
            Op::SetSlot(1),
            Op::GetSlot(0),
            Op::GetSlot(2),
            op.clone(),
            Op::SetSlot(0),
            Op::GetSlot(1),
            Op::Inc,
            Op::Dup,
            Op::SetSlot(1),
            Op::LoadInt(4),
            Op::NumLt,
            Op::JumpIfTrue(l),
            Op::GetSlot(0),
        ]
    }));
    assert_no_diffs("aot loops", diffs);
}

/// Stack shuffles feeding a non-commutative op, so a lowering that permutes
/// the wrong registers (or the wrong register *kinds*) answers differently.
#[cfg(feature = "aot")]
#[test]
fn aot_matches_interpreter_on_stack_shuffles() {
    let mut diffs = diff_aot_seq(&[Op::Sub, Op::Div, Op::Concat], &edge_operands(), 2, |_, op| {
        vec![Op::Swap, op.clone()]
    });
    diffs.extend(diff_aot_seq(&[Op::Sub, Op::Pow], &edge_operands(), 2, |_, op| {
        vec![Op::Dup2, op.clone(), Op::Rot, Op::Rot, op.clone(), op.clone()]
    }));
    diffs.extend(diff_aot_seq(&[Op::Sub, Op::Mul], &kind_operands(), 3, |_, op| {
        vec![Op::Rot, op.clone(), op.clone()]
    }));
    diffs.extend(diff_aot_seq(&[Op::Mul, Op::Spaceship], &edge_operands(), 1, |_, op| {
        vec![Op::Dup, op.clone()]
    }));
    assert_no_diffs("aot stack shuffles", diffs);
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
// The comparison is strict on the variant, `Bool` included: `BlockNum::Bool`
// carries a boolean chunk result out, so an interpreter `Bool` must arrive as
// one (see `block_agrees`), and `block_jit_preserves_boolean_result_kind`
// below pins the same contract end to end through `VM::run`.

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
        Some(fusevm::BlockNum::Bool(b)) => format!("Bool({b})"),
    }
}

/// Declining is always correct — the caller falls back to the interpreter.
/// Answering differently is not.
///
/// There is still deliberately no `Bool`-answers-`Int` arm, and that is the
/// load-bearing part of this function: accepting `Bool(true)` against
/// `BlockNum::Int(1)` is exactly what let the old flattening go unnoticed. A
/// boolean result must arrive as `BlockNum::Bool` or not at all.
///
/// What changed is that arriving is now allowed. The tier no longer has to
/// decline a chunk whose result is a `Value::Bool`: `BlockNum::Bool` carries
/// the kind out, so a comparison at the end of a chunk compiles and answers
/// with the kind the interpreter would. Every other escape route for a boolean
/// is still refused — see `bool_is_consumed_in_place` and
/// `bool_is_chunk_result` in `src/jit.rs`.
fn block_agrees(expected: &Option<Value>, got: &Option<fusevm::BlockNum>) -> bool {
    match (expected, got) {
        (_, None) => true,
        (Some(Value::Int(a)), Some(fusevm::BlockNum::Int(b))) => a == b,
        (Some(Value::Bool(a)), Some(fusevm::BlockNum::Bool(b))) => a == b,
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

// ── Block tier: control flow, slots, and warmup ──
//
// Everything above feeds the block tier one straight-line op with every slot
// declared `SlotKind::Int`. A frontend's chunks are not shaped like that: they
// branch, loop, keep `Float` (and `Bool`, and string) values in slots, and
// reach the tier through `VM::run`, which snapshots the frame's slots into a
// raw `i64` buffer and writes them back afterwards. The cases below drive
// that whole path and require the answer — and the frame it leaves behind —
// to be the same on the cold (interpreter) run and every warm (native) run.

/// The value an operand pushes, for seeding a slot with it.
fn operand_value(o: &Operand) -> Value {
    interp(&chunk_for(o, None, &Op::Nop)).expect("operand pushes a value")
}

/// `VM::run` of `chunk` on a fresh VM whose base frame holds `slots`, and then
/// once more on the same VM, described down to the result variants and the
/// frames left behind.
fn run_described(chunk: &Chunk, slots: &[Value]) -> String {
    let mut vm = VM::new(chunk.clone());
    vm.frames.last_mut().unwrap().slots = slots.to_vec();
    vm.enable_tracing_jit();
    let mut run = || match vm.run() {
        VMResult::Ok(v) => describe(&Some(v)),
        VMResult::Halted => "Halted".to_string(),
        VMResult::Error(e) => format!("Error({e})"),
    };
    let (first, rerun) = (run(), run());
    let frames: Vec<Vec<String>> = vm
        .frames
        .iter()
        .map(|f| f.slots.iter().map(|v| describe(&Some(v.clone()))).collect())
        .collect();
    format!("{first} rerun={rerun} frames={frames:?}")
}

/// Run `chunk` cold and then warm on one thread. Returns a divergence line if
/// any warm run differs from the cold one, and whether the block tier compiled
/// the chunk at all (so a corpus that never reaches the tier cannot pass).
fn block_warmup_diff(label: &str, chunk: &Chunk, slots: &[Value]) -> (Option<String>, bool) {
    let answers: Vec<String> = (0..3).map(|_| run_described(chunk, slots)).collect();
    let compiled = JitCompiler::new().block_jit_is_compiled(chunk);
    let diff = (!answers.windows(2).all(|w| w[0] == w[1]))
        .then(|| format!("{label}: cold={} warm={}", answers[0], answers[2]));
    (diff, compiled)
}

/// `block_warmup_diff` over a corpus; see `assert_trace_clean` for why an
/// all-declined corpus is a failure rather than a pass.
fn assert_block_warmup_clean(what: &str, cases: Vec<(String, Chunk, Vec<Value>)>) {
    let mut diffs = Vec::new();
    let mut compiled = 0usize;
    for (label, chunk, slots) in &cases {
        let (diff, hit) = block_warmup_diff(label, chunk, slots);
        diffs.extend(diff);
        compiled += hit as usize;
    }
    assert!(
        compiled > 0,
        "{what}: no case compiled in the block tier, so agreement proves nothing"
    );
    assert_no_diffs(what, diffs);
}

/// Prefix that gives every case its own `op_hash`, so each one gets its own
/// cold run instead of sharing a warm block-cache entry with an earlier case.
fn salted(salt: i64, ops: &[Op]) -> Chunk {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(salt), 1);
    b.emit(Op::Pop, 1);
    for o in ops {
        b.emit(o.clone(), 1);
    }
    b.build()
}

/// Seeds whose fused slot increment stays inside `i64`. The interpreter's
/// fused slot ops add with a bare `+` (see `Op::PreIncSlot` in `vm.rs`), which
/// panics on overflow in a debug build, so the `i64::MAX`-adjacent edges are
/// not a question the interpreter can answer there.
fn fused_slot_seeds() -> Vec<Operand> {
    all_operands()
        .into_iter()
        .filter(|o| operand_value(o).to_int().unsigned_abs() < 1 << 62)
        .collect()
}

#[test]
fn block_jit_slot_loops_are_stable_across_warmup() {
    // s = seed; i = 0; do { s = s OP k; i += 1 } while (i < 3); s
    let mut cases = Vec::new();
    let mut salt = 8_000_000;
    for op in [Op::Add, Op::Sub, Op::Mul] {
        for seed in mixed_operands() {
            for k in mixed_operands() {
                salt += 1;
                let mut ops = vec![Op::LoadInt(0), Op::SetSlot(1)];
                let body = ops.len() + 2; // after the salt prefix
                ops.push(Op::GetSlot(0));
                ops.extend(k.push.iter().cloned());
                ops.extend([
                    op.clone(),
                    Op::SetSlot(0),
                    Op::PreIncSlotVoid(1),
                    Op::GetSlot(1),
                    Op::LoadInt(3),
                    Op::NumLt,
                    Op::JumpIfTrue(body),
                    Op::GetSlot(0),
                ]);
                cases.push((
                    format!("s={} s{op:?}={}", seed.label, k.label),
                    salted(salt, &ops),
                    vec![operand_value(&seed), Value::Int(0)],
                ));
            }
        }
    }
    assert_block_warmup_clean("block slot loop", cases);
}

#[test]
fn block_jit_fused_slot_ops_are_stable_across_warmup() {
    let mut cases = Vec::new();
    let mut salt = 8_100_000;
    let shapes: Vec<(&str, Vec<Op>)> = vec![
        ("PreIncSlot", vec![Op::PreIncSlot(0)]),
        ("PreDecSlot", vec![Op::PreDecSlot(0)]),
        ("PostIncSlot", vec![Op::PostIncSlot(0)]),
        ("PostDecSlot", vec![Op::PostDecSlot(0)]),
        ("PreIncSlotVoid", vec![Op::PreIncSlotVoid(0), Op::GetSlot(0)]),
        (
            "AddAssignSlotVoid",
            vec![Op::AddAssignSlotVoid(0, 1), Op::GetSlot(0)],
        ),
        (
            "AddAssignSlotVoid(rhs)",
            vec![Op::AddAssignSlotVoid(1, 0), Op::GetSlot(1)],
        ),
        ("AccumSumLoop", vec![Op::AccumSumLoop(1, 0, 4), Op::GetSlot(1)]),
        // do { } while (++s < 4): body is the fused op itself (ip 2).
        (
            "SlotIncLtIntJumpBack",
            vec![Op::SlotIncLtIntJumpBack(0, 4, 2), Op::GetSlot(0)],
        ),
        // if (s < 1) 10 else 20
        (
            "SlotLtIntJumpIfFalse",
            vec![
                Op::SlotLtIntJumpIfFalse(0, 1, 5),
                Op::LoadInt(10),
                Op::Jump(6),
                Op::LoadInt(20),
                Op::Nop,
            ],
        ),
    ];
    for (name, ops) in &shapes {
        // The two loops count the seed slot up to 4, one step per iteration, so
        // a far-negative seed is a question about loop length, not about kinds.
        let counts_up = matches!(ops[0], Op::AccumSumLoop(..) | Op::SlotIncLtIntJumpBack(..));
        for seed in fused_slot_seeds() {
            if counts_up && operand_value(&seed).to_int() < -8 {
                continue;
            }
            salt += 1;
            cases.push((
                format!("{name}(s={})", seed.label),
                salted(salt, ops),
                vec![operand_value(&seed), Value::Int(2)],
            ));
        }
    }
    assert_block_warmup_clean("block fused slot op", cases);
}

#[test]
fn block_jit_branch_conditions_are_stable_across_warmup() {
    // cond ? 10 : 20, with cond every operand kind, plus NaN (truthy).
    let mut conds = all_operands();
    conds.push(float("NaN", f64::NAN));
    let mut cases = Vec::new();
    let mut salt = 8_200_000;
    for jump_if_true in [false, true] {
        for c in &conds {
            salt += 1;
            let mut ops: Vec<Op> = c.push.clone();
            let base = 2 + ops.len(); // salt prefix + condition
            ops.extend([
                if jump_if_true {
                    Op::JumpIfTrue(base + 3)
                } else {
                    Op::JumpIfFalse(base + 3)
                },
                Op::LoadInt(10),
                Op::Jump(base + 4),
                Op::LoadInt(20),
                Op::Nop,
            ]);
            cases.push((
                format!("{}({})", if jump_if_true { "jit" } else { "jif" }, c.label),
                salted(salt, &ops),
                vec![],
            ));
        }
    }
    assert_block_warmup_clean("block branch condition", cases);
}

/// Slot values whose kind the block tier's `i64` buffer cannot hold, and
/// values it can hold under a kind the chunk then changes.
#[test]
fn block_jit_slot_kinds_are_stable_across_warmup() {
    let str_slot = || Value::str("x");
    let cases: Vec<(&str, Vec<Op>, Vec<Value>)> = vec![
        // An Int stored into a slot that entered the chunk holding a Float.
        (
            "int into float slot",
            vec![Op::LoadInt(3), Op::SetSlot(0), Op::GetSlot(0)],
            vec![Value::Float(1.5)],
        ),
        (
            "int into float slot, void",
            vec![Op::LoadInt(3), Op::SetSlot(0), Op::LoadInt(1)],
            vec![Value::Float(1.5)],
        ),
        // A Bool slot read as a value, and as an arithmetic operand.
        ("bool slot read", vec![Op::GetSlot(0)], vec![Value::Bool(true)]),
        (
            "bool slot + 1",
            vec![Op::GetSlot(0), Op::LoadInt(1), Op::Add],
            vec![Value::Bool(true)],
        ),
        // A slot the chunk never touches must come back unchanged.
        (
            "untouched string slot",
            vec![Op::GetSlot(0), Op::LoadInt(1), Op::Add],
            vec![Value::Int(1), str_slot()],
        ),
        (
            "untouched bool slot",
            vec![Op::GetSlot(0), Op::LoadInt(1), Op::Add],
            vec![Value::Int(1), Value::Bool(false)],
        ),
        // A slot past the end of the frame reads as Undef and grows the frame.
        (
            "slot past frame end",
            vec![Op::LoadInt(5), Op::SetSlot(3), Op::GetSlot(3)],
            vec![],
        ),
        ("read missing slot", vec![Op::GetSlot(2)], vec![Value::Int(1)]),
        // No value left on the stack: the interpreter answers `Halted`.
        (
            "void chunk",
            vec![Op::LoadInt(4), Op::SetSlot(0)],
            vec![Value::Int(1)],
        ),
        // A scope frame: slots address the new, empty frame, and `PopFrame`
        // discards what the scope pushed.
        (
            "slot read inside scope frame",
            vec![Op::PushFrame, Op::GetSlot(0)],
            vec![Value::Int(9)],
        ),
        (
            "slot write inside scope frame",
            vec![
                Op::PushFrame,
                Op::LoadInt(5),
                Op::SetSlot(0),
                Op::GetSlot(0),
                Op::PopFrame,
                Op::GetSlot(0),
            ],
            vec![Value::Int(9)],
        ),
        (
            "PopFrame drops scope stack",
            vec![Op::LoadInt(7), Op::PushFrame, Op::LoadInt(1), Op::PopFrame],
            vec![],
        ),
        (
            "PopFrame without PushFrame",
            vec![Op::LoadInt(7), Op::PopFrame, Op::LoadInt(1)],
            vec![Value::Int(3)],
        ),
        // Control: a plain all-Int chunk that must keep compiling.
        (
            "int slot + 1",
            vec![Op::GetSlot(0), Op::LoadInt(1), Op::Add, Op::SetSlot(0), Op::GetSlot(0)],
            vec![Value::Int(1)],
        ),
        (
            "float slot + 1",
            vec![Op::GetSlot(0), Op::LoadInt(1), Op::Add, Op::SetSlot(0), Op::GetSlot(0)],
            vec![Value::Float(-0.5)],
        ),
    ];
    let mut salt = 8_300_000;
    let cases = cases
        .into_iter()
        .map(|(label, ops, slots)| {
            salt += 1;
            (label.to_string(), salted(salt, &ops), slots)
        })
        .collect();
    assert_block_warmup_clean("block slot kind", cases);
}

#[test]
fn block_jit_matches_interpreter_on_stack_ops() {
    assert_no_diffs(
        "block stack unary",
        diff_block(&[Op::Dup, Op::Nop], &all_operands(), false),
    );
    assert_no_diffs(
        "block stack binary",
        diff_block(&[Op::Swap, Op::Pop], &mixed_operands(), true),
    );
}

/// The direct entry points take the slot buffer from the caller, and the
/// compiled code addresses every slot the chunk names through it unchecked.
/// A buffer too short for the chunk must be declined, not overrun — an empty
/// one is passed as a null pointer.
#[test]
fn block_jit_declines_a_slot_buffer_too_short_for_the_chunk() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(5), 1);
    b.emit(Op::SetSlot(3), 1);
    b.emit(Op::GetSlot(3), 1);
    let chunk = b.build();
    let jit = JitCompiler::new();
    for len in 0..4 {
        let mut slots = vec![0i64; len];
        let kinds = vec![fusevm::SlotKind::Int; len];
        assert_eq!(
            jit.try_run_block_eager_typed_kinded(&chunk, &mut slots, &kinds),
            None,
            "slot 3 does not exist in a {len}-slot buffer"
        );
    }
    let mut slots = vec![0i64; 4];
    assert_eq!(
        jit.try_run_block_eager_typed_kinded(&chunk, &mut slots, &[fusevm::SlotKind::Int; 4]),
        Some(fusevm::BlockNum::Int(5)),
        "a buffer that holds slot 3 still compiles"
    );
}

/// `Op::PopFrame` truncates the operand stack to the frame's base; a no-op
/// lowering leaves the scope's values standing.
#[test]
fn block_jit_matches_interpreter_on_pop_frame() {
    for ops in [
        vec![Op::LoadInt(7), Op::PushFrame, Op::LoadInt(1), Op::PopFrame],
        vec![Op::LoadInt(7), Op::PushFrame, Op::LoadFloat(0.5), Op::PopFrame],
        vec![Op::LoadInt(7), Op::LoadInt(8), Op::PopFrame, Op::LoadInt(1), Op::Add],
    ] {
        let mut b = ChunkBuilder::new();
        for o in &ops {
            b.emit(o.clone(), 1);
        }
        let chunk = b.build();
        let expected = interp(&chunk);
        let got = block(&chunk);
        assert!(
            block_agrees(&expected, &got),
            "{ops:?}: interp={} block={}",
            describe(&expected),
            describe_block(&got)
        );
    }
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

/// Where the value under test lives while the loop runs.
///
/// The tracing tier carries every slot in one `i64` register whose kind is
/// fixed at the anchor (`slot_kinds_at_anchor`), so the shapes differ in
/// exactly the ways that stress that model.
#[derive(Clone, Copy, Debug, PartialEq)]
enum TraceShape {
    /// `slot1 = a OP b` every iteration — a fresh value, stable kind.
    Fresh,
    /// `slot2 = a OP b; slot1 = -slot2; slot2 = 0` — a scratch slot that is
    /// `Int` at the anchor but holds the op's result *inside* the iteration.
    /// Compilers reuse temporaries this way; the trace must not compute with
    /// the mid-iteration value under the anchor's kind (a plain copy would
    /// round-trip the bits and hide it, hence the `Negate`).
    Scratch,
    /// `slot1 = a` before the loop, then `slot1 = slot1 OP b` every
    /// iteration — a loop-carried value, so 200 applications walk it to
    /// overflow, infinity, NaN, or a fixed point.
    Carried,
}

/// Wrap `a OP b` in a hot do-while loop so the op is recorded into a trace,
/// and hand the computed value back through a slot.
///
/// ```text
///   LoadInt(salt); Pop            // unique op_hash per case
///   [Carried: <push a> SetSlot(1)]
///   LoadInt(0); SetSlot(0)
/// anchor:
///   PreIncSlotVoid(0)             // loop counter
///   <the shape's body around OP>
///   GetSlot(0) LoadInt(200) NumLt JumpIfTrue(anchor)
///   GetSlot(1)                    // the value under test
/// ```
fn trace_chunk_for(a: &Operand, b: Option<&Operand>, op: &Op, salt: i64) -> (Chunk, usize) {
    trace_chunk_shaped(a, b, op, salt, TraceShape::Fresh)
}

fn trace_chunk_shaped(
    a: &Operand,
    b: Option<&Operand>,
    op: &Op,
    salt: i64,
    shape: TraceShape,
) -> (Chunk, usize) {
    let mut bd = ChunkBuilder::new();
    bd.emit(Op::LoadInt(salt), 1);
    bd.emit(Op::Pop, 1);
    if shape == TraceShape::Carried {
        for o in &a.push {
            bd.emit(o.clone(), 1);
        }
        bd.emit(Op::SetSlot(1), 1);
    }
    bd.emit(Op::LoadInt(0), 1);
    bd.emit(Op::SetSlot(0), 1);
    let anchor = bd.current_pos();
    bd.emit(Op::PreIncSlotVoid(0), 1);
    if shape == TraceShape::Carried {
        bd.emit(Op::GetSlot(1), 1);
    } else {
        for o in &a.push {
            bd.emit(o.clone(), 1);
        }
    }
    if let Some(b) = b {
        for o in &b.push {
            bd.emit(o.clone(), 1);
        }
    }
    bd.emit(op.clone(), 1);
    if shape == TraceShape::Scratch {
        bd.emit(Op::SetSlot(2), 1);
        bd.emit(Op::GetSlot(2), 1);
        bd.emit(Op::Negate, 1);
        bd.emit(Op::SetSlot(1), 1);
        bd.emit(Op::LoadInt(0), 1);
        bd.emit(Op::SetSlot(2), 1);
    } else {
        bd.emit(Op::SetSlot(1), 1);
    }
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
    run_seeded(chunk, tracing, &[])
}

/// [`run_with_slots`] with the frame's leading slots seeded from `seed`; the
/// rest of the first four are `Int(0)`.
fn run_seeded(chunk: &Chunk, tracing: bool, seed: &[Value]) -> Option<Value> {
    let mut vm = VM::new(chunk.clone());
    if tracing {
        vm.enable_tracing_jit();
    }
    let frame = vm.frames.last_mut().unwrap();
    frame.slots.extend(seed.iter().cloned());
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
    diff_trace_shaped(ops, operands, binary, salt0, TraceShape::Fresh)
}

/// [`diff_trace`] over any [`TraceShape`].
fn diff_trace_shaped(
    ops: &[Op],
    operands: &[Operand],
    binary: bool,
    salt0: i64,
    shape: TraceShape,
) -> (Vec<String>, usize) {
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
                let (chunk, anchor) = trace_chunk_shaped(a, b, op, salt, shape);
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
                        "{:?}{}({}{}): interp={} trace={}",
                        op,
                        if shape == TraceShape::Fresh {
                            String::new()
                        } else {
                            format!("[{shape:?}]")
                        },
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

/// The float intrinsics and transcendentals the trace tier lowers through
/// `emit_data_op`'s `MathIds` libcalls, plus the awk ops the block tier admits.
/// The `Awk*Jit` trap/warn ops have no trace lowering, so their cases are
/// refused at compile time; they stay in the corpus so a lowering added later
/// is scored the moment it compiles.
#[test]
fn tracing_jit_matches_interpreter_on_float_intrinsics_and_transcendentals() {
    assert_trace_clean(
        "trace transcendentals",
        diff_trace(
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
                Op::AwkSqrtJit,
                Op::AwkLogJit,
                Op::AwkComplJit,
            ],
            &all_operands(),
            false,
            4_000_000,
        ),
    );
}

#[test]
fn tracing_jit_matches_interpreter_on_binary_float_and_awk_ops() {
    assert_trace_clean(
        "trace binary float/awk",
        diff_trace(
            &[
                Op::PowFloat,
                Op::Atan2Float,
                Op::AwkAtan2,
                Op::AwkAnd(2),
                Op::AwkOr(2),
                Op::AwkXor(2),
                Op::AwkDivJit,
                Op::AwkModJit,
                Op::AwkLshiftJit,
                Op::AwkRshiftJit,
            ],
            &all_operands(),
            true,
            5_000_000,
        ),
    );
}

/// `MulModFloor` pops `[a, b, k]` and `MulAddModFloor` `[a, b, c, k]`; the
/// leading operands are folded into one composite push so `diff_trace` crosses
/// them against every divisor `k`.
#[test]
fn tracing_jit_matches_interpreter_on_fused_mod_floor() {
    let base = mixed_operands();
    let pick = |label: &str| base.iter().find(|o| o.label == label).unwrap().clone();
    let firsts: Vec<Operand> = ["0", "-1", "7", "i64::MAX", "i64::MIN", "2^53+1", "0.5"]
        .iter()
        .map(|l| pick(l))
        .collect();
    let compose = |parts: &[&Operand]| Operand {
        label: Box::leak(
            parts
                .iter()
                .map(|p| p.label)
                .collect::<Vec<_>>()
                .join(", ")
                .into_boxed_str(),
        ),
        push: parts.iter().flat_map(|p| p.push.clone()).collect(),
    };
    let mut pairs = Vec::new();
    let mut triples = Vec::new();
    for a in &firsts {
        for b in &firsts {
            pairs.push(compose(&[a, b]));
            triples.push(compose(&[a, b, &pick("3")]));
        }
    }
    let ks: Vec<Operand> = [
        "0", "1", "-1", "3", "-2", "7", "i64::MAX", "i64::MIN", "0.5",
    ]
    .iter()
    .map(|l| pick(l))
    .collect();
    let (mut diffs, mut compiled) = (Vec::new(), 0usize);
    for (op, leads, salt) in [
        (Op::MulModFloor, &pairs, 6_000_000),
        (Op::MulAddModFloor, &triples, 6_500_000),
    ] {
        // `diff_trace` crosses an operand list with itself; leads × divisors
        // is the product that matters, so fold each pair into one operand.
        let crossed: Vec<Operand> = leads
            .iter()
            .flat_map(|l| ks.iter().map(move |k| compose(&[l, k])))
            .collect();
        let (d, c) = diff_trace(&[op], &crossed, false, salt);
        diffs.extend(d);
        compiled += c;
    }
    assert_trace_clean("trace fused mod-floor", (diffs, compiled));
}

/// A scratch slot that is `Int` at the anchor but carries the op's result
/// within the iteration — the reuse pattern of any compiler's temporaries.
#[test]
fn tracing_jit_matches_interpreter_through_a_scratch_slot() {
    let (mut diffs, mut compiled) = diff_trace_shaped(
        &[
            Op::Add,
            Op::Sub,
            Op::Mul,
            Op::Div,
            Op::Mod,
            Op::Pow,
            Op::PowFloat,
            Op::Atan2Float,
            Op::BitAnd,
            Op::Shl,
        ],
        &mixed_operands(),
        true,
        8_000_000,
        TraceShape::Scratch,
    );
    let (d, c) = diff_trace_shaped(
        &[
            Op::Negate,
            Op::Inc,
            Op::Dec,
            Op::AbsFloat,
            Op::TruncInt,
            Op::SqrtFloat,
            Op::SinFloat,
            Op::FloorFloat,
            Op::AwkMkbool,
        ],
        &mixed_operands(),
        false,
        8_500_000,
        TraceShape::Scratch,
    );
    diffs.extend(d);
    compiled += c;
    assert_trace_clean("trace scratch slot", (diffs, compiled));
}

/// A loop-carried value: 200 applications of `x = x OP b` walk the slot to
/// the `i64` edges (wrapping `Add`/`Mul`/`Shl`, `Negate(i64::MIN)`), to
/// infinity and NaN, and through every kind transition the op can make.
#[test]
fn tracing_jit_matches_interpreter_on_loop_carried_values() {
    let (mut diffs, mut compiled) = diff_trace_shaped(
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
            Op::Spaceship,
            Op::GcdInt,
            Op::LcmInt,
            Op::Atan2Float,
        ],
        &mixed_operands(),
        true,
        9_000_000,
        TraceShape::Carried,
    );
    let (d, c) = diff_trace_shaped(
        &[
            Op::Negate,
            Op::Inc,
            Op::Dec,
            Op::BitNot,
            Op::AbsInt,
            Op::TruncInt,
            Op::AbsFloat,
            Op::CeilFloat,
            Op::FloorFloat,
            Op::RoundFloat,
            Op::SqrtFloat,
            Op::ExpFloat,
            Op::CosFloat,
            Op::AwkMkbool,
        ],
        &all_operands(),
        false,
        9_500_000,
        TraceShape::Carried,
    );
    diffs.extend(d);
    compiled += c;
    assert_trace_clean("trace loop-carried", (diffs, compiled));
}

/// A trace compiled for one operand kind, then entered with another.
///
/// The operand is read from slot 3, which the loop never writes, and every
/// seed runs the *same* chunk on one thread — so the first seed compiles the
/// trace and every later one meets it at the entry guard. The guard compares
/// slot kinds only, and `refresh_slot_buffers` files a `Bool` (and anything
/// non-numeric) under `SlotKind::Int`; the trace must not then compute with
/// it as an integer where the interpreter coerces it.
///
/// Every seed after the first is a *re-run* of the chunk, which is what warms
/// the block tier (`block_threshold` defaults to 1) — and these chunks are
/// block-eligible. The block threshold is pinned out of reach for this thread
/// so the answers scored here are the tracing tier's; the block tier's
/// handling of the same slots is a separate matrix.
#[test]
fn tracing_jit_entry_guard_matches_interpreter_across_operand_kinds() {
    let jit = JitCompiler::new();
    jit.set_config(fusevm::TraceJitConfig {
        block_threshold: u32::MAX,
        ..jit.get_config()
    });
    let seeds = [
        Value::Int(7),
        Value::Float(0.5),
        Value::Bool(true),
        Value::Int(i64::MAX),
        Value::Bool(false),
        Value::Float(-0.0),
        Value::str("12abc"),
        Value::Int(-3),
        Value::Undef,
        Value::Float(2.5),
    ];
    let slot3 = Operand {
        label: "slot3",
        push: vec![Op::GetSlot(3)],
    };
    let mut diffs = Vec::new();
    let mut compiled = 0usize;
    let mut salt = 11_000_000;
    let cases: Vec<(Op, Option<Operand>)> = vec![
        (Op::Add, Some(int("1", 1))),
        (Op::Sub, Some(float("0.5", 0.5))),
        (Op::Mul, Some(int("3", 3))),
        (Op::BitAnd, Some(int("7", 7))),
        (Op::Shl, Some(int("1", 1))),
        (Op::Spaceship, Some(int("1", 1))),
        (Op::Negate, None),
        (Op::Inc, None),
        (Op::AbsInt, None),
        (Op::BitNot, None),
        (Op::SqrtFloat, None),
        (Op::AwkMkbool, None),
        (Op::TruncInt, None),
    ];
    for (op, b) in &cases {
        // The first seed decides what the recording sees: an `Int`, a `Bool`,
        // a string, a `Float`. Each rotation is a fresh chunk.
        for start in [0, 2, 6, 9] {
            salt += 1;
            let (chunk, anchor) = trace_chunk_for(&slot3, b.as_ref(), op, salt);
            let ordered: Vec<&Value> = seeds.iter().cycle().skip(start).take(seeds.len()).collect();
            let mut ever = false;
            for seed in ordered {
                let s = [Value::Int(0), Value::Int(0), Value::Int(0), seed.clone()];
                let traced = run_seeded(&chunk, true, &s);
                ever |= jit.trace_is_compiled(&chunk, anchor);
                if !ever {
                    continue;
                }
                let expected = run_seeded(&chunk, false, &s);
                let agree = match (&expected, &traced) {
                    (Some(e), Some(t)) => same(e, t),
                    (None, None) => true,
                    _ => false,
                };
                if !agree {
                    diffs.push(format!(
                        "{op:?}(slot3={seed:?}{}): interp={} trace={}",
                        b.as_ref()
                            .map(|b| format!(", {}", b.label))
                            .unwrap_or_default(),
                        describe(&expected),
                        describe(&traced)
                    ));
                }
            }
            compiled += ever as usize;
        }
    }
    assert_trace_clean("trace entry guard", (diffs, compiled));
}
