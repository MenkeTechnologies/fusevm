//! Interpreter-vs-AOT parity for the awk builtins and the clock op that the
//! block JIT lowers natively and the AOT register-native tier now lowers too:
//! `AwkSin`, `AwkCos`, `AwkExp`, `AwkAtan2`, `AwkMkbool`, `AwkAnd`, `AwkOr`,
//! `AwkXor`, `TimeInt`.
//!
//! Each case asserts two things: that the AOT tier really lowered the chunk
//! natively (a fallback to the threaded lowering runs every op through the
//! interpreter and would agree trivially), and that the result — `Value`
//! variant, float bits, or the exact error text — equals the interpreter's.

#![cfg(feature = "aot")]

use fusevm::aot::{lowering_for, run_chunk_native, Lowering};
use fusevm::{Chunk, ChunkBuilder, DefaultAwkHost, Op, VMResult, Value, VM};

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

/// Non-finite floats are computed in-chunk: `1e300 * 1e300` is `inf`, and
/// `inf - inf` is a NaN.
fn nonfinite() -> Vec<Operand> {
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

/// Every scalar kind the AOT lattice carries in a register: Int, Float, Bool,
/// and `$?` (`Status`, truthy when zero, `to_float` is the code).
fn scalar_operands() -> Vec<Operand> {
    let mut v = vec![
        int("0", 0),
        int("1", 1),
        int("-1", -1),
        int("12", 12),
        int("10", 10),
        int("i64::MIN", i64::MIN),
        int("i64::MAX", i64::MAX),
        int("2^53+1", (1i64 << 53) + 1),
        float("0.0", 0.0),
        float("-0.0", -0.0),
        float("0.5", 0.5),
        float("-0.25", -0.25),
        float("3.75", 3.75),
        float("1e30", 1e30),
        float("2^54", 2f64.powi(54)),
        float("710.0", 710.0),
        Operand {
            label: "true",
            push: vec![Op::LoadTrue],
        },
        Operand {
            label: "false",
            push: vec![Op::LoadFalse],
        },
        Operand {
            label: "$?=0",
            push: vec![Op::LoadInt(0), Op::SetStatus, Op::GetStatus],
        },
        Operand {
            label: "$?=3",
            push: vec![Op::LoadInt(3), Op::SetStatus, Op::GetStatus],
        },
        Operand {
            label: "$?=-2",
            push: vec![Op::LoadInt(-2), Op::SetStatus, Op::GetStatus],
        },
    ];
    v.extend(nonfinite());
    v
}

/// A compact cut for the variadic folds, where three operands cross cubically.
fn fold_operands() -> Vec<Operand> {
    vec![
        int("0", 0),
        int("12", 12),
        int("-1", -1),
        int("i64::MAX", i64::MAX),
        float("-0.0", -0.0),
        float("3.75", 3.75),
        float("-0.25", -0.25),
        float("2^54", 2f64.powi(54)),
        Operand {
            label: "true",
            push: vec![Op::LoadTrue],
        },
        Operand {
            label: "$?=3",
            push: vec![Op::LoadInt(3), Op::SetStatus, Op::GetStatus],
        },
        nonfinite().swap_remove(2), // NaN
        nonfinite().swap_remove(1), // -inf
    ]
}

fn chunk_of(operands: &[&Operand], tail: &[Op]) -> Chunk {
    let mut b = ChunkBuilder::new();
    for op in operands.iter().flat_map(|o| &o.push) {
        b.emit(op.clone(), 1);
    }
    for op in tail {
        b.emit(op.clone(), 1);
    }
    b.build()
}

/// A run's observable outcome: the result value, or the error text.
#[derive(Debug)]
enum Outcome {
    Ok(Value),
    Err(String),
    Halted,
}

fn outcome(r: VMResult) -> Outcome {
    match r {
        VMResult::Ok(v) => Outcome::Ok(v),
        VMResult::Error(e) => Outcome::Err(e),
        VMResult::Halted => Outcome::Halted,
    }
}

/// Bit-exact: `Float` compares raw bits (so `-0.0` and NaN sign matter).
fn same(a: &Outcome, b: &Outcome) -> bool {
    match (a, b) {
        (Outcome::Ok(Value::Float(x)), Outcome::Ok(Value::Float(y))) => x.to_bits() == y.to_bits(),
        (Outcome::Ok(x), Outcome::Ok(y)) => x == y,
        (Outcome::Err(x), Outcome::Err(y)) => x == y,
        (Outcome::Halted, Outcome::Halted) => true,
        _ => false,
    }
}

fn interp(chunk: &Chunk, register: fn(&mut VM)) -> Outcome {
    let mut vm = VM::new(chunk.clone());
    register(&mut vm);
    outcome(vm.run())
}

fn aot(chunk: &Chunk, register: fn(&mut VM)) -> Outcome {
    match run_chunk_native(chunk, register) {
        Ok(r) => outcome(r),
        Err(e) => Outcome::Err(format!("aot build failed: {e}")),
    }
}

fn no_host(_: &mut VM) {}

fn awk_host(vm: &mut VM) {
    vm.set_awk_host(Box::new(DefaultAwkHost));
}

/// Whether the AOT tier lowers every op of `chunk` to native IR with no
/// static deopt point.
fn fully_native(chunk: &Chunk) -> bool {
    lowering_for(chunk)
        == Lowering::Native {
            covered: chunk.ops.len(),
            deopts: 0,
        }
}

/// Check one chunk: it must lower fully native, and agree with the interpreter.
fn check(label: &str, chunk: &Chunk, register: fn(&mut VM), diffs: &mut Vec<String>) {
    if !fully_native(chunk) {
        diffs.push(format!(
            "{label}: not fully native: {:?}",
            lowering_for(chunk)
        ));
        return;
    }
    let (want, got) = (interp(chunk, register), aot(chunk, register));
    if !same(&want, &got) {
        diffs.push(format!("{label}: interp={want:?} aot={got:?}"));
    }
}

fn assert_no_diffs(what: &str, diffs: Vec<String>) {
    for d in &diffs {
        eprintln!("AOT DIVERGENCE {what}: {d}");
    }
    assert!(diffs.is_empty(), "{} divergence(s) in {what}", diffs.len());
}

#[test]
fn awk_unary_transcendentals_match_interpreter() {
    let mut diffs = Vec::new();
    for op in [Op::AwkSin, Op::AwkCos, Op::AwkExp] {
        for a in &scalar_operands() {
            let chunk = chunk_of(&[a], &[op.clone()]);
            check(&format!("{op:?}({})", a.label), &chunk, no_host, &mut diffs);
        }
    }
    assert_no_diffs("awk sin/cos/exp", diffs);
}

#[test]
fn awk_atan2_matches_interpreter_on_every_operand_pair() {
    let ops = scalar_operands();
    let mut diffs = Vec::new();
    for y in &ops {
        for x in &ops {
            let chunk = chunk_of(&[y, x], &[Op::AwkAtan2]);
            check(
                &format!("AwkAtan2({}, {})", y.label, x.label),
                &chunk,
                no_host,
                &mut diffs,
            );
        }
    }
    assert_no_diffs("awk atan2", diffs);
}

/// A NaN result from a transcendental is `+nan` in the interpreter
/// (`awk_canon_nan`), whatever sign the platform libm produced.
#[test]
fn awk_transcendental_nan_results_are_positive() {
    let nan_in = nonfinite().swap_remove(2);
    let inf = nonfinite().swap_remove(0);
    for (label, chunk) in [
        ("sin(inf)", chunk_of(&[&inf], &[Op::AwkSin])),
        ("cos(NaN)", chunk_of(&[&nan_in], &[Op::AwkCos])),
        (
            "atan2(NaN, 1)",
            chunk_of(&[&nan_in, &int("1", 1)], &[Op::AwkAtan2]),
        ),
    ] {
        assert!(fully_native(&chunk), "{label}: {:?}", lowering_for(&chunk));
        match aot(&chunk, no_host) {
            Outcome::Ok(Value::Float(f)) => {
                assert_eq!(f.to_bits(), f64::NAN.to_bits(), "{label}");
            }
            other => panic!("{label}: {other:?}"),
        }
    }
}

#[test]
fn awk_mkbool_matches_interpreter_on_every_kind() {
    let mut operands = scalar_operands();
    let mut diffs = Vec::new();
    for a in &operands {
        let chunk = chunk_of(&[a], &[Op::AwkMkbool]);
        check(
            &format!("AwkMkbool({})", a.label),
            &chunk,
            no_host,
            &mut diffs,
        );
    }
    // Boxed strings: truthiness is the string's (`""` and `"0"` are false),
    // not the arena handle's.
    for s in ["", "0", "0.0", "abc"] {
        let mut b = ChunkBuilder::new();
        let k = b.add_constant(Value::str(s));
        b.emit(Op::LoadConst(k), 1);
        b.emit(Op::AwkMkbool, 1);
        let chunk = b.build();
        check(&format!("AwkMkbool({s:?})"), &chunk, no_host, &mut diffs);
    }
    // Same, with an awk host registered (it answers `awk_mkbool` too).
    operands.truncate(4);
    for a in &operands {
        let chunk = chunk_of(&[a], &[Op::AwkMkbool]);
        check(
            &format!("host AwkMkbool({})", a.label),
            &chunk,
            awk_host,
            &mut diffs,
        );
    }
    assert_no_diffs("awk mkbool", diffs);
}

fn fold_ops(argc: u8) -> [Op; 3] {
    [Op::AwkAnd(argc), Op::AwkOr(argc), Op::AwkXor(argc)]
}

/// Two- and three-operand folds over every kind, negatives included: the
/// interpreter raises gawk's fatal naming the right-most negative argument.
#[test]
fn awk_bitwise_folds_match_interpreter_without_host() {
    let ops = fold_operands();
    let mut diffs = Vec::new();
    for op in fold_ops(2) {
        for a in &ops {
            for b in &ops {
                let chunk = chunk_of(&[a, b], &[op.clone()]);
                check(
                    &format!("{op:?}({}, {})", a.label, b.label),
                    &chunk,
                    no_host,
                    &mut diffs,
                );
            }
        }
    }
    for op in fold_ops(3) {
        for a in &ops {
            for b in &ops {
                for c in &ops {
                    let chunk = chunk_of(&[a, b, c], &[op.clone()]);
                    check(
                        &format!("{op:?}({}, {}, {})", a.label, b.label, c.label),
                        &chunk,
                        no_host,
                        &mut diffs,
                    );
                }
            }
        }
    }
    assert_no_diffs("awk and/or/xor", diffs);
}

/// gawk 5.4.1 values (`printf "%d"`), the same cases the interpreter's own
/// unit test pins, reached through the AOT tier.
#[test]
fn awk_bitwise_folds_match_gawk_through_aot() {
    let p = |e: i32| 2f64.powi(e);
    let cases: [(Op, Vec<f64>, f64); 4] = [
        (Op::AwkOr(2), vec![p(54), 1.0], 1.0),
        (
            Op::AwkAnd(2),
            vec![p(53) + p(54), p(60) + p(53)],
            9_007_199_254_740_992.0,
        ),
        (Op::AwkXor(2), vec![p(53) + 1.0, 1.0], 1.0),
        (Op::AwkAnd(3), vec![15.0, 9.0, 5.0], 1.0),
    ];
    for (op, args, want) in cases {
        let operands: Vec<Operand> = args.iter().map(|&f| float("x", f)).collect();
        let refs: Vec<&Operand> = operands.iter().collect();
        let chunk = chunk_of(&refs, &[op.clone()]);
        assert!(fully_native(&chunk), "{op:?}{args:?}");
        match aot(&chunk, no_host) {
            Outcome::Ok(Value::Float(f)) => assert_eq!(f, want, "{op:?}{args:?}"),
            other => panic!("{op:?}{args:?}: {other:?}"),
        }
    }
    let chunk = chunk_of(
        &[&float("1", 1.0), &float("-2", -2.0), &float("-3", -3.0)],
        &[Op::AwkAnd(3)],
    );
    match aot(&chunk, no_host) {
        Outcome::Err(e) => assert_eq!(e, "and: argument 3 negative value -3 is not allowed"),
        other => panic!("and(1, -2, -3): {other:?}"),
    }
}

/// With an awk host registered the interpreter hands the fold to the host,
/// whose default coerces a negative operand to 0 instead of raising the fatal.
/// The AOT tier must defer to it (runtime host guard → deopt).
#[test]
fn awk_bitwise_folds_defer_to_a_registered_host() {
    let ops = fold_operands();
    let mut diffs = Vec::new();
    for op in fold_ops(2) {
        for a in &ops {
            for b in &ops {
                let chunk = chunk_of(&[a, b], &[op.clone()]);
                check(
                    &format!("host {op:?}({}, {})", a.label, b.label),
                    &chunk,
                    awk_host,
                    &mut diffs,
                );
            }
        }
    }
    assert_no_diffs("awk and/or/xor with host", diffs);

    // The host answers where the host-less fold would be fatal.
    let chunk = chunk_of(&[&int("-1", -1), &int("12", 12)], &[Op::AwkOr(2)]);
    assert!(matches!(
        interp(&chunk, awk_host),
        Outcome::Ok(Value::Float(_))
    ));
    assert!(matches!(aot(&chunk, no_host), Outcome::Err(_)));
}

/// The host guard's deopt must spill register-held slot state: a slot set
/// natively before the fold is read back by the resumed interpreter.
#[test]
fn awk_fold_host_deopt_writes_back_slots() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(5), 1);
    b.emit(Op::SetSlot(0), 1);
    b.emit(Op::LoadInt(12), 1);
    b.emit(Op::LoadInt(10), 1);
    b.emit(Op::AwkAnd(2), 1);
    b.emit(Op::GetSlot(0), 1);
    b.emit(Op::Add, 1);
    let chunk = b.build();
    assert!(fully_native(&chunk), "{:?}", lowering_for(&chunk));
    for (label, register) in [("no host", no_host as fn(&mut VM)), ("host", awk_host)] {
        let (want, got) = (interp(&chunk, register), aot(&chunk, register));
        assert!(same(&want, &got), "{label}: interp={want:?} aot={got:?}");
        assert!(
            matches!(got, Outcome::Ok(Value::Float(f)) if f == 13.0),
            "{label}: {got:?}"
        );
    }
}

/// Fewer than two operands is gawk's own fatal; the AOT tier leaves it to the
/// interpreter (a static deopt point) and reports the same text.
#[test]
fn awk_fold_with_one_operand_deopts_with_the_interpreters_error() {
    for op in fold_ops(1) {
        let chunk = chunk_of(&[&int("1", 1)], &[op.clone()]);
        assert!(!fully_native(&chunk), "{op:?}");
        let (want, got) = (interp(&chunk, no_host), aot(&chunk, no_host));
        assert!(same(&want, &got), "{op:?}: interp={want:?} aot={got:?}");
        assert!(matches!(got, Outcome::Err(_)), "{op:?}: {got:?}");
    }
}

/// A boxed operand is not lowered: the op deopts and the interpreter coerces it.
#[test]
fn awk_ops_on_boxed_operands_deopt_and_agree() {
    let mut diffs = Vec::new();
    for tail in [
        vec![Op::AwkSin],
        vec![Op::AwkExp],
        vec![Op::LoadInt(2), Op::AwkAtan2],
        vec![Op::LoadInt(6), Op::AwkXor(2)],
    ] {
        let mut b = ChunkBuilder::new();
        let k = b.add_constant(Value::str("3"));
        b.emit(Op::LoadConst(k), 1);
        for op in &tail {
            b.emit(op.clone(), 1);
        }
        let chunk = b.build();
        let (want, got) = (interp(&chunk, no_host), aot(&chunk, no_host));
        if !same(&want, &got) {
            diffs.push(format!("{tail:?}: interp={want:?} aot={got:?}"));
        }
    }
    assert_no_diffs("awk ops on boxed operands", diffs);
}

/// `TimeInt` lowers to the interpreter's clock read: an `Int` (never a Float),
/// within a second of the interpreter's answer, and usable by integer ops.
#[test]
fn time_int_lowers_and_matches_interpreter() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::TimeInt, 1);
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::Add, 1);
    let chunk = b.build();
    assert!(fully_native(&chunk), "{:?}", lowering_for(&chunk));
    let got = aot(&chunk, no_host);
    let want = interp(&chunk, no_host);
    match (&want, &got) {
        (Outcome::Ok(Value::Int(w)), Outcome::Ok(Value::Int(g))) => {
            assert!((w - g).abs() <= 1, "interp={w} aot={g}");
            assert!(*g > 1_600_000_000, "aot={g}");
        }
        _ => panic!("interp={want:?} aot={got:?}"),
    }
}
