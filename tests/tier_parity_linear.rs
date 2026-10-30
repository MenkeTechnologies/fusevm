//! Interpreter-vs-linear-JIT parity for the awk ops the linear tier lowers:
//! the transcendentals (`AwkSin`/`AwkCos`/`AwkExp`/`AwkAtan2`), the guarded
//! div/mod, sqrt/log with their negative-argument warning, the gawk bitwise
//! ops with their negative-operand fatal, `$N` reads, and `Nop`.
//!
//! The interpreter (`VM::run`) is the spec. For every chunk:
//! - an interpreter `Ok(v)` must come back from `try_run_linear` as the same
//!   `Value` variant with the same bits (NaN matches any NaN);
//! - an interpreter `Error` (zero divisor, negative bitwise operand) must make
//!   the linear tier decline (`None`), so the caller's interpreter fallback
//!   raises the fatal — and the trap channel must be drained afterwards.
//!
//! Every chunk is asserted linear-eligible first, so a regression that stops
//! compiling an op fails here instead of passing vacuously.

#![cfg(feature = "jit")]

use fusevm::{Chunk, ChunkBuilder, JitCompiler, Op, VMResult, Value, VM};

/// One operand, as the op sequence that pushes it. Infinities and NaN are
/// built arithmetically because `Op::LoadFloat` of a non-finite constant is
/// not linear-eligible.
#[derive(Clone)]
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

fn pos_inf() -> Vec<Op> {
    vec![Op::LoadFloat(1e308), Op::LoadFloat(10.0), Op::Mul]
}

fn neg_inf() -> Vec<Op> {
    vec![Op::LoadFloat(-1e308), Op::LoadFloat(10.0), Op::Mul]
}

fn operands() -> Vec<Operand> {
    let mut nan = pos_inf();
    nan.extend(pos_inf());
    nan.push(Op::Sub);
    vec![
        int("0", 0),
        int("1", 1),
        int("-1", -1),
        int("7", 7),
        int("63", 63),
        int("64", 64),
        int("i64::MIN", i64::MIN),
        int("i64::MAX", i64::MAX),
        float("0.0", 0.0),
        float("-0.0", -0.0),
        float("0.5", 0.5),
        float("-2.5", -2.5),
        float("3.0", 3.0),
        float("65.0", 65.0),
        float("2^53+2", 2f64.powi(53) + 2.0),
        float("2^63", 2f64.powi(63)),
        float("1e300", 1e300),
        Operand {
            label: "+inf",
            push: pos_inf(),
        },
        Operand {
            label: "-inf",
            push: neg_inf(),
        },
        Operand {
            label: "nan",
            push: nan,
        },
    ]
}

fn build(ops: &[Op]) -> Chunk {
    let mut b = ChunkBuilder::new();
    for op in ops {
        b.emit(op.clone(), 1);
    }
    b.build()
}

fn same_value(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Float(x), Value::Float(y)) => {
            (x.is_nan() && y.is_nan()) || x.to_bits() == y.to_bits()
        }
        (Value::Int(x), Value::Int(y)) => x == y,
        (Value::Bool(x), Value::Bool(y)) => x == y,
        _ => false,
    }
}

/// Assert the linear tier agrees with the interpreter on `ops`.
fn assert_parity(ops: &[Op], what: &str) {
    let jit = JitCompiler::new();
    assert!(
        jit.is_linear_eligible(&build(ops)),
        "{what}: expected linear-eligible: {ops:?}"
    );
    let interp = VM::new(build(ops)).run();
    let linear = jit.try_run_linear(&build(ops), &[]);
    match interp {
        VMResult::Ok(v) => match &linear {
            Some(l) => assert!(
                same_value(&v, l),
                "{what}: interpreter {v:?}, linear {l:?}: {ops:?}"
            ),
            None => panic!("{what}: interpreter {v:?}, linear declined: {ops:?}"),
        },
        VMResult::Error(e) => {
            assert!(
                linear.is_none(),
                "{what}: interpreter raised {e:?}, linear answered {linear:?}: {ops:?}"
            );
            assert!(
                !jit.take_awk_div_trap(),
                "{what}: linear left the trap channel set: {ops:?}"
            );
        }
        other => panic!("{what}: unexpected interpreter result {other:?}"),
    }
}

fn unary_cases(op: Op) {
    for a in operands() {
        let mut ops = a.push.clone();
        ops.push(op.clone());
        assert_parity(&ops, &format!("{op:?}({})", a.label));
    }
}

fn binary_cases(op: Op) {
    for a in operands() {
        for b in operands() {
            let mut ops = a.push.clone();
            ops.extend(b.push.iter().cloned());
            ops.push(op.clone());
            assert_parity(&ops, &format!("{op:?}({}, {})", a.label, b.label));
        }
    }
}

#[test]
fn awk_sin_cos_exp_match_interpreter() {
    unary_cases(Op::AwkSin);
    unary_cases(Op::AwkCos);
    unary_cases(Op::AwkExp);
}

#[test]
fn awk_atan2_matches_interpreter() {
    binary_cases(Op::AwkAtan2);
}

#[test]
fn awk_div_jit_matches_interpreter_and_traps_on_zero() {
    binary_cases(Op::AwkDivJit);
}

#[test]
fn awk_mod_jit_matches_interpreter_and_traps_on_zero() {
    binary_cases(Op::AwkModJit);
}

/// Negative operands print the awk warning in both tiers and answer NaN.
#[test]
fn awk_sqrt_log_jit_match_interpreter() {
    unary_cases(Op::AwkSqrtJit);
    unary_cases(Op::AwkLogJit);
}

/// Shift counts of 64 or more answer 0; a negative operand is the gawk fatal.
#[test]
fn awk_shifts_match_interpreter() {
    binary_cases(Op::AwkLshiftJit);
    binary_cases(Op::AwkRshiftJit);
}

#[test]
fn awk_compl_matches_interpreter() {
    unary_cases(Op::AwkComplJit);
}

/// and/or/xor fold as `u64` and answer Float (never Int), over 2 and 3
/// operands, including a negative operand at every position.
#[test]
fn awk_and_or_xor_match_interpreter() {
    let ops3 = |a: &Operand, b: &Operand, c: &Operand, op: Op| {
        let mut ops = a.push.clone();
        ops.extend(b.push.iter().cloned());
        ops.extend(c.push.iter().cloned());
        ops.push(op);
        ops
    };
    for mk in [Op::AwkAnd as fn(u8) -> Op, Op::AwkOr, Op::AwkXor] {
        binary_cases(mk(2));
        let all = operands();
        for a in &all {
            for b in &all {
                let c = float("12.0", 12.0);
                let op = mk(3);
                let what = format!("{op:?}({}, {}, 12.0)", a.label, b.label);
                assert_parity(&ops3(a, b, &c, op.clone()), &what);
                let what = format!("{op:?}(12.0, {}, {})", a.label, b.label);
                assert_parity(&ops3(&c, a, b, op), &what);
            }
        }
    }
}

extern "C" fn field_hook(idx: i64) -> f64 {
    match idx {
        0 => -0.0,
        1 => 2.5,
        2 => f64::INFINITY,
        _ => idx as f64 * -3.0,
    }
}

/// `$N` goes through the thread-local host hook in both tiers; with no hook
/// installed both answer 0.0.
#[test]
fn awk_get_field_num_matches_interpreter() {
    fusevm::set_awk_field_num_hook(None);
    assert_parity(&[Op::AwkGetFieldNum(3)], "$3 without hook");
    fusevm::set_awk_field_num_hook(Some(field_hook));
    for idx in [0u16, 1, 2, 9, u16::MAX] {
        assert_parity(&[Op::AwkGetFieldNum(idx)], &format!("${idx}"));
    }
    assert_parity(
        &[Op::AwkGetFieldNum(1), Op::AwkGetFieldNum(9), Op::AwkDivJit],
        "$1 / $9",
    );
    fusevm::set_awk_field_num_hook(None);
}

#[test]
fn nop_is_transparent() {
    assert_parity(
        &[
            Op::Nop,
            Op::LoadInt(5),
            Op::Nop,
            Op::LoadFloat(1.5),
            Op::Add,
            Op::Nop,
        ],
        "nops around add",
    );
    assert!(!JitCompiler::new().is_linear_eligible(&build(&[Op::Nop])));
}

/// The trap sentinel must match the function's return type. Here the chunk
/// answers an integer, so the trap exit returns through the `i64` register.
#[test]
fn trap_in_integer_result_chunk() {
    for divisor in [0.0, -0.0, 2.0] {
        let ops = [
            Op::LoadFloat(1.0),
            Op::LoadFloat(divisor),
            Op::AwkDivJit,
            Op::Pop,
            Op::LoadInt(3),
        ];
        assert_parity(&ops, &format!("div by {divisor} then Int"));
    }
    let ops = [
        Op::LoadInt(4),
        Op::LoadInt(-1),
        Op::AwkLshiftJit,
        Op::Pop,
        Op::LoadInt(3),
    ];
    assert_parity(&ops, "negative lshift then Int");
}

/// A warning printed before a trap would print again when the interpreter
/// re-runs the declined chunk, so that ordering is not linear-eligible. The
/// reverse ordering is: the trap exits before the warning in both tiers.
#[test]
fn warning_before_trap_declines() {
    let jit = JitCompiler::new();
    let warn_then_trap = [
        Op::LoadFloat(-1.0),
        Op::AwkSqrtJit,
        Op::LoadFloat(0.0),
        Op::AwkDivJit,
    ];
    assert!(!jit.is_linear_eligible(&build(&warn_then_trap)));
    assert_parity(
        &[
            Op::LoadFloat(1.0),
            Op::LoadFloat(0.0),
            Op::AwkModJit,
            Op::AwkLogJit,
        ],
        "trap then log",
    );
}

/// Ops kept out of the linear tier: host-dispatched `AwkInt`, a fold with
/// fewer than two arguments, and boolean operands (the interpreter coerces
/// them through `to_float`; native code has no boolean operand lowering).
#[test]
fn excluded_shapes_stay_ineligible() {
    let jit = JitCompiler::new();
    for ops in [
        vec![Op::LoadFloat(2.5), Op::AwkInt],
        vec![Op::LoadFloat(2.0), Op::AwkAnd(1)],
        vec![Op::LoadTrue, Op::AwkSin],
        vec![Op::LoadTrue, Op::LoadFloat(2.0), Op::AwkDivJit],
    ] {
        assert!(!jit.is_linear_eligible(&build(&ops)), "{ops:?}");
    }
}

#[cfg(feature = "jit-disk-cache")]
mod disk_cache {
    use super::*;

    fn fresh_dir(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "fusevm_tier_parity_linear_{tag}_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    /// The linear blob file for `ops` (see `cache_path` in `src/jit.rs`).
    /// Checked by name, not by counting files: the cache directory override is
    /// process-global, so concurrently running tests may write into it too.
    fn lin_blob(dir: &std::path::Path, ops: &[Op]) -> std::path::PathBuf {
        let op_hash = build(ops).op_hash;
        dir.join(format!("{op_hash:016x}.{:016x}.lin.fjit", 0))
    }

    fn run_in(dir: &std::path::Path, ops: &[Op]) -> Option<Value> {
        let jit = JitCompiler::new();
        jit.set_jit_cache_dir(Some(dir.to_path_buf()));
        let r = jit.try_run_linear(&build(ops), &[]);
        jit.set_jit_cache_dir(None);
        r
    }

    /// Each new libcall (div trap, bit trap, negative-argument warning, `$N`
    /// hook) has a stable host id, so chunks using it persist as native code
    /// and reload in a fresh thread with the same answer — trap included.
    #[test]
    fn guarded_awk_chunks_persist_and_reload() {
        let dir = fresh_dir("guarded");
        let cases: Vec<(Vec<Op>, Option<Value>)> = vec![
            (
                vec![Op::LoadFloat(7.0), Op::LoadFloat(2.0), Op::AwkDivJit],
                Some(Value::Float(3.5)),
            ),
            (
                vec![Op::LoadFloat(7.0), Op::LoadFloat(0.0), Op::AwkModJit],
                None,
            ),
            (vec![Op::LoadFloat(-1.0), Op::AwkComplJit], None),
            (
                vec![Op::LoadFloat(12.0), Op::LoadFloat(10.0), Op::AwkAnd(2)],
                Some(Value::Float(8.0)),
            ),
            (
                vec![Op::LoadFloat(16.0), Op::AwkSqrtJit],
                Some(Value::Float(4.0)),
            ),
            (vec![Op::AwkGetFieldNum(4)], Some(Value::Float(0.0))),
        ];
        for (i, (ops, want)) in cases.iter().enumerate() {
            assert_eq!(&run_in(&dir, ops), want, "first run {ops:?}");
            assert!(
                lin_blob(&dir, ops).exists(),
                "case {i} not persisted: {ops:?}"
            );
            let (dir2, ops2) = (dir.clone(), ops.clone());
            let loads_before = fusevm::jit::stats().disk_loads;
            let second = std::thread::spawn(move || run_in(&dir2, &ops2))
                .join()
                .unwrap();
            assert_eq!(&second, want, "reload {ops:?}");
            assert!(
                fusevm::jit::stats().disk_loads > loads_before,
                "case {i} was rebuilt, not loaded: {ops:?}"
            );
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
