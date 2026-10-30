//! VM-level dispatch of the AWK op range with and without a registered host.
//!
//! The AWK builtins reach the VM two ways — dedicated `Op::Awk*` variants and
//! the generic `Op::ExtendedWide(AWK_*, payload)` encoding — and run through two
//! code paths: `dispatch_awk` (host registered) and `dispatch_awk_stub` (no
//! host). The stub duplicates the `AwkHost` default bodies so pure builtins work
//! host-free. These tests pin that the duplication stays in sync, that both
//! encodings route identically, and that the stub keeps the stack balanced for
//! the host-only ops it cannot compute.

use fusevm::awk_builtins::*;
use fusevm::{ChunkBuilder, DefaultAwkHost, Op, VMResult, Value, VM};

/// A value pushed before the ops under test; the stack must end as
/// `[SENTINEL, ...results]` or the op popped/pushed the wrong number of values.
const SENTINEL: i64 = 0x5e17;

/// The stack as the last op left it: `VM::run` pops the top value into
/// `VMResult::Ok`, so put it back.
fn full_stack(r: &VMResult, vm: &VM) -> Vec<Value> {
    let mut s = vm.stack.clone();
    if let VMResult::Ok(v) = r {
        s.push(v.clone());
    }
    s
}

fn exec(ops: &[Op], host: bool) -> (VMResult, Vec<Value>) {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(SENTINEL), 1);
    for op in ops {
        b.emit(op.clone(), 1);
    }
    let mut vm = VM::new(b.build());
    if host {
        vm.set_awk_host(Box::new(DefaultAwkHost));
    }
    let r = vm.run();
    let stack = full_stack(&r, &vm);
    (r, stack)
}

/// Stack after the sentinel, asserting the sentinel survived.
fn results(ops: &[Op], host: bool) -> Vec<Value> {
    let (r, stack) = exec(ops, host);
    assert!(
        !matches!(r, VMResult::Error(_)),
        "unexpected error {r:?} for {ops:?}"
    );
    assert_eq!(
        stack.first(),
        Some(&Value::Int(SENTINEL)),
        "sentinel lost — op over-popped: {ops:?} -> {stack:?}"
    );
    stack[1..].to_vec()
}

/// Value equality that treats NaN == NaN and distinguishes -0.0 from 0.0, so
/// the comparison checks the exact result rather than IEEE `==`.
fn same(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Float(x), Value::Float(y)) => x.to_bits() == y.to_bits(),
        _ => a == b,
    }
}

/// Build `[operand pushes..., op]` where string operands are constants.
fn chunk_ops(b: &mut ChunkBuilder, operands: &[Value]) {
    for v in operands {
        match v {
            Value::Int(n) => {
                b.emit(Op::LoadInt(*n), 1);
            }
            Value::Float(f) => {
                b.emit(Op::LoadFloat(*f), 1);
            }
            other => {
                let k = b.add_constant(other.clone());
                b.emit(Op::LoadConst(k), 1);
            }
        }
    }
}

fn run_with_operands(operands: &[Value], op: Op, host: bool) -> (VMResult, Vec<Value>) {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(SENTINEL), 1);
    chunk_ops(&mut b, operands);
    b.emit(op, 1);
    let mut vm = VM::new(b.build());
    if host {
        vm.set_awk_host(Box::new(DefaultAwkHost));
    }
    let r = vm.run();
    let stack = full_stack(&r, &vm);
    (r, stack)
}

fn one_result(operands: &[Value], op: Op, host: bool) -> Value {
    let (r, stack) = run_with_operands(operands, op.clone(), host);
    assert!(
        !matches!(r, VMResult::Error(_)),
        "unexpected error {r:?} for {op:?} {operands:?}"
    );
    assert_eq!(
        stack.len(),
        2,
        "{op:?} {operands:?} must leave exactly [sentinel, result]: {stack:?}"
    );
    assert_eq!(stack[0], Value::Int(SENTINEL));
    stack[1].clone()
}

// ───────────────────────────────────────────────────────────────────────────
// Stub vs DefaultAwkHost: the host-free duplicate must agree with the trait
// defaults for every pure builtin, through both op encodings.
// ───────────────────────────────────────────────────────────────────────────

/// (dedicated op, ExtendedWide id, ExtendedWide payload, operands)
fn pure_cases() -> Vec<(Op, u16, usize, Vec<Value>)> {
    let f = Value::Float;
    let st = |x: &str| Value::str(x);
    let mut v = Vec::new();
    for x in [3.7, -2.9, 0.0, -0.0, 1e300, f64::NAN, f64::INFINITY, -1.0] {
        v.push((Op::AwkInt, AWK_INT, 0, vec![f(x)]));
        v.push((Op::AwkSqrt, AWK_SQRT, 0, vec![f(x)]));
        v.push((Op::AwkSin, AWK_SIN, 0, vec![f(x)]));
        v.push((Op::AwkCos, AWK_COS, 0, vec![f(x)]));
        v.push((Op::AwkExp, AWK_EXP, 0, vec![f(x)]));
        v.push((Op::AwkLog, AWK_LOG, 0, vec![f(x)]));
        v.push((Op::AwkAtan2, AWK_ATAN2, 0, vec![f(x), f(-1.5)]));
        v.push((Op::AwkAtan2, AWK_ATAN2, 0, vec![f(1.5), f(x)]));
        v.push((Op::AwkMkbool, AWK_MKBOOL, 1, vec![f(x)]));
        v.push((Op::AwkChr, AWK_CHR, 1, vec![f(x)]));
    }
    for t in ["", "héllo", "0x1F", "017", "  42abc", "ABC def", "Ÿ"] {
        v.push((Op::AwkStrtonum, AWK_STRTONUM, 0, vec![st(t)]));
        v.push((Op::AwkOrd, AWK_ORD, 1, vec![st(t)]));
        v.push((Op::AwkToLower, AWK_TOLOWER, 0, vec![st(t)]));
        v.push((Op::AwkToUpper, AWK_TOUPPER, 0, vec![st(t)]));
        v.push((Op::AwkLength(1), AWK_LENGTH, 1, vec![st(t)]));
        v.push((Op::AwkMkbool, AWK_MKBOOL, 1, vec![st(t)]));
        v.push((Op::AwkIndex, AWK_INDEX, 0, vec![st(t), st("l")]));
        v.push((Op::AwkSubstr(2), AWK_SUBSTR, 2, vec![st(t), Value::Int(2)]));
        v.push((
            Op::AwkSubstr(3),
            AWK_SUBSTR,
            3,
            vec![st(t), Value::Int(-1), Value::Int(3)],
        ));
    }
    for (a, b) in [(7.0, 2.0), (-7.0, 2.0), (7.0, 0.0), (7.0, 0.5), (0.0, -3.0)] {
        v.push((Op::AwkIntdiv, AWK_INTDIV, 2, vec![f(a), f(b)]));
        v.push((Op::AwkIntdiv0, AWK_INTDIV0, 2, vec![f(a), f(b)]));
    }
    for spec in ["1970 01 01 00 00 00", "2000 02 30 00 00 00", "garbage"] {
        v.push((
            Op::AwkMktime(2),
            AWK_MKTIME,
            2,
            vec![st(spec), Value::Int(1)],
        ));
    }
    v.push((
        Op::AwkStrftime(3),
        AWK_STRFTIME,
        3,
        vec![st("%Y-%m-%d %H:%M:%S"), f(86_400.5), Value::Int(1)],
    ));
    v
}

#[test]
fn stub_and_default_host_agree_on_every_pure_builtin() {
    for (op, id, payload, operands) in pure_cases() {
        let stub = one_result(&operands, op.clone(), false);
        let hosted = one_result(&operands, op.clone(), true);
        assert!(
            same(&stub, &hosted),
            "{op:?} {operands:?}: stub {stub:?} != DefaultAwkHost {hosted:?}"
        );
        // The generic encoding must route to the same body in both modes.
        let wide = Op::ExtendedWide(id, payload);
        for host in [false, true] {
            let w = one_result(&operands, wide.clone(), host);
            assert!(
                same(&w, &stub),
                "{wide:?} (host={host}) {operands:?}: {w:?} != {op:?} {stub:?}"
            );
        }
    }
}

/// Spot-check the agreed values themselves so a change to both copies at once
/// (which the agreement test cannot see) is still caught.
#[test]
fn pure_builtin_values_through_the_vm() {
    let f = Value::Float;
    let st = |x: &str| Value::str(x);
    for host in [false, true] {
        assert_eq!(one_result(&[f(-2.9)], Op::AwkInt, host), Value::Int(-2));
        // Beyond i64 range int() keeps a truncated Float rather than saturating.
        assert_eq!(one_result(&[f(1e300)], Op::AwkInt, host), f(1e300));
        assert_eq!(one_result(&[st("0x1F")], Op::AwkStrtonum, host), f(31.0));
        assert_eq!(one_result(&[st("017")], Op::AwkStrtonum, host), f(15.0));
        assert_eq!(one_result(&[st("Ÿ")], Op::AwkOrd, host), f(376.0));
        assert_eq!(one_result(&[f(0x263A as f64)], Op::AwkChr, host), st("☺"));
        // A surrogate is not a scalar value: empty string, not a panic.
        assert_eq!(one_result(&[f(55296.0)], Op::AwkChr, host), st(""));
        assert_eq!(
            one_result(&[st("héllo")], Op::AwkLength(1), host),
            Value::Int(5),
            "length counts chars, not UTF-8 bytes"
        );
        assert_eq!(
            one_result(&[st("héllo"), st("l")], Op::AwkIndex, host),
            Value::Int(3)
        );
        assert_eq!(
            one_result(&[st("héllo"), Value::Int(2)], Op::AwkSubstr(2), host),
            st("éllo")
        );
        assert_eq!(
            one_result(&[f(-7.0), f(2.0)], Op::AwkIntdiv, host),
            f(-3.0),
            "intdiv truncates toward zero"
        );
        assert_eq!(
            one_result(&[f(7.0), f(0.0)], Op::AwkIntdiv, host),
            Value::Undef
        );
        // A divisor that truncates to 0 is still a zero divisor.
        assert_eq!(
            one_result(&[f(7.0), f(0.5)], Op::AwkIntdiv, host),
            Value::Undef
        );
        assert_eq!(one_result(&[f(7.0), f(0.5)], Op::AwkIntdiv0, host), f(0.0));
        assert_eq!(
            one_result(
                &[st("1970 01 02 00 00 00"), Value::Int(1)],
                Op::AwkMktime(2),
                host
            ),
            f(86_400.0)
        );
        assert_eq!(
            one_result(
                &[st("2000 02 30 00 00 00"), Value::Int(1)],
                Op::AwkMktime(2),
                host
            ),
            f(-1.0),
            "an impossible date is -1, not a normalized timestamp"
        );
        assert_eq!(
            one_result(
                &[st("%Y-%m-%d %H:%M:%S"), f(86_400.5), Value::Int(1)],
                Op::AwkStrftime(3),
                host
            ),
            st("1970-01-02 00:00:00")
        );
    }
}

/// `intdiv(i64::MIN, -1)` overflows a naive `i64 / i64`; it must produce the
/// mathematically correct quotient (2^63), not panic the VM.
#[test]
fn intdiv_of_i64_min_by_minus_one_does_not_panic() {
    let min = Value::Float(i64::MIN as f64);
    let neg1 = Value::Float(-1.0);
    for host in [false, true] {
        for op in [Op::AwkIntdiv, Op::AwkIntdiv0] {
            assert_eq!(
                one_result(&[min.clone(), neg1.clone()], op.clone(), host),
                Value::Float(2f64.powi(63)),
                "{op:?} host={host}"
            );
        }
    }
}

// ───────────────────────────────────────────────────────────────────────────
// Bitwise builtins: no host → the VM raises gawk's fatal on a negative
// operand; a registered host owns the semantics (defaults coerce to 0).
// ───────────────────────────────────────────────────────────────────────────

#[test]
fn bitwise_negative_operand_is_fatal_only_without_a_host() {
    let cases: Vec<(Op, Op, Vec<Value>)> = vec![
        (
            Op::AwkAnd(2),
            Op::ExtendedWide(AWK_AND, 2),
            vec![Value::Float(-1.0), Value::Float(3.0)],
        ),
        (
            Op::AwkOr(2),
            Op::ExtendedWide(AWK_OR, 2),
            vec![Value::Float(1.0), Value::Float(-3.0)],
        ),
        (
            Op::AwkXor(2),
            Op::ExtendedWide(AWK_XOR, 2),
            vec![Value::Float(-1.0), Value::Float(1.0)],
        ),
        (
            Op::AwkCompl,
            Op::ExtendedWide(AWK_COMPL, 0),
            vec![Value::Float(-1.0)],
        ),
        (
            Op::AwkLshift,
            Op::ExtendedWide(AWK_LSHIFT, 0),
            vec![Value::Float(-1.0), Value::Float(1.0)],
        ),
        (
            Op::AwkRshift,
            Op::ExtendedWide(AWK_RSHIFT, 0),
            vec![Value::Float(8.0), Value::Float(-1.0)],
        ),
    ];
    for (op, wide, operands) in cases {
        for o in [op.clone(), wide] {
            let (r, _) = run_with_operands(&operands, o.clone(), false);
            match r {
                VMResult::Error(msg) => assert!(
                    msg.contains("negative"),
                    "{o:?}: fatal text should name the negative operand: {msg}"
                ),
                other => panic!("{o:?} {operands:?} without host must error, got {other:?}"),
            }
            // With a host there is no error channel: the run completes and
            // leaves exactly one result.
            one_result(&operands, o, true);
        }
    }
}

#[test]
fn bitwise_results_agree_between_native_and_default_host() {
    let f = Value::Float;
    let cases: Vec<(Op, Vec<Value>, f64)> = vec![
        (
            Op::AwkAnd(3),
            vec![f(0xff as f64), f(0x0f as f64), f(0x3c as f64)],
            0x0c as f64,
        ),
        (
            Op::AwkOr(2),
            vec![f(0x10 as f64), f(0x01 as f64)],
            0x11 as f64,
        ),
        (
            Op::AwkXor(2),
            vec![f(0xff as f64), f(0x0f as f64)],
            0xf0 as f64,
        ),
        // Operands truncate toward zero before the bit op.
        (Op::AwkAnd(2), vec![f(7.9), f(3.2)], 3.0),
        (Op::AwkLshift, vec![f(1.0), f(10.0)], 1024.0),
        (Op::AwkRshift, vec![f(1024.0), f(3.0)], 128.0),
        (Op::AwkLshift, vec![f(1.0), f(64.0)], 0.0),
    ];
    for (op, operands, want) in cases {
        for host in [false, true] {
            assert_eq!(
                one_result(&operands, op.clone(), host),
                f(want),
                "{op:?} {operands:?} host={host}"
            );
        }
    }
}

// ───────────────────────────────────────────────────────────────────────────
// Host-only ops without a host: inert, but the stack must stay balanced.
// ───────────────────────────────────────────────────────────────────────────

/// Statement-form ops consume their operands and push nothing; value-form ops
/// consume their operands and push exactly one neutral default.
#[test]
fn stub_keeps_stack_balanced_for_host_only_ops() {
    let i = Op::LoadInt;
    // (ops after the sentinel, expected results above the sentinel)
    let cases: Vec<(Vec<Op>, Vec<Value>)> = vec![
        (
            vec![i(1), Op::ExtendedWide(AWK_FIELD_GET, 0)],
            vec![Value::str("")],
        ),
        (vec![i(9), i(1), Op::ExtendedWide(AWK_FIELD_SET, 0)], vec![]),
        (vec![Op::ExtendedWide(AWK_NF, 0)], vec![Value::Int(0)]),
        (vec![i(1), Op::ExtendedWide(AWK_SET_RECORD, 0)], vec![]),
        (
            vec![Op::ExtendedWide(AWK_SPECIAL_GET, 0)],
            vec![Value::Undef],
        ),
        (vec![i(1), Op::ExtendedWide(AWK_SPECIAL_SET, 0)], vec![]),
        (vec![i(1), i(2), i(3), Op::AwkPrint(3)], vec![]),
        (vec![i(1), i(2), Op::AwkPrintf(2)], vec![]),
        (vec![i(1), i(2), Op::AwkSprintf(2)], vec![Value::str("")]),
        (vec![Op::AwkLength(0)], vec![Value::Int(0)]),
        (vec![i(1), i(2), i(3), Op::AwkSplit(3)], vec![Value::Int(0)]),
        (vec![i(1), i(2), Op::AwkSub(2)], vec![Value::Int(0)]),
        (vec![i(1), i(2), i(3), Op::AwkGsub(3)], vec![Value::Int(0)]),
        (vec![i(1), i(2), Op::AwkMatch], vec![Value::Int(0)]),
        (
            vec![i(1), i(2), i(3), i(4), Op::AwkGensub(4)],
            vec![Value::str("")],
        ),
        (vec![i(1), Op::AwkArrayGet(0)], vec![Value::str("")]),
        (vec![i(9), i(1), Op::AwkArraySet(0)], vec![]),
        (vec![i(1), Op::AwkArrayExists(0)], vec![Value::Bool(false)]),
        (vec![i(1), Op::AwkArrayDelete(0)], vec![]),
        (vec![Op::AwkArrayClear(0)], vec![]),
        (vec![Op::AwkArrayLen(0)], vec![Value::Int(0)]),
    ];
    for (ops, want) in cases {
        assert_eq!(results(&ops, false), want, "no host: {ops:?}");
        // DefaultAwkHost returns the same neutral values for these ops.
        assert_eq!(results(&ops, true), want, "DefaultAwkHost: {ops:?}");
    }
}

/// `getline` pops a source operand only for the file/command forms.
#[test]
fn getline_pops_operand_only_for_file_and_cmd_sources() {
    use fusevm::awk_builtins::getline_source as g;
    for host in [false, true] {
        for src in [g::MAIN, g::MAIN_VAR] {
            let r = results(&[Op::ExtendedWide(AWK_GETLINE, src)], host);
            assert_eq!(r, vec![Value::Int(0)], "src={src} host={host}");
        }
        for src in [g::FILE, g::FILE_VAR, g::CMD, g::CMD_VAR] {
            let r = results(&[Op::LoadInt(7), Op::ExtendedWide(AWK_GETLINE, src)], host);
            assert_eq!(r, vec![Value::Int(0)], "src={src} host={host}");
        }
    }
}

// ───────────────────────────────────────────────────────────────────────────
// PRNG: VM-owned seed, identical with or without a host.
// ───────────────────────────────────────────────────────────────────────────

#[test]
fn srand_returns_previous_seed_and_reseeding_replays_rand() {
    for host in [false, true] {
        let ops = [
            Op::LoadInt(42),
            Op::AwkSrand(1), // -> previous seed (initial)
            Op::AwkRand,
            Op::AwkRand,
            Op::LoadInt(42),
            Op::AwkSrand(1), // -> 42's successor state, not 42
            Op::AwkRand,
            Op::AwkRand,
            Op::LoadFloat(7.9),
            Op::AwkSrand(1), // -> previous
            Op::Pop,
            Op::LoadInt(0),
            Op::AwkSrand(1), // -> 7 (7.9 truncated as u32)
        ];
        let r = results(&ops, host);
        assert_eq!(r.len(), 7, "host={host}: {r:?}");
        let (a1, a2, b1, b2) = (&r[1], &r[2], &r[4], &r[5]);
        assert_eq!((a1, a2), (b1, b2), "same seed must replay the sequence");
        assert_ne!(a1, a2, "consecutive rand() values must differ");
        for v in [a1, a2] {
            let x = v.to_float();
            assert!((0.0..1.0).contains(&x), "rand() out of [0,1): {x}");
        }
        assert_ne!(
            r[3],
            Value::Float(42.0),
            "srand returns the current state, which rand() advanced past 42"
        );
        assert_eq!(r[6], Value::Float(7.0), "srand(7.9) seeds with 7");
    }
}

#[test]
fn rand_sequence_is_identical_with_and_without_host() {
    let ops = [
        Op::LoadInt(1),
        Op::AwkSrand(1),
        Op::Pop,
        Op::AwkRand,
        Op::AwkRand,
        Op::AwkRand,
    ];
    assert_eq!(results(&ops, false), results(&ops, true));
}

// ───────────────────────────────────────────────────────────────────────────
// Op-id range helpers
// ───────────────────────────────────────────────────────────────────────────

#[test]
fn awk_op_id_ranges_cover_every_constant_and_nothing_outside() {
    let all = [
        AWK_FIELD_GET,
        AWK_FIELD_SET,
        AWK_NF,
        AWK_SET_RECORD,
        AWK_SPECIAL_GET,
        AWK_SPECIAL_SET,
        AWK_PRINT,
        AWK_PRINTF,
        AWK_SPRINTF,
        AWK_GETLINE,
        AWK_LENGTH,
        AWK_SUBSTR,
        AWK_INDEX,
        AWK_SPLIT,
        AWK_SUB,
        AWK_GSUB,
        AWK_MATCH,
        AWK_TOLOWER,
        AWK_TOUPPER,
        AWK_INT,
        AWK_SQRT,
        AWK_SIN,
        AWK_COS,
        AWK_EXP,
        AWK_LOG,
        AWK_ATAN2,
        AWK_ARRAY_GET,
        AWK_ARRAY_SET,
        AWK_ARRAY_EXISTS,
        AWK_ARRAY_DELETE,
        AWK_ARRAY_CLEAR,
        AWK_ARRAY_LEN,
        AWK_AND,
        AWK_OR,
        AWK_XOR,
        AWK_COMPL,
        AWK_LSHIFT,
        AWK_RSHIFT,
        AWK_STRTONUM,
        AWK_SYSTIME,
        AWK_RAND,
        AWK_SRAND,
        AWK_STRFTIME,
        AWK_MKTIME,
        AWK_ORD,
        AWK_CHR,
        AWK_MKBOOL,
        AWK_INTDIV,
        AWK_INTDIV0,
        AWK_GENSUB,
    ];
    let mut seen = std::collections::HashSet::new();
    for id in all {
        assert!(is_awk_op(id), "{id} outside the AWK range");
        assert!(seen.insert(id), "duplicate AWK op id {id}");
    }
    let bitwise = [AWK_AND, AWK_OR, AWK_XOR, AWK_COMPL, AWK_LSHIFT, AWK_RSHIFT];
    for id in all {
        assert_eq!(is_awk_bitwise_op(id), bitwise.contains(&id), "id {id}");
    }
    assert!(!is_awk_op(AWK_OP_BASE - 1));
    assert!(!is_awk_op(AWK_OP_END));
    assert!(is_awk_op(AWK_OP_END - 1));
}

/// `ExtendedWide` ids below the AWK range go to the generic wide handler, never
/// the AWK dispatcher — with no handler installed they are a no-op.
#[test]
fn extended_wide_below_awk_range_bypasses_awk_dispatch() {
    let r = results(
        &[Op::LoadInt(5), Op::ExtendedWide(AWK_OP_BASE - 1, 3)],
        false,
    );
    assert_eq!(r, vec![Value::Int(5)]);
}
