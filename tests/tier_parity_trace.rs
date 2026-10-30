//! Interpreter-vs-trace differential tests for the slot and fused counter-loop
//! ops the tracing tier lowers: `PreDecSlot`, `PostIncSlot`, `PostDecSlot`,
//! `SlotLtIntJumpIfFalse`, `SlotIncLtIntJumpBack` (as an inner branch and as
//! the loop's closing back-edge) and `AccumSumLoop`.
//!
//! Every case runs the same chunk twice — tracing off (the reference) and
//! tracing on — and compares the result and every frame slot bit for bit.
//! A case that is meant to reach the tier also asserts the trace compiled and
//! that native code actually ran (`trace_hits` moved), so a clean diff cannot
//! come from the interpreter answering both runs.

#![cfg(feature = "jit")]

use std::collections::HashMap;
use std::sync::{Mutex, MutexGuard, OnceLock};

use fusevm::jit::stats;
use fusevm::{Chunk, ChunkBuilder, JitCompiler, Op, VMResult, Value, VM};

/// `trace_hits` is process-global; every test holds this lock so the delta it
/// reads belongs to its own run.
fn serial() -> MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|e| e.into_inner())
}

/// One assembler item: an op, a label, or an op whose jump target is a label.
enum A {
    O(Op),
    L(&'static str),
    J(fn(usize) -> Op, &'static str),
}

fn o(op: Op) -> A {
    A::O(op)
}

/// Assemble `items` after a unique `LoadInt(salt); Pop` prefix (so every case
/// has its own `op_hash` and trace-cache key). `subs` registers sub entries by
/// label; their names are interned first, so the `n`th name is index `n`.
fn asm(
    salt: i64,
    subs: &[(&str, &'static str)],
    items: Vec<A>,
) -> (Chunk, HashMap<&'static str, usize>) {
    let mut labels = HashMap::new();
    let mut ip = 2;
    for it in &items {
        match it {
            A::L(l) => {
                labels.insert(*l, ip);
            }
            _ => ip += 1,
        }
    }
    let mut b = ChunkBuilder::new();
    for (i, (name, _)) in subs.iter().enumerate() {
        assert_eq!(b.add_name(name) as usize, i);
    }
    b.emit(Op::LoadInt(salt), 1);
    b.emit(Op::Pop, 1);
    for it in items {
        match it {
            A::O(op) => {
                b.emit(op, 1);
            }
            A::J(mk, l) => {
                b.emit(mk(labels[l]), 1);
            }
            A::L(_) => {}
        }
    }
    for (i, (_, l)) in subs.iter().enumerate() {
        b.add_sub_entry(i as u16, labels[l]);
    }
    (b.build(), labels)
}

/// Outcome of one run: the result value (or the error text) and the main
/// frame's slots.
#[derive(Debug)]
struct Run {
    result: Result<Option<Value>, String>,
    slots: Vec<Value>,
}

fn run(chunk: &Chunk, tracing: bool, seed: &[Value], nslots: usize) -> Run {
    let mut vm = VM::new(chunk.clone());
    if tracing {
        vm.enable_tracing_jit();
    }
    let frame = vm.frames.last_mut().unwrap();
    frame.slots.extend(seed.iter().cloned());
    while frame.slots.len() < nslots {
        frame.slots.push(Value::Int(0));
    }
    let result = match vm.run() {
        VMResult::Ok(v) => Ok(Some(v)),
        VMResult::Halted => Ok(vm.stack.last().cloned()),
        VMResult::Error(e) => Err(e),
    };
    let slots = vm
        .frames
        .first()
        .map(|f| f.slots.clone())
        .unwrap_or_default();
    Run { result, slots }
}

/// Bit-exact `Value` equality: floats compare on raw bits (NaNs match).
fn same(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Float(x), Value::Float(y)) => {
            x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan())
        }
        _ => a == b,
    }
}

fn same_run(a: &Run, b: &Run) -> bool {
    let res = match (&a.result, &b.result) {
        (Ok(Some(x)), Ok(Some(y))) => same(x, y),
        (Ok(None), Ok(None)) => true,
        (Err(x), Err(y)) => x == y,
        _ => false,
    };
    res && a.slots.len() == b.slots.len() && a.slots.iter().zip(&b.slots).all(|(x, y)| same(x, y))
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Expect {
    /// The trace at the anchor must compile and its native code must run.
    Native,
    /// The trace must NOT compile (the tier refuses the shape); parity still holds.
    Refused,
    /// Only parity is asserted (a guard may or may not let the trace run).
    Parity,
}

/// Run `chunk` both ways, assert the runs agree, then check the tier outcome.
/// Returns the traced run for case-specific assertions.
fn check(
    what: &str,
    chunk: &Chunk,
    anchor: usize,
    seed: &[Value],
    nslots: usize,
    expect: Expect,
) -> Run {
    let _g = serial();
    let reference = run(chunk, false, seed, nslots);
    let hits0 = stats().trace_hits;
    let traced = run(chunk, true, seed, nslots);
    let hits = stats().trace_hits - hits0;
    assert!(
        same_run(&reference, &traced),
        "{what}: interpreter and trace disagree\n  interp: {reference:?}\n  trace:  {traced:?}"
    );
    let compiled = JitCompiler::new().trace_is_compiled(chunk, anchor);
    match expect {
        Expect::Native => {
            assert!(compiled, "{what}: trace at anchor {anchor} did not compile");
            assert!(hits > 0, "{what}: trace compiled but native code never ran");
        }
        Expect::Refused => assert!(
            !compiled,
            "{what}: trace at anchor {anchor} compiled but should be refused"
        ),
        Expect::Parity => {}
    }
    traced
}

/// `slot0` counts 1..=n in a do-while closed by `JumpIfTrue(anchor)`; `body`
/// runs every iteration after the increment. Result: `GetSlot(1)`.
fn counted_loop(salt: i64, n: i64, body: Vec<A>) -> (Chunk, usize) {
    let mut items = vec![
        o(Op::LoadInt(0)),
        o(Op::SetSlot(0)),
        A::L("top"),
        o(Op::PreIncSlotVoid(0)),
    ];
    items.extend(body);
    items.extend([
        o(Op::GetSlot(0)),
        o(Op::LoadInt(n)),
        o(Op::NumLt),
        A::J(Op::JumpIfTrue, "top"),
        o(Op::GetSlot(1)),
    ]);
    let (chunk, l) = asm(salt, &[], items);
    (chunk, l["top"])
}

// ── PreDecSlot / PostIncSlot / PostDecSlot ──

/// Each op's pushed value (old for post-, new for pre-) is folded into slot 2,
/// so both the slot write and the pushed operand are scored.
#[test]
fn slot_inc_dec_ops_match_interpreter() {
    for (k, op) in [Op::PreDecSlot(1), Op::PostIncSlot(1), Op::PostDecSlot(1)]
        .into_iter()
        .enumerate()
    {
        let (chunk, anchor) = counted_loop(
            100 + k as i64,
            300,
            vec![
                o(op.clone()),
                o(Op::GetSlot(2)),
                o(Op::Add),
                o(Op::SetSlot(2)),
            ],
        );
        let seed = [Value::Int(0), Value::Int(7), Value::Int(0)];
        check(&format!("{op:?}"), &chunk, anchor, &seed, 4, Expect::Native);
    }
}

/// The slot walks across `i64::MAX` / `i64::MIN` inside the trace: native code
/// must wrap exactly as the interpreter's `wrapping_add` / `wrapping_sub`.
#[test]
fn slot_inc_dec_ops_wrap_on_overflow() {
    let cases = [
        (Op::PostIncSlot(1), i64::MAX - 100),
        (Op::PostDecSlot(1), i64::MIN + 100),
        (Op::PreDecSlot(1), i64::MIN + 100),
    ];
    for (k, (op, start)) in cases.into_iter().enumerate() {
        let (chunk, anchor) =
            counted_loop(110 + k as i64, 300, vec![o(op.clone()), o(Op::SetSlot(2))]);
        let seed = [Value::Int(0), Value::Int(start), Value::Int(0)];
        let traced = check(
            &format!("{op:?} from {start}"),
            &chunk,
            anchor,
            &seed,
            4,
            Expect::Native,
        );
        let wrapped = match op {
            Op::PostIncSlot(_) => start.wrapping_add(300),
            _ => start.wrapping_sub(300),
        };
        assert!(
            same(&traced.slots[1], &Value::Int(wrapped)),
            "{op:?}: slot1 = {:?}",
            traced.slots[1]
        );
    }
}

/// A slot that is `Float` at the anchor would become `Int` in the interpreter;
/// the anchor-kinded spill cannot carry that, so the compile is refused.
#[test]
fn slot_inc_dec_on_float_slot_is_refused() {
    for (k, op) in [Op::PreDecSlot(1), Op::PostIncSlot(1), Op::PostDecSlot(1)]
        .into_iter()
        .enumerate()
    {
        let (chunk, anchor) = counted_loop(
            120 + k as i64,
            300,
            vec![
                o(op.clone()),
                o(Op::SetSlot(2)),
                o(Op::LoadFloat(1.5)),
                o(Op::SetSlot(1)),
            ],
        );
        let seed = [Value::Int(0), Value::Float(1.5), Value::Int(0)];
        check(
            &format!("{op:?} on Float slot"),
            &chunk,
            anchor,
            &seed,
            4,
            Expect::Refused,
        );
    }
}

/// A `Bool` or string in the slot at every anchor fails the entry guard, so the
/// interpreter keeps the loop; a non-native slot the trace never names (slot 3)
/// survives a native run untouched.
#[test]
fn slot_inc_dec_with_bool_and_str_slots() {
    let (chunk, anchor) = counted_loop(
        130,
        300,
        vec![
            o(Op::PostIncSlot(1)),
            o(Op::SetSlot(2)),
            o(Op::LoadTrue),
            o(Op::SetSlot(1)),
        ],
    );
    let seed = [Value::Int(0), Value::Bool(true), Value::Int(0)];
    check(
        "PostIncSlot on Bool slot",
        &chunk,
        anchor,
        &seed,
        4,
        Expect::Parity,
    );

    let (chunk, anchor) = counted_loop(131, 300, vec![o(Op::PreDecSlot(1)), o(Op::SetSlot(2))]);
    let seed = [
        Value::Int(0),
        Value::Int(5),
        Value::Int(0),
        Value::str("keep"),
    ];
    let traced = check(
        "PreDecSlot beside a Str slot",
        &chunk,
        anchor,
        &seed,
        4,
        Expect::Native,
    );
    assert!(same(&traced.slots[3], &Value::str("keep")));

    let seed = [
        Value::Int(0),
        Value::Int(5),
        Value::Int(0),
        Value::Bool(false),
    ];
    let traced = check(
        "PreDecSlot beside a Bool slot",
        &chunk,
        anchor,
        &seed,
        4,
        Expect::Native,
    );
    assert!(same(&traced.slots[3], &Value::Bool(false)));
}

/// The pushed post-increment value is still on the abstract stack when a
/// branch flips direction mid-loop (at `i == 150`): the side exit must spill
/// the slot and materialize that stack entry for the interpreter.
#[test]
fn post_inc_value_survives_a_mid_loop_side_exit() {
    let (chunk, anchor) = counted_loop(
        140,
        300,
        vec![
            o(Op::PostIncSlot(1)),
            o(Op::GetSlot(0)),
            o(Op::LoadInt(150)),
            o(Op::NumLt),
            A::J(Op::JumpIfFalse, "skip"),
            o(Op::LoadInt(1000)),
            o(Op::Add),
            A::L("skip"),
            o(Op::GetSlot(2)),
            o(Op::Add),
            o(Op::SetSlot(2)),
        ],
    );
    let seed = [Value::Int(0), Value::Int(0), Value::Int(0)];
    check(
        "PostIncSlot across side exit",
        &chunk,
        anchor,
        &seed,
        4,
        Expect::Native,
    );
}

/// `MAX_TRACE_SLOT` (64) bounds every slot-bearing op: slot 64 refuses the
/// trace, slot 63 is admitted.
#[test]
fn slot_inc_dec_respects_max_trace_slot() {
    let (chunk, anchor) = counted_loop(150, 300, vec![o(Op::PostDecSlot(64)), o(Op::SetSlot(1))]);
    check("PostDecSlot(64)", &chunk, anchor, &[], 70, Expect::Refused);
    let (chunk, anchor) = counted_loop(151, 300, vec![o(Op::PostDecSlot(63)), o(Op::SetSlot(1))]);
    check("PostDecSlot(63)", &chunk, anchor, &[], 70, Expect::Native);
}

/// Inside an inlined callee the op works on the callee's own (fresh, zero)
/// slots and pushes into the callee's stack frame.
#[test]
fn slot_inc_dec_inside_inlined_callee() {
    let (chunk, l) = asm(
        160,
        &[("f", "f")],
        vec![
            A::J(Op::Jump, "main"),
            A::L("f"),
            o(Op::PreDecSlot(0)),  // -1 from a fresh frame
            o(Op::PostIncSlot(0)), // pushes -1, slot0 = 0
            o(Op::Add),
            o(Op::Add), // arg + (-1) + (-1)
            o(Op::ReturnValue),
            A::L("main"),
            o(Op::LoadInt(0)),
            o(Op::SetSlot(0)),
            A::L("top"),
            o(Op::PreIncSlotVoid(0)),
            o(Op::GetSlot(0)),
            o(Op::Call(0, 1)),
            o(Op::GetSlot(1)),
            o(Op::Add),
            o(Op::SetSlot(1)),
            o(Op::GetSlot(0)),
            o(Op::LoadInt(300)),
            o(Op::NumLt),
            A::J(Op::JumpIfTrue, "top"),
            o(Op::GetSlot(1)),
        ],
    );
    check(
        "slot ops in callee",
        &chunk,
        l["top"],
        &[],
        4,
        Expect::Native,
    );
}

// ── SlotLtIntJumpIfFalse ──

/// An if/else on `slot0 < 120` inside the loop body. The recorded direction
/// (then-branch) stops holding at 120, so the guard side-exits with a value
/// still on the abstract stack.
#[test]
fn slot_lt_branch_side_exits_mid_loop() {
    let (chunk, anchor) = counted_loop(
        200,
        300,
        vec![
            o(Op::LoadInt(5)), // on the stack across the guard
            A::J(|t| Op::SlotLtIntJumpIfFalse(0, 120, t), "else"),
            o(Op::PostIncSlot(1)),
            o(Op::Add),
            o(Op::SetSlot(2)),
            A::J(Op::Jump, "end"),
            A::L("else"),
            o(Op::PreDecSlot(1)),
            o(Op::Add),
            o(Op::SetSlot(2)),
            A::L("end"),
        ],
    );
    check(
        "SlotLtIntJumpIfFalse",
        &chunk,
        anchor,
        &[],
        4,
        Expect::Native,
    );
}

/// Limits at the `i32` edges against slots beyond them: the comparison is a
/// signed 64-bit compare of the slot against the sign-extended limit.
#[test]
fn slot_lt_branch_at_i32_limits() {
    let cases = [
        (i32::MAX, i64::from(i32::MAX) - 150),
        (i32::MIN, i64::from(i32::MIN) - 150),
        (0, -150),
    ];
    for (k, (limit, start)) in cases.into_iter().enumerate() {
        let (chunk, l) = asm(
            210 + k as i64,
            &[],
            vec![
                o(Op::LoadInt(0)),
                o(Op::SetSlot(0)),
                A::L("top"),
                o(Op::PreIncSlotVoid(0)),
                o(Op::PreIncSlotVoid(1)),
                A::J(move |t| Op::SlotLtIntJumpIfFalse(1, 0, t), "else"),
                o(Op::PreIncSlotVoid(2)),
                A::L("else"),
                o(Op::GetSlot(0)),
                o(Op::LoadInt(300)),
                o(Op::NumLt),
                A::J(Op::JumpIfTrue, "top"),
                o(Op::GetSlot(2)),
            ],
        );
        // `fn` pointers cannot capture: patch the limit in afterwards.
        let mut chunk = chunk;
        for op in &mut chunk.ops {
            if let Op::SlotLtIntJumpIfFalse(_, lim, _) = op {
                *lim = limit;
            }
        }
        let chunk = rehash(chunk);
        let seed = [Value::Int(0), Value::Int(start), Value::Int(0)];
        check(
            &format!("SlotLtIntJumpIfFalse limit {limit}"),
            &chunk,
            l["top"],
            &seed,
            4,
            Expect::Native,
        );
    }
}

/// Recompute `op_hash` after editing `chunk.ops` in place.
fn rehash(mut chunk: Chunk) -> Chunk {
    chunk.op_hash = chunk.compute_op_hash();
    chunk
}

/// The slot read by the guard is `Float` at the anchor: refused (int-only).
#[test]
fn slot_lt_branch_on_float_slot_is_refused() {
    let (chunk, anchor) = counted_loop(
        220,
        300,
        vec![
            A::J(|t| Op::SlotLtIntJumpIfFalse(1, 10, t), "else"),
            o(Op::PreIncSlotVoid(2)),
            A::L("else"),
        ],
    );
    let seed = [Value::Int(0), Value::Float(2.5), Value::Int(0)];
    check(
        "SlotLtIntJumpIfFalse on Float slot",
        &chunk,
        anchor,
        &seed,
        4,
        Expect::Refused,
    );
}

// ── SlotIncLtIntJumpBack as the closing back-edge ──

/// `do { body } while (++slot0 < n)` fused into one op: the VM anchors and
/// closes the trace on it.
fn fused_loop(salt: i64, n: i32, body: Vec<A>) -> (Chunk, usize) {
    let mut items = vec![o(Op::LoadInt(0)), o(Op::SetSlot(0)), A::L("top")];
    items.extend(body);
    // `fn` pointers cannot capture `n`: emit limit 0 and patch it below.
    items.extend([
        A::L("back"),
        A::J(|t| Op::SlotIncLtIntJumpBack(0, 0, t), "top"),
        o(Op::GetSlot(1)),
    ]);
    let (mut chunk, l) = asm(salt, &[], items);
    chunk.ops[l["back"]] = Op::SlotIncLtIntJumpBack(0, n, l["top"]);
    (rehash(chunk), l["top"])
}

#[test]
fn fused_back_edge_closes_a_trace() {
    let (chunk, anchor) = fused_loop(
        300,
        400,
        vec![
            o(Op::GetSlot(0)),
            o(Op::GetSlot(1)),
            o(Op::Add),
            o(Op::SetSlot(1)),
        ],
    );
    let traced = check(
        "SlotIncLtIntJumpBack close",
        &chunk,
        anchor,
        &[],
        4,
        Expect::Native,
    );
    assert!(same(&traced.slots[0], &Value::Int(400)));
    assert!(same(&traced.slots[1], &Value::Int((0..400).sum())));
}

/// The loop body carries a slot across `i64::MAX` while the fused back-edge
/// drives the loop.
#[test]
fn fused_back_edge_with_wrapping_body() {
    let (chunk, anchor) = fused_loop(301, 400, vec![o(Op::PostIncSlot(1)), o(Op::SetSlot(2))]);
    let seed = [Value::Int(0), Value::Int(i64::MAX - 200), Value::Int(0)];
    check(
        "SlotIncLtIntJumpBack + wrap",
        &chunk,
        anchor,
        &seed,
        4,
        Expect::Native,
    );
}

/// A branch inside a fused-back-edge loop flips at `i == 150`: the main trace
/// side-exits there repeatedly, which arms a side trace that also closes on
/// the fused back-edge (the side-trace form exits to the loop header).
#[test]
fn fused_back_edge_with_side_exits_and_side_trace() {
    let (chunk, anchor) = fused_loop(
        302,
        600,
        vec![
            A::J(|t| Op::SlotLtIntJumpIfFalse(0, 150, t), "else"),
            o(Op::PostIncSlot(1)),
            o(Op::Pop),
            A::J(Op::Jump, "end"),
            A::L("else"),
            o(Op::GetSlot(0)),
            o(Op::GetSlot(2)),
            o(Op::Add),
            o(Op::SetSlot(2)),
            A::L("end"),
        ],
    );
    check(
        "SlotIncLtIntJumpBack with side exits",
        &chunk,
        anchor,
        &[],
        4,
        Expect::Native,
    );
}

/// An inner fused loop (`do { slot2 += 1 } while (++slot3 < 2)`) whose start
/// alternates with the outer counter's parity, so the inner trip count — and
/// the direction of the inner, non-closing back-edge — changes every outer
/// iteration.
#[test]
fn inner_fused_back_edge_is_a_guarded_branch() {
    let (chunk, l) = asm(
        310,
        &[],
        vec![
            o(Op::LoadInt(0)),
            o(Op::SetSlot(0)),
            A::L("top"),
            o(Op::PreIncSlotVoid(0)),
            o(Op::GetSlot(0)),
            o(Op::LoadInt(1)),
            o(Op::BitAnd),
            o(Op::SetSlot(3)),
            A::L("inner"),
            o(Op::PreIncSlotVoid(2)),
            A::J(|t| Op::SlotIncLtIntJumpBack(3, 2, t), "inner"),
            o(Op::GetSlot(0)),
            o(Op::LoadInt(300)),
            o(Op::NumLt),
            A::J(Op::JumpIfTrue, "top"),
            o(Op::GetSlot(2)),
        ],
    );
    let traced = check(
        "inner SlotIncLtIntJumpBack",
        &chunk,
        l["top"],
        &[],
        4,
        Expect::Native,
    );
    // 150 even outer iterations run the inner body twice, 150 odd ones once.
    assert!(same(&traced.slots[2], &Value::Int(450)));
}

// ── AccumSumLoop ──

/// `slot1 += sum(slot2 .. 10)` every outer iteration, with `slot2` restarted
/// from the counter's low bits so the inner trip count varies (including
/// zero-trip starts at or past the limit).
#[test]
fn accum_sum_loop_matches_interpreter() {
    let (chunk, anchor) = counted_loop(
        400,
        300,
        vec![
            o(Op::GetSlot(0)),
            o(Op::LoadInt(15)),
            o(Op::BitAnd),
            o(Op::SetSlot(2)),
            o(Op::AccumSumLoop(1, 2, 10)),
        ],
    );
    check("AccumSumLoop", &chunk, anchor, &[], 4, Expect::Native);
}

/// The running sum crosses `i64::MAX`: native code wraps like `wrapping_add`.
#[test]
fn accum_sum_loop_wraps_on_overflow() {
    let (chunk, anchor) = counted_loop(
        401,
        300,
        vec![
            o(Op::LoadInt(0)),
            o(Op::SetSlot(2)),
            o(Op::AccumSumLoop(1, 2, 100)),
        ],
    );
    let seed = [Value::Int(0), Value::Int(i64::MAX - 1000), Value::Int(0)];
    let traced = check(
        "AccumSumLoop wrap",
        &chunk,
        anchor,
        &seed,
        4,
        Expect::Native,
    );
    let mut want = i64::MAX - 1000;
    for _ in 0..300 {
        want = want.wrapping_add(4950);
    }
    assert!(same(&traced.slots[1], &Value::Int(want)));
}

/// `sum_slot == i_slot`: the interpreter writes `sum` then `i`, so the slot
/// ends holding `i`. The trace must write back in the same order.
#[test]
fn accum_sum_loop_with_aliased_slots() {
    let (chunk, anchor) = counted_loop(
        402,
        300,
        vec![
            o(Op::GetSlot(0)),
            o(Op::LoadInt(7)),
            o(Op::BitAnd),
            o(Op::SetSlot(2)),
            o(Op::AccumSumLoop(2, 2, 20)),
            o(Op::GetSlot(2)),
            o(Op::GetSlot(1)),
            o(Op::Add),
            o(Op::SetSlot(1)),
        ],
    );
    check(
        "AccumSumLoop aliased",
        &chunk,
        anchor,
        &[],
        4,
        Expect::Native,
    );
}

/// A `Float` accumulator at the anchor is refused; a `Float` that the body
/// reassigns each iteration keeps the parity check honest.
#[test]
fn accum_sum_loop_on_float_slot_is_refused() {
    let (chunk, anchor) = counted_loop(
        403,
        300,
        vec![
            o(Op::LoadInt(0)),
            o(Op::SetSlot(2)),
            o(Op::AccumSumLoop(1, 2, 5)),
            o(Op::LoadFloat(0.5)),
            o(Op::SetSlot(1)),
        ],
    );
    let seed = [Value::Int(0), Value::Float(0.5)];
    check(
        "AccumSumLoop on Float slot",
        &chunk,
        anchor,
        &seed,
        4,
        Expect::Refused,
    );
}

// ── Still refused ──

/// `JumpIfTrueKeep` / `JumpIfFalseKeep` keep a (usually `Bool`) condition on
/// the stack, which the trace's Int/Float stack model cannot hand back exactly,
/// so the tier refuses them. The loop result must still match.
#[test]
fn keep_branches_stay_refused() {
    for (k, keep) in [Op::JumpIfTrueKeep as fn(usize) -> Op, Op::JumpIfFalseKeep]
        .into_iter()
        .enumerate()
    {
        let (chunk, anchor) = counted_loop(
            500 + k as i64,
            300,
            vec![
                o(Op::GetSlot(0)),
                o(Op::LoadInt(100)),
                o(Op::NumLt),
                A::J(keep, "k"),
                A::L("k"),
                o(Op::Pop),
                o(Op::PreIncSlotVoid(1)),
            ],
        );
        check("Keep branch", &chunk, anchor, &[], 4, Expect::Refused);
    }
}
