//! Interpreter-vs-block-JIT parity for the ops only a whole block chunk lowers:
//! `GetVar`/`SetVar`/`DeclareVar` (globals promoted from the globals buffer),
//! `JumpIfTrueKeep`/`JumpIfFalseKeep`, and `LogAnd`/`LogOr`.
//!
//! Every case runs through `VM::run` on three fresh VMs on one thread, each
//! run twice. With the default `block_threshold` of 1 the first VM is the
//! interpreter (cold) and the later ones run the compiled chunk (warm), so the
//! described outcome — result, rerun result, the whole global table and the
//! base frame — must be identical across all three. Each case also states
//! whether the block tier must compile it: globals that are not a native
//! `Int`/`Float` (or lie past the table) must be declined, and the
//! native-kind ones must not be, so a corpus cannot pass by never reaching the
//! tier.

#![cfg(feature = "jit")]

use std::sync::atomic::{AtomicI64, Ordering};

use fusevm::{Chunk, ChunkBuilder, JitCompiler, Op, VMResult, Value, VM};

/// Every `Value` kind a global can hold, labelled.
fn global_kinds() -> Vec<(&'static str, Value)> {
    vec![
        ("int0", Value::Int(0)),
        ("int7", Value::Int(7)),
        ("intmax", Value::Int(i64::MAX)),
        ("intmin", Value::Int(i64::MIN)),
        ("float0", Value::Float(0.0)),
        ("negzero", Value::Float(-0.0)),
        ("float2.5", Value::Float(2.5)),
        ("nan", Value::Float(f64::NAN)),
        ("inf", Value::Float(f64::INFINITY)),
        ("true", Value::Bool(true)),
        ("false", Value::Bool(false)),
        ("str7", Value::str("7")),
        ("strempty", Value::str("")),
        ("undef", Value::Undef),
        ("array", Value::array(vec![Value::Int(1)])),
    ]
}

fn is_native(v: &Value) -> bool {
    matches!(v, Value::Int(_) | Value::Float(_))
}

/// Unique `op_hash` per case: a `LoadInt(salt); Pop` prefix, with every jump
/// target rebased past it. Without it two cases with equal ops would share a
/// warm block-cache entry and the second would never see a cold run.
fn salted(ops: &[Op]) -> Chunk {
    static SALT: AtomicI64 = AtomicI64::new(0x5eed_0000);
    let salt = SALT.fetch_add(1, Ordering::Relaxed);
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(salt), 1);
    b.emit(Op::Pop, 1);
    for op in ops {
        let op = match op {
            Op::Jump(t) => Op::Jump(t + 2),
            Op::JumpIfTrue(t) => Op::JumpIfTrue(t + 2),
            Op::JumpIfFalse(t) => Op::JumpIfFalse(t + 2),
            Op::JumpIfTrueKeep(t) => Op::JumpIfTrueKeep(t + 2),
            Op::JumpIfFalseKeep(t) => Op::JumpIfFalseKeep(t + 2),
            other => other.clone(),
        };
        b.emit(op, 1);
    }
    b.build()
}

/// `{:?}` of a value, with floats spelled by their bits so `-0.0`/`0.0` and
/// distinct NaN payloads cannot compare equal by accident.
fn describe(v: &Value) -> String {
    match v {
        Value::Float(f) => format!("Float({f:?}/{:#x})", f.to_bits()),
        other => format!("{other:?}"),
    }
}

/// `VM::run` twice on one VM seeded with `globals` and base-frame `slots`.
fn run_described(chunk: &Chunk, globals: &[Value], slots: &[Value]) -> String {
    let mut vm = VM::new(chunk.clone());
    vm.globals = globals.to_vec();
    vm.frames.last_mut().unwrap().slots = slots.to_vec();
    vm.enable_tracing_jit();
    let mut run = || match vm.run() {
        VMResult::Ok(v) => describe(&v),
        VMResult::Halted => "Halted".to_string(),
        VMResult::Error(e) => format!("Error({e})"),
    };
    let (first, rerun) = (run(), run());
    let g: Vec<String> = vm.globals.iter().map(describe).collect();
    let f: Vec<String> = vm.frames[0].slots.iter().map(describe).collect();
    format!("{first} rerun={rerun} globals={g:?} slots={f:?}")
}

struct Case {
    label: String,
    chunk: Chunk,
    globals: Vec<Value>,
    slots: Vec<Value>,
    /// Whether the block tier must compile the chunk for these inputs.
    compiles: bool,
}

fn case(label: String, ops: &[Op], globals: Vec<Value>, compiles: bool) -> Case {
    Case {
        label,
        chunk: salted(ops),
        globals,
        slots: Vec::new(),
        compiles,
    }
}

/// Cold-vs-warm agreement plus the expected compile decision for every case.
fn assert_parity(what: &str, cases: Vec<Case>) {
    let mut diffs = Vec::new();
    for c in &cases {
        let answers: Vec<String> = (0..3)
            .map(|_| run_described(&c.chunk, &c.globals, &c.slots))
            .collect();
        if !answers.windows(2).all(|w| w[0] == w[1]) {
            diffs.push(format!(
                "{}: cold={} warm={}",
                c.label, answers[0], answers[2]
            ));
        }
        let compiled = JitCompiler::new().block_jit_is_compiled(&c.chunk);
        if compiled != c.compiles {
            diffs.push(format!(
                "{}: block compiled={compiled}, expected {} (answer {})",
                c.label, c.compiles, answers[0]
            ));
        }
    }
    assert!(
        cases.iter().any(|c| c.compiles),
        "{what}: no case is expected to compile, so agreement proves nothing"
    );
    assert!(
        diffs.is_empty(),
        "{what}: {} divergence(s):\n{}",
        diffs.len(),
        diffs.join("\n")
    );
}

#[test]
fn global_reads_agree_for_every_kind_and_leave_other_globals_untouched() {
    // g0 + g1, with g2 never referenced: whatever g2 holds must come back as is.
    let ops = [Op::GetVar(0), Op::GetVar(1), Op::Add];
    let mut cases = Vec::new();
    for (la, a) in global_kinds() {
        for (lb, b) in global_kinds() {
            for (lc, c) in [("int3", Value::Int(3)), ("str", Value::str("keep"))] {
                cases.push(case(
                    format!("{la}+{lb} (g2={lc})"),
                    &ops,
                    vec![a.clone(), b.clone(), c],
                    is_native(&a) && is_native(&b),
                ));
            }
        }
    }
    assert_parity("global read", cases);
}

#[test]
fn global_writes_agree_and_keep_their_kind() {
    let mut cases = Vec::new();
    for (l, g) in global_kinds() {
        // g0 = g0 + 1; g0 — an Int global stays Int, a Float global stays Float.
        cases.push(case(
            format!("{l}: g0 += 1"),
            &[
                Op::GetVar(0),
                Op::LoadInt(1),
                Op::Add,
                Op::SetVar(0),
                Op::GetVar(0),
            ],
            vec![g.clone(), Value::str("untouched")],
            is_native(&g),
        ));
        // A float store: only a Float global keeps its kind under it, so an
        // Int global must be declined (the interpreter turns it into Float).
        cases.push(case(
            format!("{l}: declare g0 = 2.5"),
            &[Op::LoadFloat(2.5), Op::DeclareVar(0), Op::GetVar(0)],
            vec![g.clone()],
            matches!(g, Value::Float(_)),
        ));
        // An int store into each kind, as a void chunk (`Halted`).
        cases.push(case(
            format!("{l}: g0 = 9 (void)"),
            &[Op::LoadInt(9), Op::SetVar(0)],
            vec![g.clone(), Value::Int(1)],
            matches!(g, Value::Int(_)),
        ));
    }
    assert_parity("global write", cases);
}

#[test]
fn global_loops_agree_cold_and_warm() {
    // while (g0 < 10) { g1 = g1 + g0; g0 = g0 + 1 }; g1
    let ops = [
        Op::GetVar(0),       // 0
        Op::LoadInt(10),     // 1
        Op::NumLt,           // 2
        Op::JumpIfFalse(14), // 3
        Op::GetVar(1),       // 4
        Op::GetVar(0),       // 5
        Op::Add,             // 6
        Op::SetVar(1),       // 7
        Op::GetVar(0),       // 8
        Op::LoadInt(1),      // 9
        Op::Add,             // 10
        Op::SetVar(0),       // 11
        Op::Jump(0),         // 12
        Op::Nop,             // 13
        Op::GetVar(1),       // 14
    ];
    let mut cases = Vec::new();
    for (l0, g0) in [("int0", Value::Int(0)), ("int-5", Value::Int(-5))] {
        for (l1, g1) in global_kinds() {
            cases.push(case(
                format!("i={l0} acc={l1}"),
                &ops,
                vec![g0.clone(), g1.clone(), Value::str("argv0")],
                // The accumulator must be an Int: `Float + Int` is Float, so a
                // Float accumulator stays Float and compiles too.
                is_native(&g1),
            ));
        }
    }
    // A Float counter: `g0 + 1` stays Float, compiled with Float compares.
    cases.push(case(
        "i=float0 acc=float0".to_string(),
        &ops,
        vec![Value::Float(0.0), Value::Float(0.0)],
        true,
    ));
    assert_parity("global loop", cases);
}

#[test]
fn globals_past_the_table_are_declined() {
    // The interpreter reads a missing global as Undef and grows the table on a
    // write; the block tier has no buffer entry for it and must decline.
    let cases = vec![
        case(
            "read past".to_string(),
            &[Op::GetVar(3), Op::LoadInt(1), Op::Add],
            vec![Value::Int(1)],
            false,
        ),
        case(
            "write past".to_string(),
            &[Op::LoadInt(4), Op::SetVar(3), Op::GetVar(0)],
            vec![Value::Int(1)],
            false,
        ),
        case(
            "in range".to_string(),
            &[Op::LoadInt(4), Op::SetVar(1), Op::GetVar(0)],
            vec![Value::Int(1), Value::Int(0)],
            true,
        ),
    ];
    assert_parity("global bounds", cases);
}

#[test]
fn global_chunks_are_declined_by_the_direct_entry_points() {
    // `try_run_block*` callers pass no globals buffer; a chunk that references
    // one must fall back rather than dereference a null pointer.
    let chunk = salted(&[Op::GetVar(0), Op::LoadInt(1), Op::Add]);
    let jit = JitCompiler::new();
    assert!(jit.is_block_eligible(&chunk));
    assert_eq!(jit.try_run_block_eager(&chunk, &mut []), None);
    assert_eq!(jit.try_run_block_eager(&chunk, &mut [5, 6]), None);
}

/// Whether the block tier can lower a truth operand: a non-finite float
/// literal is declined by `LoadFloat` codegen (the interpreter runs it).
fn lowerable(op: &Op) -> bool {
    !matches!(op, Op::LoadFloat(f) if !f.is_finite())
}

/// Whether two operands meet at a join with the same register kind. The block
/// tier carries live stack values across a join as typed block parameters, so
/// an `Int` arriving on one edge and a `Float` on the other is declined.
fn same_kind(a: &Op, b: &Op) -> bool {
    matches!(a, Op::LoadFloat(_)) == matches!(b, Op::LoadFloat(_))
}

/// Truthiness edges for a kept jump condition and logical operands.
fn truth_operands() -> Vec<(&'static str, Op)> {
    vec![
        ("0", Op::LoadInt(0)),
        ("1", Op::LoadInt(1)),
        ("-1", Op::LoadInt(-1)),
        ("min", Op::LoadInt(i64::MIN)),
        ("0.0", Op::LoadFloat(0.0)),
        ("-0.0", Op::LoadFloat(-0.0)),
        ("0.5", Op::LoadFloat(0.5)),
        ("nan", Op::LoadFloat(f64::NAN)),
    ]
}

#[test]
fn keep_jumps_agree_on_every_truthiness_edge() {
    let mut cases = Vec::new();
    for (la, a) in truth_operands() {
        for (lb, b) in truth_operands() {
            // a && b as a value: JumpIfFalseKeep(end); Pop; b; end:
            cases.push(case(
                format!("{la} &&v {lb}"),
                &[
                    a.clone(),
                    Op::JumpIfFalseKeep(4),
                    Op::Pop,
                    b.clone(),
                    Op::Nop,
                ],
                Vec::new(),
                lowerable(&a) && lowerable(&b) && same_kind(&a, &b),
            ));
            // a || b as a value: JumpIfTrueKeep(end); Pop; b; end:
            cases.push(case(
                format!("{la} ||v {lb}"),
                &[
                    a.clone(),
                    Op::JumpIfTrueKeep(4),
                    Op::Pop,
                    b.clone(),
                    Op::Nop,
                ],
                Vec::new(),
                lowerable(&a) && lowerable(&b) && same_kind(&a, &b),
            ));
        }
        // The kept value flowing into arithmetic after the join.
        cases.push(case(
            format!("({la} || 3) * 2"),
            &[
                a.clone(),
                Op::JumpIfTrueKeep(4),
                Op::Pop,
                Op::LoadInt(3),
                Op::LoadInt(2),
                Op::Mul,
            ],
            Vec::new(),
            lowerable(&a) && same_kind(&a, &Op::LoadInt(3)),
        ));
    }
    // A kept global: g0 || g1, the globals' own kinds included.
    for (la, a) in global_kinds() {
        for (lb, b) in [("int4", Value::Int(4)), ("float0", Value::Float(0.0))] {
            // Both edges of the join must carry the same register kind.
            let compiles =
                is_native(&a) && matches!(a, Value::Float(_)) == matches!(b, Value::Float(_));
            cases.push(case(
                format!("g0={la} ||v g1={lb}"),
                &[
                    Op::GetVar(0),
                    Op::JumpIfTrueKeep(4),
                    Op::Pop,
                    Op::GetVar(1),
                    Op::Nop,
                ],
                vec![a.clone(), b],
                compiles,
            ));
        }
    }
    assert_parity("keep jump", cases);
}

#[test]
fn keep_jump_leaving_a_boolean_is_declined() {
    // A comparison feeding a kept jump stays on the stack as a `Bool`, which
    // the block lattice cannot carry; the chunk must be declined.
    let cases = vec![
        case(
            "1 < 2 ||v 5".to_string(),
            &[
                Op::LoadInt(1),
                Op::LoadInt(2),
                Op::NumLt,
                Op::JumpIfTrueKeep(6),
                Op::Pop,
                Op::LoadInt(5),
                Op::Nop,
            ],
            Vec::new(),
            false,
        ),
        case(
            "7 ||v 5".to_string(),
            &[
                Op::LoadInt(7),
                Op::JumpIfTrueKeep(4),
                Op::Pop,
                Op::LoadInt(5),
                Op::Nop,
            ],
            Vec::new(),
            true,
        ),
    ];
    assert_parity("keep jump bool", cases);
}

#[test]
fn log_and_or_agree_as_result_and_as_branch_condition() {
    let mut cases = Vec::new();
    for (la, a) in truth_operands() {
        for (lb, b) in truth_operands() {
            for op in [Op::LogAnd, Op::LogOr] {
                // As the chunk result: a `Bool`, cold and warm.
                cases.push(case(
                    format!("{la} {op:?} {lb}"),
                    &[a.clone(), b.clone(), op.clone()],
                    Vec::new(),
                    lowerable(&a) && lowerable(&b),
                ));
                // As a branch condition: `if (a op b) 10 else 20`.
                cases.push(case(
                    format!("if {la} {op:?} {lb}"),
                    &[
                        a.clone(),
                        b.clone(),
                        op.clone(),
                        Op::JumpIfFalse(6),
                        Op::LoadInt(10),
                        Op::Jump(7),
                        Op::LoadInt(20),
                        Op::Nop,
                    ],
                    Vec::new(),
                    lowerable(&a) && lowerable(&b),
                ));
            }
        }
    }
    // Over globals of every kind: native ones compile, the rest are declined.
    for (la, a) in global_kinds() {
        cases.push(case(
            format!("g0={la} LogOr 0"),
            &[Op::GetVar(0), Op::LoadInt(0), Op::LogOr],
            vec![a.clone()],
            is_native(&a),
        ));
    }
    assert_parity("log and/or", cases);
}

#[test]
fn log_and_or_escaping_as_a_value_is_declined() {
    // `(1 && 2) + 1`: the interpreter coerces the `Bool` through `to_float`
    // (`Float(2.0)`); the block lattice would answer `Int(2)`. Declined.
    let cases = vec![
        case(
            "(1 LogAnd 2) + 1".to_string(),
            &[
                Op::LoadInt(1),
                Op::LoadInt(2),
                Op::LogAnd,
                Op::LoadInt(1),
                Op::Add,
            ],
            Vec::new(),
            false,
        ),
        case(
            "g0 = 1 LogOr 0".to_string(),
            &[Op::LoadInt(1), Op::LoadInt(0), Op::LogOr, Op::SetVar(0)],
            vec![Value::Int(5)],
            false,
        ),
        case(
            "1 LogOr 0; pop; 4".to_string(),
            &[
                Op::LoadInt(1),
                Op::LoadInt(0),
                Op::LogOr,
                Op::Pop,
                Op::LoadInt(4),
            ],
            Vec::new(),
            true,
        ),
    ];
    assert_parity("log and/or escape", cases);
}

/// Native code persisted by one thread and reloaded by another must give the
/// interpreter's answer for a globals chunk, and the kinds of the referenced
/// globals must key the blob: an Int-global blob is not reused for Float.
#[cfg(feature = "jit-disk-cache")]
#[test]
fn global_chunks_round_trip_through_the_disk_cache() {
    let dir = std::env::temp_dir().join(format!(
        "fusevm_tier_parity_block_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let jit = JitCompiler::new();
    jit.set_jit_cache_dir(Some(dir.clone()));
    let chunk = salted(&[
        Op::GetVar(0),
        Op::LoadInt(3),
        Op::Mul,
        Op::SetVar(1),
        Op::GetVar(1),
    ]);
    for globals in [
        vec![Value::Int(5), Value::Int(0)],
        vec![Value::Float(1.5), Value::Float(0.0)],
    ] {
        let expected = {
            let mut vm = VM::new(chunk.clone());
            vm.globals = globals.clone();
            let r = vm.run();
            format!("{r:?} {:?}", vm.globals)
        };
        // Thread A compiles and writes the blob, thread B loads it; each is a
        // fresh thread, so its in-memory block cache starts cold.
        for _ in 0..2 {
            let (chunk, seed) = (chunk.clone(), globals.clone());
            let got = std::thread::spawn(move || {
                let mut last = String::new();
                for _ in 0..2 {
                    let mut vm = VM::new(chunk.clone());
                    vm.globals = seed.clone();
                    vm.enable_tracing_jit();
                    let r = vm.run();
                    last = format!("{r:?} {:?}", vm.globals);
                }
                (last, JitCompiler::new().block_jit_is_compiled(&chunk))
            })
            .join()
            .unwrap();
            assert_eq!(got.0, expected, "globals {globals:?}");
            assert!(got.1, "globals {globals:?}: block tier did not compile");
        }
    }
    let prefix = format!("{:016x}.", chunk.op_hash);
    let blobs = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        // The cache directory is process-wide, so other tests in this binary
        // write to it too; count only this chunk's blobs.
        .filter(|n| n.starts_with(&prefix) && n.contains(".blk."))
        .count();
    assert_eq!(blobs, 2, "one block blob per global-kind signature");
    let _ = std::fs::remove_dir_all(&dir);
}
