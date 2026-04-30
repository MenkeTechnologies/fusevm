//! Differential fuzz harness — interpreter vs tracing JIT.
//!
//! Generates random valid bytecode (counter loops with random
//! arithmetic/bitwise/comparison body ops), runs each chunk through both
//! the pure interpreter and the tracing-JIT-enabled VM, and asserts the
//! two paths produce identical results. Catches latent bugs in the trace
//! recorder, eligibility checks, deopt path, and IR codegen that the
//! curated unit tests might miss.
//!
//! Gated behind `--features jit`. Without the feature flag, tracing JIT
//! is a no-op so the diff would be uninteresting.

#![cfg(feature = "jit")]

use fusevm::{ChunkBuilder, Op, TraceJitConfig, VMResult, Value, VM};

/// Linear-congruential RNG. Stable, no `rand` dep.
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_mul(0x2545_F491_4F6C_DD1D) | 1,
        }
    }
    fn next(&mut self) -> u64 {
        // Park-Miller-style multiplicative LCG.
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }
    fn range(&mut self, n: u64) -> u64 {
        if n == 0 {
            0
        } else {
            self.next() % n
        }
    }
}

/// Build a counter-loop chunk with `body_len` random body ops drawn from
/// a JIT-friendly subset (arithmetic, bitwise, comparison, slot ops).
/// All ops are stack-balanced — the body's net stack effect is zero, so
/// the closing branch always sees the loop-condition value at top.
///
/// Body ops obey this stack-machine invariant: track the abstract stack
/// depth during generation, only emit ops whose pop count ≤ current depth.
fn build_random_loop(rng: &mut Lcg, limit: i64, body_len: usize) -> fusevm::Chunk {
    let mut b = ChunkBuilder::new();

    // Init slot 0 to 0 (counter).
    b.emit(Op::LoadInt(0), 1);
    b.emit(Op::SetSlot(0), 1);
    // Init slot 1 to a random scratch value.
    b.emit(Op::LoadInt((rng.range(64) as i64) + 1), 1);
    b.emit(Op::SetSlot(1), 1);

    let anchor = b.current_pos();

    // Increment the counter as the first body op (guarantees forward
    // progress even if the random body is a no-op).
    b.emit(Op::PreIncSlotVoid(0), 1);

    // Track abstract stack depth during random body generation so we
    // never emit an op that would pop from an empty stack.
    let mut depth: usize = 0;
    for _ in 0..body_len {
        // Pick an op based on current stack depth.
        let op = match depth {
            0 => match rng.range(4) {
                0 => Op::LoadInt(rng.range(0xFFFF) as i64),
                1 => Op::GetSlot(0),
                2 => Op::GetSlot(1),
                _ => Op::Nop,
            },
            1 => match rng.range(6) {
                0 => Op::LoadInt(rng.range(0xFFFF) as i64),
                1 => Op::GetSlot(0),
                2 => Op::GetSlot(1),
                3 => Op::Pop,
                4 => Op::Dup,
                _ => Op::Nop,
            },
            _ => match rng.range(18) {
                // Exclude Mod / Div: native srem/sdiv on divisor=0 raises
                // a hardware trap (SIGILL on Apple Silicon) while the
                // interpreter returns 0. That's a known-divergent op for
                // a divisor of 0; fuzzing it gives false positives that
                // would actually be safety bugs not test bugs.
                0 => Op::Add,
                1 => Op::Sub,
                2 => Op::Mul,
                3 => Op::BitAnd,
                4 => Op::BitOr,
                5 => Op::BitXor,
                6 => Op::NumEq,
                7 => Op::NumNe,
                8 => Op::NumLt,
                9 => Op::NumGt,
                10 => Op::Pop,
                11 => Op::Dup,
                12 => Op::SetSlot(rng.range(2) as u16),
                13 => Op::Swap,
                14 => Op::LoadInt(rng.range(0xFFFF) as i64),
                15 => Op::GetSlot(0),
                16 => Op::GetSlot(1),
                _ => Op::AddAssignSlotVoid(rng.range(2) as u16, rng.range(2) as u16),
            },
        };

        // Update depth simulation.
        match &op {
            Op::LoadInt(_) | Op::GetSlot(_) | Op::Dup => depth += 1,
            Op::Pop => depth = depth.saturating_sub(1),
            Op::SetSlot(_) => depth = depth.saturating_sub(1),
            Op::Add
            | Op::Sub
            | Op::Mul
            | Op::BitAnd
            | Op::BitOr
            | Op::BitXor
            | Op::NumEq
            | Op::NumNe
            | Op::NumLt
            | Op::NumGt => {
                depth = depth.saturating_sub(1);
            }
            _ => {}
        }
        b.emit(op, 1);
    }

    // Pop everything the random body left on the stack so the closing
    // branch sees a balanced state.
    while depth > 0 {
        b.emit(Op::Pop, 1);
        depth -= 1;
    }

    // Closing comparison + backward branch.
    b.emit(Op::GetSlot(0), 1);
    b.emit(Op::LoadInt(limit), 1);
    b.emit(Op::NumLt, 1);
    let close = b.emit(Op::JumpIfTrue(0), 1);
    b.patch_jump(close, anchor);
    b.emit(Op::GetSlot(0), 1);
    b.build()
}

fn ensure_slots(vm: &mut VM, n: usize) {
    let frame = vm.frames.last_mut().unwrap();
    while frame.slots.len() < n {
        frame.slots.push(Value::Int(0));
    }
}

fn run_interp(chunk: &fusevm::Chunk) -> VMResult {
    let mut vm = VM::new(chunk.clone());
    ensure_slots(&mut vm, 2);
    vm.run()
}

fn run_trace_jit(chunk: &fusevm::Chunk) -> VMResult {
    let mut vm = VM::new(chunk.clone());
    vm.enable_tracing_jit();
    ensure_slots(&mut vm, 2);
    vm.run()
}

fn results_equivalent(a: &VMResult, b: &VMResult) -> bool {
    use VMResult::*;
    match (a, b) {
        (Ok(x), Ok(y)) => values_equivalent(x, y),
        // Either both halt or both error in the same way.
        (Halted, Halted) => true,
        (Error(_), Error(_)) => true,
        _ => false,
    }
}

fn values_equivalent(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Int(x), Value::Int(y)) => x == y,
        (Value::Float(x), Value::Float(y)) => {
            // NaN != NaN per IEEE; treat both-NaN as equivalent.
            (x.is_nan() && y.is_nan()) || x == y
        }
        // Cross-type compare: trace JIT promotes Int→Float in some paths;
        // accept if the numeric values match.
        (Value::Int(x), Value::Float(y)) | (Value::Float(y), Value::Int(x)) => (*x as f64) == *y,
        (Value::Bool(x), Value::Bool(y)) => x == y,
        (Value::Undef, Value::Undef) => true,
        _ => format!("{:?}", a) == format!("{:?}", b),
    }
}

#[test]
fn fuzz_random_loops_match_interpreter() {
    // Lower the trace threshold so traces compile within a few iterations
    // — keeps fuzz cases small and fast.
    let jit = fusevm::JitCompiler::new();
    let original_cfg = jit.get_config();
    jit.set_config(TraceJitConfig {
        trace_threshold: 5,
        ..original_cfg
    });

    let mut rng = Lcg::new(0xCAFE_BABE);
    let mut diffs: Vec<(u64, VMResult, VMResult)> = Vec::new();

    // 100 distinct random programs. Each runs ~100 iterations so the
    // trace has time to record + execute.
    for case in 0..100 {
        let seed_for_case = rng.next();
        let mut case_rng = Lcg::new(seed_for_case);
        let body_len = (case_rng.range(15) as usize) + 1;
        let limit = (case_rng.range(80) as i64) + 50; // 50..130 iterations
        let chunk = build_random_loop(&mut case_rng, limit, body_len);

        let interp_result = run_interp(&chunk);
        let jit_result = run_trace_jit(&chunk);

        if !results_equivalent(&interp_result, &jit_result) {
            diffs.push((seed_for_case, interp_result, jit_result));
        }

        // Sanity: if too many diffs accumulate, bail with detail.
        if diffs.len() >= 5 {
            break;
        }
        let _ = case;
    }

    jit.set_config(original_cfg);

    if !diffs.is_empty() {
        for (seed, interp, jit_r) in &diffs {
            eprintln!(
                "DIFF seed={:#x}\n  interp = {:?}\n  trace  = {:?}",
                seed, interp, jit_r
            );
        }
        panic!(
            "tracing JIT diverged from interpreter on {} fuzz case(s)",
            diffs.len()
        );
    }
}

#[test]
fn fuzz_repeat_runs_are_deterministic() {
    // Same seed → same output. Catches state leaks across `vm.run` calls.
    let mut rng = Lcg::new(0xFEED_FACE);
    let body_len = (rng.range(10) as usize) + 3;
    let limit = 80;
    let chunk = build_random_loop(&mut rng, limit, body_len);

    let r1 = run_trace_jit(&chunk);
    let r2 = run_trace_jit(&chunk);
    let r3 = run_trace_jit(&chunk);

    assert!(
        results_equivalent(&r1, &r2),
        "two consecutive trace runs of the same chunk must agree: {:?} vs {:?}",
        r1,
        r2
    );
    assert!(
        results_equivalent(&r2, &r3),
        "third run must also agree: {:?} vs {:?}",
        r2,
        r3
    );
}
