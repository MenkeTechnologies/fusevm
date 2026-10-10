//! Seeded differential fuzz: interpreter vs every other way to execute a chunk.
//!
//! The interpreter is the semantic spec. For each random structured program
//! this file requires that
//!
//! * the block tier and the tracing tier, driven through `VM::run` (cold and
//!   warm, several VMs sharing the thread's code caches), leave the same
//!   result, frame slots and globals;
//! * the block tier's direct entry points (`try_run_block*`) agree on result
//!   and slot buffer whenever they accept a chunk;
//! * the linear tier, the AOT tier and the persistent JIT cache (reloaded on a
//!   fresh thread) agree on the result;
//! * a bincode / JSON / rkyv round-trip of the chunk executes identically.
//!
//! Declining a chunk is always correct; answering differently never is. A
//! panic in the interpreter is reported as a divergence rather than aborting
//! the sweep, because the interpreter must not panic on a valid chunk.
//!
//! Every case is a pure function of `(seed)`. `FUSEVM_FUZZ_SEEDS` (count) and
//! `FUSEVM_FUZZ_SEED_BASE` widen or move the sweep for a longer local run.

use fusevm::{Chunk, ChunkBuilder, Op, VMResult, Value, VM};
use std::panic::{catch_unwind, AssertUnwindSafe};

/// Frame slots every program may touch: `0..DATA_SLOTS` are data, the rest are
/// loop counters (one per nesting level) that statements never write.
const SLOTS: usize = 8;
const DATA_SLOTS: u16 = 4;
const COUNTER_BASE: u16 = 4;

struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x2545_F491_4F6C_DD1D) ^ 0x9E37_79B9_7F4A_7C15)
    }
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 11
    }
    fn range(&mut self, n: u64) -> u64 {
        self.next() % n
    }
    fn chance(&mut self, num: u64, den: u64) -> bool {
        self.range(den) < num
    }
}

/// Which op families a generated program may use.
#[derive(Clone, Copy)]
struct Opts {
    /// `if`/`else`, conditional expressions and short-circuit keeps.
    branches: bool,
    /// Bounded counter loops, plain and fused.
    loops: bool,
    /// `GetVar`/`SetVar` on named globals.
    globals: bool,
    /// Strings, arrays and hashes: ops no native tier lowers.
    rich: bool,
    /// Host-free AWK ops, including the zero-divisor / negative-shift traps.
    awk: bool,
    /// Stay inside what the block and trace tiers admit: a boolean only
    /// where a jump consumes it, a constant divisor. Without this the
    /// generator is free to feed booleans to arithmetic and divide by a
    /// computed zero, which the native tiers decline wholesale.
    native: bool,
}

impl Opts {
    const STRAIGHT: Opts = Opts {
        branches: false,
        loops: false,
        globals: false,
        rich: false,
        awk: false,
        native: true,
    };
    #[cfg(feature = "jit")]
    const NATIVE: Opts = Opts {
        branches: true,
        loops: true,
        globals: false,
        rich: false,
        awk: true,
        native: true,
    };
    const FULL: Opts = Opts {
        branches: true,
        loops: true,
        globals: true,
        rich: true,
        awk: true,
        native: false,
    };
}

// Name pool indices. `GetVar`/`SetVar`/array/hash ops address `vm.globals` by
// the name index, so these double as global slots.
const G0: u16 = 0;
const G1: u16 = 1;
const ARR: u16 = 2;
const HSH: u16 = 3;

struct Program {
    chunk: Chunk,
    /// Initial values of frame slots `0..SLOTS`.
    init: Vec<Value>,
}

/// Operand domain of a native-profile program. The block tier gives each
/// value an int or float register and declines where branches disagree, so
/// most of its compiles come from programs that stay in one domain.
#[derive(Clone, Copy, PartialEq)]
enum Domain {
    Int,
    Float,
    Mixed,
}

struct Gen {
    rng: Lcg,
    o: Opts,
    dom: Domain,
    ops: Vec<Op>,
}

const UNARY: &[Op] = &[
    Op::Negate,
    Op::Inc,
    Op::Dec,
    Op::BitNot,
    Op::AbsInt,
    Op::TruncInt,
    Op::AbsFloat,
    Op::SqrtFloat,
    Op::SinFloat,
    Op::CosFloat,
    Op::ExpFloat,
    Op::LogFloat,
    Op::Log2Float,
    Op::Log10Float,
    Op::TruncFloat,
    Op::RoundFloat,
    Op::CeilFloat,
    Op::FloorFloat,
    Op::TanFloat,
    Op::AtanFloat,
    Op::TanhFloat,
    Op::AwkMkbool,
];

const UNARY_INT: &[Op] = &[
    Op::Negate,
    Op::Inc,
    Op::Dec,
    Op::BitNot,
    Op::AbsInt,
    Op::TruncInt,
];

const UNARY_FLOAT: &[Op] = &[
    Op::Negate,
    Op::AbsFloat,
    Op::SqrtFloat,
    Op::SinFloat,
    Op::CosFloat,
    Op::ExpFloat,
    Op::LogFloat,
    Op::Log2Float,
    Op::Log10Float,
    Op::RoundFloat,
    Op::CeilFloat,
    Op::FloorFloat,
    Op::TanFloat,
    Op::AtanFloat,
    Op::TanhFloat,
];

const BINARY_INT: &[Op] = &[
    Op::Add,
    Op::Sub,
    Op::Mul,
    Op::BitAnd,
    Op::BitOr,
    Op::BitXor,
    Op::Shl,
    Op::Shr,
    Op::Spaceship,
    Op::GcdInt,
    Op::LcmInt,
];

const BINARY_FLOAT: &[Op] = &[
    Op::Add,
    Op::Sub,
    Op::Mul,
    Op::Pow,
    Op::PowFloat,
    Op::Atan2Float,
];

/// Unary ops the block tier does not admit (host-dispatched or truthiness
/// coercions), usable wherever `native` is off.
const UNARY_INTERP: &[Op] = &[Op::RubyTruthy, Op::AwkInt, Op::LogNot];

/// Binary ops with a numeric (never boolean) result and no divisor.
const BINARY: &[Op] = &[
    Op::Add,
    Op::Sub,
    Op::Mul,
    Op::Pow,
    Op::BitAnd,
    Op::BitOr,
    Op::BitXor,
    Op::Shl,
    Op::Shr,
    Op::Spaceship,
    Op::GcdInt,
    Op::LcmInt,
    Op::PowFloat,
    Op::Atan2Float,
];

/// Comparisons: a boolean result.
const COMPARE: &[Op] = &[
    Op::NumEq,
    Op::NumNe,
    Op::NumLt,
    Op::NumGt,
    Op::NumLe,
    Op::NumGe,
];

/// Division and remainder; the block tier needs a constant divisor.
const DIVMOD: &[Op] = &[Op::Div, Op::Mod];

const AWK_UNARY: &[Op] = &[Op::AwkSqrtJit, Op::AwkLogJit, Op::AwkComplJit];
const AWK_BINARY: &[Op] = &[
    Op::AwkDivJit,
    Op::AwkModJit,
    Op::AwkLshiftJit,
    Op::AwkRshiftJit,
];
const RICH_BINARY: &[Op] = &[
    Op::Concat,
    Op::StrEq,
    Op::StrNe,
    Op::StrLt,
    Op::StrCmp,
    Op::LogAnd,
    Op::LogOr,
];

impl Gen {
    fn emit(&mut self, op: Op) -> usize {
        self.ops.push(op);
        self.ops.len() - 1
    }

    fn pos(&self) -> usize {
        self.ops.len()
    }

    fn pick<'a>(&mut self, pool: &'a [Op]) -> &'a Op {
        &pool[self.rng.range(pool.len() as u64) as usize]
    }

    fn int(&mut self) -> i64 {
        const EDGES: [i64; 17] = [
            0,
            1,
            -1,
            2,
            3,
            7,
            63,
            64,
            65,
            -2,
            i64::MIN,
            i64::MAX,
            i64::MIN + 1,
            (1 << 53) + 1,
            1 << 31,
            -(1 << 31),
            255,
        ];
        match self.rng.range(4) {
            0 => EDGES[self.rng.range(EDGES.len() as u64) as usize],
            1 => self.rng.next() as i64,
            _ => self.rng.range(41) as i64 - 20,
        }
    }

    fn float(&mut self) -> f64 {
        const EDGES: [f64; 12] = [
            0.0,
            -0.0,
            0.5,
            -2.5,
            1.0,
            3.0,
            1e18,
            1e30,
            -1e30,
            9.223372036854775808e18,
            1e-300,
            255.75,
        ];
        match self.rng.range(3) {
            0 => EDGES[self.rng.range(EDGES.len() as u64) as usize],
            _ => (self.rng.range(2001) as f64 - 1000.0) / 8.0,
        }
    }

    fn unary(&mut self) -> Op {
        match self.dom {
            Domain::Int => self.pick(UNARY_INT).clone(),
            Domain::Float => self.pick(UNARY_FLOAT).clone(),
            Domain::Mixed => self.pick(UNARY).clone(),
        }
    }

    fn binary(&mut self) -> Op {
        match self.dom {
            Domain::Int => self.pick(BINARY_INT).clone(),
            Domain::Float => self.pick(BINARY_FLOAT).clone(),
            Domain::Mixed => self.pick(BINARY).clone(),
        }
    }

    fn data_slot(&mut self) -> u16 {
        self.rng.range(DATA_SLOTS as u64) as u16
    }

    fn leaf(&mut self) {
        let mut pick = self.rng.range(8);
        if pick >= 6 && !(self.o.globals || self.o.rich) {
            pick = self.rng.range(6);
        }
        let op = match pick {
            0 | 1 if self.dom != Domain::Float => Op::LoadInt(self.int()),
            0..=2 if self.dom != Domain::Int => Op::LoadFloat(self.float()),
            0..=2 => Op::LoadInt(self.int()),
            // A bare boolean is only legal where a jump consumes it.
            3 if !self.o.native => {
                if self.rng.chance(1, 2) {
                    Op::LoadTrue
                } else {
                    Op::LoadFalse
                }
            }
            3 | 4 | 5 => {
                let hi = if self.dom == Domain::Float {
                    DATA_SLOTS
                } else {
                    SLOTS as u16
                };
                Op::GetSlot(self.rng.range(hi as u64) as u16)
            }
            _ => {
                if self.o.globals && (!self.o.rich || self.rng.chance(1, 2)) {
                    Op::GetVar(self.rng.range(2) as u16)
                } else {
                    match self.rng.range(3) {
                        0 => Op::LoadConst(self.rng.range(4) as u16),
                        1 => Op::ArrayLen(ARR),
                        _ => Op::LoadUndef,
                    }
                }
            }
        };
        self.emit(op);
    }

    /// `a / k` or `a % k`. Native tiers take only a constant divisor, and a
    /// zero one is kept in the mix: the interpreter answers it, the native
    /// tiers must decline it rather than trap.
    fn divmod(&mut self, depth: u32) {
        self.expr(depth);
        if self.o.native && !self.rng.chance(1, 8) {
            let as_float = match self.dom {
                Domain::Int => false,
                Domain::Float => true,
                Domain::Mixed => self.rng.chance(1, 2),
            };
            if as_float {
                let k = self.float();
                self.emit(Op::LoadFloat(if k == 0.0 { 3.0 } else { k }));
            } else {
                let k = self.int();
                self.emit(Op::LoadInt(if k.saturating_abs() < 2 { 3 } else { k }));
            }
        } else {
            self.expr(depth);
        }
        let op = self.pick(DIVMOD).clone();
        self.emit(op);
    }

    /// A branch condition: any value, or a comparison (which a jump consumes
    /// in place, so it stays native-eligible).
    fn cond(&mut self, depth: u32) {
        match self.rng.range(4) {
            0 | 1 => {
                self.expr(depth);
                self.expr(depth);
                let op = self.pick(COMPARE).clone();
                self.emit(op);
            }
            2 if self.o.native => {
                self.expr(depth);
                self.emit(Op::LogNot);
            }
            _ => self.expr(depth),
        }
    }

    /// Emit ops that push exactly one value.
    fn expr(&mut self, depth: u32) {
        if depth == 0 || self.rng.chance(1, 5) {
            return self.leaf();
        }
        match self.rng.range(15) {
            0 | 1 => {
                self.expr(depth - 1);
                let op = if self.o.awk && self.dom == Domain::Mixed && self.rng.chance(1, 6) {
                    self.pick(AWK_UNARY).clone()
                } else if self.o.rich && self.rng.chance(1, 10) {
                    Op::StringLen
                } else if !self.o.native && self.rng.chance(1, 6) {
                    self.pick(UNARY_INTERP).clone()
                } else {
                    self.unary()
                };
                self.emit(op);
            }
            2..=5 => {
                self.expr(depth - 1);
                self.expr(depth - 1);
                let op = if self.o.awk && self.dom == Domain::Mixed && self.rng.chance(1, 6) {
                    self.pick(AWK_BINARY).clone()
                } else if self.o.rich && self.rng.chance(1, 8) {
                    self.pick(RICH_BINARY).clone()
                } else if !self.o.native && self.rng.chance(1, 5) {
                    self.pick(COMPARE).clone()
                } else {
                    self.binary()
                };
                self.emit(op);
            }
            6 => self.divmod(depth - 1),
            7 => {
                // (a * b) % k and (a * b + c) % k, the fused modular forms.
                let wide = self.rng.chance(1, 2);
                for _ in 0..(3 + wide as u32) {
                    self.expr(depth - 1);
                }
                self.emit(if wide {
                    Op::MulAddModFloor
                } else {
                    Op::MulModFloor
                });
            }
            8 => {
                // Operand reuse: the value feeds both sides of a binary op.
                self.expr(depth - 1);
                self.emit(Op::Dup);
                let op = self.binary();
                self.emit(op);
            }
            9 => {
                self.expr(depth - 1);
                self.expr(depth - 1);
                self.emit(Op::Swap);
                let op = self.binary();
                self.emit(op);
            }
            10 if !self.o.native => {
                // [a b] -> [a b a b] -> three reductions.
                self.expr(depth - 1);
                self.expr(depth - 1);
                self.emit(Op::Dup2);
                for _ in 0..3 {
                    let op = self.binary();
                    self.emit(op);
                }
            }
            11 => {
                for _ in 0..3 {
                    self.expr(depth - 1);
                }
                self.emit(Op::Rot);
                for _ in 0..2 {
                    let op = self.binary();
                    self.emit(op);
                }
            }
            12 | 14 if self.o.branches => {
                // cond ? a : b
                self.cond(depth - 1);
                let to_else = self.emit(Op::JumpIfFalse(0));
                self.expr(depth - 1);
                let to_end = self.emit(Op::Jump(0));
                self.ops[to_else] = Op::JumpIfFalse(self.pos());
                self.expr(depth - 1);
                self.ops[to_end] = Op::Jump(self.pos());
            }
            13 if self.o.branches && !self.o.native => {
                // a && b / a || b with the value kept.
                self.expr(depth - 1);
                let keep = if self.rng.chance(1, 2) {
                    self.emit(Op::JumpIfFalseKeep(0))
                } else {
                    self.emit(Op::JumpIfTrueKeep(0))
                };
                self.emit(Op::Pop);
                self.expr(depth - 1);
                let end = self.pos();
                self.ops[keep] = match self.ops[keep] {
                    Op::JumpIfFalseKeep(_) => Op::JumpIfFalseKeep(end),
                    _ => Op::JumpIfTrueKeep(end),
                };
            }
            _ => {
                self.expr(depth - 1);
                let op = self.unary();
                self.emit(op);
            }
        }
    }

    fn stmts(&mut self, n: u64, depth: u32, loops: u16) {
        for _ in 0..n {
            self.stmt(depth, loops);
        }
    }

    /// Emit ops with a net-zero stack effect.
    fn stmt(&mut self, depth: u32, loops: u16) {
        match self.rng.range(14) {
            0..=3 => {
                self.expr(depth);
                let s = self.data_slot();
                self.emit(Op::SetSlot(s));
            }
            4 => {
                self.expr(depth);
                self.emit(Op::Pop);
            }
            5 if self.dom != Domain::Float => {
                let s = self.data_slot();
                self.emit(Op::PreIncSlotVoid(s));
            }
            6 if self.dom != Domain::Float => {
                let (a, b) = (self.data_slot(), self.data_slot());
                self.emit(Op::AddAssignSlotVoid(a, b));
            }
            7 if self.dom != Domain::Float => {
                let s = self.data_slot();
                let op = match self.rng.range(4) {
                    0 => Op::PreIncSlot(s),
                    1 => Op::PreDecSlot(s),
                    2 => Op::PostIncSlot(s),
                    _ => Op::PostDecSlot(s),
                };
                self.emit(op);
                self.emit(Op::Pop);
            }
            8 if self.o.globals => {
                self.expr(depth);
                let g = self.rng.range(2) as u16;
                self.emit(Op::SetVar(g));
            }
            9 if self.o.rich => {
                self.expr(depth);
                self.emit(Op::ArrayPush(ARR));
            }
            10 if self.o.rich => {
                self.expr(depth);
                let k = self.rng.range(4) as u16;
                self.emit(Op::LoadConst(k));
                self.emit(Op::HashSet(HSH));
            }
            11 if self.o.branches => {
                self.cond(depth);
                let cond = if self.rng.chance(1, 2) {
                    self.emit(Op::JumpIfFalse(0))
                } else {
                    self.emit(Op::JumpIfTrue(0))
                };
                let n = 1 + self.rng.range(2);
                self.stmts(n, depth, loops);
                let has_else = self.rng.chance(1, 2);
                let mut to_end = None;
                if has_else {
                    to_end = Some(self.emit(Op::Jump(0)));
                }
                let else_at = self.pos();
                self.ops[cond] = match self.ops[cond] {
                    Op::JumpIfFalse(_) => Op::JumpIfFalse(else_at),
                    _ => Op::JumpIfTrue(else_at),
                };
                if let Some(j) = to_end {
                    let n = 1 + self.rng.range(2);
                    self.stmts(n, depth, loops);
                    self.ops[j] = Op::Jump(self.pos());
                }
            }
            12 | 13 if self.o.loops && loops < 3 => self.counter_loop(depth, loops),
            _ => {
                self.expr(depth);
                let s = self.data_slot();
                self.emit(Op::SetSlot(s));
            }
        }
    }

    fn counter_loop(&mut self, depth: u32, loops: u16) {
        let c = COUNTER_BASE + loops;
        let limit = 1 + self.rng.range(if loops == 0 { 40 } else { 6 }) as i32;
        let body = 1 + self.rng.range(3);
        self.emit(Op::LoadInt(0));
        self.emit(Op::SetSlot(c));
        match self.rng.range(5) {
            0 => {
                // while (c < limit) { body; ++c }
                let top = self.pos();
                self.emit(Op::GetSlot(c));
                self.emit(Op::LoadInt(limit as i64));
                self.emit(Op::NumLt);
                let exit = self.emit(Op::JumpIfFalse(0));
                self.stmts(body, depth, loops + 1);
                self.emit(Op::PreIncSlotVoid(c));
                self.emit(Op::Jump(top));
                self.ops[exit] = Op::JumpIfFalse(self.pos());
            }
            1 => {
                // The same loop through the fused compare-and-exit.
                let top = self.pos();
                let exit = self.emit(Op::SlotLtIntJumpIfFalse(c, limit, 0));
                self.stmts(body, depth, loops + 1);
                self.emit(Op::PreIncSlotVoid(c));
                self.emit(Op::Jump(top));
                self.ops[exit] = Op::SlotLtIntJumpIfFalse(c, limit, self.pos());
            }
            2 => {
                // do { body } while (++c < limit)
                let top = self.pos();
                self.stmts(body, depth, loops + 1);
                self.emit(Op::SlotIncLtIntJumpBack(c, limit, top));
            }
            3 => {
                // do { body } while (c++ ... ) through the unfused tail.
                let top = self.pos();
                self.stmts(body, depth, loops + 1);
                self.emit(Op::PreIncSlot(c));
                self.emit(Op::LoadInt(limit as i64));
                self.emit(Op::NumLt);
                self.emit(Op::JumpIfTrue(top));
            }
            _ => {
                let sum = self.data_slot();
                self.emit(Op::AccumSumLoop(sum, c, limit));
            }
        }
    }
}

fn generate(seed: u64, o: Opts) -> Program {
    let dom = match (o.native, seed % 4) {
        (true, 0 | 1) => Domain::Int,
        (true, 2) => Domain::Float,
        _ => Domain::Mixed,
    };
    let mut g = Gen {
        rng: Lcg::new(seed),
        o,
        dom,
        ops: Vec::new(),
    };
    if o.globals {
        g.emit(Op::LoadInt(0));
        g.emit(Op::SetVar(G0));
        g.emit(Op::LoadInt(5));
        g.emit(Op::SetVar(G1));
    }
    if o.rich {
        g.emit(Op::DeclareArray(ARR));
        g.emit(Op::DeclareHash(HSH));
    }
    // A straight-line program is one expression: statements write slots, and
    // the linear tier reads its slots from an immutable slice.
    let straight = !o.branches && !o.loops;
    if !straight {
        let n = 1 + g.rng.range(6);
        g.stmts(n, 3, 0);
    }
    if g.rng.chance(1, 4) {
        g.cond(2);
    } else {
        g.expr(3);
    }

    let mut b = ChunkBuilder::new();
    for name in ["g0", "g1", "arr", "hsh"] {
        b.add_name(name);
    }
    for c in ["", "7", "abc", "3.5x"] {
        b.add_constant(Value::str(c));
    }
    // Block and trace lowerings of integer arithmetic switch on this flag; a
    // chunk that sets it must still answer what the flagless interpreter does.
    if g.rng.chance(1, 4) {
        b.set_int_overflow_deopt(true);
    }
    for op in &g.ops {
        b.emit(op.clone(), 1);
    }

    let mut init = Vec::new();
    for i in 0..SLOTS {
        init.push(if i >= DATA_SLOTS as usize {
            Value::Int(0)
        } else if dom == Domain::Float || (dom == Domain::Mixed && g.rng.chance(1, 3)) {
            Value::Float(g.float())
        } else {
            Value::Int(g.int())
        });
    }
    Program {
        chunk: b.build(),
        init,
    }
}

// ── Observation ──────────────────────────────────────────────────────────

/// Variant-strict rendering: `Int(1)`, `Bool(true)` and `Float(1.0)` differ,
/// floats compare on raw bits (a NaN is one NaN), hash order is sorted.
fn show(v: &Value) -> String {
    match v {
        Value::Float(f) if f.is_nan() => "Float(NaN)".to_string(),
        Value::Float(f) => format!("Float({:#x})", f.to_bits()),
        Value::Array(a) => format!("[{}]", a.iter().map(show).collect::<Vec<_>>().join(",")),
        Value::Hash(h) => {
            let mut kv: Vec<_> = h
                .iter()
                .map(|(k, v)| format!("{k:?}:{}", show(v)))
                .collect();
            kv.sort();
            format!("{{{}}}", kv.join(","))
        }
        other => format!("{other:?}"),
    }
}

fn show_result(r: &VMResult) -> String {
    match r {
        VMResult::Ok(v) => format!("Ok({})", show(v)),
        VMResult::Halted => "Halted".to_string(),
        VMResult::Error(e) => format!("Error({e})"),
    }
}

/// Result, then (unless the run errored, where partial state is unspecified)
/// the base frame's slots and the globals.
fn observe(vm: &VM, r: &VMResult) -> String {
    let mut s = show_result(r);
    if !matches!(r, VMResult::Error(_)) {
        let slots: Vec<String> = vm.frames[0].slots.iter().take(SLOTS).map(show).collect();
        let globals: Vec<String> = vm.globals.iter().map(show).collect();
        s.push_str(&format!(" slots={slots:?} globals={globals:?}"));
    }
    s
}

fn guarded(f: impl FnOnce() -> String) -> String {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(s) => s,
        Err(p) => {
            let msg = p
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_default();
            format!("PANIC({msg})")
        }
    }
}

fn run_interp(p: &Program) -> String {
    guarded(|| {
        let mut vm = VM::new(p.chunk.clone());
        vm.frames[0].slots = p.init.clone();
        let r = vm.run();
        observe(&vm, &r)
    })
}

#[cfg(feature = "jit")]
fn run_jit_vm(p: &Program) -> String {
    guarded(|| {
        let mut vm = VM::new(p.chunk.clone());
        vm.enable_tracing_jit();
        vm.frames[0].slots = p.init.clone();
        let r = vm.run();
        observe(&vm, &r)
    })
}

// ── Sweep plumbing ───────────────────────────────────────────────────────

fn env_u64(key: &str, default: u64) -> u64 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Run `check(seed)` over the sweep; each `Some(msg)` is one divergence.
fn sweep(what: &str, default_seeds: u64, o: Opts, check: impl Fn(&Program) -> Option<String>) {
    let n = env_u64("FUSEVM_FUZZ_SEEDS", default_seeds);
    let base = env_u64("FUSEVM_FUZZ_SEED_BASE", 0);
    let mut failures = Vec::new();
    for seed in base..base + n {
        let p = generate(seed, o);
        if let Some(msg) = check(&p) {
            failures.push(format!(
                "seed {seed}: {msg}\n  init={:?}\n  ops={:?}",
                p.init.iter().map(show).collect::<Vec<_>>(),
                p.chunk.ops
            ));
            if failures.len() >= 3 {
                break;
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{what}: {} divergence(s)\n{}",
        failures.len(),
        failures.join("\n")
    );
}

fn diff(label: &str, expected: &str, got: &str) -> Option<String> {
    (expected != got).then(|| format!("{label}\n  interp = {expected}\n  {label:<6} = {got}"))
}

/// The sweep must exercise more than error paths and constants.
fn assert_variety(o: Opts, min_ok_pct: u64) {
    let n = 200;
    let ok = (0..n)
        .filter(|&s| run_interp(&generate(s, o)).starts_with("Ok("))
        .count() as u64;
    assert!(
        ok * 100 >= n * min_ok_pct,
        "only {ok}/{n} generated programs ran to a value; the generator is mostly producing errors"
    );
}

// ── Always-on tests (no JIT feature) ─────────────────────────────────────

#[test]
fn generator_programs_run_to_values() {
    assert_variety(Opts::FULL, 60);
    assert_variety(Opts::STRAIGHT, 60);
}

#[test]
fn interpreter_runs_are_deterministic() {
    sweep("determinism", 150, Opts::FULL, |p| {
        diff("rerun", &run_interp(p), &run_interp(p))
    });
}

#[test]
fn bincode_round_trip_executes_identically() {
    sweep("bincode round-trip", 150, Opts::FULL, |p| {
        let bytes = bincode::serialize(&p.chunk).unwrap();
        let back: Chunk = bincode::deserialize(&bytes).unwrap();
        let q = Program {
            chunk: back,
            init: p.init.clone(),
        };
        diff("bincode", &run_interp(p), &run_interp(&q))
    });
}

#[test]
fn json_round_trip_executes_identically() {
    sweep("json round-trip", 100, Opts::FULL, |p| {
        let text = serde_json::to_string(&p.chunk).unwrap();
        let back: Chunk = serde_json::from_str(&text).unwrap();
        let q = Program {
            chunk: back,
            init: p.init.clone(),
        };
        diff("json", &run_interp(p), &run_interp(&q))
    });
}

#[cfg(feature = "rkyv-archive")]
#[test]
fn rkyv_round_trip_executes_identically() {
    use rkyv::Deserialize;
    sweep("rkyv round-trip", 150, Opts::FULL, |p| {
        let bytes = rkyv::to_bytes::<_, 4096>(&p.chunk).unwrap();
        let archived = rkyv::check_archived_root::<Chunk>(&bytes[..]).unwrap();
        let back: Chunk = archived
            .deserialize(&mut rkyv::de::deserializers::SharedDeserializeMap::new())
            .unwrap();
        let q = Program {
            chunk: back,
            init: p.init.clone(),
        };
        diff("rkyv", &run_interp(p), &run_interp(&q))
    });
}

// ── JIT tiers ────────────────────────────────────────────────────────────

#[cfg(feature = "jit")]
mod jit {
    use super::*;
    use fusevm::{BlockNum, JitCompiler, SlotKind, TraceJitConfig};
    use std::cell::Cell;
    use std::sync::{Mutex, MutexGuard, OnceLock};

    /// Holds the lock that serializes the JIT tests of this file (the on-disk
    /// cache directory is process-global) and points that cache at a private
    /// directory, so a sweep neither reads blobs a previous build of the
    /// compiler persisted under `~/.cache` nor leaves its own there.
    struct Env {
        _lock: MutexGuard<'static, ()>,
        #[cfg(feature = "jit-disk-cache")]
        dir: std::path::PathBuf,
    }

    impl Env {
        fn new(block_threshold: u32, trace_threshold: u32) -> Env {
            static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
            let lock = LOCK
                .get_or_init(|| Mutex::new(()))
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            let jit = JitCompiler::new();
            jit.set_config(TraceJitConfig {
                block_threshold,
                trace_threshold,
                ..jit.get_config()
            });
            #[cfg(feature = "jit-disk-cache")]
            let dir = {
                let dir = std::env::temp_dir().join(format!(
                    "fusevm_tier_fuzz_{}_{:?}",
                    std::process::id(),
                    std::thread::current().id()
                ));
                let _ = std::fs::remove_dir_all(&dir);
                jit.set_jit_cache_dir(Some(dir.clone()));
                dir
            };
            Env {
                _lock: lock,
                #[cfg(feature = "jit-disk-cache")]
                dir,
            }
        }
    }

    impl Drop for Env {
        fn drop(&mut self) {
            #[cfg(feature = "jit-disk-cache")]
            {
                JitCompiler::new().set_jit_cache_dir(None);
                let _ = std::fs::remove_dir_all(&self.dir);
            }
        }
    }

    /// A tier that accepts nothing makes agreement vacuous. Require that at
    /// least `pct` percent of the sweep reached native code.
    fn assert_reached(what: &str, reached: u64, n: u64, pct: u64) {
        assert!(
            reached * 100 >= n * pct,
            "{what}: only {reached}/{n} programs reached native code; agreement proves little"
        );
    }

    fn sweep_len(default_seeds: u64) -> u64 {
        env_u64("FUSEVM_FUZZ_SEEDS", default_seeds)
    }

    /// Block tier through `VM::run`: compile on first use, then warm reruns on
    /// fresh VMs. Every run must match the interpreter, cold or warm.
    #[test]
    fn block_tier_via_vm_matches_interpreter() {
        let _env = Env::new(0, 1_000_000);
        let jit = JitCompiler::new();
        let compiled = Cell::new(0u64);
        sweep("block via VM::run", 250, Opts::NATIVE, |p| {
            let expected = run_interp(p);
            let d = (0..3).find_map(|i| diff(&format!("run{i}"), &expected, &run_jit_vm(p)));
            compiled.set(compiled.get() + jit.block_jit_is_compiled(&p.chunk) as u64);
            d
        });
        assert_reached("block via VM::run", compiled.get(), sweep_len(250), 10);
    }

    /// Tracing tier: the block tier is held off so loops run in the
    /// interpreter, record, compile and take over. Reruns reuse the cached
    /// traces.
    #[test]
    fn trace_tier_via_vm_matches_interpreter() {
        let _env = Env::new(1_000_000, 2);
        let before = fusevm::jit::stats();
        sweep("trace via VM::run", 250, Opts::NATIVE, |p| {
            let expected = run_interp(p);
            (0..3).find_map(|i| diff(&format!("run{i}"), &expected, &run_jit_vm(p)))
        });
        let after = fusevm::jit::stats();
        assert!(
            after.trace_compiles > before.trace_compiles && after.trace_hits > before.trace_hits,
            "no trace was compiled and entered; agreement proves nothing"
        );
    }

    /// Programs the native tiers mostly decline (strings, arrays, hashes,
    /// globals): declining must fall back cleanly, and a native loop with a
    /// non-native statement in it must not corrupt state.
    #[test]
    fn mixed_programs_fall_back_cleanly() {
        let _env = Env::new(0, 2);
        sweep("mixed via VM::run", 200, Opts::FULL, |p| {
            let expected = run_interp(p);
            (0..3).find_map(|i| diff(&format!("run{i}"), &expected, &run_jit_vm(p)))
        });
    }

    fn kinds_of(init: &[Value]) -> Vec<SlotKind> {
        init.iter()
            .map(|v| match v {
                Value::Float(_) => SlotKind::Float,
                _ => SlotKind::Int,
            })
            .collect()
    }

    fn raw_slots(init: &[Value]) -> Vec<i64> {
        init.iter()
            .map(|v| match v {
                Value::Float(f) => f.to_bits() as i64,
                other => other.to_int(),
            })
            .collect()
    }

    fn decode_slots(raw: &[i64], kinds: &[SlotKind]) -> Vec<String> {
        raw.iter()
            .zip(kinds)
            .map(|(&r, k)| match k {
                SlotKind::Int => show(&Value::Int(r)),
                SlotKind::Float => show(&Value::Float(f64::from_bits(r as u64))),
            })
            .collect()
    }

    /// What the interpreter leaves for a direct block call: the result value
    /// and the slots, or `None` when it errored (state is then unspecified).
    fn expected_block(p: &Program) -> Option<(Value, Vec<String>)> {
        let mut vm = VM::new(p.chunk.clone());
        vm.frames[0].slots = p.init.clone();
        match catch_unwind(AssertUnwindSafe(|| vm.run())) {
            Ok(VMResult::Ok(v)) => {
                let s = vm.frames[0].slots.iter().take(SLOTS).map(show).collect();
                Some((v, s))
            }
            _ => None,
        }
    }

    fn block_num_value(n: BlockNum) -> Value {
        match n {
            BlockNum::Int(i) => Value::Int(i),
            BlockNum::Float(f) => Value::Float(f),
            BlockNum::Bool(b) => Value::Bool(b),
        }
    }

    /// The `i64` a plain `try_run_block*` hands back for `v`: floats truncate,
    /// booleans are 0/1.
    fn as_plain_i64(v: &Value) -> i64 {
        match v {
            Value::Float(f) => *f as i64,
            Value::Bool(b) => *b as i64,
            other => other.to_int(),
        }
    }

    /// One block-tier answer, or why there is none.
    enum Block {
        Declined,
        Trap,
        Panic(String),
        Answer(String),
    }

    fn block_call(
        raw: &mut [i64],
        kinds: &[SlotKind],
        call: impl FnOnce(&JitCompiler, &mut [i64]) -> Option<String>,
    ) -> Block {
        let jit = JitCompiler::new();
        match catch_unwind(AssertUnwindSafe(|| {
            let r = call(&jit, raw);
            (r, jit.take_awk_div_trap())
        })) {
            Err(_) => Block::Panic("panic".into()),
            Ok((_, true)) => Block::Trap,
            Ok((None, false)) => Block::Declined,
            Ok((Some(s), false)) => {
                Block::Answer(format!("{s} slots={:?}", decode_slots(raw, kinds)))
            }
        }
    }

    /// Compare one accepted block answer against the interpreter's.
    fn check_block(
        label: &str,
        got: Block,
        expected: &Option<(Value, Vec<String>)>,
        plain: bool,
    ) -> Option<String> {
        match (got, expected) {
            (Block::Declined, _) => None,
            (Block::Panic(m), _) => Some(format!("{label}: {m}")),
            // A trap is the host-visible zero-divisor / bit-op fatal; the
            // interpreter reports it as an error.
            (Block::Trap, None) => None,
            (Block::Trap, Some(_)) => {
                Some(format!("{label}: trapped where the interpreter answered"))
            }
            (Block::Answer(a), None) => {
                Some(format!("{label}: interpreter errored, block answered {a}"))
            }
            (Block::Answer(a), Some((v, slots))) => {
                let want = if plain {
                    format!("Some({}) slots={slots:?}", as_plain_i64(v))
                } else {
                    format!("{} slots={slots:?}", show(v))
                };
                diff(label, &want, &a)
            }
        }
    }

    /// Chunks the direct entry points accept are compared on the result and on
    /// the slot buffer they leave behind.
    #[test]
    fn direct_block_entry_points_match_interpreter() {
        let _env = Env::new(1, 1_000_000);
        let accepted = Cell::new(0u64);
        sweep("direct block API", 250, Opts::NATIVE, |p| {
            let expected = expected_block(p);
            let kinds = kinds_of(&p.init);

            // Eager and typed: the result keeps its variant.
            let mut raw = raw_slots(&p.init);
            let got = block_call(&mut raw, &kinds, |jit, raw| {
                jit.try_run_block_eager_typed_kinded(&p.chunk, raw, &kinds)
                    .map(|n| show(&block_num_value(n)))
            });
            accepted.set(accepted.get() + matches!(got, Block::Answer(_)) as u64);
            if let Some(d) = check_block("eager", got, &expected, false) {
                return Some(d);
            }

            // Tiered, truncating to i64. Repeated so both the warm-up fallback
            // and the compiled call are seen; a warm-up call answers `None`.
            for call in 0..3 {
                let mut raw = raw_slots(&p.init);
                let got = block_call(&mut raw, &kinds, |jit, raw| {
                    jit.try_run_block_kinded(&p.chunk, raw, &kinds)
                        .map(|n| format!("Some({n})"))
                });
                if let Some(d) = check_block(&format!("kinded#{call}"), got, &expected, true) {
                    return Some(d);
                }
            }

            // The kind-less entry point reads every slot as an int.
            if kinds.iter().all(|k| *k == SlotKind::Int) {
                for call in 0..3 {
                    let mut raw = raw_slots(&p.init);
                    let got = block_call(&mut raw, &kinds, |jit, raw| {
                        jit.try_run_block(&p.chunk, raw)
                            .map(|n| format!("Some({n})"))
                    });
                    if let Some(d) = check_block(&format!("plain#{call}"), got, &expected, true) {
                        return Some(d);
                    }
                }
            }
            None
        });
        assert_reached("direct block API", accepted.get(), sweep_len(250), 10);
    }

    #[test]
    fn linear_tier_matches_interpreter() {
        let _env = Env::new(1, 1_000_000);
        let accepted = Cell::new(0u64);
        sweep("linear tier", 400, Opts::STRAIGHT, |p| {
            let jit = JitCompiler::new();
            // The linear tier's slot slice holds ints only, so seed the
            // interpreter the same way.
            let q = Program {
                chunk: p.chunk.clone(),
                init: p.init.iter().map(|v| Value::Int(v.to_int())).collect(),
            };
            let slots: Vec<i64> = q.init.iter().map(Value::to_int).collect();
            let expected = expected_block(&q).map(|(v, _)| v);
            let got = guarded(|| match jit.try_run_linear(&q.chunk, &slots) {
                None => "declined".to_string(),
                Some(v) => show(&v),
            });
            if got != "declined" {
                accepted.set(accepted.get() + 1);
            }
            match (&expected, got.as_str()) {
                (_, "declined") => None,
                (None, other) => Some(format!("interpreter errored, linear answered {other}")),
                (Some(v), other) => diff("linear", &show(v), other),
            }
        });
        assert_reached("linear tier", accepted.get(), sweep_len(400), 5);
    }

    #[cfg(feature = "aot")]
    #[test]
    fn aot_tier_matches_interpreter() {
        let _env = Env::new(1, 1_000_000);
        sweep("aot tier", 60, Opts::FULL, |p| {
            let want = run_interp(p);
            let init = p.init.clone();
            // The AOT tier assumes slots start unassigned unless the chunk
            // declares them caller-seeded.
            let mut seeded = p.chunk.clone();
            seeded.aot_seeded_slots = SLOTS as u16;
            let got = guarded(|| {
                match fusevm::aot::run_chunk_native(&seeded, move |vm| {
                    vm.frames[0].slots = init.clone();
                }) {
                    Ok(r) => show_result(&r),
                    Err(e) => format!("declined({e})"),
                }
            });
            if got.starts_with("declined") {
                return None;
            }
            // The AOT entry point returns the result only.
            let want_result = want.split(" slots=").next().unwrap().to_string();
            diff("aot", &want_result, &got)
        });
    }

    /// Native code persisted by one thread and loaded by another (a fresh
    /// thread has empty in-memory caches) must answer what the first run did.
    #[cfg(feature = "jit-disk-cache")]
    #[test]
    fn persisted_native_code_matches_interpreter_after_reload() {
        let _env = Env::new(0, 1_000_000);
        let before = fusevm::jit::stats();
        sweep("disk cache", 80, Opts::NATIVE, |p| {
            let kinds = kinds_of(&p.init);
            let expected = expected_block(p);
            let run = || {
                let mut raw = raw_slots(&p.init);
                let jit = JitCompiler::new();
                let r = jit.try_run_block_eager_typed_kinded(&p.chunk, &mut raw, &kinds);
                let trapped = jit.take_awk_div_trap();
                (r.map(block_num_value), trapped, decode_slots(&raw, &kinds))
            };
            let describe = |t: Option<(Option<Value>, bool, Vec<String>)>| match t {
                None => "PANIC".to_string(),
                Some((None, _, _)) => "declined".to_string(),
                Some((_, true, _)) => "trap".to_string(),
                Some((Some(v), false, s)) => format!("{} slots={s:?}", show(&v)),
            };
            let first = describe(catch_unwind(AssertUnwindSafe(run)).ok());
            let second = describe(std::thread::scope(|s| {
                s.spawn(|| catch_unwind(AssertUnwindSafe(run)).ok())
                    .join()
                    .unwrap()
            }));
            if let Some(d) = diff("reload", &first, &second) {
                return Some(d);
            }
            match (&expected, first.as_str()) {
                (_, "declined" | "trap") => None,
                (None, other) => Some(format!("interpreter errored, cache answered {other}")),
                (Some((v, slots)), other) => {
                    diff("disk", &format!("{} slots={slots:?}", show(v)), other)
                }
            }
        });
        let after = fusevm::jit::stats();
        assert!(
            after.disk_loads > before.disk_loads,
            "no native blob was loaded back from disk; the reload path was not exercised"
        );
    }

    /// A chunk that went through serde must key its native code like the
    /// original, and two different ones must not share a cache entry.
    #[test]
    fn deserialized_chunks_do_not_alias_in_the_jit_caches() {
        let _env = Env::new(0, 2);
        sweep("deserialized chunks in the JIT", 150, Opts::NATIVE, |p| {
            let bytes = bincode::serialize(&p.chunk).unwrap();
            let back: Chunk = bincode::deserialize(&bytes).unwrap();
            if back.op_hash != p.chunk.op_hash {
                return Some(format!(
                    "op_hash {:#x} did not survive the round-trip ({:#x})",
                    p.chunk.op_hash, back.op_hash
                ));
            }
            let q = Program {
                chunk: back,
                init: p.init.clone(),
            };
            let expected = run_interp(p);
            (0..2).find_map(|i| diff(&format!("run{i}"), &expected, &run_jit_vm(&q)))
        });
    }
}

// ── Regressions pinned from the sweeps above ─────────────────────────────

#[cfg(feature = "jit")]
fn shown(r: VMResult) -> String {
    show_result(&r)
}

#[cfg(feature = "jit")]
fn ok(v: Value) -> String {
    show_result(&VMResult::Ok(v))
}

fn chunk_of(ops: Vec<Op>) -> Chunk {
    let mut b = ChunkBuilder::new();
    for op in ops {
        b.emit(op, 1);
    }
    b.build()
}

#[test]
fn deserialized_chunk_keeps_its_op_hash() {
    let original = chunk_of(vec![Op::LoadInt(1), Op::LoadInt(2), Op::Add]);
    let wire = bincode::serialize(&original).unwrap();
    let back: Chunk = bincode::deserialize(&wire).unwrap();
    assert_ne!(original.op_hash, 0);
    assert_eq!(back.op_hash, original.op_hash);

    // Sub-chunks are deserialized through the same path.
    let mut b = ChunkBuilder::new();
    b.add_sub_chunk(original.clone());
    let outer = b.build();
    let back: Chunk = bincode::deserialize(&bincode::serialize(&outer).unwrap()).unwrap();
    assert_eq!(back.op_hash, outer.op_hash);
    assert_eq!(back.sub_chunks[0].op_hash, original.op_hash);
}

#[cfg(feature = "jit")]
mod jit_regressions {
    use super::*;
    use fusevm::{JitCompiler, TraceJitConfig};

    fn warm_config(block_threshold: u32, trace_threshold: u32) {
        let jit = JitCompiler::new();
        jit.set_config(TraceJitConfig {
            block_threshold,
            trace_threshold,
            ..jit.get_config()
        });
    }

    fn run_with_slots(chunk: &Chunk, slots: &[Value], jit: bool) -> VMResult {
        let mut vm = VM::new(chunk.clone());
        if jit {
            vm.enable_tracing_jit();
        }
        vm.frames[0].slots = slots.to_vec();
        vm.run()
    }

    /// Two deserialized chunks used to share the JIT cache key `0`: once the
    /// first was compiled, the second ran the first's native code.
    #[test]
    fn deserialized_chunks_do_not_share_native_code() {
        warm_config(0, 1_000_000);
        let a = chunk_of(vec![Op::LoadInt(1), Op::LoadInt(2), Op::Add]);
        let b = chunk_of(vec![Op::LoadInt(5), Op::LoadInt(7), Op::Mul]);
        let a: Chunk = bincode::deserialize(&bincode::serialize(&a).unwrap()).unwrap();
        let b: Chunk = bincode::deserialize(&bincode::serialize(&b).unwrap()).unwrap();
        for _ in 0..4 {
            assert_eq!(shown(run_with_slots(&a, &[], true)), ok(Value::Int(3)));
            assert_eq!(shown(run_with_slots(&b, &[], true)), ok(Value::Int(35)));
        }
    }

    /// A chunk built as a struct literal has no hash either; `VM::new` gives it
    /// one rather than leaving it on the shared key.
    #[test]
    fn literal_chunks_do_not_share_native_code() {
        warm_config(0, 1_000_000);
        let a = Chunk {
            ops: vec![Op::LoadInt(10), Op::LoadInt(4), Op::Sub],
            ..Default::default()
        };
        let b = Chunk {
            ops: vec![Op::LoadInt(10), Op::LoadInt(4), Op::Mul],
            ..Default::default()
        };
        for _ in 0..4 {
            assert_eq!(shown(run_with_slots(&a, &[], true)), ok(Value::Int(6)));
            assert_eq!(shown(run_with_slots(&b, &[], true)), ok(Value::Int(40)));
        }
    }

    /// `c ? slot0 : 11` feeding `%`. The op before the `Mod` is the constant
    /// `11`, but the `then` arm jumps past it with `slot0` on the stack; with
    /// `slot0 == 0` the native `srem` trapped (SIGILL) where the interpreter
    /// answers `0`. Same shape for `Div`, whose interpreter answer is `Undef`.
    #[test]
    fn division_reached_by_a_jump_is_not_judged_by_the_op_before_it() {
        warm_config(0, 1_000_000);
        for (divide, expected) in [(Op::Mod, Value::Int(0)), (Op::Div, Value::Undef)] {
            let chunk = chunk_of(vec![
                Op::LoadInt(100),
                Op::GetSlot(1),
                Op::JumpIfFalse(5),
                Op::GetSlot(0),
                Op::Jump(6),
                Op::LoadInt(11),
                divide.clone(),
            ]);
            let slots = [Value::Int(0), Value::Int(1)];
            let jit = JitCompiler::new();
            assert!(
                !jit.is_block_eligible(&chunk),
                "{divide:?}: a joined division was admitted to the block tier"
            );
            // The constant arm still divides correctly: nothing is lost by the
            // conservative decline.
            for i in 0..4 {
                assert_eq!(
                    shown(run_with_slots(&chunk, &slots, true)),
                    ok(expected.clone()),
                    "{divide:?} run {i}"
                );
            }
            let other_arm = [Value::Int(0), Value::Int(0)];
            let want = match divide {
                Op::Mod => Value::Int(100 % 11),
                _ => Value::Float(100.0 / 11.0),
            };
            assert_eq!(
                shown(run_with_slots(&chunk, &other_arm, true)),
                ok(want),
                "{divide:?} constant arm"
            );
        }
    }

    /// A float condition that is NaN is truthy (`NaN != 0.0`), like the
    /// interpreter's `is_truthy`. The trace tier tested float branch conditions
    /// with an *ordered* not-equal, which is false for NaN — and which
    /// Cranelift's aarch64 backend does not implement at all, so compiling such a
    /// trace panicked there.
    #[test]
    fn nan_is_truthy_in_a_compiled_trace() {
        warm_config(1_000_000, 2);
        use Op::*;
        let chunk = chunk_of(vec![
            LoadInt(0),
            SetSlot(0),
            LoadFloat(0.0),
            SetSlot(1),
            // 4: loop top. inf - inf is NaN, which is truthy.
            LoadFloat(1e300),
            LoadFloat(1e300),
            Mul,
            Dup,
            Sub,
            JumpIfFalse(14),
            GetSlot(0),
            LoadInt(1),
            Add,
            SetSlot(0),
            // 14: loop tail; a float counter, and the loop closes on the float
            // difference itself being truthy.
            GetSlot(1),
            LoadFloat(1.0),
            Add,
            SetSlot(1),
            GetSlot(1),
            LoadFloat(50.0),
            Sub,
            JumpIfTrue(4),
            GetSlot(0),
        ]);
        let slots = [Value::Int(0), Value::Int(0)];
        let want = shown(run_with_slots(&chunk, &slots, false));
        assert_eq!(want, ok(Value::Int(50)));
        let before = fusevm::jit::stats();
        for i in 0..3 {
            assert_eq!(shown(run_with_slots(&chunk, &slots, true)), want, "run {i}");
        }
        let after = fusevm::jit::stats();
        assert!(
            after.trace_hits > before.trace_hits,
            "the loop never ran in a compiled trace"
        );
    }
}

#[cfg(feature = "aot")]
#[test]
fn in_process_aot_runs_a_chunk_with_seeded_slots() {
    // `aot_seeded_slots` makes the native entry load its slots through
    // `fusevm_aot_load_slot_int` and `fusevm_aot_guard_taken`; the in-process
    // runner did not register either symbol and failed to resolve them.
    let mut chunk = chunk_of(vec![Op::GetSlot(0), Op::GetSlot(1), Op::Add]);
    chunk.aot_seeded_slots = 2;
    let r = fusevm::aot::run_chunk_native(&chunk, |vm| {
        vm.frames[0].slots = vec![Value::Int(3), Value::Int(4)];
    });
    assert_eq!(shown(r.unwrap()), ok(Value::Int(7)));
}
