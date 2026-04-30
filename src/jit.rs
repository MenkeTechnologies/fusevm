//! fusevm JIT — Cranelift codegen for universal bytecodes.
//!
//! Compiles fusevm::Op sequences to native machine code via Cranelift.
//! Language-specific ops (Extended) are handled by a `JitExtension` trait
//! that frontends register.
//!
//! # Architecture
//!
//! ```text
//! fusevm::Chunk
//!     │
//!     ▼
//! JitCompiler::compile()
//!     │
//!     ├── Universal ops: LoadInt, Add, Jump, NumEq, Call, ...
//!     │   → Cranelift IR directly
//!     │
//!     └── Extended(id, arg)
//!         → JitExtension::emit_extended() (registered by frontend)
//!     │
//!     ▼
//! NativeCode { fn_ptr } → call directly
//! ```
//!
//! # Two compilation tiers
//!
//! ## Linear JIT
//! Compiles straight-line (no branches) sequences in a single Cranelift basic block.
//! Stack must end with exactly one value.
//!
//! ## Block JIT
//! Compiles bytecodes with control flow (loops, conditionals, short-circuit &&/||) via
//! a basic-block CFG with abstract stack merges at block joins.
//!
//! # Eligible universal ops (both tiers)
//!
//! `LoadInt`, `LoadFloat`, `LoadConst` (int/float constants), `LoadTrue`, `LoadFalse`,
//! `Pop`, `Dup`, `Swap`, `Rot`,
//! `GetSlot`/`SetSlot`,
//! `Add`/`Sub`/`Mul`/`Div`/`Mod`/`Pow`/`Negate`/`Inc`/`Dec`,
//! `NumEq`/`NumNe`/`NumLt`/`NumGt`/`NumLe`/`NumGe`/`Spaceship`,
//! `BitAnd`/`BitOr`/`BitXor`/`BitNot`/`Shl`/`Shr`,
//! `LogNot`,
//! `Jump`/`JumpIfTrue`/`JumpIfFalse`/`JumpIfTrueKeep`/`JumpIfFalseKeep` (block tier only),
//! `PreIncSlot`/`PreIncSlotVoid`/`SlotLtIntJumpIfFalse`/`SlotIncLtIntJumpBack`/`AccumSumLoop`/`AddAssignSlotVoid`.

/// Extension trait for language-specific JIT codegen.
/// Frontends implement this to JIT their Extended ops.
pub trait JitExtension: Send + Sync {
    /// Whether this extension can JIT-compile the given extended op ID.
    fn can_jit(&self, ext_id: u16) -> bool;

    /// Number of extended ops this extension handles.
    fn op_count(&self) -> usize;

    /// Human-readable name for debugging.
    fn name(&self) -> &str;
}

/// Slot type tag observed at trace recording time.
///
/// Used by the tracing JIT entry guard: if a slot's runtime type at trace
/// invocation differs from the type recorded at compile time, the trace
/// is skipped and the interpreter handles the iteration.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum SlotKind {
    Int,
    Float,
}

/// Serializable trace metadata for persistent cache export/import (phase 7).
///
/// Captures everything needed to re-compile a previously-installed trace
/// in a fresh process: the original recorded op sequence, the parallel
/// bytecode IPs, the slot-type entry guard, and the fallthrough IP. Bind
/// to the chunk via `chunk_op_hash` so a stale metadata file (chunk has
/// changed) is rejected on import rather than silently mis-compiled.
///
/// Persistence format is intentionally serde-based so callers can pick
/// whatever encoding fits their environment (JSON, bincode, custom binary).
/// `fusevm` itself doesn't ship a file format — `JitCompiler::export_trace`
/// returns the struct, `import_trace` consumes one. The user owns I/O.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct TraceMetadata {
    pub chunk_op_hash: u64,
    pub anchor_ip: usize,
    pub fallthrough_ip: usize,
    pub ops: Vec<crate::op::Op>,
    pub recorded_ips: Vec<usize>,
    pub slot_kinds_at_anchor: Vec<SlotKind>,
}

/// Outcome of consulting the trace cache at a backward-branch site.
#[derive(Debug)]
pub enum TraceLookup {
    /// Header not yet hot — interpreter continues. Counter was bumped.
    NotHot,
    /// Header crossed `TRACE_THRESHOLD` and no compiled trace exists yet.
    /// The interpreter should arm the recorder for the next iteration.
    StartRecording,
    /// A compiled trace ran. The interpreter should resume at this IP.
    Ran { resume_ip: usize },
    /// A compiled trace exists but the slot type guard failed.
    /// Interpreter handles this iteration normally.
    GuardMismatch,
    /// Trace was previously aborted or blacklisted; never retry.
    Skip,
}

/// Maximum number of inlined callee frames a trace can materialize on
/// side-exit. Traces requiring deeper inlining at any side-exit point are
/// rejected at compile time. 4 covers typical shell/script helper patterns.
pub const MAX_DEOPT_FRAMES: usize = 4;

/// Maximum slot indices per inlined frame the trace can materialize.
pub const MAX_DEOPT_SLOTS_PER_FRAME: usize = 16;

/// Maximum abstract-stack values a trace can write back to the interpreter
/// stack on side-exit. Phase 5+. Branches with deeper stack are rejected
/// at compile time.
pub const MAX_DEOPT_STACK: usize = 32;

/// Tunable thresholds for the tracing JIT.
///
/// All hot/threshold/cap values are surfaced here so callers can adjust
/// the JIT for their workload without recompiling fusevm. Defaults match
/// the constants used through phase 9; aggressive workloads (very hot
/// short loops) might want lower thresholds, while cold-start workloads
/// (script that runs once) might want higher thresholds to avoid spending
/// compile time on traces that won't pay off.
///
/// Apply via `JitCompiler::set_config(...)` — affects subsequent calls
/// from the current thread.
#[derive(Clone, Copy, Debug)]
pub struct TraceJitConfig {
    /// Backedges through a loop header before recording arms.
    pub trace_threshold: u32,
    /// Mid-trace side-exits before the trace is auto-blacklisted.
    pub max_side_exits: u32,
    /// Maximum self-recursive levels the recorder will inline.
    pub max_inline_recursion: usize,
    /// Maximum chained side traces dispatched per backward-branch hop.
    pub max_trace_chain: usize,
    /// Maximum recorded ops in a single trace before recording aborts.
    pub max_trace_len: usize,
}

impl TraceJitConfig {
    /// Defaults matching the phase-1-through-9 constants.
    pub const fn defaults() -> Self {
        Self {
            trace_threshold: 50,
            max_side_exits: 50,
            max_inline_recursion: 4,
            max_trace_chain: 4,
            max_trace_len: 256,
        }
    }
}

impl Default for TraceJitConfig {
    fn default() -> Self {
        Self::defaults()
    }
}

/// Materialization record for a single inlined frame at side-exit time.
///
/// Phase 4 of the tracing JIT: when a trace deopts inside a callee, the
/// interpreter expects `vm.frames` to reflect the call stack the bytecode
/// would naturally have at the deopt IP. The compiled trace populates one
/// `DeoptFrame` per inlined frame (caller→callee order) into `DeoptInfo`,
/// and the VM materializes them after the trace returns.
#[derive(Clone, Copy)]
#[repr(C)]
pub struct DeoptFrame {
    /// IP just after the corresponding `Op::Call` in the parent frame.
    /// Set by the interpreter when this synthetic frame's `Return` fires.
    pub return_ip: usize,
    /// Number of slot values written into `slots`.
    pub slot_count: usize,
    /// Slot values, indexed [0..slot_count]. Each is interpreted as an
    /// i64 carried by `Value::Int`. Beyond `slot_count` is undefined.
    pub slots: [i64; MAX_DEOPT_SLOTS_PER_FRAME],
}

impl DeoptFrame {
    /// Zero-init record (used to pre-fill the buffer before each invocation).
    pub const fn zeroed() -> Self {
        Self {
            return_ip: 0,
            slot_count: 0,
            slots: [0; MAX_DEOPT_SLOTS_PER_FRAME],
        }
    }
}

/// Tag for a single abstract-stack entry written to `DeoptInfo.stack_buf`.
/// 0 = `Value::Int(i64)`, 1 = `Value::Float(f64)`. Other values reserved.
pub const STACK_KIND_INT: u8 = 0;
pub const STACK_KIND_FLOAT: u8 = 1;

/// Out-parameter the trace fn writes on every invocation.
///
/// `resume_ip` is set by the trace on every exit (normal loop fallthrough
/// OR side-exit). `frame_count` is 0 for caller-frame side-exits (no
/// materialization needed) and 1..=MAX_DEOPT_FRAMES for callee-frame
/// side-exits. `stack_count` is the number of abstract-stack values the
/// trace had built up at the side-exit; the VM pushes them onto
/// `vm.stack` after the trace returns, materializing each entry as
/// `Value::Int` or `Value::Float` based on the parallel `stack_kinds`
/// tag (Float values are bit-cast through `f64::from_bits`).
#[derive(Clone, Copy)]
#[repr(C)]
pub struct DeoptInfo {
    pub resume_ip: usize,
    pub frame_count: usize,
    pub stack_count: usize,
    pub frames: [DeoptFrame; MAX_DEOPT_FRAMES],
    pub stack_buf: [i64; MAX_DEOPT_STACK],
    pub stack_kinds: [u8; MAX_DEOPT_STACK],
}

impl DeoptInfo {
    /// Zero-init buffer.
    pub const fn zeroed() -> Self {
        Self {
            resume_ip: 0,
            frame_count: 0,
            stack_count: 0,
            frames: [DeoptFrame::zeroed(); MAX_DEOPT_FRAMES],
            stack_buf: [0; MAX_DEOPT_STACK],
            stack_kinds: [0; MAX_DEOPT_STACK],
        }
    }
}

// ── Cranelift JIT implementation (feature-gated) ──

#[cfg(feature = "jit")]
mod cranelift_jit_impl {
    use std::collections::HashMap;
    use std::sync::OnceLock;

    use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};
    use cranelift_codegen::ir::immediates::Ieee64;
    use cranelift_codegen::ir::types;
    use cranelift_codegen::ir::{AbiParam, BlockArg, InstBuilder, MemFlags, UserFuncName, Value};
    use cranelift_codegen::isa::OwnedTargetIsa;
    use cranelift_codegen::settings::{self, Configurable};
    use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
    use cranelift_jit::{JITBuilder, JITModule};
    use cranelift_module::{default_libcall_names, Linkage, Module};

    use crate::chunk::Chunk;
    use crate::op::Op;
    use crate::value::Value as FuseValue;

    // ── Types ──

    /// Whether the native function returns an integer or a float.
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    pub(crate) enum JitTy {
        Int,
        Float,
    }

    type LinearFn0 = unsafe extern "C" fn() -> i64;
    type LinearFn0F = unsafe extern "C" fn() -> f64;
    type LinearFnSlots = unsafe extern "C" fn(*const i64) -> i64;
    type LinearFnSlotsF = unsafe extern "C" fn(*const i64) -> f64;

    enum LinearRun {
        Nullary(LinearFn0),
        NullaryF(LinearFn0F),
        Slots(LinearFnSlots),
        SlotsF(LinearFnSlotsF),
    }

    pub(crate) enum JitResult {
        Int(i64),
        Float(f64),
    }

    pub(crate) struct CompiledLinear {
        #[allow(dead_code)]
        module: JITModule,
        run: LinearRun,
    }

    impl CompiledLinear {
        pub(crate) fn invoke(&self, slots: *const i64) -> JitResult {
            match &self.run {
                LinearRun::Nullary(f) => JitResult::Int(unsafe { f() }),
                LinearRun::NullaryF(f) => JitResult::Float(unsafe { f() }),
                LinearRun::Slots(f) => JitResult::Int(unsafe { f(slots) }),
                LinearRun::SlotsF(f) => JitResult::Float(unsafe { f(slots) }),
            }
        }

        pub(crate) fn result_to_value(&self, j: JitResult) -> FuseValue {
            match j {
                JitResult::Int(n) => FuseValue::Int(n),
                JitResult::Float(f) => FuseValue::Float(f),
            }
        }
    }

    // ── Abstract stack simulation ──

    #[derive(Clone, Copy, PartialEq, Debug)]
    pub(crate) enum Cell {
        Const(i64),
        ConstF(f64),
        Dyn,
        DynF,
    }

    impl Cell {
        fn is_float(self) -> bool {
            matches!(self, Cell::ConstF(_) | Cell::DynF)
        }
        fn either_float(a: Cell, b: Cell) -> bool {
            a.is_float() || b.is_float()
        }
    }

    fn cell_to_jit_ty(c: Cell) -> JitTy {
        match c {
            Cell::ConstF(_) | Cell::DynF => JitTy::Float,
            Cell::Const(_) | Cell::Dyn => JitTy::Int,
        }
    }

    fn pop2_strict(stack: &mut Vec<Cell>) -> Option<(Cell, Cell)> {
        let b = stack.pop()?;
        let a = stack.pop()?;
        Some((a, b))
    }

    fn fold_arith(
        a: Cell,
        b: Cell,
        int_op: fn(i64, i64) -> i64,
        f_op: fn(f64, f64) -> f64,
    ) -> Cell {
        match (a, b) {
            (Cell::Const(x), Cell::Const(y)) => Cell::Const(int_op(x, y)),
            (Cell::ConstF(x), Cell::ConstF(y)) => Cell::ConstF(f_op(x, y)),
            _ if Cell::either_float(a, b) => Cell::DynF,
            _ => Cell::Dyn,
        }
    }

    fn fold_cmp_cell(op: &Op, a: Cell, b: Cell) -> Cell {
        fn float_cmp(op: &Op, x: f64, y: f64) -> i64 {
            if x.is_nan() || y.is_nan() {
                return 0;
            }
            match op {
                Op::NumEq => i64::from(x == y),
                Op::NumNe => i64::from(x != y),
                Op::NumLt => i64::from(x < y),
                Op::NumGt => i64::from(x > y),
                Op::NumLe => i64::from(x <= y),
                Op::NumGe => i64::from(x >= y),
                Op::Spaceship => match x.partial_cmp(&y) {
                    Some(std::cmp::Ordering::Less) => -1,
                    Some(std::cmp::Ordering::Equal) => 0,
                    Some(std::cmp::Ordering::Greater) => 1,
                    None => 0,
                },
                _ => 0,
            }
        }
        match (a, b) {
            (Cell::Const(x), Cell::Const(y)) => {
                let v = match op {
                    Op::NumEq => i64::from(x == y),
                    Op::NumNe => i64::from(x != y),
                    Op::NumLt => i64::from(x < y),
                    Op::NumGt => i64::from(x > y),
                    Op::NumLe => i64::from(x <= y),
                    Op::NumGe => i64::from(x >= y),
                    Op::Spaceship => match x.cmp(&y) {
                        std::cmp::Ordering::Less => -1,
                        std::cmp::Ordering::Equal => 0,
                        std::cmp::Ordering::Greater => 1,
                    },
                    _ => 0,
                };
                Cell::Const(v)
            }
            (Cell::ConstF(x), Cell::ConstF(y)) => Cell::Const(float_cmp(op, x, y)),
            (Cell::Const(x), Cell::ConstF(y)) => Cell::Const(float_cmp(op, x as f64, y)),
            (Cell::ConstF(x), Cell::Const(y)) => Cell::Const(float_cmp(op, x, y as f64)),
            _ => Cell::Dyn,
        }
    }

    /// One op for abstract stack simulation (validation).
    fn simulate_one_op(op: &Op, stack: &mut Vec<Cell>, constants: &[FuseValue]) -> Option<()> {
        match op {
            Op::LoadInt(n) => stack.push(Cell::Const(*n)),
            Op::LoadFloat(f) => {
                if !f.is_finite() {
                    return None;
                }
                let n = *f as i64;
                if (n as f64) == *f {
                    stack.push(Cell::Const(n));
                } else {
                    stack.push(Cell::ConstF(*f));
                }
            }
            Op::LoadConst(idx) => {
                let val = constants.get(*idx as usize)?;
                match val {
                    FuseValue::Int(n) => stack.push(Cell::Const(*n)),
                    FuseValue::Float(f) => stack.push(Cell::ConstF(*f)),
                    _ => return None,
                }
            }
            Op::LoadTrue => stack.push(Cell::Const(1)),
            Op::LoadFalse => stack.push(Cell::Const(0)),
            Op::Add => {
                let (a, b) = pop2_strict(stack)?;
                stack.push(fold_arith(a, b, i64::wrapping_add, |x, y| x + y));
            }
            Op::Sub => {
                let (a, b) = pop2_strict(stack)?;
                stack.push(fold_arith(a, b, i64::wrapping_sub, |x, y| x - y));
            }
            Op::Mul => {
                let (a, b) = pop2_strict(stack)?;
                stack.push(fold_arith(a, b, i64::wrapping_mul, |x, y| x * y));
            }
            Op::Div => {
                let (a, b) = pop2_strict(stack)?;
                if Cell::either_float(a, b) {
                    match (a, b) {
                        (Cell::ConstF(x), Cell::ConstF(y)) => stack.push(Cell::ConstF(x / y)),
                        _ => stack.push(Cell::DynF),
                    }
                } else {
                    match (a, b) {
                        (Cell::Const(x), Cell::Const(y)) if y != 0 && x % y == 0 => {
                            stack.push(Cell::Const(x / y));
                        }
                        _ => return None,
                    }
                }
            }
            Op::Mod => {
                let (a, b) = pop2_strict(stack)?;
                if Cell::either_float(a, b) {
                    match (a, b) {
                        (Cell::ConstF(x), Cell::ConstF(y)) => stack.push(Cell::ConstF(x % y)),
                        _ => stack.push(Cell::DynF),
                    }
                } else {
                    match b {
                        Cell::Const(0) => return None,
                        Cell::Const(y) => stack.push(match a {
                            Cell::Const(x) => Cell::Const(x % y),
                            _ => Cell::Dyn,
                        }),
                        _ => return None,
                    }
                }
            }
            Op::Pow => {
                let (a, b) = pop2_strict(stack)?;
                if Cell::either_float(a, b) {
                    match (a, b) {
                        (Cell::ConstF(x), Cell::ConstF(y)) => stack.push(Cell::ConstF(x.powf(y))),
                        _ => stack.push(Cell::DynF),
                    }
                } else {
                    match (a, b) {
                        (Cell::Const(x), Cell::Const(y)) if (0..=63).contains(&y) => {
                            stack.push(Cell::Const(x.wrapping_pow(y as u32)));
                        }
                        (Cell::Dyn, Cell::Const(y)) if (0..=63).contains(&y) => {
                            stack.push(Cell::Dyn);
                        }
                        _ => return None,
                    }
                }
            }
            Op::Negate => {
                let a = stack.pop()?;
                stack.push(match a {
                    Cell::Const(n) => Cell::Const(n.wrapping_neg()),
                    Cell::ConstF(f) => Cell::ConstF(-f),
                    Cell::DynF => Cell::DynF,
                    Cell::Dyn => Cell::Dyn,
                });
            }
            Op::Inc => {
                let a = stack.pop()?;
                stack.push(match a {
                    Cell::Const(n) => Cell::Const(n.wrapping_add(1)),
                    _ => Cell::Dyn,
                });
            }
            Op::Dec => {
                let a = stack.pop()?;
                stack.push(match a {
                    Cell::Const(n) => Cell::Const(n.wrapping_sub(1)),
                    _ => Cell::Dyn,
                });
            }
            Op::Pop => {
                stack.pop()?;
            }
            Op::Dup => {
                let v = stack.last().copied()?;
                stack.push(v);
            }
            Op::Swap => {
                let b = stack.pop()?;
                let a = stack.pop()?;
                stack.push(b);
                stack.push(a);
            }
            Op::Rot => {
                let c = stack.pop()?;
                let b = stack.pop()?;
                let a = stack.pop()?;
                stack.push(b);
                stack.push(c);
                stack.push(a);
            }
            Op::NumEq
            | Op::NumNe
            | Op::NumLt
            | Op::NumGt
            | Op::NumLe
            | Op::NumGe
            | Op::Spaceship => {
                let (a, b) = pop2_strict(stack)?;
                stack.push(fold_cmp_cell(op, a, b));
            }
            Op::BitXor | Op::BitAnd | Op::BitOr | Op::Shl | Op::Shr => {
                let (a, b) = pop2_strict(stack)?;
                if Cell::either_float(a, b) {
                    return None;
                }
                stack.push(match (a, b) {
                    (Cell::Const(x), Cell::Const(y)) => Cell::Const(match op {
                        Op::BitXor => x ^ y,
                        Op::BitAnd => x & y,
                        Op::BitOr => x | y,
                        Op::Shl => x.wrapping_shl((y as u32) & 63),
                        Op::Shr => x.wrapping_shr((y as u32) & 63),
                        _ => unreachable!(),
                    }),
                    _ => Cell::Dyn,
                });
            }
            Op::BitNot => {
                let a = stack.pop()?;
                if a.is_float() {
                    return None;
                }
                stack.push(match a {
                    Cell::Const(n) => Cell::Const(!n),
                    _ => Cell::Dyn,
                });
            }
            Op::LogNot => {
                let a = stack.pop()?;
                stack.push(match a {
                    Cell::Const(n) => Cell::Const(if n != 0 { 0 } else { 1 }),
                    Cell::ConstF(f) => Cell::Const(if f != 0.0 { 0 } else { 1 }),
                    _ => Cell::Dyn,
                });
            }
            Op::GetSlot(_) => stack.push(Cell::Dyn),
            Op::SetSlot(_) => {
                stack.pop()?;
            }
            Op::PreIncSlot(_) => stack.push(Cell::Dyn),
            Op::PreIncSlotVoid(_) => {}
            Op::SlotLtIntJumpIfFalse(_, _, _) => {} // control flow, handled at block level
            Op::SlotIncLtIntJumpBack(_, _, _) => {} // control flow, handled at block level
            Op::AccumSumLoop(_, _, _) => {}         // self-contained fused op
            Op::AddAssignSlotVoid(_, _) => {}
            _ => return None,
        }
        Some(())
    }

    fn validate_linear_seq(seq: &[Op], constants: &[FuseValue]) -> bool {
        if seq.is_empty() {
            return false;
        }
        let mut stack: Vec<Cell> = Vec::new();
        for op in seq {
            if simulate_one_op(op, &mut stack, constants).is_none() {
                return false;
            }
            if stack.len() > 256 {
                return false;
            }
        }
        stack.len() == 1
    }

    fn linear_result_cell(seq: &[Op], constants: &[FuseValue]) -> Option<Cell> {
        let mut stack: Vec<Cell> = Vec::new();
        for op in seq {
            simulate_one_op(op, &mut stack, constants)?;
            if stack.len() > 256 {
                return None;
            }
        }
        if stack.len() != 1 {
            return None;
        }
        stack.pop()
    }

    fn needs_slots(seq: &[Op]) -> bool {
        seq.iter().any(|o| {
            matches!(
                o,
                Op::GetSlot(_)
                    | Op::SetSlot(_)
                    | Op::PreIncSlot(_)
                    | Op::PreIncSlotVoid(_)
                    | Op::SlotLtIntJumpIfFalse(_, _, _)
                    | Op::SlotIncLtIntJumpBack(_, _, _)
                    | Op::AccumSumLoop(_, _, _)
                    | Op::AddAssignSlotVoid(_, _)
            )
        })
    }

    // ── Cranelift setup ──

    fn isa_flags() -> cranelift_codegen::settings::Flags {
        let mut flag_builder = settings::builder();
        let _ = flag_builder.set("use_colocated_libcalls", "false");
        let _ = flag_builder.set("is_pic", "false");
        let _ = flag_builder.set("opt_level", "speed");
        settings::Flags::new(flag_builder)
    }

    static JIT_OWNED_ISA: OnceLock<Option<OwnedTargetIsa>> = OnceLock::new();

    fn cached_owned_isa() -> Option<&'static OwnedTargetIsa> {
        JIT_OWNED_ISA
            .get_or_init(|| {
                let isa_builder = cranelift_native::builder().ok()?;
                isa_builder.finish(isa_flags()).ok()
            })
            .as_ref()
    }

    /// Integer `**` matching vm.rs when both operands are `i64` and `0 <= exp <= 63`.
    #[no_mangle]
    pub extern "C" fn fusevm_jit_pow_i64(base: i64, exp: i64) -> i64 {
        if (0..=63).contains(&exp) {
            base.wrapping_pow(exp as u32)
        } else {
            0
        }
    }

    /// Float `**` — delegates to `f64::powf`.
    #[no_mangle]
    pub extern "C" fn fusevm_jit_pow_f64(base: f64, exp: f64) -> f64 {
        base.powf(exp)
    }

    /// Float `%` — delegates to `f64 % f64`.
    #[no_mangle]
    pub extern "C" fn fusevm_jit_fmod_f64(a: f64, b: f64) -> f64 {
        a % b
    }

    /// `!` on an i64 value (0 → 1, nonzero → 0).
    #[no_mangle]
    pub extern "C" fn fusevm_jit_lognot_i64(n: i64) -> i64 {
        if n != 0 {
            0
        } else {
            1
        }
    }

    fn new_jit_module() -> Option<JITModule> {
        let isa = cached_owned_isa()?.clone();
        let mut builder = JITBuilder::with_isa(isa, default_libcall_names());
        builder.symbol("fusevm_jit_pow_i64", fusevm_jit_pow_i64 as *const u8);
        builder.symbol("fusevm_jit_pow_f64", fusevm_jit_pow_f64 as *const u8);
        builder.symbol("fusevm_jit_fmod_f64", fusevm_jit_fmod_f64 as *const u8);
        builder.symbol("fusevm_jit_lognot_i64", fusevm_jit_lognot_i64 as *const u8);
        Some(JITModule::new(builder))
    }

    // ── Cranelift IR helpers ──

    fn intcmp_to_01(bcx: &mut FunctionBuilder, cc: IntCC, a: Value, b: Value) -> Value {
        let pred = bcx.ins().icmp(cc, a, b);
        let one = bcx.ins().iconst(types::I64, 1);
        let zero = bcx.ins().iconst(types::I64, 0);
        bcx.ins().select(pred, one, zero)
    }

    fn spaceship_i64(bcx: &mut FunctionBuilder, a: Value, b: Value) -> Value {
        let lt = bcx.ins().icmp(IntCC::SignedLessThan, a, b);
        let gt = bcx.ins().icmp(IntCC::SignedGreaterThan, a, b);
        let m1 = bcx.ins().iconst(types::I64, -1);
        let z = bcx.ins().iconst(types::I64, 0);
        let p1 = bcx.ins().iconst(types::I64, 1);
        let mid = bcx.ins().select(gt, p1, z);
        bcx.ins().select(lt, m1, mid)
    }

    fn floatcmp_to_01(bcx: &mut FunctionBuilder, cc: FloatCC, a: Value, b: Value) -> Value {
        let pred = bcx.ins().fcmp(cc, a, b);
        let one = bcx.ins().iconst(types::I64, 1);
        let zero = bcx.ins().iconst(types::I64, 0);
        bcx.ins().select(pred, one, zero)
    }

    fn spaceship_f64(bcx: &mut FunctionBuilder, a: Value, b: Value) -> Value {
        let lt = bcx.ins().fcmp(FloatCC::LessThan, a, b);
        let gt = bcx.ins().fcmp(FloatCC::GreaterThan, a, b);
        let m1 = bcx.ins().iconst(types::I64, -1);
        let z = bcx.ins().iconst(types::I64, 0);
        let p1 = bcx.ins().iconst(types::I64, 1);
        let mid = bcx.ins().select(gt, p1, z);
        bcx.ins().select(lt, m1, mid)
    }

    fn i64_to_f64(bcx: &mut FunctionBuilder, v: Value) -> Value {
        bcx.ins().fcvt_from_sint(types::F64, v)
    }

    fn f64_to_i64_trunc(bcx: &mut FunctionBuilder, v: Value) -> Value {
        bcx.ins().fcvt_to_sint(types::I64, v)
    }

    fn pop_pair_promote(
        bcx: &mut FunctionBuilder,
        stack: &mut Vec<(Value, JitTy)>,
    ) -> Option<(Value, Value, JitTy)> {
        let (b, tb) = stack.pop()?;
        let (a, ta) = stack.pop()?;
        Some(match (ta, tb) {
            (JitTy::Int, JitTy::Int) => (a, b, JitTy::Int),
            (JitTy::Float, JitTy::Float) => (a, b, JitTy::Float),
            (JitTy::Int, JitTy::Float) => (i64_to_f64(bcx, a), b, JitTy::Float),
            (JitTy::Float, JitTy::Int) => (a, i64_to_f64(bcx, b), JitTy::Float),
        })
    }

    fn scalar_store_i64(bcx: &mut FunctionBuilder, v: Value, ty: JitTy) -> Value {
        match ty {
            JitTy::Int => v,
            JitTy::Float => f64_to_i64_trunc(bcx, v),
        }
    }

    // ── Cranelift IR emission per op ──

    fn emit_data_op(
        bcx: &mut FunctionBuilder,
        op: &Op,
        stack: &mut Vec<(Value, JitTy)>,
        slot_base: Option<Value>,
        pow_i64_ref: Option<cranelift_codegen::ir::FuncRef>,
        pow_f64_ref: Option<cranelift_codegen::ir::FuncRef>,
        fmod_f64_ref: Option<cranelift_codegen::ir::FuncRef>,
        lognot_ref: Option<cranelift_codegen::ir::FuncRef>,
        constants: &[FuseValue],
    ) -> Option<()> {
        match op {
            Op::LoadInt(n) => {
                stack.push((bcx.ins().iconst(types::I64, *n), JitTy::Int));
            }
            Op::LoadFloat(f) => {
                if !f.is_finite() {
                    return None;
                }
                let n = *f as i64;
                if (n as f64) == *f {
                    stack.push((bcx.ins().iconst(types::I64, n), JitTy::Int));
                } else {
                    stack.push((
                        bcx.ins().f64const(Ieee64::with_bits(f.to_bits())),
                        JitTy::Float,
                    ));
                }
            }
            Op::LoadConst(idx) => {
                let val = constants.get(*idx as usize)?;
                match val {
                    FuseValue::Int(n) => {
                        stack.push((bcx.ins().iconst(types::I64, *n), JitTy::Int));
                    }
                    FuseValue::Float(f) => {
                        stack.push((
                            bcx.ins().f64const(Ieee64::with_bits(f.to_bits())),
                            JitTy::Float,
                        ));
                    }
                    _ => return None,
                }
            }
            Op::LoadTrue => stack.push((bcx.ins().iconst(types::I64, 1), JitTy::Int)),
            Op::LoadFalse => stack.push((bcx.ins().iconst(types::I64, 0), JitTy::Int)),
            Op::Add => {
                let (a, b, ty) = pop_pair_promote(bcx, stack)?;
                match ty {
                    JitTy::Int => stack.push((bcx.ins().iadd(a, b), JitTy::Int)),
                    JitTy::Float => stack.push((bcx.ins().fadd(a, b), JitTy::Float)),
                }
            }
            Op::Sub => {
                let (a, b, ty) = pop_pair_promote(bcx, stack)?;
                match ty {
                    JitTy::Int => stack.push((bcx.ins().isub(a, b), JitTy::Int)),
                    JitTy::Float => stack.push((bcx.ins().fsub(a, b), JitTy::Float)),
                }
            }
            Op::Mul => {
                let (a, b, ty) = pop_pair_promote(bcx, stack)?;
                match ty {
                    JitTy::Int => stack.push((bcx.ins().imul(a, b), JitTy::Int)),
                    JitTy::Float => stack.push((bcx.ins().fmul(a, b), JitTy::Float)),
                }
            }
            Op::Div => {
                let (a, b, ty) = pop_pair_promote(bcx, stack)?;
                match ty {
                    JitTy::Int => stack.push((bcx.ins().sdiv(a, b), JitTy::Int)),
                    JitTy::Float => stack.push((bcx.ins().fdiv(a, b), JitTy::Float)),
                }
            }
            Op::Mod => {
                let (a, b, ty) = pop_pair_promote(bcx, stack)?;
                match ty {
                    JitTy::Int => stack.push((bcx.ins().srem(a, b), JitTy::Int)),
                    JitTy::Float => {
                        let fr = fmod_f64_ref?;
                        let call = bcx.ins().call(fr, &[a, b]);
                        stack.push((*bcx.inst_results(call).first()?, JitTy::Float));
                    }
                }
            }
            Op::Pow => {
                let (a, b, ty) = pop_pair_promote(bcx, stack)?;
                match ty {
                    JitTy::Int => {
                        let fr = pow_i64_ref?;
                        let call = bcx.ins().call(fr, &[a, b]);
                        stack.push((*bcx.inst_results(call).first()?, JitTy::Int));
                    }
                    JitTy::Float => {
                        let fr = pow_f64_ref?;
                        let call = bcx.ins().call(fr, &[a, b]);
                        stack.push((*bcx.inst_results(call).first()?, JitTy::Float));
                    }
                }
            }
            Op::Negate => {
                let (a, ty) = stack.pop()?;
                match ty {
                    JitTy::Int => stack.push((bcx.ins().ineg(a), JitTy::Int)),
                    JitTy::Float => stack.push((bcx.ins().fneg(a), JitTy::Float)),
                }
            }
            Op::Inc => {
                let (a, ty) = stack.pop()?;
                let a = scalar_store_i64(bcx, a, ty);
                let one = bcx.ins().iconst(types::I64, 1);
                stack.push((bcx.ins().iadd(a, one), JitTy::Int));
            }
            Op::Dec => {
                let (a, ty) = stack.pop()?;
                let a = scalar_store_i64(bcx, a, ty);
                let one = bcx.ins().iconst(types::I64, 1);
                stack.push((bcx.ins().isub(a, one), JitTy::Int));
            }
            Op::NumEq => {
                let (a, b, ty) = pop_pair_promote(bcx, stack)?;
                let v = match ty {
                    JitTy::Int => intcmp_to_01(bcx, IntCC::Equal, a, b),
                    JitTy::Float => floatcmp_to_01(bcx, FloatCC::Equal, a, b),
                };
                stack.push((v, JitTy::Int));
            }
            Op::NumNe => {
                let (a, b, ty) = pop_pair_promote(bcx, stack)?;
                let v = match ty {
                    JitTy::Int => intcmp_to_01(bcx, IntCC::NotEqual, a, b),
                    JitTy::Float => floatcmp_to_01(bcx, FloatCC::NotEqual, a, b),
                };
                stack.push((v, JitTy::Int));
            }
            Op::NumLt => {
                let (a, b, ty) = pop_pair_promote(bcx, stack)?;
                let v = match ty {
                    JitTy::Int => intcmp_to_01(bcx, IntCC::SignedLessThan, a, b),
                    JitTy::Float => floatcmp_to_01(bcx, FloatCC::LessThan, a, b),
                };
                stack.push((v, JitTy::Int));
            }
            Op::NumGt => {
                let (a, b, ty) = pop_pair_promote(bcx, stack)?;
                let v = match ty {
                    JitTy::Int => intcmp_to_01(bcx, IntCC::SignedGreaterThan, a, b),
                    JitTy::Float => floatcmp_to_01(bcx, FloatCC::GreaterThan, a, b),
                };
                stack.push((v, JitTy::Int));
            }
            Op::NumLe => {
                let (a, b, ty) = pop_pair_promote(bcx, stack)?;
                let v = match ty {
                    JitTy::Int => intcmp_to_01(bcx, IntCC::SignedLessThanOrEqual, a, b),
                    JitTy::Float => floatcmp_to_01(bcx, FloatCC::LessThanOrEqual, a, b),
                };
                stack.push((v, JitTy::Int));
            }
            Op::NumGe => {
                let (a, b, ty) = pop_pair_promote(bcx, stack)?;
                let v = match ty {
                    JitTy::Int => intcmp_to_01(bcx, IntCC::SignedGreaterThanOrEqual, a, b),
                    JitTy::Float => floatcmp_to_01(bcx, FloatCC::GreaterThanOrEqual, a, b),
                };
                stack.push((v, JitTy::Int));
            }
            Op::Spaceship => {
                let (a, b, ty) = pop_pair_promote(bcx, stack)?;
                let v = match ty {
                    JitTy::Int => spaceship_i64(bcx, a, b),
                    JitTy::Float => spaceship_f64(bcx, a, b),
                };
                stack.push((v, JitTy::Int));
            }
            Op::LogNot => {
                let (a, ty) = stack.pop()?;
                let fr = lognot_ref?;
                let a_int = match ty {
                    JitTy::Int => a,
                    JitTy::Float => {
                        let z = bcx.ins().f64const(Ieee64::with_bits(0.0f64.to_bits()));
                        let pred = bcx.ins().fcmp(FloatCC::OrderedNotEqual, a, z);
                        let one = bcx.ins().iconst(types::I64, 1);
                        let zero = bcx.ins().iconst(types::I64, 0);
                        bcx.ins().select(pred, one, zero)
                    }
                };
                let call = bcx.ins().call(fr, &[a_int]);
                stack.push((*bcx.inst_results(call).first()?, JitTy::Int));
            }
            Op::Pop => {
                stack.pop()?;
            }
            Op::Dup => {
                let v = *stack.last()?;
                stack.push(v);
            }
            Op::Swap => {
                let (b, tb) = stack.pop()?;
                let (a, ta) = stack.pop()?;
                if ta != tb {
                    return None;
                }
                stack.push((b, tb));
                stack.push((a, ta));
            }
            Op::Rot => {
                let (c, tc) = stack.pop()?;
                let (b, tb) = stack.pop()?;
                let (a, ta) = stack.pop()?;
                if ta != tb || tb != tc {
                    return None;
                }
                stack.push((b, tb));
                stack.push((c, tc));
                stack.push((a, ta));
            }
            Op::BitXor => {
                let (b, tb) = stack.pop()?;
                let (a, ta) = stack.pop()?;
                if ta != JitTy::Int || tb != JitTy::Int {
                    return None;
                }
                stack.push((bcx.ins().bxor(a, b), JitTy::Int));
            }
            Op::BitAnd => {
                let (b, tb) = stack.pop()?;
                let (a, ta) = stack.pop()?;
                if ta != JitTy::Int || tb != JitTy::Int {
                    return None;
                }
                stack.push((bcx.ins().band(a, b), JitTy::Int));
            }
            Op::BitOr => {
                let (b, tb) = stack.pop()?;
                let (a, ta) = stack.pop()?;
                if ta != JitTy::Int || tb != JitTy::Int {
                    return None;
                }
                stack.push((bcx.ins().bor(a, b), JitTy::Int));
            }
            Op::BitNot => {
                let (a, ty) = stack.pop()?;
                if ty != JitTy::Int {
                    return None;
                }
                let ones = bcx.ins().iconst(types::I64, -1);
                stack.push((bcx.ins().bxor(a, ones), JitTy::Int));
            }
            Op::Shl => {
                let (b, tb) = stack.pop()?;
                let (a, ta) = stack.pop()?;
                if ta != JitTy::Int || tb != JitTy::Int {
                    return None;
                }
                let mask = bcx.ins().iconst(types::I64, 63);
                let mb = bcx.ins().band(b, mask);
                stack.push((bcx.ins().ishl(a, mb), JitTy::Int));
            }
            Op::Shr => {
                let (b, tb) = stack.pop()?;
                let (a, ta) = stack.pop()?;
                if ta != JitTy::Int || tb != JitTy::Int {
                    return None;
                }
                let mask = bcx.ins().iconst(types::I64, 63);
                let mb = bcx.ins().band(b, mask);
                stack.push((bcx.ins().sshr(a, mb), JitTy::Int));
            }
            Op::GetSlot(slot) => {
                let base = slot_base?;
                let off = (*slot as i32) * 8;
                stack.push((
                    bcx.ins().load(types::I64, MemFlags::trusted(), base, off),
                    JitTy::Int,
                ));
            }
            Op::SetSlot(slot) => {
                let base = slot_base?;
                let (v, ty) = stack.pop()?;
                let v = scalar_store_i64(bcx, v, ty);
                bcx.ins()
                    .store(MemFlags::trusted(), v, base, (*slot as i32) * 8);
            }
            Op::PreIncSlot(slot) => {
                let base = slot_base?;
                let off = (*slot as i32) * 8;
                let old = bcx.ins().load(types::I64, MemFlags::trusted(), base, off);
                let one = bcx.ins().iconst(types::I64, 1);
                let new = bcx.ins().iadd(old, one);
                bcx.ins().store(MemFlags::trusted(), new, base, off);
                stack.push((new, JitTy::Int));
            }
            Op::PreIncSlotVoid(slot) => {
                let base = slot_base?;
                let off = (*slot as i32) * 8;
                let old = bcx.ins().load(types::I64, MemFlags::trusted(), base, off);
                let one = bcx.ins().iconst(types::I64, 1);
                let new = bcx.ins().iadd(old, one);
                bcx.ins().store(MemFlags::trusted(), new, base, off);
            }
            Op::AddAssignSlotVoid(a_slot, b_slot) => {
                let base = slot_base?;
                let va =
                    bcx.ins()
                        .load(types::I64, MemFlags::trusted(), base, (*a_slot as i32) * 8);
                let vb =
                    bcx.ins()
                        .load(types::I64, MemFlags::trusted(), base, (*b_slot as i32) * 8);
                let sum = bcx.ins().iadd(va, vb);
                bcx.ins()
                    .store(MemFlags::trusted(), sum, base, (*a_slot as i32) * 8);
            }
            _ => return None,
        }
        Some(())
    }

    // ── Linear JIT compilation ──

    /// Compile a straight-line sequence of ops to native code.
    pub(crate) fn compile_linear(chunk: &Chunk) -> Option<CompiledLinear> {
        let seq = &chunk.ops;
        if !validate_linear_seq(seq, &chunk.constants) {
            return None;
        }
        let ret_cell = linear_result_cell(seq, &chunk.constants)?;
        let ret_ty = cell_to_jit_ty(ret_cell);
        let need_slots = needs_slots(seq);
        let mut module = new_jit_module()?;

        let needs_pow = seq.iter().any(|o| matches!(o, Op::Pow));
        let pow_i64_id = if needs_pow {
            let mut ps = module.make_signature();
            ps.params.push(AbiParam::new(types::I64));
            ps.params.push(AbiParam::new(types::I64));
            ps.returns.push(AbiParam::new(types::I64));
            Some(
                module
                    .declare_function("fusevm_jit_pow_i64", Linkage::Import, &ps)
                    .ok()?,
            )
        } else {
            None
        };
        let pow_f64_id = if needs_pow {
            let mut ps = module.make_signature();
            ps.params.push(AbiParam::new(types::F64));
            ps.params.push(AbiParam::new(types::F64));
            ps.returns.push(AbiParam::new(types::F64));
            Some(
                module
                    .declare_function("fusevm_jit_pow_f64", Linkage::Import, &ps)
                    .ok()?,
            )
        } else {
            None
        };
        let needs_fmod = seq.iter().any(|o| matches!(o, Op::Mod));
        let fmod_f64_id = if needs_fmod {
            let mut ps = module.make_signature();
            ps.params.push(AbiParam::new(types::F64));
            ps.params.push(AbiParam::new(types::F64));
            ps.returns.push(AbiParam::new(types::F64));
            Some(
                module
                    .declare_function("fusevm_jit_fmod_f64", Linkage::Import, &ps)
                    .ok()?,
            )
        } else {
            None
        };
        let needs_lognot = seq.iter().any(|o| matches!(o, Op::LogNot));
        let lognot_id = if needs_lognot {
            let mut ps = module.make_signature();
            ps.params.push(AbiParam::new(types::I64));
            ps.returns.push(AbiParam::new(types::I64));
            Some(
                module
                    .declare_function("fusevm_jit_lognot_i64", Linkage::Import, &ps)
                    .ok()?,
            )
        } else {
            None
        };

        let ptr_ty = module.target_config().pointer_type();
        let mut sig = module.make_signature();
        if need_slots {
            sig.params.push(AbiParam::new(ptr_ty));
        }
        sig.returns.push(AbiParam::new(match ret_ty {
            JitTy::Int => types::I64,
            JitTy::Float => types::F64,
        }));

        let fid = module
            .declare_function("linear", Linkage::Local, &sig)
            .ok()?;
        let mut ctx = module.make_context();
        ctx.func.signature = sig;
        ctx.func.name = UserFuncName::user(0, fid.as_u32());

        let mut fctx = FunctionBuilderContext::new();
        {
            let mut bcx = FunctionBuilder::new(&mut ctx.func, &mut fctx);
            let entry = bcx.create_block();
            bcx.append_block_params_for_function_params(entry);
            bcx.switch_to_block(entry);

            let slot_base = if need_slots {
                Some(bcx.block_params(entry)[0])
            } else {
                None
            };

            let pow_i64_ref = pow_i64_id.map(|pid| module.declare_func_in_func(pid, bcx.func));
            let pow_f64_ref = pow_f64_id.map(|pid| module.declare_func_in_func(pid, bcx.func));
            let fmod_f64_ref = fmod_f64_id.map(|pid| module.declare_func_in_func(pid, bcx.func));
            let lognot_ref = lognot_id.map(|lid| module.declare_func_in_func(lid, bcx.func));

            let mut stack: Vec<(cranelift_codegen::ir::Value, JitTy)> = Vec::with_capacity(32);
            for op in seq {
                emit_data_op(
                    &mut bcx,
                    op,
                    &mut stack,
                    slot_base,
                    pow_i64_ref,
                    pow_f64_ref,
                    fmod_f64_ref,
                    lognot_ref,
                    &chunk.constants,
                )?;
            }
            let (v, ty) = stack.pop()?;
            let ret_v = match (ret_ty, ty) {
                (JitTy::Int, JitTy::Int) | (JitTy::Float, JitTy::Float) => v,
                (JitTy::Float, JitTy::Int) => i64_to_f64(&mut bcx, v),
                (JitTy::Int, JitTy::Float) => f64_to_i64_trunc(&mut bcx, v),
            };
            bcx.ins().return_(&[ret_v]);
            bcx.seal_all_blocks();
            bcx.finalize();
        }

        module.define_function(fid, &mut ctx).ok()?;
        module.clear_context(&mut ctx);
        module.finalize_definitions().ok()?;
        let ptr = module.get_finalized_function(fid);
        let run = match (need_slots, ret_ty) {
            (false, JitTy::Int) => {
                LinearRun::Nullary(unsafe { std::mem::transmute::<*const u8, LinearFn0>(ptr) })
            }
            (false, JitTy::Float) => {
                LinearRun::NullaryF(unsafe { std::mem::transmute::<*const u8, LinearFn0F>(ptr) })
            }
            (true, JitTy::Int) => {
                LinearRun::Slots(unsafe { std::mem::transmute::<*const u8, LinearFnSlots>(ptr) })
            }
            (true, JitTy::Float) => {
                LinearRun::SlotsF(unsafe { std::mem::transmute::<*const u8, LinearFnSlotsF>(ptr) })
            }
        };
        Some(CompiledLinear { module, run })
    }

    // ── Linear JIT cache (per-thread, lock-free) ──

    thread_local! {
        static LINEAR_CACHE_TLS: std::cell::RefCell<HashMap<u64, Box<CompiledLinear>>> =
            std::cell::RefCell::new(HashMap::new());
    }

    /// Try to JIT-compile and run a chunk's ops as a linear sequence.
    /// Returns `Some(Value)` on success, `None` if the chunk isn't eligible.
    pub(crate) fn try_run_linear(chunk: &Chunk, slots: &[i64]) -> Option<FuseValue> {
        let key = chunk.op_hash;
        let slot_ptr = if slots.is_empty() {
            std::ptr::null()
        } else {
            slots.as_ptr()
        };

        // Cache hit: invoke and return
        let cached = LINEAR_CACHE_TLS.with(|cache_cell| {
            let cache = cache_cell.borrow();
            cache.get(&key).map(|c| {
                let result = c.invoke(slot_ptr);
                c.result_to_value(result)
            })
        });
        if let Some(v) = cached {
            return Some(v);
        }

        // Cache miss: compile, invoke, store
        let compiled = compile_linear(chunk)?;
        let result = compiled.invoke(slot_ptr);
        let value = compiled.result_to_value(result);

        LINEAR_CACHE_TLS.with(|cache_cell| {
            cache_cell.borrow_mut().insert(key, Box::new(compiled));
        });

        Some(value)
    }

    /// Check if a chunk is eligible for linear JIT compilation.
    pub(crate) fn is_linear_eligible(chunk: &Chunk) -> bool {
        validate_linear_seq(&chunk.ops, &chunk.constants)
    }

    // ── Block JIT compilation ──

    use std::collections::BTreeSet;

    // Specialized block-JIT function pointer types.
    // Saves a register by omitting the slot pointer when the chunk doesn't use slots.
    type BlockFnSlotsI = unsafe extern "C" fn(*mut i64) -> i64;
    type BlockFnSlotsF = unsafe extern "C" fn(*mut i64) -> f64;
    type BlockFnNoSlotsI = unsafe extern "C" fn() -> i64;
    type BlockFnNoSlotsF = unsafe extern "C" fn() -> f64;

    #[allow(dead_code)] // Variants reserved for future signature specialization.
    pub(crate) enum BlockRun {
        SlotsI(BlockFnSlotsI),
        SlotsF(BlockFnSlotsF),
        NoSlotsI(BlockFnNoSlotsI),
        NoSlotsF(BlockFnNoSlotsF),
    }

    pub(crate) struct CompiledBlock {
        #[allow(dead_code)]
        module: JITModule,
        run: BlockRun,
    }

    impl CompiledBlock {
        /// Invoke and return the result as i64 (truncating float results).
        pub(crate) fn invoke(&self, slots: &mut [i64]) -> i64 {
            let ptr = if slots.is_empty() {
                std::ptr::null_mut()
            } else {
                slots.as_mut_ptr()
            };
            match &self.run {
                BlockRun::SlotsI(f) => unsafe { f(ptr) },
                BlockRun::NoSlotsI(f) => unsafe { f() },
                BlockRun::SlotsF(f) => (unsafe { f(ptr) }) as i64,
                BlockRun::NoSlotsF(f) => (unsafe { f() }) as i64,
            }
        }
    }

    fn find_leaders(ops: &[Op]) -> BTreeSet<usize> {
        let mut leaders = BTreeSet::new();
        if ops.is_empty() {
            return leaders;
        }
        leaders.insert(0);
        for (ip, op) in ops.iter().enumerate() {
            match op {
                Op::Jump(t) => {
                    leaders.insert(*t);
                    if ip + 1 < ops.len() {
                        leaders.insert(ip + 1);
                    }
                }
                Op::JumpIfTrue(t) | Op::JumpIfFalse(t) => {
                    leaders.insert(*t);
                    if ip + 1 < ops.len() {
                        leaders.insert(ip + 1);
                    }
                }
                Op::SlotLtIntJumpIfFalse(_, _, t) | Op::SlotIncLtIntJumpBack(_, _, t) => {
                    leaders.insert(*t);
                    if ip + 1 < ops.len() {
                        leaders.insert(ip + 1);
                    }
                }
                Op::AccumSumLoop(_, _, _) => {
                    if ip + 1 < ops.len() {
                        leaders.insert(ip + 1);
                    }
                }
                _ => {}
            }
        }
        leaders
    }

    fn is_block_eligible_op(op: &Op) -> bool {
        matches!(
            op,
            Op::Nop
                | Op::LoadInt(_)
                | Op::LoadFloat(_)
                | Op::LoadConst(_)
                | Op::LoadTrue
                | Op::LoadFalse
                | Op::Pop
                | Op::Dup
                | Op::Swap
                | Op::Rot
                | Op::Add
                | Op::Sub
                | Op::Mul
                | Op::Div
                | Op::Mod
                | Op::Pow
                | Op::Negate
                | Op::Inc
                | Op::Dec
                | Op::NumEq
                | Op::NumNe
                | Op::NumLt
                | Op::NumGt
                | Op::NumLe
                | Op::NumGe
                | Op::Spaceship
                | Op::BitAnd
                | Op::BitOr
                | Op::BitXor
                | Op::BitNot
                | Op::Shl
                | Op::Shr
                | Op::LogNot
                | Op::GetSlot(_)
                | Op::SetSlot(_)
                | Op::PreIncSlot(_)
                | Op::PreIncSlotVoid(_)
                | Op::AddAssignSlotVoid(_, _)
                | Op::Jump(_)
                | Op::JumpIfTrue(_)
                | Op::JumpIfFalse(_)
                | Op::SlotLtIntJumpIfFalse(_, _, _)
                | Op::SlotIncLtIntJumpBack(_, _, _)
                | Op::AccumSumLoop(_, _, _)
                | Op::PushFrame
                | Op::PopFrame
        )
    }

    thread_local! {
        /// Per-thread cache of block-JIT eligibility decisions. Keyed on
        /// `chunk.op_hash` so the same chunk's eligibility is decided once
        /// then reused across `is_block_eligible` calls (notably from
        /// `VM::run`'s phase-10 auto-dispatch path, which would otherwise
        /// linear-scan the ops on every invocation).
        static BLOCK_ELIGIBLE_TLS: RefCell<HashMap<u64, bool>> =
            RefCell::new(HashMap::new());
    }

    #[inline]
    pub(crate) fn is_block_eligible(chunk: &Chunk) -> bool {
        // Fast path: cached decision.
        if let Some(hit) = BLOCK_ELIGIBLE_TLS.with(|c| c.borrow().get(&chunk.op_hash).copied()) {
            return hit;
        }
        // Slow path: scan ops, cache result.
        let ops = &chunk.ops;
        let result = !ops.is_empty() && ops.iter().all(is_block_eligible_op);
        BLOCK_ELIGIBLE_TLS.with(|c| c.borrow_mut().insert(chunk.op_hash, result));
        result
    }

    /// Find the largest contiguous JIT-eligible region in a chunk.
    /// A region is closed (all jump targets within the region must also be in
    /// the region — bytecode-level jumps to outside the region disqualify it).
    ///
    /// Returns `(start, end)` op indices (end exclusive), or None if no
    /// eligible region of useful size exists. Useful size = at least 8 ops.
    pub(crate) fn find_jit_region(ops: &[Op]) -> Option<(usize, usize)> {
        let mut best: Option<(usize, usize)> = None;
        let mut start: Option<usize> = None;

        for (ip, op) in ops.iter().enumerate() {
            if is_block_eligible_op(op) {
                if start.is_none() {
                    start = Some(ip);
                }
            } else if let Some(s) = start.take() {
                let len = ip - s;
                if len >= 4 && best.map_or(true, |(bs, be)| len > be - bs) {
                    best = Some((s, ip));
                }
            }
        }
        // Tail region
        if let Some(s) = start {
            let len = ops.len() - s;
            if len >= 4 && best.map_or(true, |(bs, be)| len > be - bs) {
                best = Some((s, ops.len()));
            }
        }

        // Verify all jumps within the region target inside the region.
        // Rebase jumps locally if so; otherwise reject.
        let (s, e) = best?;
        for op in &ops[s..e] {
            match op {
                Op::Jump(t)
                | Op::JumpIfTrue(t)
                | Op::JumpIfFalse(t)
                | Op::SlotLtIntJumpIfFalse(_, _, t)
                | Op::SlotIncLtIntJumpBack(_, _, t) => {
                    if *t < s || *t >= e {
                        return None;
                    }
                }
                _ => {}
            }
        }
        Some((s, e))
    }

    /// Extract a JIT region as a standalone sub-chunk with rebased jump targets.
    /// The returned chunk has its op_hash recomputed.
    pub(crate) fn extract_region(chunk: &Chunk, start: usize, end: usize) -> Chunk {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut sub = Chunk {
            ops: Vec::with_capacity(end - start),
            constants: chunk.constants.clone(),
            names: chunk.names.clone(),
            lines: chunk.lines[start..end].to_vec(),
            sub_entries: Vec::new(),
            block_ranges: Vec::new(),
            sub_chunks: Vec::new(),
            source: chunk.source.clone(),
            op_hash: 0,
        };
        for op in &chunk.ops[start..end] {
            // Rebase jump targets to be local to the sub-chunk.
            let new_op = match op {
                Op::Jump(t) => Op::Jump(t - start),
                Op::JumpIfTrue(t) => Op::JumpIfTrue(t - start),
                Op::JumpIfFalse(t) => Op::JumpIfFalse(t - start),
                Op::SlotLtIntJumpIfFalse(s, l, t) => Op::SlotLtIntJumpIfFalse(*s, *l, t - start),
                Op::SlotIncLtIntJumpBack(s, l, t) => Op::SlotIncLtIntJumpBack(*s, *l, t - start),
                other => other.clone(),
            };
            sub.ops.push(new_op);
        }
        let mut h = DefaultHasher::new();
        sub.ops.hash(&mut h);
        sub.constants.hash(&mut h);
        sub.op_hash = h.finish();
        sub
    }

    /// Collect all slot indices referenced by the chunk for promotion.
    fn collect_slots(ops: &[Op]) -> Vec<u16> {
        let mut slots = std::collections::BTreeSet::new();
        for op in ops {
            match op {
                Op::GetSlot(s)
                | Op::SetSlot(s)
                | Op::PreIncSlot(s)
                | Op::PreIncSlotVoid(s)
                | Op::SlotLtIntJumpIfFalse(s, _, _)
                | Op::SlotIncLtIntJumpBack(s, _, _) => {
                    slots.insert(*s);
                }
                Op::AddAssignSlotVoid(a, b) => {
                    slots.insert(*a);
                    slots.insert(*b);
                }
                Op::AccumSumLoop(s, i, _) => {
                    slots.insert(*s);
                    slots.insert(*i);
                }
                _ => {}
            }
        }
        slots.into_iter().collect()
    }

    fn cond_to_i1(bcx: &mut FunctionBuilder, v: Value, ty: JitTy) -> cranelift_codegen::ir::Value {
        match ty {
            JitTy::Int => bcx.ins().icmp_imm(IntCC::NotEqual, v, 0),
            JitTy::Float => {
                let z = bcx.ins().f64const(Ieee64::with_bits(0.0f64.to_bits()));
                bcx.ins().fcmp(FloatCC::OrderedNotEqual, v, z)
            }
        }
    }

    pub(crate) fn compile_block(chunk: &Chunk) -> Option<CompiledBlock> {
        let ops = &chunk.ops;
        if !is_block_eligible(chunk) {
            return None;
        }

        let leaders = find_leaders(ops);
        let leader_vec: Vec<usize> = leaders.iter().copied().collect();
        let mut module = new_jit_module()?;

        // Declare external helpers
        let needs_pow = ops.iter().any(|o| matches!(o, Op::Pow));
        let pow_i64_id = if needs_pow {
            let mut ps = module.make_signature();
            ps.params.push(AbiParam::new(types::I64));
            ps.params.push(AbiParam::new(types::I64));
            ps.returns.push(AbiParam::new(types::I64));
            Some(
                module
                    .declare_function("fusevm_jit_pow_i64", Linkage::Import, &ps)
                    .ok()?,
            )
        } else {
            None
        };
        let pow_f64_id = if needs_pow {
            let mut ps = module.make_signature();
            ps.params.push(AbiParam::new(types::F64));
            ps.params.push(AbiParam::new(types::F64));
            ps.returns.push(AbiParam::new(types::F64));
            Some(
                module
                    .declare_function("fusevm_jit_pow_f64", Linkage::Import, &ps)
                    .ok()?,
            )
        } else {
            None
        };
        let needs_fmod = ops.iter().any(|o| matches!(o, Op::Mod));
        let fmod_f64_id = if needs_fmod {
            let mut ps = module.make_signature();
            ps.params.push(AbiParam::new(types::F64));
            ps.params.push(AbiParam::new(types::F64));
            ps.returns.push(AbiParam::new(types::F64));
            Some(
                module
                    .declare_function("fusevm_jit_fmod_f64", Linkage::Import, &ps)
                    .ok()?,
            )
        } else {
            None
        };
        let needs_lognot = ops.iter().any(|o| matches!(o, Op::LogNot));
        let lognot_id = if needs_lognot {
            let mut ps = module.make_signature();
            ps.params.push(AbiParam::new(types::I64));
            ps.returns.push(AbiParam::new(types::I64));
            Some(
                module
                    .declare_function("fusevm_jit_lognot_i64", Linkage::Import, &ps)
                    .ok()?,
            )
        } else {
            None
        };

        let ptr_ty = module.target_config().pointer_type();
        let mut sig = module.make_signature();
        sig.params.push(AbiParam::new(ptr_ty)); // *mut i64 slots
        sig.returns.push(AbiParam::new(types::I64)); // result

        let fid = module
            .declare_function("block_jit", Linkage::Local, &sig)
            .ok()?;
        let mut ctx = module.make_context();
        ctx.func.signature = sig;
        ctx.func.name = UserFuncName::user(0, fid.as_u32());

        let mut fctx = FunctionBuilderContext::new();
        {
            let mut bcx = FunctionBuilder::new(&mut ctx.func, &mut fctx);

            // Create Cranelift blocks for each bytecode leader
            let mut block_map: HashMap<usize, cranelift_codegen::ir::Block> = HashMap::new();
            for &leader_ip in &leader_vec {
                block_map.insert(leader_ip, bcx.create_block());
            }

            // Entry block setup
            let entry = block_map[&0];
            bcx.append_block_params_for_function_params(entry);
            bcx.switch_to_block(entry);
            let slot_base = bcx.block_params(entry)[0];

            // ── Slot promotion: declare a Cranelift Variable per used slot, ──
            // ── load each from the slot pointer at entry. After this, all   ──
            // ── slot ops use Variables (register-allocated by Cranelift).   ──
            let used_slots = collect_slots(ops);
            let mut slot_vars: HashMap<u16, Variable> = HashMap::new();
            for &slot in &used_slots {
                let var = bcx.declare_var(types::I64);
                let val = bcx.ins().load(
                    types::I64,
                    MemFlags::trusted(),
                    slot_base,
                    (slot as i32) * 8,
                );
                bcx.def_var(var, val);
                slot_vars.insert(slot, var);
            }

            let pow_i64_ref = pow_i64_id.map(|pid| module.declare_func_in_func(pid, bcx.func));
            let pow_f64_ref = pow_f64_id.map(|pid| module.declare_func_in_func(pid, bcx.func));
            let fmod_f64_ref = fmod_f64_id.map(|pid| module.declare_func_in_func(pid, bcx.func));
            let lognot_ref = lognot_id.map(|lid| module.declare_func_in_func(lid, bcx.func));

            // Process each basic block
            let mut block_terminated = false;

            for (block_idx, &leader_ip) in leader_vec.iter().enumerate() {
                let block_end = if block_idx + 1 < leader_vec.len() {
                    leader_vec[block_idx + 1]
                } else {
                    ops.len()
                };

                // Switch to this block (unless it's the entry which we already started)
                if leader_ip > 0 {
                    if !block_terminated {
                        // Previous block fell through — emit jump
                        bcx.ins().jump(block_map[&leader_ip], &[]);
                    }
                    bcx.switch_to_block(block_map[&leader_ip]);
                }

                let mut stack: Vec<(cranelift_codegen::ir::Value, JitTy)> = Vec::new();
                block_terminated = false;

                for ip in leader_ip..block_end {
                    let op = &ops[ip];
                    match op {
                        Op::PushFrame | Op::PopFrame | Op::Nop => {}

                        Op::Jump(target) => {
                            let target_block = *block_map.get(target)?;
                            bcx.ins().jump(target_block, &[]);
                            block_terminated = true;
                        }

                        Op::JumpIfTrue(target) => {
                            let (cond, ty) = stack.pop()?;
                            let pred = cond_to_i1(&mut bcx, cond, ty);
                            let target_block = *block_map.get(target)?;
                            let fall = if ip + 1 < ops.len() {
                                *block_map.get(&(ip + 1))?
                            } else {
                                return None;
                            };
                            bcx.ins().brif(pred, target_block, &[], fall, &[]);
                            block_terminated = true;
                        }

                        Op::JumpIfFalse(target) => {
                            let (cond, ty) = stack.pop()?;
                            let pred = cond_to_i1(&mut bcx, cond, ty);
                            let target_block = *block_map.get(target)?;
                            let fall = if ip + 1 < ops.len() {
                                *block_map.get(&(ip + 1))?
                            } else {
                                return None;
                            };
                            bcx.ins().brif(pred, fall, &[], target_block, &[]);
                            block_terminated = true;
                        }

                        // ── Slot data ops: use Variables (register-allocated) ──
                        Op::GetSlot(slot) => {
                            let var = *slot_vars.get(slot)?;
                            let v = bcx.use_var(var);
                            stack.push((v, JitTy::Int));
                        }
                        Op::SetSlot(slot) => {
                            let var = *slot_vars.get(slot)?;
                            let (v, ty) = stack.pop()?;
                            let v_i = scalar_store_i64(&mut bcx, v, ty);
                            bcx.def_var(var, v_i);
                        }
                        Op::PreIncSlot(slot) => {
                            let var = *slot_vars.get(slot)?;
                            let old = bcx.use_var(var);
                            let one = bcx.ins().iconst(types::I64, 1);
                            let new = bcx.ins().iadd(old, one);
                            bcx.def_var(var, new);
                            stack.push((new, JitTy::Int));
                        }
                        Op::PreIncSlotVoid(slot) => {
                            let var = *slot_vars.get(slot)?;
                            let old = bcx.use_var(var);
                            let one = bcx.ins().iconst(types::I64, 1);
                            let new = bcx.ins().iadd(old, one);
                            bcx.def_var(var, new);
                        }
                        Op::AddAssignSlotVoid(a_slot, b_slot) => {
                            let a_var = *slot_vars.get(a_slot)?;
                            let b_var = *slot_vars.get(b_slot)?;
                            let va = bcx.use_var(a_var);
                            let vb = bcx.use_var(b_var);
                            let sum = bcx.ins().iadd(va, vb);
                            bcx.def_var(a_var, sum);
                        }

                        Op::SlotLtIntJumpIfFalse(slot, limit, target) => {
                            let var = *slot_vars.get(slot)?;
                            let val = bcx.use_var(var);
                            let limit_v = bcx.ins().iconst(types::I64, *limit as i64);
                            let is_lt = bcx.ins().icmp(IntCC::SignedLessThan, val, limit_v);
                            let target_block = *block_map.get(target)?;
                            let fall = if ip + 1 < ops.len() {
                                *block_map.get(&(ip + 1))?
                            } else {
                                return None;
                            };
                            // if >= limit, jump to target; otherwise fall through
                            bcx.ins().brif(is_lt, fall, &[], target_block, &[]);
                            block_terminated = true;
                        }

                        Op::SlotIncLtIntJumpBack(slot, limit, target) => {
                            let var = *slot_vars.get(slot)?;
                            let old = bcx.use_var(var);
                            let one = bcx.ins().iconst(types::I64, 1);
                            let new = bcx.ins().iadd(old, one);
                            bcx.def_var(var, new);
                            let limit_v = bcx.ins().iconst(types::I64, *limit as i64);
                            let is_lt = bcx.ins().icmp(IntCC::SignedLessThan, new, limit_v);
                            let target_block = *block_map.get(target)?;
                            let fall = if ip + 1 < ops.len() {
                                *block_map.get(&(ip + 1))?
                            } else {
                                return None;
                            };
                            bcx.ins().brif(is_lt, target_block, &[], fall, &[]);
                            block_terminated = true;
                        }

                        Op::AccumSumLoop(sum_slot, i_slot, limit) => {
                            let sum_var = *slot_vars.get(sum_slot)?;
                            let i_var = *slot_vars.get(i_slot)?;
                            let sum_init = bcx.use_var(sum_var);
                            let i_init = bcx.use_var(i_var);
                            let limit_v = bcx.ins().iconst(types::I64, *limit as i64);

                            let loop_hdr = bcx.create_block();
                            let loop_body = bcx.create_block();
                            let loop_exit = bcx.create_block();

                            bcx.ins().jump(
                                loop_hdr,
                                &[BlockArg::Value(sum_init), BlockArg::Value(i_init)],
                            );

                            // Loop header: check i < limit
                            bcx.switch_to_block(loop_hdr);
                            bcx.append_block_param(loop_hdr, types::I64); // sum
                            bcx.append_block_param(loop_hdr, types::I64); // i
                            let sum_p = bcx.block_params(loop_hdr)[0];
                            let i_p = bcx.block_params(loop_hdr)[1];
                            let cond = bcx.ins().icmp(IntCC::SignedLessThan, i_p, limit_v);
                            bcx.ins().brif(
                                cond,
                                loop_body,
                                &[],
                                loop_exit,
                                &[BlockArg::Value(sum_p), BlockArg::Value(i_p)],
                            );

                            // Loop body: sum += i; i++
                            bcx.switch_to_block(loop_body);
                            let new_sum = bcx.ins().iadd(sum_p, i_p);
                            let one = bcx.ins().iconst(types::I64, 1);
                            let new_i = bcx.ins().iadd(i_p, one);
                            bcx.ins().jump(
                                loop_hdr,
                                &[BlockArg::Value(new_sum), BlockArg::Value(new_i)],
                            );

                            // Loop exit: write back to slot variables
                            bcx.switch_to_block(loop_exit);
                            bcx.append_block_param(loop_exit, types::I64);
                            bcx.append_block_param(loop_exit, types::I64);
                            let final_sum = bcx.block_params(loop_exit)[0];
                            let final_i = bcx.block_params(loop_exit)[1];
                            bcx.def_var(sum_var, final_sum);
                            bcx.def_var(i_var, final_i);
                        }

                        // Data ops — delegate to emit_data_op
                        _ => {
                            emit_data_op(
                                &mut bcx,
                                op,
                                &mut stack,
                                Some(slot_base),
                                pow_i64_ref,
                                pow_f64_ref,
                                fmod_f64_ref,
                                lognot_ref,
                                &chunk.constants,
                            )?;
                        }
                    }
                }

                // End of this basic block — if not terminated, handle fallthrough or return
                if !block_terminated {
                    if block_end == ops.len() {
                        // Final block: spill promoted slot variables to memory, then return
                        let ret_val = if let Some((v, ty)) = stack.pop() {
                            scalar_store_i64(&mut bcx, v, ty)
                        } else {
                            bcx.ins().iconst(types::I64, 0)
                        };
                        // Write all slot Variables back to the slot pointer
                        // (caller observes the final state of the slot array)
                        for (&slot, &var) in &slot_vars {
                            let val = bcx.use_var(var);
                            bcx.ins()
                                .store(MemFlags::trusted(), val, slot_base, (slot as i32) * 8);
                        }
                        bcx.ins().return_(&[ret_val]);
                        block_terminated = true;
                    }
                    // Non-final unterminated blocks get a fallthrough jump
                    // at the top of the next iteration
                }
            }

            bcx.seal_all_blocks();
            bcx.finalize();
        }

        module.define_function(fid, &mut ctx).ok()?;
        module.clear_context(&mut ctx);
        module.finalize_definitions().ok()?;
        let ptr = module.get_finalized_function(fid);
        // Currently always SlotsI: signature is fn(*mut i64) -> i64.
        // Future specialization: detect no-slots chunks → NoSlotsI;
        // detect float-returning chunks → SlotsF/NoSlotsF.
        let run = BlockRun::SlotsI(unsafe { std::mem::transmute::<*const u8, BlockFnSlotsI>(ptr) });
        Some(CompiledBlock { module, run })
    }

    // ── Block JIT cache (per-thread, lock-free) ──
    //
    // Each thread has its own cache — JITModule is not Send anyway, and
    // VMs are single-threaded per instance. No mutex overhead per call.
    // Hot-counts are also tracked here for tiered compilation.

    use std::cell::RefCell;

    struct BlockCacheEntry {
        /// Number of times we've been asked to run this chunk.
        hot_count: u32,
        /// Compiled native code (set after threshold).
        compiled: Option<Box<CompiledBlock>>,
    }

    thread_local! {
        static BLOCK_CACHE_TLS: RefCell<HashMap<u64, BlockCacheEntry>> =
            RefCell::new(HashMap::new());
    }

    /// Tiered compilation threshold — JIT after N interpreter runs.
    /// Below this, `try_run_block` returns None so the caller falls back
    /// to the interpreter. This avoids paying compile cost for one-shot chunks.
    const HOT_THRESHOLD: u32 = 10;

    /// Try to JIT-compile and run a chunk via the block JIT.
    /// Returns `Some(result_i64)` on success, `None` if ineligible OR not yet hot.
    pub(crate) fn try_run_block(chunk: &Chunk, slots: &mut [i64]) -> Option<i64> {
        try_run_block_inner(chunk, slots, HOT_THRESHOLD)
    }

    /// Like `try_run_block` but compiles immediately (no warmup). For tests
    /// and synthetic benchmarks where you want to skip the tiered policy.
    pub(crate) fn try_run_block_eager(chunk: &Chunk, slots: &mut [i64]) -> Option<i64> {
        try_run_block_inner(chunk, slots, 0)
    }

    fn try_run_block_inner(chunk: &Chunk, slots: &mut [i64], threshold: u32) -> Option<i64> {
        let key = chunk.op_hash;

        BLOCK_CACHE_TLS.with(|cache_cell| {
            let mut cache = cache_cell.borrow_mut();
            let entry = cache.entry(key).or_insert(BlockCacheEntry {
                hot_count: 0,
                compiled: None,
            });

            if let Some(ref compiled) = entry.compiled {
                return Some(compiled.invoke(slots));
            }

            entry.hot_count = entry.hot_count.saturating_add(1);
            if entry.hot_count <= threshold {
                return None; // not hot yet — caller falls back to interpreter
            }

            // Compile on threshold cross
            let compiled = compile_block(chunk)?;
            let result = compiled.invoke(slots);
            entry.compiled = Some(Box::new(compiled));
            Some(result)
        })
    }

    // ── Tracing JIT (Tier 2 — hot paths through control flow) ──
    //
    // Tracing JIT records the actual hot path through bytecode, anchored at
    // backward branches (loop headers). Once a header crosses TRACE_THRESHOLD
    // backedge counts, the recorder is armed; on the next iteration through the
    // header it captures every executed op until execution returns to the
    // anchor IP. The captured straight-line sequence (the "trace") is compiled
    // to native code.
    //
    // # Phase 1 restrictions
    // - Loop body must consist entirely of block-JIT-eligible ops
    // - All slots referenced by the trace must hold Int values at trace entry
    //   (entry guard enforced by interpreter, not by compiled code)
    // - The trace closes on a backward Jump/JumpIfTrue/JumpIfFalse to the anchor
    // - No other backward jumps allowed (single-loop only)
    // - No internal forward jumps that escape the loop body (phase 2)
    // - No Call/Return inside the trace (phase 2)
    //
    // # Side-exit ABI
    // Trace fns: `unsafe extern "C" fn(*mut i64) -> i64`
    // Returns the IP at which the interpreter should resume:
    // - The fallthrough IP on normal loop exit (condition false)
    // - The deopt IP on internal guard failure (phase 2 — currently unused)
    //
    // # Cache key
    // (chunk.op_hash, anchor_ip) — different headers in the same chunk get
    // different traces. Per-thread, lock-free.

    /// Trace fn signature.
    ///
    /// - `slots` — pointer to the caller frame's i64 slot array.
    /// - `deopt_info` — pointer to a `DeoptInfo` the trace populates on
    ///   every exit (normal loop fallthrough OR side-exit). The trace
    ///   writes `resume_ip` always; `frame_count` and `frames[0..frame_count]`
    ///   are populated only on callee-frame side-exits.
    /// - returns: the resume IP (also written to `*deopt_info`).
    type TraceFn = unsafe extern "C" fn(*mut i64, *mut super::DeoptInfo) -> i64;

    /// A compiled trace.
    pub(crate) struct CompiledTrace {
        #[allow(dead_code)]
        module: JITModule,
        run: TraceFn,
    }

    impl CompiledTrace {
        /// Invoke the trace. Returns the IP to resume interpretation at,
        /// and populates `deopt_info` for the caller to materialize any
        /// inlined frames.
        pub(crate) fn invoke(&self, slots: *mut i64, deopt_info: &mut super::DeoptInfo) -> i64 {
            unsafe { (self.run)(slots, deopt_info as *mut _) }
        }
    }

    /// Per-trace cache entry.
    struct TraceCacheEntry {
        /// Backedge counter for this header IP.
        hot_count: u32,
        /// Compiled trace, or None if not yet compiled / failed to compile.
        compiled: Option<Box<CompiledTrace>>,
        /// True if recording was attempted and aborted (don't retry).
        aborted: bool,
        /// Number of entry-guard deopts at runtime (slot type mismatches).
        deopt_count: u32,
        /// Number of mid-trace side-exits (a brif guard fired and the
        /// interpreter resumed at a non-fallthrough IP). Phase 6: blacklist
        /// the trace after `MAX_SIDE_EXITS` to avoid pathological
        /// trace+deopt+interpret cycles.
        side_exit_count: u32,
        /// True if blacklisted after too many deopts. Skip lookup entirely.
        blacklisted: bool,
        /// IP just past the loop body (where the loop falls through on exit).
        /// Set when the trace is compiled. Compared against the trace fn's
        /// returned resume_ip to detect mid-trace side-exits.
        fallthrough_ip: usize,
        /// Slot indices touched by the trace, with their expected entry types.
        /// Phase 1: all entries are JitTy::Int. The interpreter checks these
        /// before invoking the trace.
        slot_types: Vec<(u16, JitTy)>,
        /// Phase 7: original recording metadata retained for persistent
        /// cache export. None until the trace is successfully installed.
        saved_metadata: Option<super::TraceMetadata>,
    }

    thread_local! {
        static TRACE_CACHE_TLS: RefCell<HashMap<(u64, usize), TraceCacheEntry>> =
            RefCell::new(HashMap::new());
    }

    /// Maximum slot index a trace can reference (keeps slot_types small).
    /// Not user-tunable — fundamental to the deopt buffer layout.
    pub(crate) const MAX_TRACE_SLOT: u16 = 64;

    thread_local! {
        /// Per-thread tunable thresholds for the tracing JIT. Callers
        /// override via `JitCompiler::set_config`.
        static TRACE_CONFIG: RefCell<super::TraceJitConfig> =
            RefCell::new(super::TraceJitConfig::defaults());
    }

    // Field readers used internally by trace_lookup / is_trace_eligible /
    // recorder push hook. Other thresholds (max_inline_recursion,
    // max_trace_chain) are read directly via `get_config()` from the VM
    // because they're already in scope where VM holds the config-snapshot.
    #[inline(always)]
    fn cfg_trace_threshold() -> u32 {
        TRACE_CONFIG.with(|c| c.borrow().trace_threshold)
    }
    #[inline(always)]
    fn cfg_max_side_exits() -> u32 {
        TRACE_CONFIG.with(|c| c.borrow().max_side_exits)
    }
    #[inline(always)]
    pub(crate) fn cfg_max_trace_len() -> usize {
        TRACE_CONFIG.with(|c| c.borrow().max_trace_len)
    }
    #[inline]
    pub(crate) fn set_config(cfg: super::TraceJitConfig) {
        TRACE_CONFIG.with(|c| *c.borrow_mut() = cfg);
    }
    #[inline]
    pub(crate) fn get_config() -> super::TraceJitConfig {
        TRACE_CONFIG.with(|c| *c.borrow())
    }

    fn slot_kind_to_jitty(k: super::SlotKind) -> JitTy {
        match k {
            super::SlotKind::Int => JitTy::Int,
            super::SlotKind::Float => JitTy::Float,
        }
    }

    /// Consult the trace cache at a backward-branch site.
    ///
    /// `slot_kinds_at_anchor` is the runtime types of the frame's slots at the
    /// anchor IP. Used to (a) install the entry guard at compile time, (b) check
    /// it before invoking a previously compiled trace.
    pub(crate) fn trace_lookup(
        chunk: &Chunk,
        anchor_ip: usize,
        slots: *mut i64,
        slot_kinds_at_anchor: &[super::SlotKind],
        deopt_info: &mut super::DeoptInfo,
    ) -> super::TraceLookup {
        let key = (chunk.op_hash, anchor_ip);
        TRACE_CACHE_TLS.with(|cache_cell| {
            let mut cache = cache_cell.borrow_mut();

            // Hot path: existing entry. Avoid the always-allocated
            // `TraceCacheEntry` construction that `entry().or_insert(...)`
            // would force; only build a default entry on cold miss.
            if let Some(entry) = cache.get_mut(&key) {
                if entry.blacklisted || entry.aborted {
                    return super::TraceLookup::Skip;
                }
                if let Some(ref compiled) = entry.compiled {
                    // Entry guard: verify referenced slots still match
                    // recorded types.
                    for &(slot, ty) in &entry.slot_types {
                        let actual = slot_kinds_at_anchor
                            .get(slot as usize)
                            .copied()
                            .map(slot_kind_to_jitty)
                            .unwrap_or(JitTy::Int);
                        if actual != ty {
                            entry.deopt_count = entry.deopt_count.saturating_add(1);
                            if entry.deopt_count >= 5 {
                                entry.blacklisted = true;
                            }
                            return super::TraceLookup::GuardMismatch;
                        }
                    }
                    // Reset deopt info before each invocation so stale
                    // records from prior calls don't leak through.
                    deopt_info.resume_ip = 0;
                    deopt_info.frame_count = 0;
                    deopt_info.stack_count = 0;
                    let resume_ip = compiled.invoke(slots, deopt_info) as usize;
                    return super::TraceLookup::Ran { resume_ip };
                }
                // Not compiled yet — bump hot counter, decide arming.
                entry.hot_count = entry.hot_count.saturating_add(1);
                return if entry.hot_count >= cfg_trace_threshold() {
                    super::TraceLookup::StartRecording
                } else {
                    super::TraceLookup::NotHot
                };
            }

            // Cold path: first time seeing this anchor. Insert default
            // entry, return NotHot. The threshold check on subsequent
            // calls hits the hot path above.
            cache.insert(
                key,
                TraceCacheEntry {
                    hot_count: 1,
                    compiled: None,
                    aborted: false,
                    deopt_count: 0,
                    side_exit_count: 0,
                    blacklisted: false,
                    fallthrough_ip: 0,
                    slot_types: Vec::new(),
                    saved_metadata: None,
                },
            );
            if 1 >= cfg_trace_threshold() {
                super::TraceLookup::StartRecording
            } else {
                super::TraceLookup::NotHot
            }
        })
    }

    /// Mark a trace cache entry as aborted (recording failed).
    pub(crate) fn trace_abort(chunk: &Chunk, anchor_ip: usize) {
        let key = (chunk.op_hash, anchor_ip);
        TRACE_CACHE_TLS.with(|cache_cell| {
            if let Some(entry) = cache_cell.borrow_mut().get_mut(&key) {
                entry.aborted = true;
            }
        });
    }

    /// Compile and install a trace for a closed recording.
    ///
    /// `ops` is the captured op sequence (loop body, last op is the closing
    /// backward branch). `fallthrough_ip` is where the interpreter resumes when
    /// the loop exits normally. `slot_kinds_at_anchor` is the runtime slot type
    /// snapshot at trace start; we extract only the kinds of slots the trace
    /// actually references and store them as the entry guard.
    ///
    /// Returns true if compile + install succeeded.
    pub(crate) fn trace_install(
        chunk: &Chunk,
        anchor_ip: usize,
        fallthrough_ip: usize,
        ops: &[Op],
        recorded_ips: &[usize],
        slot_kinds_at_anchor: &[super::SlotKind],
        constants: &[FuseValue],
    ) -> bool {
        trace_install_with_kind(
            chunk,
            anchor_ip,
            anchor_ip, // close_anchor == record_anchor for main traces
            fallthrough_ip,
            ops,
            recorded_ips,
            slot_kinds_at_anchor,
            constants,
        )
    }

    /// Phase 9: install a trace with explicit `record_anchor_ip` (cache key)
    /// and `close_anchor_ip` (loop header where the close branch lands).
    /// For main traces these are equal; for side traces `record_anchor_ip`
    /// is the side-exit IP. When they differ, the compiled IR's "loop
    /// continuation" branch DOES NOT jump back to its own loop header —
    /// instead it exits returning `close_anchor_ip`, so the VM can resume
    /// the main trace (or interpreter) at the loop header for the next
    /// iteration. Side traces are one-shot completions of the post-side-
    /// exit portion of the loop body, not standalone loops.
    pub(crate) fn trace_install_with_kind(
        chunk: &Chunk,
        record_anchor_ip: usize,
        close_anchor_ip: usize,
        fallthrough_ip: usize,
        ops: &[Op],
        recorded_ips: &[usize],
        slot_kinds_at_anchor: &[super::SlotKind],
        constants: &[FuseValue],
    ) -> bool {
        let is_side_trace = record_anchor_ip != close_anchor_ip;
        // Build the per-trace slot guard list from the kinds-at-anchor snapshot.
        // Float slots are supported alongside Int slots — each entry's kind
        // is checked at trace entry and used during compile to bit-cast
        // i64-stored slot values to/from f64.
        let used_slots = collect_trace_slots(ops);
        let mut slot_types: Vec<(u16, JitTy)> = Vec::with_capacity(used_slots.len());
        for &slot in &used_slots {
            let kind = slot_kinds_at_anchor
                .get(slot as usize)
                .copied()
                .unwrap_or(super::SlotKind::Int);
            let ty = slot_kind_to_jitty(kind);
            slot_types.push((slot, ty));
        }

        let key = (chunk.op_hash, record_anchor_ip);
        let compiled = match compile_trace_kinded(
            ops,
            recorded_ips,
            close_anchor_ip,
            fallthrough_ip,
            is_side_trace,
            &slot_types,
            constants,
        ) {
            Some(c) => c,
            None => {
                trace_abort(chunk, record_anchor_ip);
                return false;
            }
        };
        TRACE_CACHE_TLS.with(|cache_cell| {
            let mut cache = cache_cell.borrow_mut();
            let entry = cache.entry(key).or_insert(TraceCacheEntry {
                hot_count: 0,
                compiled: None,
                aborted: false,
                deopt_count: 0,
                side_exit_count: 0,
                blacklisted: false,
                fallthrough_ip,
                slot_types: Vec::new(),
                saved_metadata: None,
            });
            entry.compiled = Some(Box::new(compiled));
            entry.fallthrough_ip = fallthrough_ip;
            entry.slot_types = slot_types;
            // Phase 7: retain the recording so callers can export it for
            // persistent caching. The saved metadata records the
            // `close_anchor_ip` rather than `record_anchor_ip`, so on
            // import we re-derive the (record, close) pair from the
            // metadata's `chunk_op_hash` + `anchor_ip` lookup.
            entry.saved_metadata = Some(super::TraceMetadata {
                chunk_op_hash: chunk.op_hash,
                anchor_ip: close_anchor_ip,
                fallthrough_ip,
                ops: ops.to_vec(),
                recorded_ips: recorded_ips.to_vec(),
                slot_kinds_at_anchor: slot_kinds_at_anchor.to_vec(),
            });
        });
        true
    }

    /// Whether an op is allowed to appear inside a recorded trace, ignoring
    /// frame boundaries. Superset of `is_block_eligible_op`: tracing JIT
    /// additionally accepts `Op::Call` / `Op::Return` / `Op::ReturnValue`
    /// (cross-call inlining, phase 2). `Op::CallBuiltin` is rejected because
    /// builtin handlers are arbitrary Rust we can't lower to Cranelift IR.
    /// `Op::PushFrame` / `Op::PopFrame` are rejected — the trace JIT models
    /// scopes implicitly via Call/Return only. The fused control-flow ops
    /// (`SlotLtIntJumpIfFalse`, `SlotIncLtIntJumpBack`, `AccumSumLoop`) carry
    /// embedded jumps and are rejected — chunks that contain these are
    /// already block-JIT-optimized and tracing them would just compile the
    /// same loop pattern twice.
    fn is_trace_op_allowed(op: &Op) -> bool {
        match op {
            Op::Call(_, _) | Op::Return | Op::ReturnValue => true,
            Op::CallBuiltin(_, _) | Op::PushFrame | Op::PopFrame => false,
            Op::SlotLtIntJumpIfFalse(_, _, _)
            | Op::SlotIncLtIntJumpBack(_, _, _)
            | Op::AccumSumLoop(_, _, _) => false,
            _ => is_block_eligible_op(op),
        }
    }

    /// Compile-time per-frame state for the tracing JIT.
    ///
    /// Each entry on the `frames` stack tracks the slot-variable scope of a
    /// (possibly inlined) frame and the abstract-stack length at frame entry,
    /// so that `Op::Return` / `Op::ReturnValue` can truncate the abstract
    /// stack to mirror interpreter semantics.
    ///
    /// Phase 4: `return_ip` is the IP just after the corresponding `Op::Call`
    /// in the parent frame — used at side-exit time to materialize a
    /// synthetic interpreter `Frame` so the dispatch loop can resume mid-
    /// callee with a correctly shaped call stack. For the caller frame
    /// (frames\[0\]) `return_ip` is unused (the caller frame already exists
    /// in `vm.frames` at trace entry).
    struct CompileFrame {
        slot_vars: HashMap<u16, Variable>,
        stack_base: usize,
        return_ip: usize,
    }

    // ── Cranelift IR offsets into super::DeoptInfo / super::DeoptFrame ──
    //
    // These are derived from the `#[repr(C)]` layouts of the public structs
    // and verified via `const _ assertions` on each compile_trace invocation.
    // Hardcoded so we can write store offsets directly without runtime
    // pointer math at codegen time.

    const DEOPT_INFO_RESUME_IP_OFFSET: i32 = 0;
    const DEOPT_INFO_FRAME_COUNT_OFFSET: i32 = 8;
    const DEOPT_INFO_STACK_COUNT_OFFSET: i32 = 16;
    const DEOPT_INFO_FRAMES_OFFSET: i32 = 24;
    const DEOPT_FRAME_RETURN_IP_OFFSET: i32 = 0;
    const DEOPT_FRAME_SLOT_COUNT_OFFSET: i32 = 8;
    const DEOPT_FRAME_SLOTS_OFFSET: i32 = 16;
    /// Stride of a `super::DeoptFrame`. With `MAX_DEOPT_SLOTS_PER_FRAME=16`,
    /// each frame is 8 (return_ip) + 8 (slot_count) + 16*8 (slots) = 144 bytes.
    const DEOPT_FRAME_STRIDE: i32 = 16 + (super::MAX_DEOPT_SLOTS_PER_FRAME as i32) * 8;
    /// Offset of `stack_buf[0]` from the start of `DeoptInfo`. Lives after
    /// the frames array.
    const DEOPT_INFO_STACK_BUF_OFFSET: i32 =
        DEOPT_INFO_FRAMES_OFFSET + (super::MAX_DEOPT_FRAMES as i32) * DEOPT_FRAME_STRIDE;
    /// Offset of `stack_kinds[0]` from the start of `DeoptInfo`. Lives after
    /// the stack value buffer.
    const DEOPT_INFO_STACK_KINDS_OFFSET: i32 =
        DEOPT_INFO_STACK_BUF_OFFSET + (super::MAX_DEOPT_STACK as i32) * 8;

    /// Compile-time validation of struct layouts so the offsets above stay
    /// consistent with the Rust types. Triggered on every `compile_trace`
    /// call; if the layouts ever change without updating constants, builds
    /// catch the mismatch via the runtime `assert!`s below.
    fn assert_deopt_layout() {
        assert_eq!(
            std::mem::offset_of!(super::DeoptInfo, resume_ip),
            DEOPT_INFO_RESUME_IP_OFFSET as usize
        );
        assert_eq!(
            std::mem::offset_of!(super::DeoptInfo, frame_count),
            DEOPT_INFO_FRAME_COUNT_OFFSET as usize
        );
        assert_eq!(
            std::mem::offset_of!(super::DeoptInfo, stack_count),
            DEOPT_INFO_STACK_COUNT_OFFSET as usize
        );
        assert_eq!(
            std::mem::offset_of!(super::DeoptInfo, frames),
            DEOPT_INFO_FRAMES_OFFSET as usize
        );
        assert_eq!(
            std::mem::offset_of!(super::DeoptInfo, stack_buf),
            DEOPT_INFO_STACK_BUF_OFFSET as usize
        );
        assert_eq!(
            std::mem::offset_of!(super::DeoptInfo, stack_kinds),
            DEOPT_INFO_STACK_KINDS_OFFSET as usize
        );
        assert_eq!(
            std::mem::offset_of!(super::DeoptFrame, return_ip),
            DEOPT_FRAME_RETURN_IP_OFFSET as usize
        );
        assert_eq!(
            std::mem::offset_of!(super::DeoptFrame, slot_count),
            DEOPT_FRAME_SLOT_COUNT_OFFSET as usize
        );
        assert_eq!(
            std::mem::offset_of!(super::DeoptFrame, slots),
            DEOPT_FRAME_SLOTS_OFFSET as usize
        );
        assert_eq!(
            std::mem::size_of::<super::DeoptFrame>(),
            DEOPT_FRAME_STRIDE as usize
        );
    }

    /// Emit IR that writes `value` (an `i64` Cranelift value) to a field at
    /// `offset` from the deopt-info pointer.
    fn store_deopt_i64(
        bcx: &mut FunctionBuilder,
        deopt_ptr: cranelift_codegen::ir::Value,
        offset: i32,
        value: cranelift_codegen::ir::Value,
    ) {
        bcx.ins()
            .store(MemFlags::trusted(), value, deopt_ptr, offset);
    }

    /// Emit the IR sequence shared by every trace exit (normal loop
    /// fallthrough OR a per-branch side-exit):
    /// 1. Spill caller-frame slot Variables back to `*slot_base`.
    /// 2. Write `resume_ip`, `frame_count`, `stack_count` to `*deopt_info`.
    /// 3. Write per-frame materialization records.
    /// 4. Write abstract-stack values to `stack_buf` so the VM can push them
    ///    onto `vm.stack` after the trace returns.
    /// 5. Emit a `return resume_ip`.
    ///
    /// `frames_to_materialize` is a slice of (return_ip, slot_count,
    /// Vec<(slot_idx, current_var)>) tuples for callee frames. For caller-
    /// only side-exits this is empty and `frame_count` is 0.
    /// `abstract_stack` is the trace's abstract stack at the exit point;
    /// each entry's i64 value is written to `stack_buf` in order. Phase 5a
    /// only supports Int entries — Float entries should be rejected by the
    /// caller before invoking emit_exit.
    fn emit_exit(
        bcx: &mut FunctionBuilder,
        slot_base: cranelift_codegen::ir::Value,
        deopt_ptr: cranelift_codegen::ir::Value,
        caller_slot_vars: &HashMap<u16, Variable>,
        frames_to_materialize: &[(usize, usize, Vec<(u16, Variable)>)],
        abstract_stack: &[(cranelift_codegen::ir::Value, JitTy)],
        resume_ip: usize,
    ) {
        // 1. Spill caller-frame slots.
        for (&slot, &var) in caller_slot_vars {
            let val = bcx.use_var(var);
            bcx.ins()
                .store(MemFlags::trusted(), val, slot_base, (slot as i32) * 8);
        }
        // 2. Write resume_ip / frame_count / stack_count.
        let resume_v = bcx.ins().iconst(types::I64, resume_ip as i64);
        store_deopt_i64(bcx, deopt_ptr, DEOPT_INFO_RESUME_IP_OFFSET, resume_v);
        let frame_count_v = bcx
            .ins()
            .iconst(types::I64, frames_to_materialize.len() as i64);
        store_deopt_i64(bcx, deopt_ptr, DEOPT_INFO_FRAME_COUNT_OFFSET, frame_count_v);
        let stack_count_v = bcx.ins().iconst(types::I64, abstract_stack.len() as i64);
        store_deopt_i64(bcx, deopt_ptr, DEOPT_INFO_STACK_COUNT_OFFSET, stack_count_v);
        // 3. Per-frame records.
        for (i, (return_ip, slot_count, slot_vals)) in frames_to_materialize.iter().enumerate() {
            let frame_off = DEOPT_INFO_FRAMES_OFFSET + (i as i32) * DEOPT_FRAME_STRIDE;
            let rip_v = bcx.ins().iconst(types::I64, *return_ip as i64);
            store_deopt_i64(
                bcx,
                deopt_ptr,
                frame_off + DEOPT_FRAME_RETURN_IP_OFFSET,
                rip_v,
            );
            let sc_v = bcx.ins().iconst(types::I64, *slot_count as i64);
            store_deopt_i64(
                bcx,
                deopt_ptr,
                frame_off + DEOPT_FRAME_SLOT_COUNT_OFFSET,
                sc_v,
            );
            for (slot_idx, var) in slot_vals {
                let val = bcx.use_var(*var);
                let off = frame_off + DEOPT_FRAME_SLOTS_OFFSET + (*slot_idx as i32) * 8;
                store_deopt_i64(bcx, deopt_ptr, off, val);
            }
        }
        // 4. Abstract stack values + kinds. Order: trace's abstract stack[0]
        // → stack_buf[0]; the VM pushes them onto vm.stack in this order so
        // stack[N-1] ends up at the top. Float entries bit-cast f64→i64
        // (preserving the bit pattern so VM-side `f64::from_bits` recovers
        // the original value) and tag with STACK_KIND_FLOAT.
        for (i, (val, ty)) in abstract_stack.iter().enumerate() {
            let val_off = DEOPT_INFO_STACK_BUF_OFFSET + (i as i32) * 8;
            let stored: cranelift_codegen::ir::Value = match ty {
                JitTy::Int => *val,
                JitTy::Float => {
                    bcx.ins()
                        .bitcast(types::I64, cranelift_codegen::ir::MemFlags::new(), *val)
                }
            };
            bcx.ins()
                .store(MemFlags::trusted(), stored, deopt_ptr, val_off);
            // Kind tag byte.
            let kind_off = DEOPT_INFO_STACK_KINDS_OFFSET + (i as i32);
            let kind_v = bcx.ins().iconst(
                types::I8,
                match ty {
                    JitTy::Int => super::STACK_KIND_INT as i64,
                    JitTy::Float => super::STACK_KIND_FLOAT as i64,
                },
            );
            bcx.ins()
                .store(MemFlags::trusted(), kind_v, deopt_ptr, kind_off);
        }
        // 5. Return.
        bcx.ins().return_(&[resume_v]);
    }

    /// Look up (or lazily allocate) the slot variable for the current frame.
    /// Caller-frame slots are eagerly populated from the slot pointer at trace
    /// entry; if a caller-frame access misses here, `collect_trace_slots`
    /// missed a referenced slot — that's a bug, return None to fail compile.
    /// Inlined frames lazily allocate their slot vars zero-initialized,
    /// matching the interpreter's "out-of-bounds slot reads as Undef → 0".
    fn get_or_alloc_slot_var(
        frames: &mut Vec<CompileFrame>,
        slot: u16,
        bcx: &mut FunctionBuilder,
    ) -> Option<Variable> {
        let depth = frames.len().checked_sub(1)?;
        let frame = frames.last_mut()?;
        if let Some(&v) = frame.slot_vars.get(&slot) {
            return Some(v);
        }
        if depth == 0 {
            // Caller frame must have been pre-populated. Missing slot = bug.
            return None;
        }
        let var = bcx.declare_var(types::I64);
        let zero = bcx.ins().iconst(types::I64, 0);
        bcx.def_var(var, zero);
        frame.slot_vars.insert(slot, var);
        Some(var)
    }

    /// Validate a recorded trace before compilation. Phase 2 rules:
    ///
    /// - All ops are trace-allowed (block-eligible + Call/Return/ReturnValue).
    /// - Slot indices stay below `MAX_TRACE_SLOT`.
    /// - Frame depth, simulated by `Call` (push) and `Return`/`ReturnValue`
    ///   (pop), stays non-negative throughout and is exactly 0 at the close.
    /// - Inside any callee body (depth > 0), no `Jump*` ops (branchless
    ///   callees only — internal branches need side-exit machinery).
    /// - In the caller frame (depth == 0), only the FINAL op may be a Jump*,
    ///   and it must be a backward branch with target == `anchor_ip`.
    /// - Length is bounded by `MAX_TRACE_LEN`.
    pub(crate) fn is_trace_eligible(ops: &[Op], anchor_ip: usize) -> bool {
        if ops.is_empty() || ops.len() > cfg_max_trace_len() {
            return false;
        }

        // Per-op allowance + slot-index bound check.
        for op in ops {
            if !is_trace_op_allowed(op) {
                return false;
            }
            let bad_slot = match op {
                Op::GetSlot(s)
                | Op::SetSlot(s)
                | Op::PreIncSlot(s)
                | Op::PreIncSlotVoid(s)
                | Op::SlotLtIntJumpIfFalse(s, _, _)
                | Op::SlotIncLtIntJumpBack(s, _, _) => *s >= MAX_TRACE_SLOT,
                Op::AccumSumLoop(a, b, _) | Op::AddAssignSlotVoid(a, b) => {
                    *a >= MAX_TRACE_SLOT || *b >= MAX_TRACE_SLOT
                }
                _ => false,
            };
            if bad_slot {
                return false;
            }
        }

        // Last op must be a backward branch to anchor_ip in the caller frame.
        let last = ops.last().unwrap();
        let closes = matches!(
            last,
            Op::Jump(t) | Op::JumpIfTrue(t) | Op::JumpIfFalse(t)
                if *t == anchor_ip
        );
        if !closes {
            return false;
        }

        // Walk the body (everything except the closing branch) tracking frame
        // depth. Phase 4: branches allowed in any frame; max inlining depth at
        // any side-exit is bounded by `MAX_DEOPT_FRAMES`.
        let mut depth: i32 = 0;
        let mut max_depth_at_branch: i32 = 0;
        for op in &ops[..ops.len() - 1] {
            match op {
                Op::Call(_, _) => depth += 1,
                Op::Return | Op::ReturnValue => {
                    depth -= 1;
                    if depth < 0 {
                        return false;
                    }
                }
                Op::Jump(t) | Op::JumpIfTrue(t) | Op::JumpIfFalse(t) => {
                    // Backward jumps to anchor BEFORE the final close are
                    // duplicate closes — malformed trace.
                    if depth == 0 && *t == anchor_ip {
                        return false;
                    }
                    if depth > max_depth_at_branch {
                        max_depth_at_branch = depth;
                    }
                }
                Op::JumpIfTrueKeep(_) | Op::JumpIfFalseKeep(_) => {
                    // Keep variants leave the condition on the stack post-
                    // branch; we still require empty stack at branch points.
                    return false;
                }
                _ => {}
            }
        }
        // Closing branch must be in the caller frame (depth 0).
        if depth != 0 {
            return false;
        }
        // Phase 4 cap on inlined-frame materialization at side-exit.
        if max_depth_at_branch > super::MAX_DEOPT_FRAMES as i32 {
            return false;
        }
        true
    }

    /// Collect the set of slot indices referenced by the trace's CALLER
    /// frame (depth 0). Inlined callee frames have ephemeral slot variables
    /// allocated lazily at compile time — they don't need entry-guard
    /// snapshots and aren't reflected in the slot buffer.
    pub(crate) fn collect_trace_slots(ops: &[Op]) -> Vec<u16> {
        let mut seen = Vec::new();
        let mut depth: i32 = 0;
        for op in ops {
            match op {
                Op::Call(_, _) => {
                    depth += 1;
                    continue;
                }
                Op::Return | Op::ReturnValue => {
                    depth -= 1;
                    continue;
                }
                _ => {}
            }
            if depth != 0 {
                continue; // skip slot ops inside callees
            }
            let mark = |s: u16, seen: &mut Vec<u16>| {
                if !seen.contains(&s) {
                    seen.push(s);
                }
            };
            match op {
                Op::GetSlot(s) | Op::SetSlot(s) | Op::PreIncSlot(s) | Op::PreIncSlotVoid(s) => {
                    mark(*s, &mut seen)
                }
                Op::SlotLtIntJumpIfFalse(s, _, _) | Op::SlotIncLtIntJumpBack(s, _, _) => {
                    mark(*s, &mut seen)
                }
                Op::AccumSumLoop(a, b, _) => {
                    mark(*a, &mut seen);
                    mark(*b, &mut seen);
                }
                Op::AddAssignSlotVoid(a, b) => {
                    mark(*a, &mut seen);
                    mark(*b, &mut seen);
                }
                _ => {}
            }
        }
        seen
    }

    /// Compile a recorded trace to native code.
    ///
    /// IR shape:
    /// ```text
    /// entry:
    ///     load slot vars from *slots
    ///     jump loop_hdr
    /// loop_hdr:
    ///     <emit body ops, except last (the closing branch)>
    ///     <emit closing branch as conditional brif:
    ///       - if continues, branch to loop_hdr
    ///       - if exits,    branch to exit_block>
    /// exit_block:
    ///     spill slot vars back to *slots
    ///     return iconst(fallthrough_ip)
    /// ```
    /// Phase 9 + float slots: compile a trace.
    ///
    /// `is_side_trace = true` means the closing branch's "loop continuation"
    /// direction exits returning `close_anchor_ip` rather than looping back
    /// to the IR's own loop header (side traces are one-shot completions).
    ///
    /// `slot_types` carries the (slot_idx, JitTy) pairs from the trace's
    /// entry guard. Slots marked Float are stored in their Cranelift
    /// Variables as i64 bit patterns; GetSlot bit-casts to f64 on read,
    /// SetSlot bit-casts back on write. Fused arithmetic ops on slots
    /// (PreIncSlot, AddAssignSlotVoid, etc.) reject Float slots — those
    /// remain Int-only by design (they exist for tight integer counter
    /// loops).
    fn compile_trace_kinded(
        ops: &[Op],
        recorded_ips: &[usize],
        close_anchor_ip: usize,
        fallthrough_ip: usize,
        is_side_trace: bool,
        slot_types: &[(u16, JitTy)],
        constants: &[FuseValue],
    ) -> Option<CompiledTrace> {
        let _ = close_anchor_ip;
        compile_trace_inner(
            ops,
            recorded_ips,
            fallthrough_ip,
            is_side_trace,
            slot_types,
            constants,
        )
    }

    fn compile_trace_inner(
        ops: &[Op],
        recorded_ips: &[usize],
        fallthrough_ip: usize,
        is_side_trace: bool,
        slot_types: &[(u16, JitTy)],
        constants: &[FuseValue],
    ) -> Option<CompiledTrace> {
        // Quick lookup: slot index → its kind (Int / Float).
        let slot_kind_of: HashMap<u16, JitTy> = slot_types.iter().copied().collect();
        // Defensive: catch struct-layout drift between Rust types and the
        // hardcoded offsets used in IR codegen below.
        assert_deopt_layout();
        if recorded_ips.len() != ops.len() {
            return None;
        }
        let mut module = new_jit_module()?;

        // Helper signatures (call out to fmod/pow/lognot when needed).
        let needs_pow = ops.iter().any(|o| matches!(o, Op::Pow));
        let pow_i64_id = if needs_pow {
            let mut ps = module.make_signature();
            ps.params.push(AbiParam::new(types::I64));
            ps.params.push(AbiParam::new(types::I64));
            ps.returns.push(AbiParam::new(types::I64));
            Some(
                module
                    .declare_function("fusevm_jit_pow_i64", Linkage::Import, &ps)
                    .ok()?,
            )
        } else {
            None
        };
        let pow_f64_id = if needs_pow {
            let mut ps = module.make_signature();
            ps.params.push(AbiParam::new(types::F64));
            ps.params.push(AbiParam::new(types::F64));
            ps.returns.push(AbiParam::new(types::F64));
            Some(
                module
                    .declare_function("fusevm_jit_pow_f64", Linkage::Import, &ps)
                    .ok()?,
            )
        } else {
            None
        };
        let needs_fmod = ops.iter().any(|o| matches!(o, Op::Mod));
        let fmod_f64_id = if needs_fmod {
            let mut ps = module.make_signature();
            ps.params.push(AbiParam::new(types::F64));
            ps.params.push(AbiParam::new(types::F64));
            ps.returns.push(AbiParam::new(types::F64));
            Some(
                module
                    .declare_function("fusevm_jit_fmod_f64", Linkage::Import, &ps)
                    .ok()?,
            )
        } else {
            None
        };
        let needs_lognot = ops.iter().any(|o| matches!(o, Op::LogNot));
        let lognot_id = if needs_lognot {
            let mut ps = module.make_signature();
            ps.params.push(AbiParam::new(types::I64));
            ps.returns.push(AbiParam::new(types::I64));
            Some(
                module
                    .declare_function("fusevm_jit_lognot_i64", Linkage::Import, &ps)
                    .ok()?,
            )
        } else {
            None
        };

        let ptr_ty = module.target_config().pointer_type();
        let mut sig = module.make_signature();
        sig.params.push(AbiParam::new(ptr_ty)); // *mut i64 slots
        sig.params.push(AbiParam::new(ptr_ty)); // *mut DeoptInfo
        sig.returns.push(AbiParam::new(types::I64));
        let fid = module
            .declare_function("trace", Linkage::Local, &sig)
            .ok()?;
        let mut ctx = module.make_context();
        ctx.func.signature = sig;
        ctx.func.name = UserFuncName::user(0, fid.as_u32());

        // Slots referenced by trace, ordered. Each becomes a Cranelift Variable
        // promoted in the entry block and spilled in the exit block.
        let trace_slots = collect_trace_slots(ops);

        let mut fctx = FunctionBuilderContext::new();
        {
            let mut bcx = FunctionBuilder::new(&mut ctx.func, &mut fctx);
            let entry = bcx.create_block();
            let loop_hdr = bcx.create_block();
            let exit_block = bcx.create_block();

            bcx.append_block_params_for_function_params(entry);
            bcx.switch_to_block(entry);
            let slot_base = bcx.block_params(entry)[0];
            let deopt_ptr = bcx.block_params(entry)[1];

            let pow_i64_ref = pow_i64_id.map(|pid| module.declare_func_in_func(pid, bcx.func));
            let pow_f64_ref = pow_f64_id.map(|pid| module.declare_func_in_func(pid, bcx.func));
            let fmod_f64_ref = fmod_f64_id.map(|pid| module.declare_func_in_func(pid, bcx.func));
            let lognot_ref = lognot_id.map(|lid| module.declare_func_in_func(lid, bcx.func));

            // Caller frame: eagerly promote slot variables from the slot
            // pointer. Inlined frames are pushed lazily on Op::Call.
            let mut frames: Vec<CompileFrame> = Vec::with_capacity(4);
            frames.push(CompileFrame {
                slot_vars: HashMap::new(),
                stack_base: 0,
                return_ip: 0, // unused for caller frame
            });
            for &slot in &trace_slots {
                let var = bcx.declare_var(types::I64);
                let off = (slot as i32) * 8;
                let v = bcx
                    .ins()
                    .load(types::I64, MemFlags::trusted(), slot_base, off);
                bcx.def_var(var, v);
                frames[0].slot_vars.insert(slot, var);
            }

            // Jump to loop header.
            bcx.ins().jump(loop_hdr, &[]);

            // Loop header block. As internal caller-frame branches are
            // encountered, we allocate a fresh `continue_block` per branch
            // and switch to it; the IR ends up as a chain of blocks
            // separated by brif guards leading either forward (recorded
            // direction) or to a side-exit.
            bcx.switch_to_block(loop_hdr);
            let mut stack: Vec<(cranelift_codegen::ir::Value, JitTy)> = Vec::with_capacity(32);

            // Helper: emit a side-exit block that spills caller-frame slots
            // and returns the given IP. Caller positions the cursor in the
            // side-exit block on entry; we restore to the previous block on
            // exit. Closure-style is awkward across borrows, so this is a
            // macro-style inline pattern reused below.

            // Emit all ops except the final closing branch.
            let body = &ops[..ops.len() - 1];
            for (i, op) in body.iter().enumerate() {
                // For internal branches we infer the recorded direction by
                // comparing the recorded IP of the NEXT executed op against
                // the branch target. This is well-defined for all body
                // positions because `recorded_ips` is parallel to `ops` and
                // the closing branch has its own recorded IP at the end.
                let next_recorded_ip = recorded_ips.get(i + 1).copied();

                match op {
                    Op::Nop => {}

                    // ── Frame management (cross-call inlining) ──
                    Op::Call(_, argc) => {
                        // Op::Call resolution is implicit: the recorded ops
                        // immediately following are the callee body. We just
                        // open a new slot scope; abstract stack carries args
                        // in place (no movement to slots — bytecode handles
                        // arg consumption explicitly).
                        let new_base = stack.len().saturating_sub(*argc as usize);
                        // The caller's resume IP after this Call is the IP
                        // immediately after the recorded Call op. Used at
                        // side-exit to materialize this synthetic frame's
                        // return address so the interpreter resumes the
                        // caller correctly when the callee eventually
                        // returns.
                        let return_ip = recorded_ips[i] + 1;
                        frames.push(CompileFrame {
                            slot_vars: HashMap::new(),
                            stack_base: new_base,
                            return_ip,
                        });
                    }
                    Op::Return => {
                        // Callee returns no value: truncate stack to the
                        // frame's entry mark, drop the slot scope.
                        let frame = frames.pop()?;
                        if frames.is_empty() {
                            // Underflow — recorded an extra Return.
                            return None;
                        }
                        stack.truncate(frame.stack_base);
                    }
                    Op::ReturnValue => {
                        // Callee returns top-of-stack: save, truncate, push.
                        let saved = stack.pop()?;
                        let frame = frames.pop()?;
                        if frames.is_empty() {
                            return None;
                        }
                        stack.truncate(frame.stack_base);
                        stack.push(saved);
                    }

                    // ── Slot ops route through the current frame's scope ──
                    // Float-kinded caller slots are stored as i64 bit
                    // patterns; bit-cast through f64 on use. Inlined-callee
                    // slots are always Int (they're zero-init scratch).
                    Op::GetSlot(slot) => {
                        let var = get_or_alloc_slot_var(&mut frames, *slot, &mut bcx)?;
                        let raw = bcx.use_var(var);
                        let kind = if frames.len() == 1 {
                            slot_kind_of.get(slot).copied().unwrap_or(JitTy::Int)
                        } else {
                            JitTy::Int
                        };
                        match kind {
                            JitTy::Int => stack.push((raw, JitTy::Int)),
                            JitTy::Float => {
                                let f = bcx.ins().bitcast(
                                    types::F64,
                                    cranelift_codegen::ir::MemFlags::new(),
                                    raw,
                                );
                                stack.push((f, JitTy::Float));
                            }
                        }
                    }
                    Op::SetSlot(slot) => {
                        let var = get_or_alloc_slot_var(&mut frames, *slot, &mut bcx)?;
                        let (v, ty) = stack.pop()?;
                        // Coerce stored value to i64 bit pattern. For Int
                        // values this is identity; for Float values we
                        // bit-cast (preserving the f64's bit pattern).
                        let v_i = match ty {
                            JitTy::Int => v,
                            JitTy::Float => bcx.ins().bitcast(
                                types::I64,
                                cranelift_codegen::ir::MemFlags::new(),
                                v,
                            ),
                        };
                        bcx.def_var(var, v_i);
                    }
                    Op::PreIncSlot(slot) => {
                        // Fused-arithmetic ops on slots are int-only by
                        // design (emitted by the bytecode optimizer for
                        // tight integer counter loops). Reject if the slot
                        // is Float-kinded.
                        if frames.len() == 1 {
                            if let Some(JitTy::Float) = slot_kind_of.get(slot).copied() {
                                return None;
                            }
                        }
                        let var = get_or_alloc_slot_var(&mut frames, *slot, &mut bcx)?;
                        let old = bcx.use_var(var);
                        let one = bcx.ins().iconst(types::I64, 1);
                        let new = bcx.ins().iadd(old, one);
                        bcx.def_var(var, new);
                        stack.push((new, JitTy::Int));
                    }
                    Op::PreIncSlotVoid(slot) => {
                        if frames.len() == 1 {
                            if let Some(JitTy::Float) = slot_kind_of.get(slot).copied() {
                                return None;
                            }
                        }
                        let var = get_or_alloc_slot_var(&mut frames, *slot, &mut bcx)?;
                        let old = bcx.use_var(var);
                        let one = bcx.ins().iconst(types::I64, 1);
                        let new = bcx.ins().iadd(old, one);
                        bcx.def_var(var, new);
                    }
                    Op::AddAssignSlotVoid(a_slot, b_slot) => {
                        if frames.len() == 1 {
                            let a_kind = slot_kind_of.get(a_slot).copied().unwrap_or(JitTy::Int);
                            let b_kind = slot_kind_of.get(b_slot).copied().unwrap_or(JitTy::Int);
                            if a_kind != JitTy::Int || b_kind != JitTy::Int {
                                return None;
                            }
                        }
                        let a_var = get_or_alloc_slot_var(&mut frames, *a_slot, &mut bcx)?;
                        let b_var = get_or_alloc_slot_var(&mut frames, *b_slot, &mut bcx)?;
                        let va = bcx.use_var(a_var);
                        let vb = bcx.use_var(b_var);
                        let sum = bcx.ins().iadd(va, vb);
                        bcx.def_var(a_var, sum);
                    }

                    // ── Internal caller-frame branches with side-exits ──
                    Op::Jump(t) => {
                        // Unconditional jump — the recorder must have followed
                        // it. If the next recorded IP doesn't match the target,
                        // the trace is malformed.
                        if next_recorded_ip != Some(*t) {
                            return None;
                        }
                        // No IR emitted: control falls through linearly to
                        // the next recorded op.
                    }
                    Op::JumpIfTrue(t) | Op::JumpIfFalse(t) => {
                        let target = *t;
                        let (cond, ty) = stack.pop()?;
                        // Phase 5b: non-empty abstract stack at branch with
                        // mixed Int/Float entries OK — each entry's kind is
                        // tagged in `DeoptInfo.stack_kinds` so the VM can
                        // materialize Value::Int vs Value::Float correctly.
                        if stack.len() > super::MAX_DEOPT_STACK {
                            return None;
                        }
                        let took_jump = next_recorded_ip == Some(target);
                        // The un-recorded direction's target — where we'd
                        // resume in the interpreter on guard fail.
                        let side_exit_ip = if took_jump {
                            recorded_ips[i] + 1
                        } else {
                            target
                        };
                        // Coerce the condition to an i64 truthy value for brif.
                        let cond_pred = match ty {
                            JitTy::Int => cond,
                            JitTy::Float => {
                                let z = bcx.ins().f64const(Ieee64::with_bits(0.0f64.to_bits()));
                                let p = bcx.ins().fcmp(FloatCC::OrderedNotEqual, cond, z);
                                let one = bcx.ins().iconst(types::I64, 1);
                                let zero = bcx.ins().iconst(types::I64, 0);
                                bcx.ins().select(p, one, zero)
                            }
                        };
                        // Was the cond truthy at recording time?
                        let recorded_truthy = match op {
                            Op::JumpIfTrue(_) => took_jump,
                            Op::JumpIfFalse(_) => !took_jump,
                            _ => unreachable!(),
                        };
                        // Build materialization records for inlined callee
                        // frames (frames[1..]). Caller frame is implicit —
                        // it already exists in vm.frames at trace entry.
                        let mut frames_to_materialize: Vec<(usize, usize, Vec<(u16, Variable)>)> =
                            Vec::new();
                        for callee_frame in &frames[1..] {
                            let slot_count = if callee_frame.slot_vars.is_empty() {
                                0
                            } else {
                                let max = *callee_frame.slot_vars.keys().max().unwrap();
                                (max as usize) + 1
                            };
                            if slot_count > super::MAX_DEOPT_SLOTS_PER_FRAME {
                                return None;
                            }
                            let slot_vals: Vec<(u16, Variable)> = callee_frame
                                .slot_vars
                                .iter()
                                .map(|(&slot, &var)| (slot, var))
                                .collect();
                            frames_to_materialize.push((
                                callee_frame.return_ip,
                                slot_count,
                                slot_vals,
                            ));
                        }
                        if frames_to_materialize.len() > super::MAX_DEOPT_FRAMES {
                            return None;
                        }
                        let cont = bcx.create_block();
                        let side = bcx.create_block();
                        if recorded_truthy {
                            bcx.ins().brif(cond_pred, cont, &[], side, &[]);
                        } else {
                            bcx.ins().brif(cond_pred, side, &[], cont, &[]);
                        }
                        // Side-exit: spill caller slots, materialize inlined
                        // frames, write the remaining abstract stack to the
                        // deopt buffer, return resume_ip.
                        bcx.switch_to_block(side);
                        emit_exit(
                            &mut bcx,
                            slot_base,
                            deopt_ptr,
                            &frames[0].slot_vars,
                            &frames_to_materialize,
                            &stack,
                            side_exit_ip,
                        );
                        // Resume IR emission in the continue block.
                        bcx.switch_to_block(cont);
                    }
                    // Phase 3 doesn't support Keep variants (post-branch
                    // stack non-empty). Eligibility rejects upstream; this
                    // is a defensive double-check.
                    Op::JumpIfTrueKeep(_) | Op::JumpIfFalseKeep(_) => return None,

                    // Fused-loop ops contain embedded control flow; rejected.
                    Op::SlotLtIntJumpIfFalse(_, _, _)
                    | Op::SlotIncLtIntJumpBack(_, _, _)
                    | Op::AccumSumLoop(_, _, _) => return None,

                    // Frame markers and builtin calls aren't traceable.
                    Op::PushFrame | Op::PopFrame | Op::CallBuiltin(_, _) => return None,

                    // Everything else delegates to emit_data_op (slot_base is
                    // unused since slot ops are handled above).
                    _ => {
                        emit_data_op(
                            &mut bcx,
                            op,
                            &mut stack,
                            Some(slot_base),
                            pow_i64_ref,
                            pow_f64_ref,
                            fmod_f64_ref,
                            lognot_ref,
                            constants,
                        )?;
                    }
                }
            }

            // Closing branch must be in caller frame (depth == 0). Validation
            // already enforced this in `is_trace_eligible`, but re-check at
            // compile time — in case the recorder gave us a malformed trace.
            if frames.len() != 1 {
                return None;
            }

            // Closing branch — direction determines which side jumps back vs exits.
            let last = ops.last()?;
            match last {
                Op::Jump(_) => {
                    // Unconditional close — loop never exits via this op.
                    // Phase 1: still need an exit somewhere; for this case
                    // the trace is an infinite loop, which we reject upstream
                    // (eligibility currently allows it, but we'd never compile
                    // it productively). Bail.
                    return None;
                }
                Op::JumpIfTrue(t) => {
                    // True → "continue loop" direction; false → exit.
                    // For main traces "continue" loops back to loop_hdr;
                    // for side traces "continue" exits returning the close
                    // target (so the main trace / interpreter can take
                    // over the next iteration from the loop header).
                    let target_ip = *t;
                    let (cond, ty) = stack.pop()?;
                    let pred = match ty {
                        JitTy::Int => cond,
                        JitTy::Float => {
                            let z = bcx.ins().f64const(Ieee64::with_bits(0.0f64.to_bits()));
                            let p = bcx.ins().fcmp(FloatCC::OrderedNotEqual, cond, z);
                            let one = bcx.ins().iconst(types::I64, 1);
                            let zero = bcx.ins().iconst(types::I64, 0);
                            bcx.ins().select(p, one, zero)
                        }
                    };
                    if is_side_trace {
                        // Both directions exit. "Continue" exits with the
                        // close target IP; "exit" exits with the trace's
                        // fallthrough_ip (the post-loop IP).
                        let cont_exit = bcx.create_block();
                        bcx.ins().brif(pred, cont_exit, &[], exit_block, &[]);
                        bcx.switch_to_block(cont_exit);
                        emit_exit(
                            &mut bcx,
                            slot_base,
                            deopt_ptr,
                            &frames[0].slot_vars,
                            &[],
                            &[],
                            target_ip,
                        );
                    } else {
                        bcx.ins().brif(pred, loop_hdr, &[], exit_block, &[]);
                    }
                }
                Op::JumpIfFalse(t) => {
                    // False → "continue loop"; true → exit.
                    let target_ip = *t;
                    let (cond, ty) = stack.pop()?;
                    let pred = match ty {
                        JitTy::Int => cond,
                        JitTy::Float => {
                            let z = bcx.ins().f64const(Ieee64::with_bits(0.0f64.to_bits()));
                            let p = bcx.ins().fcmp(FloatCC::OrderedNotEqual, cond, z);
                            let one = bcx.ins().iconst(types::I64, 1);
                            let zero = bcx.ins().iconst(types::I64, 0);
                            bcx.ins().select(p, one, zero)
                        }
                    };
                    if is_side_trace {
                        let cont_exit = bcx.create_block();
                        bcx.ins().brif(pred, exit_block, &[], cont_exit, &[]);
                        bcx.switch_to_block(cont_exit);
                        emit_exit(
                            &mut bcx,
                            slot_base,
                            deopt_ptr,
                            &frames[0].slot_vars,
                            &[],
                            &[],
                            target_ip,
                        );
                    } else {
                        bcx.ins().brif(pred, exit_block, &[], loop_hdr, &[]);
                    }
                }
                _ => return None,
            }

            // Exit block: spill caller-frame slot vars, write deopt info
            // (frame_count = 0, stack_count = 0 — closing branch leaves an
            // empty abstract stack and is at depth 0 by eligibility), and
            // return the loop's fallthrough IP.
            bcx.switch_to_block(exit_block);
            emit_exit(
                &mut bcx,
                slot_base,
                deopt_ptr,
                &frames[0].slot_vars,
                &[],
                &[],
                fallthrough_ip,
            );

            bcx.seal_all_blocks();
            bcx.finalize();
        }

        module.define_function(fid, &mut ctx).ok()?;
        module.clear_context(&mut ctx);
        module.finalize_definitions().ok()?;
        let ptr = module.get_finalized_function(fid);
        let run = unsafe { std::mem::transmute::<*const u8, TraceFn>(ptr) };
        Some(CompiledTrace { module, run })
    }

    /// Test-only: clear the trace cache. Public for tests within this crate.
    #[cfg(test)]
    pub(crate) fn trace_cache_clear() {
        TRACE_CACHE_TLS.with(|c| c.borrow_mut().clear());
    }

    /// Public-ish: return whether a compiled trace exists for (chunk, anchor).
    pub(crate) fn trace_is_compiled(chunk: &Chunk, anchor_ip: usize) -> bool {
        let key = (chunk.op_hash, anchor_ip);
        TRACE_CACHE_TLS.with(|c| c.borrow().get(&key).map_or(false, |e| e.compiled.is_some()))
    }

    /// Public-ish: return the deopt count for a trace (for tests/blacklist obs).
    pub(crate) fn trace_deopt_count(chunk: &Chunk, anchor_ip: usize) -> u32 {
        let key = (chunk.op_hash, anchor_ip);
        TRACE_CACHE_TLS.with(|c| c.borrow().get(&key).map_or(0, |e| e.deopt_count))
    }

    /// Public-ish: return the side-exit count for a trace.
    pub(crate) fn trace_side_exit_count(chunk: &Chunk, anchor_ip: usize) -> u32 {
        let key = (chunk.op_hash, anchor_ip);
        TRACE_CACHE_TLS.with(|c| c.borrow().get(&key).map_or(0, |e| e.side_exit_count))
    }

    /// Phase 9: bump the side-exit counter for a trace when a deopt fires
    /// AND no side trace was available to absorb it. Auto-blacklists after
    /// `MAX_SIDE_EXITS`. Called from VM-side chained dispatch.
    pub(crate) fn trace_bump_side_exit(chunk: &Chunk, anchor_ip: usize) {
        let key = (chunk.op_hash, anchor_ip);
        TRACE_CACHE_TLS.with(|c| {
            if let Some(entry) = c.borrow_mut().get_mut(&key) {
                entry.side_exit_count = entry.side_exit_count.saturating_add(1);
                if entry.side_exit_count >= cfg_max_side_exits() {
                    entry.blacklisted = true;
                }
            }
        });
    }

    /// Phase 9: read the recorded close_anchor_ip / fallthrough_ip pair
    /// for an installed trace. The VM uses this when arming side-trace
    /// recording at a hot side-exit — the side trace must close at the
    /// same loop header (close_anchor_ip) and fall through to the same
    /// post-loop IP (fallthrough_ip) as the main trace.
    pub(crate) fn trace_loop_anchors(chunk: &Chunk, anchor_ip: usize) -> Option<(usize, usize)> {
        let key = (chunk.op_hash, anchor_ip);
        TRACE_CACHE_TLS.with(|c| {
            c.borrow().get(&key).and_then(|e| {
                e.saved_metadata
                    .as_ref()
                    .map(|m| (m.anchor_ip, m.fallthrough_ip))
            })
        })
    }

    /// Public-ish: whether a trace was blacklisted.
    pub(crate) fn trace_is_blacklisted(chunk: &Chunk, anchor_ip: usize) -> bool {
        let key = (chunk.op_hash, anchor_ip);
        TRACE_CACHE_TLS.with(|c| c.borrow().get(&key).map_or(false, |e| e.blacklisted))
    }

    /// Phase 7: export retained recording metadata for a compiled trace so
    /// the caller can serialize it (file, sqlite, etc.) and re-install on
    /// next process start. Returns `None` if no compiled trace exists at
    /// `(chunk, anchor_ip)`.
    pub(crate) fn trace_export(chunk: &Chunk, anchor_ip: usize) -> Option<super::TraceMetadata> {
        let key = (chunk.op_hash, anchor_ip);
        TRACE_CACHE_TLS.with(|c| c.borrow().get(&key).and_then(|e| e.saved_metadata.clone()))
    }

    /// Phase 7: re-install a previously-exported trace. Verifies the
    /// metadata's `chunk_op_hash` matches the current chunk; mismatch
    /// (chunk has been modified since export) returns false rather than
    /// silently mis-compiling. On success the cache entry is populated
    /// with a freshly-compiled trace, ready for invocation.
    pub(crate) fn trace_import(
        chunk: &Chunk,
        meta: &super::TraceMetadata,
        constants: &[FuseValue],
    ) -> bool {
        if meta.chunk_op_hash != chunk.op_hash {
            return false;
        }
        trace_install(
            chunk,
            meta.anchor_ip,
            meta.fallthrough_ip,
            &meta.ops,
            &meta.recorded_ips,
            &meta.slot_kinds_at_anchor,
            constants,
        )
    }

    /// Bulk-export every compiled trace whose `chunk_op_hash` matches the
    /// given chunk. Useful for persisting the entire cache after a hot
    /// run — pair with `trace_import_all` on the next process start.
    pub(crate) fn trace_export_all(chunk: &Chunk) -> Vec<super::TraceMetadata> {
        TRACE_CACHE_TLS.with(|c| {
            c.borrow()
                .values()
                .filter_map(|e| e.saved_metadata.clone())
                .filter(|m| m.chunk_op_hash == chunk.op_hash)
                .collect()
        })
    }

    /// Bulk-import a slice of trace metadata. Each entry must match the
    /// chunk's hash; mismatched entries are silently skipped. Returns the
    /// number successfully re-installed.
    pub(crate) fn trace_import_all(
        chunk: &Chunk,
        metas: &[super::TraceMetadata],
        constants: &[FuseValue],
    ) -> usize {
        let mut installed = 0;
        for m in metas {
            if trace_import(chunk, m, constants) {
                installed += 1;
            }
        }
        installed
    }
}

// ── Public API (always available) ──

/// JIT compiler state.
pub struct JitCompiler {
    extensions: Vec<Box<dyn JitExtension>>,
}

impl JitCompiler {
    pub fn new() -> Self {
        Self {
            extensions: Vec::new(),
        }
    }

    /// Register a language-specific JIT extension.
    pub fn register_extension(&mut self, ext: Box<dyn JitExtension>) {
        tracing::info!(
            name = ext.name(),
            ops = ext.op_count(),
            "JIT extension registered"
        );
        self.extensions.push(ext);
    }

    /// Check if a chunk is eligible for JIT compilation.
    pub fn is_eligible(&self, chunk: &crate::Chunk) -> bool {
        use crate::Op;
        for op in &chunk.ops {
            match op {
                // Universal ops — always JIT-able
                Op::Nop
                | Op::LoadInt(_)
                | Op::LoadFloat(_)
                | Op::LoadConst(_)
                | Op::LoadTrue
                | Op::LoadFalse
                | Op::LoadUndef
                | Op::Pop
                | Op::Dup
                | Op::Dup2
                | Op::Swap
                | Op::Rot
                | Op::GetVar(_)
                | Op::SetVar(_)
                | Op::DeclareVar(_)
                | Op::GetSlot(_)
                | Op::SetSlot(_)
                | Op::Add
                | Op::Sub
                | Op::Mul
                | Op::Div
                | Op::Mod
                | Op::Pow
                | Op::Negate
                | Op::Inc
                | Op::Dec
                | Op::Concat
                | Op::StringRepeat
                | Op::StringLen
                | Op::NumEq
                | Op::NumNe
                | Op::NumLt
                | Op::NumGt
                | Op::NumLe
                | Op::NumGe
                | Op::StrEq
                | Op::StrNe
                | Op::StrLt
                | Op::StrGt
                | Op::StrLe
                | Op::StrGe
                | Op::StrCmp
                | Op::Spaceship
                | Op::LogNot
                | Op::LogAnd
                | Op::LogOr
                | Op::BitAnd
                | Op::BitOr
                | Op::BitXor
                | Op::BitNot
                | Op::Shl
                | Op::Shr
                | Op::Jump(_)
                | Op::JumpIfTrue(_)
                | Op::JumpIfFalse(_)
                | Op::JumpIfTrueKeep(_)
                | Op::JumpIfFalseKeep(_)
                | Op::Call(_, _)
                | Op::Return
                | Op::ReturnValue
                | Op::PushFrame
                | Op::PopFrame
                | Op::PreIncSlot(_)
                | Op::PreIncSlotVoid(_)
                | Op::SlotLtIntJumpIfFalse(_, _, _)
                | Op::SlotIncLtIntJumpBack(_, _, _)
                | Op::AccumSumLoop(_, _, _)
                | Op::AddAssignSlotVoid(_, _)
                | Op::SetStatus
                | Op::GetStatus => continue,

                // Extended — check if any extension handles it
                Op::Extended(id, _) | Op::ExtendedWide(id, _) => {
                    let id = *id;
                    if !self.extensions.iter().any(|ext| ext.can_jit(id)) {
                        return false;
                    }
                }

                _ => return false,
            }
        }
        true
    }

    /// Try to compile and run a chunk via the linear JIT.
    /// Returns `Some(Value)` on success, `None` if not eligible or JIT feature disabled.
    #[cfg(feature = "jit")]
    pub fn try_run_linear(&self, chunk: &crate::Chunk, slots: &[i64]) -> Option<crate::Value> {
        cranelift_jit_impl::try_run_linear(chunk, slots)
    }

    /// Check if a chunk is eligible for linear JIT.
    #[cfg(feature = "jit")]
    pub fn is_linear_eligible(&self, chunk: &crate::Chunk) -> bool {
        cranelift_jit_impl::is_linear_eligible(chunk)
    }

    /// Stub when JIT feature is disabled.
    #[cfg(not(feature = "jit"))]
    pub fn try_run_linear(&self, _chunk: &crate::Chunk, _slots: &[i64]) -> Option<crate::Value> {
        None
    }

    /// Stub when JIT feature is disabled.
    #[cfg(not(feature = "jit"))]
    pub fn is_linear_eligible(&self, _chunk: &crate::Chunk) -> bool {
        false
    }

    /// Try to compile and run a chunk via the block JIT (handles loops/branches).
    /// Slots are read and written in-place. Returns `Some(result)` on success.
    ///
    /// Tiered: returns `None` for the first N invocations of a given chunk so the
    /// caller falls back to the interpreter. After the chunk crosses the
    /// hot-threshold, compiles and caches the native code.
    #[cfg(feature = "jit")]
    pub fn try_run_block(&self, chunk: &crate::Chunk, slots: &mut [i64]) -> Option<i64> {
        cranelift_jit_impl::try_run_block(chunk, slots)
    }

    /// Like `try_run_block` but skips the tiered policy — compiles immediately
    /// on first call. Use for tests, microbenchmarks, or AOT-style usage.
    #[cfg(feature = "jit")]
    pub fn try_run_block_eager(&self, chunk: &crate::Chunk, slots: &mut [i64]) -> Option<i64> {
        cranelift_jit_impl::try_run_block_eager(chunk, slots)
    }

    /// Check if a chunk is eligible for block JIT.
    #[cfg(feature = "jit")]
    pub fn is_block_eligible(&self, chunk: &crate::Chunk) -> bool {
        cranelift_jit_impl::is_block_eligible(chunk)
    }

    /// Find the largest contiguous JIT-eligible region in a chunk.
    /// Returns `(start, end)` op indices, or None if no useful region exists.
    /// Useful for partial JIT compilation of chunks that aren't entirely eligible.
    #[cfg(feature = "jit")]
    pub fn find_jit_region(&self, chunk: &crate::Chunk) -> Option<(usize, usize)> {
        cranelift_jit_impl::find_jit_region(&chunk.ops)
    }

    /// Extract a JIT region as a standalone sub-chunk with rebased jump targets.
    /// The sub-chunk can then be passed to `try_run_block_eager` to JIT-compile
    /// just that region. Use with `find_jit_region` to find an eligible range.
    #[cfg(feature = "jit")]
    pub fn extract_region(&self, chunk: &crate::Chunk, start: usize, end: usize) -> crate::Chunk {
        cranelift_jit_impl::extract_region(chunk, start, end)
    }

    /// Stub when JIT feature is disabled.
    #[cfg(not(feature = "jit"))]
    pub fn find_jit_region(&self, _chunk: &crate::Chunk) -> Option<(usize, usize)> {
        None
    }

    /// Stub when JIT feature is disabled.
    #[cfg(not(feature = "jit"))]
    pub fn extract_region(&self, chunk: &crate::Chunk, _start: usize, _end: usize) -> crate::Chunk {
        chunk.clone()
    }

    /// Stub when JIT feature is disabled.
    #[cfg(not(feature = "jit"))]
    pub fn try_run_block(&self, _chunk: &crate::Chunk, _slots: &mut [i64]) -> Option<i64> {
        None
    }

    /// Stub when JIT feature is disabled.
    #[cfg(not(feature = "jit"))]
    pub fn try_run_block_eager(&self, _chunk: &crate::Chunk, _slots: &mut [i64]) -> Option<i64> {
        None
    }

    /// Stub when JIT feature is disabled.
    #[cfg(not(feature = "jit"))]
    pub fn is_block_eligible(&self, _chunk: &crate::Chunk) -> bool {
        false
    }

    // ── Tracing JIT (Tier 2) ──

    /// Consult the trace cache at a backward-branch site.
    ///
    /// `anchor_ip` is the IP of the loop header (target of the backward branch).
    /// `slots` is a mutable slot array — passed to the trace fn if it runs.
    /// `slot_kinds_at_anchor` is the runtime types of slots at the anchor; used
    /// for the entry guard.
    /// `deopt_info` is a reusable scratch buffer the trace populates on exit.
    /// On `TraceLookup::Ran`, the caller materializes any inlined frames the
    /// trace recorded into `deopt_info.frames[..deopt_info.frame_count]`.
    ///
    /// Returns a `TraceLookup` describing what the interpreter should do next.
    #[cfg(feature = "jit")]
    pub fn trace_lookup(
        &self,
        chunk: &crate::Chunk,
        anchor_ip: usize,
        slots: &mut [i64],
        slot_kinds_at_anchor: &[SlotKind],
        deopt_info: &mut DeoptInfo,
    ) -> TraceLookup {
        let ptr = if slots.is_empty() {
            std::ptr::null_mut()
        } else {
            slots.as_mut_ptr()
        };
        cranelift_jit_impl::trace_lookup(chunk, anchor_ip, ptr, slot_kinds_at_anchor, deopt_info)
    }

    /// Compile and install a recorded trace.
    ///
    /// `ops` is the recorded op sequence; `recorded_ips` is the parallel
    /// bytecode IP each op was dispatched from (used to infer branch
    /// directions at compile time); `fallthrough_ip` is where the
    /// interpreter resumes when the loop exits normally;
    /// `slot_kinds_at_anchor` is the slot type snapshot used to install the
    /// entry guard.
    #[cfg(feature = "jit")]
    pub fn trace_install(
        &self,
        chunk: &crate::Chunk,
        anchor_ip: usize,
        fallthrough_ip: usize,
        ops: &[crate::Op],
        recorded_ips: &[usize],
        slot_kinds_at_anchor: &[SlotKind],
    ) -> bool {
        cranelift_jit_impl::trace_install(
            chunk,
            anchor_ip,
            fallthrough_ip,
            ops,
            recorded_ips,
            slot_kinds_at_anchor,
            &chunk.constants,
        )
    }

    /// Mark the trace cache entry at this anchor as aborted (recording failed).
    #[cfg(feature = "jit")]
    pub fn trace_abort(&self, chunk: &crate::Chunk, anchor_ip: usize) {
        cranelift_jit_impl::trace_abort(chunk, anchor_ip);
    }

    /// Whether a recorded sequence is eligible for trace JIT compilation.
    #[cfg(feature = "jit")]
    pub fn is_trace_eligible(&self, ops: &[crate::Op], anchor_ip: usize) -> bool {
        cranelift_jit_impl::is_trace_eligible(ops, anchor_ip)
    }

    /// Whether a compiled trace exists for (chunk, anchor_ip).
    #[cfg(feature = "jit")]
    pub fn trace_is_compiled(&self, chunk: &crate::Chunk, anchor_ip: usize) -> bool {
        cranelift_jit_impl::trace_is_compiled(chunk, anchor_ip)
    }

    /// Number of entry-guard deopts at runtime (slot type mismatch).
    #[cfg(feature = "jit")]
    pub fn trace_deopt_count(&self, chunk: &crate::Chunk, anchor_ip: usize) -> u32 {
        cranelift_jit_impl::trace_deopt_count(chunk, anchor_ip)
    }

    /// Number of mid-trace side-exits at runtime (brif guard mismatch).
    #[cfg(feature = "jit")]
    pub fn trace_side_exit_count(&self, chunk: &crate::Chunk, anchor_ip: usize) -> u32 {
        cranelift_jit_impl::trace_side_exit_count(chunk, anchor_ip)
    }

    /// Phase 9: bump the side-exit counter for a trace and auto-blacklist
    /// past the threshold. The VM's chained-dispatch path calls this only
    /// when no side trace was found at the deopt's `resume_ip` — if a side
    /// trace handled the deopt productively, the main trace shouldn't be
    /// penalized.
    #[cfg(feature = "jit")]
    pub fn trace_bump_side_exit(&self, chunk: &crate::Chunk, anchor_ip: usize) {
        cranelift_jit_impl::trace_bump_side_exit(chunk, anchor_ip);
    }

    /// Phase 9: install a trace with separate `record_anchor_ip` (cache key)
    /// and `close_anchor_ip` (loop header where the closing branch lands).
    /// For main traces the two values are identical; for side traces
    /// recorded at a hot side-exit, `record_anchor_ip` is the side-exit IP
    /// while `close_anchor_ip` is the enclosing loop's header. Side traces
    /// don't loop in their own IR — both directions of the closing branch
    /// exit, returning either the close target (continuation) or
    /// fallthrough_ip (exit). Main traces compile via the simpler
    /// `trace_install`.
    #[cfg(feature = "jit")]
    pub fn trace_install_with_kind(
        &self,
        chunk: &crate::Chunk,
        record_anchor_ip: usize,
        close_anchor_ip: usize,
        fallthrough_ip: usize,
        ops: &[crate::Op],
        recorded_ips: &[usize],
        slot_kinds_at_anchor: &[SlotKind],
    ) -> bool {
        cranelift_jit_impl::trace_install_with_kind(
            chunk,
            record_anchor_ip,
            close_anchor_ip,
            fallthrough_ip,
            ops,
            recorded_ips,
            slot_kinds_at_anchor,
            &chunk.constants,
        )
    }

    /// Phase 9: read the (close_anchor_ip, fallthrough_ip) pair recorded
    /// for an installed trace. Used by the VM when arming side-trace
    /// recording at a hot side-exit so the side trace closes correctly.
    #[cfg(feature = "jit")]
    pub fn trace_loop_anchors(
        &self,
        chunk: &crate::Chunk,
        anchor_ip: usize,
    ) -> Option<(usize, usize)> {
        cranelift_jit_impl::trace_loop_anchors(chunk, anchor_ip)
    }

    /// Whether the trace was blacklisted (too many deopts).
    #[cfg(feature = "jit")]
    pub fn trace_is_blacklisted(&self, chunk: &crate::Chunk, anchor_ip: usize) -> bool {
        cranelift_jit_impl::trace_is_blacklisted(chunk, anchor_ip)
    }

    /// Phase 7: export the compiled trace's recording metadata so callers
    /// can serialize it for persistent caching across process restarts.
    /// Returns `None` if no compiled trace exists at `(chunk, anchor_ip)`.
    #[cfg(feature = "jit")]
    pub fn trace_export(&self, chunk: &crate::Chunk, anchor_ip: usize) -> Option<TraceMetadata> {
        cranelift_jit_impl::trace_export(chunk, anchor_ip)
    }

    /// Phase 7: re-install a previously-exported trace. The metadata's
    /// `chunk_op_hash` must match `chunk.op_hash`; otherwise returns false
    /// (stale metadata, chunk has changed). On success the trace is
    /// re-compiled and ready for the next invocation through `trace_lookup`.
    #[cfg(feature = "jit")]
    pub fn trace_import(&self, chunk: &crate::Chunk, meta: &TraceMetadata) -> bool {
        cranelift_jit_impl::trace_import(chunk, meta, &chunk.constants)
    }

    /// Bulk export every compiled trace whose `chunk_op_hash` matches
    /// `chunk.op_hash`. Use to persist the full cache for this chunk to
    /// disk in a single pass.
    #[cfg(feature = "jit")]
    pub fn trace_export_all(&self, chunk: &crate::Chunk) -> Vec<TraceMetadata> {
        cranelift_jit_impl::trace_export_all(chunk)
    }

    /// Bulk import a slice of trace metadata. Entries with mismatched
    /// `chunk_op_hash` are skipped. Returns the count successfully
    /// re-installed.
    #[cfg(feature = "jit")]
    pub fn trace_import_all(&self, chunk: &crate::Chunk, metas: &[TraceMetadata]) -> usize {
        cranelift_jit_impl::trace_import_all(chunk, metas, &chunk.constants)
    }

    /// Maximum number of ops a single trace can record.
    #[cfg(feature = "jit")]
    pub fn trace_max_len(&self) -> usize {
        cranelift_jit_impl::cfg_max_trace_len()
    }

    /// Apply a tracing JIT configuration to the current thread. Affects
    /// subsequent recording, dispatch, and blacklist behavior. Existing
    /// compiled traces are unaffected.
    #[cfg(feature = "jit")]
    pub fn set_config(&self, cfg: TraceJitConfig) {
        cranelift_jit_impl::set_config(cfg);
    }

    /// Read the current thread's tracing JIT configuration.
    #[cfg(feature = "jit")]
    pub fn get_config(&self) -> TraceJitConfig {
        cranelift_jit_impl::get_config()
    }

    /// Maximum slot index a trace can reference.
    #[cfg(feature = "jit")]
    pub fn trace_max_slot(&self) -> u16 {
        cranelift_jit_impl::MAX_TRACE_SLOT
    }

    // ── Tracing JIT stubs (no-jit feature) ──

    #[cfg(not(feature = "jit"))]
    pub fn trace_lookup(
        &self,
        _chunk: &crate::Chunk,
        _anchor_ip: usize,
        _slots: &mut [i64],
        _slot_kinds_at_anchor: &[SlotKind],
        _deopt_info: &mut DeoptInfo,
    ) -> TraceLookup {
        TraceLookup::Skip
    }

    #[cfg(not(feature = "jit"))]
    pub fn trace_install(
        &self,
        _chunk: &crate::Chunk,
        _anchor_ip: usize,
        _fallthrough_ip: usize,
        _ops: &[crate::Op],
        _recorded_ips: &[usize],
        _slot_kinds_at_anchor: &[SlotKind],
    ) -> bool {
        false
    }

    #[cfg(not(feature = "jit"))]
    pub fn trace_abort(&self, _chunk: &crate::Chunk, _anchor_ip: usize) {}

    #[cfg(not(feature = "jit"))]
    pub fn is_trace_eligible(&self, _ops: &[crate::Op], _anchor_ip: usize) -> bool {
        false
    }

    #[cfg(not(feature = "jit"))]
    pub fn trace_is_compiled(&self, _chunk: &crate::Chunk, _anchor_ip: usize) -> bool {
        false
    }

    #[cfg(not(feature = "jit"))]
    pub fn trace_deopt_count(&self, _chunk: &crate::Chunk, _anchor_ip: usize) -> u32 {
        0
    }

    #[cfg(not(feature = "jit"))]
    pub fn trace_side_exit_count(&self, _chunk: &crate::Chunk, _anchor_ip: usize) -> u32 {
        0
    }

    #[cfg(not(feature = "jit"))]
    pub fn trace_bump_side_exit(&self, _chunk: &crate::Chunk, _anchor_ip: usize) {}

    #[cfg(not(feature = "jit"))]
    pub fn trace_install_with_kind(
        &self,
        _chunk: &crate::Chunk,
        _record_anchor_ip: usize,
        _close_anchor_ip: usize,
        _fallthrough_ip: usize,
        _ops: &[crate::Op],
        _recorded_ips: &[usize],
        _slot_kinds_at_anchor: &[SlotKind],
    ) -> bool {
        false
    }

    #[cfg(not(feature = "jit"))]
    pub fn trace_loop_anchors(
        &self,
        _chunk: &crate::Chunk,
        _anchor_ip: usize,
    ) -> Option<(usize, usize)> {
        None
    }

    #[cfg(not(feature = "jit"))]
    pub fn trace_is_blacklisted(&self, _chunk: &crate::Chunk, _anchor_ip: usize) -> bool {
        false
    }

    #[cfg(not(feature = "jit"))]
    pub fn trace_export(&self, _chunk: &crate::Chunk, _anchor_ip: usize) -> Option<TraceMetadata> {
        None
    }

    #[cfg(not(feature = "jit"))]
    pub fn trace_import(&self, _chunk: &crate::Chunk, _meta: &TraceMetadata) -> bool {
        false
    }

    #[cfg(not(feature = "jit"))]
    pub fn trace_export_all(&self, _chunk: &crate::Chunk) -> Vec<TraceMetadata> {
        Vec::new()
    }

    #[cfg(not(feature = "jit"))]
    pub fn trace_import_all(&self, _chunk: &crate::Chunk, _metas: &[TraceMetadata]) -> usize {
        0
    }

    #[cfg(not(feature = "jit"))]
    pub fn trace_max_len(&self) -> usize {
        256
    }

    #[cfg(not(feature = "jit"))]
    pub fn trace_max_slot(&self) -> u16 {
        64
    }

    #[cfg(not(feature = "jit"))]
    pub fn set_config(&self, _cfg: TraceJitConfig) {}

    #[cfg(not(feature = "jit"))]
    pub fn get_config(&self) -> TraceJitConfig {
        TraceJitConfig::defaults()
    }
}

impl Default for JitCompiler {
    fn default() -> Self {
        Self::new()
    }
}

/// Compiled native code handle (placeholder for block JIT — linear JIT is cached internally).
pub struct NativeCode {
    _private: (),
}
