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
/// Frontends implement this to JIT their `Op::Extended` ops.
///
/// Register an instance once at startup via [`register_global_extension`]; the
/// block JIT then consults it both for eligibility ([`JitExtension::can_jit`])
/// and codegen ([`JitExtension::emit_extended`]). Extensions may emit pure
/// integer IR or call host helpers registered via [`register_jit_helper`]
/// (through [`ExtJitCtx::call_host`]); either way the resulting chunk remains
/// eligible for the on-disk native-code cache.
pub trait JitExtension: Send + Sync {
    /// Whether this extension can JIT-compile the given extended op ID.
    fn can_jit(&self, ext_id: u16) -> bool;

    /// Number of extended ops this extension handles.
    fn op_count(&self) -> usize;

    /// Human-readable name for debugging.
    fn name(&self) -> &str;

    /// Emit Cranelift IR for `Op::Extended(ext_id, arg)`, manipulating the
    /// abstract operand stack through `cx` (pop inputs, push the result).
    ///
    /// Return `true` once the op has been fully lowered; return `false` to
    /// decline, which aborts block-JIT compilation of the whole chunk so it
    /// falls back to the interpreter. The default declines everything, so an
    /// extension that only sets `can_jit` without overriding this is a no-op.
    ///
    /// Implementations may emit self-contained integer IR (arithmetic,
    /// compares, `select`, slot loads/stores) and may also call host helpers
    /// registered via [`register_jit_helper`] through [`ExtJitCtx::call_host`]
    /// (e.g. for string/collection ops whose operands are passed as `i64`
    /// handles). Both styles remain eligible for the on-disk native-code cache:
    /// integer IR has no relocations, and host-helper calls relocate to a
    /// stable, name-derived helper id re-resolved live at load time.
    #[cfg(feature = "jit")]
    fn emit_extended(&self, ext_id: u16, arg: u8, cx: &mut ExtJitCtx) -> bool {
        let _ = (ext_id, arg, cx);
        false
    }
}

#[cfg(feature = "jit")]
pub use cranelift_jit_impl::ExtJitCtx;

/// Opaque Cranelift SSA value handle produced/consumed by [`ExtJitCtx`] helpers.
/// Frontends thread these between helper calls without depending on Cranelift.
#[cfg(feature = "jit")]
pub type JitValue = cranelift_codegen::ir::Value;

/// Global registry of JIT extensions, consulted by the block JIT for any
/// `Op::Extended` it encounters. Registration is process-global because the
/// codegen path is reached through free functions (and frontends typically
/// create throwaway [`JitCompiler`] instances per call), so per-instance
/// registration would never reach the compiler.
fn ext_registry() -> &'static std::sync::RwLock<Vec<std::sync::Arc<dyn JitExtension>>> {
    static REG: std::sync::OnceLock<std::sync::RwLock<Vec<std::sync::Arc<dyn JitExtension>>>> =
        std::sync::OnceLock::new();
    REG.get_or_init(|| std::sync::RwLock::new(Vec::new()))
}

/// Register a JIT extension process-wide. Call once, before the first chunk is
/// compiled (block-JIT eligibility is memoised per chunk, so an extension
/// registered after a chunk was first judged ineligible will not retroactively
/// make it eligible). Idempotent for a given extension `name`.
pub fn register_global_extension(ext: std::sync::Arc<dyn JitExtension>) {
    let mut reg = ext_registry().write().unwrap();
    if reg.iter().any(|e| e.name() == ext.name()) {
        return;
    }
    tracing::info!(name = ext.name(), ops = ext.op_count(), "JIT extension registered (global)");
    reg.push(ext);
}

/// Find the registered extension that handles `ext_id`, if any.
#[cfg_attr(not(feature = "jit"), allow(dead_code))]
pub(crate) fn global_extension_for(ext_id: u16) -> Option<std::sync::Arc<dyn JitExtension>> {
    ext_registry()
        .read()
        .unwrap()
        .iter()
        .find(|e| e.can_jit(ext_id))
        .cloned()
}

/// A host helper function an extension can call from JIT-compiled code. The
/// helper takes `n_args` `i64` arguments and returns one value (`i64`, or `f64`
/// when `ret_float`). String/pointer operands are passed as `i64` handles.
#[derive(Clone)]
pub(crate) struct ExtHelper {
    pub name: &'static str,
    pub ptr: usize,
    pub n_args: u8,
    pub ret_float: bool,
}

fn ext_helper_registry() -> &'static std::sync::RwLock<Vec<ExtHelper>> {
    static REG: std::sync::OnceLock<std::sync::RwLock<Vec<ExtHelper>>> = std::sync::OnceLock::new();
    REG.get_or_init(|| std::sync::RwLock::new(Vec::new()))
}

/// 32-bit FNV-1a hash, used to derive a stable host-helper id from its name so
/// the value persists across processes (and therefore across cache files).
fn fnv1a32(s: &str) -> u32 {
    let mut h: u32 = 0x811c_9dc5;
    for b in s.as_bytes() {
        h ^= *b as u32;
        h = h.wrapping_mul(0x0100_0193);
    }
    h
}

/// Stable host-helper id for a given helper name. Extension helper ids always
/// have the top bit set, keeping them disjoint from the built-in helpers
/// (`H_POW_I64`..`H_LOGNOT`, which use small ids `0..4`).
pub fn jit_helper_id(name: &str) -> u32 {
    0x8000_0000 | (fnv1a32(name) & 0x7fff_ffff)
}

/// Register a host helper callable from extension JIT codegen and return its
/// stable id (pass it to [`ExtJitCtx::call_host`]). The helper must be an
/// `extern "C"` function taking `n_args` `i64` arguments and returning an `i64`
/// (or `f64` when `ret_float`). Idempotent by `name`; call once at startup
/// alongside [`register_global_extension`]. Because the helper is identified by
/// a stable, name-derived id and resolved live at load time, chunks that call
/// it remain eligible for the on-disk native-code cache.
///
/// # Safety
/// `ptr` must point to a valid function with exactly the declared ABI; calling
/// it with a mismatched signature is undefined behaviour.
pub unsafe fn register_jit_helper(
    name: &'static str,
    ptr: *const u8,
    n_args: u8,
    ret_float: bool,
) -> u32 {
    let mut reg = ext_helper_registry().write().unwrap();
    if !reg.iter().any(|h| h.name == name) {
        reg.push(ExtHelper { name, ptr: ptr as usize, n_args, ret_float });
        tracing::info!(name, n_args, ret_float, "JIT host helper registered (global)");
    }
    jit_helper_id(name)
}

/// Look up a registered host helper by its stable id.
#[cfg_attr(not(feature = "jit"), allow(dead_code))]
pub(crate) fn ext_helper_by_id(id: u32) -> Option<ExtHelper> {
    ext_helper_registry()
        .read()
        .unwrap()
        .iter()
        .find(|h| jit_helper_id(h.name) == id)
        .cloned()
}

/// Snapshot of all registered host helpers (used to seed `JITBuilder` symbols).
#[cfg_attr(not(feature = "jit"), allow(dead_code))]
pub(crate) fn ext_helpers_snapshot() -> Vec<ExtHelper> {
    ext_helper_registry().read().unwrap().clone()
}

/// Slot type tag observed at trace recording time.
///
/// Used by the tracing JIT entry guard: if a slot's runtime type at trace
/// invocation differs from the type recorded at compile time, the trace
/// is skipped and the interpreter handles the iteration.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum SlotKind {
    /// Slot holds an `i64` at trace-record time.
    Int,
    /// Slot holds an `f64` at trace-record time.
    Float,
}

/// Result of a typed block-JIT invocation. The block tier returns an `i64`
/// register; a float result rides back as its raw `f64` bit pattern. This enum
/// lets a typed caller recover the exact value instead of truncating floats to
/// integers (the behavior of the plain `i64`-returning entry points).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BlockNum {
    /// Integer result.
    Int(i64),
    /// Float result (decoded from the returned bit pattern).
    Float(f64),
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
    /// Hash of the chunk's ops + constants pool at trace-record time.
    /// Used to detect chunk drift on import: a mismatched hash means
    /// the bytecode has changed and the persisted trace is stale.
    pub chunk_op_hash: u64,
    /// Bytecode IP where the trace was anchored (the backward-branch
    /// header that crossed `TRACE_THRESHOLD`).
    pub anchor_ip: usize,
    /// Bytecode IP the interpreter should resume at after the trace
    /// runs to completion (one past the loop's natural exit).
    pub fallthrough_ip: usize,
    /// The captured op sequence in record order; the trace is a
    /// straight-line projection of these ops with side-exit guards
    /// inserted at each branch.
    pub ops: Vec<crate::op::Op>,
    /// Parallel array to `ops`: the original bytecode IP each recorded
    /// op corresponds to. Used to materialise the interpreter's IP on
    /// deopt so execution resumes at the right place.
    pub recorded_ips: Vec<usize>,
    /// Slot type fingerprint at the entry guard. The trace's entry
    /// stub compares each slot's runtime type against this snapshot;
    /// mismatch → skip the trace and let the interpreter handle the
    /// iteration.
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
    Ran {
        /// The bytecode IP to resume the interpreter at.
        resume_ip: usize,
    },
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
    /// Whole-chunk invocations before the block JIT compiles a chunk.
    /// Below this, `try_run_block` returns `None` and the caller falls back
    /// to the interpreter — avoiding compile cost for one-shot chunks.
    /// Defaults to 1 (compile on the 2nd invocation) so re-run-heavy
    /// workloads reach native code fast; raise it for cold-start scripts that
    /// run once, or override per-process with `FUSEVM_JIT_BLOCK_THRESHOLD`.
    pub block_threshold: u32,
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
    /// Defaults matching the phase-1-through-9 constants, except
    /// `block_threshold` which defaults to 1 (compile on the 2nd invocation)
    /// to favor re-run-heavy workloads. Per-process env overrides
    /// (`FUSEVM_JIT_BLOCK_THRESHOLD` / `FUSEVM_JIT_TRACE_THRESHOLD`) are applied
    /// on top of these when a thread first touches the JIT.
    pub const fn defaults() -> Self {
        Self {
            trace_threshold: 50,
            block_threshold: 1,
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
/// `STACK_KIND_FLOAT` tag — see [`STACK_KIND_INT`] doc-comment for the contract.
pub const STACK_KIND_FLOAT: u8 = 1;

thread_local! {
    /// Trap channel for JIT-compiled `AwkDivJit`/`AwkModJit`.
    ///
    /// Float `fdiv`/`fmod` do not hardware-trap on a zero divisor (they yield
    /// inf/nan), but awk requires a fatal "division by zero" error. The block
    /// JIT therefore emits a guarded early-exit: on a zero divisor it calls
    /// [`fusevm_jit_awk_div_trap`], which records `1` (div) or `2` (mod) here,
    /// then returns a sentinel. The VM reads the code via [`take_awk_div_trap`]
    /// immediately after a block invocation returns and converts a nonzero code
    /// into the awk fatal, discarding the block's (garbage) result.
    static AWK_DIV_TRAP: std::cell::Cell<u8> = const { std::cell::Cell::new(0) };
}

/// Libcall invoked by JIT-compiled `AwkDivJit`/`AwkModJit` on a zero divisor.
/// `kind` is `1` for division, `2` for modulo. Records the trap so the VM can
/// raise the awk fatal after the block returns.
pub(crate) extern "C" fn fusevm_jit_awk_div_trap(kind: i64) {
    AWK_DIV_TRAP.with(|c| c.set(kind as u8));
}

/// Read and clear the pending div/mod trap code (`0` = none, `1` = div,
/// `2` = mod) set by [`fusevm_jit_awk_div_trap`].
pub(crate) fn take_awk_div_trap() -> u8 {
    AWK_DIV_TRAP.with(|c| c.replace(0))
}

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
    /// Bytecode IP the interpreter should resume at (set on every
    /// trace exit — normal fallthrough and side-exit alike).
    pub resume_ip: usize,
    /// Number of materialised callee frames in `frames` (0 for
    /// caller-frame side-exits, 1..=`MAX_DEOPT_FRAMES` for callee).
    pub frame_count: usize,
    /// Number of abstract-stack entries in `stack_buf` to push onto
    /// `vm.stack` after the trace returns.
    pub stack_count: usize,
    /// Callee frame snapshots (slot vectors + return IPs) for the
    /// VM to restore. Slots 0..`frame_count` are valid.
    pub frames: [DeoptFrame; MAX_DEOPT_FRAMES],
    /// Raw bit pattern for each abstract-stack entry; reinterpret as
    /// `i64` or `f64` based on `stack_kinds[i]`.
    pub stack_buf: [i64; MAX_DEOPT_STACK],
    /// Per-entry tag for `stack_buf`: `STACK_KIND_INT` or
    /// `STACK_KIND_FLOAT`.
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

    /// Keeps the executable memory backing a [`CompiledLinear`] alive. The
    /// JIT path owns a `JITModule`; the on-disk-cache path owns an mmap'd
    /// region of relocated native code.
    enum LinearBacking {
        #[allow(dead_code)]
        Jit(JITModule),
        #[cfg(feature = "jit-disk-cache")]
        #[allow(dead_code)]
        Native(disk_cache::LoadedNative),
    }

    pub(crate) struct CompiledLinear {
        #[allow(dead_code)]
        backing: LinearBacking,
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
            Op::PowFloat => {
                let (a, b) = pop2_strict(stack)?;
                match (a, b) {
                    (Cell::ConstF(x), Cell::ConstF(y)) => stack.push(Cell::ConstF(x.powf(y))),
                    _ => stack.push(Cell::DynF),
                }
            }
            // awk int(x): truncate toward zero. An integer operand is already
            // integral (identity); a float is truncated. Matches awkrs
            // `Value::Num(as_number().trunc())` (bignum path excluded upstream).
            Op::AwkInt => {
                let a = stack.pop()?;
                stack.push(match a {
                    Cell::Const(n) => Cell::Const(n),
                    Cell::ConstF(f) => Cell::ConstF(f.trunc()),
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
            Op::PreDecSlot(_) | Op::PostIncSlot(_) | Op::PostDecSlot(_) => stack.push(Cell::Dyn),
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
                    | Op::PreDecSlot(_)
                    | Op::PostIncSlot(_)
                    | Op::PostDecSlot(_)
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

    /// awk `sin(x)` — radians. NaN result canonicalized to a positive NaN so the
    /// JIT matches awkrs/gawk (`+nan`), never a platform `-nan`.
    #[no_mangle]
    pub extern "C" fn fusevm_jit_sin_f64(x: f64) -> f64 {
        let r = x.sin();
        if r.is_nan() {
            f64::NAN
        } else {
            r
        }
    }

    /// awk `cos(x)` — radians. NaN canonicalized to `+nan` (see `sin`).
    #[no_mangle]
    pub extern "C" fn fusevm_jit_cos_f64(x: f64) -> f64 {
        let r = x.cos();
        if r.is_nan() {
            f64::NAN
        } else {
            r
        }
    }

    /// awk `exp(x)` — e^x. NaN canonicalized to `+nan` (see `sin`).
    #[no_mangle]
    pub extern "C" fn fusevm_jit_exp_f64(x: f64) -> f64 {
        let r = x.exp();
        if r.is_nan() {
            f64::NAN
        } else {
            r
        }
    }

    /// awk `atan2(y, x)`. NaN canonicalized to `+nan` (see `sin`).
    #[no_mangle]
    pub extern "C" fn fusevm_jit_atan2_f64(y: f64, x: f64) -> f64 {
        let r = y.atan2(x);
        if r.is_nan() {
            f64::NAN
        } else {
            r
        }
    }

    /// FuncIds for the awk transcendental libcalls, declared lazily on a module
    /// only when the corresponding op appears in the chunk (so chunks without
    /// them compile to byte-identical native code as before).
    #[derive(Default)]
    struct MathIds {
        sin: Option<cranelift_module::FuncId>,
        cos: Option<cranelift_module::FuncId>,
        exp: Option<cranelift_module::FuncId>,
        atan2: Option<cranelift_module::FuncId>,
        awk_div_trap: Option<cranelift_module::FuncId>,
    }

    /// Resolved `FuncRef`s for the awk transcendental libcalls, valid inside one
    /// function builder.
    #[derive(Default, Clone, Copy)]
    struct MathRefs {
        sin: Option<cranelift_codegen::ir::FuncRef>,
        cos: Option<cranelift_codegen::ir::FuncRef>,
        exp: Option<cranelift_codegen::ir::FuncRef>,
        atan2: Option<cranelift_codegen::ir::FuncRef>,
        awk_div_trap: Option<cranelift_codegen::ir::FuncRef>,
    }

    fn declare_unary_f64(module: &mut JITModule, name: &str) -> Option<cranelift_module::FuncId> {
        let mut ps = module.make_signature();
        ps.params.push(AbiParam::new(types::F64));
        ps.returns.push(AbiParam::new(types::F64));
        module.declare_function(name, Linkage::Import, &ps).ok()
    }

    fn declare_binary_f64(module: &mut JITModule, name: &str) -> Option<cranelift_module::FuncId> {
        let mut ps = module.make_signature();
        ps.params.push(AbiParam::new(types::F64));
        ps.params.push(AbiParam::new(types::F64));
        ps.returns.push(AbiParam::new(types::F64));
        module.declare_function(name, Linkage::Import, &ps).ok()
    }

    fn declare_void_i64(module: &mut JITModule, name: &str) -> Option<cranelift_module::FuncId> {
        let mut ps = module.make_signature();
        ps.params.push(AbiParam::new(types::I64));
        module.declare_function(name, Linkage::Import, &ps).ok()
    }

    impl MathIds {
        /// Declare the imports needed by `ops` (only the ones actually present).
        fn declare(module: &mut JITModule, ops: &[Op]) -> Self {
            let mut m = MathIds::default();
            for op in ops {
                match op {
                    Op::AwkSin if m.sin.is_none() => {
                        m.sin = declare_unary_f64(module, "fusevm_jit_sin_f64");
                    }
                    Op::AwkCos if m.cos.is_none() => {
                        m.cos = declare_unary_f64(module, "fusevm_jit_cos_f64");
                    }
                    Op::AwkExp if m.exp.is_none() => {
                        m.exp = declare_unary_f64(module, "fusevm_jit_exp_f64");
                    }
                    Op::AwkAtan2 if m.atan2.is_none() => {
                        m.atan2 = declare_binary_f64(module, "fusevm_jit_atan2_f64");
                    }
                    Op::AwkDivJit | Op::AwkModJit if m.awk_div_trap.is_none() => {
                        m.awk_div_trap = declare_void_i64(module, "fusevm_jit_awk_div_trap");
                    }
                    _ => {}
                }
            }
            m
        }

        /// Bind the declared `FuncId`s into a function builder's namespace.
        fn resolve(&self, module: &mut JITModule, func: &mut cranelift_codegen::ir::Function) -> MathRefs {
            MathRefs {
                sin: self.sin.map(|id| module.declare_func_in_func(id, func)),
                cos: self.cos.map(|id| module.declare_func_in_func(id, func)),
                exp: self.exp.map(|id| module.declare_func_in_func(id, func)),
                atan2: self.atan2.map(|id| module.declare_func_in_func(id, func)),
                awk_div_trap: self
                    .awk_div_trap
                    .map(|id| module.declare_func_in_func(id, func)),
            }
        }
    }

    fn new_jit_module() -> Option<JITModule> {
        let isa = cached_owned_isa()?.clone();
        let mut builder = JITBuilder::with_isa(isa, default_libcall_names());
        builder.symbol("fusevm_jit_pow_i64", fusevm_jit_pow_i64 as *const u8);
        builder.symbol("fusevm_jit_pow_f64", fusevm_jit_pow_f64 as *const u8);
        builder.symbol("fusevm_jit_fmod_f64", fusevm_jit_fmod_f64 as *const u8);
        builder.symbol("fusevm_jit_lognot_i64", fusevm_jit_lognot_i64 as *const u8);
        builder.symbol("fusevm_jit_sin_f64", fusevm_jit_sin_f64 as *const u8);
        builder.symbol("fusevm_jit_cos_f64", fusevm_jit_cos_f64 as *const u8);
        builder.symbol("fusevm_jit_exp_f64", fusevm_jit_exp_f64 as *const u8);
        builder.symbol("fusevm_jit_atan2_f64", fusevm_jit_atan2_f64 as *const u8);
        builder.symbol(
            "fusevm_jit_awk_div_trap",
            super::fusevm_jit_awk_div_trap as *const u8,
        );
        for h in super::ext_helpers_snapshot() {
            builder.symbol(h.name, h.ptr as *const u8);
        }
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

    /// Pop one operand and coerce it to an `i64` SSA value using *saturating*
    /// float→int conversion, matching awkrs's `num_to_u64` = `n.trunc() as i64`
    /// (Rust's `as` saturates: NaN→0, +inf→i64::MAX, -inf→i64::MIN). Used by the
    /// AWK bitwise fold arms (`and`/`or`/`xor`). Distinct from `f64_to_i64_trunc`
    /// (`fcvt_to_sint`, which *traps* out of range) — the bitwise builtins must
    /// not trap on huge/NaN operands, they wrap like awkrs.
    fn pop_as_i64_sat(bcx: &mut FunctionBuilder, stack: &mut Vec<(Value, JitTy)>) -> Option<Value> {
        let (v, ty) = stack.pop()?;
        Some(match ty {
            JitTy::Int => v,
            JitTy::Float => bcx.ins().fcvt_to_sint_sat(types::I64, v),
        })
    }

    /// Pop one operand and coerce it to an `f64` SSA value (Int operands are
    /// widened via `fcvt_from_sint`). Used by the unary/binary transcendental
    /// libcall arms, which always operate in floating point.
    fn pop_as_f64(bcx: &mut FunctionBuilder, stack: &mut Vec<(Value, JitTy)>) -> Option<Value> {
        let (v, ty) = stack.pop()?;
        Some(match ty {
            JitTy::Float => v,
            JitTy::Int => i64_to_f64(bcx, v),
        })
    }

    /// Hash of a slot-kind snapshot (Float vs Int per slot). Folded into the
    /// block-JIT cache key (TLS + on-disk) so native code specialized for Int
    /// slots is never reused for Float slots, whose `GetSlot`/`SetSlot` bit-cast
    /// and arithmetic differ. Only the Float/Int distinction matters.
    pub(crate) fn slot_kinds_hash(kinds: &[super::SlotKind]) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        kinds.len().hash(&mut h);
        for k in kinds {
            (matches!(k, super::SlotKind::Float) as u8).hash(&mut h);
        }
        h.finish()
    }

    /// Codegen handle handed to [`super::JitExtension::emit_extended`]. Wraps the
    /// in-flight Cranelift function builder and the block's abstract operand
    /// stack, exposing just enough to lower a numeric Extended op: pop i64
    /// inputs, emit IR via [`ExtJitCtx::builder`], and push the i64 result.
    ///
    /// Floats on the stack are coerced to i64 on pop (matching the universal
    /// integer ops), so extensions operate purely on `types::I64` values.
    pub struct ExtJitCtx<'a, 'f> {
        bcx: &'a mut FunctionBuilder<'f>,
        stack: &'a mut Vec<(Value, JitTy)>,
        module: &'a mut JITModule,
        /// Cache of host helpers imported into the current function, keyed by
        /// stable helper id. Doubles as the collector of `(FuncId, stable id)`
        /// pairs threaded into the disk-cache relocation mapper.
        helpers: &'a mut HashMap<u32, (cranelift_module::FuncId, cranelift_codegen::ir::FuncRef)>,
    }

    impl<'a, 'f> ExtJitCtx<'a, 'f> {
        /// Pop the top operand as an `i64` Cranelift value (coercing a float to
        /// a truncated integer). Returns `None` if the abstract stack is empty,
        /// which the caller treats as a codegen failure.
        pub fn pop_i64(&mut self) -> Option<Value> {
            let (v, ty) = self.stack.pop()?;
            Some(scalar_store_i64(self.bcx, v, ty))
        }

        /// Push an `i64` Cranelift value as the new top operand.
        pub fn push_i64(&mut self, v: Value) {
            self.stack.push((v, JitTy::Int));
        }

        /// Call a registered host helper (see [`super::register_jit_helper`]) by
        /// its stable id, passing `args` as `i64` values and returning the
        /// helper's single result `Value` (`i64`, or `f64` for a `ret_float`
        /// helper). Returns `None` if the id is not registered. The helper is
        /// imported into the current function lazily and recorded so the chunk's
        /// native code stays disk-cacheable.
        pub fn call_host(&mut self, helper_id: u32, args: &[Value]) -> Option<Value> {
            let fref = match self.helpers.get(&helper_id) {
                Some((_, fref)) => *fref,
                None => {
                    let h = super::ext_helper_by_id(helper_id)?;
                    let mut ps = self.module.make_signature();
                    for _ in 0..h.n_args {
                        ps.params.push(AbiParam::new(types::I64));
                    }
                    ps.returns.push(AbiParam::new(if h.ret_float {
                        types::F64
                    } else {
                        types::I64
                    }));
                    let fid = self
                        .module
                        .declare_function(h.name, Linkage::Import, &ps)
                        .ok()?;
                    let fref = self.module.declare_func_in_func(fid, self.bcx.func);
                    self.helpers.insert(helper_id, (fid, fref));
                    fref
                }
            };
            let call = self.bcx.ins().call(fref, args);
            self.bcx.inst_results(call).first().copied()
        }

        /// Access the underlying Cranelift function builder to emit IR. Use with
        /// `cranelift_codegen::ir::InstBuilder` (`builder().ins()…`). Most
        /// extensions can use the typed integer helpers below instead and avoid
        /// depending on Cranelift directly.
        pub fn builder(&mut self) -> &mut FunctionBuilder<'f> {
            self.bcx
        }

        // ── Cranelift-free integer emit helpers (sufficient for numeric ops) ──

        /// Materialise an `i64` constant.
        pub fn iconst(&mut self, n: i64) -> Value {
            self.bcx.ins().iconst(types::I64, n)
        }
        /// `a + b` (wrapping i64).
        pub fn iadd(&mut self, a: Value, b: Value) -> Value {
            self.bcx.ins().iadd(a, b)
        }
        /// `a - b` (wrapping i64).
        pub fn isub(&mut self, a: Value, b: Value) -> Value {
            self.bcx.ins().isub(a, b)
        }
        /// `a * b` (wrapping i64).
        pub fn imul(&mut self, a: Value, b: Value) -> Value {
            self.bcx.ins().imul(a, b)
        }
        /// Signed `a / b` (truncated). Caller must guarantee `b != 0` and not
        /// `INT_MIN / -1`, otherwise the hardware divide traps.
        pub fn sdiv(&mut self, a: Value, b: Value) -> Value {
            self.bcx.ins().sdiv(a, b)
        }
        /// Signed `a % b` (truncated, sign of dividend). Same trap caveat as
        /// [`ExtJitCtx::sdiv`].
        pub fn srem(&mut self, a: Value, b: Value) -> Value {
            self.bcx.ins().srem(a, b)
        }
        /// Bitwise `a & b`.
        pub fn band(&mut self, a: Value, b: Value) -> Value {
            self.bcx.ins().band(a, b)
        }
        /// Bitwise `a | b`.
        pub fn bor(&mut self, a: Value, b: Value) -> Value {
            self.bcx.ins().bor(a, b)
        }
        /// Bitwise `a ^ b`.
        pub fn bxor(&mut self, a: Value, b: Value) -> Value {
            self.bcx.ins().bxor(a, b)
        }
        /// `a == b` → boolean value (use with [`ExtJitCtx::select`]).
        pub fn icmp_eq(&mut self, a: Value, b: Value) -> Value {
            self.bcx.ins().icmp(IntCC::Equal, a, b)
        }
        /// `a != b` → boolean value.
        pub fn icmp_ne(&mut self, a: Value, b: Value) -> Value {
            self.bcx.ins().icmp(IntCC::NotEqual, a, b)
        }
        /// Signed `a < b` → boolean value.
        pub fn icmp_slt(&mut self, a: Value, b: Value) -> Value {
            self.bcx.ins().icmp(IntCC::SignedLessThan, a, b)
        }
        /// Signed `a > b` → boolean value.
        pub fn icmp_sgt(&mut self, a: Value, b: Value) -> Value {
            self.bcx.ins().icmp(IntCC::SignedGreaterThan, a, b)
        }
        /// `cond ? a : b` (branchless select; `cond` from an `icmp_*` helper).
        pub fn select(&mut self, cond: Value, a: Value, b: Value) -> Value {
            self.bcx.ins().select(cond, a, b)
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
        math: MathRefs,
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
            Op::PowFloat => {
                let b = pop_as_f64(bcx, stack)?;
                let a = pop_as_f64(bcx, stack)?;
                let fr = pow_f64_ref?;
                let call = bcx.ins().call(fr, &[a, b]);
                stack.push((*bcx.inst_results(call).first()?, JitTy::Float));
            }
            Op::Negate => {
                let (a, ty) = stack.pop()?;
                match ty {
                    JitTy::Int => stack.push((bcx.ins().ineg(a), JitTy::Int)),
                    JitTy::Float => stack.push((bcx.ins().fneg(a), JitTy::Float)),
                }
            }
            // awk int(x): truncate toward zero. Integer operand is already
            // integral (identity); float uses Cranelift's `trunc`
            // (roundToIntegralTowardZero), preserving NaN/inf — matching awkrs
            // `Value::Num(as_number().trunc())`.
            Op::AwkInt => {
                let (a, ty) = stack.pop()?;
                match ty {
                    JitTy::Int => stack.push((a, JitTy::Int)),
                    JitTy::Float => stack.push((bcx.ins().trunc(a), JitTy::Float)),
                }
            }
            // awk sin/cos/exp: unary f64 transcendentals via a Rust libcall that
            // canonicalizes NaN to `+nan` (matching awkrs/gawk). Integer operands
            // are widened to f64 first. Result is always Float.
            Op::AwkSin => {
                let a = pop_as_f64(bcx, stack)?;
                let call = bcx.ins().call(math.sin?, &[a]);
                stack.push((*bcx.inst_results(call).first()?, JitTy::Float));
            }
            Op::AwkCos => {
                let a = pop_as_f64(bcx, stack)?;
                let call = bcx.ins().call(math.cos?, &[a]);
                stack.push((*bcx.inst_results(call).first()?, JitTy::Float));
            }
            Op::AwkExp => {
                let a = pop_as_f64(bcx, stack)?;
                let call = bcx.ins().call(math.exp?, &[a]);
                stack.push((*bcx.inst_results(call).first()?, JitTy::Float));
            }
            // awk atan2(y, x): the chunk pushes y then x, so x is on top.
            Op::AwkAtan2 => {
                let x = pop_as_f64(bcx, stack)?;
                let y = pop_as_f64(bcx, stack)?;
                let call = bcx.ins().call(math.atan2?, &[y, x]);
                stack.push((*bcx.inst_results(call).first()?, JitTy::Float));
            }
            // awk and/or/xor(v1, v2, ...): variadic bitwise fold. Each operand is
            // truncated toward zero and saturated to i64 (matching awkrs's
            // `num_to_u64`), folded with the integer bit-op, and pushed as Int —
            // its f64 materialization (`fcvt_from_sint`) matches awkrs's final
            // `… as i64 as f64`. Folding order is irrelevant (and/or/xor are
            // associative + commutative). The `u8` payload is the arg count
            // (≥2, guaranteed by the parser); fewer than 2 bails the block JIT.
            Op::AwkAnd(n) | Op::AwkOr(n) | Op::AwkXor(n) => {
                if (*n as usize) < 2 {
                    return None;
                }
                let mut acc = pop_as_i64_sat(bcx, stack)?;
                for _ in 1..*n {
                    let v = pop_as_i64_sat(bcx, stack)?;
                    acc = match op {
                        Op::AwkAnd(_) => bcx.ins().band(acc, v),
                        Op::AwkOr(_) => bcx.ins().bor(acc, v),
                        _ => bcx.ins().bxor(acc, v),
                    };
                }
                stack.push((acc, JitTy::Int));
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
            Op::PreDecSlot(slot) => {
                let base = slot_base?;
                let off = (*slot as i32) * 8;
                let old = bcx.ins().load(types::I64, MemFlags::trusted(), base, off);
                let one = bcx.ins().iconst(types::I64, 1);
                let new = bcx.ins().isub(old, one);
                bcx.ins().store(MemFlags::trusted(), new, base, off);
                stack.push((new, JitTy::Int));
            }
            Op::PostIncSlot(slot) => {
                let base = slot_base?;
                let off = (*slot as i32) * 8;
                let old = bcx.ins().load(types::I64, MemFlags::trusted(), base, off);
                let one = bcx.ins().iconst(types::I64, 1);
                let new = bcx.ins().iadd(old, one);
                bcx.ins().store(MemFlags::trusted(), new, base, off);
                stack.push((old, JitTy::Int));
            }
            Op::PostDecSlot(slot) => {
                let base = slot_base?;
                let off = (*slot as i32) * 8;
                let old = bcx.ins().load(types::I64, MemFlags::trusted(), base, off);
                let one = bcx.ins().iconst(types::I64, 1);
                let new = bcx.ins().isub(old, one);
                bcx.ins().store(MemFlags::trusted(), new, base, off);
                stack.push((old, JitTy::Int));
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

        let needs_pow = seq.iter().any(|o| matches!(o, Op::Pow | Op::PowFloat));
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

        let math_ids = MathIds::declare(&mut module, seq);

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
            let math = math_ids.resolve(&mut module, bcx.func);

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
                    math,
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
        Some(CompiledLinear {
            backing: LinearBacking::Jit(module),
            run,
        })
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

        // Disk-cache path (opt-in): reuse or build relocatable native code
        // persisted across process restarts, skipping Cranelift codegen.
        #[cfg(feature = "jit-disk-cache")]
        {
            if let Some(dir) = disk_cache::cache_dir() {
                if let Some(compiled) = disk_cache::try_load_or_build(chunk, &dir) {
                    let result = compiled.invoke(slot_ptr);
                    let value = compiled.result_to_value(result);
                    LINEAR_CACHE_TLS.with(|cache_cell| {
                        cache_cell.borrow_mut().insert(key, Box::new(compiled));
                    });
                    return Some(value);
                }
                // Native caching rejected this chunk (e.g. an unsupported
                // relocation): fall through to the in-memory JIT path.
            }
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

    // ── Persistent on-disk native-code cache (linear tier) ──
    //
    // Caches the *native machine code* produced by the linear JIT to disk so
    // repeated processes skip Cranelift codegen. fusevm's linear functions are
    // position-independent; the only relocations are `Abs8` absolute writes to
    // the fixed host helpers. Loading therefore reduces to: map executable
    // memory, copy the code, write the current address of each referenced host
    // helper at its relocation offset, and flip the page to executable.
    //
    // Conservative by design: any chunk whose compiled code contains a
    // relocation that is not an `Abs8` to a known host helper is rejected, and
    // the caller falls back to the in-memory JIT. This keeps the loader correct
    // even on targets whose codegen differs from the ones exercised in tests.
    #[cfg(feature = "jit-disk-cache")]
    pub(crate) mod disk_cache {
        use super::*;
        use cranelift_codegen::binemit::Reloc;
        use cranelift_codegen::ir::{
            ExtFuncData, ExternalName, Signature, UserExternalName,
        };
        use cranelift_codegen::Context;
        use std::hash::{Hash, Hasher};
        use std::path::{Path, PathBuf};
        use std::sync::{OnceLock, RwLock};

        // Stable host-helper identifiers — persisted in cache files and used to
        // re-resolve the helper address at load time. Never renumber these
        // without bumping `SCHEMA_VERSION`.
        const H_POW_I64: u32 = 0;
        const H_POW_F64: u32 = 1;
        const H_FMOD_F64: u32 = 2;
        const H_LOGNOT: u32 = 3;
        const H_SIN_F64: u32 = 4;
        const H_COS_F64: u32 = 5;
        const H_EXP_F64: u32 = 6;
        const H_ATAN2_F64: u32 = 7;
        /// Zero-divisor trap libcall for JIT-compiled `AwkDivJit`/`AwkModJit`.
        /// Registering it as a known host helper lets div/mod chunks (e.g. a
        /// front-end's float `/`) persist to the native disk cache instead of
        /// being limited to the in-memory JIT.
        pub(crate) const H_AWK_DIV_TRAP: u32 = 8;

        // Native-blob tier discriminator. Persisted in the file and verified on
        // load so a block blob can never be transmuted with a linear signature.
        pub(crate) const KIND_LINEAR: u8 = 0;
        pub(crate) const KIND_BLOCK: u8 = 1;
        pub(crate) const KIND_TRACE: u8 = 2;

        const MAGIC: &[u8; 8] = b"FJITNAT2";
        // 7 -> 8: inserted `Op::PowFloat` mid-enum, shifting the serde/Hash
        // discriminants of all following ops; bumped to invalidate any prior
        // op_hash-keyed blobs that would otherwise collide.
        const SCHEMA_VERSION: u32 = 8;

        /// Current address of a host helper by id, or `None` if unknown.
        fn host_addr(id: u32) -> Option<usize> {
            // Extension helper ids have the top bit set; resolve them live via
            // the global helper registry (the frontend must have registered the
            // helper before the cached blob is loaded, else this misses and the
            // blob is rejected → safe recompile).
            if id & 0x8000_0000 != 0 {
                return super::super::ext_helper_by_id(id).map(|h| h.ptr);
            }
            Some(match id {
                H_POW_I64 => super::fusevm_jit_pow_i64 as *const u8 as usize,
                H_POW_F64 => super::fusevm_jit_pow_f64 as *const u8 as usize,
                H_FMOD_F64 => super::fusevm_jit_fmod_f64 as *const u8 as usize,
                H_LOGNOT => super::fusevm_jit_lognot_i64 as *const u8 as usize,
                H_SIN_F64 => super::fusevm_jit_sin_f64 as *const u8 as usize,
                H_COS_F64 => super::fusevm_jit_cos_f64 as *const u8 as usize,
                H_EXP_F64 => super::fusevm_jit_exp_f64 as *const u8 as usize,
                H_ATAN2_F64 => super::fusevm_jit_atan2_f64 as *const u8 as usize,
                H_AWK_DIV_TRAP => super::super::fusevm_jit_awk_div_trap as *const u8 as usize,
                _ => return None,
            })
        }

        /// Compile a fully-built `ctx` to raw bytes + validated host-helper
        /// relocations. `map_index` maps each relocation's
        /// `UserExternalName.index` to a stable host-helper id; the block/trace
        /// native paths build their `ctx` via a `JITModule` (so the index is the
        /// helper's module `FuncId`), while the linear path imports helpers
        /// directly under their host id. Returns `None` (so the caller falls
        /// back to the in-memory JIT) if any relocation is not an `Abs8` to a
        /// known host helper.
        fn compile_and_extract(
            ctx: &mut Context,
            isa: &dyn cranelift_codegen::isa::TargetIsa,
            map_index: impl Fn(u32) -> Option<u32>,
        ) -> Option<(Vec<u8>, Vec<(u32, u32, i64)>)> {
            // Copy out code + relocs as owned data so the mutable borrow of
            // `ctx` ends before we read its name table.
            let (code, raw): (Vec<u8>, Vec<(u32, Reloc, i64, Option<ExternalName>)>) = {
                let compiled = ctx.compile(isa, &mut Default::default()).ok()?;
                let bytes = compiled.code_buffer().to_vec();
                let relocs = compiled
                    .buffer
                    .relocs()
                    .iter()
                    .map(|r| {
                        let name = match &r.target {
                            cranelift_codegen::FinalizedRelocTarget::ExternalName(n) => {
                                Some(n.clone())
                            }
                            _ => None,
                        };
                        (r.offset, r.kind, r.addend, name)
                    })
                    .collect();
                (bytes, relocs)
            };

            let mut relocs = Vec::with_capacity(raw.len());
            for (off, kind, addend, name) in raw {
                if kind != Reloc::Abs8 {
                    return None;
                }
                let uref = match name {
                    Some(ExternalName::User(u)) => u,
                    _ => return None,
                };
                let index = ctx.func.params.user_named_funcs()[uref].index;
                let id = map_index(index)?;
                host_addr(id)?;
                if (off as usize).checked_add(8)? > code.len() {
                    return None;
                }
                relocs.push((off, id, addend));
            }
            Some((code, relocs))
        }

        /// Map a block/trace `ctx`'s relocation index (a module `FuncId` as u32)
        /// to a stable host-helper id, using the helper-FuncId table captured at
        /// build time. Order:
        /// `[pow_i64, pow_f64, fmod_f64, lognot, sin, cos, exp, atan2]`.
        fn map_helper_funcid(
            helper_ids: &[Option<cranelift_module::FuncId>; 8],
            index: u32,
        ) -> Option<u32> {
            const HOST_IDS: [u32; 8] = [
                H_POW_I64, H_POW_F64, H_FMOD_F64, H_LOGNOT, H_SIN_F64, H_COS_F64, H_EXP_F64,
                H_ATAN2_F64,
            ];
            for (slot, id) in helper_ids.iter().zip(HOST_IDS.iter()) {
                if let Some(fid) = slot {
                    if fid.as_u32() == index {
                        return Some(*id);
                    }
                }
            }
            None
        }

        /// Map a relocation index (module `FuncId` as u32) to a stable
        /// host-helper id, consulting both the built-in helpers and any
        /// extension helpers (`(FuncId, stable id)` pairs) used by the chunk.
        fn map_helper_funcid_ext(
            helper_ids: &[Option<cranelift_module::FuncId>; 8],
            ext_helpers: &[(cranelift_module::FuncId, u32)],
            index: u32,
        ) -> Option<u32> {
            if let Some(id) = map_helper_funcid(helper_ids, index) {
                return Some(id);
            }
            ext_helpers
                .iter()
                .find(|(fid, _)| fid.as_u32() == index)
                .map(|(_, id)| *id)
        }

        // ── Cache directory configuration ──

        fn cache_dir_slot() -> &'static RwLock<Option<PathBuf>> {
            static SLOT: OnceLock<RwLock<Option<PathBuf>>> = OnceLock::new();
            SLOT.get_or_init(|| RwLock::new(None))
        }

        /// Override the cache directory programmatically. `Some(dir)` pins an
        /// explicit directory; `None` clears the override so resolution falls
        /// back to the `FUSEVM_JIT_CACHE_DIR` env var and then the default
        /// (`~/.cache/fusevm-jit`).
        pub(crate) fn set_cache_dir(dir: Option<PathBuf>) {
            if let Ok(mut g) = cache_dir_slot().write() {
                *g = dir;
            }
        }

        /// Whether `val` is an explicit "disabled" sentinel for the env var.
        fn is_disabled_value(val: &std::ffi::OsStr) -> bool {
            let s = val.to_string_lossy();
            let t = s.trim();
            t.is_empty()
                || matches!(
                    t.to_ascii_lowercase().as_str(),
                    "off" | "0" | "no" | "none" | "false" | "disabled"
                )
        }

        /// The default cache directory used when nothing else is configured:
        /// `$XDG_CACHE_HOME/fusevm-jit`, else `$HOME/.cache/fusevm-jit`, else a
        /// temp-dir fallback. Disk caching is *on by default* — set
        /// `FUSEVM_JIT_CACHE_DIR=off` (or call `set_jit_cache_dir`) to change it.
        fn default_cache_dir() -> Option<PathBuf> {
            if let Some(x) = std::env::var_os("XDG_CACHE_HOME") {
                if !x.is_empty() {
                    return Some(PathBuf::from(x).join("fusevm-jit"));
                }
            }
            if let Some(h) = std::env::var_os("HOME") {
                if !h.is_empty() {
                    return Some(PathBuf::from(h).join(".cache").join("fusevm-jit"));
                }
            }
            Some(std::env::temp_dir().join("fusevm-jit"))
        }

        /// The active cache directory. Resolution order:
        /// 1. programmatic override (`set_cache_dir`),
        /// 2. `FUSEVM_JIT_CACHE_DIR` env var (a "disabled" sentinel turns
        ///    caching off, any other value is used as the directory),
        /// 3. the default `~/.cache/fusevm-jit`.
        ///
        /// `None` means disk caching is disabled for this run.
        pub(crate) fn cache_dir() -> Option<PathBuf> {
            if let Ok(g) = cache_dir_slot().read() {
                if let Some(p) = g.as_ref() {
                    return Some(p.clone());
                }
            }
            match std::env::var_os("FUSEVM_JIT_CACHE_DIR") {
                Some(val) if is_disabled_value(&val) => None,
                Some(val) => Some(PathBuf::from(val)),
                None => default_cache_dir(),
            }
        }

        /// Identifies the (target, toolchain, schema) a cache file was built
        /// for. A mismatch on load means the file is stale and is ignored.
        fn fingerprint() -> u64 {
            static FP: OnceLock<u64> = OnceLock::new();
            *FP.get_or_init(|| {
                let mut h = std::collections::hash_map::DefaultHasher::new();
                SCHEMA_VERSION.hash(&mut h);
                env!("CARGO_PKG_VERSION").hash(&mut h);
                if let Some(isa) = cached_owned_isa() {
                    isa.triple().to_string().hash(&mut h);
                    (isa.pointer_type().bytes() as u32).hash(&mut h);
                }
                std::mem::size_of::<usize>().hash(&mut h);
                h.finish()
            })
        }

        // ── Serializable native blob ──

        pub(crate) struct NativeBlob {
            kind: u8,
            code: Vec<u8>,
            /// `(code offset, host helper id, addend)` per Abs8 relocation.
            relocs: Vec<(u32, u32, i64)>,
            entry: u32,
            ret_is_float: bool,
            need_slots: bool,
            /// Extra verification word: for traces this is a content hash of the
            /// recording (recorded path + slot types + fallthrough); 0 otherwise.
            aux: u64,
        }

        impl NativeBlob {
            fn to_bytes(&self, op_hash: u64) -> Vec<u8> {
                let mut b = Vec::with_capacity(72 + self.code.len() + self.relocs.len() * 16);
                b.extend_from_slice(MAGIC);
                b.extend_from_slice(&SCHEMA_VERSION.to_le_bytes());
                b.extend_from_slice(&fingerprint().to_le_bytes());
                b.extend_from_slice(&op_hash.to_le_bytes());
                b.extend_from_slice(&self.aux.to_le_bytes());
                b.push(self.kind);
                let flags = (self.ret_is_float as u8) | ((self.need_slots as u8) << 1);
                b.push(flags);
                b.extend_from_slice(&self.entry.to_le_bytes());
                b.extend_from_slice(&(self.code.len() as u32).to_le_bytes());
                b.extend_from_slice(&self.code);
                b.extend_from_slice(&(self.relocs.len() as u32).to_le_bytes());
                for (off, id, addend) in &self.relocs {
                    b.extend_from_slice(&off.to_le_bytes());
                    b.extend_from_slice(&id.to_le_bytes());
                    b.extend_from_slice(&addend.to_le_bytes());
                }
                b
            }

            fn from_bytes(buf: &[u8], expect_op_hash: u64, expect_aux: u64) -> Option<NativeBlob> {
                let mut r = Reader { buf, pos: 0 };
                if r.take(8)? != MAGIC {
                    return None;
                }
                if r.u32()? != SCHEMA_VERSION || r.u64()? != fingerprint() {
                    return None;
                }
                if r.u64()? != expect_op_hash {
                    return None;
                }
                if r.u64()? != expect_aux {
                    return None;
                }
                let kind = r.u8()?;
                let flags = r.u8()?;
                let entry = r.u32()?;
                let code_len = r.u32()? as usize;
                let code = r.take(code_len)?.to_vec();
                let n = r.u32()? as usize;
                let mut relocs = Vec::with_capacity(n);
                for _ in 0..n {
                    let off = r.u32()?;
                    let id = r.u32()?;
                    let addend = r.i64()?;
                    relocs.push((off, id, addend));
                }
                Some(NativeBlob {
                    kind,
                    code,
                    relocs,
                    entry,
                    ret_is_float: flags & 1 != 0,
                    need_slots: flags & 2 != 0,
                    aux: expect_aux,
                })
            }
        }

        struct Reader<'a> {
            buf: &'a [u8],
            pos: usize,
        }
        impl<'a> Reader<'a> {
            fn take(&mut self, n: usize) -> Option<&'a [u8]> {
                let end = self.pos.checked_add(n)?;
                let s = self.buf.get(self.pos..end)?;
                self.pos = end;
                Some(s)
            }
            fn u8(&mut self) -> Option<u8> {
                Some(self.take(1)?[0])
            }
            fn u32(&mut self) -> Option<u32> {
                Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?))
            }
            fn u64(&mut self) -> Option<u64> {
                Some(u64::from_le_bytes(self.take(8)?.try_into().ok()?))
            }
            fn i64(&mut self) -> Option<i64> {
                Some(i64::from_le_bytes(self.take(8)?.try_into().ok()?))
            }
        }

        // ── Native compilation (no Module — raw Context) ──

        /// Compile a chunk's linear sequence to relocatable native code.
        /// Returns `None` if the chunk is ineligible or its code contains a
        /// relocation the loader cannot handle.
        pub(crate) fn compile_linear_native(chunk: &Chunk) -> Option<NativeBlob> {
            let seq = &chunk.ops;
            if !validate_linear_seq(seq, &chunk.constants) {
                return None;
            }
            let ret_cell = linear_result_cell(seq, &chunk.constants)?;
            let ret_ty = cell_to_jit_ty(ret_cell);
            let need_slots = needs_slots(seq);
            let isa = cached_owned_isa()?.clone();
            let call_conv = isa.default_call_conv();
            let ptr_ty = isa.pointer_type();

            let mut sig = Signature::new(call_conv);
            if need_slots {
                sig.params.push(AbiParam::new(ptr_ty));
            }
            sig.returns.push(AbiParam::new(match ret_ty {
                JitTy::Int => types::I64,
                JitTy::Float => types::F64,
            }));

            let mut ctx = Context::new();
            ctx.func.signature = sig;
            ctx.func.name = UserFuncName::user(0, 0);
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

                let import = |bcx: &mut FunctionBuilder,
                              id: u32,
                              params: &[types::Type],
                              ret: types::Type|
                 -> cranelift_codegen::ir::FuncRef {
                    let mut s = Signature::new(call_conv);
                    for p in params {
                        s.params.push(AbiParam::new(*p));
                    }
                    s.returns.push(AbiParam::new(ret));
                    let sref = bcx.import_signature(s);
                    let nref = bcx
                        .func
                        .declare_imported_user_function(UserExternalName::new(0, id));
                    bcx.import_function(ExtFuncData {
                        name: ExternalName::user(nref),
                        signature: sref,
                        colocated: false,
                        patchable: false,
                    })
                };

                let pow_i64_ref =
                    Some(import(&mut bcx, H_POW_I64, &[types::I64, types::I64], types::I64));
                let pow_f64_ref =
                    Some(import(&mut bcx, H_POW_F64, &[types::F64, types::F64], types::F64));
                let fmod_f64_ref =
                    Some(import(&mut bcx, H_FMOD_F64, &[types::F64, types::F64], types::F64));
                let lognot_ref = Some(import(&mut bcx, H_LOGNOT, &[types::I64], types::I64));
                let math = MathRefs {
                    sin: Some(import(&mut bcx, H_SIN_F64, &[types::F64], types::F64)),
                    cos: Some(import(&mut bcx, H_COS_F64, &[types::F64], types::F64)),
                    exp: Some(import(&mut bcx, H_EXP_F64, &[types::F64], types::F64)),
                    atan2: Some(import(
                        &mut bcx,
                        H_ATAN2_F64,
                        &[types::F64, types::F64],
                        types::F64,
                    )),
                    // The linear tier never lowers AwkDivJit/AwkModJit (those are
                    // block-only ops handled in `build_block_function`), so the
                    // linear native path imports no trap ref. The block native
                    // path *does* carry them and registers the trap libcall under
                    // `H_AWK_DIV_TRAP`, so div/mod chunks persist to disk.
                    awk_div_trap: None,
                };

                let mut stack: Vec<(Value, JitTy)> = Vec::with_capacity(32);
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
                        math,
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

            // Compile + extract relocations. The linear path imports helpers
            // directly under their host id, so the relocation index *is* the
            // host id.
            let (code, relocs) = compile_and_extract(&mut ctx, &*isa, |index| {
                if host_addr(index).is_some() {
                    Some(index)
                } else {
                    None
                }
            })?;

            Some(NativeBlob {
                kind: KIND_LINEAR,
                code,
                relocs,
                entry: 0,
                ret_is_float: matches!(ret_ty, JitTy::Float),
                need_slots,
                aux: 0,
            })
        }

        // ── Executable-memory loader ──

        /// Owns an mmap'd region of relocated, executable native code. Unmapped
        /// on drop. Not `Send`/`Sync`: the linear cache is per-thread.
        pub(crate) struct LoadedNative {
            ptr: *mut u8,
            len: usize,
        }

        impl Drop for LoadedNative {
            fn drop(&mut self) {
                unsafe {
                    libc::munmap(self.ptr as *mut libc::c_void, self.len);
                }
            }
        }

        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        extern "C" {
            fn pthread_jit_write_protect_np(enabled: libc::c_int);
            fn sys_icache_invalidate(start: *mut libc::c_void, len: libc::size_t);
        }

        /// Map the blob's code into executable memory and apply its host-helper
        /// relocations. Returns the mapped region (owning handle) and the entry
        /// pointer, or `None` on any mapping/protection/resolution failure.
        fn map_relocate(blob: &NativeBlob) -> Option<(LoadedNative, *const u8)> {
            let code_len = blob.code.len();
            if code_len == 0 {
                return None;
            }
            let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
            if page <= 0 {
                return None;
            }
            let page = page as usize;
            let len = code_len.checked_add(page - 1)? / page * page;

            unsafe {
                #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
                let (prot, flags) = (
                    libc::PROT_READ | libc::PROT_WRITE | libc::PROT_EXEC,
                    libc::MAP_PRIVATE | libc::MAP_ANON | libc::MAP_JIT,
                );
                #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
                let (prot, flags) = (
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_PRIVATE | libc::MAP_ANON,
                );

                let p = libc::mmap(std::ptr::null_mut(), len, prot, flags, -1, 0);
                if p == libc::MAP_FAILED {
                    return None;
                }
                let base = p as *mut u8;

                #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
                pthread_jit_write_protect_np(0);

                std::ptr::copy_nonoverlapping(blob.code.as_ptr(), base, code_len);

                for (off, id, addend) in &blob.relocs {
                    let addr = match host_addr(*id) {
                        Some(a) => a as i64 + *addend,
                        None => {
                            libc::munmap(p, len);
                            return None;
                        }
                    };
                    if (*off as usize) + 8 > code_len {
                        libc::munmap(p, len);
                        return None;
                    }
                    let slot = base.add(*off as usize) as *mut i64;
                    slot.write_unaligned(addr);
                }

                #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
                {
                    pthread_jit_write_protect_np(1);
                    sys_icache_invalidate(p, len);
                }
                #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
                {
                    if libc::mprotect(p, len, libc::PROT_READ | libc::PROT_EXEC) != 0 {
                        libc::munmap(p, len);
                        return None;
                    }
                }

                let entry = base.add(blob.entry as usize) as *const u8;
                Some((LoadedNative { ptr: base, len }, entry))
            }
        }

        /// Map the blob's code into executable memory, apply relocations, and
        /// build a [`CompiledLinear`] that calls into it. `None` on any mapping,
        /// protection, or relocation-resolution failure.
        pub(crate) fn load_native(blob: &NativeBlob) -> Option<CompiledLinear> {
            if blob.kind != KIND_LINEAR {
                return None;
            }
            let (loaded, entry) = map_relocate(blob)?;
            let run = unsafe {
                match (blob.need_slots, blob.ret_is_float) {
                    (false, false) => {
                        LinearRun::Nullary(std::mem::transmute::<*const u8, LinearFn0>(entry))
                    }
                    (false, true) => {
                        LinearRun::NullaryF(std::mem::transmute::<*const u8, LinearFn0F>(entry))
                    }
                    (true, false) => {
                        LinearRun::Slots(std::mem::transmute::<*const u8, LinearFnSlots>(entry))
                    }
                    (true, true) => {
                        LinearRun::SlotsF(std::mem::transmute::<*const u8, LinearFnSlotsF>(entry))
                    }
                }
            };
            Some(CompiledLinear {
                backing: LinearBacking::Native(loaded),
                run,
            })
        }

        /// Map a block blob and build a [`CompiledBlock`]. Block functions are
        /// always `fn(*mut i64) -> i64` (the `SlotsI` variant).
        pub(crate) fn load_native_block(blob: &NativeBlob) -> Option<CompiledBlock> {
            if blob.kind != KIND_BLOCK {
                return None;
            }
            let (loaded, entry) = map_relocate(blob)?;
            let run = BlockRun::SlotsI(unsafe {
                std::mem::transmute::<*const u8, BlockFnSlotsI>(entry)
            });
            Some(CompiledBlock {
                backing: BlockBacking::Native(loaded),
                run,
                ret_is_float: blob.ret_is_float,
            })
        }

        /// Map a trace blob and build a [`CompiledTrace`]. Trace functions are
        /// always `fn(*mut i64, *mut DeoptInfo) -> i64`.
        pub(crate) fn load_native_trace(blob: &NativeBlob) -> Option<CompiledTrace> {
            if blob.kind != KIND_TRACE {
                return None;
            }
            let (loaded, entry) = map_relocate(blob)?;
            let run = unsafe { std::mem::transmute::<*const u8, TraceFn>(entry) };
            Some(CompiledTrace {
                backing: TraceBacking::Native(loaded),
                run,
            })
        }

        // ── File I/O ──

        fn cache_path(dir: &Path, tag: &str, op_hash: u64, sub: u64) -> PathBuf {
            dir.join(format!("{op_hash:016x}.{sub:016x}.{tag}.fjit"))
        }

        fn write_blob(dir: &Path, tag: &str, op_hash: u64, sub: u64, blob: &NativeBlob) {
            let _ = std::fs::create_dir_all(dir);
            let bytes = blob.to_bytes(op_hash);
            // Atomic publish: write to a *unique* temp file then rename, so a
            // reader never observes a partial file and concurrent writers (other
            // threads or processes, even for the same key) never share a temp
            // path and clobber each other mid-write. Uniqueness = pid + a global
            // monotonic counter.
            static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let seq = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let tmp = dir.join(format!(
                "{op_hash:016x}.{sub:016x}.{tag}.{}.{seq}.tmp",
                std::process::id()
            ));
            if std::fs::write(&tmp, &bytes).is_ok() {
                if std::fs::rename(&tmp, cache_path(dir, tag, op_hash, sub)).is_err() {
                    let _ = std::fs::remove_file(&tmp);
                } else if seq.is_multiple_of(PRUNE_INTERVAL) {
                    // Amortized cap enforcement: scan + evict roughly once per
                    // PRUNE_INTERVAL writes rather than on every write.
                    let _ = prune(dir, max_bytes());
                }
            } else {
                let _ = std::fs::remove_file(&tmp);
            }
        }

        fn read_blob(
            dir: &Path,
            tag: &str,
            op_hash: u64,
            sub: u64,
            expect_aux: u64,
        ) -> Option<NativeBlob> {
            let bytes = std::fs::read(cache_path(dir, tag, op_hash, sub)).ok()?;
            NativeBlob::from_bytes(&bytes, op_hash, expect_aux)
        }

        // ── Cache size management ──

        /// Default cap on total on-disk cache size: 256 MiB. Files are tiny
        /// (~100 bytes for linear blobs, up to a few KB for block/trace), so
        /// this holds a very large working set; override via
        /// `FUSEVM_JIT_CACHE_MAX_BYTES` or `set_max_bytes`.
        const DEFAULT_MAX_BYTES: u64 = 256 * 1024 * 1024;

        /// How often (in `write_blob` calls) to opportunistically prune, so the
        /// directory scan cost is amortized instead of paid on every write.
        const PRUNE_INTERVAL: u64 = 128;

        fn max_bytes_slot() -> &'static RwLock<Option<u64>> {
            static SLOT: OnceLock<RwLock<Option<u64>>> = OnceLock::new();
            SLOT.get_or_init(|| RwLock::new(None))
        }

        /// Programmatic override for the cache size cap. `Some(0)` means
        /// unlimited (never prune); `None` falls back to the
        /// `FUSEVM_JIT_CACHE_MAX_BYTES` env var, then `DEFAULT_MAX_BYTES`.
        pub(crate) fn set_max_bytes(limit: Option<u64>) {
            if let Ok(mut g) = max_bytes_slot().write() {
                *g = limit;
            }
        }

        /// Resolved cap in bytes; `0` means unlimited (pruning disabled).
        pub(crate) fn max_bytes() -> u64 {
            if let Ok(g) = max_bytes_slot().read() {
                if let Some(v) = *g {
                    return v;
                }
            }
            match std::env::var("FUSEVM_JIT_CACHE_MAX_BYTES") {
                Ok(s) => parse_size(&s).unwrap_or(DEFAULT_MAX_BYTES),
                Err(_) => DEFAULT_MAX_BYTES,
            }
        }

        /// Parse a size string: a plain byte count, an optional binary suffix
        /// (`k`/`m`/`g`, case-insensitive), or a disable sentinel
        /// (`0`/`off`/`none`/`unlimited`) → `0`. Returns `None` if unparsable.
        fn parse_size(s: &str) -> Option<u64> {
            let lower = s.trim().to_ascii_lowercase();
            if matches!(lower.as_str(), "0" | "off" | "none" | "unlimited") {
                return Some(0);
            }
            let (num, mult) = if let Some(p) = lower.strip_suffix('g') {
                (p, 1u64 << 30)
            } else if let Some(p) = lower.strip_suffix('m') {
                (p, 1u64 << 20)
            } else if let Some(p) = lower.strip_suffix('k') {
                (p, 1u64 << 10)
            } else {
                (lower.as_str(), 1u64)
            };
            num.trim()
                .parse::<u64>()
                .ok()
                .map(|n| n.saturating_mul(mult))
        }

        /// `(path, size, mtime)` for every cache blob in `dir`.
        fn fjit_entries(dir: &Path) -> Vec<(PathBuf, u64, std::time::SystemTime)> {
            let mut v = Vec::new();
            if let Ok(rd) = std::fs::read_dir(dir) {
                for e in rd.flatten() {
                    let p = e.path();
                    if p.extension().and_then(|s| s.to_str()) == Some("fjit") {
                        if let Ok(m) = e.metadata() {
                            let t = m.modified().unwrap_or(std::time::UNIX_EPOCH);
                            v.push((p, m.len(), t));
                        }
                    }
                }
            }
            v
        }

        /// Total bytes of cache blobs in `dir`.
        pub(crate) fn cache_size_bytes(dir: &Path) -> u64 {
            fjit_entries(dir).iter().map(|(_, s, _)| *s).sum()
        }

        /// Remove every cache blob in `dir`; returns the count removed.
        pub(crate) fn clear(dir: &Path) -> usize {
            let mut n = 0;
            for (p, _, _) in fjit_entries(dir) {
                if std::fs::remove_file(&p).is_ok() {
                    n += 1;
                }
            }
            n
        }

        /// Evict oldest-first until total size ≤ 80% of `max`. No-op when
        /// `max == 0` (unlimited) or already under cap. Returns bytes freed.
        pub(crate) fn prune(dir: &Path, max: u64) -> u64 {
            if max == 0 {
                return 0;
            }
            let mut entries = fjit_entries(dir);
            let total: u64 = entries.iter().map(|(_, s, _)| *s).sum();
            if total <= max {
                return 0;
            }
            // Low-water mark avoids pruning on every subsequent write once at cap.
            let target = max - max / 5;
            entries.sort_by_key(|(_, _, t)| *t); // oldest first
            let mut cur = total;
            let mut freed = 0;
            for (p, size, _) in entries {
                if cur <= target {
                    break;
                }
                if std::fs::remove_file(&p).is_ok() {
                    cur = cur.saturating_sub(size);
                    freed += size;
                }
            }
            freed
        }

        #[cfg(test)]
        mod size_tests {
            use super::{cache_size_bytes, clear, parse_size, prune};
            use std::path::PathBuf;

            #[test]
            fn parse_size_units_and_sentinels() {
                assert_eq!(parse_size("1024"), Some(1024));
                assert_eq!(parse_size(" 2k "), Some(2 * 1024));
                assert_eq!(parse_size("3M"), Some(3 * 1024 * 1024));
                assert_eq!(parse_size("1g"), Some(1024 * 1024 * 1024));
                assert_eq!(parse_size("0"), Some(0));
                assert_eq!(parse_size("off"), Some(0));
                assert_eq!(parse_size("unlimited"), Some(0));
                assert_eq!(parse_size("garbage"), None);
            }

            fn tmp_dir(tag: &str) -> PathBuf {
                let d = std::env::temp_dir().join(format!(
                    "fusevm-jit-size-test-{}-{}-{}",
                    tag,
                    std::process::id(),
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_nanos()
                ));
                std::fs::create_dir_all(&d).unwrap();
                d
            }

            fn write_blob_file(dir: &std::path::Path, name: &str, bytes: usize) {
                std::fs::write(dir.join(format!("{name}.fjit")), vec![0u8; bytes]).unwrap();
                // Space out mtimes so oldest-first eviction is deterministic.
                std::thread::sleep(std::time::Duration::from_millis(10));
            }

            #[test]
            fn prune_evicts_oldest_until_under_low_water() {
                let dir = tmp_dir("prune");
                // 10 blobs × 100 bytes = 1000 bytes total.
                for i in 0..10 {
                    write_blob_file(&dir, &format!("{i:02}"), 100);
                }
                assert_eq!(cache_size_bytes(&dir), 1000);

                // Cap 500 → low-water 400; evict oldest until ≤ 400.
                let freed = prune(&dir, 500);
                let remaining = cache_size_bytes(&dir);
                assert!(remaining <= 400, "expected ≤400 after prune, got {remaining}");
                assert_eq!(freed, 1000 - remaining);

                // The newest blob (09) must survive; the oldest (00) must not.
                assert!(dir.join("09.fjit").exists(), "newest blob should be kept");
                assert!(!dir.join("00.fjit").exists(), "oldest blob should be evicted");

                // max == 0 is unlimited: no-op.
                assert_eq!(prune(&dir, 0), 0);
                std::fs::remove_dir_all(&dir).ok();
            }

            #[test]
            fn clear_removes_all_blobs() {
                let dir = tmp_dir("clear");
                for i in 0..5 {
                    write_blob_file(&dir, &format!("{i}"), 50);
                }
                // A non-blob file must be left untouched.
                std::fs::write(dir.join("keep.txt"), b"x").unwrap();
                assert_eq!(clear(&dir), 5);
                assert_eq!(cache_size_bytes(&dir), 0);
                assert!(dir.join("keep.txt").exists());
                std::fs::remove_dir_all(&dir).ok();
            }
        }

        /// Load native code for `chunk`'s linear sequence from `dir`, compiling
        /// and persisting it first if absent. Returns `None` if the chunk cannot
        /// be natively cached (caller should fall back to the in-memory JIT).
        pub(crate) fn try_load_or_build(chunk: &Chunk, dir: &Path) -> Option<CompiledLinear> {
            if let Some(blob) = read_blob(dir, "lin", chunk.op_hash, 0, 0) {
                if let Some(compiled) = load_native(&blob) {
                    return Some(compiled);
                }
            }
            let blob = compile_linear_native(chunk)?;
            write_blob(dir, "lin", chunk.op_hash, 0, &blob);
            load_native(&blob)
        }

        /// Block-tier equivalent of [`try_load_or_build`]. Keyed by `op_hash`.
        pub(crate) fn try_load_or_build_block(
            chunk: &Chunk,
            dir: &Path,
            slot_kinds: &[crate::SlotKind],
        ) -> Option<CompiledBlock> {
            // The slot-kinds hash is the verification aux word: native code
            // compiled for Int slots must not be reused for Float slots (the
            // bit-cast/arithmetic differs), so a kinds mismatch rejects the file.
            let kinds_hash = slot_kinds_hash(slot_kinds);
            if let Some(blob) = read_blob(dir, "blk", chunk.op_hash, kinds_hash, kinds_hash) {
                if let Some(compiled) = load_native_block(&blob) {
                    return Some(compiled);
                }
            }
            let blob = compile_block_native(chunk, slot_kinds)?;
            write_blob(dir, "blk", chunk.op_hash, kinds_hash, &blob);
            load_native_block(&blob)
        }

        /// Trace-tier equivalent. Keyed by `(op_hash, record_anchor_ip)`; the
        /// `meta_hash` (content hash of the recording) is the verification aux
        /// word, so a stale file recorded for a different path is rejected.
        #[allow(clippy::too_many_arguments)]
        pub(crate) fn try_load_or_build_trace(
            dir: &Path,
            op_hash: u64,
            record_anchor_ip: usize,
            meta_hash: u64,
            ops: &[Op],
            recorded_ips: &[usize],
            fallthrough_ip: usize,
            is_side_trace: bool,
            slot_types: &[(u16, JitTy)],
            constants: &[FuseValue],
        ) -> Option<CompiledTrace> {
            let sub = record_anchor_ip as u64;
            if let Some(blob) = read_blob(dir, "trc", op_hash, sub, meta_hash) {
                if let Some(compiled) = load_native_trace(&blob) {
                    return Some(compiled);
                }
            }
            let blob = compile_trace_native(
                ops,
                recorded_ips,
                fallthrough_ip,
                is_side_trace,
                slot_types,
                constants,
                meta_hash,
            )?;
            write_blob(dir, "trc", op_hash, sub, &blob);
            load_native_trace(&blob)
        }

        /// Compile a chunk's block JIT to relocatable native code, or `None` if
        /// the chunk is ineligible or its code contains an unsupported
        /// relocation.
        pub(crate) fn compile_block_native(
            chunk: &Chunk,
            slot_kinds: &[crate::SlotKind],
        ) -> Option<NativeBlob> {
            let (
                BuiltFn {
                    module,
                    mut ctx,
                    fid: _,
                    helper_ids,
                    ext_helpers,
                },
                ret_is_float,
            ) = build_block_function(chunk, slot_kinds)?;
            let (code, relocs) = {
                let isa = module.isa();
                compile_and_extract(&mut ctx, isa, |index| {
                    map_helper_funcid_ext(&helper_ids, &ext_helpers, index)
                })?
            };
            Some(NativeBlob {
                kind: KIND_BLOCK,
                code,
                relocs,
                entry: 0,
                ret_is_float,
                need_slots: true,
                aux: slot_kinds_hash(slot_kinds),
            })
        }

        /// Compile a recorded trace to relocatable native code, or `None` if it
        /// contains an unsupported relocation.
        #[allow(clippy::too_many_arguments)]
        pub(crate) fn compile_trace_native(
            ops: &[Op],
            recorded_ips: &[usize],
            fallthrough_ip: usize,
            is_side_trace: bool,
            slot_types: &[(u16, JitTy)],
            constants: &[FuseValue],
            meta_hash: u64,
        ) -> Option<NativeBlob> {
            let BuiltFn {
                module,
                mut ctx,
                fid: _,
                helper_ids,
                ext_helpers: _,
            } = build_trace_function(
                ops,
                recorded_ips,
                fallthrough_ip,
                is_side_trace,
                slot_types,
                constants,
            )?;
            let (code, relocs) = {
                let isa = module.isa();
                compile_and_extract(&mut ctx, isa, |index| map_helper_funcid(&helper_ids, index))?
            };
            Some(NativeBlob {
                kind: KIND_TRACE,
                code,
                relocs,
                entry: 0,
                ret_is_float: false,
                need_slots: true,
                aux: meta_hash,
            })
        }
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

    /// Keeps the executable memory backing a [`CompiledBlock`] alive — either a
    /// `JITModule` (in-memory JIT path) or an mmap'd region of relocated native
    /// code loaded from the on-disk cache.
    pub(crate) enum BlockBacking {
        #[allow(dead_code)]
        Jit(JITModule),
        #[cfg(feature = "jit-disk-cache")]
        #[allow(dead_code)]
        Native(disk_cache::LoadedNative),
    }

    pub(crate) struct CompiledBlock {
        #[allow(dead_code)]
        backing: BlockBacking,
        run: BlockRun,
        /// Whether the chunk's result is a float, returned as its raw `f64` bit
        /// pattern in the `i64` register (the block signature is always
        /// `fn(*mut i64) -> i64`; float results are `bitcast`, not truncated, so a
        /// typed caller can recover the exact value via [`f64::from_bits`]).
        ret_is_float: bool,
    }

    impl CompiledBlock {
        /// Invoke and return the result as i64 (truncating float results, so all
        /// existing integer-contract callers — incl. the awk/VM path — observe the
        /// exact same value they did before float returns were bit-encoded).
        #[allow(dead_code)]
        pub(crate) fn invoke(&self, slots: &mut [i64]) -> i64 {
            let (bits, is_float) = self.invoke_typed(slots);
            if is_float {
                f64::from_bits(bits as u64) as i64
            } else {
                bits
            }
        }

        /// Invoke and return `(raw_bits, ret_is_float)`. For a float result
        /// `raw_bits` is the `f64` bit pattern (recover via `f64::from_bits`); for an
        /// integer result it is the plain `i64`.
        pub(crate) fn invoke_typed(&self, slots: &mut [i64]) -> (i64, bool) {
            let ptr = if slots.is_empty() {
                std::ptr::null_mut()
            } else {
                slots.as_mut_ptr()
            };
            let raw = match &self.run {
                BlockRun::SlotsI(f) => unsafe { f(ptr) },
                BlockRun::NoSlotsI(f) => unsafe { f() },
                BlockRun::SlotsF(f) => (unsafe { f(ptr) }).to_bits() as i64,
                BlockRun::NoSlotsF(f) => (unsafe { f() }).to_bits() as i64,
            };
            (raw, self.ret_is_float)
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
        if let Op::Extended(id, _) = op {
            return super::global_extension_for(*id).is_some();
        }
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
                | Op::PowFloat
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
                | Op::PreDecSlot(_)
                | Op::PostIncSlot(_)
                | Op::PostDecSlot(_)
                | Op::AddAssignSlotVoid(_, _)
                | Op::Jump(_)
                | Op::JumpIfTrue(_)
                | Op::JumpIfFalse(_)
                | Op::SlotLtIntJumpIfFalse(_, _, _)
                | Op::SlotIncLtIntJumpBack(_, _, _)
                | Op::AccumSumLoop(_, _, _)
                | Op::PushFrame
                | Op::PopFrame
                | Op::AwkInt
                | Op::AwkSin
                | Op::AwkCos
                | Op::AwkExp
                | Op::AwkAtan2
                | Op::AwkAnd(_)
                | Op::AwkOr(_)
                | Op::AwkXor(_)
                | Op::AwkDivJit
                | Op::AwkModJit
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

    /// Whether the block JIT cache has a compiled (post-warmup) entry for
    /// this chunk. Used by the VM to avoid paying slot-buffer refresh cost
    /// when block JIT is still warming up — until then `try_run_block`
    /// returns `None` and the refresh was wasted.
    #[inline]
    pub(crate) fn block_jit_is_compiled(chunk: &Chunk) -> bool {
        BLOCK_CACHE_TLS.with(|cache_cell| {
            cache_cell
                .borrow()
                .iter()
                .any(|(&(op_hash, _kinds), e)| op_hash == chunk.op_hash && e.compiled.is_some())
        })
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
                | Op::PreDecSlot(s)
                | Op::PostIncSlot(s)
                | Op::PostDecSlot(s)
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

    /// The product of building a tier's Cranelift function *before* emission:
    /// the owning `JITModule`, the populated `Context`, the function id, and the
    /// host-helper FuncIds (order `[pow_i64, pow_f64, fmod_f64, lognot]`) used to
    /// map relocations back to stable host-helper ids when caching native code.
    /// Consumed either by the in-memory JIT path (define + finalize) or the
    /// disk-cache native path (raw `ctx.compile` + relocation extraction).
    pub(crate) struct BuiltFn {
        module: JITModule,
        ctx: cranelift_codegen::Context,
        fid: cranelift_module::FuncId,
        #[allow(dead_code)]
        helper_ids: [Option<cranelift_module::FuncId>; 8],
        /// `(FuncId, stable host-helper id)` for each extension host helper the
        /// chunk calls — threaded into the cache relocation mapper.
        #[allow(dead_code)]
        ext_helpers: Vec<(cranelift_module::FuncId, u32)>,
    }

    /// Pass the current operand `stack` as block parameters when branching to
    /// `target`, so operand-stack values that are live across a basic-block boundary
    /// survive the control-flow merge. (The block JIT otherwise keeps the operand
    /// stack local to each Cranelift block, which silently drops values produced in
    /// one block and consumed after a merge — e.g. a ternary `?:` / `if`-expression
    /// / `&&` / `||` result.)
    ///
    /// The *first* predecessor to branch to a block fixes its parameter list (one
    /// param per live stack entry, typed by [`JitTy`]); every later predecessor must
    /// agree on arity and types, otherwise the chunk is rejected (`None`) and falls
    /// back to the interpreter. Branching back to the entry block is unsupported
    /// (its params are the function params), and also yields `None`. Returns the
    /// block arguments to hand to the branch instruction.
    fn branch_block_args(
        bcx: &mut FunctionBuilder,
        entry: cranelift_codegen::ir::Block,
        target: cranelift_codegen::ir::Block,
        stack: &[(Value, JitTy)],
        param_tys: &mut HashMap<cranelift_codegen::ir::Block, Vec<JitTy>>,
    ) -> Option<Vec<BlockArg>> {
        if target == entry && !stack.is_empty() {
            return None;
        }
        let tys: Vec<JitTy> = stack.iter().map(|(_, t)| *t).collect();
        match param_tys.get(&target) {
            Some(existing) => {
                if *existing != tys {
                    return None;
                }
            }
            None => {
                for ty in &tys {
                    let clty = match ty {
                        JitTy::Int => types::I64,
                        JitTy::Float => types::F64,
                    };
                    bcx.append_block_param(target, clty);
                }
                param_tys.insert(target, tys);
            }
        }
        Some(stack.iter().map(|(v, _)| BlockArg::Value(*v)).collect())
    }

    /// Build the block-JIT Cranelift function for `chunk`. Returns the built
    /// function plus `ret_is_float`: whether the chunk's result is a float, which
    /// the function returns as the raw `f64` bit pattern in its `i64` return
    /// register (the signature is always `fn(*mut i64) -> i64`). Returns `None` if
    /// the chunk is block-ineligible, or if two distinct return points disagree on
    /// the result's int/float-ness (an inconsistent ABI the caller can't decode) —
    /// in which case the caller falls back to the interpreter.
    pub(crate) fn build_block_function(
        chunk: &Chunk,
        slot_kinds: &[super::SlotKind],
    ) -> Option<(BuiltFn, bool)> {
        let ops = &chunk.ops;
        if !is_block_eligible(chunk) {
            return None;
        }

        // A slot is Float-kinded when the runtime value it holds at block entry
        // is a float (awk numeric vars). Float slots are stored as i64 bit
        // patterns in the slot array; `GetSlot`/`SetSlot` bit-cast through f64.
        // Integer-only fused slot superinstructions bail (return None) on a
        // Float slot so the chunk falls back to the interpreter.
        let slot_is_float =
            |slot: u16| matches!(slot_kinds.get(slot as usize), Some(super::SlotKind::Float));

        let leaders = find_leaders(ops);
        let leader_vec: Vec<usize> = leaders.iter().copied().collect();
        let mut module = new_jit_module()?;

        // Collects extension host helpers imported while lowering `Op::Extended`
        // arms, as `helper_id -> (FuncId, FuncRef)`. Drained into `BuiltFn::
        // ext_helpers` so `compile_block_native` can relocate them in cached
        // native code. Stays empty for chunks with no extension calls (all
        // AWK-op and pure-numeric chunks).
        let mut ext_helper_map: std::collections::HashMap<
            u32,
            (cranelift_module::FuncId, cranelift_codegen::ir::FuncRef),
        > = std::collections::HashMap::new();

        // Declare external helpers
        let needs_pow = ops.iter().any(|o| matches!(o, Op::Pow | Op::PowFloat));
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
        let needs_fmod = ops.iter().any(|o| matches!(o, Op::Mod | Op::AwkModJit));
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

        let math_ids = MathIds::declare(&mut module, ops);

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
        // Set inside the builder block at the chunk's return point(s); read after
        // the block closes to tag the built function as float- or int-returning.
        let ret_is_float_final: bool;
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
            let math = math_ids.resolve(&mut module, bcx.func);

            // Process each basic block

            // Operand-stack values that are live across a basic-block boundary are
            // carried as Cranelift block parameters (see `branch_block_args`). This
            // records the JitTy of each block's params so successor blocks can seed
            // their local operand stack and later predecessors can verify a match.
            let mut block_param_tys: HashMap<cranelift_codegen::ir::Block, Vec<JitTy>> =
                HashMap::new();

            // Whether the chunk's result is a float, decided at the (one or more)
            // return points. All return points must agree; a disagreement makes the
            // i64-encoded result undecodable, so we bail to the interpreter.
            let mut ret_is_float: Option<bool> = None;

            for (block_idx, &leader_ip) in leader_vec.iter().enumerate() {
                let block_end = if block_idx + 1 < leader_vec.len() {
                    leader_vec[block_idx + 1]
                } else {
                    ops.len()
                };

                let cur_block = block_map[&leader_ip];
                // Switch to this block (unless it's the entry which we already started).
                // Fallthrough jumps are emitted at the END of the predecessor block so
                // the residual operand stack can be passed as block arguments.
                if leader_ip > 0 {
                    bcx.switch_to_block(cur_block);
                }

                let mut stack: Vec<(cranelift_codegen::ir::Value, JitTy)> = Vec::new();
                // Seed the local operand stack from this block's parameters (values
                // carried in from predecessors across the merge).
                if leader_ip > 0 {
                    if let Some(tys) = block_param_tys.get(&cur_block) {
                        let params: Vec<Value> = bcx.block_params(cur_block).to_vec();
                        if params.len() != tys.len() {
                            return None;
                        }
                        for (i, ty) in tys.iter().enumerate() {
                            stack.push((params[i], *ty));
                        }
                    }
                }
                let mut block_terminated = false;

                for ip in leader_ip..block_end {
                    let op = &ops[ip];
                    match op {
                        Op::PushFrame | Op::PopFrame | Op::Nop => {}

                        Op::Jump(target) => {
                            let target_block = *block_map.get(target)?;
                            let args = branch_block_args(
                                &mut bcx,
                                entry,
                                target_block,
                                &stack,
                                &mut block_param_tys,
                            )?;
                            bcx.ins().jump(target_block, &args);
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
                            let targs = branch_block_args(
                                &mut bcx,
                                entry,
                                target_block,
                                &stack,
                                &mut block_param_tys,
                            )?;
                            let fargs = branch_block_args(
                                &mut bcx,
                                entry,
                                fall,
                                &stack,
                                &mut block_param_tys,
                            )?;
                            bcx.ins().brif(pred, target_block, &targs, fall, &fargs);
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
                            let targs = branch_block_args(
                                &mut bcx,
                                entry,
                                target_block,
                                &stack,
                                &mut block_param_tys,
                            )?;
                            let fargs = branch_block_args(
                                &mut bcx,
                                entry,
                                fall,
                                &stack,
                                &mut block_param_tys,
                            )?;
                            bcx.ins().brif(pred, fall, &fargs, target_block, &targs);
                            block_terminated = true;
                        }

                        // ── Slot data ops: use Variables (register-allocated) ──
                        // Float-kinded slots hold an i64 bit pattern; bit-cast
                        // through f64 on use/def so downstream arithmetic is
                        // float-aware (mirrors the tracing-JIT slot handling).
                        Op::GetSlot(slot) => {
                            let var = *slot_vars.get(slot)?;
                            let raw = bcx.use_var(var);
                            if slot_is_float(*slot) {
                                let f = bcx.ins().bitcast(
                                    types::F64,
                                    cranelift_codegen::ir::MemFlags::new(),
                                    raw,
                                );
                                stack.push((f, JitTy::Float));
                            } else {
                                stack.push((raw, JitTy::Int));
                            }
                        }
                        Op::SetSlot(slot) => {
                            let var = *slot_vars.get(slot)?;
                            let (v, ty) = stack.pop()?;
                            let v_i = if slot_is_float(*slot) {
                                // Store the f64 bit pattern. Coerce an Int
                                // operand to f64 first (integer-valued result of
                                // e.g. a comparison or AwkInt), then bit-cast.
                                let f = match ty {
                                    JitTy::Float => v,
                                    JitTy::Int => i64_to_f64(&mut bcx, v),
                                };
                                bcx.ins().bitcast(
                                    types::I64,
                                    cranelift_codegen::ir::MemFlags::new(),
                                    f,
                                )
                            } else {
                                scalar_store_i64(&mut bcx, v, ty)
                            };
                            bcx.def_var(var, v_i);
                        }
                        Op::PreIncSlot(slot) => {
                            if slot_is_float(*slot) {
                                return None;
                            }
                            let var = *slot_vars.get(slot)?;
                            let old = bcx.use_var(var);
                            let one = bcx.ins().iconst(types::I64, 1);
                            let new = bcx.ins().iadd(old, one);
                            bcx.def_var(var, new);
                            stack.push((new, JitTy::Int));
                        }
                        Op::PreIncSlotVoid(slot) => {
                            if slot_is_float(*slot) {
                                return None;
                            }
                            let var = *slot_vars.get(slot)?;
                            let old = bcx.use_var(var);
                            let one = bcx.ins().iconst(types::I64, 1);
                            let new = bcx.ins().iadd(old, one);
                            bcx.def_var(var, new);
                        }
                        Op::PreDecSlot(slot) => {
                            if slot_is_float(*slot) {
                                return None;
                            }
                            let var = *slot_vars.get(slot)?;
                            let old = bcx.use_var(var);
                            let one = bcx.ins().iconst(types::I64, 1);
                            let new = bcx.ins().isub(old, one);
                            bcx.def_var(var, new);
                            stack.push((new, JitTy::Int));
                        }
                        Op::PostIncSlot(slot) => {
                            if slot_is_float(*slot) {
                                return None;
                            }
                            let var = *slot_vars.get(slot)?;
                            let old = bcx.use_var(var);
                            let one = bcx.ins().iconst(types::I64, 1);
                            let new = bcx.ins().iadd(old, one);
                            bcx.def_var(var, new);
                            stack.push((old, JitTy::Int));
                        }
                        Op::PostDecSlot(slot) => {
                            if slot_is_float(*slot) {
                                return None;
                            }
                            let var = *slot_vars.get(slot)?;
                            let old = bcx.use_var(var);
                            let one = bcx.ins().iconst(types::I64, 1);
                            let new = bcx.ins().isub(old, one);
                            bcx.def_var(var, new);
                            stack.push((old, JitTy::Int));
                        }
                        Op::AddAssignSlotVoid(a_slot, b_slot) => {
                            if slot_is_float(*a_slot) || slot_is_float(*b_slot) {
                                return None;
                            }
                            let a_var = *slot_vars.get(a_slot)?;
                            let b_var = *slot_vars.get(b_slot)?;
                            let va = bcx.use_var(a_var);
                            let vb = bcx.use_var(b_var);
                            let sum = bcx.ins().iadd(va, vb);
                            bcx.def_var(a_var, sum);
                        }

                        Op::SlotLtIntJumpIfFalse(slot, limit, target) => {
                            if slot_is_float(*slot) {
                                return None;
                            }
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
                            let targs = branch_block_args(
                                &mut bcx,
                                entry,
                                target_block,
                                &stack,
                                &mut block_param_tys,
                            )?;
                            let fargs = branch_block_args(
                                &mut bcx,
                                entry,
                                fall,
                                &stack,
                                &mut block_param_tys,
                            )?;
                            bcx.ins().brif(is_lt, fall, &fargs, target_block, &targs);
                            block_terminated = true;
                        }

                        Op::SlotIncLtIntJumpBack(slot, limit, target) => {
                            if slot_is_float(*slot) {
                                return None;
                            }
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
                            let targs = branch_block_args(
                                &mut bcx,
                                entry,
                                target_block,
                                &stack,
                                &mut block_param_tys,
                            )?;
                            let fargs = branch_block_args(
                                &mut bcx,
                                entry,
                                fall,
                                &stack,
                                &mut block_param_tys,
                            )?;
                            bcx.ins().brif(is_lt, target_block, &targs, fall, &fargs);
                            block_terminated = true;
                        }

                        Op::AccumSumLoop(sum_slot, i_slot, limit) => {
                            if slot_is_float(*sum_slot) || slot_is_float(*i_slot) {
                                return None;
                            }
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

                        // Extended — delegate to a registered JIT extension.
                        Op::Extended(id, arg) => {
                            let ext = super::global_extension_for(*id)?;
                            let mut cx = ExtJitCtx {
                                bcx: &mut bcx,
                                stack: &mut stack,
                                module: &mut module,
                                helpers: &mut ext_helper_map,
                            };
                            if !ext.emit_extended(*id, *arg, &mut cx) {
                                return None;
                            }
                        }

                        // AWK div/mod with a guarded zero-divisor trap. Float
                        // `fdiv`/`fmod` don't hardware-trap (they yield inf/nan),
                        // but awk requires a fatal on a zero divisor. Emit an
                        // early-exit: compare the divisor to 0.0; on equality
                        // branch to a trap block that records the trap code via
                        // the `fusevm_jit_awk_div_trap` libcall and returns a
                        // sentinel (the VM reads the code after the block and
                        // raises the awk error, discarding this result). The
                        // operands match `Op::Div`/`Op::AwkDiv`: pop divisor (top)
                        // then dividend.
                        Op::AwkDivJit | Op::AwkModJit => {
                            let trap_ref = math.awk_div_trap?;
                            let divisor = pop_as_f64(&mut bcx, &mut stack)?;
                            let dividend = pop_as_f64(&mut bcx, &mut stack)?;
                            let zero = bcx.ins().f64const(0.0);
                            let is_zero = bcx.ins().fcmp(FloatCC::Equal, divisor, zero);
                            let trap_block = bcx.create_block();
                            let cont_block = bcx.create_block();
                            bcx.ins().brif(is_zero, trap_block, &[], cont_block, &[]);

                            // Trap path: record code (1 = div, 2 = mod), return 0.
                            bcx.switch_to_block(trap_block);
                            let code = if matches!(op, Op::AwkDivJit) { 1 } else { 2 };
                            let code_v = bcx.ins().iconst(types::I64, code);
                            bcx.ins().call(trap_ref, &[code_v]);
                            let sentinel = bcx.ins().iconst(types::I64, 0);
                            bcx.ins().return_(&[sentinel]);

                            // Continue path: compute the quotient/remainder.
                            bcx.switch_to_block(cont_block);
                            let res = if matches!(op, Op::AwkDivJit) {
                                bcx.ins().fdiv(dividend, divisor)
                            } else {
                                let fref = fmod_f64_ref?;
                                let call = bcx.ins().call(fref, &[dividend, divisor]);
                                bcx.inst_results(call)[0]
                            };
                            stack.push((res, JitTy::Float));
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
                                math,
                                &chunk.constants,
                            )?;
                        }
                    }
                }

                // End of this basic block — if not terminated, handle fallthrough or return
                if !block_terminated {
                    if block_end == ops.len() {
                        // Final block: spill promoted slot variables to memory, then return
                        let (ret_val, is_f) = if let Some((v, ty)) = stack.pop() {
                            match ty {
                                // A float result is returned as its raw f64 bit
                                // pattern (bit-cast, NOT truncated), so a typed
                                // caller recovers the exact value. The integer
                                // `invoke` re-truncates, preserving old behavior.
                                JitTy::Float => {
                                    let bits = bcx.ins().bitcast(
                                        types::I64,
                                        cranelift_codegen::ir::MemFlags::new(),
                                        v,
                                    );
                                    (bits, true)
                                }
                                JitTy::Int => (v, false),
                            }
                        } else {
                            (bcx.ins().iconst(types::I64, 0), false)
                        };
                        // All return points must agree on int vs float.
                        match ret_is_float {
                            Some(prev) if prev != is_f => return None,
                            _ => ret_is_float = Some(is_f),
                        }
                        // Write all slot Variables back to the slot pointer
                        // (caller observes the final state of the slot array)
                        for (&slot, &var) in &slot_vars {
                            let val = bcx.use_var(var);
                            bcx.ins()
                                .store(MemFlags::trusted(), val, slot_base, (slot as i32) * 8);
                        }
                        bcx.ins().return_(&[ret_val]);
                    } else {
                        // Non-final unterminated block: fall through to the next leader,
                        // carrying any live operand-stack values as block arguments.
                        let next_block = *block_map.get(&block_end)?;
                        let args = branch_block_args(
                            &mut bcx,
                            entry,
                            next_block,
                            &stack,
                            &mut block_param_tys,
                        )?;
                        bcx.ins().jump(next_block, &args);
                    }
                }
            }

            bcx.seal_all_blocks();
            bcx.finalize();
            ret_is_float_final = ret_is_float.unwrap_or(false);
        }

        Some((
            BuiltFn {
                module,
                ctx,
                fid,
                helper_ids: [
                    pow_i64_id,
                    pow_f64_id,
                    fmod_f64_id,
                    lognot_id,
                    math_ids.sin,
                    math_ids.cos,
                    math_ids.exp,
                    math_ids.atan2,
                ],
                ext_helpers: {
                    // Extension host helpers called by this chunk, as `(FuncId,
                    // stable id)` pairs, so `compile_block_native` can relocate
                    // them in cached native code. Empty for chunks that call no
                    // host helpers (e.g. all AWK-op or pure-numeric chunks).
                    #[cfg_attr(not(feature = "jit-disk-cache"), allow(unused_mut))]
                    let mut v: Vec<(cranelift_module::FuncId, u32)> = ext_helper_map
                        .into_iter()
                        .map(|(id, (fid, _))| (fid, id))
                        .collect();
                    // The `AwkDivJit`/`AwkModJit` zero-divisor trap libcall is a
                    // built-in helper with a stable disk-cache id, so div/mod
                    // chunks persist as native code instead of being limited to
                    // the in-memory JIT.
                    #[cfg(feature = "jit-disk-cache")]
                    if let Some(fid) = math_ids.awk_div_trap {
                        v.push((fid, disk_cache::H_AWK_DIV_TRAP));
                    }
                    v
                },
            },
            ret_is_float_final,
        ))
    }

    /// In-memory block JIT: build the function and finalize it through the
    /// `JITModule`. Block functions are always `fn(*mut i64) -> i64`.
    pub(crate) fn compile_block(
        chunk: &Chunk,
        slot_kinds: &[super::SlotKind],
    ) -> Option<CompiledBlock> {
        let (
            BuiltFn {
                mut module,
                mut ctx,
                fid,
                helper_ids: _,
                ext_helpers: _,
            },
            ret_is_float,
        ) = build_block_function(chunk, slot_kinds)?;
        module.define_function(fid, &mut ctx).ok()?;
        module.clear_context(&mut ctx);
        module.finalize_definitions().ok()?;
        let ptr = module.get_finalized_function(fid);
        // Always SlotsI: signature is fn(*mut i64) -> i64. A float result rides
        // back in the i64 register as its f64 bit pattern; `ret_is_float` tells a
        // typed caller to decode it.
        let run = BlockRun::SlotsI(unsafe { std::mem::transmute::<*const u8, BlockFnSlotsI>(ptr) });
        Some(CompiledBlock {
            backing: BlockBacking::Jit(module),
            run,
            ret_is_float,
        })
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
        static BLOCK_CACHE_TLS: RefCell<HashMap<(u64, u64), BlockCacheEntry>> =
            RefCell::new(HashMap::new());
    }

    /// Try to JIT-compile and run a chunk via the block JIT.
    /// Returns `Some(result_i64)` on success, `None` if ineligible OR not yet hot.
    ///
    /// The warmup threshold (whole-chunk invocations before compiling) is read
    /// from the per-thread [`TraceJitConfig::block_threshold`] (default 10), so
    /// callers can tune it for their workload via `JitCompiler::set_config`.
    /// Below the threshold this returns `None` and the caller falls back to the
    /// interpreter — avoiding compile cost for one-shot chunks.
    pub(crate) fn try_run_block(chunk: &Chunk, slots: &mut [i64]) -> Option<i64> {
        try_run_block_inner(chunk, slots, &[], cfg_block_threshold()).map(decode_block_i64)
    }

    /// Slot-kind-aware variant: `slot_kinds[i]` gives the runtime kind of slot
    /// `i` (Float for awk numeric vars). Float slots are stored as i64 bit
    /// patterns and bit-cast through f64 in the compiled code.
    pub(crate) fn try_run_block_kinded(
        chunk: &Chunk,
        slots: &mut [i64],
        slot_kinds: &[super::SlotKind],
    ) -> Option<i64> {
        try_run_block_inner(chunk, slots, slot_kinds, cfg_block_threshold()).map(decode_block_i64)
    }

    /// Typed slot-kind-aware variant: returns the chunk result as a
    /// [`BlockNum`], preserving float results exactly (rather than truncating
    /// them to i64 like [`try_run_block_kinded`]). Used by frontends whose chunk
    /// result may be a float value.
    pub(crate) fn try_run_block_typed_kinded(
        chunk: &Chunk,
        slots: &mut [i64],
        slot_kinds: &[super::SlotKind],
    ) -> Option<super::BlockNum> {
        try_run_block_inner(chunk, slots, slot_kinds, cfg_block_threshold()).map(decode_block_num)
    }

    /// Like `try_run_block` but compiles immediately (no warmup). For tests
    /// and synthetic benchmarks where you want to skip the tiered policy.
    pub(crate) fn try_run_block_eager(chunk: &Chunk, slots: &mut [i64]) -> Option<i64> {
        try_run_block_inner(chunk, slots, &[], 0).map(decode_block_i64)
    }

    /// Slot-kind-aware eager variant (see [`try_run_block_kinded`]).
    pub(crate) fn try_run_block_eager_kinded(
        chunk: &Chunk,
        slots: &mut [i64],
        slot_kinds: &[super::SlotKind],
    ) -> Option<i64> {
        try_run_block_inner(chunk, slots, slot_kinds, 0).map(decode_block_i64)
    }

    /// Typed eager slot-kind-aware variant (see [`try_run_block_typed_kinded`]).
    pub(crate) fn try_run_block_eager_typed_kinded(
        chunk: &Chunk,
        slots: &mut [i64],
        slot_kinds: &[super::SlotKind],
    ) -> Option<super::BlockNum> {
        try_run_block_inner(chunk, slots, slot_kinds, 0).map(decode_block_num)
    }

    /// Decode a `(raw_bits, ret_is_float)` block result to an i64, truncating a
    /// float result (preserving the historical integer contract).
    fn decode_block_i64((raw, is_float): (i64, bool)) -> i64 {
        if is_float {
            f64::from_bits(raw as u64) as i64
        } else {
            raw
        }
    }

    /// Decode a `(raw_bits, ret_is_float)` block result to a typed [`BlockNum`].
    fn decode_block_num((raw, is_float): (i64, bool)) -> super::BlockNum {
        if is_float {
            super::BlockNum::Float(f64::from_bits(raw as u64))
        } else {
            super::BlockNum::Int(raw)
        }
    }

    fn try_run_block_inner(
        chunk: &Chunk,
        slots: &mut [i64],
        slot_kinds: &[super::SlotKind],
        threshold: u32,
    ) -> Option<(i64, bool)> {
        let key = (chunk.op_hash, slot_kinds_hash(slot_kinds));

        BLOCK_CACHE_TLS.with(|cache_cell| {
            let mut cache = cache_cell.borrow_mut();
            let entry = cache.entry(key).or_insert(BlockCacheEntry {
                hot_count: 0,
                compiled: None,
            });

            if let Some(ref compiled) = entry.compiled {
                return Some(compiled.invoke_typed(slots));
            }

            entry.hot_count = entry.hot_count.saturating_add(1);
            if entry.hot_count <= threshold {
                return None; // not hot yet — caller falls back to interpreter
            }

            // Disk-cache path (on by default): reuse or build relocatable native
            // code persisted across process restarts, skipping Cranelift codegen.
            #[cfg(feature = "jit-disk-cache")]
            {
                if let Some(dir) = disk_cache::cache_dir() {
                    if let Some(compiled) =
                        disk_cache::try_load_or_build_block(chunk, &dir, slot_kinds)
                    {
                        let result = compiled.invoke_typed(slots);
                        entry.compiled = Some(Box::new(compiled));
                        return Some(result);
                    }
                    // Native caching rejected this chunk: fall through to JIT.
                }
            }

            // Compile on threshold cross
            let compiled = compile_block(chunk, slot_kinds)?;
            let result = compiled.invoke_typed(slots);
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

    /// Keeps the executable memory backing a [`CompiledTrace`] alive — either a
    /// `JITModule` (in-memory JIT path) or an mmap'd region of relocated native
    /// code loaded from the on-disk cache.
    pub(crate) enum TraceBacking {
        #[allow(dead_code)]
        Jit(JITModule),
        #[cfg(feature = "jit-disk-cache")]
        #[allow(dead_code)]
        Native(disk_cache::LoadedNative),
    }

    /// A compiled trace.
    pub(crate) struct CompiledTrace {
        #[allow(dead_code)]
        backing: TraceBacking,
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
        /// override via `JitCompiler::set_config`. Initialized from the
        /// compiled defaults, with optional process-wide env overrides
        /// (`FUSEVM_JIT_BLOCK_THRESHOLD` / `FUSEVM_JIT_TRACE_THRESHOLD`) so
        /// re-run-heavy workloads can dial in warmup without recompiling.
        static TRACE_CONFIG: RefCell<super::TraceJitConfig> =
            RefCell::new(config_from_env());
    }

    /// Build the starting `TraceJitConfig` for a thread: the compiled defaults
    /// with any `FUSEVM_JIT_*_THRESHOLD` env overrides applied. Unset or
    /// unparsable vars leave the corresponding default untouched.
    fn config_from_env() -> super::TraceJitConfig {
        apply_threshold_overrides(
            super::TraceJitConfig::defaults(),
            std::env::var("FUSEVM_JIT_BLOCK_THRESHOLD").ok(),
            std::env::var("FUSEVM_JIT_TRACE_THRESHOLD").ok(),
        )
    }

    /// Pure override logic split out from `config_from_env` for testability:
    /// applies parsed block/trace threshold strings onto `cfg`, ignoring
    /// missing or unparsable values.
    fn apply_threshold_overrides(
        mut cfg: super::TraceJitConfig,
        block: Option<String>,
        trace: Option<String>,
    ) -> super::TraceJitConfig {
        if let Some(n) = block.and_then(|v| v.trim().parse::<u32>().ok()) {
            cfg.block_threshold = n;
        }
        if let Some(n) = trace.and_then(|v| v.trim().parse::<u32>().ok()) {
            cfg.trace_threshold = n;
        }
        cfg
    }

    #[cfg(test)]
    mod env_override_tests {
        use super::apply_threshold_overrides;
        use crate::TraceJitConfig;

        #[test]
        fn overrides_parse_and_apply() {
            let cfg = apply_threshold_overrides(
                TraceJitConfig::defaults(),
                Some("0".to_string()),
                Some(" 8 ".to_string()),
            );
            assert_eq!(cfg.block_threshold, 0);
            assert_eq!(cfg.trace_threshold, 8);
        }

        #[test]
        fn missing_or_garbage_leaves_defaults() {
            let d = TraceJitConfig::defaults();
            let cfg = apply_threshold_overrides(d, None, Some("not-a-number".to_string()));
            assert_eq!(cfg.block_threshold, d.block_threshold);
            assert_eq!(cfg.trace_threshold, d.trace_threshold);
        }
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
    fn cfg_block_threshold() -> u32 {
        TRACE_CONFIG.with(|c| c.borrow().block_threshold)
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

        // Disk-cache path (on by default): reuse or build relocatable native
        // trace code persisted across process restarts. The trace is keyed by
        // (op_hash, record_anchor_ip); a content hash of the recording guards
        // against a stale file recorded for a different path.
        #[cfg_attr(not(feature = "jit-disk-cache"), allow(unused_mut))]
        let mut compiled: Option<CompiledTrace> = None;
        #[cfg(feature = "jit-disk-cache")]
        {
            if let Some(dir) = disk_cache::cache_dir() {
                let meta_hash = trace_meta_hash(
                    ops,
                    recorded_ips,
                    fallthrough_ip,
                    is_side_trace,
                    &slot_types,
                    constants,
                );
                compiled = disk_cache::try_load_or_build_trace(
                    &dir,
                    chunk.op_hash,
                    record_anchor_ip,
                    meta_hash,
                    ops,
                    recorded_ips,
                    fallthrough_ip,
                    is_side_trace,
                    &slot_types,
                    constants,
                );
            }
        }

        let compiled = match compiled {
            Some(c) => c,
            None => match compile_trace_kinded(
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
            },
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
            // Post/pre-dec slot ops are emitted by the bytecode optimizer for
            // counter loops and are handled by the block JIT (whole-chunk);
            // the trace tier has no codegen for them, so reject to fall back.
            Op::PreDecSlot(_) | Op::PostIncSlot(_) | Op::PostDecSlot(_) => false,
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

    /// Content hash of everything that determines a trace's native code, used
    /// as the disk-cache verification word so a cache file recorded for a
    /// different path/types is rejected even when the `(op_hash, anchor)` key
    /// collides. Stable across processes of the same build.
    #[cfg(feature = "jit-disk-cache")]
    fn trace_meta_hash(
        ops: &[Op],
        recorded_ips: &[usize],
        fallthrough_ip: usize,
        is_side_trace: bool,
        slot_types: &[(u16, JitTy)],
        constants: &[FuseValue],
    ) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        format!("{ops:?}").hash(&mut h);
        recorded_ips.hash(&mut h);
        fallthrough_ip.hash(&mut h);
        is_side_trace.hash(&mut h);
        for (slot, ty) in slot_types {
            slot.hash(&mut h);
            (matches!(ty, JitTy::Float) as u8).hash(&mut h);
        }
        format!("{constants:?}").hash(&mut h);
        h.finish()
    }

    fn build_trace_function(
        ops: &[Op],
        recorded_ips: &[usize],
        fallthrough_ip: usize,
        is_side_trace: bool,
        slot_types: &[(u16, JitTy)],
        constants: &[FuseValue],
    ) -> Option<BuiltFn> {
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
        let needs_pow = ops.iter().any(|o| matches!(o, Op::Pow | Op::PowFloat));
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

        let math_ids = MathIds::declare(&mut module, ops);

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
            let math = math_ids.resolve(&mut module, bcx.func);

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
                            math,
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

        Some(BuiltFn {
            module,
            ctx,
            fid,
            helper_ids: [
                pow_i64_id,
                pow_f64_id,
                fmod_f64_id,
                lognot_id,
                math_ids.sin,
                math_ids.cos,
                math_ids.exp,
                math_ids.atan2,
            ],
            ext_helpers: Vec::new(),
        })
    }

    fn compile_trace_inner(
        ops: &[Op],
        recorded_ips: &[usize],
        fallthrough_ip: usize,
        is_side_trace: bool,
        slot_types: &[(u16, JitTy)],
        constants: &[FuseValue],
    ) -> Option<CompiledTrace> {
        let BuiltFn {
            mut module,
            mut ctx,
            fid,
            helper_ids: _,
            ext_helpers: _,
        } = build_trace_function(
            ops,
            recorded_ips,
            fallthrough_ip,
            is_side_trace,
            slot_types,
            constants,
        )?;
        module.define_function(fid, &mut ctx).ok()?;
        module.clear_context(&mut ctx);
        module.finalize_definitions().ok()?;
        let ptr = module.get_finalized_function(fid);
        let run = unsafe { std::mem::transmute::<*const u8, TraceFn>(ptr) };
        Some(CompiledTrace {
            backing: TraceBacking::Jit(module),
            run,
        })
    }

    /// Test-only: clear the trace cache. Public for tests within this crate.
    #[cfg(test)]
    #[allow(dead_code)]
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
    /// Construct a fresh JIT compiler with no extensions registered.
    /// Frontends call `register_extension` to plug in custom-op codegen
    /// for any `Op::Extended(tag, _)` they emit.
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
                | Op::PowFloat
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
    /// Public method `try_run_linear` — see the implementing block's surrounding context for the call contract.
    pub fn try_run_linear(&self, chunk: &crate::Chunk, slots: &[i64]) -> Option<crate::Value> {
        cranelift_jit_impl::try_run_linear(chunk, slots)
    }

    /// Check if a chunk is eligible for linear JIT.
    #[cfg(feature = "jit")]
    /// Public method `is_linear_eligible` — see the implementing block's surrounding context for the call contract.
    pub fn is_linear_eligible(&self, chunk: &crate::Chunk) -> bool {
        cranelift_jit_impl::is_linear_eligible(chunk)
    }

    /// Stub when JIT feature is disabled.
    #[cfg(not(feature = "jit"))]
    /// No-op stub for `try_run_linear` when the `jit` cargo feature is disabled. The real implementation lives behind `#[cfg(feature = "jit")]`.
    pub fn try_run_linear(&self, _chunk: &crate::Chunk, _slots: &[i64]) -> Option<crate::Value> {
        None
    }

    /// Stub when JIT feature is disabled.
    #[cfg(not(feature = "jit"))]
    /// No-op stub for `is_linear_eligible` when the `jit` cargo feature is disabled. The real implementation lives behind `#[cfg(feature = "jit")]`.
    pub fn is_linear_eligible(&self, _chunk: &crate::Chunk) -> bool {
        false
    }

    /// Set the directory used to persist native linear-JIT code across
    /// processes (the on-disk JIT cache). Pass `None` to clear the override so
    /// resolution falls back to the `FUSEVM_JIT_CACHE_DIR` environment variable
    /// and then the default (`~/.cache/fusevm-jit`).
    ///
    /// Disk caching is **on by default** when the `jit-disk-cache` feature is
    /// enabled; set `FUSEVM_JIT_CACHE_DIR=off` to disable it. Without the
    /// feature this is a no-op.
    #[cfg(feature = "jit-disk-cache")]
    pub fn set_jit_cache_dir(&self, dir: Option<std::path::PathBuf>) {
        cranelift_jit_impl::disk_cache::set_cache_dir(dir);
    }

    /// The active on-disk JIT cache directory, or `None` if caching is
    /// disabled (`FUSEVM_JIT_CACHE_DIR=off`). Defaults to `~/.cache/fusevm-jit`.
    #[cfg(feature = "jit-disk-cache")]
    pub fn jit_cache_dir(&self) -> Option<std::path::PathBuf> {
        cranelift_jit_impl::disk_cache::cache_dir()
    }

    /// Total size in bytes of the on-disk JIT cache, or `None` if caching is
    /// disabled. Counts only `*.fjit` blobs (ignores in-flight temp files).
    #[cfg(feature = "jit-disk-cache")]
    pub fn jit_cache_size_bytes(&self) -> Option<u64> {
        cranelift_jit_impl::disk_cache::cache_dir()
            .map(|d| cranelift_jit_impl::disk_cache::cache_size_bytes(&d))
    }

    /// Set the cap on total on-disk cache size. `Some(0)` means unlimited
    /// (never evict); `None` restores the default resolution
    /// (`FUSEVM_JIT_CACHE_MAX_BYTES` env var, then 256 MiB). When the cache
    /// exceeds the cap, the oldest blobs are evicted (down to 80% of the cap)
    /// opportunistically as new entries are written.
    #[cfg(feature = "jit-disk-cache")]
    pub fn set_jit_cache_max_bytes(&self, limit: Option<u64>) {
        cranelift_jit_impl::disk_cache::set_max_bytes(limit);
    }

    /// Force an immediate eviction pass against the current cap. Returns the
    /// number of bytes freed (0 if caching is disabled, the cap is unlimited,
    /// or the cache is already under the cap).
    #[cfg(feature = "jit-disk-cache")]
    pub fn prune_jit_cache(&self) -> u64 {
        match cranelift_jit_impl::disk_cache::cache_dir() {
            Some(d) => {
                cranelift_jit_impl::disk_cache::prune(&d, cranelift_jit_impl::disk_cache::max_bytes())
            }
            None => 0,
        }
    }

    /// Delete every blob in the on-disk JIT cache. Returns the number of files
    /// removed (0 if caching is disabled). The cache repopulates lazily on the
    /// next run.
    #[cfg(feature = "jit-disk-cache")]
    pub fn clear_jit_cache(&self) -> usize {
        match cranelift_jit_impl::disk_cache::cache_dir() {
            Some(d) => cranelift_jit_impl::disk_cache::clear(&d),
            None => 0,
        }
    }

    /// No-op stub when the `jit-disk-cache` feature is disabled.
    #[cfg(not(feature = "jit-disk-cache"))]
    pub fn set_jit_cache_dir(&self, _dir: Option<std::path::PathBuf>) {}

    /// Always `None` when the `jit-disk-cache` feature is disabled.
    #[cfg(not(feature = "jit-disk-cache"))]
    pub fn jit_cache_dir(&self) -> Option<std::path::PathBuf> {
        None
    }

    /// Always `None` when the `jit-disk-cache` feature is disabled.
    #[cfg(not(feature = "jit-disk-cache"))]
    pub fn jit_cache_size_bytes(&self) -> Option<u64> {
        None
    }

    /// No-op stub when the `jit-disk-cache` feature is disabled.
    #[cfg(not(feature = "jit-disk-cache"))]
    pub fn set_jit_cache_max_bytes(&self, _limit: Option<u64>) {}

    /// Always `0` when the `jit-disk-cache` feature is disabled.
    #[cfg(not(feature = "jit-disk-cache"))]
    pub fn prune_jit_cache(&self) -> u64 {
        0
    }

    /// Always `0` when the `jit-disk-cache` feature is disabled.
    #[cfg(not(feature = "jit-disk-cache"))]
    pub fn clear_jit_cache(&self) -> usize {
        0
    }

    /// Try to compile and run a chunk via the block JIT (handles loops/branches).
    /// Slots are read and written in-place. Returns `Some(result)` on success.
    ///
    /// Tiered: returns `None` for the first N invocations of a given chunk so the
    /// caller falls back to the interpreter. After the chunk crosses the
    /// hot-threshold, compiles and caches the native code.
    #[cfg(feature = "jit")]
    /// Public method `try_run_block` — see the implementing block's surrounding context for the call contract.
    pub fn try_run_block(&self, chunk: &crate::Chunk, slots: &mut [i64]) -> Option<i64> {
        cranelift_jit_impl::try_run_block(chunk, slots)
    }

    /// Slot-kind-aware [`try_run_block`]: `slot_kinds[i]` is the runtime kind of
    /// slot `i`. Float slots (awk numeric vars) are stored as i64 bit patterns
    /// and bit-cast through f64 in the compiled native code. The slot kinds are
    /// part of the cache key, so Int- and Float-specialized blocks never alias.
    #[cfg(feature = "jit")]
    pub fn try_run_block_kinded(
        &self,
        chunk: &crate::Chunk,
        slots: &mut [i64],
        slot_kinds: &[SlotKind],
    ) -> Option<i64> {
        cranelift_jit_impl::try_run_block_kinded(chunk, slots, slot_kinds)
    }

    /// Like `try_run_block` but skips the tiered policy — compiles immediately
    /// on first call. Use for tests, microbenchmarks, or AOT-style usage.
    #[cfg(feature = "jit")]
    /// Public method `try_run_block_eager` — see the implementing block's surrounding context for the call contract.
    pub fn try_run_block_eager(&self, chunk: &crate::Chunk, slots: &mut [i64]) -> Option<i64> {
        cranelift_jit_impl::try_run_block_eager(chunk, slots)
    }

    /// Slot-kind-aware eager variant (see [`Self::try_run_block_kinded`]).
    #[cfg(feature = "jit")]
    pub fn try_run_block_eager_kinded(
        &self,
        chunk: &crate::Chunk,
        slots: &mut [i64],
        slot_kinds: &[SlotKind],
    ) -> Option<i64> {
        cranelift_jit_impl::try_run_block_eager_kinded(chunk, slots, slot_kinds)
    }

    /// Typed slot-kind-aware [`try_run_block_kinded`]: returns the chunk result
    /// as a [`BlockNum`], preserving float results exactly instead of truncating
    /// them to `i64`. Use this when the chunk's result may be a float value (the
    /// plain `i64` entry points truncate floats for backward compatibility).
    #[cfg(feature = "jit")]
    pub fn try_run_block_typed_kinded(
        &self,
        chunk: &crate::Chunk,
        slots: &mut [i64],
        slot_kinds: &[SlotKind],
    ) -> Option<BlockNum> {
        cranelift_jit_impl::try_run_block_typed_kinded(chunk, slots, slot_kinds)
    }

    /// Typed eager slot-kind-aware variant (see [`Self::try_run_block_typed_kinded`]).
    #[cfg(feature = "jit")]
    pub fn try_run_block_eager_typed_kinded(
        &self,
        chunk: &crate::Chunk,
        slots: &mut [i64],
        slot_kinds: &[SlotKind],
    ) -> Option<BlockNum> {
        cranelift_jit_impl::try_run_block_eager_typed_kinded(chunk, slots, slot_kinds)
    }

    /// Reads and clears the JIT zero-divisor trap channel set by a compiled
    /// [`Op::AwkDivJit`]/[`Op::AwkModJit`] that hit a `b == 0` divisor.
    ///
    /// Returns `true` if a trap fired since the last call (the JIT-returned
    /// result for that segment is a sentinel and must be discarded). Front-ends
    /// that lower their own float division to `Op::AwkDivJit` should call this
    /// immediately after a `try_run_block*` invocation; on `true`, decline the
    /// JIT result and re-run on the interpreter so the front-end's own
    /// division-by-zero semantics (error message, etc.) take effect.
    #[cfg(feature = "jit")]
    pub fn take_awk_div_trap(&self) -> bool {
        take_awk_div_trap() != 0
    }

    /// Check if a chunk is eligible for block JIT.
    #[cfg(feature = "jit")]
    /// Public method `is_block_eligible` — see the implementing block's surrounding context for the call contract.
    pub fn is_block_eligible(&self, chunk: &crate::Chunk) -> bool {
        cranelift_jit_impl::is_block_eligible(chunk)
    }

    /// Whether a compiled block-JIT entry exists for this chunk (i.e.,
    /// the chunk has crossed `try_run_block`'s warmup threshold and the
    /// next call will run native code, not return `None`). Lets the VM
    /// skip slot-buffer refresh when block JIT isn't ready yet.
    #[cfg(feature = "jit")]
    /// Public method `block_jit_is_compiled` — see the implementing block's surrounding context for the call contract.
    pub fn block_jit_is_compiled(&self, chunk: &crate::Chunk) -> bool {
        cranelift_jit_impl::block_jit_is_compiled(chunk)
    }

    /// Find the largest contiguous JIT-eligible region in a chunk.
    /// Returns `(start, end)` op indices, or None if no useful region exists.
    /// Useful for partial JIT compilation of chunks that aren't entirely eligible.
    #[cfg(feature = "jit")]
    /// Public method `find_jit_region` — see the implementing block's surrounding context for the call contract.
    pub fn find_jit_region(&self, chunk: &crate::Chunk) -> Option<(usize, usize)> {
        cranelift_jit_impl::find_jit_region(&chunk.ops)
    }

    /// Extract a JIT region as a standalone sub-chunk with rebased jump targets.
    /// The sub-chunk can then be passed to `try_run_block_eager` to JIT-compile
    /// just that region. Use with `find_jit_region` to find an eligible range.
    #[cfg(feature = "jit")]
    /// Public method `extract_region` — see the implementing block's surrounding context for the call contract.
    pub fn extract_region(&self, chunk: &crate::Chunk, start: usize, end: usize) -> crate::Chunk {
        cranelift_jit_impl::extract_region(chunk, start, end)
    }

    /// Stub when JIT feature is disabled.
    #[cfg(not(feature = "jit"))]
    /// No-op stub for `find_jit_region` when the `jit` cargo feature is disabled. The real implementation lives behind `#[cfg(feature = "jit")]`.
    pub fn find_jit_region(&self, _chunk: &crate::Chunk) -> Option<(usize, usize)> {
        None
    }

    /// Stub when JIT feature is disabled.
    #[cfg(not(feature = "jit"))]
    /// No-op stub for `extract_region` when the `jit` cargo feature is disabled. The real implementation lives behind `#[cfg(feature = "jit")]`.
    pub fn extract_region(&self, chunk: &crate::Chunk, _start: usize, _end: usize) -> crate::Chunk {
        chunk.clone()
    }

    /// Stub when JIT feature is disabled.
    #[cfg(not(feature = "jit"))]
    /// No-op stub for `try_run_block` when the `jit` cargo feature is disabled. The real implementation lives behind `#[cfg(feature = "jit")]`.
    pub fn try_run_block(&self, _chunk: &crate::Chunk, _slots: &mut [i64]) -> Option<i64> {
        None
    }

    /// Stub when JIT feature is disabled.
    #[cfg(not(feature = "jit"))]
    /// No-op stub for `try_run_block_eager` when the `jit` cargo feature is disabled. The real implementation lives behind `#[cfg(feature = "jit")]`.
    pub fn try_run_block_eager(&self, _chunk: &crate::Chunk, _slots: &mut [i64]) -> Option<i64> {
        None
    }

    /// Stub when JIT feature is disabled.
    #[cfg(not(feature = "jit"))]
    /// No-op stub for `is_block_eligible` when the `jit` cargo feature is disabled. The real implementation lives behind `#[cfg(feature = "jit")]`.
    pub fn is_block_eligible(&self, _chunk: &crate::Chunk) -> bool {
        false
    }

    /// Stub when JIT feature is disabled.
    #[cfg(not(feature = "jit"))]
    /// No-op stub for `block_jit_is_compiled` when the `jit` cargo feature is disabled. The real implementation lives behind `#[cfg(feature = "jit")]`.
    pub fn block_jit_is_compiled(&self, _chunk: &crate::Chunk) -> bool {
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
    /// Public method `trace_lookup` — see the implementing block's surrounding context for the call contract.
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
    /// Public method `trace_install` — see the implementing block's surrounding context for the call contract.
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
    /// Public method `trace_abort` — see the implementing block's surrounding context for the call contract.
    pub fn trace_abort(&self, chunk: &crate::Chunk, anchor_ip: usize) {
        cranelift_jit_impl::trace_abort(chunk, anchor_ip);
    }

    /// Whether a recorded sequence is eligible for trace JIT compilation.
    #[cfg(feature = "jit")]
    /// Public method `is_trace_eligible` — see the implementing block's surrounding context for the call contract.
    pub fn is_trace_eligible(&self, ops: &[crate::Op], anchor_ip: usize) -> bool {
        cranelift_jit_impl::is_trace_eligible(ops, anchor_ip)
    }

    /// Whether a compiled trace exists for (chunk, anchor_ip).
    #[cfg(feature = "jit")]
    /// Public method `trace_is_compiled` — see the implementing block's surrounding context for the call contract.
    pub fn trace_is_compiled(&self, chunk: &crate::Chunk, anchor_ip: usize) -> bool {
        cranelift_jit_impl::trace_is_compiled(chunk, anchor_ip)
    }

    /// Number of entry-guard deopts at runtime (slot type mismatch).
    #[cfg(feature = "jit")]
    /// Public method `trace_deopt_count` — see the implementing block's surrounding context for the call contract.
    pub fn trace_deopt_count(&self, chunk: &crate::Chunk, anchor_ip: usize) -> u32 {
        cranelift_jit_impl::trace_deopt_count(chunk, anchor_ip)
    }

    /// Number of mid-trace side-exits at runtime (brif guard mismatch).
    #[cfg(feature = "jit")]
    /// Public method `trace_side_exit_count` — see the implementing block's surrounding context for the call contract.
    pub fn trace_side_exit_count(&self, chunk: &crate::Chunk, anchor_ip: usize) -> u32 {
        cranelift_jit_impl::trace_side_exit_count(chunk, anchor_ip)
    }

    /// Phase 9: bump the side-exit counter for a trace and auto-blacklist
    /// past the threshold. The VM's chained-dispatch path calls this only
    /// when no side trace was found at the deopt's `resume_ip` — if a side
    /// trace handled the deopt productively, the main trace shouldn't be
    /// penalized.
    #[cfg(feature = "jit")]
    /// Public method `trace_bump_side_exit` — see the implementing block's surrounding context for the call contract.
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
    /// Public method `trace_install_with_kind` — see the implementing block's surrounding context for the call contract.
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
    /// Public method `trace_loop_anchors` — see the implementing block's surrounding context for the call contract.
    pub fn trace_loop_anchors(
        &self,
        chunk: &crate::Chunk,
        anchor_ip: usize,
    ) -> Option<(usize, usize)> {
        cranelift_jit_impl::trace_loop_anchors(chunk, anchor_ip)
    }

    /// Whether the trace was blacklisted (too many deopts).
    #[cfg(feature = "jit")]
    /// Public method `trace_is_blacklisted` — see the implementing block's surrounding context for the call contract.
    pub fn trace_is_blacklisted(&self, chunk: &crate::Chunk, anchor_ip: usize) -> bool {
        cranelift_jit_impl::trace_is_blacklisted(chunk, anchor_ip)
    }

    /// Phase 7: export the compiled trace's recording metadata so callers
    /// can serialize it for persistent caching across process restarts.
    /// Returns `None` if no compiled trace exists at `(chunk, anchor_ip)`.
    #[cfg(feature = "jit")]
    /// Public method `trace_export` — see the implementing block's surrounding context for the call contract.
    pub fn trace_export(&self, chunk: &crate::Chunk, anchor_ip: usize) -> Option<TraceMetadata> {
        cranelift_jit_impl::trace_export(chunk, anchor_ip)
    }

    /// Phase 7: re-install a previously-exported trace. The metadata's
    /// `chunk_op_hash` must match `chunk.op_hash`; otherwise returns false
    /// (stale metadata, chunk has changed). On success the trace is
    /// re-compiled and ready for the next invocation through `trace_lookup`.
    #[cfg(feature = "jit")]
    /// Public method `trace_import` — see the implementing block's surrounding context for the call contract.
    pub fn trace_import(&self, chunk: &crate::Chunk, meta: &TraceMetadata) -> bool {
        cranelift_jit_impl::trace_import(chunk, meta, &chunk.constants)
    }

    /// Bulk export every compiled trace whose `chunk_op_hash` matches
    /// `chunk.op_hash`. Use to persist the full cache for this chunk to
    /// disk in a single pass.
    #[cfg(feature = "jit")]
    /// Public method `trace_export_all` — see the implementing block's surrounding context for the call contract.
    pub fn trace_export_all(&self, chunk: &crate::Chunk) -> Vec<TraceMetadata> {
        cranelift_jit_impl::trace_export_all(chunk)
    }

    /// Bulk import a slice of trace metadata. Entries with mismatched
    /// `chunk_op_hash` are skipped. Returns the count successfully
    /// re-installed.
    #[cfg(feature = "jit")]
    /// Public method `trace_import_all` — see the implementing block's surrounding context for the call contract.
    pub fn trace_import_all(&self, chunk: &crate::Chunk, metas: &[TraceMetadata]) -> usize {
        cranelift_jit_impl::trace_import_all(chunk, metas, &chunk.constants)
    }

    /// Maximum number of ops a single trace can record.
    #[cfg(feature = "jit")]
    /// Public method `trace_max_len` — see the implementing block's surrounding context for the call contract.
    pub fn trace_max_len(&self) -> usize {
        cranelift_jit_impl::cfg_max_trace_len()
    }

    /// Apply a tracing JIT configuration to the current thread. Affects
    /// subsequent recording, dispatch, and blacklist behavior. Existing
    /// compiled traces are unaffected.
    #[cfg(feature = "jit")]
    /// Public method `set_config` — see the implementing block's surrounding context for the call contract.
    pub fn set_config(&self, cfg: TraceJitConfig) {
        cranelift_jit_impl::set_config(cfg);
    }

    /// Read the current thread's tracing JIT configuration.
    #[cfg(feature = "jit")]
    /// Public method `get_config` — see the implementing block's surrounding context for the call contract.
    pub fn get_config(&self) -> TraceJitConfig {
        cranelift_jit_impl::get_config()
    }

    /// Maximum slot index a trace can reference.
    #[cfg(feature = "jit")]
    /// Public method `trace_max_slot` — see the implementing block's surrounding context for the call contract.
    pub fn trace_max_slot(&self) -> u16 {
        cranelift_jit_impl::MAX_TRACE_SLOT
    }

    // ── Tracing JIT stubs (no-jit feature) ──

    #[cfg(not(feature = "jit"))]
    /// No-op stub for `trace_lookup` when the `jit` cargo feature is disabled. The real implementation lives behind `#[cfg(feature = "jit")]`.
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
    /// No-op stub for `trace_install` when the `jit` cargo feature is disabled. The real implementation lives behind `#[cfg(feature = "jit")]`.
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
    /// No-op stub for `trace_abort` when the `jit` cargo feature is disabled. The real implementation lives behind `#[cfg(feature = "jit")]`.
    pub fn trace_abort(&self, _chunk: &crate::Chunk, _anchor_ip: usize) {}

    #[cfg(not(feature = "jit"))]
    /// No-op stub for `is_trace_eligible` when the `jit` cargo feature is disabled. The real implementation lives behind `#[cfg(feature = "jit")]`.
    pub fn is_trace_eligible(&self, _ops: &[crate::Op], _anchor_ip: usize) -> bool {
        false
    }

    #[cfg(not(feature = "jit"))]
    /// No-op stub for `trace_is_compiled` when the `jit` cargo feature is disabled. The real implementation lives behind `#[cfg(feature = "jit")]`.
    pub fn trace_is_compiled(&self, _chunk: &crate::Chunk, _anchor_ip: usize) -> bool {
        false
    }

    #[cfg(not(feature = "jit"))]
    /// No-op stub for `trace_deopt_count` when the `jit` cargo feature is disabled. The real implementation lives behind `#[cfg(feature = "jit")]`.
    pub fn trace_deopt_count(&self, _chunk: &crate::Chunk, _anchor_ip: usize) -> u32 {
        0
    }

    #[cfg(not(feature = "jit"))]
    /// No-op stub for `trace_side_exit_count` when the `jit` cargo feature is disabled. The real implementation lives behind `#[cfg(feature = "jit")]`.
    pub fn trace_side_exit_count(&self, _chunk: &crate::Chunk, _anchor_ip: usize) -> u32 {
        0
    }

    #[cfg(not(feature = "jit"))]
    /// No-op stub for `trace_bump_side_exit` when the `jit` cargo feature is disabled. The real implementation lives behind `#[cfg(feature = "jit")]`.
    pub fn trace_bump_side_exit(&self, _chunk: &crate::Chunk, _anchor_ip: usize) {}

    #[cfg(not(feature = "jit"))]
    #[allow(clippy::too_many_arguments)]
    /// Public method `trace_install_with_kind` — see the implementing block's surrounding context for the call contract.
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
    /// No-op stub for `trace_loop_anchors` when the `jit` cargo feature is disabled. The real implementation lives behind `#[cfg(feature = "jit")]`.
    pub fn trace_loop_anchors(
        &self,
        _chunk: &crate::Chunk,
        _anchor_ip: usize,
    ) -> Option<(usize, usize)> {
        None
    }

    #[cfg(not(feature = "jit"))]
    /// No-op stub for `trace_is_blacklisted` when the `jit` cargo feature is disabled. The real implementation lives behind `#[cfg(feature = "jit")]`.
    pub fn trace_is_blacklisted(&self, _chunk: &crate::Chunk, _anchor_ip: usize) -> bool {
        false
    }

    #[cfg(not(feature = "jit"))]
    /// No-op stub for `trace_export` when the `jit` cargo feature is disabled. The real implementation lives behind `#[cfg(feature = "jit")]`.
    pub fn trace_export(&self, _chunk: &crate::Chunk, _anchor_ip: usize) -> Option<TraceMetadata> {
        None
    }

    #[cfg(not(feature = "jit"))]
    /// No-op stub for `trace_import` when the `jit` cargo feature is disabled. The real implementation lives behind `#[cfg(feature = "jit")]`.
    pub fn trace_import(&self, _chunk: &crate::Chunk, _meta: &TraceMetadata) -> bool {
        false
    }

    #[cfg(not(feature = "jit"))]
    /// No-op stub for `trace_export_all` when the `jit` cargo feature is disabled. The real implementation lives behind `#[cfg(feature = "jit")]`.
    pub fn trace_export_all(&self, _chunk: &crate::Chunk) -> Vec<TraceMetadata> {
        Vec::new()
    }

    #[cfg(not(feature = "jit"))]
    /// No-op stub for `trace_import_all` when the `jit` cargo feature is disabled. The real implementation lives behind `#[cfg(feature = "jit")]`.
    pub fn trace_import_all(&self, _chunk: &crate::Chunk, _metas: &[TraceMetadata]) -> usize {
        0
    }

    #[cfg(not(feature = "jit"))]
    /// No-op stub for `trace_max_len` when the `jit` cargo feature is disabled. The real implementation lives behind `#[cfg(feature = "jit")]`.
    pub fn trace_max_len(&self) -> usize {
        256
    }

    #[cfg(not(feature = "jit"))]
    /// No-op stub for `trace_max_slot` when the `jit` cargo feature is disabled. The real implementation lives behind `#[cfg(feature = "jit")]`.
    pub fn trace_max_slot(&self) -> u16 {
        64
    }

    #[cfg(not(feature = "jit"))]
    /// No-op stub for `set_config` when the `jit` cargo feature is disabled. The real implementation lives behind `#[cfg(feature = "jit")]`.
    pub fn set_config(&self, _cfg: TraceJitConfig) {}

    #[cfg(not(feature = "jit"))]
    /// No-op stub for `get_config` when the `jit` cargo feature is disabled. The real implementation lives behind `#[cfg(feature = "jit")]`.
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

#[cfg(all(test, feature = "jit"))]
mod awk_div_trap_tests {
    use super::{take_awk_div_trap, JitCompiler};
    use crate::{ChunkBuilder, Op, SlotKind};

    fn run_div_mod(op: Op, dividend: f64, divisor: f64) -> u8 {
        // slot0 = OP(slot0, slot1); awk pops divisor (top) then dividend, so
        // GetSlot(0)=dividend first, GetSlot(1)=divisor second.
        let mut b = ChunkBuilder::new();
        b.emit(Op::GetSlot(0), 1);
        b.emit(Op::GetSlot(1), 1);
        b.emit(op.clone(), 1);
        b.emit(Op::Dup, 1);
        b.emit(Op::SetSlot(0), 1);
        b.emit(Op::Pop, 1);
        let chunk = b.build();

        let jit = JitCompiler::new();
        assert!(jit.is_block_eligible(&chunk), "{op:?} must be block-eligible");

        let _ = take_awk_div_trap(); // clear any prior state
        let kinds = [SlotKind::Float, SlotKind::Float];
        let mut slots = vec![dividend.to_bits() as i64, divisor.to_bits() as i64];
        jit.try_run_block_eager_kinded(&chunk, &mut slots, &kinds)
            .unwrap_or_else(|| panic!("{op:?} chunk must compile"));
        take_awk_div_trap()
    }

    #[test]
    fn awk_div_jit_traps_on_zero_divisor() {
        // Zero divisor → guarded early-exit fires the trap libcall (code 1 = div).
        assert_eq!(run_div_mod(Op::AwkDivJit, 7.0, 0.0), 1, "div-by-zero must trap (code 1)");
        // Nonzero divisor → no trap.
        assert_eq!(run_div_mod(Op::AwkDivJit, 7.0, 2.0), 0, "nonzero div must not trap");
    }

    #[test]
    fn awk_mod_jit_traps_on_zero_divisor() {
        // Zero divisor → guarded early-exit fires the trap libcall (code 2 = mod).
        assert_eq!(run_div_mod(Op::AwkModJit, 7.0, 0.0), 2, "mod-by-zero must trap (code 2)");
        assert_eq!(run_div_mod(Op::AwkModJit, 7.0, 3.0), 0, "nonzero mod must not trap");
    }
}
