//! ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//! EXTENSION — NO `csrc/` COUNTERPART. Closed-world ahead-of-time compiler:
//! lowers a whole [`Chunk`] to native machine code via Cranelift. Shared by
//! every fusevm frontend (stryke / zshrs / awkrs / vimlrs) so AOT lives here
//! once; each frontend's `--build` calls this, then links the result.
//!
//! # Model — "AOT threaded code"
//!
//! The bytecode dispatch loop ([`VM::run`]) is replaced by a native function
//! with one Cranelift block per op. Each op block calls the per-op runtime
//! step (`VM::aot_exec_op`, reached through the `extern "C"`
//! [`fusevm_aot_exec_op`] shim), which runs that op's semantics via the same
//! `VM::exec_op` the interpreter uses — the single source of truth — and
//! returns the **next instruction index** (or `-1` to terminate). The native
//! code branches on that:
//!
//! ```text
//!   entry → dispatch(0)
//!   dispatch(ip): br_table ip → [block_0, …, block_{n-1}]  (default → ret)
//!   block_i: next = exec_op(vm, i);
//!            if next < 0 → ret  else → dispatch(next)
//!   ret:     finish(vm); return
//! ```
//!
//! Routing every op through `dispatch` (rather than static fall-through) keeps
//! the lowering uniform and correct for data-dependent targets — `Op::Jump`,
//! the `JumpIf*` family, and intra-chunk `Op::Call`/`Op::Return`, whose target
//! is only known at run time — without the native code ever reading the `VM`
//! struct layout. Op *semantics* still live in the runtime (linked in), so the
//! interpreter dispatch loop is gone but the work each op does is unchanged.
//!
//! # Native op specialization (in progress)
//!
//! Layered on top of the threaded path, [`build_entry`] lowers chunks that are
//! integer/boolean computations directly to native IR. `analyze_native` runs
//! an abstract interpretation over the operand stack — tracking int-vs-bool
//! `Kind`s, finding basic-block leaders, and checking join consistency — and,
//! when the whole chunk qualifies, `build_entry_native` emits one Cranelift
//! block per leader with the operand stack held in frontend `Variable`s (an
//! `i64` and an `f64` variable per position; the plan's `Kind`s say which is
//! live). This covers integer **and float** arithmetic/comparisons — including
//! `int→float` promotion mirroring the interpreter — modulo (`Mod`: integer
//! `srem` guarding the divisors that would trap, or an `fmod` libcall for
//! floats) and power (`Pow`/`PowFloat` via a `powf` libcall), the math
//! intrinsics (`AbsFloat`/`SqrtFloat`/`Ceil`/`Floor`/`Trunc`/`RoundFloat` as
//! single instructions; `Sin`/`Cos`/`Tan`/`Exp`/`Log`/… and `Atan2Float` via
//! libcalls; `AbsInt`/`TruncInt`; `GcdInt`/`LcmInt` as internal Euclid loops;
//! and the awk scalar ops — `AwkDiv`/`AwkMod` and their JIT twins
//! `AwkDivJit`/`AwkModJit`/`AwkLshiftJit`/`AwkRshiftJit`/`AwkComplJit` (a native
//! error-return branch on a bad operand), plus `AwkSqrtJit`/`AwkLogJit`
//! (warn-and-return-NaN on a negative argument) — integer bitwise/shift ops
//! (`BitAnd`/…/`Shl`/`Shr`), `Inc`/`Dec`, booleans (`LoadTrue`/`LoadFalse`/
//! `LogNot`/`LogAnd`/`LogOr`), three-way compare (`Spaceship`), stack shuffles
//! (`Dup`/`Dup2`/`Swap`/`Rot`/`Pop`), control flow (`Jump`/`JumpIf*`, including
//! the value-keeping `JumpIf*Keep`), integer slots and globals
//! (`GetVar`/`SetVar`/`DeclareVar`) — both held in SSA registers, guarded by a
//! definite-assignment analysis — and the slot super-ops the
//! compiler emits for hot loops (`PreIncSlot`/`PostIncSlot`/…,
//! `AddAssignSlotVoid`, the fused `SlotLtIntJumpIfFalse` /
//! `SlotIncLtIntJumpBack`, and `AccumSumLoop` — whose internal
//! `while i < limit { sum += i; i += 1 }` is emitted as a real native loop).
//! The lowered ops use no per-op interpreter dispatch; only the final result is
//! boxed back into the VM, so a fully-scalar loop runs entirely in registers.
//!
//! # Inline/shim boundary (in progress)
//!
//! Chunks that mix scalar work with heap ops no longer fall back wholesale. The
//! *spill* half of the boundary is implemented for **sink ops** (`Print`/
//! `PrintLn`): the native code spills the top `n` register scalars onto the
//! boxed `vm.stack` (per `Kind`), runs the op via the [`fusevm_aot_exec_op`]
//! shim, and continues — `vm.stack` is empty before and after, so no reload or
//! type guard is needed. This lets a hot numeric loop with embedded output run
//! native.
//!
//! The *reload* half is started too, for **source ops whose result kind is
//! statically known** (`AwkGetFieldNum`, always `Float`): run the op via the
//! shim, then reload its pushed value into a register via `fusevm_aot_pop_float`
//! — no type guard needed because the kind is static.
//!
//! Slots and globals are typed per a chunk-wide kind inferred during analysis,
//! so a float slot/global (e.g. an awk `sum += 0.5` accumulator) lowers to an
//! `f64` register; a slot used with mixed Int/Float kinds falls back.
//!
//! # Deopt (one-way exit to the interpreter)
//!
//! A chunk no longer has to be lowerable end to end. Anything the native path
//! can't handle at a given op — a string/array/hash/heap op, a heap constant
//! load, or an *operand-type* mismatch (a `Bool` fed to `Add`, a non-scalar
//! stored to a slot) — becomes a **deopt point**: the analysis lowers everything
//! around it and stops propagating past it, and codegen emits a deopt there. So
//! a hot numeric loop alongside an occasional string op runs native, deopting
//! only where it must. The chunk falls back to threaded wholesale only for the
//! genuinely structural reasons that leave nothing to lower (stack underflow, an
//! inconsistent kind join, a mixed-kind slot, a non-numeric final result, or a
//! conditionally-assigned slot at a deopt point).
//!
//! For cases the register model can't represent, native code takes a *deopt*:
//! `emit_deopt` writes the **definitely-assigned** register-cached slots/globals
//! back to the VM (an unassigned one is left `Undef`, not a register's zero).
//! Because a register can't distinguish a real `0` from `Undef`, a chunk where a
//! slot/global is only *maybe* assigned (conditionally written) at a deopt point
//! falls back to threaded rather than risk the resumed interpreter reading the
//! wrong value. The deopt also spills
//! the live operand stack onto the boxed stack, then `fusevm_aot_resume`s the
//! interpreter at the deopt ip (which owns the rest of the run). `Op::Div` uses
//! this for its rare divide-by-zero case (which yields `Undef`): the common path
//! is a native `fdiv`, and only a zero divisor deopts.
//!
//! `GetStatus` (`$?`) is lowered as a statically-typed `Status` source (it
//! always pushes a `Value::Status`): the code is carried in an `i64` register,
//! float-promotes in arithmetic, is truthy when **zero** (shell success), and
//! boxes back as `Value::Status`.
//!
//! Still deferred: source ops whose result kind is **truly dynamic** (array/hash
//! reads, `ReadLine`, calls) — these reuse the deopt machinery but need a
//! runtime type guard on the boxed value (deopt when it isn't the expected
//! scalar). Anything not yet specialized falls back wholesale to
//! `build_entry_threaded`.
//!
//! [`build_entry`] is generic over [`Module`], so the in-memory JIT path used
//! to validate the compiler ([`run_chunk_native`]) and the on-disk object path
//! (`ObjectModule`, wired in the linker stage) share identical codegen.
//! ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

// The AOT shims and entry pass `*mut VM` across `extern "C"` boundaries, but
// both sides are Rust compilation units linked together (the emitted object, the
// fusevm runtime, and the frontend). The pointer is an opaque handle — C code
// never dereferences it — so the FFI-safety lint is a false positive here.
#![allow(improper_ctypes, improper_ctypes_definitions)]

use crate::chunk::Chunk;
use crate::op::Op;
use crate::value::Value;
use crate::vm::{VMResult, VM};

use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};
use cranelift_codegen::ir::immediates::Ieee64;
use cranelift_codegen::ir::{types, AbiParam, Block, BlockArg, FuncRef, InstBuilder};
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Switch, Variable};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{default_libcall_names, DataDescription, FuncId, Linkage, Module};
use cranelift_object::{ObjectBuilder, ObjectModule};
use std::collections::{BTreeSet, HashMap};
use std::path::Path;

/// Exported symbol of the embedded serialized chunk (defined in the object).
pub const AOT_CHUNK_BLOB_SYMBOL: &str = "fusevm_aot_chunk_blob";
/// Exported symbol holding the embedded chunk's length (u64, little-endian).
pub const AOT_CHUNK_LEN_SYMBOL: &str = "fusevm_aot_chunk_len";

/// Exported symbol name of the compiled entry function.
pub const AOT_ENTRY_SYMBOL: &str = "fusevm_aot_entry";

/// Per-op runtime step, called by the native driver once per op. Runs the op at
/// `ip` via the shared `VM::aot_exec_op` and returns the next instruction
/// index, or `-1` when the run terminates.
///
/// # Safety
/// `vm` must be a valid, uniquely-borrowed `*mut VM` for the duration of the
/// call (the driver always passes the live VM it was handed).
#[no_mangle]
pub extern "C" fn fusevm_aot_exec_op(vm: *mut VM, ip: u64) -> i64 {
    debug_assert!(!vm.is_null());
    // SAFETY: see the function contract; the driver owns the VM for the run.
    unsafe { (*vm).aot_exec_op(ip as usize) }
}

/// Finalize the run, computing the tail result if no op stored one. Called once
/// by the native driver's return path.
///
/// # Safety
/// Same contract as [`fusevm_aot_exec_op`].
#[no_mangle]
pub extern "C" fn fusevm_aot_finish(vm: *mut VM) {
    debug_assert!(!vm.is_null());
    // SAFETY: see the function contract; the driver owns the VM for the run.
    unsafe { (*vm).aot_finish() }
}

/// Store an integer result computed by natively-lowered AOT code (see
/// `VM::aot_set_int_result`). The native fast path keeps intermediate values
/// in registers, so it reports its final result through this shim rather than
/// the boxed stack-tail logic in [`fusevm_aot_finish`].
///
/// # Safety
/// Same contract as [`fusevm_aot_exec_op`].
#[no_mangle]
pub extern "C" fn fusevm_aot_set_int_result(vm: *mut VM, n: i64) {
    debug_assert!(!vm.is_null());
    // SAFETY: see the function contract; the driver owns the VM for the run.
    unsafe { (*vm).aot_set_int_result(n) }
}

/// Store a float result from natively-lowered AOT code (see
/// `VM::aot_set_float_result`). The float analog of
/// [`fusevm_aot_set_int_result`].
///
/// # Safety
/// Same contract as [`fusevm_aot_exec_op`].
#[no_mangle]
pub extern "C" fn fusevm_aot_set_float_result(vm: *mut VM, f: f64) {
    debug_assert!(!vm.is_null());
    // SAFETY: see the function contract; the driver owns the VM for the run.
    unsafe { (*vm).aot_set_float_result(f) }
}

/// Box the boxed-stack top into the value arena, returning its handle
/// (`VM::aot_box`). Used to thread a heap value through a register.
///
/// # Safety
/// Same contract as [`fusevm_aot_exec_op`].
#[no_mangle]
pub extern "C" fn fusevm_aot_box(vm: *mut VM) -> i64 {
    debug_assert!(!vm.is_null());
    // SAFETY: see the function contract; the driver owns the VM for the run.
    unsafe { (*vm).aot_box() }
}

/// Push the arena value for `handle` back onto the boxed stack
/// (`VM::aot_unbox`), so a shimmed op can consume it.
///
/// # Safety
/// Same contract as [`fusevm_aot_exec_op`].
#[no_mangle]
pub extern "C" fn fusevm_aot_unbox(vm: *mut VM, handle: i64) {
    debug_assert!(!vm.is_null());
    // SAFETY: see the function contract; the driver owns the VM for the run.
    unsafe { (*vm).aot_unbox(handle) }
}

/// Store a boxed (heap) result from a register handle
/// (`VM::aot_set_obj_result`).
///
/// # Safety
/// Same contract as [`fusevm_aot_exec_op`].
#[no_mangle]
pub extern "C" fn fusevm_aot_set_obj_result(vm: *mut VM, handle: i64) {
    debug_assert!(!vm.is_null());
    // SAFETY: see the function contract; the driver owns the VM for the run.
    unsafe { (*vm).aot_set_obj_result(handle) }
}

/// Clone the value behind `handle` into a fresh owned handle, returned
/// (`VM::aot_clone`). Used when reading an `Obj` slot.
///
/// # Safety
/// Same contract as [`fusevm_aot_exec_op`].
#[no_mangle]
pub extern "C" fn fusevm_aot_clone(vm: *mut VM, handle: i64) -> i64 {
    debug_assert!(!vm.is_null());
    // SAFETY: see the function contract; the driver owns the VM for the run.
    unsafe { (*vm).aot_clone(handle) }
}

/// Free an owned handle (`VM::aot_free`); negative is the empty-slot sentinel.
///
/// # Safety
/// Same contract as [`fusevm_aot_exec_op`].
#[no_mangle]
pub extern "C" fn fusevm_aot_free(vm: *mut VM, handle: i64) {
    debug_assert!(!vm.is_null());
    // SAFETY: see the function contract; the driver owns the VM for the run.
    unsafe { (*vm).aot_free(handle) }
}

/// Deopt writeback of an `Obj` slot from a register handle
/// (`VM::aot_store_slot_obj`).
///
/// # Safety
/// Same contract as [`fusevm_aot_exec_op`].
#[no_mangle]
pub extern "C" fn fusevm_aot_store_slot_obj(vm: *mut VM, idx: u32, handle: i64) {
    debug_assert!(!vm.is_null());
    // SAFETY: see the function contract; the driver owns the VM for the run.
    unsafe { (*vm).aot_store_slot_obj(idx, handle) }
}

/// Global analog of [`fusevm_aot_store_slot_obj`] (`VM::aot_store_global_obj`).
///
/// # Safety
/// Same contract as [`fusevm_aot_exec_op`].
#[no_mangle]
pub extern "C" fn fusevm_aot_store_global_obj(vm: *mut VM, idx: u32, handle: i64) {
    debug_assert!(!vm.is_null());
    // SAFETY: see the function contract; the driver owns the VM for the run.
    unsafe { (*vm).aot_store_global_obj(idx, handle) }
}

/// Store an error result (by code) from natively-lowered AOT code that hit a
/// runtime fault (e.g. awk division by zero). The native side branches to a
/// block that calls this and returns early, leaving `aot_result` an `Error`.
///
/// # Safety
/// Same contract as [`fusevm_aot_exec_op`].
#[no_mangle]
pub extern "C" fn fusevm_aot_set_error(vm: *mut VM, code: u32) {
    debug_assert!(!vm.is_null());
    // SAFETY: see the function contract; the driver owns the VM for the run.
    unsafe { (*vm).aot_set_error(code) }
}

/// Deopt writeback: store a register-cached int slot back to the VM frame (see
/// `VM::aot_store_slot_int`).
///
/// # Safety
/// Same contract as [`fusevm_aot_exec_op`].
#[no_mangle]
pub extern "C" fn fusevm_aot_store_slot_int(vm: *mut VM, idx: u32, n: i64) {
    debug_assert!(!vm.is_null());
    // SAFETY: see the function contract; the driver owns the VM for the run.
    unsafe { (*vm).aot_store_slot_int(idx, n) }
}

/// Float analog of [`fusevm_aot_store_slot_int`].
///
/// # Safety
/// Same contract as [`fusevm_aot_exec_op`].
#[no_mangle]
pub extern "C" fn fusevm_aot_store_slot_float(vm: *mut VM, idx: u32, f: f64) {
    debug_assert!(!vm.is_null());
    // SAFETY: see the function contract; the driver owns the VM for the run.
    unsafe { (*vm).aot_store_slot_float(idx, f) }
}

/// Global analog of [`fusevm_aot_store_slot_int`].
///
/// # Safety
/// Same contract as [`fusevm_aot_exec_op`].
#[no_mangle]
pub extern "C" fn fusevm_aot_store_global_int(vm: *mut VM, idx: u32, n: i64) {
    debug_assert!(!vm.is_null());
    // SAFETY: see the function contract; the driver owns the VM for the run.
    unsafe { (*vm).aot_store_global_int(idx, n) }
}

/// Float global analog of [`fusevm_aot_store_slot_int`].
///
/// # Safety
/// Same contract as [`fusevm_aot_exec_op`].
#[no_mangle]
pub extern "C" fn fusevm_aot_store_global_float(vm: *mut VM, idx: u32, f: f64) {
    debug_assert!(!vm.is_null());
    // SAFETY: see the function contract; the driver owns the VM for the run.
    unsafe { (*vm).aot_store_global_float(idx, f) }
}

/// Deopt exit: resume the interpreter from `ip` with reconstructed state (see
/// `VM::aot_resume`).
///
/// # Safety
/// Same contract as [`fusevm_aot_exec_op`].
#[no_mangle]
pub extern "C" fn fusevm_aot_resume(vm: *mut VM, ip: u32) {
    debug_assert!(!vm.is_null());
    // SAFETY: see the function contract; the driver owns the VM for the run.
    unsafe { (*vm).aot_resume(ip) }
}

/// Reload a float from the boxed operand stack into a native register — the
/// reload half of the boundary, for source ops with a statically-`Float` result
/// (see `VM::aot_pop_float`).
///
/// # Safety
/// Same contract as [`fusevm_aot_exec_op`].
#[no_mangle]
pub extern "C" fn fusevm_aot_pop_float(vm: *mut VM) -> f64 {
    debug_assert!(!vm.is_null());
    // SAFETY: see the function contract; the driver owns the VM for the run.
    unsafe { (*vm).aot_pop_float() }
}

/// Reload an integer from the boxed operand stack (see `VM::aot_pop_int`).
///
/// # Safety
/// Same contract as [`fusevm_aot_exec_op`].
#[no_mangle]
pub extern "C" fn fusevm_aot_pop_int(vm: *mut VM) -> i64 {
    debug_assert!(!vm.is_null());
    // SAFETY: see the function contract; the driver owns the VM for the run.
    unsafe { (*vm).aot_pop_int() }
}

/// Spill a `Status` code onto the boxed operand stack (see
/// `VM::aot_push_status`).
///
/// # Safety
/// Same contract as [`fusevm_aot_exec_op`].
#[no_mangle]
pub extern "C" fn fusevm_aot_push_status(vm: *mut VM, n: i64) {
    debug_assert!(!vm.is_null());
    // SAFETY: see the function contract; the driver owns the VM for the run.
    unsafe { (*vm).aot_push_status(n) }
}

/// Store a `Status` result from natively-lowered AOT code (see
/// `VM::aot_set_status_result`).
///
/// # Safety
/// Same contract as [`fusevm_aot_exec_op`].
#[no_mangle]
pub extern "C" fn fusevm_aot_set_status_result(vm: *mut VM, n: i64) {
    debug_assert!(!vm.is_null());
    // SAFETY: see the function contract; the driver owns the VM for the run.
    unsafe { (*vm).aot_set_status_result(n) }
}

/// Spill an integer register onto the boxed operand stack ahead of a shimmed
/// (non-lowered) op — the spill half of the inline/shim boundary (see
/// `VM::aot_push_int`).
///
/// # Safety
/// Same contract as [`fusevm_aot_exec_op`].
#[no_mangle]
pub extern "C" fn fusevm_aot_push_int(vm: *mut VM, n: i64) {
    debug_assert!(!vm.is_null());
    // SAFETY: see the function contract; the driver owns the VM for the run.
    unsafe { (*vm).aot_push_int(n) }
}

/// Float analog of [`fusevm_aot_push_int`].
///
/// # Safety
/// Same contract as [`fusevm_aot_exec_op`].
#[no_mangle]
pub extern "C" fn fusevm_aot_push_float(vm: *mut VM, f: f64) {
    debug_assert!(!vm.is_null());
    // SAFETY: see the function contract; the driver owns the VM for the run.
    unsafe { (*vm).aot_push_float(f) }
}

/// Bool analog of [`fusevm_aot_push_int`].
///
/// # Safety
/// Same contract as [`fusevm_aot_exec_op`].
#[no_mangle]
pub extern "C" fn fusevm_aot_push_bool(vm: *mut VM, n: i64) {
    debug_assert!(!vm.is_null());
    // SAFETY: see the function contract; the driver owns the VM for the run.
    unsafe { (*vm).aot_push_bool(n) }
}

/// Emit an awk negative-argument warning to stderr, keyed by `code` (0 = sqrt,
/// 1 = log), formatting `value` exactly as the interpreter does. Pure (no VM
/// access); used by `AwkSqrtJit`/`AwkLogJit`'s negative branch.
#[no_mangle]
pub extern "C" fn fusevm_aot_awk_warn(code: u32, value: f64) {
    let func = if code == 0 { "sqrt" } else { "log" };
    eprintln!("awk: warning: {func}: received negative argument {value}");
}

/// Math libcall for `Op::Pow` — there is no single instruction for `powf`, so
/// native code calls it exactly as compiled Rust/C would. Pure (no VM access).
#[no_mangle]
pub extern "C" fn fusevm_aot_powf(a: f64, b: f64) -> f64 {
    a.powf(b)
}

/// Math libcall for the float case of `Op::Mod` (`f64` remainder). Matches the
/// interpreter's `a % b`. Pure (no VM access).
#[no_mangle]
pub extern "C" fn fusevm_aot_fmod(a: f64, b: f64) -> f64 {
    a % b
}

/// Unary transcendental dispatcher for the `*Float` math ops, keyed by `id` (see
/// the id assignment in `build_entry_native`'s codegen). One libcall import
/// covers all of them. Pure (no VM access).
#[no_mangle]
pub extern "C" fn fusevm_aot_unary_math(id: u32, x: f64) -> f64 {
    match id {
        0 => x.sin(),
        1 => x.cos(),
        2 => x.tan(),
        3 => x.asin(),
        4 => x.acos(),
        5 => x.atan(),
        6 => x.sinh(),
        7 => x.cosh(),
        8 => x.tanh(),
        9 => x.exp(),
        10 => x.ln(),
        11 => x.log2(),
        12 => x.log10(),
        _ => f64::NAN,
    }
}

/// Math libcall for `Op::Atan2Float` (`y.atan2(x)`). Pure (no VM access).
#[no_mangle]
pub extern "C" fn fusevm_aot_atan2(y: f64, x: f64) -> f64 {
    y.atan2(x)
}

/// Build a Cranelift ISA for the host. `is_pic=false` matches the in-process
/// JIT (used by [`run_chunk_native`]); the object path overrides this.
fn host_isa() -> Result<cranelift_codegen::isa::OwnedTargetIsa, String> {
    let mut fb = settings::builder();
    let _ = fb.set("use_colocated_libcalls", "false");
    let _ = fb.set("is_pic", "false");
    let _ = fb.set("opt_level", "speed");
    let flags = settings::Flags::new(fb);
    let isa_builder = cranelift_native::builder().map_err(|e| format!("aot: native ISA: {e}"))?;
    isa_builder
        .finish(flags)
        .map_err(|e| format!("aot: ISA finish: {e}"))
}

/// Lower `chunk` to a native entry function in `module` and return its id. The
/// emitted function has signature `extern "C" fn(*mut VM) -> i64` (the return
/// is a status word, always 0; the run's result is left in the VM). Generic
/// over [`Module`] so JIT and object backends share this codegen.
///
/// Two codegen strategies, selected per chunk:
///
/// * `build_entry_native` — when `analyze_native` approves the whole chunk
///   (an integer/boolean computation over the supported op set, including
///   control flow and slots), every op is lowered to real native IR
///   (`iadd`/`icmp`/`brif`/…) with the operand stack and slots held in
///   registers and **no per-op interpreter dispatch**; only the final result
///   is boxed back into the VM.
/// * `build_entry_threaded` — the general fallback: one block per op, each
///   running its semantics through the [`fusevm_aot_exec_op`] shim. Handles
///   everything the native path doesn't yet cover (strings, arrays, hashes,
///   calls, float arithmetic, …).
pub fn build_entry<M: Module>(module: &mut M, chunk: &Chunk) -> Result<FuncId, String> {
    match analyze_native(chunk) {
        Some(plan) => build_entry_native(module, chunk, &plan),
        None => build_entry_threaded(module, chunk),
    }
}

/// Abstract value kind tracked while validating a chunk for native lowering.
/// The native path must know each stack value's representation to emit the
/// right instruction (integer vs float) and the right result/branch handling,
/// and to reject divergent cases: a `Bool` fed into arithmetic or stored in a
/// slot, or left as the program result, would not match the interpreter (which
/// would float-promote, or produce a `Value::Bool`). `Int`/`Bool`/`Status` are
/// carried as `i64` at runtime; `Float` as `f64`. `Status` (an exit code from
/// `$?`) holds the code as `i64` but coerces to float in arithmetic — like the
/// interpreter's non-Int branch — boxes back as `Value::Status`, and is truthy
/// when the code is **zero** (shell success), inverted from `Int`.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Int,
    Bool,
    Float,
    Status,
    /// A boxed heap value (string/array/hash/…) that can't fit a scalar
    /// register. Carried as an `i64` *handle* into the VM's value arena (so it
    /// rides the `ivars`, like an integer); produced by heap sources and
    /// consumed by heap ops run through the `exec_op` shim. Not numeric and not
    /// int-like — feeding it to arithmetic/bitwise deopts.
    Obj,
}

impl Kind {
    /// Numeric kinds may feed arithmetic and comparisons (a `Bool` may not).
    /// `Status` participates as a float-coercing operand.
    fn is_numeric(self) -> bool {
        matches!(self, Kind::Int | Kind::Float | Kind::Status)
    }

    /// Int-like kinds are carried as `i64` and feed integer bitwise/shift ops
    /// and `Inc`/`Dec` directly (a `Bool`'s 0/1 equals its `to_int`, so it works
    /// unchanged); `Float`/`Status` do not (they coerce).
    fn is_intlike(self) -> bool {
        matches!(self, Kind::Int | Kind::Bool)
    }

    /// Whether an arithmetic/comparison with this operand promotes to float.
    /// Both Int ⇒ integer; anything else (Float or Status) ⇒ float.
    fn promotes_float(self) -> bool {
        !matches!(self, Kind::Int)
    }
}

/// Record that the slot/global keyed by `key` holds kind `k` (its single kind
/// for the whole chunk, since its register has one type). Returns `false` on a
/// mixed-kind conflict (e.g. a slot stored both Int and Float), which
/// disqualifies native lowering.
fn set_var_kind(map: &mut HashMap<u32, Kind>, key: u32, k: Kind) -> bool {
    match map.get(&key) {
        Some(&existing) => existing == k,
        None => {
            map.insert(key, k);
            true
        }
    }
}

/// The kind of the slot/global keyed by `key` in a finished plan (absent ⇒ an
/// unused index ⇒ `Int`).
fn var_ty_kind(plan: &NativePlan, key: u32) -> Kind {
    plan.var_kinds.get(&key).copied().unwrap_or(Kind::Int)
}

/// Stack effect of a "boxed heap op" that the native path runs through the
/// `exec_op` shim: `(operands_popped, pushed_result_kind)`. The operands are
/// staged onto the boxed stack (scalars spilled by kind, `Obj` handles unboxed),
/// the op runs, then its result is reloaded — boxed into the arena for an `Obj`
/// result, or popped into a register for a scalar one (`None` ⇒ pushes nothing).
/// Pop counts MUST exactly match the interpreter's op (else the boxed stack
/// desyncs). `Concat` has its own arm; everything here is fixed- or
/// operand-encoded-arity with no register-resident operands of its own.
fn heap_op_effect(op: &Op) -> Option<(usize, Option<Kind>)> {
    Some(match op {
        Op::StringRepeat => (2, Some(Kind::Obj)),
        Op::StringLen => (1, Some(Kind::Int)),
        // String comparisons: pop 2 (stringified operands), push Bool; `StrCmp`
        // pushes Int (-1/0/1). Pure stack ops (no named var, no alias hazard).
        Op::StrEq | Op::StrNe | Op::StrLt | Op::StrGt | Op::StrLe | Op::StrGe => {
            (2, Some(Kind::Bool))
        }
        Op::StrCmp => (2, Some(Kind::Int)),
        Op::MakeArray(n) => (*n as usize, Some(Kind::Obj)),
        Op::MakeHash(n) => (*n as usize, Some(Kind::Obj)),
        Op::Range => (2, Some(Kind::Obj)),
        Op::RangeStep => (3, Some(Kind::Obj)),
        // Input + shell/string expansion sources (host-dispatched in the shim;
        // pure stack effect, push a boxed string/array). Keep these on the
        // native driver instead of the threaded fallback.
        Op::ReadLine => (0, Some(Kind::Obj)),
        Op::Glob | Op::GlobRecursive => (1, Some(Kind::Obj)),
        Op::TildeExpand | Op::BraceExpand | Op::WordSplit => (1, Some(Kind::Obj)),
        // Shell-conditional predicates → Bool: glob match (`[[ x = pat ]]`/case),
        // regex match (`=~`), file tests (`-f`/`-d`/...). Pure stack effect.
        Op::StrMatch | Op::RegexMatch => (2, Some(Kind::Bool)),
        Op::TestFile(_) => (1, Some(Kind::Bool)),
        // `${var...}` parameter expansion: pops `argc` modifier args + the name,
        // pushes the expanded value (Obj). The arg count per modifier MUST match
        // the interpreter exactly. The name is a stack value (not a global index),
        // so no register-alias hazard.
        Op::ExpandParam(m) => {
            use crate::op::param_mod::*;
            let argc = match *m {
                DEFAULT | ASSIGN | ERROR | ALTERNATE | STRIP_SHORT | STRIP_LONG
                | RSTRIP_SHORT | RSTRIP_LONG => 1,
                SUBST_FIRST | SUBST_ALL | SLICE => 2,
                _ => 0,
            };
            (argc + 1, Some(Kind::Obj))
        }
        // ── AWK string/print ops ──────────────────────────────────────────
        // Dispatched through the frontend awk host in the shim; the awk record /
        // fields / arrays live in that host (not in fusevm registers), so these
        // have a pure operand-stack effect and no register-alias hazard. Results
        // are boxed (`Obj`) since awk values carry numeric-string duality. The
        // `u8` payload is the operand count (matching the op's contract). NOT
        // differential-testable in fusevm core (needs the awk host) — verified
        // end-to-end in awkrs.
        Op::AwkToLower | Op::AwkToUpper => (1, Some(Kind::Obj)),
        Op::AwkIndex => (2, Some(Kind::Obj)),
        Op::AwkLength(n) | Op::AwkSubstr(n) | Op::AwkSprintf(n) => {
            (*n as usize, Some(Kind::Obj))
        }
        Op::AwkPrint(n) | Op::AwkPrintf(n) => (*n as usize, None),
        // NOTE: `split`/`sub`/`gsub`/`match` are intentionally NOT lowered here.
        // The engine could (pure stack effect, counts boxed), but awkrs's own AOT
        // compiler rejects them upstream ("unsupported builtin call"), so they
        // can't be verified end-to-end yet. Lowering them requires the awkrs
        // frontend to emit them first — added here only once verifiable.
        // Named array/hash element ops operate on VM-scope heap state
        // (`self.globals[name]`) entirely inside the shim; the native code only
        // stages the stack operands and boxes the result. SOUND ONLY when the
        // name is not also a register-cached global — `analyze_native` bails on
        // that overlap (see `heap_op_name`).
        Op::ArrayGet(_) => (1, Some(Kind::Obj)),
        Op::ArraySet(_) => (2, None),
        Op::ArrayPush(_) => (1, None),
        Op::ArrayLen(_) => (0, Some(Kind::Int)),
        Op::ArrayPop(_) => (0, Some(Kind::Obj)),
        Op::ArrayShift(_) => (0, Some(Kind::Obj)),
        Op::HashGet(_) => (1, Some(Kind::Obj)),
        Op::HashSet(_) => (2, None),
        Op::DeclareArray(_) => (0, None),
        Op::DeclareHash(_) => (0, None),
        // Whole-array / hash-view ops, also on `self.globals[name]`.
        Op::GetArray(_) => (0, Some(Kind::Obj)),
        Op::SetArray(_) => (1, None),
        Op::HashKeys(_) => (0, Some(Kind::Obj)),
        Op::HashValues(_) => (0, Some(Kind::Obj)),
        Op::HashDelete(_) => (1, Some(Kind::Obj)),
        Op::HashExists(_) => (1, Some(Kind::Bool)),
        _ => return None,
    })
}

/// Name-pool index of a heap op that reads/writes `self.globals[name]` in the
/// shim. Such a name must NOT also be a register-cached global (the native path
/// would hold a stale register for it), so `analyze_native` bails on overlap.
fn heap_op_name(op: &Op) -> Option<u16> {
    match op {
        Op::ArrayGet(n)
        | Op::ArraySet(n)
        | Op::ArrayPush(n)
        | Op::ArrayLen(n)
        | Op::ArrayPop(n)
        | Op::ArrayShift(n)
        | Op::HashGet(n)
        | Op::HashSet(n)
        | Op::DeclareArray(n)
        | Op::DeclareHash(n)
        | Op::GetArray(n)
        | Op::SetArray(n)
        | Op::HashKeys(n)
        | Op::HashValues(n)
        | Op::HashDelete(n)
        | Op::HashExists(n) => Some(*n),
        _ => None,
    }
}

/// Imported-shim handles used to emit a deopt exit.
struct DeoptRefs {
    store_slot_int: FuncRef,
    store_slot_float: FuncRef,
    store_slot_obj: FuncRef,
    store_global_int: FuncRef,
    store_global_float: FuncRef,
    store_global_obj: FuncRef,
    push_int: FuncRef,
    push_float: FuncRef,
    push_bool: FuncRef,
    push_status: FuncRef,
    /// Unbox a register handle back onto the boxed stack (for `Kind::Obj`
    /// operand positions when spilling at a deopt).
    unbox: FuncRef,
    resume: FuncRef,
}

/// Emit a one-way deopt exit into the current block: reconstruct the full VM
/// state the resumed interpreter needs — write register-cached slots/globals
/// back to the VM, spill the live operand-stack registers (`kinds`) onto the
/// boxed stack bottom-first — then `resume` interpretation at `deopt_ip` and
/// return. The block is terminated by the `return`.
#[allow(clippy::too_many_arguments)]
fn emit_deopt(
    b: &mut FunctionBuilder,
    plan: &NativePlan,
    vm_var: Variable,
    ivars: &[Variable],
    fvars: &[Variable],
    slot_vars: &[Variable],
    global_vars: &[Variable],
    refs: &DeoptRefs,
    kinds: &[Kind],
    assigned: &BTreeSet<u32>,
    deopt_ip: usize,
) {
    // Slots/globals: write back ONLY the definitely-assigned ones, boxed per
    // kind. An unassigned register holds a dead zero; writing it would make the
    // resumed interpreter see 0 where it expects `Undef`.
    for (i, &sv) in slot_vars.iter().enumerate() {
        if !assigned.contains(&(i as u32)) {
            continue;
        }
        let v = b.use_var(sv);
        let vm = b.use_var(vm_var);
        let idx = b.ins().iconst(types::I32, i as i64);
        match var_ty_kind(plan, i as u32) {
            Kind::Float => {
                b.ins().call(refs.store_slot_float, &[vm, idx, v]);
            }
            Kind::Obj => {
                b.ins().call(refs.store_slot_obj, &[vm, idx, v]);
            }
            _ => {
                b.ins().call(refs.store_slot_int, &[vm, idx, v]);
            }
        }
    }
    for (i, &gv) in global_vars.iter().enumerate() {
        if !assigned.contains(&(i as u32 | GLOBAL_TAG)) {
            continue;
        }
        let v = b.use_var(gv);
        let vm = b.use_var(vm_var);
        let idx = b.ins().iconst(types::I32, i as i64);
        match var_ty_kind(plan, i as u32 | GLOBAL_TAG) {
            Kind::Float => {
                b.ins().call(refs.store_global_float, &[vm, idx, v]);
            }
            Kind::Obj => {
                b.ins().call(refs.store_global_obj, &[vm, idx, v]);
            }
            _ => {
                b.ins().call(refs.store_global_int, &[vm, idx, v]);
            }
        }
    }
    // Operand stack: spill bottom-most first (interpreter order).
    for (pos, &k) in kinds.iter().enumerate() {
        let vm = b.use_var(vm_var);
        match k {
            Kind::Float => {
                let v = b.use_var(fvars[pos]);
                b.ins().call(refs.push_float, &[vm, v]);
            }
            Kind::Bool => {
                let v = b.use_var(ivars[pos]);
                b.ins().call(refs.push_bool, &[vm, v]);
            }
            Kind::Status => {
                let v = b.use_var(ivars[pos]);
                b.ins().call(refs.push_status, &[vm, v]);
            }
            Kind::Int => {
                let v = b.use_var(ivars[pos]);
                b.ins().call(refs.push_int, &[vm, v]);
            }
            // Obj: the register holds an arena handle; unbox it back onto the
            // boxed stack so the resumed interpreter sees the real value.
            Kind::Obj => {
                let v = b.use_var(ivars[pos]);
                b.ins().call(refs.unbox, &[vm, v]);
            }
        }
    }
    let vm = b.use_var(vm_var);
    let ipc = b.ins().iconst(types::I32, deopt_ip as i64);
    b.ins().call(refs.resume, &[vm, ipc]);
    let z = b.ins().iconst(types::I64, 0);
    b.ins().return_(&[z]);
}

/// Upper bound on distinct slots (or globals) a chunk may use and still lower
/// natively (each becomes an SSA register variable); larger spaces fall back to
/// threaded. Both stay well under `GLOBAL_TAG`, so the two index spaces never
/// collide in the unified definite-assignment set.
const NATIVE_SLOT_LIMIT: usize = 4096;

/// Tag bit OR-ed into a global's index to key it in the unified
/// definite-assignment set, keeping it disjoint from slot keys (both index
/// spaces are bounded by `NATIVE_SLOT_LIMIT`, far below this bit).
const GLOBAL_TAG: u32 = 0x1_0000;

/// A validated plan for native lowering: where the basic-block boundaries are,
/// the operand-stack *kinds* on entry to each block (length = depth), the
/// maximum stack depth (how many SSA stack variables to allocate), the stack
/// kinds when control falls off the end (whose top selects the result), and how
/// many slot register variables to allocate.
struct NativePlan {
    /// Basic-block leader ips (each becomes one Cranelift block).
    leaders: BTreeSet<usize>,
    /// Operand-stack kinds on entry to each leader (its depth is the length).
    entry_kinds: HashMap<usize, Vec<Kind>>,
    /// Maximum operand-stack depth reached anywhere (SSA var count).
    max_depth: usize,
    /// Operand-stack kinds when control falls off the end (empty ⇒ `Halted`).
    end_kinds: Vec<Kind>,
    /// Number of slot register variables to allocate (max slot index + 1).
    slot_count: usize,
    /// Number of global register variables to allocate (max global index + 1).
    global_count: usize,
    /// Kind of each slot/global (key = slot index, or global index | GLOBAL_TAG),
    /// so codegen types its register `i64` or `f64`. Absent ⇒ unused ⇒ `Int`.
    var_kinds: HashMap<u32, Kind>,
    /// Ips of non-lowerable ops reached on a native path: codegen emits a deopt
    /// there (spill state, resume the interpreter) instead of lowering the op.
    deopt_points: BTreeSet<usize>,
    /// Definite-assignment set at each ip that may emit a deopt (the
    /// `deopt_points`, plus every `Div`). Codegen writes back only these
    /// slots/globals, so an unassigned slot stays `Undef` for the resumed
    /// interpreter rather than becoming a register's zero.
    inits_at: HashMap<usize, BTreeSet<u32>>,
}

/// Whether `chunk` lowers natively — thin wrapper over `analyze_native` used
/// by tests to assert which path a chunk takes.
#[cfg(test)]
fn native_lowerable(chunk: &Chunk) -> bool {
    analyze_native(chunk).is_some()
}

/// Control-flow successor ips of `ops[ip]` in the native CFG — the same edges
/// `analyze_native` propagates over. A deopt point has none (native execution
/// ends there). MUST stay in sync with the branch arms in `analyze_native`.
fn native_successors(chunk: &Chunk, ip: usize, deopt_points: &BTreeSet<usize>) -> Vec<usize> {
    if deopt_points.contains(&ip) {
        return Vec::new();
    }
    match &chunk.ops[ip] {
        Op::Jump(t) => vec![*t],
        Op::JumpIfTrue(t)
        | Op::JumpIfFalse(t)
        | Op::JumpIfTrueKeep(t)
        | Op::JumpIfFalseKeep(t)
        | Op::SlotLtIntJumpIfFalse(_, _, t)
        | Op::SlotIncLtIntJumpBack(_, _, t) => vec![*t, ip + 1],
        _ => vec![ip + 1],
    }
}

/// Slots/globals this op assigns, keyed as in the definite-assignment set.
/// MUST mirror the gen sites in `analyze_native`.
fn op_assigns(op: &Op, out: &mut BTreeSet<u32>) {
    match op {
        Op::SetSlot(s)
        | Op::PreIncSlot(s)
        | Op::PreDecSlot(s)
        | Op::PostIncSlot(s)
        | Op::PostDecSlot(s)
        | Op::PreIncSlotVoid(s)
        | Op::SlotIncLtIntJumpBack(s, _, _) => {
            out.insert(u32::from(*s));
        }
        Op::SetVar(g) | Op::DeclareVar(g) => {
            out.insert(u32::from(*g) | GLOBAL_TAG);
        }
        Op::AddAssignSlotVoid(a, _) => {
            out.insert(u32::from(*a));
        }
        Op::AccumSumLoop(sum, i, _) => {
            out.insert(u32::from(*sum));
            out.insert(u32::from(*i));
        }
        _ => {}
    }
}

/// May-assigned (union) dataflow: slots/globals assigned on *some* native path
/// to each ip (entry state). Where the definite-assignment set is the
/// intersection (must-assigned), this is the union. A deopt point where the two
/// differ has a conditionally-assigned slot whose register value can't be
/// soundly written back — a register can't tell a real `0` from `Undef` — so
/// such a chunk falls back to threaded.
fn may_assigned(
    chunk: &Chunk,
    n: usize,
    deopt_points: &BTreeSet<usize>,
) -> HashMap<usize, BTreeSet<u32>> {
    let mut may: HashMap<usize, BTreeSet<u32>> = HashMap::new();
    may.insert(0, BTreeSet::new());
    let mut work = vec![0usize];
    while let Some(ip) = work.pop() {
        if ip >= n {
            continue;
        }
        let mut out = may.get(&ip).cloned().unwrap_or_default();
        op_assigns(&chunk.ops[ip], &mut out);
        for s in native_successors(chunk, ip, deopt_points) {
            if s >= n {
                continue;
            }
            let newly = !may.contains_key(&s);
            let e = may.entry(s).or_default();
            let before = e.len();
            e.extend(out.iter().copied());
            // Queue on first visit (so every reachable ip is processed) or when
            // the set grows (re-converge).
            if newly || e.len() != before {
                work.push(s);
            }
        }
    }
    may
}

/// Validate `chunk` for native lowering and, if it qualifies, return a
/// `NativePlan`. Performs an abstract interpretation over the operand stack:
/// it tracks per-position `Kind`s, discovers basic-block leaders, and checks
/// that the stack state is identical on every edge into a join point (which
/// well-formed structured bytecode always satisfies). Any unsupported op, stack
/// underflow, type violation, inconsistent join, or non-integer result makes it
/// return `None`, so the threaded path takes over.
fn analyze_native(chunk: &Chunk) -> Option<NativePlan> {
    let ops = &chunk.ops;
    let n = ops.len();
    if n == 0 {
        return Some(NativePlan {
            leaders: BTreeSet::new(),
            entry_kinds: HashMap::new(),
            max_depth: 0,
            end_kinds: Vec::new(),
            slot_count: 0,
            global_count: 0,
            var_kinds: HashMap::new(),
            deopt_points: BTreeSet::new(),
            inits_at: HashMap::new(),
        });
    }

    // Per-ip entry state: the operand-stack kinds and the set of slots/globals
    // that are *definitely* assigned on every path here (globals tagged with
    // `GLOBAL_TAG` to share one set). `state` doubles as the visited set; the
    // program-end pseudo-point is `n`. Kinds use must-match-exactly merging at
    // joins (structured bytecode guarantees this); init sets use intersection
    // and re-converge (a node is re-queued when its set shrinks).
    let mut state: HashMap<usize, (Vec<Kind>, BTreeSet<u32>)> = HashMap::new();
    let mut leaders: BTreeSet<usize> = BTreeSet::new();
    let mut max_depth = 0usize;
    let mut max_slot: Option<u16> = None;
    let mut max_global: Option<u16> = None;
    // Single kind per slot/global (Int or Float), determined as the analysis
    // runs; a mixed-kind variable disqualifies the chunk.
    let mut var_kind: HashMap<u32, Kind> = HashMap::new();
    // Ips of non-lowerable ops reached natively; each becomes a deopt exit.
    let mut deopt_points: BTreeSet<usize> = BTreeSet::new();
    let mut end_state: Option<Vec<Kind>> = None;

    state.insert(0, (Vec::new(), BTreeSet::new()));
    leaders.insert(0);
    let mut work = vec![0usize];

    while let Some(ip) = work.pop() {
        let (mut st, inits) = state.get(&ip)?.clone();
        max_depth = max_depth.max(st.len());

        // In a per-op arm: when the op can't be lowered with these operand kinds,
        // make it a deopt point (the interpreter runs it) rather than bailing the
        // whole chunk. `continue` skips the (now empty) successor propagation.
        macro_rules! deopt_unless {
            ($ok:expr) => {
                if !($ok) {
                    deopt_points.insert(ip);
                    continue;
                }
            };
        }

        // Bound the slot/global spaces we'd materialize as registers — over
        // every slot or global any op touches.
        let mut tslot = [0u16; 2];
        let mut nslot = 0usize;
        let mut tglob: Option<u16> = None;
        match &ops[ip] {
            Op::GetSlot(s)
            | Op::SetSlot(s)
            | Op::PreIncSlot(s)
            | Op::PreIncSlotVoid(s)
            | Op::PreDecSlot(s)
            | Op::PostIncSlot(s)
            | Op::PostDecSlot(s)
            | Op::SlotLtIntJumpIfFalse(s, _, _)
            | Op::SlotIncLtIntJumpBack(s, _, _) => {
                tslot[0] = *s;
                nslot = 1;
            }
            Op::AddAssignSlotVoid(a, b) | Op::AccumSumLoop(a, b, _) => {
                tslot[0] = *a;
                tslot[1] = *b;
                nslot = 2;
            }
            Op::GetVar(g) | Op::SetVar(g) | Op::DeclareVar(g) => tglob = Some(*g),
            _ => {}
        }
        for &s in &tslot[..nslot] {
            if s as usize >= NATIVE_SLOT_LIMIT {
                return None;
            }
            max_slot = Some(max_slot.map_or(s, |m| m.max(s)));
        }
        if let Some(g) = tglob {
            if g as usize >= NATIVE_SLOT_LIMIT {
                return None;
            }
            max_global = Some(max_global.map_or(g, |m| m.max(g)));
        }

        // Successors after applying ops[ip]: (successor ip, kinds, init set).
        let mut succs: Vec<(usize, Vec<Kind>, BTreeSet<u32>)> = Vec::new();
        match &ops[ip] {
            Op::Nop => succs.push((ip + 1, st, inits)),
            Op::LoadInt(_) => {
                st.push(Kind::Int);
                max_depth = max_depth.max(st.len());
                succs.push((ip + 1, st, inits));
            }
            // Read a slot's value with the slot's tracked kind (Int or Float).
            Op::GetSlot(s) => {
                let k = var_kind.get(&u32::from(*s)).copied().unwrap_or(Kind::Int);
                st.push(k);
                max_depth = max_depth.max(st.len());
                succs.push((ip + 1, st, inits));
            }
            // Read a global's value with its tracked kind.
            Op::GetVar(g) => {
                let k = var_kind
                    .get(&(u32::from(*g) | GLOBAL_TAG))
                    .copied()
                    .unwrap_or(Kind::Int);
                st.push(k);
                max_depth = max_depth.max(st.len());
                succs.push((ip + 1, st, inits));
            }
            Op::LoadFloat(_) => {
                st.push(Kind::Float);
                max_depth = max_depth.max(st.len());
                succs.push((ip + 1, st, inits));
            }
            Op::LoadConst(idx) => {
                // Numeric constants lower; a heap constant (string, etc.) is a
                // deopt point — the interpreter pushes it and runs on from there.
                match chunk.constants.get(*idx as usize)? {
                    Value::Int(_) => {
                        st.push(Kind::Int);
                        max_depth = max_depth.max(st.len());
                        succs.push((ip + 1, st, inits));
                    }
                    Value::Float(_) => {
                        st.push(Kind::Float);
                        max_depth = max_depth.max(st.len());
                        succs.push((ip + 1, st, inits));
                    }
                    // A heap constant (string/array/…) loads as a boxed handle:
                    // the shim pushes the real value, the native code boxes it
                    // into the arena and threads the handle (`Kind::Obj`).
                    _ => {
                        st.push(Kind::Obj);
                        max_depth = max_depth.max(st.len());
                        succs.push((ip + 1, st, inits));
                    }
                }
            }
            // String concat: pops two values (scalar or boxed), pushes a boxed
            // string. Runs through the shim — scalar operands are spilled by
            // kind, boxed operands are unboxed — then the result is re-boxed.
            Op::Concat => {
                let _b = st.pop()?;
                let _a = st.pop()?;
                st.push(Kind::Obj);
                max_depth = max_depth.max(st.len());
                succs.push((ip + 1, st, inits));
            }
            // Arithmetic promotes: result is Float if either operand is Float,
            // else Int — mirroring the interpreter's `arith_int_fast`.
            Op::Add | Op::Sub | Op::Mul => {
                let b = st.pop()?;
                let a = st.pop()?;
                deopt_unless!(a.is_numeric() && b.is_numeric());
                let r = if a.promotes_float() || b.promotes_float() {
                    Kind::Float
                } else {
                    Kind::Int
                };
                st.push(r);
                succs.push((ip + 1, st, inits));
            }
            // Modulo: both Int → integer `srem` (with the `y==0 → 0` guard);
            // otherwise the float branch via an `fmod` libcall. (Bool operands
            // fall back, like the other arithmetic.)
            Op::Mod => {
                let b = st.pop()?;
                let a = st.pop()?;
                deopt_unless!(a.is_numeric() && b.is_numeric());
                let r = if a.promotes_float() || b.promotes_float() {
                    Kind::Float
                } else {
                    Kind::Int
                };
                st.push(r);
                succs.push((ip + 1, st, inits));
            }
            // Division: always Float, except divide-by-zero yields `Undef` — a
            // value the register model can't represent, so the zero case takes a
            // deopt exit (rare). Operands must be numeric.
            Op::Div => {
                let b = st.pop()?;
                let a = st.pop()?;
                deopt_unless!(a.is_numeric() && b.is_numeric());
                st.push(Kind::Float);
                succs.push((ip + 1, st, inits));
            }
            // `a.powf(b)` — always Float, via a `powf` libcall.
            Op::Pow => {
                let b = st.pop()?;
                let a = st.pop()?;
                deopt_unless!(a.is_numeric() && b.is_numeric());
                st.push(Kind::Float);
                succs.push((ip + 1, st, inits));
            }
            // Unary float math (single instructions + transcendental libcalls):
            // operand coerced to f64, result Float.
            Op::AbsFloat
            | Op::SqrtFloat
            | Op::CeilFloat
            | Op::FloorFloat
            | Op::TruncFloat
            | Op::RoundFloat
            | Op::SinFloat
            | Op::CosFloat
            | Op::TanFloat
            | Op::AsinFloat
            | Op::AcosFloat
            | Op::AtanFloat
            | Op::SinhFloat
            | Op::CoshFloat
            | Op::TanhFloat
            | Op::ExpFloat
            | Op::LogFloat
            | Op::Log2Float
            | Op::Log10Float => {
                deopt_unless!(st.pop()?.is_numeric());
                st.push(Kind::Float);
                succs.push((ip + 1, st, inits));
            }
            // `to_int` of the operand → Int (Float truncates toward zero,
            // saturating like `f as i64`).
            Op::TruncInt => {
                deopt_unless!(st.pop()?.is_numeric());
                st.push(Kind::Int);
                succs.push((ip + 1, st, inits));
            }
            // Integer absolute value (wrapping); int-like operand only.
            Op::AbsInt => {
                deopt_unless!(st.pop()?.is_intlike());
                st.push(Kind::Int);
                succs.push((ip + 1, st, inits));
            }
            // Binary float math libcalls → Float.
            Op::PowFloat | Op::Atan2Float => {
                let b = st.pop()?;
                let a = st.pop()?;
                deopt_unless!(a.is_numeric() && b.is_numeric());
                st.push(Kind::Float);
                succs.push((ip + 1, st, inits));
            }
            // gcd/lcm via internal Euclid loops on magnitudes → Int. Int-like
            // operands only (the interpreter's `to_int().unsigned_abs()`).
            Op::GcdInt | Op::LcmInt => {
                let b = st.pop()?;
                let a = st.pop()?;
                deopt_unless!(a.is_intlike() && b.is_intlike());
                st.push(Kind::Int);
                succs.push((ip + 1, st, inits));
            }
            // awk `/` and `%` — float result, but a runtime error (not `Undef`)
            // on a zero divisor, so they get a native error-return branch. The
            // Jit and non-Jit variants are identical in shape and message.
            Op::AwkDivJit | Op::AwkModJit | Op::AwkDiv | Op::AwkMod => {
                let b = st.pop()?;
                let a = st.pop()?;
                deopt_unless!(a.is_numeric() && b.is_numeric());
                st.push(Kind::Float);
                succs.push((ip + 1, st, inits));
            }
            // awk lshift/rshift — two operands, error on negatives, Float result.
            Op::AwkLshiftJit | Op::AwkRshiftJit => {
                let b = st.pop()?;
                let a = st.pop()?;
                deopt_unless!(a.is_numeric() && b.is_numeric());
                st.push(Kind::Float);
                succs.push((ip + 1, st, inits));
            }
            // awk compl — one operand, error on negative, Float result.
            Op::AwkComplJit => {
                deopt_unless!(st.pop()?.is_numeric());
                st.push(Kind::Float);
                succs.push((ip + 1, st, inits));
            }
            // awk sqrt/log — one operand, Float result. Negative input warns to
            // stderr and yields NaN (no error/halt), so no fallback is needed.
            Op::AwkSqrtJit | Op::AwkLogJit => {
                deopt_unless!(st.pop()?.is_numeric());
                st.push(Kind::Float);
                succs.push((ip + 1, st, inits));
            }
            // Sink ops (inline/shim boundary): consume `n` scalars and push
            // nothing, straight-line. The spilled operands are scalar by
            // construction (everything on the native stack is), so no reload or
            // type guard is needed — just spill and let the interpreter run it.
            Op::Print(n) | Op::PrintLn(n) => {
                let n = *n as usize;
                if st.len() < n {
                    return None;
                }
                st.truncate(st.len() - n);
                succs.push((ip + 1, st, inits));
            }
            // `SetStatus` is a silent sink: pop one scalar, set `$?`, push none.
            Op::SetStatus => {
                if st.pop().is_none() {
                    return None;
                }
                succs.push((ip + 1, st, inits));
            }
            // Source op (reload half): awk numeric `$N` field read. Pops nothing,
            // pushes a value that is *statically* Float, so it reloads into a
            // float register with no type guard.
            Op::AwkGetFieldNum(_) => {
                st.push(Kind::Float);
                max_depth = max_depth.max(st.len());
                succs.push((ip + 1, st, inits));
            }
            Op::Negate => {
                let a = st.pop()?;
                deopt_unless!(a.is_numeric());
                // Int → Int; Float/Status → Float (`-to_float`).
                st.push(if a == Kind::Int { Kind::Int } else { Kind::Float });
                succs.push((ip + 1, st, inits));
            }
            // `$?` — always pushes a `Status` value; reloads into a register.
            Op::GetStatus => {
                st.push(Kind::Status);
                max_depth = max_depth.max(st.len());
                succs.push((ip + 1, st, inits));
            }
            Op::NumEq | Op::NumNe | Op::NumLt | Op::NumGt | Op::NumLe | Op::NumGe => {
                let b = st.pop()?;
                let a = st.pop()?;
                deopt_unless!(a.is_numeric() && b.is_numeric());
                st.push(Kind::Bool);
                succs.push((ip + 1, st, inits));
            }
            Op::LoadTrue | Op::LoadFalse => {
                st.push(Kind::Bool);
                max_depth = max_depth.max(st.len());
                succs.push((ip + 1, st, inits));
            }
            // Inc/Dec coerce to int and yield int; int-like operands only.
            Op::Inc | Op::Dec => {
                deopt_unless!(st.pop()?.is_intlike());
                st.push(Kind::Int);
                succs.push((ip + 1, st, inits));
            }
            // `!truthy` for any kind → Bool.
            Op::LogNot => {
                st.pop()?;
                st.push(Kind::Bool);
                succs.push((ip + 1, st, inits));
            }
            // Bitwise/shift: int-like operands (carried as i64), int result.
            Op::BitAnd | Op::BitOr | Op::BitXor | Op::Shl | Op::Shr => {
                let b = st.pop()?;
                let a = st.pop()?;
                deopt_unless!(a.is_intlike() && b.is_intlike());
                st.push(Kind::Int);
                succs.push((ip + 1, st, inits));
            }
            Op::BitNot => {
                deopt_unless!(st.pop()?.is_intlike());
                st.push(Kind::Int);
                succs.push((ip + 1, st, inits));
            }
            // Stack shuffles — kind-preserving permutations of the top values.
            Op::Dup2 => {
                let len = st.len();
                if len < 2 {
                    return None;
                }
                let a = st[len - 2];
                let b = st[len - 1];
                st.push(a);
                st.push(b);
                max_depth = max_depth.max(st.len());
                succs.push((ip + 1, st, inits));
            }
            Op::Swap => {
                let len = st.len();
                if len < 2 {
                    return None;
                }
                st.swap(len - 1, len - 2);
                succs.push((ip + 1, st, inits));
            }
            Op::Rot => {
                let len = st.len();
                if len < 3 {
                    return None;
                }
                // [a, b, c] → [b, c, a]
                st.swap(len - 3, len - 2);
                st.swap(len - 2, len - 1);
                succs.push((ip + 1, st, inits));
            }
            // Logical and/or: truthiness of both operands → Bool (not
            // short-circuit — both operands are already evaluated on the stack).
            Op::LogAnd | Op::LogOr => {
                st.pop()?;
                st.pop()?;
                st.push(Kind::Bool);
                succs.push((ip + 1, st, inits));
            }
            // Three-way compare → Int (-1 / 0 / 1).
            Op::Spaceship => {
                let b = st.pop()?;
                let a = st.pop()?;
                deopt_unless!(a.is_numeric() && b.is_numeric());
                st.push(Kind::Int);
                succs.push((ip + 1, st, inits));
            }
            // Branch on truthiness but keep the value on the stack (both arms).
            Op::JumpIfTrueKeep(t) | Op::JumpIfFalseKeep(t) => {
                let t = *t;
                if t > n {
                    return None;
                }
                if st.is_empty() {
                    return None;
                }
                if t < n {
                    leaders.insert(t);
                }
                if ip + 1 < n {
                    leaders.insert(ip + 1);
                }
                succs.push((t, st.clone(), inits.clone()));
                succs.push((ip + 1, st, inits));
            }
            // Slot read-modify-write super-ops. They read via `to_int` (so an
            // unassigned slot coerces to 0, matching a zero-init register — no
            // definite-assignment needed), write an int slot (gen), and push the
            // new (Pre*) or old (Post*) value as Int. Int slots only.
            Op::PreIncSlot(s) | Op::PreDecSlot(s) | Op::PostIncSlot(s) | Op::PostDecSlot(s) => {
                if !set_var_kind(&mut var_kind, u32::from(*s), Kind::Int) {
                    return None; // these ops are int-only
                }
                let mut ni = inits;
                ni.insert(u32::from(*s));
                st.push(Kind::Int);
                max_depth = max_depth.max(st.len());
                succs.push((ip + 1, st, ni));
            }
            Op::PreIncSlotVoid(s) => {
                if !set_var_kind(&mut var_kind, u32::from(*s), Kind::Int) {
                    return None;
                }
                let mut ni = inits;
                ni.insert(u32::from(*s));
                succs.push((ip + 1, st, ni));
            }
            Op::AddAssignSlotVoid(a, b) => {
                if !set_var_kind(&mut var_kind, u32::from(*a), Kind::Int)
                    || !set_var_kind(&mut var_kind, u32::from(*b), Kind::Int)
                {
                    return None;
                }
                let mut ni = inits;
                ni.insert(u32::from(*a)); // writes slot a (reads a and b coerce)
                succs.push((ip + 1, st, ni));
            }
            // Runs a whole `while i < limit { sum += i; i += 1 }` internally;
            // straight-line from the outside (one successor). Writes both slots.
            Op::AccumSumLoop(sum_s, i_s, _limit) => {
                if !set_var_kind(&mut var_kind, u32::from(*sum_s), Kind::Int)
                    || !set_var_kind(&mut var_kind, u32::from(*i_s), Kind::Int)
                {
                    return None;
                }
                let mut ni = inits;
                ni.insert(u32::from(*sum_s));
                ni.insert(u32::from(*i_s));
                succs.push((ip + 1, st, ni));
            }
            // Fused loop ops: a slot/limit comparison branch and an
            // increment-compare-jumpback. No stack change; the read coerces.
            Op::SlotLtIntJumpIfFalse(s, _limit, t) => {
                if !set_var_kind(&mut var_kind, u32::from(*s), Kind::Int) {
                    return None;
                }
                let t = *t;
                if t > n {
                    return None;
                }
                if t < n {
                    leaders.insert(t);
                }
                if ip + 1 < n {
                    leaders.insert(ip + 1);
                }
                succs.push((t, st.clone(), inits.clone()));
                succs.push((ip + 1, st, inits));
            }
            Op::SlotIncLtIntJumpBack(s, _limit, t) => {
                if !set_var_kind(&mut var_kind, u32::from(*s), Kind::Int) {
                    return None;
                }
                let t = *t;
                if t > n {
                    return None;
                }
                let mut ni = inits;
                ni.insert(u32::from(*s)); // increments (writes) slot s
                if t < n {
                    leaders.insert(t);
                }
                if ip + 1 < n {
                    leaders.insert(ip + 1);
                }
                succs.push((t, st.clone(), ni.clone()));
                succs.push((ip + 1, st, ni));
            }
            // Store a slot. The operand must be Int or Float (Bool/etc. aren't
            // storable as a typed register); its kind becomes the slot's kind.
            Op::SetSlot(s) => {
                let k = st.pop()?;
                deopt_unless!(k == Kind::Int || k == Kind::Float || k == Kind::Obj);
                let key = u32::from(*s);
                if !set_var_kind(&mut var_kind, key, k) {
                    return None; // mixed-kind slot
                }
                let mut ni = inits;
                ni.insert(key); // this slot is now definitely assigned
                succs.push((ip + 1, st, ni));
            }
            // Globals mirror slots: `SetVar`/`DeclareVar` store Int, Float, or a
            // boxed `Obj` and set the global's kind; `GetVar` (above) reads it.
            Op::SetVar(g) | Op::DeclareVar(g) => {
                let k = st.pop()?;
                deopt_unless!(k == Kind::Int || k == Kind::Float || k == Kind::Obj);
                let key = u32::from(*g) | GLOBAL_TAG;
                if !set_var_kind(&mut var_kind, key, k) {
                    return None;
                }
                let mut ni = inits;
                ni.insert(key);
                succs.push((ip + 1, st, ni));
            }
            Op::Pop => {
                st.pop()?;
                succs.push((ip + 1, st, inits));
            }
            Op::Dup => {
                let k = *st.last()?;
                // Duplicating a boxed handle would alias one arena slot to two
                // owners (double-free on consume); deopt instead.
                deopt_unless!(k != Kind::Obj);
                st.push(k);
                max_depth = max_depth.max(st.len());
                succs.push((ip + 1, st, inits));
            }
            Op::Jump(t) => {
                let t = *t;
                if t > n {
                    return None;
                }
                if t < n {
                    leaders.insert(t);
                }
                succs.push((t, st, inits));
            }
            Op::JumpIfTrue(t) | Op::JumpIfFalse(t) => {
                let t = *t;
                if t > n {
                    return None;
                }
                st.pop()?; // condition consumed (truthiness works for any kind)
                if t < n {
                    leaders.insert(t);
                }
                if ip + 1 < n {
                    leaders.insert(ip + 1);
                }
                succs.push((t, st.clone(), inits.clone()));
                succs.push((ip + 1, st, inits));
            }
            // Boxed heap ops (string repeat/len, array/hash construction,
            // ranges): pop their operands, push the result kind (Obj or scalar).
            // Run through the shim with box/unbox staging in codegen.
            op if heap_op_effect(op).is_some() => {
                let (pops, res) = heap_op_effect(op).unwrap();
                for _ in 0..pops {
                    st.pop()?;
                }
                if let Some(k) = res {
                    st.push(k);
                    max_depth = max_depth.max(st.len());
                }
                succs.push((ip + 1, st, inits));
            }
            // A non-lowerable op: deopt here rather than bail the whole chunk.
            // Don't propagate past it — the resumed interpreter owns everything
            // after this point on this path. `state[ip]` already holds the entry
            // (kinds, inits) the deopt needs; `succs` stays empty.
            _ => {
                deopt_points.insert(ip);
            }
        }

        for (s, sk, si) in succs {
            if s == n {
                match &end_state {
                    None => end_state = Some(sk),
                    Some(prev) if *prev != sk => return None,
                    Some(_) => {}
                }
                continue;
            }
            match state.get_mut(&s) {
                None => {
                    state.insert(s, (sk, si));
                    work.push(s);
                }
                Some((pk, pi)) => {
                    if *pk != sk {
                        return None;
                    }
                    // Intersect the init-slot sets; re-converge if this shrank it.
                    let before = pi.len();
                    pi.retain(|x| si.contains(x));
                    if pi.len() != before {
                        work.push(s);
                    }
                }
            }
        }
    }

    // Definite assignment: every reachable `GetSlot`/`GetVar` must read a
    // slot/global assigned on all paths to it. Otherwise the interpreter would
    // observe `Undef` where the native path would see a register, so we fall
    // back. (The slot/global RMW ops coerce via `to_int`, so they don't need
    // this — an unassigned read coerces to 0, matching a zero-init register.)
    for (&ip, (_kinds, inits)) in &state {
        match &ops[ip] {
            Op::GetSlot(s) if !inits.contains(&u32::from(*s)) => return None,
            Op::GetVar(g) if !inits.contains(&(u32::from(*g) | GLOBAL_TAG)) => return None,
            _ => {}
        }
    }

    // Named array/hash ops read/write `self.globals[name]` inside the shim. If
    // that same name is also a register-cached global (`GetVar`/`SetVar`), the
    // register holds the live value and the shim would see a stale
    // `self.globals[name]` — so bail to the threaded path on any such overlap.
    for &ip in state.keys() {
        if let Some(name) = heap_op_name(&ops[ip]) {
            if var_kind.contains_key(&(u32::from(name) | GLOBAL_TAG)) {
                return None;
            }
        }
    }

    // The result is the stack top when control falls off the end; it must be
    // numeric (a `Bool` there would box to the wrong `Value` variant).
    let end_kinds = match end_state {
        Some(es) => {
            if let Some(top) = es.last() {
                // A numeric or boxed (`Obj`) top can be boxed as the result; a
                // bare `Bool` cannot (it would box to the wrong Value variant).
                if !top.is_numeric() && *top != Kind::Obj {
                    return None;
                }
            }
            es
        }
        // No path reaches the end (e.g. an infinite loop): the ret block is
        // unreachable, so there is no result to box.
        None => Vec::new(),
    };

    // If every reachable op is a deopt point there is nothing to lower, so the
    // threaded path (no deopt overhead) is strictly better.
    if state.len() == deopt_points.len() {
        return None;
    }

    // Soundness: a deopt resumes the interpreter, which reads slots/globals from
    // the VM. `emit_deopt` writes back only the *definitely* assigned ones, so a
    // slot that is only *maybe* assigned at a deopt point (conditionally written
    // before it) would be read as `Undef` instead of its live register value.
    // Reject such chunks (they fall back to threaded). (`Div`'s deopt is exempt:
    // its successors are native and definite-assignment-checked.)
    if !deopt_points.is_empty() {
        let may = may_assigned(chunk, n, &deopt_points);
        for &ip in &deopt_points {
            let must = state.get(&ip).map(|(_, m)| m);
            if must != may.get(&ip) {
                return None;
            }
        }
    }

    // Definite-assignment set at each ip that may emit a deopt: the deopt points
    // plus every `Div` (whose divide-by-zero deopts). Codegen writes back only
    // these slots/globals.
    let mut inits_at: HashMap<usize, BTreeSet<u32>> = HashMap::new();
    for (&ip, (_k, inits)) in &state {
        if deopt_points.contains(&ip) || matches!(ops[ip], Op::Div) {
            inits_at.insert(ip, inits.clone());
        }
    }

    let entry_kinds = leaders
        .iter()
        .map(|&l| (l, state.get(&l).map(|(k, _)| k.clone()).unwrap_or_default()))
        .collect();
    let slot_count = max_slot.map_or(0, |m| m as usize + 1);
    let global_count = max_global.map_or(0, |m| m as usize + 1);

    Some(NativePlan {
        leaders,
        entry_kinds,
        max_depth,
        end_kinds,
        slot_count,
        global_count,
        var_kinds: var_kind,
        deopt_points,
        inits_at,
    })
}

/// Native fast path: emit real integer/float IR for a chunk that
/// `analyze_native` has approved, following the `plan` it produced. One
/// Cranelift block per basic-block leader; the operand stack lives in frontend
/// [`Variable`]s (so SSA/phi construction at joins — including loop back-edges —
/// is automatic on `seal_all_blocks`).
///
/// Each stack position has two parallel variables — one `i64` (`Kind::Int`
/// and `Kind::Bool`) and one `f64` (`Kind::Float`) — and codegen replays
/// the plan's per-position `Kind`s to read/write the right one and to insert
/// `int→float` promotions where the interpreter would. Integer slots get their
/// own SSA register variables (analysis proved definite assignment). Only the
/// final result is boxed back into the VM — no per-op dispatch, no per-op
/// shim calls.
fn build_entry_native<M: Module>(
    module: &mut M,
    chunk: &Chunk,
    plan: &NativePlan,
) -> Result<FuncId, String> {
    let ptr_ty = module.target_config().pointer_type();

    // Imported typed shims.
    let mut ires_sig = module.make_signature();
    ires_sig.params.push(AbiParam::new(ptr_ty));
    ires_sig.params.push(AbiParam::new(types::I64));
    let ires_id = module
        .declare_function("fusevm_aot_set_int_result", Linkage::Import, &ires_sig)
        .map_err(|e| format!("aot: declare set_int_result: {e}"))?;

    let mut fres_sig = module.make_signature();
    fres_sig.params.push(AbiParam::new(ptr_ty));
    fres_sig.params.push(AbiParam::new(types::F64));
    let fres_id = module
        .declare_function("fusevm_aot_set_float_result", Linkage::Import, &fres_sig)
        .map_err(|e| format!("aot: declare set_float_result: {e}"))?;

    let mut serr_sig = module.make_signature();
    serr_sig.params.push(AbiParam::new(ptr_ty));
    serr_sig.params.push(AbiParam::new(types::I32));
    let serr_id = module
        .declare_function("fusevm_aot_set_error", Linkage::Import, &serr_sig)
        .map_err(|e| format!("aot: declare set_error: {e}"))?;

    // Math libcalls: fn(f64, f64) -> f64 (powf, fmod).
    let mut math_sig = module.make_signature();
    math_sig.params.push(AbiParam::new(types::F64));
    math_sig.params.push(AbiParam::new(types::F64));
    math_sig.returns.push(AbiParam::new(types::F64));
    let powf_id = module
        .declare_function("fusevm_aot_powf", Linkage::Import, &math_sig)
        .map_err(|e| format!("aot: declare powf: {e}"))?;
    let fmod_id = module
        .declare_function("fusevm_aot_fmod", Linkage::Import, &math_sig)
        .map_err(|e| format!("aot: declare fmod: {e}"))?;
    let atan2_id = module
        .declare_function("fusevm_aot_atan2", Linkage::Import, &math_sig)
        .map_err(|e| format!("aot: declare atan2: {e}"))?;

    // Unary transcendental dispatcher: fn(u32 id, f64) -> f64.
    let mut unary_sig = module.make_signature();
    unary_sig.params.push(AbiParam::new(types::I32));
    unary_sig.params.push(AbiParam::new(types::F64));
    unary_sig.returns.push(AbiParam::new(types::F64));
    let unary_id = module
        .declare_function("fusevm_aot_unary_math", Linkage::Import, &unary_sig)
        .map_err(|e| format!("aot: declare unary_math: {e}"))?;

    // awk negative-arg warning: fn(u32 code, f64) -> ().
    let mut warn_sig = module.make_signature();
    warn_sig.params.push(AbiParam::new(types::I32));
    warn_sig.params.push(AbiParam::new(types::F64));
    let warn_id = module
        .declare_function("fusevm_aot_awk_warn", Linkage::Import, &warn_sig)
        .map_err(|e| format!("aot: declare awk_warn: {e}"))?;

    // Inline/shim boundary: the per-op interpreter step (for sink ops) and the
    // scalar spill helpers that stage operands onto the boxed stack first.
    let mut exec_sig = module.make_signature();
    exec_sig.params.push(AbiParam::new(ptr_ty));
    exec_sig.params.push(AbiParam::new(types::I64));
    exec_sig.returns.push(AbiParam::new(types::I64));
    let exec_id = module
        .declare_function("fusevm_aot_exec_op", Linkage::Import, &exec_sig)
        .map_err(|e| format!("aot: declare exec_op: {e}"))?;

    let mut pushi_sig = module.make_signature();
    pushi_sig.params.push(AbiParam::new(ptr_ty));
    pushi_sig.params.push(AbiParam::new(types::I64));
    let pushi_id = module
        .declare_function("fusevm_aot_push_int", Linkage::Import, &pushi_sig)
        .map_err(|e| format!("aot: declare push_int: {e}"))?;
    let pushb_id = module
        .declare_function("fusevm_aot_push_bool", Linkage::Import, &pushi_sig)
        .map_err(|e| format!("aot: declare push_bool: {e}"))?;

    let mut pushf_sig = module.make_signature();
    pushf_sig.params.push(AbiParam::new(ptr_ty));
    pushf_sig.params.push(AbiParam::new(types::F64));
    let pushf_id = module
        .declare_function("fusevm_aot_push_float", Linkage::Import, &pushf_sig)
        .map_err(|e| format!("aot: declare push_float: {e}"))?;

    let mut popf_sig = module.make_signature();
    popf_sig.params.push(AbiParam::new(ptr_ty));
    popf_sig.returns.push(AbiParam::new(types::F64));
    let popf_id = module
        .declare_function("fusevm_aot_pop_float", Linkage::Import, &popf_sig)
        .map_err(|e| format!("aot: declare pop_float: {e}"))?;

    // Boxed-value (heap) shims: box (vm) -> i64 handle, unbox/set_obj_result
    // (vm, i64). `unbox`/`set_obj_result` share `pushi_sig` ((vm, i64)).
    let mut box_sig = module.make_signature();
    box_sig.params.push(AbiParam::new(ptr_ty));
    box_sig.returns.push(AbiParam::new(types::I64));
    let box_id = module
        .declare_function("fusevm_aot_box", Linkage::Import, &box_sig)
        .map_err(|e| format!("aot: declare box: {e}"))?;
    let unbox_id = module
        .declare_function("fusevm_aot_unbox", Linkage::Import, &pushi_sig)
        .map_err(|e| format!("aot: declare unbox: {e}"))?;
    let obj_res_id = module
        .declare_function("fusevm_aot_set_obj_result", Linkage::Import, &pushi_sig)
        .map_err(|e| format!("aot: declare set_obj_result: {e}"))?;

    // Deopt writeback shims: store_slot/global (vm, i32, i64|f64) and
    // resume (vm, i32).
    let mut ssi_sig = module.make_signature();
    ssi_sig.params.push(AbiParam::new(ptr_ty));
    ssi_sig.params.push(AbiParam::new(types::I32));
    ssi_sig.params.push(AbiParam::new(types::I64));
    let ssi_id = module
        .declare_function("fusevm_aot_store_slot_int", Linkage::Import, &ssi_sig)
        .map_err(|e| format!("aot: declare store_slot_int: {e}"))?;
    let sgi_id = module
        .declare_function("fusevm_aot_store_global_int", Linkage::Import, &ssi_sig)
        .map_err(|e| format!("aot: declare store_global_int: {e}"))?;

    let mut ssf_sig = module.make_signature();
    ssf_sig.params.push(AbiParam::new(ptr_ty));
    ssf_sig.params.push(AbiParam::new(types::I32));
    ssf_sig.params.push(AbiParam::new(types::F64));
    let ssf_id = module
        .declare_function("fusevm_aot_store_slot_float", Linkage::Import, &ssf_sig)
        .map_err(|e| format!("aot: declare store_slot_float: {e}"))?;
    let sgf_id = module
        .declare_function("fusevm_aot_store_global_float", Linkage::Import, &ssf_sig)
        .map_err(|e| format!("aot: declare store_global_float: {e}"))?;
    // Obj (boxed) slot/global writeback share the (vm, i32, i64) shape.
    let ssobj_id = module
        .declare_function("fusevm_aot_store_slot_obj", Linkage::Import, &ssi_sig)
        .map_err(|e| format!("aot: declare store_slot_obj: {e}"))?;
    let sgobj_id = module
        .declare_function("fusevm_aot_store_global_obj", Linkage::Import, &ssi_sig)
        .map_err(|e| format!("aot: declare store_global_obj: {e}"))?;

    // Heap-handle ownership shims: clone (vm, i64) -> i64, free (vm, i64).
    let mut clone_sig = module.make_signature();
    clone_sig.params.push(AbiParam::new(ptr_ty));
    clone_sig.params.push(AbiParam::new(types::I64));
    clone_sig.returns.push(AbiParam::new(types::I64));
    let clone_id = module
        .declare_function("fusevm_aot_clone", Linkage::Import, &clone_sig)
        .map_err(|e| format!("aot: declare clone: {e}"))?;
    let free_id = module
        .declare_function("fusevm_aot_free", Linkage::Import, &pushi_sig)
        .map_err(|e| format!("aot: declare free: {e}"))?;

    let mut resume_sig = module.make_signature();
    resume_sig.params.push(AbiParam::new(ptr_ty));
    resume_sig.params.push(AbiParam::new(types::I32));
    let resume_id = module
        .declare_function("fusevm_aot_resume", Linkage::Import, &resume_sig)
        .map_err(|e| format!("aot: declare resume: {e}"))?;

    // Status source/result: pop_int (vm) -> i64, push_status/set_status_result
    // (vm, i64).
    let mut popi_sig = module.make_signature();
    popi_sig.params.push(AbiParam::new(ptr_ty));
    popi_sig.returns.push(AbiParam::new(types::I64));
    let popi_id = module
        .declare_function("fusevm_aot_pop_int", Linkage::Import, &popi_sig)
        .map_err(|e| format!("aot: declare pop_int: {e}"))?;
    let push_status_id = module
        .declare_function("fusevm_aot_push_status", Linkage::Import, &pushi_sig)
        .map_err(|e| format!("aot: declare push_status: {e}"))?;
    let sres_id = module
        .declare_function("fusevm_aot_set_status_result", Linkage::Import, &pushi_sig)
        .map_err(|e| format!("aot: declare set_status_result: {e}"))?;

    // Exported entry: fn(vm) -> i64.
    let mut entry_sig = module.make_signature();
    entry_sig.params.push(AbiParam::new(ptr_ty));
    entry_sig.returns.push(AbiParam::new(types::I64));
    let entry_id = module
        .declare_function(AOT_ENTRY_SYMBOL, Linkage::Export, &entry_sig)
        .map_err(|e| format!("aot: declare entry: {e}"))?;

    let mut ctx = module.make_context();
    ctx.func.signature = entry_sig;
    let mut fbctx = FunctionBuilderContext::new();
    {
        let mut b = FunctionBuilder::new(&mut ctx.func, &mut fbctx);
        let ires_ref = module.declare_func_in_func(ires_id, b.func);
        let fres_ref = module.declare_func_in_func(fres_id, b.func);
        let serr_ref = module.declare_func_in_func(serr_id, b.func);
        let powf_ref = module.declare_func_in_func(powf_id, b.func);
        let fmod_ref = module.declare_func_in_func(fmod_id, b.func);
        let atan2_ref = module.declare_func_in_func(atan2_id, b.func);
        let unary_ref = module.declare_func_in_func(unary_id, b.func);
        let warn_ref = module.declare_func_in_func(warn_id, b.func);
        let exec_ref = module.declare_func_in_func(exec_id, b.func);
        let pushi_ref = module.declare_func_in_func(pushi_id, b.func);
        let pushb_ref = module.declare_func_in_func(pushb_id, b.func);
        let pushf_ref = module.declare_func_in_func(pushf_id, b.func);
        let popf_ref = module.declare_func_in_func(popf_id, b.func);
        let popi_ref = module.declare_func_in_func(popi_id, b.func);
        let sres_ref = module.declare_func_in_func(sres_id, b.func);
        let push_status_ref = module.declare_func_in_func(push_status_id, b.func);
        let box_ref = module.declare_func_in_func(box_id, b.func);
        let unbox_ref = module.declare_func_in_func(unbox_id, b.func);
        let obj_res_ref = module.declare_func_in_func(obj_res_id, b.func);
        let clone_ref = module.declare_func_in_func(clone_id, b.func);
        let free_ref = module.declare_func_in_func(free_id, b.func);
        let deopt_refs = DeoptRefs {
            store_slot_int: module.declare_func_in_func(ssi_id, b.func),
            store_slot_float: module.declare_func_in_func(ssf_id, b.func),
            store_slot_obj: module.declare_func_in_func(ssobj_id, b.func),
            store_global_int: module.declare_func_in_func(sgi_id, b.func),
            store_global_float: module.declare_func_in_func(sgf_id, b.func),
            store_global_obj: module.declare_func_in_func(sgobj_id, b.func),
            push_int: pushi_ref,
            push_float: pushf_ref,
            push_bool: pushb_ref,
            push_status: push_status_ref,
            unbox: unbox_ref,
            resume: module.declare_func_in_func(resume_id, b.func),
        };

        // The VM pointer and each operand-stack position are frontend Variables;
        // `seal_all_blocks` then builds the SSA phis at all joins for us. Each
        // stack position has an i64 var and an f64 var; the plan's Kinds say
        // which is live (consistent across all edges into any join). Slots are
        // int-only register variables (definite-assignment proven by analysis).
        let vm_var = b.declare_var(ptr_ty);
        let ivars: Vec<Variable> = (0..plan.max_depth)
            .map(|_| b.declare_var(types::I64))
            .collect();
        let fvars: Vec<Variable> = (0..plan.max_depth)
            .map(|_| b.declare_var(types::F64))
            .collect();
        // Slots/globals are typed per the plan's kind (i64 or f64).
        let var_ty = |key: u32| -> types::Type {
            if plan.var_kinds.get(&key) == Some(&Kind::Float) {
                types::F64
            } else {
                types::I64
            }
        };
        let slot_vars: Vec<Variable> = (0..plan.slot_count)
            .map(|i| b.declare_var(var_ty(i as u32)))
            .collect();
        let global_vars: Vec<Variable> = (0..plan.global_count)
            .map(|i| b.declare_var(var_ty(i as u32 | GLOBAL_TAG)))
            .collect();

        let n = chunk.ops.len();
        let entry_block = b.create_block();
        b.append_block_params_for_function_params(entry_block);
        let ret_block = b.create_block();
        let blocks: HashMap<usize, Block> = plan
            .leaders
            .iter()
            .map(|&l| (l, b.create_block()))
            .collect();

        // Resolve a branch/fallthrough target to its block (program end → ret).
        let block_for = |ip: usize| -> Block {
            if ip >= n {
                ret_block
            } else {
                blocks[&ip]
            }
        };

        // Entry: stash the VM pointer, zero-init slot/global registers (definite
        // assignment guarantees this 0 is never actually read; it only keeps the
        // variables defined on every edge for Cranelift's SSA construction),
        // then jump to the first block (or ret if empty).
        b.switch_to_block(entry_block);
        let vm_param = b.block_params(entry_block)[0];
        b.def_var(vm_var, vm_param);
        if !slot_vars.is_empty() || !global_vars.is_empty() {
            let zi = b.ins().iconst(types::I64, 0);
            let zf = b.ins().f64const(Ieee64::with_bits(0f64.to_bits()));
            // `Obj` slots start at the -1 "empty handle" sentinel so the first
            // `SetSlot` frees nothing; scalar slots zero-init (never read before
            // assignment, per definite-assignment).
            let neg1 = b.ins().iconst(types::I64, -1);
            for (i, &sv) in slot_vars.iter().enumerate() {
                let init = match var_ty_kind(plan, i as u32) {
                    Kind::Obj => neg1,
                    Kind::Float => zf,
                    _ => zi,
                };
                b.def_var(sv, init);
            }
            for (i, &gv) in global_vars.iter().enumerate() {
                let init = match var_ty_kind(plan, i as u32 | GLOBAL_TAG) {
                    Kind::Obj => neg1,
                    Kind::Float => zf,
                    _ => zi,
                };
                b.def_var(gv, init);
            }
        }
        if n == 0 {
            b.ins().jump(ret_block, &[]);
        } else {
            b.ins().jump(blocks[&0], &[]);
        }

        // `kinds` is the compile-time operand-stack kind stack (its length is
        // the depth). Push: write var at index `kinds.len()`, then push a Kind.
        // Pop: pop a Kind, then read the var at the new `kinds.len()`. All the
        // `unwrap`s / index math are sound: `analyze_native` proved depths, the
        // op set, and the per-position kinds.
        let mut kinds: Vec<Kind> = Vec::new();
        let mut terminated = true; // no open block yet
        for ip in 0..n {
            if let Some(&blk) = blocks.get(&ip) {
                if !terminated {
                    b.ins().jump(blk, &[]); // fallthrough into the leader
                }
                b.switch_to_block(blk);
                kinds = plan.entry_kinds[&ip].clone();
                terminated = false;
            } else if terminated {
                continue; // unreachable op after an unconditional terminator
            }

            // Non-lowerable op reached natively: deopt here (reconstruct state,
            // resume the interpreter) instead of lowering it. Handled uniformly
            // for every deopt point — string/heap ops and heap constant loads.
            if plan.deopt_points.contains(&ip) {
                emit_deopt(
                    &mut b,
                    plan,
                    vm_var,
                    &ivars,
                    &fvars,
                    &slot_vars,
                    &global_vars,
                    &deopt_refs,
                    &kinds,
                    &plan.inits_at[&ip],
                    ip,
                );
                terminated = true;
                continue;
            }

            match &chunk.ops[ip] {
                Op::Nop => {}
                Op::LoadInt(k) => {
                    let v = b.ins().iconst(types::I64, *k);
                    let idx = kinds.len();
                    b.def_var(ivars[idx], v);
                    kinds.push(Kind::Int);
                }
                Op::LoadFloat(f) => {
                    let v = b.ins().f64const(Ieee64::with_bits(f.to_bits()));
                    let idx = kinds.len();
                    b.def_var(fvars[idx], v);
                    kinds.push(Kind::Float);
                }
                Op::LoadConst(ci) => {
                    let idx = kinds.len();
                    match &chunk.constants[*ci as usize] {
                        Value::Int(c) => {
                            let v = b.ins().iconst(types::I64, *c);
                            b.def_var(ivars[idx], v);
                            kinds.push(Kind::Int);
                        }
                        Value::Float(f) => {
                            let v = b.ins().f64const(Ieee64::with_bits(f.to_bits()));
                            b.def_var(fvars[idx], v);
                            kinds.push(Kind::Float);
                        }
                        // Heap constant: run the shim (pushes the real Value onto
                        // the boxed stack), then box it into the arena and thread
                        // the handle through a register as `Kind::Obj`.
                        _ => {
                            let vm = b.use_var(vm_var);
                            let ipc = b.ins().iconst(types::I64, ip as i64);
                            b.ins().call(exec_ref, &[vm, ipc]);
                            let vm2 = b.use_var(vm_var);
                            let call = b.ins().call(box_ref, &[vm2]);
                            let h = b.inst_results(call)[0];
                            b.def_var(ivars[idx], h);
                            kinds.push(Kind::Obj);
                        }
                    }
                }
                // Concat: spill both operands onto the boxed stack (scalars by
                // kind, `Obj` handles unboxed), run the shim (pops 2, pushes the
                // joined string), then box the result handle. Net: pop 2, push 1
                // `Obj`. vm.stack is empty before and after.
                Op::Concat => {
                    let ky = kinds.pop().unwrap();
                    let iy = kinds.len();
                    let kx = kinds.pop().unwrap();
                    let ix = kinds.len();
                    // Bottom-most first (interpreter order): x then y.
                    for (pos, k) in [(ix, kx), (iy, ky)] {
                        let vm = b.use_var(vm_var);
                        match k {
                            Kind::Float => {
                                let v = b.use_var(fvars[pos]);
                                b.ins().call(pushf_ref, &[vm, v]);
                            }
                            Kind::Bool => {
                                let v = b.use_var(ivars[pos]);
                                b.ins().call(pushb_ref, &[vm, v]);
                            }
                            Kind::Status => {
                                let v = b.use_var(ivars[pos]);
                                b.ins().call(push_status_ref, &[vm, v]);
                            }
                            Kind::Int => {
                                let v = b.use_var(ivars[pos]);
                                b.ins().call(pushi_ref, &[vm, v]);
                            }
                            Kind::Obj => {
                                let v = b.use_var(ivars[pos]);
                                b.ins().call(unbox_ref, &[vm, v]);
                            }
                        }
                    }
                    let vm = b.use_var(vm_var);
                    let ipc = b.ins().iconst(types::I64, ip as i64);
                    b.ins().call(exec_ref, &[vm, ipc]);
                    let vm2 = b.use_var(vm_var);
                    let call = b.ins().call(box_ref, &[vm2]);
                    let h = b.inst_results(call)[0];
                    b.def_var(ivars[ix], h);
                    kinds.push(Kind::Obj);
                }
                // Boxed heap ops (string repeat/len, array/hash construction,
                // ranges): stage the top `pops` operands onto the boxed stack
                // (scalars by kind, Obj handles unboxed; bottom-most first), run
                // the shim, then reload the result (box an Obj, pop a scalar).
                op if heap_op_effect(op).is_some() => {
                    let (pops, res) = heap_op_effect(op).unwrap();
                    let base = kinds.len() - pops;
                    for pos in base..kinds.len() {
                        let vm = b.use_var(vm_var);
                        match kinds[pos] {
                            Kind::Float => {
                                let v = b.use_var(fvars[pos]);
                                b.ins().call(pushf_ref, &[vm, v]);
                            }
                            Kind::Bool => {
                                let v = b.use_var(ivars[pos]);
                                b.ins().call(pushb_ref, &[vm, v]);
                            }
                            Kind::Status => {
                                let v = b.use_var(ivars[pos]);
                                b.ins().call(push_status_ref, &[vm, v]);
                            }
                            Kind::Int => {
                                let v = b.use_var(ivars[pos]);
                                b.ins().call(pushi_ref, &[vm, v]);
                            }
                            Kind::Obj => {
                                let v = b.use_var(ivars[pos]);
                                b.ins().call(unbox_ref, &[vm, v]);
                            }
                        }
                    }
                    let vm = b.use_var(vm_var);
                    let ipc = b.ins().iconst(types::I64, ip as i64);
                    b.ins().call(exec_ref, &[vm, ipc]);
                    kinds.truncate(base);
                    match res {
                        Some(Kind::Obj) => {
                            let vm2 = b.use_var(vm_var);
                            let call = b.ins().call(box_ref, &[vm2]);
                            b.def_var(ivars[base], b.inst_results(call)[0]);
                            kinds.push(Kind::Obj);
                        }
                        Some(Kind::Float) => {
                            let vm2 = b.use_var(vm_var);
                            let call = b.ins().call(popf_ref, &[vm2]);
                            b.def_var(fvars[base], b.inst_results(call)[0]);
                            kinds.push(Kind::Float);
                        }
                        // Int/Bool/Status scalar result: pop into an i64 register
                        // under its exact kind (e.g. StringLen→Int, HashExists→Bool).
                        Some(k) => {
                            let vm2 = b.use_var(vm_var);
                            let call = b.ins().call(popi_ref, &[vm2]);
                            b.def_var(ivars[base], b.inst_results(call)[0]);
                            kinds.push(k);
                        }
                        None => {}
                    }
                }
                // `a` is the lower slot, `b`(=y) the top; matches the
                // interpreter's `arith_int_fast` (int if both Int, else float).
                op @ (Op::Add | Op::Sub | Op::Mul) => {
                    let ky = kinds.pop().unwrap();
                    let iy = kinds.len();
                    let kx = kinds.pop().unwrap();
                    let ix = kinds.len();
                    if ky.promotes_float() || kx.promotes_float() {
                        let y = load_f64(&mut b, &ivars, &fvars, iy, ky);
                        let x = load_f64(&mut b, &ivars, &fvars, ix, kx);
                        let r = match op {
                            Op::Add => b.ins().fadd(x, y),
                            Op::Sub => b.ins().fsub(x, y),
                            _ => b.ins().fmul(x, y),
                        };
                        b.def_var(fvars[ix], r);
                        kinds.push(Kind::Float);
                    } else {
                        let y = b.use_var(ivars[iy]);
                        let x = b.use_var(ivars[ix]);
                        let r = match op {
                            Op::Add => b.ins().iadd(x, y),
                            Op::Sub => b.ins().isub(x, y),
                            _ => b.ins().imul(x, y),
                        };
                        b.def_var(ivars[ix], r);
                        kinds.push(Kind::Int);
                    }
                }
                // Modulo. Both Int → integer `srem`, guarding the two divisors
                // that would trap natively but yield 0 in the interpreter:
                // `y == 0` (its explicit `→ 0`) and `y == -1` (where
                // `x % -1 == 0` for all x, also dodging the `srem(INT_MIN, -1)`
                // overflow trap). Otherwise the float branch via `fmod`.
                Op::Mod => {
                    let ky = kinds.pop().unwrap();
                    let iy = kinds.len();
                    let kx = kinds.pop().unwrap();
                    let ix = kinds.len();
                    if ky.promotes_float() || kx.promotes_float() {
                        let y = load_f64(&mut b, &ivars, &fvars, iy, ky);
                        let x = load_f64(&mut b, &ivars, &fvars, ix, kx);
                        let call = b.ins().call(fmod_ref, &[x, y]);
                        let r = b.inst_results(call)[0];
                        b.def_var(fvars[ix], r);
                        kinds.push(Kind::Float);
                    } else {
                        let y = b.use_var(ivars[iy]);
                        let x = b.use_var(ivars[ix]);
                        let zero = b.ins().iconst(types::I64, 0);
                        let one = b.ins().iconst(types::I64, 1);
                        let y_is_zero = b.ins().icmp_imm(IntCC::Equal, y, 0);
                        let y_is_neg1 = b.ins().icmp_imm(IntCC::Equal, y, -1);
                        let special = b.ins().bor(y_is_zero, y_is_neg1);
                        let safe_y = b.ins().select(special, one, y);
                        let rem = b.ins().srem(x, safe_y);
                        let r = b.ins().select(special, zero, rem);
                        b.def_var(ivars[ix], r);
                        kinds.push(Kind::Int);
                    }
                }
                // Division: native `fdiv` on the common (nonzero-divisor) path;
                // a divide-by-zero deopts to the interpreter (which produces the
                // `Undef` the register model can't hold). Operands are NOT popped
                // before the guard, so the deopt spills the full stack and the
                // interpreter re-executes the Div at `ip`.
                Op::Div => {
                    let d = kinds.len();
                    let ky = kinds[d - 1];
                    let kx = kinds[d - 2];
                    let divisor = load_f64(&mut b, &ivars, &fvars, d - 1, ky);
                    let dividend = load_f64(&mut b, &ivars, &fvars, d - 2, kx);
                    let zero = b.ins().f64const(Ieee64::with_bits(0f64.to_bits()));
                    let is_zero = b.ins().fcmp(FloatCC::Equal, divisor, zero);
                    let deopt_blk = b.create_block();
                    let ok_blk = b.create_block();
                    b.ins().brif(is_zero, deopt_blk, &[], ok_blk, &[]);

                    b.switch_to_block(deopt_blk);
                    emit_deopt(
                        &mut b,
                        plan,
                        vm_var,
                        &ivars,
                        &fvars,
                        &slot_vars,
                        &global_vars,
                        &deopt_refs,
                        &kinds,
                        &plan.inits_at[&ip],
                        ip,
                    );

                    b.switch_to_block(ok_blk);
                    let q = b.ins().fdiv(dividend, divisor);
                    kinds.pop();
                    kinds.pop();
                    let ix = kinds.len();
                    b.def_var(fvars[ix], q);
                    kinds.push(Kind::Float);
                }
                // Power: always Float, via the `powf` libcall.
                Op::Pow => {
                    let ky = kinds.pop().unwrap();
                    let iy = kinds.len();
                    let kx = kinds.pop().unwrap();
                    let ix = kinds.len();
                    let y = load_f64(&mut b, &ivars, &fvars, iy, ky);
                    let x = load_f64(&mut b, &ivars, &fvars, ix, kx);
                    let call = b.ins().call(powf_ref, &[x, y]);
                    let r = b.inst_results(call)[0];
                    b.def_var(fvars[ix], r);
                    kinds.push(Kind::Float);
                }
                // Unary float math: single Cranelift instructions where they
                // exist (`RoundFloat`'s round-ties-even is `nearest`).
                op @ (Op::AbsFloat
                | Op::SqrtFloat
                | Op::CeilFloat
                | Op::FloorFloat
                | Op::TruncFloat
                | Op::RoundFloat) => {
                    let ka = kinds.pop().unwrap();
                    let idx = kinds.len();
                    let x = load_f64(&mut b, &ivars, &fvars, idx, ka);
                    let r = match op {
                        Op::AbsFloat => b.ins().fabs(x),
                        Op::SqrtFloat => b.ins().sqrt(x),
                        Op::CeilFloat => b.ins().ceil(x),
                        Op::FloorFloat => b.ins().floor(x),
                        Op::TruncFloat => b.ins().trunc(x),
                        _ => b.ins().nearest(x), // RoundFloat (ties to even)
                    };
                    b.def_var(fvars[idx], r);
                    kinds.push(Kind::Float);
                }
                // Unary transcendentals: one shared libcall, keyed by id.
                op @ (Op::SinFloat
                | Op::CosFloat
                | Op::TanFloat
                | Op::AsinFloat
                | Op::AcosFloat
                | Op::AtanFloat
                | Op::SinhFloat
                | Op::CoshFloat
                | Op::TanhFloat
                | Op::ExpFloat
                | Op::LogFloat
                | Op::Log2Float
                | Op::Log10Float) => {
                    let id: i64 = match op {
                        Op::SinFloat => 0,
                        Op::CosFloat => 1,
                        Op::TanFloat => 2,
                        Op::AsinFloat => 3,
                        Op::AcosFloat => 4,
                        Op::AtanFloat => 5,
                        Op::SinhFloat => 6,
                        Op::CoshFloat => 7,
                        Op::TanhFloat => 8,
                        Op::ExpFloat => 9,
                        Op::LogFloat => 10,
                        Op::Log2Float => 11,
                        _ => 12, // Log10Float
                    };
                    let ka = kinds.pop().unwrap();
                    let idx = kinds.len();
                    let x = load_f64(&mut b, &ivars, &fvars, idx, ka);
                    let idc = b.ins().iconst(types::I32, id);
                    let call = b.ins().call(unary_ref, &[idc, x]);
                    let r = b.inst_results(call)[0];
                    b.def_var(fvars[idx], r);
                    kinds.push(Kind::Float);
                }
                // `to_int`: Float truncates (saturating, like `f as i64`); an
                // int-like value passes through unchanged.
                Op::TruncInt => {
                    let ka = kinds.pop().unwrap();
                    let idx = kinds.len();
                    let r = if ka == Kind::Float {
                        let f = b.use_var(fvars[idx]);
                        b.ins().fcvt_to_sint_sat(types::I64, f)
                    } else {
                        b.use_var(ivars[idx])
                    };
                    b.def_var(ivars[idx], r);
                    kinds.push(Kind::Int);
                }
                // Integer absolute value, wrapping (`abs(i64::MIN) == i64::MIN`).
                Op::AbsInt => {
                    let _ = kinds.pop().unwrap();
                    let idx = kinds.len();
                    let x = b.use_var(ivars[idx]);
                    let neg = b.ins().ineg(x);
                    let is_neg = b.ins().icmp_imm(IntCC::SignedLessThan, x, 0);
                    let r = b.ins().select(is_neg, neg, x);
                    b.def_var(ivars[idx], r);
                    kinds.push(Kind::Int);
                }
                // Binary float math libcalls. `a` is the lower operand, `b` the
                // top — matching the interpreter's pop order.
                op @ (Op::PowFloat | Op::Atan2Float) => {
                    let kb = kinds.pop().unwrap();
                    let iy = kinds.len();
                    let ka = kinds.pop().unwrap();
                    let ix = kinds.len();
                    let top = load_f64(&mut b, &ivars, &fvars, iy, kb);
                    let low = load_f64(&mut b, &ivars, &fvars, ix, ka);
                    let call = match op {
                        // a.powf(b): base=low, exponent=top
                        Op::PowFloat => b.ins().call(powf_ref, &[low, top]),
                        // y.atan2(x): pops x (top) then y (low) → atan2(y, x)
                        _ => b.ins().call(atan2_ref, &[low, top]),
                    };
                    let r = b.inst_results(call)[0];
                    b.def_var(fvars[ix], r);
                    kinds.push(Kind::Float);
                }
                // gcd via an internal Euclid loop on the magnitudes (`urem`,
                // guarded by the `y != 0` loop condition). Result is `x` as i64.
                Op::GcdInt => {
                    let _ = kinds.pop().unwrap();
                    let iy = kinds.len();
                    let _ = kinds.pop().unwrap();
                    let ix = kinds.len();
                    let av = b.use_var(ivars[ix]);
                    let a0 = uabs(&mut b, av);
                    let bv = b.use_var(ivars[iy]);
                    let b0 = uabs(&mut b, bv);
                    let xv = b.declare_var(types::I64);
                    let yv = b.declare_var(types::I64);
                    b.def_var(xv, a0);
                    b.def_var(yv, b0);
                    let header = b.create_block();
                    let body = b.create_block();
                    let exit = b.create_block();
                    b.ins().jump(header, &[]);
                    b.switch_to_block(header);
                    let y = b.use_var(yv);
                    let cont = b.ins().icmp_imm(IntCC::NotEqual, y, 0);
                    b.ins().brif(cont, body, &[], exit, &[]);
                    b.switch_to_block(body);
                    let x = b.use_var(xv);
                    let y2 = b.use_var(yv);
                    let t = b.ins().urem(x, y2);
                    b.def_var(xv, y2); // x = y
                    b.def_var(yv, t); // y = x % y
                    b.ins().jump(header, &[]);
                    b.switch_to_block(exit);
                    let g = b.use_var(xv);
                    b.def_var(ivars[ix], g);
                    kinds.push(Kind::Int);
                }
                // lcm = (a/gcd).saturating_mul(b), capped at i64::MAX; 0 if either
                // operand is 0. Euclid loop plus a saturating u64 multiply.
                Op::LcmInt => {
                    let _ = kinds.pop().unwrap();
                    let iy = kinds.len();
                    let _ = kinds.pop().unwrap();
                    let ix = kinds.len();
                    let av = b.use_var(ivars[ix]);
                    let a0 = uabs(&mut b, av);
                    let bv = b.use_var(ivars[iy]);
                    let b0 = uabs(&mut b, bv);
                    let rv = b.declare_var(types::I64);
                    let a_is0 = b.ins().icmp_imm(IntCC::Equal, a0, 0);
                    let b_is0 = b.ins().icmp_imm(IntCC::Equal, b0, 0);
                    let any0 = b.ins().bor(a_is0, b_is0);
                    let zero_b = b.create_block();
                    let comp_b = b.create_block();
                    let done_b = b.create_block();
                    b.ins().brif(any0, zero_b, &[], comp_b, &[]);

                    b.switch_to_block(zero_b);
                    let z = b.ins().iconst(types::I64, 0);
                    b.def_var(rv, z);
                    b.ins().jump(done_b, &[]);

                    b.switch_to_block(comp_b);
                    let xv = b.declare_var(types::I64);
                    let yv = b.declare_var(types::I64);
                    b.def_var(xv, a0);
                    b.def_var(yv, b0);
                    let gh = b.create_block();
                    let gb = b.create_block();
                    let ge = b.create_block();
                    b.ins().jump(gh, &[]);
                    b.switch_to_block(gh);
                    let y = b.use_var(yv);
                    let cont = b.ins().icmp_imm(IntCC::NotEqual, y, 0);
                    b.ins().brif(cont, gb, &[], ge, &[]);
                    b.switch_to_block(gb);
                    let x = b.use_var(xv);
                    let y2 = b.use_var(yv);
                    let t = b.ins().urem(x, y2);
                    b.def_var(xv, y2);
                    b.def_var(yv, t);
                    b.ins().jump(gh, &[]);
                    b.switch_to_block(ge);
                    let g = b.use_var(xv);
                    let p = b.ins().udiv(a0, g); // a / gcd
                    let prod = b.ins().imul(p, b0); // low 64 bits of p*b
                    let hi = b.ins().umulhi(p, b0); // high bits ⇒ overflow if != 0
                    let ovf = b.ins().icmp_imm(IntCC::NotEqual, hi, 0);
                    let umax = b.ins().iconst(types::I64, -1); // u64::MAX bits
                    let sat = b.ins().select(ovf, umax, prod);
                    let imax = b.ins().iconst(types::I64, i64::MAX);
                    let lt = b.ins().icmp(IntCC::UnsignedLessThan, sat, imax);
                    let res = b.ins().select(lt, sat, imax); // min(sat, i64::MAX)
                    b.def_var(rv, res);
                    b.ins().jump(done_b, &[]);

                    b.switch_to_block(done_b);
                    let r = b.use_var(rv);
                    b.def_var(ivars[ix], r);
                    kinds.push(Kind::Int);
                }
                // awk `/` and `%`: float result, with a runtime error on a zero
                // divisor (the native code branches to set the error and returns
                // early; the ok path continues). Codes match `VM::aot_set_error`.
                op @ (Op::AwkDivJit | Op::AwkModJit | Op::AwkDiv | Op::AwkMod) => {
                    let is_div = matches!(op, Op::AwkDivJit | Op::AwkDiv);
                    let kb = kinds.pop().unwrap();
                    let iy = kinds.len();
                    let ka = kinds.pop().unwrap();
                    let ix = kinds.len();
                    let divisor = load_f64(&mut b, &ivars, &fvars, iy, kb);
                    let dividend = load_f64(&mut b, &ivars, &fvars, ix, ka);
                    let zero = b.ins().f64const(Ieee64::with_bits(0f64.to_bits()));
                    let is_zero = b.ins().fcmp(FloatCC::Equal, divisor, zero);
                    let err_blk = b.create_block();
                    let ok_blk = b.create_block();
                    b.ins().brif(is_zero, err_blk, &[], ok_blk, &[]);

                    b.switch_to_block(err_blk);
                    let vm = b.use_var(vm_var);
                    // code 0 = "division by zero attempted", 1 = the `%' message.
                    let code = b.ins().iconst(types::I32, if is_div { 0 } else { 1 });
                    b.ins().call(serr_ref, &[vm, code]);
                    let status = b.ins().iconst(types::I64, 0);
                    b.ins().return_(&[status]);

                    b.switch_to_block(ok_blk);
                    let r = if is_div {
                        b.ins().fdiv(dividend, divisor)
                    } else {
                        let call = b.ins().call(fmod_ref, &[dividend, divisor]);
                        b.inst_results(call)[0]
                    };
                    b.def_var(fvars[ix], r);
                    kinds.push(Kind::Float);
                }
                // awk lshift/rshift: error on negative operands, else convert to
                // int, shift (logical for rshift) with `(n as u32) & 0x3f`, and
                // convert back to float — mirroring the interpreter exactly.
                op @ (Op::AwkLshiftJit | Op::AwkRshiftJit) => {
                    let kb = kinds.pop().unwrap();
                    let iy = kinds.len();
                    let ka = kinds.pop().unwrap();
                    let ix = kinds.len();
                    let n = load_f64(&mut b, &ivars, &fvars, iy, kb);
                    let a = load_f64(&mut b, &ivars, &fvars, ix, ka);
                    let zero = b.ins().f64const(Ieee64::with_bits(0f64.to_bits()));
                    let a_neg = b.ins().fcmp(FloatCC::LessThan, a, zero);
                    let n_neg = b.ins().fcmp(FloatCC::LessThan, n, zero);
                    let bad = b.ins().bor(a_neg, n_neg);
                    let err_blk = b.create_block();
                    let ok_blk = b.create_block();
                    b.ins().brif(bad, err_blk, &[], ok_blk, &[]);

                    b.switch_to_block(err_blk);
                    let vm = b.use_var(vm_var);
                    let code = b
                        .ins()
                        .iconst(types::I32, if matches!(op, Op::AwkLshiftJit) { 2 } else { 3 });
                    b.ins().call(serr_ref, &[vm, code]);
                    let st0 = b.ins().iconst(types::I64, 0);
                    b.ins().return_(&[st0]);

                    b.switch_to_block(ok_blk);
                    let ai = b.ins().fcvt_to_sint_sat(types::I64, a);
                    let shift_u32 = b.ins().fcvt_to_uint_sat(types::I32, n);
                    let shift_u64 = b.ins().uextend(types::I64, shift_u32);
                    let shift = b.ins().band_imm(shift_u64, 0x3f);
                    let r = if matches!(op, Op::AwkLshiftJit) {
                        let sh = b.ins().ishl(ai, shift);
                        b.ins().fcvt_from_sint(types::F64, sh)
                    } else {
                        // logical right shift on the u64 bit pattern → f64
                        let sh = b.ins().ushr(ai, shift);
                        b.ins().fcvt_from_uint(types::F64, sh)
                    };
                    b.def_var(fvars[ix], r);
                    kinds.push(Kind::Float);
                }
                // awk compl: error on negative, else `!(a as i64)` as float.
                Op::AwkComplJit => {
                    let ka = kinds.pop().unwrap();
                    let idx = kinds.len();
                    let a = load_f64(&mut b, &ivars, &fvars, idx, ka);
                    let zero = b.ins().f64const(Ieee64::with_bits(0f64.to_bits()));
                    let a_neg = b.ins().fcmp(FloatCC::LessThan, a, zero);
                    let err_blk = b.create_block();
                    let ok_blk = b.create_block();
                    b.ins().brif(a_neg, err_blk, &[], ok_blk, &[]);

                    b.switch_to_block(err_blk);
                    let vm = b.use_var(vm_var);
                    let code = b.ins().iconst(types::I32, 4);
                    b.ins().call(serr_ref, &[vm, code]);
                    let st0 = b.ins().iconst(types::I64, 0);
                    b.ins().return_(&[st0]);

                    b.switch_to_block(ok_blk);
                    let ai = b.ins().fcvt_to_sint_sat(types::I64, a);
                    let v = b.ins().bnot(ai);
                    let r = b.ins().fcvt_from_sint(types::F64, v);
                    b.def_var(fvars[idx], r);
                    kinds.push(Kind::Float);
                }
                // awk sqrt/log: warn-to-stderr + NaN on negative input (no halt),
                // else the result. The two arms merge their value in fvars[idx].
                op @ (Op::AwkSqrtJit | Op::AwkLogJit) => {
                    let ka = kinds.pop().unwrap();
                    let idx = kinds.len();
                    let a = load_f64(&mut b, &ivars, &fvars, idx, ka);
                    let zero = b.ins().f64const(Ieee64::with_bits(0f64.to_bits()));
                    let neg = b.ins().fcmp(FloatCC::LessThan, a, zero);
                    let warn_blk = b.create_block();
                    let ok_blk = b.create_block();
                    let cont_blk = b.create_block();
                    b.ins().brif(neg, warn_blk, &[], ok_blk, &[]);

                    b.switch_to_block(warn_blk);
                    let code = b
                        .ins()
                        .iconst(types::I32, if matches!(op, Op::AwkSqrtJit) { 0 } else { 1 });
                    b.ins().call(warn_ref, &[code, a]);
                    let nan = b.ins().f64const(Ieee64::with_bits(f64::NAN.to_bits()));
                    b.def_var(fvars[idx], nan);
                    b.ins().jump(cont_blk, &[]);

                    b.switch_to_block(ok_blk);
                    let r = if matches!(op, Op::AwkSqrtJit) {
                        b.ins().sqrt(a)
                    } else {
                        let ln_id = b.ins().iconst(types::I32, 10); // ln
                        let call = b.ins().call(unary_ref, &[ln_id, a]);
                        b.inst_results(call)[0]
                    };
                    b.def_var(fvars[idx], r);
                    b.ins().jump(cont_blk, &[]);

                    b.switch_to_block(cont_blk);
                    kinds.push(Kind::Float);
                }
                // Sink ops: spill the top n scalar registers onto the boxed
                // stack (bottom-most first, matching the interpreter's order),
                // then run the op through the shim. vm.stack is empty before and
                // after (the op pops exactly what we pushed). `SetStatus` is the
                // n=1 silent case.
                Op::Print(_) | Op::PrintLn(_) | Op::SetStatus => {
                    let n = match &chunk.ops[ip] {
                        Op::Print(c) | Op::PrintLn(c) => *c as usize,
                        _ => 1, // SetStatus
                    };
                    let base = kinds.len() - n;
                    for pos in base..kinds.len() {
                        let vm = b.use_var(vm_var);
                        match kinds[pos] {
                            Kind::Float => {
                                let v = b.use_var(fvars[pos]);
                                b.ins().call(pushf_ref, &[vm, v]);
                            }
                            Kind::Bool => {
                                let v = b.use_var(ivars[pos]);
                                b.ins().call(pushb_ref, &[vm, v]);
                            }
                            Kind::Status => {
                                let v = b.use_var(ivars[pos]);
                                b.ins().call(push_status_ref, &[vm, v]);
                            }
                            Kind::Int => {
                                let v = b.use_var(ivars[pos]);
                                b.ins().call(pushi_ref, &[vm, v]);
                            }
                            Kind::Obj => {
                                let v = b.use_var(ivars[pos]);
                                b.ins().call(unbox_ref, &[vm, v]);
                            }
                        }
                    }
                    let vm = b.use_var(vm_var);
                    let ipc = b.ins().iconst(types::I64, ip as i64);
                    // Straight-line sink: the returned next-ip is ignored.
                    b.ins().call(exec_ref, &[vm, ipc]);
                    kinds.truncate(base);
                }
                // Source op: run via the shim (it pushes a Float onto vm.stack),
                // then reload that Float into a register. Statically Float, so no
                // guard. vm.stack is empty before and after.
                Op::AwkGetFieldNum(_) => {
                    let vm = b.use_var(vm_var);
                    let ipc = b.ins().iconst(types::I64, ip as i64);
                    b.ins().call(exec_ref, &[vm, ipc]);
                    let vm2 = b.use_var(vm_var);
                    let call = b.ins().call(popf_ref, &[vm2]);
                    let v = b.inst_results(call)[0];
                    let idx = kinds.len();
                    b.def_var(fvars[idx], v);
                    kinds.push(Kind::Float);
                }
                // `$?` source: run via the shim (it pushes a Status), reload the
                // code into a register. Statically Status, so no guard.
                Op::GetStatus => {
                    let vm = b.use_var(vm_var);
                    let ipc = b.ins().iconst(types::I64, ip as i64);
                    b.ins().call(exec_ref, &[vm, ipc]);
                    let vm2 = b.use_var(vm_var);
                    let call = b.ins().call(popi_ref, &[vm2]);
                    let v = b.inst_results(call)[0];
                    let idx = kinds.len();
                    b.def_var(ivars[idx], v);
                    kinds.push(Kind::Status);
                }
                Op::Negate => {
                    let k = kinds.pop().unwrap();
                    let idx = kinds.len();
                    if k == Kind::Int {
                        let x = b.use_var(ivars[idx]);
                        let r = b.ins().ineg(x);
                        b.def_var(ivars[idx], r);
                        kinds.push(Kind::Int);
                    } else {
                        // Float or Status → `-to_float`, yielding Float.
                        let x = load_f64(&mut b, &ivars, &fvars, idx, k);
                        let r = b.ins().fneg(x);
                        b.def_var(fvars[idx], r);
                        kinds.push(Kind::Float);
                    }
                }
                op @ (Op::NumEq | Op::NumNe | Op::NumLt | Op::NumGt | Op::NumLe | Op::NumGe) => {
                    let ky = kinds.pop().unwrap();
                    let iy = kinds.len();
                    let kx = kinds.pop().unwrap();
                    let ix = kinds.len();
                    let pred = if ky == Kind::Float || kx == Kind::Float {
                        let y = load_f64(&mut b, &ivars, &fvars, iy, ky);
                        let x = load_f64(&mut b, &ivars, &fvars, ix, kx);
                        let cc = match op {
                            Op::NumEq => FloatCC::Equal,
                            Op::NumNe => FloatCC::NotEqual,
                            Op::NumLt => FloatCC::LessThan,
                            Op::NumGt => FloatCC::GreaterThan,
                            Op::NumLe => FloatCC::LessThanOrEqual,
                            _ => FloatCC::GreaterThanOrEqual,
                        };
                        b.ins().fcmp(cc, x, y)
                    } else {
                        let y = b.use_var(ivars[iy]);
                        let x = b.use_var(ivars[ix]);
                        let cc = match op {
                            Op::NumEq => IntCC::Equal,
                            Op::NumNe => IntCC::NotEqual,
                            Op::NumLt => IntCC::SignedLessThan,
                            Op::NumGt => IntCC::SignedGreaterThan,
                            Op::NumLe => IntCC::SignedLessThanOrEqual,
                            _ => IntCC::SignedGreaterThanOrEqual,
                        };
                        b.ins().icmp(cc, x, y)
                    };
                    // Normalize the i8 predicate to i64 0/1 (the interpreter
                    // pushes a `Bool`, carried here as an integer 0/1).
                    let one = b.ins().iconst(types::I64, 1);
                    let zero = b.ins().iconst(types::I64, 0);
                    let r = b.ins().select(pred, one, zero);
                    b.def_var(ivars[ix], r);
                    kinds.push(Kind::Bool);
                }
                // Slots are register-resident, typed Int or Float per the plan.
                Op::GetSlot(slot) => {
                    let k = var_ty_kind(plan, *slot as u32);
                    let v = b.use_var(slot_vars[*slot as usize]);
                    let idx = kinds.len();
                    if k == Kind::Obj {
                        // Reading clones the value into a fresh owned handle; the
                        // slot keeps its own handle.
                        let vm = b.use_var(vm_var);
                        let call = b.ins().call(clone_ref, &[vm, v]);
                        b.def_var(ivars[idx], b.inst_results(call)[0]);
                    } else {
                        store_raw(&mut b, &ivars, &fvars, idx, k, v);
                    }
                    kinds.push(k);
                }
                Op::SetSlot(slot) => {
                    let k = kinds.pop().unwrap(); // == the slot's kind
                    let idx = kinds.len();
                    let v = load_raw(&mut b, &ivars, &fvars, idx, k);
                    let sv = slot_vars[*slot as usize];
                    if k == Kind::Obj {
                        // Free the slot's previous handle (init sentinel -1 ⇒
                        // no-op) before it is overwritten.
                        let old = b.use_var(sv);
                        let vm = b.use_var(vm_var);
                        b.ins().call(free_ref, &[vm, old]);
                    }
                    b.def_var(sv, v);
                }
                // Globals mirror slots (Int/Float/Obj register variables).
                Op::GetVar(g) => {
                    let k = var_ty_kind(plan, *g as u32 | GLOBAL_TAG);
                    let v = b.use_var(global_vars[*g as usize]);
                    let idx = kinds.len();
                    if k == Kind::Obj {
                        let vm = b.use_var(vm_var);
                        let call = b.ins().call(clone_ref, &[vm, v]);
                        b.def_var(ivars[idx], b.inst_results(call)[0]);
                    } else {
                        store_raw(&mut b, &ivars, &fvars, idx, k, v);
                    }
                    kinds.push(k);
                }
                Op::SetVar(g) | Op::DeclareVar(g) => {
                    let k = kinds.pop().unwrap();
                    let idx = kinds.len();
                    let v = load_raw(&mut b, &ivars, &fvars, idx, k);
                    let gv = global_vars[*g as usize];
                    if k == Kind::Obj {
                        let old = b.use_var(gv);
                        let vm = b.use_var(vm_var);
                        b.ins().call(free_ref, &[vm, old]);
                    }
                    b.def_var(gv, v);
                }
                // Slot read-modify-write super-ops (slots are i64 registers).
                op @ (Op::PreIncSlot(slot) | Op::PreDecSlot(slot)) => {
                    let sv = slot_vars[*slot as usize];
                    let old = b.use_var(sv);
                    let nv = b
                        .ins()
                        .iadd_imm(old, if matches!(op, Op::PreIncSlot(_)) { 1 } else { -1 });
                    b.def_var(sv, nv);
                    let idx = kinds.len();
                    b.def_var(ivars[idx], nv); // pre: push the new value
                    kinds.push(Kind::Int);
                }
                op @ (Op::PostIncSlot(slot) | Op::PostDecSlot(slot)) => {
                    let sv = slot_vars[*slot as usize];
                    let old = b.use_var(sv);
                    let nv = b
                        .ins()
                        .iadd_imm(old, if matches!(op, Op::PostIncSlot(_)) { 1 } else { -1 });
                    b.def_var(sv, nv);
                    let idx = kinds.len();
                    b.def_var(ivars[idx], old); // post: push the old value
                    kinds.push(Kind::Int);
                }
                Op::PreIncSlotVoid(slot) => {
                    let sv = slot_vars[*slot as usize];
                    let old = b.use_var(sv);
                    let nv = b.ins().iadd_imm(old, 1);
                    b.def_var(sv, nv);
                }
                Op::AddAssignSlotVoid(a, b_slot) => {
                    let av = slot_vars[*a as usize];
                    let x = b.use_var(av);
                    let y = b.use_var(slot_vars[*b_slot as usize]);
                    let sum = b.ins().iadd(x, y);
                    b.def_var(av, sum);
                }
                // Emit the internal `while i < limit { sum += i; i += 1 }` as a
                // real native loop over the two slot registers. Three on-the-fly
                // blocks (not plan leaders); `seal_all_blocks` builds the phis.
                // Execution continues in `exit` after this op.
                Op::AccumSumLoop(sum_s, i_s, limit) => {
                    let sum_v = slot_vars[*sum_s as usize];
                    let i_v = slot_vars[*i_s as usize];
                    let header = b.create_block();
                    let body = b.create_block();
                    let exit = b.create_block();
                    b.ins().jump(header, &[]);

                    b.switch_to_block(header);
                    let lim = b.ins().iconst(types::I64, *limit as i64);
                    let i_cur = b.use_var(i_v);
                    let cond = b.ins().icmp(IntCC::SignedLessThan, i_cur, lim);
                    b.ins().brif(cond, body, &[], exit, &[]);

                    b.switch_to_block(body);
                    let s = b.use_var(sum_v);
                    let i2 = b.use_var(i_v);
                    let ns = b.ins().iadd(s, i2);
                    b.def_var(sum_v, ns);
                    let ni = b.ins().iadd_imm(i2, 1);
                    b.def_var(i_v, ni);
                    b.ins().jump(header, &[]);

                    // Fall through into `exit`; subsequent ops emit here.
                    b.switch_to_block(exit);
                }
                Op::SlotLtIntJumpIfFalse(slot, limit, t) => {
                    let v = b.use_var(slot_vars[*slot as usize]);
                    // jump to target when slot >= limit, i.e. NOT (slot < limit).
                    let lt = b.ins().icmp_imm(IntCC::SignedLessThan, v, *limit as i64);
                    b.ins()
                        .brif(lt, block_for(ip + 1), &[], block_for(*t), &[]);
                    terminated = true;
                }
                Op::SlotIncLtIntJumpBack(slot, limit, t) => {
                    let sv = slot_vars[*slot as usize];
                    let old = b.use_var(sv);
                    let nv = b.ins().iadd_imm(old, 1);
                    b.def_var(sv, nv);
                    // jump back to target while the incremented slot < limit.
                    let lt = b.ins().icmp_imm(IntCC::SignedLessThan, nv, *limit as i64);
                    b.ins()
                        .brif(lt, block_for(*t), &[], block_for(ip + 1), &[]);
                    terminated = true;
                }
                Op::Pop => {
                    let k = kinds.pop().unwrap();
                    if k == Kind::Obj {
                        // Discarding a boxed value frees its handle.
                        let h = b.use_var(ivars[kinds.len()]);
                        let vm = b.use_var(vm_var);
                        b.ins().call(free_ref, &[vm, h]);
                    }
                }
                Op::Dup => {
                    let k = *kinds.last().unwrap();
                    let src = kinds.len() - 1;
                    let dst = kinds.len();
                    if k == Kind::Float {
                        let v = b.use_var(fvars[src]);
                        b.def_var(fvars[dst], v);
                    } else {
                        let v = b.use_var(ivars[src]);
                        b.def_var(ivars[dst], v);
                    }
                    kinds.push(k);
                }
                Op::LoadTrue | Op::LoadFalse => {
                    let v = b
                        .ins()
                        .iconst(types::I64, matches!(&chunk.ops[ip], Op::LoadTrue) as i64);
                    let idx = kinds.len();
                    b.def_var(ivars[idx], v);
                    kinds.push(Kind::Bool);
                }
                op @ (Op::Inc | Op::Dec) => {
                    let _ = kinds.pop().unwrap(); // int-like ⇒ value lives in ivars
                    let idx = kinds.len();
                    let x = b.use_var(ivars[idx]);
                    let r = match op {
                        Op::Inc => b.ins().iadd_imm(x, 1),
                        _ => b.ins().iadd_imm(x, -1),
                    };
                    b.def_var(ivars[idx], r);
                    kinds.push(Kind::Int);
                }
                Op::LogNot => {
                    let k = kinds.pop().unwrap();
                    let idx = kinds.len();
                    let pred = truthy(&mut b, &ivars, &fvars, idx, k);
                    // !truthy: truthy ⇒ 0, falsy ⇒ 1.
                    let one = b.ins().iconst(types::I64, 1);
                    let zero = b.ins().iconst(types::I64, 0);
                    let r = b.ins().select(pred, zero, one);
                    b.def_var(ivars[idx], r);
                    kinds.push(Kind::Bool);
                }
                op @ (Op::BitAnd | Op::BitOr | Op::BitXor | Op::Shl | Op::Shr) => {
                    let _ = kinds.pop().unwrap();
                    let iy = kinds.len();
                    let _ = kinds.pop().unwrap();
                    let ix = kinds.len();
                    let y = b.use_var(ivars[iy]);
                    let x = b.use_var(ivars[ix]);
                    // Cranelift masks the shift amount to the operand width
                    // (0–63 for i64), matching the interpreter's `& 63`.
                    let r = match op {
                        Op::BitAnd => b.ins().band(x, y),
                        Op::BitOr => b.ins().bor(x, y),
                        Op::BitXor => b.ins().bxor(x, y),
                        Op::Shl => b.ins().ishl(x, y),
                        _ => b.ins().sshr(x, y),
                    };
                    b.def_var(ivars[ix], r);
                    kinds.push(Kind::Int);
                }
                Op::BitNot => {
                    let _ = kinds.pop().unwrap();
                    let idx = kinds.len();
                    let x = b.use_var(ivars[idx]);
                    let r = b.ins().bnot(x);
                    b.def_var(ivars[idx], r);
                    kinds.push(Kind::Int);
                }
                Op::Dup2 => {
                    let len = kinds.len();
                    let ka = kinds[len - 2];
                    let kb = kinds[len - 1];
                    let a = load_raw(&mut b, &ivars, &fvars, len - 2, ka);
                    let bv = load_raw(&mut b, &ivars, &fvars, len - 1, kb);
                    store_raw(&mut b, &ivars, &fvars, len, ka, a);
                    store_raw(&mut b, &ivars, &fvars, len + 1, kb, bv);
                    kinds.push(ka);
                    kinds.push(kb);
                }
                Op::Swap => {
                    let len = kinds.len();
                    let (ix, iy) = (len - 2, len - 1);
                    let kx = kinds[ix];
                    let ky = kinds[iy];
                    let x = load_raw(&mut b, &ivars, &fvars, ix, kx);
                    let y = load_raw(&mut b, &ivars, &fvars, iy, ky);
                    store_raw(&mut b, &ivars, &fvars, ix, ky, y);
                    store_raw(&mut b, &ivars, &fvars, iy, kx, x);
                    kinds.swap(ix, iy);
                }
                Op::Rot => {
                    let len = kinds.len();
                    let (ia, ib, ic) = (len - 3, len - 2, len - 1);
                    let ka = kinds[ia];
                    let kb = kinds[ib];
                    let kc = kinds[ic];
                    let a = load_raw(&mut b, &ivars, &fvars, ia, ka);
                    let bv = load_raw(&mut b, &ivars, &fvars, ib, kb);
                    let c = load_raw(&mut b, &ivars, &fvars, ic, kc);
                    // [a, b, c] → [b, c, a]
                    store_raw(&mut b, &ivars, &fvars, ia, kb, bv);
                    store_raw(&mut b, &ivars, &fvars, ib, kc, c);
                    store_raw(&mut b, &ivars, &fvars, ic, ka, a);
                    kinds[ia] = kb;
                    kinds[ib] = kc;
                    kinds[ic] = ka;
                }
                op @ (Op::LogAnd | Op::LogOr) => {
                    let kb = kinds.pop().unwrap();
                    let iy = kinds.len();
                    let ka = kinds.pop().unwrap();
                    let ix = kinds.len();
                    let ta = truthy(&mut b, &ivars, &fvars, ix, ka);
                    let tb = truthy(&mut b, &ivars, &fvars, iy, kb);
                    let combined = match op {
                        Op::LogAnd => b.ins().band(ta, tb),
                        _ => b.ins().bor(ta, tb),
                    };
                    let one = b.ins().iconst(types::I64, 1);
                    let zero = b.ins().iconst(types::I64, 0);
                    let r = b.ins().select(combined, one, zero);
                    b.def_var(ivars[ix], r);
                    kinds.push(Kind::Bool);
                }
                Op::Spaceship => {
                    let kb = kinds.pop().unwrap();
                    let iy = kinds.len();
                    let ka = kinds.pop().unwrap();
                    let ix = kinds.len();
                    let m1 = b.ins().iconst(types::I64, -1);
                    let zero = b.ins().iconst(types::I64, 0);
                    let p1 = b.ins().iconst(types::I64, 1);
                    let (lt, gt) = if ka.promotes_float() || kb.promotes_float() {
                        let y = load_f64(&mut b, &ivars, &fvars, iy, kb);
                        let x = load_f64(&mut b, &ivars, &fvars, ix, ka);
                        (
                            b.ins().fcmp(FloatCC::LessThan, x, y),
                            b.ins().fcmp(FloatCC::GreaterThan, x, y),
                        )
                    } else {
                        let y = b.use_var(ivars[iy]);
                        let x = b.use_var(ivars[ix]);
                        (
                            b.ins().icmp(IntCC::SignedLessThan, x, y),
                            b.ins().icmp(IntCC::SignedGreaterThan, x, y),
                        )
                    };
                    // x<y ? -1 : (x>y ? 1 : 0)
                    let mid = b.ins().select(gt, p1, zero);
                    let r = b.ins().select(lt, m1, mid);
                    b.def_var(ivars[ix], r);
                    kinds.push(Kind::Int);
                }
                Op::JumpIfTrueKeep(t) => {
                    // Peek (don't pop): the value stays live on both arms.
                    let k = *kinds.last().unwrap();
                    let idx = kinds.len() - 1;
                    let cond = truthy(&mut b, &ivars, &fvars, idx, k);
                    b.ins()
                        .brif(cond, block_for(*t), &[], block_for(ip + 1), &[]);
                    terminated = true;
                }
                Op::JumpIfFalseKeep(t) => {
                    let k = *kinds.last().unwrap();
                    let idx = kinds.len() - 1;
                    let cond = truthy(&mut b, &ivars, &fvars, idx, k);
                    b.ins()
                        .brif(cond, block_for(ip + 1), &[], block_for(*t), &[]);
                    terminated = true;
                }
                Op::Jump(t) => {
                    b.ins().jump(block_for(*t), &[]);
                    terminated = true;
                }
                Op::JumpIfTrue(t) => {
                    let k = kinds.pop().unwrap();
                    let idx = kinds.len();
                    let cond = truthy(&mut b, &ivars, &fvars, idx, k);
                    b.ins()
                        .brif(cond, block_for(*t), &[], block_for(ip + 1), &[]);
                    terminated = true;
                }
                Op::JumpIfFalse(t) => {
                    let k = kinds.pop().unwrap();
                    let idx = kinds.len();
                    let cond = truthy(&mut b, &ivars, &fvars, idx, k);
                    // truthy ⇒ fallthrough; false ⇒ branch to the target.
                    b.ins()
                        .brif(cond, block_for(ip + 1), &[], block_for(*t), &[]);
                    terminated = true;
                }
                // Deopt points are handled before the match; any op reaching
                // here was admitted as lowerable by `analyze_native`.
                other => unreachable!("analyze_native admitted unsupported op {other:?}"),
            }
        }

        // Fall off the end of the last block into ret.
        if !terminated {
            b.ins().jump(ret_block, &[]);
        }

        // Ret: box the stack top as the result, mirroring the interpreter's
        // stack-tail rule (empty ⇒ `aot_result` stays unset ⇒ `Halted`). The
        // top kind is Int or Float (analysis rejects a Bool result).
        b.switch_to_block(ret_block);
        if let Some(&topk) = plan.end_kinds.last() {
            let idx = plan.end_kinds.len() - 1;
            let vm = b.use_var(vm_var);
            match topk {
                Kind::Float => {
                    let v = b.use_var(fvars[idx]);
                    b.ins().call(fres_ref, &[vm, v]);
                }
                Kind::Status => {
                    let v = b.use_var(ivars[idx]);
                    b.ins().call(sres_ref, &[vm, v]);
                }
                // Obj: the register holds an arena handle; box it as the result.
                Kind::Obj => {
                    let v = b.use_var(ivars[idx]);
                    b.ins().call(obj_res_ref, &[vm, v]);
                }
                // Int (Bool is rejected as a result by analysis).
                _ => {
                    let v = b.use_var(ivars[idx]);
                    b.ins().call(ires_ref, &[vm, v]);
                }
            }
        }
        let status = b.ins().iconst(types::I64, 0);
        b.ins().return_(&[status]);

        b.seal_all_blocks();
        b.finalize();
    }
    module
        .define_function(entry_id, &mut ctx)
        .map_err(|e| format!("aot: define native entry: {e}"))?;
    module.clear_context(&mut ctx);
    Ok(entry_id)
}

/// Read operand-stack position `idx` as an `f64`, promoting an integer with
/// `int→float` conversion exactly as the interpreter's `to_float` would. Never
/// called on a `Bool` (analysis rejects booleans feeding arithmetic).
fn load_f64(
    b: &mut FunctionBuilder,
    ivars: &[Variable],
    fvars: &[Variable],
    idx: usize,
    k: Kind,
) -> cranelift_codegen::ir::Value {
    if k == Kind::Float {
        b.use_var(fvars[idx])
    } else {
        let v = b.use_var(ivars[idx]);
        b.ins().fcvt_from_sint(types::F64, v)
    }
}

/// Read operand-stack position `idx` in its own representation (no coercion),
/// picking the i64 or f64 variable per its `Kind`. Used by the stack-shuffle
/// ops, which move values around unchanged.
fn load_raw(
    b: &mut FunctionBuilder,
    ivars: &[Variable],
    fvars: &[Variable],
    idx: usize,
    k: Kind,
) -> cranelift_codegen::ir::Value {
    if k == Kind::Float {
        b.use_var(fvars[idx])
    } else {
        b.use_var(ivars[idx])
    }
}

/// Unsigned absolute value of an `i64` (same bit pattern as `i64::unsigned_abs`,
/// including `i64::MIN` → `2^63`), as an i64 holding the u64 bits. Used by the
/// `GcdInt`/`LcmInt` Euclid loops, which operate on magnitudes.
fn uabs(b: &mut FunctionBuilder, v: cranelift_codegen::ir::Value) -> cranelift_codegen::ir::Value {
    let neg = b.ins().ineg(v);
    let is_neg = b.ins().icmp_imm(IntCC::SignedLessThan, v, 0);
    b.ins().select(is_neg, neg, v)
}

/// Write `v` to operand-stack position `idx`, into the variable matching `k`.
fn store_raw(
    b: &mut FunctionBuilder,
    ivars: &[Variable],
    fvars: &[Variable],
    idx: usize,
    k: Kind,
    v: cranelift_codegen::ir::Value,
) {
    if k == Kind::Float {
        b.def_var(fvars[idx], v);
    } else {
        b.def_var(ivars[idx], v);
    }
}

/// Build an i8 truthiness predicate for the value at position `idx`, matching
/// [`crate::value::Value::is_truthy`]: integers/bools are nonzero-true; floats
/// use `!= 0.0` (unordered, so `NaN` is truthy).
fn truthy(
    b: &mut FunctionBuilder,
    ivars: &[Variable],
    fvars: &[Variable],
    idx: usize,
    k: Kind,
) -> cranelift_codegen::ir::Value {
    match k {
        Kind::Float => {
            let f = b.use_var(fvars[idx]);
            let z = b.ins().f64const(Ieee64::with_bits(0f64.to_bits()));
            b.ins().fcmp(FloatCC::NotEqual, f, z)
        }
        // Status is truthy when the code is ZERO (shell success) — inverted.
        Kind::Status => {
            let v = b.use_var(ivars[idx]);
            b.ins().icmp_imm(IntCC::Equal, v, 0)
        }
        // Int / Bool: nonzero is truthy.
        _ => {
            let v = b.use_var(ivars[idx]);
            b.ins().icmp_imm(IntCC::NotEqual, v, 0)
        }
    }
}

/// Threaded fallback codegen: one Cranelift block per op, each running the op's
/// semantics through the [`fusevm_aot_exec_op`] shim and re-dispatching on the
/// returned next-ip. See the module docs for the control-flow shape.
fn build_entry_threaded<M: Module>(module: &mut M, chunk: &Chunk) -> Result<FuncId, String> {
    let ptr_ty = module.target_config().pointer_type();

    // Imported runtime shims.
    let mut exec_sig = module.make_signature();
    exec_sig.params.push(AbiParam::new(ptr_ty));
    exec_sig.params.push(AbiParam::new(types::I64));
    exec_sig.returns.push(AbiParam::new(types::I64));
    let exec_id = module
        .declare_function("fusevm_aot_exec_op", Linkage::Import, &exec_sig)
        .map_err(|e| format!("aot: declare exec_op: {e}"))?;

    let mut fin_sig = module.make_signature();
    fin_sig.params.push(AbiParam::new(ptr_ty));
    let fin_id = module
        .declare_function("fusevm_aot_finish", Linkage::Import, &fin_sig)
        .map_err(|e| format!("aot: declare finish: {e}"))?;

    // Exported entry: fn(vm) -> i64.
    let mut entry_sig = module.make_signature();
    entry_sig.params.push(AbiParam::new(ptr_ty));
    entry_sig.returns.push(AbiParam::new(types::I64));
    let entry_id = module
        .declare_function(AOT_ENTRY_SYMBOL, Linkage::Export, &entry_sig)
        .map_err(|e| format!("aot: declare entry: {e}"))?;

    let mut ctx = module.make_context();
    ctx.func.signature = entry_sig;
    let mut fbctx = FunctionBuilderContext::new();
    {
        let mut b = FunctionBuilder::new(&mut ctx.func, &mut fbctx);
        let exec_ref = module.declare_func_in_func(exec_id, b.func);
        let fin_ref = module.declare_func_in_func(fin_id, b.func);

        let n = chunk.ops.len();
        let entry_block = b.create_block();
        b.append_block_params_for_function_params(entry_block);
        let dispatch_block = b.create_block();
        b.append_block_param(dispatch_block, types::I64); // current ip
        let ret_block = b.create_block();
        let op_blocks: Vec<_> = (0..n).map(|_| b.create_block()).collect();

        // `vm` pointer threaded to every block via a frontend Variable.
        let vm_var = b.declare_var(ptr_ty);

        // entry: stash vm, jump into dispatch at ip 0 (or straight to ret if empty).
        b.switch_to_block(entry_block);
        let vm_param = b.block_params(entry_block)[0];
        b.def_var(vm_var, vm_param);
        if n == 0 {
            b.ins().jump(ret_block, &[]);
        } else {
            let zero = b.ins().iconst(types::I64, 0);
            b.ins().jump(dispatch_block, &[BlockArg::Value(zero)]);
        }

        // dispatch: br_table on the ip param to the matching op block.
        b.switch_to_block(dispatch_block);
        let ip_val = b.block_params(dispatch_block)[0];
        if n == 0 {
            b.ins().jump(ret_block, &[]);
        } else {
            let mut switch = Switch::new();
            for (i, blk) in op_blocks.iter().enumerate() {
                switch.set_entry(i as u128, *blk);
            }
            switch.emit(&mut b, ip_val, ret_block);
        }

        // one block per op: run it, then terminate or re-dispatch on the result.
        for (i, blk) in op_blocks.iter().enumerate() {
            b.switch_to_block(*blk);
            let vm = b.use_var(vm_var);
            let ipc = b.ins().iconst(types::I64, i as i64);
            let call = b.ins().call(exec_ref, &[vm, ipc]);
            let next = b.inst_results(call)[0];
            let is_term = b.ins().icmp_imm(IntCC::SignedLessThan, next, 0);
            b.ins().brif(
                is_term,
                ret_block,
                &[],
                dispatch_block,
                &[BlockArg::Value(next)],
            );
        }

        // ret: compute the tail result and return.
        b.switch_to_block(ret_block);
        let vm = b.use_var(vm_var);
        b.ins().call(fin_ref, &[vm]);
        let status = b.ins().iconst(types::I64, 0);
        b.ins().return_(&[status]);

        b.seal_all_blocks();
        b.finalize();
    }
    module
        .define_function(entry_id, &mut ctx)
        .map_err(|e| format!("aot: define entry: {e}"))?;
    module.clear_context(&mut ctx);
    Ok(entry_id)
}

/// Compile `chunk` to native code in-process via Cranelift, run it on a fresh
/// VM, and return the result. The emitted native entry drives the whole program
/// (no interpreter dispatch loop). `register` installs any frontend builtins on
/// the VM before the run. This validates the closed-world compiler end to end
/// and is the in-memory analog of the on-disk object path.
pub fn run_chunk_native(chunk: &Chunk, register: impl FnOnce(&mut VM)) -> Result<VMResult, String> {
    let isa = host_isa()?;
    let mut builder = JITBuilder::with_isa(isa, default_libcall_names());
    builder.symbol("fusevm_aot_exec_op", fusevm_aot_exec_op as *const u8);
    builder.symbol("fusevm_aot_finish", fusevm_aot_finish as *const u8);
    builder.symbol(
        "fusevm_aot_set_int_result",
        fusevm_aot_set_int_result as *const u8,
    );
    builder.symbol(
        "fusevm_aot_set_float_result",
        fusevm_aot_set_float_result as *const u8,
    );
    builder.symbol("fusevm_aot_powf", fusevm_aot_powf as *const u8);
    builder.symbol("fusevm_aot_fmod", fusevm_aot_fmod as *const u8);
    builder.symbol("fusevm_aot_unary_math", fusevm_aot_unary_math as *const u8);
    builder.symbol("fusevm_aot_atan2", fusevm_aot_atan2 as *const u8);
    builder.symbol("fusevm_aot_set_error", fusevm_aot_set_error as *const u8);
    builder.symbol("fusevm_aot_awk_warn", fusevm_aot_awk_warn as *const u8);
    builder.symbol("fusevm_aot_push_int", fusevm_aot_push_int as *const u8);
    builder.symbol("fusevm_aot_push_float", fusevm_aot_push_float as *const u8);
    builder.symbol("fusevm_aot_push_bool", fusevm_aot_push_bool as *const u8);
    builder.symbol("fusevm_aot_pop_float", fusevm_aot_pop_float as *const u8);
    builder.symbol(
        "fusevm_aot_store_slot_int",
        fusevm_aot_store_slot_int as *const u8,
    );
    builder.symbol(
        "fusevm_aot_store_slot_float",
        fusevm_aot_store_slot_float as *const u8,
    );
    builder.symbol(
        "fusevm_aot_store_global_int",
        fusevm_aot_store_global_int as *const u8,
    );
    builder.symbol(
        "fusevm_aot_store_global_float",
        fusevm_aot_store_global_float as *const u8,
    );
    builder.symbol("fusevm_aot_resume", fusevm_aot_resume as *const u8);
    builder.symbol("fusevm_aot_pop_int", fusevm_aot_pop_int as *const u8);
    builder.symbol("fusevm_aot_push_status", fusevm_aot_push_status as *const u8);
    builder.symbol("fusevm_aot_box", fusevm_aot_box as *const u8);
    builder.symbol("fusevm_aot_unbox", fusevm_aot_unbox as *const u8);
    builder.symbol("fusevm_aot_clone", fusevm_aot_clone as *const u8);
    builder.symbol("fusevm_aot_free", fusevm_aot_free as *const u8);
    builder.symbol(
        "fusevm_aot_store_slot_obj",
        fusevm_aot_store_slot_obj as *const u8,
    );
    builder.symbol(
        "fusevm_aot_store_global_obj",
        fusevm_aot_store_global_obj as *const u8,
    );
    builder.symbol(
        "fusevm_aot_set_obj_result",
        fusevm_aot_set_obj_result as *const u8,
    );
    builder.symbol(
        "fusevm_aot_set_status_result",
        fusevm_aot_set_status_result as *const u8,
    );
    let mut module = JITModule::new(builder);

    let entry_id = build_entry(&mut module, chunk)?;
    module
        .finalize_definitions()
        .map_err(|e| format!("aot: finalize: {e}"))?;
    let code = module.get_finalized_function(entry_id);
    // SAFETY: `code` is the finalized entry with the declared C ABI.
    let entry: extern "C" fn(*mut VM) -> i64 = unsafe { std::mem::transmute(code) };

    let mut vm = VM::new(chunk.clone());
    register(&mut vm);
    let _ = entry(&mut vm as *mut VM);
    Ok(vm.take_aot_result())
}

/// Build a Cranelift ISA for emitting a relocatable object (`is_pic=true`).
fn object_isa() -> Result<cranelift_codegen::isa::OwnedTargetIsa, String> {
    let mut fb = settings::builder();
    let _ = fb.set("use_colocated_libcalls", "false");
    let _ = fb.set("is_pic", "true");
    let _ = fb.set("opt_level", "speed");
    let flags = settings::Flags::new(fb);
    let isa_builder = cranelift_native::builder().map_err(|e| format!("aot: native ISA: {e}"))?;
    isa_builder
        .finish(flags)
        .map_err(|e| format!("aot: ISA finish: {e}"))
}

/// Ahead-of-time compile `chunk` to a relocatable native object file at
/// `out_path`. The object exports:
///   - [`AOT_ENTRY_SYMBOL`] — the native driver `fn(*mut VM) -> i64`,
///   - [`AOT_CHUNK_BLOB_SYMBOL`] / [`AOT_CHUNK_LEN_SYMBOL`] — the serialized
///     chunk and its length, so the runtime can rebuild the VM that the driver
///     reads ops/constants from,
/// and imports the runtime shims (`fusevm_aot_exec_op`, `fusevm_aot_finish`) plus
/// the frontend hook `fusevm_aot_register_builtins`, all resolved when the object
/// is linked against the frontend runtime (see [`fusevm_aot_run_embedded`]).
pub fn compile_object(chunk: &Chunk, out_path: &Path) -> Result<(), String> {
    let isa = object_isa()?;
    let builder = ObjectBuilder::new(isa, "fusevm_aot", default_libcall_names())
        .map_err(|e| format!("aot: object builder: {e}"))?;
    let mut module = ObjectModule::new(builder);

    let blob = bincode::serialize(chunk).map_err(|e| format!("aot: serialize chunk: {e}"))?;
    let blob_id = module
        .declare_data(AOT_CHUNK_BLOB_SYMBOL, Linkage::Export, false, false)
        .map_err(|e| format!("aot: declare blob: {e}"))?;
    let mut blob_desc = DataDescription::new();
    blob_desc.define(blob.clone().into_boxed_slice());
    module
        .define_data(blob_id, &blob_desc)
        .map_err(|e| format!("aot: define blob: {e}"))?;

    let len_id = module
        .declare_data(AOT_CHUNK_LEN_SYMBOL, Linkage::Export, false, false)
        .map_err(|e| format!("aot: declare len: {e}"))?;
    let mut len_desc = DataDescription::new();
    len_desc.define(
        (blob.len() as u64)
            .to_le_bytes()
            .to_vec()
            .into_boxed_slice(),
    );
    module
        .define_data(len_id, &len_desc)
        .map_err(|e| format!("aot: define len: {e}"))?;

    build_entry(&mut module, chunk)?;

    let product = module.finish();
    let bytes = product
        .emit()
        .map_err(|e| format!("aot: emit object: {e}"))?;
    std::fs::write(out_path, bytes)
        .map_err(|e| format!("aot: write {}: {e}", out_path.display()))?;
    Ok(())
}

/// Runtime entry for a linked AOT binary. Deserializes the embedded chunk,
/// builds a VM, lets the frontend register its builtins, runs the native driver,
/// and maps the result to a process exit code. A frontend's AOT `main` calls
/// this; the symbols it references are satisfied by the linked object
/// ([`compile_object`]) and the frontend runtime.
///
/// The frontend must provide `extern "C" fn fusevm_aot_register_builtins(*mut VM)`.
#[no_mangle]
pub extern "C" fn fusevm_aot_run_embedded() -> i64 {
    extern "C" {
        static fusevm_aot_chunk_blob: u8;
        static fusevm_aot_chunk_len: u64;
        fn fusevm_aot_entry(vm: *mut VM) -> i64;
        fn fusevm_aot_register_builtins(vm: *mut VM);
    }
    // SAFETY: the linked object defines the blob/len symbols; the driver and
    // register hook are resolved at link time.
    let chunk: Chunk = unsafe {
        let len = fusevm_aot_chunk_len as usize;
        let bytes = std::slice::from_raw_parts(&fusevm_aot_chunk_blob as *const u8, len);
        match bincode::deserialize(bytes) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("fusevm aot: corrupt embedded chunk: {e}");
                return 1;
            }
        }
    };
    let mut vm = VM::new(chunk);
    // SAFETY: frontend-provided; registers builtins on the fresh VM.
    unsafe { fusevm_aot_register_builtins(&mut vm as *mut VM) };
    // SAFETY: the compiled entry has the declared C ABI and reads `vm`.
    unsafe { fusevm_aot_entry(&mut vm as *mut VM) };
    match vm.take_aot_result() {
        VMResult::Ok(Value::Int(n)) => n,
        VMResult::Ok(Value::Status(s)) => s as i64,
        VMResult::Ok(_) => 0,
        VMResult::Halted => 0,
        VMResult::Error(e) => {
            eprintln!("fusevm aot: {e}");
            1
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunk::ChunkBuilder;
    use crate::op::Op;
    use crate::value::Value;

    /// Native AOT result must equal the interpreter's for the same chunk.
    fn assert_native_matches_interp(chunk: Chunk) {
        let interp = {
            let mut vm = VM::new(chunk.clone());
            vm.run()
        };
        let native = run_chunk_native(&chunk, |_| {}).expect("native compile/run");
        match (interp, native) {
            (VMResult::Ok(a), VMResult::Ok(b)) => assert_eq!(a, b, "value mismatch"),
            (VMResult::Halted, VMResult::Halted) => {}
            (VMResult::Error(a), VMResult::Error(b)) => assert_eq!(a, b, "error mismatch"),
            (i, n) => panic!("interp {i:?} != native {n:?}"),
        }
    }

    #[test]
    fn native_straight_line_arithmetic() {
        let mut b = ChunkBuilder::new();
        b.emit(Op::LoadInt(2), 1);
        b.emit(Op::LoadInt(3), 1);
        b.emit(Op::Add, 1);
        b.emit(Op::LoadInt(4), 1);
        b.emit(Op::Mul, 1);
        let chunk = b.build();
        // This chunk must take the native fast path (no shim), and be correct.
        assert!(native_lowerable(&chunk), "arithmetic chunk should lower natively");
        assert_native_matches_interp(chunk);
    }

    #[test]
    fn native_arithmetic_with_negate_dup_pop() {
        // ((7 dup +) negate)  →  -14, with a dropped extra value.
        let mut b = ChunkBuilder::new();
        b.emit(Op::LoadInt(7), 1);
        b.emit(Op::Dup, 1);
        b.emit(Op::Add, 1);
        b.emit(Op::Negate, 1);
        b.emit(Op::LoadInt(99), 1);
        b.emit(Op::Pop, 1);
        let chunk = b.build();
        assert!(native_lowerable(&chunk));
        assert_native_matches_interp(chunk);
    }

    #[test]
    fn native_lowerable_rejects_unsupported_and_unsafe() {
        // A heap constant now lowers natively as a boxed handle (Kind::Obj),
        // returned as the program result — verified against the interpreter.
        let mut s = ChunkBuilder::new();
        let c = s.add_constant(Value::str("x"));
        s.emit(Op::LoadConst(c), 1);
        let schunk = s.build();
        assert!(native_lowerable(&schunk), "heap LoadConst lowers as Obj");
        assert_native_matches_interp(schunk);

        // A boolean left as the program result would box to the wrong Value
        // variant (`Bool` vs the native path's `Int`), so it is rejected.
        let mut cmp = ChunkBuilder::new();
        cmp.emit(Op::LoadInt(1), 1);
        cmp.emit(Op::LoadInt(2), 1);
        cmp.emit(Op::NumLt, 1);
        assert!(!native_lowerable(&cmp.build()), "bool result rejected");

        // Storing a boolean into a slot can't be a typed register, but it now
        // partial-lowers: the comparison runs native and the SetSlot deopts to
        // the interpreter. The result must still match.
        let mut bslot = ChunkBuilder::new();
        bslot.emit(Op::LoadInt(1), 1);
        bslot.emit(Op::LoadInt(2), 1);
        bslot.emit(Op::NumLt, 1);
        bslot.emit(Op::SetSlot(0), 1);
        assert_native_matches_interp(bslot.build());

        // Underflow (Add with one operand) is rejected, not miscompiled.
        let mut bad = ChunkBuilder::new();
        bad.emit(Op::LoadInt(1), 1);
        bad.emit(Op::Add, 1);
        assert!(!native_lowerable(&bad.build()), "underflow must be rejected");
    }

    #[test]
    fn native_unset_slot_read_falls_back() {
        // Reading a slot never written would yield `Undef` in the interpreter
        // but 0 in a register — so definite-assignment must reject it, and the
        // threaded path must still match the interpreter.
        let mut b = ChunkBuilder::new();
        b.emit(Op::GetSlot(0), 1);
        let chunk = b.build();
        assert!(!native_lowerable(&chunk), "unset slot read must fall back");
        assert_native_matches_interp(chunk);
    }

    #[test]
    fn native_slot_set_on_one_branch_falls_back() {
        // slot 0 is assigned only on the taken-if path; at the join it is not
        // definitely assigned, so the read must fall back to threaded.
        let mut b = ChunkBuilder::new();
        b.emit(Op::LoadInt(0), 1); // condition: false ⇒ skip the set
        let jf = b.emit(Op::JumpIfFalse(0), 1);
        b.emit(Op::LoadInt(7), 1);
        b.emit(Op::SetSlot(0), 1);
        let skip = b.current_pos();
        b.patch_jump(jf, skip);
        b.emit(Op::GetSlot(0), 1);
        let chunk = b.build();
        assert!(
            !native_lowerable(&chunk),
            "slot assigned on only one path must fall back"
        );
        assert_native_matches_interp(chunk);
    }

    #[test]
    fn native_slot_set_on_both_branches_lowers() {
        // slot 0 assigned on both arms ⇒ definitely assigned at the join ⇒ the
        // read lowers natively. (cond false ⇒ else ⇒ slot0 = 2.)
        let mut b = ChunkBuilder::new();
        b.emit(Op::LoadInt(0), 1);
        let jf = b.emit(Op::JumpIfFalse(0), 1);
        b.emit(Op::LoadInt(1), 1);
        b.emit(Op::SetSlot(0), 1);
        let j = b.emit(Op::Jump(0), 1);
        let else_ip = b.current_pos();
        b.patch_jump(jf, else_ip);
        b.emit(Op::LoadInt(2), 1);
        b.emit(Op::SetSlot(0), 1);
        let end_ip = b.current_pos();
        b.patch_jump(j, end_ip);
        b.emit(Op::GetSlot(0), 1);
        let chunk = b.build();
        assert!(native_lowerable(&chunk), "both-path assignment should lower");
        match run_chunk_native(&chunk, |_| {}).expect("native run") {
            VMResult::Ok(v) => assert_eq!(v, Value::Int(2)),
            other => panic!("expected Ok(2), got {other:?}"),
        }
    }

    #[test]
    fn native_float_slot_round_trip() {
        // x = 1.5; x = x + 0.25; return x → 1.75, with a float-typed slot.
        let mut b = ChunkBuilder::new();
        b.emit(Op::LoadFloat(1.5), 1);
        b.emit(Op::SetSlot(0), 1);
        b.emit(Op::GetSlot(0), 1);
        b.emit(Op::LoadFloat(0.25), 1);
        b.emit(Op::Add, 1);
        b.emit(Op::SetSlot(0), 1);
        b.emit(Op::GetSlot(0), 1);
        let chunk = b.build();
        assert!(native_lowerable(&chunk), "float slot should lower");
        match run_chunk_native(&chunk, |_| {}).expect("run") {
            VMResult::Ok(v) => assert_eq!(v, Value::Float(1.75)),
            other => panic!("got {other:?}"),
        }
    }

    #[test]
    fn native_float_accumulator_loop() {
        // sum = 0.0; for i in 0..5 { sum += 0.5 }; return sum → 2.5.
        // A float slot accumulator — the canonical awk pattern that previously
        // fell back to the interpreter.
        let mut b = ChunkBuilder::new();
        b.emit(Op::LoadFloat(0.0), 1);
        b.emit(Op::SetSlot(0), 1); // sum (float)
        b.emit(Op::LoadInt(0), 1);
        b.emit(Op::SetSlot(1), 1); // i (int)
        let top = b.current_pos();
        b.emit(Op::GetSlot(1), 1);
        b.emit(Op::LoadInt(5), 1);
        b.emit(Op::NumLt, 1);
        let exit = b.emit(Op::JumpIfFalse(0), 1);
        b.emit(Op::GetSlot(0), 1); // sum (float)
        b.emit(Op::LoadFloat(0.5), 1);
        b.emit(Op::Add, 1);
        b.emit(Op::SetSlot(0), 1);
        b.emit(Op::GetSlot(1), 1);
        b.emit(Op::LoadInt(1), 1);
        b.emit(Op::Add, 1);
        b.emit(Op::SetSlot(1), 1);
        b.emit(Op::Jump(top), 1);
        let end = b.current_pos();
        b.patch_jump(exit, end);
        b.emit(Op::GetSlot(0), 1);
        let chunk = b.build();
        assert!(native_lowerable(&chunk), "float accumulator loop should lower");
        match run_chunk_native(&chunk, |_| {}).expect("run") {
            VMResult::Ok(v) => assert_eq!(v, Value::Float(2.5)),
            other => panic!("got {other:?}"),
        }
        // (int slot `i` and float slot `sum` coexist in the same loop.)
    }

    #[test]
    fn native_float_global() {
        // Float global round-trips like a float slot.
        let mut b = ChunkBuilder::new();
        b.emit(Op::LoadFloat(3.5), 1);
        b.emit(Op::SetVar(0), 1);
        b.emit(Op::GetVar(0), 1);
        b.emit(Op::LoadFloat(1.5), 1);
        b.emit(Op::Add, 1);
        let chunk = b.build();
        assert!(native_lowerable(&chunk));
        match run_chunk_native(&chunk, |_| {}).expect("run") {
            VMResult::Ok(v) => assert_eq!(v, Value::Float(5.0)),
            other => panic!("got {other:?}"),
        }
    }

    #[test]
    fn native_mixed_kind_slot_falls_back() {
        // A slot stored as both Int and Float can't be a single typed register,
        // so the chunk falls back to threaded (and still matches the interp).
        let mut b = ChunkBuilder::new();
        b.emit(Op::LoadInt(1), 1);
        b.emit(Op::SetSlot(0), 1);
        b.emit(Op::LoadFloat(2.5), 1);
        b.emit(Op::SetSlot(0), 1); // same slot, different kind → conflict
        b.emit(Op::GetSlot(0), 1);
        let chunk = b.build();
        assert!(!native_lowerable(&chunk), "mixed-kind slot must fall back");
        assert_native_matches_interp(chunk);
    }

    #[test]
    fn native_int_slots_round_trip() {
        // x = 41; x = x + 1; return x  →  42, fully native (slots supported).
        let mut b = ChunkBuilder::new();
        b.emit(Op::LoadInt(41), 1);
        b.emit(Op::SetSlot(0), 1);
        b.emit(Op::GetSlot(0), 1);
        b.emit(Op::LoadInt(1), 1);
        b.emit(Op::Add, 1);
        b.emit(Op::SetSlot(0), 1);
        b.emit(Op::GetSlot(0), 1);
        let chunk = b.build();
        assert!(native_lowerable(&chunk));
        assert_native_matches_interp(chunk);
    }

    #[test]
    fn native_empty_is_lowerable() {
        assert!(native_lowerable(&ChunkBuilder::new().build()));
    }

    #[test]
    fn native_conditional_branch() {
        // if (1 < 2) push 10 else push 20  →  10
        let mut b = ChunkBuilder::new();
        b.emit(Op::LoadInt(1), 1);
        b.emit(Op::LoadInt(2), 1);
        b.emit(Op::NumLt, 1);
        let jf = b.emit(Op::JumpIfFalse(0), 1);
        b.emit(Op::LoadInt(10), 1);
        let j = b.emit(Op::Jump(0), 1);
        let else_ip = b.emit(Op::LoadInt(20), 1);
        let end_ip = b.current_pos();
        b.patch_jump(jf, else_ip);
        b.patch_jump(j, end_ip);
        let chunk = b.build();
        // Control flow + comparison now lower natively.
        assert!(native_lowerable(&chunk), "branch chunk should lower natively");
        assert_native_matches_interp(chunk);
    }

    #[test]
    fn native_counted_loop() {
        // sum = 0; for i in 0..5 { sum += i }  via slots
        let mut b = ChunkBuilder::new();
        // slot 0 = sum, slot 1 = i
        b.emit(Op::LoadInt(0), 1);
        b.emit(Op::SetSlot(0), 1);
        b.emit(Op::LoadInt(0), 1);
        b.emit(Op::SetSlot(1), 1);
        let top = b.current_pos();
        b.emit(Op::GetSlot(1), 1);
        b.emit(Op::LoadInt(5), 1);
        b.emit(Op::NumLt, 1);
        let exit = b.emit(Op::JumpIfFalse(0), 1);
        b.emit(Op::GetSlot(0), 1);
        b.emit(Op::GetSlot(1), 1);
        b.emit(Op::Add, 1);
        b.emit(Op::SetSlot(0), 1);
        b.emit(Op::GetSlot(1), 1);
        b.emit(Op::LoadInt(1), 1);
        b.emit(Op::Add, 1);
        b.emit(Op::SetSlot(1), 1);
        b.emit(Op::Jump(top), 1);
        let end_ip = b.current_pos();
        b.patch_jump(exit, end_ip);
        b.emit(Op::GetSlot(0), 1);
        let chunk = b.build();
        // A real loop with a back-edge, slots, and a comparison — all native.
        assert!(native_lowerable(&chunk), "loop chunk should lower natively");
        assert_native_matches_interp(chunk);
    }

    #[test]
    fn native_loop_sum_value_is_correct() {
        // Same loop, but assert the concrete value (0+1+2+3+4 = 10), so a
        // miscompiled back-edge/phi would be caught, not just an interp match.
        let mut b = ChunkBuilder::new();
        b.emit(Op::LoadInt(0), 1);
        b.emit(Op::SetSlot(0), 1);
        b.emit(Op::LoadInt(0), 1);
        b.emit(Op::SetSlot(1), 1);
        let top = b.current_pos();
        b.emit(Op::GetSlot(1), 1);
        b.emit(Op::LoadInt(5), 1);
        b.emit(Op::NumLt, 1);
        let exit = b.emit(Op::JumpIfFalse(0), 1);
        b.emit(Op::GetSlot(0), 1);
        b.emit(Op::GetSlot(1), 1);
        b.emit(Op::Add, 1);
        b.emit(Op::SetSlot(0), 1);
        b.emit(Op::GetSlot(1), 1);
        b.emit(Op::LoadInt(1), 1);
        b.emit(Op::Add, 1);
        b.emit(Op::SetSlot(1), 1);
        b.emit(Op::Jump(top), 1);
        let end_ip = b.current_pos();
        b.patch_jump(exit, end_ip);
        b.emit(Op::GetSlot(0), 1);
        let chunk = b.build();
        assert!(native_lowerable(&chunk));
        match run_chunk_native(&chunk, |_| {}).expect("native run") {
            VMResult::Ok(v) => assert_eq!(v, Value::Int(10), "0+1+2+3+4 == 10"),
            other => panic!("expected Ok(10), got {other:?}"),
        }
    }

    #[test]
    fn native_differential_arithmetic_and_branches() {
        // Deterministic differential fuzz: build many random integer programs
        // (balanced RPN arithmetic, optionally guarded by a comparison + branch)
        // and require the native compile to match the interpreter exactly.
        let mut seed: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut next = || {
            // SplitMix64 — deterministic, no external rng.
            seed = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = seed;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        };

        // Emit a balanced arithmetic expression leaving exactly one Int on the
        // stack (depth grows with LoadInt, shrinks with a binop).
        fn emit_expr(b: &mut ChunkBuilder, next: &mut impl FnMut() -> u64, terms: usize) {
            b.emit(Op::LoadInt((next() % 21) as i64 - 10), 1);
            for _ in 0..terms {
                b.emit(Op::LoadInt((next() % 21) as i64 - 10), 1);
                match next() % 3 {
                    0 => b.emit(Op::Add, 1),
                    1 => b.emit(Op::Sub, 1),
                    _ => b.emit(Op::Mul, 1),
                };
            }
        }

        for _ in 0..400 {
            let mut b = ChunkBuilder::new();
            if next() % 2 == 0 {
                // Plain arithmetic expression.
                let terms = (next() % 5) as usize;
                emit_expr(&mut b, &mut next, terms);
            } else {
                // if (a < b) <expr1> else <expr2>
                b.emit(Op::LoadInt((next() % 11) as i64 - 5), 1);
                b.emit(Op::LoadInt((next() % 11) as i64 - 5), 1);
                match next() % 4 {
                    0 => b.emit(Op::NumLt, 1),
                    1 => b.emit(Op::NumGt, 1),
                    2 => b.emit(Op::NumEq, 1),
                    _ => b.emit(Op::NumGe, 1),
                };
                let jf = b.emit(Op::JumpIfFalse(0), 1);
                let then_terms = (next() % 4) as usize;
                emit_expr(&mut b, &mut next, then_terms);
                let j = b.emit(Op::Jump(0), 1);
                let else_ip = b.current_pos();
                b.patch_jump(jf, else_ip);
                let else_terms = (next() % 4) as usize;
                emit_expr(&mut b, &mut next, else_terms);
                let end_ip = b.current_pos();
                b.patch_jump(j, end_ip);
            }
            let chunk = b.build();
            assert!(native_lowerable(&chunk), "generated chunk must lower natively");
            assert_native_matches_interp(chunk);
        }
    }

    #[test]
    fn native_differential_integer_bitwise() {
        // Deterministic differential fuzz over the integer op set, including
        // bitwise/shift and Inc/Dec. Every value is Int, so each chunk lowers
        // and must match the interpreter exactly (wrapping on both sides).
        let mut seed: u64 = 0x1234_5678_9ABC_DEF0;
        let mut next = || {
            seed = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = seed;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        };

        for _ in 0..400 {
            let mut b = ChunkBuilder::new();
            b.emit(Op::LoadInt((next() % 17) as i64 - 8), 1);
            let terms = next() % 8;
            for _ in 0..terms {
                if next() % 3 == 0 {
                    // unary, int → int (depth unchanged)
                    match next() % 3 {
                        0 => b.emit(Op::Inc, 1),
                        1 => b.emit(Op::Dec, 1),
                        _ => b.emit(Op::BitNot, 1),
                    };
                } else {
                    // binary: push another operand, then combine (depth unchanged)
                    b.emit(Op::LoadInt((next() % 17) as i64 - 8), 1);
                    match next() % 9 {
                        0 => b.emit(Op::Add, 1),
                        1 => b.emit(Op::Sub, 1),
                        2 => b.emit(Op::Mul, 1),
                        3 => b.emit(Op::BitAnd, 1),
                        4 => b.emit(Op::BitOr, 1),
                        5 => b.emit(Op::BitXor, 1),
                        6 => b.emit(Op::Shl, 1),
                        7 => b.emit(Op::Shr, 1),
                        // operands span -8..8, so this exercises `% 0` and `% -1`.
                        _ => b.emit(Op::Mod, 1),
                    };
                }
            }
            let chunk = b.build();
            assert!(native_lowerable(&chunk), "integer chunk must lower");
            assert_native_matches_interp(chunk);
        }
    }

    #[test]
    fn native_bool_lognot_and_bitwise() {
        // !true → false; (false) & 5 == 0. A Bool flows through LogNot into a
        // bitwise op (Bool is int-like) and yields an Int result — all native.
        let mut b = ChunkBuilder::new();
        b.emit(Op::LoadTrue, 1);
        b.emit(Op::LogNot, 1); // false (0)
        b.emit(Op::LoadInt(5), 1);
        b.emit(Op::BitAnd, 1); // 0 & 5 = 0
        let chunk = b.build();
        assert!(native_lowerable(&chunk));
        match run_chunk_native(&chunk, |_| {}).expect("native run") {
            VMResult::Ok(v) => assert_eq!(v, Value::Int(0)),
            other => panic!("expected Ok(0), got {other:?}"),
        }

        // A bare Bool result is not numeric ⇒ falls back; threaded must match.
        let mut bb = ChunkBuilder::new();
        bb.emit(Op::LoadFalse, 1);
        bb.emit(Op::LogNot, 1); // true — but a Bool result, so not lowered
        let chunk = bb.build();
        assert!(!native_lowerable(&chunk), "bare bool result falls back");
        assert_native_matches_interp(chunk);
    }

    /// Run a lowerable chunk natively and assert its integer result.
    fn assert_native_int(chunk: Chunk, expect: i64) {
        assert!(native_lowerable(&chunk), "chunk should lower natively");
        match run_chunk_native(&chunk, |_| {}).expect("native run") {
            VMResult::Ok(v) => assert_eq!(v, Value::Int(expect)),
            other => panic!("expected Ok({expect}), got {other:?}"),
        }
    }

    /// Run a lowerable chunk natively and assert its (finite) float result.
    fn assert_native_int_or_float(chunk: Chunk, expect: f64) {
        assert!(native_lowerable(&chunk), "chunk should lower natively");
        match run_chunk_native(&chunk, |_| {}).expect("native run") {
            VMResult::Ok(v) => assert_eq!(v, Value::Float(expect)),
            other => panic!("expected Ok({expect}), got {other:?}"),
        }
    }

    #[test]
    fn native_partial_deopt() {
        // A chunk mixing native arithmetic with a non-lowerable op (Concat) now
        // lowers: the arithmetic runs native, the Concat deopts to the
        // interpreter, and the result matches end-to-end.
        let mut b = ChunkBuilder::new();
        b.emit(Op::LoadInt(2), 1);
        b.emit(Op::LoadInt(3), 1);
        b.emit(Op::Add, 1); // 5, native
        b.emit(Op::LoadInt(4), 1); // native
        b.emit(Op::Concat, 1); // non-lowerable → deopt; "5" + "4" = "54"
        let chunk = b.build();
        assert!(native_lowerable(&chunk), "mixed chunk should lower (with deopt)");
        match run_chunk_native(&chunk, |_| {}).expect("run") {
            VMResult::Ok(v) => assert_eq!(v, Value::str("54")),
            other => panic!("got {other:?}"),
        }
        assert_native_matches_interp(chunk);

        // Slot state set NATIVELY must survive the deopt: slot 0 = 99 in a
        // register, then a Concat deopts; the resumed interpreter reads slot 0.
        let mut b = ChunkBuilder::new();
        b.emit(Op::LoadInt(99), 1);
        b.emit(Op::SetSlot(0), 1); // native register
        b.emit(Op::LoadInt(1), 1);
        b.emit(Op::LoadInt(2), 1);
        b.emit(Op::Concat, 1); // deopt
        b.emit(Op::Pop, 1);
        b.emit(Op::GetSlot(0), 1); // interpreter reads slot 0 → must be 99
        let chunk = b.build();
        assert!(native_lowerable(&chunk));
        match run_chunk_native(&chunk, |_| {}).expect("run") {
            VMResult::Ok(v) => assert_eq!(v, Value::Int(99), "slot survived partial deopt"),
            other => panic!("got {other:?}"),
        }
        assert_native_matches_interp(chunk);

        // The correctness fix: an UNassigned slot must stay Undef across the
        // deopt (not be written back as a register's 0).
        let mut b = ChunkBuilder::new();
        b.emit(Op::LoadInt(1), 1);
        b.emit(Op::LoadInt(2), 1);
        b.emit(Op::LoadUndef, 1); // not lowered → deopt; slot 0 never assigned
        b.emit(Op::Pop, 1);
        b.emit(Op::GetSlot(0), 1); // interpreter reads an unassigned slot → Undef
        let chunk = b.build();
        assert!(native_lowerable(&chunk));
        match run_chunk_native(&chunk, |_| {}).expect("run") {
            VMResult::Ok(Value::Undef) => {}
            other => panic!("expected Undef (unassigned slot), got {other:?}"),
        }
        assert_native_matches_interp(chunk);
    }

    #[test]
    fn native_differential_partial_deopt() {
        // Native integer expressions, half of them followed by a non-lowerable
        // Concat (which deopts). Both the pure-native and the native-prefix-then-
        // deopt forms must match the interpreter.
        let mut seed: u64 = 0x6C62_72E0_7E1A_55D9;
        let mut next = || {
            seed = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = seed;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        };
        let mut deopted = 0;
        for _ in 0..400 {
            let mut b = ChunkBuilder::new();
            b.emit(Op::LoadInt((next() % 21) as i64 - 10), 1);
            for _ in 0..(next() % 4) {
                b.emit(Op::LoadInt((next() % 21) as i64 - 10), 1);
                match next() % 3 {
                    0 => b.emit(Op::Add, 1),
                    1 => b.emit(Op::Sub, 1),
                    _ => b.emit(Op::Mul, 1),
                };
            }
            if next() % 2 == 0 {
                // append a Concat with a literal → non-lowerable, deopts.
                b.emit(Op::LoadInt((next() % 21) as i64 - 10), 1);
                b.emit(Op::Concat, 1);
                deopted += 1;
            }
            let chunk = b.build();
            assert!(native_lowerable(&chunk), "expr (±Concat) should lower");
            assert_native_matches_interp(chunk);
        }
        assert!(deopted > 100, "expected many deopting chunks, got {deopted}");
    }

    #[test]
    fn native_conditional_slot_before_deopt_is_sound() {
        // Regression: a slot assigned on only ONE branch before a deopt can't be
        // soundly written back (its register can't distinguish a real value from
        // Undef), so the chunk must fall back to threaded — never miscompile.
        //   if (1) slot0 = 5;  "7"~"8" (deopt);  pop;  return slot0   → 5
        let build = |cond: i64| {
            let mut b = ChunkBuilder::new();
            b.emit(Op::LoadInt(cond), 1);
            let jf = b.emit(Op::JumpIfFalse(0), 1);
            b.emit(Op::LoadInt(5), 1);
            b.emit(Op::SetSlot(0), 1); // conditional assignment
            let skip = b.current_pos();
            b.patch_jump(jf, skip);
            b.emit(Op::LoadInt(7), 1);
            b.emit(Op::LoadInt(8), 1);
            b.emit(Op::Concat, 1); // deopt
            b.emit(Op::Pop, 1);
            b.emit(Op::GetSlot(0), 1);
            b.build()
        };
        // Maybe-assigned slot before a deopt ⇒ not lowered, falls back.
        let chunk = build(1);
        assert!(
            !native_lowerable(&chunk),
            "conditional slot before deopt must fall back, not miscompile"
        );
        // Both branch values must match the interpreter via the threaded path.
        assert_native_matches_interp(build(1)); // cond true  → slot0 = 5
        assert_native_matches_interp(build(0)); // cond false → slot0 = Undef
    }

    #[test]
    fn native_differential_cond_assign_deopt() {
        // Hammer the must/may-assigned soundness boundary: random combinations of
        // unconditional/conditional slot assignment before a deopt, then a read.
        // Whether the chunk lowers (must-assigned) or falls back (maybe-assigned),
        // the result must match the interpreter.
        let mut seed: u64 = 0x9F1C_2B3D_4E5A_6071;
        let mut next = || {
            seed = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = seed;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        };
        for _ in 0..400 {
            let mut b = ChunkBuilder::new();
            let init = next() % 2 == 0; // unconditional init before the branch
            if init {
                b.emit(Op::LoadInt((next() % 9) as i64), 1);
                b.emit(Op::SetSlot(0), 1);
            }
            b.emit(Op::LoadInt((next() % 2) as i64), 1); // cond 0/1
            let jf = b.emit(Op::JumpIfFalse(0), 1);
            b.emit(Op::LoadInt((next() % 9) as i64 + 10), 1); // true-branch value
            b.emit(Op::SetSlot(0), 1);
            let j = b.emit(Op::Jump(0), 1);
            let else_ip = b.current_pos();
            b.patch_jump(jf, else_ip);
            if next() % 2 == 0 {
                // else branch also assigns (→ definitely assigned at the join)
                b.emit(Op::LoadInt((next() % 9) as i64 + 20), 1);
                b.emit(Op::SetSlot(0), 1);
            }
            let end_ip = b.current_pos();
            b.patch_jump(j, end_ip);
            // a deopt, then read slot 0 as the result
            b.emit(Op::LoadInt(7), 1);
            b.emit(Op::LoadInt(8), 1);
            b.emit(Op::Concat, 1);
            b.emit(Op::Pop, 1);
            b.emit(Op::GetSlot(0), 1);
            // Result must match the interpreter on whichever path was chosen.
            assert_native_matches_interp(b.build());
        }
    }

    #[test]
    fn native_boxed_concat_lowers() {
        // Heap string constants + Concat now lower natively via boxed handles
        // (no deopt): each LoadConst boxes into the arena, Concat unboxes/joins/
        // re-boxes, and the Obj result is returned. Must match the interpreter.
        let mut b = ChunkBuilder::new();
        let a = b.add_constant(Value::str("a"));
        let c = b.add_constant(Value::str("b"));
        b.emit(Op::LoadConst(a), 1);
        b.emit(Op::LoadConst(c), 1);
        b.emit(Op::Concat, 1); // "ab"
        let chunk = b.build();
        assert!(native_lowerable(&chunk), "boxed concat should lower natively");
        assert_native_matches_interp(chunk);

        // Mixed scalar/boxed operands: "n=" . (2 + 3) → "n=5". The arithmetic
        // runs in registers, the Int is spilled, the string is unboxed.
        let mut b = ChunkBuilder::new();
        let pre = b.add_constant(Value::str("n="));
        b.emit(Op::LoadConst(pre), 1);
        b.emit(Op::LoadInt(2), 1);
        b.emit(Op::LoadInt(3), 1);
        b.emit(Op::Add, 1);
        b.emit(Op::Concat, 1);
        assert_native_matches_interp(b.build());

        // Three-way concat (boxed result fed back into Concat): "a"."b"."c".
        let mut b = ChunkBuilder::new();
        let k = |s| Value::str(s);
        let (x, y, z) = (b.add_constant(k("a")), b.add_constant(k("b")), b.add_constant(k("c")));
        b.emit(Op::LoadConst(x), 1);
        b.emit(Op::LoadConst(y), 1);
        b.emit(Op::Concat, 1);
        b.emit(Op::LoadConst(z), 1);
        b.emit(Op::Concat, 1);
        assert_native_matches_interp(b.build());
    }

    #[test]
    fn native_named_array_hash_ops_lower() {
        // Array element round-trip: a[0]=42; a[0] → 42 (boxed Obj result).
        let mut b = ChunkBuilder::new();
        b.emit(Op::DeclareArray(0), 1);
        b.emit(Op::LoadInt(42), 1);
        b.emit(Op::LoadInt(0), 1);
        b.emit(Op::ArraySet(0), 1); // a[0] = 42  (stack [value, index])
        b.emit(Op::LoadInt(0), 1);
        b.emit(Op::ArrayGet(0), 1); // → 42
        let chunk = b.build();
        assert!(native_lowerable(&chunk), "array element ops lower");
        assert_native_matches_interp(chunk);

        // Push in a loop, then length → 3 (scalar Int result).
        let mut b = ChunkBuilder::new();
        b.emit(Op::DeclareArray(0), 1);
        b.emit(Op::LoadInt(0), 1);
        b.emit(Op::SetSlot(0), 1); // i = 0
        b.emit(Op::GetSlot(0), 1); // leader ip 3
        b.emit(Op::ArrayPush(0), 1); // push i
        b.emit(Op::SlotIncLtIntJumpBack(0, 3, 3), 1); // i++; if i<3 goto 3
        b.emit(Op::ArrayLen(0), 1); // → 3
        let chunk = b.build();
        assert!(native_lowerable(&chunk), "array push loop + len lowers");
        assert_native_matches_interp(chunk);

        // Hash element round-trip: h{"k"}=1; h{"k"} → 1.
        let mut b = ChunkBuilder::new();
        let k = b.add_constant(Value::str("k"));
        b.emit(Op::DeclareHash(0), 1);
        b.emit(Op::LoadInt(1), 1);
        b.emit(Op::LoadConst(k), 1);
        b.emit(Op::HashSet(0), 1); // h{"k"} = 1  (stack [value, key])
        b.emit(Op::LoadConst(k), 1);
        b.emit(Op::HashGet(0), 1); // → 1
        assert_native_matches_interp(b.build());
    }

    #[test]
    fn native_expand_param_ops_lower() {
        // `${var:-def}` (1 arg + name = 2 pops) and `${#var}` (LENGTH: 0 args +
        // name = 1 pop). No host ⇒ "" deterministically; pop counts must match
        // the interpreter exactly or the boxed stack desyncs.
        let mut b = ChunkBuilder::new();
        let name = b.add_constant(Value::str("v"));
        let def = b.add_constant(Value::str("d"));
        b.emit(Op::LoadConst(name), 1);
        b.emit(Op::LoadConst(def), 1);
        b.emit(Op::ExpandParam(crate::op::param_mod::DEFAULT), 1);
        let chunk = b.build();
        assert!(native_lowerable(&chunk), "ExpandParam(DEFAULT) lowers");
        assert_native_matches_interp(chunk);

        let mut b = ChunkBuilder::new();
        let name = b.add_constant(Value::str("v"));
        b.emit(Op::LoadConst(name), 1);
        b.emit(Op::ExpandParam(crate::op::param_mod::LENGTH), 1);
        let chunk = b.build();
        assert!(native_lowerable(&chunk), "ExpandParam(LENGTH) lowers");
        assert_native_matches_interp(chunk);
    }

    #[test]
    fn native_shell_predicate_ops_lower() {
        // Glob/regex match (no host ⇒ deterministic) and file tests lower as
        // boxed-heap Bool predicates; result consumed by MakeArray for a valid
        // end. Verified against the interpreter.
        let mut b = ChunkBuilder::new();
        let s = b.add_constant(Value::str("abc"));
        let p = b.add_constant(Value::str("abc"));
        b.emit(Op::LoadConst(s), 1);
        b.emit(Op::LoadConst(p), 1);
        b.emit(Op::StrMatch, 1); // no host ⇒ s == pat ⇒ true
        b.emit(Op::MakeArray(1), 1);
        let chunk = b.build();
        assert!(native_lowerable(&chunk), "StrMatch lowers");
        assert_native_matches_interp(chunk);

        // TestFile: "/" is a directory ⇒ true.
        let mut b = ChunkBuilder::new();
        let root = b.add_constant(Value::str("/"));
        b.emit(Op::LoadConst(root), 1);
        b.emit(Op::TestFile(crate::op::file_test::IS_DIR), 1);
        b.emit(Op::MakeArray(1), 1);
        let chunk = b.build();
        assert!(native_lowerable(&chunk), "TestFile lowers");
        assert_native_matches_interp(chunk);
    }

    #[test]
    fn native_expansion_source_ops_lower() {
        // Host-dispatched string-expansion ops lower as boxed sources; with no
        // host installed both interp and native take the same fallback, so the
        // result matches. (Exercises the (1 → Obj) general-arm path.)
        for op in [Op::BraceExpand, Op::WordSplit, Op::TildeExpand] {
            let mut b = ChunkBuilder::new();
            let s = b.add_constant(Value::str("a b c"));
            b.emit(Op::LoadConst(s), 1);
            b.emit(op, 1);
            let chunk = b.build();
            assert!(native_lowerable(&chunk), "expansion op lowers");
            assert_native_matches_interp(chunk);
        }
    }

    #[test]
    fn native_string_compares_lower() {
        // StrCmp → Int (-1/0/1), a valid program result.
        let mut b = ChunkBuilder::new();
        let a = b.add_constant(Value::str("abc"));
        let c = b.add_constant(Value::str("abd"));
        b.emit(Op::LoadConst(a), 1);
        b.emit(Op::LoadConst(c), 1);
        b.emit(Op::StrCmp, 1); // -1
        let chunk = b.build();
        assert!(native_lowerable(&chunk), "StrCmp lowers");
        assert_native_matches_interp(chunk);

        // StrEq / StrLt → Bool, consumed by MakeArray so it's a valid end.
        for (op, l, r) in [
            (Op::StrEq, "x", "x"),
            (Op::StrEq, "x", "y"),
            (Op::StrLt, "a", "b"),
            (Op::StrGe, "b", "a"),
        ] {
            let mut b = ChunkBuilder::new();
            let lc = b.add_constant(Value::str(l));
            let rc = b.add_constant(Value::str(r));
            b.emit(Op::LoadConst(lc), 1);
            b.emit(Op::LoadConst(rc), 1);
            b.emit(op, 1);
            b.emit(Op::MakeArray(1), 1); // [bool]
            let chunk = b.build();
            assert!(native_lowerable(&chunk), "string compare lowers");
            assert_native_matches_interp(chunk);
        }
    }

    #[test]
    fn native_whole_array_and_hash_view_ops_lower() {
        // SetArray/GetArray whole-array round-trip: a = [1,2]; @a → [1,2].
        let mut b = ChunkBuilder::new();
        b.emit(Op::LoadInt(1), 1);
        b.emit(Op::LoadInt(2), 1);
        b.emit(Op::MakeArray(2), 1);
        b.emit(Op::SetArray(0), 1); // a = [1,2]
        b.emit(Op::GetArray(0), 1); // → [1,2]
        let chunk = b.build();
        assert!(native_lowerable(&chunk), "GetArray/SetArray lower");
        assert_native_matches_interp(chunk);

        // HashDelete returns the removed value: h{"x"}=5; delete h{"x"} → 5.
        let mut b = ChunkBuilder::new();
        let x = b.add_constant(Value::str("x"));
        b.emit(Op::DeclareHash(0), 1);
        b.emit(Op::LoadInt(5), 1);
        b.emit(Op::LoadConst(x), 1);
        b.emit(Op::HashSet(0), 1);
        b.emit(Op::LoadConst(x), 1);
        b.emit(Op::HashDelete(0), 1); // → 5
        let chunk = b.build();
        assert!(native_lowerable(&chunk), "HashDelete lowers");
        assert_native_matches_interp(chunk);

        // HashExists (Bool result) consumed by MakeArray: [exists h{"x"}].
        let mut b = ChunkBuilder::new();
        let x = b.add_constant(Value::str("x"));
        b.emit(Op::DeclareHash(0), 1);
        b.emit(Op::LoadInt(1), 1);
        b.emit(Op::LoadConst(x), 1);
        b.emit(Op::HashSet(0), 1);
        b.emit(Op::LoadConst(x), 1);
        b.emit(Op::HashExists(0), 1); // Bool(true)
        b.emit(Op::MakeArray(1), 1); // [true]
        let chunk = b.build();
        assert!(native_lowerable(&chunk), "HashExists (bool result) lowers");
        assert_native_matches_interp(chunk);
    }

    #[test]
    fn native_array_global_alias_bails() {
        // Name 0 used as BOTH a register-cached global (SetVar/GetVar) and a
        // named array (ArrayPush) → the register would shadow the shim's view of
        // self.globals[0], so the chunk must fall back to threaded (not lower),
        // and threaded must still match the interpreter.
        let mut b = ChunkBuilder::new();
        b.emit(Op::DeclareArray(0), 1);
        b.emit(Op::LoadInt(7), 1);
        b.emit(Op::SetVar(0), 1); // name 0 as a global register
        b.emit(Op::LoadInt(9), 1);
        b.emit(Op::ArrayPush(0), 1); // name 0 as an array → overlap
        b.emit(Op::GetVar(0), 1);
        let chunk = b.build();
        assert!(!native_lowerable(&chunk), "global/array name overlap must bail");
        assert_native_matches_interp(chunk);
    }

    #[test]
    fn native_heap_ops_lower() {
        // String repeat/len, array & hash construction, and ranges now lower as
        // boxed heap ops (staged operands → shim → reloaded result), matching
        // the interpreter.
        let mut b = ChunkBuilder::new();
        let ab = b.add_constant(Value::str("ab"));
        b.emit(Op::LoadConst(ab), 1);
        b.emit(Op::LoadInt(3), 1);
        b.emit(Op::StringRepeat, 1); // "ababab"
        assert_native_matches_interp(b.build());

        let mut b = ChunkBuilder::new();
        let hello = b.add_constant(Value::str("hello"));
        b.emit(Op::LoadConst(hello), 1);
        b.emit(Op::StringLen, 1); // Int 5 (scalar result)
        let chunk = b.build();
        assert!(native_lowerable(&chunk), "StringLen lowers");
        assert_native_matches_interp(chunk);

        // MakeArray with mixed scalar/computed/boxed elements: [1, "x", 2+3].
        let mut b = ChunkBuilder::new();
        let x = b.add_constant(Value::str("x"));
        b.emit(Op::LoadInt(1), 1);
        b.emit(Op::LoadConst(x), 1);
        b.emit(Op::LoadInt(2), 1);
        b.emit(Op::LoadInt(3), 1);
        b.emit(Op::Add, 1);
        b.emit(Op::MakeArray(3), 1);
        assert_native_matches_interp(b.build());

        // Range and a StringLen of nothing-special; both boxed-Obj / scalar.
        let mut b = ChunkBuilder::new();
        b.emit(Op::LoadInt(1), 1);
        b.emit(Op::LoadInt(5), 1);
        b.emit(Op::Range, 1); // [1,2,3,4,5]
        assert_native_matches_interp(b.build());

        // MakeHash(2): {"k" => 1}.
        let mut b = ChunkBuilder::new();
        let kk = b.add_constant(Value::str("k"));
        b.emit(Op::LoadConst(kk), 1);
        b.emit(Op::LoadInt(1), 1);
        b.emit(Op::MakeHash(2), 1);
        assert_native_matches_interp(b.build());
    }

    #[test]
    fn native_obj_slot_accumulator_loop_lowers() {
        // The classic heap-in-loop case now runs fully native: an Obj-typed slot
        // holds a string accumulator, concatenated each iteration. Ownership
        // (clone-on-read, free-on-overwrite, take-on-consume) keeps it correct;
        // the arena stays bounded. Result must match the interpreter ("xxx").
        let mut b = ChunkBuilder::new();
        let empty = b.add_constant(Value::str(""));
        let x = b.add_constant(Value::str("x"));
        b.emit(Op::LoadInt(0), 1); // i = 0
        b.emit(Op::SetSlot(0), 1);
        b.emit(Op::LoadConst(empty), 1); // s = "" (Obj slot)
        b.emit(Op::SetSlot(1), 1);
        // loop body (leader at ip 4):
        b.emit(Op::GetSlot(1), 1); // s   (clone)
        b.emit(Op::LoadConst(x), 1); // "x"
        b.emit(Op::Concat, 1); // s . "x"
        b.emit(Op::SetSlot(1), 1); // s = ...
        b.emit(Op::SlotIncLtIntJumpBack(0, 3, 4), 1); // i++; if i<3 goto 4
        b.emit(Op::GetSlot(1), 1); // result: s
        let chunk = b.build();
        assert!(native_lowerable(&chunk), "Obj-slot accumulator loop lowers");
        assert_native_matches_interp(chunk);
    }

    #[test]
    fn native_obj_accumulator_differential_fuzz() {
        // Stress the handle-ownership/freelist across many iteration counts and
        // string pieces: every accumulator loop must lower and match the interp.
        // (A double-free / use-after-free or leak-as-wrong-value would diverge.)
        let mut seed: u64 = 0x1234_5678_9ABC_DEF0;
        let mut next = || {
            seed = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = seed;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        };
        for _ in 0..200 {
            let iters = (next() % 12) as i32; // 0..11 iterations
            let piece = ["x", "ab", "", "Z9"][(next() % 4) as usize];
            let mut b = ChunkBuilder::new();
            let empty = b.add_constant(Value::str("acc:"));
            let p = b.add_constant(Value::str(piece));
            b.emit(Op::LoadInt(0), 1);
            b.emit(Op::SetSlot(0), 1);
            b.emit(Op::LoadConst(empty), 1);
            b.emit(Op::SetSlot(1), 1);
            b.emit(Op::GetSlot(1), 1); // leader ip 4
            b.emit(Op::LoadConst(p), 1);
            b.emit(Op::Concat, 1);
            b.emit(Op::SetSlot(1), 1);
            b.emit(Op::SlotIncLtIntJumpBack(0, iters, 4), 1);
            b.emit(Op::GetSlot(1), 1);
            let chunk = b.build();
            // iters<=0 still executes the body once (post-inc test), matching
            // the interpreter; either way native must agree.
            assert_native_matches_interp(chunk);
        }
    }

    #[test]
    fn native_obj_global_roundtrip_lowers() {
        // An Obj global: store a string, read it back as the result.
        let mut b = ChunkBuilder::new();
        let s = b.add_constant(Value::str("hi"));
        b.emit(Op::LoadConst(s), 1);
        b.emit(Op::SetVar(0), 1);
        b.emit(Op::GetVar(0), 1);
        let chunk = b.build();
        assert!(native_lowerable(&chunk), "Obj global round-trips");
        assert_native_matches_interp(chunk);
    }

    #[test]
    fn native_boxed_value_then_deopt_spills() {
        // A boxed value live across a deopt point must be unboxed back onto the
        // stack so the resumed interpreter sees the real string, not a handle.
        // StringRepeat isn't lowered → deopt after the boxed Concat.
        let mut b = ChunkBuilder::new();
        let s = b.add_constant(Value::str("ab"));
        b.emit(Op::LoadConst(s), 1);
        let two = b.add_constant(Value::str("x"));
        b.emit(Op::LoadConst(two), 1);
        b.emit(Op::Concat, 1); // boxed "abx"
        b.emit(Op::LoadInt(2), 1);
        b.emit(Op::StringRepeat, 1); // not lowered → deopt; spills the Obj
        assert_native_matches_interp(b.build());
    }

    #[test]
    fn native_type_mismatch_deopts() {
        // A Bool (comparison result) fed into Add is a type mismatch for the
        // native path; it deopts and the interpreter float-promotes. The native
        // prefix still lowers, and the result matches.
        let mut b = ChunkBuilder::new();
        b.emit(Op::LoadInt(1), 1);
        b.emit(Op::LoadInt(2), 1);
        b.emit(Op::NumLt, 1); // Bool(true)
        b.emit(Op::LoadInt(5), 1);
        b.emit(Op::Add, 1); // Bool + Int → deopt → Float
        let chunk = b.build();
        assert!(native_lowerable(&chunk), "type-mismatch chunk partial-lowers");
        assert_native_matches_interp(chunk);

        // SetSlot of a Bool deopts; the slot is then read back by the interp.
        let mut b = ChunkBuilder::new();
        b.emit(Op::LoadInt(3), 1);
        b.emit(Op::LoadInt(3), 1);
        b.emit(Op::NumEq, 1); // Bool(true)
        b.emit(Op::SetSlot(0), 1); // deopt
        b.emit(Op::GetSlot(0), 1);
        assert_native_matches_interp(b.build());
    }

    #[test]
    fn native_heap_const_deopts_not_bails() {
        // A heap (string) constant load is now a deopt point: a native prefix
        // before it still lowers, instead of bailing the whole chunk.
        let mut b = ChunkBuilder::new();
        b.emit(Op::LoadInt(40), 1);
        b.emit(Op::LoadInt(2), 1);
        b.emit(Op::Add, 1); // 42, native
        let c = b.add_constant(Value::str("!"));
        b.emit(Op::LoadConst(c), 1); // heap constant → deopt
        b.emit(Op::Concat, 1); // "42" + "!" = "42!"
        let chunk = b.build();
        assert!(native_lowerable(&chunk), "native prefix + heap const should lower");
        match run_chunk_native(&chunk, |_| {}).expect("run") {
            VMResult::Ok(v) => assert_eq!(v, Value::str("42!")),
            other => panic!("got {other:?}"),
        }
        assert_native_matches_interp(chunk);

        // A chunk that is ONLY a heap-const load now lowers as a boxed Obj
        // source returning that value — matching the interpreter.
        let mut b = ChunkBuilder::new();
        let c = b.add_constant(Value::str("x"));
        b.emit(Op::LoadConst(c), 1);
        let chunk = b.build();
        assert!(native_lowerable(&chunk), "heap LoadConst lowers as Obj");
        assert_native_matches_interp(chunk);
    }

    #[test]
    fn native_div() {
        // Normal division → Float.
        let mut b = ChunkBuilder::new();
        b.emit(Op::LoadFloat(7.0), 1);
        b.emit(Op::LoadFloat(2.0), 1);
        b.emit(Op::Div, 1);
        let chunk = b.build();
        assert!(native_lowerable(&chunk), "Div should lower");
        match run_chunk_native(&chunk, |_| {}).expect("run") {
            VMResult::Ok(v) => assert_eq!(v, Value::Float(3.5)),
            other => panic!("got {other:?}"),
        }

        // Divide-by-zero deopts to the interpreter, which yields Undef — must
        // match the interpreter exactly.
        let mut b = ChunkBuilder::new();
        b.emit(Op::LoadFloat(5.0), 1);
        b.emit(Op::LoadFloat(0.0), 1);
        b.emit(Op::Div, 1);
        let chunk = b.build();
        assert!(native_lowerable(&chunk));
        match run_chunk_native(&chunk, |_| {}).expect("run") {
            VMResult::Ok(Value::Undef) => {}
            other => panic!("expected Undef from div-by-zero, got {other:?}"),
        }
        assert_native_matches_interp(chunk);
    }

    #[test]
    fn native_div_deopt_reconstructs_slot_state() {
        // The deopt must write register-cached slots back to the VM so the
        // resumed interpreter sees them. Here slot 0 = 99 is set natively, then a
        // divide-by-zero forces deopt; the interpreter divides (→ Undef), pops
        // it, and the leftover 99 (which it must read correctly) is the result.
        let mut b = ChunkBuilder::new();
        b.emit(Op::LoadInt(99), 1);
        b.emit(Op::SetSlot(0), 1); // slot cached in a register natively
        b.emit(Op::LoadFloat(5.0), 1);
        b.emit(Op::LoadFloat(0.0), 1);
        b.emit(Op::Div, 1); // deopt; interpreter pushes Undef
        b.emit(Op::Pop, 1); // discard the Undef
        b.emit(Op::GetSlot(0), 1); // interpreter reads slot 0 → must be 99
        let chunk = b.build();
        assert!(native_lowerable(&chunk));
        match run_chunk_native(&chunk, |_| {}).expect("run") {
            VMResult::Ok(v) => assert_eq!(v, Value::Int(99), "slot survived deopt"),
            other => panic!("got {other:?}"),
        }
        assert_native_matches_interp(chunk);
    }

    #[test]
    fn native_differential_div() {
        // Random divisions incl. zero divisors: nonzero → native fdiv, zero →
        // deopt to interpreter (Undef). Both must match the interpreter.
        let mut seed: u64 = 0x53C5_AA21_99BE_4471;
        let mut next = || {
            seed = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = seed;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        };
        for _ in 0..300 {
            // small ints incl. 0 as divisor, so ~1/9 of cases deopt.
            let a = (next() % 21) as i64 - 10;
            let bdiv = (next() % 9) as i64 - 4;
            let mut b = ChunkBuilder::new();
            b.emit(Op::LoadInt(a), 1);
            b.emit(Op::LoadInt(bdiv), 1);
            b.emit(Op::Div, 1);
            let chunk = b.build();
            assert!(native_lowerable(&chunk));
            assert_native_matches_interp(chunk);
        }
    }

    #[test]
    fn native_integer_mod() {
        // Normal, plus the two guarded divisors (% 0 and % -1 both → 0).
        for (x, y, want) in [
            (7, 3, 1),
            (-7, 3, -1),  // truncated remainder (sign of dividend)
            (7, -3, 1),
            (10, 0, 0),   // interpreter's y==0 → 0
            (i64::MIN, -1, 0), // x % -1 == 0, and dodges the srem trap
            (123_456, 1000, 456),
        ] {
            let mut b = ChunkBuilder::new();
            b.emit(Op::LoadInt(x), 1);
            b.emit(Op::LoadInt(y), 1);
            b.emit(Op::Mod, 1);
            let chunk = b.build();
            assert!(native_lowerable(&chunk), "int mod should lower");
            assert_native_int(chunk, want);
        }

        // A float operand takes the fmod libcall path: 7.5 % 2.0 = 1.5.
        let mut b = ChunkBuilder::new();
        b.emit(Op::LoadFloat(7.5), 1);
        b.emit(Op::LoadFloat(2.0), 1);
        b.emit(Op::Mod, 1);
        let chunk = b.build();
        assert!(native_lowerable(&chunk), "float mod lowers via fmod");
        match run_chunk_native(&chunk, |_| {}).expect("native run") {
            VMResult::Ok(v) => assert_eq!(v, Value::Float(1.5)),
            other => panic!("expected Ok(1.5), got {other:?}"),
        }
    }

    #[test]
    fn native_float_math_ops() {
        // Each shim calls the same Rust method the interpreter does, so finite
        // results match bit-for-bit. Inputs chosen to stay in-domain (no NaN,
        // which would fail `==`).
        let cases: &[(Op, f64)] = &[
            (Op::AbsFloat, -3.5),
            (Op::SqrtFloat, 2.0),
            (Op::CeilFloat, 2.3),
            (Op::FloorFloat, 2.7),
            (Op::TruncFloat, -2.7),
            (Op::RoundFloat, 2.5), // ties to even → 2.0
            (Op::SinFloat, 1.0),
            (Op::CosFloat, 1.0),
            (Op::TanFloat, 1.0),
            (Op::AsinFloat, 0.5),
            (Op::AcosFloat, 0.5),
            (Op::AtanFloat, 2.0),
            (Op::SinhFloat, 1.0),
            (Op::CoshFloat, 1.0),
            (Op::TanhFloat, 1.0),
            (Op::ExpFloat, 1.5),
            (Op::LogFloat, 2.0),
            (Op::Log2Float, 8.0),
            (Op::Log10Float, 1000.0),
        ];
        for (op, input) in cases {
            let mut b = ChunkBuilder::new();
            b.emit(Op::LoadFloat(*input), 1);
            b.emit(op.clone(), 1);
            let chunk = b.build();
            assert!(native_lowerable(&chunk), "{op:?} should lower");
            assert_native_matches_interp(chunk);
        }

        // RoundFloat ties-to-even concretely: 2.5 → 2.0, 3.5 → 4.0.
        for (input, want) in [(2.5_f64, 2.0_f64), (3.5, 4.0)] {
            let mut b = ChunkBuilder::new();
            b.emit(Op::LoadFloat(input), 1);
            b.emit(Op::RoundFloat, 1);
            match run_chunk_native(&b.build(), |_| {}).expect("run") {
                VMResult::Ok(v) => assert_eq!(v, Value::Float(want)),
                other => panic!("got {other:?}"),
            }
        }
    }

    #[test]
    fn native_awk_div_mod() {
        // Good cases: float division and modulo.
        let mut b = ChunkBuilder::new();
        b.emit(Op::LoadFloat(7.0), 1);
        b.emit(Op::LoadFloat(2.0), 1);
        b.emit(Op::AwkDivJit, 1);
        let chunk = b.build();
        assert!(native_lowerable(&chunk));
        match run_chunk_native(&chunk, |_| {}).expect("run") {
            VMResult::Ok(v) => assert_eq!(v, Value::Float(3.5)),
            other => panic!("got {other:?}"),
        }

        let mut b = ChunkBuilder::new();
        b.emit(Op::LoadFloat(7.5), 1);
        b.emit(Op::LoadFloat(2.0), 1);
        b.emit(Op::AwkModJit, 1);
        match run_chunk_native(&b.build(), |_| {}).expect("run") {
            VMResult::Ok(v) => assert_eq!(v, Value::Float(1.5)),
            other => panic!("got {other:?}"),
        }

        // Divide-by-zero takes the native error-return branch; the error message
        // must match the interpreter's exactly.
        let mut b = ChunkBuilder::new();
        b.emit(Op::LoadFloat(5.0), 1);
        b.emit(Op::LoadFloat(0.0), 1);
        b.emit(Op::AwkDivJit, 1);
        let chunk = b.build();
        assert!(native_lowerable(&chunk));
        match run_chunk_native(&chunk, |_| {}).expect("run") {
            VMResult::Error(e) => assert_eq!(e, "division by zero attempted"),
            other => panic!("expected div-by-zero error, got {other:?}"),
        }
        assert_native_matches_interp(chunk);

        // Modulo by zero: the `%` variant's distinct message.
        let mut b = ChunkBuilder::new();
        b.emit(Op::LoadFloat(5.0), 1);
        b.emit(Op::LoadFloat(0.0), 1);
        b.emit(Op::AwkModJit, 1);
        let chunk = b.build();
        match run_chunk_native(&chunk, |_| {}).expect("run") {
            VMResult::Error(e) => assert_eq!(e, "division by zero attempted in `%'"),
            other => panic!("expected mod-by-zero error, got {other:?}"),
        }
        assert_native_matches_interp(chunk);
    }

    #[test]
    fn native_sink_print() {
        // A chunk with a heap/IO op (Print) now lowers natively: the arithmetic
        // runs in registers, Print is spilled to the boxed stack and shimmed.
        // We assert via a *leftover* value beneath the printed args, which proves
        // the sink popped exactly `n` (stdout itself isn't capturable here).

        // 99, (5+3=8), PrintLn(1) → prints "8", pops it, leaves 99.
        let mut b = ChunkBuilder::new();
        b.emit(Op::LoadInt(99), 1);
        b.emit(Op::LoadInt(5), 1);
        b.emit(Op::LoadInt(3), 1);
        b.emit(Op::Add, 1);
        b.emit(Op::PrintLn(1), 1);
        let chunk = b.build();
        assert!(native_lowerable(&chunk), "scalar+Print should now lower");
        assert_native_int(chunk, 99);

        // Print(2) pops two; leftover 7 remains.
        let mut b = ChunkBuilder::new();
        b.emit(Op::LoadInt(7), 1);
        b.emit(Op::LoadInt(1), 1);
        b.emit(Op::LoadInt(2), 1);
        b.emit(Op::Print(2), 1);
        assert_native_int(b.build(), 7);

        // Float and bool args exercise the per-kind spill helpers.
        let mut b = ChunkBuilder::new();
        b.emit(Op::LoadInt(42), 1);
        b.emit(Op::LoadFloat(2.5), 1);
        b.emit(Op::PrintLn(1), 1);
        assert_native_int(b.build(), 42);

        let mut b = ChunkBuilder::new();
        b.emit(Op::LoadInt(42), 1);
        b.emit(Op::LoadInt(1), 1);
        b.emit(Op::LoadInt(2), 1);
        b.emit(Op::NumLt, 1); // Bool
        b.emit(Op::PrintLn(1), 1);
        let chunk = b.build();
        assert!(native_lowerable(&chunk));
        assert_native_int(chunk, 42);

        // And the result still matches the interpreter (which prints identically).
        let mut b = ChunkBuilder::new();
        b.emit(Op::LoadInt(10), 1);
        b.emit(Op::LoadInt(20), 1);
        b.emit(Op::Add, 1);
        b.emit(Op::PrintLn(1), 1);
        assert_native_matches_interp(b.build());
    }

    #[test]
    fn native_getstatus_source() {
        // Bare `$?` result is a Status; with no prior SetStatus the code is 0.
        let mut b = ChunkBuilder::new();
        b.emit(Op::GetStatus, 1);
        let chunk = b.build();
        assert!(native_lowerable(&chunk), "GetStatus should lower");
        match run_chunk_native(&chunk, |_| {}).expect("run") {
            VMResult::Ok(v) => assert_eq!(v, Value::Status(0)),
            other => panic!("got {other:?}"),
        }

        // Set $? = 3, then return it → Status(3).
        let mut b = ChunkBuilder::new();
        b.emit(Op::LoadInt(3), 1);
        b.emit(Op::SetStatus, 1);
        b.emit(Op::GetStatus, 1);
        let chunk = b.build();
        assert!(native_lowerable(&chunk));
        match run_chunk_native(&chunk, |_| {}).expect("run") {
            VMResult::Ok(v) => assert_eq!(v, Value::Status(3)),
            other => panic!("got {other:?}"),
        }

        // Status float-promotes in arithmetic: $?(=0) + 2.5 → Float(2.5).
        let mut b = ChunkBuilder::new();
        b.emit(Op::GetStatus, 1);
        b.emit(Op::LoadFloat(2.5), 1);
        b.emit(Op::Add, 1);
        assert_native_matches_interp(b.build());
    }

    #[test]
    fn native_getstatus_inverted_truthiness() {
        // Status truthiness is INVERTED: code 0 is truthy (shell success).
        // `if $? { 10 } else { 20 }` with $?=0 (success) → takes the truthy arm.
        let build = |code: i64| {
            let mut b = ChunkBuilder::new();
            b.emit(Op::LoadInt(code), 1);
            b.emit(Op::SetStatus, 1);
            b.emit(Op::GetStatus, 1);
            let jf = b.emit(Op::JumpIfFalse(0), 1);
            b.emit(Op::LoadInt(10), 1); // taken when $? truthy (code == 0)
            let j = b.emit(Op::Jump(0), 1);
            let else_ip = b.emit(Op::LoadInt(20), 1);
            let end = b.current_pos();
            b.patch_jump(jf, else_ip);
            b.patch_jump(j, end);
            b.build()
        };
        // code 0 → truthy → 10; code 1 → falsy → 20. Match the interpreter.
        for code in [0, 1, 2, -1] {
            let chunk = build(code);
            assert!(native_lowerable(&chunk));
            assert_native_matches_interp(chunk);
        }
    }

    #[test]
    fn native_differential_getstatus() {
        // $? set to a random code, then used in arithmetic / comparison /
        // branch / as result — all must match the interpreter (esp. truthiness).
        let mut seed: u64 = 0xABCD_1234_5678_9F01;
        let mut next = || {
            seed = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = seed;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        };
        for _ in 0..300 {
            let code = (next() % 5) as i64 - 2; // small incl. 0
            let mut b = ChunkBuilder::new();
            b.emit(Op::LoadInt(code), 1);
            b.emit(Op::SetStatus, 1);
            b.emit(Op::GetStatus, 1);
            match next() % 4 {
                0 => {
                    // arithmetic: $? + k
                    b.emit(Op::LoadInt((next() % 7) as i64 - 3), 1);
                    b.emit(Op::Add, 1);
                }
                1 => {
                    // comparison feeding a branch
                    b.emit(Op::LoadInt(0), 1);
                    b.emit(Op::NumGt, 1);
                    let jf = b.emit(Op::JumpIfFalse(0), 1);
                    b.emit(Op::LoadInt(1), 1);
                    let j = b.emit(Op::Jump(0), 1);
                    let e = b.emit(Op::LoadInt(2), 1);
                    let end = b.current_pos();
                    b.patch_jump(jf, e);
                    b.patch_jump(j, end);
                }
                2 => {
                    // direct truthiness branch (inverted)
                    let jf = b.emit(Op::JumpIfFalse(0), 1);
                    b.emit(Op::LoadInt(100), 1);
                    let j = b.emit(Op::Jump(0), 1);
                    let e = b.emit(Op::LoadInt(200), 1);
                    let end = b.current_pos();
                    b.patch_jump(jf, e);
                    b.patch_jump(j, end);
                }
                _ => { /* bare $? result */ }
            }
            let chunk = b.build();
            if native_lowerable(&chunk) {
                assert_native_matches_interp(chunk);
            }
        }
    }

    #[test]
    fn native_source_awk_field() {
        // AwkGetFieldNum is a source whose result is statically Float; it reloads
        // into a register with no guard. With no awk host the field hook yields
        // 0.0, so $1 + 2.0 == 2.0 — and it must match the interpreter (same hook).
        let mut b = ChunkBuilder::new();
        b.emit(Op::AwkGetFieldNum(1), 1);
        b.emit(Op::LoadFloat(2.0), 1);
        b.emit(Op::Add, 1);
        let chunk = b.build();
        assert!(native_lowerable(&chunk), "awk field source should lower");
        match run_chunk_native(&chunk, |_| {}).expect("run") {
            VMResult::Ok(v) => assert_eq!(v, Value::Float(2.0)),
            other => panic!("got {other:?}"),
        }
        assert_native_matches_interp(chunk);

        // The reloaded value feeds further native arithmetic and a comparison.
        let mut b = ChunkBuilder::new();
        b.emit(Op::AwkGetFieldNum(2), 1);
        b.emit(Op::AwkGetFieldNum(3), 1);
        b.emit(Op::NumLt, 1); // 0.0 < 0.0 → false
        b.emit(Op::LoadInt(5), 1);
        b.emit(Op::Add, 1); // false(0) + 5 = 5  (bool promotes via... )
        let chunk = b.build();
        // NumLt result is Bool; Bool + Int isn't lowered (Add needs numeric), so
        // this particular chunk falls back — still must match the interpreter.
        assert_native_matches_interp(chunk);
    }

    #[test]
    fn native_setstatus_sink() {
        // SetStatus is a silent sink (no stdout): pops one, leaves 77.
        let mut b = ChunkBuilder::new();
        b.emit(Op::LoadInt(77), 1);
        b.emit(Op::LoadInt(0), 1);
        b.emit(Op::SetStatus, 1);
        let chunk = b.build();
        assert!(native_lowerable(&chunk), "SetStatus should lower as a sink");
        assert_native_int(chunk, 77);
        // Result also matches the interpreter (both set $? and leave 77).
        let mut b = ChunkBuilder::new();
        b.emit(Op::LoadInt(77), 1);
        b.emit(Op::LoadInt(0), 1);
        b.emit(Op::SetStatus, 1);
        assert_native_matches_interp(b.build());
    }

    #[test]
    fn native_loop_with_print_lowers() {
        // sum=0; for i in 0..3 { print(i); sum += i }; return sum → 3, fully
        // native incl. the per-iteration sink. (Prints 0,1,2 to stdout.)
        let mut b = ChunkBuilder::new();
        b.emit(Op::LoadInt(0), 1);
        b.emit(Op::SetSlot(0), 1); // sum
        b.emit(Op::LoadInt(0), 1);
        b.emit(Op::SetSlot(1), 1); // i
        let top = b.current_pos();
        b.emit(Op::GetSlot(1), 1);
        b.emit(Op::LoadInt(3), 1);
        b.emit(Op::NumLt, 1);
        let exit = b.emit(Op::JumpIfFalse(0), 1);
        b.emit(Op::GetSlot(1), 1);
        b.emit(Op::Print(1), 1); // sink inside the hot loop
        b.emit(Op::GetSlot(0), 1);
        b.emit(Op::GetSlot(1), 1);
        b.emit(Op::Add, 1);
        b.emit(Op::SetSlot(0), 1);
        b.emit(Op::GetSlot(1), 1);
        b.emit(Op::LoadInt(1), 1);
        b.emit(Op::Add, 1);
        b.emit(Op::SetSlot(1), 1);
        b.emit(Op::Jump(top), 1);
        let end = b.current_pos();
        b.patch_jump(exit, end);
        b.emit(Op::GetSlot(0), 1);
        let chunk = b.build();
        assert!(native_lowerable(&chunk), "loop with embedded sink should lower");
        assert_native_int(chunk, 3); // 0+1+2
    }

    #[test]
    fn native_awk_sqrt_log() {
        // sqrt(16) = 4.0 exactly.
        let mut b = ChunkBuilder::new();
        b.emit(Op::LoadFloat(16.0), 1);
        b.emit(Op::AwkSqrtJit, 1);
        assert_native_int_or_float(b.build(), 4.0);

        // Non-negative inputs match the interpreter (finite results / -inf).
        for x in [2.0_f64, 0.5, 100.0, 1.0, 0.0] {
            for op in [Op::AwkSqrtJit, Op::AwkLogJit] {
                let mut b = ChunkBuilder::new();
                b.emit(Op::LoadFloat(x), 1);
                b.emit(op, 1);
                let chunk = b.build();
                assert!(native_lowerable(&chunk));
                assert_native_matches_interp(chunk);
            }
        }

        // Negative input warns (to stderr) and yields NaN; verify the NaN result
        // (the `==`-based diff harness can't compare NaN).
        for op in [Op::AwkSqrtJit, Op::AwkLogJit] {
            let mut b = ChunkBuilder::new();
            b.emit(Op::LoadFloat(-4.0), 1);
            b.emit(op, 1);
            match run_chunk_native(&b.build(), |_| {}).expect("run") {
                VMResult::Ok(Value::Float(v)) => assert!(v.is_nan(), "expected NaN"),
                other => panic!("expected NaN float, got {other:?}"),
            }
        }
    }

    #[test]
    fn native_awk_div_mod_nonjit() {
        // The non-Jit AwkDiv/AwkMod lower identically to the Jit variants.
        let mut b = ChunkBuilder::new();
        b.emit(Op::LoadFloat(7.0), 1);
        b.emit(Op::LoadFloat(2.0), 1);
        b.emit(Op::AwkDiv, 1);
        let chunk = b.build();
        assert!(native_lowerable(&chunk));
        match run_chunk_native(&chunk, |_| {}).expect("run") {
            VMResult::Ok(v) => assert_eq!(v, Value::Float(3.5)),
            other => panic!("got {other:?}"),
        }

        let mut b = ChunkBuilder::new();
        b.emit(Op::LoadFloat(7.5), 1);
        b.emit(Op::LoadFloat(2.0), 1);
        b.emit(Op::AwkMod, 1);
        match run_chunk_native(&b.build(), |_| {}).expect("run") {
            VMResult::Ok(v) => assert_eq!(v, Value::Float(1.5)),
            other => panic!("got {other:?}"),
        }

        // Divide-by-zero error-returns with the exact interpreter message.
        for (op, msg) in [
            (Op::AwkDiv, "division by zero attempted"),
            (Op::AwkMod, "division by zero attempted in `%'"),
        ] {
            let mut b = ChunkBuilder::new();
            b.emit(Op::LoadFloat(5.0), 1);
            b.emit(Op::LoadFloat(0.0), 1);
            b.emit(op, 1);
            let chunk = b.build();
            assert!(native_lowerable(&chunk));
            match run_chunk_native(&chunk, |_| {}).expect("run") {
                VMResult::Error(e) => assert_eq!(e, msg),
                other => panic!("expected error, got {other:?}"),
            }
            assert_native_matches_interp(chunk);
        }
    }

    #[test]
    fn native_awk_bitwise() {
        // lshift(3, 4) = 48; rshift(48, 4) = 3; compl(0) = -1.
        let bin = |a: f64, n: f64, op: Op| {
            let mut b = ChunkBuilder::new();
            b.emit(Op::LoadFloat(a), 1);
            b.emit(Op::LoadFloat(n), 1);
            b.emit(op, 1);
            b.build()
        };
        assert_native_int_or_float(bin(3.0, 4.0, Op::AwkLshiftJit), 48.0);
        assert_native_int_or_float(bin(48.0, 4.0, Op::AwkRshiftJit), 3.0);

        let mut b = ChunkBuilder::new();
        b.emit(Op::LoadFloat(0.0), 1);
        b.emit(Op::AwkComplJit, 1);
        assert_native_int_or_float(b.build(), -1.0); // !0 == -1

        // Negative operands error with the exact interpreter message.
        for (op, msg) in [
            (Op::AwkLshiftJit, "lshift: negative values are not allowed"),
            (Op::AwkRshiftJit, "rshift: negative values are not allowed"),
        ] {
            let mut b = ChunkBuilder::new();
            b.emit(Op::LoadFloat(-1.0), 1);
            b.emit(Op::LoadFloat(2.0), 1);
            b.emit(op, 1);
            let chunk = b.build();
            match run_chunk_native(&chunk, |_| {}).expect("run") {
                VMResult::Error(e) => assert_eq!(e, msg),
                other => panic!("expected error, got {other:?}"),
            }
            assert_native_matches_interp(chunk);
        }
        let mut b = ChunkBuilder::new();
        b.emit(Op::LoadFloat(-1.0), 1);
        b.emit(Op::AwkComplJit, 1);
        let chunk = b.build();
        match run_chunk_native(&chunk, |_| {}).expect("run") {
            VMResult::Error(e) => assert_eq!(e, "compl: negative value is not allowed"),
            other => panic!("expected error, got {other:?}"),
        }
        assert_native_matches_interp(chunk);
    }

    #[test]
    fn native_differential_awk_bitwise() {
        // Non-negative float operands (so no error): the int↔float conversion
        // and shift semantics must match the interpreter exactly.
        let mut seed: u64 = 0x84A3_11C5_7E29_B6F1;
        let mut next = || {
            seed = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = seed;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        };
        for _ in 0..300 {
            let a = (next() % 1_000_000) as f64;
            let n = (next() % 70) as f64; // spans the 0x3f mask boundary
            let mut b = ChunkBuilder::new();
            b.emit(Op::LoadFloat(a), 1);
            if next() & 1 == 0 {
                b.emit(Op::LoadFloat(n), 1);
                b.emit(if next() & 2 == 0 { Op::AwkLshiftJit } else { Op::AwkRshiftJit }, 1);
            } else {
                b.emit(Op::AwkComplJit, 1);
            }
            let chunk = b.build();
            assert!(native_lowerable(&chunk));
            assert_native_matches_interp(chunk);
        }
    }

    #[test]
    fn native_gcd_lcm() {
        let gcd = |a: i64, bb: i64| {
            let mut b = ChunkBuilder::new();
            b.emit(Op::LoadInt(a), 1);
            b.emit(Op::LoadInt(bb), 1);
            b.emit(Op::GcdInt, 1);
            b.build()
        };
        assert_native_int(gcd(12, 18), 6);
        assert_native_int(gcd(17, 5), 1);
        assert_native_int(gcd(0, 5), 5);
        assert_native_int(gcd(-12, 18), 6); // unsigned_abs

        let lcm = |a: i64, bb: i64| {
            let mut b = ChunkBuilder::new();
            b.emit(Op::LoadInt(a), 1);
            b.emit(Op::LoadInt(bb), 1);
            b.emit(Op::LcmInt, 1);
            b.build()
        };
        assert_native_int(lcm(4, 6), 12);
        assert_native_int(lcm(21, 6), 42);
        assert_native_int(lcm(0, 5), 0);
        assert_native_int(lcm(i64::MAX, 2), i64::MAX); // capped at i64::MAX
    }

    #[test]
    fn native_differential_gcd_lcm() {
        // Full-range random pairs: exercises unsigned_abs of negatives,
        // i64::MIN, and lcm's saturating multiply — all must match the interp.
        let mut seed: u64 = 0xD1B5_4A32_D192_ED03;
        let mut next = || {
            seed = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = seed;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        };
        for _ in 0..300 {
            let a = next() as i64;
            let bb = next() as i64;
            let op = if next() & 1 == 0 { Op::GcdInt } else { Op::LcmInt };
            let mut b = ChunkBuilder::new();
            b.emit(Op::LoadInt(a), 1);
            b.emit(Op::LoadInt(bb), 1);
            b.emit(op, 1);
            let chunk = b.build();
            assert!(native_lowerable(&chunk));
            assert_native_matches_interp(chunk);
        }
    }

    #[test]
    fn native_int_math_ops() {
        // TruncInt of a float truncates toward zero; AbsInt is wrapping abs.
        let mut b = ChunkBuilder::new();
        b.emit(Op::LoadFloat(3.9), 1);
        b.emit(Op::TruncInt, 1);
        assert_native_int(b.build(), 3);

        let mut b = ChunkBuilder::new();
        b.emit(Op::LoadFloat(-3.9), 1);
        b.emit(Op::TruncInt, 1);
        assert_native_int(b.build(), -3);

        let mut b = ChunkBuilder::new();
        b.emit(Op::LoadInt(-7), 1);
        b.emit(Op::AbsInt, 1);
        assert_native_int(b.build(), 7);

        // wrapping: abs(i64::MIN) == i64::MIN
        let mut b = ChunkBuilder::new();
        b.emit(Op::LoadInt(i64::MIN), 1);
        b.emit(Op::AbsInt, 1);
        assert_native_int(b.build(), i64::MIN);
    }

    #[test]
    fn native_binary_float_math() {
        // PowFloat: 2 ** 10 = 1024.0.
        let mut b = ChunkBuilder::new();
        b.emit(Op::LoadFloat(2.0), 1);
        b.emit(Op::LoadFloat(10.0), 1);
        b.emit(Op::PowFloat, 1);
        match run_chunk_native(&b.build(), |_| {}).expect("run") {
            VMResult::Ok(v) => assert_eq!(v, Value::Float(1024.0)),
            other => panic!("got {other:?}"),
        }

        // Atan2Float: operand order y (lower), x (top) → y.atan2(x).
        let mut b = ChunkBuilder::new();
        b.emit(Op::LoadFloat(2.0), 1); // y
        b.emit(Op::LoadFloat(3.0), 1); // x
        b.emit(Op::Atan2Float, 1);
        assert_native_matches_interp(b.build());
    }

    #[test]
    fn native_pow() {
        // 2 ** 10 = 1024.0 (always Float, via powf).
        let mut b = ChunkBuilder::new();
        b.emit(Op::LoadInt(2), 1);
        b.emit(Op::LoadInt(10), 1);
        b.emit(Op::Pow, 1);
        let chunk = b.build();
        assert!(native_lowerable(&chunk), "pow lowers via powf");
        match run_chunk_native(&chunk, |_| {}).expect("native run") {
            VMResult::Ok(v) => assert_eq!(v, Value::Float(1024.0)),
            other => panic!("expected Ok(1024.0), got {other:?}"),
        }

        // Mixed/float base: 9.0 ** 0.5 = 3.0; matches the interpreter exactly.
        let mut b = ChunkBuilder::new();
        b.emit(Op::LoadFloat(9.0), 1);
        b.emit(Op::LoadFloat(0.5), 1);
        b.emit(Op::Pow, 1);
        let chunk = b.build();
        assert!(native_lowerable(&chunk));
        assert_native_matches_interp(chunk);
    }

    #[test]
    fn native_globals_round_trip() {
        // SetVar/GetVar: x = 42; return x → 42.
        let mut b = ChunkBuilder::new();
        b.emit(Op::LoadInt(42), 1);
        b.emit(Op::SetVar(0), 1);
        b.emit(Op::GetVar(0), 1);
        assert_native_int(b.build(), 42);

        // DeclareVar behaves like SetVar.
        let mut b = ChunkBuilder::new();
        b.emit(Op::LoadInt(7), 1);
        b.emit(Op::DeclareVar(0), 1);
        b.emit(Op::GetVar(0), 1);
        assert_native_int(b.build(), 7);

        // Two globals in arithmetic: 10 + 5 = 15.
        let mut b = ChunkBuilder::new();
        b.emit(Op::LoadInt(10), 1);
        b.emit(Op::SetVar(0), 1);
        b.emit(Op::LoadInt(5), 1);
        b.emit(Op::SetVar(1), 1);
        b.emit(Op::GetVar(0), 1);
        b.emit(Op::GetVar(1), 1);
        b.emit(Op::Add, 1);
        assert_native_int(b.build(), 15);
    }

    #[test]
    fn native_uninit_global_read_falls_back() {
        // Reading an unassigned global yields `Undef` in the interpreter, so it
        // must fall back; the threaded path still matches.
        let mut b = ChunkBuilder::new();
        b.emit(Op::GetVar(0), 1);
        let chunk = b.build();
        assert!(!native_lowerable(&chunk), "unset global read must fall back");
        assert_native_matches_interp(chunk);
    }

    #[test]
    fn native_slot_incdec_superops() {
        // PreIncSlot: 5 → slot=6, push 6.
        let mut b = ChunkBuilder::new();
        b.emit(Op::LoadInt(5), 1);
        b.emit(Op::SetSlot(0), 1);
        b.emit(Op::PreIncSlot(0), 1);
        assert_native_int(b.build(), 6);

        // PostIncSlot: push old 5; slot becomes 6 (observed via GetSlot).
        let mut b = ChunkBuilder::new();
        b.emit(Op::LoadInt(5), 1);
        b.emit(Op::SetSlot(0), 1);
        b.emit(Op::PostIncSlot(0), 1);
        assert_native_int(b.build(), 5);
        let mut b = ChunkBuilder::new();
        b.emit(Op::LoadInt(5), 1);
        b.emit(Op::SetSlot(0), 1);
        b.emit(Op::PostIncSlot(0), 1);
        b.emit(Op::Pop, 1);
        b.emit(Op::GetSlot(0), 1);
        assert_native_int(b.build(), 6);

        // PreDecSlot: 5 → 4. PostDecSlot: push old 5.
        let mut b = ChunkBuilder::new();
        b.emit(Op::LoadInt(5), 1);
        b.emit(Op::SetSlot(0), 1);
        b.emit(Op::PreDecSlot(0), 1);
        assert_native_int(b.build(), 4);
        let mut b = ChunkBuilder::new();
        b.emit(Op::LoadInt(5), 1);
        b.emit(Op::SetSlot(0), 1);
        b.emit(Op::PostDecSlot(0), 1);
        assert_native_int(b.build(), 5);

        // PreIncSlotVoid on an *unassigned* slot: coerces to 0, becomes 1; the
        // op also marks the slot assigned so the later GetSlot lowers.
        let mut b = ChunkBuilder::new();
        b.emit(Op::PreIncSlotVoid(0), 1);
        b.emit(Op::GetSlot(0), 1);
        assert_native_int(b.build(), 1);

        // AddAssignSlotVoid: slot0(10) += slot1(7) → 17.
        let mut b = ChunkBuilder::new();
        b.emit(Op::LoadInt(10), 1);
        b.emit(Op::SetSlot(0), 1);
        b.emit(Op::LoadInt(7), 1);
        b.emit(Op::SetSlot(1), 1);
        b.emit(Op::AddAssignSlotVoid(0, 1), 1);
        b.emit(Op::GetSlot(0), 1);
        assert_native_int(b.build(), 17);
    }

    #[test]
    fn native_accum_sum_loop() {
        // AccumSumLoop runs `while i < 5 { sum += i; i += 1 }` internally:
        // sum = 0+1+2+3+4 = 10, and i ends at 5.
        let mut b = ChunkBuilder::new();
        b.emit(Op::LoadInt(0), 1);
        b.emit(Op::SetSlot(0), 1); // sum
        b.emit(Op::LoadInt(0), 1);
        b.emit(Op::SetSlot(1), 1); // i
        b.emit(Op::AccumSumLoop(0, 1, 5), 1);
        b.emit(Op::GetSlot(0), 1);
        assert_native_int(b.build(), 10);

        // The counter slot is left at the limit.
        let mut b = ChunkBuilder::new();
        b.emit(Op::LoadInt(0), 1);
        b.emit(Op::SetSlot(0), 1);
        b.emit(Op::LoadInt(0), 1);
        b.emit(Op::SetSlot(1), 1);
        b.emit(Op::AccumSumLoop(0, 1, 5), 1);
        b.emit(Op::GetSlot(1), 1);
        assert_native_int(b.build(), 5);
    }

    #[test]
    fn native_fused_loop_ops() {
        // sum=0; i=0; if i<5 { do { sum+=i } while (++i < 5) }; return sum  → 10
        // built with the fused loop super-ops. Targets are computed by hand
        // since `patch_jump` only handles the plain jump ops.
        //  0 LoadInt 0           6 GetSlot 1
        //  1 SetSlot 0 (sum)     7 Add
        //  2 LoadInt 0           8 SetSlot 0
        //  3 SetSlot 1 (i)       9 SlotIncLtIntJumpBack(1,5,5)
        //  4 SlotLtIntJumpIfFalse(1,5,10)   10 GetSlot 0
        let mut b = ChunkBuilder::new();
        b.emit(Op::LoadInt(0), 1);
        b.emit(Op::SetSlot(0), 1);
        b.emit(Op::LoadInt(0), 1);
        b.emit(Op::SetSlot(1), 1);
        b.emit(Op::SlotLtIntJumpIfFalse(1, 5, 10), 1);
        b.emit(Op::GetSlot(0), 1);
        b.emit(Op::GetSlot(1), 1);
        b.emit(Op::Add, 1);
        b.emit(Op::SetSlot(0), 1);
        b.emit(Op::SlotIncLtIntJumpBack(1, 5, 5), 1);
        b.emit(Op::GetSlot(0), 1);
        let chunk = b.build();
        assert!(native_lowerable(&chunk), "fused loop should lower natively");
        assert_native_int(chunk, 10);
    }

    #[test]
    fn native_stack_shuffles() {
        // Swap: [3,7] → [7,3]; 7 - 3 = 4.
        let mut b = ChunkBuilder::new();
        b.emit(Op::LoadInt(3), 1);
        b.emit(Op::LoadInt(7), 1);
        b.emit(Op::Swap, 1);
        b.emit(Op::Sub, 1);
        assert_native_int(b.build(), 4);

        // Dup2: [2,5] → [2,5,2,5]; +,+,+ = 14.
        let mut b = ChunkBuilder::new();
        b.emit(Op::LoadInt(2), 1);
        b.emit(Op::LoadInt(5), 1);
        b.emit(Op::Dup2, 1);
        b.emit(Op::Add, 1);
        b.emit(Op::Add, 1);
        b.emit(Op::Add, 1);
        assert_native_int(b.build(), 14);

        // Rot: [1,2,3] → [2,3,1]; 3-1=2, 2-2=0.
        let mut b = ChunkBuilder::new();
        b.emit(Op::LoadInt(1), 1);
        b.emit(Op::LoadInt(2), 1);
        b.emit(Op::LoadInt(3), 1);
        b.emit(Op::Rot, 1);
        b.emit(Op::Sub, 1);
        b.emit(Op::Sub, 1);
        assert_native_int(b.build(), 0);
    }

    #[test]
    fn native_logical_and_spaceship() {
        // LogAnd(1,2) = true; Inc → 2 (Bool is int-like for Inc).
        let mut b = ChunkBuilder::new();
        b.emit(Op::LoadInt(1), 1);
        b.emit(Op::LoadInt(2), 1);
        b.emit(Op::LogAnd, 1);
        b.emit(Op::Inc, 1);
        assert_native_int(b.build(), 2);

        // LogOr(0,0) = false; Inc → 1.
        let mut b = ChunkBuilder::new();
        b.emit(Op::LoadInt(0), 1);
        b.emit(Op::LoadInt(0), 1);
        b.emit(Op::LogOr, 1);
        b.emit(Op::Inc, 1);
        assert_native_int(b.build(), 1);

        // Spaceship: 3 <=> 7 = -1.
        let mut b = ChunkBuilder::new();
        b.emit(Op::LoadInt(3), 1);
        b.emit(Op::LoadInt(7), 1);
        b.emit(Op::Spaceship, 1);
        assert_native_int(b.build(), -1);

        // Float spaceship: 5.0 <=> 5.0 = 0; mixed int/float promotes.
        let mut b = ChunkBuilder::new();
        b.emit(Op::LoadInt(5), 1);
        b.emit(Op::LoadFloat(5.0), 1);
        b.emit(Op::Spaceship, 1);
        assert_native_int(b.build(), 0);
    }

    #[test]
    fn native_keep_jumps() {
        // JumpIfTrueKeep: 5 is truthy ⇒ jump keeping 5 ⇒ result 5.
        let mut b = ChunkBuilder::new();
        b.emit(Op::LoadInt(5), 1);
        let jt = b.emit(Op::JumpIfTrueKeep(0), 1);
        b.emit(Op::Pop, 1); // not-taken path rebuilds a depth-1 Int stack
        b.emit(Op::LoadInt(20), 1);
        let l = b.current_pos();
        b.patch_jump(jt, l);
        assert_native_int(b.build(), 5);

        // JumpIfFalseKeep: 0 is falsy ⇒ jump keeping 0 ⇒ result 0.
        let mut b = ChunkBuilder::new();
        b.emit(Op::LoadInt(0), 1);
        let jf = b.emit(Op::JumpIfFalseKeep(0), 1);
        b.emit(Op::Pop, 1);
        b.emit(Op::LoadInt(7), 1);
        let l = b.current_pos();
        b.patch_jump(jf, l);
        assert_native_int(b.build(), 0);
    }

    #[test]
    fn native_float_arithmetic() {
        // 1.5 * 2.0 + 1.0  →  4.0, fully native.
        let mut b = ChunkBuilder::new();
        b.emit(Op::LoadFloat(1.5), 1);
        b.emit(Op::LoadFloat(2.0), 1);
        b.emit(Op::Mul, 1);
        b.emit(Op::LoadFloat(1.0), 1);
        b.emit(Op::Add, 1);
        let chunk = b.build();
        assert!(native_lowerable(&chunk));
        match run_chunk_native(&chunk, |_| {}).expect("native run") {
            VMResult::Ok(v) => assert_eq!(v, Value::Float(4.0)),
            other => panic!("expected Ok(4.0), got {other:?}"),
        }
    }

    #[test]
    fn native_mixed_int_float_promotion() {
        // 3 + 0.5  →  Float(3.5): Int promotes to float, mirroring the interp.
        let mut b = ChunkBuilder::new();
        b.emit(Op::LoadInt(3), 1);
        b.emit(Op::LoadFloat(0.5), 1);
        b.emit(Op::Add, 1);
        let chunk = b.build();
        assert!(native_lowerable(&chunk));
        assert_native_matches_interp(chunk);
    }

    #[test]
    fn native_float_comparison_branch() {
        // if (1.5 < 2.5) 3.0 else 4.0  →  3.0
        let mut b = ChunkBuilder::new();
        b.emit(Op::LoadFloat(1.5), 1);
        b.emit(Op::LoadFloat(2.5), 1);
        b.emit(Op::NumLt, 1);
        let jf = b.emit(Op::JumpIfFalse(0), 1);
        b.emit(Op::LoadFloat(3.0), 1);
        let j = b.emit(Op::Jump(0), 1);
        let else_ip = b.emit(Op::LoadFloat(4.0), 1);
        let end_ip = b.current_pos();
        b.patch_jump(jf, else_ip);
        b.patch_jump(j, end_ip);
        let chunk = b.build();
        assert!(native_lowerable(&chunk));
        assert_native_matches_interp(chunk);
    }

    #[test]
    fn native_float_const_pool() {
        // LoadConst of a numeric constant lowers; the float result must match.
        let mut b = ChunkBuilder::new();
        let c = b.add_constant(Value::Float(2.25));
        b.emit(Op::LoadConst(c), 1);
        b.emit(Op::LoadFloat(4.0), 1);
        b.emit(Op::Mul, 1);
        let chunk = b.build();
        assert!(native_lowerable(&chunk), "numeric const should lower");
        assert_native_matches_interp(chunk);
    }

    #[test]
    fn native_differential_floats() {
        // Deterministic differential fuzz over mixed int/float arithmetic and
        // float comparisons. Values stay small and use only + - *, so no NaN
        // is produced and native/interp f64 bits match exactly.
        let mut seed: u64 = 0x2545_F491_4F6C_DD1D;
        let mut next = || {
            seed = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = seed;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        };

        // Push a numeric literal: roughly half int, half (quarter-valued) float.
        fn push_num(b: &mut ChunkBuilder, next: &mut impl FnMut() -> u64) {
            if next() % 2 == 0 {
                b.emit(Op::LoadInt((next() % 13) as i64 - 6), 1);
            } else {
                let q = (next() % 25) as f64 / 4.0 - 3.0; // multiples of 0.25
                b.emit(Op::LoadFloat(q), 1);
            }
        }
        fn push_binop(b: &mut ChunkBuilder, next: &mut impl FnMut() -> u64) {
            match next() % 3 {
                0 => b.emit(Op::Add, 1),
                1 => b.emit(Op::Sub, 1),
                _ => b.emit(Op::Mul, 1),
            };
        }
        fn emit_expr(b: &mut ChunkBuilder, next: &mut impl FnMut() -> u64, terms: usize) {
            push_num(b, next);
            for _ in 0..terms {
                push_num(b, next);
                push_binop(b, next);
            }
        }

        // A branch whose two arms leave different kinds (Int vs Float) is a
        // legitimate join mismatch the native path rejects (it can't merge an
        // i64 and an f64 into one stack slot); such chunks fall back. So the
        // property is "if it lowers, it matches" — we also count lowerings to
        // ensure the test is not vacuous.
        let mut lowered = 0;
        for _ in 0..400 {
            let mut b = ChunkBuilder::new();
            if next() % 2 == 0 {
                let terms = (next() % 4) as usize;
                emit_expr(&mut b, &mut next, terms);
            } else {
                // if (num <cmp> num) <expr> else <expr>
                push_num(&mut b, &mut next);
                push_num(&mut b, &mut next);
                match next() % 4 {
                    0 => b.emit(Op::NumLt, 1),
                    1 => b.emit(Op::NumGt, 1),
                    2 => b.emit(Op::NumLe, 1),
                    _ => b.emit(Op::NumGe, 1),
                };
                let jf = b.emit(Op::JumpIfFalse(0), 1);
                let then_terms = (next() % 3) as usize;
                emit_expr(&mut b, &mut next, then_terms);
                let j = b.emit(Op::Jump(0), 1);
                let else_ip = b.current_pos();
                b.patch_jump(jf, else_ip);
                let else_terms = (next() % 3) as usize;
                emit_expr(&mut b, &mut next, else_terms);
                let end_ip = b.current_pos();
                b.patch_jump(j, end_ip);
            }
            let chunk = b.build();
            if native_lowerable(&chunk) {
                lowered += 1;
                assert_native_matches_interp(chunk);
            }
        }
        assert!(lowered > 50, "expected many float chunks to lower, got {lowered}");
    }

    #[test]
    fn native_empty_chunk() {
        let b = ChunkBuilder::new();
        assert_native_matches_interp(b.build());
    }

    #[test]
    fn native_const_pool() {
        let mut b = ChunkBuilder::new();
        let c = b.add_constant(Value::str("hello"));
        b.emit(Op::LoadConst(c), 1);
        assert_native_matches_interp(b.build());
    }

    #[test]
    fn emits_relocatable_object_with_exported_symbols() {
        let mut b = ChunkBuilder::new();
        b.emit(Op::LoadInt(42), 1);
        let chunk = b.build();
        let obj = std::env::temp_dir().join(format!("fusevm_aot_t_{}.o", std::process::id()));
        compile_object(&chunk, &obj).expect("emit object");
        assert!(
            std::fs::metadata(&obj)
                .map(|m| m.len() > 0)
                .unwrap_or(false),
            "non-empty .o"
        );
        // The driver + embedded-chunk symbols must be exported in the real object.
        if let Ok(o) = std::process::Command::new("nm")
            .arg("-g")
            .arg(&obj)
            .output()
        {
            let syms = String::from_utf8_lossy(&o.stdout);
            assert!(syms.contains("fusevm_aot_entry"), "entry exported:\n{syms}");
            assert!(
                syms.contains("fusevm_aot_chunk_blob"),
                "chunk blob exported:\n{syms}"
            );
            assert!(
                syms.contains("fusevm_aot_chunk_len"),
                "chunk len exported:\n{syms}"
            );
        }
        let _ = std::fs::remove_file(&obj);
    }
}

