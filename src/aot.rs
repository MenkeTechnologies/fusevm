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
//! integer/boolean computations directly to native IR. [`analyze_native`] runs
//! an abstract interpretation over the operand stack — tracking int-vs-bool
//! [`Kind`]s, finding basic-block leaders, and checking join consistency — and,
//! when the whole chunk qualifies, [`build_entry_native`] emits one Cranelift
//! block per leader with the operand stack held in frontend `Variable`s (an
//! `i64` and an `f64` variable per position; the plan's [`Kind`]s say which is
//! live). This covers integer **and float** arithmetic/comparisons — including
//! `int→float` promotion mirroring the interpreter — control flow
//! (`Jump`/`JumpIf*`), and integer slots (via typed accessor shims), with
//! **no per-op interpreter dispatch**; only the final result is boxed back into
//! the VM.
//!
//! Still deferred to later stages: `Div`/`Mod`/`Pow` (divide-by-zero `Undef`,
//! integer-div traps, and `powf`), float-typed slots, runtime type guards for
//! genuinely dynamic values, and mixed inline/shim regions (which need a
//! boxed↔unboxed spill/reload boundary so an unlowered op can fall back
//! mid-chunk). Anything not yet specialized falls back wholesale to
//! [`build_entry_threaded`].
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
use cranelift_codegen::ir::{types, AbiParam, Block, BlockArg, InstBuilder};
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
/// [`VM::aot_set_int_result`]). The native fast path keeps intermediate values
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
/// [`VM::aot_set_float_result`]). The float analog of
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

/// Read an integer slot for natively-lowered AOT code (see
/// [`VM::aot_slot_get_int`]). Slots stay boxed in the VM; the native path holds
/// only the operand stack in registers and touches slots through this shim.
///
/// # Safety
/// Same contract as [`fusevm_aot_exec_op`].
#[no_mangle]
pub extern "C" fn fusevm_aot_slot_get_int(vm: *mut VM, slot: u32) -> i64 {
    debug_assert!(!vm.is_null());
    // SAFETY: see the function contract; the driver owns the VM for the run.
    unsafe { (*vm).aot_slot_get_int(slot) }
}

/// Write an integer slot for natively-lowered AOT code (see
/// [`VM::aot_slot_set_int`]).
///
/// # Safety
/// Same contract as [`fusevm_aot_exec_op`].
#[no_mangle]
pub extern "C" fn fusevm_aot_slot_set_int(vm: *mut VM, slot: u32, n: i64) {
    debug_assert!(!vm.is_null());
    // SAFETY: see the function contract; the driver owns the VM for the run.
    unsafe { (*vm).aot_slot_set_int(slot, n) }
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
/// * [`build_entry_native`] — when [`analyze_native`] approves the whole chunk
///   (an integer/boolean computation over the supported op set, including
///   control flow and slots), every op is lowered to real native IR
///   (`iadd`/`icmp`/`brif`/…) with the operand stack held in registers and
///   **no per-op interpreter dispatch**. Slots are reached through typed
///   accessor shims; only the final result is boxed back into the VM.
/// * [`build_entry_threaded`] — the general fallback: one block per op, each
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
/// would float-promote, or produce a `Value::Bool`). `Int` and `Bool` are both
/// carried as `i64` at runtime; `Float` as `f64`.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Int,
    Bool,
    Float,
}

impl Kind {
    /// Numeric kinds may feed arithmetic and comparisons (a `Bool` may not).
    fn is_numeric(self) -> bool {
        matches!(self, Kind::Int | Kind::Float)
    }
}

/// A validated plan for native lowering: where the basic-block boundaries are,
/// the operand-stack *kinds* on entry to each block (length = depth), the
/// maximum stack depth (how many SSA stack variables to allocate), and the
/// stack kinds when control falls off the end (whose top selects the result).
struct NativePlan {
    /// Basic-block leader ips (each becomes one Cranelift block).
    leaders: BTreeSet<usize>,
    /// Operand-stack kinds on entry to each leader (its depth is the length).
    entry_kinds: HashMap<usize, Vec<Kind>>,
    /// Maximum operand-stack depth reached anywhere (SSA var count).
    max_depth: usize,
    /// Operand-stack kinds when control falls off the end (empty ⇒ `Halted`).
    end_kinds: Vec<Kind>,
}

/// Whether `chunk` lowers natively — thin wrapper over [`analyze_native`], kept
/// for readability at call sites and in tests.
fn native_lowerable(chunk: &Chunk) -> bool {
    analyze_native(chunk).is_some()
}

/// Validate `chunk` for native lowering and, if it qualifies, return a
/// [`NativePlan`]. Performs an abstract interpretation over the operand stack:
/// it tracks per-position [`Kind`]s, discovers basic-block leaders, and checks
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
        });
    }

    // Entry stack state (kinds) per discovered ip; `state` doubles as the
    // visited set. The program-end pseudo-point is `n`.
    let mut state: HashMap<usize, Vec<Kind>> = HashMap::new();
    let mut leaders: BTreeSet<usize> = BTreeSet::new();
    let mut max_depth = 0usize;
    let mut end_state: Option<Vec<Kind>> = None;

    state.insert(0, Vec::new());
    leaders.insert(0);
    let mut work = vec![0usize];

    while let Some(ip) = work.pop() {
        let mut st = state.get(&ip)?.clone();
        max_depth = max_depth.max(st.len());

        // Successors after applying ops[ip]: (successor ip, resulting stack).
        let mut succs: Vec<(usize, Vec<Kind>)> = Vec::new();
        match &ops[ip] {
            Op::Nop => succs.push((ip + 1, st)),
            Op::LoadInt(_) | Op::GetSlot(_) => {
                st.push(Kind::Int);
                max_depth = max_depth.max(st.len());
                succs.push((ip + 1, st));
            }
            Op::LoadFloat(_) => {
                st.push(Kind::Float);
                max_depth = max_depth.max(st.len());
                succs.push((ip + 1, st));
            }
            Op::LoadConst(idx) => {
                // Only numeric constants are lowerable; others fall back.
                let k = match chunk.constants.get(*idx as usize)? {
                    Value::Int(_) => Kind::Int,
                    Value::Float(_) => Kind::Float,
                    _ => return None,
                };
                st.push(k);
                max_depth = max_depth.max(st.len());
                succs.push((ip + 1, st));
            }
            // Arithmetic promotes: result is Float if either operand is Float,
            // else Int — mirroring the interpreter's `arith_int_fast`.
            Op::Add | Op::Sub | Op::Mul => {
                let b = st.pop()?;
                let a = st.pop()?;
                if !a.is_numeric() || !b.is_numeric() {
                    return None;
                }
                let r = if a == Kind::Float || b == Kind::Float {
                    Kind::Float
                } else {
                    Kind::Int
                };
                st.push(r);
                succs.push((ip + 1, st));
            }
            Op::Negate => {
                let a = st.pop()?;
                if !a.is_numeric() {
                    return None;
                }
                st.push(a); // Int → Int, Float → Float
                succs.push((ip + 1, st));
            }
            Op::NumEq | Op::NumNe | Op::NumLt | Op::NumGt | Op::NumLe | Op::NumGe => {
                let b = st.pop()?;
                let a = st.pop()?;
                if !a.is_numeric() || !b.is_numeric() {
                    return None;
                }
                st.push(Kind::Bool);
                succs.push((ip + 1, st));
            }
            Op::SetSlot(_) => {
                // Slots are int-only for now (float slots are a later stage).
                if st.pop()? != Kind::Int {
                    return None;
                }
                succs.push((ip + 1, st));
            }
            Op::Pop => {
                st.pop()?;
                succs.push((ip + 1, st));
            }
            Op::Dup => {
                let k = *st.last()?;
                st.push(k);
                max_depth = max_depth.max(st.len());
                succs.push((ip + 1, st));
            }
            Op::Jump(t) => {
                let t = *t;
                if t > n {
                    return None;
                }
                if t < n {
                    leaders.insert(t);
                }
                succs.push((t, st));
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
                succs.push((t, st.clone()));
                succs.push((ip + 1, st));
            }
            _ => return None,
        }

        for (s, sstate) in succs {
            if s == n {
                match &end_state {
                    None => end_state = Some(sstate),
                    Some(prev) if *prev != sstate => return None,
                    Some(_) => {}
                }
                continue;
            }
            match state.get(&s) {
                None => {
                    state.insert(s, sstate);
                    work.push(s);
                }
                Some(prev) if *prev != sstate => return None,
                Some(_) => {}
            }
        }
    }

    // The result is the stack top when control falls off the end; it must be
    // numeric (a `Bool` there would box to the wrong `Value` variant).
    let end_kinds = match end_state {
        Some(es) => {
            if let Some(top) = es.last() {
                if !top.is_numeric() {
                    return None;
                }
            }
            es
        }
        // No path reaches the end (e.g. an infinite loop): the ret block is
        // unreachable, so there is no result to box.
        None => Vec::new(),
    };

    let entry_kinds = leaders
        .iter()
        .map(|&l| (l, state.get(&l).cloned().unwrap_or_default()))
        .collect();

    Some(NativePlan {
        leaders,
        entry_kinds,
        max_depth,
        end_kinds,
    })
}

/// Native fast path: emit real integer/float IR for a chunk that
/// [`analyze_native`] has approved, following the `plan` it produced. One
/// Cranelift block per basic-block leader; the operand stack lives in frontend
/// [`Variable`]s (so SSA/phi construction at joins — including loop back-edges —
/// is automatic on `seal_all_blocks`).
///
/// Each stack position has two parallel variables — one `i64` ([`Kind::Int`]
/// and [`Kind::Bool`]) and one `f64` ([`Kind::Float`]) — and codegen replays
/// the plan's per-position [`Kind`]s to read/write the right one and to insert
/// `int→float` promotions where the interpreter would. Slots are reached
/// through typed accessor shims; only the final result is boxed back into the
/// VM. No per-op dispatch and no interpreter shim calls for the lowered ops.
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

    let mut sget_sig = module.make_signature();
    sget_sig.params.push(AbiParam::new(ptr_ty));
    sget_sig.params.push(AbiParam::new(types::I32));
    sget_sig.returns.push(AbiParam::new(types::I64));
    let sget_id = module
        .declare_function("fusevm_aot_slot_get_int", Linkage::Import, &sget_sig)
        .map_err(|e| format!("aot: declare slot_get: {e}"))?;

    let mut sset_sig = module.make_signature();
    sset_sig.params.push(AbiParam::new(ptr_ty));
    sset_sig.params.push(AbiParam::new(types::I32));
    sset_sig.params.push(AbiParam::new(types::I64));
    let sset_id = module
        .declare_function("fusevm_aot_slot_set_int", Linkage::Import, &sset_sig)
        .map_err(|e| format!("aot: declare slot_set: {e}"))?;

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
        let sget_ref = module.declare_func_in_func(sget_id, b.func);
        let sset_ref = module.declare_func_in_func(sset_id, b.func);

        // The VM pointer and each operand-stack position are frontend Variables;
        // `seal_all_blocks` then builds the SSA phis at all joins for us. Each
        // position has an i64 var and an f64 var; the plan's Kinds say which is
        // live (consistent across all edges into any join).
        let vm_var = b.declare_var(ptr_ty);
        let ivars: Vec<Variable> = (0..plan.max_depth)
            .map(|_| b.declare_var(types::I64))
            .collect();
        let fvars: Vec<Variable> = (0..plan.max_depth)
            .map(|_| b.declare_var(types::F64))
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

        // Entry: stash the VM pointer, jump to the first block (or ret if empty).
        b.switch_to_block(entry_block);
        let vm_param = b.block_params(entry_block)[0];
        b.def_var(vm_var, vm_param);
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
                        _ => unreachable!("analyze_native admitted only numeric constants"),
                    }
                }
                // `a` is the lower slot, `b`(=y) the top; matches the
                // interpreter's `arith_int_fast` (int if both Int, else float).
                op @ (Op::Add | Op::Sub | Op::Mul) => {
                    let ky = kinds.pop().unwrap();
                    let iy = kinds.len();
                    let kx = kinds.pop().unwrap();
                    let ix = kinds.len();
                    if ky == Kind::Float || kx == Kind::Float {
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
                Op::Negate => {
                    let k = kinds.pop().unwrap();
                    let idx = kinds.len();
                    if k == Kind::Float {
                        let x = b.use_var(fvars[idx]);
                        let r = b.ins().fneg(x);
                        b.def_var(fvars[idx], r);
                        kinds.push(Kind::Float);
                    } else {
                        let x = b.use_var(ivars[idx]);
                        let r = b.ins().ineg(x);
                        b.def_var(ivars[idx], r);
                        kinds.push(Kind::Int);
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
                Op::GetSlot(slot) => {
                    let vm = b.use_var(vm_var);
                    let sc = b.ins().iconst(types::I32, *slot as i64);
                    let call = b.ins().call(sget_ref, &[vm, sc]);
                    let r = b.inst_results(call)[0];
                    let idx = kinds.len();
                    b.def_var(ivars[idx], r);
                    kinds.push(Kind::Int);
                }
                Op::SetSlot(slot) => {
                    kinds.pop().unwrap(); // analysis guarantees Int
                    let idx = kinds.len();
                    let v = b.use_var(ivars[idx]);
                    let vm = b.use_var(vm_var);
                    let sc = b.ins().iconst(types::I32, *slot as i64);
                    b.ins().call(sset_ref, &[vm, sc, v]);
                }
                Op::Pop => {
                    kinds.pop().unwrap();
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
            if topk == Kind::Float {
                let v = b.use_var(fvars[idx]);
                b.ins().call(fres_ref, &[vm, v]);
            } else {
                let v = b.use_var(ivars[idx]);
                b.ins().call(ires_ref, &[vm, v]);
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
    if k == Kind::Float {
        let f = b.use_var(fvars[idx]);
        let z = b.ins().f64const(Ieee64::with_bits(0f64.to_bits()));
        b.ins().fcmp(FloatCC::NotEqual, f, z)
    } else {
        let v = b.use_var(ivars[idx]);
        b.ins().icmp_imm(IntCC::NotEqual, v, 0)
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
    builder.symbol(
        "fusevm_aot_slot_get_int",
        fusevm_aot_slot_get_int as *const u8,
    );
    builder.symbol(
        "fusevm_aot_slot_set_int",
        fusevm_aot_slot_set_int as *const u8,
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
        // An unsupported op (string concat) routes to the threaded fallback.
        let mut s = ChunkBuilder::new();
        let c = s.add_constant(Value::str("x"));
        s.emit(Op::LoadConst(c), 1);
        assert!(!native_lowerable(&s.build()), "LoadConst not lowered");

        // A boolean left as the program result would box to the wrong Value
        // variant (`Bool` vs the native path's `Int`), so it is rejected.
        let mut cmp = ChunkBuilder::new();
        cmp.emit(Op::LoadInt(1), 1);
        cmp.emit(Op::LoadInt(2), 1);
        cmp.emit(Op::NumLt, 1);
        assert!(!native_lowerable(&cmp.build()), "bool result rejected");

        // Storing a boolean into a slot would also diverge (interp stores Bool,
        // the native shim stores Int).
        let mut bslot = ChunkBuilder::new();
        bslot.emit(Op::LoadInt(1), 1);
        bslot.emit(Op::LoadInt(2), 1);
        bslot.emit(Op::NumLt, 1);
        bslot.emit(Op::SetSlot(0), 1);
        assert!(!native_lowerable(&bslot.build()), "bool→slot rejected");

        // Underflow (Add with one operand) is rejected, not miscompiled.
        let mut bad = ChunkBuilder::new();
        bad.emit(Op::LoadInt(1), 1);
        bad.emit(Op::Add, 1);
        assert!(!native_lowerable(&bad.build()), "underflow must be rejected");
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
