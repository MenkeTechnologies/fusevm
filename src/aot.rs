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
//! Specializing hot ops to inline IR (real native arithmetic/loops) is a later
//! pass layered on top of this; this stage delivers the native artifact.
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
use crate::value::Value;
use crate::vm::{VMResult, VM};

use cranelift_codegen::ir::condcodes::IntCC;
use cranelift_codegen::ir::{types, AbiParam, BlockArg, InstBuilder};
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Switch};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{default_libcall_names, DataDescription, FuncId, Linkage, Module};
use cranelift_object::{ObjectBuilder, ObjectModule};
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
pub fn build_entry<M: Module>(module: &mut M, chunk: &Chunk) -> Result<FuncId, String> {
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
        assert_native_matches_interp(b.build());
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
        assert_native_matches_interp(b.build());
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
        assert_native_matches_interp(b.build());
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
