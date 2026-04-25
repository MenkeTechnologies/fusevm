//! fusevm JIT — Cranelift codegen for universal bytecodes.
//!
//! Compiles fusevm::Op sequences to native machine code via Cranelift.
//! Language-specific ops (Extended) are handled by a JitExtension trait
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
//! cranelift_jit::JITModule::finalize()
//!     │
//!     ▼
//! NativeCode (function pointer — call directly)
//! ```
//!
//! # Universal Ops JIT'd
//!
//! Constants: LoadInt, LoadFloat, LoadConst, LoadTrue, LoadFalse, LoadUndef
//! Stack: Pop, Dup, Swap, Rot
//! Variables: GetVar, SetVar, GetSlot, SetSlot
//! Arithmetic: Add, Sub, Mul, Div, Mod, Pow, Negate, Inc, Dec
//! String: Concat, StringLen
//! Comparison: NumEq/Ne/Lt/Gt/Le/Ge, StrEq/Ne/Lt/Gt/Le/Ge, Spaceship
//! Logic/Bitwise: LogNot, BitAnd/Or/Xor/Not, Shl, Shr
//! Control: Jump, JumpIfTrue/False, JumpIfTrueKeep/FalseKeep
//! Functions: Call, Return, ReturnValue, PushFrame, PopFrame
//! Fused: PreIncSlot, SlotLtIntJumpIfFalse, SlotIncLtIntJumpBack, AccumSumLoop
//!
//! # Extension Trait
//!
//! ```rust,ignore
//! pub trait JitExtension {
//!     fn can_jit(&self, ext_id: u16) -> bool;
//!     fn emit_extended(
//!         &self,
//!         builder: &mut FunctionBuilder,
//!         ext_id: u16,
//!         ext_arg: u8,
//!         stack: &mut Vec<Value>,
//!     );
//! }
//! ```
//!
//! stryke registers ~149 extended ops (PerlValue, NaN-boxing, typeglobs, etc.)
//! zshrs registers ~20 shell ops (Exec, Redirect, Glob, TestFile, etc.)

// TODO: Cranelift codegen implementation
// This is the extraction target from strykelang/jit.rs.
// The ~45 universal op handlers move here.
// The ~149 stryke-specific handlers stay in strykelang/jit.rs
// and implement JitExtension.
//
// Cranelift dependencies (add to Cargo.toml when implementing):
//   cranelift-jit = "0.130"
//   cranelift-codegen = "0.130"
//   cranelift-frontend = "0.130"
//   cranelift-native = "0.130"
//   cranelift-module = "0.130"
//
// Gated behind: [features] jit = ["cranelift-jit", ...]

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

/// Placeholder — JIT compilation result.
/// Will hold a function pointer to native code when Cranelift is wired.
pub struct NativeCode {
    _private: (),
}

/// JIT compiler state.
/// Will hold Cranelift JITModule when implemented.
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
    /// Returns true if all ops are either universal or handled by an extension.
    pub fn is_eligible(&self, chunk: &crate::Chunk) -> bool {
        use crate::Op;
        for op in &chunk.ops {
            match op {
                // Universal ops — always JIT-able
                Op::Nop
                | Op::LoadInt(_) | Op::LoadFloat(_) | Op::LoadConst(_)
                | Op::LoadTrue | Op::LoadFalse | Op::LoadUndef
                | Op::Pop | Op::Dup | Op::Dup2 | Op::Swap | Op::Rot
                | Op::GetVar(_) | Op::SetVar(_) | Op::DeclareVar(_)
                | Op::GetSlot(_) | Op::SetSlot(_)
                | Op::Add | Op::Sub | Op::Mul | Op::Div | Op::Mod | Op::Pow
                | Op::Negate | Op::Inc | Op::Dec
                | Op::Concat | Op::StringRepeat | Op::StringLen
                | Op::NumEq | Op::NumNe | Op::NumLt | Op::NumGt | Op::NumLe | Op::NumGe
                | Op::StrEq | Op::StrNe | Op::StrLt | Op::StrGt | Op::StrLe | Op::StrGe
                | Op::StrCmp | Op::Spaceship
                | Op::LogNot | Op::LogAnd | Op::LogOr
                | Op::BitAnd | Op::BitOr | Op::BitXor | Op::BitNot | Op::Shl | Op::Shr
                | Op::Jump(_) | Op::JumpIfTrue(_) | Op::JumpIfFalse(_)
                | Op::JumpIfTrueKeep(_) | Op::JumpIfFalseKeep(_)
                | Op::Call(_, _) | Op::Return | Op::ReturnValue
                | Op::PushFrame | Op::PopFrame
                | Op::PreIncSlot(_) | Op::PreIncSlotVoid(_)
                | Op::SlotLtIntJumpIfFalse(_, _, _)
                | Op::SlotIncLtIntJumpBack(_, _, _)
                | Op::AccumSumLoop(_, _, _)
                | Op::AddAssignSlotVoid(_, _)
                | Op::SetStatus | Op::GetStatus => continue,

                // Extended — check if any extension handles it
                Op::Extended(id, _) | Op::ExtendedWide(id, _) => {
                    let id = *id;
                    if !self.extensions.iter().any(|ext| ext.can_jit(id)) {
                        return false;
                    }
                }

                // Shell ops, arrays, hashes, etc. — not yet JIT-eligible
                // These will be added as Cranelift codegen is implemented
                _ => return false,
            }
        }
        true
    }
}

impl Default for JitCompiler {
    fn default() -> Self {
        Self::new()
    }
}
