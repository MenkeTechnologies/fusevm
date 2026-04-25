//! fusevm — Language-agnostic bytecode VM with fused superinstructions.
//!
//! Any language frontend can compile to fusevm opcodes and get:
//! - Fused superinstructions for hot loops (AccumSumLoop, etc.)
//! - Extension opcode dispatch for language-specific ops
//! - Stack-based execution with slot-indexed fast paths
//! - Cranelift JIT compilation (planned)
//!
//! ## Architecture
//!
//! ```text
//! stryke source ──→ stryke compiler ──┐
//!                                      ├──→ fusevm::Op ──→ VM::run()
//! zshrs source  ──→ shell compiler  ──┘
//! ```
//!
//! ## Usage
//!
//! ```rust
//! use fusevm::{Op, ChunkBuilder, VM, VMResult, Value};
//!
//! let mut b = ChunkBuilder::new();
//! b.emit(Op::LoadInt(40), 1);
//! b.emit(Op::LoadInt(2), 1);
//! b.emit(Op::Add, 1);
//!
//! let mut vm = VM::new(b.build());
//! match vm.run() {
//!     VMResult::Ok(val) => println!("result: {}", val.to_str()),
//!     VMResult::Error(e) => eprintln!("error: {}", e),
//!     VMResult::Halted => {}
//! }
//! ```

pub mod chunk;
pub mod jit;
pub mod op;
pub mod value;
pub mod vm;

pub use chunk::{Chunk, ChunkBuilder};
pub use op::Op;
pub use value::Value;
pub use jit::{JitCompiler, JitExtension, NativeCode};
pub use vm::{Frame, VM, VMResult};
