//! fusevm — Language-agnostic bytecode VM with fused superinstructions.
//!
//! Any language frontend can compile to fusevm opcodes and get:
//! - Fused superinstructions for hot loops (`AccumSumLoop`, etc.)
//! - Extension opcode dispatch for language-specific ops
//! - Stack-based execution with slot-indexed fast paths
//! - Three-tier Cranelift JIT:
//!   - Linear (straight-line, compile on first call)
//!   - Block (whole-chunk CFG, threshold 10)
//!   - Tracing (hot loop body, threshold 50, full side-exit machinery)
//!
//! ## Tracing JIT capability matrix
//!
//! | Capability | Status |
//! |---|---|
//! | Loop bodies, int slots, no calls | Phase 1 |
//! | Cross-call inlining (branchless callees) | Phase 2 |
//! | Caller-frame `if`/`else` with side-exits | Phase 3 |
//! | Callee-frame branches, frame materialization on deopt | Phase 4 |
//! | Value-stack reconstruction on deopt (Int + Float) | Phase 5 + 5b |
//! | Side-exit deopt counter + auto-blacklist | Phase 6 |
//! | Persistent metadata export/import (`TraceMetadata`) | Phase 7 |
//! | Bounded recursion inlining (depth ≤ 4) | Phase 8 |
//! | Side-trace stitching from hot deopt sites | Phase 9 |
//!
//! ## Architecture
//!
//! ```text
//! stryke source ──→ stryke compiler ──┐
//! awkrs source ──→ awkrs compiler     ├──→ fusevm::Op ──→ VM::run()
//! zshrs source  ──→ shell compiler  ──┘             ↳ optional tracing JIT
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
pub mod host;
pub mod jit;
pub mod op;
pub mod shell_builtins;
pub mod value;
pub mod vm;

pub use chunk::{Chunk, ChunkBuilder};
pub use host::{DefaultHost, ShellHost};
pub use jit::{
    DeoptFrame, DeoptInfo, JitCompiler, JitExtension, NativeCode, SlotKind, TraceJitConfig,
    TraceLookup, TraceMetadata,
};
pub use op::Op;
pub use value::Value;
pub use vm::{Frame, VMPool, VMResult, VM};
