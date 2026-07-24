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

#[cfg(feature = "aot")]
pub mod aot;
pub mod awk_builtins;
pub mod awk_host;
pub mod chunk;
/// Inline Rust FFI runtime (`rust { ... }` blocks). Behind the `ffi` feature —
/// pulls in `libc`/`sha2`/`base64` and shells out to `rustc` at runtime.
#[cfg(feature = "ffi")]
pub mod ffi;
pub mod host;
pub mod jit;
pub mod op;
/// Source-level `rust { ... }` desugarer shared by every frontend. Behind the
/// `ffi` feature (needs `base64` to encode block bodies).
#[cfg(feature = "ffi")]
pub mod rust_sugar;
/// Cooperative goroutine scheduler + channels. A green-thread layer over the
/// single-VM dispatch loop: frontends emit `Op::Go`/`ChanMake`/`ChanSend`/
/// `ChanRecv`/`ChanClose` and drive them with [`sched::Scheduler`].
pub mod sched;
pub mod shell_builtins;
/// Portable wall-clock reads (`chrono`-backed) so the VM's clock ops work on
/// `wasm32` — where `std::time::SystemTime::now()` panics — as well as native.
pub mod sysclock;
pub mod value;
pub mod vm;

pub use awk_host::{AwkHost, AwkLvalue, DefaultAwkHost};
pub use chunk::{Chunk, ChunkBuilder};
pub use host::{DefaultHost, ShellHost};
pub use jit::{
    set_awk_field_num_hook, BlockNum, DeoptFrame, DeoptInfo, JitCompiler, JitExtension, NativeCode,
    SlotKind, TraceJitConfig, TraceLookup, TraceMetadata,
};
pub use op::Op;

/// Reduce an exact `i128` numerator by `k`, **flooring** — the remainder takes
/// the sign of the divisor (`-7 mod 3 == 2`), which is Python's and elisp's `%`
/// rather than C's truncating one.
///
/// The single implementation behind [`Op::MulModFloor`] / [`Op::MulAddModFloor`]
/// in all three tiers (interpreter, JIT libcall, AOT libcall), so they cannot
/// drift. `k == 0` yields `0`, matching [`Op::Mod`]'s zero divisor. The result
/// always fits `i64` because `|r| < |k|`.
#[inline]
pub fn floor_rem_i128(n: i128, k: i64) -> i64 {
    if k == 0 {
        return 0;
    }
    let k = k as i128;
    let r = n % k;
    // Signs disagree (and there is a remainder to shift) -> add one divisor.
    if r != 0 && (r < 0) != (k < 0) {
        (r + k) as i64
    } else {
        r as i64
    }
}
#[cfg(feature = "ffi")]
pub use rust_sugar::RustSugar;
pub use sched::{SchedError, SchedReq, Scheduler, SelectCase};
pub use value::Value;
pub use vm::{Frame, NumOp, NumericHook, VMPool, VMResult, VM};
