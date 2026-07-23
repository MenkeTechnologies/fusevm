//! The fusevm execution engine — stack-based bytecode dispatch loop.
//!
//! This is the hot path. Every cycle counts. The dispatch loop uses
//! a flat `match` on `Op` variants — Rust compiles this to a jump table.
//!
//! Frontends register extension handlers via `ExtensionHandler` for
//! language-specific opcodes (`Op::Extended`, `Op::ExtendedWide`).
//!
//! ## Optimizations
//!
//! - **Type-specialized integer fast paths**: Add, Sub, Mul, Mod, comparisons
//!   check for `Int×Int` first and skip `to_float()` coercion entirely.
//! - **Zero-clone dispatch**: ops are borrowed from the chunk, not cloned per cycle.
//!   `LoadConst` copies scalars (Int/Float/Bool) without touching Arc refcounts.
//! - **In-place container mutation**: array/hash ops (Push, Pop, Shift, Set,
//!   HashSet, HashDelete) mutate globals directly — no clone-modify-writeback.
//! - **`Cow<str>` string coercion**: `as_str_cow()` borrows `Str` variants without
//!   allocation. Used in string comparisons, Concat, Print, hash key lookup.
//! - **Inline builtin cache**: `CallBuiltin` dispatches through a pre-registered
//!   function pointer table — no name lookup at runtime.
//! - **Fused superinstructions**: hot loop patterns run as single ops
//!   (AccumSumLoop, SlotIncLtIntJumpBack, etc.)
//! - **Pre-allocated collections**: Range, MakeHash, HashKeys/Values use exact
//!   or estimated capacity. ConcatConstLoop pre-sizes the string buffer.

use crate::awk_host::AwkHost;
use crate::chunk::Chunk;
use crate::host::ShellHost;
#[cfg(feature = "jit")]
use crate::jit::{DeoptInfo, JitCompiler, SlotKind, TraceLookup};
use crate::op::Op;
use crate::value::Value;

// Tracing-JIT thresholds previously sourced from `vm.rs` constants are
// now read through `JitCompiler::get_config()` so callers can override
// per-thread via `JitCompiler::set_config(...)`. The defaults match the
// historical phase-by-phase constants:
//   trace_threshold      = 50  (backedges before recording arms)
//   max_side_exits       = 50  (side-exits before main-trace blacklist)
//   max_inline_recursion = 4   (self-recursive call depth cap)
//   max_trace_chain      = 4   (chained side-trace dispatch depth cap)
//   max_trace_len        = 256 (recorded ops cap)

/// In-progress trace recording state.
///
/// The recorder is armed when `trace_lookup` returns `StartRecording`.
/// While armed, every dispatched op is appended to `ops` before the op's
/// effect is applied. The recording closes when the interpreter takes a
/// backward jump that lands at `close_anchor_ip`.
///
/// Phase 9 split: `record_anchor_ip` is where the recording STARTED (the
/// trace cache key); `close_anchor_ip` is where the recording is expected
/// to LAND on close. For main traces the two are identical (a loop header
/// is both the recording start and the closing-branch target). For side
/// traces (recordings armed at a hot side-exit), `record_anchor_ip` is the
/// side-exit IP the recorder started from, while `close_anchor_ip` remains
/// the enclosing loop's header — so the trace closes correctly when the
/// loop's backward branch fires.
///
/// `entered_ips` simulates the inlined frame stack so the recorder can
/// (a) bound self-recursion to `MAX_INLINE_RECURSION` levels, and
/// (b) reject unbalanced Returns. The values pushed are bytecode entry IPs
/// resolved from `Op::Call(name_idx, _)`.
#[cfg(feature = "jit")]
struct TraceRecorder {
    /// Phase 9: IP recording started from. Used as the trace cache key
    /// `(chunk.op_hash, record_anchor_ip)`.
    record_anchor_ip: usize,
    /// Phase 9: IP the closing backward branch is expected to land at.
    /// For main traces this equals `record_anchor_ip`; for side traces
    /// it's the enclosing loop's header.
    close_anchor_ip: usize,
    /// IP just past the closing branch (where the interpreter resumes on
    /// normal loop exit).
    fallthrough_ip: usize,
    /// Recorded op sequence (body + closing branch as final op).
    ops: Vec<Op>,
    /// Original bytecode IP each recorded op was dispatched from. Parallel to
    /// `ops`. Used at compile time to infer direction taken at conditional
    /// branches: for op at index `i`, if `recorded_ips[i+1]` equals the op's
    /// jump target, the jump was taken; otherwise the fallthrough was.
    recorded_ips: Vec<usize>,
    /// Slot type snapshot at recording start; installed as the entry guard.
    slot_kinds_at_anchor: Vec<SlotKind>,
    /// Stack of bytecode entry IPs for currently inlined callees. Empty in
    /// the caller frame; pushed on Op::Call, popped on Op::Return /
    /// Op::ReturnValue. Used for recursion detection.
    entered_ips: Vec<usize>,
    /// True if any condition aborted the recording. Causes cleanup-only on
    /// next jump dispatch.
    aborted: bool,
}

/// Call frame on the frame stack.
#[derive(Debug, Clone)]
pub struct Frame {
    /// Return address (ip to resume after call)
    pub return_ip: usize,
    /// Base pointer into the value stack (locals start here)
    pub stack_base: usize,
    /// Local variable slots (indexed by `GetSlot`/`SetSlot`)
    pub slots: Vec<Value>,
}

/// Extension handler for language-specific opcodes.
/// Frontends register this at VM init.
pub type ExtensionHandler = Box<dyn FnMut(&mut VM, u16, u8) + Send>;
/// Wide extension handler (usize payload).
pub type ExtensionWideHandler = Box<dyn FnMut(&mut VM, u16, usize) + Send>;
/// Builtin function handler: (vm, argc) → Value
pub type BuiltinHandler = fn(&mut VM, u8) -> Value;

/// The virtual machine.
pub struct VM {
    /// Value stack
    pub stack: Vec<Value>,
    /// Call frame stack
    pub frames: Vec<Frame>,
    /// Global variables (name pool index → value)
    pub globals: Vec<Value>,
    /// Instruction pointer
    pub ip: usize,
    /// Current chunk being executed
    pub chunk: Chunk,
    /// Last exit status ($?)
    pub last_status: i32,
    /// Optional stdout sink for `Op::Print`/`Op::PrintLn`; see [`OutputSink`].
    /// `None` writes to `std::io::stdout()`. Installed by web-worker frontends.
    output_sink: Option<OutputSink>,
    /// Optional stdin source for `Op::ReadLine`; see [`InputSource`]. `None`
    /// reads `std::io::stdin()`. Installed by web-worker frontends.
    input_source: Option<InputSource>,
    /// Host callback for arithmetic fusevm cannot complete natively (non-numeric
    /// operand, or integer overflow). `None` — the default — keeps the
    /// coerce-and-wrap semantics zshrs/awkrs/stryke rely on. See [`NumericHook`].
    numeric_hook: Option<NumericHook>,
    /// The inclusive range the VM may keep as a native `Value::Int`. `None` (the
    /// default) is the whole `i64`. A Lisp with tagged fixnums has a narrower one
    /// — Emacs's is ±2^61 — and every result outside it is a bignum, so strict
    /// mode delegates those to the [`NumericHook`] exactly as it delegates an
    /// `i64` overflow. Only consulted in strict mode.
    fixnum_range: Option<(i64, i64)>,
    /// Whether every slot of the current frame held an `Int`/`Float`/`Bool` at
    /// the last `refresh_slot_buffers`. In strict numeric mode the block JIT is
    /// skipped when this is false: a slot holding a string or an object handle is
    /// passed to native code as the integer `0`, which would silently coerce
    /// exactly the operand the `NumericHook` exists to reject.
    #[cfg(feature = "jit")]
    slots_all_numeric: bool,
    /// Extension handler for `Op::Extended`
    ext_handler: Option<ExtensionHandler>,
    /// Extension handler for `Op::ExtendedWide`
    ext_wide_handler: Option<ExtensionWideHandler>,
    /// Inline builtin cache: builtin_id → function pointer (no lookup at dispatch)
    builtin_table: Vec<Option<BuiltinHandler>>,
    /// Frontend-supplied shell host (glob/expand/redirect/pipeline/etc).
    /// When `None`, shell ops fall back to minimal stub behavior.
    pub host: Option<Box<dyn ShellHost>>,
    /// Frontend-supplied AWK host (fields/record/print/getline/string builtins).
    /// The VM routes the reserved AWK op range (`Op::ExtendedWide` with
    /// `id >= awk_builtins::AWK_OP_BASE`) here. When `None`, AWK ops are inert
    /// stubs and the universal ops still execute normally.
    pub awk_host: Option<Box<dyn AwkHost>>,
    /// AWK PRNG seed for native `rand`/`srand` (glibc LCG, gawk-compatible).
    /// Execution-intrinsic VM state (like a register), not part of AWK's data
    /// model, so it lives here and is handled VM-side in both dispatch paths.
    /// Initialized to 1 to match awkrs's default seed sequence.
    awk_rand_seed: u64,
    /// AWK control-flow signal raised by `Op::AwkSignal(code)` (the chunk halts
    /// and the frontend driver reads this after `run()`): `next`/`nextfile`/
    /// `exit` and range-pattern flow that has no `fusevm::Value` representation.
    /// `None` unless an awk frontend emitted `Op::AwkSignal`; zshrs/stryke never
    /// do, so for them this stays `None` and `Halted` behaves exactly as before.
    awk_signal: Option<u8>,
    /// Cooperative-concurrency scheduling request raised by a goroutine/channel
    /// op (`Op::Go`/`ChanMake`/`ChanSend`/`ChanRecv`/`ChanClose`): the op stores
    /// the request here and halts the chunk, and a [`crate::sched::Scheduler`]
    /// driver reads it via [`VM::take_sched`] after `run()` returns, then resumes
    /// this VM (or another goroutine). `None` unless a frontend emits those ops,
    /// so zshrs/stryke/etc. behave exactly as before.
    sched: Option<crate::sched::SchedReq>,
    /// Halted flag
    halted: bool,
    /// Tracing JIT enabled. When true, backward branches consult the trace
    /// cache and may invoke compiled traces or arm the recorder.
    #[cfg(feature = "jit")]
    tracing_jit: bool,
    /// JIT compiler instance — stateless wrapper over the thread-local cache.
    #[cfg(feature = "jit")]
    jit: JitCompiler,
    /// Active trace recording, if any.
    #[cfg(feature = "jit")]
    recorder: Option<TraceRecorder>,
    /// Reusable scratch i64 buffer of slot values, passed to compiled traces.
    #[cfg(feature = "jit")]
    slot_buf: Vec<i64>,
    /// Reusable scratch slot-kind snapshot for the trace entry guard.
    #[cfg(feature = "jit")]
    slot_kinds_buf: Vec<SlotKind>,
    /// Reusable scratch buffer the trace fn populates on every invocation
    /// with the resume IP and (on callee-frame side-exits) inlined-frame
    /// materialization records the VM uses to reshape `vm.frames`.
    /// Stored inline (~888 bytes) to avoid heap indirection on the hot
    /// trace path; the size cost is paid once per VM and the access
    /// savings hit every invocation.
    #[cfg(feature = "jit")]
    deopt_info: DeoptInfo,
    /// Cached block-JIT eligibility for `self.chunk`. `None` until first
    /// `VM::run` call evaluates it; reused across subsequent runs since
    /// `Chunk` is immutable for the VM's lifetime. Saves the TLS HashMap
    /// lookup that `JitCompiler::is_block_eligible` would otherwise
    /// perform on every run.
    #[cfg(feature = "jit")]
    block_eligible_cached: Option<bool>,
    /// Result captured by the AOT closed-world driver (`fusevm::aot`). The
    /// native entry function stores the terminating [`VMResult`] here via
    /// [`VM::aot_finish`]; the caller takes it after the driver returns.
    #[cfg(feature = "aot")]
    aot_result: Option<VMResult>,
    /// Value arena for natively-lowered heap values. The native fast path keeps
    /// scalars in registers, but a boxed value (string/array/…) can't fit one,
    /// so it lives here and the register holds an `i64` *handle* (an index). See
    /// [`VM::aot_box`] / [`VM::aot_unbox`].
    #[cfg(feature = "aot")]
    aot_arena: Vec<Value>,
    /// Freelist of reusable [`VM::aot_arena`] slots. Native heap handles are
    /// owned (one live owner each): consuming a handle (`aot_unbox`, result,
    /// pop) frees its slot here so a loop that rebuilds a value each iteration
    /// reuses arena slots instead of growing without bound.
    #[cfg(feature = "aot")]
    aot_free: Vec<u32>,
}

/// Result of VM execution
#[derive(Debug)]
pub enum VMResult {
    Ok(Value),
    /// Halted (no more instructions)
    Halted,
    /// Runtime error
    Error(String),
}

/// Outcome of executing one op via [`VM::exec_op`]. `Cont` means the dispatch
/// loop proceeds to the next op (running its recorder-finalize step); `Ret`
/// means the op terminated the run and the loop returns the wrapped
/// [`VMResult`]. This is the control-flow contract shared by the interpreter
/// loop and the AOT closed-world compiler (`fusevm::aot`).
pub(crate) enum ExecFlow {
    Cont,
    Ret(VMResult),
}

/// The arithmetic or comparison op that delegated to a [`NumericHook`].
///
/// `Neg` is unary: its handler receives the operand as `a` and `Value::Undef`
/// as `b`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NumOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Pow,
    Neg,
    Lt,
    Gt,
    Le,
    Ge,
    Eq,
    Ne,
}

/// Host callback for arithmetic fusevm cannot complete natively.
///
/// fusevm's numeric ops are awk/shell-flavoured by default: a non-numeric
/// operand is coerced (`to_float`, so `"a"` becomes `0.0`) and integer overflow
/// wraps. That is correct for zshrs, awkrs and stryke, and wrong for a frontend
/// whose language signals on non-numbers or promotes on overflow.
///
/// Installing a hook with [`VM::set_numeric_hook`] switches the VM to *strict*
/// numeric mode, where an op that cannot be computed exactly in `i64`/`f64`
/// hands off to the host instead of guessing:
///
/// - an operand that is not `Int` or `Float` (a string, a `Value::Obj` handle,
///   a bool, `Undef`) — the host decides whether that is an error (elisp:
///   `(wrong-type-argument number-or-marker-p "a")`) or a value it knows how to
///   add (elisp: a marker, or a bignum living in its own object heap);
/// - integer `Add`/`Sub`/`Mul`/`Neg` that overflows `i64` — the host returns the
///   exact result (elisp: a bignum), rather than the wrapped one.
///
/// `Err` raises a VM error carrying the message. Mixed int/float and
/// float/float arithmetic never delegates: it is exact in `f64` and stays on
/// the fast path.
pub type NumericHook =
    std::sync::Arc<dyn Fn(NumOp, &Value, &Value) -> Result<Value, String> + Send + Sync>;

/// Sink for VM stdout (`Op::Print` / `Op::PrintLn`).
///
/// `None` — the default — writes to `std::io::stdout()`, byte-for-byte as
/// before. A frontend running fusevm in a browser web worker installs a sink,
/// because wasm has no real stdout: typically a closure appending to an
/// `Arc<Mutex<String>>` the frontend drains after `run()` and forwards to the
/// JS host via `postMessage`. The `Send` bound keeps [`VM`] `Send` for
/// [`VMPool`]; a worker sink stays `Send` by writing to an `Arc<Mutex<_>>`
/// rather than capturing a JS handle directly.
pub type OutputSink = Box<dyn FnMut(&str) + Send>;

/// Source for VM stdin (`Op::ReadLine`).
///
/// `None` — the default — reads a line from `std::io::stdin()`. When installed,
/// the closure returns one line per call (newline already trimmed) or `None` at
/// end of input, which `Op::ReadLine` pushes as `Value::Undef`. Blocking
/// interactive stdin inside a worker needs `SharedArrayBuffer` + `Atomics.wait`
/// and is the frontend's concern; this hook covers the common pre-loaded-input
/// case.
pub type InputSource = Box<dyn FnMut() -> Option<String> + Send>;

/// Whether a value is one fusevm's numeric ops can compute on natively.
#[inline(always)]
fn is_native_num(v: &Value) -> bool {
    matches!(v, Value::Int(_) | Value::Float(_))
}

impl VM {
    /// Construct a fresh VM bound to the given chunk. Allocates the
    /// per-name slot vector, seeds the call-frame stack with a root
    /// frame, and zeros every per-thread counter (cycle / deopt /
    /// trace stats). The chunk's `op_hash` is preserved verbatim so
    /// subsequent JIT-cache lookups can short-circuit recompilation.
    pub fn new(chunk: Chunk) -> Self {
        let num_names = chunk.names.len();
        let mut frames = Vec::with_capacity(32);
        frames.push(Frame {
            return_ip: 0,
            stack_base: 0,
            slots: Vec::with_capacity(16),
        });
        Self {
            stack: Vec::with_capacity(256),
            frames,
            globals: vec![Value::Undef; num_names],
            ip: 0,
            chunk,
            last_status: 0,
            output_sink: None,
            input_source: None,
            numeric_hook: None,
            fixnum_range: None,
            #[cfg(feature = "jit")]
            slots_all_numeric: true,
            ext_handler: None,
            ext_wide_handler: None,
            builtin_table: Vec::new(),
            host: None,
            awk_host: None,
            awk_rand_seed: 1,
            awk_signal: None,
            sched: None,
            halted: false,
            #[cfg(feature = "jit")]
            tracing_jit: false,
            #[cfg(feature = "jit")]
            jit: JitCompiler::new(),
            #[cfg(feature = "jit")]
            recorder: None,
            #[cfg(feature = "jit")]
            slot_buf: Vec::new(),
            #[cfg(feature = "jit")]
            slot_kinds_buf: Vec::new(),
            #[cfg(feature = "jit")]
            deopt_info: DeoptInfo::zeroed(),
            #[cfg(feature = "jit")]
            block_eligible_cached: None,
            #[cfg(feature = "aot")]
            aot_result: None,
            #[cfg(feature = "aot")]
            aot_arena: Vec::new(),
            #[cfg(feature = "aot")]
            aot_free: Vec::new(),
        }
    }

    /// Install a [`NumericHook`], switching this VM to strict numeric mode:
    /// arithmetic that cannot be computed exactly in `i64`/`f64` — a
    /// non-numeric operand, or integer overflow — is handed to `hook` instead
    /// of being coerced or wrapped.
    ///
    /// Strict mode also constrains the JIT, which otherwise coerces and wraps
    /// in native code exactly like the default interpreter: the block tier is
    /// skipped whenever a live slot holds a non-numeric value, and JIT-compiled
    /// integer `Add`/`Sub`/`Mul` are emitted with overflow checks that bail back
    /// to the interpreter (where the hook runs). Native code compiled in strict
    /// mode is cached separately from the coercing kind, so the two never mix.
    pub fn set_numeric_hook(&mut self, hook: NumericHook) {
        self.numeric_hook = Some(hook);
        #[cfg(feature = "jit")]
        {
            // Eligibility is cached per chunk hash and is policy-dependent.
            self.block_eligible_cached = None;
        }
    }

    /// Install an [`OutputSink`] so `Op::Print`/`Op::PrintLn` route through
    /// `sink` instead of `std::io::stdout()`. Frontends running fusevm in a
    /// browser web worker use this to capture output and bridge it to the JS
    /// host (wasm has no real stdout). With no sink installed, output is
    /// byte-for-byte identical to the previous direct-stdout behaviour.
    pub fn set_output_sink(&mut self, sink: OutputSink) {
        self.output_sink = Some(sink);
    }

    /// Install an [`InputSource`] so `Op::ReadLine` pulls from `source` instead
    /// of `std::io::stdin()`. The closure returns one line per call (newline
    /// trimmed) or `None` at end of input (pushed as `Value::Undef`).
    pub fn set_input_source(&mut self, source: InputSource) {
        self.input_source = Some(source);
    }

    /// Write `s` to the installed [`OutputSink`], or to `std::io::stdout()` when
    /// none is set. Callers buffer the print args into one `s` first so the sink
    /// borrow stays disjoint from the value stack.
    fn emit_output(&mut self, s: &str) {
        if let Some(sink) = self.output_sink.as_mut() {
            sink(s);
        } else {
            use std::io::Write;
            let stdout = std::io::stdout();
            let mut lock = stdout.lock();
            let _ = lock.write_all(s.as_bytes());
        }
    }

    /// Whether a [`NumericHook`] is installed (strict numeric mode).
    pub fn is_strict_numeric(&self) -> bool {
        self.numeric_hook.is_some()
    }

    /// Narrow the range the VM keeps as a native `Value::Int` (strict mode only).
    ///
    /// A Lisp whose integers are tagged has fewer than 64 bits for a fixnum —
    /// Emacs gets 62, so `most-positive-fixnum` is 2^61-1 — and an arithmetic
    /// result outside that range is a bignum, *even though it still fits an
    /// `i64`*. Setting the range makes strict mode delegate those results to the
    /// [`NumericHook`] alongside true `i64` overflow, so the host can widen them.
    /// JIT-compiled code carries the same bounds check (two ALU ops folded into
    /// the overflow accumulator — still no branch on the hot path).
    ///
    /// Without this, the host would see only `i64` overflow and integers in the
    /// 2^61..2^63 band would masquerade as fixnums.
    pub fn set_fixnum_range(&mut self, lo: i64, hi: i64) {
        self.fixnum_range = Some((lo, hi));
        #[cfg(feature = "jit")]
        {
            self.block_eligible_cached = None;
        }
    }

    /// Whether `n` is representable as a native fixnum under the current range.
    #[inline(always)]
    fn in_fixnum_range(&self, n: i64) -> bool {
        match self.fixnum_range {
            Some((lo, hi)) => n >= lo && n <= hi,
            None => true,
        }
    }

    /// Enable the tracing JIT for this VM. After this call, hot loops
    /// (loops crossing the backedge threshold) will be recorded and JIT-
    /// compiled at runtime; subsequent iterations dispatch through the
    /// compiled trace.
    ///
    /// Phase 1 limits: only int-slot loops with a single backward branch and
    /// no internal jumps are traceable. Loops outside that envelope continue
    /// to run in the interpreter.
    #[cfg(feature = "jit")]
    pub fn enable_tracing_jit(&mut self) {
        self.tracing_jit = true;
    }

    /// Disable the tracing JIT. Existing compiled traces remain in the
    /// thread-local cache but are no longer consulted from this VM.
    #[cfg(feature = "jit")]
    pub fn disable_tracing_jit(&mut self) {
        self.tracing_jit = false;
        self.recorder = None;
    }

    /// Reset the VM for re-use with a new chunk, preserving internal
    /// `Vec` allocations to avoid the construction cost of `VM::new`.
    ///
    /// State that's cleared:
    /// - Value stack (truncated, capacity preserved)
    /// - Frame stack (rebuilt with one entry pointing at the new chunk)
    /// - Globals (resized to match the new chunk's name pool)
    /// - Instruction pointer, halted flag, exit status
    /// - Tracing JIT recorder / slot buffers / deopt info
    /// - Cached block-JIT eligibility (the new chunk has a different hash)
    ///
    /// State that's preserved:
    /// - Tracing JIT enabled flag
    /// - Extension handlers (`ext_handler`, `ext_wide_handler`)
    /// - Builtin table
    /// - Shell host
    ///
    /// This pairs with [`VMPool`] for hot-path callers that run many
    /// chunks back-to-back and want to skip the per-call allocation cost
    /// of `VM::new`.
    pub fn reset(&mut self, chunk: Chunk) {
        self.stack.clear();
        self.frames.clear();
        let num_names = chunk.names.len();
        self.globals.clear();
        self.globals.resize(num_names, Value::Undef);
        self.frames.push(Frame {
            return_ip: 0,
            stack_base: 0,
            slots: Vec::with_capacity(16),
        });
        self.ip = 0;
        self.last_status = 0;
        self.halted = false;
        self.awk_rand_seed = 1;
        self.chunk = chunk;
        #[cfg(feature = "jit")]
        {
            self.recorder = None;
            self.slot_buf.clear();
            self.slot_kinds_buf.clear();
            self.deopt_info = DeoptInfo::zeroed();
            self.block_eligible_cached = None;
        }
    }

    /// Register the frontend shell host. Replaces any prior host.
    pub fn set_shell_host(&mut self, host: Box<dyn ShellHost>) {
        self.host = Some(host);
    }

    /// Register the frontend AWK host. Replaces any prior host. The VM then
    /// routes the reserved AWK op range to it (see [`crate::awk_host::AwkHost`]).
    pub fn set_awk_host(&mut self, host: Box<dyn AwkHost>) {
        self.awk_host = Some(host);
    }

    /// AWK control-flow signal raised by the most recent `run()`, if any. An
    /// awk frontend reads this after `run()` returns to map `Op::AwkSignal`
    /// codes (`awk_builtins::signal::{NEXT,NEXTFILE,EXIT}`) onto its own
    /// record/file/exit control flow. `None` when no signal was raised (always
    /// the case for zshrs/stryke, which never emit `Op::AwkSignal`).
    pub fn awk_signal(&self) -> Option<u8> {
        self.awk_signal
    }

    /// Take the pending cooperative-concurrency scheduling request, if any. The
    /// [`crate::sched::Scheduler`] driver calls this after `run()` returns: `Some`
    /// means a goroutine/channel op halted the VM to request scheduling; `None`
    /// means the VM finished (or halted for another reason). Clears the slot.
    pub fn take_sched(&mut self) -> Option<crate::sched::SchedReq> {
        self.sched.take()
    }

    /// Clear the halt flag so a parked goroutine VM resumes on the next `run()`
    /// (which continues from the current `ip`, past the op that parked it). Used
    /// by [`crate::sched::Scheduler`]; a plain schedulerless `run()` never needs it.
    pub fn clear_halt(&mut self) {
        self.halted = false;
    }

    /// Register a handler for `Op::Extended(id, arg)` opcodes.
    pub fn set_extension_handler(&mut self, handler: ExtensionHandler) {
        self.ext_handler = Some(handler);
    }

    /// Register a handler for `Op::ExtendedWide(id, payload)` opcodes.
    pub fn set_extension_wide_handler(&mut self, handler: ExtensionWideHandler) {
        self.ext_wide_handler = Some(handler);
    }

    /// Register a builtin function by ID. `CallBuiltin(id, argc)` dispatches
    /// directly through the function pointer — no name lookup at runtime.
    pub fn register_builtin(&mut self, id: u16, handler: BuiltinHandler) {
        let idx = id as usize;
        if idx >= self.builtin_table.len() {
            self.builtin_table.resize(idx + 1, None);
        }
        self.builtin_table[idx] = Some(handler);
    }

    /// Run a shell builtin by **name** at run time — the runtime analog of the
    /// compile-time [`Op::CallBuiltin`] opcode. The compiler resolves a literal
    /// command name to a builtin id and emits `CallBuiltin`, so builtins only
    /// dispatch for names known at compile time. A host that resolves a name
    /// only at run time — `builtin NAME`, `$var` command indirection, `eval`
    /// of a computed name — needs this to reach the same builtins.
    ///
    /// Resolves `name` via [`crate::shell_builtins::builtin_id`], pushes `args`
    /// as string values in argument order (identical to what the compiler emits
    /// ahead of `CallBuiltin`, which the handler's arg-pop reverses back), then
    /// invokes the registered handler and returns its status value. Returns
    /// `None` when `name` is not a known builtin or has no registered handler,
    /// so the caller falls through to function / external lookup.
    pub fn run_builtin_by_name(&mut self, name: &str, args: &[String]) -> Option<Value> {
        let id = crate::shell_builtins::builtin_id(name)?;
        let handler = match self.builtin_table.get(id as usize) {
            Some(Some(h)) => *h,
            _ => return None,
        };
        // `CallBuiltin`'s argc is a u8, so builtins take at most 255 args;
        // push exactly that many so the handler's arg-pop stays balanced.
        let argc = args.len().min(u8::MAX as usize);
        for a in &args[..argc] {
            self.push(Value::Str(std::sync::Arc::new(a.clone())));
        }
        Some(handler(self, argc as u8))
    }

    /// Externally request the VM to halt after the current op finishes.
    /// Used by host-side shell semantics like `set -e` post-command checks
    /// and `exit` from inside builtins to stop dispatch at a safe point.
    pub fn request_halt(&mut self) {
        self.halted = true;
    }

    // ── Tracing JIT integration helpers ──

    /// Snapshot the current frame's slots into the i64 + slot-kind buffers.
    ///
    /// Slots that don't fit cleanly into i64 (Array/Hash/String/etc) are
    /// reported as `SlotKind::Int` with i64 value 0 — they will fail the
    /// trace's entry guard if the recorded trace expected Int there, which
    /// is the desired behavior (skip the trace, fall back to interpreter).
    ///
    /// Specialized fast paths for 0-slot and 1-slot frames (the common
    /// case for tight inner loops) — these skip Vec resize bookkeeping
    /// and the iterator loop, saving ~20-50 ns per `vm.run()` invocation.
    #[cfg(feature = "jit")]
    #[inline]
    fn refresh_slot_buffers(&mut self) {
        let frame = match self.frames.last() {
            Some(f) => f,
            None => return,
        };
        let n = frame.slots.len();
        let mut all_num = true;
        match n {
            0 => {
                self.slot_buf.clear();
                self.slot_kinds_buf.clear();
            }
            1 => {
                let (i, kind) = match &frame.slots[0] {
                    Value::Int(v) => (*v, SlotKind::Int),
                    Value::Float(f) => (f.to_bits() as i64, SlotKind::Float),
                    Value::Bool(b) => (*b as i64, SlotKind::Int),
                    _ => {
                        all_num = false;
                        (0, SlotKind::Int)
                    }
                };
                if self.slot_buf.is_empty() {
                    self.slot_buf.push(i);
                    self.slot_kinds_buf.push(kind);
                } else {
                    self.slot_buf.truncate(1);
                    self.slot_buf[0] = i;
                    self.slot_kinds_buf.truncate(1);
                    self.slot_kinds_buf[0] = kind;
                }
            }
            _ => {
                self.slot_buf.clear();
                self.slot_kinds_buf.clear();
                self.slot_buf.reserve(n);
                self.slot_kinds_buf.reserve(n);
                for v in &frame.slots {
                    let (i, kind) = match v {
                        Value::Int(n) => (*n, SlotKind::Int),
                        Value::Float(f) => (f.to_bits() as i64, SlotKind::Float),
                        Value::Bool(b) => (*b as i64, SlotKind::Int),
                        _ => {
                            all_num = false;
                            (0, SlotKind::Int)
                        }
                    };
                    self.slot_buf.push(i);
                    self.slot_kinds_buf.push(kind);
                }
            }
        }
        self.slots_all_numeric = all_num;
    }

    /// Copy the i64 slot buffer back into the current frame's slots,
    /// materializing Int and Float kinds. Float slots are stored as i64
    /// bit patterns in the buffer; recover via `f64::from_bits`. Slots of
    /// other kinds (Array, Hash, etc.) are left untouched — those slots
    /// would have prevented trace install if referenced.
    ///
    /// Specialized for 0/1-slot frames (common case).
    #[cfg(feature = "jit")]
    #[inline]
    fn write_slots_back(&mut self) {
        let frame = match self.frames.last_mut() {
            Some(f) => f,
            None => return,
        };
        let n = frame.slots.len().min(self.slot_buf.len());
        match n {
            0 => {}
            1 => match self.slot_kinds_buf.first() {
                Some(SlotKind::Int) => frame.slots[0] = Value::Int(self.slot_buf[0]),
                Some(SlotKind::Float) => {
                    frame.slots[0] = Value::Float(f64::from_bits(self.slot_buf[0] as u64));
                }
                None => {}
            },
            _ => {
                for i in 0..n {
                    match self.slot_kinds_buf.get(i) {
                        Some(SlotKind::Int) => {
                            frame.slots[i] = Value::Int(self.slot_buf[i]);
                        }
                        Some(SlotKind::Float) => {
                            frame.slots[i] = Value::Float(f64::from_bits(self.slot_buf[i] as u64));
                        }
                        None => {}
                    }
                }
            }
        }
    }

    /// Consult the trace cache at a backward-branch site and return the IP
    /// the interpreter should resume at. If a compiled trace runs, slot state
    /// is copied back from the trace's i64 buffer into the frame, and any
    /// inlined callee frames the trace recorded at a side-exit are
    /// materialized as synthetic `Frame`s on `self.frames` so the
    /// interpreter can resume mid-callee with a correctly shaped call stack.
    #[cfg(feature = "jit")]
    fn lookup_trace_for_backward(&mut self, anchor_ip: usize, fallthrough_ip: usize) -> usize {
        self.refresh_slot_buffers();
        let lookup = self.jit.trace_lookup(
            &self.chunk,
            anchor_ip,
            &mut self.slot_buf,
            &self.slot_kinds_buf,
            &mut self.deopt_info,
        );
        match lookup {
            TraceLookup::Ran { resume_ip } => {
                self.write_slots_back();
                self.materialize_deopt_frames();
                // Phase 9: if the trace deopted (returned non-fallthrough),
                // try to chain into a side trace at the resume IP.
                self.chain_side_traces(anchor_ip, fallthrough_ip, resume_ip)
            }
            TraceLookup::StartRecording => {
                // Main-trace path: record_anchor_ip == close_anchor_ip
                // (the loop header). Side-trace recording is armed via the
                // chained-dispatch path below.
                self.recorder = Some(TraceRecorder {
                    record_anchor_ip: anchor_ip,
                    close_anchor_ip: anchor_ip,
                    fallthrough_ip,
                    ops: Vec::new(),
                    recorded_ips: Vec::new(),
                    slot_kinds_at_anchor: self.slot_kinds_buf.clone(),
                    entered_ips: Vec::new(),
                    aborted: false,
                });
                anchor_ip
            }
            TraceLookup::NotHot | TraceLookup::GuardMismatch | TraceLookup::Skip => anchor_ip,
        }
    }

    /// Phase 9: chained dispatch through linked traces.
    ///
    /// When the main trace's `compiled.invoke` returns a non-fallthrough
    /// resume IP (a brif guard fired and we side-exited), this method
    /// attempts to dispatch a side trace registered at that resume IP.
    /// Iterates up to `MAX_TRACE_CHAIN` times so a sequence of linked
    /// side-exits can resolve through their respective side traces.
    ///
    /// Side-trace recording is armed when a side-exit IP becomes hot (the
    /// `StartRecording` branch). The recorder is set up with
    /// `close_anchor_ip = main_anchor`, so the side trace closes correctly
    /// when the enclosing loop's backward branch fires.
    ///
    /// The main trace's `side_exit_count` is incremented only when the
    /// chain bottoms out without finding a side trace — exits that are
    /// being absorbed productively shouldn't push the main trace toward
    /// blacklisting.
    #[cfg(feature = "jit")]
    fn chain_side_traces(
        &mut self,
        main_anchor: usize,
        main_fallthrough: usize,
        first_resume: usize,
    ) -> usize {
        let mut current = first_resume;
        if current == main_fallthrough {
            return current;
        }
        let max_chain = self.jit.get_config().max_trace_chain;
        for _ in 0..max_chain {
            // The chained trace at `current` may itself have a different
            // fallthrough; we re-fetch on each iteration.
            let chained_fallthrough = self
                .jit
                .trace_loop_anchors(&self.chunk, current)
                .map(|(_, fallthrough)| fallthrough);

            self.refresh_slot_buffers();
            let lookup = self.jit.trace_lookup(
                &self.chunk,
                current,
                &mut self.slot_buf,
                &self.slot_kinds_buf,
                &mut self.deopt_info,
            );
            match lookup {
                TraceLookup::Ran { resume_ip } => {
                    self.write_slots_back();
                    self.materialize_deopt_frames();
                    current = resume_ip;
                    if Some(current) == chained_fallthrough {
                        return current;
                    }
                }
                TraceLookup::StartRecording => {
                    // Arm side-trace recording. The side trace's close
                    // anchor is the main loop's header; its fallthrough is
                    // the main loop's post-loop IP. Slot-kind snapshot is
                    // taken at THIS moment (post-deopt state).
                    self.recorder = Some(TraceRecorder {
                        record_anchor_ip: current,
                        close_anchor_ip: main_anchor,
                        fallthrough_ip: main_fallthrough,
                        ops: Vec::new(),
                        recorded_ips: Vec::new(),
                        slot_kinds_at_anchor: self.slot_kinds_buf.clone(),
                        entered_ips: Vec::new(),
                        aborted: false,
                    });
                    return current;
                }
                _ => {
                    // No side trace available; count toward main trace's
                    // blacklist budget.
                    self.jit.trace_bump_side_exit(&self.chunk, main_anchor);
                    return current;
                }
            }
        }
        current
    }

    /// Push synthetic `Frame`s onto `self.frames` for each inlined callee
    /// frame the trace recorded at side-exit, then push any remaining
    /// abstract-stack values from the trace onto `self.stack` as
    /// `Value::Int`. Order matters: stack values are pushed BEFORE the
    /// frames are pushed, because each frame's `stack_base` snapshots
    /// `self.stack.len()` AFTER the stack values are placed — that way
    /// when the synthetic frame eventually returns and truncates to
    /// `stack_base`, those values are correctly retained. Phase 5+.
    #[cfg(feature = "jit")]
    fn materialize_deopt_frames(&mut self) {
        // 1. Push the abstract stack (in trace order; entry [0] ends up at
        //    the bottom, [N-1] at the top). Float entries are bit-cast back
        //    via `f64::from_bits`; everything else is treated as `i64`.
        let stack_count = self.deopt_info.stack_count;
        for i in 0..stack_count {
            let raw = self.deopt_info.stack_buf[i];
            let v = match self.deopt_info.stack_kinds[i] {
                crate::jit::STACK_KIND_FLOAT => Value::Float(f64::from_bits(raw as u64)),
                _ => Value::Int(raw),
            };
            self.stack.push(v);
        }
        // 2. Materialize inlined frames.
        let count = self.deopt_info.frame_count;
        if count == 0 {
            return;
        }
        for i in 0..count {
            let df = &self.deopt_info.frames[i];
            let mut slots: Vec<Value> = Vec::with_capacity(df.slot_count);
            for j in 0..df.slot_count {
                slots.push(Value::Int(df.slots[j]));
            }
            self.frames.push(Frame {
                return_ip: df.return_ip,
                stack_base: self.stack.len(),
                slots,
            });
        }
    }

    /// Finalize the active recording: either close (install) when the just-
    /// dispatched jump landed back at the anchor and the trace is eligible,
    /// or abort and discard. Safe to call only when `self.recorder.is_some()`.
    #[cfg(feature = "jit")]
    fn finalize_recorder(&mut self) {
        let Some(rec) = self.recorder.as_ref() else {
            return;
        };
        let aborted = rec.aborted;
        let close_anchor = rec.close_anchor_ip;
        let record_anchor = rec.record_anchor_ip;
        // Trace closes when the just-dispatched jump lands at the recorded
        // close anchor. For main traces this is the loop header; for side
        // traces it's the enclosing loop's header (the side trace
        // started at a side-exit IP but still closes when the loop iterates).
        if !aborted && self.ip == close_anchor {
            let r = self.recorder.take().unwrap();
            if self.jit.is_trace_eligible(&r.ops, r.close_anchor_ip) {
                // Phase 9: when record != close (side trace), install via
                // the kinded variant so the IR's "continuation" branch
                // exits rather than looping back.
                let installed = self.jit.trace_install_with_kind(
                    &self.chunk,
                    r.record_anchor_ip,
                    r.close_anchor_ip,
                    r.fallthrough_ip,
                    &r.ops,
                    &r.recorded_ips,
                    &r.slot_kinds_at_anchor,
                );
                if !installed {
                    self.jit.trace_abort(&self.chunk, r.record_anchor_ip);
                }
            } else {
                self.jit.trace_abort(&self.chunk, r.record_anchor_ip);
            }
        } else {
            // Trace dispatch landed somewhere unexpected — abort. The
            // record_anchor_ip captured before take() is the cache key.
            let _ = self.recorder.take();
            self.jit.trace_abort(&self.chunk, record_anchor);
        }
    }

    // ── Stack operations ──

    /// Push `val` onto the value stack. Inlined for hot-path callers
    /// (extension handlers, builtin shims) that bypass the dispatch
    /// loop's own push.
    #[inline(always)]
    pub fn push(&mut self, val: Value) {
        self.stack.push(val);
    }

    /// Pop the top of the value stack, returning `Value::Undef` if the
    /// stack is empty. Returning Undef rather than panicking matches
    /// Perl's "underflow is undef" semantic and lets extension/builtin
    /// handlers stay panic-free under malformed bytecode.
    #[inline(always)]
    pub fn pop(&mut self) -> Value {
        self.stack.pop().unwrap_or(Value::Undef)
    }

    /// Borrow the top of the value stack without popping. Returns a
    /// reference to `Value::Undef` when the stack is empty.
    #[inline(always)]
    pub fn peek(&self) -> &Value {
        self.stack.last().unwrap_or(&Value::Undef)
    }

    // ── Type-specialized helpers (avoid to_float coercion on hot paths) ──

    /// Pop two values; if both Int, apply int_op. Otherwise promote to float.
    ///
    /// `ck_op` is the checked form of `int_op` and is consulted only in strict
    /// numeric mode (a [`NumericHook`] is installed), where an integer result
    /// that does not fit `i64` — and any operand that is not a number — goes to
    /// the host instead of being wrapped or coerced. Returns the error message
    /// if the hook signalled.
    #[inline(always)]
    fn arith_int_fast(
        &mut self,
        op: NumOp,
        int_op: fn(i64, i64) -> i64,
        ck_op: fn(i64, i64) -> Option<i64>,
        float_op: fn(f64, f64) -> f64,
    ) -> Option<String> {
        let len = self.stack.len();
        if len < 2 {
            return None;
        }
        // Borrow both slots without popping (avoid branch + unwrap_or)
        let b = &self.stack[len - 1];
        let a = &self.stack[len - 2];
        let result = match (a, b, &self.numeric_hook) {
            // Fast path, both policies: two fixnums.
            (Value::Int(x), Value::Int(y), None) => Value::Int(int_op(*x, *y)),
            (Value::Int(x), Value::Int(y), Some(hook)) => match ck_op(*x, *y) {
                // Exact, and inside the host's fixnum range.
                Some(r) if self.in_fixnum_range(r) => Value::Int(r),
                // Overflowed `i64`, or left the host's fixnum range: either way
                // only the host can represent it (a bignum).
                _ => match hook(op, a, b) {
                    Ok(v) => v,
                    Err(e) => return Some(e),
                },
            },
            // Mixed int/float and float/float are exact in f64 under either
            // policy — never delegate.
            (a, b, _) if is_native_num(a) && is_native_num(b) => {
                Value::Float(float_op(a.to_float(), b.to_float()))
            }
            // A non-number. Coerce (awk/shell) or delegate (strict).
            (a, b, None) => Value::Float(float_op(a.to_float(), b.to_float())),
            (a, b, Some(hook)) => match hook(op, a, b) {
                Ok(v) => v,
                Err(e) => return Some(e),
            },
        };
        self.stack.truncate(len - 2);
        self.stack.push(result);
        None
    }

    /// Pop two values; compare as int if both Int, otherwise float.
    /// Push Bool(true/false).
    ///
    /// Comparison cannot overflow, so in strict numeric mode only a non-numeric
    /// operand delegates to the [`NumericHook`].
    #[inline(always)]
    fn cmp_int_fast(
        &mut self,
        op: NumOp,
        int_cmp: fn(i64, i64) -> bool,
        float_cmp: fn(f64, f64) -> bool,
    ) -> Option<String> {
        let len = self.stack.len();
        if len < 2 {
            return None;
        }
        let b = &self.stack[len - 1];
        let a = &self.stack[len - 2];
        let result = match (a, b, &self.numeric_hook) {
            (Value::Int(x), Value::Int(y), _) => Value::Bool(int_cmp(*x, *y)),
            (a, b, _) if is_native_num(a) && is_native_num(b) => {
                Value::Bool(float_cmp(a.to_float(), b.to_float()))
            }
            (a, b, None) => Value::Bool(float_cmp(a.to_float(), b.to_float())),
            (a, b, Some(hook)) => match hook(op, a, b) {
                Ok(v) => v,
                Err(e) => return Some(e),
            },
        };
        self.stack.truncate(len - 2);
        self.stack.push(result);
        None
    }

    /// Pop two operands and hand them to the [`NumericHook`]. Only called in
    /// strict mode, for the ops (`Div`, `Pow`) whose native path is float-only
    /// and therefore has no overflow case to check — just a type case.
    #[inline(always)]
    fn delegate_binary(&mut self, op: NumOp) -> Option<String> {
        let b = self.pop();
        let a = self.pop();
        let hook = match &self.numeric_hook {
            Some(h) => h,
            None => return None,
        };
        match hook(op, &a, &b) {
            Ok(v) => {
                self.push(v);
                None
            }
            Err(e) => Some(e),
        }
    }

    /// Unary negate, strict-aware: `i64::MIN` has no positive counterpart and a
    /// non-number is not negatable, so both go to the [`NumericHook`].
    #[inline(always)]
    fn negate_strict(&mut self) -> Option<String> {
        let val = self.pop();
        let result = match (&val, &self.numeric_hook) {
            (Value::Int(n), None) => Value::Int(n.wrapping_neg()),
            (Value::Int(n), Some(hook)) => match n.checked_neg() {
                Some(r) if self.in_fixnum_range(r) => Value::Int(r),
                _ => match hook(NumOp::Neg, &val, &Value::Undef) {
                    Ok(v) => v,
                    Err(e) => return Some(e),
                },
            },
            (Value::Float(f), _) => Value::Float(-f),
            (v, None) => Value::Float(-v.to_float()),
            (v, Some(hook)) => match hook(NumOp::Neg, v, &Value::Undef) {
                Ok(v) => v,
                Err(e) => return Some(e),
            },
        };
        self.push(result);
        None
    }

    // ── Main dispatch loop ──

    /// Execute the loaded chunk until completion or error.
    ///
    /// Phase 10: tiered auto-dispatch. When `tracing_jit` is enabled the
    /// VM consults all three Cranelift tiers in priority order:
    ///
    /// 1. **Block JIT** — if the entire chunk is block-eligible, the
    ///    block-JIT cache returns `Some(result)` after its own warmup
    ///    threshold and the whole chunk runs in native code with zero
    ///    interpreter dispatch.
    /// 2. **Tracing JIT** — when block JIT doesn't apply, the dispatch
    ///    loop runs with the recorder armed at backward branches; hot
    ///    loops compile to traces that take over subsequent iterations.
    /// 3. **Interpreter** — fallback for cold code and chunks neither
    ///    tier handles.
    ///
    /// Block JIT is tried first because, when it applies, it has zero
    /// VM-side overhead (direct fn-ptr through the slot pointer). For
    /// chunks block JIT can't take, control falls through to the
    /// interpreter with tracing JIT integrated. The two tiers don't
    /// compete on the same chunk: block-eligible chunks short-circuit
    /// before tracing JIT records anything.
    pub fn run(&mut self) -> VMResult {
        // A fresh execution: clear any AWK signal raised by a prior run on a
        // reused VM. (zshrs/stryke never emit `Op::AwkSignal`, so this stays
        // `None` for them.)
        self.awk_signal = None;
        // A prior park request is consumed by the scheduler before it resumes
        // this VM; clear any stragglers so a plain (schedulerless) run is unaffected.
        self.sched = None;
        // Phase 10: try block JIT first for fully-eligible chunks. The
        // block-JIT cache has its own threshold (10 invocations); the
        // call returns None until it warms up, at which point the whole
        // chunk runs in native code.
        //
        // VM-side eligibility cache (`block_eligible_cached`) saves the
        // TLS HashMap lookup `JitCompiler::is_block_eligible` would
        // otherwise do on every run. The slot buffer must be valid on
        // every invocation because `try_run_block` may compile + invoke
        // the same call (on threshold cross), so we can't skip the
        // refresh based on warmup state.
        // Publish this VM's numeric policy to the thread's JIT: it decides op
        // eligibility, salts every cache key, and switches integer arithmetic to
        // its overflow-checked lowering.
        #[cfg(feature = "jit")]
        crate::jit::set_strict_numeric(self.numeric_hook.is_some(), self.fixnum_range);

        #[cfg(feature = "jit")]
        if self.tracing_jit && self.frames.len() == 1 && self.ip == 0 {
            let eligible = match self.block_eligible_cached {
                Some(v) => v,
                None => {
                    let v = self.jit.is_block_eligible(&self.chunk);
                    self.block_eligible_cached = Some(v);
                    v
                }
            };
            if eligible {
                self.refresh_slot_buffers();
                // Strict numeric mode: a non-numeric slot reaches native code as
                // the integer 0 (`refresh_slot_buffers` has no richer encoding),
                // which is precisely the silent coercion the `NumericHook` exists
                // to reject. Falling out of this block runs the chunk in the
                // interpreter, where the hook sees the real value.
                let strict_blocked = self.numeric_hook.is_some() && !self.slots_all_numeric;
                if !strict_blocked {
                    if let Some(result) = self.jit.try_run_block_typed_kinded(
                        &self.chunk,
                        &mut self.slot_buf,
                        &self.slot_kinds_buf,
                    ) {
                        // Strict numeric mode: an integer op in the compiled block
                        // overflowed i64. The native result is the wrapped (wrong)
                        // value, so discard it *and* skip the slot writeback — the
                        // block's ops are pure, so nothing has escaped — and let the
                        // interpreter re-run the chunk, where the `NumericHook`
                        // produces the exact result (elisp: a bignum).
                        if crate::jit::take_num_overflow_trap() {
                            self.ip = 0;
                        } else {
                            // A JIT-compiled AwkDivJit/AwkModJit may have hit a zero
                            // divisor and set the thread-local trap code before
                            // returning. Honor it as the awk fatal, discarding the
                            // (garbage) block result and slot writeback.
                            match crate::jit::take_awk_div_trap() {
                                1 => {
                                    return VMResult::Error(
                                        "division by zero attempted".to_string(),
                                    )
                                }
                                2 => {
                                    return VMResult::Error(
                                        "division by zero attempted in `%'".to_string(),
                                    )
                                }
                                3 => {
                                    return VMResult::Error(
                                        "lshift: negative values are not allowed".to_string(),
                                    )
                                }
                                4 => {
                                    return VMResult::Error(
                                        "rshift: negative values are not allowed".to_string(),
                                    )
                                }
                                5 => {
                                    return VMResult::Error(
                                        "compl: negative value is not allowed".to_string(),
                                    )
                                }
                                _ => {}
                            }
                            self.write_slots_back();
                            self.halted = true;
                            // The block tier returns its result in an i64 register, with a
                            // float riding back as its raw bit pattern. Decode through
                            // `BlockNum` — wrapping the register in `Value::Int`
                            // unconditionally (as this did before) truncated every float
                            // chunk result to an integer on the second and later runs of
                            // a chunk, once the block cache was warm: `LoadFloat(2.5)`
                            // returned `Int(2)`.
                            return VMResult::Ok(match result {
                                crate::jit::BlockNum::Int(n) => Value::Int(n),
                                crate::jit::BlockNum::Float(f) => Value::Float(f),
                            });
                        }
                    }
                }
            }
        }

        let ops = &self.chunk.ops as *const Vec<Op>;
        // SAFETY: self.chunk.ops is not mutated during execution.
        // We take a pointer to avoid borrow checker issues with &self.chunk.ops
        // while mutating self.stack/frames/globals.
        let ops = unsafe { &*ops };

        while self.ip < ops.len() && !self.halted {
            // Zero-clone: borrow the op instead of cloning
            let ip = self.ip;
            self.ip += 1;

            // Tracing JIT: capture this op into the active recording, if any.
            // Push happens before dispatch so the closing branch is included.
            // Track whether the recorder was armed BEFORE this dispatch step —
            // a recorder armed inside this step (via `StartRecording`) must not
            // finalize until the NEXT iteration, otherwise the very dispatch
            // step that armed it would also close it with an empty op list.
            #[cfg(feature = "jit")]
            let recorder_was_armed = self.recorder.is_some();
            #[cfg(not(feature = "jit"))]
            let recorder_was_armed = false;
            #[cfg(feature = "jit")]
            if self.recorder.is_some() {
                let cfg = self.jit.get_config();
                let max_len = cfg.max_trace_len;
                let max_inline_recursion = cfg.max_inline_recursion;
                let cur_op = ops[ip].clone();
                // Resolve sub-entry up front (immutable chunk borrow) so the
                // mutable recorder borrow below doesn't collide.
                let resolved_entry = if let Op::Call(name_idx, _) = &cur_op {
                    self.chunk.find_sub(*name_idx)
                } else {
                    None
                };
                let rec = self.recorder.as_mut().unwrap();
                if !rec.aborted {
                    rec.ops.push(cur_op.clone());
                    rec.recorded_ips.push(ip);
                    if rec.ops.len() > max_len {
                        rec.aborted = true;
                    } else {
                        // Maintain inlined-frame stack and abort on patterns
                        // the trace JIT can't represent in phase 2.
                        match cur_op {
                            Op::Call(_, _) => match resolved_entry {
                                Some(entry_ip) => {
                                    // Phase 8: bounded recursion. Allow a
                                    // self-call to be inlined up to
                                    // `MAX_INLINE_RECURSION` levels deep
                                    // before aborting. Each push is one
                                    // level — the depth limit naturally
                                    // bounds tail-recursive helpers without
                                    // explicit base-case detection.
                                    let occurrences = rec
                                        .entered_ips
                                        .iter()
                                        .filter(|&&ip| ip == entry_ip)
                                        .count();
                                    if occurrences >= max_inline_recursion {
                                        rec.aborted = true;
                                    } else {
                                        rec.entered_ips.push(entry_ip);
                                    }
                                }
                                None => rec.aborted = true, // unknown sub
                            },
                            Op::Return | Op::ReturnValue => {
                                if rec.entered_ips.is_empty() {
                                    rec.aborted = true; // unbalanced — would
                                                        // pop the caller frame
                                } else {
                                    rec.entered_ips.pop();
                                }
                            }
                            Op::CallBuiltin(_, _) => rec.aborted = true,
                            // Most AWK ops are host calls (like CallBuiltin) and
                            // can't appear in a compiled trace — abort. Pure ops
                            // with native CLIF codegen in `emit_data_op`
                            // (e.g. AwkInt → trunc) are omitted here so they can
                            // be recorded and compiled.
                            Op::AwkFieldGet
                            | Op::AwkFieldSet
                            | Op::AwkNf
                            | Op::AwkSetRecord
                            | Op::AwkSpecialGet(_)
                            | Op::AwkSpecialSet(_)
                            | Op::AwkPrint(_)
                            | Op::AwkPrintf(_)
                            | Op::AwkSprintf(_)
                            | Op::AwkGetline(_)
                            | Op::AwkLength(_)
                            | Op::AwkSubstr(_)
                            | Op::AwkIndex
                            | Op::AwkSplit(_)
                            | Op::AwkSub(_)
                            | Op::AwkGsub(_)
                            | Op::AwkGensub(_)
                            | Op::AwkMatch
                            | Op::AwkToLower
                            | Op::AwkToUpper
                            | Op::AwkSqrt
                            | Op::AwkSin
                            | Op::AwkCos
                            | Op::AwkExp
                            | Op::AwkLog
                            | Op::AwkAtan2
                            | Op::AwkDiv
                            | Op::AwkMod
                            | Op::AwkDivJit
                            | Op::AwkModJit
                            | Op::AwkSqrtJit
                            | Op::AwkLogJit
                            | Op::AwkLshiftJit
                            | Op::AwkRshiftJit
                            | Op::AwkComplJit
                            | Op::AwkAnd(_)
                            | Op::AwkOr(_)
                            | Op::AwkXor(_)
                            | Op::AwkCompl
                            | Op::AwkLshift
                            | Op::AwkRshift
                            | Op::AwkStrtonum
                            | Op::AwkSystime
                            | Op::AwkRand
                            | Op::AwkSrand(_)
                            | Op::AwkStrftime(_)
                            | Op::AwkMktime(_)
                            | Op::AwkOrd
                            | Op::AwkChr
                            | Op::AwkMkbool
                            | Op::AwkIntdiv
                            | Op::AwkIntdiv0
                            | Op::AwkArrayGet(_)
                            | Op::AwkArraySet(_)
                            | Op::AwkArrayExists(_)
                            | Op::AwkArrayDelete(_)
                            | Op::AwkArrayClear(_)
                            | Op::AwkArrayLen(_) => rec.aborted = true,
                            Op::AwkSignal(_) => rec.aborted = true,
                            // Concurrency ops halt for the scheduler; never trace.
                            Op::Go(_, _)
                            | Op::ChanMake
                            | Op::ChanSend
                            | Op::ChanRecv
                            | Op::ChanClose
                            | Op::Select(_, _)
                            | Op::CallDynamic(_) => rec.aborted = true,
                            _ => {}
                        }
                    }
                }
            }

            match self.exec_op(ops, ip, recorder_was_armed) {
                ExecFlow::Cont => {}
                ExecFlow::Ret(r) => return r,
            }
        }

        if let Some(val) = self.stack.pop() {
            VMResult::Ok(val)
        } else {
            VMResult::Halted
        }
    }

    /// Execute the single op at `ops[ip]` — the **single source of truth** for op
    /// semantics, shared by the interpreter loop ([`VM::run`]) and the AOT
    /// closed-world compiler (`fusevm::aot`), which emits native code that calls
    /// this for every op it does not specialize. Returns [`ExecFlow::Ret`] when
    /// an op terminates the run, else [`ExecFlow::Cont`]. Arms that previously
    /// `continue`d the dispatch loop now `return ExecFlow::Cont`, which skips the
    /// trailing recorder-finalize step exactly as `continue` did.
    #[cfg_attr(not(feature = "jit"), allow(unused_variables))]
    pub(crate) fn exec_op(&mut self, ops: &[Op], ip: usize, recorder_was_armed: bool) -> ExecFlow {
        use crate::awk_builtins as ab;
        match &ops[ip] {
            Op::Nop => {}

            // ── Constants ──
            Op::LoadInt(n) => self.push(Value::Int(*n)),
            Op::LoadFloat(f) => self.push(Value::Float(*f)),
            Op::LoadConst(idx) => {
                let val = match self.chunk.constants.get(*idx as usize) {
                    Some(Value::Int(n)) => Value::Int(*n),
                    Some(Value::Float(f)) => Value::Float(*f),
                    Some(Value::Bool(b)) => Value::Bool(*b),
                    Some(other) => other.clone(),
                    None => Value::Undef,
                };
                self.push(val);
            }
            Op::LoadTrue => self.push(Value::Bool(true)),
            Op::LoadFalse => self.push(Value::Bool(false)),
            Op::LoadUndef => self.push(Value::Undef),

            // ── Stack ──
            Op::Pop => {
                self.pop();
            }
            Op::Dup => {
                let val = self.peek().clone();
                self.push(val);
            }
            Op::Dup2 => {
                let len = self.stack.len();
                if len >= 2 {
                    let a = self.stack[len - 2].clone();
                    let b = self.stack[len - 1].clone();
                    self.push(a);
                    self.push(b);
                }
            }
            Op::Swap => {
                let len = self.stack.len();
                if len >= 2 {
                    self.stack.swap(len - 1, len - 2);
                }
            }
            Op::Rot => {
                let len = self.stack.len();
                if len >= 3 {
                    // [a, b, c] → [b, c, a] via two swaps instead of O(n) remove
                    self.stack.swap(len - 3, len - 2);
                    self.stack.swap(len - 2, len - 1);
                }
            }

            // ── Variables ──
            Op::GetVar(idx) => {
                let val = self.get_var(*idx);
                self.push(val);
            }
            Op::SetVar(idx) => {
                let val = self.pop();
                self.set_var(*idx, val);
            }
            Op::DeclareVar(idx) => {
                let val = self.pop();
                self.set_var(*idx, val);
            }
            Op::GetSlot(slot) => {
                let val = self.get_slot(*slot);
                self.push(val);
            }
            Op::SetSlot(slot) => {
                let val = self.pop();
                self.set_slot(*slot, val);
            }
            Op::SlotArrayGet(slot) => {
                let index = self.pop().to_int() as usize;
                let val = self.get_slot(*slot);
                let result = if let Value::Array(ref arr) = val {
                    arr.get(index).cloned().unwrap_or(Value::Undef)
                } else {
                    Value::Undef
                };
                self.push(result);
            }
            Op::SlotArraySet(slot) => {
                let index = self.pop().to_int() as usize;
                let val = self.pop();
                let arr_val = self.get_slot(*slot);
                if let Value::Array(mut arr) = arr_val {
                    if index >= arr.len() {
                        arr.resize(index + 1, Value::Undef);
                    }
                    arr[index] = val;
                    self.set_slot(*slot, Value::Array(arr));
                }
            }

            // ── Arithmetic (type-specialized: Int×Int avoids to_float) ──
            Op::Add => {
                if let Some(e) =
                    self.arith_int_fast(NumOp::Add, i64::wrapping_add, i64::checked_add, |a, b| {
                        a + b
                    })
                {
                    return ExecFlow::Ret(VMResult::Error(e));
                }
            }
            Op::Sub => {
                if let Some(e) =
                    self.arith_int_fast(NumOp::Sub, i64::wrapping_sub, i64::checked_sub, |a, b| {
                        a - b
                    })
                {
                    return ExecFlow::Ret(VMResult::Error(e));
                }
            }
            Op::Mul => {
                if let Some(e) =
                    self.arith_int_fast(NumOp::Mul, i64::wrapping_mul, i64::checked_mul, |a, b| {
                        a * b
                    })
                {
                    return ExecFlow::Ret(VMResult::Error(e));
                }
            }
            Op::Div => {
                // Strict mode delegates a non-numeric operand; the zero-divisor
                // and always-float results are unchanged (a frontend whose `/`
                // has other semantics — elisp's integer division — implements it
                // as a builtin rather than lowering to this op).
                if self.numeric_hook.is_some() {
                    let len = self.stack.len();
                    if len >= 2
                        && !(is_native_num(&self.stack[len - 1])
                            && is_native_num(&self.stack[len - 2]))
                    {
                        if let Some(e) = self.delegate_binary(NumOp::Div) {
                            return ExecFlow::Ret(VMResult::Error(e));
                        }
                        return ExecFlow::Cont;
                    }
                }
                let b = self.pop();
                let a = self.pop();
                let divisor = b.to_float();
                self.push(if divisor == 0.0 {
                    Value::Undef
                } else {
                    Value::Float(a.to_float() / divisor)
                });
            }
            Op::Mod => {
                if let Some(e) = self.arith_int_fast(
                    NumOp::Mod,
                    |x, y| if y != 0 { x % y } else { 0 },
                    |x, y| if y != 0 { x.checked_rem(y) } else { Some(0) },
                    |a, b| a % b,
                ) {
                    return ExecFlow::Ret(VMResult::Error(e));
                }
            }
            Op::Pow => {
                if self.numeric_hook.is_some() {
                    let len = self.stack.len();
                    if len >= 2
                        && !(is_native_num(&self.stack[len - 1])
                            && is_native_num(&self.stack[len - 2]))
                    {
                        if let Some(e) = self.delegate_binary(NumOp::Pow) {
                            return ExecFlow::Ret(VMResult::Error(e));
                        }
                        return ExecFlow::Cont;
                    }
                }
                let b = self.pop();
                let a = self.pop();
                self.push(Value::Float(a.to_float().powf(b.to_float())));
            }
            Op::Negate => {
                if let Some(e) = self.negate_strict() {
                    return ExecFlow::Ret(VMResult::Error(e));
                }
            }
            Op::Inc => {
                let val = self.pop();
                let result = match (&val, &self.numeric_hook) {
                    (Value::Int(n), None) => Value::Int(n.wrapping_add(1)),
                    (Value::Int(n), Some(hook)) => match n.checked_add(1) {
                        Some(r) if self.in_fixnum_range(r) => Value::Int(r),
                        _ => match hook(NumOp::Add, &val, &Value::Int(1)) {
                            Ok(v) => v,
                            Err(e) => return ExecFlow::Ret(VMResult::Error(e)),
                        },
                    },
                    (v, None) => Value::Int(v.to_int().wrapping_add(1)),
                    (v, Some(hook)) => match hook(NumOp::Add, v, &Value::Int(1)) {
                        Ok(v) => v,
                        Err(e) => return ExecFlow::Ret(VMResult::Error(e)),
                    },
                };
                self.push(result);
            }
            Op::Dec => {
                let val = self.pop();
                let result = match (&val, &self.numeric_hook) {
                    (Value::Int(n), None) => Value::Int(n.wrapping_sub(1)),
                    (Value::Int(n), Some(hook)) => match n.checked_sub(1) {
                        Some(r) if self.in_fixnum_range(r) => Value::Int(r),
                        _ => match hook(NumOp::Sub, &val, &Value::Int(1)) {
                            Ok(v) => v,
                            Err(e) => return ExecFlow::Ret(VMResult::Error(e)),
                        },
                    },
                    (v, None) => Value::Int(v.to_int().wrapping_sub(1)),
                    (v, Some(hook)) => match hook(NumOp::Sub, v, &Value::Int(1)) {
                        Ok(v) => v,
                        Err(e) => return ExecFlow::Ret(VMResult::Error(e)),
                    },
                };
                self.push(result);
            }

            // ── String ──
            Op::Concat => {
                let b = self.pop();
                let a = self.pop();
                let a_s = a.as_str_cow();
                let b_s = b.as_str_cow();
                let mut s = String::with_capacity(a_s.len() + b_s.len());
                s.push_str(&a_s);
                s.push_str(&b_s);
                self.push(Value::str(s));
            }
            Op::StringRepeat => {
                let count = self.pop().to_int();
                let s = self.pop().to_str();
                self.push(Value::str(s.repeat(count.max(0) as usize)));
            }
            Op::StringLen => {
                let s = self.pop();
                self.push(Value::Int(s.len() as i64));
            }

            // ── Comparison (type-specialized: Int×Int avoids to_float) ──
            Op::NumEq => {
                if let Some(e) = self.cmp_int_fast(NumOp::Eq, |x, y| x == y, |a, b| a == b) {
                    return ExecFlow::Ret(VMResult::Error(e));
                }
            }
            Op::NumNe => {
                if let Some(e) = self.cmp_int_fast(NumOp::Ne, |x, y| x != y, |a, b| a != b) {
                    return ExecFlow::Ret(VMResult::Error(e));
                }
            }
            Op::NumLt => {
                if let Some(e) = self.cmp_int_fast(NumOp::Lt, |x, y| x < y, |a, b| a < b) {
                    return ExecFlow::Ret(VMResult::Error(e));
                }
            }
            Op::NumGt => {
                if let Some(e) = self.cmp_int_fast(NumOp::Gt, |x, y| x > y, |a, b| a > b) {
                    return ExecFlow::Ret(VMResult::Error(e));
                }
            }
            Op::NumLe => {
                if let Some(e) = self.cmp_int_fast(NumOp::Le, |x, y| x <= y, |a, b| a <= b) {
                    return ExecFlow::Ret(VMResult::Error(e));
                }
            }
            Op::NumGe => {
                if let Some(e) = self.cmp_int_fast(NumOp::Ge, |x, y| x >= y, |a, b| a >= b) {
                    return ExecFlow::Ret(VMResult::Error(e));
                }
            }
            Op::Spaceship => {
                let len = self.stack.len();
                if len >= 2 {
                    let b = &self.stack[len - 1];
                    let a = &self.stack[len - 2];
                    let result = match (a, b) {
                        (Value::Int(x), Value::Int(y)) => x.cmp(y) as i64,
                        _ => {
                            let af = a.to_float();
                            let bf = b.to_float();
                            if af < bf {
                                -1
                            } else if af > bf {
                                1
                            } else {
                                0
                            }
                        }
                    };
                    self.stack.truncate(len - 2);
                    self.stack.push(Value::Int(result));
                }
            }

            // ── Comparison (string — borrow via Cow to avoid allocation) ──
            Op::StrEq => {
                let b = self.pop();
                let a = self.pop();
                self.push(Value::Bool(a.as_str_cow() == b.as_str_cow()));
            }
            Op::StrNe => {
                let b = self.pop();
                let a = self.pop();
                self.push(Value::Bool(a.as_str_cow() != b.as_str_cow()));
            }
            Op::StrLt => {
                let b = self.pop();
                let a = self.pop();
                self.push(Value::Bool(a.as_str_cow() < b.as_str_cow()));
            }
            Op::StrGt => {
                let b = self.pop();
                let a = self.pop();
                self.push(Value::Bool(a.as_str_cow() > b.as_str_cow()));
            }
            Op::StrLe => {
                let b = self.pop();
                let a = self.pop();
                self.push(Value::Bool(a.as_str_cow() <= b.as_str_cow()));
            }
            Op::StrGe => {
                let b = self.pop();
                let a = self.pop();
                self.push(Value::Bool(a.as_str_cow() >= b.as_str_cow()));
            }
            Op::StrCmp => {
                let b = self.pop();
                let a = self.pop();
                self.push(Value::Int(match a.as_str_cow().cmp(&b.as_str_cow()) {
                    std::cmp::Ordering::Less => -1,
                    std::cmp::Ordering::Equal => 0,
                    std::cmp::Ordering::Greater => 1,
                }));
            }

            // ── Logical / Bitwise ──
            Op::LogNot => {
                let val = self.pop();
                self.push(Value::Bool(!val.is_truthy()));
            }
            Op::LogAnd => {
                let b = self.pop();
                let a = self.pop();
                self.push(Value::Bool(a.is_truthy() && b.is_truthy()));
            }
            Op::LogOr => {
                let b = self.pop();
                let a = self.pop();
                self.push(Value::Bool(a.is_truthy() || b.is_truthy()));
            }
            Op::BitAnd => {
                let b = self.pop();
                let a = self.pop();
                self.push(Value::Int(a.to_int() & b.to_int()));
            }
            Op::BitOr => {
                let b = self.pop();
                let a = self.pop();
                self.push(Value::Int(a.to_int() | b.to_int()));
            }
            Op::BitXor => {
                let b = self.pop();
                let a = self.pop();
                self.push(Value::Int(a.to_int() ^ b.to_int()));
            }
            Op::BitNot => {
                let val = self.pop();
                self.push(Value::Int(!val.to_int()));
            }
            Op::Shl => {
                let b = self.pop();
                let a = self.pop();
                self.push(Value::Int(a.to_int() << (b.to_int() as u32 & 63)));
            }
            Op::Shr => {
                let b = self.pop();
                let a = self.pop();
                self.push(Value::Int(a.to_int() >> (b.to_int() as u32 & 63)));
            }

            // ── Control flow ──
            Op::Jump(target) => {
                let target = *target;
                #[cfg(feature = "jit")]
                if self.tracing_jit && self.recorder.is_none() && target <= ip {
                    self.ip = self.lookup_trace_for_backward(target, ip + 1);
                } else {
                    self.ip = target;
                }
                #[cfg(not(feature = "jit"))]
                {
                    self.ip = target;
                }
            }
            Op::JumpIfTrue(target) => {
                let target = *target;
                if self.pop().is_truthy() {
                    #[cfg(feature = "jit")]
                    if self.tracing_jit && self.recorder.is_none() && target <= ip {
                        self.ip = self.lookup_trace_for_backward(target, ip + 1);
                    } else {
                        self.ip = target;
                    }
                    #[cfg(not(feature = "jit"))]
                    {
                        self.ip = target;
                    }
                }
            }
            Op::JumpIfFalse(target) => {
                let target = *target;
                if !self.pop().is_truthy() {
                    #[cfg(feature = "jit")]
                    if self.tracing_jit && self.recorder.is_none() && target <= ip {
                        self.ip = self.lookup_trace_for_backward(target, ip + 1);
                    } else {
                        self.ip = target;
                    }
                    #[cfg(not(feature = "jit"))]
                    {
                        self.ip = target;
                    }
                }
            }
            Op::JumpIfTrueKeep(target) => {
                let target = *target;
                if self.peek().is_truthy() {
                    #[cfg(feature = "jit")]
                    if self.tracing_jit && self.recorder.is_none() && target <= ip {
                        self.ip = self.lookup_trace_for_backward(target, ip + 1);
                    } else {
                        self.ip = target;
                    }
                    #[cfg(not(feature = "jit"))]
                    {
                        self.ip = target;
                    }
                }
            }
            Op::JumpIfFalseKeep(target) => {
                let target = *target;
                if !self.peek().is_truthy() {
                    #[cfg(feature = "jit")]
                    if self.tracing_jit && self.recorder.is_none() && target <= ip {
                        self.ip = self.lookup_trace_for_backward(target, ip + 1);
                    } else {
                        self.ip = target;
                    }
                    #[cfg(not(feature = "jit"))]
                    {
                        self.ip = target;
                    }
                }
            }

            // ── Functions ──
            Op::Call(name_idx, argc) => {
                if let Some(entry_ip) = self.chunk.find_sub(*name_idx) {
                    self.frames.push(Frame {
                        return_ip: self.ip,
                        stack_base: self.stack.len() - *argc as usize,
                        slots: Vec::new(),
                    });
                    self.ip = entry_ip;
                } else {
                    return ExecFlow::Ret(VMResult::Error(format!(
                        "undefined function: {}",
                        self.chunk
                            .names
                            .get(*name_idx as usize)
                            .map(|s| s.as_str())
                            .unwrap_or("?")
                    )));
                }
            }
            Op::Return => {
                if let Some(frame) = self.frames.pop() {
                    self.stack.truncate(frame.stack_base);
                    self.ip = frame.return_ip;
                } else {
                    self.halted = true;
                }
            }
            Op::ReturnValue => {
                let val = self.pop();
                if let Some(frame) = self.frames.pop() {
                    self.stack.truncate(frame.stack_base);
                    self.ip = frame.return_ip;
                    self.push(val);
                } else {
                    self.halted = true;
                    return ExecFlow::Ret(VMResult::Ok(val));
                }
            }

            // ── Scope ──
            Op::PushFrame => {
                self.frames.push(Frame {
                    return_ip: self.ip,
                    stack_base: self.stack.len(),
                    slots: Vec::new(),
                });
            }
            Op::PopFrame => {
                if let Some(frame) = self.frames.pop() {
                    self.stack.truncate(frame.stack_base);
                }
            }

            // ── I/O (write directly, no intermediate Vec) ──
            Op::Print(n) => {
                let n = *n;
                let start = self.stack.len().saturating_sub(n as usize);
                let mut out = String::new();
                for v in &self.stack[start..] {
                    out.push_str(&v.as_str_cow());
                }
                self.stack.truncate(start);
                self.emit_output(&out);
            }
            Op::PrintLn(n) => {
                let n = *n;
                let start = self.stack.len().saturating_sub(n as usize);
                let mut out = String::new();
                for v in &self.stack[start..] {
                    out.push_str(&v.as_str_cow());
                }
                out.push('\n');
                self.stack.truncate(start);
                self.emit_output(&out);
            }
            Op::ReadLine => {
                if let Some(src) = self.input_source.as_mut() {
                    match src() {
                        Some(l) => self.push(Value::str(l)),
                        None => self.push(Value::Undef),
                    }
                } else {
                    let mut line = String::new();
                    let _ = std::io::stdin().read_line(&mut line);
                    self.push(Value::str(line.trim_end_matches('\n')));
                }
            }

            // ── Fused superinstructions ──
            Op::PreIncSlot(slot) => {
                let val = self.get_slot(*slot).to_int() + 1;
                self.set_slot(*slot, Value::Int(val));
                self.push(Value::Int(val));
            }
            Op::PreIncSlotVoid(slot) => {
                let val = self.get_slot(*slot).to_int() + 1;
                self.set_slot(*slot, Value::Int(val));
            }
            Op::SlotLtIntJumpIfFalse(slot, limit, target) => {
                if self.get_slot(*slot).to_int() >= *limit as i64 {
                    self.ip = *target;
                }
            }
            Op::SlotIncLtIntJumpBack(slot, limit, target) => {
                let val = self.get_slot(*slot).to_int() + 1;
                self.set_slot(*slot, Value::Int(val));
                if val < *limit as i64 {
                    self.ip = *target;
                }
            }
            Op::AccumSumLoop(sum_slot, i_slot, limit) => {
                let mut sum = self.get_slot(*sum_slot).to_int();
                let mut i = self.get_slot(*i_slot).to_int();
                let lim = *limit as i64;
                while i < lim {
                    sum += i;
                    i += 1;
                }
                self.set_slot(*sum_slot, Value::Int(sum));
                self.set_slot(*i_slot, Value::Int(i));
            }
            Op::AddAssignSlotVoid(a, b) => {
                let sum = self.get_slot(*a).to_int() + self.get_slot(*b).to_int();
                self.set_slot(*a, Value::Int(sum));
            }
            Op::PreDecSlot(slot) => {
                let val = self.get_slot(*slot).to_int() - 1;
                self.set_slot(*slot, Value::Int(val));
                self.push(Value::Int(val));
            }
            Op::PostIncSlot(slot) => {
                let old = self.get_slot(*slot).to_int();
                self.set_slot(*slot, Value::Int(old + 1));
                self.push(Value::Int(old));
            }
            Op::PostDecSlot(slot) => {
                let old = self.get_slot(*slot).to_int();
                self.set_slot(*slot, Value::Int(old - 1));
                self.push(Value::Int(old));
            }

            // ── Status ──
            Op::SetStatus => {
                self.last_status = self.pop().to_int() as i32;
            }
            Op::GetStatus => {
                self.push(Value::Status(self.last_status));
            }

            // ── Extension dispatch ──
            Op::Extended(id, arg) => {
                let (id, arg) = (*id, *arg);
                if let Some(mut handler) = self.ext_handler.take() {
                    handler(self, id, arg);
                    self.ext_handler = Some(handler);
                }
            }
            Op::ExtendedWide(id, payload) => {
                let (id, payload) = (*id, *payload);
                if crate::awk_builtins::is_awk_op(id) {
                    self.dispatch_awk(id, payload);
                } else if let Some(mut handler) = self.ext_wide_handler.take() {
                    handler(self, id, payload);
                    self.ext_wide_handler = Some(handler);
                }
            }

            // ── Arrays ──
            Op::GetArray(idx) => {
                let val = self.get_var(*idx);
                self.push(val);
            }
            Op::SetArray(idx) => {
                let val = self.pop();
                self.set_var(*idx, val);
            }
            Op::DeclareArray(idx) => {
                self.set_var(*idx, Value::Array(Vec::new()));
            }
            Op::ArrayGet(arr_idx) => {
                let index = self.pop().to_int() as usize;
                let idx = *arr_idx as usize;
                let val = if idx < self.globals.len() {
                    if let Value::Array(ref arr) = self.globals[idx] {
                        arr.get(index).cloned().unwrap_or(Value::Undef)
                    } else {
                        Value::Undef
                    }
                } else {
                    Value::Undef
                };
                self.push(val);
            }
            Op::ArraySet(arr_idx) => {
                let index = self.pop().to_int() as usize;
                let val = self.pop();
                let idx = *arr_idx as usize;
                if idx >= self.globals.len() {
                    self.globals.resize(idx + 1, Value::Undef);
                }
                if let Value::Array(ref mut vec) = self.globals[idx] {
                    if index >= vec.len() {
                        vec.resize(index + 1, Value::Undef);
                    }
                    vec[index] = val;
                }
            }
            Op::ArrayPush(arr_idx) => {
                let val = self.pop();
                let idx = *arr_idx as usize;
                if idx >= self.globals.len() {
                    self.globals.resize(idx + 1, Value::Undef);
                }
                if let Value::Array(ref mut vec) = self.globals[idx] {
                    vec.push(val);
                }
            }
            Op::ArrayPop(arr_idx) => {
                let idx = *arr_idx as usize;
                let val = if idx < self.globals.len() {
                    if let Value::Array(ref mut vec) = self.globals[idx] {
                        vec.pop().unwrap_or(Value::Undef)
                    } else {
                        Value::Undef
                    }
                } else {
                    Value::Undef
                };
                self.push(val);
            }
            Op::ArrayShift(arr_idx) => {
                let idx = *arr_idx as usize;
                let val = if idx < self.globals.len() {
                    if let Value::Array(ref mut vec) = self.globals[idx] {
                        if vec.is_empty() {
                            Value::Undef
                        } else {
                            vec.remove(0)
                        }
                    } else {
                        Value::Undef
                    }
                } else {
                    Value::Undef
                };
                self.push(val);
            }
            Op::ArrayLen(arr_idx) => {
                let idx = *arr_idx as usize;
                let len = if idx < self.globals.len() {
                    if let Value::Array(ref vec) = self.globals[idx] {
                        vec.len() as i64
                    } else {
                        0
                    }
                } else {
                    0
                };
                self.push(Value::Int(len));
            }
            Op::MakeArray(n) => {
                let n = *n;
                let start = self.stack.len().saturating_sub(n as usize);
                let elements: Vec<Value> = self.stack.drain(start..).collect();
                self.push(Value::Array(elements));
            }

            // ── Hashes ──
            Op::GetHash(idx) => {
                let val = self.get_var(*idx);
                self.push(val);
            }
            Op::SetHash(idx) => {
                let val = self.pop();
                self.set_var(*idx, val);
            }
            Op::DeclareHash(idx) => {
                self.set_var(*idx, Value::Hash(std::collections::HashMap::new()));
            }
            Op::HashGet(hash_idx) => {
                let key_val = self.pop();
                let key = key_val.as_str_cow();
                let idx = *hash_idx as usize;
                let val = if idx < self.globals.len() {
                    if let Value::Hash(ref map) = self.globals[idx] {
                        map.get(key.as_ref()).cloned().unwrap_or(Value::Undef)
                    } else {
                        Value::Undef
                    }
                } else {
                    Value::Undef
                };
                self.push(val);
            }
            Op::HashSet(hash_idx) => {
                let key = self.pop().to_str();
                let val = self.pop();
                let idx = *hash_idx as usize;
                if idx >= self.globals.len() {
                    self.globals.resize(idx + 1, Value::Undef);
                }
                if let Value::Hash(ref mut map) = self.globals[idx] {
                    map.insert(key, val);
                }
            }
            Op::HashDelete(hash_idx) => {
                let key_val = self.pop();
                let key = key_val.as_str_cow();
                let idx = *hash_idx as usize;
                let val = if idx < self.globals.len() {
                    if let Value::Hash(ref mut map) = self.globals[idx] {
                        map.remove(key.as_ref()).unwrap_or(Value::Undef)
                    } else {
                        Value::Undef
                    }
                } else {
                    Value::Undef
                };
                self.push(val);
            }
            Op::HashExists(hash_idx) => {
                let key_val = self.pop();
                let key = key_val.as_str_cow();
                let idx = *hash_idx as usize;
                let val = if idx < self.globals.len() {
                    if let Value::Hash(ref map) = self.globals[idx] {
                        map.contains_key(key.as_ref())
                    } else {
                        false
                    }
                } else {
                    false
                };
                self.push(Value::Bool(val));
            }
            Op::HashKeys(hash_idx) => {
                let idx = *hash_idx as usize;
                let arr = if idx < self.globals.len() {
                    if let Value::Hash(ref map) = self.globals[idx] {
                        let mut keys = Vec::with_capacity(map.len());
                        keys.extend(map.keys().map(|k| Value::str(k.as_str())));
                        keys
                    } else {
                        Vec::new()
                    }
                } else {
                    Vec::new()
                };
                self.push(Value::Array(arr));
            }
            Op::HashValues(hash_idx) => {
                let idx = *hash_idx as usize;
                let arr = if idx < self.globals.len() {
                    if let Value::Hash(ref map) = self.globals[idx] {
                        let mut vals = Vec::with_capacity(map.len());
                        vals.extend(map.values().cloned());
                        vals
                    } else {
                        Vec::new()
                    }
                } else {
                    Vec::new()
                };
                self.push(Value::Array(arr));
            }
            Op::MakeHash(n) => {
                let n = *n;
                let start = self.stack.len().saturating_sub(n as usize);
                let pairs: Vec<Value> = self.stack.drain(start..).collect();
                let mut map = std::collections::HashMap::with_capacity(pairs.len() / 2);
                let mut iter = pairs.into_iter();
                while let Some(key) = iter.next() {
                    if let Some(val) = iter.next() {
                        map.insert(key.to_str(), val);
                    }
                }
                self.push(Value::Hash(map));
            }

            // ── Range ──
            Op::Range => {
                let to = self.pop().to_int();
                let from = self.pop().to_int();
                let cap = (to - from + 1).max(0) as usize;
                let mut arr = Vec::with_capacity(cap);
                arr.extend((from..=to).map(Value::Int));
                self.push(Value::Array(arr));
            }
            Op::RangeStep => {
                let step = self.pop().to_int();
                let to = self.pop().to_int();
                let from = self.pop().to_int();
                let cap = if step > 0 {
                    ((to - from) / step + 1).max(0) as usize
                } else if step < 0 {
                    ((from - to) / (-step) + 1).max(0) as usize
                } else {
                    0
                };
                let mut arr = Vec::with_capacity(cap);
                if step > 0 {
                    let mut i = from;
                    while i <= to {
                        arr.push(Value::Int(i));
                        i += step;
                    }
                } else if step < 0 {
                    let mut i = from;
                    while i >= to {
                        arr.push(Value::Int(i));
                        i += step;
                    }
                }
                self.push(Value::Array(arr));
            }

            // ── Shell ops ──
            Op::TestFile(test_type) => {
                let test_type = *test_type;
                let path = self.pop().to_str();
                let result = match test_type {
                    crate::op::file_test::EXISTS => std::path::Path::new(&path).exists(),
                    crate::op::file_test::IS_FILE => std::path::Path::new(&path).is_file(),
                    crate::op::file_test::IS_DIR => std::path::Path::new(&path).is_dir(),
                    crate::op::file_test::IS_SYMLINK => std::path::Path::new(&path).is_symlink(),
                    crate::op::file_test::IS_READABLE | crate::op::file_test::IS_WRITABLE => {
                        std::path::Path::new(&path).exists()
                    }
                    crate::op::file_test::IS_EXECUTABLE => {
                        #[cfg(unix)]
                        {
                            use std::os::unix::fs::PermissionsExt;
                            std::fs::metadata(&path)
                                .map(|m| m.permissions().mode() & 0o111 != 0)
                                .unwrap_or(false)
                        }
                        #[cfg(not(unix))]
                        {
                            std::path::Path::new(&path).exists()
                        }
                    }
                    crate::op::file_test::IS_NONEMPTY => std::fs::metadata(&path)
                        .map(|m| m.len() > 0)
                        .unwrap_or(false),
                    crate::op::file_test::IS_SOCKET => {
                        #[cfg(unix)]
                        {
                            use std::os::unix::fs::FileTypeExt;
                            std::fs::symlink_metadata(&path)
                                .map(|m| m.file_type().is_socket())
                                .unwrap_or(false)
                        }
                        #[cfg(not(unix))]
                        {
                            false
                        }
                    }
                    crate::op::file_test::IS_FIFO => {
                        #[cfg(unix)]
                        {
                            use std::os::unix::fs::FileTypeExt;
                            std::fs::symlink_metadata(&path)
                                .map(|m| m.file_type().is_fifo())
                                .unwrap_or(false)
                        }
                        #[cfg(not(unix))]
                        {
                            false
                        }
                    }
                    crate::op::file_test::IS_BLOCK_DEV => {
                        #[cfg(unix)]
                        {
                            use std::os::unix::fs::FileTypeExt;
                            std::fs::symlink_metadata(&path)
                                .map(|m| m.file_type().is_block_device())
                                .unwrap_or(false)
                        }
                        #[cfg(not(unix))]
                        {
                            false
                        }
                    }
                    crate::op::file_test::IS_CHAR_DEV => {
                        #[cfg(unix)]
                        {
                            use std::os::unix::fs::FileTypeExt;
                            std::fs::symlink_metadata(&path)
                                .map(|m| m.file_type().is_char_device())
                                .unwrap_or(false)
                        }
                        #[cfg(not(unix))]
                        {
                            false
                        }
                    }
                    _ => false,
                };
                self.push(Value::Bool(result));
            }

            Op::Exec(argc) => {
                let argc = *argc;
                let start = self.stack.len().saturating_sub(argc as usize);
                // Flatten Value::Array entries into argv. Shell array splice
                // (`${arr[@]}`) pushes a single Array value at compile-time
                // even though it expands to N argv slots at runtime. Without
                // this flat_map, `cmd ${arr[@]}` would pass the whole array
                // as one space-joined arg instead of N separate args.
                let args: Vec<String> = self
                    .stack
                    .drain(start..)
                    .flat_map(|v| match v {
                        Value::Array(items) => {
                            items.into_iter().map(|i| i.to_str()).collect::<Vec<_>>()
                        }
                        other => vec![other.to_str()],
                    })
                    .collect();
                if let Some(cmd) = args.first() {
                    // Check if it's a shell function
                    let name_idx = self.chunk.names.iter().position(|n| n == cmd);
                    if let Some(name_idx) = name_idx {
                        if let Some(entry_ip) = self.chunk.find_sub(name_idx as u16) {
                            // Push arguments for the function (skip command name)
                            for arg in &args[1..] {
                                self.push(Value::str(arg));
                            }
                            // Push frame and call
                            self.frames.push(Frame {
                                return_ip: self.ip,
                                stack_base: self.stack.len() - (args.len() - 1),
                                slots: Vec::with_capacity(8),
                            });
                            self.ip = entry_ip;
                            return ExecFlow::Cont;
                        }
                    }

                    match cmd.as_str() {
                        "true" => self.push(Value::Status(0)),
                        "false" => self.push(Value::Status(1)),
                        "echo" => {
                            println!("{}", args[1..].join(" "));
                            self.push(Value::Status(0));
                        }
                        "test" | "[" => {
                            self.push(Value::Status(0));
                        }
                        _ => {
                            // Route through the host's `exec` so frontends
                            // (zshrs) can apply intercepts/AOP advice/job
                            // tracking on dynamic command names like
                            // `cmd=ls; $cmd`. The default ShellHost::exec
                            // implementation falls back to Command::new,
                            // so behavior is identical when no host is
                            // wired. Without host, we keep the inline
                            // Command::new path so the VM still runs in
                            // host-less embeddings (tests, REPL stubs).
                            let status = if let Some(h) = self.host.as_mut() {
                                h.exec(args.clone())
                            } else {
                                #[cfg(not(target_arch = "wasm32"))]
                                {
                                    use std::process::{Command, Stdio};
                                    Command::new(cmd)
                                        .args(&args[1..])
                                        .stdout(Stdio::inherit())
                                        .stderr(Stdio::inherit())
                                        .status()
                                        .map(|s| s.code().unwrap_or(1))
                                        .unwrap_or(127)
                                }
                                // No process model in a browser worker; a
                                // host-less wasm embedding reports 127.
                                #[cfg(target_arch = "wasm32")]
                                {
                                    let _ = cmd;
                                    127
                                }
                            };
                            self.push(Value::Status(status));
                        }
                    }
                } else {
                    self.push(Value::Status(0));
                }
            }
            Op::ExecBg(argc) => {
                let argc = *argc;
                let start = self.stack.len().saturating_sub(argc as usize);
                // Same Array-flattening as Op::Exec — see comment there.
                let args: Vec<String> = self
                    .stack
                    .drain(start..)
                    .flat_map(|v| match v {
                        Value::Array(items) => {
                            items.into_iter().map(|i| i.to_str()).collect::<Vec<_>>()
                        }
                        other => vec![other.to_str()],
                    })
                    .collect();
                if let Some(cmd) = args.first() {
                    // Route bg exec through the host. Frontends override
                    // to register the spawned pid in their job table; the
                    // default impl spawns and detaches. We DON'T wait on
                    // the bg child here — that's the host's responsibility
                    // (zshrs uses BUILTIN_RUN_BG which forks before
                    // emitting Op::ExecBg, so this path is rare for
                    // shell-level bg). Without host, fall back to inline
                    // Command::new spawn for host-less embeddings.
                    if let Some(h) = self.host.as_mut() {
                        let _ = h.exec_bg(args.clone());
                    } else {
                        #[cfg(not(target_arch = "wasm32"))]
                        {
                            use std::process::{Command, Stdio};
                            let _ = Command::new(cmd)
                                .args(&args[1..])
                                .stdout(Stdio::null())
                                .stderr(Stdio::null())
                                .spawn();
                        }
                        // No process model in a browser worker; drop the spawn.
                        #[cfg(target_arch = "wasm32")]
                        {
                            let _ = cmd;
                        }
                    }
                }
                self.push(Value::Status(0));
            }

            // ── Shell ops ── (route through host when set, fall back to stubs)
            Op::PipelineBegin(n) => {
                let n = *n;
                if let Some(h) = self.host.as_mut() {
                    h.pipeline_begin(n);
                }
            }
            Op::PipelineStage => {
                if let Some(h) = self.host.as_mut() {
                    h.pipeline_stage();
                }
            }
            Op::PipelineEnd => {
                let status = if let Some(h) = self.host.as_mut() {
                    h.pipeline_end()
                } else {
                    self.last_status
                };
                self.last_status = status;
                self.push(Value::Status(status));
            }
            Op::SubshellBegin => {
                if let Some(h) = self.host.as_mut() {
                    h.subshell_begin();
                }
            }
            Op::SubshellEnd => {
                if let Some(h) = self.host.as_mut() {
                    if let Some(status) = h.subshell_end() {
                        self.last_status = status;
                    }
                }
            }
            Op::Redirect(fd, op) => {
                let fd = *fd;
                let op = *op;
                let target = self.pop().to_str();
                if let Some(h) = self.host.as_mut() {
                    h.redirect(fd, op, &target);
                }
            }
            Op::HereDoc(idx) => {
                let content = self
                    .chunk
                    .constants
                    .get(*idx as usize)
                    .map(|v| v.to_str())
                    .unwrap_or_default();
                if let Some(h) = self.host.as_mut() {
                    h.heredoc(&content);
                }
            }
            Op::HereString => {
                let s = self.pop().to_str();
                if let Some(h) = self.host.as_mut() {
                    h.herestring(&s);
                }
            }
            Op::CmdSubst(idx) => {
                let result = match self.chunk.sub_chunks.get(*idx as usize) {
                    Some(sub) => {
                        // Split borrow: self.host and self.chunk are disjoint fields
                        let sub_ref: *const Chunk = sub;
                        // SAFETY: sub_chunks is not mutated during op dispatch
                        let sub_ref = unsafe { &*sub_ref };
                        if let Some(h) = self.host.as_mut() {
                            h.cmd_subst(sub_ref)
                        } else {
                            String::new()
                        }
                    }
                    None => String::new(),
                };
                self.push(Value::str(result));
            }
            Op::ProcessSubIn(idx) => {
                let result = match self.chunk.sub_chunks.get(*idx as usize) {
                    Some(sub) => {
                        let sub_ref: *const Chunk = sub;
                        let sub_ref = unsafe { &*sub_ref };
                        if let Some(h) = self.host.as_mut() {
                            h.process_sub_in(sub_ref)
                        } else {
                            String::new()
                        }
                    }
                    None => String::new(),
                };
                self.push(Value::str(result));
            }
            Op::ProcessSubOut(idx) => {
                let result = match self.chunk.sub_chunks.get(*idx as usize) {
                    Some(sub) => {
                        let sub_ref: *const Chunk = sub;
                        let sub_ref = unsafe { &*sub_ref };
                        if let Some(h) = self.host.as_mut() {
                            h.process_sub_out(sub_ref)
                        } else {
                            String::new()
                        }
                    }
                    None => String::new(),
                };
                self.push(Value::str(result));
            }
            Op::Glob | Op::GlobRecursive => {
                let recursive = matches!(&ops[ip], Op::GlobRecursive);
                let pat_val = self.pop();
                let pattern = pat_val.to_str();
                let matches: Vec<String> = if let Some(h) = self.host.as_mut() {
                    h.glob(&pattern, recursive)
                } else {
                    glob::glob(&pattern)
                        .into_iter()
                        .flat_map(|paths| paths.filter_map(|p| p.ok()))
                        .map(|p| p.to_string_lossy().into_owned())
                        .collect()
                };
                let arr: Vec<Value> = matches.into_iter().map(Value::str).collect();
                self.push(Value::Array(arr));
            }
            Op::TrapSet(idx) => {
                // stack: [signal_name]
                let sig = self.pop().to_str();
                if let Some(sub) = self.chunk.sub_chunks.get(*idx as usize) {
                    let sub_ref: *const Chunk = sub;
                    let sub_ref = unsafe { &*sub_ref };
                    if let Some(h) = self.host.as_mut() {
                        h.trap_set(&sig, sub_ref);
                    }
                }
            }
            Op::TrapCheck => {
                if let Some(h) = self.host.as_mut() {
                    h.trap_check();
                }
            }
            Op::TildeExpand => {
                let s = self.pop().to_str();
                let result = if let Some(h) = self.host.as_mut() {
                    h.tilde_expand(&s)
                } else {
                    s
                };
                self.push(Value::str(result));
            }
            Op::BraceExpand => {
                let s = self.pop().to_str();
                let result = if let Some(h) = self.host.as_mut() {
                    h.brace_expand(&s)
                } else {
                    vec![s]
                };
                let arr: Vec<Value> = result.into_iter().map(Value::str).collect();
                self.push(Value::Array(arr));
            }
            Op::WordSplit => {
                let s = self.pop().to_str();
                let result = if let Some(h) = self.host.as_mut() {
                    h.word_split(&s)
                } else {
                    s.split_whitespace().map(|w| w.to_string()).collect()
                };
                let arr: Vec<Value> = result.into_iter().map(Value::str).collect();
                self.push(Value::Array(arr));
            }
            Op::ExpandParam(modifier) => {
                // Stack layout per modifier:
                //   DEFAULT/ASSIGN/ERROR/ALTERNATE/STRIP*/RSTRIP*: [name, arg]
                //   SUBST_FIRST/SUBST_ALL: [name, pat, rep]
                //   SLICE: [name, off, len]
                //   LENGTH/UPPER/LOWER/UPPER_FIRST/LOWER_FIRST/INDIRECT/KEYS: [name]
                let m = *modifier;
                let argc = match m {
                    crate::op::param_mod::DEFAULT
                    | crate::op::param_mod::ASSIGN
                    | crate::op::param_mod::ERROR
                    | crate::op::param_mod::ALTERNATE
                    | crate::op::param_mod::STRIP_SHORT
                    | crate::op::param_mod::STRIP_LONG
                    | crate::op::param_mod::RSTRIP_SHORT
                    | crate::op::param_mod::RSTRIP_LONG => 1,
                    crate::op::param_mod::SUBST_FIRST
                    | crate::op::param_mod::SUBST_ALL
                    | crate::op::param_mod::SLICE => 2,
                    _ => 0,
                };
                let mut args: Vec<Value> = Vec::with_capacity(argc);
                for _ in 0..argc {
                    args.push(self.pop());
                }
                args.reverse();
                let name = self.pop().to_str();
                let result = if let Some(h) = self.host.as_mut() {
                    h.expand_param(&name, m, &args)
                } else {
                    Value::str("")
                };
                self.push(result);
            }
            Op::StrMatch => {
                let pat = self.pop().to_str();
                let s = self.pop().to_str();
                let result = if let Some(h) = self.host.as_mut() {
                    h.str_match(&s, &pat)
                } else {
                    s == pat
                };
                self.push(Value::Bool(result));
            }
            Op::RegexMatch => {
                let re = self.pop().to_str();
                let s = self.pop().to_str();
                let result = if let Some(h) = self.host.as_mut() {
                    h.regex_match(&s, &re)
                } else {
                    false
                };
                self.push(Value::Bool(result));
            }
            Op::WithRedirectsBegin(n) => {
                let n = *n;
                if let Some(h) = self.host.as_mut() {
                    h.with_redirects_begin(n);
                }
            }
            Op::WithRedirectsEnd => {
                if let Some(h) = self.host.as_mut() {
                    h.with_redirects_end();
                }
            }
            Op::CallFunction(name_idx, argc) => {
                let name = self
                    .chunk
                    .names
                    .get(*name_idx as usize)
                    .cloned()
                    .unwrap_or_default();
                let argc = *argc as usize;
                let start = self.stack.len().saturating_sub(argc);
                // Flatten arrays (see Op::Exec for rationale).
                let args: Vec<String> = self
                    .stack
                    .drain(start..)
                    .flat_map(|v| match v {
                        Value::Array(items) => {
                            items.into_iter().map(|i| i.to_str()).collect::<Vec<_>>()
                        }
                        other => vec![other.to_str()],
                    })
                    .collect();
                let status = if self.host.is_some() {
                    // alias/function/host-table builtin resolution.
                    let cf = self
                        .host
                        .as_mut()
                        .unwrap()
                        .call_function(&name, args.clone());
                    match cf {
                        Some(s) => s,
                        None => {
                            // Not a user function or a host-table builtin. Try a
                            // VM-registered builtin by NAME — the run-time analog
                            // of the `CallBuiltin` opcode — BEFORE external exec,
                            // matching the shell's function -> builtin -> external
                            // resolution order. Without this, a builtin only ever
                            // reached through its compile-time `CallBuiltin`
                            // (literal name) is unreachable when the command name
                            // is resolved at run time (`$var`, `eval`, indirection).
                            if let Some(v) = self.run_builtin_by_name(&name, &args) {
                                v.to_int() as i32
                            } else {
                                let mut full = Vec::with_capacity(args.len() + 1);
                                full.push(name.clone());
                                full.extend(args);
                                self.host.as_mut().unwrap().exec(full)
                            }
                        }
                    }
                } else {
                    // No host — fall back to in-chunk function lookup, then external exec
                    let nidx = *name_idx;
                    if let Some(entry_ip) = self.chunk.find_sub(nidx) {
                        for arg in &args {
                            self.push(Value::str(arg));
                        }
                        self.frames.push(Frame {
                            return_ip: self.ip,
                            stack_base: self.stack.len() - args.len(),
                            slots: Vec::with_capacity(8),
                        });
                        self.ip = entry_ip;
                        return ExecFlow::Cont;
                    }
                    let mut full = Vec::with_capacity(args.len() + 1);
                    full.push(name);
                    full.extend(args);
                    #[cfg(not(target_arch = "wasm32"))]
                    {
                        use std::process::Command;
                        Command::new(&full[0])
                            .args(&full[1..])
                            .status()
                            .map(|s| s.code().unwrap_or(1))
                            .unwrap_or(127)
                    }
                    // No process model in a browser worker; a host-less wasm
                    // embedding reports 127 (command not found).
                    #[cfg(target_arch = "wasm32")]
                    {
                        let _ = &full;
                        127
                    }
                };
                self.last_status = status;
                self.push(Value::Status(status));
            }

            // ── Remaining fused ops ──
            Op::ConcatConstLoop(const_idx, s_slot, i_slot, limit) => {
                let c_str = self
                    .chunk
                    .constants
                    .get(*const_idx as usize)
                    .map(|v| v.as_str_cow())
                    .unwrap_or(std::borrow::Cow::Borrowed(""));
                let mut s = self.get_slot(*s_slot).to_str();
                let mut i = self.get_slot(*i_slot).to_int();
                let lim = *limit as i64;
                let iters = (lim - i).max(0) as usize;
                s.reserve(c_str.len() * iters);
                while i < lim {
                    s.push_str(&c_str);
                    i += 1;
                }
                self.set_slot(*s_slot, Value::str(s));
                self.set_slot(*i_slot, Value::Int(i));
            }
            Op::PushIntRangeLoop(arr_idx, i_slot, limit) => {
                let mut i = self.get_slot(*i_slot).to_int();
                let lim = *limit as i64;
                let arr = self.get_var(*arr_idx);
                let mut vec = if let Value::Array(v) = arr {
                    v
                } else {
                    Vec::new()
                };
                vec.reserve((lim - i).max(0) as usize);
                while i < lim {
                    vec.push(Value::Int(i));
                    i += 1;
                }
                self.set_var(*arr_idx, Value::Array(vec));
                self.set_slot(*i_slot, Value::Int(i));
            }

            // ── Higher-order (stubs) ──
            Op::MapBlock(_)
            | Op::GrepBlock(_)
            | Op::SortBlock(_)
            | Op::SortDefault
            | Op::ForEachBlock(_) => {}

            // ── Builtins (inline cache) ──
            Op::CallBuiltin(id, argc) => {
                let (id, argc) = (*id, *argc);
                if let Some(Some(handler)) = self.builtin_table.get(id as usize) {
                    let result = handler(self, argc);
                    self.push(result);
                }
            }

            // ── AWK ops (first-class; dispatched to the AwkHost, same path
            //    as the reserved ExtendedWide AWK range) ──
            Op::AwkFieldGet => self.dispatch_awk(ab::AWK_FIELD_GET, 0),
            Op::AwkFieldSet => self.dispatch_awk(ab::AWK_FIELD_SET, 0),
            Op::AwkNf => self.dispatch_awk(ab::AWK_NF, 0),
            Op::AwkSetRecord => self.dispatch_awk(ab::AWK_SET_RECORD, 0),
            Op::AwkSpecialGet(n) => self.dispatch_awk(ab::AWK_SPECIAL_GET, *n as usize),
            Op::AwkSpecialSet(n) => self.dispatch_awk(ab::AWK_SPECIAL_SET, *n as usize),
            Op::AwkPrint(argc) => self.dispatch_awk(ab::AWK_PRINT, *argc as usize),
            Op::AwkPrintf(argc) => self.dispatch_awk(ab::AWK_PRINTF, *argc as usize),
            Op::AwkSprintf(argc) => self.dispatch_awk(ab::AWK_SPRINTF, *argc as usize),
            Op::AwkGetline(src) => self.dispatch_awk(ab::AWK_GETLINE, *src as usize),
            Op::AwkLength(argc) => self.dispatch_awk(ab::AWK_LENGTH, *argc as usize),
            Op::AwkSubstr(argc) => self.dispatch_awk(ab::AWK_SUBSTR, *argc as usize),
            Op::AwkIndex => self.dispatch_awk(ab::AWK_INDEX, 0),
            Op::AwkSplit(argc) => self.dispatch_awk(ab::AWK_SPLIT, *argc as usize),
            Op::AwkSub(argc) => self.dispatch_awk(ab::AWK_SUB, *argc as usize),
            Op::AwkGsub(argc) => self.dispatch_awk(ab::AWK_GSUB, *argc as usize),
            Op::AwkMatch => self.dispatch_awk(ab::AWK_MATCH, 0),
            Op::AwkToLower => self.dispatch_awk(ab::AWK_TOLOWER, 0),
            Op::AwkToUpper => self.dispatch_awk(ab::AWK_TOUPPER, 0),
            Op::AwkInt => self.dispatch_awk(ab::AWK_INT, 0),
            Op::AwkSqrt => self.dispatch_awk(ab::AWK_SQRT, 0),
            Op::AwkSin => self.dispatch_awk(ab::AWK_SIN, 0),
            Op::AwkCos => self.dispatch_awk(ab::AWK_COS, 0),
            Op::AwkExp => self.dispatch_awk(ab::AWK_EXP, 0),
            Op::AwkLog => self.dispatch_awk(ab::AWK_LOG, 0),
            Op::AwkAtan2 => self.dispatch_awk(ab::AWK_ATAN2, 0),
            // awk `a / b` and `a % b`: pop b then a (same order as Op::Div),
            // raise the POSIX fatal error on a zero divisor instead of
            // yielding Undef. Distinct from the shared shell-arithmetic ops.
            Op::AwkDiv => {
                let b = self.pop();
                let a = self.pop();
                let divisor = b.to_float();
                if divisor == 0.0 {
                    return ExecFlow::Ret(VMResult::Error(
                        "division by zero attempted".to_string(),
                    ));
                }
                self.push(Value::Float(a.to_float() / divisor));
            }
            Op::AwkMod => {
                let b = self.pop();
                let a = self.pop();
                let divisor = b.to_float();
                if divisor == 0.0 {
                    return ExecFlow::Ret(VMResult::Error(
                        "division by zero attempted in `%'".to_string(),
                    ));
                }
                self.push(Value::Float(a.to_float() % divisor));
            }
            // Block-JIT-eligible div/mod (see `Op::AwkDivJit`). The
            // interpreter behavior is byte-identical to AwkDiv/AwkMod; the
            // distinct opcode only changes JIT eligibility.
            Op::AwkDivJit => {
                let b = self.pop();
                let a = self.pop();
                let divisor = b.to_float();
                if divisor == 0.0 {
                    return ExecFlow::Ret(VMResult::Error(
                        "division by zero attempted".to_string(),
                    ));
                }
                self.push(Value::Float(a.to_float() / divisor));
            }
            Op::AwkModJit => {
                let b = self.pop();
                let a = self.pop();
                let divisor = b.to_float();
                if divisor == 0.0 {
                    return ExecFlow::Ret(VMResult::Error(
                        "division by zero attempted in `%'".to_string(),
                    ));
                }
                self.push(Value::Float(a.to_float() % divisor));
            }
            // awk sqrt(x) — interpreter path. On negative input, emit the
            // generic "awk: warning: sqrt: received negative argument <x>"
            // warning to stderr (the JIT-trapped path uses the same generic
            // format via the warn libcall, so the two tiers agree).
            Op::AwkSqrtJit => {
                let a = self.pop().to_float();
                if a < 0.0 {
                    eprintln!("awk: warning: sqrt: received negative argument {a}");
                    self.push(Value::Float(f64::NAN));
                } else {
                    self.push(Value::Float(a.sqrt()));
                }
            }
            // awk log(x) — interpreter path. Negative emits the generic warn
            // and pushes NaN; zero returns -inf naturally (no lint warn in
            // this tier — host frontends that want LINT=1 behavior must use
            // the existing `Op::AwkLog` host-dispatched variant).
            Op::AwkLogJit => {
                let a = self.pop().to_float();
                if a < 0.0 {
                    eprintln!("awk: warning: log: received negative argument {a}");
                    self.push(Value::Float(f64::NAN));
                } else {
                    self.push(Value::Float(a.ln()));
                }
            }
            // awk lshift(a, n) — fatal on negative operands. Stack [a, n]:
            // pop n then a (matches the awk evaluation order pushed by
            // frontends).
            Op::AwkLshiftJit => {
                let n = self.pop().to_float();
                let a = self.pop().to_float();
                if a < 0.0 || n < 0.0 {
                    return ExecFlow::Ret(VMResult::Error(
                        "lshift: negative values are not allowed".to_string(),
                    ));
                }
                let shifted = (a as i64).wrapping_shl((n as u32) & 0x3f);
                self.push(Value::Float(shifted as f64));
            }
            // awk rshift(a, n) — same guard as lshift but logical right.
            Op::AwkRshiftJit => {
                let n = self.pop().to_float();
                let a = self.pop().to_float();
                if a < 0.0 || n < 0.0 {
                    return ExecFlow::Ret(VMResult::Error(
                        "rshift: negative values are not allowed".to_string(),
                    ));
                }
                let shifted = ((a as i64) as u64).wrapping_shr((n as u32) & 0x3f);
                self.push(Value::Float(shifted as f64));
            }
            // awk compl(a) — fatal on negative. `!a` in u64 space then back
            // to f64 (the high bits saturate the f64 mantissa, matching
            // awkrs's `num_to_u64` semantics).
            Op::AwkComplJit => {
                let a = self.pop().to_float();
                if a < 0.0 {
                    return ExecFlow::Ret(VMResult::Error(
                        "compl: negative value is not allowed".to_string(),
                    ));
                }
                let v = !(a as i64);
                self.push(Value::Float(v as f64));
            }
            // awk `$N` numeric read — interpreter path. Calls the same
            // host-installed hook as the JIT-compiled variant so behavior
            // matches across tiers. Returns 0.0 when no hook is set, which
            // matches awk's "missing field" coercion.
            Op::AwkGetFieldNum(field_idx) => {
                let v = crate::jit::fusevm_jit_awk_get_field_num(*field_idx as i64);
                self.push(Value::Float(v));
            }
            Op::PowFloat => {
                let b = self.pop();
                let a = self.pop();
                self.push(Value::Float(a.to_float().powf(b.to_float())));
            }
            Op::SqrtFloat => {
                let a = self.pop();
                self.push(Value::Float(a.to_float().sqrt()));
            }
            Op::SinFloat => {
                let a = self.pop();
                self.push(Value::Float(a.to_float().sin()));
            }
            Op::CosFloat => {
                let a = self.pop();
                self.push(Value::Float(a.to_float().cos()));
            }
            Op::ExpFloat => {
                let a = self.pop();
                self.push(Value::Float(a.to_float().exp()));
            }
            Op::Atan2Float => {
                let x = self.pop();
                let y = self.pop();
                self.push(Value::Float(y.to_float().atan2(x.to_float())));
            }
            Op::LogFloat => {
                let a = self.pop();
                self.push(Value::Float(a.to_float().ln()));
            }
            Op::AbsFloat => {
                let a = self.pop();
                self.push(Value::Float(a.to_float().abs()));
            }
            Op::TruncInt => {
                let a = self.pop();
                self.push(Value::Int(a.to_int()));
            }
            Op::CeilFloat => {
                let a = self.pop();
                self.push(Value::Float(a.to_float().ceil()));
            }
            Op::FloorFloat => {
                let a = self.pop();
                self.push(Value::Float(a.to_float().floor()));
            }
            Op::TruncFloat => {
                let a = self.pop();
                self.push(Value::Float(a.to_float().trunc()));
            }
            Op::RoundFloat => {
                let a = self.pop();
                self.push(Value::Float(a.to_float().round_ties_even()));
            }
            Op::TanFloat => {
                let a = self.pop();
                self.push(Value::Float(a.to_float().tan()));
            }
            Op::AsinFloat => {
                let a = self.pop();
                self.push(Value::Float(a.to_float().asin()));
            }
            Op::AcosFloat => {
                let a = self.pop();
                self.push(Value::Float(a.to_float().acos()));
            }
            Op::AtanFloat => {
                let a = self.pop();
                self.push(Value::Float(a.to_float().atan()));
            }
            Op::SinhFloat => {
                let a = self.pop();
                self.push(Value::Float(a.to_float().sinh()));
            }
            Op::CoshFloat => {
                let a = self.pop();
                self.push(Value::Float(a.to_float().cosh()));
            }
            Op::TanhFloat => {
                let a = self.pop();
                self.push(Value::Float(a.to_float().tanh()));
            }
            Op::Log2Float => {
                let a = self.pop();
                self.push(Value::Float(a.to_float().log2()));
            }
            Op::Log10Float => {
                let a = self.pop();
                self.push(Value::Float(a.to_float().log10()));
            }
            Op::AbsInt => {
                let a = self.pop();
                self.push(Value::Int(a.to_int().wrapping_abs()));
            }
            Op::GcdInt => {
                let b = self.pop().to_int().unsigned_abs();
                let a = self.pop().to_int().unsigned_abs();
                let mut x = a;
                let mut y = b;
                while y != 0 {
                    let t = x % y;
                    x = y;
                    y = t;
                }
                self.push(Value::Int(x as i64));
            }
            Op::LcmInt => {
                let b = self.pop().to_int().unsigned_abs();
                let a = self.pop().to_int().unsigned_abs();
                if a == 0 || b == 0 {
                    self.push(Value::Int(0));
                } else {
                    let mut x = a;
                    let mut y = b;
                    while y != 0 {
                        let t = x % y;
                        x = y;
                        y = t;
                    }
                    let prod = (a / x).saturating_mul(b);
                    self.push(Value::Int(prod.min(i64::MAX as u64) as i64));
                }
            }
            Op::TimeInt => {
                self.push(Value::Int(crate::sysclock::unix_secs()));
            }
            Op::AwkArrayGet(n) => self.dispatch_awk(ab::AWK_ARRAY_GET, *n as usize),
            Op::AwkArraySet(n) => self.dispatch_awk(ab::AWK_ARRAY_SET, *n as usize),
            Op::AwkArrayExists(n) => self.dispatch_awk(ab::AWK_ARRAY_EXISTS, *n as usize),
            Op::AwkArrayDelete(n) => self.dispatch_awk(ab::AWK_ARRAY_DELETE, *n as usize),
            Op::AwkArrayClear(n) => self.dispatch_awk(ab::AWK_ARRAY_CLEAR, *n as usize),
            Op::AwkArrayLen(n) => self.dispatch_awk(ab::AWK_ARRAY_LEN, *n as usize),
            Op::AwkAnd(argc) => self.dispatch_awk(ab::AWK_AND, *argc as usize),
            Op::AwkOr(argc) => self.dispatch_awk(ab::AWK_OR, *argc as usize),
            Op::AwkXor(argc) => self.dispatch_awk(ab::AWK_XOR, *argc as usize),
            Op::AwkCompl => self.dispatch_awk(ab::AWK_COMPL, 0),
            Op::AwkLshift => self.dispatch_awk(ab::AWK_LSHIFT, 0),
            Op::AwkRshift => self.dispatch_awk(ab::AWK_RSHIFT, 0),
            Op::AwkStrtonum => self.dispatch_awk(ab::AWK_STRTONUM, 0),
            Op::AwkSystime => self.dispatch_awk(ab::AWK_SYSTIME, 0),
            Op::AwkRand => self.dispatch_awk(ab::AWK_RAND, 0),
            Op::AwkSrand(argc) => self.dispatch_awk(ab::AWK_SRAND, *argc as usize),
            Op::AwkStrftime(argc) => self.dispatch_awk(ab::AWK_STRFTIME, *argc as usize),
            Op::AwkMktime(argc) => self.dispatch_awk(ab::AWK_MKTIME, *argc as usize),
            Op::AwkOrd => self.dispatch_awk(ab::AWK_ORD, 1),
            Op::AwkChr => self.dispatch_awk(ab::AWK_CHR, 1),
            Op::AwkMkbool => self.dispatch_awk(ab::AWK_MKBOOL, 1),
            Op::AwkIntdiv => self.dispatch_awk(ab::AWK_INTDIV, 2),
            Op::AwkIntdiv0 => self.dispatch_awk(ab::AWK_INTDIV0, 2),
            Op::AwkGensub(argc) => self.dispatch_awk(ab::AWK_GENSUB, *argc as usize),
            Op::AwkSignal(code) => {
                // Raise the AWK control-flow signal and halt this chunk; the
                // frontend driver reads `self.awk_signal()` after `run()`.
                self.awk_signal = Some(*code);
                self.halted = true;
            }

            // ── cooperative concurrency: raise a scheduling request + halt ──
            // The op has already advanced `self.ip`, so on resume `run()`
            // continues past it — the scheduler delivers any result (a channel
            // id, a received value) by pushing onto this VM's stack first.
            Op::Go(name_idx, argc) => {
                let n = *argc as usize;
                let mut args = Vec::with_capacity(n);
                for _ in 0..n {
                    args.push(self.pop());
                }
                args.reverse();
                self.sched = Some(crate::sched::SchedReq::Go {
                    name_idx: *name_idx,
                    args,
                });
                self.halted = true;
            }
            Op::ChanMake => {
                let cap = self.pop().to_int().max(0) as usize;
                self.sched = Some(crate::sched::SchedReq::Make { cap });
                self.halted = true;
            }
            Op::ChanSend => {
                let val = self.pop();
                let ch = self.pop().to_int();
                self.sched = Some(crate::sched::SchedReq::Send { ch, val });
                self.halted = true;
            }
            Op::ChanRecv => {
                let ch = self.pop().to_int();
                self.sched = Some(crate::sched::SchedReq::Recv { ch });
                self.halted = true;
            }
            Op::ChanClose => {
                let ch = self.pop().to_int();
                self.sched = Some(crate::sched::SchedReq::Close { ch });
                self.halted = true;
            }
            Op::CallDynamic(argc) => {
                // The subroutine name-index is on top; the `argc` args are below.
                let name_idx = self.pop().to_int() as u16;
                if let Some(entry_ip) = self.chunk.find_sub(name_idx) {
                    self.frames.push(Frame {
                        return_ip: self.ip,
                        stack_base: self.stack.len() - *argc as usize,
                        slots: Vec::new(),
                    });
                    self.ip = entry_ip;
                } else {
                    return ExecFlow::Ret(VMResult::Error(
                        "call of a nil or unknown function value".to_string(),
                    ));
                }
            }
            Op::Select(num_cases, has_default) => {
                let n = *num_cases as usize;
                let mut raw = Vec::with_capacity(n * 3);
                for _ in 0..n * 3 {
                    raw.push(self.pop());
                }
                raw.reverse();
                let cases = raw
                    .chunks_exact(3)
                    .map(|c| crate::sched::SelectCase {
                        ch: c[0].to_int(),
                        recv: c[1].to_int() != 0,
                        val: c[2].clone(),
                    })
                    .collect();
                self.sched = Some(crate::sched::SchedReq::Select {
                    cases,
                    has_default: *has_default != 0,
                });
                self.halted = true;
            }
        }

        // Tracing JIT: finalize an active recording on either:
        //   (a) the recorder was marked aborted earlier (e.g. trace
        //       exceeded MAX_TRACE_LEN, observed CallBuiltin, etc.) —
        //       discard and clean up the cache entry, OR
        //   (b) the just-dispatched jump landed at the anchor IP —
        //       this is the loop-closing backward branch.
        // Internal mid-trace branches that DON'T land at the anchor
        // continue recording; their direction is captured in
        // `recorded_ips` for later compile-time guard emission.
        // Only run finalize if the recorder was armed *before* this step;
        // a recorder freshly armed inside this step starts recording on
        // the next iteration.
        #[cfg(feature = "jit")]
        if recorder_was_armed && self.recorder.is_some() {
            let aborted = self.recorder.as_ref().map_or(false, |r| r.aborted);
            // Phase 9: close on the recorded `close_anchor_ip` rather
            // than `record_anchor_ip` — for side traces these differ.
            let close_ip = self
                .recorder
                .as_ref()
                .map(|r| r.close_anchor_ip)
                .unwrap_or(0);
            let was_jump = matches!(
                &ops[ip],
                Op::Jump(_)
                    | Op::JumpIfTrue(_)
                    | Op::JumpIfFalse(_)
                    | Op::JumpIfTrueKeep(_)
                    | Op::JumpIfFalseKeep(_)
            );
            let landed_at_anchor = self.ip == close_ip;
            if aborted || (was_jump && landed_at_anchor) {
                self.finalize_recorder();
            }
        }
        ExecFlow::Cont
    }

    /// AOT closed-world per-op step. Mirrors one iteration of the [`VM::run`]
    /// dispatch loop for the op at `ip`: advances `self.ip` to `ip + 1`, runs
    /// the op via [`VM::exec_op`], and returns the **next instruction index**
    /// for the native driver to branch to — or `-1` when the run terminates
    /// (an op returned [`ExecFlow::Ret`] or set the halted flag). On terminate,
    /// any explicit result is stashed in `self.aot_result`. The returned index
    /// is `ip + 1` for ordinary ops and the jump/call/return target for
    /// control-flow ops, so the native driver branches without ever reading the
    /// `VM` struct layout.
    #[cfg(feature = "aot")]
    pub(crate) fn aot_exec_op(&mut self, ip: usize) -> i64 {
        // SAFETY: chunk.ops is not mutated during execution; alias it past the
        // borrow checker exactly as VM::run does.
        let ops = &self.chunk.ops as *const Vec<Op>;
        let ops = unsafe { &*ops };
        self.ip = ip + 1;
        match self.exec_op(ops, ip, false) {
            ExecFlow::Ret(r) => {
                self.aot_result = Some(r);
                return -1;
            }
            ExecFlow::Cont => {}
        }
        if self.halted {
            return -1;
        }
        self.ip as i64
    }

    /// Finalize an AOT run: if no op stored an explicit result, apply the same
    /// tail logic as [`VM::run`] (pop the stack top as the value, else
    /// `Halted`). Called once by the native driver's return path.
    #[cfg(feature = "aot")]
    pub(crate) fn aot_finish(&mut self) {
        if self.aot_result.is_none() {
            self.aot_result = Some(match self.stack.pop() {
                Some(v) => VMResult::Ok(v),
                None => VMResult::Halted,
            });
        }
    }

    /// Store an explicit integer result computed by natively-lowered AOT code.
    /// The native fast path holds intermediate values in registers (never on
    /// `self.stack`), so it reports its final value through this hook instead of
    /// the stack-tail logic in [`VM::aot_finish`].
    #[cfg(feature = "aot")]
    pub(crate) fn aot_set_int_result(&mut self, n: i64) {
        self.aot_result = Some(VMResult::Ok(Value::Int(n)));
    }

    /// Store an explicit float result computed by natively-lowered AOT code.
    /// The float analog of [`VM::aot_set_int_result`].
    #[cfg(feature = "aot")]
    pub(crate) fn aot_set_float_result(&mut self, f: f64) {
        self.aot_result = Some(VMResult::Ok(Value::Float(f)));
    }

    /// Allocate an arena slot holding `v`, returning its handle. Reuses a freed
    /// slot when available so loops that rebuild values stay bounded.
    #[cfg(feature = "aot")]
    fn aot_alloc(&mut self, v: Value) -> i64 {
        if let Some(h) = self.aot_free.pop() {
            self.aot_arena[h as usize] = v;
            h as i64
        } else {
            let h = self.aot_arena.len() as i64;
            self.aot_arena.push(v);
            h
        }
    }

    /// Take the value out of an arena slot, freeing the slot for reuse. A handle
    /// is owned by exactly one place, so consuming it returns its slot.
    #[cfg(feature = "aot")]
    fn aot_take(&mut self, handle: i64) -> Value {
        match self.aot_arena.get_mut(handle as usize) {
            Some(slot) => {
                let v = std::mem::replace(slot, Value::Undef);
                self.aot_free.push(handle as u32);
                v
            }
            None => Value::Undef,
        }
    }

    /// Box the boxed-stack top into the value arena, returning its owning handle.
    /// A shimmed op leaves its (boxed) result on `self.stack`; the native code
    /// stashes it here and threads the handle through a register, the way scalars
    /// are threaded directly.
    #[cfg(feature = "aot")]
    pub(crate) fn aot_box(&mut self) -> i64 {
        let v = self.stack.pop().unwrap_or(Value::Undef);
        self.aot_alloc(v)
    }

    /// Push the value for `handle` back onto the boxed stack so a shimmed op can
    /// consume it, *consuming* the handle (its slot is freed). The inverse of
    /// [`VM::aot_box`]; every owned handle is unboxed exactly once.
    #[cfg(feature = "aot")]
    pub(crate) fn aot_unbox(&mut self, handle: i64) {
        let v = self.aot_take(handle);
        self.stack.push(v);
    }

    /// Clone the value behind `handle` into a fresh owned handle (used when a
    /// slot is *read*: the slot keeps its handle, the stack gets its own copy).
    #[cfg(feature = "aot")]
    pub(crate) fn aot_clone(&mut self, handle: i64) -> i64 {
        let v = self
            .aot_arena
            .get(handle as usize)
            .cloned()
            .unwrap_or(Value::Undef);
        self.aot_alloc(v)
    }

    /// Free an owned handle without using its value (e.g. a slot overwritten, or
    /// an `Obj` popped). A negative handle is the "empty slot" sentinel: no-op.
    #[cfg(feature = "aot")]
    pub(crate) fn aot_free(&mut self, handle: i64) {
        if handle >= 0 && (handle as usize) < self.aot_arena.len() {
            self.aot_arena[handle as usize] = Value::Undef;
            self.aot_free.push(handle as u32);
        }
    }

    /// Store an explicit boxed (heap) result from a register handle (consuming
    /// it). The arena analog of [`VM::aot_set_int_result`].
    #[cfg(feature = "aot")]
    pub(crate) fn aot_set_obj_result(&mut self, handle: i64) {
        let v = self.aot_take(handle);
        self.aot_result = Some(VMResult::Ok(v));
    }

    /// Write a register-held `Obj` slot/global back to the VM on deopt: store a
    /// *clone* of the handle's value (the run ends in the interpreter afterward,
    /// so the arena is abandoned — cloning avoids disturbing other live handles).
    #[cfg(feature = "aot")]
    pub(crate) fn aot_store_slot_obj(&mut self, idx: u32, handle: i64) {
        let v = self
            .aot_arena
            .get(handle as usize)
            .cloned()
            .unwrap_or(Value::Undef);
        self.set_slot(idx as u16, v);
    }

    /// Global analog of [`VM::aot_store_slot_obj`].
    #[cfg(feature = "aot")]
    pub(crate) fn aot_store_global_obj(&mut self, idx: u32, handle: i64) {
        let v = self
            .aot_arena
            .get(handle as usize)
            .cloned()
            .unwrap_or(Value::Undef);
        self.set_var(idx as u16, v);
    }

    /// Spill a scalar from a native register onto the boxed operand stack, so a
    /// shimmed (non-lowered) op can consume it. Boxed by kind to match exactly
    /// what the interpreter would have on the stack.
    #[cfg(feature = "aot")]
    pub(crate) fn aot_push_int(&mut self, n: i64) {
        self.stack.push(Value::Int(n));
    }

    /// Float analog of [`VM::aot_push_int`].
    #[cfg(feature = "aot")]
    pub(crate) fn aot_push_float(&mut self, f: f64) {
        self.stack.push(Value::Float(f));
    }

    /// Bool analog of [`VM::aot_push_int`] (the register carries 0/1).
    #[cfg(feature = "aot")]
    pub(crate) fn aot_push_bool(&mut self, n: i64) {
        self.stack.push(Value::Bool(n != 0));
    }

    /// Write a register-resident slot back to the VM frame on deopt, so the
    /// resumed interpreter sees the current value. (Native code caches slots in
    /// registers; the frame is otherwise stale.)
    #[cfg(feature = "aot")]
    pub(crate) fn aot_store_slot_int(&mut self, idx: u32, n: i64) {
        self.set_slot(idx as u16, Value::Int(n));
    }

    /// Float analog of [`VM::aot_store_slot_int`].
    #[cfg(feature = "aot")]
    pub(crate) fn aot_store_slot_float(&mut self, idx: u32, f: f64) {
        self.set_slot(idx as u16, Value::Float(f));
    }

    /// Global analog of [`VM::aot_store_slot_int`].
    #[cfg(feature = "aot")]
    pub(crate) fn aot_store_global_int(&mut self, idx: u32, n: i64) {
        self.set_var(idx as u16, Value::Int(n));
    }

    /// Float global analog of [`VM::aot_store_slot_int`].
    #[cfg(feature = "aot")]
    pub(crate) fn aot_store_global_float(&mut self, idx: u32, f: f64) {
        self.set_var(idx as u16, Value::Float(f));
    }

    /// Deopt exit: resume interpretation from `ip` with the VM state the native
    /// code just reconstructed (operand stack spilled, slots/globals written
    /// back), and capture the rest-of-chunk result. One-way — native code does
    /// not re-enter after this.
    #[cfg(feature = "aot")]
    pub(crate) fn aot_resume(&mut self, ip: u32) {
        self.ip = ip as usize;
        let r = self.run();
        self.aot_result = Some(r);
    }

    /// Reload a float off the boxed operand stack into a native register — the
    /// reload half of the boundary, for source ops whose result kind is
    /// statically `Float` (so no runtime type guard is needed).
    #[cfg(feature = "aot")]
    pub(crate) fn aot_pop_float(&mut self) -> f64 {
        self.stack.pop().map(|v| v.to_float()).unwrap_or(0.0)
    }

    /// Reload an integer (via `to_int`) off the boxed operand stack — used for
    /// `GetStatus`, whose `Value::Status` carries its code as the int.
    #[cfg(feature = "aot")]
    pub(crate) fn aot_pop_int(&mut self) -> i64 {
        self.stack.pop().map(|v| v.to_int()).unwrap_or(0)
    }

    /// Spill a `Status` code from a register onto the boxed stack (deopt/sink).
    #[cfg(feature = "aot")]
    pub(crate) fn aot_push_status(&mut self, n: i64) {
        self.stack.push(Value::Status(n as i32));
    }

    /// Store a `Status` result computed by natively-lowered AOT code.
    #[cfg(feature = "aot")]
    pub(crate) fn aot_set_status_result(&mut self, n: i64) {
        self.aot_result = Some(VMResult::Ok(Value::Status(n as i32)));
    }

    /// Store an error result from natively-lowered AOT code, keyed by a small
    /// code so the native side passes an integer rather than a string. The
    /// messages match the interpreter's for the corresponding ops exactly.
    #[cfg(feature = "aot")]
    pub(crate) fn aot_set_error(&mut self, code: u32) {
        let msg = match code {
            0 => "division by zero attempted",
            1 => "division by zero attempted in `%'",
            2 => "lshift: negative values are not allowed",
            3 => "rshift: negative values are not allowed",
            4 => "compl: negative value is not allowed",
            _ => "aot: runtime error",
        };
        self.aot_result = Some(VMResult::Error(msg.to_string()));
    }

    /// Take the result captured by the AOT driver, leaving `None` behind.
    #[cfg(feature = "aot")]
    pub(crate) fn take_aot_result(&mut self) -> VMResult {
        self.aot_result.take().unwrap_or(VMResult::Halted)
    }

    // ── Helpers ──

    /// Dispatch one AWK op (`Op::ExtendedWide(id, payload)` with `id` in the
    /// reserved AWK range) through the registered [`AwkHost`]. Value operands
    /// come from the stack (pushed in source order); `payload` carries the
    /// inline integer operand (field index / argument count / name-pool index).
    ///
    /// When no AWK host is registered, AWK ops are inert: ops that yield a value
    /// push a neutral default so the stack stays balanced, statement-form ops
    /// (`print`/`delete`/field-set) simply drop their operands.
    ///
    /// [`AwkHost`]: crate::awk_host::AwkHost
    fn dispatch_awk(&mut self, id: u16, payload: usize) {
        use crate::awk_builtins as ab;
        use crate::awk_host::AwkLvalue;

        // Take the host out to satisfy the borrow checker (handlers reach back
        // into `self` via the stack); restore it afterwards. Mirrors the
        // `ext_handler` take/restore pattern used for `Op::Extended`.
        let mut host = match self.awk_host.take() {
            Some(h) => h,
            None => {
                self.dispatch_awk_stub(id, payload);
                return;
            }
        };

        // Pop `n` value operands, returned in source (pushed) order.
        macro_rules! pop_n {
            ($n:expr) => {{
                let n = $n;
                let mut v: Vec<Value> = (0..n).map(|_| self.pop()).collect();
                v.reverse();
                v
            }};
        }
        let name_at = |vm: &Self, idx: usize| -> String {
            vm.chunk.names.get(idx).cloned().unwrap_or_default()
        };

        match id {
            ab::AWK_FIELD_GET => {
                let i = self.pop().to_int();
                let v = host.field_get(i);
                self.push(v);
            }
            ab::AWK_FIELD_SET => {
                let i = self.pop().to_int();
                let v = self.pop();
                host.field_set(i, v);
            }
            ab::AWK_NF => {
                let n = host.nf();
                self.push(Value::Int(n));
            }
            ab::AWK_SET_RECORD => {
                let v = self.pop();
                host.set_record(v);
            }
            ab::AWK_SPECIAL_GET => {
                let name = name_at(self, payload);
                let v = host.special_get(&name);
                self.push(v);
            }
            ab::AWK_SPECIAL_SET => {
                let name = name_at(self, payload);
                let v = self.pop();
                host.special_set(&name, v);
            }
            ab::AWK_PRINT => {
                let args = pop_n!(payload);
                host.print(&args);
            }
            ab::AWK_PRINTF => {
                let mut args = pop_n!(payload);
                let fmt = if args.is_empty() {
                    String::new()
                } else {
                    args.remove(0).to_str()
                };
                host.printf(&fmt, &args);
            }
            ab::AWK_SPRINTF => {
                let mut args = pop_n!(payload);
                let fmt = if args.is_empty() {
                    String::new()
                } else {
                    args.remove(0).to_str()
                };
                let v = host.sprintf(&fmt, &args);
                self.push(v);
            }
            ab::AWK_GETLINE => {
                // For file/command sources the operand string is on the stack.
                let operand = match payload {
                    ab::getline_source::FILE
                    | ab::getline_source::FILE_VAR
                    | ab::getline_source::CMD
                    | ab::getline_source::CMD_VAR => Some(self.pop().to_str()),
                    _ => None,
                };
                let status = host.getline(payload, operand.as_deref(), None);
                self.push(Value::Int(status));
            }
            ab::AWK_LENGTH => {
                let arg = if payload == 0 { None } else { Some(self.pop()) };
                let n = host.length(arg.as_ref());
                self.push(Value::Int(n));
            }
            ab::AWK_SUBSTR => {
                let args = pop_n!(payload);
                let s = args.first().cloned().unwrap_or(Value::str(""));
                let m = args.get(1).map(|v| v.to_int()).unwrap_or(1);
                let n = args.get(2).map(|v| v.to_int());
                let v = host.substr(&s, m, n);
                self.push(v);
            }
            ab::AWK_INDEX => {
                let t = self.pop();
                let s = self.pop();
                let r = host.index(&s, &t);
                self.push(Value::Int(r));
            }
            ab::AWK_SPLIT => {
                let args = pop_n!(payload);
                let s = args.first().cloned().unwrap_or(Value::str(""));
                let arr = args.get(1).map(|v| v.to_str()).unwrap_or_default();
                let fs = args.get(2);
                let n = host.split(&s, &arr, fs);
                self.push(Value::Int(n));
            }
            ab::AWK_SUB | ab::AWK_GSUB => {
                // Stack: [re, repl, target_name]. The target name string lets
                // the host write back to the right lvalue (a var here; field /
                // array targets use their own dedicated emission).
                let args = pop_n!(payload);
                let re = args.first().cloned().unwrap_or(Value::str(""));
                let repl = args.get(1).cloned().unwrap_or(Value::str(""));
                let target = args
                    .get(2)
                    .map(|v| AwkLvalue::Var(v.to_str()))
                    .unwrap_or(AwkLvalue::Field(0));
                let n = if id == ab::AWK_SUB {
                    host.sub(&re, &repl, &target)
                } else {
                    host.gsub(&re, &repl, &target)
                };
                self.push(Value::Int(n));
            }
            ab::AWK_MATCH => {
                let re = self.pop();
                let s = self.pop();
                let r = host.match_re(&s, &re);
                self.push(Value::Int(r));
            }
            ab::AWK_GENSUB => {
                // Stack: [re, repl, how, target?] (payload = argc, 3..=4).
                let args = pop_n!(payload);
                let re = args.first().cloned().unwrap_or(Value::str(""));
                let repl = args.get(1).cloned().unwrap_or(Value::str(""));
                let how = args.get(2).cloned().unwrap_or(Value::str("g"));
                let target = args.get(3);
                let v = host.gensub(&re, &repl, &how, target);
                self.push(v);
            }
            ab::AWK_TOLOWER => {
                let s = self.pop();
                let v = host.tolower(&s);
                self.push(v);
            }
            ab::AWK_TOUPPER => {
                let s = self.pop();
                let v = host.toupper(&s);
                self.push(v);
            }
            ab::AWK_INT => {
                let x = self.pop();
                let v = host.int(&x);
                self.push(v);
            }
            ab::AWK_SQRT => {
                let x = self.pop();
                let v = host.sqrt(&x);
                self.push(v);
            }
            ab::AWK_SIN => {
                let x = self.pop();
                let v = host.sin(&x);
                self.push(v);
            }
            ab::AWK_COS => {
                let x = self.pop();
                let v = host.cos(&x);
                self.push(v);
            }
            ab::AWK_EXP => {
                let x = self.pop();
                let v = host.exp(&x);
                self.push(v);
            }
            ab::AWK_LOG => {
                let x = self.pop();
                let v = host.log(&x);
                self.push(v);
            }
            ab::AWK_ATAN2 => {
                let x = self.pop();
                let y = self.pop();
                let v = host.atan2(&y, &x);
                self.push(v);
            }
            ab::AWK_AND => {
                let args = pop_n!(payload);
                let v = host.and(&args);
                self.push(v);
            }
            ab::AWK_OR => {
                let args = pop_n!(payload);
                let v = host.or(&args);
                self.push(v);
            }
            ab::AWK_XOR => {
                let args = pop_n!(payload);
                let v = host.xor(&args);
                self.push(v);
            }
            ab::AWK_COMPL => {
                let v = self.pop();
                let r = host.compl(&v);
                self.push(r);
            }
            ab::AWK_LSHIFT => {
                let n = self.pop();
                let v = self.pop();
                let r = host.lshift(&v, &n);
                self.push(r);
            }
            ab::AWK_RSHIFT => {
                let n = self.pop();
                let v = self.pop();
                let r = host.rshift(&v, &n);
                self.push(r);
            }
            ab::AWK_STRTONUM => {
                let s = self.pop();
                let r = host.strtonum(&s);
                self.push(r);
            }
            ab::AWK_SYSTIME => {
                let r = host.systime();
                self.push(r);
            }
            ab::AWK_RAND => {
                let r = crate::awk_host::awk_rand(&mut self.awk_rand_seed);
                self.push(Value::Float(r));
            }
            ab::AWK_SRAND => {
                let n = if payload >= 1 {
                    Some(self.pop().to_float() as u32 as u64)
                } else {
                    None
                };
                let r = crate::awk_host::awk_srand(&mut self.awk_rand_seed, n);
                self.push(Value::Float(r));
            }
            ab::AWK_STRFTIME => {
                let args = pop_n!(payload);
                let r = host.strftime(&args);
                self.push(r);
            }
            ab::AWK_MKTIME => {
                let args = pop_n!(payload);
                let r = host.mktime(&args);
                self.push(r);
            }
            ab::AWK_ORD => {
                let a = self.pop();
                let r = host.ord(&a);
                self.push(r);
            }
            ab::AWK_CHR => {
                let a = self.pop();
                let r = host.chr(&a);
                self.push(r);
            }
            ab::AWK_MKBOOL => {
                let a = self.pop();
                let r = host.mkbool(&a);
                self.push(r);
            }
            ab::AWK_INTDIV => {
                let b = self.pop();
                let a = self.pop();
                let r = host.intdiv(&a, &b);
                self.push(r);
            }
            ab::AWK_INTDIV0 => {
                let b = self.pop();
                let a = self.pop();
                let r = host.intdiv0(&a, &b);
                self.push(r);
            }
            ab::AWK_ARRAY_GET => {
                let name = name_at(self, payload);
                let key = self.pop();
                let v = host.array_get(&name, &key);
                self.push(v);
            }
            ab::AWK_ARRAY_SET => {
                let name = name_at(self, payload);
                let key = self.pop();
                let v = self.pop();
                host.array_set(&name, &key, v);
            }
            ab::AWK_ARRAY_EXISTS => {
                let name = name_at(self, payload);
                let key = self.pop();
                let b = host.array_exists(&name, &key);
                self.push(Value::Bool(b));
            }
            ab::AWK_ARRAY_DELETE => {
                let name = name_at(self, payload);
                let key = self.pop();
                host.array_delete(&name, &key);
            }
            ab::AWK_ARRAY_CLEAR => {
                let name = name_at(self, payload);
                host.array_clear(&name);
            }
            ab::AWK_ARRAY_LEN => {
                let name = name_at(self, payload);
                let n = host.array_len(&name);
                self.push(Value::Int(n));
            }
            // Unknown AWK op id: drop nothing, push Undef to keep callers that
            // expect a value from glitching. (Reserved range, forward-compat.)
            _ => self.push(Value::Undef),
        }

        self.awk_host = Some(host);
    }

    /// Inert fallback for AWK ops when no [`AwkHost`] is registered. Keeps the
    /// stack balanced: value-producing ops push a neutral default; statement
    /// ops drop their operands.
    fn dispatch_awk_stub(&mut self, id: u16, payload: usize) {
        use crate::awk_builtins as ab;
        use crate::awk_host::{
            awk_canon_nan, awk_chr, awk_compl, awk_fold_and, awk_fold_or, awk_fold_xor, awk_index,
            awk_int, awk_intdiv, awk_intdiv0, awk_length, awk_lshift, awk_mkbool, awk_mktime,
            awk_ord, awk_rand, awk_rshift, awk_srand, awk_strftime, awk_strtonum, awk_substr,
            awk_systime, awk_tolower, awk_toupper,
        };
        match id {
            // value-producing: pop declared operands, push neutral default
            ab::AWK_FIELD_GET => {
                self.pop();
                self.push(Value::str(""));
            }
            // `length(s)` (scalar form, payload>0) is host-independent — compute
            // it natively. `length($0)` (payload==0) and `length(arr)` need the
            // host, so they still yield 0 here.
            ab::AWK_LENGTH if payload > 0 => {
                let s = self.pop();
                self.push(Value::Int(awk_length(Some(&s))));
            }
            ab::AWK_NF | ab::AWK_LENGTH | ab::AWK_ARRAY_LEN => {
                self.push(Value::Int(0));
            }
            ab::AWK_SPECIAL_GET => self.push(Value::Undef),
            ab::AWK_SPRINTF => {
                for _ in 0..payload {
                    self.pop();
                }
                self.push(Value::str(""));
            }
            // Host-independent string builtins: compute the real result so these
            // AWK ops execute natively even with no registered host. Operand pop
            // order mirrors the host path in `dispatch_awk`.
            ab::AWK_SUBSTR => {
                let mut args: Vec<Value> = (0..payload).map(|_| self.pop()).collect();
                args.reverse();
                let s = args.first().cloned().unwrap_or(Value::str(""));
                let m = args.get(1).map(|v| v.to_int()).unwrap_or(1);
                let n = args.get(2).map(|v| v.to_int());
                self.push(awk_substr(&s, m, n));
            }
            ab::AWK_TOLOWER => {
                let s = self.pop();
                self.push(awk_tolower(&s));
            }
            ab::AWK_TOUPPER => {
                let s = self.pop();
                self.push(awk_toupper(&s));
            }
            // Host-independent numeric builtins: pure f64 math, computed
            // natively even with no registered host.
            ab::AWK_INT => {
                let x = self.pop();
                self.push(awk_int(&x));
            }
            ab::AWK_SQRT => {
                let x = self.pop();
                self.push(Value::Float(x.to_float().sqrt()));
            }
            ab::AWK_SIN => {
                let x = self.pop();
                self.push(Value::Float(awk_canon_nan(x.to_float().sin())));
            }
            ab::AWK_COS => {
                let x = self.pop();
                self.push(Value::Float(awk_canon_nan(x.to_float().cos())));
            }
            ab::AWK_EXP => {
                let x = self.pop();
                self.push(Value::Float(awk_canon_nan(x.to_float().exp())));
            }
            ab::AWK_LOG => {
                let x = self.pop();
                self.push(Value::Float(x.to_float().ln()));
            }
            ab::AWK_ATAN2 => {
                let x = self.pop();
                let y = self.pop();
                self.push(Value::Float(awk_canon_nan(
                    y.to_float().atan2(x.to_float()),
                )));
            }
            // Host-independent bitwise builtins (gawk): pure integer math.
            ab::AWK_AND => {
                let args: Vec<Value> = {
                    let mut v: Vec<Value> = (0..payload).map(|_| self.pop()).collect();
                    v.reverse();
                    v
                };
                self.push(Value::Int(awk_fold_and(&args)));
            }
            ab::AWK_OR => {
                let args: Vec<Value> = {
                    let mut v: Vec<Value> = (0..payload).map(|_| self.pop()).collect();
                    v.reverse();
                    v
                };
                self.push(Value::Int(awk_fold_or(&args)));
            }
            ab::AWK_XOR => {
                let args: Vec<Value> = {
                    let mut v: Vec<Value> = (0..payload).map(|_| self.pop()).collect();
                    v.reverse();
                    v
                };
                self.push(Value::Int(awk_fold_xor(&args)));
            }
            ab::AWK_COMPL => {
                let v = self.pop();
                self.push(Value::Int(awk_compl(&v)));
            }
            ab::AWK_LSHIFT => {
                let n = self.pop();
                let v = self.pop();
                self.push(Value::Int(awk_lshift(&v, &n)));
            }
            ab::AWK_RSHIFT => {
                let n = self.pop();
                let v = self.pop();
                self.push(Value::Int(awk_rshift(&v, &n)));
            }
            ab::AWK_STRTONUM => {
                let s = self.pop();
                self.push(Value::Float(awk_strtonum(&s.to_str())));
            }
            ab::AWK_SYSTIME => {
                self.push(Value::Float(awk_systime()));
            }
            ab::AWK_RAND => {
                let r = awk_rand(&mut self.awk_rand_seed);
                self.push(Value::Float(r));
            }
            ab::AWK_SRAND => {
                let n = if payload >= 1 {
                    Some(self.pop().to_float() as u32 as u64)
                } else {
                    None
                };
                let r = awk_srand(&mut self.awk_rand_seed, n);
                self.push(Value::Float(r));
            }
            ab::AWK_STRFTIME => {
                let mut args: Vec<Value> = (0..payload).map(|_| self.pop()).collect();
                args.reverse();
                self.push(awk_strftime(&args));
            }
            ab::AWK_MKTIME => {
                let mut args: Vec<Value> = (0..payload).map(|_| self.pop()).collect();
                args.reverse();
                self.push(awk_mktime(&args));
            }
            ab::AWK_ORD => {
                let a = self.pop();
                self.push(awk_ord(&a));
            }
            ab::AWK_CHR => {
                let a = self.pop();
                self.push(awk_chr(&a));
            }
            ab::AWK_MKBOOL => {
                let a = self.pop();
                self.push(awk_mkbool(&a));
            }
            ab::AWK_INTDIV => {
                let b = self.pop();
                let a = self.pop();
                self.push(awk_intdiv(&a, &b));
            }
            ab::AWK_INTDIV0 => {
                let b = self.pop();
                let a = self.pop();
                self.push(awk_intdiv0(&a, &b));
            }
            ab::AWK_INDEX => {
                let t = self.pop();
                let s = self.pop();
                self.push(Value::Int(awk_index(&s, &t)));
            }
            ab::AWK_MATCH => {
                self.pop();
                self.pop();
                self.push(Value::Int(0));
            }
            ab::AWK_SPLIT | ab::AWK_SUB | ab::AWK_GSUB => {
                for _ in 0..payload {
                    self.pop();
                }
                self.push(Value::Int(0));
            }
            ab::AWK_GENSUB => {
                for _ in 0..payload {
                    self.pop();
                }
                self.push(Value::str(""));
            }
            ab::AWK_GETLINE => {
                if matches!(
                    payload,
                    ab::getline_source::FILE
                        | ab::getline_source::FILE_VAR
                        | ab::getline_source::CMD
                        | ab::getline_source::CMD_VAR
                ) {
                    self.pop();
                }
                self.push(Value::Int(0));
            }
            ab::AWK_ARRAY_GET => {
                self.pop();
                self.push(Value::str(""));
            }
            ab::AWK_ARRAY_EXISTS => {
                self.pop();
                self.push(Value::Bool(false));
            }
            // statement-form ops: drop operands, push nothing
            ab::AWK_FIELD_SET | ab::AWK_ARRAY_SET => {
                self.pop();
                self.pop();
            }
            ab::AWK_SET_RECORD | ab::AWK_SPECIAL_SET | ab::AWK_ARRAY_DELETE => {
                self.pop();
            }
            ab::AWK_PRINT | ab::AWK_PRINTF => {
                for _ in 0..payload {
                    self.pop();
                }
            }
            ab::AWK_ARRAY_CLEAR => {}
            _ => {}
        }
    }

    fn get_var(&self, idx: u16) -> Value {
        self.globals
            .get(idx as usize)
            .cloned()
            .unwrap_or(Value::Undef)
    }

    fn set_var(&mut self, idx: u16, val: Value) {
        let idx = idx as usize;
        if idx >= self.globals.len() {
            self.globals.resize(idx + 1, Value::Undef);
        }
        self.globals[idx] = val;
    }

    /// Read a slot from the current (top) call frame.
    ///
    /// Returns `Value::Undef` when there is no active frame or the slot
    /// index is out of range. Public so frontend extension handlers
    /// (`set_extension_handler`) can read slot operands without reaching
    /// into `frames` directly.
    pub fn get_slot(&self, slot: u16) -> Value {
        self.frames
            .last()
            .and_then(|f| f.slots.get(slot as usize))
            .cloned()
            .unwrap_or(Value::Undef)
    }

    /// Write a slot in the current (top) call frame, growing the frame's
    /// slot vector as needed. No-op when there is no active frame.
    ///
    /// Public so frontend extension handlers can write slot results back
    /// without reaching into `frames` directly.
    pub fn set_slot(&mut self, slot: u16, val: Value) {
        if let Some(frame) = self.frames.last_mut() {
            let idx = slot as usize;
            if idx >= frame.slots.len() {
                frame.slots.resize(idx + 1, Value::Undef);
            }
            frame.slots[idx] = val;
        }
    }
}

/// Pool of reusable `VM` instances.
///
/// `VM::new` does ~3 `Vec` allocations (stack, frames, globals) at
/// construction. Callers that run many small chunks back-to-back —
/// REPL-style invocation, batch script execution, eval loops — pay
/// that cost on every call. `VMPool` recycles the allocations: the
/// first `acquire` allocates, subsequent acquires pop a previously-
/// released VM and reset it via `VM::reset`.
///
/// # Example
///
/// ```
/// use fusevm::{ChunkBuilder, Op, VMPool, VMResult, Value};
///
/// let mut pool = VMPool::new();
///
/// for _ in 0..1000 {
///     let mut b = ChunkBuilder::new();
///     b.emit(Op::LoadInt(40), 1);
///     b.emit(Op::LoadInt(2), 1);
///     b.emit(Op::Add, 1);
///
///     let mut vm = pool.acquire(b.build());
///     let result = vm.run();
///     assert!(matches!(result, VMResult::Ok(Value::Int(42))));
///     pool.release(vm);
/// }
/// ```
pub struct VMPool {
    pool: Vec<VM>,
}

impl VMPool {
    /// Construct an empty pool.
    pub fn new() -> Self {
        Self { pool: Vec::new() }
    }

    /// Construct with a pre-allocated capacity.
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            pool: Vec::with_capacity(cap),
        }
    }

    /// Acquire a VM ready to run `chunk`. Pops a recycled VM if
    /// available; otherwise constructs a fresh one. The returned VM
    /// inherits the pool's previously-released VMs' allocations
    /// (Vec capacities preserved).
    pub fn acquire(&mut self, chunk: Chunk) -> VM {
        if let Some(mut vm) = self.pool.pop() {
            vm.reset(chunk);
            vm
        } else {
            VM::new(chunk)
        }
    }

    /// Return a VM to the pool for later reuse. The VM's allocations
    /// are kept; only state is cleared on the next `acquire`.
    pub fn release(&mut self, vm: VM) {
        self.pool.push(vm);
    }

    /// Run a closure against an acquired VM, returning it to the pool
    /// after the closure finishes (RAII-style scope).
    pub fn with<F, T>(&mut self, chunk: Chunk, f: F) -> T
    where
        F: FnOnce(&mut VM) -> T,
    {
        let mut vm = self.acquire(chunk);
        let r = f(&mut vm);
        self.release(vm);
        r
    }

    /// Number of VMs currently held in the pool (released, ready for
    /// reuse). Doesn't count VMs currently checked out via `acquire`.
    pub fn len(&self) -> usize {
        self.pool.len()
    }

    /// Whether the pool is empty.
    pub fn is_empty(&self) -> bool {
        self.pool.is_empty()
    }
}

impl Default for VMPool {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunk::ChunkBuilder;

    #[test]
    fn test_arithmetic() {
        let mut b = ChunkBuilder::new();
        b.emit(Op::LoadInt(10), 1);
        b.emit(Op::LoadInt(32), 1);
        b.emit(Op::Add, 1);
        let mut vm = VM::new(b.build());
        match vm.run() {
            VMResult::Ok(Value::Int(42)) => {}
            other => panic!("expected Int(42), got {:?}", other),
        }
    }

    #[test]
    fn test_jump() {
        let mut b = ChunkBuilder::new();
        b.emit(Op::LoadInt(1), 1);
        b.emit(Op::Jump(3), 1);
        b.emit(Op::LoadInt(999), 1); // skipped
                                     // ip 3:
        b.emit(Op::LoadInt(2), 1);
        b.emit(Op::Add, 1);
        let mut vm = VM::new(b.build());
        match vm.run() {
            VMResult::Ok(Value::Int(3)) => {}
            other => panic!("expected Int(3), got {:?}", other),
        }
    }

    #[test]
    fn test_fused_sum_loop() {
        // sum = 0; for i in 0..100 { sum += i }
        let mut b = ChunkBuilder::new();
        b.emit(Op::PushFrame, 1);
        b.emit(Op::LoadInt(0), 1);
        b.emit(Op::SetSlot(0), 1); // sum = 0
        b.emit(Op::LoadInt(0), 1);
        b.emit(Op::SetSlot(1), 1); // i = 0
        b.emit(Op::AccumSumLoop(0, 1, 100), 1);
        b.emit(Op::GetSlot(0), 1);

        let mut vm = VM::new(b.build());
        match vm.run() {
            VMResult::Ok(Value::Int(4950)) => {}
            other => panic!("expected Int(4950), got {:?}", other),
        }
    }

    #[test]
    fn test_call_dynamic() {
        let mut b = ChunkBuilder::new();
        let dbl = b.add_name("double");
        // main: push 21, push name_idx of "double", CallDynamic(1)
        b.emit(Op::LoadInt(21), 1);
        b.emit(Op::LoadInt(dbl as i64), 1);
        b.emit(Op::CallDynamic(1), 1);
        let end = b.emit(Op::Jump(0), 1);
        let ip = b.current_pos();
        b.add_sub_entry(dbl, ip);
        b.emit(Op::LoadInt(2), 2);
        b.emit(Op::Mul, 2);
        b.emit(Op::ReturnValue, 2);
        b.patch_jump(end, b.current_pos());
        let mut vm = VM::new(b.build());
        assert!(matches!(vm.run(), VMResult::Ok(Value::Int(42))));
    }

    #[test]
    fn test_function_call() {
        let mut b = ChunkBuilder::new();
        let double_name = b.add_name("double");

        // main: push 21, call double, result on stack
        b.emit(Op::LoadInt(21), 1);
        b.emit(Op::Call(double_name, 1), 1);
        let end_jump = b.emit(Op::Jump(0), 1); // jump past function body

        // double: arg * 2
        let double_ip = b.current_pos();
        b.add_sub_entry(double_name, double_ip);
        b.emit(Op::LoadInt(2), 2);
        b.emit(Op::Mul, 2);
        b.emit(Op::ReturnValue, 2);

        b.patch_jump(end_jump, b.current_pos());

        let mut vm = VM::new(b.build());
        match vm.run() {
            VMResult::Ok(Value::Int(42)) => {}
            other => panic!("expected Int(42), got {:?}", other),
        }
    }

    #[test]
    fn test_builtin_cache() {
        let mut b = ChunkBuilder::new();
        b.emit(Op::LoadInt(10), 1);
        b.emit(Op::CallBuiltin(0, 1), 1);
        let mut vm = VM::new(b.build());
        vm.register_builtin(0, |vm, _argc| {
            let val = vm.pop();
            Value::Int(val.to_int() * 2)
        });
        match vm.run() {
            VMResult::Ok(Value::Int(20)) => {}
            other => panic!("expected Int(20), got {:?}", other),
        }
    }

    // ── helpers ──

    fn run_one(ops: Vec<Op>) -> VMResult {
        let mut b = ChunkBuilder::new();
        for op in ops {
            b.emit(op, 1);
        }
        VM::new(b.build()).run()
    }

    fn expect_int(ops: Vec<Op>, want: i64) {
        match run_one(ops) {
            VMResult::Ok(Value::Int(n)) => assert_eq!(n, want),
            other => panic!("expected Int({}), got {:?}", want, other),
        }
    }

    fn expect_bool(ops: Vec<Op>, want: bool) {
        match run_one(ops) {
            VMResult::Ok(Value::Bool(b)) => assert_eq!(b, want),
            other => panic!("expected Bool({}), got {:?}", want, other),
        }
    }

    // ── Arithmetic ──

    #[test]
    fn arithmetic_sub_mul_div_mod() {
        expect_int(vec![Op::LoadInt(20), Op::LoadInt(8), Op::Sub], 12);
        expect_int(vec![Op::LoadInt(6), Op::LoadInt(7), Op::Mul], 42);
        expect_int(vec![Op::LoadInt(20), Op::LoadInt(3), Op::Mod], 2);
        // Div returns Float for int operands (no truncating int division).
        match run_one(vec![Op::LoadInt(20), Op::LoadInt(5), Op::Div]) {
            VMResult::Ok(Value::Float(f)) => assert!((f - 4.0).abs() < 1e-9),
            VMResult::Ok(Value::Int(4)) => {} // tolerate either impl
            other => panic!("got {:?}", other),
        }
    }

    #[test]
    fn arithmetic_negate_and_inc_dec() {
        expect_int(vec![Op::LoadInt(5), Op::Negate], -5);
        expect_int(vec![Op::LoadInt(5), Op::Inc], 6);
        expect_int(vec![Op::LoadInt(5), Op::Dec], 4);
    }

    #[test]
    fn arithmetic_negate_preserves_float_zero_kind_and_sign() {
        // Interpreter reference for the JIT float-kind tests: a zero-valued
        // float operand stays Float (with the correct sign), never Int(0).
        match run_one(vec![Op::LoadFloat(-0.0), Op::Negate]) {
            VMResult::Ok(Value::Float(f)) => assert_eq!(f.to_bits(), 0.0f64.to_bits()),
            other => panic!("expected Float(0.0), got {:?}", other),
        }
        match run_one(vec![Op::LoadFloat(0.0), Op::Negate]) {
            VMResult::Ok(Value::Float(f)) => assert_eq!(f.to_bits(), (-0.0f64).to_bits()),
            other => panic!("expected Float(-0.0), got {:?}", other),
        }
        match run_one(vec![Op::LoadFloat(-0.0), Op::LoadInt(0), Op::Sub]) {
            VMResult::Ok(Value::Float(f)) => assert_eq!(f.to_bits(), (-0.0f64).to_bits()),
            other => panic!("expected Float(-0.0), got {:?}", other),
        }
    }

    #[test]
    fn arithmetic_pow_returns_float() {
        match run_one(vec![Op::LoadInt(3), Op::LoadInt(4), Op::Pow]) {
            VMResult::Ok(Value::Float(f)) => assert!((f - 81.0).abs() < 1e-9),
            VMResult::Ok(Value::Int(81)) => {} // tolerate either impl
            other => panic!("got {:?}", other),
        }
    }

    // ── Comparison ──

    #[test]
    fn num_comparisons_produce_booleans() {
        expect_bool(vec![Op::LoadInt(1), Op::LoadInt(1), Op::NumEq], true);
        expect_bool(vec![Op::LoadInt(1), Op::LoadInt(2), Op::NumEq], false);
        expect_bool(vec![Op::LoadInt(1), Op::LoadInt(2), Op::NumLt], true);
        expect_bool(vec![Op::LoadInt(1), Op::LoadInt(2), Op::NumGt], false);
        expect_bool(vec![Op::LoadInt(2), Op::LoadInt(2), Op::NumLe], true);
        expect_bool(vec![Op::LoadInt(2), Op::LoadInt(2), Op::NumGe], true);
        expect_bool(vec![Op::LoadInt(2), Op::LoadInt(2), Op::NumNe], false);
    }

    #[test]
    fn spaceship_returns_neg_zero_pos() {
        expect_int(vec![Op::LoadInt(1), Op::LoadInt(2), Op::Spaceship], -1);
        expect_int(vec![Op::LoadInt(2), Op::LoadInt(2), Op::Spaceship], 0);
        expect_int(vec![Op::LoadInt(3), Op::LoadInt(2), Op::Spaceship], 1);
    }

    #[test]
    fn string_comparisons() {
        let mut b = ChunkBuilder::new();
        let a = b.add_constant(Value::str("alpha"));
        let z = b.add_constant(Value::str("beta"));
        b.emit(Op::LoadConst(a), 1);
        b.emit(Op::LoadConst(z), 1);
        b.emit(Op::StrLt, 1);
        let mut vm = VM::new(b.build());
        assert!(matches!(vm.run(), VMResult::Ok(Value::Bool(true))));
    }

    #[test]
    fn string_eq_and_ne() {
        let mut b = ChunkBuilder::new();
        let s1 = b.add_constant(Value::str("hi"));
        let s2 = b.add_constant(Value::str("hi"));
        b.emit(Op::LoadConst(s1), 1);
        b.emit(Op::LoadConst(s2), 1);
        b.emit(Op::StrEq, 1);
        assert!(matches!(
            VM::new(b.build()).run(),
            VMResult::Ok(Value::Bool(true))
        ));
    }

    // ── Stack manipulation ──

    #[test]
    fn pop_discards_top() {
        // push 1, push 2, pop, → result 1
        expect_int(vec![Op::LoadInt(1), Op::LoadInt(2), Op::Pop], 1);
    }

    #[test]
    fn dup_duplicates_top() {
        // 5, dup, add → 10
        expect_int(vec![Op::LoadInt(5), Op::Dup, Op::Add], 10);
    }

    #[test]
    fn swap_exchanges_top_two() {
        // 10, 3, swap, sub → 10 - 3 = 7 (after swap top is 10, next is 3 → 3 - 10 = -7?)
        // Sub semantics: pops b then a, returns a - b. After swap, top=10, below=3.
        // Pop b=10, a=3 → 3 - 10 = -7.
        expect_int(vec![Op::LoadInt(10), Op::LoadInt(3), Op::Swap, Op::Sub], -7);
    }

    #[test]
    fn dup2_duplicates_top_two_values() {
        // Dup2 on [3,4] yields [3,4,3,4]. Two Adds collapse to 11 on top.
        expect_int(
            vec![Op::LoadInt(3), Op::LoadInt(4), Op::Dup2, Op::Add, Op::Add],
            11,
        );
    }

    // ── Logical / Bitwise ──

    #[test]
    fn log_not_inverts_truthiness() {
        expect_bool(vec![Op::LoadInt(0), Op::LogNot], true);
        expect_bool(vec![Op::LoadInt(1), Op::LogNot], false);
        expect_bool(vec![Op::LoadTrue, Op::LogNot], false);
        expect_bool(vec![Op::LoadFalse, Op::LogNot], true);
    }

    #[test]
    fn bitwise_ops() {
        expect_int(
            vec![Op::LoadInt(0b1100), Op::LoadInt(0b1010), Op::BitAnd],
            0b1000,
        );
        expect_int(
            vec![Op::LoadInt(0b1100), Op::LoadInt(0b1010), Op::BitOr],
            0b1110,
        );
        expect_int(
            vec![Op::LoadInt(0b1100), Op::LoadInt(0b1010), Op::BitXor],
            0b0110,
        );
        expect_int(vec![Op::LoadInt(1), Op::LoadInt(4), Op::Shl], 16);
        expect_int(vec![Op::LoadInt(64), Op::LoadInt(2), Op::Shr], 16);
    }

    #[test]
    fn bit_not_inverts_bits() {
        expect_int(vec![Op::LoadInt(0), Op::BitNot], -1);
    }

    // ── Strings ──

    #[test]
    fn concat_joins_strings() {
        let mut b = ChunkBuilder::new();
        let h = b.add_constant(Value::str("hello "));
        let w = b.add_constant(Value::str("world"));
        b.emit(Op::LoadConst(h), 1);
        b.emit(Op::LoadConst(w), 1);
        b.emit(Op::Concat, 1);
        match VM::new(b.build()).run() {
            VMResult::Ok(v) => assert_eq!(v.to_str(), "hello world"),
            other => panic!("got {:?}", other),
        }
    }

    #[test]
    fn string_repeat_op() {
        let mut b = ChunkBuilder::new();
        let s = b.add_constant(Value::str("ab"));
        b.emit(Op::LoadConst(s), 1);
        b.emit(Op::LoadInt(3), 1);
        b.emit(Op::StringRepeat, 1);
        match VM::new(b.build()).run() {
            VMResult::Ok(v) => assert_eq!(v.to_str(), "ababab"),
            other => panic!("got {:?}", other),
        }
    }

    #[test]
    fn string_len_returns_int() {
        let mut b = ChunkBuilder::new();
        let s = b.add_constant(Value::str("abcd"));
        b.emit(Op::LoadConst(s), 1);
        b.emit(Op::StringLen, 1);
        match VM::new(b.build()).run() {
            VMResult::Ok(Value::Int(4)) => {}
            other => panic!("got {:?}", other),
        }
    }

    // ── Constants & literals ──

    #[test]
    fn load_true_false_undef() {
        assert!(matches!(
            run_one(vec![Op::LoadTrue]),
            VMResult::Ok(Value::Bool(true))
        ));
        assert!(matches!(
            run_one(vec![Op::LoadFalse]),
            VMResult::Ok(Value::Bool(false))
        ));
        assert!(matches!(
            run_one(vec![Op::LoadUndef]),
            VMResult::Ok(Value::Undef)
        ));
    }

    #[test]
    fn load_const_string() {
        let mut b = ChunkBuilder::new();
        let c = b.add_constant(Value::str("xyz"));
        b.emit(Op::LoadConst(c), 1);
        match VM::new(b.build()).run() {
            VMResult::Ok(v) => assert_eq!(v.to_str(), "xyz"),
            other => panic!("got {:?}", other),
        }
    }

    // ── Control flow ──

    #[test]
    fn jump_if_true_taken() {
        // load true; JumpIfTrue past "load 0"; load 1 → 1
        let mut b = ChunkBuilder::new();
        b.emit(Op::LoadTrue, 1);
        let j = b.emit(Op::JumpIfTrue(0), 1);
        b.emit(Op::LoadInt(0), 1);
        b.patch_jump(j, b.current_pos());
        b.emit(Op::LoadInt(1), 1);
        assert!(matches!(
            VM::new(b.build()).run(),
            VMResult::Ok(Value::Int(1))
        ));
    }

    #[test]
    fn jump_if_false_not_taken_for_true() {
        let mut b = ChunkBuilder::new();
        b.emit(Op::LoadTrue, 1);
        let j = b.emit(Op::JumpIfFalse(0), 1);
        b.emit(Op::LoadInt(7), 1); // executed
        b.patch_jump(j, b.current_pos());
        match VM::new(b.build()).run() {
            VMResult::Ok(Value::Int(7)) => {}
            other => panic!("got {:?}", other),
        }
    }

    // ── Frame / scope ──

    #[test]
    fn push_pop_frame_with_slots() {
        // PushFrame creates a new frame with its own slot table; GetSlot reads
        // back what SetSlot wrote. We omit PopFrame to keep the result on the
        // stack at end-of-chunk.
        let mut b = ChunkBuilder::new();
        b.emit(Op::PushFrame, 1);
        b.emit(Op::LoadInt(99), 1);
        b.emit(Op::SetSlot(0), 1);
        b.emit(Op::GetSlot(0), 1);
        match VM::new(b.build()).run() {
            VMResult::Ok(Value::Int(99)) => {}
            other => panic!("got {:?}", other),
        }
    }

    // ── Public VM helpers: push/pop/peek ──

    #[test]
    fn vm_push_pop_peek_round_trip() {
        let chunk = ChunkBuilder::new().build();
        let mut vm = VM::new(chunk);
        vm.push(Value::Int(1));
        vm.push(Value::Int(2));
        assert_eq!(*vm.peek(), Value::Int(2));
        assert_eq!(vm.pop(), Value::Int(2));
        assert_eq!(vm.pop(), Value::Int(1));
    }

    // ── register_builtin: overwrite + grow ──

    #[test]
    fn register_builtin_overwrites_existing_handler() {
        let mut b = ChunkBuilder::new();
        b.emit(Op::LoadInt(1), 1);
        b.emit(Op::CallBuiltin(0, 1), 1);
        let mut vm = VM::new(b.build());
        vm.register_builtin(0, |vm, _| {
            vm.pop();
            Value::Int(111)
        });
        // Overwrite with a different handler before run.
        vm.register_builtin(0, |vm, _| {
            vm.pop();
            Value::Int(222)
        });
        assert!(matches!(vm.run(), VMResult::Ok(Value::Int(222))));
    }

    #[test]
    fn run_builtin_by_name_dispatches_and_passes_args() {
        // A handler registered at a real builtin id must be reachable by NAME
        // at runtime, receiving its args in argument order.
        let id = crate::shell_builtins::builtin_id("true").expect("`true` is a builtin");
        let mut vm = VM::new(ChunkBuilder::new().build());
        vm.register_builtin(id, |vm, argc| {
            // args pushed in order → popped-and-reversed back to order.
            let mut got = Vec::new();
            for _ in 0..argc {
                got.push(vm.pop().to_str());
            }
            got.reverse();
            assert_eq!(got, vec!["x".to_string(), "y".to_string()]);
            Value::Int(7)
        });
        let out = vm.run_builtin_by_name("true", &["x".to_string(), "y".to_string()]);
        assert!(matches!(out, Some(Value::Int(7))));
        // Unknown / unregistered names return None (caller falls through).
        assert!(vm
            .run_builtin_by_name("definitely_not_a_builtin_xyz", &[])
            .is_none());
    }

    #[test]
    fn register_builtin_grows_table_to_high_id() {
        // High id should expand the builtin_table to accommodate.
        let chunk = ChunkBuilder::new().build();
        let mut vm = VM::new(chunk);
        vm.register_builtin(500, |_, _| Value::Int(0));
        // Indirect proof: re-registering at lower id still works (no panic).
        vm.register_builtin(1, |_, _| Value::Int(0));
    }

    // ── reset() ──

    #[test]
    fn reset_clears_state_and_runs_new_chunk() {
        let mut b1 = ChunkBuilder::new();
        b1.emit(Op::LoadInt(1), 1);
        let mut vm = VM::new(b1.build());
        assert!(matches!(vm.run(), VMResult::Ok(Value::Int(1))));

        let mut b2 = ChunkBuilder::new();
        b2.emit(Op::LoadInt(2), 1);
        b2.emit(Op::LoadInt(3), 1);
        b2.emit(Op::Add, 1);
        vm.reset(b2.build());
        assert!(matches!(vm.run(), VMResult::Ok(Value::Int(5))));
    }

    // ── Extension handler ──

    #[test]
    fn extension_handler_invoked_with_payload() {
        use std::sync::{Arc, Mutex};
        let captured: Arc<Mutex<Option<(u16, u8)>>> = Arc::new(Mutex::new(None));
        let captured_cl = Arc::clone(&captured);

        let mut b = ChunkBuilder::new();
        b.emit(Op::Extended(7, 42), 1);
        let mut vm = VM::new(b.build());
        vm.set_extension_handler(Box::new(move |vm, id, arg| {
            *captured_cl.lock().unwrap() = Some((id, arg));
            vm.push(Value::Int(123));
        }));
        match vm.run() {
            VMResult::Ok(Value::Int(123)) => {}
            other => panic!("got {:?}", other),
        }
        assert_eq!(*captured.lock().unwrap(), Some((7, 42)));
    }

    #[test]
    fn extension_wide_handler_invoked_with_payload() {
        use std::sync::{Arc, Mutex};
        let captured: Arc<Mutex<Option<(u16, usize)>>> = Arc::new(Mutex::new(None));
        let captured_cl = Arc::clone(&captured);
        let mut b = ChunkBuilder::new();
        b.emit(Op::ExtendedWide(9, 9999), 1);
        let mut vm = VM::new(b.build());
        vm.set_extension_wide_handler(Box::new(move |vm, id, payload| {
            *captured_cl.lock().unwrap() = Some((id, payload));
            vm.push(Value::Int(0));
        }));
        let _ = vm.run();
        assert_eq!(*captured.lock().unwrap(), Some((9, 9999)));
    }

    // ── VMPool ──

    #[test]
    fn vmpool_new_default_and_with_capacity_start_empty() {
        let p = VMPool::new();
        assert!(p.is_empty());
        assert_eq!(p.len(), 0);
        let p = VMPool::with_capacity(8);
        assert!(p.is_empty());
        let p: VMPool = Default::default();
        assert!(p.is_empty());
    }

    #[test]
    fn vmpool_release_then_acquire_reuses_vm() {
        let mut pool = VMPool::new();
        let chunk1 = {
            let mut b = ChunkBuilder::new();
            b.emit(Op::LoadInt(1), 1);
            b.build()
        };
        let vm = pool.acquire(chunk1);
        assert_eq!(pool.len(), 0);
        pool.release(vm);
        assert_eq!(pool.len(), 1);

        let chunk2 = {
            let mut b = ChunkBuilder::new();
            b.emit(Op::LoadInt(2), 1);
            b.build()
        };
        let mut vm = pool.acquire(chunk2);
        assert_eq!(pool.len(), 0);
        assert!(matches!(vm.run(), VMResult::Ok(Value::Int(2))));
    }

    #[test]
    fn vmpool_with_returns_value_and_recycles_vm() {
        let mut pool = VMPool::new();
        let chunk = {
            let mut b = ChunkBuilder::new();
            b.emit(Op::LoadInt(10), 1);
            b.emit(Op::LoadInt(5), 1);
            b.emit(Op::Add, 1);
            b.build()
        };
        let result = pool.with(chunk, |vm| match vm.run() {
            VMResult::Ok(Value::Int(n)) => n,
            other => panic!("got {:?}", other),
        });
        assert_eq!(result, 15);
        assert_eq!(pool.len(), 1, "VM should be returned to pool after with()");
    }

    // ── AWK host dispatch ──────────────────────────────────────────────────

    /// Recording AWK host: captures every routed call so tests can assert the
    /// VM popped/pushed the right operands and dispatched to the right method.
    #[derive(Default)]
    struct RecordingAwkHost {
        record: String,
        fields: Vec<String>,
        printed: Vec<Vec<String>>,
        field_sets: Vec<(i64, String)>,
        special_sets: Vec<(String, String)>,
        array: std::collections::HashMap<String, String>,
    }

    impl crate::awk_host::AwkHost for RecordingAwkHost {
        fn field_get(&mut self, i: i64) -> Value {
            Value::str(self.fields.get(i as usize).cloned().unwrap_or_default())
        }
        fn field_set(&mut self, i: i64, v: Value) {
            self.field_sets.push((i, v.to_str()));
        }
        fn nf(&mut self) -> i64 {
            self.fields.len() as i64
        }
        fn set_record(&mut self, v: Value) {
            self.record = v.to_str();
            self.fields = self.record.split(' ').map(|s| s.to_string()).collect();
        }
        fn special_get(&mut self, name: &str) -> Value {
            match name {
                "NR" => Value::Int(7),
                _ => Value::str(""),
            }
        }
        fn special_set(&mut self, name: &str, v: Value) {
            self.special_sets.push((name.to_string(), v.to_str()));
        }
        fn print(&mut self, args: &[Value]) {
            self.printed.push(args.iter().map(|v| v.to_str()).collect());
        }
        fn array_get(&mut self, _arr: &str, key: &Value) -> Value {
            Value::str(self.array.get(&key.to_str()).cloned().unwrap_or_default())
        }
        fn array_set(&mut self, _arr: &str, key: &Value, v: Value) {
            self.array.insert(key.to_str(), v.to_str());
        }
    }

    fn awk_op(b: &mut ChunkBuilder, id: u16, payload: usize) {
        b.emit(Op::ExtendedWide(id, payload), 1);
    }

    #[test]
    fn awk_field_get_routes_to_host() {
        use crate::awk_builtins::*;
        let chunk = {
            let mut b = ChunkBuilder::new();
            b.emit(Op::LoadInt(2), 1); // $2
            awk_op(&mut b, AWK_FIELD_GET, 0);
            b.build()
        };
        let mut vm = VM::new(chunk);
        let host = RecordingAwkHost {
            fields: vec!["a".into(), "b".into(), "c".into()],
            ..Default::default()
        };
        vm.set_awk_host(Box::new(host));
        match vm.run() {
            VMResult::Ok(v) => assert_eq!(v.to_str(), "c"),
            other => panic!("got {:?}", other),
        }
    }

    #[test]
    fn awk_print_pops_args_in_source_order() {
        use crate::awk_builtins::*;
        let chunk = {
            let mut b = ChunkBuilder::new();
            let x = b.add_constant(Value::str("x"));
            let y = b.add_constant(Value::str("y"));
            b.emit(Op::LoadConst(x), 1);
            b.emit(Op::LoadConst(y), 1);
            awk_op(&mut b, AWK_PRINT, 2);
            b.build()
        };
        let mut vm = VM::new(chunk);
        // Use a shared host we can inspect after the run.
        struct H(std::sync::Arc<std::sync::Mutex<Vec<Vec<String>>>>);
        impl crate::awk_host::AwkHost for H {
            fn print(&mut self, args: &[Value]) {
                self.0
                    .lock()
                    .unwrap()
                    .push(args.iter().map(|v| v.to_str()).collect());
            }
        }
        let sink = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        vm.set_awk_host(Box::new(H(sink.clone())));
        let _ = vm.run();
        assert_eq!(
            sink.lock().unwrap().as_slice(),
            &[vec!["x".to_string(), "y".to_string()]]
        );
    }

    #[test]
    fn awk_field_set_pops_value_and_index() {
        use crate::awk_builtins::*;
        let chunk = {
            let mut b = ChunkBuilder::new();
            let v = b.add_constant(Value::str("Z"));
            b.emit(Op::LoadConst(v), 1); // value
            b.emit(Op::LoadInt(3), 1); // index
            awk_op(&mut b, AWK_FIELD_SET, 0);
            b.build()
        };
        struct H(std::sync::Arc<std::sync::Mutex<Vec<(i64, String)>>>);
        impl crate::awk_host::AwkHost for H {
            fn field_set(&mut self, i: i64, v: Value) {
                self.0.lock().unwrap().push((i, v.to_str()));
            }
        }
        let sink = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let mut vm = VM::new(chunk);
        vm.set_awk_host(Box::new(H(sink.clone())));
        let _ = vm.run();
        assert_eq!(sink.lock().unwrap().as_slice(), &[(3i64, "Z".to_string())]);
    }

    #[test]
    fn awk_special_get_and_array_roundtrip() {
        use crate::awk_builtins::*;
        let chunk = {
            let mut b = ChunkBuilder::new();
            let arr = b.add_name("counts");
            let k = b.add_constant(Value::str("k"));
            let val = b.add_constant(Value::str("42"));
            // counts["k"] = "42"
            b.emit(Op::LoadConst(val), 1);
            b.emit(Op::LoadConst(k), 1);
            awk_op(&mut b, AWK_ARRAY_SET, arr as usize);
            // push counts["k"] then NR
            b.emit(Op::LoadConst(k), 1);
            awk_op(&mut b, AWK_ARRAY_GET, arr as usize);
            b.build()
        };
        let mut vm = VM::new(chunk);
        vm.set_awk_host(Box::new(RecordingAwkHost::default()));
        match vm.run() {
            VMResult::Ok(v) => assert_eq!(v.to_str(), "42"),
            other => panic!("got {:?}", other),
        }
    }

    #[test]
    fn awk_ops_are_inert_without_host_but_keep_stack_balanced() {
        use crate::awk_builtins::*;
        // $1 with no host → pushes "" (stack stays balanced, run returns it).
        let chunk = {
            let mut b = ChunkBuilder::new();
            b.emit(Op::LoadInt(1), 1);
            awk_op(&mut b, AWK_FIELD_GET, 0);
            b.build()
        };
        let mut vm = VM::new(chunk);
        match vm.run() {
            VMResult::Ok(v) => assert_eq!(v.to_str(), ""),
            other => panic!("got {:?}", other),
        }
    }

    // ── First-class AWK ops (Op::Awk*) ──
    // These mirror the shell-ops design: named Op variants dispatched to the
    // same AwkHost path as the reserved ExtendedWide AWK range. The tests below
    // prove each first-class variant routes to the host identically to its
    // `ExtendedWide(AWK_*)` form.

    #[test]
    fn first_class_awk_field_get_routes_to_host() {
        let chunk = {
            let mut b = ChunkBuilder::new();
            b.emit(Op::LoadInt(2), 1); // $2
            b.emit(Op::AwkFieldGet, 1);
            b.build()
        };
        let mut vm = VM::new(chunk);
        let host = RecordingAwkHost {
            fields: vec!["a".into(), "b".into(), "c".into()],
            ..Default::default()
        };
        vm.set_awk_host(Box::new(host));
        match vm.run() {
            VMResult::Ok(v) => assert_eq!(v.to_str(), "c"),
            other => panic!("got {:?}", other),
        }
    }

    #[test]
    fn first_class_awk_print_pops_args_in_source_order() {
        struct H(std::sync::Arc<std::sync::Mutex<Vec<Vec<String>>>>);
        impl crate::awk_host::AwkHost for H {
            fn print(&mut self, args: &[Value]) {
                self.0
                    .lock()
                    .unwrap()
                    .push(args.iter().map(|v| v.to_str()).collect());
            }
        }
        let chunk = {
            let mut b = ChunkBuilder::new();
            let x = b.add_constant(Value::str("x"));
            let y = b.add_constant(Value::str("y"));
            b.emit(Op::LoadConst(x), 1);
            b.emit(Op::LoadConst(y), 1);
            b.emit(Op::AwkPrint(2), 1);
            b.build()
        };
        let mut vm = VM::new(chunk);
        let sink = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        vm.set_awk_host(Box::new(H(sink.clone())));
        let _ = vm.run();
        assert_eq!(
            sink.lock().unwrap().as_slice(),
            &[vec!["x".to_string(), "y".to_string()]]
        );
    }

    #[test]
    fn first_class_awk_array_roundtrip_matches_extendedwide() {
        // counts["k"]="42"; counts["k"] — once via Op::AwkArray*, once via the
        // ExtendedWide form; both must yield the same value.
        fn run_variant(first_class: bool) -> String {
            use crate::awk_builtins::*;
            let mut b = ChunkBuilder::new();
            let arr = b.add_name("counts");
            let k = b.add_constant(Value::str("k"));
            let val = b.add_constant(Value::str("42"));
            b.emit(Op::LoadConst(val), 1);
            b.emit(Op::LoadConst(k), 1);
            if first_class {
                b.emit(Op::AwkArraySet(arr), 1);
            } else {
                b.emit(Op::ExtendedWide(AWK_ARRAY_SET, arr as usize), 1);
            }
            b.emit(Op::LoadConst(k), 1);
            if first_class {
                b.emit(Op::AwkArrayGet(arr), 1);
            } else {
                b.emit(Op::ExtendedWide(AWK_ARRAY_GET, arr as usize), 1);
            }
            let mut vm = VM::new(b.build());
            vm.set_awk_host(Box::new(RecordingAwkHost::default()));
            match vm.run() {
                VMResult::Ok(v) => v.to_str(),
                other => panic!("got {:?}", other),
            }
        }
        assert_eq!(run_variant(true), "42");
        assert_eq!(run_variant(true), run_variant(false));
    }

    #[test]
    fn first_class_awk_ops_inert_without_host() {
        // $1 (Op::AwkFieldGet) with no host → pushes "" and stays balanced.
        let chunk = {
            let mut b = ChunkBuilder::new();
            b.emit(Op::LoadInt(1), 1);
            b.emit(Op::AwkFieldGet, 1);
            b.build()
        };
        let mut vm = VM::new(chunk);
        match vm.run() {
            VMResult::Ok(v) => assert_eq!(v.to_str(), ""),
            other => panic!("got {:?}", other),
        }
    }

    // ── Host-independent AWK string builtins execute natively (no host) ──
    // `substr`/`tolower`/`toupper`/`index`/`length(s)` need none of AWK's
    // host-side runtime state, so they compute real results even when no
    // `AwkHost` is registered (unlike field/array/print ops, which stay inert).

    fn run_native(chunk: crate::chunk::Chunk) -> Value {
        let mut vm = VM::new(chunk);
        match vm.run() {
            VMResult::Ok(v) => v,
            other => panic!("got {:?}", other),
        }
    }

    #[test]
    fn awk_substr_executes_natively_without_host() {
        // substr("hello", 2, 3) → "ell"
        let chunk = {
            let mut b = ChunkBuilder::new();
            let s = b.add_constant(Value::str("hello"));
            b.emit(Op::LoadConst(s), 1);
            b.emit(Op::LoadInt(2), 1);
            b.emit(Op::LoadInt(3), 1);
            b.emit(Op::AwkSubstr(3), 1);
            b.build()
        };
        assert_eq!(run_native(chunk).to_str(), "ell");
    }

    #[test]
    fn awk_substr_two_arg_to_end_without_host() {
        // substr("hello", 2) → "ello"
        let chunk = {
            let mut b = ChunkBuilder::new();
            let s = b.add_constant(Value::str("hello"));
            b.emit(Op::LoadConst(s), 1);
            b.emit(Op::LoadInt(2), 1);
            b.emit(Op::AwkSubstr(2), 1);
            b.build()
        };
        assert_eq!(run_native(chunk).to_str(), "ello");
    }

    #[test]
    fn awk_tolower_toupper_execute_natively_without_host() {
        let lower = {
            let mut b = ChunkBuilder::new();
            let s = b.add_constant(Value::str("MiXeD"));
            b.emit(Op::LoadConst(s), 1);
            b.emit(Op::AwkToLower, 1);
            b.build()
        };
        assert_eq!(run_native(lower).to_str(), "mixed");
        let upper = {
            let mut b = ChunkBuilder::new();
            let s = b.add_constant(Value::str("MiXeD"));
            b.emit(Op::LoadConst(s), 1);
            b.emit(Op::AwkToUpper, 1);
            b.build()
        };
        assert_eq!(run_native(upper).to_str(), "MIXED");
    }

    #[test]
    fn awk_index_executes_natively_without_host() {
        // index("hello", "ll") → 3
        let chunk = {
            let mut b = ChunkBuilder::new();
            let s = b.add_constant(Value::str("hello"));
            let t = b.add_constant(Value::str("ll"));
            b.emit(Op::LoadConst(s), 1);
            b.emit(Op::LoadConst(t), 1);
            b.emit(Op::AwkIndex, 1);
            b.build()
        };
        assert_eq!(run_native(chunk).to_int(), 3);
    }

    #[test]
    fn awk_length_scalar_executes_natively_without_host() {
        // length("héllo") → 5 chars (not bytes)
        let chunk = {
            let mut b = ChunkBuilder::new();
            let s = b.add_constant(Value::str("héllo"));
            b.emit(Op::LoadConst(s), 1);
            b.emit(Op::AwkLength(1), 1);
            b.build()
        };
        assert_eq!(run_native(chunk).to_int(), 5);
    }

    #[test]
    fn awk_native_string_ops_match_host_path() {
        // The no-host native result must equal the DefaultAwkHost result.
        use crate::awk_host::{awk_index, awk_substr, awk_tolower};
        assert_eq!(awk_substr(&Value::str("hello"), 2, Some(3)).to_str(), "ell");
        assert_eq!(awk_index(&Value::str("hello"), &Value::str("z")), 0);
        assert_eq!(awk_tolower(&Value::str("ABC")).to_str(), "abc");
    }

    // ── Host-independent AWK numeric builtins execute natively (no host) ──
    // int/sqrt/sin/cos/exp/log/atan2 are pure f64 math with no AWK runtime
    // state, so they compute real results with no `AwkHost` registered.

    #[test]
    fn awk_int_truncates_toward_zero_without_host() {
        for (input, want) in [(3.7_f64, 3_i64), (-3.7, -3), (0.0, 0)] {
            let chunk = {
                let mut b = ChunkBuilder::new();
                let x = b.add_constant(Value::Float(input));
                b.emit(Op::LoadConst(x), 1);
                b.emit(Op::AwkInt, 1);
                b.build()
            };
            assert_eq!(run_native(chunk).to_int(), want, "int({input})");
        }
    }

    #[test]
    fn awk_sqrt_exp_log_execute_natively_without_host() {
        let sqrt = {
            let mut b = ChunkBuilder::new();
            let x = b.add_constant(Value::Float(16.0));
            b.emit(Op::LoadConst(x), 1);
            b.emit(Op::AwkSqrt, 1);
            b.build()
        };
        assert_eq!(run_native(sqrt).to_float(), 4.0);
        let log = {
            let mut b = ChunkBuilder::new();
            let x = b.add_constant(Value::Float(std::f64::consts::E));
            b.emit(Op::LoadConst(x), 1);
            b.emit(Op::AwkLog, 1);
            b.build()
        };
        assert!((run_native(log).to_float() - 1.0).abs() < 1e-12);
        let exp = {
            let mut b = ChunkBuilder::new();
            let x = b.add_constant(Value::Float(0.0));
            b.emit(Op::LoadConst(x), 1);
            b.emit(Op::AwkExp, 1);
            b.build()
        };
        assert_eq!(run_native(exp).to_float(), 1.0);
    }

    #[test]
    fn awk_sin_cos_execute_natively_without_host() {
        let sin = {
            let mut b = ChunkBuilder::new();
            let x = b.add_constant(Value::Float(0.0));
            b.emit(Op::LoadConst(x), 1);
            b.emit(Op::AwkSin, 1);
            b.build()
        };
        assert_eq!(run_native(sin).to_float(), 0.0);
        let cos = {
            let mut b = ChunkBuilder::new();
            let x = b.add_constant(Value::Float(0.0));
            b.emit(Op::LoadConst(x), 1);
            b.emit(Op::AwkCos, 1);
            b.build()
        };
        assert_eq!(run_native(cos).to_float(), 1.0);
    }

    #[test]
    fn awk_atan2_pops_y_then_x_without_host() {
        // atan2(1, 1) == π/4. Stack order is [y, x] (y pushed first).
        let chunk = {
            let mut b = ChunkBuilder::new();
            let y = b.add_constant(Value::Float(1.0));
            let x = b.add_constant(Value::Float(1.0));
            b.emit(Op::LoadConst(y), 1);
            b.emit(Op::LoadConst(x), 1);
            b.emit(Op::AwkAtan2, 1);
            b.build()
        };
        assert!((run_native(chunk).to_float() - std::f64::consts::FRAC_PI_4).abs() < 1e-12);
    }

    // ── Host-independent AWK bitwise builtins execute natively (no host) ──
    // gawk and/or/xor/compl/lshift/rshift are pure integer math (operands
    // truncated to u64), ported faithfully from awkrs's f64 path.

    fn run_native_int(chunk: crate::chunk::Chunk) -> i64 {
        run_native(chunk).to_int()
    }

    #[test]
    fn awk_and_or_xor_execute_natively_without_host() {
        // and(12,10)=8, or(12,10)=14, xor(12,10)=6
        let mk = |op: Op| {
            let mut b = ChunkBuilder::new();
            b.emit(Op::LoadInt(12), 1);
            b.emit(Op::LoadInt(10), 1);
            b.emit(op, 1);
            b.build()
        };
        assert_eq!(run_native_int(mk(Op::AwkAnd(2))), 8);
        assert_eq!(run_native_int(mk(Op::AwkOr(2))), 14);
        assert_eq!(run_native_int(mk(Op::AwkXor(2))), 6);
    }

    #[test]
    fn awk_and_is_variadic_without_host() {
        // and(15, 9, 5) → 15&9=9, 9&5=1
        let chunk = {
            let mut b = ChunkBuilder::new();
            b.emit(Op::LoadInt(15), 1);
            b.emit(Op::LoadInt(9), 1);
            b.emit(Op::LoadInt(5), 1);
            b.emit(Op::AwkAnd(3), 1);
            b.build()
        };
        assert_eq!(run_native_int(chunk), 1);
    }

    #[test]
    fn awk_compl_matches_awkrs_i64_wrap_without_host() {
        // awkrs f64 path: compl(0) = (!0u64) as i64 = -1.
        let chunk = {
            let mut b = ChunkBuilder::new();
            b.emit(Op::LoadInt(0), 1);
            b.emit(Op::AwkCompl, 1);
            b.build()
        };
        assert_eq!(run_native_int(chunk), -1);
    }

    #[test]
    fn awk_lshift_rshift_execute_natively_without_host() {
        // lshift(1,4)=16; rshift(256,4)=16. Stack order is [v, n].
        let lshift = {
            let mut b = ChunkBuilder::new();
            b.emit(Op::LoadInt(1), 1);
            b.emit(Op::LoadInt(4), 1);
            b.emit(Op::AwkLshift, 1);
            b.build()
        };
        assert_eq!(run_native_int(lshift), 16);
        let rshift = {
            let mut b = ChunkBuilder::new();
            b.emit(Op::LoadInt(256), 1);
            b.emit(Op::LoadInt(4), 1);
            b.emit(Op::AwkRshift, 1);
            b.build()
        };
        assert_eq!(run_native_int(rshift), 16);
    }

    #[test]
    fn awk_bitwise_free_fns_match_gawk_semantics() {
        use crate::awk_host::{awk_compl, awk_fold_and, awk_fold_or, awk_fold_xor, awk_lshift};
        assert_eq!(awk_fold_and(&[Value::Int(12), Value::Int(10)]), 8);
        assert_eq!(awk_fold_or(&[Value::Int(12), Value::Int(10)]), 14);
        assert_eq!(awk_fold_xor(&[Value::Int(12), Value::Int(10)]), 6);
        assert_eq!(awk_compl(&Value::Int(0)), -1);
        // shift count is masked to low 6 bits (n & 0x3f)
        assert_eq!(awk_lshift(&Value::Int(1), &Value::Int(64)), 1);
    }

    #[test]
    fn awk_mktime_utc_executes_natively_without_host() {
        // mktime("2020 01 01 00 00 00", 1) in UTC = 1577836800 epoch seconds.
        let chunk = {
            let mut b = ChunkBuilder::new();
            let s = b.add_constant(Value::str("2020 01 01 00 00 00"));
            b.emit(Op::LoadConst(s), 1);
            b.emit(Op::LoadInt(1), 1); // utc = true
            b.emit(Op::AwkMktime(2), 1);
            b.build()
        };
        assert_eq!(run_native(chunk).to_float(), 1_577_836_800.0);
    }

    #[test]
    fn awk_mktime_bad_datespec_returns_minus_one_without_host() {
        // Fewer than 6 fields → -1.
        let chunk = {
            let mut b = ChunkBuilder::new();
            let s = b.add_constant(Value::str("2020 01 01"));
            b.emit(Op::LoadConst(s), 1);
            b.emit(Op::AwkMktime(1), 1);
            b.build()
        };
        assert_eq!(run_native(chunk).to_float(), -1.0);
    }

    #[test]
    fn awk_strftime_utc_executes_natively_without_host() {
        // strftime("%Y-%m-%d", 0, 1) in UTC = "1970-01-01".
        let chunk = {
            let mut b = ChunkBuilder::new();
            let fmt = b.add_constant(Value::str("%Y-%m-%d"));
            b.emit(Op::LoadConst(fmt), 1);
            b.emit(Op::LoadInt(0), 1); // ts = epoch
            b.emit(Op::LoadInt(1), 1); // utc = true
            b.emit(Op::AwkStrftime(3), 1);
            b.build()
        };
        assert_eq!(run_native(chunk).to_str(), "1970-01-01");
    }

    #[test]
    fn awk_strftime_mktime_free_fns_match_awkrs() {
        use crate::awk_host::{awk_mktime, awk_strftime};
        // UTC round trip and -1 sentinel.
        assert_eq!(
            awk_mktime(&[Value::str("2020 01 01 00 00 00"), Value::Int(1)]).to_float(),
            1_577_836_800.0
        );
        assert_eq!(awk_mktime(&[Value::str("garbage")]).to_float(), -1.0);
        assert_eq!(
            awk_strftime(&[Value::str("%H:%M:%S"), Value::Int(0), Value::Int(1)]).to_str(),
            "00:00:00"
        );
    }

    #[test]
    fn awk_ord_executes_natively_without_host() {
        // ord("A") = 65; ord("") = 0.
        let chunk = {
            let mut b = ChunkBuilder::new();
            let s = b.add_constant(Value::str("ABC"));
            b.emit(Op::LoadConst(s), 1);
            b.emit(Op::AwkOrd, 1);
            b.build()
        };
        assert_eq!(run_native(chunk).to_float(), 65.0);

        let empty = {
            let mut b = ChunkBuilder::new();
            let s = b.add_constant(Value::str(""));
            b.emit(Op::LoadConst(s), 1);
            b.emit(Op::AwkOrd, 1);
            b.build()
        };
        assert_eq!(run_native(empty).to_float(), 0.0);
    }

    #[test]
    fn awk_chr_executes_natively_without_host() {
        // chr(65) = "A"; chr of an invalid scalar (surrogate) = "".
        let chunk = {
            let mut b = ChunkBuilder::new();
            b.emit(Op::LoadInt(65), 1);
            b.emit(Op::AwkChr, 1);
            b.build()
        };
        assert_eq!(run_native(chunk).to_str(), "A");

        let bad = {
            let mut b = ChunkBuilder::new();
            b.emit(Op::LoadInt(0xD800), 1); // lone surrogate → invalid
            b.emit(Op::AwkChr, 1);
            b.build()
        };
        assert_eq!(run_native(bad).to_str(), "");
    }

    #[test]
    fn awk_mkbool_executes_natively_without_host() {
        // mkbool(7) = 1; mkbool(0) = 0; mkbool("") = 0.
        let truthy = {
            let mut b = ChunkBuilder::new();
            b.emit(Op::LoadInt(7), 1);
            b.emit(Op::AwkMkbool, 1);
            b.build()
        };
        assert_eq!(run_native(truthy).to_float(), 1.0);

        let zero = {
            let mut b = ChunkBuilder::new();
            b.emit(Op::LoadInt(0), 1);
            b.emit(Op::AwkMkbool, 1);
            b.build()
        };
        assert_eq!(run_native(zero).to_float(), 0.0);
    }

    #[test]
    fn awk_intdiv_executes_natively_without_host() {
        // intdiv(17, 5) = 3 (truncating); intdiv(x, 0) = Undef.
        let chunk = {
            let mut b = ChunkBuilder::new();
            b.emit(Op::LoadInt(17), 1);
            b.emit(Op::LoadInt(5), 1);
            b.emit(Op::AwkIntdiv, 1);
            b.build()
        };
        assert_eq!(run_native(chunk).to_float(), 3.0);

        let div0 = {
            let mut b = ChunkBuilder::new();
            b.emit(Op::LoadInt(17), 1);
            b.emit(Op::LoadInt(0), 1);
            b.emit(Op::AwkIntdiv, 1);
            b.build()
        };
        assert!(matches!(run_native(div0), Value::Undef));
    }

    #[test]
    fn awk_intdiv0_executes_natively_without_host() {
        // intdiv0(17, 5) = 3; intdiv0(x, 0) = 0 (safe variant, never errors).
        let chunk = {
            let mut b = ChunkBuilder::new();
            b.emit(Op::LoadInt(17), 1);
            b.emit(Op::LoadInt(5), 1);
            b.emit(Op::AwkIntdiv0, 1);
            b.build()
        };
        assert_eq!(run_native(chunk).to_float(), 3.0);

        let div0 = {
            let mut b = ChunkBuilder::new();
            b.emit(Op::LoadInt(17), 1);
            b.emit(Op::LoadInt(0), 1);
            b.emit(Op::AwkIntdiv0, 1);
            b.build()
        };
        assert_eq!(run_native(div0).to_float(), 0.0);
    }

    #[test]
    fn awk_div_mod_compute_and_trap_on_zero() {
        // awk `a / b` and `a % b` compute the float result for a nonzero
        // divisor and raise the POSIX fatal runtime error on a zero divisor
        // (distinct from the shell-arithmetic Op::Div/Op::Mod which yield
        // Undef / 0). Pop order mirrors Op::Div: b is on top, a beneath.
        let div = {
            let mut b = ChunkBuilder::new();
            b.emit(Op::LoadFloat(7.0), 1);
            b.emit(Op::LoadFloat(2.0), 1);
            b.emit(Op::AwkDiv, 1);
            b.build()
        };
        assert_eq!(run_native(div).to_float(), 3.5);

        let md = {
            let mut b = ChunkBuilder::new();
            b.emit(Op::LoadFloat(7.0), 1);
            b.emit(Op::LoadFloat(3.0), 1);
            b.emit(Op::AwkMod, 1);
            b.build()
        };
        assert_eq!(run_native(md).to_float(), 1.0);

        let div0 = {
            let mut b = ChunkBuilder::new();
            b.emit(Op::LoadFloat(1.0), 1);
            b.emit(Op::LoadFloat(0.0), 1);
            b.emit(Op::AwkDiv, 1);
            b.build()
        };
        match VM::new(div0).run() {
            VMResult::Error(m) => assert_eq!(m, "division by zero attempted"),
            other => panic!("expected div-by-zero trap, got {:?}", other),
        }

        let mod0 = {
            let mut b = ChunkBuilder::new();
            b.emit(Op::LoadFloat(1.0), 1);
            b.emit(Op::LoadFloat(0.0), 1);
            b.emit(Op::AwkMod, 1);
            b.build()
        };
        match VM::new(mod0).run() {
            VMResult::Error(m) => assert_eq!(m, "division by zero attempted in `%'"),
            other => panic!("expected mod-by-zero trap, got {:?}", other),
        }
    }

    #[test]
    fn awk_signal_halts_chunk_and_records_code() {
        use crate::awk_builtins::signal;
        // `Op::AwkSignal(code)` halts the chunk immediately and stashes `code`
        // in the VM for the frontend driver to read; ops after it do not run.
        let chunk = {
            let mut b = ChunkBuilder::new();
            b.emit(Op::LoadInt(1), 1);
            b.emit(Op::AwkSignal(signal::NEXTFILE), 1);
            // Unreachable once the signal halts the chunk.
            b.emit(Op::LoadInt(99), 1);
            b.build()
        };
        let mut vm = VM::new(chunk);
        let r = vm.run();
        assert_eq!(vm.awk_signal(), Some(signal::NEXTFILE));
        // The pre-signal value remains on the stack; the post-signal LoadInt(99)
        // never executed (would have been the Ok value otherwise).
        match r {
            VMResult::Ok(v) => assert_eq!(v.to_float(), 1.0),
            other => panic!("expected Ok(1), got {:?}", other),
        }

        // A signal-free chunk leaves awk_signal None (zshrs/stryke behavior).
        let plain = {
            let mut b = ChunkBuilder::new();
            b.emit(Op::LoadInt(5), 1);
            b.build()
        };
        let mut vm2 = VM::new(plain);
        let _ = vm2.run();
        assert_eq!(vm2.awk_signal(), None);
    }

    #[test]
    fn awk_gensub_stub_is_stack_balanced_without_host() {
        // Host-bound op: with no AwkHost the stub pops all argc operands and
        // pushes a neutral empty string, keeping the stack balanced.
        let chunk = {
            let mut b = ChunkBuilder::new();
            let re = b.add_constant(Value::str("x"));
            let repl = b.add_constant(Value::str("y"));
            let how = b.add_constant(Value::str("g"));
            let target = b.add_constant(Value::str("xax"));
            b.emit(Op::LoadConst(re), 1);
            b.emit(Op::LoadConst(repl), 1);
            b.emit(Op::LoadConst(how), 1);
            b.emit(Op::LoadConst(target), 1);
            b.emit(Op::AwkGensub(4), 1);
            b.build()
        };
        assert_eq!(run_native(chunk).to_str(), "");
    }

    #[test]
    fn awk_char_scalar_free_fns_match_awkrs() {
        use crate::awk_host::{awk_chr, awk_intdiv, awk_intdiv0, awk_mkbool, awk_ord};
        assert_eq!(awk_ord(&Value::str("z")).to_float(), 122.0);
        assert_eq!(awk_chr(&Value::Int(0x1F600)).to_str(), "😀");
        assert_eq!(awk_mkbool(&Value::str("0")).to_float(), 0.0);
        assert_eq!(awk_mkbool(&Value::str("x")).to_float(), 1.0);
        assert_eq!(awk_intdiv(&Value::Int(-7), &Value::Int(2)).to_float(), -3.0);
        assert!(matches!(
            awk_intdiv(&Value::Int(1), &Value::Int(0)),
            Value::Undef
        ));
        assert_eq!(
            awk_intdiv0(&Value::Int(-7), &Value::Int(2)).to_float(),
            -3.0
        );
        assert_eq!(awk_intdiv0(&Value::Int(1), &Value::Int(0)).to_float(), 0.0);
    }

    #[test]
    fn awk_rand_executes_natively_without_host() {
        // Fresh VM seeds rand_seed=1; first rand() is deterministic.
        let chunk = {
            let mut b = ChunkBuilder::new();
            b.emit(Op::AwkRand, 1);
            b.build()
        };
        let v = run_native(chunk).to_float();
        assert!((0.0..1.0).contains(&v), "rand() = {v} out of [0,1)");
        assert_eq!(
            v, 0.51385498046875,
            "first rand() from seed=1 must be stable"
        );
    }

    #[test]
    fn awk_srand_reseeds_and_returns_prev_seed_without_host() {
        // srand(42) on a fresh VM (seed=1) returns previous seed 1.0, then the
        // next rand() follows the seed-42 sequence.
        let chunk = {
            let mut b = ChunkBuilder::new();
            b.emit(Op::LoadInt(42), 1);
            b.emit(Op::AwkSrand(1), 1); // pushes prev seed (1.0)
            b.emit(Op::Pop, 1);
            b.emit(Op::AwkRand, 1);
            b.build()
        };
        assert_eq!(run_native(chunk).to_float(), 0.582305908203125);
    }

    #[test]
    fn awk_srand_no_arg_returns_prev_seed_without_host() {
        // srand() with no arg reseeds from the clock but still returns prev seed
        // (1.0 on a fresh VM).
        let chunk = {
            let mut b = ChunkBuilder::new();
            b.emit(Op::AwkSrand(0), 1);
            b.build()
        };
        assert_eq!(run_native(chunk).to_float(), 1.0);
    }

    #[test]
    fn awk_rand_srand_free_fns_match_awkrs_lcg() {
        use crate::awk_host::{awk_rand, awk_srand};
        let mut seed: u64 = 1;
        assert_eq!(awk_rand(&mut seed), 0.51385498046875);
        // srand(Some(42)) returns prev seed low-32 bits and reseeds to 42
        let prev = awk_srand(&mut seed, Some(42));
        assert_eq!(prev, 1103527590.0);
        assert_eq!(awk_rand(&mut seed), 0.582305908203125);
    }

    #[test]
    fn awk_systime_executes_natively_without_host() {
        // systime() pushes seconds since the Unix epoch; must be a large positive
        // number (well past 2020-01-01 = 1577836800) with no host registered.
        let chunk = {
            let mut b = ChunkBuilder::new();
            b.emit(Op::AwkSystime, 1);
            b.build()
        };
        let v = run_native(chunk).to_float();
        assert!(v > 1_577_836_800.0, "systime() = {v} should be past 2020");
    }

    #[test]
    fn awk_systime_free_fn_is_positive() {
        use crate::awk_host::awk_systime;
        assert!(awk_systime() > 1_577_836_800.0);
    }

    #[test]
    fn awk_strtonum_executes_natively_without_host() {
        // strtonum("0x10") → 16 (hex), no host registered.
        let chunk = {
            let mut b = ChunkBuilder::new();
            let s = b.add_constant(Value::str("0x10"));
            b.emit(Op::LoadConst(s), 1);
            b.emit(Op::AwkStrtonum, 1);
            b.build()
        };
        assert_eq!(run_native(chunk).to_float(), 16.0);
    }

    #[test]
    fn awk_strtonum_octal_and_decimal_prefix_without_host() {
        // strtonum("010") → 8 (octal); strtonum("42abc") → 42 (longest prefix).
        let octal = {
            let mut b = ChunkBuilder::new();
            let s = b.add_constant(Value::str("010"));
            b.emit(Op::LoadConst(s), 1);
            b.emit(Op::AwkStrtonum, 1);
            b.build()
        };
        assert_eq!(run_native(octal).to_float(), 8.0);
        let prefix = {
            let mut b = ChunkBuilder::new();
            let s = b.add_constant(Value::str("42abc"));
            b.emit(Op::LoadConst(s), 1);
            b.emit(Op::AwkStrtonum, 1);
            b.build()
        };
        assert_eq!(run_native(prefix).to_float(), 42.0);
    }

    #[test]
    fn awk_strtonum_free_fn_matches_awkrs_semantics() {
        use crate::awk_host::awk_strtonum;
        assert_eq!(awk_strtonum(""), 0.0);
        assert_eq!(awk_strtonum("   "), 0.0);
        assert_eq!(awk_strtonum("0x10"), 16.0);
        assert_eq!(awk_strtonum("0Xff"), 255.0);
        assert_eq!(awk_strtonum("010"), 8.0);
        assert_eq!(awk_strtonum("3.5"), 3.5);
        // invalid hex → 0
        assert_eq!(awk_strtonum("0x"), 0.0);
        assert_eq!(awk_strtonum("0xzz"), 0.0);
        // signed disqualifies hex/octal form: "+0x10" parses leading "+0" → 0
        assert_eq!(awk_strtonum("+0x10"), 0.0);
        // bare nan/inf without sign → 0 (gawk number scan rejects)
        assert_eq!(awk_strtonum("nan"), 0.0);
        assert_eq!(awk_strtonum("inf"), 0.0);
    }

    #[test]
    fn awk_builtin_substr_default_impl_is_posix() {
        use crate::awk_host::{AwkHost, DefaultAwkHost};
        let mut h = DefaultAwkHost;
        assert_eq!(h.substr(&Value::str("hello"), 2, Some(3)).to_str(), "ell");
        assert_eq!(h.substr(&Value::str("hello"), 2, None).to_str(), "ello");
        assert_eq!(h.substr(&Value::str("hello"), 0, Some(3)).to_str(), "he");
        assert_eq!(h.index(&Value::str("hello"), &Value::str("ll")), 3);
        assert_eq!(h.index(&Value::str("hello"), &Value::str("z")), 0);
    }

    #[test]
    fn awk_op_range_is_disjoint_from_generic_extended_wide() {
        use crate::awk_builtins::*;
        assert!(!is_awk_op(0));
        assert!(!is_awk_op(AWK_OP_BASE - 1));
        assert!(is_awk_op(AWK_FIELD_GET));
        assert!(is_awk_op(AWK_ARRAY_LEN));
        assert!(!is_awk_op(AWK_OP_END));
    }

    // ── Block-JIT-eligible builtins with awk negative-arg semantics ──
    // Interpreter-tier coverage for AwkSqrtJit / AwkLogJit (warn-then-NaN) and
    // AwkLshiftJit / AwkRshiftJit / AwkComplJit (fatal trap). Block-JIT codegen
    // is a separate follow-up — these stay block-JIT-ineligible for now and
    // run on the fusevm interpreter through the chunk dispatch.

    fn build_unary(op: Op, x: f64) -> crate::Chunk {
        let mut b = crate::ChunkBuilder::new();
        b.emit(Op::PushFrame, 1);
        b.emit(Op::LoadFloat(x), 1);
        b.emit(op, 1);
        b.build()
    }

    fn build_binary(op: Op, a: f64, n: f64) -> crate::Chunk {
        let mut b = crate::ChunkBuilder::new();
        b.emit(Op::PushFrame, 1);
        b.emit(Op::LoadFloat(a), 1);
        b.emit(Op::LoadFloat(n), 1);
        b.emit(op, 1);
        b.build()
    }

    #[test]
    fn awk_sqrt_jit_negative_warns_returns_nan() {
        let chunk = build_unary(Op::AwkSqrtJit, -4.0);
        let mut vm = VM::new(chunk);
        match vm.run() {
            VMResult::Ok(v) => assert!(v.to_float().is_nan(), "expected NaN, got {v:?}"),
            other => panic!("expected Ok(NaN), got {other:?}"),
        }
    }

    #[test]
    fn awk_sqrt_jit_positive_returns_sqrt() {
        let chunk = build_unary(Op::AwkSqrtJit, 16.0);
        let mut vm = VM::new(chunk);
        match vm.run() {
            VMResult::Ok(v) => assert_eq!(v.to_float(), 4.0),
            other => panic!("expected Ok(4.0), got {other:?}"),
        }
    }

    #[test]
    fn awk_log_jit_positive_returns_ln() {
        let chunk = build_unary(Op::AwkLogJit, std::f64::consts::E);
        let mut vm = VM::new(chunk);
        match vm.run() {
            VMResult::Ok(v) => assert!((v.to_float() - 1.0).abs() < 1e-10),
            other => panic!("expected Ok(~1.0), got {other:?}"),
        }
    }

    #[test]
    fn awk_log_jit_negative_warns_returns_nan() {
        let chunk = build_unary(Op::AwkLogJit, -1.0);
        let mut vm = VM::new(chunk);
        match vm.run() {
            VMResult::Ok(v) => assert!(v.to_float().is_nan()),
            other => panic!("expected Ok(NaN), got {other:?}"),
        }
    }

    #[test]
    fn awk_lshift_jit_computes_left_shift() {
        let chunk = build_binary(Op::AwkLshiftJit, 1.0, 4.0);
        let mut vm = VM::new(chunk);
        match vm.run() {
            VMResult::Ok(v) => assert_eq!(v.to_float(), 16.0, "1 << 4 == 16"),
            other => panic!("expected Ok(16.0), got {other:?}"),
        }
    }

    #[test]
    fn awk_lshift_jit_negative_amount_errors() {
        let chunk = build_binary(Op::AwkLshiftJit, 1.0, -1.0);
        let mut vm = VM::new(chunk);
        match vm.run() {
            VMResult::Error(msg) => assert!(msg.contains("lshift"), "msg = {msg:?}"),
            other => panic!("expected Error, got {other:?}"),
        }
    }

    #[test]
    fn awk_rshift_jit_computes_right_shift() {
        let chunk = build_binary(Op::AwkRshiftJit, 16.0, 2.0);
        let mut vm = VM::new(chunk);
        match vm.run() {
            VMResult::Ok(v) => assert_eq!(v.to_float(), 4.0, "16 >> 2 == 4"),
            other => panic!("expected Ok(4.0), got {other:?}"),
        }
    }

    #[test]
    fn awk_compl_jit_negates_bits() {
        // compl(15) ≈ !15 in i64 ≈ -16 ≈ as f64 ≈ -16.0
        let chunk = build_unary(Op::AwkComplJit, 15.0);
        let mut vm = VM::new(chunk);
        match vm.run() {
            VMResult::Ok(v) => assert_eq!(v.to_float(), -16.0, "compl(15) == -16"),
            other => panic!("expected Ok(-16.0), got {other:?}"),
        }
    }

    #[test]
    fn awk_compl_jit_negative_errors() {
        let chunk = build_unary(Op::AwkComplJit, -1.0);
        let mut vm = VM::new(chunk);
        match vm.run() {
            VMResult::Error(msg) => assert!(msg.contains("compl"), "msg = {msg:?}"),
            other => panic!("expected Error, got {other:?}"),
        }
    }

    // ─── I/O hooks (web-worker bridging) ──────────────────────────────
    // Target-independent: these exercise the OutputSink / InputSource paths
    // that a wasm32 web-worker frontend installs, but run on any host.

    use std::sync::{Arc, Mutex};

    #[test]
    fn output_sink_captures_print_and_println() {
        // `print "a"; println "b"` → sink sees "a" then "b\n".
        let mut b = ChunkBuilder::new();
        let a = b.add_constant(Value::str("a"));
        let bee = b.add_constant(Value::str("b"));
        b.emit(Op::LoadConst(a), 1);
        b.emit(Op::Print(1), 1);
        b.emit(Op::LoadConst(bee), 1);
        b.emit(Op::PrintLn(1), 1);
        let captured = Arc::new(Mutex::new(String::new()));
        let buf = Arc::clone(&captured);
        let mut vm = VM::new(b.build());
        vm.set_output_sink(Box::new(move |s: &str| buf.lock().unwrap().push_str(s)));
        vm.run();
        assert_eq!(*captured.lock().unwrap(), "ab\n");
    }

    #[test]
    fn output_sink_receives_multi_arg_print_concatenated() {
        // `print "x", 1, "y"` with three args → one contiguous "x1y".
        let mut b = ChunkBuilder::new();
        let x = b.add_constant(Value::str("x"));
        let y = b.add_constant(Value::str("y"));
        b.emit(Op::LoadConst(x), 1);
        b.emit(Op::LoadInt(1), 1);
        b.emit(Op::LoadConst(y), 1);
        b.emit(Op::Print(3), 1);
        let captured = Arc::new(Mutex::new(String::new()));
        let buf = Arc::clone(&captured);
        let mut vm = VM::new(b.build());
        vm.set_output_sink(Box::new(move |s: &str| buf.lock().unwrap().push_str(s)));
        vm.run();
        assert_eq!(*captured.lock().unwrap(), "x1y");
    }

    #[test]
    fn input_source_feeds_readline_then_undef_at_eof() {
        // Two ReadLines pull the source's two lines; a third past EOF is Undef.
        let mut b = ChunkBuilder::new();
        b.emit(Op::ReadLine, 1); // -> "first"  (popped last)
        b.emit(Op::Pop, 1);
        b.emit(Op::ReadLine, 1); // -> "second"
        b.emit(Op::Pop, 1);
        b.emit(Op::ReadLine, 1); // -> Undef (EOF)
        let mut pending = vec!["second".to_string(), "first".to_string()];
        let mut vm = VM::new(b.build());
        vm.set_input_source(Box::new(move || pending.pop()));
        match vm.run() {
            VMResult::Ok(Value::Undef) => {}
            other => panic!("expected Undef at EOF, got {other:?}"),
        }
    }
}
