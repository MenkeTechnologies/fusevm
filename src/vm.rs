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
    /// Extension handler for `Op::Extended`
    ext_handler: Option<ExtensionHandler>,
    /// Extension handler for `Op::ExtendedWide`
    ext_wide_handler: Option<ExtensionWideHandler>,
    /// Inline builtin cache: builtin_id → function pointer (no lookup at dispatch)
    builtin_table: Vec<Option<BuiltinHandler>>,
    /// Frontend-supplied shell host (glob/expand/redirect/pipeline/etc).
    /// When `None`, shell ops fall back to minimal stub behavior.
    pub host: Option<Box<dyn ShellHost>>,
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
    #[cfg(feature = "jit")]
    deopt_info: Box<DeoptInfo>,
}

/// Result of VM execution
#[derive(Debug)]
pub enum VMResult {
    /// Normal completion with a value
    Ok(Value),
    /// Halted (no more instructions)
    Halted,
    /// Runtime error
    Error(String),
}

impl VM {
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
            ext_handler: None,
            ext_wide_handler: None,
            builtin_table: Vec::new(),
            host: None,
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
            deopt_info: Box::new(DeoptInfo::zeroed()),
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

    /// Register the frontend shell host. Replaces any prior host.
    pub fn set_shell_host(&mut self, host: Box<dyn ShellHost>) {
        self.host = Some(host);
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

    // ── Tracing JIT integration helpers ──

    /// Snapshot the current frame's slots into the i64 + slot-kind buffers.
    ///
    /// Slots that don't fit cleanly into i64 (Array/Hash/String/etc) are
    /// reported as `SlotKind::Int` with i64 value 0 — they will fail the
    /// trace's entry guard if the recorded trace expected Int there, which
    /// is the desired behavior (skip the trace, fall back to interpreter).
    #[cfg(feature = "jit")]
    fn refresh_slot_buffers(&mut self) {
        let frame = match self.frames.last() {
            Some(f) => f,
            None => return,
        };
        let n = frame.slots.len();
        self.slot_buf.clear();
        self.slot_kinds_buf.clear();
        self.slot_buf.reserve(n);
        self.slot_kinds_buf.reserve(n);
        for v in &frame.slots {
            let (i, kind) = match v {
                Value::Int(n) => (*n, SlotKind::Int),
                Value::Float(f) => (f.to_bits() as i64, SlotKind::Float),
                Value::Bool(b) => (*b as i64, SlotKind::Int),
                _ => (0, SlotKind::Int),
            };
            self.slot_buf.push(i);
            self.slot_kinds_buf.push(kind);
        }
    }

    /// Copy the i64 slot buffer back into the current frame's slots,
    /// materializing Int and Float kinds. Float slots are stored as i64
    /// bit patterns in the buffer; recover via `f64::from_bits`. Slots of
    /// other kinds (Array, Hash, etc.) are left untouched — those slots
    /// would have prevented trace install if referenced.
    #[cfg(feature = "jit")]
    fn write_slots_back(&mut self) {
        let frame = match self.frames.last_mut() {
            Some(f) => f,
            None => return,
        };
        let n = frame.slots.len().min(self.slot_buf.len());
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

    #[inline(always)]
    pub fn push(&mut self, val: Value) {
        self.stack.push(val);
    }

    #[inline(always)]
    pub fn pop(&mut self) -> Value {
        self.stack.pop().unwrap_or(Value::Undef)
    }

    #[inline(always)]
    pub fn peek(&self) -> &Value {
        self.stack.last().unwrap_or(&Value::Undef)
    }

    // ── Type-specialized helpers (avoid to_float coercion on hot paths) ──

    /// Pop two values; if both Int, apply int_op. Otherwise promote to float.
    #[inline(always)]
    fn arith_int_fast(&mut self, int_op: fn(i64, i64) -> i64, float_op: fn(f64, f64) -> f64) {
        let len = self.stack.len();
        if len >= 2 {
            // Borrow both slots without popping (avoid branch + unwrap_or)
            let b = &self.stack[len - 1];
            let a = &self.stack[len - 2];
            let result = match (a, b) {
                (Value::Int(x), Value::Int(y)) => Value::Int(int_op(*x, *y)),
                _ => Value::Float(float_op(a.to_float(), b.to_float())),
            };
            self.stack.truncate(len - 2);
            self.stack.push(result);
        }
    }

    /// Pop two values; compare as int if both Int, otherwise float.
    /// Push Bool(true/false).
    #[inline(always)]
    fn cmp_int_fast(&mut self, int_cmp: fn(i64, i64) -> bool, float_cmp: fn(f64, f64) -> bool) {
        let len = self.stack.len();
        if len >= 2 {
            let b = &self.stack[len - 1];
            let a = &self.stack[len - 2];
            let result = match (a, b) {
                (Value::Int(x), Value::Int(y)) => int_cmp(*x, *y),
                _ => float_cmp(a.to_float(), b.to_float()),
            };
            self.stack.truncate(len - 2);
            self.stack.push(Value::Bool(result));
        }
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
        // Phase 10: try block JIT first for fully-eligible chunks. The
        // block-JIT cache has its own threshold (10 invocations); the
        // call returns None until it warms up, at which point the whole
        // chunk runs in native code.
        #[cfg(feature = "jit")]
        if self.tracing_jit && self.frames.len() == 1 && self.ip == 0 {
            if self.jit.is_block_eligible(&self.chunk) {
                self.refresh_slot_buffers();
                if let Some(result_i64) = self.jit.try_run_block(&self.chunk, &mut self.slot_buf) {
                    self.write_slots_back();
                    self.halted = true;
                    return VMResult::Ok(Value::Int(result_i64));
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
                            _ => {}
                        }
                    }
                }
            }

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
                Op::Add => self.arith_int_fast(i64::wrapping_add, |a, b| a + b),
                Op::Sub => self.arith_int_fast(i64::wrapping_sub, |a, b| a - b),
                Op::Mul => self.arith_int_fast(i64::wrapping_mul, |a, b| a * b),
                Op::Div => {
                    let b = self.pop();
                    let a = self.pop();
                    let divisor = b.to_float();
                    self.push(if divisor == 0.0 {
                        Value::Undef
                    } else {
                        Value::Float(a.to_float() / divisor)
                    });
                }
                Op::Mod => self.arith_int_fast(|x, y| if y != 0 { x % y } else { 0 }, |a, b| a % b),
                Op::Pow => {
                    let b = self.pop();
                    let a = self.pop();
                    self.push(Value::Float(a.to_float().powf(b.to_float())));
                }
                Op::Negate => {
                    let val = self.pop();
                    self.push(match val {
                        Value::Int(n) => Value::Int(n.wrapping_neg()),
                        _ => Value::Float(-val.to_float()),
                    });
                }
                Op::Inc => {
                    let val = self.pop();
                    self.push(match val {
                        Value::Int(n) => Value::Int(n.wrapping_add(1)),
                        _ => Value::Int(val.to_int().wrapping_add(1)),
                    });
                }
                Op::Dec => {
                    let val = self.pop();
                    self.push(match val {
                        Value::Int(n) => Value::Int(n.wrapping_sub(1)),
                        _ => Value::Int(val.to_int().wrapping_sub(1)),
                    });
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
                Op::NumEq => self.cmp_int_fast(|x, y| x == y, |a, b| a == b),
                Op::NumNe => self.cmp_int_fast(|x, y| x != y, |a, b| a != b),
                Op::NumLt => self.cmp_int_fast(|x, y| x < y, |a, b| a < b),
                Op::NumGt => self.cmp_int_fast(|x, y| x > y, |a, b| a > b),
                Op::NumLe => self.cmp_int_fast(|x, y| x <= y, |a, b| a <= b),
                Op::NumGe => self.cmp_int_fast(|x, y| x >= y, |a, b| a >= b),
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
                        return VMResult::Error(format!(
                            "undefined function: {}",
                            self.chunk
                                .names
                                .get(*name_idx as usize)
                                .map(|s| s.as_str())
                                .unwrap_or("?")
                        ));
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
                        return VMResult::Ok(val);
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
                    use std::io::Write;
                    let stdout = std::io::stdout();
                    let mut lock = stdout.lock();
                    for v in &self.stack[start..] {
                        let _ = write!(lock, "{}", v.as_str_cow());
                    }
                    self.stack.truncate(start);
                }
                Op::PrintLn(n) => {
                    let n = *n;
                    let start = self.stack.len().saturating_sub(n as usize);
                    use std::io::Write;
                    let stdout = std::io::stdout();
                    let mut lock = stdout.lock();
                    for v in &self.stack[start..] {
                        let _ = write!(lock, "{}", v.as_str_cow());
                    }
                    let _ = writeln!(lock);
                    self.stack.truncate(start);
                }
                Op::ReadLine => {
                    let mut line = String::new();
                    let _ = std::io::stdin().read_line(&mut line);
                    self.push(Value::str(line.trim_end_matches('\n')));
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
                    if let Some(mut handler) = self.ext_wide_handler.take() {
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
                        crate::op::file_test::IS_SYMLINK => {
                            std::path::Path::new(&path).is_symlink()
                        }
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
                                continue;
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
                                    use std::process::{Command, Stdio};
                                    Command::new(cmd)
                                        .args(&args[1..])
                                        .stdout(Stdio::inherit())
                                        .stderr(Stdio::inherit())
                                        .status()
                                        .map(|s| s.code().unwrap_or(1))
                                        .unwrap_or(127)
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
                            use std::process::{Command, Stdio};
                            let _ = Command::new(cmd)
                                .args(&args[1..])
                                .stdout(Stdio::null())
                                .stderr(Stdio::null())
                                .spawn();
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
                        h.subshell_end();
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
                    let status = if let Some(h) = self.host.as_mut() {
                        match h.call_function(&name, args.clone()) {
                            Some(s) => s,
                            None => {
                                let mut full = Vec::with_capacity(args.len() + 1);
                                full.push(name.clone());
                                full.extend(args);
                                h.exec(full)
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
                            continue;
                        }
                        let mut full = Vec::with_capacity(args.len() + 1);
                        full.push(name);
                        full.extend(args);
                        use std::process::Command;
                        Command::new(&full[0])
                            .args(&full[1..])
                            .status()
                            .map(|s| s.code().unwrap_or(1))
                            .unwrap_or(127)
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
        }

        if let Some(val) = self.stack.pop() {
            VMResult::Ok(val)
        } else {
            VMResult::Halted
        }
    }

    // ── Helpers ──

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

    fn get_slot(&self, slot: u16) -> Value {
        self.frames
            .last()
            .and_then(|f| f.slots.get(slot as usize))
            .cloned()
            .unwrap_or(Value::Undef)
    }

    fn set_slot(&mut self, slot: u16, val: Value) {
        if let Some(frame) = self.frames.last_mut() {
            let idx = slot as usize;
            if idx >= frame.slots.len() {
                frame.slots.resize(idx + 1, Value::Undef);
            }
            frame.slots[idx] = val;
        }
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
}
