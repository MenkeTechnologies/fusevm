```
 ███████╗██╗   ██╗███████╗███████╗██╗   ██╗███╗   ███╗
 ██╔════╝██║   ██║██╔════╝██╔════╝██║   ██║████╗ ████║
 █████╗  ██║   ██║███████╗█████╗  ██║   ██║██╔████╔██║
 ██╔══╝  ██║   ██║╚════██║██╔══╝  ╚██╗ ██╔╝██║╚██╔╝██║
 ██║     ╚██████╔╝███████║███████╗ ╚████╔╝ ██║ ╚═╝ ██║
 ╚═╝      ╚═════╝ ╚══════╝╚══════╝  ╚═══╝  ╚═╝     ╚═╝
```

[![CI](https://github.com/MenkeTechnologies/fusevm/actions/workflows/ci.yml/badge.svg)](https://github.com/MenkeTechnologies/fusevm/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/fusevm.svg)](https://crates.io/crates/fusevm)
[![Downloads](https://img.shields.io/crates/d/fusevm.svg)](https://crates.io/crates/fusevm)
[![Docs.rs](https://docs.rs/fusevm/badge.svg)](https://docs.rs/fusevm)
 [![Docs](https://img.shields.io/badge/docs-online-blue.svg)](https://menketechnologies.github.io/fusevm/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

### `[LANGUAGE-AGNOSTIC BYTECODE VM WITH FUSED SUPERINSTRUCTIONS POWERING THE FASTEST INTERPRETED LANGUAGES]`

> *"One VM to run them all."*

A language-agnostic bytecode virtual machine with fused superinstructions and Cranelift JIT. Any language frontend compiles to fusevm opcodes and gets fused hot-loop dispatch, extension opcode tables, stack-based execution with slot-indexed fast paths, and native code compilation via Cranelift — for free. 129 opcodes across 10 categories. Cranelift 0.130 behind `jit` feature flag.

```sh
cargo add fusevm --features jit   # with Cranelift JIT
cargo add fusevm                  # interpreter only
```

### [`Docs`](https://menketechnologies.github.io/fusevm/) · [`API Reference`](https://docs.rs/fusevm) · [`Crates.io`](https://crates.io/crates/fusevm) · [`strykelang`](https://github.com/MenkeTechnologies/strykelang) · [`zshrs`](https://github.com/MenkeTechnologies/zshrs)

---

## Table of Contents

- [\[0x00\] Overview](#0x00-overview)
- [\[0x01\] Install](#0x01-install)
- [\[0x02\] Usage](#0x02-usage)
- [\[0x03\] Architecture](#0x03-architecture)
- [\[0x04\] Fused Superinstructions](#0x04-fused-superinstructions)
- [\[0x05\] Op Categories](#0x05-op-categories)
- [\[0x06\] Extension Mechanism](#0x06-extension-mechanism)
- [\[0x07\] JIT Compilation](#0x07-jit-compilation)
- [\[0x08\] Value Representation](#0x08-value-representation)
- [\[0x09\] Benchmarks](#0x09-benchmarks)
- [\[0xFF\] License](#0xff-license)

---

## [0x00] OVERVIEW

fusevm is the shared execution engine behind [strykelang](https://github.com/MenkeTechnologies/strykelang), [zshrs](https://github.com/MenkeTechnologies/zshrs), and [awkrs](https://github.com/MenkeTechnologies/awkrs). All three compile to the same `Op` enum. The VM doesn't care which language produced the bytecodes.

```
stryke source ──► stryke compiler ──┐
                                     │
zshrs source  ──► shell compiler  ──┼──► fusevm::Op ──► VM::run() ─────┐
                                     │                                  │
awkrs source  ──► awk compiler    ──┘                                   │
                                                                        ▼
                                              JitCompiler tiers (Cranelift 0.130)
                                              ├── Linear JIT (straight-line, instant)
                                              ├── Block JIT (CFG, threshold 10)
                                              └── Tracing JIT (hot loop, threshold 50,
                                                              deopts on guard miss)
                                                          │
                                                          ▼
                                                native x86-64 / aarch64
```

- **Fused superinstructions** — the compiler detects hot patterns and emits single ops instead of multi-op sequences
- **Extension dispatch** — language-specific opcodes via `Extended(u16, u8)` with registered handler tables
- **Stack + slots** — stack-based execution with slot-indexed fast paths for locals
- **Three-tier Cranelift JIT** — Linear JIT (straight-line, compile-on-first-call), Block JIT (CFG-aware, threshold 10), Tracing JIT (records hot loop paths, threshold 50, deopts on type-guard miss)
- **Zero-clone dispatch** — ops borrowed from chunk, in-place array/hash mutation, `Cow<str>` string coercion
- **Zero runtime dependencies** — pure Rust, no allocator tricks, no unsafe

---

## [0x01] INSTALL

```sh
cargo add fusevm
# or from source
git clone https://github.com/MenkeTechnologies/fusevm && cd fusevm && cargo build
```

---

## [0x02] USAGE

```rust
use fusevm::{Op, ChunkBuilder, VM, VMResult, Value};

let mut b = ChunkBuilder::new();
b.emit(Op::LoadInt(40), 1);
b.emit(Op::LoadInt(2), 1);
b.emit(Op::Add, 1);

let mut vm = VM::new(b.build());
// Optional: enable tracing JIT — hot loops will be recorded and
// JIT-compiled at runtime. Requires `--features jit`.
#[cfg(feature = "jit")]
vm.enable_tracing_jit();

match vm.run() {
    VMResult::Ok(val) => println!("result: {}", val.to_str()),  // "42"
    VMResult::Error(e) => eprintln!("error: {}", e),
    VMResult::Halted => {}
}
```

---

## [0x03] ARCHITECTURE

```
                  ┌──────────────────────────────────┐
                  │         Language Frontend         │
                  │   (stryke, zshrs, or your own)    │
                  └──────────────┬───────────────────┘
                                 │ compile
                                 ▼
                  ┌──────────────────────────────────┐
                  │       ChunkBuilder::emit()       │
                  │   Op enum ──► Chunk (bytecodes)  │
                  └──────────────┬───────────────────┘
                                 │
                    ┌────────────┴────────────┐
                    ▼                         ▼
          ┌─────────────────┐     ┌─────────────────────┐
          │   VM::run()     │     │   JitCompiler       │
          │  match-dispatch │     │  Cranelift codegen   │
          ��  interpreter    │     │  (eligible chunks)   │
          └─────────────────┘     └─────────────────────┘
```

---

## [0x04] FUSED SUPERINSTRUCTIONS

The performance secret. The compiler detects hot patterns and emits single ops instead of multi-op sequences:

| Fused Op | Replaces | Effect |
|----------|----------|--------|
| `AccumSumLoop(sum, i, limit)` | `GetSlot + GetSlot + Add + SetSlot + PreInc + NumLt + JumpIfFalse` | Entire counted sum loop in one dispatch |
| `SlotIncLtIntJumpBack(slot, limit, target)` | `PreIncSlot + SlotLtIntJumpIfFalse` | Loop backedge in one dispatch |
| `ConcatConstLoop(const, s, i, limit)` | `LoadConst + ConcatAppendSlot + SlotIncLtIntJumpBack` | String append loop in one dispatch |
| `PushIntRangeLoop(arr, i, limit)` | `GetSlot + PushArray + ArrayLen + Pop + SlotIncLtIntJumpBack` | Array push loop in one dispatch |

Each fused op eliminates N-1 dispatch cycles, stack pushes, and branch mispredictions from the hot path.

---

## [0x05] OP CATEGORIES

129 opcodes across 10 categories:

| Category | Count | Examples |
|----------|-------|---------|
| Constants & Stack | ~12 | `LoadInt`, `LoadFloat`, `Pop`, `Dup`, `Swap` |
| Variables | ~8 | `GetVar`, `SetVar`, `GetSlot`, `SetSlot` |
| Arrays & Hashes | ~25 | `ArrayPush`, `HashGet`, `MakeArray`, `HashKeys` |
| Arithmetic | ~9 | `Add`, `Sub`, `Mul`, `Div`, `Pow` |
| Comparison | ~14 | `NumEq`, `StrLt`, `Spaceship` |
| Control Flow | ~5 | `Jump`, `JumpIfFalse`, `JumpIfTrueKeep` |
| Functions | ~3 | `Call`, `Return`, `PushFrame` |
| Shell Ops | ~24 | `Exec`, `PipelineBegin`, `Redirect`, `Glob`, `TestFile` |
| Fused | ~8 | `AccumSumLoop`, `SlotIncLtIntJumpBack` |
| Extension | 2 | `Extended(u16, u8)`, `ExtendedWide(u16, usize)` |

---

## [0x06] EXTENSION MECHANISM

Language-specific opcodes use `Extended(u16, u8)` which dispatches through a handler table registered by the frontend:

```rust
let mut vm = VM::new(chunk);
vm.set_extension_handler(Box::new(|vm, id, arg| {
    match id {
        0 => { /* language-specific op 0 */ }
        1 => { /* language-specific op 1 */ }
        _ => {}
    }
}));
```

stryke registers ~450 extended ops. zshrs registers ~20. awkrs registers ~95. They don't conflict — each frontend owns its own ID space.

### Shell Host (0.10.0+)

Shell-specific runtime ops (`Glob`, `TildeExpand`, `BraceExpand`, `WordSplit`, `ExpandParam`, `CmdSubst`, `ProcessSubIn`/`Out`, `Redirect`, `HereDoc`, `HereString`, `PipelineBegin`/`Stage`/`End`, `SubshellBegin`/`End`, `TrapSet`/`TrapCheck`, `WithRedirectsBegin`/`End`, `CallFunction`, `StrMatch`, `RegexMatch`) dispatch through the `ShellHost` trait. The frontend (zshrs) provides a real implementation; without one, the VM uses minimal stubs that keep stack discipline correct.

```rust
use fusevm::{ShellHost, VM, Chunk, Value};

struct MyHost;
impl ShellHost for MyHost {
    fn glob(&mut self, pattern: &str, _recursive: bool) -> Vec<String> { /* … */ vec![] }
    fn tilde_expand(&mut self, s: &str) -> String { /* … */ s.into() }
    fn cmd_subst(&mut self, sub: &Chunk) -> String { /* run sub, capture stdout */ String::new() }
    // … other methods have default impls
}

let mut vm = VM::new(chunk);
vm.set_shell_host(Box::new(MyHost));
```

Sub-execution (cmd substitution, process substitution, trap handlers) is delivered to the host as `&Chunk` references taken from the parent's `sub_chunks` table. Build them with `ChunkBuilder::add_sub_chunk(sub) -> u16` and reference by index in `Op::CmdSubst(idx)`, `Op::ProcessSubIn(idx)`, `Op::ProcessSubOut(idx)`, `Op::TrapSet(idx)`.

---

## [0x07] JIT COMPILATION

The `JitCompiler` compiles eligible chunks to native code via Cranelift 0.130. Enable with `cargo add fusevm --features jit`.

```rust
use fusevm::{JitCompiler, ChunkBuilder, Op, Value};

let mut b = ChunkBuilder::new();
b.emit(Op::LoadInt(40), 1);
b.emit(Op::LoadInt(2), 1);
b.emit(Op::Add, 1);
let chunk = b.build();

let jit = JitCompiler::new();
if jit.is_linear_eligible(&chunk) {
    // Compiles to native x86-64/aarch64, caches, and runs
    let result = jit.try_run_linear(&chunk, &[]);  // Some(Int(42))
}
```

### Linear JIT — eligible ops

| Category | JIT'd Ops |
|----------|-----------|
| Constants | `LoadInt`, `LoadFloat`, `LoadConst` (int/float), `LoadTrue`, `LoadFalse` |
| Arithmetic | `Add`, `Sub`, `Mul`, `Div`, `Mod`, `Pow`, `Negate`, `Inc`, `Dec` |
| Comparison | `NumEq`/`Ne`/`Lt`/`Gt`/`Le`/`Ge`, `Spaceship` |
| Bitwise | `BitAnd`/`Or`/`Xor`/`Not`, `Shl`, `Shr` |
| Logic | `LogNot` |
| Stack | `Pop`, `Dup`, `Swap`, `Rot` |
| Slots | `GetSlot`, `SetSlot`, `PreIncSlot`, `PreIncSlotVoid`, `AddAssignSlotVoid` |

Int/float promotion: when either operand is float, both are promoted to `f64`. Cranelift emits `iadd`/`fadd`/`fcvt_from_sint` as needed. Runtime helpers for `Pow` (wrapping integer + `f64::powf`) and `Mod` (float `fmod`).

### JIT tier ladder

fusevm runs three JIT tiers in increasing order of optimization power and compile cost. A given chunk can be served by exactly one tier — they cover disjoint cases:

| Tier | Trigger | Coverage | Speculation |
|------|---------|----------|-------------|
| **Linear** | `is_linear_eligible` + first call | Straight-line expression chunks; returns `Value` (int or float) | None — IR matches bytecode exactly |
| **Block** | `is_block_eligible` + 10 invocations | Whole-chunk CFG (loops, branches, fused backedges) | None — slot ops assume i64 |
| **Tracing** | 50 backedges through any loop header | Hot path through anything; recorded loop body compiled with type-specialized IR | Slot-type entry guard; deopt to interpreter on guard miss |

Tracing JIT is opt-in per VM (`vm.enable_tracing_jit()`). The recorder anchors at backward branches, captures the executed op sequence on the next iteration through the header, and installs a compiled trace that runs the loop body in native code until the loop's exit condition becomes false. Slot type changes between invocations cause the entry guard to refuse the trace; after 5 such guard mismatches the trace is blacklisted and never retried.

**Cross-call inlining (phase 2).** `Op::Call` to a sub-entry resolves to the callee's bytecode IP at recording time, and the callee body inlines into the trace IR. Each inlined frame gets its own slot-variable scope (caller slots eagerly promoted from the slot pointer; callee slots lazily allocated zero-initialized). `Op::Return` and `Op::ReturnValue` truncate the abstract stack to the frame's entry mark, mirroring interpreter semantics. Args travel via the value stack — no movement to slots is required.

**Caller-frame internal branches with side-exits (phase 3).** Loops with `if`/`else` bodies are now traceable. The recorder captures the executed direction at each conditional jump (via parallel `recorded_ips`), and the compiler emits a `brif` guard at every internal branch: the runtime condition must match the recorded direction, otherwise control transfers to a per-branch side-exit block that spills the caller's slot variables and returns the un-recorded direction's IP for the interpreter to resume from.

**Callee-frame branches with frame materialization on deopt (phase 4).** Branches are now allowed inside inlined callees, not just the caller frame. When a side-exit fires from inside an inlined callee, the trace populates a `DeoptInfo` out-parameter the VM uses to materialize synthetic `Frame`s on `vm.frames` — each with its `return_ip` pointing back to the post-`Op::Call` IP in the parent, and slot values copied from the trace's per-frame Cranelift Variables. The interpreter then resumes mid-callee with a correctly shaped call stack; when the callee eventually hits `Op::Return`, the synthetic frame is popped and execution continues in the parent. Bounds: max 4 inlined frames at any side-exit, max 16 slot indices per inlined frame.

**Value-stack reconstruction on deopt (phase 5).** The "abstract stack empty at branch" restriction is lifted: branches can fire while the trace's abstract stack still holds intermediate values. At side-exit, those values are written into `DeoptInfo.stack_buf` (capacity 32) and the VM pushes them onto `vm.stack` so the interpreter resumes with the same stack state the bytecode would have at the deopt IP. Phase 5b adds a parallel `stack_kinds` tag array so Float entries get bit-cast through `f64::from_bits` and materialized as `Value::Float` (not just `Value::Int`). This unlocks short-circuit `&&`/`||` patterns and any branch where intermediate float/int computations live on the value stack.

**Side-exit deopt counter + auto-blacklist (phase 6).** Each compiled trace's `TraceCacheEntry` tracks a `side_exit_count` distinct from the entry-guard `deopt_count`. When a brif guard inside the trace fires (the trace returns a resume IP that isn't the loop fallthrough), the counter increments; after `MAX_SIDE_EXITS` (50) misses the trace is blacklisted and never retried. This avoids the pathological case where the recorded path doesn't match runtime and every iteration pays trace+deopt+interpret cost. Note: full side-trace stitching — recording from the side-exit IP and linking the new trace into the main one — is deferred (it's substantial work on its own).

**Persistent trace metadata (phase 7).** `TraceMetadata` is a serde-serializable struct (chunk hash, anchor IP, fallthrough IP, op sequence, recorded IPs, slot-kind snapshot). `JitCompiler::trace_export` extracts it from a compiled-trace cache entry; `trace_import` re-installs it on a fresh `JitCompiler` after verifying `chunk_op_hash` still matches. Persistence format is intentionally caller-owned — fusevm doesn't ship a file layout, so users can pick JSON, bincode, sqlite, or anything else with serde support.

**Bounded recursion inlining (phase 8).** The recorder's hard-no on self-recursive calls is relaxed to a depth cap (`MAX_INLINE_RECURSION` = 4 levels). A self-call up to that depth is inlined like any other Call; deeper recursion aborts the trace and the interpreter handles it. Combined with phase 4's frame materialization, this enables tracing of tail-recursive helpers up to the cap.

**Side-trace stitching (phase 9).** When a main trace's side-exit fires repeatedly at the same IP, the recorder rearms at that IP and records a *side trace*: the bytecode path from the side-exit forward to the loop's backward branch. `TraceRecorder` splits its anchor into `record_anchor_ip` (cache key — the side-exit IP) and `close_anchor_ip` (the enclosing loop's header where the closing branch lands). Side traces compile via `trace_install_with_kind` and don't loop in their own IR — both directions of the closing branch exit, returning either the close target (so the main trace runs the next iteration) or the loop's fallthrough IP (loop done). The VM's chained-dispatch path runs after each main-trace deopt: if a side trace is registered at the resume IP, dispatch it; otherwise bump the main trace's `side_exit_count` toward auto-blacklist. Chains are bounded by `MAX_TRACE_CHAIN` (4) per backward-branch hop. Phase 6's blacklist counter is reserved for cases where no side trace is helping — productive deopts don't penalize the main trace.

**Deferred refinement.** A built-in disk-backed cache implementation is out of scope; the `TraceMetadata` API gives users a clean integration point. Phase 9 ships a working stitching path with the conservative restriction that side traces use the same eligibility rules as main traces and don't recursively spawn further side traces from their own deopts (their side-exits still bump main trace's blacklist counter).

---

## [0x08] VALUE REPRESENTATION

`Value` is a tagged enum with fast-path immediates:

| Variant | Representation | Size |
|---------|---------------|------|
| `Undef` | Tag only | 0 bytes payload |
| `Int(i64)` | Inline | 8 bytes |
| `Float(f64)` | Inline | 8 bytes |
| `Bool(bool)` | Inline | 1 byte |
| `Str(Arc<String>)` | Heap | pointer |
| `Array(Vec<Value>)` | Heap, in-place mutation | 3 words |
| `Hash(HashMap<String, Value>)` | Heap, in-place mutation | 7 words |
| `Status(i32)` | Inline | 4 bytes |
| `Ref(Box<Value>)` | Heap | pointer |
| `NativeFn(u16)` | Inline | 2 bytes |

String coercion returns `Cow<str>` via `as_str_cow()` — borrows the inner `Arc<String>` for `Str` variants, avoiding allocation on string comparisons, concatenation, hash key lookup, and I/O.

Array and hash mutations (`ArrayPush`, `ArrayPop`, `ArrayShift`, `ArraySet`, `HashSet`, `HashDelete`) operate in-place on globals — no clone-modify-writeback cycle. Read-only access (`ArrayGet`, `ArrayLen`, `HashGet`, `HashExists`, `HashKeys`, `HashValues`) borrows directly from the globals vector.

---

## [0x09] BENCHMARKS

All benchmarks run via [criterion](https://crates.io/crates/criterion) on Apple M-series. `cargo bench` for all, `cargo bench --features jit --bench jit_vs_interp` for JIT comparisons. HTML report at `target/criterion/report/index.html`.

### Classic algorithms

| Benchmark | Time | Ops/sec |
|-----------|------|---------|
| `fib_iterative(35)` | 2.7 µs | 374k |
| `fib_recursive(20)` — 21,891 calls | 1.28 ms | 783 |
| `ackermann(3,4)` — 10,547 calls | 774 µs | 1.3k |
| `sum(1..1M)` fused `AccumSumLoop` | 142 ns | 7.0M |
| `sum(1..1M)` unfused loop ops | 31.0 ms | 32 |
| `nested_loop(100×100)` | 352 µs | 2.8k |
| `dispatch_nop_1M` — raw dispatch overhead | 819 µs | **1.22 Gops/sec** |
| `string_build(10k)` via `ConcatConstLoop` | 11.9 µs | 84k |

### Interpreter vs Cranelift JIT vs native Rust

Slot-based inputs prevent constant folding — honest apples-to-apples comparison:

| Workload | Interpreter | JIT (cached) | Native Rust | JIT vs interp | JIT vs native |
|----------|-------------|--------------|-------------|---------------|---------------|
| `slot_mixed × 100` | 2.2 µs | **75 ns** | 42 ns | **29x faster** | 1.8x slower |
| `slot_bitwise × 200` | 6.6 µs | **130 ns** | 74 ns | **51x faster** | 1.8x slower |
| `slot_float × 200` | 3.1 µs | **246 ns** | 137 ns | **13x faster** | 1.8x slower |

JIT cache lookup is O(1) — chunk hash precomputed at build time (24ns overhead). The linear JIT is consistently ~1.8x slower than LLVM `-O3` on real computation and 13–51x faster than the interpreter.

### Block JIT — loops and branches compiled to native code

The block JIT handles real control flow — loops, conditionals, fused backedges:

| Benchmark | Interpreter | Block JIT | Speedup |
|-----------|-------------|-----------|---------|
| `sum(1..1M)` unfused loop | 30.0 ms | **315 µs** | **95x** |
| `nested_loop(100×100)` | 340 µs | **9.5 µs** | **36x** |

The block JIT compiles the full CFG to native code via Cranelift. All mutable state flows through the slots pointer (`*mut i64`), and `AccumSumLoop` is register-allocated with block parameters — no memory traffic in the inner loop.

### Tracing JIT — hot loop bodies compiled to native code

`cargo bench --features jit --bench jit_trace` (Apple M-series). Trace recorded at threshold 5 (default 50 in production) so the cache is primed before measurement; all reported times are steady-state hot-path execution.

**Three-tier comparison.** Block JIT bypasses the VM entirely — caller invokes the compiled fn through a direct fn-ptr with a slot pointer. Tracing JIT dispatches through `VM::run()` which adds startup, the backward-branch lookup, and slot copy-in/out per invocation. At large iteration counts both converge as native execution dominates; at small N the VM overhead is visible.

| Benchmark | Iterations | Interpreter | Block JIT | Tracing JIT | Trace vs Interp | Trace vs Block |
|---|---|---|---|---|---|---|
| `counter_loop` | 1,000 | 23.0 µs | — | **0.57 µs** | **40x** | — |
| `counter_loop` | 10,000 | 282.6 µs | 2.66 µs | **3.45 µs** | **82x** | 1.30x slower |
| `counter_loop` | 100,000 | 3,287 µs | 27.05 µs | **27.91 µs** | **118x** | 1.03x slower |
| `loop_with_branch` | 1,000 | 43.0 µs | 0.29 µs | **0.81 µs** | **53x** | 2.76x slower |
| `loop_with_branch` | 10,000 | 448.0 µs | 2.85 µs | **3.54 µs** | **127x** | 1.24x slower |
| `loop_with_branch` | 100,000 | 4,351 µs | 29.20 µs | **28.85 µs** | **151x** | **1.01x faster** |

`counter_loop` is a tight `for i { i++ }` integer counter — about as friendly to a JIT as bytecode gets. `loop_with_branch` adds an internal `if i > 0 { ... }` inside the body to exercise the phase-3 branch-guard machinery; the recorded path's brif compares slot value to zero each iteration.

The trace-vs-block gap narrows from ~2-3x at 1k iterations to within-noise at 100k. Block JIT wins on minimum overhead (no VM dispatch, direct fn-ptr) but only handles whole-chunk eligibility — anything with extension ops, calls to host builtins, or polymorphic types falls back to the interpreter. Tracing JIT covers those cases with a small constant overhead from running through `VM::run`. For purely loop-shaped integer kernels block JIT is faster; for real workloads where most code paths aren't 100% block-eligible, tracing JIT is the more general win.

### Tracking improvements

```sh
cargo bench --bench vm_bench -- --save-baseline before   # save baseline
# ... make changes ...
cargo bench --bench vm_bench -- --baseline before        # compare
open target/criterion/report/index.html                  # HTML graphs
```

---

## [0xFF] LICENSE

MIT — Copyright (c) 2026 [MenkeTechnologies](https://github.com/MenkeTechnologies)
