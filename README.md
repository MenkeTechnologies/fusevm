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

## `[PATENT PENDING]`

A language-agnostic bytecode virtual machine with fused superinstructions and 3 stage (linear, block, tracing) Cranelift JIT. Any language frontend compiles to fusevm opcodes and gets fused hot-loop dispatch, extension opcode tables, stack-based execution with slot-indexed fast paths, and native code compilation via Cranelift — for free. 224 opcodes across 17 sections, 11 fused superinstructions, 29 first-class shell ops, 61 first-class AWK ops. Cranelift 0.130 behind `jit` feature flag.

```sh
cargo add fusevm --features jit   # with Cranelift JIT
cargo add fusevm                  # interpreter only
```

### [`Read the Docs`](https://menketechnologies.github.io/fusevm/) &middot; [`Engineering Report`](https://menketechnologies.github.io/fusevm/report.html) · [`API Reference`](https://docs.rs/fusevm) · [`Crates.io`](https://crates.io/crates/fusevm) · [`strykelang`](https://github.com/MenkeTechnologies/strykelang) · [`zshrs`](https://github.com/MenkeTechnologies/zshrs)

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
- [\[0x08\] Ahead-of-Time Compilation](#0x08-ahead-of-time-compilation)
- [\[0x09\] Value Representation](#0x09-value-representation)
- [\[0x0A\] Benchmarks](#0x0a-benchmarks)
- [\[0xFF\] License](#0xff-license)

---

## [0x00] OVERVIEW

fusevm is the shared execution engine behind five language frontends — [zshrs](https://github.com/MenkeTechnologies/zshrs), [strykelang](https://github.com/MenkeTechnologies/strykelang), [awkrs](https://github.com/MenkeTechnologies/awkrs), [vimlrs](https://github.com/MenkeTechnologies/vimlrs), and [elisprs](https://github.com/MenkeTechnologies/elisprs). All five compile to the same `Op` enum. The VM doesn't care which language produced the bytecodes.

```
zshrs  source ──► shell compiler  ──┐
stryke source ──► stryke compiler ──┤
awk    source ──► awk compiler    ──┼──► fusevm::Op ──► VM::run() ─────┐
viml   source ──► viml compiler   ──┤                                  │
elisp  source ──► elisp compiler  ──┘                                  │
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
- **Lean foundational dependencies** — pure Rust, no unsafe in the core; runtime deps are durable, widely-vetted crates (`serde`, `tracing`, `glob`, `chrono`); Cranelift JIT and `libc` disk-cache are opt-in feature flags

---

## [0x01] INSTALL

```sh
cargo add fusevm
# or from source
git clone https://github.com/MenkeTechnologies/fusevm && cd fusevm && cargo build
```

**Cargo features:**

| Feature | Effect |
|---------|--------|
| `jit` | Cranelift-backed native JIT (linear, block, and tracing tiers). |
| `jit-disk-cache` | Persists compiled native code to `~/.cache/fusevm-jit` so codegen is skipped across process restarts. Implies `jit`; on by default once enabled (see [JIT Compilation](#0x07-jit-compilation)). |

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
          │   interpreter   │     │  (eligible chunks)   │
          └─────────────────┘     └─────────────────────┘
```

### Execution tiers — one semantic source of truth

fusevm has four ways to execute a chunk, and they all agree on what every op *means* because they route through — or fall back to — a single function, `VM::exec_op` (`src/vm.rs`), documented in-source as *"the single source of truth for op semantics."*

| Tier | Entry | How it runs an op |
|------|-------|-------------------|
| **Interpreter** | `VM::run` (`src/vm.rs`) | Dispatch loop calls `exec_op(ops, ip, …)` per op; the returned `ExecFlow` says continue or terminate. |
| **Linear / Block / Tracing JIT** | `JitCompiler` (`src/jit.rs`) | Emits specialized Cranelift IR for the eligible integer/float/slot subset; anything ineligible **bails or deopts back to the interpreter** — i.e. back to `exec_op`. |
| **AOT** | `aot::compile_object` (`src/aot.rs`) | Native driver, one Cranelift block per op; unspecialized ops call `exec_op` through the `extern "C"` `fusevm_aot_exec_op` shim (`VM::aot_exec_op`). |

Consequences that fall out of the single-source design:

1. **Semantics never fork.** A new op is implemented once, in `exec_op`, and every tier inherits it — the JIT/AOT specialize *performance*, never *behavior*.
2. **Specialize the hot subset, lean on the interpreter for the tail.** JIT and AOT only lower the scalar (int/float/bool/slot) ops that pay off; string/array/hash/host ops stay in `exec_op`.
3. **Deopt is just "resume `exec_op` at this ip."** On a tracing-JIT guard miss, `materialize_deopt_frames` (`src/vm.rs`) rebuilds the value stack (`stack_buf` + per-entry `stack_kinds` so floats bit-cast back through `f64::from_bits`) and the inlined call frames (`return_ip` + slot values), so the interpreter picks up mid-loop with byte-identical state.

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

224 opcodes across 17 sections in `src/op.rs`:

| Category | Count | Examples |
|----------|-------|---------|
| Constants & Stack | 12 | `LoadInt`, `LoadFloat`, `Pop`, `Dup`, `Swap` |
| Variables | 7 | `GetVar`, `SetVar`, `GetSlot`, `SetSlot`, `SlotArrayGet` |
| Arrays & Hashes | 20 | `ArrayPush`, `HashGet`, `MakeArray`, `HashKeys` |
| Arithmetic | 9 | `Add`, `Sub`, `Mul`, `Div`, `Pow` |
| String | 3 | `Concat`, `StringRepeat`, `StringLen` |
| Comparison | 14 | `NumEq`, `StrLt`, `Spaceship`, `StrCmp` |
| Logical / Bitwise | 9 | `LogNot`, `LogAnd`, `BitAnd`, `Shl`, `Shr` |
| Control Flow | 5 | `Jump`, `JumpIfFalse`, `JumpIfTrueKeep` |
| Functions / Scope | 5 | `Call`, `Return`, `PushFrame`, `PopFrame` |
| I/O | 3 | `Print`, `PrintLn`, `ReadLine` |
| Collections | 2 | `Range`, `RangeStep` |
| Higher-Order | 5 | `MapBlock`, `GrepBlock`, `SortBlock`, `ForEachBlock` |
| **Fused** | **11** | `AccumSumLoop`, `SlotIncLtIntJumpBack`, `ConcatConstLoop`, `PreIncSlot`, `PostIncSlot`, `PreDecSlot`, `PostDecSlot` |
| Builtins | 1 | `CallBuiltin(id, argc)` (140 IDs in `shell_builtins.rs`) |
| Shell Ops | 29 | `Exec`, `PipelineBegin`, `Redirect`, `Glob`, `TestFile`, `RegexMatch` |
| AWK Ops | 61 | `AwkFieldGet`, `AwkPrint`, `AwkStrtonum`, `AwkDivJit`, `AwkModJit`, `AwkGensub`, `AwkOrd`, `AwkChr`, `AwkMkbool`, `AwkIntdiv` |
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

stryke registers ~450 extended ops. zshrs registers ~20. awkrs registers ~95. elisprs registers 10. vimlrs takes the other route — ~510 builtin IDs through `CallBuiltin` rather than extended ops. They don't conflict — each frontend owns its own ID space.

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

### AWK Host (0.13.0+)

The 61 first-class `Op::Awk*` variants dispatch through the `AwkHost` trait. AWK's data model (numeric-string duality, `CONVFMT`/`OFMT` coercion, `$0`/`$n`/`NF` field coupling, `SUBSEP` arrays, regex, `getline`/`printf` IO) lives in the frontend (awkrs), so most AWK ops require a registered host; without one they stay inert but stack-balanced.

Twenty-nine builtins are the exception — they execute natively **even with no host registered**. Most are pure on `fusevm::Value`; `rand`/`srand` run against a VM-owned PRNG seed (execution-intrinsic state, reset with the VM); `strftime`/`mktime` read the system timezone but need no AWK runtime state:

- **Strings:** `substr`, `index`, `tolower`, `toupper`, scalar `length(s)`.
- **Characters (gawk):** `ord` (first char → codepoint), `chr` (codepoint → char, empty if invalid).
- **Math:** `int`, `sqrt`, `sin`, `cos`, `exp`, `log`, `atan2` (pure `f64`), `intdiv` (truncating integer quotient; `Undef` on divide-by-zero), `intdiv0` (same, but `0` on divide-by-zero), `mkbool` (`1`/`0` by truthiness).
- **Bitwise (gawk):** `and`, `or`, `xor`, `compl`, `lshift`, `rshift` (operands truncated to integers).
- **Conversion (gawk):** `strtonum` (`0x…` hex, `0…` octal, else longest decimal/float prefix).
- **Time (gawk):** `systime`, `strftime`, `mktime` (`chrono`-backed; local-tz and UTC paths).
- **PRNG (POSIX/gawk):** `rand`, `srand` (glibc LCG over a VM-owned seed initialized to 1; deterministic without a host).
- **Arithmetic (POSIX awk):** `AwkDiv` (`a / b`), `AwkMod` (`a % b`) — float divide/modulo that raise a fatal `"division by zero attempted"` / `"division by zero attempted in \`%'"` runtime error on a zero divisor (vs the shell-arithmetic `Op::Div`/`Op::Mod`, which yield `Undef`/`0`). Host-independent; interpreter-only (not block/trace-JIT-eligible, since they conditionally trap). `AwkDivJit` / `AwkModJit` are block-JIT-eligible variants with byte-identical interpreter semantics: the block JIT emits a **guarded early-exit** (compare the divisor to `0.0`; on equality call the `fusevm_jit_awk_div_trap` libcall with a code — `1` div / `2` mod — and `return` a sentinel, else `fdiv`/`fmod`). The VM's block-dispatch path reads the trap channel after the compiled run and converts a set code into the same fatal error the interpreter raises, so a JIT-compiled `for(;;) x = 1/0` traps instead of producing `inf`/`NaN` or hanging. The trap libcall is not a registered host-helper id, so `AwkDivJit`/`AwkModJit` chunks skip on-disk cache persistence (in-process JIT only) and never touch the shared cache schema — zshrs/stryke (which emit only `Op::Div`/`Op::Mod`) get byte-identical native code.

**AWK control flow** has no `fusevm::Value` representation (`next`/`nextfile`/`exit` are statements, not expressions). `Op::AwkSignal(code)` carries it host-free: it halts the current chunk and stashes `code` (`awk_builtins::signal::{NEXT, NEXTFILE, EXIT}`) in the VM, which the frontend driver reads via `VM::awk_signal()` after `run()` to drive its own record/file/exit flow. zshrs/stryke never emit it, so `awk_signal()` stays `None` for them and `Halted` is byte-identical to before — the channel is a VM-state side effect, not a new `VMResult` variant. Interpreter-only.

```rust
use fusevm::{VM, ChunkBuilder, Op, Value};

let mut b = ChunkBuilder::new();
let s = b.add_constant(Value::str("hello"));
b.emit(Op::LoadConst(s), 1);
b.emit(Op::LoadInt(2), 1);
b.emit(Op::LoadInt(3), 1);
b.emit(Op::AwkSubstr(3), 1);          // substr("hello", 2, 3)
let mut vm = VM::new(b.build());      // no set_awk_host needed
// vm.run() → "ell"
```

A registered host may still override these (e.g. locale-aware casing, MPFR-precision math, or gawk's fatal-error on negative bitwise operands); the native path is used only when no host is present. `length($0)` and `length(arr)` always need the host (field/array state). `rand`/`srand` also need the host (RNG seed state).

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
| **Block** | `is_block_eligible` + 1 invocation | Whole-chunk CFG (loops, branches, fused backedges) | None — slot ops assume i64 |
| **Tracing** | 50 backedges through any loop header | Hot path through anything; recorded loop body compiled with type-specialized IR | Slot-type entry guard; deopt to interpreter on guard miss |

#### Tuning warmup for re-run-heavy workloads

The block (default **1**) and tracing (default **50**) warmup thresholds are how many times a chunk must run before that tier compiles it. They are tunable two ways:

- **Per process, no recompile** — set environment variables (great for a shell rc when you re-run the same scripts constantly):

  ```sh
  export FUSEVM_JIT_BLOCK_THRESHOLD=0   # block-JIT the whole chunk on its FIRST run (max eager)
  export FUSEVM_JIT_TRACE_THRESHOLD=10  # arm hot-loop traces sooner
  ```

  These are read once per thread when the JIT is first touched, applied on top of the compiled defaults.

- **Per thread, programmatically** — via `TraceJitConfig` (`block_threshold` / `trace_threshold`) and `JitCompiler::set_config`.

For workloads that run the same scripts over and over, combine a low warmup with the **`jit-disk-cache`** feature (on by default): the warmup decides *when* a tier engages, and the disk cache makes the resulting native code free to reload on the next run — so you get AOT-like speed without explicitly AOT-compiling. Setting `FUSEVM_JIT_BLOCK_THRESHOLD=0` is the most aggressive: every block-eligible chunk is compiled to native on its first invocation and reloaded from `~/.cache/fusevm-jit` on subsequent runs. The trade-off is a one-time codegen cost the very first time a chunk is ever seen (paid once, then cached), so raise the thresholds again for scripts that genuinely run only once.

Tracing JIT is opt-in per VM (`vm.enable_tracing_jit()`). The recorder anchors at backward branches, captures the executed op sequence on the next iteration through the header, and installs a compiled trace that runs the loop body in native code until the loop's exit condition becomes false. Slot type changes between invocations cause the entry guard to refuse the trace; after 5 such guard mismatches the trace is blacklisted and never retried.

**Cross-call inlining (phase 2).** `Op::Call` to a sub-entry resolves to the callee's bytecode IP at recording time, and the callee body inlines into the trace IR. Each inlined frame gets its own slot-variable scope (caller slots eagerly promoted from the slot pointer; callee slots lazily allocated zero-initialized). `Op::Return` and `Op::ReturnValue` truncate the abstract stack to the frame's entry mark, mirroring interpreter semantics. Args travel via the value stack — no movement to slots is required.

**Caller-frame internal branches with side-exits (phase 3).** Loops with `if`/`else` bodies are now traceable. The recorder captures the executed direction at each conditional jump (via parallel `recorded_ips`), and the compiler emits a `brif` guard at every internal branch: the runtime condition must match the recorded direction, otherwise control transfers to a per-branch side-exit block that spills the caller's slot variables and returns the un-recorded direction's IP for the interpreter to resume from.

**Callee-frame branches with frame materialization on deopt (phase 4).** Branches are now allowed inside inlined callees, not just the caller frame. When a side-exit fires from inside an inlined callee, the trace populates a `DeoptInfo` out-parameter the VM uses to materialize synthetic `Frame`s on `vm.frames` — each with its `return_ip` pointing back to the post-`Op::Call` IP in the parent, and slot values copied from the trace's per-frame Cranelift Variables. The interpreter then resumes mid-callee with a correctly shaped call stack; when the callee eventually hits `Op::Return`, the synthetic frame is popped and execution continues in the parent. Bounds: max 4 inlined frames at any side-exit, max 16 slot indices per inlined frame.

**Value-stack reconstruction on deopt (phase 5).** The "abstract stack empty at branch" restriction is lifted: branches can fire while the trace's abstract stack still holds intermediate values. At side-exit, those values are written into `DeoptInfo.stack_buf` (capacity 32) and the VM pushes them onto `vm.stack` so the interpreter resumes with the same stack state the bytecode would have at the deopt IP. Phase 5b adds a parallel `stack_kinds` tag array so Float entries get bit-cast through `f64::from_bits` and materialized as `Value::Float` (not just `Value::Int`). This unlocks short-circuit `&&`/`||` patterns and any branch where intermediate float/int computations live on the value stack.

**Side-exit deopt counter + auto-blacklist (phase 6).** Each compiled trace's `TraceCacheEntry` tracks a `side_exit_count` distinct from the entry-guard `deopt_count`. When a brif guard inside the trace fires (the trace returns a resume IP that isn't the loop fallthrough), the counter increments; after `MAX_SIDE_EXITS` (50) misses the trace is blacklisted and never retried. This avoids the pathological case where the recorded path doesn't match runtime and every iteration pays trace+deopt+interpret cost. Note: full side-trace stitching — recording from the side-exit IP and linking the new trace into the main one — is deferred (it's substantial work on its own).

**Persistent trace metadata (phase 7).** `TraceMetadata` is a serde-serializable struct (chunk hash, anchor IP, fallthrough IP, op sequence, recorded IPs, slot-kind snapshot). `JitCompiler::trace_export` extracts it from a compiled-trace cache entry; `trace_import` re-installs it on a fresh `JitCompiler` after verifying `chunk_op_hash` still matches. Persistence format is intentionally caller-owned — fusevm doesn't ship a file layout, so users can pick JSON, bincode, sqlite, or anything else with serde support.

**Bounded recursion inlining (phase 8).** The recorder's hard-no on self-recursive calls is relaxed to a depth cap (`MAX_INLINE_RECURSION` = 4 levels). A self-call up to that depth is inlined like any other Call; deeper recursion aborts the trace and the interpreter handles it. Combined with phase 4's frame materialization, this enables tracing of tail-recursive helpers up to the cap.

**Side-trace stitching (phase 9).** When a main trace's side-exit fires repeatedly at the same IP, the recorder rearms at that IP and records a *side trace*: the bytecode path from the side-exit forward to the loop's backward branch. `TraceRecorder` splits its anchor into `record_anchor_ip` (cache key — the side-exit IP) and `close_anchor_ip` (the enclosing loop's header where the closing branch lands). Side traces compile via `trace_install_with_kind` and don't loop in their own IR — both directions of the closing branch exit, returning either the close target (so the main trace runs the next iteration) or the loop's fallthrough IP (loop done). The VM's chained-dispatch path runs after each main-trace deopt: if a side trace is registered at the resume IP, dispatch it; otherwise bump the main trace's `side_exit_count` toward auto-blacklist. Chains are bounded by `MAX_TRACE_CHAIN` (4) per backward-branch hop. Phase 6's blacklist counter is reserved for cases where no side trace is helping — productive deopts don't penalize the main trace. Side traces use the same eligibility rules as main traces and don't recursively spawn further side traces from their own deopts (their side-exits still bump the main trace's blacklist counter).

**Persistent native-code disk cache (`jit-disk-cache`).** Enable with `cargo add fusevm --features jit-disk-cache` to cache compiled **native code** to disk, skipping Cranelift codegen across process restarts — a big win for workloads that re-launch the VM repeatedly (e.g. running a large test suite over and over). The cache covers **all three tiers** (linear, block, tracing) and is **on by default once the feature is enabled**, writing to `~/.cache/fusevm-jit`. Override the directory with the `FUSEVM_JIT_CACHE_DIR` env var or `JitCompiler::set_jit_cache_dir(Some(dir))`; disable at runtime with `FUSEVM_JIT_CACHE_DIR=off` or `set_jit_cache_dir(None)`.

Cache files are tier-tagged (`.lin.` / `.blk.` / `.trc.`) and keyed by the chunk's op-hash (the tracing tier additionally keys on the record-anchor IP and verifies a content hash over the recorded ops, IPs, slot types, and constants, so divergent recorded paths never collide). Blobs store the native code plus a small relocation table re-patched on load; loading mmaps the code with W^X handling (`pthread_jit_write_protect_np` + icache invalidation on Apple Silicon, `mprotect` elsewhere). Writes publish via a unique temp file + atomic rename, so the cache is safe under many concurrent processes. The loader is **conservative**: any chunk whose code carries a relocation other than a known host-helper call falls back to the in-memory JIT, so an untested target degrades to "no caching" rather than miscompiling. The cache is **behavior-transparent** — it only eliminates Cranelift codegen time; tier selection, warmup thresholds, and results are identical to an uncached run. Benchmark (`cargo bench --features jit-disk-cache --bench jit_disk_cache`): a cached block load is ~35µs versus ~152µs for cold codegen.

**Size control.** Each blob is small — roughly 100 bytes for a linear chunk, up to a few KB for block/trace — and the cache writes one blob per unique JITable segment per script version, so it grows slowly but is never *automatically* trimmed by op-hash (an edited script produces new hashes; the old blobs linger). To keep it bounded there's a **total-size cap, default 256 MiB**, enforced by **oldest-first (mtime) eviction** down to 80% of the cap, applied opportunistically as new blobs are written (so no scan cost on most writes). Controls:

| Knob | Effect |
|------|--------|
| `FUSEVM_JIT_CACHE_MAX_BYTES` | Cap as bytes or with a `k`/`m`/`g` suffix (e.g. `512m`, `2g`). `0`/`off`/`unlimited` disables eviction. Overridden by the programmatic setter. |
| `JitCompiler::set_jit_cache_max_bytes(Some(n))` | Same cap programmatically; `Some(0)` = unlimited, `None` = restore env/default resolution. |
| `JitCompiler::jit_cache_size_bytes()` | Current total cache size in bytes (`None` if disabled). |
| `JitCompiler::prune_jit_cache()` | Force an immediate eviction pass against the cap; returns bytes freed. |
| `JitCompiler::clear_jit_cache()` | Delete every blob (repopulates lazily next run); returns files removed. |
| `rm -rf ~/.cache/fusevm-jit` | Manual nuke. |

---

## [0x08] AHEAD-OF-TIME COMPILATION

The `aot` feature (`src/aot.rs`) compiles a whole `Chunk` to a native object
file via Cranelift's `ObjectModule`, then links it against the fusevm runtime
into a standalone executable — with no interpreter dispatch loop at run time.
It's a closed-world compiler shared by every frontend, so AOT lives here once
and each frontend's `--build` calls into it.

### Threaded-code baseline

The bytecode dispatch loop (`VM::run`) is replaced by a native function with one
Cranelift block per op. Each op block calls the per-op runtime step
(`VM::aot_exec_op`, reached through the `extern "C"` `fusevm_aot_exec_op` shim),
which runs that op via the same `VM::exec_op` the interpreter uses, and returns
the **next instruction index** (or `-1` to terminate). The native code branches
on that through a central `dispatch` block:

```text
entry → dispatch(0)
dispatch(ip): br_table ip → [block_0, …, block_{n-1}]  (default → ret)
block_i:      next = exec_op(vm, i);  if next < 0 → ret  else → dispatch(next)
ret:          finish(vm); return
```

Routing every op through `dispatch` (rather than static fall-through) keeps the
lowering uniform for data-dependent targets — `Op::Jump`, the `JumpIf*` family,
and intra-chunk `Op::Call`/`Op::Return`, whose target is only known at run time —
without the native code ever reading the `VM` struct layout. The interpreter
dispatch loop is gone; the *work* each op does is unchanged.

### Native op specialization

Layered on top of the threaded path, `build_entry` lowers chunks that are
scalar computations directly to native IR (no per-op shim call). `analyze_native`
runs an abstract interpretation over the operand stack — tracking int-vs-bool
`Kind`s, finding basic-block leaders, checking join consistency — and when a
region qualifies, `build_entry_native` emits one Cranelift block per leader with
the operand stack held in frontend `Variable`s (an `i64` and an `f64` per stack
position; the plan's `Kind`s say which is live). This covers:

- **Integer and float arithmetic/comparison**, including `int→float` promotion mirroring the interpreter, `Mod` (integer `srem` with trap-divisor guards, or an `fmod` libcall for floats), and `Pow`/`PowFloat` via a `powf` libcall.
- **Math intrinsics** — `Abs`/`Sqrt`/`Ceil`/`Floor`/`Trunc`/`Round` as single instructions; `Sin`/`Cos`/`Tan`/`Exp`/`Log`/`Atan2` via libcalls; `GcdInt`/`LcmInt` as internal Euclid loops; the awk scalar ops (`AwkDiv`/`AwkMod` and their JIT twins, `AwkSqrtJit`/`AwkLogJit` with warn-and-return-NaN on a negative argument).
- **Bitwise/shift**, `Inc`/`Dec`, booleans (`LoadTrue`/`LoadFalse`/`LogNot`/`LogAnd`/`LogOr`), three-way `Spaceship`, stack shuffles, native control flow (`Jump`/`JumpIf*`, including value-keeping `JumpIf*Keep`).
- **Integer slots and globals** (`GetVar`/`SetVar`/`DeclareVar`) held in SSA registers under a definite-assignment analysis, plus the fused hot-loop slot super-ops (`PreIncSlot`/…, `AddAssignSlotVoid`, `SlotIncLtIntJumpBack`, and `AccumSumLoop` — whose internal `while i < limit { sum += i; i += 1 }` becomes a real native loop).

A fully-scalar loop runs entirely in registers; only the final result is boxed
back into the VM.

### Inline/shim boundary

Chunks that mix scalar work with heap ops don't fall back wholesale. For **sink
ops** (`Print`/`PrintLn`) the native code spills the top register scalars onto
the boxed `vm.stack` (per `Kind`), runs the op via the shim, and continues — so a
hot numeric loop with embedded output stays native. For **source ops whose result
kind is statically known** (`AwkGetFieldNum`, always `Float`) it runs the op via
the shim then reloads the pushed value into a register with no type guard. Slots
and globals are typed by a chunk-wide inferred kind, so a float accumulator
(`sum += 0.5`) lowers to an `f64` register.

### Partial deopt (one-way exit to the interpreter)

Anything the native path can't handle at a given op — a string/array/hash/heap
op, a heap constant load, or an operand-type mismatch — becomes a **deopt point**:
the analysis lowers everything around it, stops propagating past it, and codegen
emits a deopt there. `emit_deopt` writes the **definitely-assigned**
register-cached slots/globals back to the VM (a merely *maybe*-assigned slot at a
deopt point forces a wholesale threaded fallback, since a register can't
distinguish a real `0` from `Undef`), spills the live operand stack, and calls
`fusevm_aot_resume` to hand the rest of the run to the interpreter at the deopt
ip. `Op::Div` uses this for its rare divide-by-zero (native `fdiv` on the common
path, deopt only on a zero divisor); `GetStatus` (`$?`) is lowered as a
statically-typed `Status` source. A chunk falls back to threaded wholesale only
for genuinely structural reasons (stack underflow, inconsistent kind join,
mixed-kind slot, non-numeric final result).

`build_entry` is generic over Cranelift's `Module`, so the in-memory JIT path
that validates the compiler (`run_chunk_native`) and the on-disk `ObjectModule`
path share identical codegen.

| API | Purpose |
|---|---|
| `aot::compile_object(&chunk, path)` | Emit a relocatable `.o` exporting `fusevm_aot_entry` plus the serialized chunk (`fusevm_aot_chunk_blob` / `…_len`). |
| `aot::run_chunk_native(&chunk, register)` | Compile in-process via Cranelift and run it — validates codegen end to end. |
| `aot::fusevm_aot_run_embedded()` | Runtime entry for a linked binary: rebuilds the VM from the embedded chunk, calls the frontend's `fusevm_aot_register_builtins`, runs the native entry, and maps the result to an exit code. |

Link the emitted object against a frontend runtime (which provides
`fusevm_aot_register_builtins`) to produce the standalone binary. On macOS the
link needs `-framework CoreFoundation`. The `staticlib` crate-type in
`Cargo.toml` builds `libfusevm.a` so the object can be linked against the
runtime.

---

## [0x09] VALUE REPRESENTATION

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

## [0x0A] BENCHMARKS

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

**Float slots (`SlotKind::Float`).** Slots are promoted to Cranelift `i64` variables holding raw bits. When a slot's kind is `Float`, the `i64` *is* the `f64` bit pattern: `GetSlot` bitcasts `I64 → F64` (and integer operands are converted with `fcvt_from_sint` before float arithmetic), `SetSlot` bitcasts `F64 → I64`. Pass slot kinds via `try_run_block_kinded` / `try_run_block_eager_kinded`; the kind vector is folded into the native-code cache key (TLS and the on-disk `*.blk.fjit` blob) so float-specialized code is never reused for an integer slot or vice-versa. The default `try_run_block` / `try_run_block_eager` (no kinds) treat every slot as `Int` — unchanged behavior for integer consumers. This is what lets `awkrs` block-JIT-compile `f64` AWK numeric chunks (e.g. `x = int(x + c)`, lowered through `Op::AwkInt`) and persist them to the shared on-disk cache. Integer-only fused superinstructions (`PreIncSlot`, `AccumSumLoop`, `SlotIncLtIntJumpBack`, …) bail to the interpreter on a `Float` slot rather than miscompute it.

**AWK math ops in the JIT.** `Op::AwkInt` compiles natively to a Cranelift `trunc`. The transcendentals `Op::AwkSin` / `AwkCos` / `AwkExp` / `AwkAtan2` compile to Cranelift libcalls into small `extern "C"` Rust helpers (`fusevm_jit_sin_f64`, …) that canonicalize a NaN result to `+nan` to match gawk/awkrs. These follow the same `None`-guarded import pattern as the existing `pow`/`fmod`/`lognot` libcalls — the helper imports are declared only when the op appears in the chunk (`MathIds::declare`), so chunks without them compile to byte-identical native code. For the on-disk cache the helper relocations are keyed by stable host-helper ids (`H_SIN_F64`…`H_ATAN2_F64`), carried in the per-function `[Option<FuncId>; 8]` helper table and re-resolved on load via `host_addr` (cache `SCHEMA_VERSION` 16). The gawk bitwise builtins `Op::AwkAnd` / `AwkOr` / `AwkXor` (variadic, ≥2 args) also compile natively: each operand is converted to `i64` with a **saturating** `fcvt_to_sint_sat` (matching awkrs's `num_to_u64`, which truncates and saturates NaN→0 / ±inf→i64 bounds rather than trapping), folded with Cranelift `band`/`bor`/`bxor`, and pushed back as an integer. No libcall and no host needed — pure integer arithmetic — so they are admitted to `is_block_eligible_op` directly.

**Trapping div/mod in the JIT (guarded early-exit).** `Op::AwkDivJit` / `AwkModJit` are the block-JIT-eligible counterparts of the interpreter-only `AwkDiv`/`AwkMod`. Float `fdiv`/`fmod` do not hardware-trap (they yield `inf`/`NaN`), so a JIT-compiled awk division must check the divisor explicitly: the codegen pops divisor then dividend, emits `fcmp eq divisor, 0.0`, and branches — the trap block calls the `fusevm_jit_awk_div_trap(code)` libcall (`code` = `1` for div, `2` for mod) into a thread-local channel and `return`s a sentinel, while the continuation block computes `fdiv` (div) or the `fmod` libcall (mod). After the compiled block returns, the VM's block-dispatch path calls `take_awk_div_trap()` and, if a code was set, raises the same fatal `"division by zero attempted"` / `…in \`%'` error the interpreter raises — *before* writing slots back. Because the trap libcall is not a registered host-helper id, these chunks skip on-disk persistence (in-process JIT only) and add nothing to the cache schema; frontends that never emit them (zshrs/stryke) are byte-identical.

### Tracing JIT — hot loop bodies compiled to native code

`cargo bench --features jit --bench jit_trace` (Apple M-series). Trace recorded at threshold 5 (default 50 in production) so the cache is primed before measurement; all reported times are steady-state hot-path execution.

**Synergistic three-tier dispatch (phase 10).** When `enable_tracing_jit()` is called, `VM::run` consults all three Cranelift tiers in priority order: block JIT first if the chunk is fully eligible (zero VM-side overhead, direct fn-ptr through the slot pointer), tracing JIT for hot loops in chunks block JIT can't handle, interpreter for cold paths and edge cases. Block-eligible chunks short-circuit before tracing JIT records anything — the two tiers never compete on the same chunk.

| Benchmark | Iterations | Interpreter | Block JIT (direct) | Tracing-JIT VM | VM vs Interp | VM vs Block |
|---|---|---|---|---|---|---|
| `counter_loop` | 1,000 | 24.0 µs | 309 ns | **474 ns** | **51x** | 1.53x slower |
| `counter_loop` | 10,000 | 236.1 µs | 2.69 µs | **2.79 µs** | **84x** | 1.04x slower |
| `counter_loop` | 100,000 | 2,354 µs | 26.71 µs | **26.95 µs** | **87x** | 1.01x slower |
| `loop_with_branch` | 1,000 | 40.2 µs | 300 ns | **474 ns** | **85x** | 1.58x slower |
| `loop_with_branch` | 10,000 | 410.3 µs | 2.68 µs | **2.83 µs** | **145x** | 1.06x slower |
| `loop_with_branch` | 100,000 | 3,942 µs | 26.46 µs | **26.64 µs** | **148x** | 1.01x slower |

`counter_loop` is a tight `for i { i++ }` integer counter — about as friendly to a JIT as bytecode gets. `loop_with_branch` adds an internal `if i > 0 { ... }` inside the body to exercise the phase-3 branch-guard machinery; the recorded path's brif compares slot value to zero each iteration.

The "Block JIT (direct)" column measures `JitCompiler::try_run_block` invoked directly with no VM around it — the floor for what's achievable through the JIT pipeline. The "Tracing-JIT VM" column measures `VM::run()` with `enable_tracing_jit()` set on a block-eligible chunk; the VM auto-dispatches block JIT before reaching the interpreter. The remaining 1.0–1.7x gap between the two is purely VM construction + slot copy-in/out overhead per `vm.run()` call (constant, ~150-200 ns); native execution itself is identical.

For chunks that aren't block-eligible (anything with extension ops, host builtins, or polymorphic types), block JIT bows out and the same `VM::run` path falls through to the interpreter with tracing JIT's recorder armed at backward branches — that's where tracing JIT earns its keep, accelerating loops in code block JIT can't take. The two tiers cover disjoint cases at runtime.

### `VMPool` — VM reuse for callers running many small chunks

`VMPool` recycles `VM` instances so callers running many short-lived chunks (REPL, eval loops, batch evaluation) can skip the per-call `VM::new()` cost. `acquire` pops a recycled VM and resets its state via `VM::reset`; `release` returns it for reuse.

```rust
use fusevm::{ChunkBuilder, Op, VMPool, VMResult, Value};

let mut pool = VMPool::new();
for _ in 0..1000 {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(40), 1);
    b.emit(Op::LoadInt(2), 1);
    b.emit(Op::Add, 1);
    pool.with(b.build(), |vm| {
        assert!(matches!(vm.run(), VMResult::Ok(Value::Int(42))));
    });
}
```

**When the pool actually helps:** chunks where `VM::new()` cost dominates the run. Measured on a 3-op chunk (`LoadInt(40); LoadInt(2); Add`):

| Pattern | Time/call |
|---|---|
| `VM::new(chunk)` per call | 130 ns |
| `pool.acquire(chunk)` per call | 163 ns |

For tiny chunks the pool is *slower* — `reset` does more bookkeeping (drop the old chunk, clear globals, zero the deopt buffer) than `VM::new` skips. The pool wins for chunks where:
- Globals/name pool is large (>16 entries — reset's resize is amortized vs `vec![Value::Undef; n]`)
- Many slots get used (frame.slots Vec capacity is preserved across reuse)
- Tracing JIT runs (deopt buffer is already zeroed and cached eligibility carries over… well, doesn't, since chunk hash differs — gets recomputed)

Honest read: VMPool is useful for **multi-chunk evaluation loops with non-trivial chunk shapes**. For uniform tight loops, pure `VM::new` is fine. The API is shipped so callers can pick. ~10 LOC if your call site looks like `for chunk in ... { VM::new(chunk).run() }`.

**Frontend adoption.** All five sibling frontends (strykelang, awkrs, zshrs, vimlrs, elisprs) drive `fusevm::VM` through bridge layers, NOT direct emit. The common pattern is: (1) frontend-side eligibility analysis (which subroutines / bodies / per-record rules can be lowered to fusevm ops at all), (2) op-vector → `fusevm::Chunk` translation cached behind frontend-owned `OnceCell` / `HashMap` so the 2-pass translation runs once per source program region, not per call, (3) `VMPool` on the frontend `Runtime` so `VM::reset(chunk)` recycles slot/stack/globals Vec capacities across invocations, (4) narrow writeback driven by a precomputed `Vec<u16>` of `Op::SetSlot` targets so only mutated slots get copied back to the frontend's runtime. The on-disk JIT cache (keyed by op-hash) handles compiled-code persistence; the per-frontend in-process caches above handle the upstream chunk-build and runtime-setup costs the disk cache can't touch. strykelang adds `STK_VAL_LOAD_CONST` to make LoadConst-bearing chunks disk-cache safe (index-based, not per-process pointer).

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
