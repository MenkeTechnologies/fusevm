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

### `[LANGUAGE-AGNOSTIC BYTECODE VM WITH FUSED SUPERINSTRUCTIONS]`

> *"One VM to run them all."*

A language-agnostic bytecode virtual machine with fused superinstructions and Cranelift JIT. Any language frontend compiles to fusevm opcodes and gets fused hot-loop dispatch, extension opcode tables, stack-based execution with slot-indexed fast paths, and native code compilation via Cranelift — for free. 127 opcodes across 10 categories. Cranelift 0.130 behind `jit` feature flag.

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
- [\[0xFF\] License](#0xff-license)

---

## [0x00] OVERVIEW

fusevm is the shared execution engine behind [strykelang](https://github.com/MenkeTechnologies/strykelang), [zshrs](https://github.com/MenkeTechnologies/zshrs), and [awkrs](https://github.com/MenkeTechnologies/awkrs). All three compile to the same `Op` enum. The VM doesn't care which language produced the bytecodes.

```
stryke source ──► stryke compiler ──┐
                                     │
zshrs source  ──► shell compiler  ──┼──► fusevm::Op ──► VM::run()
                                     │         │
awkrs source  ──► awk compiler    ──┘    JitCompiler::try_run_linear()
                                                │
                                          Cranelift 0.130
                                         native x86-64 / aarch64
```

- **Fused superinstructions** — the compiler detects hot patterns and emits single ops instead of multi-op sequences
- **Extension dispatch** — language-specific opcodes via `Extended(u16, u8)` with registered handler tables
- **Stack + slots** — stack-based execution with slot-indexed fast paths for locals
- **Cranelift JIT** — eligibility analysis and compilation for hot chunks
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

127 opcodes across 10 categories:

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

## [0xFF] LICENSE

MIT — Copyright (c) 2026 [MenkeTechnologies](https://github.com/MenkeTechnologies)
