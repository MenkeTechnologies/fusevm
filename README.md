# fusevm

[![Crates.io](https://img.shields.io/crates/v/fusevm.svg)](https://crates.io/crates/fusevm)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Language-agnostic bytecode VM with fused superinstructions.**

Any language frontend can compile to fusevm opcodes and get:
- Fused superinstructions for hot loops (`AccumSumLoop`, `SlotIncLtIntJumpBack`, etc.)
- Extension opcode dispatch for language-specific ops
- Stack-based execution with slot-indexed fast paths
- Cranelift JIT compilation path (planned)

## Architecture

```
stryke source ──► stryke compiler ──┐
                                     ├──► fusevm::Op ──► VM::run()
zshrs source  ──► shell compiler  ──┘
```

Multiple language frontends compile to the same `Op` enum. The VM doesn't care which language produced the bytecodes.

## Usage

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

## Fused Superinstructions

The performance secret. The compiler detects hot patterns and emits single ops instead of multi-op sequences:

| Fused Op | Replaces | Speedup |
|----------|----------|---------|
| `AccumSumLoop(sum, i, limit)` | `GetSlot + GetSlot + Add + SetSlot + PreInc + NumLt + JumpIfFalse` | Entire counted sum loop in one dispatch |
| `SlotIncLtIntJumpBack(slot, limit, target)` | `PreIncSlot + SlotLtIntJumpIfFalse` | Loop backedge in one dispatch |
| `ConcatConstLoop(const, s, i, limit)` | `LoadConst + ConcatAppendSlot + SlotIncLtIntJumpBack` | String append loop in one dispatch |
| `PushIntRangeLoop(arr, i, limit)` | `GetSlot + PushArray + ArrayLen + Pop + SlotIncLtIntJumpBack` | Array push loop in one dispatch |

## Op Categories

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

## Extension Mechanism

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

stryke registers ~450 extended ops. zshrs registers ~20. They don't conflict.

## License

MIT
