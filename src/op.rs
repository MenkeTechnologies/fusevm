//! Bytecode instruction set for fusevm.
//!
//! Universal ops that any language frontend can target.
//! Language-specific ops use `Extended(u16, u8)` which dispatches
//! through a handler table registered by the frontend.

use serde::{Deserialize, Serialize};
use std::hash::{Hash, Hasher};

/// Stack-based bytecode instruction set.
///
/// Operands: u16 for pool indices (64k names/constants), usize for jump targets.
/// Language-specific operations use `Extended` with a frontend-registered handler.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Op {
    Nop,

    // ── Constants ──
    LoadInt(i64),
    LoadFloat(f64),
    LoadConst(u16), // index into constant pool
    LoadTrue,
    LoadFalse,
    LoadUndef,

    // ── Stack ──
    Pop,
    Dup,
    Dup2,
    Swap,
    Rot,

    // ── Variables (u16 = name pool index) ──
    GetVar(u16),
    SetVar(u16),
    DeclareVar(u16),
    /// Slot-indexed fast path (frame slot index, avoids name lookup)
    GetSlot(u16),
    SetSlot(u16),
    /// Slot-based array index: stack: [index], slot contains array → value
    SlotArrayGet(u16),
    /// Slot-based array set: stack: [value, index], slot contains array
    SlotArraySet(u16),

    // ── Arrays ──
    GetArray(u16),
    SetArray(u16),
    DeclareArray(u16),
    ArrayGet(u16),   // stack: [index] → value
    ArraySet(u16),   // stack: [value, index]
    ArrayPush(u16),  // stack: [value]
    ArrayPop(u16),   // → popped value
    ArrayShift(u16), // → shifted value
    ArrayLen(u16),   // → length
    MakeArray(u16),  // pop N values, push as array

    // ── Hashes ──
    GetHash(u16),
    SetHash(u16),
    DeclareHash(u16),
    HashGet(u16),    // stack: [key] → value
    HashSet(u16),    // stack: [value, key]
    HashDelete(u16), // stack: [key] → deleted value
    HashExists(u16), // stack: [key] → bool
    HashKeys(u16),   // → array of keys
    HashValues(u16), // → array of values
    MakeHash(u16),   // pop N key-value pairs, push as hash

    // ── Arithmetic ──
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Pow,
    Negate,
    Inc,
    Dec,

    // ── String ──
    Concat,
    StringRepeat,
    StringLen,

    // ── Comparison (numeric) ──
    NumEq,
    NumNe,
    NumLt,
    NumGt,
    NumLe,
    NumGe,
    Spaceship, // <=> → -1, 0, 1

    // ── Comparison (string) ──
    StrEq,
    StrNe,
    StrLt,
    StrGt,
    StrLe,
    StrGe,
    StrCmp,

    // ── Logical / Bitwise ──
    LogNot,
    LogAnd, // differs from short-circuit jumps: evaluates both
    LogOr,
    BitAnd,
    BitOr,
    BitXor,
    BitNot,
    Shl,
    Shr,

    // ── Control flow ──
    Jump(usize),
    JumpIfTrue(usize),
    JumpIfFalse(usize),
    JumpIfTrueKeep(usize),  // short-circuit ||
    JumpIfFalseKeep(usize), // short-circuit &&

    // ── Functions ──
    /// Call: name_index, arg_count
    Call(u16, u8),
    Return,
    ReturnValue,

    // ── Scope ──
    PushFrame,
    PopFrame,

    // ── I/O ──
    /// Print N values from stack to stdout
    Print(u8),
    /// Print N values + newline
    PrintLn(u8),
    /// Read line from stdin, push as string
    ReadLine,

    // ── Collections ──
    /// [from, to] → array
    Range,
    /// [from, to, step] → array
    RangeStep,

    // ── Higher-order (u16 = block index in chunk) ──
    MapBlock(u16),
    GrepBlock(u16),
    SortBlock(u16),
    SortDefault, // sort with default string comparison
    ForEachBlock(u16),

    // ── Fused superinstructions ──
    // These are the performance secret sauce.
    // The compiler detects hot loop patterns and emits these
    // instead of multi-op sequences.
    /// Slot-indexed pre-increment (no stack traffic)
    PreIncSlot(u16),
    /// `if ($slot < INT) goto target` — fused compare + branch
    SlotLtIntJumpIfFalse(u16, i32, usize),
    /// `$slot += 1; if $slot < limit goto body` — fused loop backedge
    SlotIncLtIntJumpBack(u16, i32, usize),
    /// `while $i < limit { $sum += $i; $i += 1 }` — entire counted sum loop
    AccumSumLoop(u16, u16, i32),
    /// `while $i < limit { $s .= CONST; $i += 1 }` — fused string append loop
    ConcatConstLoop(u16, u16, u16, i32),
    /// `while $i < limit { push @a, $i; $i += 1 }` — fused array push loop
    PushIntRangeLoop(u16, u16, i32),
    /// Void-context slot add-assign: `$a += $b` (no stack push)
    AddAssignSlotVoid(u16, u16),
    /// Void-context pre-increment: `++$slot` (no stack push)
    PreIncSlotVoid(u16),

    // ── Builtins ──
    /// Call a registered builtin by ID: (builtin_id, arg_count)
    /// The builtin table is registered by the frontend at VM init.
    CallBuiltin(u16, u8),

    // ── Extension point ──
    /// Language-specific opcode dispatched through a frontend handler table.
    /// u16 = extension op ID, u8 = inline operand.
    /// Frontends register a `fn(&mut VM, u16, u8)` handler at init.
    Extended(u16, u8),
    /// Extended with usize payload (for jump targets, large indices)
    ExtendedWide(u16, usize),

    // ── Shell ops (registered via Extended, but defined here for type safety) ──
    // These are first-class because process control is universal enough
    // that multiple frontends need them (shell, scripting, build tools).
    /// Spawn external command: pop N args from stack, exec, push exit status
    Exec(u8),
    /// Spawn background: like Exec but don't wait
    ExecBg(u8),
    /// Set up N-stage pipeline
    PipelineBegin(u8),
    /// Wire next pipeline stage
    PipelineStage,
    /// Wait for pipeline, push last status
    PipelineEnd,
    /// Redirect fd: (source_fd, op_byte) — target on stack
    Redirect(u8, u8),
    /// Here-document: fd on stack, content from constant pool
    HereDoc(u16),
    /// Here-string: fd on stack, word on stack
    HereString,
    /// Command substitution: capture stdout of subprogram
    CmdSubst(u16), // u16 = bytecode range index
    /// Subshell: isolate scope
    SubshellBegin,
    SubshellEnd,
    /// Process substitution <(cmd) — push FIFO path
    ProcessSubIn(u16),
    /// Process substitution >(cmd) — push FIFO path
    ProcessSubOut(u16),
    /// Glob expand: pop pattern, push array of matches
    Glob,
    /// Recursive glob (parallel): pop pattern, push array
    GlobRecursive,
    /// File test: u8 encodes test type (-f=0, -d=1, -r=2, -w=3, -x=4, -e=5, -s=6, -L=7)
    TestFile(u8),
    /// Set last exit status ($?)
    SetStatus,
    /// Get last exit status
    GetStatus,
    /// Set trap handler: signal on stack, handler bytecode range
    TrapSet(u16),
    /// Check pending traps (inserted between ops by compiler)
    TrapCheck,
    /// Expand ${var:-default} family: u8 encodes modifier type
    ExpandParam(u8),
    /// Word split by IFS
    WordSplit,
    /// Brace expand {a,b} and {1..10}
    BraceExpand,
    /// Tilde expand ~ and ~user
    TildeExpand,
}

/// File test opcodes for `TestFile(u8)`
pub mod file_test {
    pub const IS_FILE: u8 = 0;
    pub const IS_DIR: u8 = 1;
    pub const IS_READABLE: u8 = 2;
    pub const IS_WRITABLE: u8 = 3;
    pub const IS_EXECUTABLE: u8 = 4;
    pub const EXISTS: u8 = 5;
    pub const IS_NONEMPTY: u8 = 6;
    pub const IS_SYMLINK: u8 = 7;
    pub const IS_SOCKET: u8 = 8;
    pub const IS_FIFO: u8 = 9;
    pub const IS_BLOCK_DEV: u8 = 10;
    pub const IS_CHAR_DEV: u8 = 11;
}

/// Redirect op types for `Redirect(fd, op)`
pub mod redirect_op {
    pub const WRITE: u8 = 0;
    pub const APPEND: u8 = 1;
    pub const READ: u8 = 2;
    pub const READ_WRITE: u8 = 3;
    pub const CLOBBER: u8 = 4;
    pub const DUP_READ: u8 = 5;
    pub const DUP_WRITE: u8 = 6;
    pub const WRITE_BOTH: u8 = 7;
    pub const APPEND_BOTH: u8 = 8;
}

/// Parameter expansion modifier types for `ExpandParam(u8)`
pub mod param_mod {
    pub const DEFAULT: u8 = 0; // ${var:-default}
    pub const ASSIGN: u8 = 1; // ${var:=default}
    pub const ERROR: u8 = 2; // ${var:?error}
    pub const ALTERNATE: u8 = 3; // ${var:+alternate}
    pub const LENGTH: u8 = 4; // ${#var}
    pub const STRIP_SHORT: u8 = 5; // ${var#pat}
    pub const STRIP_LONG: u8 = 6; // ${var##pat}
    pub const RSTRIP_SHORT: u8 = 7; // ${var%pat}
    pub const RSTRIP_LONG: u8 = 8; // ${var%%pat}
    pub const SUBST_FIRST: u8 = 9; // ${var/pat/rep}
    pub const SUBST_ALL: u8 = 10; // ${var//pat/rep}
    pub const UPPER: u8 = 11; // ${var^^}
    pub const LOWER: u8 = 12; // ${var,,}
    pub const UPPER_FIRST: u8 = 13; // ${var^}
    pub const LOWER_FIRST: u8 = 14; // ${var,}
    pub const INDIRECT: u8 = 15; // ${!var}
    pub const KEYS: u8 = 16; // ${!arr[@]}
    pub const SLICE: u8 = 17; // ${var:off:len}
}

impl Hash for Op {
    fn hash<H: Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            Op::LoadInt(n) => n.hash(state),
            Op::LoadFloat(f) => f.to_bits().hash(state),
            Op::LoadConst(idx) => idx.hash(state),
            Op::GetVar(idx)
            | Op::SetVar(idx)
            | Op::DeclareVar(idx)
            | Op::GetSlot(idx)
            | Op::SetSlot(idx)
            | Op::SlotArrayGet(idx)
            | Op::SlotArraySet(idx)
            | Op::GetArray(idx)
            | Op::SetArray(idx)
            | Op::DeclareArray(idx)
            | Op::ArrayGet(idx)
            | Op::ArraySet(idx)
            | Op::ArrayPush(idx)
            | Op::ArrayPop(idx)
            | Op::ArrayShift(idx)
            | Op::ArrayLen(idx)
            | Op::MakeArray(idx)
            | Op::GetHash(idx)
            | Op::SetHash(idx)
            | Op::DeclareHash(idx)
            | Op::HashGet(idx)
            | Op::HashSet(idx)
            | Op::HashDelete(idx)
            | Op::HashExists(idx)
            | Op::HashKeys(idx)
            | Op::HashValues(idx)
            | Op::MakeHash(idx)
            | Op::PreIncSlot(idx)
            | Op::PreIncSlotVoid(idx)
            | Op::HereDoc(idx)
            | Op::CmdSubst(idx)
            | Op::ProcessSubIn(idx)
            | Op::ProcessSubOut(idx)
            | Op::TrapSet(idx)
            | Op::MapBlock(idx)
            | Op::GrepBlock(idx)
            | Op::SortBlock(idx)
            | Op::ForEachBlock(idx) => idx.hash(state),
            Op::Jump(t)
            | Op::JumpIfTrue(t)
            | Op::JumpIfFalse(t)
            | Op::JumpIfTrueKeep(t)
            | Op::JumpIfFalseKeep(t) => t.hash(state),
            Op::Call(name, argc) => {
                name.hash(state);
                argc.hash(state);
            }
            Op::CallBuiltin(id, argc) => {
                id.hash(state);
                argc.hash(state);
            }
            Op::Extended(id, arg) => {
                id.hash(state);
                arg.hash(state);
            }
            Op::ExtendedWide(id, payload) => {
                id.hash(state);
                payload.hash(state);
            }
            Op::Print(n) | Op::PrintLn(n) | Op::Exec(n) | Op::ExecBg(n) | Op::PipelineBegin(n) => {
                n.hash(state)
            }
            Op::Redirect(fd, op) => {
                fd.hash(state);
                op.hash(state);
            }
            Op::TestFile(t) | Op::ExpandParam(t) => t.hash(state),
            Op::SlotLtIntJumpIfFalse(slot, limit, target) => {
                slot.hash(state);
                limit.hash(state);
                target.hash(state);
            }
            Op::SlotIncLtIntJumpBack(slot, limit, target) => {
                slot.hash(state);
                limit.hash(state);
                target.hash(state);
            }
            Op::AccumSumLoop(sum, i, limit) => {
                sum.hash(state);
                i.hash(state);
                limit.hash(state);
            }
            Op::ConcatConstLoop(c, s, i, limit) => {
                c.hash(state);
                s.hash(state);
                i.hash(state);
                limit.hash(state);
            }
            Op::PushIntRangeLoop(arr, i, limit) => {
                arr.hash(state);
                i.hash(state);
                limit.hash(state);
            }
            Op::AddAssignSlotVoid(a, b) => {
                a.hash(state);
                b.hash(state);
            }
            // Nullary ops — discriminant alone is sufficient
            Op::Nop
            | Op::LoadTrue
            | Op::LoadFalse
            | Op::LoadUndef
            | Op::Pop
            | Op::Dup
            | Op::Dup2
            | Op::Swap
            | Op::Rot
            | Op::Add
            | Op::Sub
            | Op::Mul
            | Op::Div
            | Op::Mod
            | Op::Pow
            | Op::Negate
            | Op::Inc
            | Op::Dec
            | Op::Concat
            | Op::StringRepeat
            | Op::StringLen
            | Op::NumEq
            | Op::NumNe
            | Op::NumLt
            | Op::NumGt
            | Op::NumLe
            | Op::NumGe
            | Op::Spaceship
            | Op::StrEq
            | Op::StrNe
            | Op::StrLt
            | Op::StrGt
            | Op::StrLe
            | Op::StrGe
            | Op::StrCmp
            | Op::LogNot
            | Op::LogAnd
            | Op::LogOr
            | Op::BitAnd
            | Op::BitOr
            | Op::BitXor
            | Op::BitNot
            | Op::Shl
            | Op::Shr
            | Op::Return
            | Op::ReturnValue
            | Op::PushFrame
            | Op::PopFrame
            | Op::ReadLine
            | Op::Range
            | Op::RangeStep
            | Op::SortDefault
            | Op::SetStatus
            | Op::GetStatus
            | Op::PipelineStage
            | Op::PipelineEnd
            | Op::HereString
            | Op::SubshellBegin
            | Op::SubshellEnd
            | Op::Glob
            | Op::GlobRecursive
            | Op::TrapCheck
            | Op::WordSplit
            | Op::BraceExpand
            | Op::TildeExpand => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_op_size() {
        // Ops should be reasonably small for cache-friendly dispatch
        assert!(
            std::mem::size_of::<Op>() <= 24,
            "Op too large: {} bytes",
            std::mem::size_of::<Op>()
        );
    }
}
