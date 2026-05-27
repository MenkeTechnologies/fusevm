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
    /// Slot-based array index: stack: \[index\], slot contains array → value
    SlotArrayGet(u16),
    /// Slot-based array set: stack: \[value, index\], slot contains array
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
    /// Call user-defined shell function by name pool index with N args.
    /// Falls through to host.call_function() then host.exec() if not found.
    /// stack: [arg_N, ..., arg_1] → pushes Status
    CallFunction(u16, u8),
    /// Glob-pattern match: pop pattern, pop string, push Bool.
    /// Used by `[[ x = pat ]]` and `case` arm matching.
    StrMatch,
    /// Regex match: pop regex, pop string, push Bool. (`=~`)
    RegexMatch,
    /// Begin scoped redirection block: u8 = number of redirects already
    /// applied via prior Redirect ops. Saves fd state on the host's stack.
    /// Used for `cmd > out.txt` applied to compound commands and
    /// `func() { ... } > out.txt`.
    WithRedirectsBegin(u8),
    /// End scoped redirection block — restore fd state.
    WithRedirectsEnd,
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
            Op::CallFunction(name, argc) => {
                name.hash(state);
                argc.hash(state);
            }
            Op::WithRedirectsBegin(n) => n.hash(state),
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
            | Op::TildeExpand
            | Op::StrMatch
            | Op::RegexMatch
            | Op::WithRedirectsEnd => {}
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

    #[test]
    fn equal_ops_hash_equal() {
        use std::collections::hash_map::DefaultHasher;
        let h = |op: &Op| {
            let mut hs = DefaultHasher::new();
            op.hash(&mut hs);
            hs.finish()
        };
        assert_eq!(h(&Op::LoadInt(42)), h(&Op::LoadInt(42)));
        assert_eq!(h(&Op::Jump(7)), h(&Op::Jump(7)));
        assert_eq!(h(&Op::Add), h(&Op::Add));
    }

    #[test]
    fn different_ops_typically_hash_differently() {
        use std::collections::hash_map::DefaultHasher;
        let h = |op: &Op| {
            let mut hs = DefaultHasher::new();
            op.hash(&mut hs);
            hs.finish()
        };
        assert_ne!(h(&Op::LoadInt(1)), h(&Op::LoadInt(2)));
        assert_ne!(h(&Op::Add), h(&Op::Sub));
        assert_ne!(h(&Op::Jump(0)), h(&Op::JumpIfTrue(0)));
    }

    #[test]
    fn float_load_hash_uses_bit_pattern() {
        // f64 must hash via bits — NaN, -0.0 etc. need to be hashable.
        use std::collections::hash_map::DefaultHasher;
        let h = |op: &Op| {
            let mut hs = DefaultHasher::new();
            op.hash(&mut hs);
            hs.finish()
        };
        let a = Op::LoadFloat(f64::NAN);
        let b = Op::LoadFloat(f64::NAN);
        // Same bit pattern → equal hash.
        assert_eq!(h(&a), h(&b));
        // +0.0 and -0.0 are == under PartialEq but have different bits;
        // their hashes will differ — verify the impl is bit-based.
        assert_ne!(h(&Op::LoadFloat(0.0)), h(&Op::LoadFloat(-0.0)));
    }

    #[test]
    fn partialeq_works_for_payloaded_ops() {
        assert_eq!(Op::LoadInt(1), Op::LoadInt(1));
        assert_ne!(Op::LoadInt(1), Op::LoadInt(2));
        assert_eq!(Op::Jump(5), Op::Jump(5));
        assert_ne!(Op::Jump(5), Op::Jump(6));
    }

    #[test]
    fn op_clone_is_value_equal() {
        let a = Op::LoadInt(123);
        let b = a.clone();
        assert_eq!(a, b);
    }

    #[test]
    fn op_serde_roundtrip() {
        // Verify a representative selection of ops survive serde JSON roundtrip.
        let cases = vec![
            Op::Nop,
            Op::LoadInt(-5),
            Op::LoadFloat(1.5),
            Op::Add,
            Op::Jump(42),
            Op::GetSlot(7),
        ];
        for op in cases {
            let s = serde_json::to_string(&op).unwrap();
            let back: Op = serde_json::from_str(&s).unwrap();
            assert_eq!(op, back);
        }
    }

    use std::collections::hash_map::DefaultHasher;

    fn hash_of(op: &Op) -> u64 {
        let mut h = DefaultHasher::new();
        op.hash(&mut h);
        h.finish()
    }

    // ─── Hash impl coverage: every variant arm must produce consistent
    //     hashes and equal-ops-hash-equal regardless of payload class ──

    #[test]
    fn nullary_arithmetic_ops_hash_consistently() {
        for op in [Op::Add, Op::Sub, Op::Mul, Op::Div, Op::Mod, Op::Pow] {
            assert_eq!(hash_of(&op), hash_of(&op.clone()));
        }
    }

    #[test]
    fn distinct_nullary_ops_hash_differently() {
        // Discriminant alone differentiates these. Collisions are theoretically
        // allowed by Hash but DefaultHasher is unlikely to collide on so few.
        let h_add = hash_of(&Op::Add);
        let h_sub = hash_of(&Op::Sub);
        let h_mul = hash_of(&Op::Mul);
        assert_ne!(h_add, h_sub);
        assert_ne!(h_add, h_mul);
        assert_ne!(h_sub, h_mul);
    }

    #[test]
    fn same_idx_in_different_u16_arms_does_not_collide_on_discriminant() {
        // GetVar(5) and SetVar(5) share the payload value but differ in
        // discriminant — must hash differently.
        let h_get = hash_of(&Op::GetVar(5));
        let h_set = hash_of(&Op::SetVar(5));
        assert_ne!(h_get, h_set);
    }

    #[test]
    fn jump_targets_hash_independent_of_call_targets() {
        // Jump(7) and Call(7, 0) share the literal value 7 but should never
        // collide because discriminants differ.
        let h_jump = hash_of(&Op::Jump(7));
        let h_call = hash_of(&Op::Call(7, 0));
        assert_ne!(h_jump, h_call);
    }

    #[test]
    fn call_argc_changes_hash() {
        let h_zero_arg = hash_of(&Op::Call(10, 0));
        let h_two_args = hash_of(&Op::Call(10, 2));
        assert_ne!(h_zero_arg, h_two_args);
    }

    #[test]
    fn float_load_distinct_values_hash_differently() {
        // f64 → to_bits → hash. 1.0 and 2.0 have distinct bit patterns.
        let h_one = hash_of(&Op::LoadFloat(1.0));
        let h_two = hash_of(&Op::LoadFloat(2.0));
        assert_ne!(h_one, h_two);
    }

    #[test]
    fn float_load_neg_zero_and_pos_zero_hash_differently() {
        // 0.0 and -0.0 are == under PartialEq but have different bit patterns,
        // so the bit-based Hash impl produces different hashes. This is consistent
        // with serde_json roundtrip semantics for the constant pool.
        let h_pos = hash_of(&Op::LoadFloat(0.0));
        let h_neg = hash_of(&Op::LoadFloat(-0.0));
        assert_ne!(h_pos, h_neg, "+0.0 and -0.0 have distinct bit patterns");
    }

    #[test]
    fn redirect_op_uses_both_fd_and_op_in_hash() {
        let a = Op::Redirect(0, 1);
        let b = Op::Redirect(0, 2);
        let c = Op::Redirect(1, 1);
        assert_ne!(hash_of(&a), hash_of(&b), "second field changes hash");
        assert_ne!(hash_of(&a), hash_of(&c), "first field changes hash");
    }

    #[test]
    fn extended_uses_both_id_and_arg_in_hash() {
        let a = Op::Extended(1, 2);
        let b = Op::Extended(1, 3);
        let c = Op::Extended(2, 2);
        assert_ne!(hash_of(&a), hash_of(&b));
        assert_ne!(hash_of(&a), hash_of(&c));
    }

    #[test]
    fn three_field_loop_ops_use_all_fields() {
        let base = Op::SlotLtIntJumpIfFalse(1, 10, 100);
        let diff_slot = Op::SlotLtIntJumpIfFalse(2, 10, 100);
        let diff_limit = Op::SlotLtIntJumpIfFalse(1, 99, 100);
        let diff_target = Op::SlotLtIntJumpIfFalse(1, 10, 200);
        assert_ne!(hash_of(&base), hash_of(&diff_slot));
        assert_ne!(hash_of(&base), hash_of(&diff_limit));
        assert_ne!(hash_of(&base), hash_of(&diff_target));
    }

    #[test]
    fn equal_loadint_payloads_hash_equal() {
        // Same payload → same hash, no matter how the value was constructed.
        let a = Op::LoadInt(-42);
        let b = Op::LoadInt(-42);
        assert_eq!(hash_of(&a), hash_of(&b));
    }

    #[test]
    fn pop_dup_swap_rot_are_each_unique() {
        // Common stack ops — discriminant alone differentiates.
        let pop = hash_of(&Op::Pop);
        let dup = hash_of(&Op::Dup);
        let swap = hash_of(&Op::Swap);
        let rot = hash_of(&Op::Rot);
        let set: std::collections::HashSet<_> = [pop, dup, swap, rot].iter().copied().collect();
        assert_eq!(set.len(), 4, "all four nullary stack ops are distinct");
    }

    // ─── Serde round-trip extension: more payload-carrying ops ────────

    #[test]
    fn serde_roundtrip_payload_ops() {
        let cases = vec![
            Op::Call(100, 3),
            Op::CallBuiltin(0, 1),
            Op::Redirect(2, 5),
            Op::Extended(7, 9),
            Op::SlotLtIntJumpIfFalse(1, 10, 200),
        ];
        for op in cases {
            let s = serde_json::to_string(&op).expect("serialize");
            let back: Op = serde_json::from_str(&s).expect("deserialize");
            assert_eq!(op, back);
        }
    }

    #[test]
    fn serde_roundtrip_float_special_values() {
        // Special-case floats: NaN doesn't round-trip via PartialEq, so
        // only check finite ones.
        for f in [0.0, -0.0, 1.5, -1.5, f64::MIN, f64::MAX] {
            let op = Op::LoadFloat(f);
            let s = serde_json::to_string(&op).expect("ser");
            let back: Op = serde_json::from_str(&s).expect("de");
            assert_eq!(op, back, "roundtrip {f}");
        }
    }

    // ─── tests for the `block` constants module ───────────────────────

    #[test]
    fn param_mod_constants_are_unique_and_within_u8() {
        // Each ExpandParam modifier maps to a distinct u8 op-code; verify no
        // collisions on the 18-value table.
        let names = [
            param_mod::DEFAULT,
            param_mod::ASSIGN,
            param_mod::ERROR,
            param_mod::ALTERNATE,
            param_mod::LENGTH,
            param_mod::STRIP_SHORT,
            param_mod::STRIP_LONG,
            param_mod::RSTRIP_SHORT,
            param_mod::RSTRIP_LONG,
            param_mod::SUBST_FIRST,
            param_mod::SUBST_ALL,
            param_mod::UPPER,
            param_mod::LOWER,
            param_mod::UPPER_FIRST,
            param_mod::LOWER_FIRST,
            param_mod::INDIRECT,
            param_mod::KEYS,
            param_mod::SLICE,
        ];
        let set: std::collections::HashSet<_> = names.iter().copied().collect();
        assert_eq!(set.len(), names.len(), "param_mod constants must be unique");
    }
}
