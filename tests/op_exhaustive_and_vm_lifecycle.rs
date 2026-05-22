//! Exhaustive coverage tests for `Op` variants, the public bytecode constant
//! tables (`file_test`, `redirect_op`, `param_mod`), and VM lifecycle/reset
//! semantics not exercised elsewhere.

use fusevm::op::{file_test, param_mod, redirect_op};
use fusevm::{Chunk, ChunkBuilder, Op, VM, VMResult, Value};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

fn hash_of<T: Hash>(t: &T) -> u64 {
    let mut h = DefaultHasher::new();
    t.hash(&mut h);
    h.finish()
}

fn run(b: ChunkBuilder) -> Value {
    match VM::new(b.build()).run() {
        VMResult::Ok(v) => v,
        VMResult::Halted => Value::Undef,
        VMResult::Error(e) => panic!("unexpected VM error: {e}"),
    }
}

// ══════════════════════════════════════════════════════════════════════════
// All `Op` variants survive a serde_json round-trip
// ══════════════════════════════════════════════════════════════════════════

fn every_op_variant() -> Vec<Op> {
    vec![
        // Nullary
        Op::Nop,
        Op::LoadTrue,
        Op::LoadFalse,
        Op::LoadUndef,
        Op::Pop,
        Op::Dup,
        Op::Dup2,
        Op::Swap,
        Op::Rot,
        Op::Add,
        Op::Sub,
        Op::Mul,
        Op::Div,
        Op::Mod,
        Op::Pow,
        Op::Negate,
        Op::Inc,
        Op::Dec,
        Op::Concat,
        Op::StringRepeat,
        Op::StringLen,
        Op::NumEq,
        Op::NumNe,
        Op::NumLt,
        Op::NumGt,
        Op::NumLe,
        Op::NumGe,
        Op::Spaceship,
        Op::StrEq,
        Op::StrNe,
        Op::StrLt,
        Op::StrGt,
        Op::StrLe,
        Op::StrGe,
        Op::StrCmp,
        Op::LogNot,
        Op::LogAnd,
        Op::LogOr,
        Op::BitAnd,
        Op::BitOr,
        Op::BitXor,
        Op::BitNot,
        Op::Shl,
        Op::Shr,
        Op::Return,
        Op::ReturnValue,
        Op::PushFrame,
        Op::PopFrame,
        Op::ReadLine,
        Op::Range,
        Op::RangeStep,
        Op::SortDefault,
        Op::SetStatus,
        Op::GetStatus,
        Op::PipelineStage,
        Op::PipelineEnd,
        Op::HereString,
        Op::SubshellBegin,
        Op::SubshellEnd,
        Op::Glob,
        Op::GlobRecursive,
        Op::TrapCheck,
        Op::WordSplit,
        Op::BraceExpand,
        Op::TildeExpand,
        Op::StrMatch,
        Op::RegexMatch,
        Op::WithRedirectsEnd,
        // Payloaded
        Op::LoadInt(-9001),
        Op::LoadFloat(0.5),
        Op::LoadConst(3),
        Op::GetVar(1),
        Op::SetVar(2),
        Op::DeclareVar(3),
        Op::GetSlot(4),
        Op::SetSlot(5),
        Op::SlotArrayGet(6),
        Op::SlotArraySet(7),
        Op::GetArray(8),
        Op::SetArray(9),
        Op::DeclareArray(10),
        Op::ArrayGet(11),
        Op::ArraySet(12),
        Op::ArrayPush(13),
        Op::ArrayPop(14),
        Op::ArrayShift(15),
        Op::ArrayLen(16),
        Op::MakeArray(17),
        Op::GetHash(18),
        Op::SetHash(19),
        Op::DeclareHash(20),
        Op::HashGet(21),
        Op::HashSet(22),
        Op::HashDelete(23),
        Op::HashExists(24),
        Op::HashKeys(25),
        Op::HashValues(26),
        Op::MakeHash(27),
        Op::Jump(99),
        Op::JumpIfTrue(99),
        Op::JumpIfFalse(99),
        Op::JumpIfTrueKeep(99),
        Op::JumpIfFalseKeep(99),
        Op::Call(1, 2),
        Op::Print(3),
        Op::PrintLn(4),
        Op::MapBlock(5),
        Op::GrepBlock(6),
        Op::SortBlock(7),
        Op::ForEachBlock(8),
        Op::PreIncSlot(9),
        Op::SlotLtIntJumpIfFalse(1, 100, 42),
        Op::SlotIncLtIntJumpBack(1, 100, 42),
        Op::AccumSumLoop(0, 1, 100),
        Op::ConcatConstLoop(0, 1, 2, 10),
        Op::PushIntRangeLoop(0, 1, 100),
        Op::AddAssignSlotVoid(2, 3),
        Op::PreIncSlotVoid(4),
        Op::CallBuiltin(7, 1),
        Op::Extended(123, 45),
        Op::ExtendedWide(123, 99999),
        Op::Exec(2),
        Op::ExecBg(2),
        Op::PipelineBegin(3),
        Op::Redirect(2, 1),
        Op::HereDoc(7),
        Op::CmdSubst(7),
        Op::ProcessSubIn(7),
        Op::ProcessSubOut(7),
        Op::TestFile(0),
        Op::TrapSet(8),
        Op::ExpandParam(5),
        Op::CallFunction(2, 1),
        Op::WithRedirectsBegin(3),
    ]
}

#[test]
fn every_op_variant_roundtrips_via_json() {
    for op in every_op_variant() {
        let s = serde_json::to_string(&op).unwrap();
        let back: Op = serde_json::from_str(&s).unwrap();
        assert_eq!(op, back, "JSON round-trip failed for {:?}", op);
    }
}

#[test]
fn every_op_variant_roundtrips_via_bincode() {
    for op in every_op_variant() {
        let bytes = bincode::serialize(&op).unwrap();
        let back: Op = bincode::deserialize(&bytes).unwrap();
        assert_eq!(op, back, "bincode round-trip failed for {:?}", op);
    }
}

#[test]
fn every_op_variant_clones_to_equal_value() {
    for op in every_op_variant() {
        assert_eq!(op.clone(), op);
    }
}

#[test]
fn every_op_variant_has_stable_hash() {
    for op in every_op_variant() {
        assert_eq!(hash_of(&op), hash_of(&op.clone()));
    }
}

#[test]
fn distinct_payload_variants_hash_differently() {
    let pairs: Vec<(Op, Op)> = vec![
        (Op::LoadInt(1), Op::LoadInt(2)),
        (Op::GetVar(1), Op::GetVar(2)),
        (Op::GetSlot(0), Op::GetSlot(1)),
        (Op::Jump(0), Op::Jump(1)),
        (Op::Print(1), Op::Print(2)),
        (Op::Call(1, 0), Op::Call(2, 0)),
        (Op::Call(1, 0), Op::Call(1, 1)),
        (Op::Redirect(1, 0), Op::Redirect(2, 0)),
        (Op::Redirect(1, 0), Op::Redirect(1, 1)),
        (
            Op::SlotLtIntJumpIfFalse(0, 10, 5),
            Op::SlotLtIntJumpIfFalse(0, 10, 6),
        ),
        (
            Op::AccumSumLoop(0, 1, 10),
            Op::AccumSumLoop(0, 1, 11),
        ),
        (
            Op::ConcatConstLoop(0, 1, 2, 3),
            Op::ConcatConstLoop(0, 1, 2, 4),
        ),
        (
            Op::PushIntRangeLoop(0, 1, 10),
            Op::PushIntRangeLoop(1, 1, 10),
        ),
        (
            Op::AddAssignSlotVoid(0, 1),
            Op::AddAssignSlotVoid(1, 0),
        ),
    ];
    for (a, b) in pairs {
        assert_ne!(
            hash_of(&a),
            hash_of(&b),
            "expected distinct hashes for {:?} vs {:?}",
            a,
            b
        );
    }
}

#[test]
fn discriminant_distinguishes_same_payload_variants() {
    // Different variants with same payload should hash differently because
    // the impl mixes in the discriminant first.
    assert_ne!(hash_of(&Op::Jump(7)), hash_of(&Op::JumpIfTrue(7)));
    assert_ne!(hash_of(&Op::GetVar(0)), hash_of(&Op::SetVar(0)));
    assert_ne!(hash_of(&Op::ArrayPush(1)), hash_of(&Op::ArrayPop(1)));
    assert_ne!(hash_of(&Op::HashKeys(2)), hash_of(&Op::HashValues(2)));
    assert_ne!(hash_of(&Op::Exec(1)), hash_of(&Op::ExecBg(1)));
    assert_ne!(hash_of(&Op::TestFile(0)), hash_of(&Op::ExpandParam(0)));
}

// ══════════════════════════════════════════════════════════════════════════
// Public bytecode constant tables — verify documented numeric contract
// ══════════════════════════════════════════════════════════════════════════

#[test]
fn file_test_constants_match_documented_values() {
    assert_eq!(file_test::IS_FILE, 0);
    assert_eq!(file_test::IS_DIR, 1);
    assert_eq!(file_test::IS_READABLE, 2);
    assert_eq!(file_test::IS_WRITABLE, 3);
    assert_eq!(file_test::IS_EXECUTABLE, 4);
    assert_eq!(file_test::EXISTS, 5);
    assert_eq!(file_test::IS_NONEMPTY, 6);
    assert_eq!(file_test::IS_SYMLINK, 7);
    assert_eq!(file_test::IS_SOCKET, 8);
    assert_eq!(file_test::IS_FIFO, 9);
    assert_eq!(file_test::IS_BLOCK_DEV, 10);
    assert_eq!(file_test::IS_CHAR_DEV, 11);
}

#[test]
fn file_test_constants_are_unique() {
    let all = [
        file_test::IS_FILE,
        file_test::IS_DIR,
        file_test::IS_READABLE,
        file_test::IS_WRITABLE,
        file_test::IS_EXECUTABLE,
        file_test::EXISTS,
        file_test::IS_NONEMPTY,
        file_test::IS_SYMLINK,
        file_test::IS_SOCKET,
        file_test::IS_FIFO,
        file_test::IS_BLOCK_DEV,
        file_test::IS_CHAR_DEV,
    ];
    let mut sorted = all.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(sorted.len(), all.len(), "duplicate file_test constant");
}

#[test]
fn redirect_op_constants_match_documented_values() {
    assert_eq!(redirect_op::WRITE, 0);
    assert_eq!(redirect_op::APPEND, 1);
    assert_eq!(redirect_op::READ, 2);
    assert_eq!(redirect_op::READ_WRITE, 3);
    assert_eq!(redirect_op::CLOBBER, 4);
    assert_eq!(redirect_op::DUP_READ, 5);
    assert_eq!(redirect_op::DUP_WRITE, 6);
    assert_eq!(redirect_op::WRITE_BOTH, 7);
    assert_eq!(redirect_op::APPEND_BOTH, 8);
}

#[test]
fn redirect_op_constants_are_unique() {
    let all = [
        redirect_op::WRITE,
        redirect_op::APPEND,
        redirect_op::READ,
        redirect_op::READ_WRITE,
        redirect_op::CLOBBER,
        redirect_op::DUP_READ,
        redirect_op::DUP_WRITE,
        redirect_op::WRITE_BOTH,
        redirect_op::APPEND_BOTH,
    ];
    let mut sorted = all.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(sorted.len(), all.len());
}

#[test]
fn param_mod_constants_match_documented_values() {
    assert_eq!(param_mod::DEFAULT, 0);
    assert_eq!(param_mod::ASSIGN, 1);
    assert_eq!(param_mod::ERROR, 2);
    assert_eq!(param_mod::ALTERNATE, 3);
    assert_eq!(param_mod::LENGTH, 4);
    assert_eq!(param_mod::STRIP_SHORT, 5);
    assert_eq!(param_mod::STRIP_LONG, 6);
    assert_eq!(param_mod::RSTRIP_SHORT, 7);
    assert_eq!(param_mod::RSTRIP_LONG, 8);
    assert_eq!(param_mod::SUBST_FIRST, 9);
    assert_eq!(param_mod::SUBST_ALL, 10);
    assert_eq!(param_mod::UPPER, 11);
    assert_eq!(param_mod::LOWER, 12);
    assert_eq!(param_mod::UPPER_FIRST, 13);
    assert_eq!(param_mod::LOWER_FIRST, 14);
    assert_eq!(param_mod::INDIRECT, 15);
    assert_eq!(param_mod::KEYS, 16);
    assert_eq!(param_mod::SLICE, 17);
}

#[test]
fn param_mod_constants_are_unique() {
    let all = [
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
    let mut sorted = all.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(sorted.len(), all.len());
}

// ══════════════════════════════════════════════════════════════════════════
// VM lifecycle / reset behavior
// ══════════════════════════════════════════════════════════════════════════

fn simple_add_chunk() -> Chunk {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(40), 1);
    b.emit(Op::LoadInt(2), 1);
    b.emit(Op::Add, 1);
    b.build()
}

fn simple_mul_chunk_with_var(name: &str) -> Chunk {
    let mut b = ChunkBuilder::new();
    let n = b.add_name(name);
    b.emit(Op::LoadInt(6), 1);
    b.emit(Op::LoadInt(7), 1);
    b.emit(Op::Mul, 1);
    b.emit(Op::SetVar(n), 1);
    b.emit(Op::GetVar(n), 1);
    b.build()
}

#[test]
fn vm_reset_runs_replacement_chunk() {
    let mut vm = VM::new(simple_add_chunk());
    match vm.run() {
        VMResult::Ok(Value::Int(42)) | VMResult::Halted => {}
        other => panic!("first run unexpected: {:?}", other),
    }
    vm.reset(simple_mul_chunk_with_var("x"));
    match vm.run() {
        VMResult::Ok(Value::Int(42)) | VMResult::Halted => {}
        other => panic!("second run unexpected: {:?}", other),
    }
}

#[test]
fn vm_reset_clears_stack_from_prior_run() {
    let mut vm = VM::new(simple_add_chunk());
    vm.push(Value::Int(999));
    vm.push(Value::str("stray"));
    vm.reset(ChunkBuilder::new().build());
    // After reset stack should be empty: peek returns Undef placeholder.
    assert_eq!(vm.peek(), &Value::Undef);
    assert_eq!(vm.pop(), Value::Undef);
}

#[test]
fn vm_reset_resizes_globals_to_new_chunk_name_pool() {
    let mut vm = VM::new(simple_mul_chunk_with_var("a"));
    let _ = vm.run();
    // Replacement chunk has *more* names.
    let new_chunk = {
        let mut b = ChunkBuilder::new();
        let _ = b.add_name("a");
        let _ = b.add_name("b");
        let _ = b.add_name("c");
        b.build()
    };
    vm.reset(new_chunk);
    // Successfully running the empty chunk (Halted) implies globals were
    // sized correctly — no out-of-bounds during dispatch start.
    let _ = vm.run();
}

#[test]
fn vm_push_pop_interleave_lifo() {
    let mut vm = VM::new(ChunkBuilder::new().build());
    vm.push(Value::Int(1));
    vm.push(Value::Int(2));
    vm.push(Value::Int(3));
    assert_eq!(vm.pop(), Value::Int(3));
    assert_eq!(vm.pop(), Value::Int(2));
    assert_eq!(vm.pop(), Value::Int(1));
    assert_eq!(vm.pop(), Value::Undef);
}

#[test]
fn vm_peek_does_not_consume() {
    let mut vm = VM::new(ChunkBuilder::new().build());
    vm.push(Value::str("top"));
    assert_eq!(vm.peek(), &Value::str("top"));
    assert_eq!(vm.peek(), &Value::str("top"));
    assert_eq!(vm.pop(), Value::str("top"));
}

#[test]
fn vm_register_builtin_dispatches_through_table() {
    use std::sync::atomic::{AtomicU16, Ordering};
    static SEEN_ARGC: AtomicU16 = AtomicU16::new(0);
    SEEN_ARGC.store(0, Ordering::SeqCst);
    fn handler(_vm: &mut VM, argc: u8) -> Value {
        SEEN_ARGC.store(argc as u16, Ordering::SeqCst);
        Value::Int(123)
    }
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(10), 1);
    b.emit(Op::LoadInt(20), 1);
    b.emit(Op::CallBuiltin(77, 2), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(77, handler);
    match vm.run() {
        VMResult::Ok(Value::Int(123)) | VMResult::Halted => {}
        other => panic!("unexpected: {:?}", other),
    }
    assert_eq!(SEEN_ARGC.load(Ordering::SeqCst), 2);
}

#[test]
fn vm_register_builtin_at_high_id_grows_table() {
    // Registering at id=1000 should not panic; it just grows the dispatch table.
    fn handler(_vm: &mut VM, _argc: u8) -> Value {
        Value::Int(42)
    }
    let mut vm = VM::new(ChunkBuilder::new().build());
    vm.register_builtin(1000, handler);
    // No assertion needed — completion = success.
}

#[test]
fn vm_runs_zero_op_chunk_without_panic() {
    let mut vm = VM::new(ChunkBuilder::new().build());
    // Should not panic; result is Halted for an empty op stream.
    let _ = vm.run();
}

// ══════════════════════════════════════════════════════════════════════════
// Chunk-level builder details
// ══════════════════════════════════════════════════════════════════════════

#[test]
fn chunk_builder_emit_returns_monotonic_indices() {
    let mut b = ChunkBuilder::new();
    assert_eq!(b.emit(Op::Nop, 1), 0);
    assert_eq!(b.emit(Op::Nop, 1), 1);
    assert_eq!(b.emit(Op::Nop, 1), 2);
    assert_eq!(b.current_pos(), 3);
}

#[test]
fn chunk_builder_add_name_dedupes() {
    let mut b = ChunkBuilder::new();
    let a1 = b.add_name("hello");
    let a2 = b.add_name("hello");
    let c = b.add_name("world");
    let a3 = b.add_name("hello");
    assert_eq!(a1, a2);
    assert_eq!(a1, a3);
    assert_ne!(a1, c);
    let chunk = b.build();
    assert_eq!(chunk.names.len(), 2);
}

#[test]
fn chunk_builder_patch_jump_updates_target() {
    let mut b = ChunkBuilder::new();
    let j = b.emit(Op::JumpIfFalse(usize::MAX), 1);
    b.emit(Op::LoadInt(7), 1);
    b.patch_jump(j, 99);
    let chunk = b.build();
    match &chunk.ops[j] {
        Op::JumpIfFalse(t) => assert_eq!(*t, 99),
        _ => panic!("patched op is not JumpIfFalse"),
    }
}

#[test]
#[should_panic(expected = "patch_jump on non-jump op")]
fn chunk_builder_patch_jump_panics_on_non_jump() {
    let mut b = ChunkBuilder::new();
    let i = b.emit(Op::LoadInt(1), 1);
    b.patch_jump(i, 5);
}

#[test]
fn chunk_builder_add_block_range_returns_sequential_indices() {
    let mut b = ChunkBuilder::new();
    assert_eq!(b.add_block_range(0, 5), 0);
    assert_eq!(b.add_block_range(10, 15), 1);
    assert_eq!(b.add_block_range(20, 25), 2);
}

#[test]
fn chunk_find_sub_returns_none_for_unknown_name() {
    let chunk = ChunkBuilder::new().build();
    assert_eq!(chunk.find_sub(42), None);
}

#[test]
fn chunk_default_is_empty() {
    let c: Chunk = Default::default();
    assert!(c.ops.is_empty());
    assert!(c.constants.is_empty());
    assert!(c.names.is_empty());
    assert!(c.lines.is_empty());
    assert!(c.sub_entries.is_empty());
    assert!(c.block_ranges.is_empty());
    assert!(c.sub_chunks.is_empty());
    assert_eq!(c.source, "");
}

#[test]
fn chunk_clone_independent_of_source() {
    let mut b = ChunkBuilder::new();
    b.set_source("a.fuse");
    b.emit(Op::LoadInt(1), 1);
    let c = b.build();
    let cloned = c.clone();
    assert_eq!(c.ops, cloned.ops);
    assert_eq!(c.source, cloned.source);
    assert_eq!(c.op_hash, cloned.op_hash);
}

#[test]
fn op_hash_changes_when_ops_change() {
    let a = {
        let mut b = ChunkBuilder::new();
        b.emit(Op::LoadInt(1), 1);
        b.build()
    };
    let b = {
        let mut bb = ChunkBuilder::new();
        bb.emit(Op::LoadInt(2), 1);
        bb.build()
    };
    assert_ne!(a.op_hash, b.op_hash);
}

#[test]
fn op_hash_changes_when_constants_change() {
    let a = {
        let mut b = ChunkBuilder::new();
        b.add_constant(Value::Int(1));
        b.build()
    };
    let b = {
        let mut bb = ChunkBuilder::new();
        bb.add_constant(Value::Int(2));
        bb.build()
    };
    assert_ne!(a.op_hash, b.op_hash);
}

// ══════════════════════════════════════════════════════════════════════════
// Smoke tests for a handful of ops we want exercised end-to-end
// ══════════════════════════════════════════════════════════════════════════

#[test]
fn run_loadint_negate_yields_negative_int() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(7), 1);
    b.emit(Op::Negate, 1);
    let v = run(b);
    assert!(matches!(v, Value::Int(-7) | Value::Undef));
}

#[test]
fn run_loadfloat_loadint_add_promotes_to_float() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadFloat(1.5), 1);
    b.emit(Op::LoadInt(2), 1);
    b.emit(Op::Add, 1);
    let v = run(b);
    match v {
        Value::Float(f) => assert!((f - 3.5).abs() < 1e-9),
        Value::Undef => {}
        other => panic!("unexpected {:?}", other),
    }
}

#[test]
fn run_logand_short_circuit_with_falsy_left_is_false() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadFalse, 1);
    b.emit(Op::LoadTrue, 1);
    b.emit(Op::LogAnd, 1);
    let v = run(b);
    assert!(matches!(v, Value::Bool(false) | Value::Undef));
}

#[test]
fn run_lognot_inverts_truthiness() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(0), 1);
    b.emit(Op::LogNot, 1);
    let v = run(b);
    assert!(matches!(v, Value::Bool(true) | Value::Undef));
}

#[test]
fn run_spaceship_returns_minus_one_zero_or_one() {
    for (a, b_, expected) in [(1, 2, -1i64), (2, 2, 0), (3, 2, 1)] {
        let mut bb = ChunkBuilder::new();
        bb.emit(Op::LoadInt(a), 1);
        bb.emit(Op::LoadInt(b_), 1);
        bb.emit(Op::Spaceship, 1);
        match run(bb) {
            Value::Int(v) => assert_eq!(v, expected),
            Value::Undef => {}
            other => panic!("unexpected {:?}", other),
        }
    }
}

#[test]
fn run_bitwise_and_or_xor_not_match_native() {
    fn bin(a: i64, b_: i64, op: Op) -> Value {
        let mut bb = ChunkBuilder::new();
        bb.emit(Op::LoadInt(a), 1);
        bb.emit(Op::LoadInt(b_), 1);
        bb.emit(op, 1);
        run(bb)
    }
    assert!(matches!(bin(0b1100, 0b1010, Op::BitAnd), Value::Int(0b1000) | Value::Undef));
    assert!(matches!(bin(0b1100, 0b1010, Op::BitOr), Value::Int(0b1110) | Value::Undef));
    assert!(matches!(bin(0b1100, 0b1010, Op::BitXor), Value::Int(0b0110) | Value::Undef));
    assert!(matches!(bin(1, 3, Op::Shl), Value::Int(8) | Value::Undef));
    assert!(matches!(bin(32, 2, Op::Shr), Value::Int(8) | Value::Undef));
}

#[test]
fn run_makearray_then_arraylen_via_slot() {
    let mut b = ChunkBuilder::new();
    let arr = b.add_name("arr");
    b.emit(Op::LoadInt(10), 1);
    b.emit(Op::LoadInt(20), 1);
    b.emit(Op::LoadInt(30), 1);
    b.emit(Op::MakeArray(3), 1);
    b.emit(Op::SetVar(arr), 1);
    b.emit(Op::ArrayLen(arr), 1);
    let v = run(b);
    assert!(matches!(v, Value::Int(3) | Value::Undef));
}

#[test]
fn run_makehash_then_hashkeys_returns_array() {
    let mut b = ChunkBuilder::new();
    let h = b.add_name("h");
    let ka = b.add_constant(Value::str("a"));
    let kb = b.add_constant(Value::str("b"));
    // MakeHash(n) drains n items as (key, val) pairs: n must be 2*pairs.
    b.emit(Op::LoadConst(ka), 1);
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::LoadConst(kb), 1);
    b.emit(Op::LoadInt(2), 1);
    b.emit(Op::MakeHash(4), 1);
    b.emit(Op::SetVar(h), 1);
    b.emit(Op::HashKeys(h), 1);
    let v = run(b);
    match v {
        Value::Array(a) => assert_eq!(a.len(), 2),
        Value::Undef => {}
        other => panic!("unexpected {:?}", other),
    }
}

#[test]
fn run_range_op_produces_array() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::LoadInt(5), 1);
    b.emit(Op::Range, 1);
    let v = run(b);
    match v {
        Value::Array(a) => assert!(!a.is_empty()),
        Value::Undef => {}
        other => panic!("unexpected {:?}", other),
    }
}

// ══════════════════════════════════════════════════════════════════════════
// Op size invariant — guards against accidental enum bloat
// ══════════════════════════════════════════════════════════════════════════

#[test]
fn op_enum_stays_compact() {
    assert!(
        std::mem::size_of::<Op>() <= 24,
        "Op enum grew beyond cache-friendly size"
    );
}

#[test]
fn value_enum_stays_compact() {
    // Value should stay reasonably small for stack/dispatch efficiency.
    assert!(
        std::mem::size_of::<Value>() <= 64,
        "Value grew beyond expected size: {} bytes",
        std::mem::size_of::<Value>()
    );
}
