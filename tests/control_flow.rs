//! Additional integration tests focusing on under-covered control-flow,
//! shell ops with the default host, and Op::Call subroutine dispatch.

use fusevm::op::file_test;
use fusevm::{ChunkBuilder, Op, VMResult, Value, VM};

fn run(b: ChunkBuilder) -> VMResult {
    VM::new(b.build()).run()
}

// ── JumpIfTrueKeep / JumpIfFalseKeep ────────────────────────────────────────

#[test]
fn jump_if_true_keep_preserves_top_when_taken() {
    // Layout:
    //   0: LoadInt(7)
    //   1: JumpIfTrueKeep(3)  ; 7 is truthy → jump, leave 7 on stack
    //   2: LoadInt(999)        ; skipped
    //   3: <end>               ; stack top = 7
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(7), 1);
    b.emit(Op::JumpIfTrueKeep(3), 1);
    b.emit(Op::LoadInt(999), 1);
    match run(b) {
        VMResult::Ok(Value::Int(7)) => {}
        other => panic!("got {:?}", other),
    }
}

#[test]
fn jump_if_true_keep_falls_through_with_falsy() {
    // 0 is falsy → no jump → LoadInt(42) executes → top is 42 (but 0 is still
    // below it; final stack top is 42 which VMResult::Ok returns).
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(0), 1);
    b.emit(Op::JumpIfTrueKeep(3), 1);
    b.emit(Op::LoadInt(42), 1);
    match run(b) {
        VMResult::Ok(Value::Int(42)) => {}
        other => panic!("got {:?}", other),
    }
}

#[test]
fn jump_if_false_keep_short_circuits_or_chain() {
    // Simulate `7 || 99`: push 7, JumpIfFalseKeep(skip) — since 7 is truthy,
    // it does NOT jump, then we pop and push 99... but for OR semantics,
    // JumpIfTrueKeep would be used instead. We test the falsy path: push 0,
    // JumpIfFalseKeep → jumps past the discard+push, leaving 0 on stack.
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(0), 1);
    b.emit(Op::JumpIfFalseKeep(4), 1);
    b.emit(Op::Pop, 1);
    b.emit(Op::LoadInt(99), 1);
    match run(b) {
        VMResult::Ok(Value::Int(0)) => {}
        other => panic!("got {:?}", other),
    }
}

// ── Op::Call subroutine dispatch ────────────────────────────────────────────

#[test]
fn call_unknown_function_returns_error() {
    let mut b = ChunkBuilder::new();
    let n = b.add_name("not_defined");
    b.emit(Op::Call(n, 0), 1);
    match run(b) {
        VMResult::Error(msg) => assert!(msg.contains("undefined function")),
        other => panic!("got {:?}", other),
    }
}

#[test]
fn call_invokes_registered_subroutine_and_returns_value() {
    // Layout:
    //   0: Call("twice", 1)     ; with arg 21
    //   1: Return                ; main script ends
    //   2: <sub "twice">: LoadInt(2); Mul; ReturnValue
    //
    // But Call expects argc args already on stack BEFORE the call. So:
    //   0: LoadInt(21)
    //   1: Call("twice", 1)
    //   2: Return
    //   3: <sub entry>: GetSlot(0)? — actually args are at stack_base.
    //
    // Looking at the VM: Call sets stack_base = stack.len() - argc, so args
    // are sitting on the value stack from stack_base onwards. The subroutine
    // can read them directly with stack peek/pop. Simplest sub body:
    //   sub: LoadInt(2); Mul; ReturnValue
    // which pops 2 and the arg (21), multiplies → 42, ReturnValue pops result,
    // truncates stack to stack_base (which removes the original arg... but
    // we already popped it), so result is the only value pushed back.
    let mut b = ChunkBuilder::new();
    let n = b.add_name("twice");
    b.emit(Op::LoadInt(21), 1);
    let call_pos = b.emit(Op::Call(n, 1), 1);
    let _ = call_pos;
    b.emit(Op::Return, 1);
    // Sub entry
    let sub_ip = b.current_pos();
    b.emit(Op::LoadInt(2), 1);
    b.emit(Op::Mul, 1);
    b.emit(Op::ReturnValue, 1);
    b.add_sub_entry(n, sub_ip);
    match run(b) {
        VMResult::Ok(Value::Int(42)) => {}
        other => panic!("got {:?}", other),
    }
}

#[test]
fn call_with_plain_return_truncates_to_stack_base() {
    // Subroutine that pushes garbage but uses Return (not ReturnValue);
    // stack is truncated to stack_base (which includes the original args).
    let mut b = ChunkBuilder::new();
    let n = b.add_name("noisy");
    b.emit(Op::LoadInt(100), 1); // pre-call sentinel
    b.emit(Op::Call(n, 0), 1);
    b.emit(Op::Return, 1);
    let sub_ip = b.current_pos();
    b.emit(Op::LoadInt(999), 1);
    b.emit(Op::LoadInt(888), 1);
    b.emit(Op::Return, 1);
    b.add_sub_entry(n, sub_ip);
    match run(b) {
        // sentinel 100 still on stack; sub's pushes were truncated.
        VMResult::Ok(Value::Int(100)) => {}
        other => panic!("got {:?}", other),
    }
}

// ── TestFile op via default host ────────────────────────────────────────────

#[test]
fn test_file_exists_on_cargo_toml_true() {
    let mut b = ChunkBuilder::new();
    let c = b.add_constant(Value::str("Cargo.toml"));
    b.emit(Op::LoadConst(c), 1);
    b.emit(Op::TestFile(file_test::EXISTS), 1);
    match run(b) {
        VMResult::Ok(Value::Bool(true)) => {}
        other => panic!("got {:?}", other),
    }
}

#[test]
fn test_file_is_file_on_cargo_toml_true() {
    let mut b = ChunkBuilder::new();
    let c = b.add_constant(Value::str("Cargo.toml"));
    b.emit(Op::LoadConst(c), 1);
    b.emit(Op::TestFile(file_test::IS_FILE), 1);
    match run(b) {
        VMResult::Ok(Value::Bool(true)) => {}
        other => panic!("got {:?}", other),
    }
}

#[test]
fn test_file_is_dir_on_src_true() {
    let mut b = ChunkBuilder::new();
    let c = b.add_constant(Value::str("src"));
    b.emit(Op::LoadConst(c), 1);
    b.emit(Op::TestFile(file_test::IS_DIR), 1);
    match run(b) {
        VMResult::Ok(Value::Bool(true)) => {}
        other => panic!("got {:?}", other),
    }
}

#[test]
fn test_file_exists_on_nonexistent_path_false() {
    let mut b = ChunkBuilder::new();
    let c = b.add_constant(Value::str("/no/such/path/zzz_fusevm_test"));
    b.emit(Op::LoadConst(c), 1);
    b.emit(Op::TestFile(file_test::EXISTS), 1);
    match run(b) {
        VMResult::Ok(Value::Bool(false)) => {}
        other => panic!("got {:?}", other),
    }
}

#[test]
fn test_file_is_file_on_directory_returns_false() {
    let mut b = ChunkBuilder::new();
    let c = b.add_constant(Value::str("src"));
    b.emit(Op::LoadConst(c), 1);
    b.emit(Op::TestFile(file_test::IS_FILE), 1);
    match run(b) {
        VMResult::Ok(Value::Bool(false)) => {}
        other => panic!("got {:?}", other),
    }
}

// ── Range edge cases ────────────────────────────────────────────────────────

#[test]
fn range_descending_yields_empty() {
    // from=5, to=2 → Range produces empty array because (to - from + 1) = -2 < 0
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(5), 1);
    b.emit(Op::LoadInt(2), 1);
    b.emit(Op::Range, 1);
    match run(b) {
        VMResult::Ok(Value::Array(a)) => assert!(a.is_empty()),
        other => panic!("got {:?}", other),
    }
}

#[test]
fn range_single_element_when_from_equals_to() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(7), 1);
    b.emit(Op::LoadInt(7), 1);
    b.emit(Op::Range, 1);
    match run(b) {
        VMResult::Ok(Value::Array(a)) => {
            assert_eq!(a.len(), 1);
            assert_eq!(a[0].to_int(), 7);
        }
        other => panic!("got {:?}", other),
    }
}

// ── PushFrame / PopFrame slot scope ─────────────────────────────────────────

#[test]
fn slot_is_scoped_to_current_frame() {
    // Outer: slot 0 = 100
    // Inner PushFrame: slot 0 in inner frame is a fresh slot (Undef)
    // After PopFrame: outer slot 0 should still be 100
    let mut b = ChunkBuilder::new();
    b.emit(Op::PushFrame, 1);
    b.emit(Op::LoadInt(100), 1);
    b.emit(Op::SetSlot(0), 1);
    b.emit(Op::PushFrame, 1);
    b.emit(Op::LoadInt(50), 1);
    b.emit(Op::SetSlot(0), 1);
    b.emit(Op::PopFrame, 1);
    b.emit(Op::GetSlot(0), 1);
    match run(b) {
        VMResult::Ok(Value::Int(100)) => {}
        other => panic!("got {:?}", other),
    }
}

// ── HereString / HereDoc don't pop stack from host point of view ────────────

#[test]
fn here_string_pops_target_value() {
    // Verify that HereString consumes the top of stack (the target string)
    // even with no host present. Then ensure remaining stack is intact.
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(7), 1);
    let s = b.add_constant(Value::str("input"));
    b.emit(Op::LoadConst(s), 1);
    b.emit(Op::HereString, 1);
    match run(b) {
        VMResult::Ok(Value::Int(7)) => {}
        other => panic!("got {:?}", other),
    }
}

// ── ChunkBuilder.set_source persists into Chunk ─────────────────────────────

#[test]
fn chunk_builder_set_source_persists() {
    let mut b = ChunkBuilder::new();
    b.set_source("test.fuse");
    b.emit(Op::LoadInt(0), 1);
    let c = b.build();
    assert_eq!(c.source.as_str(), "test.fuse");
}

#[test]
fn chunk_builder_add_block_range_grows_block_table() {
    let mut b = ChunkBuilder::new();
    let i0 = b.add_block_range(0, 10);
    let i1 = b.add_block_range(20, 30);
    assert_eq!(i0, 0);
    assert_eq!(i1, 1);
}
