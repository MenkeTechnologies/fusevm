use fusevm::{ChunkBuilder, Op, VM, VMResult, Value};
use std::sync::{Arc, Mutex};

fn run(b: ChunkBuilder) -> Value {
    match VM::new(b.build()).run() {
        VMResult::Ok(v) => v,
        other => panic!("unexpected result: {:?}", other),
    }
}

// ── Jump opcodes ──

#[test]
fn jump_unconditional_forward() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::Jump(3), 1);
    b.emit(Op::LoadInt(99), 1);
    b.emit(Op::Jump(4), 1);
    b.emit(Op::LoadInt(7), 1);
    assert_eq!(run(b), Value::Int(7));
}

#[test]
fn jump_if_true_taken() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadTrue, 1);
    b.emit(Op::JumpIfTrue(3), 1);
    b.emit(Op::LoadInt(99), 1);
    b.emit(Op::LoadInt(5), 1);
    assert_eq!(run(b), Value::Int(5));
}

#[test]
fn jump_if_true_not_taken_pops() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadFalse, 1);
    b.emit(Op::JumpIfTrue(999), 1);
    b.emit(Op::LoadInt(8), 1);
    assert_eq!(run(b), Value::Int(8));
}

#[test]
fn jump_if_false_taken() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadFalse, 1);
    b.emit(Op::JumpIfFalse(3), 1);
    b.emit(Op::LoadInt(99), 1);
    b.emit(Op::LoadInt(12), 1);
    assert_eq!(run(b), Value::Int(12));
}

#[test]
fn jump_if_false_not_taken() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadTrue, 1);
    b.emit(Op::JumpIfFalse(999), 1);
    b.emit(Op::LoadInt(20), 1);
    assert_eq!(run(b), Value::Int(20));
}

#[test]
fn jump_if_true_keep_keeps_value() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(42), 1);
    b.emit(Op::JumpIfTrueKeep(3), 1);
    b.emit(Op::LoadInt(99), 1);
    // Reached only if no jump; with truthy 42, we jump to ip 3 → end
    assert_eq!(run(b), Value::Int(42));
}

#[test]
fn jump_if_true_keep_not_taken_pops_continues() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(0), 1); // falsy
    b.emit(Op::JumpIfTrueKeep(999), 1);
    b.emit(Op::Pop, 1);
    b.emit(Op::LoadInt(7), 1);
    assert_eq!(run(b), Value::Int(7));
}

#[test]
fn jump_if_false_keep_taken() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(0), 1);
    b.emit(Op::JumpIfFalseKeep(3), 1);
    b.emit(Op::LoadInt(99), 1);
    assert_eq!(run(b), Value::Int(0));
}

#[test]
fn jump_truthy_string_is_true() {
    let mut b = ChunkBuilder::new();
    let k = b.add_constant(Value::str("nonempty"));
    b.emit(Op::LoadConst(k), 1);
    b.emit(Op::JumpIfTrue(4), 1);
    b.emit(Op::LoadInt(99), 1);
    b.emit(Op::Jump(5), 1);
    b.emit(Op::LoadInt(1), 1);
    assert_eq!(run(b), Value::Int(1));
}

#[test]
fn jump_empty_string_is_false() {
    let mut b = ChunkBuilder::new();
    let k = b.add_constant(Value::str(""));
    b.emit(Op::LoadConst(k), 1);
    b.emit(Op::JumpIfFalse(4), 1);
    b.emit(Op::LoadInt(99), 1);
    b.emit(Op::Jump(5), 1);
    b.emit(Op::LoadInt(2), 1);
    assert_eq!(run(b), Value::Int(2));
}

#[test]
fn loop_via_jump_counts_iterations() {
    // i = 0; while i < 5 { i++; } -> i should be 5
    let mut b = ChunkBuilder::new();
    b.emit(Op::PushFrame, 1);
    b.emit(Op::LoadInt(0), 1);
    b.emit(Op::SetSlot(0), 1);
    // ip 3 — loop top: condition
    b.emit(Op::GetSlot(0), 1);
    b.emit(Op::LoadInt(5), 1);
    b.emit(Op::NumLt, 1);
    b.emit(Op::JumpIfFalse(9), 1);
    b.emit(Op::PreIncSlotVoid(0), 1);
    b.emit(Op::Jump(3), 1);
    // ip 9 — done
    b.emit(Op::GetSlot(0), 1);
    assert_eq!(run(b), Value::Int(5));
}

// ── Print / PrintLn drain stack ──

#[test]
fn print_drains_n_from_stack() {
    let mut b = ChunkBuilder::new();
    let k = b.add_constant(Value::str("x"));
    b.emit(Op::LoadConst(k), 1);
    b.emit(Op::LoadConst(k), 1);
    b.emit(Op::LoadConst(k), 1);
    b.emit(Op::Print(3), 1);
    // Now stack is empty; push sentinel
    b.emit(Op::LoadInt(42), 1);
    assert_eq!(run(b), Value::Int(42));
}

#[test]
fn println_drains_n_from_stack() {
    let mut b = ChunkBuilder::new();
    let k = b.add_constant(Value::str("y"));
    b.emit(Op::LoadConst(k), 1);
    b.emit(Op::LoadConst(k), 1);
    b.emit(Op::PrintLn(2), 1);
    b.emit(Op::LoadInt(99), 1);
    assert_eq!(run(b), Value::Int(99));
}

#[test]
fn print_zero_is_noop() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(5), 1);
    b.emit(Op::Print(0), 1);
    assert_eq!(run(b), Value::Int(5));
}

#[test]
fn print_more_than_stack_drains_all() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::Print(10), 1); // saturating_sub → 0
    b.emit(Op::LoadInt(7), 1);
    assert_eq!(run(b), Value::Int(7));
}

// ── Extension dispatch ──

#[test]
fn extended_op_invokes_handler() {
    let observed = Arc::new(Mutex::new(None));
    let observed_clone = observed.clone();
    let mut b = ChunkBuilder::new();
    b.emit(Op::Extended(7, 42), 1);
    b.emit(Op::LoadInt(1), 1);
    let mut vm = VM::new(b.build());
    vm.set_extension_handler(Box::new(move |_vm, id, arg| {
        *observed_clone.lock().unwrap() = Some((id, arg));
    }));
    match vm.run() {
        VMResult::Ok(Value::Int(1)) => {}
        other => panic!("expected Int(1), got {:?}", other),
    }
    assert_eq!(*observed.lock().unwrap(), Some((7, 42)));
}

#[test]
fn extended_op_no_handler_is_noop() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::Extended(1, 2), 1);
    b.emit(Op::LoadInt(11), 1);
    assert_eq!(run(b), Value::Int(11));
}

#[test]
fn extended_wide_op_invokes_handler() {
    let observed = Arc::new(Mutex::new(None));
    let observed_clone = observed.clone();
    let mut b = ChunkBuilder::new();
    b.emit(Op::ExtendedWide(3, 99999), 1);
    b.emit(Op::LoadInt(2), 1);
    let mut vm = VM::new(b.build());
    vm.set_extension_wide_handler(Box::new(move |_vm, id, payload| {
        *observed_clone.lock().unwrap() = Some((id, payload));
    }));
    match vm.run() {
        VMResult::Ok(Value::Int(2)) => {}
        other => panic!("expected Int(2), got {:?}", other),
    }
    assert_eq!(*observed.lock().unwrap(), Some((3, 99999)));
}

#[test]
fn extended_handler_can_push() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::Extended(0, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_extension_handler(Box::new(|vm, _id, _arg| {
        vm.push(Value::Int(123));
    }));
    match vm.run() {
        VMResult::Ok(Value::Int(123)) => {}
        other => panic!("expected Int(123), got {:?}", other),
    }
}

// ── Builtin registration ──

#[test]
fn registered_builtin_is_called() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(10), 1);
    b.emit(Op::CallBuiltin(0, 1), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(0, |vm, argc| {
        let mut sum: i64 = 0;
        for _ in 0..argc {
            sum += vm.pop().to_int();
        }
        Value::Int(sum + 1)
    });
    match vm.run() {
        VMResult::Ok(Value::Int(11)) => {}
        other => panic!("expected Int(11), got {:?}", other),
    }
}

#[test]
fn registered_builtin_high_id_resizes() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(50, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(50, |_, _| Value::Int(777));
    match vm.run() {
        VMResult::Ok(Value::Int(777)) => {}
        other => panic!("expected Int(777), got {:?}", other),
    }
}

#[test]
fn call_unregistered_builtin_is_noop() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(99, 0), 1);
    b.emit(Op::LoadInt(5), 1);
    assert_eq!(run(b), Value::Int(5));
}

// ── No-host fallbacks for shell ops ──

#[test]
fn word_split_default_whitespace() {
    let mut b = ChunkBuilder::new();
    let k = b.add_constant(Value::str("foo bar baz"));
    b.emit(Op::LoadConst(k), 1);
    b.emit(Op::WordSplit, 1);
    match run(b) {
        Value::Array(v) => {
            assert_eq!(v.len(), 3);
            assert_eq!(v[0].to_str(), "foo");
            assert_eq!(v[2].to_str(), "baz");
        }
        other => panic!("expected array, got {:?}", other),
    }
}

#[test]
fn word_split_empty_string() {
    let mut b = ChunkBuilder::new();
    let k = b.add_constant(Value::str(""));
    b.emit(Op::LoadConst(k), 1);
    b.emit(Op::WordSplit, 1);
    match run(b) {
        Value::Array(v) => assert!(v.is_empty()),
        other => panic!("expected array, got {:?}", other),
    }
}

#[test]
fn brace_expand_default_returns_singleton() {
    let mut b = ChunkBuilder::new();
    let k = b.add_constant(Value::str("a{b,c}d"));
    b.emit(Op::LoadConst(k), 1);
    b.emit(Op::BraceExpand, 1);
    match run(b) {
        Value::Array(v) => {
            assert_eq!(v.len(), 1);
            assert_eq!(v[0].to_str(), "a{b,c}d");
        }
        other => panic!("expected array, got {:?}", other),
    }
}

#[test]
fn tilde_expand_default_returns_unchanged() {
    let mut b = ChunkBuilder::new();
    let k = b.add_constant(Value::str("~/foo"));
    b.emit(Op::LoadConst(k), 1);
    b.emit(Op::TildeExpand, 1);
    match run(b) {
        Value::Str(s) => assert_eq!(&*s, "~/foo"),
        other => panic!("expected str, got {:?}", other),
    }
}

#[test]
fn str_match_default_is_equality() {
    let mut b = ChunkBuilder::new();
    let s = b.add_constant(Value::str("hello"));
    let p = b.add_constant(Value::str("hello"));
    b.emit(Op::LoadConst(s), 1);
    b.emit(Op::LoadConst(p), 1);
    b.emit(Op::StrMatch, 1);
    assert_eq!(run(b), Value::Bool(true));
}

#[test]
fn str_match_default_unequal() {
    let mut b = ChunkBuilder::new();
    let s = b.add_constant(Value::str("hello"));
    let p = b.add_constant(Value::str("world"));
    b.emit(Op::LoadConst(s), 1);
    b.emit(Op::LoadConst(p), 1);
    b.emit(Op::StrMatch, 1);
    assert_eq!(run(b), Value::Bool(false));
}

#[test]
fn regex_match_default_false() {
    let mut b = ChunkBuilder::new();
    let s = b.add_constant(Value::str("hello"));
    let r = b.add_constant(Value::str("hello"));
    b.emit(Op::LoadConst(s), 1);
    b.emit(Op::LoadConst(r), 1);
    b.emit(Op::RegexMatch, 1);
    assert_eq!(run(b), Value::Bool(false));
}

#[test]
fn expand_param_no_host_is_empty_string() {
    let mut b = ChunkBuilder::new();
    let name = b.add_constant(Value::str("X"));
    b.emit(Op::LoadConst(name), 1);
    b.emit(Op::ExpandParam(4 /* LENGTH */), 1);
    match run(b) {
        Value::Str(s) => assert_eq!(&*s, ""),
        other => panic!("expected str, got {:?}", other),
    }
}

#[test]
fn trap_check_no_host_is_noop() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::TrapCheck, 1);
    b.emit(Op::LoadInt(5), 1);
    assert_eq!(run(b), Value::Int(5));
}

// ── TestFile op (filesystem) ──

#[test]
fn test_file_exists_temp_file() {
    let dir = std::env::temp_dir();
    let path = dir.join(format!("fusevm_test_{}.txt", std::process::id()));
    std::fs::write(&path, b"hi").unwrap();
    let mut b = ChunkBuilder::new();
    let k = b.add_constant(Value::str(path.to_str().unwrap()));
    b.emit(Op::LoadConst(k), 1);
    b.emit(Op::TestFile(5 /* EXISTS */), 1);
    let r = run(b);
    let _ = std::fs::remove_file(&path);
    assert_eq!(r, Value::Bool(true));
}

#[test]
fn test_file_exists_missing_path_is_false() {
    let mut b = ChunkBuilder::new();
    let k = b.add_constant(Value::str("/this/path/should/not/exist/xyz_no"));
    b.emit(Op::LoadConst(k), 1);
    b.emit(Op::TestFile(5 /* EXISTS */), 1);
    assert_eq!(run(b), Value::Bool(false));
}

#[test]
fn test_file_is_file() {
    let dir = std::env::temp_dir();
    let path = dir.join(format!("fusevm_isfile_{}.txt", std::process::id()));
    std::fs::write(&path, b"x").unwrap();
    let mut b = ChunkBuilder::new();
    let k = b.add_constant(Value::str(path.to_str().unwrap()));
    b.emit(Op::LoadConst(k), 1);
    b.emit(Op::TestFile(0 /* IS_FILE */), 1);
    let r = run(b);
    let _ = std::fs::remove_file(&path);
    assert_eq!(r, Value::Bool(true));
}

#[test]
fn test_file_is_dir() {
    let mut b = ChunkBuilder::new();
    let k = b.add_constant(Value::str(std::env::temp_dir().to_str().unwrap()));
    b.emit(Op::LoadConst(k), 1);
    b.emit(Op::TestFile(1 /* IS_DIR */), 1);
    assert_eq!(run(b), Value::Bool(true));
}

#[test]
fn test_file_is_file_for_dir_is_false() {
    let mut b = ChunkBuilder::new();
    let k = b.add_constant(Value::str(std::env::temp_dir().to_str().unwrap()));
    b.emit(Op::LoadConst(k), 1);
    b.emit(Op::TestFile(0 /* IS_FILE */), 1);
    assert_eq!(run(b), Value::Bool(false));
}

#[test]
fn test_file_nonempty() {
    let dir = std::env::temp_dir();
    let path = dir.join(format!("fusevm_nonempty_{}.txt", std::process::id()));
    std::fs::write(&path, b"abc").unwrap();
    let mut b = ChunkBuilder::new();
    let k = b.add_constant(Value::str(path.to_str().unwrap()));
    b.emit(Op::LoadConst(k), 1);
    b.emit(Op::TestFile(6 /* IS_NONEMPTY */), 1);
    let r = run(b);
    let _ = std::fs::remove_file(&path);
    assert_eq!(r, Value::Bool(true));
}

#[test]
fn test_file_nonempty_empty_file_is_false() {
    let dir = std::env::temp_dir();
    let path = dir.join(format!("fusevm_empty_{}.txt", std::process::id()));
    std::fs::write(&path, b"").unwrap();
    let mut b = ChunkBuilder::new();
    let k = b.add_constant(Value::str(path.to_str().unwrap()));
    b.emit(Op::LoadConst(k), 1);
    b.emit(Op::TestFile(6 /* IS_NONEMPTY */), 1);
    let r = run(b);
    let _ = std::fs::remove_file(&path);
    assert_eq!(r, Value::Bool(false));
}

#[test]
fn test_file_is_readable_for_existing_path() {
    let mut b = ChunkBuilder::new();
    let k = b.add_constant(Value::str(std::env::temp_dir().to_str().unwrap()));
    b.emit(Op::LoadConst(k), 1);
    b.emit(Op::TestFile(2 /* IS_READABLE */), 1);
    assert_eq!(run(b), Value::Bool(true));
}

#[test]
fn test_file_is_symlink_for_regular_is_false() {
    let dir = std::env::temp_dir();
    let path = dir.join(format!("fusevm_notlink_{}.txt", std::process::id()));
    std::fs::write(&path, b"x").unwrap();
    let mut b = ChunkBuilder::new();
    let k = b.add_constant(Value::str(path.to_str().unwrap()));
    b.emit(Op::LoadConst(k), 1);
    b.emit(Op::TestFile(7 /* IS_SYMLINK */), 1);
    let r = run(b);
    let _ = std::fs::remove_file(&path);
    assert_eq!(r, Value::Bool(false));
}

// ── VM stack helpers (public API) ──

#[test]
fn vm_push_pop_peek_round_trip() {
    let mut vm = VM::new(ChunkBuilder::new().build());
    vm.push(Value::Int(1));
    vm.push(Value::Int(2));
    assert_eq!(vm.peek(), &Value::Int(2));
    assert_eq!(vm.pop(), Value::Int(2));
    assert_eq!(vm.pop(), Value::Int(1));
}

#[test]
fn vm_reset_clears_state() {
    let mut b1 = ChunkBuilder::new();
    b1.emit(Op::LoadInt(1), 1);
    let mut vm = VM::new(b1.build());
    let _ = vm.run();
    let mut b2 = ChunkBuilder::new();
    b2.emit(Op::LoadInt(99), 1);
    vm.reset(b2.build());
    match vm.run() {
        VMResult::Ok(Value::Int(99)) => {}
        other => panic!("expected Int(99), got {:?}", other),
    }
}

// ── Spaceship op ──

#[test]
fn spaceship_int_int_lt() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::LoadInt(2), 1);
    b.emit(Op::Spaceship, 1);
    assert_eq!(run(b), Value::Int(-1));
}

#[test]
fn spaceship_int_int_gt() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(5), 1);
    b.emit(Op::LoadInt(2), 1);
    b.emit(Op::Spaceship, 1);
    assert_eq!(run(b), Value::Int(1));
}

#[test]
fn spaceship_int_int_eq() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(3), 1);
    b.emit(Op::LoadInt(3), 1);
    b.emit(Op::Spaceship, 1);
    assert_eq!(run(b), Value::Int(0));
}

// ── Tracing-JIT toggles (jit-only API; reserved for future tests) ──

#[test]
#[cfg(feature = "jit")]
fn enable_disable_tracing_jit_does_not_break() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(5), 1);
    b.emit(Op::LoadInt(3), 1);
    b.emit(Op::Add, 1);
    let mut vm = VM::new(b.build());
    vm.enable_tracing_jit();
    vm.disable_tracing_jit();
    match vm.run() {
        VMResult::Ok(Value::Int(8)) => {}
        other => panic!("expected Int(8), got {:?}", other),
    }
}
