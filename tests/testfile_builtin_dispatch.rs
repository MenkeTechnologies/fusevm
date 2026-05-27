//! Coverage for TestFile (real filesystem), CallBuiltin dispatch,
//! Extended/ExtendedWide, builtin_id lookup table, and constant stability.

use fusevm::chunk::{Chunk, ChunkBuilder};
use fusevm::op::{file_test, param_mod, redirect_op, Op};
use fusevm::shell_builtins::*;
use fusevm::value::Value;
use fusevm::vm::{VMResult, VM};

fn run(b: ChunkBuilder) -> Value {
    match VM::new(b.build()).run() {
        VMResult::Ok(v) => v,
        other => panic!("unexpected result: {:?}", other),
    }
}

// ── TestFile against the real filesystem ────────────────────────────────

fn tmp_path(suffix: &str) -> std::path::PathBuf {
    let pid = std::process::id();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or(0);
    std::env::temp_dir().join(format!("fusevm-test-{}-{}-{}", pid, nanos, suffix))
}

fn test_file_op(path: &str, t: u8) -> Value {
    let mut b = ChunkBuilder::new();
    let p = b.add_constant(Value::str(path));
    b.emit(Op::LoadConst(p), 1);
    b.emit(Op::TestFile(t), 1);
    run(b)
}

#[test]
fn test_file_exists_true_for_real_file() {
    let p = tmp_path("exists.txt");
    std::fs::write(&p, b"hello").unwrap();
    let s = p.to_str().unwrap();
    assert_eq!(test_file_op(s, file_test::EXISTS), Value::Bool(true));
    assert_eq!(test_file_op(s, file_test::IS_FILE), Value::Bool(true));
    assert_eq!(test_file_op(s, file_test::IS_DIR), Value::Bool(false));
    let _ = std::fs::remove_file(&p);
}

#[test]
fn test_file_exists_false_for_missing() {
    let p = tmp_path("missing.txt");
    let s = p.to_str().unwrap();
    assert_eq!(test_file_op(s, file_test::EXISTS), Value::Bool(false));
    assert_eq!(test_file_op(s, file_test::IS_FILE), Value::Bool(false));
    assert_eq!(test_file_op(s, file_test::IS_DIR), Value::Bool(false));
}

#[test]
fn test_file_is_dir_true_for_real_dir() {
    let p = tmp_path("dir");
    std::fs::create_dir(&p).unwrap();
    let s = p.to_str().unwrap();
    assert_eq!(test_file_op(s, file_test::IS_DIR), Value::Bool(true));
    assert_eq!(test_file_op(s, file_test::IS_FILE), Value::Bool(false));
    let _ = std::fs::remove_dir(&p);
}

#[test]
fn test_file_nonempty_distinguishes_empty_vs_filled() {
    let empty = tmp_path("empty");
    let filled = tmp_path("filled");
    std::fs::write(&empty, b"").unwrap();
    std::fs::write(&filled, b"x").unwrap();
    assert_eq!(
        test_file_op(empty.to_str().unwrap(), file_test::IS_NONEMPTY),
        Value::Bool(false)
    );
    assert_eq!(
        test_file_op(filled.to_str().unwrap(), file_test::IS_NONEMPTY),
        Value::Bool(true)
    );
    let _ = std::fs::remove_file(&empty);
    let _ = std::fs::remove_file(&filled);
}

#[test]
fn test_file_readable_writable_use_exists_fallback() {
    let p = tmp_path("rw");
    std::fs::write(&p, b"x").unwrap();
    let s = p.to_str().unwrap();
    assert_eq!(test_file_op(s, file_test::IS_READABLE), Value::Bool(true));
    assert_eq!(test_file_op(s, file_test::IS_WRITABLE), Value::Bool(true));
    let _ = std::fs::remove_file(&p);
}

#[test]
fn test_file_symlink_false_for_regular_file() {
    let p = tmp_path("regular");
    std::fs::write(&p, b"x").unwrap();
    let s = p.to_str().unwrap();
    assert_eq!(test_file_op(s, file_test::IS_SYMLINK), Value::Bool(false));
    let _ = std::fs::remove_file(&p);
}

#[cfg(unix)]
#[test]
fn test_file_symlink_true_for_symlink() {
    let target = tmp_path("symtarget");
    let link = tmp_path("symlink");
    std::fs::write(&target, b"x").unwrap();
    std::os::unix::fs::symlink(&target, &link).unwrap();
    let s = link.to_str().unwrap();
    assert_eq!(test_file_op(s, file_test::IS_SYMLINK), Value::Bool(true));
    let _ = std::fs::remove_file(&link);
    let _ = std::fs::remove_file(&target);
}

#[test]
fn test_file_unknown_test_type_yields_false() {
    let p = tmp_path("any");
    std::fs::write(&p, b"x").unwrap();
    assert_eq!(test_file_op(p.to_str().unwrap(), 200), Value::Bool(false));
    let _ = std::fs::remove_file(&p);
}

#[test]
fn test_file_socket_block_char_false_for_regular_file() {
    let p = tmp_path("special");
    std::fs::write(&p, b"x").unwrap();
    let s = p.to_str().unwrap();
    assert_eq!(test_file_op(s, file_test::IS_SOCKET), Value::Bool(false));
    assert_eq!(test_file_op(s, file_test::IS_BLOCK_DEV), Value::Bool(false));
    assert_eq!(test_file_op(s, file_test::IS_CHAR_DEV), Value::Bool(false));
    assert_eq!(test_file_op(s, file_test::IS_FIFO), Value::Bool(false));
    let _ = std::fs::remove_file(&p);
}

// ── CallBuiltin dispatch ─────────────────────────────────────────────────

#[test]
fn call_builtin_registered_handler_dispatches() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(7), 1);
    b.emit(Op::CallBuiltin(42, 1), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(42, |vm, _argc| {
        let v = vm.pop();
        Value::Int(v.to_int() + 100)
    });
    match vm.run() {
        VMResult::Ok(Value::Int(107)) => {}
        other => panic!("got {:?}", other),
    }
}

#[test]
fn call_builtin_unregistered_id_is_silent_noop() {
    // No handler => the op simply does nothing (no panic).
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(5), 1);
    b.emit(Op::CallBuiltin(999, 0), 1);
    let mut vm = VM::new(b.build());
    match vm.run() {
        VMResult::Ok(Value::Int(5)) => {}
        other => panic!("got {:?}", other),
    }
}

#[test]
fn call_builtin_argc_value_passed_to_handler() {
    use std::sync::atomic::{AtomicU8, Ordering};
    static CAP: AtomicU8 = AtomicU8::new(0);
    CAP.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(0, 7), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(0, |_vm, argc| {
        CAP.store(argc, Ordering::SeqCst);
        Value::Status(0)
    });
    let _ = vm.run();
    assert_eq!(CAP.load(Ordering::SeqCst), 7);
}

#[test]
fn call_builtin_handler_can_read_multiple_args_from_stack() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::LoadInt(2), 1);
    b.emit(Op::LoadInt(3), 1);
    b.emit(Op::CallBuiltin(0, 3), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(0, |vm, argc| {
        let mut sum = 0i64;
        for _ in 0..argc {
            sum += vm.pop().to_int();
        }
        Value::Int(sum)
    });
    match vm.run() {
        VMResult::Ok(Value::Int(6)) => {}
        other => panic!("got {:?}", other),
    }
}

#[test]
fn register_builtin_grows_table_on_demand() {
    // Picking a large id should auto-resize the inline table.
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(5000, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(5000, |_, _| Value::Int(99));
    match vm.run() {
        VMResult::Ok(Value::Int(99)) => {}
        other => panic!("got {:?}", other),
    }
}

// ── Extended / ExtendedWide ─────────────────────────────────────────────

#[test]
fn extended_handler_receives_id_and_arg() {
    use std::sync::{Arc, Mutex};
    let cap: Arc<Mutex<Option<(u16, u8)>>> = Arc::new(Mutex::new(None));
    let cap_cl = cap.clone();

    let mut b = ChunkBuilder::new();
    b.emit(Op::Extended(13, 99), 1);
    let mut vm = VM::new(b.build());
    vm.set_extension_handler(Box::new(move |vm, id, arg| {
        *cap_cl.lock().unwrap() = Some((id, arg));
        vm.push(Value::Int(id as i64 + arg as i64));
    }));
    match vm.run() {
        VMResult::Ok(Value::Int(112)) => {}
        other => panic!("got {:?}", other),
    }
    assert_eq!(*cap.lock().unwrap(), Some((13, 99)));
}

#[test]
fn extended_wide_handler_receives_id_and_payload() {
    use std::sync::{Arc, Mutex};
    let cap: Arc<Mutex<Option<(u16, usize)>>> = Arc::new(Mutex::new(None));
    let cap_cl = cap.clone();

    let mut b = ChunkBuilder::new();
    b.emit(Op::ExtendedWide(7, 70000), 1);
    b.emit(Op::LoadInt(0), 1);
    let mut vm = VM::new(b.build());
    vm.set_extension_wide_handler(Box::new(move |_vm, id, payload| {
        *cap_cl.lock().unwrap() = Some((id, payload));
    }));
    let _ = vm.run();
    assert_eq!(*cap.lock().unwrap(), Some((7, 70000)));
}

#[test]
fn extended_without_handler_is_silent_noop() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(5), 1);
    b.emit(Op::Extended(0, 0), 1);
    let mut vm = VM::new(b.build());
    match vm.run() {
        VMResult::Ok(Value::Int(5)) => {}
        other => panic!("got {:?}", other),
    }
}

#[test]
fn extended_wide_without_handler_is_silent_noop() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(5), 1);
    b.emit(Op::ExtendedWide(0, 0), 1);
    let mut vm = VM::new(b.build());
    match vm.run() {
        VMResult::Ok(Value::Int(5)) => {}
        other => panic!("got {:?}", other),
    }
}

// ── builtin_id name → id lookup table ───────────────────────────────────

#[test]
fn builtin_id_recognizes_posix_core() {
    assert_eq!(builtin_id("cd"), Some(BUILTIN_CD));
    assert_eq!(builtin_id("chdir"), Some(BUILTIN_CD)); // alias
    assert_eq!(builtin_id("pwd"), Some(BUILTIN_PWD));
    assert_eq!(builtin_id("echo"), Some(BUILTIN_ECHO));
    assert_eq!(builtin_id("printf"), Some(BUILTIN_PRINTF));
    assert_eq!(builtin_id("true"), Some(BUILTIN_TRUE));
    assert_eq!(builtin_id("false"), Some(BUILTIN_FALSE));
    assert_eq!(builtin_id(":"), Some(BUILTIN_COLON));
}

#[test]
fn builtin_id_aliases_dot_and_source() {
    assert_eq!(builtin_id("source"), Some(BUILTIN_SOURCE));
    assert_eq!(builtin_id("."), Some(BUILTIN_SOURCE));
}

#[test]
fn builtin_id_aliases_test_and_bracket() {
    assert_eq!(builtin_id("test"), Some(BUILTIN_TEST));
    assert_eq!(builtin_id("["), Some(BUILTIN_TEST));
}

#[test]
fn builtin_id_exit_aliases() {
    assert_eq!(builtin_id("exit"), Some(BUILTIN_EXIT));
    assert_eq!(builtin_id("bye"), Some(BUILTIN_EXIT));
    assert_eq!(builtin_id("logout"), Some(BUILTIN_EXIT));
}

#[test]
fn builtin_id_typeset_alias_for_declare() {
    assert_eq!(builtin_id("declare"), Some(BUILTIN_TYPESET));
    assert_eq!(builtin_id("typeset"), Some(BUILTIN_TYPESET));
}

#[test]
fn builtin_id_readarray_alias_for_mapfile() {
    assert_eq!(builtin_id("mapfile"), Some(BUILTIN_MAPFILE));
    assert_eq!(builtin_id("readarray"), Some(BUILTIN_MAPFILE));
}

#[test]
fn builtin_id_unknown_returns_none() {
    assert_eq!(builtin_id("not_a_real_builtin_xyzzy"), None);
    assert_eq!(builtin_id(""), None);
    assert_eq!(builtin_id("CD"), None); // case sensitive
}

#[test]
fn is_builtin_agrees_with_builtin_id() {
    for name in [
        "cd", "pwd", "echo", "printf", "test", "[", "true", "false", "source", ".", "alias", "set",
        "unset", "exit", "trap", "jobs", "fg", "bg", "break", "continue",
    ] {
        assert!(is_builtin(name), "is_builtin({:?}) should be true", name);
        assert!(builtin_id(name).is_some(), "{:?} has no id", name);
    }
    assert!(!is_builtin("definitely_not_a_builtin"));
    assert!(!is_builtin(""));
}

// ── Constant value stability (regression guard) ─────────────────────────

#[test]
fn builtin_constants_are_stable() {
    // These IDs are part of the public API: changing them invalidates any
    // serialized bytecode. Lock the values that downstream frontends emit.
    assert_eq!(BUILTIN_CD, 0);
    assert_eq!(BUILTIN_PWD, 1);
    assert_eq!(BUILTIN_ECHO, 2);
    assert_eq!(BUILTIN_PRINT, 3);
    assert_eq!(BUILTIN_PRINTF, 4);
    assert_eq!(BUILTIN_EXPORT, 5);
    assert_eq!(BUILTIN_UNSET, 6);
    assert_eq!(BUILTIN_SOURCE, 7);
    assert_eq!(BUILTIN_EXIT, 8);
    assert_eq!(BUILTIN_RETURN, 9);
    assert_eq!(BUILTIN_TRUE, 10);
    assert_eq!(BUILTIN_FALSE, 11);
    assert_eq!(BUILTIN_TEST, 12);
}

#[test]
fn file_test_constants_are_stable() {
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
fn redirect_op_constants_are_stable() {
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
fn param_mod_constants_are_stable() {
    assert_eq!(param_mod::DEFAULT, 0);
    assert_eq!(param_mod::ASSIGN, 1);
    assert_eq!(param_mod::ERROR, 2);
    assert_eq!(param_mod::ALTERNATE, 3);
    assert_eq!(param_mod::LENGTH, 4);
    assert_eq!(param_mod::SUBST_FIRST, 9);
    assert_eq!(param_mod::SUBST_ALL, 10);
    assert_eq!(param_mod::UPPER, 11);
    assert_eq!(param_mod::LOWER, 12);
    assert_eq!(param_mod::INDIRECT, 15);
    assert_eq!(param_mod::KEYS, 16);
    assert_eq!(param_mod::SLICE, 17);
}

// ── Chunk find_sub semantics ────────────────────────────────────────────

#[test]
fn find_sub_returns_none_for_unknown_name() {
    let chunk = Chunk::new();
    assert_eq!(chunk.find_sub(0), None);
    assert_eq!(chunk.find_sub(9999), None);
}

#[test]
fn op_hash_is_deterministic_across_builds() {
    // Same op stream + constants → same op_hash; useful for JIT cache lookups.
    let build = || {
        let mut b = ChunkBuilder::new();
        b.emit(Op::LoadInt(7), 1);
        b.emit(Op::LoadInt(3), 1);
        b.emit(Op::Add, 1);
        b.add_constant(Value::str("dummy"));
        b.build()
    };
    let a = build();
    let b = build();
    assert_eq!(a.op_hash, b.op_hash);
}

#[test]
fn op_hash_differs_when_constants_differ() {
    let mut ba = ChunkBuilder::new();
    ba.emit(Op::LoadInt(0), 1);
    ba.add_constant(Value::Int(1));
    let mut bb = ChunkBuilder::new();
    bb.emit(Op::LoadInt(0), 1);
    bb.add_constant(Value::Int(2));
    assert_ne!(ba.build().op_hash, bb.build().op_hash);
}

#[test]
fn op_hash_differs_when_ops_differ() {
    let mut ba = ChunkBuilder::new();
    ba.emit(Op::LoadInt(0), 1);
    ba.emit(Op::LoadInt(1), 1);
    let mut bb = ChunkBuilder::new();
    bb.emit(Op::LoadInt(0), 1);
    bb.emit(Op::LoadInt(2), 1);
    assert_ne!(ba.build().op_hash, bb.build().op_hash);
}

#[test]
fn add_name_deduplicates_repeated_calls() {
    let mut b = ChunkBuilder::new();
    let a = b.add_name("x");
    let c = b.add_name("x");
    let d = b.add_name("y");
    assert_eq!(a, c);
    assert_ne!(a, d);
    assert_eq!(b.build().names.len(), 2);
}

#[test]
fn add_constant_does_not_deduplicate() {
    // Constants pool is append-only (no dedup); each call yields a new index.
    let mut b = ChunkBuilder::new();
    let i1 = b.add_constant(Value::Int(42));
    let i2 = b.add_constant(Value::Int(42));
    assert_ne!(i1, i2);
    assert_eq!(b.build().constants.len(), 2);
}

#[test]
fn patch_jump_updates_target() {
    let mut b = ChunkBuilder::new();
    let jmp = b.emit(Op::Jump(0), 1);
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::LoadInt(2), 1);
    let end = b.current_pos();
    b.patch_jump(jmp, end);
    let chunk = b.build();
    match chunk.ops[jmp] {
        Op::Jump(t) => assert_eq!(t, end),
        _ => panic!("expected Jump"),
    }
}

#[test]
#[should_panic(expected = "patch_jump on non-jump op")]
fn patch_jump_on_non_jump_panics() {
    let mut b = ChunkBuilder::new();
    let nop = b.emit(Op::Nop, 1);
    b.patch_jump(nop, 0);
}

// ── Block ranges ────────────────────────────────────────────────────────

#[test]
fn add_block_range_returns_sequential_indices() {
    let mut b = ChunkBuilder::new();
    let i0 = b.add_block_range(0, 10);
    let i1 = b.add_block_range(11, 20);
    let i2 = b.add_block_range(21, 30);
    assert_eq!((i0, i1, i2), (0, 1, 2));
    let chunk = b.build();
    assert_eq!(chunk.block_ranges, vec![(0, 10), (11, 20), (21, 30)]);
}

// ── Op Hash trait determinism ───────────────────────────────────────────

#[test]
fn op_hash_distinguishes_same_payload_different_op() {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let h = |op: Op| {
        let mut hs = DefaultHasher::new();
        op.hash(&mut hs);
        hs.finish()
    };
    // GetVar(0) vs SetVar(0) — same payload, different discriminant.
    assert_ne!(h(Op::GetVar(0)), h(Op::SetVar(0)));
    // LoadInt(0) vs Nop
    assert_ne!(h(Op::LoadInt(0)), h(Op::Nop));
    // LoadFloat NaN equals itself by bits.
    assert_eq!(h(Op::LoadFloat(f64::NAN)), h(Op::LoadFloat(f64::NAN)));
}

// ── set_source on builder ───────────────────────────────────────────────

#[test]
fn set_source_propagates_to_built_chunk() {
    let mut b = ChunkBuilder::new();
    b.set_source("script.fuse");
    let chunk = b.build();
    assert_eq!(chunk.source, "script.fuse");
}
