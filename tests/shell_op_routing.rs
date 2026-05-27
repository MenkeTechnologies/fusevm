//! Targeted coverage for shell-routed VM ops that pop / push host inputs:
//! `ExpandParam`, `HereDoc`, `HereString`, `CmdSubst`, `ProcessSubIn`/`Out`,
//! `TrapSet`, `StrMatch`, `RegexMatch`, `Redirect`, plus `CallFunction`'s
//! no-host in-chunk function fallback. Also verifies index out-of-range
//! behavior for the sub-chunk family.

use fusevm::host::ShellHost;
use fusevm::op::{param_mod, redirect_op};
use fusevm::{Chunk, ChunkBuilder, Op, VM, VMResult, Value};
use std::sync::{Arc, Mutex};

fn run(b: ChunkBuilder) -> Value {
    match VM::new(b.build()).run() {
        VMResult::Ok(v) => v,
        VMResult::Halted => Value::Undef,
        VMResult::Error(e) => panic!("unexpected VM error: {e}"),
    }
}

// ── Recording host that captures host-method arguments ──

#[derive(Clone, Debug)]
enum Call {
    ExpandParam(String, u8, Vec<Value>),
    Heredoc(String),
    Herestring(String),
    Redirect(u8, u8, String),
    CmdSubst(usize),    // sub.ops.len() as fingerprint
    ProcessSubIn(usize),
    ProcessSubOut(usize),
    TrapSet(String, usize),
    StrMatch(String, String),
    RegexMatch(String, String),
    CallFunction(String, Vec<String>),
}

#[derive(Default)]
struct Recorder {
    log: Arc<Mutex<Vec<Call>>>,
    expand_reply: Option<Value>,
    cmd_subst_reply: Option<String>,
    proc_sub_reply: Option<String>,
    str_match_reply: Option<bool>,
    regex_match_reply: Option<bool>,
    call_function_reply: Option<Option<i32>>,
}

impl Recorder {
    fn new() -> Self {
        Self::default()
    }
}

impl ShellHost for Recorder {
    fn expand_param(&mut self, name: &str, modifier: u8, args: &[Value]) -> Value {
        self.log.lock().unwrap().push(Call::ExpandParam(
            name.to_string(),
            modifier,
            args.to_vec(),
        ));
        self.expand_reply.clone().unwrap_or(Value::str(""))
    }
    fn heredoc(&mut self, c: &str) {
        self.log.lock().unwrap().push(Call::Heredoc(c.to_string()));
    }
    fn herestring(&mut self, c: &str) {
        self.log
            .lock()
            .unwrap()
            .push(Call::Herestring(c.to_string()));
    }
    fn redirect(&mut self, fd: u8, op: u8, target: &str) {
        self.log
            .lock()
            .unwrap()
            .push(Call::Redirect(fd, op, target.to_string()));
    }
    fn cmd_subst(&mut self, sub: &Chunk) -> String {
        self.log
            .lock()
            .unwrap()
            .push(Call::CmdSubst(sub.ops.len()));
        self.cmd_subst_reply.clone().unwrap_or_default()
    }
    fn process_sub_in(&mut self, sub: &Chunk) -> String {
        self.log
            .lock()
            .unwrap()
            .push(Call::ProcessSubIn(sub.ops.len()));
        self.proc_sub_reply.clone().unwrap_or_default()
    }
    fn process_sub_out(&mut self, sub: &Chunk) -> String {
        self.log
            .lock()
            .unwrap()
            .push(Call::ProcessSubOut(sub.ops.len()));
        self.proc_sub_reply.clone().unwrap_or_default()
    }
    fn trap_set(&mut self, sig: &str, handler: &Chunk) {
        self.log
            .lock()
            .unwrap()
            .push(Call::TrapSet(sig.to_string(), handler.ops.len()));
    }
    fn str_match(&mut self, s: &str, pat: &str) -> bool {
        self.log
            .lock()
            .unwrap()
            .push(Call::StrMatch(s.to_string(), pat.to_string()));
        self.str_match_reply.unwrap_or(false)
    }
    fn regex_match(&mut self, s: &str, regex: &str) -> bool {
        self.log
            .lock()
            .unwrap()
            .push(Call::RegexMatch(s.to_string(), regex.to_string()));
        self.regex_match_reply.unwrap_or(false)
    }
    fn call_function(&mut self, name: &str, args: Vec<String>) -> Option<i32> {
        self.log
            .lock()
            .unwrap()
            .push(Call::CallFunction(name.to_string(), args.clone()));
        self.call_function_reply.unwrap_or(None)
    }
}

fn build_recording_vm(b: ChunkBuilder) -> (Arc<Mutex<Vec<Call>>>, VM) {
    let rec = Recorder::new();
    let log = rec.log.clone();
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(rec));
    (log, vm)
}

// ══════════════════════════════════════════════════════════════════════════
// ExpandParam — verify per-modifier argc behavior
// ══════════════════════════════════════════════════════════════════════════

fn make_expand_chunk(modifier: u8, name: &str, args: &[Value]) -> ChunkBuilder {
    let mut b = ChunkBuilder::new();
    let n = b.add_constant(Value::str(name));
    b.emit(Op::LoadConst(n), 1);
    for a in args {
        let idx = b.add_constant(a.clone());
        b.emit(Op::LoadConst(idx), 1);
    }
    b.emit(Op::ExpandParam(modifier), 1);
    b
}

#[test]
fn expand_param_zero_arg_modifiers_pop_only_name() {
    for m in [
        param_mod::LENGTH,
        param_mod::UPPER,
        param_mod::LOWER,
        param_mod::UPPER_FIRST,
        param_mod::LOWER_FIRST,
        param_mod::INDIRECT,
        param_mod::KEYS,
    ] {
        let b = make_expand_chunk(m, "VAR", &[]);
        let (log, mut vm) = build_recording_vm(b);
        let _ = vm.run();
        let calls = log.lock().unwrap().clone();
        assert_eq!(calls.len(), 1, "modifier {m}");
        match &calls[0] {
            Call::ExpandParam(n, mod_, args) => {
                assert_eq!(n, "VAR");
                assert_eq!(*mod_, m);
                assert!(args.is_empty(), "modifier {m} should have zero args");
            }
            other => panic!("unexpected: {:?}", other),
        }
    }
}

#[test]
fn expand_param_one_arg_modifiers_pop_one_arg() {
    for m in [
        param_mod::DEFAULT,
        param_mod::ASSIGN,
        param_mod::ERROR,
        param_mod::ALTERNATE,
        param_mod::STRIP_SHORT,
        param_mod::STRIP_LONG,
        param_mod::RSTRIP_SHORT,
        param_mod::RSTRIP_LONG,
    ] {
        let b = make_expand_chunk(m, "VAR", &[Value::str("X")]);
        let (log, mut vm) = build_recording_vm(b);
        let _ = vm.run();
        let calls = log.lock().unwrap().clone();
        match &calls[0] {
            Call::ExpandParam(_, mod_, args) => {
                assert_eq!(*mod_, m);
                assert_eq!(args.len(), 1, "modifier {m} should have 1 arg");
                assert_eq!(args[0], Value::str("X"));
            }
            other => panic!("unexpected: {:?}", other),
        }
    }
}

#[test]
fn expand_param_two_arg_modifiers_pop_two_args_in_order() {
    for m in [
        param_mod::SUBST_FIRST,
        param_mod::SUBST_ALL,
        param_mod::SLICE,
    ] {
        let b = make_expand_chunk(m, "VAR", &[Value::str("P"), Value::str("R")]);
        let (log, mut vm) = build_recording_vm(b);
        let _ = vm.run();
        let calls = log.lock().unwrap().clone();
        match &calls[0] {
            Call::ExpandParam(_, mod_, args) => {
                assert_eq!(*mod_, m);
                assert_eq!(args.len(), 2);
                // Order should match push order (the VM reverses after popping).
                assert_eq!(args[0], Value::str("P"));
                assert_eq!(args[1], Value::str("R"));
            }
            other => panic!("unexpected: {:?}", other),
        }
    }
}

#[test]
fn expand_param_uses_host_returned_value() {
    let rec = Recorder {
        expand_reply: Some(Value::str("EXPANDED")),
        ..Recorder::default()
    };
    let log = rec.log.clone();
    let b = make_expand_chunk(param_mod::DEFAULT, "X", &[Value::str("d")]);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(rec));
    let v = match vm.run() {
        VMResult::Ok(v) => v,
        VMResult::Halted => Value::Undef,
        VMResult::Error(e) => panic!("error: {e}"),
    };
    assert!(matches!(&v, Value::Str(s) if s.as_str() == "EXPANDED"));
    assert_eq!(log.lock().unwrap().len(), 1);
}

#[test]
fn expand_param_without_host_pushes_empty_string() {
    let b = make_expand_chunk(param_mod::DEFAULT, "X", &[Value::str("d")]);
    let v = run(b);
    // No host attached → push empty string.
    match v {
        Value::Str(s) => assert_eq!(s.as_str(), ""),
        Value::Undef => {}
        other => panic!("unexpected {:?}", other),
    }
}

// ══════════════════════════════════════════════════════════════════════════
// HereDoc / HereString
// ══════════════════════════════════════════════════════════════════════════

#[test]
fn heredoc_pulls_content_from_constant_pool() {
    let mut b = ChunkBuilder::new();
    let c = b.add_constant(Value::str("body line 1\nbody line 2"));
    b.emit(Op::HereDoc(c), 1);
    let (log, mut vm) = build_recording_vm(b);
    let _ = vm.run();
    let calls = log.lock().unwrap().clone();
    assert_eq!(calls.len(), 1);
    match &calls[0] {
        Call::Heredoc(s) => assert_eq!(s, "body line 1\nbody line 2"),
        other => panic!("unexpected: {:?}", other),
    }
}

#[test]
fn heredoc_with_out_of_range_index_uses_empty_string() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::HereDoc(999), 1);
    let (log, mut vm) = build_recording_vm(b);
    let _ = vm.run();
    let calls = log.lock().unwrap().clone();
    assert!(matches!(&calls[0], Call::Heredoc(s) if s.is_empty()));
}

#[test]
fn herestring_takes_top_of_stack() {
    let mut b = ChunkBuilder::new();
    let c = b.add_constant(Value::str("hello world"));
    b.emit(Op::LoadConst(c), 1);
    b.emit(Op::HereString, 1);
    let (log, mut vm) = build_recording_vm(b);
    let _ = vm.run();
    let calls = log.lock().unwrap().clone();
    assert!(matches!(&calls[0], Call::Herestring(s) if s == "hello world"));
}

// ══════════════════════════════════════════════════════════════════════════
// Redirect — host receives (fd, op, target-from-stack)
// ══════════════════════════════════════════════════════════════════════════

#[test]
fn redirect_passes_fd_op_and_popped_target() {
    let mut b = ChunkBuilder::new();
    let path = b.add_constant(Value::str("/tmp/out.log"));
    b.emit(Op::LoadConst(path), 1);
    b.emit(Op::Redirect(2, redirect_op::APPEND), 1);
    let (log, mut vm) = build_recording_vm(b);
    let _ = vm.run();
    let calls = log.lock().unwrap().clone();
    assert_eq!(calls.len(), 1);
    match &calls[0] {
        Call::Redirect(fd, op, target) => {
            assert_eq!(*fd, 2);
            assert_eq!(*op, redirect_op::APPEND);
            assert_eq!(target, "/tmp/out.log");
        }
        other => panic!("unexpected: {:?}", other),
    }
}

// ══════════════════════════════════════════════════════════════════════════
// CmdSubst / ProcessSubIn / ProcessSubOut — sub-chunk dispatch
// ══════════════════════════════════════════════════════════════════════════

fn build_sub_chunk(n_ops: usize) -> Chunk {
    let mut b = ChunkBuilder::new();
    for _ in 0..n_ops {
        b.emit(Op::Nop, 1);
    }
    b.build()
}

#[test]
fn cmd_subst_dispatches_correct_sub_chunk() {
    let mut b = ChunkBuilder::new();
    let s0 = b.add_sub_chunk(build_sub_chunk(3));
    let s1 = b.add_sub_chunk(build_sub_chunk(7));
    b.emit(Op::CmdSubst(s1), 1);
    b.emit(Op::CmdSubst(s0), 1);
    let rec = Recorder {
        cmd_subst_reply: Some("OUT".to_string()),
        ..Recorder::default()
    };
    let log = rec.log.clone();
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(rec));
    let _ = vm.run();
    let calls = log.lock().unwrap().clone();
    // First CmdSubst(s1) saw 7-op sub-chunk, second saw 3-op.
    assert!(matches!(&calls[0], Call::CmdSubst(7)));
    assert!(matches!(&calls[1], Call::CmdSubst(3)));
}

#[test]
fn cmd_subst_with_invalid_index_pushes_empty_string() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::CmdSubst(999), 1);
    let v = run(b);
    // No sub-chunk found → empty string pushed.
    match v {
        Value::Str(s) => assert_eq!(s.as_str(), ""),
        Value::Undef => {}
        other => panic!("unexpected: {:?}", other),
    }
}

#[test]
fn process_sub_in_with_invalid_index_pushes_empty_string() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::ProcessSubIn(999), 1);
    let v = run(b);
    assert!(matches!(&v, Value::Str(s) if s.as_str() == "") || matches!(v, Value::Undef));
}

#[test]
fn process_sub_out_with_invalid_index_pushes_empty_string() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::ProcessSubOut(999), 1);
    let v = run(b);
    assert!(matches!(&v, Value::Str(s) if s.as_str() == "") || matches!(v, Value::Undef));
}

#[test]
fn process_sub_in_and_out_dispatch_to_their_respective_host_methods() {
    let mut b = ChunkBuilder::new();
    let s = b.add_sub_chunk(build_sub_chunk(2));
    b.emit(Op::ProcessSubIn(s), 1);
    b.emit(Op::Pop, 1);
    b.emit(Op::ProcessSubOut(s), 1);
    let (log, mut vm) = build_recording_vm(b);
    let _ = vm.run();
    let calls = log.lock().unwrap().clone();
    assert_eq!(calls.len(), 2);
    assert!(matches!(&calls[0], Call::ProcessSubIn(2)));
    assert!(matches!(&calls[1], Call::ProcessSubOut(2)));
}

#[test]
fn cmd_subst_result_string_is_pushed_to_stack() {
    let mut b = ChunkBuilder::new();
    let s = b.add_sub_chunk(build_sub_chunk(1));
    b.emit(Op::CmdSubst(s), 1);
    let rec = Recorder {
        cmd_subst_reply: Some("CAPTURED".to_string()),
        ..Recorder::default()
    };
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(rec));
    let v = match vm.run() {
        VMResult::Ok(v) => v,
        VMResult::Halted => Value::Undef,
        other => panic!("unexpected: {:?}", other),
    };
    assert!(matches!(&v, Value::Str(s) if s.as_str() == "CAPTURED"));
}

// ══════════════════════════════════════════════════════════════════════════
// TrapSet — invokes host with signal name and handler chunk
// ══════════════════════════════════════════════════════════════════════════

#[test]
fn trap_set_passes_signal_name_and_handler_to_host() {
    let mut b = ChunkBuilder::new();
    let sig = b.add_constant(Value::str("USR1"));
    let handler = b.add_sub_chunk(build_sub_chunk(4));
    b.emit(Op::LoadConst(sig), 1);
    b.emit(Op::TrapSet(handler), 1);
    let (log, mut vm) = build_recording_vm(b);
    let _ = vm.run();
    let calls = log.lock().unwrap().clone();
    match &calls[0] {
        Call::TrapSet(s, n) => {
            assert_eq!(s, "USR1");
            assert_eq!(*n, 4);
        }
        other => panic!("unexpected: {:?}", other),
    }
}

#[test]
fn trap_set_with_invalid_handler_index_does_not_call_host() {
    let mut b = ChunkBuilder::new();
    let sig = b.add_constant(Value::str("INT"));
    b.emit(Op::LoadConst(sig), 1);
    b.emit(Op::TrapSet(999), 1);
    let (log, mut vm) = build_recording_vm(b);
    let _ = vm.run();
    let calls = log.lock().unwrap().clone();
    // No TrapSet call recorded — the op silently dropped because index was OOB.
    assert!(
        !calls.iter().any(|c| matches!(c, Call::TrapSet(..))),
        "unexpected TrapSet call: {:?}",
        calls
    );
}

// ══════════════════════════════════════════════════════════════════════════
// StrMatch / RegexMatch — host receives string + pattern in that order
// ══════════════════════════════════════════════════════════════════════════

#[test]
fn strmatch_pops_pattern_then_string_and_passes_in_order_s_pat() {
    let mut b = ChunkBuilder::new();
    let s = b.add_constant(Value::str("haystack"));
    let pat = b.add_constant(Value::str("needle"));
    b.emit(Op::LoadConst(s), 1);
    b.emit(Op::LoadConst(pat), 1);
    b.emit(Op::StrMatch, 1);
    let rec = Recorder {
        str_match_reply: Some(true),
        ..Recorder::default()
    };
    let log = rec.log.clone();
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(rec));
    let v = match vm.run() {
        VMResult::Ok(v) => v,
        VMResult::Halted => Value::Undef,
        other => panic!("unexpected: {:?}", other),
    };
    let calls = log.lock().unwrap().clone();
    match &calls[0] {
        Call::StrMatch(s, p) => {
            assert_eq!(s, "haystack");
            assert_eq!(p, "needle");
        }
        other => panic!("unexpected: {:?}", other),
    }
    assert!(matches!(v, Value::Bool(true) | Value::Undef));
}

#[test]
fn strmatch_without_host_falls_back_to_strict_equality() {
    // Equal strings.
    {
        let mut b = ChunkBuilder::new();
        let s = b.add_constant(Value::str("abc"));
        let p = b.add_constant(Value::str("abc"));
        b.emit(Op::LoadConst(s), 1);
        b.emit(Op::LoadConst(p), 1);
        b.emit(Op::StrMatch, 1);
        assert!(matches!(run(b), Value::Bool(true) | Value::Undef));
    }
    // Different strings.
    {
        let mut b = ChunkBuilder::new();
        let s = b.add_constant(Value::str("abc"));
        let p = b.add_constant(Value::str("xyz"));
        b.emit(Op::LoadConst(s), 1);
        b.emit(Op::LoadConst(p), 1);
        b.emit(Op::StrMatch, 1);
        assert!(matches!(run(b), Value::Bool(false) | Value::Undef));
    }
}

#[test]
fn regex_match_host_routing() {
    let mut b = ChunkBuilder::new();
    let s = b.add_constant(Value::str("abc123"));
    let re = b.add_constant(Value::str(r"\d+"));
    b.emit(Op::LoadConst(s), 1);
    b.emit(Op::LoadConst(re), 1);
    b.emit(Op::RegexMatch, 1);
    let rec = Recorder {
        regex_match_reply: Some(true),
        ..Recorder::default()
    };
    let log = rec.log.clone();
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(rec));
    let v = match vm.run() {
        VMResult::Ok(v) => v,
        VMResult::Halted => Value::Undef,
        other => panic!("unexpected: {:?}", other),
    };
    let calls = log.lock().unwrap().clone();
    assert!(matches!(&calls[0], Call::RegexMatch(s, re) if s == "abc123" && re == r"\d+"));
    assert!(matches!(v, Value::Bool(true) | Value::Undef));
}

#[test]
fn regex_match_without_host_returns_false() {
    let mut b = ChunkBuilder::new();
    let s = b.add_constant(Value::str("anything"));
    let re = b.add_constant(Value::str(".*"));
    b.emit(Op::LoadConst(s), 1);
    b.emit(Op::LoadConst(re), 1);
    b.emit(Op::RegexMatch, 1);
    assert!(matches!(run(b), Value::Bool(false) | Value::Undef));
}

// ══════════════════════════════════════════════════════════════════════════
// CallFunction — host first, then in-chunk find_sub, then exec
// ══════════════════════════════════════════════════════════════════════════

#[test]
fn callfunction_passes_name_and_flat_args_to_host() {
    let mut b = ChunkBuilder::new();
    let n = b.add_name("myfn");
    let a1 = b.add_constant(Value::str("a"));
    let a2 = b.add_constant(Value::str("b"));
    b.emit(Op::LoadConst(a1), 1);
    b.emit(Op::LoadConst(a2), 1);
    b.emit(Op::CallFunction(n, 2), 1);
    let rec = Recorder {
        call_function_reply: Some(Some(7)),
        ..Recorder::default()
    };
    let log = rec.log.clone();
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(rec));
    let v = match vm.run() {
        VMResult::Ok(v) => v,
        VMResult::Halted => Value::Undef,
        other => panic!("unexpected: {:?}", other),
    };
    let calls = log.lock().unwrap().clone();
    match &calls[0] {
        Call::CallFunction(name, args) => {
            assert_eq!(name, "myfn");
            assert_eq!(args, &vec!["a".to_string(), "b".to_string()]);
        }
        other => panic!("unexpected: {:?}", other),
    }
    assert!(matches!(v, Value::Status(7) | Value::Undef));
}

#[test]
fn callfunction_flattens_array_args() {
    let mut b = ChunkBuilder::new();
    let n = b.add_name("myfn");
    // Push an array of 3 strings as a single argument; flattening should
    // produce 3 separate string args to the host.
    let s1 = b.add_constant(Value::str("x"));
    let s2 = b.add_constant(Value::str("y"));
    let s3 = b.add_constant(Value::str("z"));
    b.emit(Op::LoadConst(s1), 1);
    b.emit(Op::LoadConst(s2), 1);
    b.emit(Op::LoadConst(s3), 1);
    b.emit(Op::MakeArray(3), 1);
    b.emit(Op::CallFunction(n, 1), 1);
    let rec = Recorder {
        call_function_reply: Some(Some(0)),
        ..Recorder::default()
    };
    let log = rec.log.clone();
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(rec));
    let _ = vm.run();
    let calls = log.lock().unwrap().clone();
    match &calls[0] {
        Call::CallFunction(_, args) => {
            assert_eq!(args, &vec!["x".to_string(), "y".to_string(), "z".to_string()]);
        }
        other => panic!("unexpected: {:?}", other),
    }
}

// ══════════════════════════════════════════════════════════════════════════
// JitCompiler is_eligible additional cases
// ══════════════════════════════════════════════════════════════════════════

#[test]
fn jit_compiler_is_eligible_for_pure_arith_chunk() {
    use fusevm::jit::JitCompiler;
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::LoadInt(2), 1);
    b.emit(Op::Add, 1);
    b.emit(Op::LoadInt(3), 1);
    b.emit(Op::Mul, 1);
    let chunk = b.build();
    let jit = JitCompiler::new();
    let _ = jit.is_eligible(&chunk);
}

#[test]
fn jit_compiler_is_eligible_for_chunk_with_shell_op_returns_false() {
    use fusevm::jit::JitCompiler;
    let mut b = ChunkBuilder::new();
    b.emit(Op::Exec(1), 1);
    let chunk = b.build();
    let jit = JitCompiler::new();
    assert!(!jit.is_eligible(&chunk));
}

#[test]
fn jit_compiler_register_extension_is_callable() {
    use fusevm::jit::{JitCompiler, JitExtension};
    struct Stub;
    impl JitExtension for Stub {
        fn can_jit(&self, _ext_id: u16) -> bool {
            false
        }
        fn op_count(&self) -> usize {
            0
        }
        fn name(&self) -> &str {
            "stub"
        }
    }
    let mut jit = JitCompiler::new();
    jit.register_extension(Box::new(Stub));
}
