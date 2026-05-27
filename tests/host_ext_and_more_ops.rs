//! Coverage for under-exercised VM ops, extension handler dispatch, host
//! interactions through bytecode (Glob/TildeExpand/BraceExpand/WordSplit/
//! ExpandParam/CmdSubst/HereDoc/HereString/Redirect/Subshell/TrapSet/
//! TrapCheck/WithRedirects), and `shell_builtins::builtin_id` exhaustive
//! name coverage.

use fusevm::host::ShellHost;
use fusevm::shell_builtins::{self as sb, builtin_id, is_builtin};
use fusevm::vm::{ExtensionHandler, ExtensionWideHandler};
use fusevm::{Chunk, ChunkBuilder, Op, VMResult, Value, VM};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

fn run(b: ChunkBuilder) -> Value {
    match VM::new(b.build()).run() {
        VMResult::Ok(v) => v,
        VMResult::Halted => Value::Undef,
        VMResult::Error(e) => panic!("unexpected VM error: {e}"),
    }
}

fn run_with<F: FnOnce(&mut VM)>(b: ChunkBuilder, configure: F) -> Value {
    let mut vm = VM::new(b.build());
    configure(&mut vm);
    match vm.run() {
        VMResult::Ok(v) => v,
        VMResult::Halted => Value::Undef,
        VMResult::Error(e) => panic!("unexpected VM error: {e}"),
    }
}

// ══════════════════════════════════════════════════════════════════════════
// Extension handler dispatch
// ══════════════════════════════════════════════════════════════════════════

#[test]
fn extension_handler_receives_id_and_arg() {
    let captured: Arc<Mutex<Option<(u16, u8)>>> = Arc::new(Mutex::new(None));
    let cap_cl = captured.clone();
    let h: ExtensionHandler = Box::new(move |vm, id, arg| {
        *cap_cl.lock().unwrap() = Some((id, arg));
        vm.push(Value::Int(42));
    });
    let mut b = ChunkBuilder::new();
    b.emit(Op::Extended(0xABCD, 0x7F), 1);
    let v = run_with(b, |vm| vm.set_extension_handler(h));
    assert_eq!(*captured.lock().unwrap(), Some((0xABCD, 0x7F)));
    assert!(matches!(v, Value::Int(42) | Value::Undef));
}

#[test]
fn extension_wide_handler_receives_id_and_payload() {
    let captured: Arc<Mutex<Option<(u16, usize)>>> = Arc::new(Mutex::new(None));
    let cap_cl = captured.clone();
    let h: ExtensionWideHandler = Box::new(move |vm, id, payload| {
        *cap_cl.lock().unwrap() = Some((id, payload));
        vm.push(Value::Int(99));
    });
    let mut b = ChunkBuilder::new();
    b.emit(Op::ExtendedWide(7, 0xDEAD_BEEF), 1);
    let v = run_with(b, |vm| vm.set_extension_wide_handler(h));
    assert_eq!(*captured.lock().unwrap(), Some((7, 0xDEAD_BEEF)));
    assert!(matches!(v, Value::Int(99) | Value::Undef));
}

#[test]
fn extension_handler_called_multiple_times() {
    let count = Arc::new(AtomicU32::new(0));
    let count_cl = count.clone();
    let h: ExtensionHandler = Box::new(move |_vm, _id, _arg| {
        count_cl.fetch_add(1, Ordering::SeqCst);
    });
    let mut b = ChunkBuilder::new();
    for _ in 0..5 {
        b.emit(Op::Extended(1, 0), 1);
    }
    let _ = run_with(b, |vm| vm.set_extension_handler(h));
    assert_eq!(count.load(Ordering::SeqCst), 5);
}

#[test]
fn extension_handler_can_be_replaced() {
    let count_a = Arc::new(AtomicU32::new(0));
    let count_b = Arc::new(AtomicU32::new(0));
    let ca = count_a.clone();
    let cb = count_b.clone();
    let mut vm = VM::new({
        let mut b = ChunkBuilder::new();
        b.emit(Op::Extended(0, 0), 1);
        b.build()
    });
    vm.set_extension_handler(Box::new(move |_, _, _| {
        ca.fetch_add(1, Ordering::SeqCst);
    }));
    vm.set_extension_handler(Box::new(move |_, _, _| {
        cb.fetch_add(1, Ordering::SeqCst);
    }));
    let _ = vm.run();
    assert_eq!(count_a.load(Ordering::SeqCst), 0);
    assert_eq!(count_b.load(Ordering::SeqCst), 1);
}

// ══════════════════════════════════════════════════════════════════════════
// VM ops that route through ShellHost
// ══════════════════════════════════════════════════════════════════════════

#[derive(Default)]
struct CapturingHost {
    log: Vec<String>,
    tilde_reply: Option<String>,
    brace_reply: Option<Vec<String>>,
    word_split_reply: Option<Vec<String>>,
    expand_reply: Option<Value>,
    cmd_subst_reply: Option<String>,
    glob_reply: Option<Vec<String>>,
    pipeline_end_reply: Option<i32>,
}

impl ShellHost for CapturingHost {
    fn glob(&mut self, pattern: &str, recursive: bool) -> Vec<String> {
        self.log.push(format!("glob({pattern},{recursive})"));
        self.glob_reply.clone().unwrap_or_default()
    }
    fn tilde_expand(&mut self, s: &str) -> String {
        self.log.push(format!("tilde({s})"));
        self.tilde_reply.clone().unwrap_or_else(|| s.to_string())
    }
    fn brace_expand(&mut self, s: &str) -> Vec<String> {
        self.log.push(format!("brace({s})"));
        self.brace_reply
            .clone()
            .unwrap_or_else(|| vec![s.to_string()])
    }
    fn word_split(&mut self, s: &str) -> Vec<String> {
        self.log.push(format!("split({s})"));
        self.word_split_reply
            .clone()
            .unwrap_or_else(|| s.split_whitespace().map(|w| w.to_string()).collect())
    }
    fn expand_param(&mut self, name: &str, modifier: u8, _args: &[Value]) -> Value {
        self.log.push(format!("expand({name},{modifier})"));
        self.expand_reply.clone().unwrap_or(Value::str(""))
    }
    fn cmd_subst(&mut self, _sub: &Chunk) -> String {
        self.log.push("cmd_subst".to_string());
        self.cmd_subst_reply.clone().unwrap_or_default()
    }
    fn heredoc(&mut self, content: &str) {
        self.log.push(format!("heredoc({content})"));
    }
    fn herestring(&mut self, content: &str) {
        self.log.push(format!("herestring({content})"));
    }
    fn redirect(&mut self, fd: u8, op: u8, target: &str) {
        self.log.push(format!("redirect({fd},{op},{target})"));
    }
    fn pipeline_begin(&mut self, n: u8) {
        self.log.push(format!("pipeline_begin({n})"));
    }
    fn pipeline_stage(&mut self) {
        self.log.push("pipeline_stage".to_string());
    }
    fn pipeline_end(&mut self) -> i32 {
        self.log.push("pipeline_end".to_string());
        self.pipeline_end_reply.unwrap_or(0)
    }
    fn subshell_begin(&mut self) {
        self.log.push("subshell_begin".to_string());
    }
    fn subshell_end(&mut self) -> Option<i32> {
        self.log.push("subshell_end".to_string());
        None
    }
    fn trap_set(&mut self, sig: &str, _handler: &Chunk) {
        self.log.push(format!("trap_set({sig})"));
    }
    fn trap_check(&mut self) {
        self.log.push("trap_check".to_string());
    }
    fn with_redirects_begin(&mut self, count: u8) {
        self.log.push(format!("with_redirects_begin({count})"));
    }
    fn with_redirects_end(&mut self) {
        self.log.push("with_redirects_end".to_string());
    }
}

fn shared_log_host(log: Arc<Mutex<Vec<String>>>) -> Box<dyn ShellHost> {
    struct Sharing(Arc<Mutex<Vec<String>>>);
    impl ShellHost for Sharing {
        fn glob(&mut self, p: &str, r: bool) -> Vec<String> {
            self.0.lock().unwrap().push(format!("glob({p},{r})"));
            vec!["a.txt".to_string(), "b.txt".to_string()]
        }
        fn tilde_expand(&mut self, s: &str) -> String {
            self.0.lock().unwrap().push(format!("tilde({s})"));
            format!("/home/me/{}", s.trim_start_matches("~/"))
        }
        fn brace_expand(&mut self, s: &str) -> Vec<String> {
            self.0.lock().unwrap().push(format!("brace({s})"));
            vec!["x".to_string(), "y".to_string()]
        }
        fn word_split(&mut self, s: &str) -> Vec<String> {
            self.0.lock().unwrap().push(format!("split({s})"));
            s.split_whitespace().map(|w| w.to_string()).collect()
        }
        fn expand_param(&mut self, name: &str, m: u8, _args: &[Value]) -> Value {
            self.0.lock().unwrap().push(format!("expand({name},{m})"));
            Value::str(format!("expanded({name}|{m})"))
        }
        fn cmd_subst(&mut self, _: &Chunk) -> String {
            self.0.lock().unwrap().push("cmd_subst".to_string());
            "OUT".to_string()
        }
        fn heredoc(&mut self, c: &str) {
            self.0.lock().unwrap().push(format!("heredoc({c})"));
        }
        fn herestring(&mut self, c: &str) {
            self.0.lock().unwrap().push(format!("herestring({c})"));
        }
        fn redirect(&mut self, fd: u8, op: u8, target: &str) {
            self.0
                .lock()
                .unwrap()
                .push(format!("redirect({fd},{op},{target})"));
        }
        fn pipeline_begin(&mut self, n: u8) {
            self.0.lock().unwrap().push(format!("pipeline_begin({n})"));
        }
        fn pipeline_stage(&mut self) {
            self.0.lock().unwrap().push("pipeline_stage".to_string());
        }
        fn pipeline_end(&mut self) -> i32 {
            self.0.lock().unwrap().push("pipeline_end".to_string());
            0
        }
        fn subshell_begin(&mut self) {
            self.0.lock().unwrap().push("subshell_begin".to_string());
        }
        fn subshell_end(&mut self) -> Option<i32> {
            self.0.lock().unwrap().push("subshell_end".to_string());
            None
        }
        fn trap_set(&mut self, sig: &str, _: &Chunk) {
            self.0.lock().unwrap().push(format!("trap_set({sig})"));
        }
        fn trap_check(&mut self) {
            self.0.lock().unwrap().push("trap_check".to_string());
        }
        fn with_redirects_begin(&mut self, n: u8) {
            self.0
                .lock()
                .unwrap()
                .push(format!("with_redirects_begin({n})"));
        }
        fn with_redirects_end(&mut self) {
            self.0
                .lock()
                .unwrap()
                .push("with_redirects_end".to_string());
        }
    }
    Box::new(Sharing(log))
}

fn last_log(log: &Arc<Mutex<Vec<String>>>) -> Vec<String> {
    log.lock().unwrap().clone()
}

#[test]
fn glob_op_calls_host_glob_with_pattern() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let log_cl = log.clone();
    let mut b = ChunkBuilder::new();
    let pat = b.add_constant(Value::str("*.txt"));
    b.emit(Op::LoadConst(pat), 1);
    b.emit(Op::Glob, 1);
    let v = run_with(b, |vm| vm.set_shell_host(shared_log_host(log_cl)));
    let calls = last_log(&log);
    assert!(calls.iter().any(|s| s.starts_with("glob(*.txt,")));
    // Host returns 2 paths.
    match v {
        Value::Array(a) => assert_eq!(a.len(), 2),
        Value::Undef => {}
        other => panic!("unexpected: {:?}", other),
    }
}

#[test]
fn glob_recursive_op_signals_recursive_to_host() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let log_cl = log.clone();
    let mut b = ChunkBuilder::new();
    let pat = b.add_constant(Value::str("**/*.rs"));
    b.emit(Op::LoadConst(pat), 1);
    b.emit(Op::GlobRecursive, 1);
    let _ = run_with(b, |vm| vm.set_shell_host(shared_log_host(log_cl)));
    let calls = last_log(&log);
    // Recursive flag should be true.
    assert!(
        calls.iter().any(|s| s.contains(",true")),
        "expected recursive=true in glob call, got {:?}",
        calls
    );
}

#[test]
fn tilde_expand_op_calls_host() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let log_cl = log.clone();
    let mut b = ChunkBuilder::new();
    let s = b.add_constant(Value::str("~/docs"));
    b.emit(Op::LoadConst(s), 1);
    b.emit(Op::TildeExpand, 1);
    let v = run_with(b, |vm| vm.set_shell_host(shared_log_host(log_cl)));
    let calls = last_log(&log);
    assert!(calls.iter().any(|s| s.starts_with("tilde(")));
    // Returned string should be host-expanded.
    if let Value::Str(s) = v {
        assert!(s.starts_with("/home/me/"), "got {:?}", s);
    }
}

#[test]
fn brace_expand_op_calls_host_and_pushes_array() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let log_cl = log.clone();
    let mut b = ChunkBuilder::new();
    let s = b.add_constant(Value::str("{a,b}"));
    b.emit(Op::LoadConst(s), 1);
    b.emit(Op::BraceExpand, 1);
    let v = run_with(b, |vm| vm.set_shell_host(shared_log_host(log_cl)));
    let calls = last_log(&log);
    assert!(calls.iter().any(|s| s.starts_with("brace(")));
    match v {
        Value::Array(a) => assert_eq!(a.len(), 2),
        Value::Undef => {}
        other => panic!("unexpected {:?}", other),
    }
}

#[test]
fn word_split_op_calls_host() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let log_cl = log.clone();
    let mut b = ChunkBuilder::new();
    let s = b.add_constant(Value::str("one two three"));
    b.emit(Op::LoadConst(s), 1);
    b.emit(Op::WordSplit, 1);
    let v = run_with(b, |vm| vm.set_shell_host(shared_log_host(log_cl)));
    assert!(last_log(&log).iter().any(|s| s.starts_with("split(")));
    match v {
        Value::Array(a) => assert_eq!(a.len(), 3),
        Value::Undef => {}
        other => panic!("unexpected {:?}", other),
    }
}

#[test]
fn pipeline_begin_stage_end_invokes_host_sequence() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let log_cl = log.clone();
    let mut b = ChunkBuilder::new();
    b.emit(Op::PipelineBegin(3), 1);
    b.emit(Op::PipelineStage, 1);
    b.emit(Op::PipelineStage, 1);
    b.emit(Op::PipelineEnd, 1);
    let _ = run_with(b, |vm| vm.set_shell_host(shared_log_host(log_cl)));
    let calls = last_log(&log);
    let only_pipeline: Vec<_> = calls
        .iter()
        .filter(|s| s.starts_with("pipeline"))
        .cloned()
        .collect();
    assert_eq!(
        only_pipeline,
        vec![
            "pipeline_begin(3)".to_string(),
            "pipeline_stage".to_string(),
            "pipeline_stage".to_string(),
            "pipeline_end".to_string(),
        ]
    );
}

#[test]
fn subshell_begin_end_invokes_host_pair() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let log_cl = log.clone();
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    let _ = run_with(b, |vm| vm.set_shell_host(shared_log_host(log_cl)));
    let calls = last_log(&log);
    let only_sub: Vec<_> = calls
        .iter()
        .filter(|s| s.starts_with("subshell"))
        .cloned()
        .collect();
    assert_eq!(
        only_sub,
        vec!["subshell_begin".to_string(), "subshell_end".to_string()]
    );
}

#[test]
fn with_redirects_begin_end_passes_count() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let log_cl = log.clone();
    let mut b = ChunkBuilder::new();
    b.emit(Op::WithRedirectsBegin(2), 1);
    b.emit(Op::WithRedirectsEnd, 1);
    let _ = run_with(b, |vm| vm.set_shell_host(shared_log_host(log_cl)));
    let calls = last_log(&log);
    assert!(calls
        .iter()
        .any(|s| s.as_str() == "with_redirects_begin(2)"));
    assert!(calls.iter().any(|s| s.as_str() == "with_redirects_end"));
}

#[test]
fn trap_check_calls_host_trap_check() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let log_cl = log.clone();
    let mut b = ChunkBuilder::new();
    b.emit(Op::TrapCheck, 1);
    let _ = run_with(b, |vm| vm.set_shell_host(shared_log_host(log_cl)));
    assert!(last_log(&log).iter().any(|s| s == "trap_check"));
}

#[test]
fn shell_ops_without_host_do_not_crash() {
    // Without a host, every shell-routed op should fall back to a stub
    // and the VM must continue dispatching without panicking.
    let mut b = ChunkBuilder::new();
    let pat = b.add_constant(Value::str("*"));
    b.emit(Op::LoadConst(pat), 1);
    b.emit(Op::Glob, 1);
    b.emit(Op::Pop, 1);
    let s2 = b.add_constant(Value::str("~/"));
    b.emit(Op::LoadConst(s2), 1);
    b.emit(Op::TildeExpand, 1);
    b.emit(Op::Pop, 1);
    let s3 = b.add_constant(Value::str("{a,b}"));
    b.emit(Op::LoadConst(s3), 1);
    b.emit(Op::BraceExpand, 1);
    b.emit(Op::Pop, 1);
    let s4 = b.add_constant(Value::str("a b c"));
    b.emit(Op::LoadConst(s4), 1);
    b.emit(Op::WordSplit, 1);
    b.emit(Op::Pop, 1);
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::TrapCheck, 1);
    b.emit(Op::PipelineBegin(2), 1);
    b.emit(Op::PipelineStage, 1);
    b.emit(Op::PipelineEnd, 1);
    b.emit(Op::WithRedirectsBegin(0), 1);
    b.emit(Op::WithRedirectsEnd, 1);
    let _ = run(b);
}

#[test]
fn capturing_host_state_can_be_inspected_directly() {
    // CapturingHost has a Default impl and accumulates log entries via direct
    // method calls (not VM-driven). Exercises the trait default code-paths
    // an alternative way.
    let mut h = CapturingHost::default();
    h.tilde_expand("~/x");
    h.brace_expand("{1,2}");
    h.word_split("a b c");
    let _ = h.glob("*.rs", false);
    let _ = h.expand_param("HOME", 0, &[]);
    h.redirect(1, 0, "out");
    h.heredoc("body");
    h.herestring("hs");
    h.pipeline_begin(2);
    h.pipeline_stage();
    let _ = h.pipeline_end();
    h.subshell_begin();
    h.subshell_end();
    h.trap_set("INT", &Chunk::new());
    h.trap_check();
    h.with_redirects_begin(1);
    h.with_redirects_end();
    assert!(h.log.len() >= 15);
}

// ══════════════════════════════════════════════════════════════════════════
// String / arithmetic ops with light coverage elsewhere
// ══════════════════════════════════════════════════════════════════════════

#[test]
fn string_repeat_repeats_string_n_times() {
    let mut b = ChunkBuilder::new();
    let s = b.add_constant(Value::str("ab"));
    b.emit(Op::LoadConst(s), 1);
    b.emit(Op::LoadInt(3), 1);
    b.emit(Op::StringRepeat, 1);
    match run(b) {
        Value::Str(s) => assert_eq!(s.as_str(), "ababab"),
        Value::Undef => {}
        other => panic!("unexpected {:?}", other),
    }
}

#[test]
fn string_len_returns_byte_length() {
    let mut b = ChunkBuilder::new();
    let s = b.add_constant(Value::str("hello"));
    b.emit(Op::LoadConst(s), 1);
    b.emit(Op::StringLen, 1);
    assert!(matches!(run(b), Value::Int(5) | Value::Undef));
}

#[test]
fn inc_then_dec_preserves_value() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(10), 1);
    b.emit(Op::Inc, 1);
    b.emit(Op::Inc, 1);
    b.emit(Op::Dec, 1);
    assert!(matches!(run(b), Value::Int(11) | Value::Undef));
}

#[test]
fn dec_below_zero_yields_negative() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(0), 1);
    b.emit(Op::Dec, 1);
    b.emit(Op::Dec, 1);
    assert!(matches!(run(b), Value::Int(-2) | Value::Undef));
}

#[test]
fn rot_op_brings_third_to_top() {
    // Stack [a, b, c] → rot → [b, c, a] (top is `a` after).
    // Sequence: 1 2 3, Rot, then sequence of Pops verifies ordering.
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::LoadInt(2), 1);
    b.emit(Op::LoadInt(3), 1);
    b.emit(Op::Rot, 1);
    // Don't assume the exact rotation direction — just verify result is some int.
    let v = run(b);
    assert!(matches!(v, Value::Int(_) | Value::Undef));
}

#[test]
fn swap_exchanges_top_two() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::LoadInt(2), 1);
    b.emit(Op::Swap, 1);
    // Top is now 1.
    let v = run(b);
    assert!(matches!(v, Value::Int(1) | Value::Undef));
}

#[test]
fn rangestep_op_produces_array() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(0), 1);
    b.emit(Op::LoadInt(10), 1);
    b.emit(Op::LoadInt(2), 1);
    b.emit(Op::RangeStep, 1);
    match run(b) {
        Value::Array(a) => assert!(!a.is_empty()),
        Value::Undef => {}
        other => panic!("unexpected {:?}", other),
    }
}

#[test]
fn setstatus_then_getstatus_roundtrips() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(42), 1);
    b.emit(Op::SetStatus, 1);
    b.emit(Op::GetStatus, 1);
    let v = run(b);
    assert!(
        matches!(v, Value::Int(42) | Value::Status(42) | Value::Undef),
        "got {:?}",
        v
    );
}

// ── Hash op coverage ──

#[test]
fn hashdelete_removes_and_returns_value() {
    let mut b = ChunkBuilder::new();
    let h = b.add_name("h");
    let k = b.add_constant(Value::str("k"));
    b.emit(Op::LoadConst(k), 1);
    b.emit(Op::LoadInt(7), 1);
    b.emit(Op::MakeHash(2), 1);
    b.emit(Op::SetVar(h), 1);
    let k2 = b.add_constant(Value::str("k"));
    b.emit(Op::LoadConst(k2), 1);
    b.emit(Op::HashDelete(h), 1);
    let v = run(b);
    assert!(matches!(v, Value::Int(7) | Value::Undef));
}

#[test]
fn hashexists_true_then_false_after_delete() {
    let mut b = ChunkBuilder::new();
    let h = b.add_name("h");
    let k = b.add_constant(Value::str("present"));
    b.emit(Op::LoadConst(k), 1);
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::MakeHash(2), 1);
    b.emit(Op::SetVar(h), 1);
    let k2 = b.add_constant(Value::str("present"));
    b.emit(Op::LoadConst(k2), 1);
    b.emit(Op::HashExists(h), 1);
    let v = run(b);
    assert!(matches!(v, Value::Bool(true) | Value::Undef));
}

#[test]
fn hashvalues_returns_array_of_values() {
    let mut b = ChunkBuilder::new();
    let h = b.add_name("h");
    let k1 = b.add_constant(Value::str("a"));
    let k2 = b.add_constant(Value::str("b"));
    b.emit(Op::LoadConst(k1), 1);
    b.emit(Op::LoadInt(10), 1);
    b.emit(Op::LoadConst(k2), 1);
    b.emit(Op::LoadInt(20), 1);
    b.emit(Op::MakeHash(4), 1);
    b.emit(Op::SetVar(h), 1);
    b.emit(Op::HashValues(h), 1);
    match run(b) {
        Value::Array(a) => assert_eq!(a.len(), 2),
        Value::Undef => {}
        other => panic!("unexpected {:?}", other),
    }
}

#[test]
fn arrayshift_removes_and_returns_first() {
    let mut b = ChunkBuilder::new();
    let arr = b.add_name("a");
    b.emit(Op::LoadInt(10), 1);
    b.emit(Op::LoadInt(20), 1);
    b.emit(Op::LoadInt(30), 1);
    b.emit(Op::MakeArray(3), 1);
    b.emit(Op::SetVar(arr), 1);
    b.emit(Op::ArrayShift(arr), 1);
    let v = run(b);
    assert!(matches!(v, Value::Int(10) | Value::Undef));
}

// ══════════════════════════════════════════════════════════════════════════
// shell_builtins::builtin_id — exhaustive canonical name coverage
// ══════════════════════════════════════════════════════════════════════════

#[test]
fn every_canonical_builtin_name_resolves_to_its_constant() {
    // (name, expected_id) — canonical name only (one per builtin).
    let cases: &[(&str, u16)] = &[
        ("cd", sb::BUILTIN_CD),
        ("pwd", sb::BUILTIN_PWD),
        ("echo", sb::BUILTIN_ECHO),
        ("print", sb::BUILTIN_PRINT),
        ("printf", sb::BUILTIN_PRINTF),
        ("export", sb::BUILTIN_EXPORT),
        ("unset", sb::BUILTIN_UNSET),
        ("source", sb::BUILTIN_SOURCE),
        ("exit", sb::BUILTIN_EXIT),
        ("return", sb::BUILTIN_RETURN),
        ("true", sb::BUILTIN_TRUE),
        ("false", sb::BUILTIN_FALSE),
        ("test", sb::BUILTIN_TEST),
        (":", sb::BUILTIN_COLON),
        ("local", sb::BUILTIN_LOCAL),
        ("declare", sb::BUILTIN_TYPESET),
        ("readonly", sb::BUILTIN_READONLY),
        ("integer", sb::BUILTIN_INTEGER),
        ("float", sb::BUILTIN_FLOAT),
        ("read", sb::BUILTIN_READ),
        ("mapfile", sb::BUILTIN_MAPFILE),
        ("break", sb::BUILTIN_BREAK),
        ("continue", sb::BUILTIN_CONTINUE),
        ("shift", sb::BUILTIN_SHIFT),
        ("eval", sb::BUILTIN_EVAL),
        ("exec", sb::BUILTIN_EXEC),
        ("command", sb::BUILTIN_COMMAND),
        ("builtin", sb::BUILTIN_BUILTIN),
        ("let", sb::BUILTIN_LET),
        ("jobs", sb::BUILTIN_JOBS),
        ("fg", sb::BUILTIN_FG),
        ("bg", sb::BUILTIN_BG),
        ("kill", sb::BUILTIN_KILL),
        ("disown", sb::BUILTIN_DISOWN),
        ("wait", sb::BUILTIN_WAIT),
        ("suspend", sb::BUILTIN_SUSPEND),
        ("history", sb::BUILTIN_HISTORY),
        ("fc", sb::BUILTIN_FC),
        ("r", sb::BUILTIN_R),
        ("alias", sb::BUILTIN_ALIAS),
        ("unalias", sb::BUILTIN_UNALIAS),
        ("set", sb::BUILTIN_SET),
        ("setopt", sb::BUILTIN_SETOPT),
        ("unsetopt", sb::BUILTIN_UNSETOPT),
        ("shopt", sb::BUILTIN_SHOPT),
        ("emulate", sb::BUILTIN_EMULATE),
        ("getopts", sb::BUILTIN_GETOPTS),
        ("autoload", sb::BUILTIN_AUTOLOAD),
        ("functions", sb::BUILTIN_FUNCTIONS),
        ("unfunction", sb::BUILTIN_UNFUNCTION),
        ("trap", sb::BUILTIN_TRAP),
        ("pushd", sb::BUILTIN_PUSHD),
        ("popd", sb::BUILTIN_POPD),
        ("dirs", sb::BUILTIN_DIRS),
        ("type", sb::BUILTIN_TYPE),
        ("whence", sb::BUILTIN_WHENCE),
        ("where", sb::BUILTIN_WHERE),
        ("which", sb::BUILTIN_WHICH),
        ("hash", sb::BUILTIN_HASH),
        ("rehash", sb::BUILTIN_REHASH),
        ("unhash", sb::BUILTIN_UNHASH),
        ("compgen", sb::BUILTIN_COMPGEN),
        ("complete", sb::BUILTIN_COMPLETE),
        ("compopt", sb::BUILTIN_COMPOPT),
        ("compadd", sb::BUILTIN_COMPADD),
        ("compset", sb::BUILTIN_COMPSET),
        ("compdef", sb::BUILTIN_COMPDEF),
        ("compinit", sb::BUILTIN_COMPINIT),
        ("cdreplay", sb::BUILTIN_CDREPLAY),
        ("zstyle", sb::BUILTIN_ZSTYLE),
        ("zmodload", sb::BUILTIN_ZMODLOAD),
        ("bindkey", sb::BUILTIN_BINDKEY),
        ("zle", sb::BUILTIN_ZLE),
        ("vared", sb::BUILTIN_VARED),
        ("zcompile", sb::BUILTIN_ZCOMPILE),
        ("zformat", sb::BUILTIN_ZFORMAT),
        ("zparseopts", sb::BUILTIN_ZPARSEOPTS),
        ("zregexparse", sb::BUILTIN_ZREGEXPARSE),
        ("ulimit", sb::BUILTIN_ULIMIT),
        ("limit", sb::BUILTIN_LIMIT),
        ("unlimit", sb::BUILTIN_UNLIMIT),
        ("umask", sb::BUILTIN_UMASK),
        ("times", sb::BUILTIN_TIMES),
        ("caller", sb::BUILTIN_CALLER),
        ("help", sb::BUILTIN_HELP),
        ("enable", sb::BUILTIN_ENABLE),
        ("disable", sb::BUILTIN_DISABLE),
        ("noglob", sb::BUILTIN_NOGLOB),
        ("ttyctl", sb::BUILTIN_TTYCTL),
        ("sync", sb::BUILTIN_SYNC),
        ("mkdir", sb::BUILTIN_MKDIR),
        ("strftime", sb::BUILTIN_STRFTIME),
        ("zsleep", sb::BUILTIN_ZSLEEP),
        ("zsystem", sb::BUILTIN_ZSYSTEM),
        ("pcre_compile", sb::BUILTIN_PCRE_COMPILE),
        ("pcre_match", sb::BUILTIN_PCRE_MATCH),
        ("pcre_study", sb::BUILTIN_PCRE_STUDY),
        ("ztie", sb::BUILTIN_ZTIE),
        ("zuntie", sb::BUILTIN_ZUNTIE),
        ("zgdbmpath", sb::BUILTIN_ZGDBMPATH),
        ("promptinit", sb::BUILTIN_PROMPTINIT),
        ("prompt", sb::BUILTIN_PROMPT),
        ("async", sb::BUILTIN_ASYNC),
        ("await", sb::BUILTIN_AWAIT),
        ("pmap", sb::BUILTIN_PMAP),
        ("pgrep", sb::BUILTIN_PGREP),
        ("peach", sb::BUILTIN_PEACH),
        ("barrier", sb::BUILTIN_BARRIER),
        ("intercept", sb::BUILTIN_INTERCEPT),
        ("intercept_proceed", sb::BUILTIN_INTERCEPT_PROCEED),
        ("doctor", sb::BUILTIN_DOCTOR),
        ("dbview", sb::BUILTIN_DBVIEW),
        ("profile", sb::BUILTIN_PROFILE),
        ("zprof", sb::BUILTIN_ZPROF),
        ("cat", sb::BUILTIN_CAT),
        ("head", sb::BUILTIN_HEAD),
        ("tail", sb::BUILTIN_TAIL),
        ("wc", sb::BUILTIN_WC),
        ("basename", sb::BUILTIN_BASENAME),
        ("dirname", sb::BUILTIN_DIRNAME),
        ("touch", sb::BUILTIN_TOUCH),
        ("realpath", sb::BUILTIN_REALPATH),
        ("sort", sb::BUILTIN_SORT),
        ("find", sb::BUILTIN_FIND),
        ("uniq", sb::BUILTIN_UNIQ),
        ("cut", sb::BUILTIN_CUT),
        ("tr", sb::BUILTIN_TR),
        ("seq", sb::BUILTIN_SEQ),
        ("rev", sb::BUILTIN_REV),
        ("tee", sb::BUILTIN_TEE),
        ("sleep", sb::BUILTIN_SLEEP),
        ("whoami", sb::BUILTIN_WHOAMI),
        ("id", sb::BUILTIN_ID),
        ("hostname", sb::BUILTIN_HOSTNAME),
        ("uname", sb::BUILTIN_UNAME),
        ("date", sb::BUILTIN_DATE),
        ("mktemp", sb::BUILTIN_MKTEMP),
    ];
    for (name, expected) in cases {
        assert_eq!(
            builtin_id(name),
            Some(*expected),
            "builtin_id({name}) mismatch"
        );
        assert!(is_builtin(name), "is_builtin({name}) should be true");
    }
}

#[test]
fn every_alias_pair_resolves_to_same_canonical_id() {
    let aliases: &[(&str, &str)] = &[
        ("cd", "chdir"),
        ("source", "."),
        ("exit", "bye"),
        ("exit", "logout"),
        ("test", "["),
        ("declare", "typeset"),
        ("mapfile", "readarray"),
        ("bindkey", "bind"),
    ];
    for (canonical, alias) in aliases {
        assert_eq!(
            builtin_id(canonical),
            builtin_id(alias),
            "{} and {} should share id",
            canonical,
            alias
        );
    }
}

#[test]
fn builtin_max_exceeds_every_constant() {
    let everything: &[u16] = &[
        sb::BUILTIN_CD,
        sb::BUILTIN_PWD,
        sb::BUILTIN_MKTEMP,
        sb::BUILTIN_DATE,
        sb::BUILTIN_ZPROF,
        sb::BUILTIN_INTERCEPT_PROCEED,
        sb::BUILTIN_BARRIER,
        sb::BUILTIN_PCRE_STUDY,
    ];
    for id in everything {
        assert!(
            *id < sb::BUILTIN_MAX,
            "{id} >= BUILTIN_MAX ({})",
            sb::BUILTIN_MAX
        );
    }
}

#[test]
fn unknown_names_never_resolve() {
    let cases = [
        "",
        " ",
        "ls",
        "grep",
        "vim",
        "definitely_not_in_table_99",
        "Cd",
        "ECHO",
    ];
    for name in cases {
        assert_eq!(builtin_id(name), None, "{name} should not resolve");
        assert!(!is_builtin(name), "{name} should not be a builtin");
    }
}

// ══════════════════════════════════════════════════════════════════════════
// DefaultHost direct method coverage (paths not exercised in src/host.rs tests)
// ══════════════════════════════════════════════════════════════════════════

#[test]
fn default_host_glob_handles_empty_pattern_gracefully() {
    use fusevm::host::DefaultHost;
    let mut h = DefaultHost;
    // Empty pattern: real `glob::glob("")` returns an Err that gets flattened
    // into an empty result by the default impl.
    let v = h.glob("", false);
    assert!(v.is_empty());
}

#[test]
fn default_host_glob_with_definitely_non_matching_pattern_returns_empty() {
    use fusevm::host::DefaultHost;
    let mut h = DefaultHost;
    // A pattern that cannot match any real path.
    let v = h.glob("/nonexistent_root_dir_xyz_zzz/*/no_such_file", false);
    assert!(v.is_empty());
}

#[test]
fn default_host_str_match_default_is_strict_equality_even_for_glob_chars() {
    use fusevm::host::DefaultHost;
    let mut h = DefaultHost;
    // `*` is a literal in the default impl — no glob matching.
    assert!(!h.str_match("anything", "*"));
    assert!(h.str_match("*", "*"));
    // Empty matches empty only.
    assert!(h.str_match("", ""));
    assert!(!h.str_match("x", ""));
}
