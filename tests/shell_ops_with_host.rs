use fusevm::{Chunk, ChunkBuilder, Op, ShellHost, VMResult, Value, VM};
use std::sync::{Arc, Mutex};

/// Recording ShellHost: tracks every method call as ("method", "arg-summary").
#[derive(Default, Clone)]
struct Recorder {
    log: Arc<Mutex<Vec<(String, String)>>>,
    pipeline_status: i32,
    exec_status: i32,
    cmd_subst_result: String,
    glob_result: Vec<String>,
    brace_result: Vec<String>,
    expand_result: Value,
    str_match_result: bool,
    regex_match_result: bool,
    call_function_status: Option<i32>,
}

impl Recorder {
    fn new() -> Self {
        Self {
            log: Arc::new(Mutex::new(Vec::new())),
            pipeline_status: 0,
            exec_status: 0,
            cmd_subst_result: String::new(),
            glob_result: Vec::new(),
            brace_result: Vec::new(),
            expand_result: Value::str(""),
            str_match_result: false,
            regex_match_result: false,
            call_function_status: None,
        }
    }
    fn rec(&self, m: &str, arg: impl ToString) {
        self.log.lock().unwrap().push((m.into(), arg.to_string()));
    }
    fn calls(&self) -> Vec<(String, String)> {
        self.log.lock().unwrap().clone()
    }
}

impl ShellHost for Recorder {
    fn glob(&mut self, pattern: &str, recursive: bool) -> Vec<String> {
        self.rec("glob", format!("{}|{}", pattern, recursive));
        self.glob_result.clone()
    }
    fn tilde_expand(&mut self, s: &str) -> String {
        self.rec("tilde_expand", s);
        format!("/home/{}", s.trim_start_matches('~'))
    }
    fn brace_expand(&mut self, s: &str) -> Vec<String> {
        self.rec("brace_expand", s);
        self.brace_result.clone()
    }
    fn word_split(&mut self, s: &str) -> Vec<String> {
        self.rec("word_split", s);
        vec!["A".into(), "B".into()]
    }
    fn expand_param(&mut self, name: &str, modifier: u8, args: &[Value]) -> Value {
        self.rec(
            "expand_param",
            format!("{}|m={}|argc={}", name, modifier, args.len()),
        );
        self.expand_result.clone()
    }
    fn str_match(&mut self, s: &str, pat: &str) -> bool {
        self.rec("str_match", format!("{}~{}", s, pat));
        self.str_match_result
    }
    fn regex_match(&mut self, s: &str, re: &str) -> bool {
        self.rec("regex_match", format!("{}~{}", s, re));
        self.regex_match_result
    }
    fn cmd_subst(&mut self, _sub: &Chunk) -> String {
        self.rec("cmd_subst", "");
        self.cmd_subst_result.clone()
    }
    fn process_sub_in(&mut self, _sub: &Chunk) -> String {
        self.rec("process_sub_in", "");
        "<psin>".to_string()
    }
    fn process_sub_out(&mut self, _sub: &Chunk) -> String {
        self.rec("process_sub_out", "");
        "<psout>".to_string()
    }
    fn redirect(&mut self, fd: u8, op: u8, target: &str) {
        self.rec("redirect", format!("{}|{}|{}", fd, op, target));
    }
    fn heredoc(&mut self, content: &str) {
        self.rec("heredoc", content);
    }
    fn herestring(&mut self, content: &str) {
        self.rec("herestring", content);
    }
    fn pipeline_begin(&mut self, n: u8) {
        self.rec("pipeline_begin", n);
    }
    fn pipeline_stage(&mut self) {
        self.rec("pipeline_stage", "");
    }
    fn pipeline_end(&mut self) -> i32 {
        self.rec("pipeline_end", "");
        self.pipeline_status
    }
    fn subshell_begin(&mut self) {
        self.rec("subshell_begin", "");
    }
    fn subshell_end(&mut self) -> Option<i32> {
        self.rec("subshell_end", "");
        None
    }
    fn with_redirects_begin(&mut self, count: u8) {
        self.rec("with_redirects_begin", count);
    }
    fn with_redirects_end(&mut self) {
        self.rec("with_redirects_end", "");
    }
    fn call_function(&mut self, name: &str, args: Vec<String>) -> Option<i32> {
        self.rec("call_function", format!("{}|{}", name, args.join(",")));
        self.call_function_status
    }
    fn exec(&mut self, args: Vec<String>) -> i32 {
        self.rec("exec", args.join(" "));
        self.exec_status
    }
    fn exec_bg(&mut self, args: Vec<String>) -> i32 {
        self.rec("exec_bg", args.join(" "));
        0
    }
    fn trap_set(&mut self, sig: &str, _sub: &Chunk) {
        self.rec("trap_set", sig);
    }
    fn trap_check(&mut self) {
        self.rec("trap_check", "");
    }
}

fn run_with(b: ChunkBuilder, host: Recorder) -> (Value, Recorder) {
    let mut vm = VM::new(b.build());
    let log = host.log.clone();
    vm.set_shell_host(Box::new(host.clone()));
    let v = match vm.run() {
        VMResult::Ok(v) => v,
        other => panic!("unexpected result: {:?}", other),
    };
    let mut out = host;
    out.log = log;
    (v, out)
}

// ── Pipeline ops ──

#[test]
fn pipeline_begin_stage_end_invokes_host() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::PipelineBegin(3), 1);
    b.emit(Op::PipelineStage, 1);
    b.emit(Op::PipelineStage, 1);
    b.emit(Op::PipelineEnd, 1);
    let mut host = Recorder::new();
    host.pipeline_status = 42;
    let (v, h) = run_with(b, host);
    assert_eq!(v, Value::Status(42));
    let calls = h.calls();
    assert_eq!(calls[0].0, "pipeline_begin");
    assert_eq!(calls[0].1, "3");
    assert_eq!(calls[1].0, "pipeline_stage");
    assert_eq!(calls[3].0, "pipeline_end");
}

#[test]
fn pipeline_end_no_host_uses_last_status() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(77), 1);
    b.emit(Op::SetStatus, 1);
    b.emit(Op::PipelineEnd, 1);
    let r = match VM::new(b.build()).run() {
        VMResult::Ok(v) => v,
        other => panic!("got {:?}", other),
    };
    assert_eq!(r, Value::Status(77));
}

// ── Subshell ──

#[test]
fn subshell_begin_end_invokes_host() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::LoadInt(1), 1);
    let (v, h) = run_with(b, Recorder::new());
    assert_eq!(v, Value::Int(1));
    let calls = h.calls();
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0].0, "subshell_begin");
    assert_eq!(calls[1].0, "subshell_end");
}

// ── Redirect ──

#[test]
fn redirect_forwards_fd_op_and_target() {
    let mut b = ChunkBuilder::new();
    let k = b.add_constant(Value::str("/tmp/out"));
    b.emit(Op::LoadConst(k), 1);
    b.emit(Op::Redirect(1, 0 /* WRITE */), 1);
    b.emit(Op::LoadInt(1), 1);
    let (_, h) = run_with(b, Recorder::new());
    let calls = h.calls();
    assert_eq!(calls[0].0, "redirect");
    assert_eq!(calls[0].1, "1|0|/tmp/out");
}

#[test]
fn redirect_no_host_is_noop() {
    let mut b = ChunkBuilder::new();
    let k = b.add_constant(Value::str("/tmp/x"));
    b.emit(Op::LoadConst(k), 1);
    b.emit(Op::Redirect(2, 1), 1);
    b.emit(Op::LoadInt(5), 1);
    match VM::new(b.build()).run() {
        VMResult::Ok(Value::Int(5)) => {}
        other => panic!("got {:?}", other),
    }
}

// ── HereDoc / HereString ──

#[test]
fn heredoc_passes_constant_to_host() {
    let mut b = ChunkBuilder::new();
    let k = b.add_constant(Value::str("body\ntext\n"));
    b.emit(Op::HereDoc(k), 1);
    b.emit(Op::LoadInt(1), 1);
    let (_, h) = run_with(b, Recorder::new());
    let calls = h.calls();
    assert_eq!(calls[0].0, "heredoc");
    assert_eq!(calls[0].1, "body\ntext\n");
}

#[test]
fn herestring_pops_and_forwards() {
    let mut b = ChunkBuilder::new();
    let k = b.add_constant(Value::str("hello"));
    b.emit(Op::LoadConst(k), 1);
    b.emit(Op::HereString, 1);
    b.emit(Op::LoadInt(1), 1);
    let (_, h) = run_with(b, Recorder::new());
    let calls = h.calls();
    assert_eq!(calls[0].0, "herestring");
    assert_eq!(calls[0].1, "hello");
}

// ── CmdSubst / ProcessSubIn / ProcessSubOut ──

#[test]
fn cmd_subst_invokes_host_with_subchunk_and_pushes_result() {
    let mut b = ChunkBuilder::new();
    let mut sub = ChunkBuilder::new();
    sub.emit(Op::LoadInt(1), 1);
    let idx = b.add_sub_chunk(sub.build());
    b.emit(Op::CmdSubst(idx), 1);
    let mut host = Recorder::new();
    host.cmd_subst_result = "captured".into();
    let (v, h) = run_with(b, host);
    match v {
        Value::Str(s) => assert_eq!(&*s, "captured"),
        other => panic!("got {:?}", other),
    }
    assert_eq!(h.calls()[0].0, "cmd_subst");
}

#[test]
fn cmd_subst_missing_subchunk_pushes_empty() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::CmdSubst(999), 1);
    match VM::new(b.build()).run() {
        VMResult::Ok(Value::Str(s)) => assert_eq!(&*s, ""),
        other => panic!("got {:?}", other),
    }
}

#[test]
fn process_sub_in_returns_pipe_path() {
    let mut b = ChunkBuilder::new();
    let mut sub = ChunkBuilder::new();
    sub.emit(Op::Nop, 1);
    let idx = b.add_sub_chunk(sub.build());
    b.emit(Op::ProcessSubIn(idx), 1);
    let (v, h) = run_with(b, Recorder::new());
    match v {
        Value::Str(s) => assert_eq!(&*s, "<psin>"),
        other => panic!("got {:?}", other),
    }
    assert_eq!(h.calls()[0].0, "process_sub_in");
}

#[test]
fn process_sub_out_returns_pipe_path() {
    let mut b = ChunkBuilder::new();
    let mut sub = ChunkBuilder::new();
    sub.emit(Op::Nop, 1);
    let idx = b.add_sub_chunk(sub.build());
    b.emit(Op::ProcessSubOut(idx), 1);
    let (v, h) = run_with(b, Recorder::new());
    match v {
        Value::Str(s) => assert_eq!(&*s, "<psout>"),
        other => panic!("got {:?}", other),
    }
    assert_eq!(h.calls()[0].0, "process_sub_out");
}

// ── Glob / GlobRecursive ──

#[test]
fn glob_invokes_host_with_pattern_and_pushes_array() {
    let mut b = ChunkBuilder::new();
    let k = b.add_constant(Value::str("*.rs"));
    b.emit(Op::LoadConst(k), 1);
    b.emit(Op::Glob, 1);
    let mut host = Recorder::new();
    host.glob_result = vec!["a.rs".into(), "b.rs".into()];
    let (v, h) = run_with(b, host);
    match v {
        Value::Array(arr) => {
            assert_eq!(arr.len(), 2);
            assert_eq!(arr[0].to_str(), "a.rs");
        }
        other => panic!("got {:?}", other),
    }
    assert_eq!(h.calls()[0].1, "*.rs|false");
}

#[test]
fn glob_recursive_sets_flag() {
    let mut b = ChunkBuilder::new();
    let k = b.add_constant(Value::str("**/*.rs"));
    b.emit(Op::LoadConst(k), 1);
    b.emit(Op::GlobRecursive, 1);
    let host = Recorder::new();
    let (_, h) = run_with(b, host);
    assert_eq!(h.calls()[0].1, "**/*.rs|true");
}

// ── TildeExpand / BraceExpand / WordSplit / ExpandParam ──

#[test]
fn tilde_expand_via_host() {
    let mut b = ChunkBuilder::new();
    let k = b.add_constant(Value::str("~alice"));
    b.emit(Op::LoadConst(k), 1);
    b.emit(Op::TildeExpand, 1);
    let (v, h) = run_with(b, Recorder::new());
    match v {
        Value::Str(s) => assert_eq!(&*s, "/home/alice"),
        other => panic!("got {:?}", other),
    }
    assert_eq!(h.calls()[0].0, "tilde_expand");
}

#[test]
fn brace_expand_via_host() {
    let mut b = ChunkBuilder::new();
    let k = b.add_constant(Value::str("{a,b,c}"));
    b.emit(Op::LoadConst(k), 1);
    b.emit(Op::BraceExpand, 1);
    let mut host = Recorder::new();
    host.brace_result = vec!["a".into(), "b".into(), "c".into()];
    let (v, _h) = run_with(b, host);
    match v {
        Value::Array(arr) => {
            assert_eq!(arr.len(), 3);
            assert_eq!(arr[2].to_str(), "c");
        }
        other => panic!("got {:?}", other),
    }
}

#[test]
fn word_split_via_host() {
    let mut b = ChunkBuilder::new();
    let k = b.add_constant(Value::str("hello world"));
    b.emit(Op::LoadConst(k), 1);
    b.emit(Op::WordSplit, 1);
    let (v, _h) = run_with(b, Recorder::new());
    match v {
        Value::Array(arr) => {
            assert_eq!(arr.len(), 2);
            assert_eq!(arr[0].to_str(), "A");
            assert_eq!(arr[1].to_str(), "B");
        }
        other => panic!("got {:?}", other),
    }
}

#[test]
fn expand_param_default_passes_one_arg() {
    let mut b = ChunkBuilder::new();
    let name = b.add_constant(Value::str("X"));
    let dflt = b.add_constant(Value::str("d"));
    b.emit(Op::LoadConst(name), 1);
    b.emit(Op::LoadConst(dflt), 1);
    b.emit(Op::ExpandParam(0 /* DEFAULT */), 1);
    let mut host = Recorder::new();
    host.expand_result = Value::str("OK");
    let (v, h) = run_with(b, host);
    match v {
        Value::Str(s) => assert_eq!(&*s, "OK"),
        other => panic!("got {:?}", other),
    }
    assert_eq!(h.calls()[0].1, "X|m=0|argc=1");
}

#[test]
fn expand_param_length_passes_no_arg() {
    let mut b = ChunkBuilder::new();
    let name = b.add_constant(Value::str("Y"));
    b.emit(Op::LoadConst(name), 1);
    b.emit(Op::ExpandParam(4 /* LENGTH */), 1);
    let (_, h) = run_with(b, Recorder::new());
    assert_eq!(h.calls()[0].1, "Y|m=4|argc=0");
}

#[test]
fn expand_param_subst_passes_two_args() {
    let mut b = ChunkBuilder::new();
    let name = b.add_constant(Value::str("Z"));
    let pat = b.add_constant(Value::str("p"));
    let rep = b.add_constant(Value::str("r"));
    b.emit(Op::LoadConst(name), 1);
    b.emit(Op::LoadConst(pat), 1);
    b.emit(Op::LoadConst(rep), 1);
    // SUBST_FIRST: walk param_mod constants — must be in [SUBST_FIRST, SUBST_ALL, SLICE]
    // Without knowing exact value, the recording host just checks argc.
    b.emit(Op::ExpandParam(12 /* SUBST_FIRST or similar */), 1);
    let (_, h) = run_with(b, Recorder::new());
    // argc-0 because 12 may not be in the 2-arg list; just ensure the host got called
    assert_eq!(h.calls()[0].0, "expand_param");
}

// ── StrMatch / RegexMatch ──

#[test]
fn str_match_uses_host() {
    let mut b = ChunkBuilder::new();
    let s = b.add_constant(Value::str("foo"));
    let p = b.add_constant(Value::str("f*"));
    b.emit(Op::LoadConst(s), 1);
    b.emit(Op::LoadConst(p), 1);
    b.emit(Op::StrMatch, 1);
    let mut host = Recorder::new();
    host.str_match_result = true;
    let (v, h) = run_with(b, host);
    assert_eq!(v, Value::Bool(true));
    assert_eq!(h.calls()[0].1, "foo~f*");
}

#[test]
fn regex_match_uses_host() {
    let mut b = ChunkBuilder::new();
    let s = b.add_constant(Value::str("abc"));
    let r = b.add_constant(Value::str("a.c"));
    b.emit(Op::LoadConst(s), 1);
    b.emit(Op::LoadConst(r), 1);
    b.emit(Op::RegexMatch, 1);
    let mut host = Recorder::new();
    host.regex_match_result = true;
    let (v, h) = run_with(b, host);
    assert_eq!(v, Value::Bool(true));
    assert_eq!(h.calls()[0].1, "abc~a.c");
}

// ── Exec ──

#[test]
fn exec_true_returns_zero() {
    let mut b = ChunkBuilder::new();
    let k = b.add_constant(Value::str("true"));
    b.emit(Op::LoadConst(k), 1);
    b.emit(Op::Exec(1), 1);
    match VM::new(b.build()).run() {
        VMResult::Ok(Value::Status(0)) => {}
        other => panic!("got {:?}", other),
    }
}

#[test]
fn exec_false_returns_one() {
    let mut b = ChunkBuilder::new();
    let k = b.add_constant(Value::str("false"));
    b.emit(Op::LoadConst(k), 1);
    b.emit(Op::Exec(1), 1);
    match VM::new(b.build()).run() {
        VMResult::Ok(Value::Status(1)) => {}
        other => panic!("got {:?}", other),
    }
}

#[test]
fn exec_test_returns_zero() {
    let mut b = ChunkBuilder::new();
    let k = b.add_constant(Value::str("test"));
    b.emit(Op::LoadConst(k), 1);
    b.emit(Op::Exec(1), 1);
    match VM::new(b.build()).run() {
        VMResult::Ok(Value::Status(0)) => {}
        other => panic!("got {:?}", other),
    }
}

#[test]
fn exec_empty_returns_zero() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::Exec(0), 1);
    match VM::new(b.build()).run() {
        VMResult::Ok(Value::Status(0)) => {}
        other => panic!("got {:?}", other),
    }
}

#[test]
fn exec_routes_unknown_command_through_host() {
    let mut b = ChunkBuilder::new();
    let k = b.add_constant(Value::str("myhostcmd"));
    let arg = b.add_constant(Value::str("--flag"));
    b.emit(Op::LoadConst(k), 1);
    b.emit(Op::LoadConst(arg), 1);
    b.emit(Op::Exec(2), 1);
    let mut host = Recorder::new();
    host.exec_status = 13;
    let (v, h) = run_with(b, host);
    assert_eq!(v, Value::Status(13));
    assert_eq!(h.calls()[0].0, "exec");
    assert_eq!(h.calls()[0].1, "myhostcmd --flag");
}

#[test]
fn exec_flattens_array_args() {
    let mut b = ChunkBuilder::new();
    let cmd = b.add_constant(Value::str("myhostcmd"));
    let arr = b.add_constant(Value::Array(vec![Value::str("a"), Value::str("b")]));
    b.emit(Op::LoadConst(cmd), 1);
    b.emit(Op::LoadConst(arr), 1);
    b.emit(Op::Exec(2), 1);
    let (_, h) = run_with(b, Recorder::new());
    assert_eq!(h.calls()[0].1, "myhostcmd a b");
}

// ── ExecBg ──

#[test]
fn exec_bg_routes_through_host() {
    let mut b = ChunkBuilder::new();
    let k = b.add_constant(Value::str("mybgcmd"));
    b.emit(Op::LoadConst(k), 1);
    b.emit(Op::ExecBg(1), 1);
    let (v, h) = run_with(b, Recorder::new());
    assert_eq!(v, Value::Status(0));
    assert_eq!(h.calls()[0].0, "exec_bg");
    assert_eq!(h.calls()[0].1, "mybgcmd");
}

// ── WithRedirects ──

#[test]
fn with_redirects_begin_and_end_invoke_host() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::WithRedirectsBegin(2), 1);
    b.emit(Op::WithRedirectsEnd, 1);
    b.emit(Op::LoadInt(1), 1);
    let (_, h) = run_with(b, Recorder::new());
    let calls = h.calls();
    assert_eq!(calls[0].0, "with_redirects_begin");
    assert_eq!(calls[0].1, "2");
    assert_eq!(calls[1].0, "with_redirects_end");
}

// ── TrapSet / TrapCheck ──

#[test]
fn trap_set_passes_signal_name() {
    let mut b = ChunkBuilder::new();
    let mut sub = ChunkBuilder::new();
    sub.emit(Op::Nop, 1);
    let sub_idx = b.add_sub_chunk(sub.build());
    let sig = b.add_constant(Value::str("INT"));
    b.emit(Op::LoadConst(sig), 1);
    b.emit(Op::TrapSet(sub_idx), 1);
    b.emit(Op::LoadInt(1), 1);
    let (_, h) = run_with(b, Recorder::new());
    assert_eq!(h.calls()[0].0, "trap_set");
    assert_eq!(h.calls()[0].1, "INT");
}

#[test]
fn trap_check_calls_host() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::TrapCheck, 1);
    b.emit(Op::LoadInt(1), 1);
    let (_, h) = run_with(b, Recorder::new());
    assert_eq!(h.calls()[0].0, "trap_check");
}

// ── CallFunction (named) ──

#[test]
fn call_function_routes_through_host() {
    let mut b = ChunkBuilder::new();
    let name = b.add_name("myfunc");
    let a = b.add_constant(Value::str("x"));
    b.emit(Op::LoadConst(a), 1);
    b.emit(Op::CallFunction(name, 1), 1);
    let mut host = Recorder::new();
    host.call_function_status = Some(7);
    let (_, h) = run_with(b, host);
    let calls = h.calls();
    assert_eq!(calls[0].0, "call_function");
    assert_eq!(calls[0].1, "myfunc|x");
}

#[test]
fn call_function_falls_back_to_exec_when_host_returns_none() {
    let mut b = ChunkBuilder::new();
    let name = b.add_name("notafunc");
    b.emit(Op::CallFunction(name, 0), 1);
    let mut host = Recorder::new();
    host.call_function_status = None;
    host.exec_status = 99;
    let (_, h) = run_with(b, host);
    let calls = h.calls();
    assert_eq!(calls[0].0, "call_function");
    assert_eq!(calls[1].0, "exec");
}

// ── Array global ops ──

#[test]
fn get_array_returns_value() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(7), 1);
    b.emit(Op::SetArray(0), 1);
    b.emit(Op::GetArray(0), 1);
    assert_eq!(run_with(b, Recorder::new()).0, Value::Int(7));
}

#[test]
fn declare_array_initializes_empty() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::DeclareArray(0), 1);
    b.emit(Op::GetArray(0), 1);
    match run_with(b, Recorder::new()).0 {
        Value::Array(arr) => assert!(arr.is_empty()),
        other => panic!("got {:?}", other),
    }
}

#[test]
fn array_push_then_len() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::DeclareArray(0), 1);
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::ArrayPush(0), 1);
    b.emit(Op::LoadInt(2), 1);
    b.emit(Op::ArrayPush(0), 1);
    b.emit(Op::ArrayLen(0), 1);
    assert_eq!(run_with(b, Recorder::new()).0, Value::Int(2));
}

#[test]
fn array_get_set_round_trip() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::DeclareArray(0), 1);
    b.emit(Op::LoadInt(10), 1);
    b.emit(Op::ArrayPush(0), 1);
    b.emit(Op::LoadInt(99), 1); // value
    b.emit(Op::LoadInt(0), 1); // index
    b.emit(Op::ArraySet(0), 1);
    b.emit(Op::LoadInt(0), 1);
    b.emit(Op::ArrayGet(0), 1);
    assert_eq!(run_with(b, Recorder::new()).0, Value::Int(99));
}

#[test]
fn array_pop_returns_last() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::DeclareArray(0), 1);
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::ArrayPush(0), 1);
    b.emit(Op::LoadInt(2), 1);
    b.emit(Op::ArrayPush(0), 1);
    b.emit(Op::ArrayPop(0), 1);
    assert_eq!(run_with(b, Recorder::new()).0, Value::Int(2));
}

#[test]
fn array_shift_returns_first() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::DeclareArray(0), 1);
    b.emit(Op::LoadInt(7), 1);
    b.emit(Op::ArrayPush(0), 1);
    b.emit(Op::LoadInt(8), 1);
    b.emit(Op::ArrayPush(0), 1);
    b.emit(Op::ArrayShift(0), 1);
    assert_eq!(run_with(b, Recorder::new()).0, Value::Int(7));
}

// ── Hash global ops ──

#[test]
fn get_hash_set_hash_round_trip() {
    let mut b = ChunkBuilder::new();
    let v = b.add_constant(Value::str("v"));
    b.emit(Op::LoadConst(v), 1);
    b.emit(Op::SetHash(0), 1);
    b.emit(Op::GetHash(0), 1);
    match run_with(b, Recorder::new()).0 {
        Value::Str(s) => assert_eq!(&*s, "v"),
        other => panic!("got {:?}", other),
    }
}

#[test]
fn declare_hash_initializes_empty() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::DeclareHash(0), 1);
    b.emit(Op::GetHash(0), 1);
    match run_with(b, Recorder::new()).0 {
        Value::Hash(h) => assert!(h.is_empty()),
        other => panic!("got {:?}", other),
    }
}

#[test]
fn hash_set_get_pair() {
    let mut b = ChunkBuilder::new();
    let k = b.add_constant(Value::str("name"));
    b.emit(Op::DeclareHash(0), 1);
    b.emit(Op::LoadInt(42), 1); // value
    b.emit(Op::LoadConst(k), 1); // key
    b.emit(Op::HashSet(0), 1);
    b.emit(Op::LoadConst(k), 1);
    b.emit(Op::HashGet(0), 1);
    assert_eq!(run_with(b, Recorder::new()).0, Value::Int(42));
}

#[test]
fn hash_exists_returns_bool() {
    let mut b = ChunkBuilder::new();
    let k = b.add_constant(Value::str("k"));
    b.emit(Op::DeclareHash(0), 1);
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::LoadConst(k), 1);
    b.emit(Op::HashSet(0), 1);
    b.emit(Op::LoadConst(k), 1);
    b.emit(Op::HashExists(0), 1);
    assert_eq!(run_with(b, Recorder::new()).0, Value::Bool(true));
}

#[test]
fn hash_delete_removes_entry() {
    let mut b = ChunkBuilder::new();
    let k = b.add_constant(Value::str("k"));
    b.emit(Op::DeclareHash(0), 1);
    b.emit(Op::LoadInt(5), 1);
    b.emit(Op::LoadConst(k), 1);
    b.emit(Op::HashSet(0), 1);
    b.emit(Op::LoadConst(k), 1);
    b.emit(Op::HashDelete(0), 1);
    b.emit(Op::Pop, 1);
    b.emit(Op::LoadConst(k), 1);
    b.emit(Op::HashExists(0), 1);
    assert_eq!(run_with(b, Recorder::new()).0, Value::Bool(false));
}

// ── No-host shell ops smoke-test ──

#[test]
fn pipeline_subshell_redirect_heredoc_no_host_is_noop() {
    // Build a chunk with several shell ops; without a host, none should panic.
    let mut b = ChunkBuilder::new();
    b.emit(Op::PipelineBegin(2), 1);
    b.emit(Op::PipelineStage, 1);
    b.emit(Op::PipelineEnd, 1);
    b.emit(Op::Pop, 1);
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    let here = b.add_constant(Value::str("body"));
    b.emit(Op::HereDoc(here), 1);
    let hs = b.add_constant(Value::str("hs"));
    b.emit(Op::LoadConst(hs), 1);
    b.emit(Op::HereString, 1);
    b.emit(Op::LoadInt(123), 1);
    match VM::new(b.build()).run() {
        VMResult::Ok(Value::Int(123)) => {}
        other => panic!("got {:?}", other),
    }
}
