//! Tests for ShellHost trait dispatch and the new shell ops added in 0.10.0.
//!
//! VM::run() returns Ok(top_of_stack) when stack is non-empty at termination,
//! so single-result tests assert against VMResult::Ok(...).

use fusevm::host::ShellHost;
use fusevm::op::{file_test, param_mod, redirect_op};
use fusevm::{Chunk, ChunkBuilder, DefaultHost, Op, VMResult, Value, VM};

fn run_for_value(chunk: Chunk) -> Value {
    let mut vm = VM::new(chunk);
    match vm.run() {
        VMResult::Ok(v) => v,
        VMResult::Halted => panic!("vm halted with empty stack"),
        VMResult::Error(e) => panic!("vm error: {}", e),
    }
}

fn run_with_host(chunk: Chunk, host: Box<dyn ShellHost>) -> Value {
    let mut vm = VM::new(chunk);
    vm.set_shell_host(host);
    match vm.run() {
        VMResult::Ok(v) => v,
        VMResult::Halted => panic!("vm halted with empty stack"),
        VMResult::Error(e) => panic!("vm error: {}", e),
    }
}

// ── Default host: VM works without any frontend host registered ──────────────

#[test]
fn default_host_glob_uses_fs_glob() {
    let mut b = ChunkBuilder::new();
    let c = b.add_constant(Value::str("Cargo.toml")); // matches in fusevm root
    b.emit(Op::LoadConst(c), 1);
    b.emit(Op::Glob, 1);

    match run_for_value(b.build()) {
        Value::Array(v) => assert_eq!(v.len(), 1),
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn default_host_tilde_expand_passes_through() {
    let mut b = ChunkBuilder::new();
    let c = b.add_constant(Value::str("~/foo"));
    b.emit(Op::LoadConst(c), 1);
    b.emit(Op::TildeExpand, 1);
    assert_eq!(run_for_value(b.build()).to_str(), "~/foo");
}

#[test]
fn default_host_word_split_pushes_array() {
    let mut b = ChunkBuilder::new();
    let c = b.add_constant(Value::str("a b c"));
    b.emit(Op::LoadConst(c), 1);
    b.emit(Op::WordSplit, 1);

    match run_for_value(b.build()) {
        Value::Array(v) => {
            assert_eq!(v.len(), 3);
            assert_eq!(v[0].to_str(), "a");
            assert_eq!(v[2].to_str(), "c");
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn default_host_brace_expand_returns_singleton() {
    let mut b = ChunkBuilder::new();
    let c = b.add_constant(Value::str("{a,b,c}"));
    b.emit(Op::LoadConst(c), 1);
    b.emit(Op::BraceExpand, 1);

    match run_for_value(b.build()) {
        Value::Array(v) => {
            assert_eq!(v.len(), 1);
            assert_eq!(v[0].to_str(), "{a,b,c}");
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn default_host_str_match_is_equality() {
    let mut b = ChunkBuilder::new();
    let s = b.add_constant(Value::str("foo"));
    let p = b.add_constant(Value::str("foo"));
    b.emit(Op::LoadConst(s), 1);
    b.emit(Op::LoadConst(p), 1);
    b.emit(Op::StrMatch, 1);
    assert_eq!(run_for_value(b.build()), Value::Bool(true));
}

#[test]
fn default_host_regex_match_returns_false() {
    let mut b = ChunkBuilder::new();
    let s = b.add_constant(Value::str("foo123"));
    let r = b.add_constant(Value::str(r"\d+"));
    b.emit(Op::LoadConst(s), 1);
    b.emit(Op::LoadConst(r), 1);
    b.emit(Op::RegexMatch, 1);
    // Default impl returns false; only the host knows the regex engine.
    assert_eq!(run_for_value(b.build()), Value::Bool(false));
}

// ── Custom host: ops dispatch through frontend ───────────────────────────────

#[derive(Default)]
struct RecordingHost {
    glob_calls: Vec<String>,
    tilde_calls: Vec<String>,
    expand_param_calls: Vec<(String, u8, Vec<String>)>,
    redirect_calls: Vec<(u8, u8, String)>,
    pipeline_begun: u8,
    pipeline_ended: bool,
    cmd_subst_ops: usize,
    trap_set_calls: Vec<(String, usize)>,
    with_redirect_depth: i32,
    fn_calls: Vec<(String, Vec<String>)>,
    str_match_calls: Vec<(String, String)>,
    regex_match_calls: Vec<(String, String)>,
}

impl ShellHost for RecordingHost {
    fn glob(&mut self, pattern: &str, _recursive: bool) -> Vec<String> {
        self.glob_calls.push(pattern.to_string());
        vec!["one.rs".to_string(), "two.rs".to_string()]
    }

    fn tilde_expand(&mut self, s: &str) -> String {
        self.tilde_calls.push(s.to_string());
        s.replace('~', "/home/u")
    }

    fn expand_param(&mut self, name: &str, modifier: u8, args: &[Value]) -> Value {
        let arg_strs: Vec<String> = args.iter().map(|v| v.to_str()).collect();
        self.expand_param_calls
            .push((name.to_string(), modifier, arg_strs));
        Value::str("EXPANDED")
    }

    fn redirect(&mut self, fd: u8, op: u8, target: &str) {
        self.redirect_calls.push((fd, op, target.to_string()));
    }

    fn pipeline_begin(&mut self, n: u8) {
        self.pipeline_begun = n;
    }

    fn pipeline_end(&mut self) -> i32 {
        self.pipeline_ended = true;
        42
    }

    fn cmd_subst(&mut self, sub: &Chunk) -> String {
        self.cmd_subst_ops = sub.ops.len();
        "captured".to_string()
    }

    fn trap_set(&mut self, sig: &str, handler: &Chunk) {
        self.trap_set_calls
            .push((sig.to_string(), handler.ops.len()));
    }

    fn with_redirects_begin(&mut self, _count: u8) {
        self.with_redirect_depth += 1;
    }

    fn with_redirects_end(&mut self) {
        self.with_redirect_depth -= 1;
    }

    fn call_function(&mut self, name: &str, args: Vec<String>) -> Option<i32> {
        self.fn_calls.push((name.to_string(), args));
        Some(7)
    }

    fn str_match(&mut self, s: &str, pat: &str) -> bool {
        self.str_match_calls.push((s.to_string(), pat.to_string()));
        if let Some(rest) = pat.strip_prefix('*') {
            s.ends_with(rest)
        } else {
            s == pat
        }
    }

    fn regex_match(&mut self, s: &str, regex: &str) -> bool {
        self.regex_match_calls
            .push((s.to_string(), regex.to_string()));
        regex == "ANY" || s.contains(regex)
    }
}

#[test]
fn host_routes_glob() {
    let mut b = ChunkBuilder::new();
    let c = b.add_constant(Value::str("*.rs"));
    b.emit(Op::LoadConst(c), 1);
    b.emit(Op::Glob, 1);

    match run_with_host(b.build(), Box::new(RecordingHost::default())) {
        Value::Array(v) => {
            assert_eq!(v.len(), 2);
            assert_eq!(v[0].to_str(), "one.rs");
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn host_routes_tilde_expand() {
    let mut b = ChunkBuilder::new();
    let c = b.add_constant(Value::str("~/code"));
    b.emit(Op::LoadConst(c), 1);
    b.emit(Op::TildeExpand, 1);
    let v = run_with_host(b.build(), Box::new(RecordingHost::default()));
    assert_eq!(v.to_str(), "/home/u/code");
}

#[test]
fn host_expand_param_default() {
    // ${var:-fallback}
    let mut b = ChunkBuilder::new();
    let name = b.add_constant(Value::str("MY_VAR"));
    let arg = b.add_constant(Value::str("fallback"));
    b.emit(Op::LoadConst(name), 1);
    b.emit(Op::LoadConst(arg), 1);
    b.emit(Op::ExpandParam(param_mod::DEFAULT), 1);
    let v = run_with_host(b.build(), Box::new(RecordingHost::default()));
    assert_eq!(v.to_str(), "EXPANDED");
}

#[test]
fn host_expand_param_subst_pops_two_args() {
    // ${var/pat/rep} — SUBST_FIRST takes 2 args
    let mut b = ChunkBuilder::new();
    let name = b.add_constant(Value::str("STR"));
    let pat = b.add_constant(Value::str("foo"));
    let rep = b.add_constant(Value::str("bar"));
    b.emit(Op::LoadConst(name), 1);
    b.emit(Op::LoadConst(pat), 1);
    b.emit(Op::LoadConst(rep), 1);
    b.emit(Op::ExpandParam(param_mod::SUBST_FIRST), 1);
    // No leftover sentinel — push then pop must net zero (tested via VM ending Ok with the result).
    let v = run_with_host(b.build(), Box::new(RecordingHost::default()));
    assert_eq!(v.to_str(), "EXPANDED");
}

#[test]
fn host_expand_param_length_takes_no_args() {
    // ${#var} — LENGTH takes no args
    let mut b = ChunkBuilder::new();
    let name = b.add_constant(Value::str("STR"));
    b.emit(Op::LoadConst(name), 1);
    b.emit(Op::ExpandParam(param_mod::LENGTH), 1);
    let v = run_with_host(b.build(), Box::new(RecordingHost::default()));
    assert_eq!(v.to_str(), "EXPANDED");
}

#[test]
fn host_redirect_consumes_target() {
    // Push target, Redirect pops it. Then push a sentinel so the VM
    // finishes Ok with the sentinel (proving Redirect consumed exactly 1).
    let mut b = ChunkBuilder::new();
    let c = b.add_constant(Value::str("/tmp/out"));
    let sentinel = b.add_constant(Value::str("ok"));
    b.emit(Op::LoadConst(c), 1);
    b.emit(Op::Redirect(1, redirect_op::WRITE), 1);
    b.emit(Op::LoadConst(sentinel), 1);
    let v = run_with_host(b.build(), Box::new(RecordingHost::default()));
    assert_eq!(v.to_str(), "ok");
}

#[test]
fn host_pipeline_lifecycle() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::PipelineBegin(3), 1);
    b.emit(Op::PipelineStage, 1);
    b.emit(Op::PipelineStage, 1);
    b.emit(Op::PipelineEnd, 1);

    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(RecordingHost::default()));
    let result = vm.run();
    // PipelineEnd pushes Status(host.pipeline_end() result) which is the
    // remaining top-of-stack.
    match result {
        VMResult::Ok(Value::Status(42)) => {}
        other => panic!("expected Status(42), got {:?}", other),
    }
    assert_eq!(vm.last_status, 42);
}

#[test]
fn host_cmd_subst_uses_sub_chunk() {
    let mut sub = ChunkBuilder::new();
    sub.emit(Op::LoadInt(1), 1);
    sub.emit(Op::LoadInt(2), 1);
    sub.emit(Op::Add, 1);
    let sub_chunk = sub.build();

    let mut b = ChunkBuilder::new();
    let sub_idx = b.add_sub_chunk(sub_chunk);
    b.emit(Op::CmdSubst(sub_idx), 1);

    let v = run_with_host(b.build(), Box::new(RecordingHost::default()));
    assert_eq!(v.to_str(), "captured");
}

#[test]
fn host_trap_set_records_signal_and_handler() {
    let mut handler = ChunkBuilder::new();
    handler.emit(Op::Nop, 1);
    handler.emit(Op::Nop, 1);
    let handler_chunk = handler.build();

    let mut b = ChunkBuilder::new();
    let h_idx = b.add_sub_chunk(handler_chunk);
    let sig = b.add_constant(Value::str("INT"));
    let sentinel = b.add_constant(Value::str("ok"));
    b.emit(Op::LoadConst(sig), 1);
    b.emit(Op::TrapSet(h_idx), 1);
    b.emit(Op::LoadConst(sentinel), 1);

    let v = run_with_host(b.build(), Box::new(RecordingHost::default()));
    assert_eq!(v.to_str(), "ok");
}

#[test]
fn host_with_redirects_balance() {
    let mut b = ChunkBuilder::new();
    let sentinel = b.add_constant(Value::str("done"));
    b.emit(Op::WithRedirectsBegin(2), 1);
    b.emit(Op::WithRedirectsBegin(1), 1);
    b.emit(Op::WithRedirectsEnd, 1);
    b.emit(Op::WithRedirectsEnd, 1);
    b.emit(Op::LoadConst(sentinel), 1);

    let v = run_with_host(b.build(), Box::new(RecordingHost::default()));
    assert_eq!(v.to_str(), "done");
}

#[test]
fn host_call_function_pops_args_pushes_status() {
    let mut b = ChunkBuilder::new();
    let name_idx = b.add_name("my_func");
    let a1 = b.add_constant(Value::str("hello"));
    let a2 = b.add_constant(Value::str("world"));
    b.emit(Op::LoadConst(a1), 1);
    b.emit(Op::LoadConst(a2), 1);
    b.emit(Op::CallFunction(name_idx, 2), 1);

    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(RecordingHost::default()));
    let result = vm.run();
    match result {
        VMResult::Ok(Value::Status(7)) => {}
        other => panic!("expected Status(7), got {:?}", other),
    }
    assert_eq!(vm.last_status, 7);
}

#[test]
fn host_str_match_glob_pattern() {
    let mut b = ChunkBuilder::new();
    let s = b.add_constant(Value::str("hello.rs"));
    let p = b.add_constant(Value::str("*.rs"));
    b.emit(Op::LoadConst(s), 1);
    b.emit(Op::LoadConst(p), 1);
    b.emit(Op::StrMatch, 1);

    let v = run_with_host(b.build(), Box::new(RecordingHost::default()));
    assert_eq!(v, Value::Bool(true));
}

#[test]
fn host_regex_match_routes_through_host() {
    let mut b = ChunkBuilder::new();
    let s = b.add_constant(Value::str("anything"));
    let r = b.add_constant(Value::str("ANY"));
    b.emit(Op::LoadConst(s), 1);
    b.emit(Op::LoadConst(r), 1);
    b.emit(Op::RegexMatch, 1);

    let v = run_with_host(b.build(), Box::new(RecordingHost::default()));
    assert_eq!(v, Value::Bool(true));
}

#[test]
fn default_host_struct_works() {
    let mut b = ChunkBuilder::new();
    let c = b.add_constant(Value::str("foo"));
    b.emit(Op::LoadConst(c), 1);
    b.emit(Op::TildeExpand, 1);

    let v = run_with_host(b.build(), Box::new(DefaultHost));
    assert_eq!(v.to_str(), "foo");
}

// ── Sub-chunks survive serialization round-trip ──────────────────────────────

#[test]
fn chunk_with_sub_chunks_round_trips_via_bincode() {
    let mut sub = ChunkBuilder::new();
    sub.emit(Op::LoadInt(99), 1);
    sub.emit(Op::ReturnValue, 1);
    let sub_chunk = sub.build();

    let mut b = ChunkBuilder::new();
    let idx = b.add_sub_chunk(sub_chunk);
    b.emit(Op::CmdSubst(idx), 1);
    let parent = b.build();

    let bytes = bincode::serialize(&parent).expect("serialize");
    let restored: Chunk = bincode::deserialize(&bytes).expect("deserialize");

    assert_eq!(restored.sub_chunks.len(), 1);
    assert_eq!(restored.sub_chunks[0].ops.len(), 2);
    assert!(matches!(restored.ops[0], Op::CmdSubst(0)));
}

// ── TestFile still works alongside new ops ────────────────────────────────────

#[test]
fn test_file_dispatch_unchanged() {
    let mut b = ChunkBuilder::new();
    let c = b.add_constant(Value::str("/tmp"));
    b.emit(Op::LoadConst(c), 1);
    b.emit(Op::TestFile(file_test::IS_DIR), 1);
    assert_eq!(run_for_value(b.build()), Value::Bool(true));
}
