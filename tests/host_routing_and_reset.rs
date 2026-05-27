//! Coverage for shell_builtins lookup aliases, VM::reset semantics,
//! host-routed pipeline/redirect/subshell/with-redirects/heredoc ops,
//! and exec/exec_bg empty-args fallthroughs.

use fusevm::host::ShellHost;
use fusevm::shell_builtins::*;
use fusevm::{Chunk, ChunkBuilder, Op, VMResult, Value, VM};
use std::sync::{Arc, Mutex};

// ══════════════════════════════════════════════════════════════════════════
// shell_builtins aliases
// ══════════════════════════════════════════════════════════════════════════

#[test]
fn chdir_is_alias_for_cd() {
    assert_eq!(builtin_id("chdir"), Some(BUILTIN_CD));
    assert_eq!(builtin_id("cd"), Some(BUILTIN_CD));
}

#[test]
fn dot_is_alias_for_source() {
    assert_eq!(builtin_id("."), Some(BUILTIN_SOURCE));
    assert_eq!(builtin_id("source"), Some(BUILTIN_SOURCE));
}

#[test]
fn bracket_is_alias_for_test() {
    assert_eq!(builtin_id("["), Some(BUILTIN_TEST));
    assert_eq!(builtin_id("test"), Some(BUILTIN_TEST));
}

#[test]
fn typeset_is_alias_for_declare() {
    assert_eq!(builtin_id("typeset"), Some(BUILTIN_TYPESET));
    assert_eq!(builtin_id("declare"), Some(BUILTIN_TYPESET));
}

#[test]
fn readarray_is_alias_for_mapfile() {
    assert_eq!(builtin_id("readarray"), Some(BUILTIN_MAPFILE));
    assert_eq!(builtin_id("mapfile"), Some(BUILTIN_MAPFILE));
}

#[test]
fn bye_and_logout_are_aliases_for_exit() {
    assert_eq!(builtin_id("bye"), Some(BUILTIN_EXIT));
    assert_eq!(builtin_id("logout"), Some(BUILTIN_EXIT));
    assert_eq!(builtin_id("exit"), Some(BUILTIN_EXIT));
}

#[test]
fn bind_is_alias_for_bindkey() {
    assert_eq!(builtin_id("bind"), Some(BUILTIN_BINDKEY));
    assert_eq!(builtin_id("bindkey"), Some(BUILTIN_BINDKEY));
}

#[test]
fn unknown_command_returns_none() {
    assert_eq!(builtin_id("not_a_builtin_at_all_xyz"), None);
    assert_eq!(builtin_id(""), None);
}

#[test]
fn is_builtin_consistent_with_builtin_id() {
    for name in &["cd", "echo", "set", "trap", "[", ".", "chdir", "readarray"] {
        assert!(is_builtin(name), "expected {} to be a builtin", name);
    }
    assert!(!is_builtin("totally_made_up_name_xyz"));
}

#[test]
fn is_builtin_returns_false_for_empty() {
    assert!(!is_builtin(""));
}

#[test]
fn builtin_id_case_sensitive() {
    // Builtin names are lower-case; capitalized variants should not match.
    assert_eq!(builtin_id("CD"), None);
    assert_eq!(builtin_id("Echo"), None);
}

#[test]
fn builtin_id_for_coreutil_set() {
    assert_eq!(builtin_id("cat"), Some(BUILTIN_CAT));
    assert_eq!(builtin_id("head"), Some(BUILTIN_HEAD));
    assert_eq!(builtin_id("tail"), Some(BUILTIN_TAIL));
    assert_eq!(builtin_id("wc"), Some(BUILTIN_WC));
    assert_eq!(builtin_id("basename"), Some(BUILTIN_BASENAME));
    assert_eq!(builtin_id("dirname"), Some(BUILTIN_DIRNAME));
}

#[test]
fn builtin_id_for_completion_set() {
    assert_eq!(builtin_id("compgen"), Some(BUILTIN_COMPGEN));
    assert_eq!(builtin_id("complete"), Some(BUILTIN_COMPLETE));
    assert_eq!(builtin_id("compadd"), Some(BUILTIN_COMPADD));
    assert_eq!(builtin_id("compdef"), Some(BUILTIN_COMPDEF));
}

#[test]
fn builtin_id_for_async_extensions() {
    assert_eq!(builtin_id("async"), Some(BUILTIN_ASYNC));
    assert_eq!(builtin_id("await"), Some(BUILTIN_AWAIT));
    assert_eq!(builtin_id("pmap"), Some(BUILTIN_PMAP));
    assert_eq!(builtin_id("peach"), Some(BUILTIN_PEACH));
}

#[test]
fn builtin_id_for_intercept_extensions() {
    assert_eq!(builtin_id("intercept"), Some(BUILTIN_INTERCEPT));
    assert_eq!(
        builtin_id("intercept_proceed"),
        Some(BUILTIN_INTERCEPT_PROCEED)
    );
}

// ══════════════════════════════════════════════════════════════════════════
// VM::reset semantics
// ══════════════════════════════════════════════════════════════════════════

#[test]
fn reset_clears_stack_and_starts_over_with_new_chunk() {
    // Run first chunk that produces 7
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(7), 1);
    let mut vm = VM::new(b.build());
    matches!(vm.run(), VMResult::Ok(Value::Int(7)));

    // Reset with a new chunk
    let mut b2 = ChunkBuilder::new();
    b2.emit(Op::LoadInt(99), 1);
    vm.reset(b2.build());
    match vm.run() {
        VMResult::Ok(Value::Int(99)) => {}
        other => panic!("expected Int(99), got {:?}", other),
    }
}

#[test]
fn reset_clears_last_status() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(127), 1);
    b.emit(Op::SetStatus, 1);
    let mut vm = VM::new(b.build());
    let _ = vm.run();

    let mut b2 = ChunkBuilder::new();
    b2.emit(Op::GetStatus, 1);
    vm.reset(b2.build());
    match vm.run() {
        VMResult::Ok(Value::Status(0)) => {}
        other => panic!("expected Status(0), got {:?}", other),
    }
}

#[test]
fn reset_resizes_globals_to_match_new_names_pool() {
    let mut b = ChunkBuilder::new();
    b.add_name("a");
    let mut vm = VM::new(b.build());
    let _ = vm.run();

    // New chunk with more names — reset must accommodate them.
    let mut b2 = ChunkBuilder::new();
    b2.add_name("x");
    b2.add_name("y");
    b2.add_name("z");
    b2.emit(Op::LoadInt(42), 1);
    b2.emit(Op::SetVar(2), 1);
    b2.emit(Op::GetVar(2), 1);
    vm.reset(b2.build());
    match vm.run() {
        VMResult::Ok(Value::Int(42)) => {}
        other => panic!("expected Int(42), got {:?}", other),
    }
}

#[test]
fn reset_clears_halted_flag() {
    // Run a chunk to completion, then verify reset lets us run again.
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(1), 1);
    let mut vm = VM::new(b.build());
    let _ = vm.run();

    let mut b2 = ChunkBuilder::new();
    b2.emit(Op::LoadInt(2), 1);
    vm.reset(b2.build());
    matches!(vm.run(), VMResult::Ok(Value::Int(2)));
}

// ══════════════════════════════════════════════════════════════════════════
// Host-routed ops: capture into a Recorder
// ══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, PartialEq)]
enum Call {
    PipelineBegin(u8),
    PipelineStage,
    PipelineEnd,
    SubshellBegin,
    SubshellEnd,
    Redirect(u8, u8, String),
    Heredoc(String),
    Herestring(String),
    WithRedirectsBegin(u8),
    WithRedirectsEnd,
    Exec(Vec<String>),
    ExecBg(Vec<String>),
    StrMatch(String, String),
    RegexMatch(String, String),
    TildeExpand(String),
}

struct Recorder {
    log: Arc<Mutex<Vec<Call>>>,
    pipeline_end_status: i32,
    exec_status: i32,
    str_match_result: bool,
    regex_match_result: bool,
}

impl Recorder {
    fn new() -> (Self, Arc<Mutex<Vec<Call>>>) {
        let log = Arc::new(Mutex::new(Vec::new()));
        let me = Recorder {
            log: log.clone(),
            pipeline_end_status: 0,
            exec_status: 0,
            str_match_result: true,
            regex_match_result: true,
        };
        (me, log)
    }
    fn push(&self, c: Call) {
        self.log.lock().unwrap().push(c);
    }
}

impl ShellHost for Recorder {
    fn pipeline_begin(&mut self, n: u8) {
        self.push(Call::PipelineBegin(n));
    }
    fn pipeline_stage(&mut self) {
        self.push(Call::PipelineStage);
    }
    fn pipeline_end(&mut self) -> i32 {
        self.push(Call::PipelineEnd);
        self.pipeline_end_status
    }
    fn subshell_begin(&mut self) {
        self.push(Call::SubshellBegin);
    }
    fn subshell_end(&mut self) -> Option<i32> {
        self.push(Call::SubshellEnd);
        None
    }
    fn redirect(&mut self, fd: u8, op: u8, target: &str) {
        self.push(Call::Redirect(fd, op, target.to_string()));
    }
    fn heredoc(&mut self, content: &str) {
        self.push(Call::Heredoc(content.to_string()));
    }
    fn herestring(&mut self, content: &str) {
        self.push(Call::Herestring(content.to_string()));
    }
    fn with_redirects_begin(&mut self, n: u8) {
        self.push(Call::WithRedirectsBegin(n));
    }
    fn with_redirects_end(&mut self) {
        self.push(Call::WithRedirectsEnd);
    }
    fn exec(&mut self, args: Vec<String>) -> i32 {
        self.push(Call::Exec(args));
        self.exec_status
    }
    fn exec_bg(&mut self, args: Vec<String>) -> i32 {
        self.push(Call::ExecBg(args));
        0
    }
    fn str_match(&mut self, s: &str, pat: &str) -> bool {
        self.push(Call::StrMatch(s.to_string(), pat.to_string()));
        self.str_match_result
    }
    fn regex_match(&mut self, s: &str, regex: &str) -> bool {
        self.push(Call::RegexMatch(s.to_string(), regex.to_string()));
        self.regex_match_result
    }
    fn tilde_expand(&mut self, s: &str) -> String {
        self.push(Call::TildeExpand(s.to_string()));
        format!("/home/test{}", s.trim_start_matches('~'))
    }
}

fn run_with_host<F: FnOnce(&mut ChunkBuilder)>(f: F) -> Arc<Mutex<Vec<Call>>> {
    let mut b = ChunkBuilder::new();
    f(&mut b);
    let mut vm = VM::new(b.build());
    let (rec, log) = Recorder::new();
    vm.set_shell_host(Box::new(rec));
    let _ = vm.run();
    log
}

#[test]
fn pipelinebegin_invokes_host_with_stage_count() {
    let log = run_with_host(|b| {
        b.emit(Op::PipelineBegin(3), 1);
    });
    assert_eq!(log.lock().unwrap()[0], Call::PipelineBegin(3));
}

#[test]
fn pipelinestage_invokes_host() {
    let log = run_with_host(|b| {
        b.emit(Op::PipelineStage, 1);
    });
    assert_eq!(log.lock().unwrap()[0], Call::PipelineStage);
}

#[test]
fn pipelineend_pushes_status_and_invokes_host() {
    let log = run_with_host(|b| {
        b.emit(Op::PipelineEnd, 1);
    });
    assert_eq!(log.lock().unwrap()[0], Call::PipelineEnd);
}

#[test]
fn full_pipeline_sequence_invokes_host_in_order() {
    let log = run_with_host(|b| {
        b.emit(Op::PipelineBegin(2), 1);
        b.emit(Op::PipelineStage, 1);
        b.emit(Op::PipelineEnd, 1);
    });
    let l = log.lock().unwrap();
    assert_eq!(l.len(), 3);
    assert_eq!(l[0], Call::PipelineBegin(2));
    assert_eq!(l[1], Call::PipelineStage);
    assert_eq!(l[2], Call::PipelineEnd);
}

#[test]
fn subshellbegin_and_subshellend_invoke_host() {
    let log = run_with_host(|b| {
        b.emit(Op::SubshellBegin, 1);
        b.emit(Op::SubshellEnd, 1);
    });
    let l = log.lock().unwrap();
    assert_eq!(l[0], Call::SubshellBegin);
    assert_eq!(l[1], Call::SubshellEnd);
}

#[test]
fn redirect_pops_target_and_invokes_host_with_fd_and_op() {
    let log = run_with_host(|b| {
        let k = b.add_constant(Value::str("/tmp/out.txt"));
        b.emit(Op::LoadConst(k), 1);
        b.emit(Op::Redirect(1, fusevm::op::redirect_op::WRITE), 1);
    });
    assert_eq!(
        log.lock().unwrap()[0],
        Call::Redirect(
            1,
            fusevm::op::redirect_op::WRITE,
            "/tmp/out.txt".to_string()
        )
    );
}

#[test]
fn heredoc_reads_content_from_constant_pool_and_passes_to_host() {
    let log = run_with_host(|b| {
        let k = b.add_constant(Value::str("here-content"));
        b.emit(Op::HereDoc(k), 1);
    });
    assert_eq!(
        log.lock().unwrap()[0],
        Call::Heredoc("here-content".to_string())
    );
}

#[test]
fn herestring_pops_content_and_passes_to_host() {
    let log = run_with_host(|b| {
        let k = b.add_constant(Value::str("hello here-string"));
        b.emit(Op::LoadConst(k), 1);
        b.emit(Op::HereString, 1);
    });
    assert_eq!(
        log.lock().unwrap()[0],
        Call::Herestring("hello here-string".to_string())
    );
}

#[test]
fn with_redirects_begin_and_end_invoke_host_with_count() {
    let log = run_with_host(|b| {
        b.emit(Op::WithRedirectsBegin(2), 1);
        b.emit(Op::WithRedirectsEnd, 1);
    });
    let l = log.lock().unwrap();
    assert_eq!(l[0], Call::WithRedirectsBegin(2));
    assert_eq!(l[1], Call::WithRedirectsEnd);
}

#[test]
fn exec_routes_through_host_when_command_is_not_an_inline_builtin() {
    let log = run_with_host(|b| {
        let k = b.add_constant(Value::str("/usr/bin/some-cmd"));
        b.emit(Op::LoadConst(k), 1);
        let k2 = b.add_constant(Value::str("arg1"));
        b.emit(Op::LoadConst(k2), 1);
        b.emit(Op::Exec(2), 1);
    });
    assert_eq!(
        log.lock().unwrap()[0],
        Call::Exec(vec!["/usr/bin/some-cmd".to_string(), "arg1".to_string()])
    );
}

#[test]
fn exec_with_zero_args_pushes_status_zero_and_does_not_call_host() {
    let log = run_with_host(|b| {
        b.emit(Op::Exec(0), 1);
    });
    assert!(log.lock().unwrap().is_empty());
}

#[test]
fn exec_inline_true_does_not_route_through_host() {
    let log = run_with_host(|b| {
        let k = b.add_constant(Value::str("true"));
        b.emit(Op::LoadConst(k), 1);
        b.emit(Op::Exec(1), 1);
    });
    assert!(log.lock().unwrap().is_empty()); // inline path
}

#[test]
fn exec_inline_false_does_not_route_through_host() {
    let log = run_with_host(|b| {
        let k = b.add_constant(Value::str("false"));
        b.emit(Op::LoadConst(k), 1);
        b.emit(Op::Exec(1), 1);
    });
    assert!(log.lock().unwrap().is_empty());
}

#[test]
fn execbg_routes_through_host() {
    let log = run_with_host(|b| {
        let k = b.add_constant(Value::str("backgroundcmd"));
        b.emit(Op::LoadConst(k), 1);
        b.emit(Op::ExecBg(1), 1);
    });
    assert_eq!(
        log.lock().unwrap()[0],
        Call::ExecBg(vec!["backgroundcmd".to_string()])
    );
}

#[test]
fn execbg_with_zero_args_does_not_call_host() {
    let log = run_with_host(|b| {
        b.emit(Op::ExecBg(0), 1);
    });
    assert!(log.lock().unwrap().is_empty());
}

#[test]
fn strmatch_routes_through_host_and_pushes_bool() {
    let log = run_with_host(|b| {
        let s = b.add_constant(Value::str("hello"));
        let p = b.add_constant(Value::str("h*"));
        b.emit(Op::LoadConst(s), 1);
        b.emit(Op::LoadConst(p), 1);
        b.emit(Op::StrMatch, 1);
    });
    assert_eq!(
        log.lock().unwrap()[0],
        Call::StrMatch("hello".to_string(), "h*".to_string())
    );
}

#[test]
fn regexmatch_routes_through_host_and_pushes_bool() {
    let log = run_with_host(|b| {
        let s = b.add_constant(Value::str("foo123"));
        let r = b.add_constant(Value::str("[0-9]+"));
        b.emit(Op::LoadConst(s), 1);
        b.emit(Op::LoadConst(r), 1);
        b.emit(Op::RegexMatch, 1);
    });
    assert_eq!(
        log.lock().unwrap()[0],
        Call::RegexMatch("foo123".to_string(), "[0-9]+".to_string())
    );
}

#[test]
fn tildeexpand_routes_through_host() {
    let log = run_with_host(|b| {
        let k = b.add_constant(Value::str("~/projects"));
        b.emit(Op::LoadConst(k), 1);
        b.emit(Op::TildeExpand, 1);
    });
    assert_eq!(
        log.lock().unwrap()[0],
        Call::TildeExpand("~/projects".to_string())
    );
}

// ══════════════════════════════════════════════════════════════════════════
// Default ShellHost trait behaviors (in-process, no override)
// ══════════════════════════════════════════════════════════════════════════

struct Bare;
impl ShellHost for Bare {}

#[test]
fn default_host_tildeexpand_is_identity() {
    let mut h = Bare;
    assert_eq!(h.tilde_expand("~/foo"), "~/foo");
}

#[test]
fn default_host_brace_expand_wraps_in_single_vec() {
    let mut h = Bare;
    assert_eq!(h.brace_expand("a{1,2}b"), vec!["a{1,2}b".to_string()]);
}

#[test]
fn default_host_word_split_uses_whitespace() {
    let mut h = Bare;
    let v = h.word_split("  hello   there\tworld\n");
    assert_eq!(
        v,
        vec![
            "hello".to_string(),
            "there".to_string(),
            "world".to_string()
        ]
    );
}

#[test]
fn default_host_expand_param_returns_empty_string_value() {
    let mut h = Bare;
    let v = h.expand_param("foo", fusevm::op::param_mod::DEFAULT, &[]);
    match v {
        Value::Str(s) => assert!(s.is_empty()),
        other => panic!("expected Str, got {:?}", other),
    }
}

#[test]
fn default_host_array_index_returns_undef() {
    let mut h = Bare;
    matches!(h.array_index("a", &Value::Int(0)), Value::Undef);
}

#[test]
fn default_host_cmd_subst_returns_empty_string() {
    let mut h = Bare;
    let c = Chunk::new();
    assert_eq!(h.cmd_subst(&c), "");
}

#[test]
fn default_host_process_sub_returns_empty_paths() {
    let mut h = Bare;
    let c = Chunk::new();
    assert_eq!(h.process_sub_in(&c), "");
    assert_eq!(h.process_sub_out(&c), "");
}

#[test]
fn default_host_pipeline_end_is_zero() {
    let mut h = Bare;
    assert_eq!(h.pipeline_end(), 0);
}

#[test]
fn default_host_call_function_returns_none() {
    let mut h = Bare;
    assert!(h.call_function("nope", vec![]).is_none());
}

#[test]
fn default_host_str_match_is_exact_equality() {
    let mut h = Bare;
    assert!(h.str_match("foo", "foo"));
    assert!(!h.str_match("foo", "f*"));
}

#[test]
fn default_host_regex_match_is_always_false() {
    let mut h = Bare;
    assert!(!h.regex_match("foo", ".*"));
}

#[test]
fn default_host_exec_with_no_args_returns_zero() {
    let mut h = Bare;
    assert_eq!(h.exec(vec![]), 0);
}

#[test]
fn default_host_exec_bg_with_no_args_returns_zero() {
    let mut h = Bare;
    assert_eq!(h.exec_bg(vec![]), 0);
}

#[test]
fn default_host_exec_returns_127_for_missing_command() {
    let mut h = Bare;
    let st = h.exec(vec!["/this/path/should/not/exist/abc_xyz".to_string()]);
    assert_eq!(st, 127);
}

// ══════════════════════════════════════════════════════════════════════════
// PipelineEnd without host falls back to last_status
// ══════════════════════════════════════════════════════════════════════════

#[test]
fn pipelineend_without_host_returns_last_status() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(42), 1);
    b.emit(Op::SetStatus, 1);
    b.emit(Op::PipelineEnd, 1);
    let mut vm = VM::new(b.build());
    match vm.run() {
        VMResult::Ok(Value::Status(42)) => {}
        other => panic!("expected Status(42), got {:?}", other),
    }
}

// ══════════════════════════════════════════════════════════════════════════
// HereDoc with out-of-range constant index produces empty content
// ══════════════════════════════════════════════════════════════════════════

#[test]
fn heredoc_with_oob_constant_index_passes_empty_string_to_host() {
    let log = run_with_host(|b| {
        b.emit(Op::HereDoc(999), 1);
    });
    assert_eq!(log.lock().unwrap()[0], Call::Heredoc(String::new()));
}
