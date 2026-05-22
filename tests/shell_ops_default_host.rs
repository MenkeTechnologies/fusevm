//! Coverage for shell-related ops via the DefaultHost: TildeExpand,
//! BraceExpand, WordSplit, Glob/GlobRecursive, ProcessSubIn/Out, TrapCheck,
//! WithRedirectsBegin/End, HereDoc, plus a number of ExpandParam modifiers
//! that the default host returns empty for.

use fusevm::op::param_mod;
use fusevm::{ChunkBuilder, Op, VMResult, Value, VM};

fn run(b: ChunkBuilder) -> Value {
    match VM::new(b.build()).run() {
        VMResult::Ok(v) => v,
        VMResult::Halted => Value::Undef,
        other => panic!("unexpected: {:?}", other),
    }
}

// ── TildeExpand: DefaultHost is identity ────────────────────────────────────

#[test]
fn tilde_expand_default_is_identity() {
    let mut b = ChunkBuilder::new();
    let c = b.add_constant(Value::str("~root/foo"));
    b.emit(Op::LoadConst(c), 1);
    b.emit(Op::TildeExpand, 1);
    assert_eq!(run(b).to_str(), "~root/foo");
}

#[test]
fn tilde_expand_with_empty_input() {
    let mut b = ChunkBuilder::new();
    let c = b.add_constant(Value::str(""));
    b.emit(Op::LoadConst(c), 1);
    b.emit(Op::TildeExpand, 1);
    assert_eq!(run(b).to_str(), "");
}

// ── BraceExpand: DefaultHost wraps single string in vec ─────────────────────

#[test]
fn brace_expand_default_returns_one_element() {
    let mut b = ChunkBuilder::new();
    let c = b.add_constant(Value::str("a{b,c}d"));
    b.emit(Op::LoadConst(c), 1);
    b.emit(Op::BraceExpand, 1);
    match run(b) {
        Value::Array(a) => {
            assert_eq!(a.len(), 1);
            assert_eq!(a[0].to_str(), "a{b,c}d");
        }
        other => panic!("got {:?}", other),
    }
}

// ── WordSplit: DefaultHost splits on whitespace ─────────────────────────────

#[test]
fn word_split_splits_runs_of_whitespace() {
    let mut b = ChunkBuilder::new();
    let c = b.add_constant(Value::str("a  b\tc\nd"));
    b.emit(Op::LoadConst(c), 1);
    b.emit(Op::WordSplit, 1);
    match run(b) {
        Value::Array(a) => {
            let words: Vec<String> = a.iter().map(|v| v.to_str()).collect();
            assert_eq!(words, vec!["a", "b", "c", "d"]);
        }
        other => panic!("got {:?}", other),
    }
}

#[test]
fn word_split_empty_string_produces_empty_array() {
    let mut b = ChunkBuilder::new();
    let c = b.add_constant(Value::str(""));
    b.emit(Op::LoadConst(c), 1);
    b.emit(Op::WordSplit, 1);
    match run(b) {
        Value::Array(a) => assert!(a.is_empty()),
        other => panic!("got {:?}", other),
    }
}

#[test]
fn word_split_only_whitespace_produces_empty_array() {
    let mut b = ChunkBuilder::new();
    let c = b.add_constant(Value::str("   \t\n  "));
    b.emit(Op::LoadConst(c), 1);
    b.emit(Op::WordSplit, 1);
    match run(b) {
        Value::Array(a) => assert!(a.is_empty()),
        other => panic!("got {:?}", other),
    }
}

// ── Glob via DefaultHost (uses glob crate) ──────────────────────────────────

#[test]
fn glob_finds_cargo_toml_in_cwd() {
    let mut b = ChunkBuilder::new();
    let c = b.add_constant(Value::str("Cargo.toml"));
    b.emit(Op::LoadConst(c), 1);
    b.emit(Op::Glob, 1);
    match run(b) {
        Value::Array(a) => assert_eq!(a.len(), 1),
        other => panic!("got {:?}", other),
    }
}

#[test]
fn glob_no_match_yields_empty_array() {
    let mut b = ChunkBuilder::new();
    let c = b.add_constant(Value::str("zzz_no_match_xyzzy_*.rs"));
    b.emit(Op::LoadConst(c), 1);
    b.emit(Op::Glob, 1);
    match run(b) {
        Value::Array(a) => assert!(a.is_empty()),
        other => panic!("got {:?}", other),
    }
}

#[test]
fn glob_wildcard_matches_multiple_files() {
    let mut b = ChunkBuilder::new();
    let c = b.add_constant(Value::str("src/*.rs"));
    b.emit(Op::LoadConst(c), 1);
    b.emit(Op::Glob, 1);
    match run(b) {
        Value::Array(a) => assert!(a.len() >= 4, "got {} matches", a.len()),
        other => panic!("got {:?}", other),
    }
}

// ── ExpandParam: DefaultHost returns empty for all modifiers ────────────────

#[test]
fn expand_param_default_returns_empty_for_length() {
    let mut b = ChunkBuilder::new();
    let c = b.add_constant(Value::str("var"));
    b.emit(Op::LoadConst(c), 1);
    b.emit(Op::ExpandParam(param_mod::LENGTH), 1);
    assert_eq!(run(b).to_str(), "");
}

#[test]
fn expand_param_default_returns_empty_for_upper() {
    let mut b = ChunkBuilder::new();
    let c = b.add_constant(Value::str("var"));
    b.emit(Op::LoadConst(c), 1);
    b.emit(Op::ExpandParam(param_mod::UPPER), 1);
    assert_eq!(run(b).to_str(), "");
}

#[test]
fn expand_param_default_with_default_modifier_pops_arg() {
    let mut b = ChunkBuilder::new();
    let nm = b.add_constant(Value::str("var"));
    let arg = b.add_constant(Value::str("fallback"));
    b.emit(Op::LoadConst(nm), 1);
    b.emit(Op::LoadConst(arg), 1);
    b.emit(Op::ExpandParam(param_mod::DEFAULT), 1);
    // DefaultHost returns "" but pops the arg cleanly.
    assert_eq!(run(b).to_str(), "");
}

#[test]
fn expand_param_subst_pops_two_args() {
    let mut b = ChunkBuilder::new();
    let nm = b.add_constant(Value::str("var"));
    let pat = b.add_constant(Value::str("a"));
    let rep = b.add_constant(Value::str("b"));
    b.emit(Op::LoadConst(nm), 1);
    b.emit(Op::LoadConst(pat), 1);
    b.emit(Op::LoadConst(rep), 1);
    b.emit(Op::ExpandParam(param_mod::SUBST_ALL), 1);
    assert_eq!(run(b).to_str(), "");
}

// ── TrapCheck / TrapSet with default host: no-ops ───────────────────────────

#[test]
fn trap_check_no_op_under_default_host() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::TrapCheck, 1);
    b.emit(Op::LoadInt(42), 1);
    match run(b) {
        Value::Int(42) => {}
        other => panic!("got {:?}", other),
    }
}

// ── WithRedirectsBegin / End balance under default host ─────────────────────

#[test]
fn with_redirects_begin_end_default_is_balanced() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::WithRedirectsBegin(2), 1);
    b.emit(Op::LoadInt(7), 1);
    b.emit(Op::WithRedirectsEnd, 1);
    match run(b) {
        Value::Int(7) => {}
        other => panic!("got {:?}", other),
    }
}

// ── HereDoc takes content from constant pool without popping stack ──────────

#[test]
fn here_doc_does_not_consume_stack() {
    let mut b = ChunkBuilder::new();
    let h = b.add_constant(Value::str("some heredoc body\n"));
    b.emit(Op::LoadInt(99), 1);
    b.emit(Op::HereDoc(h), 1);
    match run(b) {
        Value::Int(99) => {}
        other => panic!("got {:?}", other),
    }
}

// ── PipelineBegin / Stage / End sequence under default host ─────────────────

#[test]
fn pipeline_end_pushes_status_after_default_pipeline() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::PipelineBegin(2), 1);
    b.emit(Op::PipelineStage, 1);
    b.emit(Op::PipelineStage, 1);
    b.emit(Op::PipelineEnd, 1);
    match run(b) {
        Value::Status(0) => {} // DefaultHost::pipeline_end returns 0
        other => panic!("got {:?}", other),
    }
}

// ── SubshellBegin / End default no-ops ──────────────────────────────────────

#[test]
fn subshell_begin_end_under_default_host_does_not_crash() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::LoadInt(123), 1);
    b.emit(Op::SubshellEnd, 1);
    match run(b) {
        Value::Int(123) => {}
        other => panic!("got {:?}", other),
    }
}

// ── Redirect op consumes target string from stack ───────────────────────────

#[test]
fn redirect_pops_target_string() {
    use fusevm::op::redirect_op;
    let mut b = ChunkBuilder::new();
    let s = b.add_constant(Value::str("/tmp/out.txt"));
    b.emit(Op::LoadInt(0), 1); // sentinel below
    b.emit(Op::LoadConst(s), 1);
    b.emit(Op::Redirect(1, redirect_op::WRITE), 1);
    match run(b) {
        Value::Int(0) => {}
        other => panic!("got {:?}", other),
    }
}
