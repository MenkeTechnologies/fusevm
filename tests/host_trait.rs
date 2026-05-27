//! Direct exercises of every `ShellHost` default method on `DefaultHost`.

use fusevm::host::{DefaultHost, ShellHost};
use fusevm::op::param_mod;
use fusevm::{Chunk, Value};

// ── tilde_expand ───────────────────────────────────────────────────────────

#[test]
fn tilde_expand_preserves_all_inputs() {
    let mut h = DefaultHost;
    for s in &["", "~", "~user", "~/path", "no-tilde", "~+", "~-"] {
        assert_eq!(h.tilde_expand(s), *s);
    }
}

// ── brace_expand ───────────────────────────────────────────────────────────

#[test]
fn brace_expand_wraps_single_element() {
    let mut h = DefaultHost;
    assert_eq!(h.brace_expand("{a,b}"), vec!["{a,b}".to_string()]);
    assert_eq!(h.brace_expand(""), vec!["".to_string()]);
    assert_eq!(h.brace_expand("plain"), vec!["plain".to_string()]);
}

// ── word_split ─────────────────────────────────────────────────────────────

#[test]
fn word_split_empty_yields_empty_vec() {
    let mut h = DefaultHost;
    let r = h.word_split("");
    assert!(r.is_empty());
}

#[test]
fn word_split_collapses_internal_whitespace() {
    let mut h = DefaultHost;
    let r = h.word_split("  a   b\t\tc\nd  ");
    assert_eq!(r, vec!["a", "b", "c", "d"]);
}

#[test]
fn word_split_single_word() {
    let mut h = DefaultHost;
    assert_eq!(h.word_split("solo"), vec!["solo".to_string()]);
}

// ── glob (uses glob crate) ─────────────────────────────────────────────────

#[test]
fn glob_matches_cargo_toml() {
    let mut h = DefaultHost;
    let r = h.glob("Cargo.toml", false);
    assert!(r.iter().any(|p| p.ends_with("Cargo.toml")));
}

#[test]
fn glob_no_match_returns_empty() {
    let mut h = DefaultHost;
    let r = h.glob("this/path/definitely/does/not/exist/*.zz", false);
    assert!(r.is_empty());
}

#[test]
fn glob_recursive_flag_is_ignored() {
    // DefaultHost ignores recursive; same result either way.
    let mut h = DefaultHost;
    let a = h.glob("Cargo.toml", false);
    let b = h.glob("Cargo.toml", true);
    assert_eq!(a, b);
}

#[test]
fn glob_invalid_pattern_returns_empty() {
    // Unmatched [ — glob returns an error iterator → no matches.
    let mut h = DefaultHost;
    let r = h.glob("src/[invalid", false);
    assert!(r.is_empty());
}

// ── expand_param ───────────────────────────────────────────────────────────

#[test]
fn expand_param_default_returns_empty_string_for_all_modifiers() {
    let mut h = DefaultHost;
    let mods = [
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
    ];
    for m in mods {
        let v = h.expand_param("PATH", m, &[]);
        assert_eq!(v, Value::str(""));
    }
}

#[test]
fn expand_param_ignores_args() {
    let mut h = DefaultHost;
    let v = h.expand_param("var", param_mod::DEFAULT, &[Value::str("fallback")]);
    assert_eq!(v, Value::str(""));
}

// ── array_index ────────────────────────────────────────────────────────────

#[test]
fn array_index_returns_undef() {
    let mut h = DefaultHost;
    assert_eq!(h.array_index("arr", &Value::Int(0)), Value::Undef);
    assert_eq!(h.array_index("arr", &Value::str("key")), Value::Undef);
}

// ── cmd_subst / process_sub_in / out ───────────────────────────────────────

#[test]
fn cmd_subst_returns_empty_string() {
    let mut h = DefaultHost;
    assert_eq!(h.cmd_subst(&Chunk::new()), "");
}

#[test]
fn process_sub_in_returns_empty_string() {
    let mut h = DefaultHost;
    assert_eq!(h.process_sub_in(&Chunk::new()), "");
}

#[test]
fn process_sub_out_returns_empty_string() {
    let mut h = DefaultHost;
    assert_eq!(h.process_sub_out(&Chunk::new()), "");
}

// ── pipeline / subshell / trap / redirect no-ops ──────────────────────────

#[test]
fn pipeline_lifecycle_does_not_panic() {
    let mut h = DefaultHost;
    h.pipeline_begin(3);
    h.pipeline_stage();
    h.pipeline_stage();
    let s = h.pipeline_end();
    assert_eq!(s, 0);
}

#[test]
fn subshell_lifecycle_does_not_panic() {
    let mut h = DefaultHost;
    h.subshell_begin();
    h.subshell_end();
}

#[test]
fn trap_set_and_check_do_not_panic() {
    let mut h = DefaultHost;
    h.trap_set("INT", &Chunk::new());
    h.trap_check();
}

#[test]
fn with_redirects_balance_does_not_panic() {
    let mut h = DefaultHost;
    h.with_redirects_begin(2);
    h.with_redirects_end();
}

#[test]
fn redirect_called_with_various_fds() {
    let mut h = DefaultHost;
    h.redirect(0, 0, "in.txt");
    h.redirect(1, 1, "out.txt");
    h.redirect(2, 2, "err.txt");
}

#[test]
fn heredoc_and_herestring_no_op() {
    let mut h = DefaultHost;
    h.heredoc("EOF body\nline2\n");
    h.herestring("input");
}

// ── call_function / str_match / regex_match ────────────────────────────────

#[test]
fn call_function_returns_none_for_unknown() {
    let mut h = DefaultHost;
    assert_eq!(h.call_function("missing", vec![]), None);
    assert_eq!(h.call_function("foo", vec!["a".into(), "b".into()]), None);
}

#[test]
fn str_match_exact_equality_default() {
    let mut h = DefaultHost;
    assert!(h.str_match("abc", "abc"));
    assert!(!h.str_match("abc", "ab*"));
    assert!(!h.str_match("", "*"));
    assert!(h.str_match("", ""));
}

#[test]
fn regex_match_always_false_by_default() {
    let mut h = DefaultHost;
    assert!(!h.regex_match("abc", "abc"));
    assert!(!h.regex_match("hello", ".*"));
    assert!(!h.regex_match("", ""));
}

// ── exec / exec_bg (use harmless commands) ────────────────────────────────

#[test]
fn exec_empty_args_returns_zero() {
    let mut h = DefaultHost;
    assert_eq!(h.exec(vec![]), 0);
}

#[test]
fn exec_bg_empty_args_returns_zero() {
    let mut h = DefaultHost;
    assert_eq!(h.exec_bg(vec![]), 0);
}

#[test]
fn exec_nonexistent_command_returns_127() {
    let mut h = DefaultHost;
    let status = h.exec(vec!["/this/binary/does/not/exist/zzz_fusevm_test".into()]);
    assert_eq!(status, 127);
}

#[test]
fn exec_bg_nonexistent_command_returns_zero() {
    let mut h = DefaultHost;
    let pid = h.exec_bg(vec!["/this/binary/does/not/exist/zzz_fusevm_test".into()]);
    assert_eq!(pid, 0);
}

// ── Custom host overriding methods (proves the trait is properly dispatched) ─

struct CountingHost {
    pipeline_n: u8,
    cmd_subst_calls: u32,
    redirect_calls: u32,
    last_status: i32,
}

impl ShellHost for CountingHost {
    fn pipeline_begin(&mut self, n: u8) {
        self.pipeline_n = n;
    }
    fn pipeline_end(&mut self) -> i32 {
        self.last_status
    }
    fn cmd_subst(&mut self, _sub: &Chunk) -> String {
        self.cmd_subst_calls += 1;
        format!("call#{}", self.cmd_subst_calls)
    }
    fn redirect(&mut self, _fd: u8, _op: u8, _target: &str) {
        self.redirect_calls += 1;
    }
}

#[test]
fn custom_host_pipeline_begin_records_n() {
    let mut h = CountingHost {
        pipeline_n: 0,
        cmd_subst_calls: 0,
        redirect_calls: 0,
        last_status: 42,
    };
    h.pipeline_begin(7);
    assert_eq!(h.pipeline_n, 7);
    assert_eq!(h.pipeline_end(), 42);
}

#[test]
fn custom_host_cmd_subst_returns_per_call_string() {
    let mut h = CountingHost {
        pipeline_n: 0,
        cmd_subst_calls: 0,
        redirect_calls: 0,
        last_status: 0,
    };
    let c = Chunk::new();
    assert_eq!(h.cmd_subst(&c), "call#1");
    assert_eq!(h.cmd_subst(&c), "call#2");
    assert_eq!(h.cmd_subst_calls, 2);
}

#[test]
fn custom_host_redirect_count_accumulates() {
    let mut h = CountingHost {
        pipeline_n: 0,
        cmd_subst_calls: 0,
        redirect_calls: 0,
        last_status: 0,
    };
    h.redirect(1, 0, "out");
    h.redirect(2, 1, "err");
    h.redirect(0, 2, "in");
    assert_eq!(h.redirect_calls, 3);
}
