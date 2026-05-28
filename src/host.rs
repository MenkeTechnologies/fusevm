//! Shell host callback interface.
//!
//! Frontends that emit shell-specific bytecodes (zshrs) provide a `ShellHost`
//! implementation. The VM dispatches `Op::Glob`, `Op::TildeExpand`,
//! `Op::ExpandParam`, `Op::CmdSubst`, `Op::Redirect`, pipeline ops, etc.
//! through the host so the actual semantics live in the frontend, not the VM.
//!
//! Without a host, the relevant ops fall back to minimal stubs — the VM
//! still runs, but shell-specific ops are no-ops or pass through inputs.
//!
//! Sub-execution (cmd substitution, process substitution, trap handlers,
//! function bodies) is delivered to the host as `&Chunk` references taken
//! from the parent chunk's `sub_chunks` table. The host is responsible for
//! running them on a fresh or shared VM and reporting captured output / exit
//! status back.

use crate::chunk::Chunk;
use crate::value::Value;

/// Frontend-supplied implementation of shell-specific runtime behavior.
///
/// All methods have default no-op or identity implementations so frontends
/// only override what they need.
pub trait ShellHost: Send {
    /// Glob expand `pattern`. If `recursive`, treat `**` as a recursive marker.
    /// Returns the matched paths (empty when no match — caller decides how to
    /// handle nullglob/nomatch options).
    fn glob(&mut self, pattern: &str, recursive: bool) -> Vec<String> {
        let _ = recursive;
        glob::glob(pattern)
            .into_iter()
            .flat_map(|paths| paths.filter_map(|p| p.ok()))
            .map(|p| p.to_string_lossy().into_owned())
            .collect()
    }

    /// Tilde expansion: `~` → $HOME, `~user` → user's home, `~+`/`~-` → dir stack.
    /// Returns input unchanged if no expansion applies.
    fn tilde_expand(&mut self, s: &str) -> String {
        s.to_string()
    }

    /// Brace expansion: `{a,b}` → ["a","b"], `{1..10}` → 10 strings.
    /// Returns a single-element vec containing the input when no braces present.
    fn brace_expand(&mut self, s: &str) -> Vec<String> {
        vec![s.to_string()]
    }

    /// Word splitting using current IFS rules.
    fn word_split(&mut self, s: &str) -> Vec<String> {
        s.split_whitespace().map(|w| w.to_string()).collect()
    }

    /// Parameter expansion: `${var:-default}`, `${#var}`, `${var/pat/rep}`, etc.
    /// `modifier` is one of `crate::op::param_mod::*`.
    /// `args` are the modifier operands (already evaluated to Values) — for
    /// `${var:-x}` it's `[x]`, for `${var/p/r}` it's `[p, r]`, for `${var:o:l}`
    /// it's `[o, l]`. `LENGTH`/`UPPER`/`LOWER`/`KEYS`/`INDIRECT` take no args.
    fn expand_param(&mut self, name: &str, modifier: u8, args: &[Value]) -> Value {
        let _ = (name, modifier, args);
        Value::str("")
    }

    /// Index into an array variable: `${arr[idx]}`. `index` is the evaluated
    /// subscript (Int for indexed arrays, Str for associative).
    fn array_index(&mut self, name: &str, index: &Value) -> Value {
        let _ = (name, index);
        Value::Undef
    }

    /// Run a sub-chunk and capture its stdout as a string. (`$(cmd)`,`` `cmd` ``)
    fn cmd_subst(&mut self, sub: &Chunk) -> String {
        let _ = sub;
        String::new()
    }

    /// Process substitution input: spawn `sub`, return path to a fd/FIFO that
    /// reads its stdout. (`<(cmd)`)
    fn process_sub_in(&mut self, sub: &Chunk) -> String {
        let _ = sub;
        String::new()
    }

    /// Process substitution output: spawn `sub`, return path to a fd/FIFO that
    /// writes to its stdin. (`>(cmd)`)
    fn process_sub_out(&mut self, sub: &Chunk) -> String {
        let _ = sub;
        String::new()
    }

    /// Apply a redirection at the next exec/builtin call.
    /// `fd` is the source fd, `op` from `crate::op::redirect_op::*`, `target`
    /// is the (already-expanded) filename or fd reference.
    fn redirect(&mut self, fd: u8, op: u8, target: &str) {
        let _ = (fd, op, target);
    }

    /// Heredoc body for the next command's stdin.
    fn heredoc(&mut self, content: &str) {
        let _ = content;
    }

    /// Herestring body for the next command's stdin.
    fn herestring(&mut self, content: &str) {
        let _ = content;
    }

    /// Begin an N-stage pipeline. Subsequent `pipeline_stage` calls separate
    /// stages; `pipeline_end` waits for completion and returns final status.
    fn pipeline_begin(&mut self, n: u8) {
        let _ = n;
    }

    /// Wire next pipeline stage (set up pipe between previous and next).
    fn pipeline_stage(&mut self) {}

    /// Wait for the pipeline to complete; return last command's exit status.
    fn pipeline_end(&mut self) -> i32 {
        0
    }

    /// Begin subshell scope (snapshot/save state).
    fn subshell_begin(&mut self) {}

    /// End subshell scope (restore state). Returns `Some(status)` when
    /// the subshell terminated with a deferred exit (the host wants the
    /// VM's `last_status` updated so the parent's `$?` sees the exit
    /// value); returns `None` to leave `last_status` untouched.
    fn subshell_end(&mut self) -> Option<i32> {
        None
    }

    /// Install a trap handler for signal `sig`. The handler is a sub-chunk
    /// that the host runs when the signal fires.
    fn trap_set(&mut self, sig: &str, handler: &Chunk) {
        let _ = (sig, handler);
    }

    /// Process pending traps (called periodically by the VM dispatch loop).
    fn trap_check(&mut self) {}

    /// Begin scoped redirect block — `cmd > out.txt` style applied to a
    /// compound command. The host saves current fd state.
    fn with_redirects_begin(&mut self, count: u8) {
        let _ = count;
    }

    /// End scoped redirect block — restore fd state.
    fn with_redirects_end(&mut self) {}

    /// Call a user-defined shell function. Returns `Some(status)` when the
    /// function exists, `None` to fall through to external `exec`.
    fn call_function(&mut self, name: &str, args: Vec<String>) -> Option<i32> {
        let _ = (name, args);
        None
    }

    /// Spawn an external command and wait. Default uses `std::process::Command`.
    fn exec(&mut self, args: Vec<String>) -> i32 {
        use std::process::{Command, Stdio};
        let cmd = match args.first() {
            Some(c) => c,
            None => return 0,
        };
        Command::new(cmd)
            .args(&args[1..])
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit())
            .status()
            .map(|s| s.code().unwrap_or(1))
            .unwrap_or(127)
    }

    /// Spawn an external command in the background and detach. Returns the
    /// child pid (or 0 on failure / when the host doesn't track pids). Default
    /// uses `std::process::Command::spawn()`. Frontends override to register
    /// the pid in their job table so `jobs`, `fg`, `wait`, `disown` see it.
    fn exec_bg(&mut self, args: Vec<String>) -> i32 {
        use std::process::{Command, Stdio};
        let cmd = match args.first() {
            Some(c) => c,
            None => return 0,
        };
        Command::new(cmd)
            .args(&args[1..])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .map(|c| c.id() as i32)
            .unwrap_or(0)
    }

    /// Glob match: does `s` match the shell glob pattern `pat`?
    /// Used by `[[ x = pat ]]` and `case`. Default is exact equality.
    fn str_match(&mut self, s: &str, pat: &str) -> bool {
        s == pat
    }

    /// Regex match: `s =~ regex` (extended POSIX or PCRE per host).
    fn regex_match(&mut self, s: &str, regex: &str) -> bool {
        let _ = (s, regex);
        false
    }
}

/// Minimal default host — every method uses the trait's default impl.
/// Useful for tests and for non-shell frontends that still want shell ops
/// to be stack-discipline-correct without writing a full host.
pub struct DefaultHost;

impl ShellHost for DefaultHost {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunk::Chunk;

    #[test]
    fn tilde_expand_is_identity_by_default() {
        let mut h = DefaultHost;
        assert_eq!(h.tilde_expand("~/foo"), "~/foo");
        assert_eq!(h.tilde_expand(""), "");
    }

    #[test]
    fn brace_expand_returns_single_element_vec_by_default() {
        let mut h = DefaultHost;
        assert_eq!(h.brace_expand("{a,b}"), vec!["{a,b}".to_string()]);
        assert_eq!(h.brace_expand("plain"), vec!["plain".to_string()]);
    }

    #[test]
    fn word_split_splits_on_whitespace() {
        let mut h = DefaultHost;
        assert_eq!(h.word_split("one two  three"), vec!["one", "two", "three"]);
        assert!(h.word_split("").is_empty());
        assert!(h.word_split("   \t  ").is_empty());
    }

    #[test]
    fn expand_param_default_returns_empty_string() {
        let mut h = DefaultHost;
        let v = h.expand_param("VAR", 0, &[]);
        assert_eq!(v, Value::str(""));
    }

    #[test]
    fn array_index_default_returns_undef() {
        let mut h = DefaultHost;
        assert_eq!(h.array_index("arr", &Value::Int(0)), Value::Undef);
    }

    #[test]
    fn cmd_subst_and_process_sub_default_to_empty_string() {
        let mut h = DefaultHost;
        let c = Chunk::new();
        assert_eq!(h.cmd_subst(&c), "");
        assert_eq!(h.process_sub_in(&c), "");
        assert_eq!(h.process_sub_out(&c), "");
    }

    #[test]
    fn pipeline_end_default_is_success() {
        let mut h = DefaultHost;
        h.pipeline_begin(2);
        h.pipeline_stage();
        assert_eq!(h.pipeline_end(), 0);
    }

    #[test]
    fn call_function_default_returns_none() {
        let mut h = DefaultHost;
        assert_eq!(h.call_function("fn", vec!["a".into()]), None);
    }

    #[test]
    fn str_match_default_is_exact_equality() {
        let mut h = DefaultHost;
        assert!(h.str_match("foo", "foo"));
        assert!(!h.str_match("foo", "bar"));
        assert!(!h.str_match("foo", "f*"), "default does not glob");
    }

    #[test]
    fn regex_match_default_is_false() {
        let mut h = DefaultHost;
        assert!(!h.regex_match("anything", "."));
    }

    #[test]
    fn noop_methods_do_not_panic() {
        // Verify the methods with `()` returns and no observable state are safe to call.
        let mut h = DefaultHost;
        h.redirect(1, 0, "file");
        h.heredoc("body");
        h.herestring("body");
        h.subshell_begin();
        h.subshell_end();
        h.trap_check();
        h.with_redirects_begin(1);
        h.with_redirects_end();
        h.trap_set("INT", &Chunk::new());
    }

    #[test]
    fn exec_with_empty_args_returns_zero() {
        // First-arg guard avoids spawning anything.
        let mut h = DefaultHost;
        assert_eq!(h.exec(vec![]), 0);
        assert_eq!(h.exec_bg(vec![]), 0);
    }

    // ─── glob default uses system glob crate ──────────────────────────

    #[test]
    fn glob_default_returns_paths_for_literal_pattern() {
        // Default glob impl uses the `glob` crate. A literal path that exists
        // resolves to one entry; the implementation must not panic and must
        // return an empty Vec on no match (not error).
        let mut h = DefaultHost;
        // `/` always exists on Unix; on Windows `C:\` etc. — use temp_dir as a
        // portable existing target.
        let tmp = std::env::temp_dir();
        let tmp_str = tmp.to_string_lossy().to_string();
        let result = h.glob(&tmp_str, false);
        assert_eq!(result.len(), 1, "literal existing path matches itself");
        // Path resolution is implementation-defined; just confirm something came back.
        assert!(!result[0].is_empty());
    }

    #[test]
    fn glob_default_returns_empty_for_nonmatching_pattern() {
        // No matches → empty Vec (nullglob-style default — caller decides
        // how to handle nomatch).
        let mut h = DefaultHost;
        // Use a guaranteed-non-existent absolute pattern.
        let result = h.glob(
            "/this/path/definitely/does/not/exist/anywhere_xyz_*.tmp",
            false,
        );
        assert!(result.is_empty(), "no match → empty, got: {:?}", result);
    }

    #[test]
    fn glob_default_ignores_recursive_flag() {
        // Default impl accepts `recursive` but ignores it (no `**` semantics).
        // Verify the boolean does not change behavior for a simple literal.
        let mut h = DefaultHost;
        let tmp = std::env::temp_dir();
        let tmp_str = tmp.to_string_lossy().to_string();
        let r1 = h.glob(&tmp_str, false);
        let r2 = h.glob(&tmp_str, true);
        assert_eq!(r1, r2);
    }

    // ─── expand_param ignores its arguments by default ────────────────

    #[test]
    fn expand_param_default_ignores_modifier_and_args() {
        // No matter the modifier byte or args, the default impl returns Value::str("").
        // Pins the no-op contract — overriding hosts must replace this method.
        let mut h = DefaultHost;
        assert_eq!(h.expand_param("ANY", 0, &[]), Value::str(""));
        assert_eq!(h.expand_param("ANY", 255, &[]), Value::str(""));
        assert_eq!(
            h.expand_param("ANY", 7, &[Value::Int(42), Value::str("x")]),
            Value::str("")
        );
    }

    // ─── word_split contract: collapses runs of whitespace ────────────

    #[test]
    fn word_split_collapses_consecutive_whitespace() {
        // Mixed spaces and tabs between words should yield the words without
        // empty separators — matches POSIX IFS default behavior.
        let mut h = DefaultHost;
        assert_eq!(
            h.word_split("  a\t b\n\nc \t d  "),
            vec!["a", "b", "c", "d"]
        );
    }

    // ─── array_index returns Undef for any kind of index ──────────────

    #[test]
    fn array_index_default_returns_undef_for_any_index_type() {
        let mut h = DefaultHost;
        assert_eq!(h.array_index("a", &Value::Int(-1)), Value::Undef);
        assert_eq!(h.array_index("", &Value::str("key")), Value::Undef);
        assert_eq!(h.array_index("a", &Value::Undef), Value::Undef);
    }

    // ─── pipeline_end stays at zero across multiple beginnings ────────

    #[test]
    fn pipeline_lifecycle_does_not_drift_status_in_default_impl() {
        // Default impl doesn't track state — repeated cycles always yield 0.
        let mut h = DefaultHost;
        for _ in 0..5 {
            h.pipeline_begin(3);
            h.pipeline_stage();
            h.pipeline_stage();
            assert_eq!(h.pipeline_end(), 0);
        }
    }

    // ─── trap_set with empty signal name ──────────────────────────────

    #[test]
    fn trap_set_default_accepts_any_signal_name_without_panic() {
        let mut h = DefaultHost;
        let c = Chunk::new();
        h.trap_set("", &c);
        h.trap_set("SIGINT", &c);
        h.trap_set("EXIT", &c);
        h.trap_set("\0nonsense", &c);
        // No observable state to assert — the contract is "no panic".
    }
}
