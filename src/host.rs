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

    /// End subshell scope (restore state).
    fn subshell_end(&mut self) {}

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
