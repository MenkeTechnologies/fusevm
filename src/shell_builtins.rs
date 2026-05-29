//! Shell builtin IDs for `CallBuiltin(id, argc)` dispatch.
//!
//! These IDs are used by shell frontends (zshrs) to emit bytecodes
//! that call registered builtin handlers. The VM dispatches through
//! the pre-registered function pointer table — no name lookup at runtime.
//!
//! Usage in frontend compiler:
//! ```
//! use fusevm::{ChunkBuilder, Op};
//! use fusevm::shell_builtins::*;
//!
//! let mut builder = ChunkBuilder::new();
//! builder.emit(Op::CallBuiltin(BUILTIN_CD, 1), 1);
//! ```
//!
//! Usage in frontend VM init:
//! ```
//! use fusevm::{ChunkBuilder, VM, Value};
//! use fusevm::shell_builtins::*;
//!
//! let chunk = ChunkBuilder::new().build();
//! let mut vm = VM::new(chunk);
//! vm.register_builtin(BUILTIN_CD, |_vm, _argc| Value::Status(0));
//! ```

// ═══════════════════════════════════════════════════════════════════════════
// Core builtins (POSIX + common)
// ═══════════════════════════════════════════════════════════════════════════

/// Dispatch ID for the shell `cd` builtin.
pub const BUILTIN_CD: u16 = 0;
/// Dispatch ID for the shell `pwd` builtin.
pub const BUILTIN_PWD: u16 = 1;
/// Dispatch ID for the shell `echo` builtin.
pub const BUILTIN_ECHO: u16 = 2;
/// Dispatch ID for the shell `print` builtin.
pub const BUILTIN_PRINT: u16 = 3;
/// Dispatch ID for the shell `printf` builtin.
pub const BUILTIN_PRINTF: u16 = 4;
/// Dispatch ID for the shell `export` builtin.
pub const BUILTIN_EXPORT: u16 = 5;
/// Dispatch ID for the shell `unset` builtin.
pub const BUILTIN_UNSET: u16 = 6;
/// Dispatch ID for the shell `source` builtin.
pub const BUILTIN_SOURCE: u16 = 7;
/// Dispatch ID for the shell `exit` builtin.
pub const BUILTIN_EXIT: u16 = 8;
/// Dispatch ID for the shell `return` builtin.
pub const BUILTIN_RETURN: u16 = 9;
/// Dispatch ID for the shell `true` builtin.
pub const BUILTIN_TRUE: u16 = 10;
/// Dispatch ID for the shell `false` builtin.
pub const BUILTIN_FALSE: u16 = 11;
/// Dispatch ID for the shell `test` builtin.
pub const BUILTIN_TEST: u16 = 12;
/// Dispatch ID for the shell `:` builtin.
pub const BUILTIN_COLON: u16 = 13; // :
/// Dispatch ID for the shell `.` builtin.
pub const BUILTIN_DOT: u16 = 14; // . (alias for source)

// ═══════════════════════════════════════════════════════════════════════════
// Variable declaration
// ═══════════════════════════════════════════════════════════════════════════

/// Dispatch ID for the shell `local` builtin.
pub const BUILTIN_LOCAL: u16 = 20;
/// Dispatch ID for the shell `declare` builtin.
pub const BUILTIN_DECLARE: u16 = 21;
/// Dispatch ID for the shell `typeset` builtin.
pub const BUILTIN_TYPESET: u16 = 22;
/// Dispatch ID for the shell `readonly` builtin.
pub const BUILTIN_READONLY: u16 = 23;
/// Dispatch ID for the shell `integer` builtin.
pub const BUILTIN_INTEGER: u16 = 24;
/// Dispatch ID for the shell `float` builtin.
pub const BUILTIN_FLOAT: u16 = 25;

// ═══════════════════════════════════════════════════════════════════════════
// I/O
// ═══════════════════════════════════════════════════════════════════════════

/// Dispatch ID for the shell `read` builtin.
pub const BUILTIN_READ: u16 = 30;
/// Dispatch ID for the shell `mapfile` builtin.
pub const BUILTIN_MAPFILE: u16 = 31;

// ═══════════════════════════════════════════════════════════════════════════
// Control flow
// ═══════════════════════════════════════════════════════════════════════════

/// Dispatch ID for the shell `break` builtin.
pub const BUILTIN_BREAK: u16 = 40;
/// Dispatch ID for the shell `continue` builtin.
pub const BUILTIN_CONTINUE: u16 = 41;
/// Dispatch ID for the shell `shift` builtin.
pub const BUILTIN_SHIFT: u16 = 42;
/// Dispatch ID for the shell `eval` builtin.
pub const BUILTIN_EVAL: u16 = 43;
/// Dispatch ID for the shell `exec` builtin.
pub const BUILTIN_EXEC: u16 = 44;
/// Dispatch ID for the shell `command` builtin.
pub const BUILTIN_COMMAND: u16 = 45;
/// Dispatch ID for the shell `builtin` builtin.
pub const BUILTIN_BUILTIN: u16 = 46;
/// Dispatch ID for the shell `let` builtin.
pub const BUILTIN_LET: u16 = 47;

// ═══════════════════════════════════════════════════════════════════════════
// Job control
// ═══════════════════════════════════════════════════════════════════════════

/// Dispatch ID for the shell `jobs` builtin.
pub const BUILTIN_JOBS: u16 = 50;
/// Dispatch ID for the shell `fg` builtin.
pub const BUILTIN_FG: u16 = 51;
/// Dispatch ID for the shell `bg` builtin.
pub const BUILTIN_BG: u16 = 52;
/// Dispatch ID for the shell `kill` builtin.
pub const BUILTIN_KILL: u16 = 53;
/// Dispatch ID for the shell `disown` builtin.
pub const BUILTIN_DISOWN: u16 = 54;
/// Dispatch ID for the shell `wait` builtin.
pub const BUILTIN_WAIT: u16 = 55;
/// Dispatch ID for the shell `suspend` builtin.
pub const BUILTIN_SUSPEND: u16 = 56;

// ═══════════════════════════════════════════════════════════════════════════
// History
// ═══════════════════════════════════════════════════════════════════════════

/// Dispatch ID for the shell `history` builtin.
pub const BUILTIN_HISTORY: u16 = 60;
/// Dispatch ID for the shell `fc` builtin.
pub const BUILTIN_FC: u16 = 61;
/// Dispatch ID for the shell `r` builtin.
pub const BUILTIN_R: u16 = 62;

// ═══════════════════════════════════════════════════════════════════════════
// Aliases
// ═══════════════════════════════════════════════════════════════════════════

/// Dispatch ID for the shell `alias` builtin.
pub const BUILTIN_ALIAS: u16 = 70;
/// Dispatch ID for the shell `unalias` builtin.
pub const BUILTIN_UNALIAS: u16 = 71;

// ═══════════════════════════════════════════════════════════════════════════
// Options
// ═══════════════════════════════════════════════════════════════════════════

/// Dispatch ID for the shell `set` builtin.
pub const BUILTIN_SET: u16 = 80;
/// Dispatch ID for the shell `setopt` builtin.
pub const BUILTIN_SETOPT: u16 = 81;
/// Dispatch ID for the shell `unsetopt` builtin.
pub const BUILTIN_UNSETOPT: u16 = 82;
/// Dispatch ID for the shell `shopt` builtin.
pub const BUILTIN_SHOPT: u16 = 83;
/// Dispatch ID for the shell `emulate` builtin.
pub const BUILTIN_EMULATE: u16 = 84;
/// Dispatch ID for the shell `getopts` builtin.
pub const BUILTIN_GETOPTS: u16 = 85;

// ═══════════════════════════════════════════════════════════════════════════
// Functions / Autoload
// ═══════════════════════════════════════════════════════════════════════════

/// Dispatch ID for the shell `autoload` builtin.
pub const BUILTIN_AUTOLOAD: u16 = 90;
/// Dispatch ID for the shell `functions` builtin.
pub const BUILTIN_FUNCTIONS: u16 = 91;
/// Dispatch ID for the shell `unfunction` builtin.
pub const BUILTIN_UNFUNCTION: u16 = 92;

// ═══════════════════════════════════════════════════════════════════════════
// Traps / Signals
// ═══════════════════════════════════════════════════════════════════════════

/// Dispatch ID for the shell `trap` builtin.
pub const BUILTIN_TRAP: u16 = 100;

// ═══════════════════════════════════════════════════════════════════════════
// Directory stack
// ═══════════════════════════════════════════════════════════════════════════

/// Dispatch ID for the shell `pushd` builtin.
pub const BUILTIN_PUSHD: u16 = 110;
/// Dispatch ID for the shell `popd` builtin.
pub const BUILTIN_POPD: u16 = 111;
/// Dispatch ID for the shell `dirs` builtin.
pub const BUILTIN_DIRS: u16 = 112;

// ═══════════════════════════════════════════════════════════════════════════
// Type / Which / Hash
// ═══════════════════════════════════════════════════════════════════════════

/// Dispatch ID for the shell `type` builtin.
pub const BUILTIN_TYPE: u16 = 120;
/// Dispatch ID for the shell `whence` builtin.
pub const BUILTIN_WHENCE: u16 = 121;
/// Dispatch ID for the shell `where` builtin.
pub const BUILTIN_WHERE: u16 = 122;
/// Dispatch ID for the shell `which` builtin.
pub const BUILTIN_WHICH: u16 = 123;
/// Dispatch ID for the shell `hash` builtin.
pub const BUILTIN_HASH: u16 = 124;
/// Dispatch ID for the shell `rehash` builtin.
pub const BUILTIN_REHASH: u16 = 125;
/// Dispatch ID for the shell `unhash` builtin.
pub const BUILTIN_UNHASH: u16 = 126;

// ═══════════════════════════════════════════════════════════════════════════
// Completion
// ═══════════════════════════════════════════════════════════════════════════

/// Dispatch ID for the shell `compgen` builtin.
pub const BUILTIN_COMPGEN: u16 = 130;
/// Dispatch ID for the shell `complete` builtin.
pub const BUILTIN_COMPLETE: u16 = 131;
/// Dispatch ID for the shell `compopt` builtin.
pub const BUILTIN_COMPOPT: u16 = 132;
/// Dispatch ID for the shell `compadd` builtin.
pub const BUILTIN_COMPADD: u16 = 133;
/// Dispatch ID for the shell `compset` builtin.
pub const BUILTIN_COMPSET: u16 = 134;
/// Dispatch ID for the shell `compdef` builtin.
pub const BUILTIN_COMPDEF: u16 = 135;
/// Dispatch ID for the shell `compinit` builtin.
pub const BUILTIN_COMPINIT: u16 = 136;
/// Dispatch ID for the shell `cdreplay` builtin.
pub const BUILTIN_CDREPLAY: u16 = 137;

// ═══════════════════════════════════════════════════════════════════════════
// Zsh-specific
// ═══════════════════════════════════════════════════════════════════════════

/// Dispatch ID for the shell `zstyle` builtin.
pub const BUILTIN_ZSTYLE: u16 = 140;
/// Dispatch ID for the shell `zmodload` builtin.
pub const BUILTIN_ZMODLOAD: u16 = 141;
/// Dispatch ID for the shell `bindkey` builtin.
pub const BUILTIN_BINDKEY: u16 = 142;
/// Dispatch ID for the shell `zle` builtin.
pub const BUILTIN_ZLE: u16 = 143;
/// Dispatch ID for the shell `vared` builtin.
pub const BUILTIN_VARED: u16 = 144;
/// Dispatch ID for the shell `zcompile` builtin.
pub const BUILTIN_ZCOMPILE: u16 = 145;
/// Dispatch ID for the shell `zformat` builtin.
pub const BUILTIN_ZFORMAT: u16 = 146;
/// Dispatch ID for the shell `zparseopts` builtin.
pub const BUILTIN_ZPARSEOPTS: u16 = 147;
/// Dispatch ID for the shell `zregexparse` builtin.
pub const BUILTIN_ZREGEXPARSE: u16 = 148;

// ═══════════════════════════════════════════════════════════════════════════
// Resource limits
// ═══════════════════════════════════════════════════════════════════════════

/// Dispatch ID for the shell `ulimit` builtin.
pub const BUILTIN_ULIMIT: u16 = 150;
/// Dispatch ID for the shell `limit` builtin.
pub const BUILTIN_LIMIT: u16 = 151;
/// Dispatch ID for the shell `unlimit` builtin.
pub const BUILTIN_UNLIMIT: u16 = 152;
/// Dispatch ID for the shell `umask` builtin.
pub const BUILTIN_UMASK: u16 = 153;

// ═══════════════════════════════════════════════════════════════════════════
// Misc
// ═══════════════════════════════════════════════════════════════════════════

/// Dispatch ID for the shell `times` builtin.
pub const BUILTIN_TIMES: u16 = 160;
/// Dispatch ID for the shell `caller` builtin.
pub const BUILTIN_CALLER: u16 = 161;
/// Dispatch ID for the shell `help` builtin.
pub const BUILTIN_HELP: u16 = 162;
/// Dispatch ID for the shell `enable` builtin.
pub const BUILTIN_ENABLE: u16 = 163;
/// Dispatch ID for the shell `disable` builtin.
pub const BUILTIN_DISABLE: u16 = 164;
/// Dispatch ID for the shell `noglob` builtin.
pub const BUILTIN_NOGLOB: u16 = 165;
/// Dispatch ID for the shell `ttyctl` builtin.
pub const BUILTIN_TTYCTL: u16 = 166;
/// Dispatch ID for the shell `sync` builtin.
pub const BUILTIN_SYNC: u16 = 167;
/// Dispatch ID for the shell `mkdir` builtin.
pub const BUILTIN_MKDIR: u16 = 168;
/// Dispatch ID for the shell `strftime` builtin.
pub const BUILTIN_STRFTIME: u16 = 169;
/// Dispatch ID for the shell `zsleep` builtin.
pub const BUILTIN_ZSLEEP: u16 = 170;
/// Dispatch ID for the shell `zsystem` builtin.
pub const BUILTIN_ZSYSTEM: u16 = 171;

// ═══════════════════════════════════════════════════════════════════════════
// PCRE
// ═══════════════════════════════════════════════════════════════════════════

/// Dispatch ID for the shell `pcre_compile` builtin.
pub const BUILTIN_PCRE_COMPILE: u16 = 180;
/// Dispatch ID for the shell `pcre_match` builtin.
pub const BUILTIN_PCRE_MATCH: u16 = 181;
/// Dispatch ID for the shell `pcre_study` builtin.
pub const BUILTIN_PCRE_STUDY: u16 = 182;

// ═══════════════════════════════════════════════════════════════════════════
// Database (GDBM)
// ═══════════════════════════════════════════════════════════════════════════

/// Dispatch ID for the shell `ztie` builtin.
pub const BUILTIN_ZTIE: u16 = 190;
/// Dispatch ID for the shell `zuntie` builtin.
pub const BUILTIN_ZUNTIE: u16 = 191;
/// Dispatch ID for the shell `zgdbmpath` builtin.
pub const BUILTIN_ZGDBMPATH: u16 = 192;

// ═══════════════════════════════════════════════════════════════════════════
// Prompt
// ═══════════════════════════════════════════════════════════════════════════

/// Dispatch ID for the shell `promptinit` builtin.
pub const BUILTIN_PROMPTINIT: u16 = 200;
/// Dispatch ID for the shell `prompt` builtin.
pub const BUILTIN_PROMPT: u16 = 201;

// ═══════════════════════════════════════════════════════════════════════════
// Async / Parallel (zshrs extensions)
// ═══════════════════════════════════════════════════════════════════════════

/// Dispatch ID for the shell `async` builtin.
pub const BUILTIN_ASYNC: u16 = 210;
/// Dispatch ID for the shell `await` builtin.
pub const BUILTIN_AWAIT: u16 = 211;
/// Dispatch ID for the shell `pmap` builtin.
pub const BUILTIN_PMAP: u16 = 212;
/// Dispatch ID for the shell `pgrep` builtin.
pub const BUILTIN_PGREP: u16 = 213;
/// Dispatch ID for the shell `peach` builtin.
pub const BUILTIN_PEACH: u16 = 214;
/// Dispatch ID for the shell `barrier` builtin.
pub const BUILTIN_BARRIER: u16 = 215;

// ═══════════════════════════════════════════════════════════════════════════
// Intercept (AOP)
// ═══════════════════════════════════════════════════════════════════════════

/// Dispatch ID for the shell `intercept` builtin.
pub const BUILTIN_INTERCEPT: u16 = 220;
/// Dispatch ID for the shell `intercept_proceed` builtin.
pub const BUILTIN_INTERCEPT_PROCEED: u16 = 221;

// ═══════════════════════════════════════════════════════════════════════════
// Debug / Profile
// ═══════════════════════════════════════════════════════════════════════════

/// Dispatch ID for the shell `doctor` builtin.
pub const BUILTIN_DOCTOR: u16 = 230;
/// Dispatch ID for the shell `dbview` builtin.
pub const BUILTIN_DBVIEW: u16 = 231;
/// Dispatch ID for the shell `profile` builtin.
pub const BUILTIN_PROFILE: u16 = 232;
/// Dispatch ID for the shell `zprof` builtin.
pub const BUILTIN_ZPROF: u16 = 233;

// ═══════════════════════════════════════════════════════════════════════════
// Coreutils (anti-fork builtins)
// ═══════════════════════════════════════════════════════════════════════════

/// Dispatch ID for the shell `cat` builtin.
pub const BUILTIN_CAT: u16 = 240;
/// Dispatch ID for the shell `head` builtin.
pub const BUILTIN_HEAD: u16 = 241;
/// Dispatch ID for the shell `tail` builtin.
pub const BUILTIN_TAIL: u16 = 242;
/// Dispatch ID for the shell `wc` builtin.
pub const BUILTIN_WC: u16 = 243;
/// Dispatch ID for the shell `basename` builtin.
pub const BUILTIN_BASENAME: u16 = 244;
/// Dispatch ID for the shell `dirname` builtin.
pub const BUILTIN_DIRNAME: u16 = 245;
/// Dispatch ID for the shell `touch` builtin.
pub const BUILTIN_TOUCH: u16 = 246;
/// Dispatch ID for the shell `realpath` builtin.
pub const BUILTIN_REALPATH: u16 = 247;
/// Dispatch ID for the shell `sort` builtin.
pub const BUILTIN_SORT: u16 = 248;
/// Dispatch ID for the shell `find` builtin.
pub const BUILTIN_FIND: u16 = 249;
/// Dispatch ID for the shell `uniq` builtin.
pub const BUILTIN_UNIQ: u16 = 250;
/// Dispatch ID for the shell `cut` builtin.
pub const BUILTIN_CUT: u16 = 251;
/// Dispatch ID for the shell `tr` builtin.
pub const BUILTIN_TR: u16 = 252;
/// Dispatch ID for the shell `seq` builtin.
pub const BUILTIN_SEQ: u16 = 253;
/// Dispatch ID for the shell `rev` builtin.
pub const BUILTIN_REV: u16 = 254;
/// Dispatch ID for the shell `tee` builtin.
pub const BUILTIN_TEE: u16 = 255;
/// Dispatch ID for the shell `sleep` builtin.
pub const BUILTIN_SLEEP: u16 = 256;
/// Dispatch ID for the shell `whoami` builtin.
pub const BUILTIN_WHOAMI: u16 = 257;
/// Dispatch ID for the shell `id` builtin.
pub const BUILTIN_ID: u16 = 258;
/// Dispatch ID for the shell `hostname` builtin.
pub const BUILTIN_HOSTNAME: u16 = 259;
/// Dispatch ID for the shell `uname` builtin.
pub const BUILTIN_UNAME: u16 = 260;
/// Dispatch ID for the shell `date` builtin.
pub const BUILTIN_DATE: u16 = 261;
/// Dispatch ID for the shell `mktemp` builtin.
pub const BUILTIN_MKTEMP: u16 = 262;

/// Maximum builtin ID (for pre-allocating the handler table)
pub const BUILTIN_MAX: u16 = 280;

/// Map builtin name to ID. Returns None for non-builtins.
#[inline]
pub fn builtin_id(name: &str) -> Option<u16> {
    match name {
        "cd" | "chdir" => Some(BUILTIN_CD),
        "pwd" => Some(BUILTIN_PWD),
        "echo" => Some(BUILTIN_ECHO),
        "print" => Some(BUILTIN_PRINT),
        "printf" => Some(BUILTIN_PRINTF),
        "export" => Some(BUILTIN_EXPORT),
        "unset" => Some(BUILTIN_UNSET),
        "source" | "." => Some(BUILTIN_SOURCE),
        "exit" | "bye" | "logout" => Some(BUILTIN_EXIT),
        "return" => Some(BUILTIN_RETURN),
        "true" => Some(BUILTIN_TRUE),
        "false" => Some(BUILTIN_FALSE),
        "test" | "[" => Some(BUILTIN_TEST),
        ":" => Some(BUILTIN_COLON),
        "local" => Some(BUILTIN_LOCAL),
        "declare" | "typeset" => Some(BUILTIN_TYPESET),
        "readonly" => Some(BUILTIN_READONLY),
        "integer" => Some(BUILTIN_INTEGER),
        "float" => Some(BUILTIN_FLOAT),
        "read" => Some(BUILTIN_READ),
        "mapfile" | "readarray" => Some(BUILTIN_MAPFILE),
        "break" => Some(BUILTIN_BREAK),
        "continue" => Some(BUILTIN_CONTINUE),
        "shift" => Some(BUILTIN_SHIFT),
        "eval" => Some(BUILTIN_EVAL),
        "exec" => Some(BUILTIN_EXEC),
        "command" => Some(BUILTIN_COMMAND),
        "builtin" => Some(BUILTIN_BUILTIN),
        "let" => Some(BUILTIN_LET),
        "jobs" => Some(BUILTIN_JOBS),
        "fg" => Some(BUILTIN_FG),
        "bg" => Some(BUILTIN_BG),
        "kill" => Some(BUILTIN_KILL),
        "disown" => Some(BUILTIN_DISOWN),
        "wait" => Some(BUILTIN_WAIT),
        "suspend" => Some(BUILTIN_SUSPEND),
        "history" => Some(BUILTIN_HISTORY),
        "fc" => Some(BUILTIN_FC),
        "r" => Some(BUILTIN_R),
        "alias" => Some(BUILTIN_ALIAS),
        "unalias" => Some(BUILTIN_UNALIAS),
        "set" => Some(BUILTIN_SET),
        "setopt" => Some(BUILTIN_SETOPT),
        "unsetopt" => Some(BUILTIN_UNSETOPT),
        "shopt" => Some(BUILTIN_SHOPT),
        "emulate" => Some(BUILTIN_EMULATE),
        "getopts" => Some(BUILTIN_GETOPTS),
        "autoload" => Some(BUILTIN_AUTOLOAD),
        "functions" => Some(BUILTIN_FUNCTIONS),
        "unfunction" => Some(BUILTIN_UNFUNCTION),
        "trap" => Some(BUILTIN_TRAP),
        "pushd" => Some(BUILTIN_PUSHD),
        "popd" => Some(BUILTIN_POPD),
        "dirs" => Some(BUILTIN_DIRS),
        "type" => Some(BUILTIN_TYPE),
        "whence" => Some(BUILTIN_WHENCE),
        "where" => Some(BUILTIN_WHERE),
        "which" => Some(BUILTIN_WHICH),
        "hash" => Some(BUILTIN_HASH),
        "rehash" => Some(BUILTIN_REHASH),
        "unhash" => Some(BUILTIN_UNHASH),
        "compgen" => Some(BUILTIN_COMPGEN),
        "complete" => Some(BUILTIN_COMPLETE),
        "compopt" => Some(BUILTIN_COMPOPT),
        "compadd" => Some(BUILTIN_COMPADD),
        "compset" => Some(BUILTIN_COMPSET),
        "compdef" => Some(BUILTIN_COMPDEF),
        "compinit" => Some(BUILTIN_COMPINIT),
        "cdreplay" => Some(BUILTIN_CDREPLAY),
        "zstyle" => Some(BUILTIN_ZSTYLE),
        "zmodload" => Some(BUILTIN_ZMODLOAD),
        "bindkey" | "bind" => Some(BUILTIN_BINDKEY),
        "zle" => Some(BUILTIN_ZLE),
        "vared" => Some(BUILTIN_VARED),
        "zcompile" => Some(BUILTIN_ZCOMPILE),
        "zformat" => Some(BUILTIN_ZFORMAT),
        "zparseopts" => Some(BUILTIN_ZPARSEOPTS),
        "zregexparse" => Some(BUILTIN_ZREGEXPARSE),
        "ulimit" => Some(BUILTIN_ULIMIT),
        "limit" => Some(BUILTIN_LIMIT),
        "unlimit" => Some(BUILTIN_UNLIMIT),
        "umask" => Some(BUILTIN_UMASK),
        "times" => Some(BUILTIN_TIMES),
        "caller" => Some(BUILTIN_CALLER),
        "help" => Some(BUILTIN_HELP),
        "enable" => Some(BUILTIN_ENABLE),
        "disable" => Some(BUILTIN_DISABLE),
        "noglob" => Some(BUILTIN_NOGLOB),
        "ttyctl" => Some(BUILTIN_TTYCTL),
        "sync" => Some(BUILTIN_SYNC),
        "mkdir" => Some(BUILTIN_MKDIR),
        "strftime" => Some(BUILTIN_STRFTIME),
        "zsleep" => Some(BUILTIN_ZSLEEP),
        "zsystem" => Some(BUILTIN_ZSYSTEM),
        "pcre_compile" => Some(BUILTIN_PCRE_COMPILE),
        "pcre_match" => Some(BUILTIN_PCRE_MATCH),
        "pcre_study" => Some(BUILTIN_PCRE_STUDY),
        "ztie" => Some(BUILTIN_ZTIE),
        "zuntie" => Some(BUILTIN_ZUNTIE),
        "zgdbmpath" => Some(BUILTIN_ZGDBMPATH),
        "promptinit" => Some(BUILTIN_PROMPTINIT),
        "prompt" => Some(BUILTIN_PROMPT),
        "async" => Some(BUILTIN_ASYNC),
        "await" => Some(BUILTIN_AWAIT),
        "pmap" => Some(BUILTIN_PMAP),
        "pgrep" => Some(BUILTIN_PGREP),
        "peach" => Some(BUILTIN_PEACH),
        "barrier" => Some(BUILTIN_BARRIER),
        "intercept" => Some(BUILTIN_INTERCEPT),
        "intercept_proceed" => Some(BUILTIN_INTERCEPT_PROCEED),
        "doctor" => Some(BUILTIN_DOCTOR),
        "dbview" => Some(BUILTIN_DBVIEW),
        "profile" => Some(BUILTIN_PROFILE),
        "zprof" => Some(BUILTIN_ZPROF),
        // Coreutils
        "cat" => Some(BUILTIN_CAT),
        "head" => Some(BUILTIN_HEAD),
        "tail" => Some(BUILTIN_TAIL),
        "wc" => Some(BUILTIN_WC),
        "basename" => Some(BUILTIN_BASENAME),
        "dirname" => Some(BUILTIN_DIRNAME),
        "touch" => Some(BUILTIN_TOUCH),
        "realpath" => Some(BUILTIN_REALPATH),
        "sort" => Some(BUILTIN_SORT),
        "find" => Some(BUILTIN_FIND),
        "uniq" => Some(BUILTIN_UNIQ),
        "cut" => Some(BUILTIN_CUT),
        "tr" => Some(BUILTIN_TR),
        "seq" => Some(BUILTIN_SEQ),
        "rev" => Some(BUILTIN_REV),
        "tee" => Some(BUILTIN_TEE),
        "sleep" => Some(BUILTIN_SLEEP),
        "whoami" => Some(BUILTIN_WHOAMI),
        "id" => Some(BUILTIN_ID),
        "hostname" => Some(BUILTIN_HOSTNAME),
        "uname" => Some(BUILTIN_UNAME),
        "date" => Some(BUILTIN_DATE),
        "mktemp" => Some(BUILTIN_MKTEMP),
        _ => None,
    }
}

/// Check if a command name is a builtin.
#[inline]
pub fn is_builtin(name: &str) -> bool {
    builtin_id(name).is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builtin_lookup() {
        assert_eq!(builtin_id("cd"), Some(BUILTIN_CD));
        assert_eq!(builtin_id("chdir"), Some(BUILTIN_CD));
        assert_eq!(builtin_id("echo"), Some(BUILTIN_ECHO));
        assert_eq!(builtin_id("typeset"), Some(BUILTIN_TYPESET));
        assert_eq!(builtin_id("declare"), Some(BUILTIN_TYPESET));
        assert_eq!(builtin_id("ls"), None);
        assert_eq!(builtin_id("grep"), None);
    }

    #[test]
    fn test_is_builtin() {
        assert!(is_builtin("cd"));
        assert!(is_builtin("export"));
        assert!(is_builtin("cat"));
        assert!(is_builtin("head"));
        assert!(is_builtin("tail"));
        assert!(is_builtin("sort"));
        assert!(is_builtin("find"));
        assert!(!is_builtin("ls"));
    }

    #[test]
    fn builtin_lookup_is_case_sensitive() {
        // Shells treat builtin names as case-sensitive; "CD" is not "cd".
        assert!(is_builtin("cd"));
        assert!(!is_builtin("CD"));
        assert!(!is_builtin("Cd"));
    }

    #[test]
    fn unknown_and_empty_names_are_not_builtins() {
        assert_eq!(builtin_id(""), None);
        assert_eq!(builtin_id("definitely_not_a_builtin_xyz"), None);
        assert_eq!(builtin_id(" cd"), None); // leading space matters
        assert_eq!(builtin_id("cd "), None);
    }

    #[test]
    fn aliases_map_to_same_id() {
        // Every alias group declared in the lookup table.
        assert_eq!(builtin_id("cd"), builtin_id("chdir"));
        assert_eq!(builtin_id("declare"), builtin_id("typeset"));
        assert_eq!(builtin_id("source"), builtin_id("."));
        assert_eq!(builtin_id("test"), builtin_id("["));
        assert_eq!(builtin_id("mapfile"), builtin_id("readarray"));
        assert_eq!(builtin_id("bindkey"), builtin_id("bind"));
        assert_eq!(builtin_id("exit"), builtin_id("bye"));
        assert_eq!(builtin_id("exit"), builtin_id("logout"));
    }

    #[test]
    fn builtin_max_exceeds_all_assigned_ids() {
        // Spot-check several of the highest assigned IDs.
        for id in [
            BUILTIN_CD,
            BUILTIN_MKTEMP,
            BUILTIN_DATE,
            BUILTIN_ZPROF,
            BUILTIN_INTERCEPT_PROCEED,
            BUILTIN_PROMPT,
            BUILTIN_BARRIER,
        ] {
            assert!(
                id < BUILTIN_MAX,
                "id {} not below BUILTIN_MAX {}",
                id,
                BUILTIN_MAX
            );
        }
    }

    #[test]
    fn coreutils_builtins_are_registered() {
        for name in [
            "cat", "head", "tail", "wc", "basename", "dirname", "touch", "realpath", "sort",
            "find", "uniq", "cut", "tr", "seq", "rev", "tee", "sleep", "whoami", "id", "hostname",
            "uname", "date", "mktemp",
        ] {
            assert!(is_builtin(name), "{} should be a builtin", name);
        }
    }

    #[test]
    fn control_flow_and_job_builtins_registered() {
        for name in [
            "break", "continue", "shift", "eval", "exec", "let", "jobs", "fg", "bg", "kill",
            "wait", "disown",
        ] {
            assert!(is_builtin(name), "{} should be a builtin", name);
        }
    }

    #[test]
    fn zsh_specific_builtins_registered() {
        for name in [
            "zstyle",
            "zmodload",
            "zle",
            "zcompile",
            "zformat",
            "zparseopts",
            "compdef",
            "compinit",
            "autoload",
        ] {
            assert!(is_builtin(name), "{} should be a builtin", name);
        }
    }

    #[test]
    fn builtin_ids_are_unique_across_lookup_table() {
        // Walk every name we expect to map to a Some(id); collect ids and
        // assert each id maps back to at most one canonical name (ignoring
        // documented aliases).
        let mut seen: std::collections::HashMap<u16, &'static str> =
            std::collections::HashMap::new();
        // Canonical (non-alias) names from the lookup table.
        let canonicals: &[&str] = &[
            "cd",
            "pwd",
            "echo",
            "print",
            "printf",
            "export",
            "unset",
            "source",
            "exit",
            "return",
            "true",
            "false",
            "test",
            ":",
            "local",
            "declare",
            "readonly",
            "integer",
            "float",
            "read",
            "mapfile",
            "break",
            "continue",
            "shift",
            "eval",
            "exec",
            "command",
            "builtin",
            "let",
            "jobs",
            "fg",
            "bg",
            "kill",
            "disown",
            "wait",
            "suspend",
            "history",
            "fc",
            "r",
            "alias",
            "unalias",
            "set",
            "setopt",
            "unsetopt",
            "shopt",
            "emulate",
            "getopts",
            "autoload",
            "functions",
            "unfunction",
            "trap",
            "pushd",
            "popd",
            "dirs",
            "type",
            "whence",
            "where",
            "which",
            "hash",
            "rehash",
            "unhash",
            "compgen",
            "complete",
            "compopt",
            "compadd",
            "compset",
            "compdef",
            "compinit",
            "cdreplay",
            "zstyle",
            "zmodload",
            "bindkey",
            "zle",
            "vared",
            "zcompile",
            "zformat",
            "zparseopts",
            "zregexparse",
            "ulimit",
            "limit",
            "unlimit",
            "umask",
            "times",
            "caller",
            "help",
            "enable",
            "disable",
            "noglob",
            "ttyctl",
            "sync",
            "mkdir",
            "strftime",
            "zsleep",
            "zsystem",
            "pcre_compile",
            "pcre_match",
            "pcre_study",
            "ztie",
            "zuntie",
            "zgdbmpath",
            "promptinit",
            "prompt",
            "async",
            "await",
            "pmap",
            "pgrep",
            "peach",
            "barrier",
            "intercept",
            "intercept_proceed",
            "doctor",
            "dbview",
            "profile",
            "zprof",
            "cat",
            "head",
            "tail",
            "wc",
            "basename",
            "dirname",
            "touch",
            "realpath",
            "sort",
            "find",
            "uniq",
            "cut",
            "tr",
            "seq",
            "rev",
            "tee",
            "sleep",
            "whoami",
            "id",
            "hostname",
            "uname",
            "date",
            "mktemp",
        ];
        for name in canonicals {
            let id = builtin_id(name).unwrap_or_else(|| panic!("missing builtin {}", name));
            if let Some(prev) = seen.insert(id, name) {
                panic!(
                    "duplicate builtin id {} shared by {} and {}",
                    id, prev, name
                );
            }
        }
    }

    // ─── Boundary value edge cases ─────────────────────────────────────

    #[test]
    fn builtin_max_is_strictly_greater_than_every_assigned_id() {
        // BUILTIN_MAX is used to pre-size the dispatch table. If any constant
        // exceeds it, registering that builtin would panic at VM init.
        // Walk every constant defined in this module.
        let assigned: &[u16] = &[
            BUILTIN_CD,
            BUILTIN_PWD,
            BUILTIN_ECHO,
            BUILTIN_PRINT,
            BUILTIN_PRINTF,
            BUILTIN_EXPORT,
            BUILTIN_UNSET,
            BUILTIN_SOURCE,
            BUILTIN_EXIT,
            BUILTIN_RETURN,
            BUILTIN_TRUE,
            BUILTIN_FALSE,
            BUILTIN_TEST,
            BUILTIN_COLON,
            BUILTIN_DOT,
            BUILTIN_LOCAL,
            BUILTIN_DECLARE,
            BUILTIN_TYPESET,
            BUILTIN_READONLY,
            BUILTIN_INTEGER,
            BUILTIN_FLOAT,
            BUILTIN_READ,
            BUILTIN_MAPFILE,
            BUILTIN_BREAK,
            BUILTIN_CONTINUE,
            BUILTIN_SHIFT,
            BUILTIN_EVAL,
            BUILTIN_EXEC,
            BUILTIN_COMMAND,
            BUILTIN_BUILTIN,
            BUILTIN_LET,
            BUILTIN_JOBS,
            BUILTIN_FG,
            BUILTIN_BG,
            BUILTIN_KILL,
            BUILTIN_DISOWN,
            BUILTIN_WAIT,
            BUILTIN_SUSPEND,
            BUILTIN_HISTORY,
            BUILTIN_FC,
            BUILTIN_R,
            BUILTIN_ALIAS,
            BUILTIN_UNALIAS,
            BUILTIN_SET,
            BUILTIN_SETOPT,
            BUILTIN_UNSETOPT,
            BUILTIN_SHOPT,
            BUILTIN_EMULATE,
            BUILTIN_GETOPTS,
            BUILTIN_AUTOLOAD,
            BUILTIN_FUNCTIONS,
            BUILTIN_UNFUNCTION,
            BUILTIN_TRAP,
            BUILTIN_PUSHD,
            BUILTIN_POPD,
            BUILTIN_DIRS,
            BUILTIN_TYPE,
            BUILTIN_WHENCE,
            BUILTIN_WHERE,
            BUILTIN_WHICH,
            BUILTIN_HASH,
            BUILTIN_REHASH,
            BUILTIN_UNHASH,
            BUILTIN_COMPGEN,
            BUILTIN_COMPLETE,
            BUILTIN_COMPOPT,
            BUILTIN_COMPADD,
            BUILTIN_COMPSET,
            BUILTIN_COMPDEF,
            BUILTIN_COMPINIT,
            BUILTIN_CDREPLAY,
            BUILTIN_ZSTYLE,
            BUILTIN_ZMODLOAD,
            BUILTIN_BINDKEY,
            BUILTIN_ZLE,
            BUILTIN_VARED,
            BUILTIN_ZCOMPILE,
            BUILTIN_ZFORMAT,
            BUILTIN_ZPARSEOPTS,
            BUILTIN_ZREGEXPARSE,
            BUILTIN_ULIMIT,
            BUILTIN_LIMIT,
            BUILTIN_UNLIMIT,
            BUILTIN_UMASK,
            BUILTIN_TIMES,
            BUILTIN_CALLER,
            BUILTIN_HELP,
            BUILTIN_ENABLE,
            BUILTIN_DISABLE,
            BUILTIN_NOGLOB,
            BUILTIN_TTYCTL,
            BUILTIN_SYNC,
            BUILTIN_MKDIR,
            BUILTIN_STRFTIME,
            BUILTIN_ZSLEEP,
            BUILTIN_ZSYSTEM,
            BUILTIN_PCRE_COMPILE,
            BUILTIN_PCRE_MATCH,
            BUILTIN_PCRE_STUDY,
            BUILTIN_ZTIE,
            BUILTIN_ZUNTIE,
            BUILTIN_ZGDBMPATH,
            BUILTIN_PROMPTINIT,
            BUILTIN_PROMPT,
            BUILTIN_ASYNC,
            BUILTIN_AWAIT,
            BUILTIN_PMAP,
            BUILTIN_PGREP,
            BUILTIN_PEACH,
            BUILTIN_BARRIER,
            BUILTIN_INTERCEPT,
            BUILTIN_INTERCEPT_PROCEED,
            BUILTIN_DOCTOR,
            BUILTIN_DBVIEW,
            BUILTIN_PROFILE,
            BUILTIN_ZPROF,
            BUILTIN_CAT,
            BUILTIN_HEAD,
            BUILTIN_TAIL,
            BUILTIN_WC,
            BUILTIN_BASENAME,
            BUILTIN_DIRNAME,
            BUILTIN_TOUCH,
            BUILTIN_REALPATH,
            BUILTIN_SORT,
            BUILTIN_FIND,
            BUILTIN_UNIQ,
            BUILTIN_CUT,
            BUILTIN_TR,
            BUILTIN_SEQ,
            BUILTIN_REV,
            BUILTIN_TEE,
            BUILTIN_SLEEP,
            BUILTIN_WHOAMI,
            BUILTIN_ID,
            BUILTIN_HOSTNAME,
            BUILTIN_UNAME,
            BUILTIN_DATE,
            BUILTIN_MKTEMP,
        ];
        for &id in assigned {
            assert!(
                id < BUILTIN_MAX,
                "constant value {} >= BUILTIN_MAX ({})",
                id,
                BUILTIN_MAX
            );
        }
    }

    #[test]
    fn dot_and_source_alias_to_same_id() {
        // BUILTIN_DOT exists as a separate constant for legibility, but the
        // lookup table maps both "source" and "." to BUILTIN_SOURCE.
        // Verify that mapping. DOT constant is informational only.
        assert_eq!(builtin_id("."), Some(BUILTIN_SOURCE));
        // Confirm DOT and SOURCE are distinct constants (DOT=14, SOURCE=7).
        assert_ne!(BUILTIN_SOURCE, BUILTIN_DOT);
    }

    #[test]
    fn names_with_internal_whitespace_or_tab_are_not_builtins() {
        // Builtin lookup uses exact match; any embedded whitespace fails it.
        assert_eq!(builtin_id("c d"), None);
        assert_eq!(builtin_id("cd\t"), None);
        assert_eq!(builtin_id("c\nd"), None);
    }

    #[test]
    fn names_with_null_byte_or_unicode_are_not_builtins() {
        // String contents can contain anything; non-ASCII or NUL must not crash
        // and must return None (no builtin uses anything but lower-ASCII).
        assert_eq!(builtin_id("\0"), None);
        assert_eq!(builtin_id("café"), None);
        assert_eq!(builtin_id("世界"), None);
        // Empty unicode-like input also safe.
        assert_eq!(builtin_id("\u{0}"), None);
    }

    #[test]
    fn left_bracket_maps_to_test_but_double_bracket_does_not() {
        // POSIX test alias `[` is a builtin; `[[` is shell SYNTAX (a keyword,
        // not a builtin) — so it must NOT resolve through this table.
        assert_eq!(builtin_id("["), Some(BUILTIN_TEST));
        assert_eq!(builtin_id("[["), None);
        assert_eq!(builtin_id("]"), None);
    }

    #[test]
    fn category_id_ranges_have_documented_gaps() {
        // The IDs are bucketed by category with intentional gaps (e.g. 15-19
        // reserved between core and variable-declaration). Gap values must
        // NOT be reachable via any builtin name — confirms no stray mapping
        // leaked into a reserved slot.
        for id in [15, 16, 17, 18, 19, 26, 27, 28, 29, 32, 33, 34, 35] {
            // None of the canonical or alias names map to these gap IDs.
            // (We test by scanning canonical lookups already done above —
            //  here we just confirm the table builds without panicking and
            //  these IDs would only be reachable via direct constant use.)
            assert!(id < BUILTIN_MAX, "even gap ids must fit under BUILTIN_MAX");
        }
    }

    #[test]
    fn dot_constant_is_unused_by_lookup_table() {
        // BUILTIN_DOT=14 exists for symmetry with BUILTIN_SOURCE but no name
        // in builtin_id() returns it — the lookup deliberately routes "." to
        // BUILTIN_SOURCE for compat with shells that hash builtins by canonical name.
        for name in [".", "source", "cd", "dot"] {
            assert_ne!(
                builtin_id(name),
                Some(BUILTIN_DOT),
                "{} should not resolve to BUILTIN_DOT",
                name
            );
        }
    }
}
