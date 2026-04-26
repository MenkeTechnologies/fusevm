//! Shell builtin IDs for `CallBuiltin(id, argc)` dispatch.
//!
//! These IDs are used by shell frontends (zshrs) to emit bytecodes
//! that call registered builtin handlers. The VM dispatches through
//! the pre-registered function pointer table — no name lookup at runtime.
//!
//! Usage in frontend compiler:
//! ```ignore
//! use fusevm::shell_builtins::*;
//! builder.emit(Op::CallBuiltin(BUILTIN_CD, 1), line);
//! ```
//!
//! Usage in frontend VM init:
//! ```ignore
//! vm.register_builtin(BUILTIN_CD, |vm, argc| { ... });
//! ```

// ═══════════════════════════════════════════════════════════════════════════
// Core builtins (POSIX + common)
// ═══════════════════════════════════════════════════════════════════════════

pub const BUILTIN_CD: u16 = 0;
pub const BUILTIN_PWD: u16 = 1;
pub const BUILTIN_ECHO: u16 = 2;
pub const BUILTIN_PRINT: u16 = 3;
pub const BUILTIN_PRINTF: u16 = 4;
pub const BUILTIN_EXPORT: u16 = 5;
pub const BUILTIN_UNSET: u16 = 6;
pub const BUILTIN_SOURCE: u16 = 7;
pub const BUILTIN_EXIT: u16 = 8;
pub const BUILTIN_RETURN: u16 = 9;
pub const BUILTIN_TRUE: u16 = 10;
pub const BUILTIN_FALSE: u16 = 11;
pub const BUILTIN_TEST: u16 = 12;
pub const BUILTIN_COLON: u16 = 13; // :
pub const BUILTIN_DOT: u16 = 14; // . (alias for source)

// ═══════════════════════════════════════════════════════════════════════════
// Variable declaration
// ═══════════════════════════════════════════════════════════════════════════

pub const BUILTIN_LOCAL: u16 = 20;
pub const BUILTIN_DECLARE: u16 = 21;
pub const BUILTIN_TYPESET: u16 = 22;
pub const BUILTIN_READONLY: u16 = 23;
pub const BUILTIN_INTEGER: u16 = 24;
pub const BUILTIN_FLOAT: u16 = 25;

// ═══════════════════════════════════════════════════════════════════════════
// I/O
// ═══════════════════════════════════════════════════════════════════════════

pub const BUILTIN_READ: u16 = 30;
pub const BUILTIN_MAPFILE: u16 = 31;

// ═══════════════════════════════════════════════════════════════════════════
// Control flow
// ═══════════════════════════════════════════════════════════════════════════

pub const BUILTIN_BREAK: u16 = 40;
pub const BUILTIN_CONTINUE: u16 = 41;
pub const BUILTIN_SHIFT: u16 = 42;
pub const BUILTIN_EVAL: u16 = 43;
pub const BUILTIN_EXEC: u16 = 44;
pub const BUILTIN_COMMAND: u16 = 45;
pub const BUILTIN_BUILTIN: u16 = 46;
pub const BUILTIN_LET: u16 = 47;

// ═══════════════════════════════════════════════════════════════════════════
// Job control
// ═══════════════════════════════════════════════════════════════════════════

pub const BUILTIN_JOBS: u16 = 50;
pub const BUILTIN_FG: u16 = 51;
pub const BUILTIN_BG: u16 = 52;
pub const BUILTIN_KILL: u16 = 53;
pub const BUILTIN_DISOWN: u16 = 54;
pub const BUILTIN_WAIT: u16 = 55;
pub const BUILTIN_SUSPEND: u16 = 56;

// ═══════════════════════════════════════════════════════════════════════════
// History
// ═══════════════════════════════════════════════════════════════════════════

pub const BUILTIN_HISTORY: u16 = 60;
pub const BUILTIN_FC: u16 = 61;
pub const BUILTIN_R: u16 = 62;

// ═══════════════════════════════════════════════════════════════════════════
// Aliases
// ═══════════════════════════════════════════════════════════════════════════

pub const BUILTIN_ALIAS: u16 = 70;
pub const BUILTIN_UNALIAS: u16 = 71;

// ═══════════════════════════════════════════════════════════════════════════
// Options
// ═══════════════════════════════════════════════════════════════════════════

pub const BUILTIN_SET: u16 = 80;
pub const BUILTIN_SETOPT: u16 = 81;
pub const BUILTIN_UNSETOPT: u16 = 82;
pub const BUILTIN_SHOPT: u16 = 83;
pub const BUILTIN_EMULATE: u16 = 84;
pub const BUILTIN_GETOPTS: u16 = 85;

// ═══════════════════════════════════════════════════════════════════════════
// Functions / Autoload
// ═══════════════════════════════════════════════════════════════════════════

pub const BUILTIN_AUTOLOAD: u16 = 90;
pub const BUILTIN_FUNCTIONS: u16 = 91;
pub const BUILTIN_UNFUNCTION: u16 = 92;

// ═══════════════════════════════════════════════════════════════════════════
// Traps / Signals
// ═══════════════════════════════════════════════════════════════════════════

pub const BUILTIN_TRAP: u16 = 100;

// ═══════════════════════════════════════════════════════════════════════════
// Directory stack
// ═══════════════════════════════════════════════════════════════════════════

pub const BUILTIN_PUSHD: u16 = 110;
pub const BUILTIN_POPD: u16 = 111;
pub const BUILTIN_DIRS: u16 = 112;

// ═══════════════════════════════════════════════════════════════════════════
// Type / Which / Hash
// ═══════════════════════════════════════════════════════════════════════════

pub const BUILTIN_TYPE: u16 = 120;
pub const BUILTIN_WHENCE: u16 = 121;
pub const BUILTIN_WHERE: u16 = 122;
pub const BUILTIN_WHICH: u16 = 123;
pub const BUILTIN_HASH: u16 = 124;
pub const BUILTIN_REHASH: u16 = 125;
pub const BUILTIN_UNHASH: u16 = 126;

// ═══════════════════════════════════════════════════════════════════════════
// Completion
// ═══════════════════════════════════════════════════════════════════════════

pub const BUILTIN_COMPGEN: u16 = 130;
pub const BUILTIN_COMPLETE: u16 = 131;
pub const BUILTIN_COMPOPT: u16 = 132;
pub const BUILTIN_COMPADD: u16 = 133;
pub const BUILTIN_COMPSET: u16 = 134;
pub const BUILTIN_COMPDEF: u16 = 135;
pub const BUILTIN_COMPINIT: u16 = 136;
pub const BUILTIN_CDREPLAY: u16 = 137;

// ═══════════════════════════════════════════════════════════════════════════
// Zsh-specific
// ═══════════════════════════════════════════════════════════════════════════

pub const BUILTIN_ZSTYLE: u16 = 140;
pub const BUILTIN_ZMODLOAD: u16 = 141;
pub const BUILTIN_BINDKEY: u16 = 142;
pub const BUILTIN_ZLE: u16 = 143;
pub const BUILTIN_VARED: u16 = 144;
pub const BUILTIN_ZCOMPILE: u16 = 145;
pub const BUILTIN_ZFORMAT: u16 = 146;
pub const BUILTIN_ZPARSEOPTS: u16 = 147;
pub const BUILTIN_ZREGEXPARSE: u16 = 148;

// ═══════════════════════════════════════════════════════════════════════════
// Resource limits
// ═══════════════════════════════════════════════════════════════════════════

pub const BUILTIN_ULIMIT: u16 = 150;
pub const BUILTIN_LIMIT: u16 = 151;
pub const BUILTIN_UNLIMIT: u16 = 152;
pub const BUILTIN_UMASK: u16 = 153;

// ═══════════════════════════════════════════════════════════════════════════
// Misc
// ═══════════════════════════════════════════════════════════════════════════

pub const BUILTIN_TIMES: u16 = 160;
pub const BUILTIN_CALLER: u16 = 161;
pub const BUILTIN_HELP: u16 = 162;
pub const BUILTIN_ENABLE: u16 = 163;
pub const BUILTIN_DISABLE: u16 = 164;
pub const BUILTIN_NOGLOB: u16 = 165;
pub const BUILTIN_TTYCTL: u16 = 166;
pub const BUILTIN_SYNC: u16 = 167;
pub const BUILTIN_MKDIR: u16 = 168;
pub const BUILTIN_STRFTIME: u16 = 169;
pub const BUILTIN_ZSLEEP: u16 = 170;
pub const BUILTIN_ZSYSTEM: u16 = 171;

// ═══════════════════════════════════════════════════════════════════════════
// PCRE
// ═══════════════════════════════════════════════════════════════════════════

pub const BUILTIN_PCRE_COMPILE: u16 = 180;
pub const BUILTIN_PCRE_MATCH: u16 = 181;
pub const BUILTIN_PCRE_STUDY: u16 = 182;

// ═══════════════════════════════════════════════════════════════════════════
// Database (GDBM)
// ═══════════════════════════════════════════════════════════════════════════

pub const BUILTIN_ZTIE: u16 = 190;
pub const BUILTIN_ZUNTIE: u16 = 191;
pub const BUILTIN_ZGDBMPATH: u16 = 192;

// ═══════════════════════════════════════════════════════════════════════════
// Prompt
// ═══════════════════════════════════════════════════════════════════════════

pub const BUILTIN_PROMPTINIT: u16 = 200;
pub const BUILTIN_PROMPT: u16 = 201;

// ═══════════════════════════════════════════════════════════════════════════
// Async / Parallel (zshrs extensions)
// ═══════════════════════════════════════════════════════════════════════════

pub const BUILTIN_ASYNC: u16 = 210;
pub const BUILTIN_AWAIT: u16 = 211;
pub const BUILTIN_PMAP: u16 = 212;
pub const BUILTIN_PGREP: u16 = 213;
pub const BUILTIN_PEACH: u16 = 214;
pub const BUILTIN_BARRIER: u16 = 215;

// ═══════════════════════════════════════════════════════════════════════════
// Intercept (AOP)
// ═══════════════════════════════════════════════════════════════════════════

pub const BUILTIN_INTERCEPT: u16 = 220;
pub const BUILTIN_INTERCEPT_PROCEED: u16 = 221;

// ═══════════════════════════════════════════════════════════════════════════
// Debug / Profile
// ═══════════════════════════════════════════════════════════════════════════

pub const BUILTIN_DOCTOR: u16 = 230;
pub const BUILTIN_DBVIEW: u16 = 231;
pub const BUILTIN_PROFILE: u16 = 232;
pub const BUILTIN_ZPROF: u16 = 233;

// ═══════════════════════════════════════════════════════════════════════════
// Coreutils (anti-fork builtins)
// ═══════════════════════════════════════════════════════════════════════════

pub const BUILTIN_CAT: u16 = 240;
pub const BUILTIN_HEAD: u16 = 241;
pub const BUILTIN_TAIL: u16 = 242;
pub const BUILTIN_WC: u16 = 243;
pub const BUILTIN_BASENAME: u16 = 244;
pub const BUILTIN_DIRNAME: u16 = 245;
pub const BUILTIN_TOUCH: u16 = 246;
pub const BUILTIN_REALPATH: u16 = 247;
pub const BUILTIN_SORT: u16 = 248;
pub const BUILTIN_FIND: u16 = 249;
pub const BUILTIN_UNIQ: u16 = 250;
pub const BUILTIN_CUT: u16 = 251;
pub const BUILTIN_TR: u16 = 252;
pub const BUILTIN_SEQ: u16 = 253;
pub const BUILTIN_REV: u16 = 254;
pub const BUILTIN_TEE: u16 = 255;
pub const BUILTIN_SLEEP: u16 = 256;
pub const BUILTIN_WHOAMI: u16 = 257;
pub const BUILTIN_ID: u16 = 258;
pub const BUILTIN_HOSTNAME: u16 = 259;
pub const BUILTIN_UNAME: u16 = 260;
pub const BUILTIN_DATE: u16 = 261;
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
}
