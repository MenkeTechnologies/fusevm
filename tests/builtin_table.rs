//! Comprehensive table-driven coverage for `builtin_id` and `is_builtin`,
//! plus invariants over the constant ID space.

use fusevm::shell_builtins::*;

// ── Table-driven canonical mapping for every documented name ──────────────

#[test]
fn all_canonical_builtin_names_map_correctly() {
    let cases: &[(&str, u16)] = &[
        ("cd", BUILTIN_CD),
        ("pwd", BUILTIN_PWD),
        ("echo", BUILTIN_ECHO),
        ("print", BUILTIN_PRINT),
        ("printf", BUILTIN_PRINTF),
        ("export", BUILTIN_EXPORT),
        ("unset", BUILTIN_UNSET),
        ("source", BUILTIN_SOURCE),
        ("exit", BUILTIN_EXIT),
        ("return", BUILTIN_RETURN),
        ("true", BUILTIN_TRUE),
        ("false", BUILTIN_FALSE),
        ("test", BUILTIN_TEST),
        (":", BUILTIN_COLON),
        ("local", BUILTIN_LOCAL),
        ("declare", BUILTIN_TYPESET),
        ("readonly", BUILTIN_READONLY),
        ("integer", BUILTIN_INTEGER),
        ("float", BUILTIN_FLOAT),
        ("read", BUILTIN_READ),
        ("mapfile", BUILTIN_MAPFILE),
        ("break", BUILTIN_BREAK),
        ("continue", BUILTIN_CONTINUE),
        ("shift", BUILTIN_SHIFT),
        ("eval", BUILTIN_EVAL),
        ("exec", BUILTIN_EXEC),
        ("command", BUILTIN_COMMAND),
        ("builtin", BUILTIN_BUILTIN),
        ("let", BUILTIN_LET),
        ("jobs", BUILTIN_JOBS),
        ("fg", BUILTIN_FG),
        ("bg", BUILTIN_BG),
        ("kill", BUILTIN_KILL),
        ("disown", BUILTIN_DISOWN),
        ("wait", BUILTIN_WAIT),
        ("suspend", BUILTIN_SUSPEND),
        ("history", BUILTIN_HISTORY),
        ("fc", BUILTIN_FC),
        ("r", BUILTIN_R),
        ("alias", BUILTIN_ALIAS),
        ("unalias", BUILTIN_UNALIAS),
        ("set", BUILTIN_SET),
        ("setopt", BUILTIN_SETOPT),
        ("unsetopt", BUILTIN_UNSETOPT),
        ("shopt", BUILTIN_SHOPT),
        ("emulate", BUILTIN_EMULATE),
        ("getopts", BUILTIN_GETOPTS),
        ("autoload", BUILTIN_AUTOLOAD),
        ("functions", BUILTIN_FUNCTIONS),
        ("unfunction", BUILTIN_UNFUNCTION),
        ("trap", BUILTIN_TRAP),
        ("pushd", BUILTIN_PUSHD),
        ("popd", BUILTIN_POPD),
        ("dirs", BUILTIN_DIRS),
        ("type", BUILTIN_TYPE),
        ("whence", BUILTIN_WHENCE),
        ("where", BUILTIN_WHERE),
        ("which", BUILTIN_WHICH),
        ("hash", BUILTIN_HASH),
        ("rehash", BUILTIN_REHASH),
        ("unhash", BUILTIN_UNHASH),
        ("compgen", BUILTIN_COMPGEN),
        ("complete", BUILTIN_COMPLETE),
        ("compopt", BUILTIN_COMPOPT),
        ("compadd", BUILTIN_COMPADD),
        ("compset", BUILTIN_COMPSET),
        ("compdef", BUILTIN_COMPDEF),
        ("compinit", BUILTIN_COMPINIT),
        ("cdreplay", BUILTIN_CDREPLAY),
        ("zstyle", BUILTIN_ZSTYLE),
        ("zmodload", BUILTIN_ZMODLOAD),
        ("bindkey", BUILTIN_BINDKEY),
        ("zle", BUILTIN_ZLE),
        ("vared", BUILTIN_VARED),
        ("zcompile", BUILTIN_ZCOMPILE),
        ("zformat", BUILTIN_ZFORMAT),
        ("zparseopts", BUILTIN_ZPARSEOPTS),
        ("zregexparse", BUILTIN_ZREGEXPARSE),
        ("ulimit", BUILTIN_ULIMIT),
        ("limit", BUILTIN_LIMIT),
        ("unlimit", BUILTIN_UNLIMIT),
        ("umask", BUILTIN_UMASK),
        ("times", BUILTIN_TIMES),
        ("caller", BUILTIN_CALLER),
        ("help", BUILTIN_HELP),
        ("enable", BUILTIN_ENABLE),
        ("disable", BUILTIN_DISABLE),
        ("noglob", BUILTIN_NOGLOB),
        ("ttyctl", BUILTIN_TTYCTL),
        ("sync", BUILTIN_SYNC),
        ("mkdir", BUILTIN_MKDIR),
        ("strftime", BUILTIN_STRFTIME),
        ("zsleep", BUILTIN_ZSLEEP),
        ("zsystem", BUILTIN_ZSYSTEM),
        ("pcre_compile", BUILTIN_PCRE_COMPILE),
        ("pcre_match", BUILTIN_PCRE_MATCH),
        ("pcre_study", BUILTIN_PCRE_STUDY),
        ("ztie", BUILTIN_ZTIE),
        ("zuntie", BUILTIN_ZUNTIE),
        ("zgdbmpath", BUILTIN_ZGDBMPATH),
        ("promptinit", BUILTIN_PROMPTINIT),
        ("prompt", BUILTIN_PROMPT),
        ("async", BUILTIN_ASYNC),
        ("await", BUILTIN_AWAIT),
        ("pmap", BUILTIN_PMAP),
        ("pgrep", BUILTIN_PGREP),
        ("peach", BUILTIN_PEACH),
        ("barrier", BUILTIN_BARRIER),
        ("intercept", BUILTIN_INTERCEPT),
        ("intercept_proceed", BUILTIN_INTERCEPT_PROCEED),
        ("doctor", BUILTIN_DOCTOR),
        ("dbview", BUILTIN_DBVIEW),
        ("profile", BUILTIN_PROFILE),
        ("zprof", BUILTIN_ZPROF),
        ("cat", BUILTIN_CAT),
        ("head", BUILTIN_HEAD),
        ("tail", BUILTIN_TAIL),
        ("wc", BUILTIN_WC),
        ("basename", BUILTIN_BASENAME),
        ("dirname", BUILTIN_DIRNAME),
        ("touch", BUILTIN_TOUCH),
        ("realpath", BUILTIN_REALPATH),
        ("sort", BUILTIN_SORT),
        ("find", BUILTIN_FIND),
        ("uniq", BUILTIN_UNIQ),
        ("cut", BUILTIN_CUT),
        ("tr", BUILTIN_TR),
        ("seq", BUILTIN_SEQ),
        ("rev", BUILTIN_REV),
        ("tee", BUILTIN_TEE),
        ("sleep", BUILTIN_SLEEP),
        ("whoami", BUILTIN_WHOAMI),
        ("id", BUILTIN_ID),
        ("hostname", BUILTIN_HOSTNAME),
        ("uname", BUILTIN_UNAME),
        ("date", BUILTIN_DATE),
        ("mktemp", BUILTIN_MKTEMP),
    ];
    for (name, id) in cases {
        assert_eq!(builtin_id(name), Some(*id), "mismatch for {:?}", name);
        assert!(is_builtin(name), "{:?} should be a builtin", name);
        assert!(*id < BUILTIN_MAX, "{:?} id {} exceeds BUILTIN_MAX {}", name, id, BUILTIN_MAX);
    }
}

#[test]
fn all_aliases_share_canonical_id() {
    let alias_cases: &[(&str, &str)] = &[
        ("chdir", "cd"),
        (".", "source"),
        ("bye", "exit"),
        ("logout", "exit"),
        ("[", "test"),
        ("typeset", "declare"),
        ("readarray", "mapfile"),
        ("bind", "bindkey"),
    ];
    for (alias, canonical) in alias_cases {
        assert_eq!(
            builtin_id(alias),
            builtin_id(canonical),
            "alias {:?} should match canonical {:?}",
            alias,
            canonical
        );
    }
}

#[test]
fn unknown_names_return_none() {
    let unknowns = [
        "",
        "ls",
        "grep",
        "awk",
        "sed",
        "definitely_not_a_builtin_xyz",
        "CD",
        "PWD",
        "cd ",
        " cd",
        "\tcd",
        "cd\n",
        "echo;",
        "cd/",
    ];
    for name in unknowns {
        assert!(builtin_id(name).is_none(), "{:?} should not be a builtin", name);
        assert!(!is_builtin(name));
    }
}

#[test]
fn is_builtin_is_consistent_with_builtin_id() {
    let names = [
        "cd", "echo", "true", "false", ":", "test", "[", "type", "source",
        ".", "ls", "rm", "definitely_no_such_thing",
    ];
    for n in names {
        assert_eq!(is_builtin(n), builtin_id(n).is_some());
    }
}

#[test]
fn all_builtin_ids_are_below_max() {
    let ids = [
        BUILTIN_CD, BUILTIN_PWD, BUILTIN_ECHO, BUILTIN_PRINT, BUILTIN_PRINTF,
        BUILTIN_EXPORT, BUILTIN_UNSET, BUILTIN_SOURCE, BUILTIN_EXIT,
        BUILTIN_RETURN, BUILTIN_TRUE, BUILTIN_FALSE, BUILTIN_TEST,
        BUILTIN_COLON, BUILTIN_DOT, BUILTIN_LOCAL, BUILTIN_DECLARE,
        BUILTIN_TYPESET, BUILTIN_READONLY, BUILTIN_INTEGER, BUILTIN_FLOAT,
        BUILTIN_READ, BUILTIN_MAPFILE, BUILTIN_BREAK, BUILTIN_CONTINUE,
        BUILTIN_SHIFT, BUILTIN_EVAL, BUILTIN_EXEC, BUILTIN_COMMAND,
        BUILTIN_BUILTIN, BUILTIN_MKTEMP,
    ];
    for id in ids {
        assert!(id < BUILTIN_MAX);
    }
}

#[test]
fn declare_and_typeset_both_map_to_typeset_id() {
    // Both names alias to the same canonical id BUILTIN_TYPESET.
    assert_eq!(builtin_id("declare"), Some(BUILTIN_TYPESET));
    assert_eq!(builtin_id("typeset"), Some(BUILTIN_TYPESET));
}

#[test]
fn unicode_and_case_variants_not_builtins() {
    let variants = ["Cd", "CD", "ECHO", "Source", "ＣＤ", "ＳＯＵＲＣＥ"];
    for v in variants {
        assert!(!is_builtin(v), "{:?} should not be a builtin", v);
    }
}

#[test]
fn special_one_char_builtins() {
    // ":", ".", "[", "r" are all single-char builtins.
    assert_eq!(builtin_id(":"), Some(BUILTIN_COLON));
    assert_eq!(builtin_id("."), Some(BUILTIN_SOURCE));
    assert_eq!(builtin_id("["), Some(BUILTIN_TEST));
    assert_eq!(builtin_id("r"), Some(BUILTIN_R));
}

#[test]
fn no_builtin_returns_id_equal_to_max() {
    // BUILTIN_MAX is an exclusive upper bound — no builtin's id should equal it.
    for name in &[
        "cd", "mktemp", "zprof", "sync", "barrier", "intercept_proceed",
    ] {
        let id = builtin_id(name).expect("known builtin");
        assert!(id < BUILTIN_MAX);
    }
}
