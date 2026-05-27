//! Additional coverage for the shell-builtins lookup table and the
//! `is_builtin` helper. Most aliases (chdir/bind/.) and rarely-listed
//! ones (intercept, dbview, async/await) are not exercised by the
//! in-module test set.

use fusevm::shell_builtins::*;

// ── Aliases map to the same ID as the canonical name ────────────────────────

#[test]
fn cd_and_chdir_share_id() {
    assert_eq!(builtin_id("cd"), Some(BUILTIN_CD));
    assert_eq!(builtin_id("chdir"), Some(BUILTIN_CD));
    assert_eq!(builtin_id("cd"), builtin_id("chdir"));
}

#[test]
fn source_and_dot_share_id() {
    assert_eq!(builtin_id("source"), Some(BUILTIN_SOURCE));
    assert_eq!(builtin_id("."), Some(BUILTIN_SOURCE));
}

#[test]
fn exit_aliases_share_id() {
    assert_eq!(builtin_id("exit"), Some(BUILTIN_EXIT));
    assert_eq!(builtin_id("bye"), Some(BUILTIN_EXIT));
    assert_eq!(builtin_id("logout"), Some(BUILTIN_EXIT));
}

#[test]
fn test_and_brace_share_id() {
    assert_eq!(builtin_id("test"), Some(BUILTIN_TEST));
    assert_eq!(builtin_id("["), Some(BUILTIN_TEST));
}

#[test]
fn declare_and_typeset_share_id() {
    assert_eq!(builtin_id("declare"), Some(BUILTIN_TYPESET));
    assert_eq!(builtin_id("typeset"), Some(BUILTIN_TYPESET));
}

#[test]
fn mapfile_and_readarray_share_id() {
    assert_eq!(builtin_id("mapfile"), Some(BUILTIN_MAPFILE));
    assert_eq!(builtin_id("readarray"), Some(BUILTIN_MAPFILE));
}

#[test]
fn bindkey_and_bind_share_id() {
    assert_eq!(builtin_id("bindkey"), Some(BUILTIN_BINDKEY));
    assert_eq!(builtin_id("bind"), Some(BUILTIN_BINDKEY));
}

// ── Negative lookups: things that look like builtins but aren't ────────────

#[test]
fn unknown_returns_none() {
    assert_eq!(builtin_id("not_a_builtin_xyzzy"), None);
    assert_eq!(builtin_id(""), None);
    assert_eq!(builtin_id("Cd"), None); // case-sensitive
    assert_eq!(builtin_id("CD"), None);
    assert_eq!(builtin_id(" cd"), None); // whitespace-sensitive
    assert_eq!(builtin_id("cd "), None);
}

#[test]
fn similar_but_distinct_commands_not_recognised() {
    assert_eq!(builtin_id("ls"), None);
    assert_eq!(builtin_id("grep"), None);
    assert_eq!(builtin_id("sed"), None);
    assert_eq!(builtin_id("awk"), None);
    assert_eq!(builtin_id("python"), None);
}

#[test]
fn is_builtin_matches_builtin_id_some() {
    let names = [
        "cd", "echo", "true", "false", ":", "source", ".", "test", "[", "declare", "typeset",
        "let", "shift", "trap", "alias", "set",
    ];
    for n in names {
        assert!(is_builtin(n), "{:?} should be a builtin", n);
    }
}

#[test]
fn is_builtin_false_for_unknown() {
    assert!(!is_builtin("not_a_builtin_xyzzy"));
    assert!(!is_builtin(""));
    assert!(!is_builtin("CD"));
}

// ── Coreutils group is recognised ───────────────────────────────────────────

#[test]
fn coreutils_recognised() {
    let coreutils = [
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
    for (name, expected) in coreutils {
        assert_eq!(builtin_id(name), Some(expected), "lookup of {:?}", name);
    }
}

// ── zsh-flavour group ──────────────────────────────────────────────────────

#[test]
fn zsh_flavour_builtins_recognised() {
    let zsh = [
        "setopt",
        "unsetopt",
        "shopt",
        "emulate",
        "autoload",
        "functions",
        "unfunction",
        "zmodload",
        "zstyle",
        "zle",
        "vared",
        "zcompile",
        "zformat",
        "zparseopts",
        "zregexparse",
        "zsleep",
        "zsystem",
        "zprof",
        "zgdbmpath",
        "ztie",
        "zuntie",
    ];
    for n in zsh {
        assert!(is_builtin(n), "{:?} should be a zsh builtin", n);
    }
}

// ── Concurrency/async group ─────────────────────────────────────────────────

#[test]
fn concurrency_builtins_recognised() {
    let names = [
        ("async", BUILTIN_ASYNC),
        ("await", BUILTIN_AWAIT),
        ("pmap", BUILTIN_PMAP),
        ("pgrep", BUILTIN_PGREP),
        ("peach", BUILTIN_PEACH),
        ("barrier", BUILTIN_BARRIER),
    ];
    for (n, id) in names {
        assert_eq!(builtin_id(n), Some(id));
    }
}

// ── ID range / collision invariants ─────────────────────────────────────────

#[test]
fn all_ids_under_builtin_max() {
    // BUILTIN_MAX is the documented upper bound. Sample a handful from
    // each numeric "band" to ensure none have drifted above it.
    let ids = [
        BUILTIN_CD,
        BUILTIN_LOCAL,
        BUILTIN_READ,
        BUILTIN_BREAK,
        BUILTIN_CAT,
        BUILTIN_MKTEMP,
    ];
    for id in ids {
        assert!(
            id < BUILTIN_MAX,
            "{} should be < BUILTIN_MAX ({})",
            id,
            BUILTIN_MAX
        );
    }
}

#[test]
fn canonical_names_have_unique_ids() {
    // Exhaustive uniqueness across a representative slice of canonical names.
    let canonical = [
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
        ("read", BUILTIN_READ),
        ("break", BUILTIN_BREAK),
        ("continue", BUILTIN_CONTINUE),
        ("eval", BUILTIN_EVAL),
        ("exec", BUILTIN_EXEC),
        ("let", BUILTIN_LET),
        ("jobs", BUILTIN_JOBS),
        ("alias", BUILTIN_ALIAS),
        ("trap", BUILTIN_TRAP),
        ("type", BUILTIN_TYPE),
        ("ulimit", BUILTIN_ULIMIT),
        ("umask", BUILTIN_UMASK),
        ("cat", BUILTIN_CAT),
        ("sort", BUILTIN_SORT),
        ("mktemp", BUILTIN_MKTEMP),
    ];
    let mut seen = std::collections::HashSet::new();
    for (name, id) in canonical {
        assert!(seen.insert(id), "duplicate id {} for {:?}", id, name);
        assert_eq!(builtin_id(name), Some(id));
    }
}

// ── Control-flow group ──────────────────────────────────────────────────────

#[test]
fn control_flow_group_recognised() {
    assert_eq!(builtin_id("break"), Some(BUILTIN_BREAK));
    assert_eq!(builtin_id("continue"), Some(BUILTIN_CONTINUE));
    assert_eq!(builtin_id("shift"), Some(BUILTIN_SHIFT));
    assert_eq!(builtin_id("eval"), Some(BUILTIN_EVAL));
    assert_eq!(builtin_id("return"), Some(BUILTIN_RETURN));
    assert_eq!(builtin_id(":"), Some(BUILTIN_COLON));
}

// ── Idempotency: lookup always returns the same id ──────────────────────────

#[test]
fn repeated_lookups_are_deterministic() {
    for _ in 0..3 {
        assert_eq!(builtin_id("printf"), Some(BUILTIN_PRINTF));
        assert_eq!(builtin_id("nope_xyz"), None);
        assert!(is_builtin("source"));
        assert!(!is_builtin("nope_xyz"));
    }
}
