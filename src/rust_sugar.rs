//! Source-level desugaring for `rust { ... }` inline-FFI blocks — shared by
//! every fusevm frontend.
//!
//! Why pre-lex rather than a new AST node? So the feature needs zero grammar
//! changes in each frontend: the block is rewritten, at the source level,
//! into an ordinary call to a runtime builtin that carries the block body
//! base64-encoded. Later phases (lexer / parser / compiler) only ever see a
//! normal function call; the builtin dispatches to [`crate::ffi`].
//!
//! The scanner treats `rust` as a keyword only at a statement boundary
//! (start-of-file, or after a boundary byte / — optionally — a newline) with an
//! opening `{` following. Inside the block body it uses Rust's own
//! brace/string/comment rules to find the matching `}`. False positives on
//! exotic constructs fall through unchanged.
//!
//! The lang-specific bits — the keyword, comment syntax, whether a newline ends
//! a statement, and how the replacement call is spelled — are supplied by
//! [`RustSugar`] so one scanner serves Perl-family, C-family, and
//! indentation-based frontends alike.

use base64::Engine as _;

/// Per-frontend configuration for the `rust { ... }` desugarer.
///
/// Construct one per language (typically a `const`) and call
/// [`RustSugar::desugar`] on the raw source before lexing.
#[derive(Clone, Copy)]
pub struct RustSugar {
    /// The block-introducing keyword. Almost always `"rust"`.
    pub keyword: &'static str,
    /// Line-comment introducers to skip over (e.g. `&["//", "#"]` for PHP), or
    /// `&[]` for none. Skipping comments matters so a `;`/`}` inside one does
    /// not create a false statement boundary.
    pub line_comments: &'static [&'static str],
    /// Nesting block-comment delimiters to skip over, or `None`.
    pub block_comment: Option<(&'static str, &'static str)>,
    /// Whether a newline resets the statement boundary. C/Perl-family: `false`
    /// (statements span lines, delimited by `;`/`{`/`}`). Line-oriented
    /// frontends (shell, vimscript, python at the top level): `true`.
    pub newline_boundary: bool,
    /// Formats the replacement statement from `(base64_body, source_line)`.
    /// Must be valid top-level source in the target language and must run the
    /// referenced compile-and-register builtin before any FFI call — e.g.
    /// `__rust_compile("<b64>", 12);` for C-family, or a `BEGIN { ... }` wrap
    /// for awk.
    pub emit: fn(&str, usize) -> String,
}

impl RustSugar {
    /// Rewrite `code` so every top-level `rust { ... }` block becomes a call
    /// into the FFI runtime. Returns the input unchanged when no block exists
    /// (single substring scan on the fast path).
    ///
    /// Line numbers are preserved: the replacement is padded with as many
    /// newlines as the original block spanned, so later diagnostics keep their
    /// source position.
    pub fn desugar(&self, code: &str) -> String {
        // Fast path: no candidate keyword → no work.
        if !code.contains(self.keyword) {
            return code.to_string();
        }
        let bytes = code.as_bytes();
        let mut out = String::with_capacity(code.len());
        let mut i = 0;
        let mut can_start_stmt = true; // start-of-file is a boundary
        let mut line = 1usize;

        while i < bytes.len() {
            let c = bytes[i];

            // Skip strings/comments so a `rust {` inside `"...rust {..."` or a
            // comment is never matched.
            //
            // CRITICAL: emit every run via `out.push_str(&code[a..b])` (a slice
            // of the original `&str`) so multi-byte UTF-8 round-trips intact.
            // Casting individual bytes via `bytes[i] as char` would re-encode
            // every high byte and mangle non-ASCII source.
            if c == b'\n' {
                out.push('\n');
                i += 1;
                line += 1;
                if self.newline_boundary {
                    can_start_stmt = true;
                }
                continue;
            }
            if c == b' ' || c == b'\t' || c == b'\r' {
                out.push(c as char);
                i += 1;
                continue;
            }
            // Line comment (any configured introducer).
            if self
                .line_comments
                .iter()
                .any(|c| !c.is_empty() && code[i..].starts_with(c))
            {
                let start = i;
                while i < bytes.len() && bytes[i] != b'\n' {
                    i += 1;
                }
                out.push_str(&code[start..i]);
                continue;
            }
            // Block comment (nesting).
            if let Some((open, close)) = self.block_comment {
                if code[i..].starts_with(open) {
                    let start = i;
                    let mut depth = 1i32;
                    i += open.len();
                    while i < bytes.len() && depth > 0 {
                        if code[i..].starts_with(open) {
                            depth += 1;
                            i += open.len();
                        } else if code[i..].starts_with(close) {
                            depth -= 1;
                            i += close.len();
                        } else {
                            if bytes[i] == b'\n' {
                                line += 1;
                            }
                            i += 1;
                        }
                    }
                    out.push_str(&code[start..i]);
                    continue;
                }
            }
            if c == b'"' || c == b'\'' || c == b'`' {
                // Quoted string with `\` escapes.
                let quote = c;
                let start = i;
                i += 1;
                while i < bytes.len() {
                    let b = bytes[i];
                    if b == b'\\' && i + 1 < bytes.len() {
                        if bytes[i + 1] == b'\n' {
                            line += 1;
                        }
                        i += 2;
                        continue;
                    }
                    i += 1;
                    if b == b'\n' {
                        line += 1;
                    }
                    if b == quote {
                        break;
                    }
                }
                out.push_str(&code[start..i]);
                can_start_stmt = false;
                continue;
            }
            if c == b';' || c == b'{' || c == b'}' {
                out.push(c as char);
                i += 1;
                can_start_stmt = true;
                continue;
            }

            // Identifier start? Only single-byte ASCII letters / `_`. Any
            // non-ASCII byte leads a UTF-8 sequence and must be copied via a
            // slice (never cast to `char`).
            let is_ident_start = c.is_ascii_alphabetic() || c == b'_';
            if !is_ident_start {
                let start = i;
                let step = if c < 0x80 {
                    1
                } else if c < 0xC0 {
                    1 // stray continuation byte — advance conservatively
                } else if c < 0xE0 {
                    2
                } else if c < 0xF0 {
                    3
                } else {
                    4
                };
                i = (i + step).min(bytes.len());
                out.push_str(&code[start..i]);
                can_start_stmt = false;
                continue;
            }

            // Read the identifier.
            let start = i;
            while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
                i += 1;
            }
            let ident = &code[start..i];

            if ident == self.keyword && can_start_stmt {
                // Peek past whitespace (newlines allowed) for `{`.
                let mut j = i;
                let mut inline_newlines = 0usize;
                while j < bytes.len() && matches!(bytes[j], b' ' | b'\t' | b'\r' | b'\n') {
                    if bytes[j] == b'\n' {
                        inline_newlines += 1;
                    }
                    j += 1;
                }
                if j < bytes.len() && bytes[j] == b'{' {
                    if let Some((body, end)) = scan_rust_block(bytes, j) {
                        let block_line = line;
                        let body_newlines = body.bytes().filter(|&b| b == b'\n').count();
                        let total_newlines = inline_newlines + body_newlines;
                        let encoded =
                            base64::engine::general_purpose::STANDARD.encode(body.as_bytes());
                        out.push_str(&(self.emit)(&encoded, block_line));
                        for _ in 0..total_newlines {
                            out.push('\n');
                        }
                        line += total_newlines;
                        i = end;
                        can_start_stmt = true;
                        continue;
                    }
                    // Unbalanced — let the normal lexer report the error.
                }
            }

            out.push_str(ident);
            can_start_stmt = false;
        }
        out
    }
}

/// Scan a Rust `{ ... }` block starting at the `{` byte at `open`. Returns
/// `(body, end)` where `body` excludes the outer braces and `end` is the byte
/// index immediately after the closing `}`. Handles string literals, raw
/// strings (`r"..."` / `r#"..."#`), char literals, line comments, and nested
/// block comments per Rust's lexer.
fn scan_rust_block(bytes: &[u8], open: usize) -> Option<(&str, usize)> {
    debug_assert_eq!(bytes[open], b'{');
    let mut i = open + 1;
    let body_start = i;
    let mut depth: i32 = 1;
    while i < bytes.len() {
        let c = bytes[i];
        match c {
            b'{' => {
                depth += 1;
                i += 1;
            }
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    let body = std::str::from_utf8(&bytes[body_start..i]).ok()?;
                    return Some((body, i + 1));
                }
                i += 1;
            }
            b'/' if i + 1 < bytes.len() && bytes[i + 1] == b'/' => {
                i += 2;
                while i < bytes.len() && bytes[i] != b'\n' {
                    i += 1;
                }
            }
            b'/' if i + 1 < bytes.len() && bytes[i + 1] == b'*' => {
                i += 2;
                let mut cdepth: i32 = 1;
                while i < bytes.len() && cdepth > 0 {
                    if i + 1 < bytes.len() && bytes[i] == b'/' && bytes[i + 1] == b'*' {
                        cdepth += 1;
                        i += 2;
                    } else if i + 1 < bytes.len() && bytes[i] == b'*' && bytes[i + 1] == b'/' {
                        cdepth -= 1;
                        i += 2;
                    } else {
                        i += 1;
                    }
                }
            }
            b'"' => {
                i += 1;
                while i < bytes.len() {
                    match bytes[i] {
                        b'\\' if i + 1 < bytes.len() => i += 2,
                        b'"' => {
                            i += 1;
                            break;
                        }
                        _ => i += 1,
                    }
                }
            }
            b'r' if i + 1 < bytes.len() && (bytes[i + 1] == b'"' || bytes[i + 1] == b'#') => {
                let mut j = i + 1;
                let mut hashes = 0usize;
                while j < bytes.len() && bytes[j] == b'#' {
                    hashes += 1;
                    j += 1;
                }
                if j >= bytes.len() || bytes[j] != b'"' {
                    i += 1;
                    continue;
                }
                j += 1;
                while j < bytes.len() {
                    if bytes[j] == b'"' {
                        let mut k = j + 1;
                        let mut matched = 0;
                        while matched < hashes && k < bytes.len() && bytes[k] == b'#' {
                            matched += 1;
                            k += 1;
                        }
                        if matched == hashes {
                            j = k;
                            break;
                        }
                        j += 1;
                    } else {
                        j += 1;
                    }
                }
                i = j;
            }
            b'\'' => {
                let mut j = i + 1;
                if j < bytes.len() && bytes[j] == b'\\' && j + 1 < bytes.len() {
                    j += 2;
                    if j < bytes.len() && bytes[j] == b'{' {
                        while j < bytes.len() && bytes[j] != b'}' {
                            j += 1;
                        }
                        if j < bytes.len() {
                            j += 1;
                        }
                    }
                } else if j < bytes.len() {
                    j += 1;
                }
                if j < bytes.len() && bytes[j] == b'\'' {
                    i = j + 1;
                } else {
                    i += 1;
                }
            }
            _ => i += 1,
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    // A C-family emitter (bare call statement) used by most frontends.
    fn c_emit(b64: &str, line: usize) -> String {
        format!("__rust_compile(\"{b64}\", {line});")
    }
    const C_FAMILY: RustSugar = RustSugar {
        keyword: "rust",
        line_comments: &["//"],
        block_comment: Some(("/*", "*/")),
        newline_boundary: false,
        emit: c_emit,
    };

    // A shell/hash-comment emitter (line-oriented).
    fn hash_emit(b64: &str, line: usize) -> String {
        format!("__rust_compile \"{b64}\" {line}")
    }
    const HASH_FAMILY: RustSugar = RustSugar {
        keyword: "rust",
        line_comments: &["#"],
        block_comment: None,
        newline_boundary: true,
        emit: hash_emit,
    };

    #[test]
    fn no_keyword_pass_through() {
        let src = "console.log('hello');\n";
        assert_eq!(C_FAMILY.desugar(src), src);
    }

    #[test]
    fn keyword_in_string_not_expanded() {
        let src = "log(\"rust { not a block }\");\n";
        assert_eq!(C_FAMILY.desugar(src), src);
    }

    #[test]
    fn keyword_in_line_comment_not_expanded() {
        let src = "// rust { not a block }\nlog(1);\n";
        assert_eq!(C_FAMILY.desugar(src), src);
    }

    #[test]
    fn keyword_in_hash_comment_not_expanded() {
        let src = "# rust { not a block }\nprint(1)\n";
        assert_eq!(HASH_FAMILY.desugar(src), src);
    }

    #[test]
    fn simple_block_becomes_call() {
        let src =
            "rust { pub extern \"C\" fn add(a: i64, b: i64) -> i64 { a + b } }\nlog(add(1,2));\n";
        let out = C_FAMILY.desugar(src);
        assert!(out.contains("__rust_compile("), "no builtin call: {out}");
        assert!(!out.contains("pub extern"), "Rust body leaked: {out}");
        assert!(out.contains("log(add(1,2));"));
    }

    #[test]
    fn nested_braces_balanced() {
        let src = "rust { pub extern \"C\" fn f() -> i64 { let v = vec![1,2,3]; v.iter().sum::<i64>() } }";
        let out = C_FAMILY.desugar(src);
        assert!(out.starts_with("__rust_compile("));
        assert!(!out.contains("pub extern"));
    }

    #[test]
    fn string_with_brace_balanced() {
        let src = "rust { pub extern \"C\" fn g() -> i64 { let s = \"}\"; s.len() as i64 } }";
        let out = C_FAMILY.desugar(src);
        assert!(out.starts_with("__rust_compile("), "desugar failed: {out}");
        assert!(!out.contains("pub extern"));
    }

    #[test]
    fn raw_string_with_brace_balanced() {
        let src = "rust { pub extern \"C\" fn h() -> i64 { let s = r#\"}\"#; s.len() as i64 } }";
        assert!(C_FAMILY.desugar(src).starts_with("__rust_compile("));
    }

    #[test]
    fn line_number_preserved_by_newline_padding() {
        // The block spans two source lines; the replacement must carry two
        // newlines so `after` stays on line 3.
        let src = "rust {\n pub extern \"C\" fn z() -> i64 { 1 }\n}\nafter();\n";
        let out = C_FAMILY.desugar(src);
        let after_line = out.lines().position(|l| l.contains("after()")).unwrap();
        assert_eq!(after_line, 3, "after() must stay on line index 3:\n{out}");
    }

    #[test]
    fn identifier_starting_with_rust_not_matched() {
        let src = "let rusty = 1; function rusty2() { return 1; }\n";
        assert_eq!(C_FAMILY.desugar(src), src);
    }

    #[test]
    fn keyword_mid_expression_not_matched() {
        // `rust` after `=` is not at a statement boundary.
        let src = "let x = rust { 1 }\n";
        assert_eq!(C_FAMILY.desugar(src), src);
    }

    #[test]
    fn semicolon_inside_comment_is_not_a_false_boundary() {
        // A `;` inside a skipped line comment must NOT flip can_start_stmt and
        // let the following `rust {` (which is itself commentary) desugar.
        let src = "// end; rust { not a block }\nlog(1);\n";
        assert_eq!(C_FAMILY.desugar(src), src);
    }

    #[test]
    fn php_style_two_line_comment_styles() {
        // PHP has both `//` and `#`; a `;` in either must not create a false
        // boundary for a trailing `rust {`.
        const PHP: RustSugar = RustSugar {
            keyword: "rust",
            line_comments: &["//", "#"],
            block_comment: Some(("/*", "*/")),
            newline_boundary: true,
            emit: c_emit,
        };
        let src = "# note; rust { skip }\n// too; rust { skip }\n$x = 1;\n";
        assert_eq!(PHP.desugar(src), src);
    }

    #[test]
    fn preserves_utf8_when_fast_path_misses() {
        // Any `rust` substring makes the fast path miss and forces a full scan;
        // non-ASCII must round-trip byte-for-byte.
        let src = "// banner — § arrow → must survive; trust me\nlet b = \"── §1 ──\";\n";
        assert_eq!(C_FAMILY.desugar(src), src);
    }
}
