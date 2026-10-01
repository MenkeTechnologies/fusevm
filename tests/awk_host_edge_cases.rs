//! Edge-case pins for `awk_host` pure helper functions.
//!
//! These cover concrete bug classes the existing happy-path tests do not:
//!   - POSIX/gawk semantic edges the doc-comments claim compatibility with:
//!     negative `m` with `m+n>0` in `substr`, `lshift`/`rshift` 64-bit boundary,
//!     surrogate codepoints in `chr`, multi-byte first char in `ord`.
//!   - UTF-8 byte-vs-char arithmetic in `substr` and `index` (the helpers use
//!     Vec<char> / 1-based char position; verify they survive a multi-byte
//!     input that would corrupt a byte-slicing impl).
//!   - Fold-op zero-arg contract (the `unwrap_or(0)` branch is reachable via
//!     `Op::AwkAnd(0)` so a refactor to `.expect()` would panic at runtime).
//!   - `mktime` short-input / non-numeric-part / epoch-zero contract pins.
//!
//! Each test is hand-crafted around a specific failure mode — none of them
//! would catch a regression that a getter/mirror test would also catch.

use fusevm::awk_host::{
    awk_chr, awk_compl, awk_fold_and, awk_fold_or, awk_fold_xor, awk_index, awk_lshift, awk_mktime,
    awk_ord, awk_rshift, awk_substr,
};
use fusevm::value::Value;

// ───────────────────────────────────────────────────────────────────────────
// awk_substr: POSIX clamping + UTF-8 char counting
// ───────────────────────────────────────────────────────────────────────────

/// Negative `m` with `m + n` straddling position 1 should yield the prefix from
/// position 1 to `m + n - 1`. `substr("abcdef", -5, 10)` → "abcd" because the
/// character range is positions -5..=4, clamped to 1..=4.
#[test]
fn awk_substr_negative_start_with_overlap_returns_clamped_prefix() {
    let v = awk_substr(&Value::str("abcdef"), -5, Some(10)).to_str();
    assert_eq!(
        v, "abcd",
        "negative m with m+n>1 must return clamped prefix"
    );
}

/// Multi-byte UTF-8: `substr("héllo", 2, 3)` must return "éll" — three CHARS
/// starting from position 2. A byte-slice implementation would silently
/// fracture the é character or panic on a non-char-boundary slice.
#[test]
fn awk_substr_counts_chars_not_bytes_on_utf8() {
    let v = awk_substr(&Value::str("héllo"), 2, Some(3)).to_str();
    assert_eq!(v, "éll", "substr must count chars not bytes");
}

// ───────────────────────────────────────────────────────────────────────────
// awk_index: UTF-8 char-position
// ───────────────────────────────────────────────────────────────────────────

/// `index` must count CHARS, not BYTES. With multi-byte UTF-8 in the
/// pre-needle region the 1-based char position must match the visible
/// character count.
#[test]
fn awk_index_returns_char_position_not_byte_offset() {
    // "café" = c(1) a(2) f(3) é(4). 'é' is 2 bytes in UTF-8 → byte offset 5.
    // Looking for "x" after 'é': the char position must be 5, not 6.
    let pos = awk_index(&Value::str("caféx"), &Value::str("x"));
    assert_eq!(pos, 5, "index must return 1-based char position, not bytes");
}

// ───────────────────────────────────────────────────────────────────────────
// awk_lshift / awk_rshift: gawk shift-count boundary
// ───────────────────────────────────────────────────────────────────────────

/// gawk `do_lshift`: a shift count of 64 or more yields 0 (no `& 0x3f`
/// wrap-around), and a top-bit result survives `adjust_uint` because its low
/// bits are zero. Values checked against gawk 5.4.1.
#[test]
fn awk_lshift_count_of_64_or_more_yields_zero() {
    // gawk: lshift(1, 64) = 0, lshift(1, 65) = 0
    assert_eq!(awk_lshift(&Value::Int(1), &Value::Int(64)), 0.0);
    assert_eq!(awk_lshift(&Value::Int(1), &Value::Int(65)), 0.0);
    // gawk: lshift(1, 63) = 9223372036854775808
    assert_eq!(awk_lshift(&Value::Int(1), &Value::Int(63)), 2f64.powi(63));
}

/// `rshift` mirror of the 64-bit boundary.
#[test]
fn awk_rshift_count_of_64_or_more_yields_zero() {
    let top = Value::Float(2f64.powi(63));
    // gawk: rshift(2^63, 63) = 1, rshift(2^63, 64) = 0
    assert_eq!(awk_rshift(&top, &Value::Int(63)), 1.0);
    assert_eq!(awk_rshift(&top, &Value::Int(64)), 0.0);
}

// ───────────────────────────────────────────────────────────────────────────
// awk_compl + fold ops: empty fold contract
// ───────────────────────────────────────────────────────────────────────────

/// Empty-args folds are documented as returning 0. Pin so a refactor doesn't
/// introduce a panic on a frontend that emits a zero-arg `and()`/`or()`/
/// `xor()` (degenerate but reachable from `Op::AwkAnd(0)` with a host).
#[test]
fn awk_fold_ops_return_zero_on_empty_input() {
    assert_eq!(awk_fold_and(&[]), 0.0);
    assert_eq!(awk_fold_or(&[]), 0.0);
    assert_eq!(awk_fold_xor(&[]), 0.0);
}

/// gawk narrows `compl` with `adjust_uint`: `compl(0)` is `2^53 - 1`
/// (9007199254740991 in gawk 5.4.1), not the `-1` an `i64` complement gives.
#[test]
fn awk_compl_of_zero_is_two_pow_53_minus_one() {
    assert_eq!(awk_compl(&Value::Int(0)), 9_007_199_254_740_991.0);
}

// ───────────────────────────────────────────────────────────────────────────
// awk_ord / awk_chr: invalid codepoint + UTF-8 boundary
// ───────────────────────────────────────────────────────────────────────────

/// `chr(0xD800)` is a surrogate codepoint — not a valid Unicode scalar value.
/// Per the doc-comment ("empty string if invalid") this must round-trip to "".
/// A naive `char::from_u32_unchecked` impl would produce undefined behavior.
#[test]
fn awk_chr_returns_empty_for_surrogate_codepoint() {
    // U+D800 — first high surrogate, never a scalar value
    let v = awk_chr(&Value::Int(0xD800)).to_str();
    assert_eq!(v, "", "surrogate codepoint must yield empty string");
    // > U+10FFFF — out of Unicode range
    let v = awk_chr(&Value::Int(0x110000)).to_str();
    assert_eq!(v, "", "codepoint above U+10FFFF must yield empty string");
}

/// `ord` must take the first CHAR's scalar value, not the first BYTE.
/// "ñ" (U+00F1) is two UTF-8 bytes (0xC3 0xB1) — byte-based impl would return
/// 195, scalar-based impl returns 241.
#[test]
fn awk_ord_returns_scalar_value_of_first_char_not_first_byte() {
    let v = awk_ord(&Value::str("ñoño")).to_float();
    assert_eq!(
        v, 241.0,
        "ord must read first char's scalar, not first byte"
    );
}

// ───────────────────────────────────────────────────────────────────────────
// awk_mktime: short input + DST-ambiguous time
// ───────────────────────────────────────────────────────────────────────────

/// gawk `mktime` requires six whitespace-separated fields. Five is an error
/// (-1). The function must not panic on too-few-parts; pin the -1 contract.
#[test]
fn awk_mktime_returns_minus_one_on_too_few_parts() {
    let v = awk_mktime(&[Value::str("2020 1 1 0 0")]).to_float();
    assert_eq!(v, -1.0, "mktime with <6 parts must return -1");
}

/// gawk `mktime` returns -1 on non-numeric parts. Pin so a refactor that
/// uses `unwrap_or(0)` doesn't silently corrupt a bad datespec into epoch 0.
#[test]
fn awk_mktime_returns_minus_one_on_non_numeric_part() {
    let v = awk_mktime(&[Value::str("2020 1 1 abc 0 0")]).to_float();
    assert_eq!(v, -1.0, "mktime with non-numeric part must return -1");
}

/// Unix epoch in UTC: "1970 1 1 0 0 0" with utc=true → 0.0 seconds.
#[test]
fn awk_mktime_utc_epoch_zero() {
    let v = awk_mktime(&[Value::str("1970 1 1 0 0 0"), Value::Int(1)]).to_float();
    assert_eq!(v, 0.0, "1970-01-01 00:00:00 UTC must map to epoch 0");
}
