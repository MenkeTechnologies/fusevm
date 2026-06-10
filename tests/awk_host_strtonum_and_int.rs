//! Edge-case pins for `awk_strtonum`, `awk_int`, and `awk_intdiv0` — covering
//! parsing branches and overflow boundaries that the existing happy-path tests
//! don't reach.
//!
//! Each test is hand-crafted around a specific failure mode:
//!   - `awk_strtonum` has four mutually-exclusive parse branches (empty,
//!     bad-leading-byte, 0x-hex, leading-zero octal, longest-decimal-prefix);
//!     a refactor that reorders or drops a branch silently changes
//!     `strtonum("0xff")`, `strtonum("+0xff")`, `strtonum("042")`, or
//!     `strtonum("42abc")` to the wrong number.
//!   - `awk_int` has a narrow Float→Int demotion window driven by
//!     `t.is_finite() && t >= i64::MIN as f64 && t <= i64::MAX as f64`;
//!     a NaN or out-of-range argument MUST remain a `Value::Float`, not
//!     silently cast to `Value::Int(0)` (which is what `t as i64` would
//!     produce after Rust's 1.45 saturating cast).
//!   - `awk_intdiv0` MUST be the "safe" variant — division by zero never
//!     yields `Value::Undef` (that's `awk_intdiv`'s contract). A refactor that
//!     copy-pastes the `awk_intdiv` body and forgets to flip the `Undef`
//!     branch silently turns `intdiv0(x, 0)` into a fatal-style sentinel for
//!     callers that test with `is_truthy`.

use fusevm::awk_host::{awk_int, awk_intdiv0, awk_strtonum};
use fusevm::value::Value;

// ───────────────────────────────────────────────────────────────────────────
// awk_strtonum: branch coverage for the parse precedence
// ───────────────────────────────────────────────────────────────────────────

/// The `0x…` branch must beat the leading-decimal-prefix branch even when the
/// trailing characters would also parse as decimal. `strtonum("0x10")` is hex
/// 16, NOT decimal 10 (which would happen if the longest-prefix branch ran
/// first and grabbed "0").
#[test]
fn strtonum_hex_prefix_beats_decimal_prefix() {
    assert_eq!(
        awk_strtonum("0x10"),
        16.0,
        "0x10 must parse as hex 16, not 10"
    );
    assert_eq!(awk_strtonum("0xff"), 255.0);
    assert_eq!(awk_strtonum("0XFF"), 255.0, "uppercase 0X must also be hex");
}

/// gawk explicitly disqualifies the `0x…` / `0…` octal branches when the input
/// starts with a sign — `+` / `-` flips into the longest-decimal-prefix path.
/// `strtonum("+0x10")` must be 0.0 (longest numeric prefix of "+0x10" is "+0"
/// → 0), NOT 16.0. Pin so a refactor that drops the `unsigned_hex_or_octal`
/// gate silently breaks gawk parity.
#[test]
fn strtonum_signed_zero_x_is_not_hex() {
    assert_eq!(
        awk_strtonum("+0x10"),
        0.0,
        "+0x10: signed → hex branch skipped; longest prefix is +0 → 0",
    );
    assert_eq!(
        awk_strtonum("-0x10"),
        0.0,
        "-0x10: signed → hex branch skipped; longest prefix is -0 → 0",
    );
}

/// The leading-zero octal branch must not fire when the input contains a `.`
/// or an exponent marker — those force decimal/float parsing. `strtonum("0.5")`
/// must be 0.5, not 0 (which `i64::from_str_radix("0.5", 8)` would yield as the
/// error fallback).
#[test]
fn strtonum_octal_branch_skipped_for_dot_or_exponent() {
    assert_eq!(
        awk_strtonum("0.5"),
        0.5,
        "0.5 has '.', octal branch must skip"
    );
    assert_eq!(
        awk_strtonum("0e2"),
        0.0,
        "0e2 has 'e', octal branch must skip"
    );
    assert_eq!(
        awk_strtonum("01e2"),
        100.0,
        "01e2 has 'e' → decimal float, NOT octal 1 (gawk semantics)",
    );
}

/// `strtonum("042")` is octal 42 = decimal 34. A future refactor that drops the
/// leading-zero check or routes through `parse::<f64>()` directly would yield
/// 42.0 (decimal interpretation), silently breaking gawk parity for any awk
/// script that uses octal literals via strtonum.
#[test]
fn strtonum_octal_path_parses_leading_zero_as_base_eight() {
    assert_eq!(awk_strtonum("042"), 34.0, "042 must parse as octal 34");
    assert_eq!(awk_strtonum("010"), 8.0, "010 must parse as octal 8");
}

/// gawk treats a bare alphabetic-starting token (e.g. "inf", "nan") as 0.0,
/// even though `f64::from_str` accepts both. The function's first-byte gate
/// disqualifies them. This pin defends against a refactor that drops the
/// `matches!(first, b'+' | b'-' | b'.' | b'0'..=b'9')` filter and lets
/// rust's parser accept the bare keywords.
#[test]
fn strtonum_bare_inf_nan_disqualified_by_first_byte_filter() {
    assert_eq!(awk_strtonum("inf"), 0.0, "bare inf must be disqualified");
    assert_eq!(awk_strtonum("nan"), 0.0, "bare nan must be disqualified");
    assert_eq!(awk_strtonum("Inf"), 0.0);
    assert_eq!(awk_strtonum("NaN"), 0.0);
}

/// The longest-decimal-prefix branch must clip at the first non-numeric byte.
/// `strtonum("42abc")` is 42, NOT a parse error → 0. A naive `f64::from_str`
/// over the whole input would fall back to 0.0.
#[test]
fn strtonum_takes_longest_numeric_prefix_not_full_string() {
    assert_eq!(awk_strtonum("42abc"), 42.0);
    assert_eq!(awk_strtonum("2.5xyz"), 2.5);
    assert_eq!(awk_strtonum("-7.5q"), -7.5);
    assert_eq!(awk_strtonum("1e3foo"), 1000.0);
}

/// Leading and trailing whitespace are stripped by `s.trim()`. Pin so a
/// refactor that drops the trim silently breaks `strtonum("  42  ")`.
#[test]
fn strtonum_trims_surrounding_whitespace() {
    assert_eq!(awk_strtonum("  42  "), 42.0);
    assert_eq!(awk_strtonum("\t-3.5\n"), -3.5);
    assert_eq!(
        awk_strtonum("   "),
        0.0,
        "all-whitespace → 0 via empty branch"
    );
    assert_eq!(awk_strtonum(""), 0.0);
}

/// `strtonum("0x")` (hex prefix with no digits) must NOT panic and must yield
/// 0.0 via the `u64::from_str_radix` error fallback. A refactor that uses
/// `unwrap()` would crash the entire AWK program.
#[test]
fn strtonum_hex_prefix_without_digits_yields_zero_no_panic() {
    assert_eq!(awk_strtonum("0x"), 0.0);
    assert_eq!(awk_strtonum("0X"), 0.0);
    // garbage hex digit also yields 0.0 via fallback
    assert_eq!(awk_strtonum("0xZZ"), 0.0);
}

/// Octal path with an invalid octal digit (8 or 9) must yield 0.0 via the
/// `i64::from_str_radix` error fallback, NOT promote to decimal. gawk uses
/// `strtoul(s, &endptr, 8)` which stops at the first '8' — so "08" → 0.
#[test]
fn strtonum_invalid_octal_digit_yields_zero_via_fallback() {
    assert_eq!(
        awk_strtonum("08"),
        0.0,
        "08 has invalid octal '8' → fallback 0"
    );
    assert_eq!(awk_strtonum("09"), 0.0);
    assert_eq!(awk_strtonum("0789"), 0.0);
}

// ───────────────────────────────────────────────────────────────────────────
// awk_int: Float→Int demotion window
// ───────────────────────────────────────────────────────────────────────────

/// `awk_int(NaN)` MUST stay a `Value::Float(NaN)`, never demote to `Int(0)`.
/// The `t.is_finite()` guard is the only thing preventing the NaN→0 demotion
/// (Rust's saturating `NaN as i64` returns 0). A refactor that drops the
/// guard would silently change `int(nan)` from NaN to 0 — wrong vs gawk.
#[test]
fn awk_int_nan_preserved_as_float() {
    let v = awk_int(&Value::Float(f64::NAN));
    match v {
        Value::Float(f) => assert!(f.is_nan(), "int(nan) must be Float(NaN)"),
        other => panic!("expected Value::Float(NaN), got {:?}", other),
    }
}

/// `awk_int(+inf)` / `awk_int(-inf)` MUST stay `Value::Float(±inf)`, never
/// demote to `Int(i64::MAX / i64::MIN)`. Same reasoning as the NaN case:
/// `t.is_finite()` is the gate.
#[test]
fn awk_int_infinity_preserved_as_float() {
    let v = awk_int(&Value::Float(f64::INFINITY));
    match v {
        Value::Float(f) => assert!(f.is_infinite() && f.is_sign_positive()),
        other => panic!("expected Value::Float(+inf), got {:?}", other),
    }
    let v = awk_int(&Value::Float(f64::NEG_INFINITY));
    match v {
        Value::Float(f) => assert!(f.is_infinite() && f.is_sign_negative()),
        other => panic!("expected Value::Float(-inf), got {:?}", other),
    }
}

/// `awk_int(1e20)` is finite but past `i64::MAX as f64` (≈9.22e18). It MUST
/// stay a `Value::Float`, not silently saturate to `Int(i64::MAX)`. Pin the
/// upper-bound check.
#[test]
fn awk_int_out_of_range_finite_stays_float() {
    let v = awk_int(&Value::Float(1e20));
    match v {
        Value::Float(f) => assert_eq!(f, 1e20),
        other => panic!("expected Value::Float(1e20), got {:?}", other),
    }
    // Below i64::MIN range
    let v = awk_int(&Value::Float(-1e20));
    match v {
        Value::Float(f) => assert_eq!(f, -1e20),
        other => panic!("expected Value::Float(-1e20), got {:?}", other),
    }
}

/// `awk_int` truncates toward zero, NOT toward minus-infinity. Pin so a
/// refactor swapping `trunc()` for `floor()` silently breaks negative inputs.
#[test]
fn awk_int_truncates_toward_zero_not_floor() {
    assert_eq!(awk_int(&Value::Float(3.9)), Value::Int(3));
    assert_eq!(
        awk_int(&Value::Float(-3.9)),
        Value::Int(-3),
        "trunc(-3.9)=-3, NOT floor(-3.9)=-4",
    );
    assert_eq!(awk_int(&Value::Float(0.999)), Value::Int(0));
    assert_eq!(awk_int(&Value::Float(-0.999)), Value::Int(0));
}

/// `awk_int` on a string applies numeric coercion via `Value::to_float`. The
/// result type still follows the same finite-range demotion. Pin that "3.7" →
/// Int(3), not Float(3.0).
#[test]
fn awk_int_demotes_finite_string_coercion_to_int() {
    assert_eq!(awk_int(&Value::str("3.7")), Value::Int(3));
    assert_eq!(awk_int(&Value::str("-2.4")), Value::Int(-2));
    // Non-numeric string coerces to 0.0 → Int(0).
    assert_eq!(awk_int(&Value::str("abc")), Value::Int(0));
}

// ───────────────────────────────────────────────────────────────────────────
// awk_intdiv0: "safe" division-by-zero contract
// ───────────────────────────────────────────────────────────────────────────

/// `awk_intdiv0(a, 0)` MUST yield `Value::Float(0.0)`, NEVER `Value::Undef`.
/// Pin the difference vs `awk_intdiv`'s `Undef` contract — a refactor that
/// copy-pastes the wrong branch would silently turn intdiv0 into a fatal
/// sentinel for callers that test the result with `is_truthy`.
#[test]
fn awk_intdiv0_by_zero_returns_zero_float_not_undef() {
    let v = awk_intdiv0(&Value::Int(42), &Value::Int(0));
    assert_eq!(
        v,
        Value::Float(0.0),
        "intdiv0 by zero MUST yield Float(0.0)"
    );
    // Make sure it's NOT Undef
    assert!(
        !matches!(v, Value::Undef),
        "intdiv0 by zero must NOT yield Undef (that's awk_intdiv's contract)",
    );
}

/// `awk_intdiv0` truncates toward zero (because it casts to i64 internally).
/// Pin negative-result trunc semantics so a refactor swapping the cast for
/// `f64::div_euclid` (which rounds toward minus-infinity) silently breaks
/// negative-result programs.
#[test]
fn awk_intdiv0_truncates_negative_result_toward_zero() {
    // -7 / 2 = -3.5 → trunc → -3 (not floor → -4)
    let v = awk_intdiv0(&Value::Int(-7), &Value::Int(2));
    assert_eq!(
        v,
        Value::Float(-3.0),
        "negative result must truncate toward zero"
    );
    // 7 / -2 = -3.5 → trunc → -3
    let v = awk_intdiv0(&Value::Int(7), &Value::Int(-2));
    assert_eq!(v, Value::Float(-3.0));
}

/// Float `b` that truncates to zero must STILL trip the zero branch. A naive
/// check `bf == 0.0` catches +0.0 / -0.0 but a follow-up `as i64` cast on a
/// non-zero float like 0.5 would yield 0 → division by 0 panic. Pin the
/// internal `bi == 0` check that defends against this exact gap.
#[test]
fn awk_intdiv0_float_b_that_truncates_to_zero_still_safe() {
    // 0.5 truncates to i64 = 0 — must hit the `bi == 0` guard, not panic.
    let v = awk_intdiv0(&Value::Float(10.0), &Value::Float(0.5));
    assert_eq!(
        v,
        Value::Float(0.0),
        "b=0.5 truncates to 0 → safe-zero branch"
    );
}

/// `awk_strtonum` previously panicked when the trailing portion of the input
/// landed on a non-char-boundary byte. `awk_longest_f64_prefix` iterated
/// byte indices via `(1..=s.len()).rev()` and sliced `&s[..end]`, which
/// panics inside a multi-byte UTF-8 codepoint. Inputs like `"42€"` (€ = 3
/// bytes), `"1e2🦀"` (🦀 = 4 bytes), or `"3.14αβγ"` triggered it.
#[test]
fn awk_strtonum_no_panic_on_trailing_multi_byte_utf8() {
    // 2-byte: NBSP
    let _ = awk_strtonum("42\u{00A0}");
    // 3-byte: euro sign
    let _ = awk_strtonum("42€");
    // 3-byte: CJK ideograph
    let _ = awk_strtonum("3.14中文");
    // 4-byte: crab emoji
    let _ = awk_strtonum("1e2🦀");
    // Lone surrogate-like multi-byte at boundary
    let _ = awk_strtonum("0xFF日本語");
    // Numeric prefix followed immediately by multi-byte
    assert_eq!(awk_strtonum("42€"), 42.0);
    #[allow(clippy::approx_constant)]
    let pi_approx: f64 = 3.14;
    assert_eq!(awk_strtonum("3.14中文"), pi_approx);
}
