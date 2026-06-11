//! Edge-case pins for `awk_intdiv` (the `Undef`-on-zero variant) and
//! `awk_strftime`'s UTC formatting path — two surfaces the existing happy-path
//! and `awk_intdiv0`-focused tests do not exercise.
//!
//! Bug classes pinned here:
//!   - `awk_intdiv` truncates BOTH operands to i64 before dividing, so a nonzero
//!     fractional divisor that truncates to 0 (e.g. `0.5`) hits the `bi == 0`
//!     fallthrough and returns `Value::Undef`, NOT a quotient. This diverges
//!     from `awk_intdiv0`, which returns `Float(0.0)` for the same input. The
//!     `bi == 0` guard (distinct from the `bf == 0.0` guard) is its own branch
//!     and would silently regress if a refactor collapsed the two checks or
//!     forgot to truncate the divisor first.
//!   - `awk_intdiv` must truncate the QUOTIENT toward zero (i64 division), not
//!     floor it — `(-7) / 2 == -3`, not `-4`. A float-division refactor would
//!     break sign-rounding on negative dividends.
//!   - `awk_strftime` with an explicit UTC timestamp must be timezone-stable:
//!     the formatted output is independent of the test host's `$TZ`. A regression
//!     that dropped the `utc` flag and used `Local` would make this flaky/wrong
//!     on non-UTC CI runners (the macOS matrix leg, etc.).
//!
//! `i64::MIN` overflow inputs are deliberately avoided — that panic is a known
//! bug already pinned in `awk_host_known_bugs_DO_NOT_COMMIT.rs`; reproducing it
//! here would just duplicate a failing test.

use fusevm::awk_host::{awk_intdiv, awk_strftime};
use fusevm::value::Value;

// ───────────────────────────────────────────────────────────────────────────
// awk_intdiv: divisor-truncation divergence + quotient sign rounding
// ───────────────────────────────────────────────────────────────────────────

/// `awk_intdiv(a, 0.5)` — divisor is nonzero as a float (`bf != 0.0`) but
/// truncates to `0` as i64, so the `bi == 0` guard fires and the result is
/// `Undef`. This is the documented divergence from `awk_intdiv0`, which returns
/// `Float(0.0)` for the identical input. Pin BOTH the `Undef` result here and
/// the `bi == 0` (not `bf == 0.0`) branch that produces it.
#[test]
fn awk_intdiv_fractional_divisor_truncating_to_zero_yields_undef() {
    let v = awk_intdiv(&Value::Float(10.0), &Value::Float(0.5));
    assert!(
        matches!(v, Value::Undef),
        "awk_intdiv with a divisor that truncates to 0 must return Undef \
         (the bi==0 fallthrough), got {v:?} — this is the contract that \
         differs from awk_intdiv0's Float(0.0)"
    );
}

/// Direct zero divisor (`bf == 0.0`) also yields `Undef`, via the first guard.
/// Kept distinct from the fractional case above so a refactor that removed one
/// guard but not the other is caught by exactly one of the two tests.
#[test]
fn awk_intdiv_exact_zero_divisor_yields_undef() {
    let v = awk_intdiv(&Value::Int(7), &Value::Int(0));
    assert!(
        matches!(v, Value::Undef),
        "awk_intdiv(_, 0) must return Undef, got {v:?}"
    );
}

/// Quotient truncates toward zero, not toward negative infinity. `(-7) / 2`
/// must be `-3.0` (i64 truncation), never `-4.0` (a floor). A refactor that
/// computed `(a.to_float() / b.to_float()).floor()` instead of truncating i64
/// operands would silently produce `-4.0` here.
#[test]
fn awk_intdiv_negative_dividend_truncates_toward_zero() {
    let v = awk_intdiv(&Value::Int(-7), &Value::Int(2)).to_float();
    assert_eq!(
        v, -3.0,
        "awk_intdiv truncates toward zero: (-7)/2 == -3, not floor(-3.5) == -4"
    );
    // Symmetric case: negative divisor, positive dividend.
    let w = awk_intdiv(&Value::Int(7), &Value::Int(-2)).to_float();
    assert_eq!(w, -3.0, "7/(-2) == -3 under toward-zero truncation");
}

// ───────────────────────────────────────────────────────────────────────────
// awk_strftime: UTC path is timezone-independent
// ───────────────────────────────────────────────────────────────────────────

/// `strftime(fmt, ts, utc=1)` must format the timestamp in UTC regardless of
/// the host timezone. The 3-arg form sets the third operand truthy → UTC; the
/// epoch `0` formatted as `%Y-%m-%d %H:%M:%S` is exactly the Unix epoch in UTC.
/// This is the one strftime assertion that can be deterministic in CI without
/// pinning `$TZ`, and it guards the `utc` flag actually selecting the `Utc`
/// branch (a regression to `Local` would shift the output by the runner's
/// offset).
#[test]
fn awk_strftime_utc_epoch_zero_is_timezone_independent() {
    let v = awk_strftime(&[
        Value::str("%Y-%m-%d %H:%M:%S"),
        Value::Float(0.0),
        Value::Int(1), // utc truthy
    ])
    .to_str();
    assert_eq!(
        v, "1970-01-01 00:00:00",
        "strftime UTC epoch 0 must be the Unix epoch regardless of host TZ"
    );
}

/// A known non-epoch UTC instant, again format-pinned to a TZ-independent shape.
/// `1000000000` seconds is 2001-09-09 01:46:40 UTC. Pins that the UTC branch
/// advances the calendar correctly (not just at the epoch boundary).
#[test]
fn awk_strftime_utc_known_instant_formats_in_utc() {
    let v = awk_strftime(&[
        Value::str("%Y-%m-%d %H:%M:%S"),
        Value::Float(1_000_000_000.0),
        Value::Int(1),
    ])
    .to_str();
    assert_eq!(
        v, "2001-09-09 01:46:40",
        "strftime must format 1e9 seconds as 2001-09-09 01:46:40 in UTC"
    );
}
