//! Coverage for the AWK PRNG (`awk_rand`, `awk_srand`) and the float-operand
//! contracts of the bitwise/transcendental helpers (`awk_compl`, `awk_lshift`,
//! `awk_rshift`, `awk_canon_nan`) — none of which had any unit tests before
//! this file.
//!
//! Each test pins a specific bug class:
//!   - `awk_rand` is a glibc LCG with hard-coded constants 1103515245/12345.
//!     A refactor swapping in a different LCG (e.g. POSIX's older
//!     `rand_r(3)`) silently changes every script's output and breaks
//!     reproducibility. Pin the first value from a known seed plus the
//!     determinism contract.
//!   - `awk_srand` returns the previous seed truncated to 32 bits, matching
//!     gawk's `srand` contract. A refactor that returns the full 64-bit seed
//!     (or the new seed) silently breaks scripts that snapshot the PRNG.
//!   - The bitwise helpers truncate their float operand via
//!     `n.trunc() as i64 as u64`. A refactor that uses `to_int()` instead
//!     would route Strings through `s.parse::<i64>()` (which fails on
//!     "3.7", returning 0) vs. the current `to_float() = 3.7 → trunc = 3`.
//!   - `awk_canon_nan` is an identity for finite results — pin so a refactor
//!     that always returns `f64::NAN` for `is_nan() || is_infinite()` breaks
//!     `sqrt(inf) = inf` and `log(inf) = inf`.

use fusevm::awk_host::{awk_canon_nan, awk_compl, awk_lshift, awk_rand, awk_rshift, awk_srand};
use fusevm::value::Value;

// ───────────────────────────────────────────────────────────────────────────
// awk_rand: glibc LCG formula + determinism
// ───────────────────────────────────────────────────────────────────────────

/// Pin the exact glibc LCG formula (`seed = seed * 1103515245 + 12345`, result
/// = `((seed >> 16) & 0x7fff) / 32768.0`) by computing the expected first
/// value from seed=1 in the test. A refactor swapping constants (e.g. to
/// POSIX `rand_r`'s 0x41c64e6d / 0x3039) silently changes every awk script's
/// random output and breaks any test/replay relying on `srand(1); rand()`.
#[test]
fn awk_rand_first_value_from_seed_one_matches_glibc_lcg() {
    let mut seed: u64 = 1;
    let v = awk_rand(&mut seed);
    // Independently re-derive the expected value (without copy-pasting the
    // impl's constants — they're inlined in the doc-cited formula).
    let expected_seed = 1_u64.wrapping_mul(1103515245).wrapping_add(12345);
    let expected = f64::from((expected_seed >> 16) as u32 & 0x7fff) / 32768.0;
    assert_eq!(v, expected, "first rand() from seed=1 must match glibc LCG");
}

/// Two PRNGs seeded identically must produce identical sequences. This is the
/// reproducibility contract: scripts using `srand(N); rand()` rely on it. A
/// refactor that adds any non-deterministic mutation (e.g. system entropy
/// mixing) silently breaks deterministic test corpora.
#[test]
fn awk_rand_is_deterministic_across_independent_seeds() {
    let mut s1: u64 = 42;
    let mut s2: u64 = 42;
    for i in 0..16 {
        let a = awk_rand(&mut s1);
        let b = awk_rand(&mut s2);
        assert_eq!(a, b, "rand call #{i}: same seed must give same value");
    }
}

/// Every `awk_rand` result must be in `[0, 1)` — the closed-open interval
/// gawk documents. A refactor that drops the `& 0x7fff` mask would lift the
/// upper bound to 65535/32768 ≈ 2.0, silently breaking every awk script that
/// uses `rand()` to index into a table of size N via `int(rand() * N)`.
#[test]
fn awk_rand_result_is_in_unit_interval() {
    let mut seed: u64 = 0xDEAD_BEEF;
    for _ in 0..512 {
        let v = awk_rand(&mut seed);
        assert!(
            (0.0..1.0).contains(&v),
            "rand result {v} outside [0, 1) — mask or divisor regression"
        );
    }
}

// ───────────────────────────────────────────────────────────────────────────
// awk_srand: previous-seed contract
// ───────────────────────────────────────────────────────────────────────────

/// gawk `srand([x])` returns the previous seed truncated to its low 32 bits.
/// A refactor that returns the full 64-bit seed (or the NEW seed) silently
/// breaks scripts that do `prev = srand(N)` to snapshot/restore the PRNG.
#[test]
fn awk_srand_returns_low_32_bits_of_previous_seed() {
    // Seed with a value whose top 32 bits are non-zero so the truncation is
    // visible — if the impl skipped the mask, the result would be the full
    // 64-bit value rendered as a (huge) f64.
    let mut seed: u64 = 0xDEAD_BEEF_CAFE_BABE;
    let prev = awk_srand(&mut seed, Some(0));
    assert_eq!(
        prev, 0xCAFE_BABE_u32 as f64,
        "srand must return low 32 bits of previous seed, not full 64"
    );
    // New seed is exactly what we passed in (Some path, no clock read).
    assert_eq!(seed, 0, "explicit Some(n) must replace seed wholesale");
}

/// `srand(None)` — seed from clock — must still advance the seed (not leave
/// it at zero) and return the previous seed. Pin so a refactor that forgets
/// the SystemTime fallback leaves the PRNG stuck at the initial value.
#[test]
fn awk_srand_none_path_replaces_seed_with_nonzero_from_clock() {
    let mut seed: u64 = 7; // arbitrary starting value
    let prev = awk_srand(&mut seed, None);
    // Previous was 7 → low 32 bits = 7
    assert_eq!(prev, 7.0);
    // New seed must NOT remain 7 (would mean clock path didn't fire); also
    // must not be 0 unless the system clock is exactly at the epoch (which
    // would mean test machine time is wildly broken).
    assert_ne!(seed, 7, "clock-seed path must replace seed");
    assert_ne!(seed, 0, "clock-seed should not be zero on a sane test host");
}

// ───────────────────────────────────────────────────────────────────────────
// Bitwise float-operand truncation contract
// ───────────────────────────────────────────────────────────────────────────

/// `awk_compl` on a Float operand routes through `to_float().trunc() as i64
/// as u64`. Pin so a refactor that uses `to_int()` silently changes the
/// behavior on string Values that look like fractions (`to_int("3.7")` is
/// 0 via `s.parse::<i64>().unwrap_or(0)`, but the current path coerces via
/// `to_float()` → 3.7 → trunc 3).
#[test]
fn awk_compl_truncates_float_operand_toward_zero() {
    // 3.7 → trunc 3 → !3u64 = 0xFFFF_FFFF_FFFF_FFFC = -4 as i64
    assert_eq!(awk_compl(&Value::Float(3.7)), -4);
    // -3.7 → trunc -3 → !(-3 as u64) = !(0xFFFF...FD) = 2
    assert_eq!(awk_compl(&Value::Float(-3.7)), 2);
    // String "3.7" must also flow through to_float → 3.7 → 3, NOT
    // to_int → 0. Catches a refactor that swaps `to_float()` for `to_int()`.
    assert_eq!(
        awk_compl(&Value::str("3.7")),
        -4,
        "compl(\"3.7\") must coerce via to_float, not to_int (which yields 0)"
    );
}

/// `awk_lshift` shift count is float-truncated then masked. A non-integer
/// shift count like 2.9 must truncate to 2 (NOT round to 3). Pin so a
/// refactor that calls `.round()` instead of relying on the `as i64`
/// truncating cast silently breaks `lshift(1, 2.9)`.
#[test]
fn awk_lshift_truncates_float_shift_count_toward_zero() {
    // 2.9 → trunc 2 → 1 << 2 = 4
    assert_eq!(awk_lshift(&Value::Int(1), &Value::Float(2.9)), 4);
    // -0.5 → trunc 0 → (0 & 0x3f) = 0 → identity shift
    assert_eq!(awk_lshift(&Value::Int(42), &Value::Float(-0.5)), 42);
}

/// `awk_rshift` on a Float operand. A naive refactor that demotes via
/// `to_int()` would silently break the awkrs frontend which may pass
/// numeric-string fields as `Value::Float`.
#[test]
fn awk_rshift_truncates_float_operand_then_shifts() {
    // 16.9 → trunc 16 → 16 >> 2 = 4
    assert_eq!(awk_rshift(&Value::Float(16.9), &Value::Int(2)), 4);
    // -1.0 (as float) → trunc -1 → cast i64::MIN_to_MAX_unsigned all-ones
    // → rshift by 63 = 1 (same as the integer test, but via the float path).
    assert_eq!(awk_rshift(&Value::Float(-1.0), &Value::Int(63)), 1);
}

// ───────────────────────────────────────────────────────────────────────────
// awk_canon_nan: identity for finite/inf, pos-NaN for NaN
// ───────────────────────────────────────────────────────────────────────────

/// `awk_canon_nan` MUST pass through finite values unchanged. A refactor that
/// canonicalizes anything other than NaN (e.g. infinity → NaN) silently
/// breaks `sqrt(inf)`, `exp(-inf)=0`, `log(inf)=inf` which the trait defaults
/// all wrap through `awk_canon_nan`. Pin the finite passthrough.
#[test]
fn awk_canon_nan_passes_finite_values_unchanged() {
    for v in [0.0, -0.0, 1.5, -1.5, f64::MIN, f64::MAX, 1e300, -1e300] {
        assert_eq!(
            awk_canon_nan(v),
            v,
            "finite {v} must survive canon_nan unchanged"
        );
    }
}

/// Infinity is NOT NaN, so it must pass through. Without this pin a refactor
/// that uses `r.is_finite()` instead of `r.is_nan()` would canonicalize both
/// inf and NaN to NaN, silently breaking `exp(big_number) → inf`.
#[test]
fn awk_canon_nan_preserves_infinities() {
    assert_eq!(awk_canon_nan(f64::INFINITY), f64::INFINITY);
    assert_eq!(awk_canon_nan(f64::NEG_INFINITY), f64::NEG_INFINITY);
}

/// NaN inputs (positive or negative bit pattern) MUST all return positive
/// NaN. Without canonicalization the JIT-fast-path NaN comparison
/// `result != result` would still detect NaN, but the sign bit could leak
/// into the host's display of `+nan` vs `-nan` — awkrs/gawk normalize this.
#[test]
fn awk_canon_nan_normalizes_negative_nan_to_positive() {
    // Construct a negative-NaN bit pattern and verify the canonical form
    // matches the standard `f64::NAN`. Both are NaN, so we compare bit
    // patterns of the IEEE-754 quiet-NaN canonical form: f64::NAN.to_bits().
    let neg_nan = f64::from_bits(f64::NAN.to_bits() | (1 << 63));
    assert!(neg_nan.is_nan(), "constructed input must still be NaN");
    let canon = awk_canon_nan(neg_nan);
    assert!(canon.is_nan(), "result must be NaN");
    assert_eq!(
        canon.to_bits(),
        f64::NAN.to_bits(),
        "canon_nan must return the standard positive NaN bit pattern"
    );
}
