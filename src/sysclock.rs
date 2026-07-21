//! Portable wall-clock reads.
//!
//! `std::time::SystemTime::now()` **panics** on `wasm32-unknown-unknown` (the
//! target has no clock), which would take down any frontend running fusevm in a
//! browser web worker. `chrono::Utc::now()` is portable: on native it reads the
//! system clock, and on `wasm32` chrono's `wasmbind` feature (active whenever
//! the crate is built for a wasm target — it pulls in `js-sys`) reads
//! `Date.now()` through the JS host, which a web worker provides. Routing the
//! VM's clock ops through here keeps one code path across native and wasm with
//! no `cfg` branches at the call sites.

use chrono::Utc;

/// Whole seconds since the Unix epoch. Backs `Op::TimeInt` and awk `systime`'s
/// integer needs. Returns a monotonic-ish wall-clock value; never panics.
#[inline]
pub fn unix_secs() -> i64 {
    Utc::now().timestamp()
}

/// Fractional seconds since the Unix epoch. Backs gawk `systime()`, which
/// returns a float. Combines whole seconds with the sub-second nanos so the
/// value matches the previous `Duration::as_secs_f64()` behaviour.
#[inline]
pub fn unix_secs_f64() -> f64 {
    let now = Utc::now();
    now.timestamp() as f64 + now.timestamp_subsec_nanos() as f64 / 1_000_000_000.0
}

/// A clock-derived entropy word for seeding the awk PRNG when `srand()` is
/// called with no argument. Mirrors the prior
/// `d.as_secs() ^ (d.subsec_nanos() as u64)` mix so seed sequences are
/// unchanged on native.
#[inline]
pub fn unix_nanos_entropy() -> u64 {
    let now = Utc::now();
    (now.timestamp() as u64) ^ (now.timestamp_subsec_nanos() as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unix_secs_is_after_2020() {
        // 2020-01-01T00:00:00Z = 1_577_836_800. Any real clock read is later,
        // and on wasm this exercises the chrono/js-sys path in a browser.
        assert!(unix_secs() > 1_577_836_800, "clock read looks unset");
    }

    #[test]
    fn unix_secs_f64_tracks_unix_secs() {
        let f = unix_secs_f64();
        let i = unix_secs();
        // The float read happens first, so f <= i is possible only across a
        // second boundary; assert they agree to within 2 seconds.
        assert!((f - i as f64).abs() < 2.0, "f={f} i={i}");
    }

    #[test]
    fn entropy_varies_from_zero() {
        // Extremely unlikely to be exactly zero for a real clock read.
        assert_ne!(unix_nanos_entropy(), 0);
    }
}
