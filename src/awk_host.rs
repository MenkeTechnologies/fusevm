//! AWK host callback interface.
//!
//! Frontends that emit AWK-specific bytecode (awkrs) provide an [`AwkHost`]
//! implementation. The VM dispatches the reserved AWK op range (see
//! [`crate::awk_builtins`]) through this trait so the actual semantics — field
//! access, record resplitting, `print`/`printf` with `OFS`/`ORS`/`OFMT`,
//! `getline` I/O, the string builtins, and `SUBSEP` associative arrays — live
//! in the frontend, not the VM core.
//!
//! Why a dedicated host (rather than native `fusevm::Value` ops): AWK values
//! carry POSIX **numeric-string** duality (a field that looks numeric compares
//! numerically but prints as its original text), `CONVFMT`/`OFMT` formatting,
//! and field/`$0`/`NF` coupling. `fusevm::Value` cannot represent those, so the
//! authoritative AWK runtime state stays host-side and the VM acts as the
//! control-flow + stack engine that calls back into it.
//!
//! Without a host, the AWK ops fall back to minimal stubs — the VM still runs,
//! but AWK-specific ops are no-ops / identity / empty results.
//!
//! All methods have default implementations so frontends only override what
//! they need (and so the trait can grow without breaking existing frontends).

use crate::value::Value;
use chrono::{Local, LocalResult, NaiveDate, TimeZone, Utc};
use std::cmp::Ordering;

/// Frontend-supplied implementation of AWK-specific runtime behavior.
///
/// The VM owns a `Box<dyn AwkHost>` (registered via
/// [`crate::vm::VM::set_awk_host`]) and routes the AWK op range to these
/// methods. Implementations operate on the frontend's own AWK runtime (record,
/// fields, special vars, arrays, I/O), translating to/from `fusevm::Value` only
/// at the call boundary.
pub trait AwkHost: Send {
    // ── Fields & record ────────────────────────────────────────────────────

    /// `$i` — read field `i` (`$0` is the whole record). Out-of-range fields
    /// read as the empty string per POSIX. Returns `""` by default.
    fn field_get(&mut self, i: i64) -> Value {
        let _ = i;
        Value::str("")
    }

    /// `$i = v` — assign field `i`. Assigning a field at or beyond `NF` extends
    /// the record (filling gaps with `OFS`); assigning `$0` resplits. Rebuilds
    /// `$0`/`NF` as required. No-op by default.
    fn field_set(&mut self, i: i64, v: Value) {
        let _ = (i, v);
    }

    /// `NF` — current field count. Returns `0` by default.
    fn nf(&mut self) -> i64 {
        0
    }

    /// `$0 = v` — replace the entire record and resplit into fields using the
    /// current `FS`. No-op by default.
    fn set_record(&mut self, v: Value) {
        let _ = v;
    }

    // ── Special variables ──────────────────────────────────────────────────

    /// Read a special AWK variable by name (`FS`, `OFS`, `ORS`, `RS`, `NR`,
    /// `FNR`, `SUBSEP`, `RSTART`, `RLENGTH`, `FILENAME`, `CONVFMT`, `OFMT`, …).
    /// Returns `Undef` by default.
    fn special_get(&mut self, name: &str) -> Value {
        let _ = name;
        Value::Undef
    }

    /// Assign a special AWK variable by name. Some assignments have side effects
    /// (e.g. changing `FS` affects the next split; setting `NF` rebuilds `$0`).
    /// No-op by default.
    fn special_set(&mut self, name: &str, v: Value) {
        let _ = (name, v);
    }

    // ── Output ─────────────────────────────────────────────────────────────

    /// `print a, b, …` — emit the arguments joined by `OFS` and terminated by
    /// `ORS`. With no arguments the frontend prints `$0`. No-op by default.
    fn print(&mut self, args: &[Value]) {
        let _ = args;
    }

    /// `printf fmt, …` — format and emit (no trailing `ORS`). No-op by default.
    fn printf(&mut self, fmt: &str, args: &[Value]) {
        let _ = (fmt, args);
    }

    /// `sprintf(fmt, …)` — format and return the string. Returns `""` default.
    fn sprintf(&mut self, fmt: &str, args: &[Value]) -> Value {
        let _ = (fmt, args);
        Value::str("")
    }

    // ── Input ──────────────────────────────────────────────────────────────

    /// `getline` family. `source` is one of [`crate::awk_builtins::getline_source`].
    /// `operand` is the file path / command string for file/command sources
    /// (ignored for the main-input sources). For `*_VAR` sources the target
    /// variable is identified by `var_name`. Returns the getline status:
    /// `1` (record read), `0` (EOF), or `-1` (error). Returns `0` by default.
    fn getline(&mut self, source: usize, operand: Option<&str>, var_name: Option<&str>) -> i64 {
        let _ = (source, operand, var_name);
        0
    }

    // ── String builtins ────────────────────────────────────────────────────

    /// `length(s)` — string length in characters. `None` ⇒ `length($0)`.
    fn length(&mut self, s: Option<&Value>) -> i64 {
        awk_length(s)
    }

    /// `substr(s, m [, n])` — 1-based substring with POSIX clamping. `n` is the
    /// length; `None` means "to end of string".
    fn substr(&mut self, s: &Value, m: i64, n: Option<i64>) -> Value {
        awk_substr(s, m, n)
    }

    /// `index(s, t)` — 1-based index of `t` in `s`, or `0` if absent.
    fn index(&mut self, s: &Value, t: &Value) -> i64 {
        awk_index(s, t)
    }

    /// `split(s, arr, fs)` — split `s` into `arr` using `fs` (or `FS` when
    /// `None`). Clears `arr` first; returns the field count. No-op default → 0.
    fn split(&mut self, s: &Value, arr_name: &str, fs: Option<&Value>) -> i64 {
        let _ = (s, arr_name, fs);
        0
    }

    /// `sub(re, repl, target)` — replace the first `re` match in the value of
    /// `target` (named by `target_ref`) with `repl`; writes back. Returns the
    /// number of substitutions (0 or 1). No-op default → 0.
    fn sub(&mut self, re: &Value, repl: &Value, target_ref: &AwkLvalue) -> i64 {
        let _ = (re, repl, target_ref);
        0
    }

    /// `gsub(re, repl, target)` — like [`AwkHost::sub`] but global. Returns the
    /// number of substitutions. No-op default → 0.
    fn gsub(&mut self, re: &Value, repl: &Value, target_ref: &AwkLvalue) -> i64 {
        let _ = (re, repl, target_ref);
        0
    }

    /// `match(s, re)` — set `RSTART`/`RLENGTH` and return `RSTART` (1-based, or
    /// `0` with `RLENGTH = -1` when no match). Default → 0.
    fn match_re(&mut self, s: &Value, re: &Value) -> i64 {
        let _ = (s, re);
        0
    }

    /// `gensub(re, repl, how, target)` — return `target` (or `$0` when `target`
    /// is `None`) with `re` matches replaced per `how`, expanding `&`/`\N`
    /// backrefs. Host-bound: regex compilation honors `IGNORECASE` and the
    /// 3-arg form reads `$0`. No-op default → empty string.
    fn gensub(&mut self, re: &Value, repl: &Value, how: &Value, target: Option<&Value>) -> Value {
        let _ = (re, repl, how, target);
        Value::str("")
    }

    /// `tolower(s)`.
    fn tolower(&mut self, s: &Value) -> Value {
        awk_tolower(s)
    }

    /// `toupper(s)`.
    fn toupper(&mut self, s: &Value) -> Value {
        awk_toupper(s)
    }

    // ── Numeric builtins (pure f64; host-independent native defaults) ────────

    /// `int(x)` — truncate toward zero.
    fn int(&mut self, x: &Value) -> Value {
        awk_int(x)
    }

    /// `sqrt(x)`.
    fn sqrt(&mut self, x: &Value) -> Value {
        Value::Float(x.to_float().sqrt())
    }

    /// `sin(x)` — radians.
    fn sin(&mut self, x: &Value) -> Value {
        Value::Float(awk_canon_nan(x.to_float().sin()))
    }

    /// `cos(x)` — radians.
    fn cos(&mut self, x: &Value) -> Value {
        Value::Float(awk_canon_nan(x.to_float().cos()))
    }

    /// `exp(x)` — e^x.
    fn exp(&mut self, x: &Value) -> Value {
        Value::Float(awk_canon_nan(x.to_float().exp()))
    }

    /// `log(x)` — natural log.
    fn log(&mut self, x: &Value) -> Value {
        Value::Float(x.to_float().ln())
    }

    /// `atan2(y, x)`.
    fn atan2(&mut self, y: &Value, x: &Value) -> Value {
        Value::Float(awk_canon_nan(y.to_float().atan2(x.to_float())))
    }

    // ── Bitwise builtins (gawk; pure integer math, host-independent) ─────────

    /// `and(v1, v2, ...)` — bitwise AND of ≥2 operands.
    fn and(&mut self, args: &[Value]) -> Value {
        Value::Int(awk_fold_and(args))
    }

    /// `or(v1, v2, ...)` — bitwise OR of ≥2 operands.
    fn or(&mut self, args: &[Value]) -> Value {
        Value::Int(awk_fold_or(args))
    }

    /// `xor(v1, v2, ...)` — bitwise XOR of ≥2 operands.
    fn xor(&mut self, args: &[Value]) -> Value {
        Value::Int(awk_fold_xor(args))
    }

    /// `compl(v)` — bitwise complement.
    fn compl(&mut self, v: &Value) -> Value {
        Value::Int(awk_compl(v))
    }

    /// `lshift(v, n)` — left shift by `n & 0x3f` bits.
    fn lshift(&mut self, v: &Value, n: &Value) -> Value {
        Value::Int(awk_lshift(v, n))
    }

    /// `rshift(v, n)` — right shift by `n & 0x3f` bits.
    fn rshift(&mut self, v: &Value, n: &Value) -> Value {
        Value::Int(awk_rshift(v, n))
    }

    // ── Conversion builtins (gawk; pure string→number parse, host-free) ─────

    /// `strtonum(s)` — parse `0x…` hex, `0…` octal, else longest decimal/float
    /// prefix. Returns a number (POSIX-numeric float, like awkrs).
    fn strtonum(&mut self, s: &Value) -> Value {
        Value::Float(awk_strtonum(&s.to_str()))
    }

    // ── Time builtins (gawk; only the dep-free clock read is native) ────────

    /// `systime()` — seconds since the Unix epoch.
    fn systime(&mut self) -> Value {
        Value::Float(awk_systime())
    }

    // ── Date/time formatting (gawk; chrono-backed, host-free) ───────────────

    /// `strftime([fmt [, ts [, utc]]])` — format a timestamp via `chrono`.
    fn strftime(&mut self, args: &[Value]) -> Value {
        awk_strftime(args)
    }

    /// `mktime(datespec [, utc])` — parse a `"YYYY MM DD HH MM SS"` datespec to
    /// epoch seconds (or -1).
    fn mktime(&mut self, args: &[Value]) -> Value {
        awk_mktime(args)
    }

    // ── Character / scalar builtins (gawk; pure on `Value`, host-free) ───────

    /// `ord(s)` — Unicode scalar value of the first character (0 if empty).
    fn ord(&mut self, arg: &Value) -> Value {
        awk_ord(arg)
    }

    /// `chr(n)` — single-character string for codepoint `n` (empty if invalid).
    fn chr(&mut self, arg: &Value) -> Value {
        awk_chr(arg)
    }

    /// `mkbool(x)` — `1` if `x` is truthy, else `0`.
    fn mkbool(&mut self, arg: &Value) -> Value {
        awk_mkbool(arg)
    }

    /// `intdiv(a, b)` — truncating integer quotient (Undef on divide-by-zero).
    fn intdiv(&mut self, a: &Value, b: &Value) -> Value {
        awk_intdiv(a, b)
    }

    /// `intdiv0(a, b)` — truncating integer quotient (0 on divide-by-zero).
    fn intdiv0(&mut self, a: &Value, b: &Value) -> Value {
        awk_intdiv0(a, b)
    }

    // ── Associative arrays (string keys, SUBSEP-joined subscripts) ──────────

    /// `arr[key]` — read element, auto-vivifying to `""` per POSIX. Default → "".
    fn array_get(&mut self, arr_name: &str, key: &Value) -> Value {
        let _ = (arr_name, key);
        Value::str("")
    }

    /// `arr[key] = v` — assign element. No-op default.
    fn array_set(&mut self, arr_name: &str, key: &Value, v: Value) {
        let _ = (arr_name, key, v);
    }

    /// `(key in arr)` — membership test without vivifying. Default → false.
    fn array_exists(&mut self, arr_name: &str, key: &Value) -> bool {
        let _ = (arr_name, key);
        false
    }

    /// `delete arr[key]`. No-op default.
    fn array_delete(&mut self, arr_name: &str, key: &Value) {
        let _ = (arr_name, key);
    }

    /// `delete arr` — remove every element. No-op default.
    fn array_clear(&mut self, arr_name: &str) {
        let _ = arr_name;
    }

    /// `length(arr)` — element count. Default → 0.
    fn array_len(&mut self, arr_name: &str) -> i64 {
        let _ = arr_name;
        0
    }

    // ── Value semantics ────────────────────────────────────────────────────

    /// POSIX numeric-string aware comparison of two AWK values. The default
    /// uses `fusevm::Value`'s coercions (numeric when both look numeric, else
    /// string), which is adequate for plain constants; frontends with full
    /// numeric-string tracking should override to honor field/`getline`
    /// provenance.
    fn compare(&mut self, a: &Value, b: &Value) -> Ordering {
        let (sa, sb) = (a.to_str(), b.to_str());
        let looks_num = |s: &str| s.trim().parse::<f64>().is_ok() || s.trim().is_empty();
        if looks_num(&sa) && looks_num(&sb) {
            a.to_float()
                .partial_cmp(&b.to_float())
                .unwrap_or(Ordering::Equal)
        } else {
            sa.cmp(&sb)
        }
    }
}

/// Identifies the assignable target of `sub`/`gsub` so the host can write the
/// result back to the right place.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AwkLvalue {
    /// A scalar variable, by name.
    Var(String),
    /// A field, by index (`$0` for the whole record).
    Field(i64),
    /// An array element: array name + already-joined subscript key.
    ArrayElem(String, String),
}

/// A no-op [`AwkHost`] using every default. Useful for tests and for running
/// AWK-targeting chunks whose AWK ops should be inert (the non-AWK universal
/// ops still execute normally).
#[derive(Debug, Default, Clone, Copy)]
pub struct DefaultAwkHost;

impl AwkHost for DefaultAwkHost {}

// ── Host-independent AWK string builtins ────────────────────────────────────
//
// These five operate purely on `fusevm::Value` and need none of AWK's host-side
// runtime state (fields, special vars, arrays, regex, I/O). They back both the
// [`AwkHost`] trait defaults and the VM's no-host fast path, so the
// corresponding ops execute natively even when no frontend host is registered.
// (`length($0)` still needs the host — only the scalar `length(s)` form is here.)

/// `length(s)` — character count. `None` ⇒ `length($0)` (host-only) ⇒ `0` here.
pub fn awk_length(s: Option<&Value>) -> i64 {
    s.map(|v| v.to_str().chars().count() as i64).unwrap_or(0)
}

/// `substr(s, m [, n])` — 1-based substring with POSIX clamping. `n` is the
/// length; `None` means "to the end of the string".
pub fn awk_substr(s: &Value, m: i64, n: Option<i64>) -> Value {
    let text: Vec<char> = s.to_str().chars().collect();
    let len = text.len() as i64;
    // POSIX: characters from position m..m+n, 1-based, clamped to [1, len+1).
    let start = m.max(1);
    let end = match n {
        Some(n) => (m + n).min(len + 1),
        None => len + 1,
    };
    if start >= end || start > len {
        return Value::str("");
    }
    let s0 = (start - 1) as usize;
    let s1 = (end - 1).min(len) as usize;
    Value::str(text[s0..s1].iter().collect::<String>())
}

/// `index(s, t)` — 1-based char index of `t` in `s`, or `0` when absent.
pub fn awk_index(s: &Value, t: &Value) -> i64 {
    let hay = s.to_str();
    let needle = t.to_str();
    match hay.find(&needle) {
        // byte offset → 1-based char position
        Some(byte_off) => hay[..byte_off].chars().count() as i64 + 1,
        None => 0,
    }
}

/// `tolower(s)`.
pub fn awk_tolower(s: &Value) -> Value {
    Value::str(s.to_str().to_lowercase())
}

/// `toupper(s)`.
pub fn awk_toupper(s: &Value) -> Value {
    Value::str(s.to_str().to_uppercase())
}

/// `int(x)` — truncate toward zero. Returns an `Int` when the truncated value
/// fits `i64`, else a truncated `Float` (preserves magnitude beyond i64 range).
pub fn awk_int(x: &Value) -> Value {
    let t = x.to_float().trunc();
    if t.is_finite() && t >= i64::MIN as f64 && t <= i64::MAX as f64 {
        Value::Int(t as i64)
    } else {
        Value::Float(t)
    }
}

/// Canonicalize a NaN result to a positive NaN, matching awkrs/gawk's `+nan`
/// (awkrs normalizes the sign in vm_builtins; platform libm may return `-nan`
/// for some non-finite transcendental inputs). Finite/inf results pass through.
#[inline]
pub fn awk_canon_nan(r: f64) -> f64 {
    if r.is_nan() {
        f64::NAN
    } else {
        r
    }
}

/// Truncate a number to a `u64` bit-pattern, matching gawk's bitwise-operand
/// coercion (`n.trunc() as i64 as u64`).
#[inline]
fn awk_to_u64(n: f64) -> u64 {
    n.trunc() as i64 as u64
}

/// `and(args...)` — bitwise AND fold over ≥2 operands. Empty ⇒ 0.
pub fn awk_fold_and(args: &[Value]) -> i64 {
    args.iter()
        .map(|v| awk_to_u64(v.to_float()))
        .reduce(|a, b| a & b)
        .unwrap_or(0) as i64
}

/// `or(args...)` — bitwise OR fold.
pub fn awk_fold_or(args: &[Value]) -> i64 {
    args.iter()
        .map(|v| awk_to_u64(v.to_float()))
        .reduce(|a, b| a | b)
        .unwrap_or(0) as i64
}

/// `xor(args...)` — bitwise XOR fold.
pub fn awk_fold_xor(args: &[Value]) -> i64 {
    args.iter()
        .map(|v| awk_to_u64(v.to_float()))
        .reduce(|a, b| a ^ b)
        .unwrap_or(0) as i64
}

/// `compl(v)` — bitwise complement.
pub fn awk_compl(v: &Value) -> i64 {
    (!awk_to_u64(v.to_float())) as i64
}

/// `lshift(v, n)` — left shift by `n & 0x3f` bits.
pub fn awk_lshift(v: &Value, n: &Value) -> i64 {
    let x = awk_to_u64(v.to_float());
    let s = (awk_to_u64(n.to_float()) & 0x3f) as u32;
    (x << s) as i64
}

/// `rshift(v, n)` — right shift by `n & 0x3f` bits.
pub fn awk_rshift(v: &Value, n: &Value) -> i64 {
    let x = awk_to_u64(v.to_float());
    let s = (awk_to_u64(n.to_float()) & 0x3f) as u32;
    (x >> s) as i64
}

// ── Conversion builtins (gawk; pure string→number parse, host-free) ─────────

/// gawk `strtonum` — `0x…` hex, `0…` octal (both only without a sign), else the
/// longest leading decimal/float prefix. Ported faithfully from awkrs
/// (`builtins::awk_strtonum`, `runtime::longest_f64_prefix`).
pub fn awk_strtonum(s: &str) -> f64 {
    let t = s.trim();
    if t.is_empty() {
        return 0.0;
    }
    // gawk: a leading byte that is not sign/dot/digit (e.g. bare "nan"/"inf")
    // disqualifies the whole token.
    let first = t.as_bytes()[0];
    if !matches!(first, b'+' | b'-' | b'.' | b'0'..=b'9') {
        return 0.0;
    }
    // gawk: `0x…` / `0…` octal prefixes are only honored without a sign.
    let unsigned_hex_or_octal = !matches!(first, b'+' | b'-');
    if unsigned_hex_or_octal {
        if t.starts_with("0x") || t.starts_with("0X") {
            return u64::from_str_radix(&t[2..], 16)
                .map(|v| v as f64)
                .unwrap_or(0.0);
        }
        if t.len() > 1
            && t.starts_with('0')
            && !t.contains('.')
            && !t.contains('e')
            && !t.contains('E')
        {
            return i64::from_str_radix(t, 8).map(|v| v as f64).unwrap_or(0.0);
        }
    }
    // Longest leading numeric prefix (so "42abc" → 42).
    if let Some(prefix) = awk_longest_f64_prefix(t) {
        if let Ok(v) = prefix.parse::<f64>() {
            return v;
        }
    }
    0.0
}

/// Longest leading substring of `s` that parses as `f64` under gawk's numeric
/// coercion rule. Ported from awkrs `runtime::longest_f64_prefix`.
fn awk_longest_f64_prefix(s: &str) -> Option<&str> {
    if s.is_empty() {
        return None;
    }
    for end in (1..=s.len()).rev() {
        let p = &s[..end];
        if !awk_numeric_prefix_acceptable(p, end < s.len() && awk_next_byte_is_alnum(s, end)) {
            continue;
        }
        if p.parse::<f64>().is_ok() {
            return Some(p);
        }
    }
    None
}

#[inline]
fn awk_next_byte_is_alnum(s: &str, end: usize) -> bool {
    s.as_bytes()
        .get(end)
        .map(|c| c.is_ascii_alphanumeric())
        .unwrap_or(false)
}

/// gawk numeric coercion: accept decimal/float prefixes (must contain a digit),
/// plus signed three-letter `inf`/`nan` that stands alone. Ported from awkrs
/// `runtime::awk_numeric_prefix_acceptable`.
#[inline]
fn awk_numeric_prefix_acceptable(p: &str, has_trailing_alnum: bool) -> bool {
    if p.bytes().any(|c| c.is_ascii_digit()) {
        return true;
    }
    if has_trailing_alnum {
        return false;
    }
    let b = p.as_bytes();
    if b.len() != 4 {
        return false;
    }
    if !matches!(b[0], b'+' | b'-') {
        return false;
    }
    let tail = &p[1..];
    tail.eq_ignore_ascii_case("inf") || tail.eq_ignore_ascii_case("nan")
}

// ── Time builtins (gawk; only the dep-free clock read is native) ────────────

/// gawk `systime` — seconds since the Unix epoch as a float. Ported faithfully
/// from awkrs (`builtins::awk_systime`). Pure `std::time`, zero deps.
pub fn awk_systime() -> f64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

// ── PRNG builtins (POSIX/gawk; glibc LCG over a VM-owned seed, host-free) ────
// The seed is VM-owned execution state (not AWK data-model state), so these are
// free fns over an `&mut u64` rather than `AwkHost` trait methods. Ported
// faithfully from awkrs (`Runtime::rand`/`Runtime::srand`, runtime.rs:2578).

/// `rand()` — advance the LCG and return the next value in `[0, 1)`.
pub fn awk_rand(seed: &mut u64) -> f64 {
    *seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
    f64::from((*seed >> 16) as u32 & 0x7fff) / 32768.0
}

/// `srand([x])` — reseed the PRNG, returning the previous seed (low 32 bits).
/// `n = None` seeds from the system clock (matching awkrs); `Some(v)` uses the
/// gawk non-bignum truncation `v as u32 as u64`.
pub fn awk_srand(seed: &mut u64, n: Option<u64>) -> f64 {
    let prev = *seed;
    *seed = n.unwrap_or_else(|| {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs() ^ (d.subsec_nanos() as u64))
            .unwrap_or(1)
    });
    (prev & 0xffff_ffff) as f64
}

// ── Date/time formatting (gawk; chrono-backed, host-free) ───────────────────
// Ported faithfully from awkrs (`builtins::awk_strftime`,
// `builtins::awk_mktime_with_utc`). These need timezone/locale data (system tz
// for the local path) but no AWK runtime state, so they execute natively.

/// gawk `strftime([format [, timestamp [, utc]]])`. On an out-of-range
/// timestamp the native path yields an empty string (a registered host may
/// instead raise gawk's fatal error).
pub fn awk_strftime(args: &[Value]) -> Value {
    // gawk default format (from PROCINFO["strftime"]): includes the timezone
    // abbreviation (`%Z`) and matches gawk's output, unlike `%c`.
    let default_fmt = "%a %b %e %H:%M:%S %Z %Y";
    let (fmt, ts, utc) = match args.len() {
        0 => (default_fmt.to_string(), awk_systime(), false),
        1 => (args[0].to_str(), awk_systime(), false),
        2 => (args[0].to_str(), args[1].to_float(), false),
        _ => (
            args[0].to_str(),
            args[1].to_float(),
            args[2].to_float() != 0.0,
        ),
    };
    let secs = ts.floor() as i64;
    let nsec = ((ts - secs as f64) * 1e9).round().clamp(0.0, 1e9 - 1.0) as u32;
    let out = if utc {
        match Utc.timestamp_opt(secs, nsec).single() {
            Some(dt) => dt.format(&fmt).to_string(),
            None => String::new(),
        }
    } else {
        match Local.timestamp_opt(secs, nsec).single() {
            Some(dt) => dt.format(&fmt).to_string(),
            None => String::new(),
        }
    };
    Value::str(out)
}

/// gawk `mktime(datespec [, utc])` — `"YYYY MM DD HH MM SS"` (whitespace-
/// separated). `utc` truthy interprets the datespec in UTC, else local time.
/// Returns `-1` for unparseable or out-of-range datespecs.
pub fn awk_mktime(args: &[Value]) -> Value {
    let s = args.first().map(|v| v.to_str()).unwrap_or_default();
    let utc = args.len() >= 2 && args[1].to_float() != 0.0;
    Value::Float(awk_mktime_with_utc(&s, utc))
}

fn awk_mktime_with_utc(s: &str, utc: bool) -> f64 {
    let parts: Vec<&str> = s.split_whitespace().collect();
    if parts.len() < 6 {
        return -1.0;
    }
    let y: i32 = match parts[0].parse() {
        Ok(v) => v,
        Err(_) => return -1.0,
    };
    let mo: u32 = match parts[1].parse() {
        Ok(v) => v,
        Err(_) => return -1.0,
    };
    let d: u32 = match parts[2].parse() {
        Ok(v) => v,
        Err(_) => return -1.0,
    };
    let h: u32 = match parts[3].parse() {
        Ok(v) => v,
        Err(_) => return -1.0,
    };
    let mi: u32 = match parts[4].parse() {
        Ok(v) => v,
        Err(_) => return -1.0,
    };
    let se: u32 = match parts[5].parse() {
        Ok(v) => v,
        Err(_) => return -1.0,
    };
    let naive = match NaiveDate::from_ymd_opt(y, mo, d) {
        Some(date) => match date.and_hms_opt(h, mi, se) {
            Some(n) => n,
            None => return -1.0,
        },
        None => return -1.0,
    };
    if utc {
        match Utc.from_local_datetime(&naive) {
            LocalResult::Single(dt) => dt.timestamp() as f64,
            LocalResult::Ambiguous(_, _) | LocalResult::None => -1.0,
        }
    } else {
        match Local.from_local_datetime(&naive) {
            LocalResult::Single(dt) => dt.timestamp() as f64,
            LocalResult::Ambiguous(_, _) | LocalResult::None => -1.0,
        }
    }
}

/// gawk `ord(s)` — Unicode scalar value of the first character of `s`'s string
/// form, or `0` if empty. Faithful port of `gawk_extensions::ord`.
pub fn awk_ord(arg: &Value) -> Value {
    let s = arg.to_str();
    let n = s.chars().next().map(|c| c as u32).unwrap_or(0);
    Value::Float(f64::from(n))
}

/// gawk `chr(n)` — single-character string for codepoint `n` (truncated toward
/// zero), or the empty string if `n` is not a valid Unicode scalar value.
/// Faithful port of `gawk_extensions::chr`.
pub fn awk_chr(arg: &Value) -> Value {
    let u = arg.to_float() as u32;
    match char::from_u32(u) {
        Some(c) => Value::str(c.to_string()),
        None => Value::str(""),
    }
}

/// gawk `mkbool(x)` — `1.0` if `x` is truthy under fusevm's value semantics,
/// else `0.0`. (A numeric-string-aware host may differ on string literals; see
/// `Value::is_truthy`.)
pub fn awk_mkbool(arg: &Value) -> Value {
    Value::Float(if arg.is_truthy() { 1.0 } else { 0.0 })
}

/// awk `intdiv(a, b)` — truncating integer quotient (non-bignum path). Division
/// by zero yields `Undef`, matching fusevm's native `Div`; a host may override
/// to raise the gawk fatal "division by zero". Faithful port of
/// `bignum::awk_intdiv_values` (non-bignum branch).
pub fn awk_intdiv(a: &Value, b: &Value) -> Value {
    let bf = b.to_float();
    if bf == 0.0 {
        return Value::Undef;
    }
    let ai = a.to_float() as i64;
    let bi = bf as i64;
    if bi == 0 {
        return Value::Undef;
    }
    Value::Float((ai / bi) as f64)
}

/// gawk `intdiv0(a, b)` — like [`awk_intdiv`], but division by zero yields `0.0`
/// instead of `Undef` (the "safe" gawk variant; never errors). Faithful port of
/// `gawk_extensions::intdiv0` (non-bignum branch).
pub fn awk_intdiv0(a: &Value, b: &Value) -> Value {
    let bf = b.to_float();
    if bf == 0.0 {
        return Value::Float(0.0);
    }
    let bi = bf as i64;
    if bi == 0 {
        return Value::Float(0.0);
    }
    Value::Float((a.to_float() as i64 / bi) as f64)
}
