//! Language-agnostic value system for fusevm.
//!
//! Every value in the VM is a `Value`. Frontends (stryke, zshrs, etc.)
//! convert their native types to/from `Value` at the boundary.

use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

/// Core value type — what lives on the stack and in variables.
///
/// Designed to be small (1 word tag + 1-2 words payload) so the
/// dispatch loop stays cache-friendly.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub enum Value {
    /// No value / uninitialized
    #[default]
    Undef,
    /// Boolean (from conditionals, `[[ ]]`, etc.)
    Bool(bool),
    /// 64-bit signed integer
    Int(i64),
    /// 64-bit float
    Float(f64),
    /// Heap-allocated string (Arc for cheap clone in closures)
    Str(Arc<String>),
    /// Ordered array of values
    Array(Vec<Value>),
    /// Key-value associative array
    Hash(HashMap<String, Value>),
    /// Exit status code (shell-specific but universal enough)
    Status(i32),
    /// Reference to another value (for pass-by-reference, nested structures)
    Ref(Box<Value>),
    /// Native function pointer (builtin dispatch)
    NativeFn(u16),
}

impl Value {
    // ── Constructors ──

    /// Construct an integer `Value::Int(n)`.
    pub fn int(n: i64) -> Self {
        Value::Int(n)
    }

    /// Construct a float `Value::Float(f)`.
    pub fn float(f: f64) -> Self {
        Value::Float(f)
    }

    /// Construct a string `Value::Str` from any type that converts to `String`.
    /// The payload is wrapped in `Arc` so clones are cheap.
    pub fn str(s: impl Into<String>) -> Self {
        Value::Str(Arc::new(s.into()))
    }

    /// Construct a boolean `Value::Bool(b)`.
    pub fn bool(b: bool) -> Self {
        Value::Bool(b)
    }

    /// Construct an array `Value::Array(v)`.
    pub fn array(v: Vec<Value>) -> Self {
        Value::Array(v)
    }

    /// Construct a hash `Value::Hash(m)`.
    pub fn hash(m: HashMap<String, Value>) -> Self {
        Value::Hash(m)
    }

    /// Construct a `Value::Status(code)` — used for `$?` / pipeline-exit
    /// values so a numeric exit code stays distinguishable from plain
    /// `Value::Int(n)` in `is_truthy` / display logic.
    pub fn status(code: i32) -> Self {
        Value::Status(code)
    }

    // ── Coercions ──

    /// Truthiness: 0, 0.0, "", undef, empty array/hash are false.
    pub fn is_truthy(&self) -> bool {
        match self {
            Value::Undef => false,
            Value::Bool(b) => *b,
            Value::Int(n) => *n != 0,
            Value::Float(f) => *f != 0.0,
            Value::Str(s) => !s.is_empty() && s.as_str() != "0",
            Value::Array(a) => !a.is_empty(),
            Value::Hash(h) => !h.is_empty(),
            Value::Status(c) => *c == 0, // shell: 0 = success = true
            Value::Ref(_) => true,
            Value::NativeFn(_) => true,
        }
    }

    /// Coerce to i64.
    pub fn to_int(&self) -> i64 {
        match self {
            Value::Int(n) => *n,
            Value::Float(f) => *f as i64,
            Value::Bool(b) => *b as i64,
            Value::Str(s) => s.parse().unwrap_or(0),
            Value::Status(c) => *c as i64,
            Value::Array(a) => a.len() as i64,
            _ => 0,
        }
    }

    /// Coerce to f64.
    pub fn to_float(&self) -> f64 {
        match self {
            Value::Float(f) => *f,
            Value::Int(n) => *n as f64,
            Value::Bool(b) if *b => 1.0,
            Value::Str(s) => s.parse().unwrap_or(0.0),
            Value::Status(c) => *c as f64,
            _ => 0.0,
        }
    }

    /// Coerce to string.
    pub fn to_str(&self) -> String {
        self.as_str_cow().into_owned()
    }

    /// Coerce to string, borrowing when possible to avoid allocation.
    /// Returns `Cow::Borrowed` for `Str`, `Undef`, `Bool`, `Hash`, `Ref` variants.
    pub fn as_str_cow(&self) -> Cow<'_, str> {
        match self {
            Value::Str(s) => Cow::Borrowed(s.as_str()),
            Value::Int(n) => Cow::Owned(n.to_string()),
            Value::Float(f) => Cow::Owned(f.to_string()),
            Value::Bool(b) => Cow::Borrowed(if *b { "1" } else { "" }),
            Value::Undef => Cow::Borrowed(""),
            Value::Status(c) => Cow::Owned(c.to_string()),
            Value::Array(a) => {
                Cow::Owned(a.iter().map(|v| v.to_str()).collect::<Vec<_>>().join(" "))
            }
            Value::Hash(_) => Cow::Borrowed("(hash)"),
            Value::Ref(_) => Cow::Borrowed("(ref)"),
            Value::NativeFn(id) => Cow::Owned(format!("(builtin:{})", id)),
        }
    }

    /// String length or array length or hash size.
    pub fn len(&self) -> usize {
        match self {
            Value::Str(s) => s.len(),
            Value::Array(a) => a.len(),
            Value::Hash(h) => h.len(),
            _ => self.to_str().len(),
        }
    }

    /// `len() == 0` shorthand — clippy requires this when `len` is `pub`.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Hash for Value {
    fn hash<H: Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            Value::Undef => {}
            Value::Bool(b) => b.hash(state),
            Value::Int(n) => n.hash(state),
            Value::Float(f) => f.to_bits().hash(state),
            Value::Str(s) => s.hash(state),
            Value::Array(a) => a.hash(state),
            Value::Hash(h) => {
                h.len().hash(state);
                for (k, v) in h {
                    k.hash(state);
                    v.hash(state);
                }
            }
            Value::Status(c) => c.hash(state),
            Value::Ref(b) => b.hash(state),
            Value::NativeFn(id) => id.hash(state),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_truthiness() {
        assert!(!Value::Undef.is_truthy());
        assert!(!Value::Int(0).is_truthy());
        assert!(Value::Int(1).is_truthy());
        assert!(!Value::str("").is_truthy());
        assert!(!Value::str("0").is_truthy());
        assert!(Value::str("hello").is_truthy());
        assert!(Value::Status(0).is_truthy()); // shell: 0 = success
        assert!(!Value::Status(1).is_truthy());
    }

    #[test]
    fn test_coercions() {
        assert_eq!(Value::str("42").to_int(), 42);
        assert_eq!(Value::Int(42).to_str(), "42");
        assert_eq!(Value::Float(3.25).to_int(), 3);
        assert_eq!(Value::Bool(true).to_int(), 1);
    }

    #[test]
    fn truthiness_for_floats() {
        assert!(!Value::Float(0.0).is_truthy());
        assert!(!Value::Float(-0.0).is_truthy());
        assert!(Value::Float(0.1).is_truthy());
        assert!(Value::Float(f64::NAN).is_truthy()); // NaN != 0.0 so truthy
        assert!(Value::Float(f64::INFINITY).is_truthy());
    }

    #[test]
    fn truthiness_for_collections() {
        assert!(!Value::Array(vec![]).is_truthy());
        assert!(Value::Array(vec![Value::Undef]).is_truthy()); // non-empty array
        assert!(!Value::Hash(HashMap::new()).is_truthy());
    }

    #[test]
    fn to_int_handles_all_variants() {
        assert_eq!(Value::Undef.to_int(), 0);
        assert_eq!(Value::Int(-7).to_int(), -7);
        assert_eq!(Value::Float(3.99).to_int(), 3); // truncation
        assert_eq!(Value::Float(-3.99).to_int(), -3); // truncation toward zero
        assert_eq!(Value::Bool(true).to_int(), 1);
        assert_eq!(Value::Bool(false).to_int(), 0);
        assert_eq!(Value::Status(42).to_int(), 42);
        assert_eq!(Value::Status(-1).to_int(), -1);
        assert_eq!(Value::str("not a number").to_int(), 0);
        assert_eq!(Value::Array(vec![Value::Int(1), Value::Int(2)]).to_int(), 2);
    }

    #[test]
    fn to_float_handles_all_variants() {
        assert_eq!(Value::Int(5).to_float(), 5.0);
        assert_eq!(Value::Float(2.5).to_float(), 2.5);
        assert_eq!(Value::Bool(true).to_float(), 1.0);
        assert_eq!(Value::Bool(false).to_float(), 0.0);
        assert_eq!(Value::Status(7).to_float(), 7.0);
        assert_eq!(Value::str("3.25").to_float(), 3.25);
        assert_eq!(Value::str("garbage").to_float(), 0.0);
    }

    #[test]
    fn as_str_cow_borrowed_for_str() {
        let v = Value::str("hello");
        match v.as_str_cow() {
            Cow::Borrowed(s) => assert_eq!(s, "hello"),
            Cow::Owned(_) => panic!("expected borrowed"),
        }
    }

    #[test]
    fn as_str_cow_owned_for_int() {
        match Value::Int(42).as_str_cow() {
            Cow::Owned(s) => assert_eq!(s, "42"),
            Cow::Borrowed(_) => panic!("expected owned"),
        }
    }

    #[test]
    fn array_to_str_joins_with_space() {
        let v = Value::Array(vec![Value::Int(1), Value::Int(2), Value::Int(3)]);
        assert_eq!(v.to_str(), "1 2 3");
    }

    #[test]
    fn len_returns_correct_size_per_variant() {
        assert_eq!(Value::str("abc").len(), 3);
        assert_eq!(Value::Array(vec![Value::Int(1); 5]).len(), 5);
        assert_eq!(Value::Hash(HashMap::new()).len(), 0);
        // Int falls through to to_str().len()
        assert_eq!(Value::Int(12345).len(), 5);
    }

    #[test]
    fn is_empty_matches_len_zero() {
        assert!(Value::str("").is_empty());
        assert!(!Value::str("x").is_empty());
        assert!(Value::Array(vec![]).is_empty());
        assert!(!Value::Array(vec![Value::Int(0)]).is_empty());
    }

    #[test]
    fn equality_via_partial_eq() {
        // PartialEq allows direct comparison — useful in tests.
        assert_eq!(Value::Int(42), Value::Int(42));
        assert_ne!(Value::Int(42), Value::Int(43));
        assert_eq!(Value::str("hi"), Value::str("hi"));
        // NaN != NaN per IEEE 754
        assert_ne!(Value::Float(f64::NAN), Value::Float(f64::NAN));
    }

    #[test]
    fn hash_impl_handles_floats_via_bits() {
        // Hash impl must be consistent: f64 hashed as bit-pattern.
        // Two NaN values with same bits hash equal even though NaN != NaN.
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let nan1 = Value::Float(f64::NAN);
        let nan2 = Value::Float(f64::NAN);
        let mut h1 = DefaultHasher::new();
        nan1.hash(&mut h1);
        let mut h2 = DefaultHasher::new();
        nan2.hash(&mut h2);
        assert_eq!(h1.finish(), h2.finish());
    }

    #[test]
    fn as_str_cow_for_undef_and_bool_is_borrowed() {
        assert!(matches!(Value::Undef.as_str_cow(), Cow::Borrowed("")));
        assert!(matches!(Value::Bool(true).as_str_cow(), Cow::Borrowed("1")));
        assert!(matches!(Value::Bool(false).as_str_cow(), Cow::Borrowed("")));
    }

    #[test]
    fn as_str_cow_for_status_and_float_is_owned() {
        match Value::Status(127).as_str_cow() {
            Cow::Owned(s) => assert_eq!(s, "127"),
            _ => panic!("expected owned"),
        }
        match Value::Float(1.5).as_str_cow() {
            Cow::Owned(s) => assert_eq!(s, "1.5"),
            _ => panic!("expected owned"),
        }
    }

    #[test]
    fn native_fn_to_str_formats_id() {
        assert_eq!(Value::NativeFn(42).to_str(), "(builtin:42)");
    }

    #[test]
    fn hash_to_str_is_placeholder() {
        let mut m = HashMap::new();
        m.insert("k".to_string(), Value::Int(1));
        assert_eq!(Value::Hash(m).to_str(), "(hash)");
    }

    #[test]
    fn ref_to_str_is_placeholder_and_is_truthy() {
        let r = Value::Ref(Box::new(Value::Int(0)));
        assert!(r.is_truthy(), "Ref is always truthy regardless of inner");
        assert_eq!(r.to_str(), "(ref)");
    }

    #[test]
    fn nested_array_to_str_recurses() {
        let inner = Value::Array(vec![Value::Int(1), Value::Int(2)]);
        let outer = Value::Array(vec![inner, Value::Int(3)]);
        // Inner array stringifies to "1 2", then outer joins with space.
        assert_eq!(outer.to_str(), "1 2 3");
    }

    #[test]
    fn to_int_from_negative_string_parses() {
        assert_eq!(Value::str("-123").to_int(), -123);
    }

    #[test]
    fn to_int_unhandled_variants_return_zero() {
        assert_eq!(Value::Hash(HashMap::new()).to_int(), 0);
        assert_eq!(Value::Ref(Box::new(Value::Int(99))).to_int(), 0);
        assert_eq!(Value::NativeFn(5).to_int(), 0);
    }

    #[test]
    fn to_float_unhandled_variants_return_zero() {
        assert_eq!(Value::Undef.to_float(), 0.0);
        assert_eq!(Value::Array(vec![Value::Int(1)]).to_float(), 0.0);
        assert_eq!(Value::Bool(false).to_float(), 0.0);
    }

    #[test]
    fn len_for_hash_counts_entries() {
        let mut m = HashMap::new();
        m.insert("a".to_string(), Value::Int(1));
        m.insert("b".to_string(), Value::Int(2));
        assert_eq!(Value::Hash(m).len(), 2);
    }

    #[test]
    fn constructors_produce_expected_variants() {
        assert!(matches!(Value::int(1), Value::Int(1)));
        assert!(matches!(Value::float(1.0), Value::Float(_)));
        assert!(matches!(Value::bool(true), Value::Bool(true)));
        assert!(matches!(Value::status(7), Value::Status(7)));
        assert!(matches!(Value::array(vec![]), Value::Array(_)));
    }

    #[test]
    fn str_constructor_accepts_string_and_str() {
        let from_str = Value::str("hi");
        let from_string = Value::str(String::from("hi"));
        assert_eq!(from_str, from_string);
    }

    #[test]
    fn clone_of_str_shares_arc() {
        // Arc<String> means clones are cheap and point to same allocation.
        let a = Value::str("hello");
        let b = a.clone();
        if let (Value::Str(sa), Value::Str(sb)) = (&a, &b) {
            assert!(Arc::ptr_eq(sa, sb));
        } else {
            panic!("expected Str variants");
        }
    }

    #[test]
    fn default_is_undef() {
        let v: Value = Default::default();
        assert_eq!(v, Value::Undef);
    }

    #[test]
    fn hash_distinguishes_variants_with_same_payload_bytes() {
        // Int(0) and Bool(false) and Status(0) all have payload 0, but their
        // discriminants must make the hashes differ.
        use std::collections::hash_map::DefaultHasher;
        let h = |v: &Value| {
            let mut hs = DefaultHasher::new();
            v.hash(&mut hs);
            hs.finish()
        };
        let a = h(&Value::Int(0));
        let b = h(&Value::Bool(false));
        let c = h(&Value::Status(0));
        assert_ne!(a, b);
        assert_ne!(b, c);
        assert_ne!(a, c);
    }

    #[test]
    fn hash_for_hashmap_value_is_deterministic_per_value() {
        // Hashing the SAME Value::Hash twice yields identical hashes even
        // though HashMap iteration order is unspecified — because the impl
        // accumulates contributions and discriminant.
        use std::collections::hash_map::DefaultHasher;
        let mut m = HashMap::new();
        m.insert("a".to_string(), Value::Int(1));
        m.insert("b".to_string(), Value::Int(2));
        let v = Value::Hash(m);
        let mut h1 = DefaultHasher::new();
        v.hash(&mut h1);
        let mut h2 = DefaultHasher::new();
        v.hash(&mut h2);
        assert_eq!(h1.finish(), h2.finish());
    }

    #[test]
    fn serde_roundtrip_preserves_value() {
        // Verify Value survives serialization without information loss.
        let cases = vec![
            Value::Undef,
            Value::Bool(true),
            Value::Int(-42),
            Value::Float(3.25),
            Value::str("hello"),
            Value::Status(127),
        ];
        for original in cases {
            let json = serde_json::to_string(&original).unwrap();
            let restored: Value = serde_json::from_str(&json).unwrap();
            assert_eq!(original, restored);
        }
    }

    // ─── Coercion table pins ─────────────────────────────────────────
    //
    // The to_int / to_float / is_truthy triples are the foundation
    // every fusevm-hosted language depends on. Pin the cross-type
    // matrix so a single-branch refactor can't silently change e.g.
    // `if @arr` semantics from "non-empty" to "always true".

    #[test]
    fn to_int_string_with_non_numeric_falls_back_to_zero() {
        assert_eq!(Value::str("nope").to_int(), 0);
    }

    #[test]
    fn to_int_array_returns_length_not_first_element() {
        // Perl-/awk-style numeric coercion of an array is its length.
        let v = Value::array(vec![Value::int(99), Value::int(100), Value::int(101)]);
        assert_eq!(v.to_int(), 3, "array's to_int must be length, not [0]");
    }

    #[test]
    fn to_int_bool_true_is_one_false_is_zero() {
        assert_eq!(Value::bool(true).to_int(), 1);
        assert_eq!(Value::bool(false).to_int(), 0);
    }

    #[test]
    fn to_int_status_preserves_exit_code() {
        // `$?` users grep for specific exit codes; coercion must
        // round-trip them, not collapse to 0/1.
        for code in [0_i32, 1, 2, 42, 127, 128, 255] {
            assert_eq!(Value::Status(code).to_int(), code as i64);
        }
    }

    #[test]
    fn to_int_float_truncates_toward_zero() {
        assert_eq!(Value::Float(3.9).to_int(), 3);
        assert_eq!(Value::Float(-3.9).to_int(), -3);
        assert_eq!(Value::Float(0.5).to_int(), 0);
    }

    #[test]
    fn to_float_string_non_numeric_falls_back_to_zero() {
        assert_eq!(Value::str("nope").to_float(), 0.0);
    }

    #[test]
    fn to_float_bool_only_true_is_one() {
        // The to_float branch for Bool is only the `true` arm; pin
        // that false drops through to the default 0.0.
        assert_eq!(Value::bool(true).to_float(), 1.0);
        assert_eq!(Value::bool(false).to_float(), 0.0);
    }

    #[test]
    fn is_truthy_empty_collections_are_false() {
        assert!(!Value::array(Vec::new()).is_truthy());
        assert!(!Value::hash(HashMap::new()).is_truthy());
        assert!(!Value::str("").is_truthy());
        assert!(!Value::int(0).is_truthy());
        assert!(!Value::Float(0.0).is_truthy());
        assert!(!Value::Undef.is_truthy());
    }

    #[test]
    fn is_truthy_single_element_array_is_true() {
        assert!(Value::array(vec![Value::int(0)]).is_truthy());
    }

    // ─── as_str_cow / to_str string-rep pins ─────────────────────────
    //
    // The string representation is hit on every `print` / interpolation
    // call. Drift here silently changes every script's output. Pin
    // each non-trivial branch.

    #[test]
    fn as_str_cow_bool_renders_as_perl_compatible() {
        // Perl convention: true → "1", false → "" (empty string).
        // gawk and Ruby differ; if a host language frontend swaps
        // these, downstream `length($flag)` checks break.
        assert_eq!(Value::bool(true).to_str(), "1");
        assert_eq!(Value::bool(false).to_str(), "");
    }

    #[test]
    fn as_str_cow_undef_renders_as_empty_string() {
        assert_eq!(Value::Undef.to_str(), "");
    }

    #[test]
    fn as_str_cow_array_joins_with_single_space() {
        // Pin the separator — Perl's default `$,` is empty but the
        // common host convention is single-space; printf("%s", @a)
        // semantics rely on this.
        let v = Value::array(vec![Value::int(1), Value::int(2), Value::int(3)]);
        assert_eq!(v.to_str(), "1 2 3");
    }

    #[test]
    fn as_str_cow_hash_renders_as_opaque_label() {
        // Hashes don't have a canonical ordering, so the host
        // chose `(hash)` as a label. Pin so a future refactor
        // doesn't accidentally start exposing internal state.
        let mut h = HashMap::new();
        h.insert("x".into(), Value::int(1));
        assert_eq!(Value::hash(h).to_str(), "(hash)");
    }

    #[test]
    fn as_str_cow_status_renders_as_exit_code_only() {
        // $? must stringify as the bare integer, not "exit:N" or
        // similar — shell scripts grep for the bare code.
        assert_eq!(Value::Status(0).to_str(), "0");
        assert_eq!(Value::Status(127).to_str(), "127");
        assert_eq!(Value::Status(-1).to_str(), "-1");
    }

    // ─── len() / is_empty() pins ─────────────────────────────────────

    #[test]
    fn len_of_str_is_byte_count_not_char_count() {
        // Pin `len()` as byte count for Str (Rust's String::len
        // semantic). If a refactor switches to char count, all
        // `length()` builtins drift on UTF-8 input.
        let v = Value::str("é"); // 2 UTF-8 bytes
        assert_eq!(v.len(), 2);
    }

    #[test]
    fn len_of_array_is_element_count() {
        let v = Value::array(vec![Value::int(1); 7]);
        assert_eq!(v.len(), 7);
    }

    #[test]
    fn len_of_int_falls_through_to_str_form() {
        assert_eq!(Value::int(12345).len(), 5);
        assert_eq!(Value::int(-99).len(), 3); // "-99"
    }

    #[test]
    fn is_empty_matches_len_zero_full_matrix() {
        assert!(Value::str("").is_empty());
        assert!(Value::array(Vec::new()).is_empty());
        assert!(Value::hash(HashMap::new()).is_empty());
        assert!(!Value::int(0).is_empty()); // "0" has len 1
        assert!(!Value::str("x").is_empty());
        assert!(Value::Undef.is_empty()); // Undef stringifies to ""
    }
}
