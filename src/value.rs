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

    pub fn int(n: i64) -> Self {
        Value::Int(n)
    }

    pub fn float(f: f64) -> Self {
        Value::Float(f)
    }

    pub fn str(s: impl Into<String>) -> Self {
        Value::Str(Arc::new(s.into()))
    }

    pub fn bool(b: bool) -> Self {
        Value::Bool(b)
    }

    pub fn array(v: Vec<Value>) -> Self {
        Value::Array(v)
    }

    pub fn hash(m: HashMap<String, Value>) -> Self {
        Value::Hash(m)
    }

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
        assert_eq!(Value::Float(3.14).to_int(), 3);
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
        assert_eq!(Value::str("3.14").to_float(), 3.14);
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
    fn serde_roundtrip_preserves_value() {
        // Verify Value survives serialization without information loss.
        let cases = vec![
            Value::Undef,
            Value::Bool(true),
            Value::Int(-42),
            Value::Float(3.14),
            Value::str("hello"),
            Value::Status(127),
        ];
        for original in cases {
            let json = serde_json::to_string(&original).unwrap();
            let restored: Value = serde_json::from_str(&json).unwrap();
            assert_eq!(original, restored);
        }
    }
}
