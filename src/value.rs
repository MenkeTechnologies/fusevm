//! Language-agnostic value system for fusevm.
//!
//! Every value in the VM is a `Value`. Frontends (stryke, zshrs, etc.)
//! convert their native types to/from `Value` at the boundary.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

/// Core value type — what lives on the stack and in variables.
///
/// Designed to be small (1 word tag + 1-2 words payload) so the
/// dispatch loop stays cache-friendly.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Value {
    /// No value / uninitialized
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

impl Default for Value {
    fn default() -> Self {
        Value::Undef
    }
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
            Value::Bool(b) => if *b { 1.0 } else { 0.0 },
            Value::Str(s) => s.parse().unwrap_or(0.0),
            Value::Status(c) => *c as f64,
            _ => 0.0,
        }
    }

    /// Coerce to string.
    pub fn to_str(&self) -> String {
        match self {
            Value::Str(s) => s.as_ref().clone(),
            Value::Int(n) => n.to_string(),
            Value::Float(f) => f.to_string(),
            Value::Bool(b) => if *b { "1".to_string() } else { "".to_string() },
            Value::Undef => String::new(),
            Value::Status(c) => c.to_string(),
            Value::Array(a) => a.iter().map(|v| v.to_str()).collect::<Vec<_>>().join(" "),
            Value::Hash(_) => "(hash)".to_string(),
            Value::Ref(_) => "(ref)".to_string(),
            Value::NativeFn(id) => format!("(builtin:{})", id),
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
