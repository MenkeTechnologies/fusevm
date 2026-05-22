//! Additional coverage for `Value` coercion methods, len(), is_empty(),
//! as_str_cow ownership semantics, and Ref/NativeFn variants.

use fusevm::Value;
use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::Arc;

// ── truthiness exhaustive ──────────────────────────────────────────────────

#[test]
fn truthiness_bool_variants() {
    assert!(Value::Bool(true).is_truthy());
    assert!(!Value::Bool(false).is_truthy());
}

#[test]
fn truthiness_str_zero_is_false() {
    // Documented: "0" is falsy (shell semantics).
    assert!(!Value::str("0").is_truthy());
    assert!(Value::str("00").is_truthy());
    assert!(Value::str(" ").is_truthy());
    assert!(Value::str("false").is_truthy());
}

#[test]
fn truthiness_ref_always_true() {
    assert!(Value::Ref(Box::new(Value::Int(0))).is_truthy());
    assert!(Value::Ref(Box::new(Value::Undef)).is_truthy());
    assert!(Value::Ref(Box::new(Value::str(""))).is_truthy());
}

#[test]
fn truthiness_native_fn_always_true() {
    assert!(Value::NativeFn(0).is_truthy());
    assert!(Value::NativeFn(u16::MAX).is_truthy());
}

#[test]
fn truthiness_status_nonzero_is_false() {
    assert!(!Value::Status(1).is_truthy());
    assert!(!Value::Status(-1).is_truthy());
    assert!(!Value::Status(255).is_truthy());
    assert!(Value::Status(0).is_truthy());
}

#[test]
fn truthiness_hash_with_entries() {
    let mut m = HashMap::new();
    m.insert("k".into(), Value::Undef);
    assert!(Value::hash(m).is_truthy());
}

// ── to_int edges ───────────────────────────────────────────────────────────

#[test]
fn to_int_string_parses_leading_whitespace_fails() {
    // i64::from_str is strict — leading whitespace fails → 0.
    assert_eq!(Value::str("  42").to_int(), 0);
    assert_eq!(Value::str("42  ").to_int(), 0);
}

#[test]
fn to_int_string_negative() {
    assert_eq!(Value::str("-7").to_int(), -7);
}

#[test]
fn to_int_string_float_form_fails() {
    // "3.14" doesn't parse as i64 — returns 0.
    assert_eq!(Value::str("3.14").to_int(), 0);
}

#[test]
fn to_int_ref_returns_zero() {
    // Documented fallback: 0 for unsupported variants.
    assert_eq!(Value::Ref(Box::new(Value::Int(99))).to_int(), 0);
}

#[test]
fn to_int_native_fn_returns_zero() {
    assert_eq!(Value::NativeFn(5).to_int(), 0);
}

#[test]
fn to_int_hash_returns_zero() {
    let mut m = HashMap::new();
    m.insert("k".into(), Value::Int(1));
    assert_eq!(Value::hash(m).to_int(), 0);
}

#[test]
fn to_int_huge_float_clamps() {
    // f64 → i64 saturating cast since Rust 1.45.
    let big = 1e30_f64;
    let result = Value::Float(big).to_int();
    assert_eq!(result, i64::MAX);
    let small = -1e30_f64;
    let result = Value::Float(small).to_int();
    assert_eq!(result, i64::MIN);
}

#[test]
fn to_int_nan_float_is_zero() {
    // saturating cast: NaN → 0.
    assert_eq!(Value::Float(f64::NAN).to_int(), 0);
}

// ── to_float edges ─────────────────────────────────────────────────────────

#[test]
fn to_float_array_returns_zero() {
    assert_eq!(Value::array(vec![Value::Int(1)]).to_float(), 0.0);
}

#[test]
fn to_float_undef_returns_zero() {
    assert_eq!(Value::Undef.to_float(), 0.0);
}

#[test]
fn to_float_negative_scientific_notation() {
    assert_eq!(Value::str("-1.5e2").to_float(), -150.0);
}

#[test]
fn to_float_inf_string() {
    assert!(Value::str("inf").to_float().is_infinite());
}

// ── to_str / as_str_cow ────────────────────────────────────────────────────

#[test]
fn to_str_undef_is_empty() {
    assert_eq!(Value::Undef.to_str(), "");
}

#[test]
fn to_str_bool_true_is_one() {
    assert_eq!(Value::Bool(true).to_str(), "1");
    assert_eq!(Value::Bool(false).to_str(), "");
}

#[test]
fn to_str_array_joins_with_space() {
    let a = Value::array(vec![Value::Int(1), Value::str("x"), Value::Int(3)]);
    assert_eq!(a.to_str(), "1 x 3");
}

#[test]
fn to_str_nested_array() {
    let inner = Value::array(vec![Value::Int(1), Value::Int(2)]);
    let outer = Value::array(vec![Value::Int(0), inner, Value::Int(9)]);
    assert_eq!(outer.to_str(), "0 1 2 9");
}

#[test]
fn to_str_hash_placeholder() {
    let mut m = HashMap::new();
    m.insert("k".into(), Value::Int(1));
    assert_eq!(Value::hash(m).to_str(), "(hash)");
}

#[test]
fn to_str_ref_placeholder() {
    assert_eq!(Value::Ref(Box::new(Value::Int(7))).to_str(), "(ref)");
}

#[test]
fn to_str_native_fn_formatted() {
    assert_eq!(Value::NativeFn(42).to_str(), "(builtin:42)");
}

#[test]
fn to_str_status_is_decimal() {
    assert_eq!(Value::Status(127).to_str(), "127");
    assert_eq!(Value::Status(-1).to_str(), "-1");
}

#[test]
fn as_str_cow_borrowed_for_bool() {
    match Value::Bool(true).as_str_cow() {
        Cow::Borrowed("1") => {}
        other => panic!("expected Borrowed(\"1\"), got {:?}", other),
    }
    match Value::Bool(false).as_str_cow() {
        Cow::Borrowed("") => {}
        other => panic!("expected Borrowed(\"\"), got {:?}", other),
    }
}

#[test]
fn as_str_cow_borrowed_for_undef() {
    match Value::Undef.as_str_cow() {
        Cow::Borrowed("") => {}
        other => panic!("expected Borrowed empty, got {:?}", other),
    }
}

#[test]
fn as_str_cow_owned_for_float() {
    match Value::Float(2.5).as_str_cow() {
        Cow::Owned(s) => assert_eq!(s, "2.5"),
        Cow::Borrowed(_) => panic!("expected Owned"),
    }
}

#[test]
fn as_str_cow_borrowed_for_hash_placeholder() {
    let m = HashMap::new();
    match Value::hash(m).as_str_cow() {
        Cow::Borrowed("(hash)") => {}
        other => panic!("expected hash placeholder, got {:?}", other),
    }
}

// ── len / is_empty ─────────────────────────────────────────────────────────

#[test]
fn len_str_uses_byte_length() {
    assert_eq!(Value::str("").len(), 0);
    assert_eq!(Value::str("abc").len(), 3);
    // 6 UTF-8 bytes for "héllo": h=1,é=2,l=1,l=1,o=1 → 6
    assert_eq!(Value::str("héllo").len(), 6);
}

#[test]
fn len_array_counts_elements() {
    assert_eq!(Value::array(vec![]).len(), 0);
    assert_eq!(Value::array(vec![Value::Undef; 7]).len(), 7);
}

#[test]
fn len_hash_counts_entries() {
    let mut m = HashMap::new();
    m.insert("a".into(), Value::Int(1));
    m.insert("b".into(), Value::Int(2));
    assert_eq!(Value::hash(m).len(), 2);
}

#[test]
fn len_int_falls_back_to_str_len() {
    // Fallback path: to_str().len()
    assert_eq!(Value::Int(123).len(), 3);
    assert_eq!(Value::Int(-1).len(), 2);
}

#[test]
fn len_undef_is_zero() {
    assert_eq!(Value::Undef.len(), 0);
    assert!(Value::Undef.is_empty());
}

#[test]
fn is_empty_consistent_with_len() {
    assert!(Value::str("").is_empty());
    assert!(!Value::str("x").is_empty());
    assert!(Value::array(vec![]).is_empty());
    assert!(!Value::Int(1).is_empty()); // "1" len=1
}

// ── Constructors ───────────────────────────────────────────────────────────

#[test]
fn str_constructor_accepts_both_str_and_string() {
    let from_str = Value::str("hi");
    let from_string = Value::str(String::from("hi"));
    assert_eq!(from_str, from_string);
}

#[test]
fn arc_share_does_not_panic_with_clone() {
    let s = Value::str("shared");
    let cloned = s.clone();
    if let (Value::Str(a), Value::Str(b)) = (&s, &cloned) {
        // Cloning a Value::Str should share the Arc, not deep-copy.
        assert!(Arc::ptr_eq(a, b));
    } else {
        panic!("expected both Str");
    }
}

// ── Ref / Boxed semantics ──────────────────────────────────────────────────

#[test]
fn ref_inner_value_accessible_via_pattern() {
    let r = Value::Ref(Box::new(Value::Int(42)));
    if let Value::Ref(inner) = r {
        assert_eq!(*inner, Value::Int(42));
    } else {
        panic!("expected Ref");
    }
}

#[test]
fn nested_refs_compose() {
    let r = Value::Ref(Box::new(Value::Ref(Box::new(Value::str("deep")))));
    if let Value::Ref(outer) = r {
        if let Value::Ref(inner) = *outer {
            assert_eq!(*inner, Value::str("deep"));
        } else {
            panic!();
        }
    } else {
        panic!();
    }
}
