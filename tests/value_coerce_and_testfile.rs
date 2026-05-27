#![allow(clippy::approx_constant)]
//! Exhaustive coercion / truthiness / hash coverage for every `Value`
//! variant, plus `Op::TestFile` for every `file_test::*` constant against
//! known-existing and known-missing paths.

use fusevm::op::file_test;
use fusevm::{ChunkBuilder, Op, VMResult, Value, VM};
use std::collections::HashMap;

fn run(b: ChunkBuilder) -> Value {
    match VM::new(b.build()).run() {
        VMResult::Ok(v) => v,
        VMResult::Halted => Value::Undef,
        VMResult::Error(e) => panic!("unexpected VM error: {e}"),
    }
}

// ══════════════════════════════════════════════════════════════════════════
// Value::to_int — exhaustive per-variant behavior
// ══════════════════════════════════════════════════════════════════════════

#[test]
fn to_int_undef_is_zero() {
    assert_eq!(Value::Undef.to_int(), 0);
}

#[test]
fn to_int_int_is_identity() {
    assert_eq!(Value::Int(0).to_int(), 0);
    assert_eq!(Value::Int(i64::MAX).to_int(), i64::MAX);
    assert_eq!(Value::Int(i64::MIN).to_int(), i64::MIN);
    assert_eq!(Value::Int(-7).to_int(), -7);
}

#[test]
fn to_int_float_truncates_toward_zero() {
    assert_eq!(Value::Float(0.0).to_int(), 0);
    assert_eq!(Value::Float(3.9).to_int(), 3);
    assert_eq!(Value::Float(-3.9).to_int(), -3);
    assert_eq!(Value::Float(-0.5).to_int(), 0);
}

#[test]
fn to_int_bool_is_one_or_zero() {
    assert_eq!(Value::Bool(true).to_int(), 1);
    assert_eq!(Value::Bool(false).to_int(), 0);
}

#[test]
fn to_int_str_parses_or_zero() {
    assert_eq!(Value::str("42").to_int(), 42);
    assert_eq!(Value::str("-7").to_int(), -7);
    assert_eq!(Value::str("").to_int(), 0);
    assert_eq!(Value::str("not a number").to_int(), 0);
    assert_eq!(Value::str("3.14").to_int(), 0); // strict integer parse only
}

#[test]
fn to_int_status_is_code() {
    assert_eq!(Value::Status(0).to_int(), 0);
    assert_eq!(Value::Status(127).to_int(), 127);
    assert_eq!(Value::Status(-1).to_int(), -1);
}

#[test]
fn to_int_array_is_length() {
    assert_eq!(Value::Array(vec![]).to_int(), 0);
    assert_eq!(
        Value::Array(vec![Value::Int(1), Value::Int(2), Value::Int(3)]).to_int(),
        3
    );
}

#[test]
fn to_int_hash_is_zero() {
    let mut h = HashMap::new();
    h.insert("k".to_string(), Value::Int(1));
    assert_eq!(Value::Hash(h).to_int(), 0);
}

// ══════════════════════════════════════════════════════════════════════════
// Value::to_float
// ══════════════════════════════════════════════════════════════════════════

#[test]
fn to_float_int_is_widened() {
    assert_eq!(Value::Int(5).to_float(), 5.0);
    assert_eq!(Value::Int(-3).to_float(), -3.0);
}

#[test]
fn to_float_bool_true_is_one_false_is_zero() {
    assert_eq!(Value::Bool(true).to_float(), 1.0);
    assert_eq!(Value::Bool(false).to_float(), 0.0);
}

#[test]
fn to_float_str_parses_or_zero() {
    assert_eq!(Value::str("3.14").to_float(), 3.14);
    assert_eq!(Value::str("-2.5").to_float(), -2.5);
    assert_eq!(Value::str("garbage").to_float(), 0.0);
    assert_eq!(Value::str("").to_float(), 0.0);
}

#[test]
fn to_float_status_is_code() {
    assert_eq!(Value::Status(5).to_float(), 5.0);
}

#[test]
fn to_float_undef_array_hash_are_zero() {
    assert_eq!(Value::Undef.to_float(), 0.0);
    assert_eq!(Value::Array(vec![Value::Int(1)]).to_float(), 0.0);
    assert_eq!(Value::Hash(HashMap::new()).to_float(), 0.0);
}

#[test]
fn to_float_float_preserves_nan_and_infinity() {
    assert!(Value::Float(f64::NAN).to_float().is_nan());
    assert!(Value::Float(f64::INFINITY).to_float().is_infinite());
    assert!(Value::Float(f64::NEG_INFINITY).to_float().is_infinite());
}

// ══════════════════════════════════════════════════════════════════════════
// Value::as_str_cow
// ══════════════════════════════════════════════════════════════════════════

#[test]
fn as_str_cow_bool_true_is_string_one() {
    assert_eq!(Value::Bool(true).as_str_cow(), "1");
}

#[test]
fn as_str_cow_bool_false_is_empty_string() {
    assert_eq!(Value::Bool(false).as_str_cow(), "");
}

#[test]
fn as_str_cow_undef_is_empty_string() {
    assert_eq!(Value::Undef.as_str_cow(), "");
}

#[test]
fn as_str_cow_array_joins_with_space() {
    let v = Value::Array(vec![Value::Int(1), Value::Int(2), Value::Int(3)]);
    assert_eq!(v.as_str_cow(), "1 2 3");
}

#[test]
fn as_str_cow_empty_array_is_empty_string() {
    let v = Value::Array(vec![]);
    assert_eq!(v.as_str_cow(), "");
}

#[test]
fn as_str_cow_hash_is_literal_marker() {
    let v = Value::Hash(HashMap::new());
    assert_eq!(v.as_str_cow(), "(hash)");
}

#[test]
fn as_str_cow_status_is_decimal_code() {
    assert_eq!(Value::Status(0).as_str_cow(), "0");
    assert_eq!(Value::Status(127).as_str_cow(), "127");
    assert_eq!(Value::Status(-1).as_str_cow(), "-1");
}

#[test]
fn as_str_cow_nativefn_includes_id() {
    let s = Value::NativeFn(42).as_str_cow().into_owned();
    assert!(s.contains("42"));
    assert!(s.contains("builtin"));
}

#[test]
fn as_str_cow_int_renders_decimal() {
    assert_eq!(Value::Int(0).as_str_cow(), "0");
    assert_eq!(Value::Int(-5).as_str_cow(), "-5");
    assert_eq!(Value::Int(i64::MAX).as_str_cow(), i64::MAX.to_string());
}

// ══════════════════════════════════════════════════════════════════════════
// Value::is_truthy
// ══════════════════════════════════════════════════════════════════════════

#[test]
fn truthy_int_nonzero_true_zero_false() {
    assert!(Value::Int(1).is_truthy());
    assert!(Value::Int(-1).is_truthy());
    assert!(!Value::Int(0).is_truthy());
}

#[test]
fn truthy_float_nonzero_true_zero_false() {
    assert!(Value::Float(0.1).is_truthy());
    assert!(Value::Float(-0.1).is_truthy());
    assert!(!Value::Float(0.0).is_truthy());
}

#[test]
fn truthy_str_empty_or_zero_is_false() {
    assert!(!Value::str("").is_truthy());
    assert!(!Value::str("0").is_truthy());
    assert!(Value::str("0.0").is_truthy()); // not literally "0"
    assert!(Value::str(" ").is_truthy());
    assert!(Value::str("false").is_truthy()); // string "false" is truthy
}

#[test]
fn truthy_status_zero_only_is_true() {
    assert!(Value::Status(0).is_truthy());
    assert!(!Value::Status(1).is_truthy());
    assert!(!Value::Status(127).is_truthy());
    assert!(!Value::Status(-1).is_truthy());
}

#[test]
fn truthy_array_nonempty_true_empty_false() {
    assert!(Value::Array(vec![Value::Int(0)]).is_truthy());
    assert!(!Value::Array(vec![]).is_truthy());
}

#[test]
fn truthy_hash_nonempty_true_empty_false() {
    let mut h = HashMap::new();
    assert!(!Value::Hash(h.clone()).is_truthy());
    h.insert("k".to_string(), Value::Int(0));
    assert!(Value::Hash(h).is_truthy());
}

#[test]
fn truthy_undef_is_false() {
    assert!(!Value::Undef.is_truthy());
}

#[test]
fn truthy_nativefn_and_ref_are_true() {
    assert!(Value::NativeFn(0).is_truthy());
    assert!(Value::Ref(Box::new(Value::Int(0))).is_truthy());
}

// ══════════════════════════════════════════════════════════════════════════
// Value::len / is_empty
// ══════════════════════════════════════════════════════════════════════════

#[test]
fn len_string_is_byte_length() {
    assert_eq!(Value::str("").len(), 0);
    assert_eq!(Value::str("abc").len(), 3);
    assert_eq!(Value::str("héllo").len(), "héllo".len());
}

#[test]
fn len_array_is_element_count() {
    assert_eq!(Value::Array(vec![]).len(), 0);
    assert_eq!(Value::Array(vec![Value::Int(0); 5]).len(), 5);
}

#[test]
fn len_hash_is_entry_count() {
    let mut h = HashMap::new();
    h.insert("a".to_string(), Value::Int(1));
    h.insert("b".to_string(), Value::Int(2));
    assert_eq!(Value::Hash(h).len(), 2);
}

#[test]
fn len_int_falls_back_to_decimal_length() {
    assert_eq!(Value::Int(42).len(), 2);
    assert_eq!(Value::Int(-1).len(), 2);
}

#[test]
fn is_empty_consistent_with_len() {
    assert!(Value::str("").is_empty());
    assert!(!Value::str("x").is_empty());
    assert!(Value::Array(vec![]).is_empty());
    assert!(Value::Hash(HashMap::new()).is_empty());
}

// ══════════════════════════════════════════════════════════════════════════
// Value::Hash trait — discriminant separates variants
// ══════════════════════════════════════════════════════════════════════════

#[test]
fn hash_distinguishes_int_and_str_with_same_numeric_value() {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut a = DefaultHasher::new();
    Value::Int(42).hash(&mut a);
    let mut b = DefaultHasher::new();
    Value::str("42").hash(&mut b);
    assert_ne!(a.finish(), b.finish());
}

#[test]
fn hash_float_uses_to_bits_so_zero_and_negzero_differ() {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut a = DefaultHasher::new();
    Value::Float(0.0).hash(&mut a);
    let mut b = DefaultHasher::new();
    Value::Float(-0.0).hash(&mut b);
    assert_ne!(a.finish(), b.finish());
}

#[test]
fn hash_equal_arrays_produce_equal_hashes() {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let a = Value::Array(vec![Value::Int(1), Value::Int(2)]);
    let b = Value::Array(vec![Value::Int(1), Value::Int(2)]);
    let mut h1 = DefaultHasher::new();
    a.hash(&mut h1);
    let mut h2 = DefaultHasher::new();
    b.hash(&mut h2);
    assert_eq!(h1.finish(), h2.finish());
}

// ══════════════════════════════════════════════════════════════════════════
// Op::TestFile — every file_test::* constant
// ══════════════════════════════════════════════════════════════════════════

fn run_test_file(path: &str, test: u8) -> bool {
    let mut b = ChunkBuilder::new();
    let p = b.add_constant(Value::str(path));
    b.emit(Op::LoadConst(p), 1);
    b.emit(Op::TestFile(test), 1);
    match run(b) {
        Value::Bool(b) => b,
        Value::Undef => false,
        other => panic!("expected Bool/Undef, got {:?}", other),
    }
}

// Helper to pick an existing file. /etc/hosts is present on macOS and Linux.
const EXISTING_FILE: &str = "/etc/hosts";
const EXISTING_DIR: &str = "/tmp";
const MISSING_PATH: &str = "/tmp/__fusevm_definitely_not_here_xyz_42";

#[test]
fn test_file_exists_for_known_file_and_dir() {
    assert!(run_test_file(EXISTING_FILE, file_test::EXISTS));
    assert!(run_test_file(EXISTING_DIR, file_test::EXISTS));
}

#[test]
fn test_file_exists_false_for_missing_path() {
    assert!(!run_test_file(MISSING_PATH, file_test::EXISTS));
}

#[test]
fn test_file_is_file_for_regular_file_only() {
    assert!(run_test_file(EXISTING_FILE, file_test::IS_FILE));
    assert!(!run_test_file(EXISTING_DIR, file_test::IS_FILE));
    assert!(!run_test_file(MISSING_PATH, file_test::IS_FILE));
}

#[test]
fn test_file_is_dir_for_directory_only() {
    assert!(run_test_file(EXISTING_DIR, file_test::IS_DIR));
    assert!(!run_test_file(EXISTING_FILE, file_test::IS_DIR));
    assert!(!run_test_file(MISSING_PATH, file_test::IS_DIR));
}

#[test]
fn test_file_is_readable_writable_implies_exists() {
    // The VM implements IS_READABLE/IS_WRITABLE as `Path::exists()`.
    assert!(run_test_file(EXISTING_FILE, file_test::IS_READABLE));
    assert!(run_test_file(EXISTING_FILE, file_test::IS_WRITABLE));
    assert!(!run_test_file(MISSING_PATH, file_test::IS_READABLE));
    assert!(!run_test_file(MISSING_PATH, file_test::IS_WRITABLE));
}

#[test]
fn test_file_is_symlink_false_for_regular_file() {
    assert!(!run_test_file(EXISTING_FILE, file_test::IS_SYMLINK));
    assert!(!run_test_file(MISSING_PATH, file_test::IS_SYMLINK));
}

#[test]
fn test_file_is_executable_false_for_non_executable_file() {
    // /etc/hosts is normally readable but not executable.
    assert!(!run_test_file(EXISTING_FILE, file_test::IS_EXECUTABLE));
    assert!(!run_test_file(MISSING_PATH, file_test::IS_EXECUTABLE));
}

#[test]
fn test_file_is_nonempty_for_nonempty_file() {
    assert!(run_test_file(EXISTING_FILE, file_test::IS_NONEMPTY));
    assert!(!run_test_file(MISSING_PATH, file_test::IS_NONEMPTY));
}

#[test]
fn test_file_special_types_false_for_regular_file() {
    assert!(!run_test_file(EXISTING_FILE, file_test::IS_SOCKET));
    assert!(!run_test_file(EXISTING_FILE, file_test::IS_FIFO));
    assert!(!run_test_file(EXISTING_FILE, file_test::IS_BLOCK_DEV));
    assert!(!run_test_file(EXISTING_FILE, file_test::IS_CHAR_DEV));
}

#[test]
fn test_file_special_types_false_for_missing_path() {
    assert!(!run_test_file(MISSING_PATH, file_test::IS_SOCKET));
    assert!(!run_test_file(MISSING_PATH, file_test::IS_FIFO));
    assert!(!run_test_file(MISSING_PATH, file_test::IS_BLOCK_DEV));
    assert!(!run_test_file(MISSING_PATH, file_test::IS_CHAR_DEV));
}

#[test]
fn test_file_unknown_test_type_returns_false() {
    // The `_ => false` fallback branch covers any constant we don't define.
    assert!(!run_test_file(EXISTING_FILE, 200));
    assert!(!run_test_file(EXISTING_FILE, 255));
}

#[cfg(unix)]
#[test]
fn test_file_dev_null_is_char_device() {
    assert!(run_test_file("/dev/null", file_test::IS_CHAR_DEV));
    assert!(run_test_file("/dev/null", file_test::EXISTS));
    assert!(!run_test_file("/dev/null", file_test::IS_FILE));
    assert!(!run_test_file("/dev/null", file_test::IS_DIR));
    assert!(!run_test_file("/dev/null", file_test::IS_FIFO));
    assert!(!run_test_file("/dev/null", file_test::IS_SOCKET));
    assert!(!run_test_file("/dev/null", file_test::IS_BLOCK_DEV));
}

#[cfg(unix)]
#[test]
fn test_file_executable_for_chmod_555_temp_file() {
    use std::fs::{File, Permissions};
    use std::io::Write;
    use std::os::unix::fs::PermissionsExt;

    let tmp = std::env::temp_dir().join(format!("fusevm_exec_test_{}", std::process::id()));
    {
        let mut f = File::create(&tmp).expect("create temp");
        f.write_all(b"#!/bin/sh\n").unwrap();
    }
    std::fs::set_permissions(&tmp, Permissions::from_mode(0o555)).unwrap();
    let s = tmp.to_string_lossy().into_owned();
    assert!(run_test_file(&s, file_test::IS_EXECUTABLE));
    assert!(run_test_file(&s, file_test::IS_FILE));
    let _ = std::fs::remove_file(&tmp);
}

#[cfg(unix)]
#[test]
fn test_file_symlink_detected() {
    use std::os::unix::fs as unix_fs;
    let dir = std::env::temp_dir().join(format!("fusevm_symlink_test_{}", std::process::id()));
    let _ = std::fs::create_dir_all(&dir);
    let link = dir.join("link");
    let _ = std::fs::remove_file(&link);
    unix_fs::symlink("/etc/hosts", &link).expect("symlink");
    let s = link.to_string_lossy().into_owned();
    assert!(run_test_file(&s, file_test::IS_SYMLINK));
    // Through the symlink, exists/is_file are still true.
    assert!(run_test_file(&s, file_test::EXISTS));
    assert!(run_test_file(&s, file_test::IS_FILE));
    let _ = std::fs::remove_file(&link);
    let _ = std::fs::remove_dir(&dir);
}

// ══════════════════════════════════════════════════════════════════════════
// Value coercion via VM bytecode round-trip (push then inspect via op)
// ══════════════════════════════════════════════════════════════════════════

#[test]
fn vm_loadint_pushes_int_variant() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(123), 1);
    let v = run(b);
    assert!(matches!(v, Value::Int(123)));
}

#[test]
fn vm_loadconst_string_round_trip() {
    let mut b = ChunkBuilder::new();
    let k = b.add_constant(Value::str("hello"));
    b.emit(Op::LoadConst(k), 1);
    let v = run(b);
    assert!(matches!(&v, Value::Str(s) if s.as_str() == "hello"));
}

#[test]
fn vm_makearray_collects_n_top_values_in_push_order() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::LoadInt(2), 1);
    b.emit(Op::LoadInt(3), 1);
    b.emit(Op::MakeArray(3), 1);
    let v = run(b);
    match v {
        Value::Array(a) => {
            assert_eq!(a.len(), 3);
            assert_eq!(a[0], Value::Int(1));
            assert_eq!(a[1], Value::Int(2));
            assert_eq!(a[2], Value::Int(3));
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn vm_makearray_zero_makes_empty_array() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::MakeArray(0), 1);
    let v = run(b);
    assert!(matches!(&v, Value::Array(a) if a.is_empty()));
}
