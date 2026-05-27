//! Integration tests for array/hash/range/variable ops.
//!
//! These tests exercise the VM through the public bytecode interface,
//! covering opcodes not yet directly tested in `vm_integration.rs`
//! or in the in-module test suite.

use fusevm::{ChunkBuilder, Op, VMResult, Value, VM};

fn run(b: ChunkBuilder) -> VMResult {
    VM::new(b.build()).run()
}

// ── MakeArray ───────────────────────────────────────────────────────────────

#[test]
fn make_array_collects_n_values_from_stack() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::LoadInt(2), 1);
    b.emit(Op::LoadInt(3), 1);
    b.emit(Op::MakeArray(3), 1);
    match run(b) {
        VMResult::Ok(Value::Array(v)) => {
            assert_eq!(v, vec![Value::Int(1), Value::Int(2), Value::Int(3)]);
        }
        other => panic!("got {:?}", other),
    }
}

#[test]
fn make_array_with_zero_yields_empty_array() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::MakeArray(0), 1);
    match run(b) {
        VMResult::Ok(Value::Array(v)) => assert!(v.is_empty()),
        other => panic!("got {:?}", other),
    }
}

// ── Array global ops via name pool ──────────────────────────────────────────

#[test]
fn array_push_pop_round_trip_through_globals() {
    let mut b = ChunkBuilder::new();
    let arr = b.add_name("arr");
    b.emit(Op::DeclareArray(arr), 1);
    b.emit(Op::LoadInt(10), 1);
    b.emit(Op::ArrayPush(arr), 1);
    b.emit(Op::LoadInt(20), 1);
    b.emit(Op::ArrayPush(arr), 1);
    b.emit(Op::ArrayLen(arr), 1);
    match run(b) {
        VMResult::Ok(Value::Int(2)) => {}
        other => panic!("got {:?}", other),
    }
}

#[test]
fn array_pop_returns_last_then_decrements_len() {
    let mut b = ChunkBuilder::new();
    let arr = b.add_name("a");
    b.emit(Op::DeclareArray(arr), 1);
    b.emit(Op::LoadInt(7), 1);
    b.emit(Op::ArrayPush(arr), 1);
    b.emit(Op::LoadInt(9), 1);
    b.emit(Op::ArrayPush(arr), 1);
    b.emit(Op::ArrayPop(arr), 1); // pops 9, leaves it on the value stack
    match run(b) {
        VMResult::Ok(Value::Int(9)) => {}
        other => panic!("got {:?}", other),
    }
}

#[test]
fn array_shift_returns_first_element() {
    let mut b = ChunkBuilder::new();
    let arr = b.add_name("a");
    b.emit(Op::DeclareArray(arr), 1);
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::ArrayPush(arr), 1);
    b.emit(Op::LoadInt(2), 1);
    b.emit(Op::ArrayPush(arr), 1);
    b.emit(Op::LoadInt(3), 1);
    b.emit(Op::ArrayPush(arr), 1);
    b.emit(Op::ArrayShift(arr), 1);
    match run(b) {
        VMResult::Ok(Value::Int(1)) => {}
        other => panic!("got {:?}", other),
    }
}

#[test]
fn array_pop_on_empty_returns_undef() {
    let mut b = ChunkBuilder::new();
    let arr = b.add_name("a");
    b.emit(Op::DeclareArray(arr), 1);
    b.emit(Op::ArrayPop(arr), 1);
    match run(b) {
        VMResult::Ok(Value::Undef) => {}
        other => panic!("got {:?}", other),
    }
}

#[test]
fn array_len_on_undeclared_name_is_zero() {
    // ArrayLen on a name that was never declared as an array should be 0.
    let mut b = ChunkBuilder::new();
    let arr = b.add_name("unset");
    b.emit(Op::ArrayLen(arr), 1);
    match run(b) {
        VMResult::Ok(Value::Int(0)) => {}
        other => panic!("got {:?}", other),
    }
}

#[test]
fn array_get_returns_value_at_index() {
    let mut b = ChunkBuilder::new();
    let arr = b.add_name("a");
    b.emit(Op::DeclareArray(arr), 1);
    b.emit(Op::LoadInt(11), 1);
    b.emit(Op::ArrayPush(arr), 1);
    b.emit(Op::LoadInt(22), 1);
    b.emit(Op::ArrayPush(arr), 1);
    b.emit(Op::LoadInt(33), 1);
    b.emit(Op::ArrayPush(arr), 1);
    b.emit(Op::LoadInt(1), 1); // index
    b.emit(Op::ArrayGet(arr), 1);
    match run(b) {
        VMResult::Ok(Value::Int(22)) => {}
        other => panic!("got {:?}", other),
    }
}

// ── Hash ops ───────────────────────────────────────────────────────────────

#[test]
fn hash_set_get_round_trip() {
    let mut b = ChunkBuilder::new();
    let h = b.add_name("h");
    let kc = b.add_constant(Value::str("k"));
    let vc = b.add_constant(Value::str("v"));
    b.emit(Op::DeclareHash(h), 1);
    // HashSet pops key then value (key is on top per impl).
    b.emit(Op::LoadConst(vc), 1);
    b.emit(Op::LoadConst(kc), 1);
    b.emit(Op::HashSet(h), 1);
    b.emit(Op::LoadConst(kc), 1);
    b.emit(Op::HashGet(h), 1);
    match run(b) {
        VMResult::Ok(v) => assert_eq!(v.to_str(), "v"),
        other => panic!("got {:?}", other),
    }
}

#[test]
fn hash_exists_true_then_false_after_delete() {
    let mut b = ChunkBuilder::new();
    let h = b.add_name("h");
    let kc = b.add_constant(Value::str("name"));
    let vc = b.add_constant(Value::str("ada"));
    b.emit(Op::DeclareHash(h), 1);
    b.emit(Op::LoadConst(vc), 1);
    b.emit(Op::LoadConst(kc), 1);
    b.emit(Op::HashSet(h), 1);

    // exists → true
    b.emit(Op::LoadConst(kc), 1);
    b.emit(Op::HashExists(h), 1);
    // delete → returns prior value (kept on stack)
    b.emit(Op::LoadConst(kc), 1);
    b.emit(Op::HashDelete(h), 1);
    b.emit(Op::Pop, 1); // discard returned deleted value
                        // exists → false
    b.emit(Op::LoadConst(kc), 1);
    b.emit(Op::HashExists(h), 1);
    // The stack now has [Bool(true), Bool(false)]; MakeArray to inspect both.
    b.emit(Op::MakeArray(2), 1);
    match run(b) {
        VMResult::Ok(Value::Array(v)) => {
            assert_eq!(v, vec![Value::Bool(true), Value::Bool(false)]);
        }
        other => panic!("got {:?}", other),
    }
}

#[test]
fn hash_get_missing_key_returns_undef() {
    let mut b = ChunkBuilder::new();
    let h = b.add_name("h");
    let missing = b.add_constant(Value::str("nope"));
    b.emit(Op::DeclareHash(h), 1);
    b.emit(Op::LoadConst(missing), 1);
    b.emit(Op::HashGet(h), 1);
    match run(b) {
        VMResult::Ok(Value::Undef) => {}
        other => panic!("got {:?}", other),
    }
}

#[test]
fn hash_keys_returns_array_of_string_keys() {
    let mut b = ChunkBuilder::new();
    let h = b.add_name("h");
    let k1 = b.add_constant(Value::str("a"));
    let v1 = b.add_constant(Value::Int(1));
    let k2 = b.add_constant(Value::str("b"));
    let v2 = b.add_constant(Value::Int(2));
    b.emit(Op::DeclareHash(h), 1);
    b.emit(Op::LoadConst(v1), 1);
    b.emit(Op::LoadConst(k1), 1);
    b.emit(Op::HashSet(h), 1);
    b.emit(Op::LoadConst(v2), 1);
    b.emit(Op::LoadConst(k2), 1);
    b.emit(Op::HashSet(h), 1);
    b.emit(Op::HashKeys(h), 1);
    match run(b) {
        VMResult::Ok(Value::Array(v)) => {
            // HashMap iteration order is unspecified — verify content set.
            let mut keys: Vec<String> = v.iter().map(|x| x.to_str()).collect();
            keys.sort();
            assert_eq!(keys, vec!["a".to_string(), "b".to_string()]);
        }
        other => panic!("got {:?}", other),
    }
}

// ── Range ──────────────────────────────────────────────────────────────────

#[test]
fn range_builds_inclusive_int_array() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(2), 1);
    b.emit(Op::LoadInt(5), 1);
    b.emit(Op::Range, 1);
    match run(b) {
        VMResult::Ok(Value::Array(v)) => {
            assert_eq!(
                v,
                vec![Value::Int(2), Value::Int(3), Value::Int(4), Value::Int(5)]
            );
        }
        other => panic!("got {:?}", other),
    }
}

#[test]
fn range_with_from_greater_than_to_is_empty() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(10), 1);
    b.emit(Op::LoadInt(5), 1);
    b.emit(Op::Range, 1);
    match run(b) {
        VMResult::Ok(Value::Array(v)) => assert!(v.is_empty()),
        other => panic!("got {:?}", other),
    }
}

// ── Variables (globals via name pool) ──────────────────────────────────────

#[test]
fn set_var_then_get_var() {
    let mut b = ChunkBuilder::new();
    let x = b.add_name("x");
    b.emit(Op::LoadInt(123), 1);
    b.emit(Op::SetVar(x), 1);
    b.emit(Op::GetVar(x), 1);
    match run(b) {
        VMResult::Ok(Value::Int(123)) => {}
        other => panic!("got {:?}", other),
    }
}

#[test]
fn unset_global_reads_as_undef() {
    let mut b = ChunkBuilder::new();
    let y = b.add_name("y"); // never set
    b.emit(Op::GetVar(y), 1);
    match run(b) {
        VMResult::Ok(Value::Undef) => {}
        other => panic!("got {:?}", other),
    }
}

#[test]
fn set_var_overwrites_previous_value() {
    let mut b = ChunkBuilder::new();
    let x = b.add_name("x");
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::SetVar(x), 1);
    b.emit(Op::LoadInt(2), 1);
    b.emit(Op::SetVar(x), 1);
    b.emit(Op::GetVar(x), 1);
    match run(b) {
        VMResult::Ok(Value::Int(2)) => {}
        other => panic!("got {:?}", other),
    }
}

// ── MakeHash ───────────────────────────────────────────────────────────────

#[test]
fn make_hash_builds_map_from_pairs_on_stack() {
    // Push key0, val0, key1, val1, then MakeHash(4)
    let mut b = ChunkBuilder::new();
    let k1 = b.add_constant(Value::str("one"));
    let k2 = b.add_constant(Value::str("two"));
    b.emit(Op::LoadConst(k1), 1);
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::LoadConst(k2), 1);
    b.emit(Op::LoadInt(2), 1);
    b.emit(Op::MakeHash(4), 1);
    match run(b) {
        VMResult::Ok(Value::Hash(m)) => {
            assert_eq!(m.get("one"), Some(&Value::Int(1)));
            assert_eq!(m.get("two"), Some(&Value::Int(2)));
            assert_eq!(m.len(), 2);
        }
        other => panic!("got {:?}", other),
    }
}
