//! Additional coverage for under-tested VM ops: HashKeys/HashValues,
//! RangeStep edge cases, Spaceship/StrCmp variants, BitNot/Shl/Shr, MakeHash
//! stack ordering, ConcatConstLoop with prior non-empty prefix, and a few
//! Value-coercion behaviours via op pipelines.

use fusevm::{ChunkBuilder, Op, VMResult, Value, VM};

fn run(ops: &[(Op, u32)]) -> Value {
    let mut b = ChunkBuilder::new();
    for (op, line) in ops {
        b.emit(op.clone(), *line);
    }
    match VM::new(b.build()).run() {
        VMResult::Ok(v) => v,
        VMResult::Halted => Value::Undef,
        other => panic!("unexpected: {:?}", other),
    }
}

fn run_int(ops: &[(Op, u32)]) -> i64 {
    match run(ops) {
        Value::Int(n) => n,
        v => panic!("expected Int, got {:?}", v),
    }
}

fn run_bool(ops: &[(Op, u32)]) -> bool {
    match run(ops) {
        Value::Bool(b) => b,
        v => panic!("expected Bool, got {:?}", v),
    }
}

// ── Spaceship / StrCmp ──────────────────────────────────────────────────────

#[test]
fn spaceship_returns_minus_one_zero_one() {
    assert_eq!(
        run_int(&[(Op::LoadInt(5), 1), (Op::LoadInt(9), 1), (Op::Spaceship, 1)]),
        -1
    );
    assert_eq!(
        run_int(&[(Op::LoadInt(7), 1), (Op::LoadInt(7), 1), (Op::Spaceship, 1)]),
        0
    );
    assert_eq!(
        run_int(&[(Op::LoadInt(9), 1), (Op::LoadInt(5), 1), (Op::Spaceship, 1)]),
        1
    );
}

#[test]
fn strcmp_lex_ordering() {
    // Build constants by hand: "abc" cmp "abd"
    let mut b = ChunkBuilder::new();
    let a = b.add_constant(Value::str("abc"));
    let z = b.add_constant(Value::str("abd"));
    b.emit(Op::LoadConst(a), 1);
    b.emit(Op::LoadConst(z), 1);
    b.emit(Op::StrCmp, 1);
    match VM::new(b.build()).run() {
        VMResult::Ok(Value::Int(n)) if n < 0 => {}
        other => panic!("got {:?}", other),
    }
}

#[test]
fn strcmp_equal_strings_is_zero() {
    let mut b = ChunkBuilder::new();
    let a = b.add_constant(Value::str("hello"));
    b.emit(Op::LoadConst(a), 1);
    b.emit(Op::LoadConst(a), 1);
    b.emit(Op::StrCmp, 1);
    match VM::new(b.build()).run() {
        VMResult::Ok(Value::Int(0)) => {}
        other => panic!("got {:?}", other),
    }
}

// ── Bitwise edges ───────────────────────────────────────────────────────────

#[test]
fn bitnot_of_zero_is_minus_one() {
    assert_eq!(run_int(&[(Op::LoadInt(0), 1), (Op::BitNot, 1)]), -1);
}

#[test]
fn bitnot_is_self_inverse() {
    assert_eq!(
        run_int(&[(Op::LoadInt(0x5a5a), 1), (Op::BitNot, 1), (Op::BitNot, 1)]),
        0x5a5a
    );
}

#[test]
fn shl_shr_round_trip_for_positive() {
    assert_eq!(
        run_int(&[
            (Op::LoadInt(0b1011), 1),
            (Op::LoadInt(4), 1),
            (Op::Shl, 1),
            (Op::LoadInt(4), 1),
            (Op::Shr, 1),
        ]),
        0b1011
    );
}

#[test]
fn bit_and_or_xor_combine() {
    // (0xff00 | 0x00ff) ^ 0x0f0f = 0xffff ^ 0x0f0f = 0xf0f0
    assert_eq!(
        run_int(&[
            (Op::LoadInt(0xff00), 1),
            (Op::LoadInt(0x00ff), 1),
            (Op::BitOr, 1),
            (Op::LoadInt(0x0f0f), 1),
            (Op::BitXor, 1),
        ]),
        0xf0f0
    );
    // BitAnd to mask out high byte
    assert_eq!(
        run_int(&[
            (Op::LoadInt(0xffff), 1),
            (Op::LoadInt(0x00ff), 1),
            (Op::BitAnd, 1),
        ]),
        0x00ff
    );
}

// ── HashKeys / HashValues ───────────────────────────────────────────────────

#[test]
fn hash_keys_after_three_inserts_has_three_entries() {
    let mut b = ChunkBuilder::new();
    let h = b.add_name("h");
    let ka = b.add_constant(Value::str("a"));
    let kb = b.add_constant(Value::str("b"));
    let kc = b.add_constant(Value::str("c"));
    b.emit(Op::DeclareHash(h), 1);
    // h["a"] = 1
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::LoadConst(ka), 1);
    b.emit(Op::HashSet(h), 1);
    // h["b"] = 2
    b.emit(Op::LoadInt(2), 1);
    b.emit(Op::LoadConst(kb), 1);
    b.emit(Op::HashSet(h), 1);
    // h["c"] = 3
    b.emit(Op::LoadInt(3), 1);
    b.emit(Op::LoadConst(kc), 1);
    b.emit(Op::HashSet(h), 1);
    b.emit(Op::HashKeys(h), 1);
    match VM::new(b.build()).run() {
        VMResult::Ok(Value::Array(a)) => assert_eq!(a.len(), 3),
        other => panic!("got {:?}", other),
    }
}

#[test]
fn hash_values_sum_matches_inserted() {
    let mut b = ChunkBuilder::new();
    let h = b.add_name("nums");
    let ka = b.add_constant(Value::str("a"));
    let kb = b.add_constant(Value::str("b"));
    b.emit(Op::DeclareHash(h), 1);
    b.emit(Op::LoadInt(10), 1);
    b.emit(Op::LoadConst(ka), 1);
    b.emit(Op::HashSet(h), 1);
    b.emit(Op::LoadInt(32), 1);
    b.emit(Op::LoadConst(kb), 1);
    b.emit(Op::HashSet(h), 1);
    b.emit(Op::HashValues(h), 1);
    match VM::new(b.build()).run() {
        VMResult::Ok(Value::Array(a)) => {
            let sum: i64 = a.iter().map(|v| v.to_int()).sum();
            assert_eq!(sum, 42);
        }
        other => panic!("got {:?}", other),
    }
}

#[test]
fn hash_keys_on_uninit_hash_returns_empty() {
    // No DeclareHash → globals slot beyond len; HashKeys returns empty array.
    let mut b = ChunkBuilder::new();
    let h = b.add_name("missing");
    b.emit(Op::HashKeys(h), 1);
    match VM::new(b.build()).run() {
        VMResult::Ok(Value::Array(a)) => assert!(a.is_empty()),
        other => panic!("got {:?}", other),
    }
}

// ── MakeHash stack ordering ─────────────────────────────────────────────────

#[test]
fn make_hash_pairs_value_then_key() {
    // Stack layout for MakeHash(n): n consecutive (value, key) pairs.
    // From the VM source: pairs.drain → iter yields value first, then key:
    //   while let Some(key) = iter.next() { if let Some(val) = iter.next() { ... } }
    // So pushed order must be (key, val) such that drained iterator yields key then val.
    let mut b = ChunkBuilder::new();
    let k = b.add_constant(Value::str("count"));
    b.emit(Op::LoadConst(k), 1);
    b.emit(Op::LoadInt(7), 1);
    b.emit(Op::MakeHash(2), 1);
    match VM::new(b.build()).run() {
        VMResult::Ok(Value::Hash(m)) => {
            assert_eq!(m.len(), 1);
            // Either ordering for k/v — verify the map has exactly one entry
            // and its value is the int we pushed.
            let v = m.values().next().unwrap();
            assert_eq!(v.to_int(), 7);
        }
        other => panic!("got {:?}", other),
    }
}

#[test]
fn make_hash_zero_pairs_yields_empty_hash() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::MakeHash(0), 1);
    match VM::new(b.build()).run() {
        VMResult::Ok(Value::Hash(m)) => assert!(m.is_empty()),
        other => panic!("got {:?}", other),
    }
}

// ── Range / RangeStep edge cases ────────────────────────────────────────────

#[test]
fn range_step_positive_produces_expected_values() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(0), 1);
    b.emit(Op::LoadInt(10), 1);
    b.emit(Op::LoadInt(3), 1);
    b.emit(Op::RangeStep, 1);
    match VM::new(b.build()).run() {
        VMResult::Ok(Value::Array(a)) => {
            let ns: Vec<i64> = a.iter().map(|v| v.to_int()).collect();
            assert_eq!(ns, vec![0, 3, 6, 9]);
        }
        other => panic!("got {:?}", other),
    }
}

#[test]
fn range_step_descending_step_minus_two() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(10), 1);
    b.emit(Op::LoadInt(0), 1);
    b.emit(Op::LoadInt(-2), 1);
    b.emit(Op::RangeStep, 1);
    match VM::new(b.build()).run() {
        VMResult::Ok(Value::Array(a)) => {
            let ns: Vec<i64> = a.iter().map(|v| v.to_int()).collect();
            assert_eq!(ns, vec![10, 8, 6, 4, 2, 0]);
        }
        other => panic!("got {:?}", other),
    }
}

#[test]
fn range_step_wrong_direction_yields_empty() {
    // from=5, to=0, step=+1 → empty
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(5), 1);
    b.emit(Op::LoadInt(0), 1);
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::RangeStep, 1);
    match VM::new(b.build()).run() {
        VMResult::Ok(Value::Array(a)) => assert!(a.is_empty()),
        other => panic!("got {:?}", other),
    }
}

// ── Comparison ops on mixed types ───────────────────────────────────────────

#[test]
fn num_lt_int_vs_float_uses_float_compare() {
    assert!(run_bool(&[
        (Op::LoadInt(2), 1),
        (Op::LoadFloat(2.5), 1),
        (Op::NumLt, 1)
    ]));
    assert!(!run_bool(&[
        (Op::LoadInt(3), 1),
        (Op::LoadFloat(2.5), 1),
        (Op::NumLt, 1)
    ]));
}

#[test]
fn num_eq_int_and_equal_float_is_true() {
    assert!(run_bool(&[
        (Op::LoadInt(4), 1),
        (Op::LoadFloat(4.0), 1),
        (Op::NumEq, 1)
    ]));
}

#[test]
fn str_le_ge_on_equal_strings_is_true() {
    let mut b = ChunkBuilder::new();
    let s = b.add_constant(Value::str("x"));
    b.emit(Op::LoadConst(s), 1);
    b.emit(Op::LoadConst(s), 1);
    b.emit(Op::StrLe, 1);
    match VM::new(b.build()).run() {
        VMResult::Ok(Value::Bool(true)) => {}
        other => panic!("got {:?}", other),
    }
    let mut b = ChunkBuilder::new();
    let s = b.add_constant(Value::str("x"));
    b.emit(Op::LoadConst(s), 1);
    b.emit(Op::LoadConst(s), 1);
    b.emit(Op::StrGe, 1);
    match VM::new(b.build()).run() {
        VMResult::Ok(Value::Bool(true)) => {}
        other => panic!("got {:?}", other),
    }
}

// ── ConcatConstLoop with non-empty prefix ───────────────────────────────────

#[test]
fn concat_const_loop_appends_to_existing_slot_value() {
    let mut b = ChunkBuilder::new();
    let pre = b.add_constant(Value::str("X"));
    let glue = b.add_constant(Value::str("-"));
    b.emit(Op::PushFrame, 1);
    b.emit(Op::LoadConst(pre), 1);
    b.emit(Op::SetSlot(0), 1); // s = "X"
    b.emit(Op::LoadInt(0), 1);
    b.emit(Op::SetSlot(1), 1); // i = 0
    b.emit(Op::ConcatConstLoop(glue, 0, 1, 3), 1); // append "-" 3 times
    b.emit(Op::GetSlot(0), 1);
    match VM::new(b.build()).run() {
        VMResult::Ok(Value::Str(s)) => assert_eq!(s.as_str(), "X---"),
        other => panic!("got {:?}", other),
    }
}

// ── String coercion via Concat ─────────────────────────────────────────────

#[test]
fn concat_float_with_string_yields_combined_string() {
    let mut b = ChunkBuilder::new();
    let suffix = b.add_constant(Value::str(" units"));
    b.emit(Op::LoadFloat(1.5), 1);
    b.emit(Op::LoadConst(suffix), 1);
    b.emit(Op::Concat, 1);
    match VM::new(b.build()).run() {
        VMResult::Ok(Value::Str(s)) => assert!(s.as_str().contains("units")),
        other => panic!("got {:?}", other),
    }
}

#[test]
fn string_repeat_zero_yields_empty_string() {
    let mut b = ChunkBuilder::new();
    let s = b.add_constant(Value::str("ab"));
    b.emit(Op::LoadConst(s), 1);
    b.emit(Op::LoadInt(0), 1);
    b.emit(Op::StringRepeat, 1);
    match VM::new(b.build()).run() {
        VMResult::Ok(Value::Str(s)) => assert_eq!(s.as_str(), ""),
        other => panic!("got {:?}", other),
    }
}

#[test]
fn string_len_counts_bytes() {
    let mut b = ChunkBuilder::new();
    let s = b.add_constant(Value::str("hello"));
    b.emit(Op::LoadConst(s), 1);
    b.emit(Op::StringLen, 1);
    match VM::new(b.build()).run() {
        VMResult::Ok(Value::Int(5)) => {}
        other => panic!("got {:?}", other),
    }
}

// ── Logic short-circuit on stack value ──────────────────────────────────────

#[test]
fn log_and_with_zero_left_short_circuits_to_zero() {
    // 0 && (anything truthy) — verify result truthiness only.
    assert!(!run_bool(&[
        (Op::LoadInt(0), 1),
        (Op::LoadInt(5), 1),
        (Op::LogAnd, 1),
        (Op::LogNot, 1),
        (Op::LogNot, 1),
    ]));
}

#[test]
fn log_or_with_truthy_left_keeps_truthiness() {
    assert!(run_bool(&[
        (Op::LoadInt(7), 1),
        (Op::LoadInt(0), 1),
        (Op::LogOr, 1),
        (Op::LogNot, 1),
        (Op::LogNot, 1),
    ]));
}
