use fusevm::{ChunkBuilder, Op, VMResult, Value, VM};

// ── Helper ──

fn run(ops: &[(Op, u32)]) -> VMResult {
    let mut b = ChunkBuilder::new();
    for (op, line) in ops {
        b.emit(op.clone(), *line);
    }
    VM::new(b.build()).run()
}

fn expect_int(ops: &[(Op, u32)], expected: i64) {
    match run(ops) {
        VMResult::Ok(Value::Int(n)) => assert_eq!(n, expected),
        other => panic!("expected Int({}), got {:?}", expected, other),
    }
}

fn expect_float(ops: &[(Op, u32)], expected: f64) {
    match run(ops) {
        VMResult::Ok(Value::Float(f)) => {
            assert!(
                (f - expected).abs() < 1e-10,
                "expected {}, got {}",
                expected,
                f
            );
        }
        other => panic!("expected Float({}), got {:?}", expected, other),
    }
}

fn expect_bool(ops: &[(Op, u32)], expected: bool) {
    match run(ops) {
        VMResult::Ok(Value::Bool(b)) => assert_eq!(b, expected),
        other => panic!("expected Bool({}), got {:?}", expected, other),
    }
}

// ── Arithmetic ──

#[test]
fn add_integers() {
    expect_int(
        &[(Op::LoadInt(40), 1), (Op::LoadInt(2), 1), (Op::Add, 1)],
        42,
    );
}

#[test]
fn sub_integers() {
    expect_int(
        &[(Op::LoadInt(50), 1), (Op::LoadInt(8), 1), (Op::Sub, 1)],
        42,
    );
}

#[test]
fn mul_integers() {
    expect_int(
        &[(Op::LoadInt(6), 1), (Op::LoadInt(7), 1), (Op::Mul, 1)],
        42,
    );
}

#[test]
fn div_by_zero_yields_undef() {
    match run(&[(Op::LoadInt(1), 1), (Op::LoadInt(0), 1), (Op::Div, 1)]) {
        VMResult::Ok(Value::Undef) => {}
        other => panic!("expected Undef for 1/0, got {:?}", other),
    }
}

#[test]
fn div_float() {
    expect_float(
        &[
            (Op::LoadFloat(10.0), 1),
            (Op::LoadFloat(4.0), 1),
            (Op::Div, 1),
        ],
        2.5,
    );
}

#[test]
fn modulo() {
    expect_int(
        &[(Op::LoadInt(17), 1), (Op::LoadInt(5), 1), (Op::Mod, 1)],
        2,
    );
}

#[test]
fn power() {
    expect_float(
        &[(Op::LoadInt(2), 1), (Op::LoadInt(10), 1), (Op::Pow, 1)],
        1024.0,
    );
}

#[test]
fn negate() {
    expect_int(&[(Op::LoadInt(42), 1), (Op::Negate, 1)], -42);
}

#[test]
fn inc_dec() {
    expect_int(&[(Op::LoadInt(41), 1), (Op::Inc, 1)], 42);
    expect_int(&[(Op::LoadInt(43), 1), (Op::Dec, 1)], 42);
}

#[test]
fn mixed_int_float_add() {
    expect_float(
        &[(Op::LoadInt(1), 1), (Op::LoadFloat(2.5), 1), (Op::Add, 1)],
        3.5,
    );
}

#[test]
fn wrapping_add_overflow() {
    expect_int(
        &[
            (Op::LoadInt(i64::MAX), 1),
            (Op::LoadInt(1), 1),
            (Op::Add, 1),
        ],
        i64::MIN, // wrapping
    );
}

// ── String ──

#[test]
fn concat_strings() {
    let mut b = ChunkBuilder::new();
    let c1 = b.add_constant(Value::str("hello "));
    let c2 = b.add_constant(Value::str("world"));
    b.emit(Op::LoadConst(c1), 1);
    b.emit(Op::LoadConst(c2), 1);
    b.emit(Op::Concat, 1);
    match VM::new(b.build()).run() {
        VMResult::Ok(Value::Str(s)) => assert_eq!(s.as_str(), "hello world"),
        other => panic!("expected 'hello world', got {:?}", other),
    }
}

#[test]
fn string_repeat() {
    let mut b = ChunkBuilder::new();
    let c = b.add_constant(Value::str("ab"));
    b.emit(Op::LoadConst(c), 1);
    b.emit(Op::LoadInt(3), 1);
    b.emit(Op::StringRepeat, 1);
    match VM::new(b.build()).run() {
        VMResult::Ok(Value::Str(s)) => assert_eq!(s.as_str(), "ababab"),
        other => panic!("expected 'ababab', got {:?}", other),
    }
}

#[test]
fn string_len() {
    let mut b = ChunkBuilder::new();
    let c = b.add_constant(Value::str("hello"));
    b.emit(Op::LoadConst(c), 1);
    b.emit(Op::StringLen, 1);
    match VM::new(b.build()).run() {
        VMResult::Ok(Value::Int(5)) => {}
        other => panic!("expected Int(5), got {:?}", other),
    }
}

#[test]
fn string_repeat_zero() {
    let mut b = ChunkBuilder::new();
    let c = b.add_constant(Value::str("x"));
    b.emit(Op::LoadConst(c), 1);
    b.emit(Op::LoadInt(0), 1);
    b.emit(Op::StringRepeat, 1);
    match VM::new(b.build()).run() {
        VMResult::Ok(Value::Str(s)) => assert_eq!(s.as_str(), ""),
        other => panic!("expected empty string, got {:?}", other),
    }
}

// ── Comparison ──

#[test]
fn numeric_comparisons() {
    expect_bool(
        &[(Op::LoadInt(1), 1), (Op::LoadInt(2), 1), (Op::NumLt, 1)],
        true,
    );
    expect_bool(
        &[(Op::LoadInt(2), 1), (Op::LoadInt(1), 1), (Op::NumLt, 1)],
        false,
    );
    expect_bool(
        &[(Op::LoadInt(5), 1), (Op::LoadInt(5), 1), (Op::NumEq, 1)],
        true,
    );
    expect_bool(
        &[(Op::LoadInt(5), 1), (Op::LoadInt(5), 1), (Op::NumNe, 1)],
        false,
    );
    expect_bool(
        &[(Op::LoadInt(5), 1), (Op::LoadInt(3), 1), (Op::NumGt, 1)],
        true,
    );
    expect_bool(
        &[(Op::LoadInt(5), 1), (Op::LoadInt(5), 1), (Op::NumLe, 1)],
        true,
    );
    expect_bool(
        &[(Op::LoadInt(5), 1), (Op::LoadInt(5), 1), (Op::NumGe, 1)],
        true,
    );
}

#[test]
fn spaceship() {
    expect_int(
        &[(Op::LoadInt(1), 1), (Op::LoadInt(2), 1), (Op::Spaceship, 1)],
        -1,
    );
    expect_int(
        &[(Op::LoadInt(2), 1), (Op::LoadInt(2), 1), (Op::Spaceship, 1)],
        0,
    );
    expect_int(
        &[(Op::LoadInt(3), 1), (Op::LoadInt(2), 1), (Op::Spaceship, 1)],
        1,
    );
}

#[test]
fn string_comparisons() {
    let mut b = ChunkBuilder::new();
    let a = b.add_constant(Value::str("apple"));
    let bv = b.add_constant(Value::str("banana"));
    b.emit(Op::LoadConst(a), 1);
    b.emit(Op::LoadConst(bv), 1);
    b.emit(Op::StrLt, 1);
    match VM::new(b.build()).run() {
        VMResult::Ok(Value::Bool(true)) => {}
        other => panic!("expected 'apple' < 'banana', got {:?}", other),
    }
}

#[test]
fn strcmp_ordering() {
    let mut b = ChunkBuilder::new();
    let a = b.add_constant(Value::str("abc"));
    let bv = b.add_constant(Value::str("abc"));
    b.emit(Op::LoadConst(a), 1);
    b.emit(Op::LoadConst(bv), 1);
    b.emit(Op::StrCmp, 1);
    match VM::new(b.build()).run() {
        VMResult::Ok(Value::Int(0)) => {}
        other => panic!("expected 0 for equal strings, got {:?}", other),
    }
}

// ── Logic / Bitwise ──

#[test]
fn logical_not() {
    expect_bool(&[(Op::LoadTrue, 1), (Op::LogNot, 1)], false);
    expect_bool(&[(Op::LoadFalse, 1), (Op::LogNot, 1)], true);
    expect_bool(&[(Op::LoadInt(0), 1), (Op::LogNot, 1)], true);
}

#[test]
fn bitwise_ops() {
    expect_int(
        &[
            (Op::LoadInt(0xFF), 1),
            (Op::LoadInt(0x0F), 1),
            (Op::BitAnd, 1),
        ],
        0x0F,
    );
    expect_int(
        &[
            (Op::LoadInt(0xF0), 1),
            (Op::LoadInt(0x0F), 1),
            (Op::BitOr, 1),
        ],
        0xFF,
    );
    expect_int(
        &[
            (Op::LoadInt(0xFF), 1),
            (Op::LoadInt(0xFF), 1),
            (Op::BitXor, 1),
        ],
        0,
    );
    expect_int(
        &[(Op::LoadInt(1), 1), (Op::LoadInt(8), 1), (Op::Shl, 1)],
        256,
    );
    expect_int(
        &[(Op::LoadInt(256), 1), (Op::LoadInt(4), 1), (Op::Shr, 1)],
        16,
    );
}

// ── Stack Ops ──

#[test]
fn dup() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(21), 1);
    b.emit(Op::Dup, 1);
    b.emit(Op::Add, 1);
    match VM::new(b.build()).run() {
        VMResult::Ok(Value::Int(42)) => {}
        other => panic!("expected Int(42), got {:?}", other),
    }
}

#[test]
fn swap() {
    expect_int(
        &[
            (Op::LoadInt(10), 1),
            (Op::LoadInt(3), 1),
            (Op::Swap, 1),
            (Op::Sub, 1),
        ],
        -7, // 3 - 10
    );
}

#[test]
fn dup2() {
    // stack: [3, 4] -> dup2 -> [3, 4, 3, 4] -> add -> [3, 4, 7] -> add -> [3, 11] -> add -> [14]
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(3), 1);
    b.emit(Op::LoadInt(4), 1);
    b.emit(Op::Dup2, 1);
    b.emit(Op::Add, 1);
    b.emit(Op::Add, 1);
    b.emit(Op::Add, 1);
    match VM::new(b.build()).run() {
        VMResult::Ok(Value::Int(14)) => {}
        other => panic!("expected Int(14), got {:?}", other),
    }
}

// ── Control Flow ──

#[test]
fn jump_if_false_taken() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(99), 1);
    b.emit(Op::LoadFalse, 1);
    b.emit(Op::JumpIfFalse(4), 1);
    b.emit(Op::LoadInt(999), 1); // skipped
                                 // ip 4:
    match VM::new(b.build()).run() {
        VMResult::Ok(Value::Int(99)) => {}
        other => panic!("expected Int(99), got {:?}", other),
    }
}

#[test]
fn jump_if_true_not_taken() {
    expect_int(
        &[
            (Op::LoadInt(1), 1),
            (Op::LoadFalse, 1),
            (Op::JumpIfTrue(4), 1), // not taken
            (Op::LoadInt(2), 1),    // executed
            (Op::Add, 1),
        ],
        3,
    );
}

#[test]
fn short_circuit_or() {
    // JumpIfTrueKeep: if TOS is truthy, jump (keep value on stack)
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(42), 1);
    b.emit(Op::JumpIfTrueKeep(3), 1); // truthy, jump to end
    b.emit(Op::LoadInt(999), 1); // skipped
                                 // ip 3: stack has 42
    match VM::new(b.build()).run() {
        VMResult::Ok(Value::Int(42)) => {}
        other => panic!("expected Int(42), got {:?}", other),
    }
}

#[test]
fn short_circuit_and_false() {
    // JumpIfFalseKeep: if TOS is falsy, jump (keep value on stack)
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(0), 1);
    b.emit(Op::JumpIfFalseKeep(3), 1); // falsy (0), jump to end
    b.emit(Op::LoadInt(999), 1); // skipped
    match VM::new(b.build()).run() {
        VMResult::Ok(Value::Int(0)) => {}
        other => panic!("expected Int(0), got {:?}", other),
    }
}

// ── Variables ──

#[test]
fn global_variables() {
    let mut b = ChunkBuilder::new();
    let x = b.add_name("x");
    b.emit(Op::LoadInt(42), 1);
    b.emit(Op::SetVar(x), 1);
    b.emit(Op::GetVar(x), 1);
    match VM::new(b.build()).run() {
        VMResult::Ok(Value::Int(42)) => {}
        other => panic!("expected Int(42), got {:?}", other),
    }
}

#[test]
fn slot_variables() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::PushFrame, 1);
    b.emit(Op::LoadInt(100), 1);
    b.emit(Op::SetSlot(0), 1);
    b.emit(Op::LoadInt(200), 1);
    b.emit(Op::SetSlot(1), 1);
    b.emit(Op::GetSlot(0), 1);
    b.emit(Op::GetSlot(1), 1);
    b.emit(Op::Add, 1);
    match VM::new(b.build()).run() {
        VMResult::Ok(Value::Int(300)) => {}
        other => panic!("expected Int(300), got {:?}", other),
    }
}

#[test]
fn undeclared_var_is_undef() {
    let mut b = ChunkBuilder::new();
    let x = b.add_name("x");
    b.emit(Op::GetVar(x), 1);
    match VM::new(b.build()).run() {
        VMResult::Ok(Value::Undef) => {}
        other => panic!("expected Undef, got {:?}", other),
    }
}

// ── Arrays ──

#[test]
fn make_array_and_len() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(10), 1);
    b.emit(Op::LoadInt(20), 1);
    b.emit(Op::LoadInt(30), 1);
    b.emit(Op::MakeArray(3), 1);
    match VM::new(b.build()).run() {
        VMResult::Ok(Value::Array(arr)) => {
            assert_eq!(arr.len(), 3);
            assert_eq!(arr[0].to_int(), 10);
            assert_eq!(arr[1].to_int(), 20);
            assert_eq!(arr[2].to_int(), 30);
        }
        other => panic!("expected Array, got {:?}", other),
    }
}

#[test]
fn array_push_pop() {
    let mut b = ChunkBuilder::new();
    let arr = b.add_name("arr");
    b.emit(Op::DeclareArray(arr), 1);
    b.emit(Op::LoadInt(10), 1);
    b.emit(Op::ArrayPush(arr), 1);
    b.emit(Op::LoadInt(20), 1);
    b.emit(Op::ArrayPush(arr), 1);
    b.emit(Op::ArrayPop(arr), 1); // pops 20
    match VM::new(b.build()).run() {
        VMResult::Ok(Value::Int(20)) => {}
        other => panic!("expected Int(20), got {:?}", other),
    }
}

#[test]
fn array_shift() {
    let mut b = ChunkBuilder::new();
    let arr = b.add_name("arr");
    b.emit(Op::DeclareArray(arr), 1);
    b.emit(Op::LoadInt(10), 1);
    b.emit(Op::ArrayPush(arr), 1);
    b.emit(Op::LoadInt(20), 1);
    b.emit(Op::ArrayPush(arr), 1);
    b.emit(Op::ArrayShift(arr), 1); // shifts 10
    match VM::new(b.build()).run() {
        VMResult::Ok(Value::Int(10)) => {}
        other => panic!("expected Int(10), got {:?}", other),
    }
}

#[test]
fn array_get_set() {
    let mut b = ChunkBuilder::new();
    let arr = b.add_name("arr");
    b.emit(Op::DeclareArray(arr), 1);
    b.emit(Op::LoadInt(0), 1);
    b.emit(Op::ArrayPush(arr), 1);
    b.emit(Op::LoadInt(0), 1);
    b.emit(Op::ArrayPush(arr), 1);
    // set arr[1] = 42
    b.emit(Op::LoadInt(42), 1);
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::ArraySet(arr), 1);
    // get arr[1]
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::ArrayGet(arr), 1);
    match VM::new(b.build()).run() {
        VMResult::Ok(Value::Int(42)) => {}
        other => panic!("expected Int(42), got {:?}", other),
    }
}

#[test]
fn array_len() {
    let mut b = ChunkBuilder::new();
    let arr = b.add_name("arr");
    b.emit(Op::DeclareArray(arr), 1);
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::ArrayPush(arr), 1);
    b.emit(Op::LoadInt(2), 1);
    b.emit(Op::ArrayPush(arr), 1);
    b.emit(Op::LoadInt(3), 1);
    b.emit(Op::ArrayPush(arr), 1);
    b.emit(Op::ArrayLen(arr), 1);
    match VM::new(b.build()).run() {
        VMResult::Ok(Value::Int(3)) => {}
        other => panic!("expected Int(3), got {:?}", other),
    }
}

// ── Hashes ──

#[test]
fn hash_set_get() {
    let mut b = ChunkBuilder::new();
    let h = b.add_name("h");
    let key = b.add_constant(Value::str("name"));
    let val = b.add_constant(Value::str("fusevm"));
    b.emit(Op::DeclareHash(h), 1);
    b.emit(Op::LoadConst(val), 1);
    b.emit(Op::LoadConst(key), 1);
    b.emit(Op::HashSet(h), 1);
    b.emit(Op::LoadConst(key), 1);
    b.emit(Op::HashGet(h), 1);
    match VM::new(b.build()).run() {
        VMResult::Ok(Value::Str(s)) => assert_eq!(s.as_str(), "fusevm"),
        other => panic!("expected Str('fusevm'), got {:?}", other),
    }
}

#[test]
fn hash_exists_and_delete() {
    let mut b = ChunkBuilder::new();
    let h = b.add_name("h");
    let key = b.add_constant(Value::str("k"));
    let val = b.add_constant(Value::str("v"));
    b.emit(Op::DeclareHash(h), 1);
    b.emit(Op::LoadConst(val), 1);
    b.emit(Op::LoadConst(key), 1);
    b.emit(Op::HashSet(h), 1);
    b.emit(Op::LoadConst(key), 1);
    b.emit(Op::HashExists(h), 1);
    match VM::new(b.build()).run() {
        VMResult::Ok(Value::Bool(true)) => {}
        other => panic!("expected true, got {:?}", other),
    }
}

#[test]
fn hash_keys_values() {
    let mut b = ChunkBuilder::new();
    let h = b.add_name("h");
    let k1 = b.add_constant(Value::str("a"));
    let v1 = b.add_constant(Value::Int(1));
    b.emit(Op::DeclareHash(h), 1);
    b.emit(Op::LoadConst(v1), 1);
    b.emit(Op::LoadConst(k1), 1);
    b.emit(Op::HashSet(h), 1);
    b.emit(Op::HashKeys(h), 1);
    match VM::new(b.build()).run() {
        VMResult::Ok(Value::Array(keys)) => {
            assert_eq!(keys.len(), 1);
            assert_eq!(keys[0].to_str(), "a");
        }
        other => panic!("expected Array with one key, got {:?}", other),
    }
}

#[test]
fn make_hash() {
    let mut b = ChunkBuilder::new();
    let k = b.add_constant(Value::str("x"));
    let v = b.add_constant(Value::Int(42));
    b.emit(Op::LoadConst(k), 1);
    b.emit(Op::LoadConst(v), 1);
    b.emit(Op::MakeHash(2), 1);
    match VM::new(b.build()).run() {
        VMResult::Ok(Value::Hash(map)) => {
            assert_eq!(map.get("x").unwrap().to_int(), 42);
        }
        other => panic!("expected Hash, got {:?}", other),
    }
}

// ── Range ──

#[test]
fn range_inclusive() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::LoadInt(5), 1);
    b.emit(Op::Range, 1);
    match VM::new(b.build()).run() {
        VMResult::Ok(Value::Array(arr)) => {
            let ints: Vec<i64> = arr.iter().map(|v| v.to_int()).collect();
            assert_eq!(ints, vec![1, 2, 3, 4, 5]);
        }
        other => panic!("expected Array [1..5], got {:?}", other),
    }
}

#[test]
fn range_step() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(0), 1);
    b.emit(Op::LoadInt(10), 1);
    b.emit(Op::LoadInt(3), 1);
    b.emit(Op::RangeStep, 1);
    match VM::new(b.build()).run() {
        VMResult::Ok(Value::Array(arr)) => {
            let ints: Vec<i64> = arr.iter().map(|v| v.to_int()).collect();
            assert_eq!(ints, vec![0, 3, 6, 9]);
        }
        other => panic!("expected Array [0,3,6,9], got {:?}", other),
    }
}

#[test]
fn range_step_negative() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(10), 1);
    b.emit(Op::LoadInt(0), 1);
    b.emit(Op::LoadInt(-3), 1);
    b.emit(Op::RangeStep, 1);
    match VM::new(b.build()).run() {
        VMResult::Ok(Value::Array(arr)) => {
            let ints: Vec<i64> = arr.iter().map(|v| v.to_int()).collect();
            assert_eq!(ints, vec![10, 7, 4, 1]);
        }
        other => panic!("expected Array [10,7,4,1], got {:?}", other),
    }
}

// ── Fused Superinstructions ──

#[test]
fn fused_accum_sum_1_to_1000() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::PushFrame, 1);
    b.emit(Op::LoadInt(0), 1);
    b.emit(Op::SetSlot(0), 1);
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::SetSlot(1), 1);
    b.emit(Op::AccumSumLoop(0, 1, 1001), 1);
    b.emit(Op::GetSlot(0), 1);
    match VM::new(b.build()).run() {
        VMResult::Ok(Value::Int(500500)) => {}
        other => panic!("expected Int(500500), got {:?}", other),
    }
}

#[test]
fn fused_slot_inc_lt_jump_back() {
    // count from 0 to 10 using the fused backedge op
    let mut b = ChunkBuilder::new();
    b.emit(Op::PushFrame, 1);
    b.emit(Op::LoadInt(0), 1);
    b.emit(Op::SetSlot(0), 1); // i = 0
    b.emit(Op::LoadInt(0), 1);
    b.emit(Op::SetSlot(1), 1); // sum = 0
                               // ip 5: loop body
    let body = b.current_pos();
    b.emit(Op::GetSlot(0), 1);
    b.emit(Op::GetSlot(1), 1);
    b.emit(Op::Add, 1);
    b.emit(Op::SetSlot(1), 1); // sum += i
    b.emit(Op::SlotIncLtIntJumpBack(0, 10, body), 1); // i++; if i < 10 goto body
    b.emit(Op::GetSlot(1), 1);
    match VM::new(b.build()).run() {
        VMResult::Ok(Value::Int(45)) => {} // 0+1+2+...+9 = 45
        other => panic!("expected Int(45), got {:?}", other),
    }
}

#[test]
fn fused_pre_inc_slot() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::PushFrame, 1);
    b.emit(Op::LoadInt(41), 1);
    b.emit(Op::SetSlot(0), 1);
    b.emit(Op::PreIncSlot(0), 1); // pushes 42 to stack
    match VM::new(b.build()).run() {
        VMResult::Ok(Value::Int(42)) => {}
        other => panic!("expected Int(42), got {:?}", other),
    }
}

#[test]
fn fused_add_assign_slot_void() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::PushFrame, 1);
    b.emit(Op::LoadInt(40), 1);
    b.emit(Op::SetSlot(0), 1);
    b.emit(Op::LoadInt(2), 1);
    b.emit(Op::SetSlot(1), 1);
    b.emit(Op::AddAssignSlotVoid(0, 1), 1);
    b.emit(Op::GetSlot(0), 1);
    match VM::new(b.build()).run() {
        VMResult::Ok(Value::Int(42)) => {}
        other => panic!("expected Int(42), got {:?}", other),
    }
}

#[test]
fn fused_concat_const_loop() {
    let mut b = ChunkBuilder::new();
    let c = b.add_constant(Value::str("x"));
    b.emit(Op::PushFrame, 1);
    // s_slot=0 = ""
    let empty = b.add_constant(Value::str(""));
    b.emit(Op::LoadConst(empty), 1);
    b.emit(Op::SetSlot(0), 1);
    // i_slot=1 = 0
    b.emit(Op::LoadInt(0), 1);
    b.emit(Op::SetSlot(1), 1);
    b.emit(Op::ConcatConstLoop(c, 0, 1, 5), 1);
    b.emit(Op::GetSlot(0), 1);
    match VM::new(b.build()).run() {
        VMResult::Ok(Value::Str(s)) => assert_eq!(s.as_str(), "xxxxx"),
        other => panic!("expected 'xxxxx', got {:?}", other),
    }
}

#[test]
fn fused_push_int_range_loop() {
    let mut b = ChunkBuilder::new();
    let arr = b.add_name("arr");
    b.emit(Op::DeclareArray(arr), 1);
    b.emit(Op::PushFrame, 1);
    b.emit(Op::LoadInt(0), 1);
    b.emit(Op::SetSlot(0), 1);
    b.emit(Op::PushIntRangeLoop(arr, 0, 5), 1);
    b.emit(Op::ArrayLen(arr), 1);
    match VM::new(b.build()).run() {
        VMResult::Ok(Value::Int(5)) => {}
        other => panic!("expected Int(5), got {:?}", other),
    }
}

// ── Functions ──

#[test]
fn nested_function_calls() {
    let mut b = ChunkBuilder::new();
    let add_one = b.add_name("add_one");
    let double = b.add_name("double");

    // main: double(add_one(20))
    b.emit(Op::LoadInt(20), 1);
    b.emit(Op::Call(add_one, 1), 1);
    b.emit(Op::Call(double, 1), 1);
    let end = b.emit(Op::Jump(0), 1);

    // add_one: arg + 1
    let add_one_ip = b.current_pos();
    b.add_sub_entry(add_one, add_one_ip);
    b.emit(Op::LoadInt(1), 2);
    b.emit(Op::Add, 2);
    b.emit(Op::ReturnValue, 2);

    // double: arg * 2
    let double_ip = b.current_pos();
    b.add_sub_entry(double, double_ip);
    b.emit(Op::LoadInt(2), 3);
    b.emit(Op::Mul, 3);
    b.emit(Op::ReturnValue, 3);

    b.patch_jump(end, b.current_pos());

    match VM::new(b.build()).run() {
        VMResult::Ok(Value::Int(42)) => {} // (20+1)*2 = 42
        other => panic!("expected Int(42), got {:?}", other),
    }
}

#[test]
fn undefined_function_errors() {
    let mut b = ChunkBuilder::new();
    let name = b.add_name("nonexistent");
    b.emit(Op::Call(name, 0), 1);
    match VM::new(b.build()).run() {
        VMResult::Error(msg) => assert!(msg.contains("nonexistent"), "error: {}", msg),
        other => panic!("expected error, got {:?}", other),
    }
}

// ── Extension Dispatch ──

#[test]
fn extension_handler() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(40), 1);
    b.emit(Op::Extended(0, 2), 1); // ext op 0 with arg 2: push arg as int
    b.emit(Op::Add, 1);

    let mut vm = VM::new(b.build());
    vm.set_extension_handler(Box::new(|vm, _id, arg| {
        vm.push(Value::Int(arg as i64));
    }));
    match vm.run() {
        VMResult::Ok(Value::Int(42)) => {}
        other => panic!("expected Int(42), got {:?}", other),
    }
}

#[test]
fn extension_wide_handler() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::ExtendedWide(0, 42), 1);

    let mut vm = VM::new(b.build());
    vm.set_extension_wide_handler(Box::new(|vm, _id, payload| {
        vm.push(Value::Int(payload as i64));
    }));
    match vm.run() {
        VMResult::Ok(Value::Int(42)) => {}
        other => panic!("expected Int(42), got {:?}", other),
    }
}

#[test]
fn extension_without_handler_is_noop() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(42), 1);
    b.emit(Op::Extended(99, 0), 1); // no handler registered
    match VM::new(b.build()).run() {
        VMResult::Ok(Value::Int(42)) => {}
        other => panic!("expected Int(42), got {:?}", other),
    }
}

// ── File Tests ──

#[test]
fn test_file_exists() {
    let mut b = ChunkBuilder::new();
    let path = b.add_constant(Value::str("Cargo.toml"));
    b.emit(Op::LoadConst(path), 1);
    b.emit(Op::TestFile(fusevm::op::file_test::EXISTS), 1);
    match VM::new(b.build()).run() {
        VMResult::Ok(Value::Bool(true)) => {}
        other => panic!("expected true for Cargo.toml exists, got {:?}", other),
    }
}

#[test]
fn test_file_is_file() {
    let mut b = ChunkBuilder::new();
    let path = b.add_constant(Value::str("Cargo.toml"));
    b.emit(Op::LoadConst(path), 1);
    b.emit(Op::TestFile(fusevm::op::file_test::IS_FILE), 1);
    match VM::new(b.build()).run() {
        VMResult::Ok(Value::Bool(true)) => {}
        other => panic!("expected true for Cargo.toml is_file, got {:?}", other),
    }
}

#[test]
fn test_file_is_dir() {
    let mut b = ChunkBuilder::new();
    let path = b.add_constant(Value::str("src"));
    b.emit(Op::LoadConst(path), 1);
    b.emit(Op::TestFile(fusevm::op::file_test::IS_DIR), 1);
    match VM::new(b.build()).run() {
        VMResult::Ok(Value::Bool(true)) => {}
        other => panic!("expected true for src/ is_dir, got {:?}", other),
    }
}

#[test]
fn test_nonexistent_file() {
    let mut b = ChunkBuilder::new();
    let path = b.add_constant(Value::str("/no/such/file/ever"));
    b.emit(Op::LoadConst(path), 1);
    b.emit(Op::TestFile(fusevm::op::file_test::EXISTS), 1);
    match VM::new(b.build()).run() {
        VMResult::Ok(Value::Bool(false)) => {}
        other => panic!("expected false, got {:?}", other),
    }
}

// ── JIT Eligibility ──

#[test]
fn jit_eligible_arithmetic_only() {
    use fusevm::JitCompiler;
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::LoadInt(2), 1);
    b.emit(Op::Add, 1);
    let chunk = b.build();
    let jit = JitCompiler::new();
    assert!(jit.is_eligible(&chunk));
}

#[test]
fn jit_ineligible_with_shell_ops() {
    use fusevm::JitCompiler;
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::Exec(1), 1);
    let chunk = b.build();
    let jit = JitCompiler::new();
    assert!(!jit.is_eligible(&chunk));
}

#[test]
fn jit_ineligible_unhandled_extension() {
    use fusevm::JitCompiler;
    let mut b = ChunkBuilder::new();
    b.emit(Op::Extended(999, 0), 1);
    let chunk = b.build();
    let jit = JitCompiler::new();
    assert!(!jit.is_eligible(&chunk));
}

// ── Chunk Serialization ──

#[test]
fn chunk_serde_roundtrip() {
    let mut b = ChunkBuilder::new();
    b.add_name("x");
    b.add_constant(Value::str("hello"));
    b.emit(Op::LoadInt(42), 1);
    b.emit(Op::LoadConst(0), 1);
    b.emit(Op::Add, 1);
    b.set_source("test.stk");
    let chunk = b.build();

    let serialized = serde_json::to_string(&chunk).unwrap();
    let deserialized: fusevm::Chunk = serde_json::from_str(&serialized).unwrap();

    assert_eq!(deserialized.ops.len(), 3);
    assert_eq!(deserialized.names.len(), 1);
    assert_eq!(deserialized.constants.len(), 1);
    assert_eq!(deserialized.source, "test.stk");
}

// ── Empty Program ──

#[test]
fn empty_chunk_halts() {
    let b = ChunkBuilder::new();
    match VM::new(b.build()).run() {
        VMResult::Halted => {}
        other => panic!("expected Halted, got {:?}", other),
    }
}

// ── Complex Programs ──

#[test]
fn fibonacci_iterative() {
    // fib(10) = 55
    // a=0, b=1, for i in 0..10 { tmp=a+b; a=b; b=tmp }
    // Two-pass: first emit with placeholder exit target, then fixup
    let mut b = ChunkBuilder::new();
    b.emit(Op::PushFrame, 1);
    b.emit(Op::LoadInt(0), 1);
    b.emit(Op::SetSlot(0), 1); // a = 0
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::SetSlot(1), 1); // b = 1
    b.emit(Op::LoadInt(0), 1);
    b.emit(Op::SetSlot(2), 1); // i = 0

    let loop_top = b.current_pos(); // ip 7
                                    // Use GetSlot + LoadInt + NumGe + JumpIfTrue for the exit condition
    b.emit(Op::GetSlot(2), 1);
    b.emit(Op::LoadInt(10), 1);
    b.emit(Op::NumGe, 1);
    let exit_jump = b.emit(Op::JumpIfTrue(0), 1); // placeholder

    // tmp = a + b → slot 3
    b.emit(Op::GetSlot(0), 1);
    b.emit(Op::GetSlot(1), 1);
    b.emit(Op::Add, 1);
    b.emit(Op::SetSlot(3), 1);
    // a = b
    b.emit(Op::GetSlot(1), 1);
    b.emit(Op::SetSlot(0), 1);
    // b = tmp
    b.emit(Op::GetSlot(3), 1);
    b.emit(Op::SetSlot(1), 1);

    b.emit(Op::PreIncSlotVoid(2), 1);
    b.emit(Op::Jump(loop_top), 1);

    let exit_ip = b.current_pos();
    b.patch_jump(exit_jump, exit_ip);
    b.emit(Op::GetSlot(0), 1); // result in a

    match VM::new(b.build()).run() {
        VMResult::Ok(Value::Int(55)) => {}
        other => panic!("expected Int(55) for fib(10), got {:?}", other),
    }
}

#[test]
fn chained_arithmetic() {
    // ((2 + 3) * 4 - 6) / 2 = 7.0
    expect_float(
        &[
            (Op::LoadInt(2), 1),
            (Op::LoadInt(3), 1),
            (Op::Add, 1),
            (Op::LoadInt(4), 1),
            (Op::Mul, 1),
            (Op::LoadInt(6), 1),
            (Op::Sub, 1),
            (Op::LoadInt(2), 1),
            (Op::Div, 1),
        ],
        7.0,
    );
}
