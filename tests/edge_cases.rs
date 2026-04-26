//! Edge case tests for the fusevm interpreter.
//!
//! VM testing patterns demonstrated here:
//!
//! 1. **Build → run → assert result**: emit a chunk, run it, check the value.
//!    Most VM tests are this shape.
//!
//! 2. **Error path coverage**: deliberately trigger error conditions and assert
//!    VMResult::Error or VMResult::Halted with the expected payload.
//!
//! 3. **Boundary inputs**: test the smallest/largest/zero/empty inputs to flush
//!    out off-by-one bugs and division-by-zero classes.
//!
//! 4. **State inspection**: don't only check the return value — check globals,
//!    slots, and stack state to catch bugs the return value would mask.
//!
//! 5. **Differential testing**: run the same computation via two paths (fused
//!    vs unfused, interpreter vs JIT) and assert results match. Catches
//!    regressions in either path.

use fusevm::{ChunkBuilder, Op, VMResult, Value, VM};

// ── Helper: run a chunk and return the result Value ──

fn run(ops: &[(Op, u32)]) -> Value {
    let mut b = ChunkBuilder::new();
    for (op, line) in ops {
        b.emit(op.clone(), *line);
    }
    let mut vm = VM::new(b.build());
    match vm.run() {
        VMResult::Ok(val) => val,
        VMResult::Halted => Value::Undef,
        VMResult::Error(e) => panic!("unexpected VM error: {}", e),
    }
}

fn run_int(ops: &[(Op, u32)]) -> i64 {
    match run(ops) {
        Value::Int(n) => n,
        Value::Float(f) => f as i64,
        other => panic!("expected Int, got {:?}", other),
    }
}

fn run_float(ops: &[(Op, u32)]) -> f64 {
    match run(ops) {
        Value::Float(f) => f,
        Value::Int(n) => n as f64,
        other => panic!("expected Float, got {:?}", other),
    }
}

fn run_bool(ops: &[(Op, u32)]) -> bool {
    match run(ops) {
        Value::Bool(b) => b,
        other => panic!("expected Bool, got {:?}", other),
    }
}

// ════════════════════════════════════════════════════════════════════════════
// PATTERN 1: Numeric edge cases — overflow, divide by zero, NaN
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn integer_add_wraps_on_overflow() {
    // Verify wrapping_add semantics: i64::MAX + 1 == i64::MIN
    let n = run_int(&[
        (Op::LoadInt(i64::MAX), 1),
        (Op::LoadInt(1), 1),
        (Op::Add, 1),
    ]);
    assert_eq!(n, i64::MIN);
}

#[test]
fn integer_sub_wraps_on_underflow() {
    let n = run_int(&[
        (Op::LoadInt(i64::MIN), 1),
        (Op::LoadInt(1), 1),
        (Op::Sub, 1),
    ]);
    assert_eq!(n, i64::MAX);
}

#[test]
fn integer_mul_wraps() {
    let n = run_int(&[
        (Op::LoadInt(i64::MAX), 1),
        (Op::LoadInt(2), 1),
        (Op::Mul, 1),
    ]);
    assert_eq!(n, i64::MAX.wrapping_mul(2));
}

#[test]
fn div_by_zero_int_returns_undef() {
    // Documented behavior: integer divisor of 0 yields Undef, not panic.
    let v = run(&[(Op::LoadInt(42), 1), (Op::LoadInt(0), 1), (Op::Div, 1)]);
    assert!(matches!(v, Value::Undef));
}

#[test]
fn mod_by_zero_int_returns_zero() {
    // arith_int_fast guards: y != 0 ? x % y : 0
    let n = run_int(&[(Op::LoadInt(42), 1), (Op::LoadInt(0), 1), (Op::Mod, 1)]);
    assert_eq!(n, 0);
}

#[test]
fn shift_amount_masked_to_low_6_bits() {
    // Shl/Shr mask the shift count with & 63 to avoid UB on overshift.
    // 1 << 64 should equal 1 << 0 = 1 (after masking).
    let n = run_int(&[(Op::LoadInt(1), 1), (Op::LoadInt(64), 1), (Op::Shl, 1)]);
    assert_eq!(n, 1);

    // 1 << 65 should equal 1 << 1 = 2
    let n = run_int(&[(Op::LoadInt(1), 1), (Op::LoadInt(65), 1), (Op::Shl, 1)]);
    assert_eq!(n, 2);
}

#[test]
fn float_arithmetic_with_nan() {
    // NaN propagates through arithmetic.
    let v = run(&[
        (Op::LoadFloat(f64::NAN), 1),
        (Op::LoadFloat(1.0), 1),
        (Op::Add, 1),
    ]);
    match v {
        Value::Float(f) => assert!(f.is_nan()),
        other => panic!("expected NaN, got {:?}", other),
    }
}

#[test]
fn float_arithmetic_infinity() {
    let f = run_float(&[
        (Op::LoadFloat(f64::INFINITY), 1),
        (Op::LoadFloat(1.0), 1),
        (Op::Add, 1),
    ]);
    assert!(f.is_infinite() && f > 0.0);
}

#[test]
fn float_eq_with_nan_is_false() {
    // NaN != NaN, even via the spaceship operator.
    let b = run_bool(&[
        (Op::LoadFloat(f64::NAN), 1),
        (Op::LoadFloat(f64::NAN), 1),
        (Op::NumEq, 1),
    ]);
    assert!(!b);
}

#[test]
fn negate_min_int_wraps() {
    // -i64::MIN can't be represented; wrapping_neg gives i64::MIN back.
    let n = run_int(&[(Op::LoadInt(i64::MIN), 1), (Op::Negate, 1)]);
    assert_eq!(n, i64::MIN);
}

#[test]
fn inc_max_int_wraps_to_min() {
    // i64::MAX + 1 wraps to i64::MIN
    let n = run_int(&[(Op::LoadInt(i64::MAX), 1), (Op::Inc, 1)]);
    assert_eq!(n, i64::MIN);
}

#[test]
fn dec_min_int_wraps_to_max() {
    let n = run_int(&[(Op::LoadInt(i64::MIN), 1), (Op::Dec, 1)]);
    assert_eq!(n, i64::MAX);
}

#[test]
fn pow_with_negative_exp_returns_zero() {
    // fusevm_jit_pow_i64 treats out-of-range exponents as 0 (matches VM).
    // VM uses .powf for negative exponents which gives 0.0..1.0 truncated to int.
    let v = run(&[(Op::LoadInt(2), 1), (Op::LoadInt(-1), 1), (Op::Pow, 1)]);
    // Pow always returns Float in the VM
    if let Value::Float(f) = v {
        assert!((f - 0.5).abs() < 1e-9);
    } else {
        panic!("expected Float, got {:?}", v);
    }
}

// ════════════════════════════════════════════════════════════════════════════
// PATTERN 2: Type coercion — to_int, to_float, to_str on all variants
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn bool_coerces_to_int_in_arithmetic() {
    // True acts as 1, False as 0
    let n = run_int(&[(Op::LoadTrue, 1), (Op::LoadInt(40), 1), (Op::Add, 1)]);
    // arith_int_fast checks Int×Int — Bool×Int falls through to to_float path
    assert_eq!(n, 41);
}

#[test]
fn status_zero_is_truthy_shell_semantics() {
    // In shell, exit code 0 = success = true. Verify is_truthy preserves this.
    assert!(Value::Status(0).is_truthy());
    assert!(!Value::Status(1).is_truthy());
    assert!(!Value::Status(127).is_truthy());
}

#[test]
fn empty_string_is_falsy() {
    assert!(!Value::str("").is_truthy());
    // Special case: "0" is also falsy (matches Perl/awk semantics)
    assert!(!Value::str("0").is_truthy());
    // But "00" is truthy
    assert!(Value::str("00").is_truthy());
    assert!(Value::str("0.0").is_truthy());
}

#[test]
fn empty_array_is_falsy() {
    assert!(!Value::Array(vec![]).is_truthy());
    assert!(Value::Array(vec![Value::Int(0)]).is_truthy());
}

#[test]
fn empty_hash_is_falsy() {
    use std::collections::HashMap;
    assert!(!Value::Hash(HashMap::new()).is_truthy());
    let mut m = HashMap::new();
    m.insert("k".to_string(), Value::Int(0));
    assert!(Value::Hash(m).is_truthy());
}

#[test]
fn string_to_int_parses_or_zero() {
    assert_eq!(Value::str("42").to_int(), 42);
    assert_eq!(Value::str("-7").to_int(), -7);
    assert_eq!(Value::str("not a number").to_int(), 0);
    assert_eq!(Value::str("").to_int(), 0);
}

#[test]
fn array_to_int_returns_length() {
    let arr = Value::Array(vec![Value::Int(1), Value::Int(2), Value::Int(3)]);
    assert_eq!(arr.to_int(), 3);
}

// ════════════════════════════════════════════════════════════════════════════
// PATTERN 3: Cow<str> coercion — borrowed vs owned semantics
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn as_str_cow_borrows_for_str_variant() {
    use std::borrow::Cow;
    let v = Value::str("hello");
    let cow = v.as_str_cow();
    assert!(matches!(cow, Cow::Borrowed(_)));
    assert_eq!(&*cow, "hello");
}

#[test]
fn as_str_cow_borrows_for_bool_undef_hash() {
    use std::borrow::Cow;
    use std::collections::HashMap;
    assert!(matches!(Value::Bool(true).as_str_cow(), Cow::Borrowed("1")));
    assert!(matches!(Value::Bool(false).as_str_cow(), Cow::Borrowed("")));
    assert!(matches!(Value::Undef.as_str_cow(), Cow::Borrowed("")));
    assert!(matches!(
        Value::Hash(HashMap::new()).as_str_cow(),
        Cow::Borrowed("(hash)")
    ));
}

#[test]
fn as_str_cow_allocates_for_int_float() {
    use std::borrow::Cow;
    assert!(matches!(Value::Int(42).as_str_cow(), Cow::Owned(_)));
    assert!(matches!(Value::Float(3.14).as_str_cow(), Cow::Owned(_)));
    // But the contents are correct
    assert_eq!(&*Value::Int(42).as_str_cow(), "42");
    assert_eq!(&*Value::Float(3.14).as_str_cow(), "3.14");
}

// ════════════════════════════════════════════════════════════════════════════
// PATTERN 4: Stack manipulation edge cases
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn pop_on_empty_stack_is_safe() {
    // VM should not panic; Pop on empty is a no-op (truncate to 0).
    let v = run(&[(Op::Pop, 1)]);
    assert!(matches!(v, Value::Undef));
}

#[test]
fn dup_preserves_value() {
    // Dup pushes a copy of TOS. After Dup, two equal values on stack.
    let n = run_int(&[
        (Op::LoadInt(7), 1),
        (Op::Dup, 1),
        (Op::Add, 1), // 7 + 7
    ]);
    assert_eq!(n, 14);
}

#[test]
fn swap_exchanges_top_two() {
    // Stack: [3, 7] → Swap → [7, 3] → Sub → 7 - 3 = 4
    let n = run_int(&[
        (Op::LoadInt(3), 1),
        (Op::LoadInt(7), 1),
        (Op::Swap, 1),
        (Op::Sub, 1),
    ]);
    assert_eq!(n, 4);
}

#[test]
fn rot_left_rotates_top_three() {
    // Stack: [a=10, b=20, c=30] → Rot → [b=20, c=30, a=10]
    // Sum: 20 + 30 = 50, then + 10 = 60
    let n = run_int(&[
        (Op::LoadInt(10), 1),
        (Op::LoadInt(20), 1),
        (Op::LoadInt(30), 1),
        (Op::Rot, 1),
        (Op::Add, 1), // 20 + 30 (top two after rot)
        (Op::Add, 1), // + 10 (which is now on top)
    ]);
    assert_eq!(n, 60);
}

#[test]
fn rot_with_only_two_elements_is_noop() {
    // Rot requires len >= 3; with 2 it should be a no-op.
    let n = run_int(&[
        (Op::LoadInt(5), 1),
        (Op::LoadInt(10), 1),
        (Op::Rot, 1),
        (Op::Sub, 1), // 5 - 10 = -5 (unchanged order)
    ]);
    assert_eq!(n, -5);
}

// ════════════════════════════════════════════════════════════════════════════
// PATTERN 5: String concat with mixed types
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn concat_int_and_string() {
    let mut b = ChunkBuilder::new();
    let s_idx = b.add_constant(Value::str(" items"));
    b.emit(Op::LoadInt(42), 1);
    b.emit(Op::LoadConst(s_idx), 1);
    b.emit(Op::Concat, 1);
    let mut vm = VM::new(b.build());
    match vm.run() {
        VMResult::Ok(Value::Str(s)) => assert_eq!(s.as_str(), "42 items"),
        other => panic!("expected '42 items', got {:?}", other),
    }
}

#[test]
fn concat_two_empty_strings() {
    let mut b = ChunkBuilder::new();
    let s_idx = b.add_constant(Value::str(""));
    b.emit(Op::LoadConst(s_idx), 1);
    b.emit(Op::LoadConst(s_idx), 1);
    b.emit(Op::Concat, 1);
    let mut vm = VM::new(b.build());
    match vm.run() {
        VMResult::Ok(Value::Str(s)) => assert!(s.is_empty()),
        other => panic!("expected empty string, got {:?}", other),
    }
}

#[test]
fn concat_undef_and_string_yields_string() {
    // Undef coerces to empty string in concat
    let mut b = ChunkBuilder::new();
    let s_idx = b.add_constant(Value::str("hello"));
    b.emit(Op::LoadUndef, 1);
    b.emit(Op::LoadConst(s_idx), 1);
    b.emit(Op::Concat, 1);
    let mut vm = VM::new(b.build());
    match vm.run() {
        VMResult::Ok(Value::Str(s)) => assert_eq!(s.as_str(), "hello"),
        other => panic!("got {:?}", other),
    }
}

// ════════════════════════════════════════════════════════════════════════════
// PATTERN 6: Differential testing — same computation, different paths
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn fused_and_unfused_sum_loops_match() {
    // Run the same sum loop two ways and verify they produce identical results.
    fn fused_sum(n: i32) -> i64 {
        let mut b = ChunkBuilder::new();
        b.emit(Op::PushFrame, 1);
        b.emit(Op::LoadInt(0), 1);
        b.emit(Op::SetSlot(0), 1);
        b.emit(Op::LoadInt(0), 1);
        b.emit(Op::SetSlot(1), 1);
        b.emit(Op::AccumSumLoop(0, 1, n), 1);
        b.emit(Op::GetSlot(0), 1);
        let mut vm = VM::new(b.build());
        if let VMResult::Ok(Value::Int(n)) = vm.run() {
            n
        } else {
            panic!()
        }
    }

    fn unfused_sum(n: i64) -> i64 {
        let mut b = ChunkBuilder::new();
        b.emit(Op::PushFrame, 1);
        b.emit(Op::LoadInt(0), 1);
        b.emit(Op::SetSlot(0), 1);
        b.emit(Op::LoadInt(0), 1);
        b.emit(Op::SetSlot(1), 1);
        // ip=5: body
        b.emit(Op::GetSlot(0), 1);
        b.emit(Op::GetSlot(1), 1);
        b.emit(Op::Add, 1);
        b.emit(Op::SetSlot(0), 1);
        b.emit(Op::PreIncSlotVoid(1), 1);
        b.emit(Op::SlotLtIntJumpIfFalse(1, n as i32, 12), 1);
        b.emit(Op::Jump(5), 1);
        // ip=12
        b.emit(Op::GetSlot(0), 1);
        let mut vm = VM::new(b.build());
        if let VMResult::Ok(Value::Int(n)) = vm.run() {
            n
        } else {
            panic!()
        }
    }

    // The mathematical truth: sum 0..N = N*(N-1)/2
    for n in [10, 100, 1000, 10000] {
        let expected = (n as i64) * (n as i64 - 1) / 2;
        assert_eq!(fused_sum(n), expected, "fused mismatch at n={}", n);
        assert_eq!(
            unfused_sum(n as i64),
            expected,
            "unfused mismatch at n={}",
            n
        );
        assert_eq!(fused_sum(n), unfused_sum(n as i64));
    }
}

#[cfg(feature = "jit")]
#[test]
fn block_jit_matches_interpreter_on_loop() {
    use fusevm::JitCompiler;
    // Run the same loop via interpreter and JIT, assert identical results.
    let mut b = ChunkBuilder::new();
    b.emit(Op::PushFrame, 1);
    b.emit(Op::LoadInt(0), 1);
    b.emit(Op::SetSlot(0), 1);
    b.emit(Op::LoadInt(0), 1);
    b.emit(Op::SetSlot(1), 1);
    b.emit(Op::AccumSumLoop(0, 1, 1000), 1);
    b.emit(Op::GetSlot(0), 1);
    let chunk = b.build();

    // Interpreter
    let mut vm = VM::new(chunk.clone());
    let interp = match vm.run() {
        VMResult::Ok(Value::Int(n)) => n,
        _ => panic!(),
    };

    // JIT
    let jit = JitCompiler::new();
    let mut slots = vec![0i64; 4];
    let jit_result = jit.try_run_block_eager(&chunk, &mut slots).unwrap();

    assert_eq!(interp, 499500); // sanity: sum 0..1000
    assert_eq!(interp, jit_result);
}

// ════════════════════════════════════════════════════════════════════════════
// PATTERN 7: State inspection — beyond the return value
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn global_var_persists_after_set() {
    let mut b = ChunkBuilder::new();
    let x = b.add_name("x");
    b.emit(Op::LoadInt(42), 1);
    b.emit(Op::SetVar(x), 1);
    b.emit(Op::GetVar(x), 1);
    let mut vm = VM::new(b.build());
    let _ = vm.run();
    // Inspect the global state directly
    assert!(matches!(vm.globals[x as usize], Value::Int(42)));
}

#[test]
fn array_in_place_mutation_persists() {
    let mut b = ChunkBuilder::new();
    let arr = b.add_name("arr");
    b.emit(Op::DeclareArray(arr), 1);
    for i in 1..=5 {
        b.emit(Op::LoadInt(i), 1);
        b.emit(Op::ArrayPush(arr), 1);
    }
    b.emit(Op::LoadInt(0), 1); // need a value left on stack
    let mut vm = VM::new(b.build());
    let _ = vm.run();
    // Inspect the array directly — verify in-place mutation worked
    if let Value::Array(ref v) = vm.globals[arr as usize] {
        assert_eq!(v.len(), 5);
        assert_eq!(v[0], Value::Int(1));
        assert_eq!(v[4], Value::Int(5));
    } else {
        panic!("expected array global");
    }
}

#[test]
fn hash_in_place_mutation_persists() {
    let mut b = ChunkBuilder::new();
    let h = b.add_name("h");
    let key1 = b.add_constant(Value::str("alpha"));
    let key2 = b.add_constant(Value::str("beta"));
    b.emit(Op::DeclareHash(h), 1);
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::LoadConst(key1), 1);
    b.emit(Op::HashSet(h), 1);
    b.emit(Op::LoadInt(2), 1);
    b.emit(Op::LoadConst(key2), 1);
    b.emit(Op::HashSet(h), 1);
    b.emit(Op::LoadInt(0), 1);
    let mut vm = VM::new(b.build());
    let _ = vm.run();
    // Inspect the hash global directly
    if let Value::Hash(ref m) = vm.globals[h as usize] {
        assert_eq!(m.len(), 2);
        assert_eq!(m.get("alpha"), Some(&Value::Int(1)));
        assert_eq!(m.get("beta"), Some(&Value::Int(2)));
    } else {
        panic!("expected hash global");
    }
}

#[test]
fn last_status_tracked_across_set_get() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(127), 1);
    b.emit(Op::SetStatus, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    match vm.run() {
        VMResult::Ok(Value::Status(127)) => {}
        other => panic!("expected Status(127), got {:?}", other),
    }
    assert_eq!(vm.last_status, 127);
}

// ════════════════════════════════════════════════════════════════════════════
// PATTERN 8: Error path coverage
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn calling_unknown_function_returns_error() {
    let mut b = ChunkBuilder::new();
    let name = b.add_name("does_not_exist");
    b.emit(Op::Call(name, 0), 1);
    let mut vm = VM::new(b.build());
    match vm.run() {
        VMResult::Error(msg) => assert!(msg.contains("does_not_exist")),
        other => panic!("expected error, got {:?}", other),
    }
}

#[test]
fn array_get_out_of_bounds_returns_undef() {
    // No panic on OOB array access — fusevm semantics return Undef.
    let mut b = ChunkBuilder::new();
    let arr = b.add_name("arr");
    b.emit(Op::DeclareArray(arr), 1);
    b.emit(Op::LoadInt(7), 1); // push value
    b.emit(Op::ArrayPush(arr), 1);
    b.emit(Op::LoadInt(99), 1); // OOB index
    b.emit(Op::ArrayGet(arr), 1);
    let mut vm = VM::new(b.build());
    match vm.run() {
        VMResult::Ok(Value::Undef) => {}
        other => panic!("expected Undef, got {:?}", other),
    }
}

#[test]
fn hash_get_missing_key_returns_undef() {
    let mut b = ChunkBuilder::new();
    let h = b.add_name("h");
    let key = b.add_constant(Value::str("missing"));
    b.emit(Op::DeclareHash(h), 1);
    b.emit(Op::LoadConst(key), 1);
    b.emit(Op::HashGet(h), 1);
    let mut vm = VM::new(b.build());
    match vm.run() {
        VMResult::Ok(Value::Undef) => {}
        other => panic!("got {:?}", other),
    }
}

#[test]
fn array_shift_on_empty_returns_undef() {
    let mut b = ChunkBuilder::new();
    let arr = b.add_name("arr");
    b.emit(Op::DeclareArray(arr), 1);
    b.emit(Op::ArrayShift(arr), 1);
    let mut vm = VM::new(b.build());
    match vm.run() {
        VMResult::Ok(Value::Undef) => {}
        other => panic!("got {:?}", other),
    }
}

// ════════════════════════════════════════════════════════════════════════════
// PATTERN 9: Higher-order behavior — functions returning early/late
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn function_can_return_early_with_returnvalue() {
    // A function that returns immediately on a condition.
    let mut b = ChunkBuilder::new();
    let fname = b.add_name("early");

    // Main: call early(0) → expect 99
    b.emit(Op::LoadInt(0), 1);
    b.emit(Op::Call(fname, 1), 1);
    let main_end = b.emit(Op::Jump(0), 1);

    // early(n): if n == 0 return 99, else return -1
    let fn_ip = b.current_pos();
    b.add_sub_entry(fname, fn_ip);
    b.emit(Op::SetSlot(0), 2); // n in slot 0
    b.emit(Op::GetSlot(0), 2);
    b.emit(Op::LoadInt(0), 2);
    b.emit(Op::NumEq, 2);
    let else_jump = b.emit(Op::JumpIfFalse(0), 2);
    b.emit(Op::LoadInt(99), 2);
    b.emit(Op::ReturnValue, 2);
    let else_ip = b.current_pos();
    b.patch_jump(else_jump, else_ip);
    b.emit(Op::LoadInt(-1), 2);
    b.emit(Op::ReturnValue, 2);

    let after = b.current_pos();
    b.patch_jump(main_end, after);

    let mut vm = VM::new(b.build());
    match vm.run() {
        VMResult::Ok(Value::Int(99)) => {}
        other => panic!("expected 99, got {:?}", other),
    }
}

#[test]
fn function_recursion_terminates_at_base() {
    // Compute factorial(5) = 120 recursively.
    let mut b = ChunkBuilder::new();
    let fname = b.add_name("fact");
    b.emit(Op::LoadInt(5), 1);
    b.emit(Op::Call(fname, 1), 1);
    let main_end = b.emit(Op::Jump(0), 1);

    let fn_ip = b.current_pos();
    b.add_sub_entry(fname, fn_ip);
    b.emit(Op::SetSlot(0), 2);
    b.emit(Op::GetSlot(0), 2);
    b.emit(Op::LoadInt(1), 2);
    b.emit(Op::NumLe, 2);
    let recurse_jump = b.emit(Op::JumpIfFalse(0), 2);
    // base case: return 1
    b.emit(Op::LoadInt(1), 2);
    b.emit(Op::ReturnValue, 2);
    // recursive: return n * fact(n-1)
    let recurse_ip = b.current_pos();
    b.patch_jump(recurse_jump, recurse_ip);
    b.emit(Op::GetSlot(0), 2); // n
    b.emit(Op::GetSlot(0), 2); // n
    b.emit(Op::Dec, 2); // n-1
    b.emit(Op::Call(fname, 1), 2); // fact(n-1)
    b.emit(Op::Mul, 2); // n * fact(n-1)
    b.emit(Op::ReturnValue, 2);

    let after = b.current_pos();
    b.patch_jump(main_end, after);

    let mut vm = VM::new(b.build());
    match vm.run() {
        VMResult::Ok(Value::Int(120)) => {}
        other => panic!("expected 120, got {:?}", other),
    }
}

// ════════════════════════════════════════════════════════════════════════════
// PATTERN 10: Serialization roundtrip — chunks survive serde
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn chunk_with_floats_roundtrips_via_json() {
    let mut b = ChunkBuilder::new();
    let s_idx = b.add_constant(Value::str("hello"));
    b.emit(Op::LoadFloat(3.14159), 1);
    b.emit(Op::LoadFloat(2.71828), 1);
    b.emit(Op::Mul, 1);
    b.emit(Op::LoadConst(s_idx), 1);
    b.emit(Op::StringLen, 1);
    let chunk = b.build();

    // Serialize → deserialize
    let json = serde_json::to_string(&chunk).expect("serialize");
    let restored: fusevm::Chunk = serde_json::from_str(&json).expect("deserialize");

    // Run both, results must match
    let mut vm_a = VM::new(chunk);
    let mut vm_b = VM::new(restored);
    let a = vm_a.run();
    let b = vm_b.run();
    let (Value::Int(a_n), Value::Int(b_n)) = (
        match a {
            VMResult::Ok(v) => v,
            _ => panic!(),
        },
        match b {
            VMResult::Ok(v) => v,
            _ => panic!(),
        },
    ) else {
        panic!("expected Int results");
    };
    assert_eq!(a_n, b_n);
}

// ════════════════════════════════════════════════════════════════════════════
// PATTERN 11: Builtin dispatch with closures capturing state
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn builtin_can_be_called_multiple_times() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(10), 1);
    b.emit(Op::CallBuiltin(0, 1), 1);
    b.emit(Op::LoadInt(5), 1);
    b.emit(Op::CallBuiltin(0, 1), 1);
    b.emit(Op::Add, 1);
    let mut vm = VM::new(b.build());
    // Builtin doubles its input
    vm.register_builtin(0, |vm, _argc| {
        let v = vm.pop();
        Value::Int(v.to_int() * 2)
    });
    match vm.run() {
        VMResult::Ok(Value::Int(30)) => {} // 20 + 10
        other => panic!("got {:?}", other),
    }
}

// ════════════════════════════════════════════════════════════════════════════
// PATTERN 12: Range bounds — empty ranges, reversed bounds, single element
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn empty_range_produces_empty_array() {
    // 5..3 produces no elements
    let v = run(&[(Op::LoadInt(5), 1), (Op::LoadInt(3), 1), (Op::Range, 1)]);
    if let Value::Array(ref a) = v {
        assert!(a.is_empty());
    } else {
        panic!("expected Array, got {:?}", v);
    }
}

#[test]
fn single_element_range() {
    // 5..5 produces [5]
    let v = run(&[(Op::LoadInt(5), 1), (Op::LoadInt(5), 1), (Op::Range, 1)]);
    if let Value::Array(ref a) = v {
        assert_eq!(a.len(), 1);
        assert_eq!(a[0], Value::Int(5));
    } else {
        panic!()
    }
}

#[test]
fn range_step_zero_yields_empty() {
    // step = 0 should not infinite-loop
    let v = run(&[
        (Op::LoadInt(0), 1),
        (Op::LoadInt(10), 1),
        (Op::LoadInt(0), 1),
        (Op::RangeStep, 1),
    ]);
    if let Value::Array(ref a) = v {
        assert!(a.is_empty());
    } else {
        panic!()
    }
}

#[test]
fn range_step_negative_descends() {
    // 10..1 step -2 → [10, 8, 6, 4, 2]
    let v = run(&[
        (Op::LoadInt(10), 1),
        (Op::LoadInt(1), 1),
        (Op::LoadInt(-2), 1),
        (Op::RangeStep, 1),
    ]);
    if let Value::Array(ref a) = v {
        assert_eq!(a.len(), 5);
        assert_eq!(a[0], Value::Int(10));
        assert_eq!(a[4], Value::Int(2));
    } else {
        panic!()
    }
}

// ════════════════════════════════════════════════════════════════════════════
// PATTERN 13: Logical ops — short-circuit semantics
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn log_and_evaluates_both_sides() {
    // LogAnd is the eager variant — evaluates both, unlike JumpIfFalseKeep.
    let b = run_bool(&[(Op::LoadTrue, 1), (Op::LoadFalse, 1), (Op::LogAnd, 1)]);
    assert!(!b);
}

#[test]
fn log_or_with_falsy_strings() {
    // "" || "0" both falsy → false
    let mut bb = ChunkBuilder::new();
    let e1 = bb.add_constant(Value::str(""));
    let e2 = bb.add_constant(Value::str("0"));
    bb.emit(Op::LoadConst(e1), 1);
    bb.emit(Op::LoadConst(e2), 1);
    bb.emit(Op::LogOr, 1);
    let mut vm = VM::new(bb.build());
    match vm.run() {
        VMResult::Ok(Value::Bool(false)) => {}
        other => panic!("got {:?}", other),
    }
}

#[test]
fn log_not_inverts_truthiness() {
    let b = run_bool(&[(Op::LoadInt(0), 1), (Op::LogNot, 1)]);
    assert!(b);
    let b = run_bool(&[(Op::LoadInt(42), 1), (Op::LogNot, 1)]);
    assert!(!b);
}
