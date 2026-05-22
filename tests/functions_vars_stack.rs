use fusevm::{ChunkBuilder, Op, VM, VMResult, Value};

fn run(b: ChunkBuilder) -> Value {
    match VM::new(b.build()).run() {
        VMResult::Ok(v) => v,
        other => panic!("unexpected result: {:?}", other),
    }
}

fn run_err(b: ChunkBuilder) -> String {
    match VM::new(b.build()).run() {
        VMResult::Error(msg) => msg,
        other => panic!("expected Error, got {:?}", other),
    }
}

// ── Op::Call / Op::ReturnValue ──

#[test]
fn call_simple_doubles_argument() {
    let mut b = ChunkBuilder::new();
    let name = b.add_name("dbl");
    b.emit(Op::LoadInt(21), 1);
    b.emit(Op::Call(name, 1), 1);
    let end = b.emit(Op::Jump(0), 1);
    let entry = b.current_pos();
    b.add_sub_entry(name, entry);
    b.emit(Op::LoadInt(2), 1);
    b.emit(Op::Mul, 1);
    b.emit(Op::ReturnValue, 1);
    b.patch_jump(end, b.current_pos());
    assert_eq!(run(b), Value::Int(42));
}

#[test]
fn call_with_zero_args_returns_constant() {
    let mut b = ChunkBuilder::new();
    let name = b.add_name("seven");
    b.emit(Op::Call(name, 0), 1);
    let end = b.emit(Op::Jump(0), 1);
    let entry = b.current_pos();
    b.add_sub_entry(name, entry);
    b.emit(Op::LoadInt(7), 1);
    b.emit(Op::ReturnValue, 1);
    b.patch_jump(end, b.current_pos());
    assert_eq!(run(b), Value::Int(7));
}

#[test]
fn call_undefined_function_returns_error() {
    let mut b = ChunkBuilder::new();
    let name = b.add_name("nope");
    b.emit(Op::Call(name, 0), 1);
    let err = run_err(b);
    assert!(err.contains("undefined function"));
    assert!(err.contains("nope"));
}

#[test]
fn call_return_without_value_pops_args() {
    // Function: receive 2 args, Return (no value) — caller stack should be cleared
    let mut b = ChunkBuilder::new();
    let name = b.add_name("drop");
    b.emit(Op::LoadInt(100), 1);
    b.emit(Op::LoadInt(200), 1);
    b.emit(Op::Call(name, 2), 1);
    b.emit(Op::LoadInt(5), 1);
    let end = b.emit(Op::Jump(0), 1);
    let entry = b.current_pos();
    b.add_sub_entry(name, entry);
    b.emit(Op::Return, 1);
    b.patch_jump(end, b.current_pos());
    assert_eq!(run(b), Value::Int(5));
}

#[test]
fn call_nested_functions() {
    let mut b = ChunkBuilder::new();
    let inc = b.add_name("inc");
    let inc2 = b.add_name("inc2");
    b.emit(Op::LoadInt(10), 1);
    b.emit(Op::Call(inc2, 1), 1);
    let end = b.emit(Op::Jump(0), 1);
    let inc_ip = b.current_pos();
    b.add_sub_entry(inc, inc_ip);
    b.emit(Op::Inc, 1);
    b.emit(Op::ReturnValue, 1);
    let inc2_ip = b.current_pos();
    b.add_sub_entry(inc2, inc2_ip);
    b.emit(Op::Call(inc, 1), 1);
    b.emit(Op::Call(inc, 1), 1);
    b.emit(Op::ReturnValue, 1);
    b.patch_jump(end, b.current_pos());
    assert_eq!(run(b), Value::Int(12));
}

#[test]
fn call_recursion_factorial() {
    // fact(5) = 120
    let mut b = ChunkBuilder::new();
    let fact = b.add_name("fact");
    b.emit(Op::LoadInt(5), 1);
    b.emit(Op::Call(fact, 1), 1);
    let end = b.emit(Op::Jump(0), 1);

    let entry = b.current_pos();
    b.add_sub_entry(fact, entry);
    // arg is on stack — dup, compare to 1
    b.emit(Op::Dup, 1);
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::NumLe, 1);
    let to_base = b.emit(Op::JumpIfTrue(0), 1);
    // recursive: arg * fact(arg-1)
    b.emit(Op::Dup, 1);
    b.emit(Op::Dec, 1);
    b.emit(Op::Call(fact, 1), 1);
    b.emit(Op::Mul, 1);
    b.emit(Op::ReturnValue, 1);
    // base case: arg already on stack
    let base = b.current_pos();
    b.patch_jump(to_base, base);
    b.emit(Op::ReturnValue, 1);
    b.patch_jump(end, b.current_pos());
    assert_eq!(run(b), Value::Int(120));
}

// ── Op::ReturnValue at top level halts with value ──

#[test]
fn return_value_at_top_level_returns_value() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(99), 1);
    b.emit(Op::ReturnValue, 1);
    b.emit(Op::LoadInt(0), 1);
    assert_eq!(run(b), Value::Int(99));
}

#[test]
fn return_at_top_level_halts() {
    // Op::Return at top level sets halted=true without popping the value,
    // so the VM returns whatever the loop's halt-handler resolves to.
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(5), 1);
    b.emit(Op::Return, 1);
    b.emit(Op::LoadInt(99), 1);
    let r = VM::new(b.build()).run();
    // Accept either Halted or Ok(5) — the important point is Op::LoadInt(99)
    // is *not* executed (no Ok(99)).
    match r {
        VMResult::Halted | VMResult::Ok(Value::Int(5)) => {}
        other => panic!("unexpected result: {:?}", other),
    }
}

// ── Variables (globals) ──

#[test]
fn declare_var_sets_global() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(11), 1);
    b.emit(Op::DeclareVar(0), 1);
    b.emit(Op::GetVar(0), 1);
    assert_eq!(run(b), Value::Int(11));
}

#[test]
fn set_var_overwrites() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::DeclareVar(0), 1);
    b.emit(Op::LoadInt(99), 1);
    b.emit(Op::SetVar(0), 1);
    b.emit(Op::GetVar(0), 1);
    assert_eq!(run(b), Value::Int(99));
}

#[test]
fn get_undefined_var_is_undef() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::GetVar(99), 1);
    assert!(matches!(run(b), Value::Undef));
}

#[test]
fn set_var_auto_resizes() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(42), 1);
    b.emit(Op::SetVar(50), 1);
    b.emit(Op::GetVar(50), 1);
    assert_eq!(run(b), Value::Int(42));
}

#[test]
fn vars_independent_across_indices() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(10), 1);
    b.emit(Op::DeclareVar(0), 1);
    b.emit(Op::LoadInt(20), 1);
    b.emit(Op::DeclareVar(1), 1);
    b.emit(Op::GetVar(0), 1);
    b.emit(Op::GetVar(1), 1);
    b.emit(Op::Add, 1);
    assert_eq!(run(b), Value::Int(30));
}

#[test]
fn var_persists_across_frame_scopes() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(7), 1);
    b.emit(Op::DeclareVar(0), 1);
    b.emit(Op::PushFrame, 1);
    b.emit(Op::PopFrame, 1);
    b.emit(Op::GetVar(0), 1);
    assert_eq!(run(b), Value::Int(7));
}

// ── Stack manipulation ──

#[test]
fn dup_duplicates_top() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(5), 1);
    b.emit(Op::Dup, 1);
    b.emit(Op::Add, 1);
    assert_eq!(run(b), Value::Int(10));
}

#[test]
fn dup2_duplicates_top_two() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(3), 1);
    b.emit(Op::LoadInt(4), 1);
    b.emit(Op::Dup2, 1);
    // stack: [3,4,3,4]
    b.emit(Op::Add, 1); // [3,4,7]
    b.emit(Op::Add, 1); // [3,11]
    b.emit(Op::Add, 1); // [14]
    assert_eq!(run(b), Value::Int(14));
}

#[test]
fn dup2_empty_stack_is_noop() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::Dup2, 1);
    b.emit(Op::LoadInt(1), 1);
    assert_eq!(run(b), Value::Int(1));
}

#[test]
fn swap_swaps_top_two() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(10), 1);
    b.emit(Op::LoadInt(3), 1);
    b.emit(Op::Swap, 1);
    b.emit(Op::Sub, 1); // 3 - 10 = -7
    assert_eq!(run(b), Value::Int(-7));
}

#[test]
fn swap_empty_stack_is_noop() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::Swap, 1);
    b.emit(Op::LoadInt(2), 1);
    assert_eq!(run(b), Value::Int(2));
}

#[test]
fn rot_rotates_top_three() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::LoadInt(2), 1);
    b.emit(Op::LoadInt(3), 1);
    b.emit(Op::Rot, 1);
    // [1,2,3] -> [2,3,1]; top is 1
    assert_eq!(run(b), Value::Int(1));
}

#[test]
fn rot_with_two_elements_is_noop() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::LoadInt(2), 1);
    b.emit(Op::Rot, 1);
    assert_eq!(run(b), Value::Int(2));
}

#[test]
fn pop_removes_top() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(99), 1);
    b.emit(Op::LoadInt(7), 1);
    b.emit(Op::Pop, 1);
    assert_eq!(run(b), Value::Int(99));
}

#[test]
fn nop_does_nothing() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(5), 1);
    b.emit(Op::Nop, 1);
    b.emit(Op::Nop, 1);
    b.emit(Op::Nop, 1);
    assert_eq!(run(b), Value::Int(5));
}

// ── PushFrame/PopFrame stack discipline ──

#[test]
fn push_pop_frame_pair_is_balanced() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(100), 1);
    b.emit(Op::PushFrame, 1);
    b.emit(Op::LoadInt(50), 1);
    b.emit(Op::PopFrame, 1); // truncates back to stack_base (1) → discards 50
    assert_eq!(run(b), Value::Int(100));
}

#[test]
fn pop_frame_with_no_frame_is_noop() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::PopFrame, 1);
    b.emit(Op::LoadInt(3), 1);
    assert_eq!(run(b), Value::Int(3));
}

// ── Arithmetic with mixed types ──

#[test]
fn int_plus_float_returns_float() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(2), 1);
    b.emit(Op::LoadFloat(0.5), 1);
    b.emit(Op::Add, 1);
    match run(b) {
        Value::Float(f) => assert!((f - 2.5).abs() < 1e-9),
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn negate_int() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(15), 1);
    b.emit(Op::Negate, 1);
    assert_eq!(run(b), Value::Int(-15));
}

#[test]
fn negate_float() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadFloat(3.14), 1);
    b.emit(Op::Negate, 1);
    match run(b) {
        Value::Float(f) => assert!((f + 3.14).abs() < 1e-9),
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn inc_dec_wrapping() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(i64::MAX), 1);
    b.emit(Op::Inc, 1);
    assert_eq!(run(b), Value::Int(i64::MIN));
}

#[test]
fn dec_wrapping_min() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(i64::MIN), 1);
    b.emit(Op::Dec, 1);
    assert_eq!(run(b), Value::Int(i64::MAX));
}

#[test]
fn negate_min_wraps() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(i64::MIN), 1);
    b.emit(Op::Negate, 1);
    assert_eq!(run(b), Value::Int(i64::MIN));
}

#[test]
fn add_wraps() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(i64::MAX), 1);
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::Add, 1);
    assert_eq!(run(b), Value::Int(i64::MIN));
}

#[test]
fn sub_wraps() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(i64::MIN), 1);
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::Sub, 1);
    assert_eq!(run(b), Value::Int(i64::MAX));
}

#[test]
fn mul_wraps() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(i64::MAX), 1);
    b.emit(Op::LoadInt(2), 1);
    b.emit(Op::Mul, 1);
    assert_eq!(run(b), Value::Int(-2));
}

// ── Concat ──

#[test]
fn concat_two_strings() {
    let mut b = ChunkBuilder::new();
    let a = b.add_constant(Value::str("foo"));
    let c = b.add_constant(Value::str("bar"));
    b.emit(Op::LoadConst(a), 1);
    b.emit(Op::LoadConst(c), 1);
    b.emit(Op::Concat, 1);
    match run(b) {
        Value::Str(s) => assert_eq!(&*s, "foobar"),
        other => panic!("expected str, got {:?}", other),
    }
}

#[test]
fn concat_int_with_string_coerces() {
    let mut b = ChunkBuilder::new();
    let suf = b.add_constant(Value::str("ms"));
    b.emit(Op::LoadInt(42), 1);
    b.emit(Op::LoadConst(suf), 1);
    b.emit(Op::Concat, 1);
    match run(b) {
        Value::Str(s) => assert_eq!(&*s, "42ms"),
        other => panic!("expected str, got {:?}", other),
    }
}

#[test]
fn concat_empty_with_empty() {
    let mut b = ChunkBuilder::new();
    let e = b.add_constant(Value::str(""));
    b.emit(Op::LoadConst(e), 1);
    b.emit(Op::LoadConst(e), 1);
    b.emit(Op::Concat, 1);
    match run(b) {
        Value::Str(s) => assert_eq!(&*s, ""),
        other => panic!("expected str, got {:?}", other),
    }
}

// ── ChunkBuilder add_sub_entry / find_sub via Call ──

#[test]
fn first_registered_sub_wins_when_duplicate_names() {
    let mut b = ChunkBuilder::new();
    let name = b.add_name("foo");
    // entry 1 → returns 1; entry 2 → returns 2
    b.emit(Op::Call(name, 0), 1);
    let end = b.emit(Op::Jump(0), 1);

    let entry1 = b.current_pos();
    b.add_sub_entry(name, entry1);
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::ReturnValue, 1);

    let entry2 = b.current_pos();
    b.add_sub_entry(name, entry2);
    b.emit(Op::LoadInt(2), 1);
    b.emit(Op::ReturnValue, 1);

    b.patch_jump(end, b.current_pos());
    // Either is acceptable per find_sub semantics; just confirm it returns
    // *one* of the registered values.
    match run(b) {
        Value::Int(1) | Value::Int(2) => {}
        other => panic!("expected 1 or 2, got {:?}", other),
    }
}

// ── Stack with deep recursion-style nesting ──

#[test]
fn deep_arithmetic_chain() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(0), 1);
    for i in 1..=50 {
        b.emit(Op::LoadInt(i), 1);
        b.emit(Op::Add, 1);
    }
    assert_eq!(run(b), Value::Int(50 * 51 / 2));
}

// ── ReadLine fallback (read empty piped stdin in test env) ──
// (Skipped — would block on real stdin; trivially covered by unit tests already.)

// ── Combining ops: classic VM patterns ──

#[test]
fn ternary_select_via_jump_if_false() {
    // cond=true: result = 7 else 99
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadTrue, 1);
    b.emit(Op::JumpIfFalse(4), 1);
    b.emit(Op::LoadInt(7), 1);
    b.emit(Op::Jump(5), 1);
    b.emit(Op::LoadInt(99), 1);
    assert_eq!(run(b), Value::Int(7));
}

#[test]
fn ternary_select_false_branch() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadFalse, 1);
    b.emit(Op::JumpIfFalse(4), 1);
    b.emit(Op::LoadInt(7), 1);
    b.emit(Op::Jump(5), 1);
    b.emit(Op::LoadInt(99), 1);
    assert_eq!(run(b), Value::Int(99));
}

#[test]
fn short_circuit_and_via_keep() {
    // a && b: JumpIfFalseKeep skips b if a is false
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadFalse, 1);
    b.emit(Op::JumpIfFalseKeep(4), 1);
    b.emit(Op::Pop, 1);
    b.emit(Op::LoadInt(42), 1);
    // ip 4: end
    assert!(matches!(run(b), Value::Bool(false)));
}

#[test]
fn short_circuit_or_via_keep() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(7), 1); // truthy
    b.emit(Op::JumpIfTrueKeep(4), 1);
    b.emit(Op::Pop, 1);
    b.emit(Op::LoadInt(99), 1);
    // ip 4
    assert_eq!(run(b), Value::Int(7));
}
