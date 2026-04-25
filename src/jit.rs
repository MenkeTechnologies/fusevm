//! fusevm JIT — Cranelift codegen for universal bytecodes.
//!
//! Compiles fusevm::Op sequences to native machine code via Cranelift.
//! Language-specific ops (Extended) are handled by a `JitExtension` trait
//! that frontends register.
//!
//! # Architecture
//!
//! ```text
//! fusevm::Chunk
//!     │
//!     ▼
//! JitCompiler::compile()
//!     │
//!     ├── Universal ops: LoadInt, Add, Jump, NumEq, Call, ...
//!     │   → Cranelift IR directly
//!     │
//!     └── Extended(id, arg)
//!         → JitExtension::emit_extended() (registered by frontend)
//!     │
//!     ▼
//! NativeCode { fn_ptr } → call directly
//! ```
//!
//! # Two compilation tiers
//!
//! ## Linear JIT
//! Compiles straight-line (no branches) sequences in a single Cranelift basic block.
//! Stack must end with exactly one value.
//!
//! ## Block JIT
//! Compiles bytecodes with control flow (loops, conditionals, short-circuit &&/||) via
//! a basic-block CFG with abstract stack merges at block joins.
//!
//! # Eligible universal ops (both tiers)
//!
//! `LoadInt`, `LoadFloat`, `LoadConst` (int/float constants), `LoadTrue`, `LoadFalse`,
//! `Pop`, `Dup`, `Swap`, `Rot`,
//! `GetSlot`/`SetSlot`,
//! `Add`/`Sub`/`Mul`/`Div`/`Mod`/`Pow`/`Negate`/`Inc`/`Dec`,
//! `NumEq`/`NumNe`/`NumLt`/`NumGt`/`NumLe`/`NumGe`/`Spaceship`,
//! `BitAnd`/`BitOr`/`BitXor`/`BitNot`/`Shl`/`Shr`,
//! `LogNot`,
//! `Jump`/`JumpIfTrue`/`JumpIfFalse`/`JumpIfTrueKeep`/`JumpIfFalseKeep` (block tier only),
//! `PreIncSlot`/`PreIncSlotVoid`/`SlotLtIntJumpIfFalse`/`SlotIncLtIntJumpBack`/`AccumSumLoop`/`AddAssignSlotVoid`.

/// Extension trait for language-specific JIT codegen.
/// Frontends implement this to JIT their Extended ops.
pub trait JitExtension: Send + Sync {
    /// Whether this extension can JIT-compile the given extended op ID.
    fn can_jit(&self, ext_id: u16) -> bool;

    /// Number of extended ops this extension handles.
    fn op_count(&self) -> usize;

    /// Human-readable name for debugging.
    fn name(&self) -> &str;
}

// ── Cranelift JIT implementation (feature-gated) ──

#[cfg(feature = "jit")]
mod cranelift_jit_impl {
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};

    use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};
    use cranelift_codegen::ir::immediates::Ieee64;
    use cranelift_codegen::ir::types;
    use cranelift_codegen::ir::{AbiParam, InstBuilder, MemFlags, UserFuncName, Value};
    use cranelift_codegen::isa::OwnedTargetIsa;
    use cranelift_codegen::settings::{self, Configurable};
    use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
    use cranelift_jit::{JITBuilder, JITModule};
    use cranelift_module::{default_libcall_names, Linkage, Module};

    use crate::chunk::Chunk;
    use crate::op::Op;
    use crate::value::Value as FuseValue;

    // ── Types ──

    /// Whether the native function returns an integer or a float.
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    pub(crate) enum JitTy {
        Int,
        Float,
    }

    type LinearFn0 = unsafe extern "C" fn() -> i64;
    type LinearFn0F = unsafe extern "C" fn() -> f64;
    type LinearFnSlots = unsafe extern "C" fn(*const i64) -> i64;
    type LinearFnSlotsF = unsafe extern "C" fn(*const i64) -> f64;

    enum LinearRun {
        Nullary(LinearFn0),
        NullaryF(LinearFn0F),
        Slots(LinearFnSlots),
        SlotsF(LinearFnSlotsF),
    }

    pub(crate) enum JitResult {
        Int(i64),
        Float(f64),
    }

    pub(crate) struct CompiledLinear {
        #[allow(dead_code)]
        module: JITModule,
        run: LinearRun,
    }

    impl CompiledLinear {
        pub(crate) fn invoke(&self, slots: *const i64) -> JitResult {
            match &self.run {
                LinearRun::Nullary(f) => JitResult::Int(unsafe { f() }),
                LinearRun::NullaryF(f) => JitResult::Float(unsafe { f() }),
                LinearRun::Slots(f) => JitResult::Int(unsafe { f(slots) }),
                LinearRun::SlotsF(f) => JitResult::Float(unsafe { f(slots) }),
            }
        }

        pub(crate) fn result_to_value(&self, j: JitResult) -> FuseValue {
            match j {
                JitResult::Int(n) => FuseValue::Int(n),
                JitResult::Float(f) => FuseValue::Float(f),
            }
        }
    }

    // ── Abstract stack simulation ──

    #[derive(Clone, Copy, PartialEq, Debug)]
    pub(crate) enum Cell {
        Const(i64),
        ConstF(f64),
        Dyn,
        DynF,
    }

    impl Cell {
        fn is_float(self) -> bool {
            matches!(self, Cell::ConstF(_) | Cell::DynF)
        }
        fn either_float(a: Cell, b: Cell) -> bool {
            a.is_float() || b.is_float()
        }
    }

    fn cell_to_jit_ty(c: Cell) -> JitTy {
        match c {
            Cell::ConstF(_) | Cell::DynF => JitTy::Float,
            Cell::Const(_) | Cell::Dyn => JitTy::Int,
        }
    }

    fn pop2_strict(stack: &mut Vec<Cell>) -> Option<(Cell, Cell)> {
        let b = stack.pop()?;
        let a = stack.pop()?;
        Some((a, b))
    }

    fn fold_arith(
        a: Cell,
        b: Cell,
        int_op: fn(i64, i64) -> i64,
        f_op: fn(f64, f64) -> f64,
    ) -> Cell {
        match (a, b) {
            (Cell::Const(x), Cell::Const(y)) => Cell::Const(int_op(x, y)),
            (Cell::ConstF(x), Cell::ConstF(y)) => Cell::ConstF(f_op(x, y)),
            _ if Cell::either_float(a, b) => Cell::DynF,
            _ => Cell::Dyn,
        }
    }

    fn fold_cmp_cell(op: &Op, a: Cell, b: Cell) -> Cell {
        fn float_cmp(op: &Op, x: f64, y: f64) -> i64 {
            if x.is_nan() || y.is_nan() {
                return 0;
            }
            match op {
                Op::NumEq => i64::from(x == y),
                Op::NumNe => i64::from(x != y),
                Op::NumLt => i64::from(x < y),
                Op::NumGt => i64::from(x > y),
                Op::NumLe => i64::from(x <= y),
                Op::NumGe => i64::from(x >= y),
                Op::Spaceship => match x.partial_cmp(&y) {
                    Some(std::cmp::Ordering::Less) => -1,
                    Some(std::cmp::Ordering::Equal) => 0,
                    Some(std::cmp::Ordering::Greater) => 1,
                    None => 0,
                },
                _ => 0,
            }
        }
        match (a, b) {
            (Cell::Const(x), Cell::Const(y)) => {
                let v = match op {
                    Op::NumEq => i64::from(x == y),
                    Op::NumNe => i64::from(x != y),
                    Op::NumLt => i64::from(x < y),
                    Op::NumGt => i64::from(x > y),
                    Op::NumLe => i64::from(x <= y),
                    Op::NumGe => i64::from(x >= y),
                    Op::Spaceship => match x.cmp(&y) {
                        std::cmp::Ordering::Less => -1,
                        std::cmp::Ordering::Equal => 0,
                        std::cmp::Ordering::Greater => 1,
                    },
                    _ => 0,
                };
                Cell::Const(v)
            }
            (Cell::ConstF(x), Cell::ConstF(y)) => Cell::Const(float_cmp(op, x, y)),
            (Cell::Const(x), Cell::ConstF(y)) => Cell::Const(float_cmp(op, x as f64, y)),
            (Cell::ConstF(x), Cell::Const(y)) => Cell::Const(float_cmp(op, x, y as f64)),
            _ => Cell::Dyn,
        }
    }

    /// One op for abstract stack simulation (validation).
    fn simulate_one_op(op: &Op, stack: &mut Vec<Cell>, constants: &[FuseValue]) -> Option<()> {
        match op {
            Op::LoadInt(n) => stack.push(Cell::Const(*n)),
            Op::LoadFloat(f) => {
                if !f.is_finite() {
                    return None;
                }
                let n = *f as i64;
                if (n as f64) == *f {
                    stack.push(Cell::Const(n));
                } else {
                    stack.push(Cell::ConstF(*f));
                }
            }
            Op::LoadConst(idx) => {
                let val = constants.get(*idx as usize)?;
                match val {
                    FuseValue::Int(n) => stack.push(Cell::Const(*n)),
                    FuseValue::Float(f) => stack.push(Cell::ConstF(*f)),
                    _ => return None,
                }
            }
            Op::LoadTrue => stack.push(Cell::Const(1)),
            Op::LoadFalse => stack.push(Cell::Const(0)),
            Op::Add => {
                let (a, b) = pop2_strict(stack)?;
                stack.push(fold_arith(a, b, i64::wrapping_add, |x, y| x + y));
            }
            Op::Sub => {
                let (a, b) = pop2_strict(stack)?;
                stack.push(fold_arith(a, b, i64::wrapping_sub, |x, y| x - y));
            }
            Op::Mul => {
                let (a, b) = pop2_strict(stack)?;
                stack.push(fold_arith(a, b, i64::wrapping_mul, |x, y| x * y));
            }
            Op::Div => {
                let (a, b) = pop2_strict(stack)?;
                if Cell::either_float(a, b) {
                    match (a, b) {
                        (Cell::ConstF(x), Cell::ConstF(y)) => stack.push(Cell::ConstF(x / y)),
                        _ => stack.push(Cell::DynF),
                    }
                } else {
                    match (a, b) {
                        (Cell::Const(x), Cell::Const(y)) if y != 0 && x % y == 0 => {
                            stack.push(Cell::Const(x / y));
                        }
                        _ => return None,
                    }
                }
            }
            Op::Mod => {
                let (a, b) = pop2_strict(stack)?;
                if Cell::either_float(a, b) {
                    match (a, b) {
                        (Cell::ConstF(x), Cell::ConstF(y)) => stack.push(Cell::ConstF(x % y)),
                        _ => stack.push(Cell::DynF),
                    }
                } else {
                    match b {
                        Cell::Const(0) => return None,
                        Cell::Const(y) => stack.push(match a {
                            Cell::Const(x) => Cell::Const(x % y),
                            _ => Cell::Dyn,
                        }),
                        _ => return None,
                    }
                }
            }
            Op::Pow => {
                let (a, b) = pop2_strict(stack)?;
                if Cell::either_float(a, b) {
                    match (a, b) {
                        (Cell::ConstF(x), Cell::ConstF(y)) => stack.push(Cell::ConstF(x.powf(y))),
                        _ => stack.push(Cell::DynF),
                    }
                } else {
                    match (a, b) {
                        (Cell::Const(x), Cell::Const(y)) if (0..=63).contains(&y) => {
                            stack.push(Cell::Const(x.wrapping_pow(y as u32)));
                        }
                        (Cell::Dyn, Cell::Const(y)) if (0..=63).contains(&y) => {
                            stack.push(Cell::Dyn);
                        }
                        _ => return None,
                    }
                }
            }
            Op::Negate => {
                let a = stack.pop()?;
                stack.push(match a {
                    Cell::Const(n) => Cell::Const(n.wrapping_neg()),
                    Cell::ConstF(f) => Cell::ConstF(-f),
                    Cell::DynF => Cell::DynF,
                    Cell::Dyn => Cell::Dyn,
                });
            }
            Op::Inc => {
                let a = stack.pop()?;
                stack.push(match a {
                    Cell::Const(n) => Cell::Const(n.wrapping_add(1)),
                    _ => Cell::Dyn,
                });
            }
            Op::Dec => {
                let a = stack.pop()?;
                stack.push(match a {
                    Cell::Const(n) => Cell::Const(n.wrapping_sub(1)),
                    _ => Cell::Dyn,
                });
            }
            Op::Pop => {
                stack.pop()?;
            }
            Op::Dup => {
                let v = stack.last().copied()?;
                stack.push(v);
            }
            Op::Swap => {
                let b = stack.pop()?;
                let a = stack.pop()?;
                stack.push(b);
                stack.push(a);
            }
            Op::Rot => {
                let c = stack.pop()?;
                let b = stack.pop()?;
                let a = stack.pop()?;
                stack.push(b);
                stack.push(c);
                stack.push(a);
            }
            Op::NumEq
            | Op::NumNe
            | Op::NumLt
            | Op::NumGt
            | Op::NumLe
            | Op::NumGe
            | Op::Spaceship => {
                let (a, b) = pop2_strict(stack)?;
                stack.push(fold_cmp_cell(op, a, b));
            }
            Op::BitXor | Op::BitAnd | Op::BitOr | Op::Shl | Op::Shr => {
                let (a, b) = pop2_strict(stack)?;
                if Cell::either_float(a, b) {
                    return None;
                }
                stack.push(match (a, b) {
                    (Cell::Const(x), Cell::Const(y)) => Cell::Const(match op {
                        Op::BitXor => x ^ y,
                        Op::BitAnd => x & y,
                        Op::BitOr => x | y,
                        Op::Shl => x.wrapping_shl((y as u32) & 63),
                        Op::Shr => x.wrapping_shr((y as u32) & 63),
                        _ => unreachable!(),
                    }),
                    _ => Cell::Dyn,
                });
            }
            Op::BitNot => {
                let a = stack.pop()?;
                if a.is_float() {
                    return None;
                }
                stack.push(match a {
                    Cell::Const(n) => Cell::Const(!n),
                    _ => Cell::Dyn,
                });
            }
            Op::LogNot => {
                let a = stack.pop()?;
                stack.push(match a {
                    Cell::Const(n) => Cell::Const(if n != 0 { 0 } else { 1 }),
                    Cell::ConstF(f) => Cell::Const(if f != 0.0 { 0 } else { 1 }),
                    _ => Cell::Dyn,
                });
            }
            Op::GetSlot(_) => stack.push(Cell::Dyn),
            Op::SetSlot(_) => {
                stack.pop()?;
            }
            Op::PreIncSlot(_) => stack.push(Cell::Dyn),
            Op::PreIncSlotVoid(_) => {}
            Op::SlotLtIntJumpIfFalse(_, _, _) => {} // control flow, handled at block level
            Op::SlotIncLtIntJumpBack(_, _, _) => {} // control flow, handled at block level
            Op::AccumSumLoop(_, _, _) => {}         // self-contained fused op
            Op::AddAssignSlotVoid(_, _) => {}
            _ => return None,
        }
        Some(())
    }

    fn validate_linear_seq(seq: &[Op], constants: &[FuseValue]) -> bool {
        if seq.is_empty() {
            return false;
        }
        let mut stack: Vec<Cell> = Vec::new();
        for op in seq {
            if simulate_one_op(op, &mut stack, constants).is_none() {
                return false;
            }
            if stack.len() > 256 {
                return false;
            }
        }
        stack.len() == 1
    }

    fn linear_result_cell(seq: &[Op], constants: &[FuseValue]) -> Option<Cell> {
        let mut stack: Vec<Cell> = Vec::new();
        for op in seq {
            simulate_one_op(op, &mut stack, constants)?;
            if stack.len() > 256 {
                return None;
            }
        }
        if stack.len() != 1 {
            return None;
        }
        stack.pop()
    }

    fn needs_slots(seq: &[Op]) -> bool {
        seq.iter().any(|o| {
            matches!(
                o,
                Op::GetSlot(_)
                    | Op::SetSlot(_)
                    | Op::PreIncSlot(_)
                    | Op::PreIncSlotVoid(_)
                    | Op::SlotLtIntJumpIfFalse(_, _, _)
                    | Op::SlotIncLtIntJumpBack(_, _, _)
                    | Op::AccumSumLoop(_, _, _)
                    | Op::AddAssignSlotVoid(_, _)
            )
        })
    }

    // ── Cranelift setup ──

    fn isa_flags() -> cranelift_codegen::settings::Flags {
        let mut flag_builder = settings::builder();
        let _ = flag_builder.set("use_colocated_libcalls", "false");
        let _ = flag_builder.set("is_pic", "false");
        let _ = flag_builder.set("opt_level", "speed");
        settings::Flags::new(flag_builder)
    }

    static JIT_OWNED_ISA: OnceLock<Option<OwnedTargetIsa>> = OnceLock::new();

    fn cached_owned_isa() -> Option<&'static OwnedTargetIsa> {
        JIT_OWNED_ISA
            .get_or_init(|| {
                let isa_builder = cranelift_native::builder().ok()?;
                isa_builder.finish(isa_flags()).ok()
            })
            .as_ref()
    }

    /// Integer `**` matching vm.rs when both operands are `i64` and `0 <= exp <= 63`.
    #[no_mangle]
    pub extern "C" fn fusevm_jit_pow_i64(base: i64, exp: i64) -> i64 {
        if (0..=63).contains(&exp) {
            base.wrapping_pow(exp as u32)
        } else {
            0
        }
    }

    /// Float `**` — delegates to `f64::powf`.
    #[no_mangle]
    pub extern "C" fn fusevm_jit_pow_f64(base: f64, exp: f64) -> f64 {
        base.powf(exp)
    }

    /// Float `%` — delegates to `f64 % f64`.
    #[no_mangle]
    pub extern "C" fn fusevm_jit_fmod_f64(a: f64, b: f64) -> f64 {
        a % b
    }

    /// `!` on an i64 value (0 → 1, nonzero → 0).
    #[no_mangle]
    pub extern "C" fn fusevm_jit_lognot_i64(n: i64) -> i64 {
        if n != 0 {
            0
        } else {
            1
        }
    }

    fn new_jit_module() -> Option<JITModule> {
        let isa = cached_owned_isa()?.clone();
        let mut builder = JITBuilder::with_isa(isa, default_libcall_names());
        builder.symbol("fusevm_jit_pow_i64", fusevm_jit_pow_i64 as *const u8);
        builder.symbol("fusevm_jit_pow_f64", fusevm_jit_pow_f64 as *const u8);
        builder.symbol("fusevm_jit_fmod_f64", fusevm_jit_fmod_f64 as *const u8);
        builder.symbol("fusevm_jit_lognot_i64", fusevm_jit_lognot_i64 as *const u8);
        Some(JITModule::new(builder))
    }

    // ── Cranelift IR helpers ──

    fn intcmp_to_01(bcx: &mut FunctionBuilder, cc: IntCC, a: Value, b: Value) -> Value {
        let pred = bcx.ins().icmp(cc, a, b);
        let one = bcx.ins().iconst(types::I64, 1);
        let zero = bcx.ins().iconst(types::I64, 0);
        bcx.ins().select(pred, one, zero)
    }

    fn spaceship_i64(bcx: &mut FunctionBuilder, a: Value, b: Value) -> Value {
        let lt = bcx.ins().icmp(IntCC::SignedLessThan, a, b);
        let gt = bcx.ins().icmp(IntCC::SignedGreaterThan, a, b);
        let m1 = bcx.ins().iconst(types::I64, -1);
        let z = bcx.ins().iconst(types::I64, 0);
        let p1 = bcx.ins().iconst(types::I64, 1);
        let mid = bcx.ins().select(gt, p1, z);
        bcx.ins().select(lt, m1, mid)
    }

    fn floatcmp_to_01(bcx: &mut FunctionBuilder, cc: FloatCC, a: Value, b: Value) -> Value {
        let pred = bcx.ins().fcmp(cc, a, b);
        let one = bcx.ins().iconst(types::I64, 1);
        let zero = bcx.ins().iconst(types::I64, 0);
        bcx.ins().select(pred, one, zero)
    }

    fn spaceship_f64(bcx: &mut FunctionBuilder, a: Value, b: Value) -> Value {
        let lt = bcx.ins().fcmp(FloatCC::LessThan, a, b);
        let gt = bcx.ins().fcmp(FloatCC::GreaterThan, a, b);
        let m1 = bcx.ins().iconst(types::I64, -1);
        let z = bcx.ins().iconst(types::I64, 0);
        let p1 = bcx.ins().iconst(types::I64, 1);
        let mid = bcx.ins().select(gt, p1, z);
        bcx.ins().select(lt, m1, mid)
    }

    fn i64_to_f64(bcx: &mut FunctionBuilder, v: Value) -> Value {
        bcx.ins().fcvt_from_sint(types::F64, v)
    }

    fn f64_to_i64_trunc(bcx: &mut FunctionBuilder, v: Value) -> Value {
        bcx.ins().fcvt_to_sint(types::I64, v)
    }

    fn pop_pair_promote(
        bcx: &mut FunctionBuilder,
        stack: &mut Vec<(Value, JitTy)>,
    ) -> Option<(Value, Value, JitTy)> {
        let (b, tb) = stack.pop()?;
        let (a, ta) = stack.pop()?;
        Some(match (ta, tb) {
            (JitTy::Int, JitTy::Int) => (a, b, JitTy::Int),
            (JitTy::Float, JitTy::Float) => (a, b, JitTy::Float),
            (JitTy::Int, JitTy::Float) => (i64_to_f64(bcx, a), b, JitTy::Float),
            (JitTy::Float, JitTy::Int) => (a, i64_to_f64(bcx, b), JitTy::Float),
        })
    }

    fn scalar_store_i64(bcx: &mut FunctionBuilder, v: Value, ty: JitTy) -> Value {
        match ty {
            JitTy::Int => v,
            JitTy::Float => f64_to_i64_trunc(bcx, v),
        }
    }

    // ── Cranelift IR emission per op ──

    fn emit_data_op(
        bcx: &mut FunctionBuilder,
        op: &Op,
        stack: &mut Vec<(Value, JitTy)>,
        slot_base: Option<Value>,
        pow_i64_ref: Option<cranelift_codegen::ir::FuncRef>,
        pow_f64_ref: Option<cranelift_codegen::ir::FuncRef>,
        fmod_f64_ref: Option<cranelift_codegen::ir::FuncRef>,
        lognot_ref: Option<cranelift_codegen::ir::FuncRef>,
        constants: &[FuseValue],
    ) -> Option<()> {
        match op {
            Op::LoadInt(n) => {
                stack.push((bcx.ins().iconst(types::I64, *n), JitTy::Int));
            }
            Op::LoadFloat(f) => {
                if !f.is_finite() {
                    return None;
                }
                let n = *f as i64;
                if (n as f64) == *f {
                    stack.push((bcx.ins().iconst(types::I64, n), JitTy::Int));
                } else {
                    stack.push((
                        bcx.ins().f64const(Ieee64::with_bits(f.to_bits())),
                        JitTy::Float,
                    ));
                }
            }
            Op::LoadConst(idx) => {
                let val = constants.get(*idx as usize)?;
                match val {
                    FuseValue::Int(n) => {
                        stack.push((bcx.ins().iconst(types::I64, *n), JitTy::Int));
                    }
                    FuseValue::Float(f) => {
                        stack.push((
                            bcx.ins().f64const(Ieee64::with_bits(f.to_bits())),
                            JitTy::Float,
                        ));
                    }
                    _ => return None,
                }
            }
            Op::LoadTrue => stack.push((bcx.ins().iconst(types::I64, 1), JitTy::Int)),
            Op::LoadFalse => stack.push((bcx.ins().iconst(types::I64, 0), JitTy::Int)),
            Op::Add => {
                let (a, b, ty) = pop_pair_promote(bcx, stack)?;
                match ty {
                    JitTy::Int => stack.push((bcx.ins().iadd(a, b), JitTy::Int)),
                    JitTy::Float => stack.push((bcx.ins().fadd(a, b), JitTy::Float)),
                }
            }
            Op::Sub => {
                let (a, b, ty) = pop_pair_promote(bcx, stack)?;
                match ty {
                    JitTy::Int => stack.push((bcx.ins().isub(a, b), JitTy::Int)),
                    JitTy::Float => stack.push((bcx.ins().fsub(a, b), JitTy::Float)),
                }
            }
            Op::Mul => {
                let (a, b, ty) = pop_pair_promote(bcx, stack)?;
                match ty {
                    JitTy::Int => stack.push((bcx.ins().imul(a, b), JitTy::Int)),
                    JitTy::Float => stack.push((bcx.ins().fmul(a, b), JitTy::Float)),
                }
            }
            Op::Div => {
                let (a, b, ty) = pop_pair_promote(bcx, stack)?;
                match ty {
                    JitTy::Int => stack.push((bcx.ins().sdiv(a, b), JitTy::Int)),
                    JitTy::Float => stack.push((bcx.ins().fdiv(a, b), JitTy::Float)),
                }
            }
            Op::Mod => {
                let (a, b, ty) = pop_pair_promote(bcx, stack)?;
                match ty {
                    JitTy::Int => stack.push((bcx.ins().srem(a, b), JitTy::Int)),
                    JitTy::Float => {
                        let fr = fmod_f64_ref?;
                        let call = bcx.ins().call(fr, &[a, b]);
                        stack.push((*bcx.inst_results(call).first()?, JitTy::Float));
                    }
                }
            }
            Op::Pow => {
                let (a, b, ty) = pop_pair_promote(bcx, stack)?;
                match ty {
                    JitTy::Int => {
                        let fr = pow_i64_ref?;
                        let call = bcx.ins().call(fr, &[a, b]);
                        stack.push((*bcx.inst_results(call).first()?, JitTy::Int));
                    }
                    JitTy::Float => {
                        let fr = pow_f64_ref?;
                        let call = bcx.ins().call(fr, &[a, b]);
                        stack.push((*bcx.inst_results(call).first()?, JitTy::Float));
                    }
                }
            }
            Op::Negate => {
                let (a, ty) = stack.pop()?;
                match ty {
                    JitTy::Int => stack.push((bcx.ins().ineg(a), JitTy::Int)),
                    JitTy::Float => stack.push((bcx.ins().fneg(a), JitTy::Float)),
                }
            }
            Op::Inc => {
                let (a, ty) = stack.pop()?;
                let a = scalar_store_i64(bcx, a, ty);
                let one = bcx.ins().iconst(types::I64, 1);
                stack.push((bcx.ins().iadd(a, one), JitTy::Int));
            }
            Op::Dec => {
                let (a, ty) = stack.pop()?;
                let a = scalar_store_i64(bcx, a, ty);
                let one = bcx.ins().iconst(types::I64, 1);
                stack.push((bcx.ins().isub(a, one), JitTy::Int));
            }
            Op::NumEq => {
                let (a, b, ty) = pop_pair_promote(bcx, stack)?;
                let v = match ty {
                    JitTy::Int => intcmp_to_01(bcx, IntCC::Equal, a, b),
                    JitTy::Float => floatcmp_to_01(bcx, FloatCC::Equal, a, b),
                };
                stack.push((v, JitTy::Int));
            }
            Op::NumNe => {
                let (a, b, ty) = pop_pair_promote(bcx, stack)?;
                let v = match ty {
                    JitTy::Int => intcmp_to_01(bcx, IntCC::NotEqual, a, b),
                    JitTy::Float => floatcmp_to_01(bcx, FloatCC::NotEqual, a, b),
                };
                stack.push((v, JitTy::Int));
            }
            Op::NumLt => {
                let (a, b, ty) = pop_pair_promote(bcx, stack)?;
                let v = match ty {
                    JitTy::Int => intcmp_to_01(bcx, IntCC::SignedLessThan, a, b),
                    JitTy::Float => floatcmp_to_01(bcx, FloatCC::LessThan, a, b),
                };
                stack.push((v, JitTy::Int));
            }
            Op::NumGt => {
                let (a, b, ty) = pop_pair_promote(bcx, stack)?;
                let v = match ty {
                    JitTy::Int => intcmp_to_01(bcx, IntCC::SignedGreaterThan, a, b),
                    JitTy::Float => floatcmp_to_01(bcx, FloatCC::GreaterThan, a, b),
                };
                stack.push((v, JitTy::Int));
            }
            Op::NumLe => {
                let (a, b, ty) = pop_pair_promote(bcx, stack)?;
                let v = match ty {
                    JitTy::Int => intcmp_to_01(bcx, IntCC::SignedLessThanOrEqual, a, b),
                    JitTy::Float => floatcmp_to_01(bcx, FloatCC::LessThanOrEqual, a, b),
                };
                stack.push((v, JitTy::Int));
            }
            Op::NumGe => {
                let (a, b, ty) = pop_pair_promote(bcx, stack)?;
                let v = match ty {
                    JitTy::Int => intcmp_to_01(bcx, IntCC::SignedGreaterThanOrEqual, a, b),
                    JitTy::Float => floatcmp_to_01(bcx, FloatCC::GreaterThanOrEqual, a, b),
                };
                stack.push((v, JitTy::Int));
            }
            Op::Spaceship => {
                let (a, b, ty) = pop_pair_promote(bcx, stack)?;
                let v = match ty {
                    JitTy::Int => spaceship_i64(bcx, a, b),
                    JitTy::Float => spaceship_f64(bcx, a, b),
                };
                stack.push((v, JitTy::Int));
            }
            Op::LogNot => {
                let (a, ty) = stack.pop()?;
                let fr = lognot_ref?;
                let a_int = match ty {
                    JitTy::Int => a,
                    JitTy::Float => {
                        let z = bcx.ins().f64const(Ieee64::with_bits(0.0f64.to_bits()));
                        let pred = bcx.ins().fcmp(FloatCC::OrderedNotEqual, a, z);
                        let one = bcx.ins().iconst(types::I64, 1);
                        let zero = bcx.ins().iconst(types::I64, 0);
                        bcx.ins().select(pred, one, zero)
                    }
                };
                let call = bcx.ins().call(fr, &[a_int]);
                stack.push((*bcx.inst_results(call).first()?, JitTy::Int));
            }
            Op::Pop => {
                stack.pop()?;
            }
            Op::Dup => {
                let v = *stack.last()?;
                stack.push(v);
            }
            Op::Swap => {
                let (b, tb) = stack.pop()?;
                let (a, ta) = stack.pop()?;
                if ta != tb {
                    return None;
                }
                stack.push((b, tb));
                stack.push((a, ta));
            }
            Op::Rot => {
                let (c, tc) = stack.pop()?;
                let (b, tb) = stack.pop()?;
                let (a, ta) = stack.pop()?;
                if ta != tb || tb != tc {
                    return None;
                }
                stack.push((b, tb));
                stack.push((c, tc));
                stack.push((a, ta));
            }
            Op::BitXor => {
                let (b, tb) = stack.pop()?;
                let (a, ta) = stack.pop()?;
                if ta != JitTy::Int || tb != JitTy::Int {
                    return None;
                }
                stack.push((bcx.ins().bxor(a, b), JitTy::Int));
            }
            Op::BitAnd => {
                let (b, tb) = stack.pop()?;
                let (a, ta) = stack.pop()?;
                if ta != JitTy::Int || tb != JitTy::Int {
                    return None;
                }
                stack.push((bcx.ins().band(a, b), JitTy::Int));
            }
            Op::BitOr => {
                let (b, tb) = stack.pop()?;
                let (a, ta) = stack.pop()?;
                if ta != JitTy::Int || tb != JitTy::Int {
                    return None;
                }
                stack.push((bcx.ins().bor(a, b), JitTy::Int));
            }
            Op::BitNot => {
                let (a, ty) = stack.pop()?;
                if ty != JitTy::Int {
                    return None;
                }
                let ones = bcx.ins().iconst(types::I64, -1);
                stack.push((bcx.ins().bxor(a, ones), JitTy::Int));
            }
            Op::Shl => {
                let (b, tb) = stack.pop()?;
                let (a, ta) = stack.pop()?;
                if ta != JitTy::Int || tb != JitTy::Int {
                    return None;
                }
                let mask = bcx.ins().iconst(types::I64, 63);
                let mb = bcx.ins().band(b, mask);
                stack.push((bcx.ins().ishl(a, mb), JitTy::Int));
            }
            Op::Shr => {
                let (b, tb) = stack.pop()?;
                let (a, ta) = stack.pop()?;
                if ta != JitTy::Int || tb != JitTy::Int {
                    return None;
                }
                let mask = bcx.ins().iconst(types::I64, 63);
                let mb = bcx.ins().band(b, mask);
                stack.push((bcx.ins().sshr(a, mb), JitTy::Int));
            }
            Op::GetSlot(slot) => {
                let base = slot_base?;
                let off = (*slot as i32) * 8;
                stack.push((
                    bcx.ins().load(types::I64, MemFlags::trusted(), base, off),
                    JitTy::Int,
                ));
            }
            Op::SetSlot(slot) => {
                let base = slot_base?;
                let (v, ty) = stack.pop()?;
                let v = scalar_store_i64(bcx, v, ty);
                bcx.ins()
                    .store(MemFlags::trusted(), v, base, (*slot as i32) * 8);
            }
            Op::PreIncSlot(slot) => {
                let base = slot_base?;
                let off = (*slot as i32) * 8;
                let old = bcx.ins().load(types::I64, MemFlags::trusted(), base, off);
                let one = bcx.ins().iconst(types::I64, 1);
                let new = bcx.ins().iadd(old, one);
                bcx.ins().store(MemFlags::trusted(), new, base, off);
                stack.push((new, JitTy::Int));
            }
            Op::PreIncSlotVoid(slot) => {
                let base = slot_base?;
                let off = (*slot as i32) * 8;
                let old = bcx.ins().load(types::I64, MemFlags::trusted(), base, off);
                let one = bcx.ins().iconst(types::I64, 1);
                let new = bcx.ins().iadd(old, one);
                bcx.ins().store(MemFlags::trusted(), new, base, off);
            }
            Op::AddAssignSlotVoid(a_slot, b_slot) => {
                let base = slot_base?;
                let va =
                    bcx.ins()
                        .load(types::I64, MemFlags::trusted(), base, (*a_slot as i32) * 8);
                let vb =
                    bcx.ins()
                        .load(types::I64, MemFlags::trusted(), base, (*b_slot as i32) * 8);
                let sum = bcx.ins().iadd(va, vb);
                bcx.ins()
                    .store(MemFlags::trusted(), sum, base, (*a_slot as i32) * 8);
            }
            _ => return None,
        }
        Some(())
    }

    // ── Linear JIT compilation ──

    /// Compile a straight-line sequence of ops to native code.
    pub(crate) fn compile_linear(chunk: &Chunk) -> Option<CompiledLinear> {
        let seq = &chunk.ops;
        if !validate_linear_seq(seq, &chunk.constants) {
            return None;
        }
        let ret_cell = linear_result_cell(seq, &chunk.constants)?;
        let ret_ty = cell_to_jit_ty(ret_cell);
        let need_slots = needs_slots(seq);
        let mut module = new_jit_module()?;

        let needs_pow = seq.iter().any(|o| matches!(o, Op::Pow));
        let pow_i64_id = if needs_pow {
            let mut ps = module.make_signature();
            ps.params.push(AbiParam::new(types::I64));
            ps.params.push(AbiParam::new(types::I64));
            ps.returns.push(AbiParam::new(types::I64));
            Some(
                module
                    .declare_function("fusevm_jit_pow_i64", Linkage::Import, &ps)
                    .ok()?,
            )
        } else {
            None
        };
        let pow_f64_id = if needs_pow {
            let mut ps = module.make_signature();
            ps.params.push(AbiParam::new(types::F64));
            ps.params.push(AbiParam::new(types::F64));
            ps.returns.push(AbiParam::new(types::F64));
            Some(
                module
                    .declare_function("fusevm_jit_pow_f64", Linkage::Import, &ps)
                    .ok()?,
            )
        } else {
            None
        };
        let needs_fmod = seq.iter().any(|o| matches!(o, Op::Mod));
        let fmod_f64_id = if needs_fmod {
            let mut ps = module.make_signature();
            ps.params.push(AbiParam::new(types::F64));
            ps.params.push(AbiParam::new(types::F64));
            ps.returns.push(AbiParam::new(types::F64));
            Some(
                module
                    .declare_function("fusevm_jit_fmod_f64", Linkage::Import, &ps)
                    .ok()?,
            )
        } else {
            None
        };
        let needs_lognot = seq.iter().any(|o| matches!(o, Op::LogNot));
        let lognot_id = if needs_lognot {
            let mut ps = module.make_signature();
            ps.params.push(AbiParam::new(types::I64));
            ps.returns.push(AbiParam::new(types::I64));
            Some(
                module
                    .declare_function("fusevm_jit_lognot_i64", Linkage::Import, &ps)
                    .ok()?,
            )
        } else {
            None
        };

        let ptr_ty = module.target_config().pointer_type();
        let mut sig = module.make_signature();
        if need_slots {
            sig.params.push(AbiParam::new(ptr_ty));
        }
        sig.returns.push(AbiParam::new(match ret_ty {
            JitTy::Int => types::I64,
            JitTy::Float => types::F64,
        }));

        let fid = module
            .declare_function("linear", Linkage::Local, &sig)
            .ok()?;
        let mut ctx = module.make_context();
        ctx.func.signature = sig;
        ctx.func.name = UserFuncName::user(0, fid.as_u32());

        let mut fctx = FunctionBuilderContext::new();
        {
            let mut bcx = FunctionBuilder::new(&mut ctx.func, &mut fctx);
            let entry = bcx.create_block();
            bcx.append_block_params_for_function_params(entry);
            bcx.switch_to_block(entry);

            let slot_base = if need_slots {
                Some(bcx.block_params(entry)[0])
            } else {
                None
            };

            let pow_i64_ref = pow_i64_id.map(|pid| module.declare_func_in_func(pid, bcx.func));
            let pow_f64_ref = pow_f64_id.map(|pid| module.declare_func_in_func(pid, bcx.func));
            let fmod_f64_ref = fmod_f64_id.map(|pid| module.declare_func_in_func(pid, bcx.func));
            let lognot_ref = lognot_id.map(|lid| module.declare_func_in_func(lid, bcx.func));

            let mut stack: Vec<(cranelift_codegen::ir::Value, JitTy)> = Vec::with_capacity(32);
            for op in seq {
                emit_data_op(
                    &mut bcx,
                    op,
                    &mut stack,
                    slot_base,
                    pow_i64_ref,
                    pow_f64_ref,
                    fmod_f64_ref,
                    lognot_ref,
                    &chunk.constants,
                )?;
            }
            let (v, ty) = stack.pop()?;
            let ret_v = match (ret_ty, ty) {
                (JitTy::Int, JitTy::Int) | (JitTy::Float, JitTy::Float) => v,
                (JitTy::Float, JitTy::Int) => i64_to_f64(&mut bcx, v),
                (JitTy::Int, JitTy::Float) => f64_to_i64_trunc(&mut bcx, v),
            };
            bcx.ins().return_(&[ret_v]);
            bcx.seal_all_blocks();
            bcx.finalize();
        }

        module.define_function(fid, &mut ctx).ok()?;
        module.clear_context(&mut ctx);
        module.finalize_definitions().ok()?;
        let ptr = module.get_finalized_function(fid);
        let run = match (need_slots, ret_ty) {
            (false, JitTy::Int) => {
                LinearRun::Nullary(unsafe { std::mem::transmute::<*const u8, LinearFn0>(ptr) })
            }
            (false, JitTy::Float) => {
                LinearRun::NullaryF(unsafe { std::mem::transmute::<*const u8, LinearFn0F>(ptr) })
            }
            (true, JitTy::Int) => {
                LinearRun::Slots(unsafe { std::mem::transmute::<*const u8, LinearFnSlots>(ptr) })
            }
            (true, JitTy::Float) => {
                LinearRun::SlotsF(unsafe { std::mem::transmute::<*const u8, LinearFnSlotsF>(ptr) })
            }
        };
        Some(CompiledLinear { module, run })
    }

    // ── Linear JIT cache ──

    static LINEAR_CACHE: OnceLock<Mutex<HashMap<u64, Box<CompiledLinear>>>> = OnceLock::new();

    fn linear_cache() -> &'static Mutex<HashMap<u64, Box<CompiledLinear>>> {
        LINEAR_CACHE.get_or_init(|| Mutex::new(HashMap::new()))
    }

    fn hash_chunk_ops(chunk: &Chunk) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut h = DefaultHasher::new();
        chunk.ops.len().hash(&mut h);
        for op in &chunk.ops {
            format!("{op:?}").hash(&mut h);
        }
        for c in &chunk.constants {
            format!("{c:?}").hash(&mut h);
        }
        h.finish()
    }

    /// Try to JIT-compile and run a chunk's ops as a linear sequence.
    /// Returns `Some(Value)` on success, `None` if the chunk isn't eligible.
    pub(crate) fn try_run_linear(chunk: &Chunk, slots: &[i64]) -> Option<FuseValue> {
        let key = hash_chunk_ops(chunk);
        let cache = linear_cache();

        // Check cache first
        if let Ok(guard) = cache.lock() {
            if let Some(compiled) = guard.get(&key) {
                let slot_ptr = if slots.is_empty() {
                    std::ptr::null()
                } else {
                    slots.as_ptr()
                };
                let result = compiled.invoke(slot_ptr);
                return Some(compiled.result_to_value(result));
            }
        }

        // Compile
        let compiled = compile_linear(chunk)?;
        let slot_ptr = if slots.is_empty() {
            std::ptr::null()
        } else {
            slots.as_ptr()
        };
        let result = compiled.invoke(slot_ptr);
        let value = compiled.result_to_value(result);

        if let Ok(mut guard) = cache.lock() {
            guard.insert(key, Box::new(compiled));
        }

        Some(value)
    }

    /// Check if a chunk is eligible for linear JIT compilation.
    pub(crate) fn is_linear_eligible(chunk: &Chunk) -> bool {
        validate_linear_seq(&chunk.ops, &chunk.constants)
    }
}

// ── Public API (always available) ──

/// JIT compiler state.
pub struct JitCompiler {
    extensions: Vec<Box<dyn JitExtension>>,
}

impl JitCompiler {
    pub fn new() -> Self {
        Self {
            extensions: Vec::new(),
        }
    }

    /// Register a language-specific JIT extension.
    pub fn register_extension(&mut self, ext: Box<dyn JitExtension>) {
        tracing::info!(
            name = ext.name(),
            ops = ext.op_count(),
            "JIT extension registered"
        );
        self.extensions.push(ext);
    }

    /// Check if a chunk is eligible for JIT compilation.
    pub fn is_eligible(&self, chunk: &crate::Chunk) -> bool {
        use crate::Op;
        for op in &chunk.ops {
            match op {
                // Universal ops — always JIT-able
                Op::Nop
                | Op::LoadInt(_)
                | Op::LoadFloat(_)
                | Op::LoadConst(_)
                | Op::LoadTrue
                | Op::LoadFalse
                | Op::LoadUndef
                | Op::Pop
                | Op::Dup
                | Op::Dup2
                | Op::Swap
                | Op::Rot
                | Op::GetVar(_)
                | Op::SetVar(_)
                | Op::DeclareVar(_)
                | Op::GetSlot(_)
                | Op::SetSlot(_)
                | Op::Add
                | Op::Sub
                | Op::Mul
                | Op::Div
                | Op::Mod
                | Op::Pow
                | Op::Negate
                | Op::Inc
                | Op::Dec
                | Op::Concat
                | Op::StringRepeat
                | Op::StringLen
                | Op::NumEq
                | Op::NumNe
                | Op::NumLt
                | Op::NumGt
                | Op::NumLe
                | Op::NumGe
                | Op::StrEq
                | Op::StrNe
                | Op::StrLt
                | Op::StrGt
                | Op::StrLe
                | Op::StrGe
                | Op::StrCmp
                | Op::Spaceship
                | Op::LogNot
                | Op::LogAnd
                | Op::LogOr
                | Op::BitAnd
                | Op::BitOr
                | Op::BitXor
                | Op::BitNot
                | Op::Shl
                | Op::Shr
                | Op::Jump(_)
                | Op::JumpIfTrue(_)
                | Op::JumpIfFalse(_)
                | Op::JumpIfTrueKeep(_)
                | Op::JumpIfFalseKeep(_)
                | Op::Call(_, _)
                | Op::Return
                | Op::ReturnValue
                | Op::PushFrame
                | Op::PopFrame
                | Op::PreIncSlot(_)
                | Op::PreIncSlotVoid(_)
                | Op::SlotLtIntJumpIfFalse(_, _, _)
                | Op::SlotIncLtIntJumpBack(_, _, _)
                | Op::AccumSumLoop(_, _, _)
                | Op::AddAssignSlotVoid(_, _)
                | Op::SetStatus
                | Op::GetStatus => continue,

                // Extended — check if any extension handles it
                Op::Extended(id, _) | Op::ExtendedWide(id, _) => {
                    let id = *id;
                    if !self.extensions.iter().any(|ext| ext.can_jit(id)) {
                        return false;
                    }
                }

                _ => return false,
            }
        }
        true
    }

    /// Try to compile and run a chunk via the linear JIT.
    /// Returns `Some(Value)` on success, `None` if not eligible or JIT feature disabled.
    #[cfg(feature = "jit")]
    pub fn try_run_linear(&self, chunk: &crate::Chunk, slots: &[i64]) -> Option<crate::Value> {
        cranelift_jit_impl::try_run_linear(chunk, slots)
    }

    /// Check if a chunk is eligible for linear JIT.
    #[cfg(feature = "jit")]
    pub fn is_linear_eligible(&self, chunk: &crate::Chunk) -> bool {
        cranelift_jit_impl::is_linear_eligible(chunk)
    }

    /// Stub when JIT feature is disabled.
    #[cfg(not(feature = "jit"))]
    pub fn try_run_linear(&self, _chunk: &crate::Chunk, _slots: &[i64]) -> Option<crate::Value> {
        None
    }

    /// Stub when JIT feature is disabled.
    #[cfg(not(feature = "jit"))]
    pub fn is_linear_eligible(&self, _chunk: &crate::Chunk) -> bool {
        false
    }
}

impl Default for JitCompiler {
    fn default() -> Self {
        Self::new()
    }
}

/// Compiled native code handle (placeholder for block JIT — linear JIT is cached internally).
pub struct NativeCode {
    _private: (),
}
