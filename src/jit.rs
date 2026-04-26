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
    use std::sync::OnceLock;

    use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};
    use cranelift_codegen::ir::immediates::Ieee64;
    use cranelift_codegen::ir::types;
    use cranelift_codegen::ir::{AbiParam, BlockArg, InstBuilder, MemFlags, UserFuncName, Value};
    use cranelift_codegen::isa::OwnedTargetIsa;
    use cranelift_codegen::settings::{self, Configurable};
    use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
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

    // ── Linear JIT cache (per-thread, lock-free) ──

    thread_local! {
        static LINEAR_CACHE_TLS: std::cell::RefCell<HashMap<u64, Box<CompiledLinear>>> =
            std::cell::RefCell::new(HashMap::new());
    }

    /// Try to JIT-compile and run a chunk's ops as a linear sequence.
    /// Returns `Some(Value)` on success, `None` if the chunk isn't eligible.
    pub(crate) fn try_run_linear(chunk: &Chunk, slots: &[i64]) -> Option<FuseValue> {
        let key = chunk.op_hash;
        let slot_ptr = if slots.is_empty() {
            std::ptr::null()
        } else {
            slots.as_ptr()
        };

        // Cache hit: invoke and return
        let cached = LINEAR_CACHE_TLS.with(|cache_cell| {
            let cache = cache_cell.borrow();
            cache.get(&key).map(|c| {
                let result = c.invoke(slot_ptr);
                c.result_to_value(result)
            })
        });
        if let Some(v) = cached {
            return Some(v);
        }

        // Cache miss: compile, invoke, store
        let compiled = compile_linear(chunk)?;
        let result = compiled.invoke(slot_ptr);
        let value = compiled.result_to_value(result);

        LINEAR_CACHE_TLS.with(|cache_cell| {
            cache_cell.borrow_mut().insert(key, Box::new(compiled));
        });

        Some(value)
    }

    /// Check if a chunk is eligible for linear JIT compilation.
    pub(crate) fn is_linear_eligible(chunk: &Chunk) -> bool {
        validate_linear_seq(&chunk.ops, &chunk.constants)
    }

    // ── Block JIT compilation ──

    use std::collections::BTreeSet;

    // Specialized block-JIT function pointer types.
    // Saves a register by omitting the slot pointer when the chunk doesn't use slots.
    type BlockFnSlotsI = unsafe extern "C" fn(*mut i64) -> i64;
    type BlockFnSlotsF = unsafe extern "C" fn(*mut i64) -> f64;
    type BlockFnNoSlotsI = unsafe extern "C" fn() -> i64;
    type BlockFnNoSlotsF = unsafe extern "C" fn() -> f64;

    #[allow(dead_code)] // Variants reserved for future signature specialization.
    pub(crate) enum BlockRun {
        SlotsI(BlockFnSlotsI),
        SlotsF(BlockFnSlotsF),
        NoSlotsI(BlockFnNoSlotsI),
        NoSlotsF(BlockFnNoSlotsF),
    }

    pub(crate) struct CompiledBlock {
        #[allow(dead_code)]
        module: JITModule,
        run: BlockRun,
    }

    impl CompiledBlock {
        /// Invoke and return the result as i64 (truncating float results).
        pub(crate) fn invoke(&self, slots: &mut [i64]) -> i64 {
            let ptr = if slots.is_empty() {
                std::ptr::null_mut()
            } else {
                slots.as_mut_ptr()
            };
            match &self.run {
                BlockRun::SlotsI(f) => unsafe { f(ptr) },
                BlockRun::NoSlotsI(f) => unsafe { f() },
                BlockRun::SlotsF(f) => (unsafe { f(ptr) }) as i64,
                BlockRun::NoSlotsF(f) => (unsafe { f() }) as i64,
            }
        }
    }

    fn find_leaders(ops: &[Op]) -> BTreeSet<usize> {
        let mut leaders = BTreeSet::new();
        if ops.is_empty() {
            return leaders;
        }
        leaders.insert(0);
        for (ip, op) in ops.iter().enumerate() {
            match op {
                Op::Jump(t) => {
                    leaders.insert(*t);
                    if ip + 1 < ops.len() {
                        leaders.insert(ip + 1);
                    }
                }
                Op::JumpIfTrue(t) | Op::JumpIfFalse(t) => {
                    leaders.insert(*t);
                    if ip + 1 < ops.len() {
                        leaders.insert(ip + 1);
                    }
                }
                Op::SlotLtIntJumpIfFalse(_, _, t) | Op::SlotIncLtIntJumpBack(_, _, t) => {
                    leaders.insert(*t);
                    if ip + 1 < ops.len() {
                        leaders.insert(ip + 1);
                    }
                }
                Op::AccumSumLoop(_, _, _) => {
                    if ip + 1 < ops.len() {
                        leaders.insert(ip + 1);
                    }
                }
                _ => {}
            }
        }
        leaders
    }

    fn is_block_eligible_op(op: &Op) -> bool {
        matches!(
            op,
            Op::Nop
                | Op::LoadInt(_)
                | Op::LoadFloat(_)
                | Op::LoadConst(_)
                | Op::LoadTrue
                | Op::LoadFalse
                | Op::Pop
                | Op::Dup
                | Op::Swap
                | Op::Rot
                | Op::Add
                | Op::Sub
                | Op::Mul
                | Op::Div
                | Op::Mod
                | Op::Pow
                | Op::Negate
                | Op::Inc
                | Op::Dec
                | Op::NumEq
                | Op::NumNe
                | Op::NumLt
                | Op::NumGt
                | Op::NumLe
                | Op::NumGe
                | Op::Spaceship
                | Op::BitAnd
                | Op::BitOr
                | Op::BitXor
                | Op::BitNot
                | Op::Shl
                | Op::Shr
                | Op::LogNot
                | Op::GetSlot(_)
                | Op::SetSlot(_)
                | Op::PreIncSlot(_)
                | Op::PreIncSlotVoid(_)
                | Op::AddAssignSlotVoid(_, _)
                | Op::Jump(_)
                | Op::JumpIfTrue(_)
                | Op::JumpIfFalse(_)
                | Op::SlotLtIntJumpIfFalse(_, _, _)
                | Op::SlotIncLtIntJumpBack(_, _, _)
                | Op::AccumSumLoop(_, _, _)
                | Op::PushFrame
                | Op::PopFrame
        )
    }

    pub(crate) fn is_block_eligible(chunk: &Chunk) -> bool {
        let ops = &chunk.ops;
        !ops.is_empty() && ops.iter().all(is_block_eligible_op)
    }

    /// Find the largest contiguous JIT-eligible region in a chunk.
    /// A region is closed (all jump targets within the region must also be in
    /// the region — bytecode-level jumps to outside the region disqualify it).
    ///
    /// Returns `(start, end)` op indices (end exclusive), or None if no
    /// eligible region of useful size exists. Useful size = at least 8 ops.
    pub(crate) fn find_jit_region(ops: &[Op]) -> Option<(usize, usize)> {
        let mut best: Option<(usize, usize)> = None;
        let mut start: Option<usize> = None;

        for (ip, op) in ops.iter().enumerate() {
            if is_block_eligible_op(op) {
                if start.is_none() {
                    start = Some(ip);
                }
            } else if let Some(s) = start.take() {
                let len = ip - s;
                if len >= 4 && best.map_or(true, |(bs, be)| len > be - bs) {
                    best = Some((s, ip));
                }
            }
        }
        // Tail region
        if let Some(s) = start {
            let len = ops.len() - s;
            if len >= 4 && best.map_or(true, |(bs, be)| len > be - bs) {
                best = Some((s, ops.len()));
            }
        }

        // Verify all jumps within the region target inside the region.
        // Rebase jumps locally if so; otherwise reject.
        let (s, e) = best?;
        for op in &ops[s..e] {
            match op {
                Op::Jump(t)
                | Op::JumpIfTrue(t)
                | Op::JumpIfFalse(t)
                | Op::SlotLtIntJumpIfFalse(_, _, t)
                | Op::SlotIncLtIntJumpBack(_, _, t) => {
                    if *t < s || *t >= e {
                        return None;
                    }
                }
                _ => {}
            }
        }
        Some((s, e))
    }

    /// Extract a JIT region as a standalone sub-chunk with rebased jump targets.
    /// The returned chunk has its op_hash recomputed.
    pub(crate) fn extract_region(chunk: &Chunk, start: usize, end: usize) -> Chunk {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut sub = Chunk {
            ops: Vec::with_capacity(end - start),
            constants: chunk.constants.clone(),
            names: chunk.names.clone(),
            lines: chunk.lines[start..end].to_vec(),
            sub_entries: Vec::new(),
            block_ranges: Vec::new(),
            source: chunk.source.clone(),
            op_hash: 0,
        };
        for op in &chunk.ops[start..end] {
            // Rebase jump targets to be local to the sub-chunk.
            let new_op = match op {
                Op::Jump(t) => Op::Jump(t - start),
                Op::JumpIfTrue(t) => Op::JumpIfTrue(t - start),
                Op::JumpIfFalse(t) => Op::JumpIfFalse(t - start),
                Op::SlotLtIntJumpIfFalse(s, l, t) => Op::SlotLtIntJumpIfFalse(*s, *l, t - start),
                Op::SlotIncLtIntJumpBack(s, l, t) => Op::SlotIncLtIntJumpBack(*s, *l, t - start),
                other => other.clone(),
            };
            sub.ops.push(new_op);
        }
        let mut h = DefaultHasher::new();
        sub.ops.hash(&mut h);
        sub.constants.hash(&mut h);
        sub.op_hash = h.finish();
        sub
    }

    /// Collect all slot indices referenced by the chunk for promotion.
    fn collect_slots(ops: &[Op]) -> Vec<u16> {
        let mut slots = std::collections::BTreeSet::new();
        for op in ops {
            match op {
                Op::GetSlot(s)
                | Op::SetSlot(s)
                | Op::PreIncSlot(s)
                | Op::PreIncSlotVoid(s)
                | Op::SlotLtIntJumpIfFalse(s, _, _)
                | Op::SlotIncLtIntJumpBack(s, _, _) => {
                    slots.insert(*s);
                }
                Op::AddAssignSlotVoid(a, b) => {
                    slots.insert(*a);
                    slots.insert(*b);
                }
                Op::AccumSumLoop(s, i, _) => {
                    slots.insert(*s);
                    slots.insert(*i);
                }
                _ => {}
            }
        }
        slots.into_iter().collect()
    }

    fn cond_to_i1(bcx: &mut FunctionBuilder, v: Value, ty: JitTy) -> cranelift_codegen::ir::Value {
        match ty {
            JitTy::Int => bcx.ins().icmp_imm(IntCC::NotEqual, v, 0),
            JitTy::Float => {
                let z = bcx.ins().f64const(Ieee64::with_bits(0.0f64.to_bits()));
                bcx.ins().fcmp(FloatCC::OrderedNotEqual, v, z)
            }
        }
    }

    pub(crate) fn compile_block(chunk: &Chunk) -> Option<CompiledBlock> {
        let ops = &chunk.ops;
        if !is_block_eligible(chunk) {
            return None;
        }

        let leaders = find_leaders(ops);
        let leader_vec: Vec<usize> = leaders.iter().copied().collect();
        let mut module = new_jit_module()?;

        // Declare external helpers
        let needs_pow = ops.iter().any(|o| matches!(o, Op::Pow));
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
        let needs_fmod = ops.iter().any(|o| matches!(o, Op::Mod));
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
        let needs_lognot = ops.iter().any(|o| matches!(o, Op::LogNot));
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
        sig.params.push(AbiParam::new(ptr_ty)); // *mut i64 slots
        sig.returns.push(AbiParam::new(types::I64)); // result

        let fid = module
            .declare_function("block_jit", Linkage::Local, &sig)
            .ok()?;
        let mut ctx = module.make_context();
        ctx.func.signature = sig;
        ctx.func.name = UserFuncName::user(0, fid.as_u32());

        let mut fctx = FunctionBuilderContext::new();
        {
            let mut bcx = FunctionBuilder::new(&mut ctx.func, &mut fctx);

            // Create Cranelift blocks for each bytecode leader
            let mut block_map: HashMap<usize, cranelift_codegen::ir::Block> = HashMap::new();
            for &leader_ip in &leader_vec {
                block_map.insert(leader_ip, bcx.create_block());
            }

            // Entry block setup
            let entry = block_map[&0];
            bcx.append_block_params_for_function_params(entry);
            bcx.switch_to_block(entry);
            let slot_base = bcx.block_params(entry)[0];

            // ── Slot promotion: declare a Cranelift Variable per used slot, ──
            // ── load each from the slot pointer at entry. After this, all   ──
            // ── slot ops use Variables (register-allocated by Cranelift).   ──
            let used_slots = collect_slots(ops);
            let mut slot_vars: HashMap<u16, Variable> = HashMap::new();
            for &slot in &used_slots {
                let var = bcx.declare_var(types::I64);
                let val = bcx.ins().load(
                    types::I64,
                    MemFlags::trusted(),
                    slot_base,
                    (slot as i32) * 8,
                );
                bcx.def_var(var, val);
                slot_vars.insert(slot, var);
            }

            let pow_i64_ref = pow_i64_id.map(|pid| module.declare_func_in_func(pid, bcx.func));
            let pow_f64_ref = pow_f64_id.map(|pid| module.declare_func_in_func(pid, bcx.func));
            let fmod_f64_ref = fmod_f64_id.map(|pid| module.declare_func_in_func(pid, bcx.func));
            let lognot_ref = lognot_id.map(|lid| module.declare_func_in_func(lid, bcx.func));

            // Process each basic block
            let mut block_terminated = false;

            for (block_idx, &leader_ip) in leader_vec.iter().enumerate() {
                let block_end = if block_idx + 1 < leader_vec.len() {
                    leader_vec[block_idx + 1]
                } else {
                    ops.len()
                };

                // Switch to this block (unless it's the entry which we already started)
                if leader_ip > 0 {
                    if !block_terminated {
                        // Previous block fell through — emit jump
                        bcx.ins().jump(block_map[&leader_ip], &[]);
                    }
                    bcx.switch_to_block(block_map[&leader_ip]);
                }

                let mut stack: Vec<(cranelift_codegen::ir::Value, JitTy)> = Vec::new();
                block_terminated = false;

                for ip in leader_ip..block_end {
                    let op = &ops[ip];
                    match op {
                        Op::PushFrame | Op::PopFrame | Op::Nop => {}

                        Op::Jump(target) => {
                            let target_block = *block_map.get(target)?;
                            bcx.ins().jump(target_block, &[]);
                            block_terminated = true;
                        }

                        Op::JumpIfTrue(target) => {
                            let (cond, ty) = stack.pop()?;
                            let pred = cond_to_i1(&mut bcx, cond, ty);
                            let target_block = *block_map.get(target)?;
                            let fall = if ip + 1 < ops.len() {
                                *block_map.get(&(ip + 1))?
                            } else {
                                return None;
                            };
                            bcx.ins().brif(pred, target_block, &[], fall, &[]);
                            block_terminated = true;
                        }

                        Op::JumpIfFalse(target) => {
                            let (cond, ty) = stack.pop()?;
                            let pred = cond_to_i1(&mut bcx, cond, ty);
                            let target_block = *block_map.get(target)?;
                            let fall = if ip + 1 < ops.len() {
                                *block_map.get(&(ip + 1))?
                            } else {
                                return None;
                            };
                            bcx.ins().brif(pred, fall, &[], target_block, &[]);
                            block_terminated = true;
                        }

                        // ── Slot data ops: use Variables (register-allocated) ──
                        Op::GetSlot(slot) => {
                            let var = *slot_vars.get(slot)?;
                            let v = bcx.use_var(var);
                            stack.push((v, JitTy::Int));
                        }
                        Op::SetSlot(slot) => {
                            let var = *slot_vars.get(slot)?;
                            let (v, ty) = stack.pop()?;
                            let v_i = scalar_store_i64(&mut bcx, v, ty);
                            bcx.def_var(var, v_i);
                        }
                        Op::PreIncSlot(slot) => {
                            let var = *slot_vars.get(slot)?;
                            let old = bcx.use_var(var);
                            let one = bcx.ins().iconst(types::I64, 1);
                            let new = bcx.ins().iadd(old, one);
                            bcx.def_var(var, new);
                            stack.push((new, JitTy::Int));
                        }
                        Op::PreIncSlotVoid(slot) => {
                            let var = *slot_vars.get(slot)?;
                            let old = bcx.use_var(var);
                            let one = bcx.ins().iconst(types::I64, 1);
                            let new = bcx.ins().iadd(old, one);
                            bcx.def_var(var, new);
                        }
                        Op::AddAssignSlotVoid(a_slot, b_slot) => {
                            let a_var = *slot_vars.get(a_slot)?;
                            let b_var = *slot_vars.get(b_slot)?;
                            let va = bcx.use_var(a_var);
                            let vb = bcx.use_var(b_var);
                            let sum = bcx.ins().iadd(va, vb);
                            bcx.def_var(a_var, sum);
                        }

                        Op::SlotLtIntJumpIfFalse(slot, limit, target) => {
                            let var = *slot_vars.get(slot)?;
                            let val = bcx.use_var(var);
                            let limit_v = bcx.ins().iconst(types::I64, *limit as i64);
                            let is_lt = bcx.ins().icmp(IntCC::SignedLessThan, val, limit_v);
                            let target_block = *block_map.get(target)?;
                            let fall = if ip + 1 < ops.len() {
                                *block_map.get(&(ip + 1))?
                            } else {
                                return None;
                            };
                            // if >= limit, jump to target; otherwise fall through
                            bcx.ins().brif(is_lt, fall, &[], target_block, &[]);
                            block_terminated = true;
                        }

                        Op::SlotIncLtIntJumpBack(slot, limit, target) => {
                            let var = *slot_vars.get(slot)?;
                            let old = bcx.use_var(var);
                            let one = bcx.ins().iconst(types::I64, 1);
                            let new = bcx.ins().iadd(old, one);
                            bcx.def_var(var, new);
                            let limit_v = bcx.ins().iconst(types::I64, *limit as i64);
                            let is_lt = bcx.ins().icmp(IntCC::SignedLessThan, new, limit_v);
                            let target_block = *block_map.get(target)?;
                            let fall = if ip + 1 < ops.len() {
                                *block_map.get(&(ip + 1))?
                            } else {
                                return None;
                            };
                            bcx.ins().brif(is_lt, target_block, &[], fall, &[]);
                            block_terminated = true;
                        }

                        Op::AccumSumLoop(sum_slot, i_slot, limit) => {
                            let sum_var = *slot_vars.get(sum_slot)?;
                            let i_var = *slot_vars.get(i_slot)?;
                            let sum_init = bcx.use_var(sum_var);
                            let i_init = bcx.use_var(i_var);
                            let limit_v = bcx.ins().iconst(types::I64, *limit as i64);

                            let loop_hdr = bcx.create_block();
                            let loop_body = bcx.create_block();
                            let loop_exit = bcx.create_block();

                            bcx.ins().jump(
                                loop_hdr,
                                &[BlockArg::Value(sum_init), BlockArg::Value(i_init)],
                            );

                            // Loop header: check i < limit
                            bcx.switch_to_block(loop_hdr);
                            bcx.append_block_param(loop_hdr, types::I64); // sum
                            bcx.append_block_param(loop_hdr, types::I64); // i
                            let sum_p = bcx.block_params(loop_hdr)[0];
                            let i_p = bcx.block_params(loop_hdr)[1];
                            let cond = bcx.ins().icmp(IntCC::SignedLessThan, i_p, limit_v);
                            bcx.ins().brif(
                                cond,
                                loop_body,
                                &[],
                                loop_exit,
                                &[BlockArg::Value(sum_p), BlockArg::Value(i_p)],
                            );

                            // Loop body: sum += i; i++
                            bcx.switch_to_block(loop_body);
                            let new_sum = bcx.ins().iadd(sum_p, i_p);
                            let one = bcx.ins().iconst(types::I64, 1);
                            let new_i = bcx.ins().iadd(i_p, one);
                            bcx.ins().jump(
                                loop_hdr,
                                &[BlockArg::Value(new_sum), BlockArg::Value(new_i)],
                            );

                            // Loop exit: write back to slot variables
                            bcx.switch_to_block(loop_exit);
                            bcx.append_block_param(loop_exit, types::I64);
                            bcx.append_block_param(loop_exit, types::I64);
                            let final_sum = bcx.block_params(loop_exit)[0];
                            let final_i = bcx.block_params(loop_exit)[1];
                            bcx.def_var(sum_var, final_sum);
                            bcx.def_var(i_var, final_i);
                        }

                        // Data ops — delegate to emit_data_op
                        _ => {
                            emit_data_op(
                                &mut bcx,
                                op,
                                &mut stack,
                                Some(slot_base),
                                pow_i64_ref,
                                pow_f64_ref,
                                fmod_f64_ref,
                                lognot_ref,
                                &chunk.constants,
                            )?;
                        }
                    }
                }

                // End of this basic block — if not terminated, handle fallthrough or return
                if !block_terminated {
                    if block_end == ops.len() {
                        // Final block: spill promoted slot variables to memory, then return
                        let ret_val = if let Some((v, ty)) = stack.pop() {
                            scalar_store_i64(&mut bcx, v, ty)
                        } else {
                            bcx.ins().iconst(types::I64, 0)
                        };
                        // Write all slot Variables back to the slot pointer
                        // (caller observes the final state of the slot array)
                        for (&slot, &var) in &slot_vars {
                            let val = bcx.use_var(var);
                            bcx.ins()
                                .store(MemFlags::trusted(), val, slot_base, (slot as i32) * 8);
                        }
                        bcx.ins().return_(&[ret_val]);
                        block_terminated = true;
                    }
                    // Non-final unterminated blocks get a fallthrough jump
                    // at the top of the next iteration
                }
            }

            bcx.seal_all_blocks();
            bcx.finalize();
        }

        module.define_function(fid, &mut ctx).ok()?;
        module.clear_context(&mut ctx);
        module.finalize_definitions().ok()?;
        let ptr = module.get_finalized_function(fid);
        // Currently always SlotsI: signature is fn(*mut i64) -> i64.
        // Future specialization: detect no-slots chunks → NoSlotsI;
        // detect float-returning chunks → SlotsF/NoSlotsF.
        let run = BlockRun::SlotsI(unsafe { std::mem::transmute::<*const u8, BlockFnSlotsI>(ptr) });
        Some(CompiledBlock { module, run })
    }

    // ── Block JIT cache (per-thread, lock-free) ──
    //
    // Each thread has its own cache — JITModule is not Send anyway, and
    // VMs are single-threaded per instance. No mutex overhead per call.
    // Hot-counts are also tracked here for tiered compilation.

    use std::cell::RefCell;

    struct BlockCacheEntry {
        /// Number of times we've been asked to run this chunk.
        hot_count: u32,
        /// Compiled native code (set after threshold).
        compiled: Option<Box<CompiledBlock>>,
    }

    thread_local! {
        static BLOCK_CACHE_TLS: RefCell<HashMap<u64, BlockCacheEntry>> =
            RefCell::new(HashMap::new());
    }

    /// Tiered compilation threshold — JIT after N interpreter runs.
    /// Below this, `try_run_block` returns None so the caller falls back
    /// to the interpreter. This avoids paying compile cost for one-shot chunks.
    const HOT_THRESHOLD: u32 = 10;

    /// Try to JIT-compile and run a chunk via the block JIT.
    /// Returns `Some(result_i64)` on success, `None` if ineligible OR not yet hot.
    pub(crate) fn try_run_block(chunk: &Chunk, slots: &mut [i64]) -> Option<i64> {
        try_run_block_inner(chunk, slots, HOT_THRESHOLD)
    }

    /// Like `try_run_block` but compiles immediately (no warmup). For tests
    /// and synthetic benchmarks where you want to skip the tiered policy.
    pub(crate) fn try_run_block_eager(chunk: &Chunk, slots: &mut [i64]) -> Option<i64> {
        try_run_block_inner(chunk, slots, 0)
    }

    fn try_run_block_inner(chunk: &Chunk, slots: &mut [i64], threshold: u32) -> Option<i64> {
        let key = chunk.op_hash;

        BLOCK_CACHE_TLS.with(|cache_cell| {
            let mut cache = cache_cell.borrow_mut();
            let entry = cache.entry(key).or_insert(BlockCacheEntry {
                hot_count: 0,
                compiled: None,
            });

            if let Some(ref compiled) = entry.compiled {
                return Some(compiled.invoke(slots));
            }

            entry.hot_count = entry.hot_count.saturating_add(1);
            if entry.hot_count <= threshold {
                return None; // not hot yet — caller falls back to interpreter
            }

            // Compile on threshold cross
            let compiled = compile_block(chunk)?;
            let result = compiled.invoke(slots);
            entry.compiled = Some(Box::new(compiled));
            Some(result)
        })
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

    /// Try to compile and run a chunk via the block JIT (handles loops/branches).
    /// Slots are read and written in-place. Returns `Some(result)` on success.
    ///
    /// Tiered: returns `None` for the first N invocations of a given chunk so the
    /// caller falls back to the interpreter. After the chunk crosses the
    /// hot-threshold, compiles and caches the native code.
    #[cfg(feature = "jit")]
    pub fn try_run_block(&self, chunk: &crate::Chunk, slots: &mut [i64]) -> Option<i64> {
        cranelift_jit_impl::try_run_block(chunk, slots)
    }

    /// Like `try_run_block` but skips the tiered policy — compiles immediately
    /// on first call. Use for tests, microbenchmarks, or AOT-style usage.
    #[cfg(feature = "jit")]
    pub fn try_run_block_eager(&self, chunk: &crate::Chunk, slots: &mut [i64]) -> Option<i64> {
        cranelift_jit_impl::try_run_block_eager(chunk, slots)
    }

    /// Check if a chunk is eligible for block JIT.
    #[cfg(feature = "jit")]
    pub fn is_block_eligible(&self, chunk: &crate::Chunk) -> bool {
        cranelift_jit_impl::is_block_eligible(chunk)
    }

    /// Find the largest contiguous JIT-eligible region in a chunk.
    /// Returns `(start, end)` op indices, or None if no useful region exists.
    /// Useful for partial JIT compilation of chunks that aren't entirely eligible.
    #[cfg(feature = "jit")]
    pub fn find_jit_region(&self, chunk: &crate::Chunk) -> Option<(usize, usize)> {
        cranelift_jit_impl::find_jit_region(&chunk.ops)
    }

    /// Extract a JIT region as a standalone sub-chunk with rebased jump targets.
    /// The sub-chunk can then be passed to `try_run_block_eager` to JIT-compile
    /// just that region. Use with `find_jit_region` to find an eligible range.
    #[cfg(feature = "jit")]
    pub fn extract_region(&self, chunk: &crate::Chunk, start: usize, end: usize) -> crate::Chunk {
        cranelift_jit_impl::extract_region(chunk, start, end)
    }

    /// Stub when JIT feature is disabled.
    #[cfg(not(feature = "jit"))]
    pub fn find_jit_region(&self, _chunk: &crate::Chunk) -> Option<(usize, usize)> {
        None
    }

    /// Stub when JIT feature is disabled.
    #[cfg(not(feature = "jit"))]
    pub fn extract_region(&self, chunk: &crate::Chunk, _start: usize, _end: usize) -> crate::Chunk {
        chunk.clone()
    }

    /// Stub when JIT feature is disabled.
    #[cfg(not(feature = "jit"))]
    pub fn try_run_block(&self, _chunk: &crate::Chunk, _slots: &mut [i64]) -> Option<i64> {
        None
    }

    /// Stub when JIT feature is disabled.
    #[cfg(not(feature = "jit"))]
    pub fn try_run_block_eager(&self, _chunk: &crate::Chunk, _slots: &mut [i64]) -> Option<i64> {
        None
    }

    /// Stub when JIT feature is disabled.
    #[cfg(not(feature = "jit"))]
    pub fn is_block_eligible(&self, _chunk: &crate::Chunk) -> bool {
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
