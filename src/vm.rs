//! The fusevm execution engine — stack-based bytecode dispatch loop.
//!
//! This is the hot path. Every cycle counts. The dispatch loop uses
//! a flat `match` on `Op` variants — Rust compiles this to a jump table.
//!
//! Frontends register extension handlers via `ExtensionHandler` for
//! language-specific opcodes (`Op::Extended`, `Op::ExtendedWide`).
//!
//! ## Optimizations
//!
//! - **Type-specialized integer fast paths**: Add, Sub, Mul, Mod, comparisons
//!   check for `Int×Int` first and skip `to_float()` coercion entirely.
//! - **Zero-clone dispatch**: ops are borrowed from the chunk, not cloned per cycle.
//!   `LoadConst` copies scalars (Int/Float/Bool) without touching Arc refcounts.
//! - **In-place container mutation**: array/hash ops (Push, Pop, Shift, Set,
//!   HashSet, HashDelete) mutate globals directly — no clone-modify-writeback.
//! - **`Cow<str>` string coercion**: `as_str_cow()` borrows `Str` variants without
//!   allocation. Used in string comparisons, Concat, Print, hash key lookup.
//! - **Inline builtin cache**: `CallBuiltin` dispatches through a pre-registered
//!   function pointer table — no name lookup at runtime.
//! - **Fused superinstructions**: hot loop patterns run as single ops
//!   (AccumSumLoop, SlotIncLtIntJumpBack, etc.)
//! - **Pre-allocated collections**: Range, MakeHash, HashKeys/Values use exact
//!   or estimated capacity. ConcatConstLoop pre-sizes the string buffer.

use crate::chunk::Chunk;
use crate::op::Op;
use crate::value::Value;

/// Call frame on the frame stack.
#[derive(Debug, Clone)]
pub struct Frame {
    /// Return address (ip to resume after call)
    pub return_ip: usize,
    /// Base pointer into the value stack (locals start here)
    pub stack_base: usize,
    /// Local variable slots (indexed by `GetSlot`/`SetSlot`)
    pub slots: Vec<Value>,
}

/// Extension handler for language-specific opcodes.
/// Frontends register this at VM init.
pub type ExtensionHandler = Box<dyn FnMut(&mut VM, u16, u8) + Send>;
/// Wide extension handler (usize payload).
pub type ExtensionWideHandler = Box<dyn FnMut(&mut VM, u16, usize) + Send>;
/// Builtin function handler: (vm, argc) → Value
pub type BuiltinHandler = fn(&mut VM, u8) -> Value;

/// The virtual machine.
pub struct VM {
    /// Value stack
    pub stack: Vec<Value>,
    /// Call frame stack
    pub frames: Vec<Frame>,
    /// Global variables (name pool index → value)
    pub globals: Vec<Value>,
    /// Instruction pointer
    pub ip: usize,
    /// Current chunk being executed
    pub chunk: Chunk,
    /// Last exit status ($?)
    pub last_status: i32,
    /// Extension handler for `Op::Extended`
    ext_handler: Option<ExtensionHandler>,
    /// Extension handler for `Op::ExtendedWide`
    ext_wide_handler: Option<ExtensionWideHandler>,
    /// Inline builtin cache: builtin_id → function pointer (no lookup at dispatch)
    builtin_table: Vec<Option<BuiltinHandler>>,
    /// Halted flag
    halted: bool,
}

/// Result of VM execution
#[derive(Debug)]
pub enum VMResult {
    /// Normal completion with a value
    Ok(Value),
    /// Halted (no more instructions)
    Halted,
    /// Runtime error
    Error(String),
}

impl VM {
    pub fn new(chunk: Chunk) -> Self {
        let num_names = chunk.names.len();
        let mut frames = Vec::with_capacity(32);
        frames.push(Frame {
            return_ip: 0,
            stack_base: 0,
            slots: Vec::with_capacity(16),
        });
        Self {
            stack: Vec::with_capacity(256),
            frames,
            globals: vec![Value::Undef; num_names],
            ip: 0,
            chunk,
            last_status: 0,
            ext_handler: None,
            ext_wide_handler: None,
            builtin_table: Vec::new(),
            halted: false,
        }
    }

    /// Register a handler for `Op::Extended(id, arg)` opcodes.
    pub fn set_extension_handler(&mut self, handler: ExtensionHandler) {
        self.ext_handler = Some(handler);
    }

    /// Register a handler for `Op::ExtendedWide(id, payload)` opcodes.
    pub fn set_extension_wide_handler(&mut self, handler: ExtensionWideHandler) {
        self.ext_wide_handler = Some(handler);
    }

    /// Register a builtin function by ID. `CallBuiltin(id, argc)` dispatches
    /// directly through the function pointer — no name lookup at runtime.
    pub fn register_builtin(&mut self, id: u16, handler: BuiltinHandler) {
        let idx = id as usize;
        if idx >= self.builtin_table.len() {
            self.builtin_table.resize(idx + 1, None);
        }
        self.builtin_table[idx] = Some(handler);
    }

    // ── Stack operations ──

    #[inline(always)]
    pub fn push(&mut self, val: Value) {
        self.stack.push(val);
    }

    #[inline(always)]
    pub fn pop(&mut self) -> Value {
        self.stack.pop().unwrap_or(Value::Undef)
    }

    #[inline(always)]
    pub fn peek(&self) -> &Value {
        self.stack.last().unwrap_or(&Value::Undef)
    }

    // ── Type-specialized helpers (avoid to_float coercion on hot paths) ──

    /// Pop two values; if both Int, apply int_op. Otherwise promote to float.
    #[inline(always)]
    fn arith_int_fast(&mut self, int_op: fn(i64, i64) -> i64, float_op: fn(f64, f64) -> f64) {
        let len = self.stack.len();
        if len >= 2 {
            // Borrow both slots without popping (avoid branch + unwrap_or)
            let b = &self.stack[len - 1];
            let a = &self.stack[len - 2];
            let result = match (a, b) {
                (Value::Int(x), Value::Int(y)) => Value::Int(int_op(*x, *y)),
                _ => Value::Float(float_op(a.to_float(), b.to_float())),
            };
            self.stack.truncate(len - 2);
            self.stack.push(result);
        }
    }

    /// Pop two values; compare as int if both Int, otherwise float.
    /// Push Bool(true/false).
    #[inline(always)]
    fn cmp_int_fast(&mut self, int_cmp: fn(i64, i64) -> bool, float_cmp: fn(f64, f64) -> bool) {
        let len = self.stack.len();
        if len >= 2 {
            let b = &self.stack[len - 1];
            let a = &self.stack[len - 2];
            let result = match (a, b) {
                (Value::Int(x), Value::Int(y)) => int_cmp(*x, *y),
                _ => float_cmp(a.to_float(), b.to_float()),
            };
            self.stack.truncate(len - 2);
            self.stack.push(Value::Bool(result));
        }
    }

    // ── Main dispatch loop ──

    /// Execute the loaded chunk until completion or error.
    pub fn run(&mut self) -> VMResult {
        let ops = &self.chunk.ops as *const Vec<Op>;
        // SAFETY: self.chunk.ops is not mutated during execution.
        // We take a pointer to avoid borrow checker issues with &self.chunk.ops
        // while mutating self.stack/frames/globals.
        let ops = unsafe { &*ops };

        while self.ip < ops.len() && !self.halted {
            // Zero-clone: borrow the op instead of cloning
            let ip = self.ip;
            self.ip += 1;

            match &ops[ip] {
                Op::Nop => {}

                // ── Constants ──
                Op::LoadInt(n) => self.push(Value::Int(*n)),
                Op::LoadFloat(f) => self.push(Value::Float(*f)),
                Op::LoadConst(idx) => {
                    let val = match self.chunk.constants.get(*idx as usize) {
                        Some(Value::Int(n)) => Value::Int(*n),
                        Some(Value::Float(f)) => Value::Float(*f),
                        Some(Value::Bool(b)) => Value::Bool(*b),
                        Some(other) => other.clone(),
                        None => Value::Undef,
                    };
                    self.push(val);
                }
                Op::LoadTrue => self.push(Value::Bool(true)),
                Op::LoadFalse => self.push(Value::Bool(false)),
                Op::LoadUndef => self.push(Value::Undef),

                // ── Stack ──
                Op::Pop => {
                    self.pop();
                }
                Op::Dup => {
                    let val = self.peek().clone();
                    self.push(val);
                }
                Op::Dup2 => {
                    let len = self.stack.len();
                    if len >= 2 {
                        let a = self.stack[len - 2].clone();
                        let b = self.stack[len - 1].clone();
                        self.push(a);
                        self.push(b);
                    }
                }
                Op::Swap => {
                    let len = self.stack.len();
                    if len >= 2 {
                        self.stack.swap(len - 1, len - 2);
                    }
                }
                Op::Rot => {
                    let len = self.stack.len();
                    if len >= 3 {
                        // [a, b, c] → [b, c, a] via two swaps instead of O(n) remove
                        self.stack.swap(len - 3, len - 2);
                        self.stack.swap(len - 2, len - 1);
                    }
                }

                // ── Variables ──
                Op::GetVar(idx) => {
                    let val = self.get_var(*idx);
                    self.push(val);
                }
                Op::SetVar(idx) => {
                    let val = self.pop();
                    self.set_var(*idx, val);
                }
                Op::DeclareVar(idx) => {
                    let val = self.pop();
                    self.set_var(*idx, val);
                }
                Op::GetSlot(slot) => {
                    let val = self.get_slot(*slot);
                    self.push(val);
                }
                Op::SetSlot(slot) => {
                    let val = self.pop();
                    self.set_slot(*slot, val);
                }
                Op::SlotArrayGet(slot) => {
                    let index = self.pop().to_int() as usize;
                    let val = self.get_slot(*slot);
                    let result = if let Value::Array(ref arr) = val {
                        arr.get(index).cloned().unwrap_or(Value::Undef)
                    } else {
                        Value::Undef
                    };
                    self.push(result);
                }
                Op::SlotArraySet(slot) => {
                    let index = self.pop().to_int() as usize;
                    let val = self.pop();
                    let arr_val = self.get_slot(*slot);
                    if let Value::Array(mut arr) = arr_val {
                        if index >= arr.len() {
                            arr.resize(index + 1, Value::Undef);
                        }
                        arr[index] = val;
                        self.set_slot(*slot, Value::Array(arr));
                    }
                }

                // ── Arithmetic (type-specialized: Int×Int avoids to_float) ──
                Op::Add => self.arith_int_fast(i64::wrapping_add, |a, b| a + b),
                Op::Sub => self.arith_int_fast(i64::wrapping_sub, |a, b| a - b),
                Op::Mul => self.arith_int_fast(i64::wrapping_mul, |a, b| a * b),
                Op::Div => {
                    let b = self.pop();
                    let a = self.pop();
                    let divisor = b.to_float();
                    self.push(if divisor == 0.0 {
                        Value::Undef
                    } else {
                        Value::Float(a.to_float() / divisor)
                    });
                }
                Op::Mod => self.arith_int_fast(|x, y| if y != 0 { x % y } else { 0 }, |a, b| a % b),
                Op::Pow => {
                    let b = self.pop();
                    let a = self.pop();
                    self.push(Value::Float(a.to_float().powf(b.to_float())));
                }
                Op::Negate => {
                    let val = self.pop();
                    self.push(match val {
                        Value::Int(n) => Value::Int(n.wrapping_neg()),
                        _ => Value::Float(-val.to_float()),
                    });
                }
                Op::Inc => {
                    let val = self.pop();
                    self.push(match val {
                        Value::Int(n) => Value::Int(n.wrapping_add(1)),
                        _ => Value::Int(val.to_int().wrapping_add(1)),
                    });
                }
                Op::Dec => {
                    let val = self.pop();
                    self.push(match val {
                        Value::Int(n) => Value::Int(n.wrapping_sub(1)),
                        _ => Value::Int(val.to_int().wrapping_sub(1)),
                    });
                }

                // ── String ──
                Op::Concat => {
                    let b = self.pop();
                    let a = self.pop();
                    let a_s = a.as_str_cow();
                    let b_s = b.as_str_cow();
                    let mut s = String::with_capacity(a_s.len() + b_s.len());
                    s.push_str(&a_s);
                    s.push_str(&b_s);
                    self.push(Value::str(s));
                }
                Op::StringRepeat => {
                    let count = self.pop().to_int();
                    let s = self.pop().to_str();
                    self.push(Value::str(s.repeat(count.max(0) as usize)));
                }
                Op::StringLen => {
                    let s = self.pop();
                    self.push(Value::Int(s.len() as i64));
                }

                // ── Comparison (type-specialized: Int×Int avoids to_float) ──
                Op::NumEq => self.cmp_int_fast(|x, y| x == y, |a, b| a == b),
                Op::NumNe => self.cmp_int_fast(|x, y| x != y, |a, b| a != b),
                Op::NumLt => self.cmp_int_fast(|x, y| x < y, |a, b| a < b),
                Op::NumGt => self.cmp_int_fast(|x, y| x > y, |a, b| a > b),
                Op::NumLe => self.cmp_int_fast(|x, y| x <= y, |a, b| a <= b),
                Op::NumGe => self.cmp_int_fast(|x, y| x >= y, |a, b| a >= b),
                Op::Spaceship => {
                    let len = self.stack.len();
                    if len >= 2 {
                        let b = &self.stack[len - 1];
                        let a = &self.stack[len - 2];
                        let result = match (a, b) {
                            (Value::Int(x), Value::Int(y)) => x.cmp(y) as i64,
                            _ => {
                                let af = a.to_float();
                                let bf = b.to_float();
                                if af < bf {
                                    -1
                                } else if af > bf {
                                    1
                                } else {
                                    0
                                }
                            }
                        };
                        self.stack.truncate(len - 2);
                        self.stack.push(Value::Int(result));
                    }
                }

                // ── Comparison (string — borrow via Cow to avoid allocation) ──
                Op::StrEq => {
                    let b = self.pop();
                    let a = self.pop();
                    self.push(Value::Bool(a.as_str_cow() == b.as_str_cow()));
                }
                Op::StrNe => {
                    let b = self.pop();
                    let a = self.pop();
                    self.push(Value::Bool(a.as_str_cow() != b.as_str_cow()));
                }
                Op::StrLt => {
                    let b = self.pop();
                    let a = self.pop();
                    self.push(Value::Bool(a.as_str_cow() < b.as_str_cow()));
                }
                Op::StrGt => {
                    let b = self.pop();
                    let a = self.pop();
                    self.push(Value::Bool(a.as_str_cow() > b.as_str_cow()));
                }
                Op::StrLe => {
                    let b = self.pop();
                    let a = self.pop();
                    self.push(Value::Bool(a.as_str_cow() <= b.as_str_cow()));
                }
                Op::StrGe => {
                    let b = self.pop();
                    let a = self.pop();
                    self.push(Value::Bool(a.as_str_cow() >= b.as_str_cow()));
                }
                Op::StrCmp => {
                    let b = self.pop();
                    let a = self.pop();
                    self.push(Value::Int(match a.as_str_cow().cmp(&b.as_str_cow()) {
                        std::cmp::Ordering::Less => -1,
                        std::cmp::Ordering::Equal => 0,
                        std::cmp::Ordering::Greater => 1,
                    }));
                }

                // ── Logical / Bitwise ──
                Op::LogNot => {
                    let val = self.pop();
                    self.push(Value::Bool(!val.is_truthy()));
                }
                Op::LogAnd => {
                    let b = self.pop();
                    let a = self.pop();
                    self.push(Value::Bool(a.is_truthy() && b.is_truthy()));
                }
                Op::LogOr => {
                    let b = self.pop();
                    let a = self.pop();
                    self.push(Value::Bool(a.is_truthy() || b.is_truthy()));
                }
                Op::BitAnd => {
                    let b = self.pop();
                    let a = self.pop();
                    self.push(Value::Int(a.to_int() & b.to_int()));
                }
                Op::BitOr => {
                    let b = self.pop();
                    let a = self.pop();
                    self.push(Value::Int(a.to_int() | b.to_int()));
                }
                Op::BitXor => {
                    let b = self.pop();
                    let a = self.pop();
                    self.push(Value::Int(a.to_int() ^ b.to_int()));
                }
                Op::BitNot => {
                    let val = self.pop();
                    self.push(Value::Int(!val.to_int()));
                }
                Op::Shl => {
                    let b = self.pop();
                    let a = self.pop();
                    self.push(Value::Int(a.to_int() << (b.to_int() as u32 & 63)));
                }
                Op::Shr => {
                    let b = self.pop();
                    let a = self.pop();
                    self.push(Value::Int(a.to_int() >> (b.to_int() as u32 & 63)));
                }

                // ── Control flow ──
                Op::Jump(target) => self.ip = *target,
                Op::JumpIfTrue(target) => {
                    if self.pop().is_truthy() {
                        self.ip = *target;
                    }
                }
                Op::JumpIfFalse(target) => {
                    if !self.pop().is_truthy() {
                        self.ip = *target;
                    }
                }
                Op::JumpIfTrueKeep(target) => {
                    if self.peek().is_truthy() {
                        self.ip = *target;
                    }
                }
                Op::JumpIfFalseKeep(target) => {
                    if !self.peek().is_truthy() {
                        self.ip = *target;
                    }
                }

                // ── Functions ──
                Op::Call(name_idx, argc) => {
                    if let Some(entry_ip) = self.chunk.find_sub(*name_idx) {
                        self.frames.push(Frame {
                            return_ip: self.ip,
                            stack_base: self.stack.len() - *argc as usize,
                            slots: Vec::new(),
                        });
                        self.ip = entry_ip;
                    } else {
                        return VMResult::Error(format!(
                            "undefined function: {}",
                            self.chunk
                                .names
                                .get(*name_idx as usize)
                                .map(|s| s.as_str())
                                .unwrap_or("?")
                        ));
                    }
                }
                Op::Return => {
                    if let Some(frame) = self.frames.pop() {
                        self.stack.truncate(frame.stack_base);
                        self.ip = frame.return_ip;
                    } else {
                        self.halted = true;
                    }
                }
                Op::ReturnValue => {
                    let val = self.pop();
                    if let Some(frame) = self.frames.pop() {
                        self.stack.truncate(frame.stack_base);
                        self.ip = frame.return_ip;
                        self.push(val);
                    } else {
                        self.halted = true;
                        return VMResult::Ok(val);
                    }
                }

                // ── Scope ──
                Op::PushFrame => {
                    self.frames.push(Frame {
                        return_ip: self.ip,
                        stack_base: self.stack.len(),
                        slots: Vec::new(),
                    });
                }
                Op::PopFrame => {
                    if let Some(frame) = self.frames.pop() {
                        self.stack.truncate(frame.stack_base);
                    }
                }

                // ── I/O (write directly, no intermediate Vec) ──
                Op::Print(n) => {
                    let n = *n;
                    let start = self.stack.len().saturating_sub(n as usize);
                    use std::io::Write;
                    let stdout = std::io::stdout();
                    let mut lock = stdout.lock();
                    for v in &self.stack[start..] {
                        let _ = write!(lock, "{}", v.as_str_cow());
                    }
                    self.stack.truncate(start);
                }
                Op::PrintLn(n) => {
                    let n = *n;
                    let start = self.stack.len().saturating_sub(n as usize);
                    use std::io::Write;
                    let stdout = std::io::stdout();
                    let mut lock = stdout.lock();
                    for v in &self.stack[start..] {
                        let _ = write!(lock, "{}", v.as_str_cow());
                    }
                    let _ = writeln!(lock);
                    self.stack.truncate(start);
                }
                Op::ReadLine => {
                    let mut line = String::new();
                    let _ = std::io::stdin().read_line(&mut line);
                    self.push(Value::str(line.trim_end_matches('\n')));
                }

                // ── Fused superinstructions ──
                Op::PreIncSlot(slot) => {
                    let val = self.get_slot(*slot).to_int() + 1;
                    self.set_slot(*slot, Value::Int(val));
                    self.push(Value::Int(val));
                }
                Op::PreIncSlotVoid(slot) => {
                    let val = self.get_slot(*slot).to_int() + 1;
                    self.set_slot(*slot, Value::Int(val));
                }
                Op::SlotLtIntJumpIfFalse(slot, limit, target) => {
                    if self.get_slot(*slot).to_int() >= *limit as i64 {
                        self.ip = *target;
                    }
                }
                Op::SlotIncLtIntJumpBack(slot, limit, target) => {
                    let val = self.get_slot(*slot).to_int() + 1;
                    self.set_slot(*slot, Value::Int(val));
                    if val < *limit as i64 {
                        self.ip = *target;
                    }
                }
                Op::AccumSumLoop(sum_slot, i_slot, limit) => {
                    let mut sum = self.get_slot(*sum_slot).to_int();
                    let mut i = self.get_slot(*i_slot).to_int();
                    let lim = *limit as i64;
                    while i < lim {
                        sum += i;
                        i += 1;
                    }
                    self.set_slot(*sum_slot, Value::Int(sum));
                    self.set_slot(*i_slot, Value::Int(i));
                }
                Op::AddAssignSlotVoid(a, b) => {
                    let sum = self.get_slot(*a).to_int() + self.get_slot(*b).to_int();
                    self.set_slot(*a, Value::Int(sum));
                }

                // ── Status ──
                Op::SetStatus => {
                    self.last_status = self.pop().to_int() as i32;
                }
                Op::GetStatus => {
                    self.push(Value::Status(self.last_status));
                }

                // ── Extension dispatch ──
                Op::Extended(id, arg) => {
                    let (id, arg) = (*id, *arg);
                    if let Some(mut handler) = self.ext_handler.take() {
                        handler(self, id, arg);
                        self.ext_handler = Some(handler);
                    }
                }
                Op::ExtendedWide(id, payload) => {
                    let (id, payload) = (*id, *payload);
                    if let Some(mut handler) = self.ext_wide_handler.take() {
                        handler(self, id, payload);
                        self.ext_wide_handler = Some(handler);
                    }
                }

                // ── Arrays ──
                Op::GetArray(idx) => {
                    let val = self.get_var(*idx);
                    self.push(val);
                }
                Op::SetArray(idx) => {
                    let val = self.pop();
                    self.set_var(*idx, val);
                }
                Op::DeclareArray(idx) => {
                    self.set_var(*idx, Value::Array(Vec::new()));
                }
                Op::ArrayGet(arr_idx) => {
                    let index = self.pop().to_int() as usize;
                    let idx = *arr_idx as usize;
                    let val = if idx < self.globals.len() {
                        if let Value::Array(ref arr) = self.globals[idx] {
                            arr.get(index).cloned().unwrap_or(Value::Undef)
                        } else {
                            Value::Undef
                        }
                    } else {
                        Value::Undef
                    };
                    self.push(val);
                }
                Op::ArraySet(arr_idx) => {
                    let index = self.pop().to_int() as usize;
                    let val = self.pop();
                    let idx = *arr_idx as usize;
                    if idx >= self.globals.len() {
                        self.globals.resize(idx + 1, Value::Undef);
                    }
                    if let Value::Array(ref mut vec) = self.globals[idx] {
                        if index >= vec.len() {
                            vec.resize(index + 1, Value::Undef);
                        }
                        vec[index] = val;
                    }
                }
                Op::ArrayPush(arr_idx) => {
                    let val = self.pop();
                    let idx = *arr_idx as usize;
                    if idx >= self.globals.len() {
                        self.globals.resize(idx + 1, Value::Undef);
                    }
                    if let Value::Array(ref mut vec) = self.globals[idx] {
                        vec.push(val);
                    }
                }
                Op::ArrayPop(arr_idx) => {
                    let idx = *arr_idx as usize;
                    let val = if idx < self.globals.len() {
                        if let Value::Array(ref mut vec) = self.globals[idx] {
                            vec.pop().unwrap_or(Value::Undef)
                        } else {
                            Value::Undef
                        }
                    } else {
                        Value::Undef
                    };
                    self.push(val);
                }
                Op::ArrayShift(arr_idx) => {
                    let idx = *arr_idx as usize;
                    let val = if idx < self.globals.len() {
                        if let Value::Array(ref mut vec) = self.globals[idx] {
                            if vec.is_empty() {
                                Value::Undef
                            } else {
                                vec.remove(0)
                            }
                        } else {
                            Value::Undef
                        }
                    } else {
                        Value::Undef
                    };
                    self.push(val);
                }
                Op::ArrayLen(arr_idx) => {
                    let idx = *arr_idx as usize;
                    let len = if idx < self.globals.len() {
                        if let Value::Array(ref vec) = self.globals[idx] {
                            vec.len() as i64
                        } else {
                            0
                        }
                    } else {
                        0
                    };
                    self.push(Value::Int(len));
                }
                Op::MakeArray(n) => {
                    let n = *n;
                    let start = self.stack.len().saturating_sub(n as usize);
                    let elements: Vec<Value> = self.stack.drain(start..).collect();
                    self.push(Value::Array(elements));
                }

                // ── Hashes ──
                Op::GetHash(idx) => {
                    let val = self.get_var(*idx);
                    self.push(val);
                }
                Op::SetHash(idx) => {
                    let val = self.pop();
                    self.set_var(*idx, val);
                }
                Op::DeclareHash(idx) => {
                    self.set_var(*idx, Value::Hash(std::collections::HashMap::new()));
                }
                Op::HashGet(hash_idx) => {
                    let key_val = self.pop();
                    let key = key_val.as_str_cow();
                    let idx = *hash_idx as usize;
                    let val = if idx < self.globals.len() {
                        if let Value::Hash(ref map) = self.globals[idx] {
                            map.get(key.as_ref()).cloned().unwrap_or(Value::Undef)
                        } else {
                            Value::Undef
                        }
                    } else {
                        Value::Undef
                    };
                    self.push(val);
                }
                Op::HashSet(hash_idx) => {
                    let key = self.pop().to_str();
                    let val = self.pop();
                    let idx = *hash_idx as usize;
                    if idx >= self.globals.len() {
                        self.globals.resize(idx + 1, Value::Undef);
                    }
                    if let Value::Hash(ref mut map) = self.globals[idx] {
                        map.insert(key, val);
                    }
                }
                Op::HashDelete(hash_idx) => {
                    let key_val = self.pop();
                    let key = key_val.as_str_cow();
                    let idx = *hash_idx as usize;
                    let val = if idx < self.globals.len() {
                        if let Value::Hash(ref mut map) = self.globals[idx] {
                            map.remove(key.as_ref()).unwrap_or(Value::Undef)
                        } else {
                            Value::Undef
                        }
                    } else {
                        Value::Undef
                    };
                    self.push(val);
                }
                Op::HashExists(hash_idx) => {
                    let key_val = self.pop();
                    let key = key_val.as_str_cow();
                    let idx = *hash_idx as usize;
                    let val = if idx < self.globals.len() {
                        if let Value::Hash(ref map) = self.globals[idx] {
                            map.contains_key(key.as_ref())
                        } else {
                            false
                        }
                    } else {
                        false
                    };
                    self.push(Value::Bool(val));
                }
                Op::HashKeys(hash_idx) => {
                    let idx = *hash_idx as usize;
                    let arr = if idx < self.globals.len() {
                        if let Value::Hash(ref map) = self.globals[idx] {
                            let mut keys = Vec::with_capacity(map.len());
                            keys.extend(map.keys().map(|k| Value::str(k.as_str())));
                            keys
                        } else {
                            Vec::new()
                        }
                    } else {
                        Vec::new()
                    };
                    self.push(Value::Array(arr));
                }
                Op::HashValues(hash_idx) => {
                    let idx = *hash_idx as usize;
                    let arr = if idx < self.globals.len() {
                        if let Value::Hash(ref map) = self.globals[idx] {
                            let mut vals = Vec::with_capacity(map.len());
                            vals.extend(map.values().cloned());
                            vals
                        } else {
                            Vec::new()
                        }
                    } else {
                        Vec::new()
                    };
                    self.push(Value::Array(arr));
                }
                Op::MakeHash(n) => {
                    let n = *n;
                    let start = self.stack.len().saturating_sub(n as usize);
                    let pairs: Vec<Value> = self.stack.drain(start..).collect();
                    let mut map = std::collections::HashMap::with_capacity(pairs.len() / 2);
                    let mut iter = pairs.into_iter();
                    while let Some(key) = iter.next() {
                        if let Some(val) = iter.next() {
                            map.insert(key.to_str(), val);
                        }
                    }
                    self.push(Value::Hash(map));
                }

                // ── Range ──
                Op::Range => {
                    let to = self.pop().to_int();
                    let from = self.pop().to_int();
                    let cap = (to - from + 1).max(0) as usize;
                    let mut arr = Vec::with_capacity(cap);
                    arr.extend((from..=to).map(Value::Int));
                    self.push(Value::Array(arr));
                }
                Op::RangeStep => {
                    let step = self.pop().to_int();
                    let to = self.pop().to_int();
                    let from = self.pop().to_int();
                    let cap = if step > 0 {
                        ((to - from) / step + 1).max(0) as usize
                    } else if step < 0 {
                        ((from - to) / (-step) + 1).max(0) as usize
                    } else {
                        0
                    };
                    let mut arr = Vec::with_capacity(cap);
                    if step > 0 {
                        let mut i = from;
                        while i <= to {
                            arr.push(Value::Int(i));
                            i += step;
                        }
                    } else if step < 0 {
                        let mut i = from;
                        while i >= to {
                            arr.push(Value::Int(i));
                            i += step;
                        }
                    }
                    self.push(Value::Array(arr));
                }

                // ── Shell ops ──
                Op::TestFile(test_type) => {
                    let test_type = *test_type;
                    let path = self.pop().to_str();
                    let result = match test_type {
                        crate::op::file_test::EXISTS => std::path::Path::new(&path).exists(),
                        crate::op::file_test::IS_FILE => std::path::Path::new(&path).is_file(),
                        crate::op::file_test::IS_DIR => std::path::Path::new(&path).is_dir(),
                        crate::op::file_test::IS_SYMLINK => {
                            std::path::Path::new(&path).is_symlink()
                        }
                        crate::op::file_test::IS_READABLE | crate::op::file_test::IS_WRITABLE => {
                            std::path::Path::new(&path).exists()
                        }
                        crate::op::file_test::IS_EXECUTABLE => {
                            #[cfg(unix)]
                            {
                                use std::os::unix::fs::PermissionsExt;
                                std::fs::metadata(&path)
                                    .map(|m| m.permissions().mode() & 0o111 != 0)
                                    .unwrap_or(false)
                            }
                            #[cfg(not(unix))]
                            {
                                std::path::Path::new(&path).exists()
                            }
                        }
                        crate::op::file_test::IS_NONEMPTY => std::fs::metadata(&path)
                            .map(|m| m.len() > 0)
                            .unwrap_or(false),
                        crate::op::file_test::IS_SOCKET => {
                            #[cfg(unix)]
                            {
                                use std::os::unix::fs::FileTypeExt;
                                std::fs::symlink_metadata(&path)
                                    .map(|m| m.file_type().is_socket())
                                    .unwrap_or(false)
                            }
                            #[cfg(not(unix))]
                            {
                                false
                            }
                        }
                        crate::op::file_test::IS_FIFO => {
                            #[cfg(unix)]
                            {
                                use std::os::unix::fs::FileTypeExt;
                                std::fs::symlink_metadata(&path)
                                    .map(|m| m.file_type().is_fifo())
                                    .unwrap_or(false)
                            }
                            #[cfg(not(unix))]
                            {
                                false
                            }
                        }
                        crate::op::file_test::IS_BLOCK_DEV => {
                            #[cfg(unix)]
                            {
                                use std::os::unix::fs::FileTypeExt;
                                std::fs::symlink_metadata(&path)
                                    .map(|m| m.file_type().is_block_device())
                                    .unwrap_or(false)
                            }
                            #[cfg(not(unix))]
                            {
                                false
                            }
                        }
                        crate::op::file_test::IS_CHAR_DEV => {
                            #[cfg(unix)]
                            {
                                use std::os::unix::fs::FileTypeExt;
                                std::fs::symlink_metadata(&path)
                                    .map(|m| m.file_type().is_char_device())
                                    .unwrap_or(false)
                            }
                            #[cfg(not(unix))]
                            {
                                false
                            }
                        }
                        _ => false,
                    };
                    self.push(Value::Bool(result));
                }

                Op::Exec(argc) => {
                    let argc = *argc;
                    let start = self.stack.len().saturating_sub(argc as usize);
                    let args: Vec<String> = self.stack.drain(start..).map(|v| v.to_str()).collect();
                    if let Some(cmd) = args.first() {
                        // Check if it's a shell function
                        let name_idx = self.chunk.names.iter().position(|n| n == cmd);
                        if let Some(name_idx) = name_idx {
                            if let Some(entry_ip) = self.chunk.find_sub(name_idx as u16) {
                                // Push arguments for the function (skip command name)
                                for arg in &args[1..] {
                                    self.push(Value::str(arg));
                                }
                                // Push frame and call
                                self.frames.push(Frame {
                                    return_ip: self.ip,
                                    stack_base: self.stack.len() - (args.len() - 1),
                                    slots: Vec::with_capacity(8),
                                });
                                self.ip = entry_ip;
                                continue;
                            }
                        }

                        match cmd.as_str() {
                            "true" => self.push(Value::Status(0)),
                            "false" => self.push(Value::Status(1)),
                            "echo" => {
                                println!("{}", args[1..].join(" "));
                                self.push(Value::Status(0));
                            }
                            "test" | "[" => {
                                self.push(Value::Status(0));
                            }
                            _ => {
                                use std::process::{Command, Stdio};
                                match Command::new(cmd)
                                    .args(&args[1..])
                                    .stdout(Stdio::inherit())
                                    .stderr(Stdio::inherit())
                                    .status()
                                {
                                    Ok(status) => {
                                        self.push(Value::Status(status.code().unwrap_or(1)))
                                    }
                                    Err(_) => self.push(Value::Status(127)),
                                }
                            }
                        }
                    } else {
                        self.push(Value::Status(0));
                    }
                }
                Op::ExecBg(argc) => {
                    let argc = *argc;
                    let start = self.stack.len().saturating_sub(argc as usize);
                    let args: Vec<String> = self.stack.drain(start..).map(|v| v.to_str()).collect();
                    if let Some(cmd) = args.first() {
                        use std::process::{Command, Stdio};
                        let _ = Command::new(cmd)
                            .args(&args[1..])
                            .stdout(Stdio::null())
                            .stderr(Stdio::null())
                            .spawn();
                    }
                    self.push(Value::Status(0));
                }

                // ── Shell stubs ──
                Op::PipelineBegin(_) | Op::PipelineStage | Op::SubshellBegin | Op::SubshellEnd => {}
                Op::PipelineEnd => {
                    self.push(Value::Status(self.last_status));
                }
                Op::Redirect(_, _) => {
                    let _ = self.pop();
                }
                Op::HereDoc(_) => {}
                Op::HereString => {
                    let _ = self.pop();
                }
                Op::CmdSubst(_) => {
                    self.push(Value::str(""));
                }
                Op::ProcessSubIn(_) | Op::ProcessSubOut(_) => {
                    self.push(Value::str(""));
                }
                Op::Glob | Op::GlobRecursive => {
                    let pat_val = self.pop();
                    let pattern = pat_val.as_str_cow();
                    let matches: Vec<Value> = glob::glob(&pattern)
                        .into_iter()
                        .flat_map(|paths| paths.filter_map(|p| p.ok()))
                        .map(|p| Value::str(p.to_string_lossy()))
                        .collect();
                    self.push(Value::Array(matches));
                }
                Op::TrapSet(_) | Op::TrapCheck => {}
                Op::ExpandParam(_) | Op::WordSplit | Op::BraceExpand | Op::TildeExpand => {}

                // ── Remaining fused ops ──
                Op::ConcatConstLoop(const_idx, s_slot, i_slot, limit) => {
                    let c_str = self
                        .chunk
                        .constants
                        .get(*const_idx as usize)
                        .map(|v| v.as_str_cow())
                        .unwrap_or(std::borrow::Cow::Borrowed(""));
                    let mut s = self.get_slot(*s_slot).to_str();
                    let mut i = self.get_slot(*i_slot).to_int();
                    let lim = *limit as i64;
                    let iters = (lim - i).max(0) as usize;
                    s.reserve(c_str.len() * iters);
                    while i < lim {
                        s.push_str(&c_str);
                        i += 1;
                    }
                    self.set_slot(*s_slot, Value::str(s));
                    self.set_slot(*i_slot, Value::Int(i));
                }
                Op::PushIntRangeLoop(arr_idx, i_slot, limit) => {
                    let mut i = self.get_slot(*i_slot).to_int();
                    let lim = *limit as i64;
                    let arr = self.get_var(*arr_idx);
                    let mut vec = if let Value::Array(v) = arr {
                        v
                    } else {
                        Vec::new()
                    };
                    vec.reserve((lim - i).max(0) as usize);
                    while i < lim {
                        vec.push(Value::Int(i));
                        i += 1;
                    }
                    self.set_var(*arr_idx, Value::Array(vec));
                    self.set_slot(*i_slot, Value::Int(i));
                }

                // ── Higher-order (stubs) ──
                Op::MapBlock(_)
                | Op::GrepBlock(_)
                | Op::SortBlock(_)
                | Op::SortDefault
                | Op::ForEachBlock(_) => {}

                // ── Builtins (inline cache) ──
                Op::CallBuiltin(id, argc) => {
                    let (id, argc) = (*id, *argc);
                    if let Some(Some(handler)) = self.builtin_table.get(id as usize) {
                        let result = handler(self, argc);
                        self.push(result);
                    }
                }
            }
        }

        if let Some(val) = self.stack.pop() {
            VMResult::Ok(val)
        } else {
            VMResult::Halted
        }
    }

    // ── Helpers ──

    fn get_var(&self, idx: u16) -> Value {
        self.globals
            .get(idx as usize)
            .cloned()
            .unwrap_or(Value::Undef)
    }

    fn set_var(&mut self, idx: u16, val: Value) {
        let idx = idx as usize;
        if idx >= self.globals.len() {
            self.globals.resize(idx + 1, Value::Undef);
        }
        self.globals[idx] = val;
    }

    fn get_slot(&self, slot: u16) -> Value {
        self.frames
            .last()
            .and_then(|f| f.slots.get(slot as usize))
            .cloned()
            .unwrap_or(Value::Undef)
    }

    fn set_slot(&mut self, slot: u16, val: Value) {
        if let Some(frame) = self.frames.last_mut() {
            let idx = slot as usize;
            if idx >= frame.slots.len() {
                frame.slots.resize(idx + 1, Value::Undef);
            }
            frame.slots[idx] = val;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunk::ChunkBuilder;

    #[test]
    fn test_arithmetic() {
        let mut b = ChunkBuilder::new();
        b.emit(Op::LoadInt(10), 1);
        b.emit(Op::LoadInt(32), 1);
        b.emit(Op::Add, 1);
        let mut vm = VM::new(b.build());
        match vm.run() {
            VMResult::Ok(Value::Int(42)) => {}
            other => panic!("expected Int(42), got {:?}", other),
        }
    }

    #[test]
    fn test_jump() {
        let mut b = ChunkBuilder::new();
        b.emit(Op::LoadInt(1), 1);
        b.emit(Op::Jump(3), 1);
        b.emit(Op::LoadInt(999), 1); // skipped
                                     // ip 3:
        b.emit(Op::LoadInt(2), 1);
        b.emit(Op::Add, 1);
        let mut vm = VM::new(b.build());
        match vm.run() {
            VMResult::Ok(Value::Int(3)) => {}
            other => panic!("expected Int(3), got {:?}", other),
        }
    }

    #[test]
    fn test_fused_sum_loop() {
        // sum = 0; for i in 0..100 { sum += i }
        let mut b = ChunkBuilder::new();
        b.emit(Op::PushFrame, 1);
        b.emit(Op::LoadInt(0), 1);
        b.emit(Op::SetSlot(0), 1); // sum = 0
        b.emit(Op::LoadInt(0), 1);
        b.emit(Op::SetSlot(1), 1); // i = 0
        b.emit(Op::AccumSumLoop(0, 1, 100), 1);
        b.emit(Op::GetSlot(0), 1);

        let mut vm = VM::new(b.build());
        match vm.run() {
            VMResult::Ok(Value::Int(4950)) => {}
            other => panic!("expected Int(4950), got {:?}", other),
        }
    }

    #[test]
    fn test_function_call() {
        let mut b = ChunkBuilder::new();
        let double_name = b.add_name("double");

        // main: push 21, call double, result on stack
        b.emit(Op::LoadInt(21), 1);
        b.emit(Op::Call(double_name, 1), 1);
        let end_jump = b.emit(Op::Jump(0), 1); // jump past function body

        // double: arg * 2
        let double_ip = b.current_pos();
        b.add_sub_entry(double_name, double_ip);
        b.emit(Op::LoadInt(2), 2);
        b.emit(Op::Mul, 2);
        b.emit(Op::ReturnValue, 2);

        b.patch_jump(end_jump, b.current_pos());

        let mut vm = VM::new(b.build());
        match vm.run() {
            VMResult::Ok(Value::Int(42)) => {}
            other => panic!("expected Int(42), got {:?}", other),
        }
    }

    #[test]
    fn test_builtin_cache() {
        let mut b = ChunkBuilder::new();
        b.emit(Op::LoadInt(10), 1);
        b.emit(Op::CallBuiltin(0, 1), 1);
        let mut vm = VM::new(b.build());
        vm.register_builtin(0, |vm, _argc| {
            let val = vm.pop();
            Value::Int(val.to_int() * 2)
        });
        match vm.run() {
            VMResult::Ok(Value::Int(20)) => {}
            other => panic!("expected Int(20), got {:?}", other),
        }
    }
}
