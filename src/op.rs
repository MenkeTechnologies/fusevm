//! Bytecode instruction set for fusevm.
//!
//! Universal ops that any language frontend can target.
//! Language-specific ops use `Extended(u16, u8)` which dispatches
//! through a handler table registered by the frontend.

use serde::{Deserialize, Serialize};
use std::hash::{Hash, Hasher};

/// Stack-based bytecode instruction set.
///
/// Operands: u16 for pool indices (64k names/constants), usize for jump targets.
/// Language-specific operations use `Extended` with a frontend-registered handler.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Op {
    /// No-op; consumed for cycle-counting and as a branch sentinel.
    Nop,

    // ── Constants ──
    /// Push int value onto the stack.
    LoadInt(i64),
    /// Push float value onto the stack.
    LoadFloat(f64),
    /// index into constant pool
    LoadConst(u16),
    /// Push true value onto the stack.
    LoadTrue,
    /// Push false value onto the stack.
    LoadFalse,
    /// Push undef value onto the stack.
    LoadUndef,

    // ── Stack ──
    /// Discard the top of the stack.
    Pop,
    /// Duplicate the top of the stack.
    Dup,
    /// Duplicate the top two values on the stack.
    Dup2,
    /// Swap the top two stack values.
    Swap,
    /// Rotate the top three stack values (a b c → b c a).
    Rot,

    // ── Variables (u16 = name pool index) ──
    /// Read var into the stack via name-pool index.
    GetVar(u16),
    /// Pop top of stack into var via name-pool index.
    SetVar(u16),
    /// Declare a fresh var binding in the current scope.
    DeclareVar(u16),
    /// Slot-indexed fast path (frame slot index, avoids name lookup)
    GetSlot(u16),
    /// Pop top of stack into slot via name-pool index.
    SetSlot(u16),
    /// Slot-based array index: stack: \[index\], slot contains array → value
    SlotArrayGet(u16),
    /// Slot-based array set: stack: \[value, index\], slot contains array
    SlotArraySet(u16),

    // ── Arrays ──
    /// Read array into the stack via name-pool index.
    GetArray(u16),
    /// Pop top of stack into array via name-pool index.
    SetArray(u16),
    /// Declare a fresh array binding in the current scope.
    DeclareArray(u16),
    /// stack: \[index\] → value
    ArrayGet(u16),
    /// stack: \[value, index\]
    ArraySet(u16),
    /// stack: \[value\]
    ArrayPush(u16),
    /// → popped value
    ArrayPop(u16),
    /// → shifted value
    ArrayShift(u16),
    /// → length
    ArrayLen(u16),
    /// pop N values, push as array
    MakeArray(u16),

    // ── Hashes ──
    /// Read hash into the stack via name-pool index.
    GetHash(u16),
    /// Pop top of stack into hash via name-pool index.
    SetHash(u16),
    /// Declare a fresh hash binding in the current scope.
    DeclareHash(u16),
    /// stack: \[key\] → value
    HashGet(u16),
    /// stack: \[value, key\]
    HashSet(u16),
    /// stack: \[key\] → deleted value
    HashDelete(u16),
    /// stack: \[key\] → bool
    HashExists(u16),
    /// → array of keys
    HashKeys(u16),
    /// → array of values
    HashValues(u16),
    /// pop N key-value pairs, push as hash
    MakeHash(u16),

    // ── Arithmetic ──
    /// Numeric `a + b` (pops 2, pushes sum).
    Add,
    /// Numeric `a - b` (pops 2, pushes difference).
    Sub,
    /// Numeric `a * b` (pops 2, pushes product).
    Mul,
    /// Numeric `a / b` (pops 2, pushes quotient).
    Div,
    /// Numeric `a % b` (pops 2, pushes remainder).
    Mod,
    /// Numeric `a ** b` (pops 2, pushes power).
    Pow,
    /// Numeric unary minus (`-x`).
    Negate,
    /// In-place += 1 on top-of-stack numeric.
    Inc,
    /// In-place -= 1 on top-of-stack numeric.
    Dec,

    // ── String ──
    /// String concatenation `a . b`.
    Concat,
    /// Repeat a string N times (`a x b` in Perl).
    StringRepeat,
    /// Length of top-of-stack string in bytes.
    StringLen,

    // ── Comparison (numeric) ──
    /// Numeric `eq` comparison.
    NumEq,
    /// Numeric `ne` comparison.
    NumNe,
    /// Numeric `lt` comparison.
    NumLt,
    /// Numeric `gt` comparison.
    NumGt,
    /// Numeric `le` comparison.
    NumLe,
    /// Numeric `ge` comparison.
    NumGe,
    /// <=> → -1, 0, 1
    Spaceship,

    // ── Comparison (string) ──
    /// String `eq` comparison.
    StrEq,
    /// String `ne` comparison.
    StrNe,
    /// String `lt` comparison.
    StrLt,
    /// String `gt` comparison.
    StrGt,
    /// String `le` comparison.
    StrLe,
    /// String `ge` comparison.
    StrGe,
    /// String `cmp` comparison.
    StrCmp,

    // ── Logical / Bitwise ──
    /// Boolean `!` (eager, evaluates both operands for binary forms).
    LogNot,
    /// differs from short-circuit jumps: evaluates both
    LogAnd,
    /// Boolean `||` (eager, evaluates both operands for binary forms).
    LogOr,
    /// Bitwise `and` operation.
    BitAnd,
    /// Bitwise `or` operation.
    BitOr,
    /// Bitwise `xor` operation.
    BitXor,
    /// Bitwise `not` operation.
    BitNot,
    /// Bit-shift left (`<<`/`>>`).
    Shl,
    /// Bit-shift right (`<<`/`>>`).
    Shr,

    // ── Control flow ──
    /// Unconditional jump to the given chunk offset.
    Jump(usize),
    /// Jump if top-of-stack is true; pops predicate.
    JumpIfTrue(usize),
    /// Jump if top-of-stack is false; pops predicate.
    JumpIfFalse(usize),
    /// short-circuit ||
    JumpIfTrueKeep(usize),
    /// short-circuit &&
    JumpIfFalseKeep(usize),

    // ── Functions ──
    /// Call: name_index, arg_count
    Call(u16, u8),
    /// Return from current function with no value (pushes Undef).
    Return,
    /// Return from current function using top-of-stack as the result.
    ReturnValue,

    // ── Scope ──
    /// Push a call frame on the frame stack.
    PushFrame,
    /// Pop a call frame on the frame stack.
    PopFrame,

    // ── I/O ──
    /// Print N values from stack to stdout
    Print(u8),
    /// Print N values + newline
    PrintLn(u8),
    /// Read line from stdin, push as string
    ReadLine,

    // ── Collections ──
    /// [from, to] → array
    Range,
    /// [from, to, step] → array
    RangeStep,

    // ── Higher-order (u16 = block index in chunk) ──
    /// Higher-order `map` over a slot-array using the block at the given chunk index.
    MapBlock(u16),
    /// Higher-order `grep` over a slot-array using the block at the given chunk index.
    GrepBlock(u16),
    /// Higher-order `sort` over a slot-array using the block at the given chunk index.
    SortBlock(u16),
    /// sort with default string comparison
    SortDefault,
    /// Higher-order `foreach` over a slot-array using the block at the given chunk index.
    ForEachBlock(u16),

    // ── Fused superinstructions ──
    // These are the performance secret sauce.
    // The compiler detects hot loop patterns and emits these
    // instead of multi-op sequences.
    /// Slot-indexed pre-increment (no stack traffic)
    PreIncSlot(u16),
    /// `if ($slot < INT) goto target` — fused compare + branch
    SlotLtIntJumpIfFalse(u16, i32, usize),
    /// `$slot += 1; if $slot < limit goto body` — fused loop backedge
    SlotIncLtIntJumpBack(u16, i32, usize),
    /// `while $i < limit { $sum += $i; $i += 1 }` — entire counted sum loop
    AccumSumLoop(u16, u16, i32),
    /// `while $i < limit { $s .= CONST; $i += 1 }` — fused string append loop
    ConcatConstLoop(u16, u16, u16, i32),
    /// `while $i < limit { push @a, $i; $i += 1 }` — fused array push loop
    PushIntRangeLoop(u16, u16, i32),
    /// Void-context slot add-assign: `$a += $b` (no stack push)
    AddAssignSlotVoid(u16, u16),
    /// Void-context pre-increment: `++$slot` (no stack push)
    PreIncSlotVoid(u16),
    /// Slot-indexed pre-decrement: `--$slot`, pushes the new value
    PreDecSlot(u16),
    /// Slot-indexed post-increment: `$slot++`, pushes the old value
    PostIncSlot(u16),
    /// Slot-indexed post-decrement: `$slot--`, pushes the old value
    PostDecSlot(u16),

    // ── Builtins ──
    /// Call a registered builtin by ID: (builtin_id, arg_count)
    /// The builtin table is registered by the frontend at VM init.
    CallBuiltin(u16, u8),

    // ── Extension point ──
    /// Language-specific opcode dispatched through a frontend handler table.
    /// u16 = extension op ID, u8 = inline operand.
    /// Frontends register a `fn(&mut VM, u16, u8)` handler at init.
    Extended(u16, u8),
    /// Extended with usize payload (for jump targets, large indices)
    ExtendedWide(u16, usize),

    // ── Shell ops (registered via Extended, but defined here for type safety) ──
    // These are first-class because process control is universal enough
    // that multiple frontends need them (shell, scripting, build tools).
    /// Spawn external command: pop N args from stack, exec, push exit status
    Exec(u8),
    /// Spawn background: like Exec but don't wait
    ExecBg(u8),
    /// Set up N-stage pipeline
    PipelineBegin(u8),
    /// Wire next pipeline stage
    PipelineStage,
    /// Wait for pipeline, push last status
    PipelineEnd,
    /// Redirect fd: (source_fd, op_byte) — target on stack
    Redirect(u8, u8),
    /// Here-document: fd on stack, content from constant pool
    HereDoc(u16),
    /// Here-string: fd on stack, word on stack
    HereString,
    /// Command substitution: capture stdout of subprogram
    CmdSubst(u16), // u16 = bytecode range index
    /// Subshell: isolate scope
    SubshellBegin,
    /// Shell Ops (Registered Via Extended, But Defined Here For Type Safety) operation (`SubshellEnd`).
    SubshellEnd,
    /// Process substitution <(cmd) — push FIFO path
    ProcessSubIn(u16),
    /// Process substitution >(cmd) — push FIFO path
    ProcessSubOut(u16),
    /// Glob expand: pop pattern, push array of matches
    Glob,
    /// Recursive glob (parallel): pop pattern, push array
    GlobRecursive,
    /// File test: u8 encodes test type (-f=0, -d=1, -r=2, -w=3, -x=4, -e=5, -s=6, -L=7)
    TestFile(u8),
    /// Set last exit status ($?)
    SetStatus,
    /// Get last exit status
    GetStatus,
    /// Set trap handler: signal on stack, handler bytecode range
    TrapSet(u16),
    /// Check pending traps (inserted between ops by compiler)
    TrapCheck,
    /// Expand ${var:-default} family: u8 encodes modifier type
    ExpandParam(u8),
    /// Word split by IFS
    WordSplit,
    /// Brace expand {a,b} and {1..10}
    BraceExpand,
    /// Tilde expand ~ and ~user
    TildeExpand,
    /// Call user-defined shell function by name pool index with N args.
    /// Falls through to host.call_function() then host.exec() if not found.
    /// stack: \[arg_N, ..., arg_1\] → pushes Status
    CallFunction(u16, u8),
    /// Glob-pattern match: pop pattern, pop string, push Bool.
    /// Used by `[[ x = pat ]]` and `case` arm matching.
    StrMatch,
    /// Regex match: pop regex, pop string, push Bool. (`=~`)
    RegexMatch,
    /// Begin scoped redirection block: u8 = number of redirects already
    /// applied via prior Redirect ops. Saves fd state on the host's stack.
    /// Used for `cmd > out.txt` applied to compound commands and
    /// `func() { ... } > out.txt`.
    WithRedirectsBegin(u8),
    /// End scoped redirection block — restore fd state.
    WithRedirectsEnd,

    // ── AWK ops (first-class, like the shell ops above) ──
    // AWK semantics — field access, record I/O, `print`/`printf` with
    // `OFS`/`ORS`/`OFMT`, `getline`, the string builtins, and `SUBSEP`
    // associative arrays — are a coherent universal vocabulary, so they are
    // defined here as named variants for type safety and visible IR rather
    // than opaque `ExtendedWide` magic numbers. They are still *dispatched*
    // through the frontend-registered [`crate::awk_host::AwkHost`] (AWK values
    // carry POSIX numeric-string duality the core `Value` can't represent), and
    // each variant maps 1:1 onto a `crate::awk_builtins::AWK_*` op id. The VM
    // routes them to the same `dispatch_awk` path as the reserved
    // `ExtendedWide` AWK range, so behavior is identical; these just give
    // frontends a typed, self-documenting alternative to hand-rolled ids.
    /// `$i` — read field `i`; index popped from stack, pushes the field value.
    AwkFieldGet,
    /// `$i = v` — assign field `i`. Stack `[value, index]`; rebuilds `$0`/`NF`.
    AwkFieldSet,
    /// `NF` — push the current field count.
    AwkNf,
    /// `$0 = v` — replace the whole record (resplit). Stack `[value]`.
    AwkSetRecord,
    /// Read a special AWK variable by name-pool index (`FS`/`NR`/`OFS`/...).
    AwkSpecialGet(u16),
    /// Assign a special AWK variable by name-pool index. Stack `[value]`.
    AwkSpecialSet(u16),
    /// `print a, b, ...` — pops `u8` args, joins with `OFS`, ends with `ORS`.
    AwkPrint(u8),
    /// `printf fmt, ...` — pops `u8` args (last popped is the format).
    AwkPrintf(u8),
    /// `sprintf(fmt, ...)` — like `AwkPrintf` but pushes the formatted string.
    AwkSprintf(u8),
    /// `getline` family. `u8` encodes the source kind (see
    /// `crate::awk_builtins::getline_source`); pushes status (1/0/-1).
    AwkGetline(u8),
    /// `length` / `length(x)` — pops `u8` args (0 ⇒ `length($0)`).
    AwkLength(u8),
    /// `substr(s, m [, n])` — pops `u8` args (2 or 3).
    AwkSubstr(u8),
    /// `index(s, t)` — stack `[s, t]`; pushes 1-based position or 0.
    AwkIndex,
    /// `split(s, arr [, fs])` — pops `u8` args; pushes field count.
    AwkSplit(u8),
    /// `sub(re, repl [, target])` — pops `u8` args; pushes substitution count.
    AwkSub(u8),
    /// `gsub(re, repl [, target])` — pops `u8` args; pushes substitution count.
    AwkGsub(u8),
    /// `match(s, re)` — stack `[s, re]`; sets `RSTART`/`RLENGTH`, pushes `RSTART`.
    AwkMatch,
    /// `tolower(s)` — stack `[s]`.
    AwkToLower,
    /// `toupper(s)` — stack `[s]`.
    AwkToUpper,
    /// `int(x)` — truncate toward zero. Stack `[x]`. Host-independent.
    AwkInt,
    /// `sqrt(x)` — square root. Stack `[x]`. Host-independent.
    AwkSqrt,
    /// `sin(x)` — sine (radians). Stack `[x]`. Host-independent.
    AwkSin,
    /// `cos(x)` — cosine (radians). Stack `[x]`. Host-independent.
    AwkCos,
    /// `exp(x)` — e^x. Stack `[x]`. Host-independent.
    AwkExp,
    /// `log(x)` — natural log. Stack `[x]`. Host-independent.
    AwkLog,
    /// `atan2(y, x)` — arctangent of `y/x`. Stack `[y, x]`. Host-independent.
    AwkAtan2,
    /// awk `a / b` — float divide. Stack `[a, b]`. Raises a fatal
    /// "division by zero attempted" runtime error when `b == 0` (POSIX awk),
    /// distinct from the shell-arithmetic `Op::Div` which yields `Undef`.
    /// Host-independent.
    AwkDiv,
    /// awk `a % b` — float modulo (`fmod`). Stack `[a, b]`. Raises a fatal
    /// "division by zero attempted in `%'" runtime error when `b == 0` (POSIX
    /// awk), distinct from `Op::Mod`. Host-independent.
    AwkMod,
    /// Block-JIT-eligible variant of [`Op::AwkDiv`] — identical semantics
    /// (float divide, fatal "division by zero attempted" on `b == 0`), but the
    /// block JIT compiles it natively with a guarded early-exit: on a zero
    /// divisor the compiled code sets a thread-local trap code and returns, and
    /// the VM raises the fatal immediately after the block function returns
    /// (`AwkDiv` stays interpreter-only so its existing behavior is untouched).
    /// Emitted only by the awkrs fusevm bridge for offloaded numeric chunks.
    /// Host-independent.
    AwkDivJit,
    /// Block-JIT-eligible variant of [`Op::AwkMod`] — identical semantics
    /// (float modulo, fatal "division by zero attempted in `%'" on `b == 0`)
    /// with the same guarded early-exit trap as [`Op::AwkDivJit`].
    /// Host-independent.
    AwkModJit,
    /// Block-JIT-eligible variant of `sqrt(x)` — `f64::sqrt` on the non-negative
    /// path, libcall warning + `+NaN` on the negative path (matching the awk
    /// "received negative argument" warning). Uses Cranelift block params to
    /// phi-merge the two paths into a single SSA result, so the chunk stays
    /// disk-cacheable through the warn libcall (registered as a named host
    /// helper). Host-independent.
    AwkSqrtJit,
    /// Block-JIT-eligible variant of `log(x)` — `f64::ln` on the non-negative
    /// path (which yields `-inf` for `0.0` naturally), libcall warning + `+NaN`
    /// on the negative path. Same phi-merge pattern as [`Op::AwkSqrtJit`].
    /// The host-default `lint_warn` on zero (gawk LINT=1) is omitted here.
    /// Host-independent.
    AwkLogJit,
    /// Block-JIT-eligible variant of `lshift(a, n)` — pops `[a, n]`. Raises
    /// the awk fatal "negative values are not allowed" when either operand is
    /// negative (guarded early-exit via the trap channel, code 3). The
    /// non-negative path computes `(a as i64) << (n as i64 & 0x3f)` and pushes
    /// the result as a float, matching the awkrs interpreter behavior.
    /// Host-independent.
    AwkLshiftJit,
    /// Block-JIT-eligible variant of `rshift(a, n)` — same guard as
    /// [`Op::AwkLshiftJit`] (trap code 4), uses unsigned-right-shift on the
    /// non-negative path. Host-independent.
    AwkRshiftJit,
    /// Block-JIT-eligible variant of `compl(a)` — pops `[a]`. Raises fatal
    /// "negative value is not allowed" when `a < 0` (trap code 5). Non-negative
    /// path computes `!(a as i64)` and pushes as float. Host-independent.
    AwkComplJit,
    /// Block-JIT-eligible read of awk's `$N` field as a number, where `N` is a
    /// compile-time constant (`u16`). Calls the host-installed hook
    /// `crate::set_awk_field_num_hook`, which the host must set to an
    /// `extern "C" fn(i64) -> f64` BEFORE invoking any chunk containing this
    /// op. Pushes the returned `f64` onto the stack. The hook reads the active
    /// awk record's field via a host-side thread-local Runtime pointer; fusevm
    /// itself stays host-agnostic. If no hook is installed at chunk-run time,
    /// the libcall returns `0.0` (matching awk's `$N` for missing fields).
    AwkGetFieldNum(u16),
    /// Always-float exponentiation: pops `[base, exp]`, pushes
    /// `Float(base.powf(exp))`. Unlike [`Op::Pow`] (whose JIT lowering keeps an
    /// integer result for two static-`Int` operands), this op coerces BOTH
    /// operands to `f64` in every tier, so the JIT and interpreter agree and the
    /// chunk stays disk-cacheable (reuses the `pow_f64` host helper). Intended
    /// for frontends whose `**` is always floating-point (e.g. strykelang).
    /// Host-independent.
    PowFloat,
    /// Always-float square root: pops `[x]`, pushes `Float(x.sqrt())` (NaN for a
    /// negative operand). Lowers to a native Cranelift `fsqrt` in the JIT and
    /// `f64::sqrt` in the interpreter — identical across tiers, with no host
    /// helper or relocation, so the chunk is trivially disk-cacheable. Unlike
    /// [`Op::AwkSqrt`] (which dispatches through the awk host and is not
    /// JIT-lowerable), this op is host-independent. Intended for frontends whose
    /// `sqrt` is always floating-point (e.g. strykelang).
    SqrtFloat,
    /// Always-float sine (radians): pops `[x]`, pushes `Float(x.sin())`. Lowers
    /// to the `fusevm_jit_sin_f64` host helper in the JIT (the same one
    /// [`Op::AwkSin`] uses) and `f64::sin` in the interpreter. Unlike
    /// [`Op::AwkSin`] (which dispatches through the awk host and would panic in a
    /// non-awk fallback), this op is host-independent and runs in the plain
    /// interpreter, so the chunk is disk-cacheable. Intended for frontends whose
    /// `sin` is always floating-point (e.g. strykelang).
    SinFloat,
    /// Always-float cosine (radians): pops `[x]`, pushes `Float(x.cos())`. See
    /// [`Op::SinFloat`]; reuses the `fusevm_jit_cos_f64` host helper.
    CosFloat,
    /// Always-float exponential: pops `[x]`, pushes `Float(x.exp())`. See
    /// [`Op::SinFloat`]; reuses the `fusevm_jit_exp_f64` host helper.
    ExpFloat,
    /// Always-float two-argument arctangent: the chunk pushes `y` then `x` (so
    /// `x` is on top), pops both, and pushes `Float(y.atan2(x))`. See
    /// [`Op::SinFloat`]; reuses the `fusevm_jit_atan2_f64` host helper. Intended
    /// for frontends whose `atan2` is always floating-point (e.g. strykelang).
    Atan2Float,
    /// Always-float natural logarithm: pops `[x]`, pushes `Float(x.ln())`. Lowers
    /// to the `fusevm_jit_log_f64` host helper in the JIT and `f64::ln` in the
    /// interpreter. Host-independent (unlike [`Op::AwkLog`], which dispatches
    /// through the awk host). Intended for frontends whose `log` is the always-
    /// floating-point natural log (e.g. strykelang).
    LogFloat,
    /// Always-float absolute value: pops `[x]`, pushes `Float(x.abs())`. Lowers
    /// to the native Cranelift `fabs` in the JIT (no host helper, like
    /// [`Op::SqrtFloat`]) and `f64::abs` in the interpreter. Intended for
    /// frontends whose `abs` is floating-point (e.g. strykelang); integral
    /// results format identically to their integer form.
    AbsFloat,
    /// Truncate-to-integer (`int(x)`): pops `[x]`, pushes `Int` of `x`
    /// truncated toward zero. An integer operand is returned unchanged (full
    /// i64 precision); a float is converted via a saturating cast (matching
    /// Rust's `f as i64` and Cranelift `fcvt_to_sint_sat`). Unlike [`Op::AwkInt`]
    /// (which keeps awk's float result), this yields a genuine integer — intended
    /// for frontends whose `int` returns an integer (e.g. strykelang/Perl).
    TruncInt,
    /// Ceiling (always-float). Native Cranelift `ceil`.
    CeilFloat,
    /// Floor (always-float). Native Cranelift `floor`.
    FloorFloat,
    /// Truncate toward zero (always-float). Native Cranelift `trunc`.
    TruncFloat,
    /// Round to nearest, ties to even (always-float). Native Cranelift `nearest`.
    RoundFloat,
    /// Tangent. Host helper `fusevm_jit_tan_f64`.
    TanFloat,
    /// Arcsine. Host helper `fusevm_jit_asin_f64`.
    AsinFloat,
    /// Arccosine. Host helper `fusevm_jit_acos_f64`.
    AcosFloat,
    /// 1-arg arctangent. Host helper `fusevm_jit_atan_f64`.
    AtanFloat,
    /// Hyperbolic sine. Host helper `fusevm_jit_sinh_f64`.
    SinhFloat,
    /// Hyperbolic cosine. Host helper `fusevm_jit_cosh_f64`.
    CoshFloat,
    /// Hyperbolic tangent. Host helper `fusevm_jit_tanh_f64`.
    TanhFloat,
    /// Base-2 logarithm. Host helper `fusevm_jit_log2_f64`.
    Log2Float,
    /// Base-10 logarithm. Host helper `fusevm_jit_log10_f64`.
    Log10Float,
    /// Integer absolute value (`abs(i64)`). Native Cranelift `iabs`.
    AbsInt,
    /// Greatest common divisor of two non-negative i64s (`gcd(|a|, |b|)`).
    /// Host helper `fusevm_jit_gcd_i64`; returns 0 if both inputs are 0.
    GcdInt,
    /// Least common multiple of two non-negative i64s (`lcm(|a|, |b|)`).
    /// Host helper `fusevm_jit_lcm_i64`; returns 0 if either input is 0; saturates
    /// at i64::MAX on overflow.
    LcmInt,
    /// Unix epoch seconds as i64 (`time()`). Host helper `fusevm_jit_time_i64`.
    /// Pure-ish — depends on system clock; not const-foldable.
    TimeInt,
    /// `arr[k]` — stack `[key]`; pushes the element (auto-vivifies to "").
    /// `u16` = name-pool index of the array variable.
    AwkArrayGet(u16),
    /// `arr[k] = v` — stack `[value, key]`. `u16` = array name-pool index.
    AwkArraySet(u16),
    /// `(k in arr)` — stack `[key]`; pushes Bool. `u16` = array name-pool index.
    AwkArrayExists(u16),
    /// `delete arr[k]` — stack `[key]`. `u16` = array name-pool index.
    AwkArrayDelete(u16),
    /// `delete arr` — clear the whole array. `u16` = array name-pool index.
    AwkArrayClear(u16),
    /// `length(arr)` — push the element count. `u16` = array name-pool index.
    AwkArrayLen(u16),
    /// `and(v1, v2, ...)` — bitwise AND; pops `u8` args (≥2). Host-independent.
    AwkAnd(u8),
    /// `or(v1, v2, ...)` — bitwise OR; pops `u8` args (≥2). Host-independent.
    AwkOr(u8),
    /// `xor(v1, v2, ...)` — bitwise XOR; pops `u8` args (≥2). Host-independent.
    AwkXor(u8),
    /// `compl(v)` — bitwise complement. Stack `[v]`. Host-independent.
    AwkCompl,
    /// `lshift(v, n)` — left shift. Stack `[v, n]`. Host-independent.
    AwkLshift,
    /// `rshift(v, n)` — right shift. Stack `[v, n]`. Host-independent.
    AwkRshift,
    /// `strtonum(s)` — parse hex/octal/decimal string to number. Stack `[s]`. Host-independent.
    AwkStrtonum,
    /// `systime()` — seconds since the Unix epoch. Nullary; pushes a number. Host-independent.
    AwkSystime,
    /// `rand()` — next PRNG value in `[0, 1)`. Nullary; pushes a number. VM-owned seed.
    AwkRand,
    /// `srand([x])` — reseed PRNG, push previous seed. Payload = argc (0 or 1). VM-owned seed.
    AwkSrand(u8),
    /// `strftime([fmt[, ts[, utc]]])` — format timestamp. Payload = argc (0..=3). Host-independent.
    AwkStrftime(u8),
    /// `mktime(datespec[, utc])` — datespec → epoch seconds (or -1). Payload = argc (1..=2). Host-independent.
    AwkMktime(u8),
    /// `ord(s)` — Unicode scalar of first char (0 if empty). Host-independent.
    AwkOrd,
    /// `chr(n)` — single-char string for codepoint `n` (empty if invalid). Host-independent.
    AwkChr,
    /// `mkbool(x)` — `1` if truthy else `0`. Host-independent.
    AwkMkbool,
    /// `intdiv(a, b)` — truncating integer quotient (Undef on divide-by-zero). Host-independent.
    AwkIntdiv,
    /// `intdiv0(a, b)` — truncating integer quotient (0 on divide-by-zero; never errors). Host-independent.
    AwkIntdiv0,
    /// `gensub(re, repl, how [, target])` — pops `u8` args; pushes the result string. Host-bound (IGNORECASE, `$0`).
    AwkGensub(u8),
    /// AWK control-flow signal — halts the current chunk and stashes `code` in
    /// the VM for the frontend driver to read after `run()` (`0`=next,
    /// `1`=nextfile, `2`=exit; see `crate::awk_builtins::signal`). Has no
    /// `fusevm::Value` representation, so it is a VM-state side effect rather
    /// than a stack value. zshrs/stryke never emit it. Interpreter-only.
    AwkSignal(u8),
}

/// File test opcodes for `TestFile(u8)`
pub mod file_test {
    /// Regular file (`-f`).
    pub const IS_FILE: u8 = 0;
    /// Directory (`-d`).
    pub const IS_DIR: u8 = 1;
    /// Readable to current uid (`-r`).
    pub const IS_READABLE: u8 = 2;
    /// Writable to current uid (`-w`).
    pub const IS_WRITABLE: u8 = 3;
    /// Executable (`-x`).
    pub const IS_EXECUTABLE: u8 = 4;
    /// Exists (any type, `-e`).
    pub const EXISTS: u8 = 5;
    /// Exists and size > 0 (`-s`).
    pub const IS_NONEMPTY: u8 = 6;
    /// Symbolic link (`-L` / `-h`).
    pub const IS_SYMLINK: u8 = 7;
    /// Unix-domain socket (`-S`).
    pub const IS_SOCKET: u8 = 8;
    /// Named pipe / FIFO (`-p`).
    pub const IS_FIFO: u8 = 9;
    /// Block device (`-b`).
    pub const IS_BLOCK_DEV: u8 = 10;
    /// Character device (`-c`).
    pub const IS_CHAR_DEV: u8 = 11;
}

/// Redirect op types for `Redirect(fd, op)`
pub mod redirect_op {
    /// `> file` — truncate-then-write.
    pub const WRITE: u8 = 0;
    /// `>> file` — open for append.
    pub const APPEND: u8 = 1;
    /// `< file` — open for read.
    pub const READ: u8 = 2;
    /// `<> file` — open for read+write.
    pub const READ_WRITE: u8 = 3;
    /// `>| file` — forced truncate even under `noclobber`.
    pub const CLOBBER: u8 = 4;
    /// `<& N` — duplicate input fd N.
    pub const DUP_READ: u8 = 5;
    /// `>& N` — duplicate output fd N.
    pub const DUP_WRITE: u8 = 6;
    /// `&> file` — redirect both stdout and stderr (truncate).
    pub const WRITE_BOTH: u8 = 7;
    /// `&>> file` — redirect both stdout and stderr (append).
    pub const APPEND_BOTH: u8 = 8;
}

/// Parameter expansion modifier types for `ExpandParam(u8)`
pub mod param_mod {
    /// `${var:-default}` — substitute default if unset/empty.
    pub const DEFAULT: u8 = 0;
    /// `${var:=default}` — assign default if unset/empty.
    pub const ASSIGN: u8 = 1;
    /// `${var:?error}` — error if unset/empty.
    pub const ERROR: u8 = 2;
    /// `${var:+alternate}` — substitute alternate when set.
    pub const ALTERNATE: u8 = 3;
    /// `${#var}` — string length.
    pub const LENGTH: u8 = 4;
    /// `${var#pat}` — strip shortest matching prefix.
    pub const STRIP_SHORT: u8 = 5;
    /// `${var##pat}` — strip longest matching prefix.
    pub const STRIP_LONG: u8 = 6;
    /// `${var%pat}` — strip shortest matching suffix.
    pub const RSTRIP_SHORT: u8 = 7;
    /// `${var%%pat}` — strip longest matching suffix.
    pub const RSTRIP_LONG: u8 = 8;
    /// `${var/pat/rep}` — substitute first match.
    pub const SUBST_FIRST: u8 = 9;
    /// `${var//pat/rep}` — substitute every match.
    pub const SUBST_ALL: u8 = 10;
    /// `${var^^}` — uppercase every char.
    pub const UPPER: u8 = 11;
    /// `${var,,}` — lowercase every char.
    pub const LOWER: u8 = 12;
    /// `${var^}` — uppercase first char.
    pub const UPPER_FIRST: u8 = 13;
    /// `${var,}` — lowercase first char.
    pub const LOWER_FIRST: u8 = 14;
    /// `${!var}` — indirect expand (use value of `var` as name).
    pub const INDIRECT: u8 = 15;
    /// `${!arr[@]}` — array key/index list.
    pub const KEYS: u8 = 16;
    /// `${var:off:len}` — substring slice.
    pub const SLICE: u8 = 17;
}

impl Hash for Op {
    fn hash<H: Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            Op::LoadInt(n) => n.hash(state),
            Op::LoadFloat(f) => f.to_bits().hash(state),
            Op::LoadConst(idx) => idx.hash(state),
            Op::GetVar(idx)
            | Op::SetVar(idx)
            | Op::DeclareVar(idx)
            | Op::GetSlot(idx)
            | Op::SetSlot(idx)
            | Op::SlotArrayGet(idx)
            | Op::SlotArraySet(idx)
            | Op::GetArray(idx)
            | Op::SetArray(idx)
            | Op::DeclareArray(idx)
            | Op::ArrayGet(idx)
            | Op::ArraySet(idx)
            | Op::ArrayPush(idx)
            | Op::ArrayPop(idx)
            | Op::ArrayShift(idx)
            | Op::ArrayLen(idx)
            | Op::MakeArray(idx)
            | Op::GetHash(idx)
            | Op::SetHash(idx)
            | Op::DeclareHash(idx)
            | Op::HashGet(idx)
            | Op::HashSet(idx)
            | Op::HashDelete(idx)
            | Op::HashExists(idx)
            | Op::HashKeys(idx)
            | Op::HashValues(idx)
            | Op::MakeHash(idx)
            | Op::PreIncSlot(idx)
            | Op::PreIncSlotVoid(idx)
            | Op::PreDecSlot(idx)
            | Op::PostIncSlot(idx)
            | Op::PostDecSlot(idx)
            | Op::HereDoc(idx)
            | Op::CmdSubst(idx)
            | Op::ProcessSubIn(idx)
            | Op::ProcessSubOut(idx)
            | Op::TrapSet(idx)
            | Op::MapBlock(idx)
            | Op::GrepBlock(idx)
            | Op::SortBlock(idx)
            | Op::ForEachBlock(idx)
            | Op::AwkSpecialGet(idx)
            | Op::AwkSpecialSet(idx)
            | Op::AwkArrayGet(idx)
            | Op::AwkArraySet(idx)
            | Op::AwkArrayExists(idx)
            | Op::AwkArrayDelete(idx)
            | Op::AwkArrayClear(idx)
            | Op::AwkArrayLen(idx) => idx.hash(state),
            Op::Jump(t)
            | Op::JumpIfTrue(t)
            | Op::JumpIfFalse(t)
            | Op::JumpIfTrueKeep(t)
            | Op::JumpIfFalseKeep(t) => t.hash(state),
            Op::Call(name, argc) => {
                name.hash(state);
                argc.hash(state);
            }
            Op::CallBuiltin(id, argc) => {
                id.hash(state);
                argc.hash(state);
            }
            Op::CallFunction(name, argc) => {
                name.hash(state);
                argc.hash(state);
            }
            Op::WithRedirectsBegin(n) => n.hash(state),
            Op::Extended(id, arg) => {
                id.hash(state);
                arg.hash(state);
            }
            Op::ExtendedWide(id, payload) => {
                id.hash(state);
                payload.hash(state);
            }
            Op::Print(n) | Op::PrintLn(n) | Op::Exec(n) | Op::ExecBg(n) | Op::PipelineBegin(n) => {
                n.hash(state)
            }
            Op::Redirect(fd, op) => {
                fd.hash(state);
                op.hash(state);
            }
            Op::TestFile(t) | Op::ExpandParam(t) => t.hash(state),
            // AWK ops with a u8 payload (argc / getline source kind).
            Op::AwkPrint(n)
            | Op::AwkPrintf(n)
            | Op::AwkSprintf(n)
            | Op::AwkGetline(n)
            | Op::AwkLength(n)
            | Op::AwkSubstr(n)
            | Op::AwkSplit(n)
            | Op::AwkSub(n)
            | Op::AwkGsub(n)
            | Op::AwkAnd(n)
            | Op::AwkOr(n)
            | Op::AwkXor(n)
            | Op::AwkSrand(n)
            | Op::AwkStrftime(n)
            | Op::AwkMktime(n)
            | Op::AwkGensub(n)
            | Op::AwkSignal(n) => n.hash(state),
            Op::SlotLtIntJumpIfFalse(slot, limit, target) => {
                slot.hash(state);
                limit.hash(state);
                target.hash(state);
            }
            Op::SlotIncLtIntJumpBack(slot, limit, target) => {
                slot.hash(state);
                limit.hash(state);
                target.hash(state);
            }
            Op::AccumSumLoop(sum, i, limit) => {
                sum.hash(state);
                i.hash(state);
                limit.hash(state);
            }
            Op::ConcatConstLoop(c, s, i, limit) => {
                c.hash(state);
                s.hash(state);
                i.hash(state);
                limit.hash(state);
            }
            Op::PushIntRangeLoop(arr, i, limit) => {
                arr.hash(state);
                i.hash(state);
                limit.hash(state);
            }
            Op::AddAssignSlotVoid(a, b) => {
                a.hash(state);
                b.hash(state);
            }
            // Nullary ops — discriminant alone is sufficient
            Op::Nop
            | Op::LoadTrue
            | Op::LoadFalse
            | Op::LoadUndef
            | Op::Pop
            | Op::Dup
            | Op::Dup2
            | Op::Swap
            | Op::Rot
            | Op::Add
            | Op::Sub
            | Op::Mul
            | Op::Div
            | Op::Mod
            | Op::Pow
            | Op::PowFloat
            | Op::SqrtFloat
            | Op::SinFloat
            | Op::CosFloat
            | Op::ExpFloat
            | Op::Atan2Float
            | Op::LogFloat
            | Op::AbsFloat
            | Op::TruncInt
            | Op::CeilFloat
            | Op::FloorFloat
            | Op::TruncFloat
            | Op::RoundFloat
            | Op::TanFloat
            | Op::AsinFloat
            | Op::AcosFloat
            | Op::AtanFloat
            | Op::SinhFloat
            | Op::CoshFloat
            | Op::TanhFloat
            | Op::Log2Float
            | Op::Log10Float
            | Op::AbsInt
            | Op::GcdInt
            | Op::LcmInt
            | Op::TimeInt
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
            | Op::Spaceship
            | Op::StrEq
            | Op::StrNe
            | Op::StrLt
            | Op::StrGt
            | Op::StrLe
            | Op::StrGe
            | Op::StrCmp
            | Op::LogNot
            | Op::LogAnd
            | Op::LogOr
            | Op::BitAnd
            | Op::BitOr
            | Op::BitXor
            | Op::BitNot
            | Op::Shl
            | Op::Shr
            | Op::Return
            | Op::ReturnValue
            | Op::PushFrame
            | Op::PopFrame
            | Op::ReadLine
            | Op::Range
            | Op::RangeStep
            | Op::SortDefault
            | Op::SetStatus
            | Op::GetStatus
            | Op::PipelineStage
            | Op::PipelineEnd
            | Op::HereString
            | Op::SubshellBegin
            | Op::SubshellEnd
            | Op::Glob
            | Op::GlobRecursive
            | Op::TrapCheck
            | Op::WordSplit
            | Op::BraceExpand
            | Op::TildeExpand
            | Op::StrMatch
            | Op::RegexMatch
            | Op::WithRedirectsEnd
            | Op::AwkFieldGet
            | Op::AwkFieldSet
            | Op::AwkNf
            | Op::AwkSetRecord
            | Op::AwkIndex
            | Op::AwkMatch
            | Op::AwkToLower
            | Op::AwkToUpper
            | Op::AwkInt
            | Op::AwkSqrt
            | Op::AwkSin
            | Op::AwkCos
            | Op::AwkExp
            | Op::AwkLog
            | Op::AwkAtan2
            | Op::AwkDiv
            | Op::AwkMod
            | Op::AwkDivJit
            | Op::AwkModJit
            | Op::AwkSqrtJit
            | Op::AwkLogJit
            | Op::AwkLshiftJit
            | Op::AwkRshiftJit
            | Op::AwkComplJit
            | Op::AwkGetFieldNum(_)
            | Op::AwkCompl
            | Op::AwkLshift
            | Op::AwkRshift
            | Op::AwkStrtonum
            | Op::AwkSystime
            | Op::AwkRand
            | Op::AwkOrd
            | Op::AwkChr
            | Op::AwkMkbool
            | Op::AwkIntdiv
            | Op::AwkIntdiv0 => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_op_size() {
        // Ops should be reasonably small for cache-friendly dispatch
        assert!(
            std::mem::size_of::<Op>() <= 24,
            "Op too large: {} bytes",
            std::mem::size_of::<Op>()
        );
    }

    #[test]
    fn equal_ops_hash_equal() {
        use std::collections::hash_map::DefaultHasher;
        let h = |op: &Op| {
            let mut hs = DefaultHasher::new();
            op.hash(&mut hs);
            hs.finish()
        };
        assert_eq!(h(&Op::LoadInt(42)), h(&Op::LoadInt(42)));
        assert_eq!(h(&Op::Jump(7)), h(&Op::Jump(7)));
        assert_eq!(h(&Op::Add), h(&Op::Add));
    }

    #[test]
    fn different_ops_typically_hash_differently() {
        use std::collections::hash_map::DefaultHasher;
        let h = |op: &Op| {
            let mut hs = DefaultHasher::new();
            op.hash(&mut hs);
            hs.finish()
        };
        assert_ne!(h(&Op::LoadInt(1)), h(&Op::LoadInt(2)));
        assert_ne!(h(&Op::Add), h(&Op::Sub));
        assert_ne!(h(&Op::Jump(0)), h(&Op::JumpIfTrue(0)));
    }

    #[test]
    fn float_load_hash_uses_bit_pattern() {
        // f64 must hash via bits — NaN, -0.0 etc. need to be hashable.
        use std::collections::hash_map::DefaultHasher;
        let h = |op: &Op| {
            let mut hs = DefaultHasher::new();
            op.hash(&mut hs);
            hs.finish()
        };
        let a = Op::LoadFloat(f64::NAN);
        let b = Op::LoadFloat(f64::NAN);
        // Same bit pattern → equal hash.
        assert_eq!(h(&a), h(&b));
        // +0.0 and -0.0 are == under PartialEq but have different bits;
        // their hashes will differ — verify the impl is bit-based.
        assert_ne!(h(&Op::LoadFloat(0.0)), h(&Op::LoadFloat(-0.0)));
    }

    #[test]
    fn partialeq_works_for_payloaded_ops() {
        assert_eq!(Op::LoadInt(1), Op::LoadInt(1));
        assert_ne!(Op::LoadInt(1), Op::LoadInt(2));
        assert_eq!(Op::Jump(5), Op::Jump(5));
        assert_ne!(Op::Jump(5), Op::Jump(6));
    }

    #[test]
    fn op_clone_is_value_equal() {
        let a = Op::LoadInt(123);
        let b = a.clone();
        assert_eq!(a, b);
    }

    #[test]
    fn op_serde_roundtrip() {
        // Verify a representative selection of ops survive serde JSON roundtrip.
        let cases = vec![
            Op::Nop,
            Op::LoadInt(-5),
            Op::LoadFloat(1.5),
            Op::Add,
            Op::Jump(42),
            Op::GetSlot(7),
        ];
        for op in cases {
            let s = serde_json::to_string(&op).unwrap();
            let back: Op = serde_json::from_str(&s).unwrap();
            assert_eq!(op, back);
        }
    }

    use std::collections::hash_map::DefaultHasher;

    fn hash_of(op: &Op) -> u64 {
        let mut h = DefaultHasher::new();
        op.hash(&mut h);
        h.finish()
    }

    // ─── Hash impl coverage: every variant arm must produce consistent
    //     hashes and equal-ops-hash-equal regardless of payload class ──

    #[test]
    fn nullary_arithmetic_ops_hash_consistently() {
        for op in [Op::Add, Op::Sub, Op::Mul, Op::Div, Op::Mod, Op::Pow] {
            assert_eq!(hash_of(&op), hash_of(&op.clone()));
        }
    }

    #[test]
    fn distinct_nullary_ops_hash_differently() {
        // Discriminant alone differentiates these. Collisions are theoretically
        // allowed by Hash but DefaultHasher is unlikely to collide on so few.
        let h_add = hash_of(&Op::Add);
        let h_sub = hash_of(&Op::Sub);
        let h_mul = hash_of(&Op::Mul);
        assert_ne!(h_add, h_sub);
        assert_ne!(h_add, h_mul);
        assert_ne!(h_sub, h_mul);
    }

    #[test]
    fn same_idx_in_different_u16_arms_does_not_collide_on_discriminant() {
        // GetVar(5) and SetVar(5) share the payload value but differ in
        // discriminant — must hash differently.
        let h_get = hash_of(&Op::GetVar(5));
        let h_set = hash_of(&Op::SetVar(5));
        assert_ne!(h_get, h_set);
    }

    #[test]
    fn jump_targets_hash_independent_of_call_targets() {
        // Jump(7) and Call(7, 0) share the literal value 7 but should never
        // collide because discriminants differ.
        let h_jump = hash_of(&Op::Jump(7));
        let h_call = hash_of(&Op::Call(7, 0));
        assert_ne!(h_jump, h_call);
    }

    #[test]
    fn call_argc_changes_hash() {
        let h_zero_arg = hash_of(&Op::Call(10, 0));
        let h_two_args = hash_of(&Op::Call(10, 2));
        assert_ne!(h_zero_arg, h_two_args);
    }

    #[test]
    fn float_load_distinct_values_hash_differently() {
        // f64 → to_bits → hash. 1.0 and 2.0 have distinct bit patterns.
        let h_one = hash_of(&Op::LoadFloat(1.0));
        let h_two = hash_of(&Op::LoadFloat(2.0));
        assert_ne!(h_one, h_two);
    }

    #[test]
    fn float_load_neg_zero_and_pos_zero_hash_differently() {
        // 0.0 and -0.0 are == under PartialEq but have different bit patterns,
        // so the bit-based Hash impl produces different hashes. This is consistent
        // with serde_json roundtrip semantics for the constant pool.
        let h_pos = hash_of(&Op::LoadFloat(0.0));
        let h_neg = hash_of(&Op::LoadFloat(-0.0));
        assert_ne!(h_pos, h_neg, "+0.0 and -0.0 have distinct bit patterns");
    }

    #[test]
    fn redirect_op_uses_both_fd_and_op_in_hash() {
        let a = Op::Redirect(0, 1);
        let b = Op::Redirect(0, 2);
        let c = Op::Redirect(1, 1);
        assert_ne!(hash_of(&a), hash_of(&b), "second field changes hash");
        assert_ne!(hash_of(&a), hash_of(&c), "first field changes hash");
    }

    #[test]
    fn extended_uses_both_id_and_arg_in_hash() {
        let a = Op::Extended(1, 2);
        let b = Op::Extended(1, 3);
        let c = Op::Extended(2, 2);
        assert_ne!(hash_of(&a), hash_of(&b));
        assert_ne!(hash_of(&a), hash_of(&c));
    }

    #[test]
    fn three_field_loop_ops_use_all_fields() {
        let base = Op::SlotLtIntJumpIfFalse(1, 10, 100);
        let diff_slot = Op::SlotLtIntJumpIfFalse(2, 10, 100);
        let diff_limit = Op::SlotLtIntJumpIfFalse(1, 99, 100);
        let diff_target = Op::SlotLtIntJumpIfFalse(1, 10, 200);
        assert_ne!(hash_of(&base), hash_of(&diff_slot));
        assert_ne!(hash_of(&base), hash_of(&diff_limit));
        assert_ne!(hash_of(&base), hash_of(&diff_target));
    }

    #[test]
    fn equal_loadint_payloads_hash_equal() {
        // Same payload → same hash, no matter how the value was constructed.
        let a = Op::LoadInt(-42);
        let b = Op::LoadInt(-42);
        assert_eq!(hash_of(&a), hash_of(&b));
    }

    #[test]
    fn pop_dup_swap_rot_are_each_unique() {
        // Common stack ops — discriminant alone differentiates.
        let pop = hash_of(&Op::Pop);
        let dup = hash_of(&Op::Dup);
        let swap = hash_of(&Op::Swap);
        let rot = hash_of(&Op::Rot);
        let set: std::collections::HashSet<_> = [pop, dup, swap, rot].iter().copied().collect();
        assert_eq!(set.len(), 4, "all four nullary stack ops are distinct");
    }

    // ─── Serde round-trip extension: more payload-carrying ops ────────

    #[test]
    fn serde_roundtrip_payload_ops() {
        let cases = vec![
            Op::Call(100, 3),
            Op::CallBuiltin(0, 1),
            Op::Redirect(2, 5),
            Op::Extended(7, 9),
            Op::SlotLtIntJumpIfFalse(1, 10, 200),
        ];
        for op in cases {
            let s = serde_json::to_string(&op).expect("serialize");
            let back: Op = serde_json::from_str(&s).expect("deserialize");
            assert_eq!(op, back);
        }
    }

    #[test]
    fn serde_roundtrip_float_special_values() {
        // Special-case floats: NaN doesn't round-trip via PartialEq, so
        // only check finite ones.
        for f in [0.0, -0.0, 1.5, -1.5, f64::MIN, f64::MAX] {
            let op = Op::LoadFloat(f);
            let s = serde_json::to_string(&op).expect("ser");
            let back: Op = serde_json::from_str(&s).expect("de");
            assert_eq!(op, back, "roundtrip {f}");
        }
    }

    // ─── tests for the `block` constants module ───────────────────────

    #[test]
    fn param_mod_constants_are_unique_and_within_u8() {
        // Each ExpandParam modifier maps to a distinct u8 op-code; verify no
        // collisions on the 18-value table.
        let names = [
            param_mod::DEFAULT,
            param_mod::ASSIGN,
            param_mod::ERROR,
            param_mod::ALTERNATE,
            param_mod::LENGTH,
            param_mod::STRIP_SHORT,
            param_mod::STRIP_LONG,
            param_mod::RSTRIP_SHORT,
            param_mod::RSTRIP_LONG,
            param_mod::SUBST_FIRST,
            param_mod::SUBST_ALL,
            param_mod::UPPER,
            param_mod::LOWER,
            param_mod::UPPER_FIRST,
            param_mod::LOWER_FIRST,
            param_mod::INDIRECT,
            param_mod::KEYS,
            param_mod::SLICE,
        ];
        let set: std::collections::HashSet<_> = names.iter().copied().collect();
        assert_eq!(set.len(), names.len(), "param_mod constants must be unique");
    }
}
