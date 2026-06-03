//! AWK operation IDs for `Op::ExtendedWide(id, payload)` dispatch.
//!
//! These IDs are used by AWK frontends (awkrs) to emit bytecode that drives
//! AWK-specific semantics — field access, record I/O, `print`/`printf`, the
//! string builtins (`sub`/`gsub`/`split`/`sprintf`/`match`/...), and special
//! variables (`NF`/`NR`/`FS`/`OFS`/...) — through a registered [`AwkHost`].
//!
//! Unlike the universal arithmetic/array ops (which the VM executes natively),
//! AWK ops cannot use `fusevm::Value` faithfully: AWK has POSIX numeric-string
//! duality, `CONVFMT`/`OFMT` formatting, field/`$0` coupling, and `SUBSEP`
//! associative arrays. Those semantics live host-side. The VM dispatches the
//! AWK op range to `vm.awk_host` (see [`crate::vm::VM::set_awk_host`]); operands
//! that aren't carried in the `payload` come from the value stack.
//!
//! [`AwkHost`]: crate::awk_host::AwkHost
//!
//! # Encoding
//!
//! AWK ops are emitted as `Op::ExtendedWide(id, payload)` where `id` is one of
//! the constants below (all `>= AWK_OP_BASE`). `payload` carries the inline
//! integer operand for ops that need one (field index for field ops, argument
//! count for variadic ops, name-pool index for special-variable ops). Value
//! operands are pushed on the stack in source order before the op.
//!
//! ```
//! use fusevm::{ChunkBuilder, Op};
//! use fusevm::awk_builtins::*;
//!
//! // `$1` — push the field index, then read the field.
//! let mut b = ChunkBuilder::new();
//! b.emit(Op::LoadInt(1), 1);
//! b.emit(Op::ExtendedWide(AWK_FIELD_GET, 0), 1);
//! ```

/// First AWK op ID. The VM routes `ExtendedWide` ops with `id >= AWK_OP_BASE`
/// (and `< AWK_OP_END`) to the registered [`AwkHost`]. Chosen high to leave the
/// low `Extended`/`ExtendedWide` id space free for generic frontend handlers
/// (stryke, etc.) that use `set_extension_wide_handler`.
///
/// [`AwkHost`]: crate::awk_host::AwkHost
pub const AWK_OP_BASE: u16 = 40_000;
/// One past the last AWK op ID (exclusive upper bound of the reserved range).
pub const AWK_OP_END: u16 = 41_000;

/// True when `id` falls in the reserved AWK op range.
#[inline]
pub fn is_awk_op(id: u16) -> bool {
    (AWK_OP_BASE..AWK_OP_END).contains(&id)
}

// ═══════════════════════════════════════════════════════════════════════════
// Fields & record  (payload = field index for the *_FIELD_* ops)
// ═══════════════════════════════════════════════════════════════════════════

/// `$i` — read field `i`. `i` is popped from the stack; pushes the field value.
pub const AWK_FIELD_GET: u16 = AWK_OP_BASE;
/// `$i = v` — assign field `i`. Stack: `[value, index]`; rebuilds `$0`/`NF`.
pub const AWK_FIELD_SET: u16 = AWK_OP_BASE + 1;
/// `NF` — push the current field count.
pub const AWK_NF: u16 = AWK_OP_BASE + 2;
/// `$0 = v` — replace the whole record. Stack: `[value]`; resplits fields.
pub const AWK_SET_RECORD: u16 = AWK_OP_BASE + 3;

// ═══════════════════════════════════════════════════════════════════════════
// Special variables  (payload = name-pool index of the special var name)
// ═══════════════════════════════════════════════════════════════════════════

/// Read a special AWK variable by name-pool index (`FS`/`OFS`/`ORS`/`RS`/
/// `NR`/`FNR`/`SUBSEP`/`RSTART`/`RLENGTH`/`FILENAME`/`CONVFMT`/`OFMT`/...).
pub const AWK_SPECIAL_GET: u16 = AWK_OP_BASE + 4;
/// Assign a special AWK variable by name-pool index. Stack: `[value]`.
pub const AWK_SPECIAL_SET: u16 = AWK_OP_BASE + 5;

// ═══════════════════════════════════════════════════════════════════════════
// Output  (payload = argument count)
// ═══════════════════════════════════════════════════════════════════════════

/// `print a, b, ...` — pops `payload` args, joins with `OFS`, ends with `ORS`.
pub const AWK_PRINT: u16 = AWK_OP_BASE + 16;
/// `printf fmt, a, ...` — pops `payload` args (last popped is the format).
pub const AWK_PRINTF: u16 = AWK_OP_BASE + 17;
/// `sprintf(fmt, a, ...)` — like `AWK_PRINTF` but pushes the formatted string.
pub const AWK_SPRINTF: u16 = AWK_OP_BASE + 18;

// ═══════════════════════════════════════════════════════════════════════════
// Input  (payload = getline source kind; see `getline_source`)
// ═══════════════════════════════════════════════════════════════════════════

/// `getline` family. `payload` encodes the source (plain/var/file/cmd). Stack
/// holds the source operand (file/cmd string) when applicable; pushes status
/// (1 = record read, 0 = EOF, -1 = error).
pub const AWK_GETLINE: u16 = AWK_OP_BASE + 24;

// ═══════════════════════════════════════════════════════════════════════════
// String builtins  (payload = argument count where variadic)
// ═══════════════════════════════════════════════════════════════════════════

/// `length` / `length(x)` — pops `payload` args (0 ⇒ `length($0)`).
pub const AWK_LENGTH: u16 = AWK_OP_BASE + 32;
/// `substr(s, m [, n])` — pops `payload` args (2 or 3).
pub const AWK_SUBSTR: u16 = AWK_OP_BASE + 33;
/// `index(s, t)` — stack `[s, t]`; pushes 1-based position or 0.
pub const AWK_INDEX: u16 = AWK_OP_BASE + 34;
/// `split(s, arr [, fs])` — pops `payload` args; pushes field count.
pub const AWK_SPLIT: u16 = AWK_OP_BASE + 35;
/// `sub(re, repl [, target])` — pops `payload` args; pushes substitution count.
pub const AWK_SUB: u16 = AWK_OP_BASE + 36;
/// `gsub(re, repl [, target])` — pops `payload` args; pushes substitution count.
pub const AWK_GSUB: u16 = AWK_OP_BASE + 37;
/// `match(s, re)` — stack `[s, re]`; sets `RSTART`/`RLENGTH`, pushes `RSTART`.
pub const AWK_MATCH: u16 = AWK_OP_BASE + 38;
/// `tolower(s)` — stack `[s]`.
pub const AWK_TOLOWER: u16 = AWK_OP_BASE + 39;
/// `toupper(s)` — stack `[s]`.
pub const AWK_TOUPPER: u16 = AWK_OP_BASE + 40;

/// AWK control-flow signal codes carried by `Op::AwkSignal(code)`. The op halts
/// the chunk; the frontend driver reads `VM::awk_signal()` and maps the code to
/// its own control-flow (next record / next file / exit).
pub mod signal {
    /// `next` — skip remaining rules for the current record.
    pub const NEXT: u8 = 0;
    /// `nextfile` — skip the rest of the current input file.
    pub const NEXTFILE: u8 = 1;
    /// `exit [code]` — stop main-input processing and run `END`.
    pub const EXIT: u8 = 2;
}

// ═══════════════════════════════════════════════════════════════════════════
// Numeric builtins  (no payload; pure f64 math, host-independent)
// ═══════════════════════════════════════════════════════════════════════════

/// `int(x)` — truncate toward zero. Stack `[x]`.
pub const AWK_INT: u16 = AWK_OP_BASE + 41;
/// `sqrt(x)` — square root. Stack `[x]`.
pub const AWK_SQRT: u16 = AWK_OP_BASE + 42;
/// `sin(x)` — sine (radians). Stack `[x]`.
pub const AWK_SIN: u16 = AWK_OP_BASE + 43;
/// `cos(x)` — cosine (radians). Stack `[x]`.
pub const AWK_COS: u16 = AWK_OP_BASE + 44;
/// `exp(x)` — e^x. Stack `[x]`.
pub const AWK_EXP: u16 = AWK_OP_BASE + 45;
/// `log(x)` — natural log. Stack `[x]`.
pub const AWK_LOG: u16 = AWK_OP_BASE + 46;
/// `atan2(y, x)` — arctangent of `y/x`. Stack `[y, x]`.
pub const AWK_ATAN2: u16 = AWK_OP_BASE + 47;

// ═══════════════════════════════════════════════════════════════════════════
// Associative arrays (AWK string-key / SUBSEP semantics)
// (payload = name-pool index of the array variable)
// ═══════════════════════════════════════════════════════════════════════════

/// `arr[k]` — stack `[key]`; pushes the element (auto-vivifies to "" per POSIX).
pub const AWK_ARRAY_GET: u16 = AWK_OP_BASE + 48;
/// `arr[k] = v` — stack `[value, key]`.
pub const AWK_ARRAY_SET: u16 = AWK_OP_BASE + 49;
/// `(k in arr)` — stack `[key]`; pushes Bool.
pub const AWK_ARRAY_EXISTS: u16 = AWK_OP_BASE + 50;
/// `delete arr[k]` — stack `[key]`.
pub const AWK_ARRAY_DELETE: u16 = AWK_OP_BASE + 51;
/// `delete arr` — clear the whole array (no stack operand).
pub const AWK_ARRAY_CLEAR: u16 = AWK_OP_BASE + 52;
/// `length(arr)` — push the element count.
pub const AWK_ARRAY_LEN: u16 = AWK_OP_BASE + 53;

// ═══════════════════════════════════════════════════════════════════════════
// Bitwise builtins (gawk extensions; pure integer math, host-independent)
// Operands are truncated to integers (`n.trunc() as i64 as u64`); results are
// the u64 bit-pattern reinterpreted as a signed integer.
// ═══════════════════════════════════════════════════════════════════════════

/// `and(v1, v2, ...)` — bitwise AND of ≥2 args. Payload = argument count.
pub const AWK_AND: u16 = AWK_OP_BASE + 54;
/// `or(v1, v2, ...)` — bitwise OR of ≥2 args. Payload = argument count.
pub const AWK_OR: u16 = AWK_OP_BASE + 55;
/// `xor(v1, v2, ...)` — bitwise XOR of ≥2 args. Payload = argument count.
pub const AWK_XOR: u16 = AWK_OP_BASE + 56;
/// `compl(v)` — bitwise complement. Stack `[v]`.
pub const AWK_COMPL: u16 = AWK_OP_BASE + 57;
/// `lshift(v, n)` — left shift `v` by `n & 0x3f` bits. Stack `[v, n]`.
pub const AWK_LSHIFT: u16 = AWK_OP_BASE + 58;
/// `rshift(v, n)` — right shift `v` by `n & 0x3f` bits. Stack `[v, n]`.
pub const AWK_RSHIFT: u16 = AWK_OP_BASE + 59;

// ═══════════════════════════════════════════════════════════════════════════
// Conversion builtins (gawk extensions; pure string→number parse, host-free)
// ═══════════════════════════════════════════════════════════════════════════

/// `strtonum(s)` — parse `0x…` hex, `0…` octal, else longest decimal/float
/// prefix. Stack `[s]`; returns a number. Host-independent.
pub const AWK_STRTONUM: u16 = AWK_OP_BASE + 60;

// ═══════════════════════════════════════════════════════════════════════════
// Time builtins (gawk extensions; only the dep-free clock read is native)
// ═══════════════════════════════════════════════════════════════════════════

/// `systime()` — seconds since the Unix epoch. Nullary; pushes a number.
/// Host-independent (reads the system clock via `std::time`, no deps).
pub const AWK_SYSTIME: u16 = AWK_OP_BASE + 61;

// ═══════════════════════════════════════════════════════════════════════════
// PRNG builtins (POSIX/gawk; glibc LCG over a VM-owned seed, host-free)
// ═══════════════════════════════════════════════════════════════════════════

/// `rand()` — next pseudo-random number in `[0, 1)`. Nullary; pushes a number.
/// Advances the VM-owned LCG seed. Host-independent.
pub const AWK_RAND: u16 = AWK_OP_BASE + 62;
/// `srand([x])` — reseed the PRNG, returning the previous seed (low 32 bits).
/// Payload = argument count (0 → seed from clock, 1 → seed from popped value).
pub const AWK_SRAND: u16 = AWK_OP_BASE + 63;

// ═══════════════════════════════════════════════════════════════════════════
// Date/time formatting builtins (gawk extensions; chrono-backed, host-free)
// ═══════════════════════════════════════════════════════════════════════════

/// `strftime([fmt [, ts [, utc]]])` — format a timestamp. Payload = argument
/// count (0..=3). Pushes a string. Host-independent (chrono + system tz).
pub const AWK_STRFTIME: u16 = AWK_OP_BASE + 64;
/// `mktime(datespec [, utc])` — `"YYYY MM DD HH MM SS"` → epoch seconds (or -1).
/// Payload = argument count (1..=2). Pushes a number. Host-independent.
pub const AWK_MKTIME: u16 = AWK_OP_BASE + 65;

// ═══════════════════════════════════════════════════════════════════════════
// Character / scalar builtins (gawk extensions; pure on `Value`, host-free)
// ═══════════════════════════════════════════════════════════════════════════

/// `ord(s)` — Unicode scalar value of the first character of `s` (0 if empty).
/// Nullary payload-free; pops one argument, pushes a number.
pub const AWK_ORD: u16 = AWK_OP_BASE + 66;
/// `chr(n)` — single-character string for codepoint `n` (empty if invalid).
/// Pops one argument, pushes a string.
pub const AWK_CHR: u16 = AWK_OP_BASE + 67;
/// `mkbool(x)` — `1` if `x` is truthy, else `0`. Pops one argument, pushes a number.
pub const AWK_MKBOOL: u16 = AWK_OP_BASE + 68;
/// `intdiv(a, b)` — integer (truncating) quotient `a / b` as a number.
/// Non-bignum path only; division by zero pushes `Undef` (a host may raise the
/// gawk fatal). Pops two arguments, pushes a number.
pub const AWK_INTDIV: u16 = AWK_OP_BASE + 69;
/// `intdiv0(a, b)` — like `intdiv`, but division by zero yields `0` (the "safe"
/// gawk variant; never errors). Pops two arguments, pushes a number.
pub const AWK_INTDIV0: u16 = AWK_OP_BASE + 70;

// ═══════════════════════════════════════════════════════════════════════════
// Regex builtin completing the family (host-bound — needs the regex cache and
// the `IGNORECASE` runtime variable, like `sub`/`gsub`/`match`/`split`)
// ═══════════════════════════════════════════════════════════════════════════

/// `gensub(re, repl, how [, target])` — return `target` (or `$0` when omitted)
/// with `re` matches replaced per `how` (`"g"`/`"G"` = all, positive integer =
/// that occurrence), expanding `&` and `\1`..`\9` backrefs in `repl`. Payload =
/// argument count (3..=4). Pushes the result string. Host-bound: regex
/// compilation honors `IGNORECASE` and the 3-arg form reads `$0`.
pub const AWK_GENSUB: u16 = AWK_OP_BASE + 71;

/// `getline` source kinds carried in the `AWK_GETLINE` payload.
pub mod getline_source {
    /// `getline` — next record from the main input into `$0` (updates NF/NR/FNR).
    pub const MAIN: usize = 0;
    /// `getline var` — next main-input record into a variable (updates NR/FNR).
    pub const MAIN_VAR: usize = 1;
    /// `getline < file` — next record of `file` into `$0`.
    pub const FILE: usize = 2;
    /// `getline var < file` — next record of `file` into a variable.
    pub const FILE_VAR: usize = 3;
    /// `cmd | getline` — next line of `cmd` output into `$0`.
    pub const CMD: usize = 4;
    /// `cmd | getline var` — next line of `cmd` output into a variable.
    pub const CMD_VAR: usize = 5;
}
