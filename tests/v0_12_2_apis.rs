//! Tests pinning the v0.12.2 API additions:
//!   - `ShellHost::subshell_end -> Option<i32>` (Some(N) updates vm.last_status)
//!   - `VM::request_halt()` (host-callable mid-execution halt)
//!
//! These behaviors are documented in fusevm v0.12.2 release notes and
//! zshrs's TODO.md #6. Until now no test pinned the actual contract —
//! only that the trait sig type-checks. A regression that no-ops
//! Some(status) or that ignores request_halt would slip silently.

use fusevm::{ChunkBuilder, Op, ShellHost, VMResult, Value, VM};
use std::sync::atomic::{AtomicU32, Ordering};

// ─── ShellHost::subshell_end Option<i32> contract ──────────────────────

struct StatusReturningHost(i32);

impl ShellHost for StatusReturningHost {
    fn subshell_end(&mut self) -> Option<i32> {
        Some(self.0)
    }
}

#[test]
fn subshell_end_some_status_updates_vm_last_status() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    let mut vm = VM::new(b.build());
    vm.last_status = 0;
    vm.set_shell_host(Box::new(StatusReturningHost(42)));
    let _ = vm.run();
    assert_eq!(
        vm.last_status, 42,
        "Some(42) returned from subshell_end MUST propagate into vm.last_status"
    );
}

#[test]
fn subshell_end_some_zero_overwrites_nonzero_last_status() {
    // The zero case is the interesting one — early implementations
    // sometimes special-case 0 as "no change" defeating the purpose.
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    let mut vm = VM::new(b.build());
    vm.last_status = 99;
    vm.set_shell_host(Box::new(StatusReturningHost(0)));
    let _ = vm.run();
    assert_eq!(
        vm.last_status, 0,
        "Some(0) MUST overwrite prior last_status — Z does not mean 'unchanged'"
    );
}

struct NoneHost;
impl ShellHost for NoneHost {
    fn subshell_end(&mut self) -> Option<i32> {
        None
    }
}

#[test]
fn subshell_end_none_leaves_last_status_untouched() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    let mut vm = VM::new(b.build());
    vm.last_status = 99;
    vm.set_shell_host(Box::new(NoneHost));
    let _ = vm.run();
    assert_eq!(
        vm.last_status, 99,
        "None MUST leave prior last_status untouched — the no-update contract"
    );
}

#[test]
fn subshell_end_default_trait_impl_returns_none() {
    // The trait's default subshell_end returns None — a host that
    // doesn't override gets prior behavior (no last_status writes).
    struct BareHost;
    impl ShellHost for BareHost {} // takes the default impl
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    let mut vm = VM::new(b.build());
    vm.last_status = 7;
    vm.set_shell_host(Box::new(BareHost));
    let _ = vm.run();
    assert_eq!(vm.last_status, 7);
}

// ─── VM::request_halt contract ──────────────────────────────────────────

// Static counters — BuiltinHandler is a raw fn pointer (no captures),
// so test state lives in static atomics. Tests reset before each use.
static AFTER_HALT_COUNT: AtomicU32 = AtomicU32::new(0);
static TOUCHED: AtomicU32 = AtomicU32::new(0);

fn builtin_request_halt(vm: &mut VM, _argc: u8) -> Value {
    vm.request_halt();
    Value::Int(0)
}

fn builtin_count_after_halt(_vm: &mut VM, _argc: u8) -> Value {
    AFTER_HALT_COUNT.fetch_add(1, Ordering::SeqCst);
    Value::Int(0)
}

fn builtin_touched(_vm: &mut VM, _argc: u8) -> Value {
    TOUCHED.store(1, Ordering::SeqCst);
    Value::Int(0)
}

fn builtin_triple_halt(vm: &mut VM, _argc: u8) -> Value {
    vm.request_halt();
    vm.request_halt();
    vm.request_halt();
    Value::Int(0)
}

#[test]
fn request_halt_from_builtin_stops_dispatch() {
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    // Builtin id 100: requests halt.
    b.emit(Op::CallBuiltin(100, 0), 1);
    // Builtin id 200: increments counter — should NOT execute.
    b.emit(Op::CallBuiltin(200, 0), 1);

    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let r = vm.run();
    // Builtin 100 pushed Value::Int(0) before halting → stack non-empty
    // at exit → run() returns VMResult::Ok(Value::Int(0)) not Halted.
    // VMResult::Halted only fires when the stack is empty after halt.
    // The load-bearing assertion is the side-effect counter below.
    assert!(
        !matches!(r, VMResult::Error(_)),
        "request_halt MUST cleanly exit dispatch, got error"
    );
    assert_eq!(
        AFTER_HALT_COUNT.load(Ordering::SeqCst),
        0,
        "the op after CallBuiltin(100) MUST NOT execute when halt is requested"
    );
}

#[test]
fn request_halt_is_idempotent() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(50, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(50, builtin_triple_halt);
    let r = vm.run();
    // 3× request_halt is just 3× set-bool-true — must not panic or error.
    assert!(
        !matches!(r, VMResult::Error(_)),
        "multiple request_halt calls must be safe"
    );
}

#[test]
fn request_halt_before_run_short_circuits_immediately() {
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(7, builtin_touched);
    vm.request_halt();
    let _ = vm.run();
    assert_eq!(
        TOUCHED.load(Ordering::SeqCst),
        0,
        "pre-run request_halt MUST prevent any op execution"
    );
}

// ─── Combined ───────────────────────────────────────────────────────────

#[test]
fn subshell_some_status_after_normal_run_matches_returned_value() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(7)));
    let _ = vm.run();
    assert_eq!(vm.last_status, 7);
}

#[test]
fn shell_op_runs_without_host_when_no_host_is_set() {
    // No set_shell_host → SubshellEnd op no-ops silently.
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    let mut vm: VM = VM::new(b.build());
    let r = vm.run();
    assert!(matches!(r, VMResult::Ok(_) | VMResult::Halted));
}
