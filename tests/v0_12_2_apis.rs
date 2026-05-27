//! Tests pinning the v0.12.2 API additions:
//!   - `ShellHost::subshell_end -> Option<i32>` (Some(N) updates vm.last_status)
//!   - `VM::request_halt()` (host-callable mid-execution halt)
//!
//! These behaviors are documented in fusevm v0.12.2 release notes and
//! zshrs's TODO.md #6. Until now no test pinned the actual contract —
//! only that the trait sig type-checks. A regression that no-ops
//! Some(status) or that ignores request_halt would slip silently.

use fusevm::{ChunkBuilder, DefaultHost, Op, ShellHost, VMResult, Value, VM};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Mutex;

/// Serializes tests that share static atomics (Rust runs #[test] in parallel).
static PIN_TEST_LOCK: Mutex<()> = Mutex::new(());

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

#[test]
fn subshell_end_negative_status_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(-1)));
    let _ = vm.run();
    assert_eq!(
        vm.last_status, -1,
        "negative exit codes MUST propagate verbatim — shell $? is signed i32"
    );
}

#[test]
fn subshell_end_some_does_not_push_onto_stack() {
    // SubshellEnd only updates last_status; it must not push Status onto
    // the value stack (PipelineEnd does that — different op, different contract).
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(99), 1);
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(5)));
    match vm.run() {
        VMResult::Ok(Value::Int(99)) => {}
        other => panic!(
            "SubshellEnd MUST NOT push onto stack — expected sole stack value Int(99), got {:?}",
            other
        ),
    }
    assert_eq!(vm.last_status, 5);
}

#[test]
fn subshell_end_getstatus_sees_host_returned_status() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(13)));
    match vm.run() {
        VMResult::Ok(Value::Status(13)) => {}
        other => panic!(
            "GetStatus after subshell_end(Some(13)) MUST read updated last_status, got {:?}",
            other
        ),
    }
}

struct FirstSomeThenNoneHost {
    calls: u32,
    status: i32,
}

impl ShellHost for FirstSomeThenNoneHost {
    fn subshell_end(&mut self) -> Option<i32> {
        self.calls += 1;
        if self.calls == 1 {
            Some(self.status)
        } else {
            None
        }
    }
}

#[test]
fn subshell_end_none_after_some_preserves_status_from_first() {
    // Nested subshells: inner returns None → outer's prior write must stick.
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(FirstSomeThenNoneHost {
        calls: 0,
        status: 88,
    }));
    match vm.run() {
        VMResult::Ok(Value::Status(88)) => {}
        other => panic!(
            "None on second subshell_end MUST NOT revert first Some(88), got {:?}",
            other
        ),
    }
}

struct AlternatingStatusHost {
    calls: u32,
}

impl ShellHost for AlternatingStatusHost {
    fn subshell_end(&mut self) -> Option<i32> {
        self.calls += 1;
        match self.calls {
            1 => Some(3),
            2 => Some(9),
            _ => None,
        }
    }
}

#[test]
fn subshell_end_later_some_overwrites_earlier_different_status() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(AlternatingStatusHost { calls: 0 }));
    match vm.run() {
        VMResult::Ok(Value::Status(9)) => {}
        other => panic!(
            "later Some(9) MUST replace earlier Some(3), got {:?}",
            other
        ),
    }
}

#[test]
fn subshell_end_some_overwrites_prior_setstatus() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(50), 1);
    b.emit(Op::SetStatus, 1);
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(3)));
    match vm.run() {
        VMResult::Ok(Value::Status(3)) => {}
        other => panic!(
            "subshell_end(Some(3)) MUST overwrite SetStatus(50), got {:?}",
            other
        ),
    }
}

#[test]
fn subshell_end_without_begin_still_updates_last_status() {
    // Orphan SubshellEnd — host may still return deferred exit status.
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(17)));
    match vm.run() {
        VMResult::Ok(Value::Status(17)) => {}
        other => panic!(
            "SubshellEnd without SubshellBegin MUST still honor host Some(17), got {:?}",
            other
        ),
    }
}

#[test]
fn subshell_end_with_default_host_leaves_last_status() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    let mut vm = VM::new(b.build());
    vm.last_status = 44;
    vm.set_shell_host(Box::new(DefaultHost));
    let _ = vm.run();
    assert_eq!(
        vm.last_status, 44,
        "DefaultHost::subshell_end returns None — last_status MUST stay unchanged"
    );
}

#[test]
fn subshell_end_no_host_leaves_last_status() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.last_status = 66;
    match vm.run() {
        VMResult::Ok(Value::Status(66)) => {}
        other => panic!(
            "SubshellEnd without host MUST NOT touch last_status, got {:?}",
            other
        ),
    }
}

struct SubshellThenDefaultPipelineHost;

impl ShellHost for SubshellThenDefaultPipelineHost {
    fn subshell_end(&mut self) -> Option<i32> {
        Some(4)
    }
    // pipeline_end uses trait default → 0
}

#[test]
fn subshell_end_with_host_then_pipelineend_uses_host_not_last_status() {
    // When a host IS set, PipelineEnd calls host.pipeline_end() — it does
    // NOT fall back to last_status even if subshell_end just wrote 4 there.
    // A regression that "helpfully" reads last_status when pipeline_end is 0
    // would break hosts that intentionally return 0.
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::PipelineEnd, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(SubshellThenDefaultPipelineHost));
    match vm.run() {
        VMResult::Ok(Value::Status(0)) => {}
        other => panic!(
            "PipelineEnd with host MUST use host.pipeline_end(), not subshell last_status(4), got {:?}",
            other
        ),
    }
    assert_eq!(
        vm.last_status, 0,
        "PipelineEnd MUST overwrite last_status with host.pipeline_end() result"
    );
}

struct NoneThenSomeHost {
    calls: u32,
}

impl ShellHost for NoneThenSomeHost {
    fn subshell_end(&mut self) -> Option<i32> {
        self.calls += 1;
        if self.calls == 1 {
            None
        } else {
            Some(42)
        }
    }
}

#[test]
fn subshell_end_none_then_some_updates_on_second() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.last_status = 11;
    vm.set_shell_host(Box::new(NoneThenSomeHost { calls: 0 }));
    match vm.run() {
        VMResult::Ok(Value::Status(42)) => {}
        other => panic!(
            "second subshell_end(Some(42)) MUST apply after first None, got {:?}",
            other
        ),
    }
}

#[test]
fn subshell_begin_alone_does_not_update_last_status() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.last_status = 31;
    vm.set_shell_host(Box::new(StatusReturningHost(99)));
    match vm.run() {
        VMResult::Ok(Value::Status(31)) => {}
        other => panic!(
            "SubshellBegin alone MUST NOT invoke subshell_end or change last_status, got {:?}",
            other
        ),
    }
}

#[test]
fn subshell_end_exit_code_255_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(255)));
    match vm.run() {
        VMResult::Ok(Value::Status(255)) => {}
        other => panic!("shell exit 255 MUST propagate as Status(255), got {:?}", other),
    }
}

#[test]
fn subshell_end_reset_clears_last_status_from_subshell() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    let chunk = b.build();

    let mut vm = VM::new(chunk.clone());
    vm.set_shell_host(Box::new(StatusReturningHost(42)));
    let _ = vm.run();
    assert_eq!(vm.last_status, 42);

    let mut b2 = ChunkBuilder::new();
    b2.emit(Op::GetStatus, 1);
    vm.reset(b2.build());
    match vm.run() {
        VMResult::Ok(Value::Status(0)) => {}
        other => panic!(
            "reset MUST zero last_status even after subshell_end(Some(42)), got {:?}",
            other
        ),
    }
}

struct CustomPipelineHost(i32);

impl ShellHost for CustomPipelineHost {
    fn subshell_end(&mut self) -> Option<i32> {
        Some(4)
    }

    fn pipeline_end(&mut self) -> i32 {
        self.0
    }
}

#[test]
fn subshell_then_custom_pipelineend_uses_host_pipeline_result() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::PipelineEnd, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(CustomPipelineHost(127)));
    match vm.run() {
        VMResult::Ok(Value::Status(127)) => {}
        other => panic!(
            "custom host.pipeline_end(127) MUST win over subshell last_status(4), got {:?}",
            other
        ),
    }
    assert_eq!(vm.last_status, 127);
}

static SUBSHELL_END_CALLS: AtomicU32 = AtomicU32::new(0);

struct CountingSubshellEndHost(i32);

impl ShellHost for CountingSubshellEndHost {
    fn subshell_end(&mut self) -> Option<i32> {
        SUBSHELL_END_CALLS.fetch_add(1, Ordering::SeqCst);
        Some(self.0)
    }
}

#[test]
fn subshell_end_without_host_does_not_call_host() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    SUBSHELL_END_CALLS.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    let mut vm = VM::new(b.build());
    vm.last_status = 5;
    let _ = vm.run();
    assert_eq!(SUBSHELL_END_CALLS.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 5);
}

#[test]
fn subshell_end_with_host_invokes_host_once_per_op() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    SUBSHELL_END_CALLS.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::SubshellEnd, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(CountingSubshellEndHost(1)));
    let _ = vm.run();
    assert_eq!(
        SUBSHELL_END_CALLS.load(Ordering::SeqCst),
        2,
        "each SubshellEnd op MUST invoke host.subshell_end exactly once"
    );
}

static SUBSHELL_BEGIN_CALLS: AtomicU32 = AtomicU32::new(0);

struct BeginEndCountingHost(i32);

impl ShellHost for BeginEndCountingHost {
    fn subshell_begin(&mut self) {
        SUBSHELL_BEGIN_CALLS.fetch_add(1, Ordering::SeqCst);
    }

    fn subshell_end(&mut self) -> Option<i32> {
        Some(self.0)
    }
}

#[test]
fn subshell_begin_end_pair_invokes_begin_once_per_begin_op() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    SUBSHELL_BEGIN_CALLS.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(BeginEndCountingHost(2)));
    let _ = vm.run();
    assert_eq!(
        SUBSHELL_BEGIN_CALLS.load(Ordering::SeqCst),
        1,
        "each SubshellBegin MUST invoke host.subshell_begin exactly once"
    );
    assert_eq!(vm.last_status, 2);
}

#[test]
fn subshell_end_i32_min_status_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(i32::MIN)));
    match vm.run() {
        VMResult::Ok(Value::Status(i32::MIN)) => {}
        other => panic!(
            "i32::MIN exit status MUST propagate verbatim, got {:?}",
            other
        ),
    }
}

#[test]
fn subshell_end_setstatus_after_overwrites_subshell_status() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::LoadInt(19), 1);
    b.emit(Op::SetStatus, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(5)));
    match vm.run() {
        VMResult::Ok(Value::Status(19)) => {}
        other => panic!(
            "SetStatus after subshell_end MUST overwrite subshell last_status, got {:?}",
            other
        ),
    }
}

#[test]
fn subshell_end_getstatus_before_and_after_reads_different_values() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::GetStatus, 1);
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.last_status = 2;
    vm.set_shell_host(Box::new(StatusReturningHost(8)));
    match vm.run() {
        VMResult::Ok(Value::Status(8)) => {}
        other => panic!("expected final stack top Status(8), got {:?}", other),
    }
    assert_eq!(vm.last_status, 8);
}

struct IncrementingSubshellHost {
    next: i32,
}

impl ShellHost for IncrementingSubshellHost {
    fn subshell_end(&mut self) -> Option<i32> {
        let s = self.next;
        self.next += 1;
        Some(s)
    }
}

#[test]
fn subshell_end_three_some_in_sequence_last_wins() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(IncrementingSubshellHost { next: 10 }));
    match vm.run() {
        VMResult::Ok(Value::Status(12)) => {}
        other => panic!(
            "third subshell_end(Some(12)) MUST be visible to GetStatus, got {:?}",
            other
        ),
    }
}

#[test]
fn subshell_end_none_after_setstatus_preserves_setstatus_value() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(23), 1);
    b.emit(Op::SetStatus, 1);
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(NoneHost));
    match vm.run() {
        VMResult::Ok(Value::Status(23)) => {}
        other => panic!(
            "subshell_end(None) MUST leave SetStatus(23) intact, got {:?}",
            other
        ),
    }
}

#[test]
fn subshell_end_reset_preserves_shell_host() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let chunk = b.build();

    let mut vm = VM::new(chunk.clone());
    vm.set_shell_host(Box::new(StatusReturningHost(91)));
    let _ = vm.run();
    assert_eq!(vm.last_status, 91);

    vm.reset(chunk);
    match vm.run() {
        VMResult::Ok(Value::Status(91)) => {}
        other => panic!(
            "reset MUST preserve shell host for subshell_end on rerun, got {:?}",
            other
        ),
    }
}

#[test]
fn subshell_end_i32_max_status_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(i32::MAX)));
    match vm.run() {
        VMResult::Ok(Value::Status(i32::MAX)) => {}
        other => panic!(
            "i32::MAX exit status MUST propagate verbatim, got {:?}",
            other
        ),
    }
}

#[test]
fn subshell_begin_without_host_is_noop() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.last_status = 5;
    match vm.run() {
        VMResult::Ok(Value::Status(5)) => {}
        other => panic!(
            "SubshellBegin without host MUST NOT change last_status, got {:?}",
            other
        ),
    }
}

struct ThirdSomeSubshellHost {
    calls: u32,
}

impl ShellHost for ThirdSomeSubshellHost {
    fn subshell_end(&mut self) -> Option<i32> {
        self.calls += 1;
        if self.calls == 3 {
            Some(30)
        } else {
            None
        }
    }
}

#[test]
fn subshell_end_some_on_third_none_before_preserves_until_some() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.last_status = 1;
    vm.set_shell_host(Box::new(ThirdSomeSubshellHost { calls: 0 }));
    match vm.run() {
        VMResult::Ok(Value::Status(30)) => {}
        other => panic!(
            "third subshell_end(Some(30)) MUST apply; first two None leave 1, got {:?}",
            other
        ),
    }
}

#[test]
fn subshell_double_begin_single_end_still_updates_status() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    SUBSHELL_BEGIN_CALLS.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(BeginEndCountingHost(6)));
    match vm.run() {
        VMResult::Ok(Value::Status(6)) => {}
        other => panic!(
            "SubshellEnd after two SubshellBegin MUST still apply Some(6), got {:?}",
            other
        ),
    }
    assert_eq!(
        SUBSHELL_BEGIN_CALLS.load(Ordering::SeqCst),
        2,
        "each SubshellBegin MUST invoke host even when paired with one SubshellEnd"
    );
}

static PIPELINE_END_CALLS: AtomicU32 = AtomicU32::new(0);

struct CountingPipelineHost;

impl ShellHost for CountingPipelineHost {
    fn pipeline_end(&mut self) -> i32 {
        PIPELINE_END_CALLS.fetch_add(1, Ordering::SeqCst);
        0
    }
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

fn builtin_halt_only(vm: &mut VM, _argc: u8) -> Value {
    vm.request_halt();
    // CallBuiltin always pushes handler return value — Undef keeps stack non-empty
    // when combined with prior pushes, but an empty pre-run stack + pre-run halt
    // is the Halted path (see request_halt_pre_run_empty_stack_returns_halted).
    Value::Undef
}

#[test]
fn request_halt_from_builtin_stops_dispatch() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
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
    let _guard = PIN_TEST_LOCK.lock().unwrap();
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

#[test]
fn request_halt_pre_run_empty_stack_returns_halted() {
    let b = ChunkBuilder::new();
    let mut vm = VM::new(b.build());
    vm.request_halt();
    match vm.run() {
        VMResult::Halted => {}
        other => panic!(
            "pre-run request_halt on empty stack MUST return VMResult::Halted, got {:?}",
            other
        ),
    }
}

#[test]
fn request_halt_preserves_prior_last_status() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(60, 0), 1);
    let mut vm = VM::new(b.build());
    vm.last_status = 55;
    vm.register_builtin(60, builtin_request_halt);
    let _ = vm.run();
    assert_eq!(
        vm.last_status, 55,
        "request_halt MUST NOT clobber last_status — only stops dispatch"
    );
}

#[test]
fn request_halt_reset_clears_halt_and_allows_rerun() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(7, builtin_touched);
    vm.request_halt();
    let _ = vm.run();
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);

    let mut b2 = ChunkBuilder::new();
    b2.emit(Op::LoadInt(42), 1);
    vm.reset(b2.build());
    match vm.run() {
        VMResult::Ok(Value::Int(42)) => {}
        other => panic!(
            "reset MUST clear request_halt so VM can run again, got {:?}",
            other
        ),
    }
}

#[test]
fn request_halt_stops_before_subsequent_ops_leave_stack_intact() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(11), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::LoadInt(22), 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    match vm.run() {
        VMResult::Ok(Value::Int(0)) => {}
        other => panic!(
            "halt after builtin MUST leave prior stack values + builtin result, got {:?}",
            other
        ),
    }
    assert_eq!(
        AFTER_HALT_COUNT.load(Ordering::SeqCst),
        0,
        "LoadInt(22) and CallBuiltin(200) MUST NOT run after halt"
    );
}

#[test]
fn request_halt_from_builtin_with_only_undef_on_stack() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(61, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(61, builtin_halt_only);
    match vm.run() {
        VMResult::Ok(Value::Undef) => {}
        other => panic!(
            "halt via builtin still pushes handler result before run() pops it, got {:?}",
            other
        ),
    }
}

#[test]
fn request_halt_from_extension_handler_stops_dispatch() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::Extended(1, 0), 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_extension_handler(Box::new(|vm, _id, _arg| {
        vm.request_halt();
    }));
    vm.register_builtin(200, builtin_count_after_halt);
    let r = vm.run();
    assert!(
        !matches!(r, VMResult::Error(_)),
        "request_halt from extension handler MUST exit cleanly, got {:?}",
        r
    );
    assert_eq!(
        AFTER_HALT_COUNT.load(Ordering::SeqCst),
        0,
        "ops after Extended MUST NOT run when extension handler requests halt"
    );
}

#[test]
fn request_halt_second_run_without_reset_does_not_resume() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);

    let r = vm.run();
    assert!(
        matches!(r, VMResult::Ok(_) | VMResult::Halted),
        "second run() without reset MUST NOT error, got {:?}",
        r
    );
    assert_eq!(
        AFTER_HALT_COUNT.load(Ordering::SeqCst),
        0,
        "halted flag MUST persist across run() calls until reset"
    );
}

#[test]
fn request_halt_from_nested_call_stops_caller_ops() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    let name = b.add_name("haltfn");
    b.emit(Op::Call(name, 0), 1);
    b.emit(Op::LoadInt(99), 1);
    let skip = b.emit(Op::Jump(0), 1);
    let entry = b.current_pos();
    b.add_sub_entry(name, entry);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::ReturnValue, 1);
    b.patch_jump(skip, b.current_pos());

    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    match vm.run() {
        VMResult::Ok(Value::Int(0)) => {}
        other => panic!(
            "halt inside callee MUST return builtin result without running caller LoadInt(99), got {:?}",
            other
        ),
    }
}

#[test]
fn request_halt_reset_preserves_shell_host() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let chunk = b.build();

    let mut vm = VM::new(chunk.clone());
    vm.set_shell_host(Box::new(StatusReturningHost(33)));
    vm.request_halt();
    let _ = vm.run();

    vm.reset(chunk);
    match vm.run() {
        VMResult::Ok(Value::Status(33)) => {}
        other => panic!(
            "reset MUST preserve shell host so subshell_end still works, got {:?}",
            other
        ),
    }
}

#[test]
fn request_halt_reset_preserves_registered_builtins() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(7, 0), 1);
    let chunk = b.build();

    let mut vm = VM::new(chunk.clone());
    vm.register_builtin(7, builtin_touched);
    vm.request_halt();
    let _ = vm.run();
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);

    vm.reset(chunk);
    let _ = vm.run();
    assert_eq!(
        TOUCHED.load(Ordering::SeqCst),
        1,
        "reset MUST preserve builtin table registered before halt"
    );
}

#[test]
fn request_halt_does_not_clear_setstatus_during_run() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(77), 1);
    b.emit(Op::SetStatus, 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    let _ = vm.run();
    assert_eq!(
        vm.last_status, 77,
        "request_halt MUST NOT undo SetStatus written earlier in the same run"
    );
}

#[test]
fn request_halt_from_extension_wide_handler_stops_dispatch() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::ExtendedWide(9, 0xBEEF), 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_extension_wide_handler(Box::new(|vm, _id, _payload| {
        vm.request_halt();
    }));
    vm.register_builtin(200, builtin_count_after_halt);
    let r = vm.run();
    assert!(
        !matches!(r, VMResult::Error(_)),
        "request_halt from ExtendedWide handler MUST exit cleanly, got {:?}",
        r
    );
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
}

#[test]
fn request_halt_stops_before_jump_target() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    let jmp = b.emit(Op::Jump(0), 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    b.patch_jump(jmp, b.current_pos());

    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(
        AFTER_HALT_COUNT.load(Ordering::SeqCst),
        0,
        "Jump target MUST NOT execute after request_halt"
    );
}

#[test]
fn request_halt_allows_ops_before_halt_to_complete() {
    static PRE_HALT: AtomicU32 = AtomicU32::new(0);
    fn mark_then_halt(vm: &mut VM, _argc: u8) -> Value {
        PRE_HALT.store(1, Ordering::SeqCst);
        vm.request_halt();
        Value::Int(0)
    }
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    PRE_HALT.store(0, Ordering::SeqCst);
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);

    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(5), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::LoadInt(6), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, mark_then_halt);
    match vm.run() {
        VMResult::Ok(Value::Int(0)) => {}
        other => panic!(
            "ops before halt MUST complete; halt builtin MUST push its result, got {:?}",
            other
        ),
    }
    assert_eq!(PRE_HALT.load(Ordering::SeqCst), 1);
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
}

#[test]
fn request_halt_prevents_subshell_end_from_running() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.last_status = 10;
    vm.set_shell_host(Box::new(StatusReturningHost(99)));
    vm.register_builtin(100, builtin_request_halt);
    match vm.run() {
        VMResult::Ok(Value::Int(0)) => {}
        other => panic!("halt MUST return builtin result, got {:?}", other),
    }
    assert_eq!(
        vm.last_status, 10,
        "subshell_end MUST NOT run after request_halt — last_status stays 10 not 99"
    );
}

#[test]
fn request_halt_pre_run_skips_ops_so_stack_stays_empty() {
    // Pre-run halt fires before dispatch — LoadInt in the chunk never runs.
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(123), 1);
    let mut vm = VM::new(b.build());
    vm.request_halt();
    match vm.run() {
        VMResult::Halted => {}
        other => panic!(
            "pre-run halt MUST skip all ops including LoadInt — stack empty → Halted, got {:?}",
            other
        ),
    }
}

#[test]
fn request_halt_after_subshell_end_preserves_subshell_status() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(61)));
    vm.register_builtin(100, builtin_request_halt);
    let _ = vm.run();
    assert_eq!(
        vm.last_status, 61,
        "request_halt after subshell_end MUST preserve subshell-written last_status"
    );
}

#[test]
fn request_halt_mid_chunk_then_reset_allows_subshell_end() {
    let mut halt_chunk = ChunkBuilder::new();
    halt_chunk.emit(Op::CallBuiltin(100, 0), 1);
    let halt_chunk = halt_chunk.build();

    let mut subshell_chunk = ChunkBuilder::new();
    subshell_chunk.emit(Op::SubshellBegin, 1);
    subshell_chunk.emit(Op::SubshellEnd, 1);
    subshell_chunk.emit(Op::GetStatus, 1);
    let subshell_chunk = subshell_chunk.build();

    let mut vm = VM::new(halt_chunk);
    vm.set_shell_host(Box::new(StatusReturningHost(71)));
    vm.register_builtin(100, builtin_request_halt);
    let _ = vm.run();
    assert_eq!(vm.last_status, 0);

    vm.reset(subshell_chunk);
    match vm.run() {
        VMResult::Ok(Value::Status(71)) => {}
        other => panic!(
            "after reset, subshell_end MUST work post-halt run, got {:?}",
            other
        ),
    }
}

#[test]
fn request_halt_stops_before_conditional_jump_target() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::NumEq, 1);
    let jmp = b.emit(Op::JumpIfTrue(0), 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    b.patch_jump(jmp, b.current_pos());

    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(
        AFTER_HALT_COUNT.load(Ordering::SeqCst),
        0,
        "JumpIfTrue target MUST NOT execute after request_halt"
    );
}

#[test]
fn request_halt_stops_before_push_frame() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::PushFrame, 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(
        AFTER_HALT_COUNT.load(Ordering::SeqCst),
        0,
        "PushFrame and subsequent ops MUST NOT run after request_halt"
    );
}

fn builtin_halt_report_argc(vm: &mut VM, argc: u8) -> Value {
    vm.request_halt();
    Value::Int(argc as i64)
}

#[test]
fn request_halt_from_builtin_with_args_still_halts() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::LoadInt(2), 1);
    b.emit(Op::CallBuiltin(100, 2), 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_halt_report_argc);
    vm.register_builtin(200, builtin_count_after_halt);
    match vm.run() {
        VMResult::Ok(Value::Int(2)) => {}
        other => panic!(
            "halt builtin MUST receive argc=2 and push it before stopping, got {:?}",
            other
        ),
    }
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
}

#[test]
fn request_halt_reset_preserves_extension_handler() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    static EXT_TOUCHED: AtomicU32 = AtomicU32::new(0);
    let mut halt_chunk = ChunkBuilder::new();
    halt_chunk.emit(Op::CallBuiltin(100, 0), 1);
    let halt_chunk = halt_chunk.build();

    let mut ext_chunk = ChunkBuilder::new();
    ext_chunk.emit(Op::Extended(3, 7), 1);
    let ext_chunk = ext_chunk.build();

    let mut vm = VM::new(halt_chunk);
    vm.set_extension_handler(Box::new(|_vm, _id, _arg| {
        EXT_TOUCHED.fetch_add(1, Ordering::SeqCst);
    }));
    vm.register_builtin(100, builtin_request_halt);
    let _ = vm.run();

    EXT_TOUCHED.store(0, Ordering::SeqCst);
    vm.reset(ext_chunk);
    let _ = vm.run();
    assert_eq!(
        EXT_TOUCHED.load(Ordering::SeqCst),
        1,
        "reset MUST preserve extension handler registered before halt"
    );
}

#[test]
fn request_halt_multiple_ops_before_all_complete() {
    static PRE_COUNT: AtomicU32 = AtomicU32::new(0);
    fn count_then_halt(vm: &mut VM, _argc: u8) -> Value {
        PRE_COUNT.fetch_add(1, Ordering::SeqCst);
        vm.request_halt();
        Value::Int(0)
    }
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    PRE_COUNT.store(0, Ordering::SeqCst);
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);

    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::LoadInt(2), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::LoadInt(3), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, count_then_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(PRE_COUNT.load(Ordering::SeqCst), 1);
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
}

#[test]
fn request_halt_does_not_run_setstatus_after_halt_point() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::LoadInt(99), 1);
    b.emit(Op::SetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.last_status = 1;
    vm.register_builtin(100, builtin_request_halt);
    let _ = vm.run();
    assert_eq!(
        vm.last_status, 1,
        "SetStatus after halt MUST NOT run — last_status stays pre-halt value"
    );
}

#[test]
fn subshell_end_then_request_halt_getstatus_not_reached() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(45)));
    vm.register_builtin(100, builtin_request_halt);
    match vm.run() {
        VMResult::Ok(Value::Int(0)) => {}
        other => panic!("expected builtin halt result Int(0), got {:?}", other),
    }
    assert_eq!(
        vm.last_status, 45,
        "subshell status MUST be set before halt; GetStatus after halt MUST NOT run"
    );
}

#[test]
fn request_halt_stops_before_jump_if_false_target() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::LoadInt(2), 1);
    b.emit(Op::NumEq, 1);
    let jmp = b.emit(Op::JumpIfFalse(0), 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    b.patch_jump(jmp, b.current_pos());

    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(
        AFTER_HALT_COUNT.load(Ordering::SeqCst),
        0,
        "JumpIfFalse target MUST NOT execute after request_halt"
    );
}

#[test]
fn request_halt_stops_before_pop_frame() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::PushFrame, 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::PopFrame, 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(
        AFTER_HALT_COUNT.load(Ordering::SeqCst),
        0,
        "PopFrame and subsequent ops MUST NOT run after request_halt"
    );
}

#[test]
fn request_halt_prevents_subshell_begin_from_running() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    SUBSHELL_BEGIN_CALLS.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(BeginEndCountingHost(9)));
    vm.register_builtin(100, builtin_request_halt);
    let _ = vm.run();
    assert_eq!(
        SUBSHELL_BEGIN_CALLS.load(Ordering::SeqCst),
        0,
        "SubshellBegin MUST NOT run after request_halt"
    );
}

#[test]
fn request_halt_prevents_pipeline_end_from_running() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    PIPELINE_END_CALLS.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::PipelineEnd, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(CountingPipelineHost));
    vm.register_builtin(100, builtin_request_halt);
    let _ = vm.run();
    assert_eq!(
        PIPELINE_END_CALLS.load(Ordering::SeqCst),
        0,
        "PipelineEnd MUST NOT run after request_halt"
    );
}

#[test]
fn request_halt_as_first_op_returns_ok_not_halted() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    match vm.run() {
        VMResult::Ok(Value::Int(0)) => {}
        VMResult::Halted => panic!(
            "halt via first-op CallBuiltin pushes result — MUST return Ok(Int(0)), not Halted"
        ),
        other => panic!("unexpected result {:?}", other),
    }
}

#[test]
fn request_halt_reset_preserves_extension_wide_handler() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    static EXT_WIDE_TOUCHED: AtomicU32 = AtomicU32::new(0);
    let mut halt_chunk = ChunkBuilder::new();
    halt_chunk.emit(Op::CallBuiltin(100, 0), 1);
    let halt_chunk = halt_chunk.build();

    let mut wide_chunk = ChunkBuilder::new();
    wide_chunk.emit(Op::ExtendedWide(5, 0xCAFE), 1);
    let wide_chunk = wide_chunk.build();

    let mut vm = VM::new(halt_chunk);
    vm.set_extension_wide_handler(Box::new(|_vm, _id, _payload| {
        EXT_WIDE_TOUCHED.fetch_add(1, Ordering::SeqCst);
    }));
    vm.register_builtin(100, builtin_request_halt);
    let _ = vm.run();

    EXT_WIDE_TOUCHED.store(0, Ordering::SeqCst);
    vm.reset(wide_chunk);
    let _ = vm.run();
    assert_eq!(
        EXT_WIDE_TOUCHED.load(Ordering::SeqCst),
        1,
        "reset MUST preserve extension wide handler registered before halt"
    );
}

#[test]
fn request_halt_nested_call_does_not_return_to_caller() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    let name = b.add_name("inner");
    b.emit(Op::LoadInt(100), 1);
    b.emit(Op::Call(name, 0), 1);
    b.emit(Op::LoadInt(200), 1);
    let skip = b.emit(Op::Jump(0), 1);
    let entry = b.current_pos();
    b.add_sub_entry(name, entry);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::ReturnValue, 1);
    b.patch_jump(skip, b.current_pos());

    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    match vm.run() {
        VMResult::Ok(Value::Int(0)) => {}
        other => panic!(
            "halt in callee MUST NOT ReturnValue to caller — expect builtin result, got {:?}",
            other
        ),
    }
}

#[test]
fn request_halt_pre_run_then_reset_runs_chunk_normally() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(7, 0), 1);
    let chunk = b.build();

    let mut vm = VM::new(chunk.clone());
    vm.register_builtin(7, builtin_touched);
    vm.request_halt();
    match vm.run() {
        VMResult::Halted => {}
        other => panic!("pre-run halt expected Halted, got {:?}", other),
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);

    vm.reset(chunk);
    let _ = vm.run();
    assert_eq!(
        TOUCHED.load(Ordering::SeqCst),
        1,
        "reset after pre-run halt MUST allow normal execution"
    );
}

#[test]
fn subshell_end_and_halt_in_same_chunk_subshell_runs_first() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    SUBSHELL_BEGIN_CALLS.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::SubshellBegin, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(BeginEndCountingHost(52)));
    vm.register_builtin(100, builtin_request_halt);
    let _ = vm.run();
    assert_eq!(vm.last_status, 52);
    assert_eq!(
        SUBSHELL_BEGIN_CALLS.load(Ordering::SeqCst),
        1,
        "only subshell ops before halt MUST run — second SubshellBegin MUST NOT"
    );
}

#[test]
fn subshell_only_ops_empty_stack_returns_halted() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(7)));
    match vm.run() {
        VMResult::Halted => {}
        other => panic!(
            "subshell-only chunk MUST leave stack empty → Halted, got {:?}",
            other
        ),
    }
    assert_eq!(vm.last_status, 7);
}

#[test]
fn subshell_end_exit_code_128_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(128)));
    match vm.run() {
        VMResult::Ok(Value::Status(128)) => {}
        other => panic!("exit code 128 MUST propagate as Status(128), got {:?}", other),
    }
}

struct PipelineThenSubshellHost;

impl ShellHost for PipelineThenSubshellHost {
    fn pipeline_end(&mut self) -> i32 {
        3
    }

    fn subshell_end(&mut self) -> Option<i32> {
        Some(9)
    }
}

#[test]
fn subshell_end_after_pipelineend_overwrites_pipeline_last_status() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::PipelineEnd, 1);
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(PipelineThenSubshellHost));
    match vm.run() {
        VMResult::Ok(Value::Status(9)) => {}
        other => panic!(
            "subshell_end(Some(9)) MUST overwrite pipeline_end(3) last_status, got {:?}",
            other
        ),
    }
}

#[test]
fn subshell_end_manual_last_status_write_visible_to_getstatus() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.last_status = 14;
    match vm.run() {
        VMResult::Ok(Value::Status(14)) => {}
        other => panic!(
            "GetStatus MUST read vm.last_status field set before run(), got {:?}",
            other
        ),
    }
}

static SUBSHELL_END_NONE_CALLS: AtomicU32 = AtomicU32::new(0);

struct NoneThenStatusHost(i32);

impl ShellHost for NoneThenStatusHost {
    fn subshell_end(&mut self) -> Option<i32> {
        SUBSHELL_END_NONE_CALLS.fetch_add(1, Ordering::SeqCst);
        if SUBSHELL_END_NONE_CALLS.load(Ordering::SeqCst) == 1 {
            None
        } else {
            Some(self.0)
        }
    }
}

#[test]
fn subshell_end_orphan_end_after_none_still_applies_some() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    SUBSHELL_END_NONE_CALLS.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.last_status = 2;
    vm.set_shell_host(Box::new(NoneThenStatusHost(16)));
    match vm.run() {
        VMResult::Ok(Value::Status(16)) => {}
        other => panic!(
            "second orphan SubshellEnd(Some(16)) MUST apply after first None, got {:?}",
            other
        ),
    }
}

static PIPELINE_BEGIN_CALLS: AtomicU32 = AtomicU32::new(0);

struct CountingPipelineBeginHost;

impl ShellHost for CountingPipelineBeginHost {
    fn pipeline_begin(&mut self, _n: u8) {
        PIPELINE_BEGIN_CALLS.fetch_add(1, Ordering::SeqCst);
    }
}

#[test]
fn request_halt_stops_before_top_level_return() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::Return, 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(
        AFTER_HALT_COUNT.load(Ordering::SeqCst),
        0,
        "top-level Return MUST NOT run after request_halt"
    );
}

#[test]
fn request_halt_stops_before_pipeline_begin() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    PIPELINE_BEGIN_CALLS.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::PipelineBegin(2), 1);
    b.emit(Op::PipelineEnd, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(CountingPipelineBeginHost));
    vm.register_builtin(100, builtin_request_halt);
    let _ = vm.run();
    assert_eq!(
        PIPELINE_BEGIN_CALLS.load(Ordering::SeqCst),
        0,
        "PipelineBegin MUST NOT run after request_halt"
    );
}

#[test]
fn request_halt_run_pops_only_stack_top() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(22), 1);
    b.emit(Op::LoadInt(11), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    match vm.run() {
        VMResult::Ok(Value::Int(0)) => {}
        other => panic!(
            "run() MUST pop only stack top (builtin result 0), not deeper Int(22), got {:?}",
            other
        ),
    }
}

#[test]
fn request_halt_skips_getstatus_after_subshell_wrote_status() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(37)));
    vm.register_builtin(100, builtin_request_halt);
    match vm.run() {
        VMResult::Ok(Value::Int(0)) => {}
        other => panic!(
            "halt MUST return builtin Int(0), not GetStatus(37) pushed after halt point, got {:?}",
            other
        ),
    }
    assert_eq!(vm.last_status, 37);
}

#[test]
fn request_halt_stops_before_later_callbuiltin() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    let _ = vm.run();
    assert_eq!(
        TOUCHED.load(Ordering::SeqCst),
        0,
        "CallBuiltin after halt MUST NOT run even when registered"
    );
}

#[test]
fn request_halt_both_extension_handlers_only_first_runs() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    static EXT_NARROW: AtomicU32 = AtomicU32::new(0);
    static EXT_WIDE: AtomicU32 = AtomicU32::new(0);
    let mut b = ChunkBuilder::new();
    b.emit(Op::Extended(1, 0), 1);
    b.emit(Op::ExtendedWide(2, 0), 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_extension_handler(Box::new(|vm, _id, _arg| {
        EXT_NARROW.fetch_add(1, Ordering::SeqCst);
        vm.request_halt();
    }));
    vm.set_extension_wide_handler(Box::new(|_vm, _id, _payload| {
        EXT_WIDE.fetch_add(1, Ordering::SeqCst);
    }));
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(EXT_NARROW.load(Ordering::SeqCst), 1);
    assert_eq!(
        EXT_WIDE.load(Ordering::SeqCst),
        0,
        "ExtendedWide MUST NOT run after Extended handler requests halt"
    );
}


static PIPELINE_STAGE_CALLS: AtomicU32 = AtomicU32::new(0);

struct CountingPipelineStageHost;

impl ShellHost for CountingPipelineStageHost {
    fn pipeline_stage(&mut self) {
        PIPELINE_STAGE_CALLS.fetch_add(1, Ordering::SeqCst);
    }
}

#[test]
fn request_halt_stops_before_pipeline_stage() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    PIPELINE_STAGE_CALLS.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::PipelineStage, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(CountingPipelineStageHost));
    vm.register_builtin(100, builtin_request_halt);
    let _ = vm.run();
    assert_eq!(
        PIPELINE_STAGE_CALLS.load(Ordering::SeqCst),
        0,
        "PipelineStage MUST NOT run after request_halt"
    );
}

#[test]
fn subshell_end_exit_code_1_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(1)));
    match vm.run() {
        VMResult::Ok(Value::Status(1)) => {}
        other => panic!("exit code 1 MUST propagate as Status(1), got {:?}", other),
    }
}


#[test]
fn subshell_end_getstatus_leaves_prior_status_on_stack_below() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::GetStatus, 1);
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.last_status = 2;
    vm.set_shell_host(Box::new(StatusReturningHost(8)));
    match vm.run() {
        VMResult::Ok(Value::Status(8)) => {}
        other => panic!(
            "run() MUST pop top Status(8) from second GetStatus, got {:?}",
            other
        ),
    }
}

#[test]
fn request_halt_stops_before_return_value_top_level() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(55), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::ReturnValue, 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    match vm.run() {
        VMResult::Ok(Value::Int(0)) => {}
        other => panic!(
            "halt MUST return builtin Int(0), not ReturnValue(55), got {:?}",
            other
        ),
    }
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
}


#[test]
fn request_halt_wide_extension_halts_before_narrow() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    static EXT_N: AtomicU32 = AtomicU32::new(0);
    static EXT_W: AtomicU32 = AtomicU32::new(0);
    let mut b = ChunkBuilder::new();
    b.emit(Op::ExtendedWide(1, 0), 1);
    b.emit(Op::Extended(2, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_extension_wide_handler(Box::new(|vm, _id, _payload| {
        EXT_W.fetch_add(1, Ordering::SeqCst);
        vm.request_halt();
    }));
    vm.set_extension_handler(Box::new(|_vm, _id, _arg| {
        EXT_N.fetch_add(1, Ordering::SeqCst);
    }));
    let _ = vm.run();
    assert_eq!(EXT_W.load(Ordering::SeqCst), 1);
    assert_eq!(
        EXT_N.load(Ordering::SeqCst),
        0,
        "Extended MUST NOT run after ExtendedWide requests halt"
    );
}

#[test]
fn subshell_end_some_overwrites_setstatus_zero() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(0), 1);
    b.emit(Op::SetStatus, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(6)));
    match vm.run() {
        VMResult::Ok(Value::Status(6)) => {}
        other => panic!(
            "subshell_end(Some(6)) MUST overwrite SetStatus(0), got {:?}",
            other
        ),
    }
}


#[test]
fn request_halt_prevents_subshell_end_host_invocation() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    SUBSHELL_END_CALLS.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::SubshellEnd, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(CountingSubshellEndHost(4)));
    vm.register_builtin(100, builtin_request_halt);
    let _ = vm.run();
    assert_eq!(
        SUBSHELL_END_CALLS.load(Ordering::SeqCst),
        0,
        "SubshellEnd MUST NOT invoke host after request_halt"
    );
}

#[test]
fn subshell_triple_begin_end_updates_to_last_some() {
    let mut b = ChunkBuilder::new();
    for _ in 0..3 {
        b.emit(Op::SubshellBegin, 1);
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(IncrementingSubshellHost { next: 10 }));
    match vm.run() {
        VMResult::Ok(Value::Status(12)) => {}
        other => panic!(
            "third subshell_end(Some(12)) MUST win in triple pairs, got {:?}",
            other
        ),
    }
}


#[test]
fn request_halt_stops_before_jump_if_true_keep() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::NumEq, 1);
    let jmp = b.emit(Op::JumpIfTrueKeep(0), 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    b.patch_jump(jmp, b.current_pos());
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(
        AFTER_HALT_COUNT.load(Ordering::SeqCst),
        0,
        "JumpIfTrueKeep target MUST NOT run after request_halt"
    );
}

#[test]
fn subshell_end_last_status_set_even_when_run_returns_halted() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(21)));
    assert!(matches!(vm.run(), VMResult::Halted));
    assert_eq!(vm.last_status, 21);
}


#[test]
fn request_halt_stops_second_extension_in_chain() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    static EXT_A: AtomicU32 = AtomicU32::new(0);
    static EXT_B: AtomicU32 = AtomicU32::new(0);
    let mut b = ChunkBuilder::new();
    b.emit(Op::Extended(1, 0), 1);
    b.emit(Op::Extended(2, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_extension_handler(Box::new(|vm, id, _arg| {
        if id == 1 {
            EXT_A.fetch_add(1, Ordering::SeqCst);
            vm.request_halt();
        } else {
            EXT_B.fetch_add(1, Ordering::SeqCst);
        }
    }));
    let _ = vm.run();
    assert_eq!(EXT_A.load(Ordering::SeqCst), 1);
    assert_eq!(EXT_B.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_begin_does_not_change_status_before_end() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(4), 1);
    b.emit(Op::SetStatus, 1);
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(99)));
    match vm.run() {
        VMResult::Ok(Value::Status(4)) => {}
        other => panic!(
            "SubshellBegin MUST NOT apply subshell_end status early, got {:?}",
            other
        ),
    }
}


#[test]
fn request_halt_reset_clears_setstatus_from_halted_run() {
    let mut halt_b = ChunkBuilder::new();
    halt_b.emit(Op::LoadInt(50), 1);
    halt_b.emit(Op::SetStatus, 1);
    halt_b.emit(Op::CallBuiltin(100, 0), 1);
    let halt_chunk = halt_b.build();

    let mut status_b = ChunkBuilder::new();
    status_b.emit(Op::GetStatus, 1);
    let status_chunk = status_b.build();

    let mut vm = VM::new(halt_chunk);
    vm.register_builtin(100, builtin_request_halt);
    let _ = vm.run();
    assert_eq!(vm.last_status, 50);

    vm.reset(status_chunk);
    match vm.run() {
        VMResult::Ok(Value::Status(0)) => {}
        other => panic!(
            "reset MUST zero last_status from prior halted run, got {:?}",
            other
        ),
    }
}

#[test]
fn subshell_end_between_pipeline_ops_does_not_call_pipeline_end() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    PIPELINE_END_CALLS.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::PipelineBegin(2), 1);
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(CountingSubshellEndHost(3)));
    let _ = vm.run();
    assert_eq!(vm.last_status, 3);
    assert_eq!(
        PIPELINE_END_CALLS.load(Ordering::SeqCst),
        0,
        "SubshellEnd MUST NOT implicitly invoke pipeline_end"
    );
}


#[test]
fn request_halt_nested_call_with_args_still_halts() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    let name = b.add_name("fn");
    b.emit(Op::LoadInt(10), 1);
    b.emit(Op::LoadInt(20), 1);
    b.emit(Op::Call(name, 2), 1);
    b.emit(Op::LoadInt(30), 1);
    let skip = b.emit(Op::Jump(0), 1);
    let entry = b.current_pos();
    b.add_sub_entry(name, entry);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::ReturnValue, 1);
    b.patch_jump(skip, b.current_pos());
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    match vm.run() {
        VMResult::Ok(Value::Int(0)) => {}
        other => panic!("nested halt with args MUST return builtin result, got {:?}", other),
    }
}

#[test]
fn subshell_end_without_host_orphan_leaves_last_status() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.last_status = 18;
    match vm.run() {
        VMResult::Ok(Value::Status(18)) => {}
        other => panic!(
            "orphan SubshellEnd without host MUST leave last_status(18), got {:?}",
            other
        ),
    }
}


#[test]
fn request_halt_three_consecutive_runs_never_resume() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    let _ = vm.run();
    let _ = vm.run();
    let _ = vm.run();
    assert_eq!(
        TOUCHED.load(Ordering::SeqCst),
        0,
        "three run() calls without reset MUST NOT resume dispatch"
    );
}

#[test]
fn subshell_end_getstatus_then_loadint_returns_loadint() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    b.emit(Op::LoadInt(44), 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(5)));
    match vm.run() {
        VMResult::Ok(Value::Int(44)) => {}
        other => panic!(
            "final stack top MUST be LoadInt(44) not Status(5), got {:?}",
            other
        ),
    }
}


#[test]
fn request_halt_reset_subshell_getstatus_lifecycle() {
    let mut halt_b = ChunkBuilder::new();
    halt_b.emit(Op::CallBuiltin(100, 0), 1);
    let halt_chunk = halt_b.build();

    let mut sub_b = ChunkBuilder::new();
    sub_b.emit(Op::SubshellBegin, 1);
    sub_b.emit(Op::SubshellEnd, 1);
    sub_b.emit(Op::GetStatus, 1);
    let sub_chunk = sub_b.build();

    let mut vm = VM::new(halt_chunk);
    vm.set_shell_host(Box::new(StatusReturningHost(93)));
    vm.register_builtin(100, builtin_request_halt);
    let _ = vm.run();

    vm.reset(sub_chunk);
    match vm.run() {
        VMResult::Ok(Value::Status(93)) => {}
        other => panic!(
            "full lifecycle: halt → reset → subshell_end(Some(93)) → GetStatus, got {:?}",
            other
        ),
    }
}

#[test]
fn subshell_halt_reset_subshell_preserves_host_and_status() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    SUBSHELL_BEGIN_CALLS.store(0, Ordering::SeqCst);

    let mut chunk_a = ChunkBuilder::new();
    chunk_a.emit(Op::SubshellBegin, 1);
    chunk_a.emit(Op::SubshellEnd, 1);
    chunk_a.emit(Op::CallBuiltin(100, 0), 1);
    let chunk_a = chunk_a.build();

    let mut chunk_b = ChunkBuilder::new();
    chunk_b.emit(Op::SubshellBegin, 1);
    chunk_b.emit(Op::SubshellEnd, 1);
    chunk_b.emit(Op::GetStatus, 1);
    let chunk_b = chunk_b.build();

    let mut vm = VM::new(chunk_a);
    vm.set_shell_host(Box::new(BeginEndCountingHost(64)));
    vm.register_builtin(100, builtin_request_halt);
    let _ = vm.run();
    assert_eq!(vm.last_status, 64);
    assert_eq!(SUBSHELL_BEGIN_CALLS.load(Ordering::SeqCst), 1);

    vm.reset(chunk_b);
    match vm.run() {
        VMResult::Ok(Value::Status(64)) => {}
        other => panic!(
            "after halt+reset, host MUST still drive subshell_end(Some(64)), got {:?}",
            other
        ),
    }
    assert_eq!(SUBSHELL_BEGIN_CALLS.load(Ordering::SeqCst), 2);
}


#[test]
fn request_halt_stops_before_jump_if_false_keep() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::LoadInt(2), 1);
    b.emit(Op::NumEq, 1);
    let jmp = b.emit(Op::JumpIfFalseKeep(0), 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    b.patch_jump(jmp, b.current_pos());
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(
        AFTER_HALT_COUNT.load(Ordering::SeqCst),
        0,
        "JumpIfFalseKeep target MUST NOT run after request_halt"
    );
}

#[test]
fn subshell_end_exit_code_2_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(2)));
    match vm.run() {
        VMResult::Ok(Value::Status(2)) => {}
        other => panic!("exit code 2 MUST propagate as Status(2), got {:?}", other),
    }
}


#[test]
fn request_halt_after_pushframe_popframe_not_reached() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::PushFrame, 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::PopFrame, 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(
        AFTER_HALT_COUNT.load(Ordering::SeqCst),
        0,
        "PopFrame after halt MUST NOT run"
    );
}

#[test]
fn subshell_end_with_loadint_only_on_stack_returns_int() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(77), 1);
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(15)));
    match vm.run() {
        VMResult::Ok(Value::Int(77)) => {}
        other => panic!(
            "SubshellEnd MUST NOT push — sole stack value stays Int(77), got {:?}",
            other
        ),
    }
    assert_eq!(vm.last_status, 15);
}


#[test]
fn request_halt_stops_before_loadint() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::LoadInt(999), 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    match vm.run() {
        VMResult::Ok(Value::Int(0)) => {}
        other => panic!("halt MUST return Int(0) not LoadInt(999), got {:?}", other),
    }
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_getstatus_twice_reflects_updated_status() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::GetStatus, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.last_status = 2;
    vm.set_shell_host(Box::new(StatusReturningHost(8)));
    match vm.run() {
        VMResult::Ok(Value::Status(8)) => {}
        other => panic!(
            "second GetStatus MUST read subshell-updated last_status(8), got {:?}",
            other
        ),
    }
}


#[test]
fn subshell_four_end_ops_incrementing_status() {
    let mut b = ChunkBuilder::new();
    for _ in 0..4 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(IncrementingSubshellHost { next: 20 }));
    match vm.run() {
        VMResult::Ok(Value::Status(23)) => {}
        other => panic!(
            "fourth subshell_end(Some(23)) MUST win, got {:?}",
            other
        ),
    }
}

#[test]
fn request_halt_after_pipeline_begin_before_end() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    PIPELINE_END_CALLS.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::PipelineBegin(2), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::PipelineEnd, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(CountingPipelineHost));
    vm.register_builtin(100, builtin_request_halt);
    let _ = vm.run();
    assert_eq!(
        PIPELINE_END_CALLS.load(Ordering::SeqCst),
        0,
        "PipelineEnd MUST NOT run after halt even when PipelineBegin already ran"
    );
}


#[test]
fn subshell_setstatus_then_end_getstatus_chain() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(10), 1);
    b.emit(Op::SetStatus, 1);
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(20)));
    match vm.run() {
        VMResult::Ok(Value::Status(20)) => {}
        other => panic!(
            "subshell_end(Some(20)) MUST override prior SetStatus(10), got {:?}",
            other
        ),
    }
}

#[test]
fn request_halt_stops_subshell_begin_after_loadint() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    SUBSHELL_BEGIN_CALLS.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::SubshellBegin, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(BeginEndCountingHost(1)));
    vm.register_builtin(100, builtin_request_halt);
    let _ = vm.run();
    assert_eq!(
        SUBSHELL_BEGIN_CALLS.load(Ordering::SeqCst),
        0,
        "SubshellBegin after halt MUST NOT run even when prior LoadInt completed"
    );
}


#[test]
fn request_halt_only_wide_handler_no_narrow() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    static ONLY_WIDE: AtomicU32 = AtomicU32::new(0);
    let mut b = ChunkBuilder::new();
    b.emit(Op::ExtendedWide(3, 1), 1);
    b.emit(Op::Extended(4, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_extension_wide_handler(Box::new(|vm, _id, _payload| {
        ONLY_WIDE.fetch_add(1, Ordering::SeqCst);
        vm.request_halt();
    }));
    let _ = vm.run();
    assert_eq!(ONLY_WIDE.load(Ordering::SeqCst), 1);
}

#[test]
fn subshell_end_some_zero_after_some_negative() {
    struct NegThenZeroHost {
        calls: u32,
    }
    impl ShellHost for NegThenZeroHost {
        fn subshell_end(&mut self) -> Option<i32> {
            self.calls += 1;
            match self.calls {
                1 => Some(-1),
                2 => Some(0),
                _ => None,
            }
        }
    }
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(NegThenZeroHost { calls: 0 }));
    match vm.run() {
        VMResult::Ok(Value::Status(0)) => {}
        other => panic!(
            "second subshell_end(Some(0)) MUST overwrite Some(-1), got {:?}",
            other
        ),
    }
}


#[test]
fn request_halt_double_pre_run_reset_then_executes() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let noop = ChunkBuilder::new();
    let noop_chunk = noop.build();
    let mut run_b = ChunkBuilder::new();
    run_b.emit(Op::CallBuiltin(7, 0), 1);
    let run_chunk = run_b.build();

    let mut vm = VM::new(noop_chunk.clone());
    vm.register_builtin(7, builtin_touched);
    vm.request_halt();
    assert!(matches!(vm.run(), VMResult::Halted));
    vm.reset(noop_chunk);
    vm.request_halt();
    assert!(matches!(vm.run(), VMResult::Halted));
    vm.reset(run_chunk);
    let _ = vm.run();
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 1);
}

#[test]
fn subshell_some_max_then_none_preserves_max() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(FirstSomeThenNoneHost {
        calls: 0,
        status: i32::MAX,
    }));
    match vm.run() {
        VMResult::Ok(Value::Status(i32::MAX)) => {}
        other => panic!(
            "None after Some(MAX) MUST preserve last_status, got {:?}",
            other
        ),
    }
}


#[test]
fn request_halt_callee_return_not_reached() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    let name = b.add_name("f");
    b.emit(Op::Call(name, 0), 1);
    b.emit(Op::LoadInt(300), 1);
    let skip = b.emit(Op::Jump(0), 1);
    let entry = b.current_pos();
    b.add_sub_entry(name, entry);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::Return, 1);
    b.patch_jump(skip, b.current_pos());
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    match vm.run() {
        VMResult::Ok(Value::Int(0)) => {}
        other => panic!("callee Return MUST NOT run after halt, got {:?}", other),
    }
}

#[test]
fn subshell_one_begin_two_end_second_status_wins() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(IncrementingSubshellHost { next: 10 }));
    match vm.run() {
        VMResult::Ok(Value::Status(11)) => {}
        other => panic!(
            "second SubshellEnd(Some(11)) MUST win over first Some(10), got {:?}",
            other
        ),
    }
}


#[test]
fn request_halt_reset_extension_handler_runs_on_new_chunk() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    static EXT_RUNS: AtomicU32 = AtomicU32::new(0);
    let halt_chunk = ChunkBuilder::new().build();
    let mut ext_b = ChunkBuilder::new();
    ext_b.emit(Op::Extended(8, 1), 1);
    let ext_chunk = ext_b.build();

    let mut vm = VM::new(halt_chunk);
    vm.set_extension_handler(Box::new(|_vm, _id, _arg| {
        EXT_RUNS.fetch_add(1, Ordering::SeqCst);
    }));
    vm.request_halt();
    let _ = vm.run();
    EXT_RUNS.store(0, Ordering::SeqCst);
    vm.reset(ext_chunk);
    let _ = vm.run();
    assert_eq!(EXT_RUNS.load(Ordering::SeqCst), 1);
}

#[test]
fn subshell_after_pipelineend_getstatus_reads_subshell_status() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::PipelineEnd, 1);
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(PipelineThenSubshellHost));
    match vm.run() {
        VMResult::Ok(Value::Status(9)) => {}
        other => panic!(
            "GetStatus after subshell MUST read 9 not pipeline 3, got {:?}",
            other
        ),
    }
}


#[test]
fn subshell_halt_reset_getstatus_full_chain() {
    let mut run_b = ChunkBuilder::new();
    run_b.emit(Op::LoadInt(50), 1);
    run_b.emit(Op::SetStatus, 1);
    run_b.emit(Op::SubshellBegin, 1);
    run_b.emit(Op::SubshellEnd, 1);
    run_b.emit(Op::CallBuiltin(100, 0), 1);
    let run_chunk = run_b.build();

    let mut status_b = ChunkBuilder::new();
    status_b.emit(Op::GetStatus, 1);
    let status_chunk = status_b.build();

    let mut vm = VM::new(run_chunk);
    vm.set_shell_host(Box::new(StatusReturningHost(60)));
    vm.register_builtin(100, builtin_request_halt);
    let _ = vm.run();
    assert_eq!(vm.last_status, 60);

    vm.reset(status_chunk);
    match vm.run() {
        VMResult::Ok(Value::Status(0)) => {}
        other => panic!(
            "reset after subshell+halt MUST zero last_status before GetStatus, got {:?}",
            other
        ),
    }
}

#[test]
fn subshell_two_begin_end_pairs_last_status_wins() {
    let mut b = ChunkBuilder::new();
    for _ in 0..2 {
        b.emit(Op::SubshellBegin, 1);
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(AlternatingStatusHost { calls: 0 }));
    match vm.run() {
        VMResult::Ok(Value::Status(9)) => {}
        other => panic!(
            "second pair subshell_end(Some(9)) MUST win, got {:?}",
            other
        ),
    }
}


#[test]
fn request_halt_stops_before_setstatus_op() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::LoadInt(88), 1);
    b.emit(Op::SetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.last_status = 1;
    vm.register_builtin(100, builtin_request_halt);
    let _ = vm.run();
    assert_eq!(
        vm.last_status, 1,
        "SetStatus MUST NOT run after request_halt"
    );
}

#[test]
fn subshell_end_exit_code_127_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(127)));
    match vm.run() {
        VMResult::Ok(Value::Status(127)) => {}
        other => panic!("exit code 127 MUST propagate, got {:?}", other),
    }
}


#[test]
fn request_halt_after_getstatus_before_second_getstatus() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(55), 1);
    b.emit(Op::SetStatus, 1);
    b.emit(Op::GetStatus, 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    match vm.run() {
        VMResult::Ok(Value::Int(0)) => {}
        other => panic!(
            "halt MUST return builtin Int(0) not second GetStatus(55), got {:?}",
            other
        ),
    }
    assert_eq!(vm.last_status, 55);
}

#[test]
fn subshell_end_two_loadints_top_value_returned() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::LoadInt(2), 1);
    b.emit(Op::SubshellEnd, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(4)));
    match vm.run() {
        VMResult::Ok(Value::Int(2)) => {}
        other => panic!(
            "SubshellEnd MUST NOT push — run() pops top Int(2), got {:?}",
            other
        ),
    }
}


#[test]
fn request_halt_between_pipeline_begin_and_stage() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    PIPELINE_STAGE_CALLS.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::PipelineBegin(2), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::PipelineStage, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(CountingPipelineStageHost));
    vm.register_builtin(100, builtin_request_halt);
    let _ = vm.run();
    assert_eq!(
        PIPELINE_STAGE_CALLS.load(Ordering::SeqCst),
        0,
        "PipelineStage MUST NOT run after request_halt"
    );
}

#[test]
fn subshell_end_invokes_host_every_time_regardless_of_value() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    SUBSHELL_END_CALLS.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::SubshellEnd, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(CountingSubshellEndHost(5)));
    let _ = vm.run();
    assert_eq!(
        SUBSHELL_END_CALLS.load(Ordering::SeqCst),
        2,
        "each SubshellEnd MUST call host even when returning same status"
    );
    assert_eq!(vm.last_status, 5);
}


#[test]
fn request_halt_extension_handler_reads_subshell_last_status() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    static SEEN_STATUS: AtomicU32 = AtomicU32::new(0);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::Extended(5, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(42)));
    vm.set_extension_handler(Box::new(|vm, _id, _arg| {
        SEEN_STATUS.store(vm.last_status as u32, Ordering::SeqCst);
        vm.request_halt();
    }));
    let _ = vm.run();
    assert_eq!(
        SEEN_STATUS.load(Ordering::SeqCst),
        42,
        "extension handler MUST see subshell-updated last_status before halt"
    );
}

#[test]
fn subshell_end_exit_code_126_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(126)));
    match vm.run() {
        VMResult::Ok(Value::Status(126)) => {}
        other => panic!("exit code 126 MUST propagate, got {:?}", other),
    }
}


#[test]
fn subshell_setstatus_after_end_overwrites_subshell_status() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::LoadInt(3), 1);
    b.emit(Op::SetStatus, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(8)));
    match vm.run() {
        VMResult::Ok(Value::Status(3)) => {}
        other => panic!(
            "SetStatus(3) after subshell MUST overwrite last_status(8), got {:?}",
            other
        ),
    }
}

#[test]
fn request_halt_stops_before_call_op() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    let name = b.add_name("fn");
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::Call(name, 0), 1);
    let skip = b.emit(Op::Jump(0), 1);
    let entry = b.current_pos();
    b.add_sub_entry(name, entry);
    b.emit(Op::CallBuiltin(7, 0), 1);
    b.emit(Op::ReturnValue, 1);
    b.patch_jump(skip, b.current_pos());
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    let _ = vm.run();
    assert_eq!(
        TOUCHED.load(Ordering::SeqCst),
        0,
        "Call op MUST NOT run after request_halt"
    );
}


#[test]
fn request_halt_double_reset_then_run() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let noop = ChunkBuilder::new().build();
    let mut run_b = ChunkBuilder::new();
    run_b.emit(Op::CallBuiltin(7, 0), 1);
    let run_chunk = run_b.build();

    let mut vm = VM::new(noop.clone());
    vm.register_builtin(7, builtin_touched);
    vm.request_halt();
    let _ = vm.run();
    vm.reset(noop.clone());
    vm.reset(run_chunk);
    let _ = vm.run();
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 1);
}

#[test]
fn subshell_end_after_begin_only_pair_updates_status() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.last_status = 0;
    vm.set_shell_host(Box::new(StatusReturningHost(33)));
    match vm.run() {
        VMResult::Ok(Value::Status(33)) => {}
        other => panic!(
            "SubshellEnd MUST update status even with trailing SubshellBegin, got {:?}",
            other
        ),
    }
}


#[test]
fn subshell_getstatus_subshell_getstatus_two_updates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::GetStatus, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.last_status = 1;
    vm.set_shell_host(Box::new(AlternatingStatusHost { calls: 0 }));
    match vm.run() {
        VMResult::Ok(Value::Status(9)) => {}
        other => panic!(
            "final GetStatus MUST read second subshell Some(9), got {:?}",
            other
        ),
    }
}

#[test]
fn request_halt_nested_two_level_call_stops_at_halt() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    let inner = b.add_name("inner");
    let outer = b.add_name("outer");
    b.emit(Op::Call(outer, 0), 1);
    b.emit(Op::LoadInt(500), 1);
    let skip_outer = b.emit(Op::Jump(0), 1);
    let outer_entry = b.current_pos();
    b.add_sub_entry(outer, outer_entry);
    b.emit(Op::Call(inner, 0), 1);
    b.emit(Op::ReturnValue, 1);
    let skip_inner = b.emit(Op::Jump(0), 1);
    let inner_entry = b.current_pos();
    b.add_sub_entry(inner, inner_entry);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::ReturnValue, 1);
    b.patch_jump(skip_inner, b.current_pos());
    b.patch_jump(skip_outer, b.current_pos());

    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    match vm.run() {
        VMResult::Ok(Value::Int(0)) => {}
        other => panic!("two-level nested halt MUST return builtin result, got {:?}", other),
    }
}


#[test]
fn subshell_five_end_incrementing_status() {
    let mut b = ChunkBuilder::new();
    for _ in 0..5 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(IncrementingSubshellHost { next: 1 }));
    match vm.run() {
        VMResult::Ok(Value::Status(5)) => {}
        other => panic!("fifth subshell_end(Some(5)) MUST win, got {:?}", other),
    }
}

#[test]
fn request_halt_pre_run_many_ops_none_execute() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(7, 0), 1);
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::LoadInt(2), 1);
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(99)));
    vm.register_builtin(7, builtin_touched);
    vm.request_halt();
    let _ = vm.run();
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 0);
}


#[test]
fn request_halt_after_getstatus_on_stack_returns_builtin_not_status() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(55), 1);
    b.emit(Op::SetStatus, 1);
    b.emit(Op::GetStatus, 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    match vm.run() {
        VMResult::Ok(Value::Int(0)) => {}
        other => panic!(
            "halt MUST return Int(0) not Status(55) from stack below, got {:?}",
            other
        ),
    }
}

#[test]
fn subshell_begin_end_halt_begin_count_is_one() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    SUBSHELL_BEGIN_CALLS.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(BeginEndCountingHost(70)));
    vm.register_builtin(100, builtin_request_halt);
    let _ = vm.run();
    assert_eq!(SUBSHELL_BEGIN_CALLS.load(Ordering::SeqCst), 1);
    assert_eq!(vm.last_status, 70);
}


#[test]
fn subshell_halt_reset_subshell_halt_reset_getstatus() {
    let mut sub_b = ChunkBuilder::new();
    sub_b.emit(Op::SubshellBegin, 1);
    sub_b.emit(Op::SubshellEnd, 1);
    sub_b.emit(Op::CallBuiltin(100, 0), 1);
    let sub_chunk = sub_b.build();

    let mut status_b = ChunkBuilder::new();
    status_b.emit(Op::GetStatus, 1);
    let status_chunk = status_b.build();

    let mut vm = VM::new(sub_chunk.clone());
    vm.set_shell_host(Box::new(StatusReturningHost(81)));
    vm.register_builtin(100, builtin_request_halt);
    let _ = vm.run();
    assert_eq!(vm.last_status, 81);

    vm.reset(sub_chunk);
    let _ = vm.run();
    assert_eq!(vm.last_status, 81);

    vm.reset(status_chunk);
    match vm.run() {
        VMResult::Ok(Value::Status(0)) => {}
        other => panic!(
            "final reset MUST clear last_status before GetStatus, got {:?}",
            other
        ),
    }
}

#[test]
fn subshell_none_none_some_via_third_host() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.last_status = 0;
    vm.set_shell_host(Box::new(ThirdSomeSubshellHost { calls: 0 }));
    match vm.run() {
        VMResult::Ok(Value::Status(30)) => {}
        other => panic!(
            "ThirdSomeSubshellHost MUST apply Some(30) on third end, got {:?}",
            other
        ),
    }
}


#[test]
fn request_halt_after_subshell_begin_before_end() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    SUBSHELL_BEGIN_CALLS.store(0, Ordering::SeqCst);
    SUBSHELL_END_CALLS.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::SubshellEnd, 1);
    let mut vm = VM::new(b.build());
    vm.last_status = 0;
    vm.set_shell_host(Box::new(BeginEndCountingHost(11)));
    vm.register_builtin(100, builtin_request_halt);
    let _ = vm.run();
    assert_eq!(SUBSHELL_BEGIN_CALLS.load(Ordering::SeqCst), 1);
    assert_eq!(
        SUBSHELL_END_CALLS.load(Ordering::SeqCst),
        0,
        "SubshellEnd MUST NOT run after halt even when SubshellBegin already ran"
    );
    assert_eq!(vm.last_status, 0);
}

#[test]
fn subshell_end_exit_code_124_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(124)));
    match vm.run() {
        VMResult::Ok(Value::Status(124)) => {}
        other => panic!("exit code 124 MUST propagate, got {:?}", other),
    }
}


#[test]
fn subshell_six_end_incrementing_status() {
    let mut b = ChunkBuilder::new();
    for _ in 0..6 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(IncrementingSubshellHost { next: 10 }));
    match vm.run() {
        VMResult::Ok(Value::Status(15)) => {}
        other => panic!("sixth subshell_end(Some(15)) MUST win, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_backward_jump_target() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    let fwd = b.emit(Op::Jump(0), 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    b.patch_jump(fwd, b.current_pos());
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(
        AFTER_HALT_COUNT.load(Ordering::SeqCst),
        0,
        "backward Jump target MUST NOT run after request_halt"
    );
}


#[test]
fn request_halt_wide_handler_reads_setstatus_last_status() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    static WIDE_SEEN: AtomicU32 = AtomicU32::new(0);
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(37), 1);
    b.emit(Op::SetStatus, 1);
    b.emit(Op::ExtendedWide(1, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_extension_wide_handler(Box::new(|vm, _id, _payload| {
        WIDE_SEEN.store(vm.last_status as u32, Ordering::SeqCst);
        vm.request_halt();
    }));
    let _ = vm.run();
    assert_eq!(
        WIDE_SEEN.load(Ordering::SeqCst),
        37,
        "ExtendedWide handler MUST see SetStatus-written last_status"
    );
}

#[test]
fn subshell_replaced_host_none_on_second_reset_chunk() {
    let mut b1 = ChunkBuilder::new();
    b1.emit(Op::SubshellEnd, 1);
    let chunk1 = b1.build();
    let mut b2 = ChunkBuilder::new();
    b2.emit(Op::SubshellEnd, 1);
    b2.emit(Op::GetStatus, 1);
    let chunk2 = b2.build();

    let mut vm = VM::new(chunk1);
    vm.set_shell_host(Box::new(StatusReturningHost(5)));
    let _ = vm.run();
    assert_eq!(vm.last_status, 5);

    vm.set_shell_host(Box::new(NoneHost));
    vm.reset(chunk2);
    match vm.run() {
        VMResult::Ok(Value::Status(0)) => {}
        other => panic!(
            "NoneHost after reset MUST leave last_status(0) not prior 5, got {:?}",
            other
        ),
    }
}


#[test]
fn request_halt_stops_before_top_level_return_value() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(88), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::ReturnValue, 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    match vm.run() {
        VMResult::Ok(Value::Int(0)) => {}
        other => panic!("ReturnValue(88) MUST NOT run after halt, got {:?}", other),
    }
}

#[test]
fn subshell_end_status_minus_two_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(-2)));
    match vm.run() {
        VMResult::Ok(Value::Status(-2)) => {}
        other => panic!("status -2 MUST propagate, got {:?}", other),
    }
}


#[test]
fn request_halt_triple_reset_then_run() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let noop = ChunkBuilder::new().build();
    let mut run_b = ChunkBuilder::new();
    run_b.emit(Op::CallBuiltin(7, 0), 1);
    let run_chunk = run_b.build();

    let mut vm = VM::new(noop.clone());
    vm.register_builtin(7, builtin_touched);
    vm.request_halt();
    let _ = vm.run();
    vm.reset(noop.clone());
    vm.reset(noop.clone());
    vm.reset(run_chunk);
    let _ = vm.run();
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 1);
}

#[test]
fn subshell_three_begins_one_end_still_applies_status() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    SUBSHELL_BEGIN_CALLS.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(BeginEndCountingHost(44)));
    match vm.run() {
        VMResult::Ok(Value::Status(44)) => {}
        other => panic!(
            "single SubshellEnd MUST apply after triple SubshellBegin, got {:?}",
            other
        ),
    }
    assert_eq!(SUBSHELL_BEGIN_CALLS.load(Ordering::SeqCst), 3);
}


#[test]
fn subshell_getstatus_loadint_getstatus_final_is_loadint() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    b.emit(Op::LoadInt(9), 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(7)));
    match vm.run() {
        VMResult::Ok(Value::Status(7)) => {}
        other => panic!(
            "top GetStatus(7) MUST be below LoadInt(9) — run pops Status(7), got {:?}",
            other
        ),
    }
}

#[test]
fn request_halt_between_two_subshell_begins() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    SUBSHELL_BEGIN_CALLS.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(BeginEndCountingHost(55)));
    vm.register_builtin(100, builtin_request_halt);
    let _ = vm.run();
    assert_eq!(
        SUBSHELL_BEGIN_CALLS.load(Ordering::SeqCst),
        1,
        "second SubshellBegin MUST NOT run after halt"
    );
    assert_eq!(vm.last_status, 0);
}


struct MaxThenMinHost {
    calls: u32,
}

impl ShellHost for MaxThenMinHost {
    fn subshell_end(&mut self) -> Option<i32> {
        self.calls += 1;
        match self.calls {
            1 => Some(i32::MAX),
            2 => Some(i32::MIN),
            _ => None,
        }
    }
}

#[test]
fn subshell_max_then_min_second_wins() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(MaxThenMinHost { calls: 0 }));
    match vm.run() {
        VMResult::Ok(Value::Status(i32::MIN)) => {}
        other => panic!(
            "second subshell_end(MIN) MUST overwrite first MAX, got {:?}",
            other
        ),
    }
}

#[test]
fn request_halt_halt_flag_persists_after_ok_return() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    assert!(matches!(vm.run(), VMResult::Ok(Value::Int(0))));
    let _ = vm.run();
    assert_eq!(
        TOUCHED.load(Ordering::SeqCst),
        0,
        "halted flag MUST persist after Ok return from first run()"
    );
}


#[test]
fn subshell_incrementing_from_100_to_104() {
    let mut b = ChunkBuilder::new();
    for _ in 0..5 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(IncrementingSubshellHost { next: 100 }));
    match vm.run() {
        VMResult::Ok(Value::Status(104)) => {}
        other => panic!("fifth Some(104) MUST win, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_pipeline_end_without_begin() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    PIPELINE_END_CALLS.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::PipelineEnd, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(CountingPipelineHost));
    vm.register_builtin(100, builtin_request_halt);
    let _ = vm.run();
    assert_eq!(PIPELINE_END_CALLS.load(Ordering::SeqCst), 0);
}


#[test]
fn subshell_halt_subshell_getstatus_same_vm_no_reset() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(92)));
    vm.register_builtin(100, builtin_request_halt);
    let _ = vm.run();
    assert_eq!(vm.last_status, 92);
    match vm.run() {
        VMResult::Ok(Value::Int(0)) | VMResult::Halted => {}
        other => panic!("second run without reset unexpected {:?}", other),
    }
}

#[test]
fn request_halt_extension_handler_receives_id_and_arg() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    static SEEN_ID: AtomicU32 = AtomicU32::new(0);
    static SEEN_ARG: AtomicU32 = AtomicU32::new(0);
    let mut b = ChunkBuilder::new();
    b.emit(Op::Extended(42, 7), 1);
    let mut vm = VM::new(b.build());
    vm.set_extension_handler(Box::new(|vm, id, arg| {
        SEEN_ID.store(id as u32, Ordering::SeqCst);
        SEEN_ARG.store(arg as u32, Ordering::SeqCst);
        vm.request_halt();
    }));
    let _ = vm.run();
    assert_eq!(SEEN_ID.load(Ordering::SeqCst), 42);
    assert_eq!(SEEN_ARG.load(Ordering::SeqCst), 7);
}


#[test]
fn full_lifecycle_setstatus_subshell_halt_reset_getstatus() {
    let mut run_b = ChunkBuilder::new();
    run_b.emit(Op::LoadInt(41), 1);
    run_b.emit(Op::SetStatus, 1);
    run_b.emit(Op::SubshellEnd, 1);
    run_b.emit(Op::CallBuiltin(100, 0), 1);
    let run_chunk = run_b.build();

    let mut status_b = ChunkBuilder::new();
    status_b.emit(Op::GetStatus, 1);
    let status_chunk = status_b.build();

    let mut vm = VM::new(run_chunk);
    vm.set_shell_host(Box::new(StatusReturningHost(83)));
    vm.register_builtin(100, builtin_request_halt);
    let _ = vm.run();
    assert_eq!(vm.last_status, 83);

    vm.reset(status_chunk);
    match vm.run() {
        VMResult::Ok(Value::Status(0)) => {}
        other => panic!("reset MUST clear subshell+setstatus state, got {:?}", other),
    }
}

#[test]
fn subshell_ten_end_incrementing_from_zero() {
    let mut b = ChunkBuilder::new();
    for _ in 0..10 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(IncrementingSubshellHost { next: 0 }));
    match vm.run() {
        VMResult::Ok(Value::Status(9)) => {}
        other => panic!("tenth subshell_end(Some(9)) MUST win, got {:?}", other),
    }
}


#[test]
fn subshell_end_exit_code_125_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(125)));
    match vm.run() {
        VMResult::Ok(Value::Status(125)) => {}
        other => panic!("exit code 125 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_between_two_subshell_ends() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    SUBSHELL_END_CALLS.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::SubshellEnd, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(CountingSubshellEndHost(3)));
    vm.register_builtin(100, builtin_request_halt);
    let _ = vm.run();
    assert_eq!(
        SUBSHELL_END_CALLS.load(Ordering::SeqCst),
        1,
        "second SubshellEnd MUST NOT run after request_halt"
    );
    assert_eq!(vm.last_status, 3);
}


#[test]
fn request_halt_three_builtins_before_halt_all_run() {
    static PRE: AtomicU32 = AtomicU32::new(0);
    fn pre(_vm: &mut VM, _argc: u8) -> Value {
        PRE.fetch_add(1, Ordering::SeqCst);
        Value::Int(0)
    }
    fn halt(vm: &mut VM, _argc: u8) -> Value {
        PRE.fetch_add(1, Ordering::SeqCst);
        vm.request_halt();
        Value::Int(0)
    }
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    PRE.store(0, Ordering::SeqCst);
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(10, 0), 1);
    b.emit(Op::CallBuiltin(11, 0), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(10, pre);
    vm.register_builtin(11, pre);
    vm.register_builtin(100, halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(PRE.load(Ordering::SeqCst), 3);
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_seven_end_incrementing_status() {
    let mut b = ChunkBuilder::new();
    for _ in 0..7 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(IncrementingSubshellHost { next: 0 }));
    match vm.run() {
        VMResult::Ok(Value::Status(6)) => {}
        other => panic!("seventh subshell_end(Some(6)) MUST win, got {:?}", other),
    }
}


#[test]
fn request_halt_wide_extension_receives_id_and_payload() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    static SEEN_ID: AtomicU32 = AtomicU32::new(0);
    static SEEN_PAYLOAD: AtomicU32 = AtomicU32::new(0);
    let mut b = ChunkBuilder::new();
    b.emit(Op::ExtendedWide(99, 0xABCD), 1);
    let mut vm = VM::new(b.build());
    vm.set_extension_wide_handler(Box::new(|vm, id, payload| {
        SEEN_ID.store(id as u32, Ordering::SeqCst);
        SEEN_PAYLOAD.store(payload as u32, Ordering::SeqCst);
        vm.request_halt();
    }));
    let _ = vm.run();
    assert_eq!(SEEN_ID.load(Ordering::SeqCst), 99);
    assert_eq!(SEEN_PAYLOAD.load(Ordering::SeqCst), 0xABCD);
}

#[test]
fn subshell_pipeline_stage_does_not_change_last_status() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    PIPELINE_STAGE_CALLS.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::PipelineBegin(2), 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::PipelineStage, 1);
    b.emit(Op::GetStatus, 1);
    struct StageAndSubshellHost;
    impl ShellHost for StageAndSubshellHost {
        fn pipeline_stage(&mut self) {
            PIPELINE_STAGE_CALLS.fetch_add(1, Ordering::SeqCst);
        }
        fn subshell_end(&mut self) -> Option<i32> {
            Some(12)
        }
    }
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StageAndSubshellHost));
    match vm.run() {
        VMResult::Ok(Value::Status(12)) => {}
        other => panic!(
            "PipelineStage MUST NOT overwrite subshell last_status(12), got {:?}",
            other
        ),
    }
    assert_eq!(PIPELINE_STAGE_CALLS.load(Ordering::SeqCst), 1);
}


#[test]
fn request_halt_stops_before_second_push_frame() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::PushFrame, 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::PushFrame, 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_130_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(130)));
    match vm.run() {
        VMResult::Ok(Value::Status(130)) => {}
        other => panic!("exit code 130 MUST propagate, got {:?}", other),
    }
}


#[test]
fn request_halt_reset_mid_subshell_chunk_allows_end() {
    let mut partial = ChunkBuilder::new();
    partial.emit(Op::SubshellBegin, 1);
    partial.emit(Op::CallBuiltin(100, 0), 1);
    let partial = partial.build();

    let mut full = ChunkBuilder::new();
    full.emit(Op::SubshellBegin, 1);
    full.emit(Op::SubshellEnd, 1);
    full.emit(Op::GetStatus, 1);
    let full = full.build();

    let mut vm = VM::new(partial);
    vm.set_shell_host(Box::new(StatusReturningHost(47)));
    vm.register_builtin(100, builtin_request_halt);
    let _ = vm.run();
    assert_eq!(vm.last_status, 0);

    vm.reset(full);
    match vm.run() {
        VMResult::Ok(Value::Status(47)) => {}
        other => panic!(
            "reset MUST allow subshell_end after halted partial chunk, got {:?}",
            other
        ),
    }
}

#[test]
fn subshell_getstatus_twice_without_intervening_end() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(19)));
    match vm.run() {
        VMResult::Ok(Value::Status(19)) => {}
        other => panic!(
            "both GetStatus MUST read subshell-updated 19, got {:?}",
            other
        ),
    }
}


#[test]
fn request_halt_after_subshell_begin_end_not_reached() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    SUBSHELL_END_CALLS.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(CountingSubshellEndHost(25)));
    vm.register_builtin(100, builtin_request_halt);
    let _ = vm.run();
    assert_eq!(
        SUBSHELL_END_CALLS.load(Ordering::SeqCst),
        1,
        "only first SubshellEnd MUST run — second pair blocked by halt"
    );
    assert_eq!(vm.last_status, 25);
}

#[test]
fn subshell_end_loadint_getstatus_stack_order() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::LoadInt(66), 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(4)));
    match vm.run() {
        VMResult::Ok(Value::Status(4)) => {}
        other => panic!(
            "GetStatus(4) MUST be stack top over LoadInt(66), got {:?}",
            other
        ),
    }
}


struct MinThenMaxHost {
    calls: u32,
}

impl ShellHost for MinThenMaxHost {
    fn subshell_end(&mut self) -> Option<i32> {
        self.calls += 1;
        match self.calls {
            1 => Some(i32::MIN),
            2 => Some(i32::MAX),
            _ => None,
        }
    }
}

#[test]
fn subshell_min_then_max_second_wins() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(MinThenMaxHost { calls: 0 }));
    match vm.run() {
        VMResult::Ok(Value::Status(i32::MAX)) => {}
        other => panic!(
            "second subshell_end(MAX) MUST overwrite MIN, got {:?}",
            other
        ),
    }
}

#[test]
fn request_halt_four_consecutive_runs_never_resume() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    for _ in 0..4 {
        let _ = vm.run();
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}


#[test]
fn subshell_twelve_end_incrementing_from_one() {
    let mut b = ChunkBuilder::new();
    for _ in 0..12 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(IncrementingSubshellHost { next: 1 }));
    match vm.run() {
        VMResult::Ok(Value::Status(12)) => {}
        other => panic!("twelfth subshell_end(Some(12)) MUST win, got {:?}", other),
    }
}

#[test]
fn request_halt_extended_without_handler_continues() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::Extended(1, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(7, builtin_touched);
    let _ = vm.run();
    assert_eq!(
        TOUCHED.load(Ordering::SeqCst),
        1,
        "CallBuiltin MUST run when Extended has no handler registered"
    );
}


#[test]
fn subshell_after_pipeline_begin_stage_without_end() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    PIPELINE_STAGE_CALLS.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::PipelineBegin(2), 1);
    b.emit(Op::PipelineStage, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    struct StageSubshellHost;
    impl ShellHost for StageSubshellHost {
        fn pipeline_stage(&mut self) {
            PIPELINE_STAGE_CALLS.fetch_add(1, Ordering::SeqCst);
        }
        fn subshell_end(&mut self) -> Option<i32> {
            Some(28)
        }
    }
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StageSubshellHost));
    match vm.run() {
        VMResult::Ok(Value::Status(28)) => {}
        other => panic!(
            "subshell_end(Some(28)) MUST apply without PipelineEnd, got {:?}",
            other
        ),
    }
    assert_eq!(PIPELINE_STAGE_CALLS.load(Ordering::SeqCst), 1);
}

#[test]
fn request_halt_with_host_set_builtin_still_halts() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    SUBSHELL_END_CALLS.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::SubshellEnd, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(CountingSubshellEndHost(99)));
    vm.register_builtin(100, builtin_request_halt);
    let _ = vm.run();
    assert_eq!(SUBSHELL_END_CALLS.load(Ordering::SeqCst), 0);
}


#[test]
fn full_lifecycle_halt_reset_subshell_getstatus_twice() {
    let halt_chunk = {
        let mut b = ChunkBuilder::new();
        b.emit(Op::CallBuiltin(100, 0), 1);
        b.build()
    };
    let sub_chunk = {
        let mut b = ChunkBuilder::new();
        b.emit(Op::SubshellEnd, 1);
        b.emit(Op::GetStatus, 1);
        b.build()
    };
    let status_chunk = {
        let mut b = ChunkBuilder::new();
        b.emit(Op::GetStatus, 1);
        b.build()
    };

    let mut vm = VM::new(halt_chunk);
    vm.set_shell_host(Box::new(StatusReturningHost(95)));
    vm.register_builtin(100, builtin_request_halt);
    let _ = vm.run();

    vm.reset(sub_chunk.clone());
    match vm.run() {
        VMResult::Ok(Value::Status(95)) => {}
        other => panic!("subshell after reset MUST use preserved host, got {:?}", other),
    }

    vm.reset(status_chunk);
    match vm.run() {
        VMResult::Ok(Value::Status(0)) => {}
        other => panic!("final reset MUST zero last_status, got {:?}", other),
    }
}

#[test]
fn subshell_twenty_end_incrementing_from_zero() {
    let mut b = ChunkBuilder::new();
    for _ in 0..20 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(IncrementingSubshellHost { next: 0 }));
    match vm.run() {
        VMResult::Ok(Value::Status(19)) => {}
        other => panic!("twentieth subshell_end(Some(19)) MUST win, got {:?}", other),
    }
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
