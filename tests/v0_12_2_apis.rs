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

// ─── Additional pin tests (handwritten) ───────────────────────────────────

#[test]
fn subshell_end_exit_code_3_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(3)));
    match vm.run() {
        VMResult::Ok(Value::Status(3)) => {}
        other => panic!("exit code 3 MUST propagate as Status(3), got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_num_ne_conditional_jump() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::LoadInt(2), 1);
    b.emit(Op::NumNe, 1);
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
        "NumNe/JumpIfTrue target MUST NOT run after request_halt"
    );
}

#[test]
fn subshell_end_exit_code_4_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(4)));
    match vm.run() {
        VMResult::Ok(Value::Status(4)) => {}
        other => panic!("exit code 4 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_triple_halt_still_blocks_following_builtin() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(50, 0), 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(50, builtin_triple_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(
        AFTER_HALT_COUNT.load(Ordering::SeqCst),
        0,
        "triple request_halt MUST still stop dispatch before next op"
    );
}

#[test]
fn subshell_end_exit_code_5_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(5)));
    match vm.run() {
        VMResult::Ok(Value::Status(5)) => {}
        other => panic!("exit code 5 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_halt_only_pushes_undef_and_blocks_loadint() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(61, 0), 1);
    b.emit(Op::LoadInt(777), 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(61, builtin_halt_only);
    vm.register_builtin(200, builtin_count_after_halt);
    match vm.run() {
        VMResult::Ok(Value::Undef) => {}
        other => panic!(
            "halt_only MUST return Undef on stack top, not LoadInt(777), got {:?}",
            other
        ),
    }
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_7_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(7)));
    match vm.run() {
        VMResult::Ok(Value::Status(7)) => {}
        other => panic!("exit code 7 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_pre_run_subshell_chunk_skips_host_until_reset() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    SUBSHELL_END_CALLS.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let chunk = b.build();
    let mut vm = VM::new(chunk.clone());
    vm.set_shell_host(Box::new(CountingSubshellEndHost(31)));
    vm.request_halt();
    match vm.run() {
        VMResult::Halted => {}
        other => panic!("pre-run halt on subshell chunk MUST return Halted, got {:?}", other),
    }
    assert_eq!(SUBSHELL_END_CALLS.load(Ordering::SeqCst), 0);
    vm.reset(chunk);
    match vm.run() {
        VMResult::Ok(Value::Status(31)) => {}
        other => panic!("after reset subshell_end MUST run, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_9_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(9)));
    match vm.run() {
        VMResult::Ok(Value::Status(9)) => {}
        other => panic!("exit code 9 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_load_true_after_halt_point() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::LoadTrue, 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_11_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(11)));
    match vm.run() {
        VMResult::Ok(Value::Status(11)) => {}
        other => panic!("exit code 11 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_load_false_after_halt_point() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::LoadFalse, 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_12_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(12)));
    match vm.run() {
        VMResult::Ok(Value::Status(12)) => {}
        other => panic!("exit code 12 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_between_two_pipeline_stages_blocks_second_stage() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    PIPELINE_STAGE_CALLS.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::PipelineBegin(3), 1);
    b.emit(Op::PipelineStage, 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::PipelineStage, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(CountingPipelineStageHost));
    vm.register_builtin(100, builtin_request_halt);
    let _ = vm.run();
    assert_eq!(
        PIPELINE_STAGE_CALLS.load(Ordering::SeqCst),
        1,
        "second PipelineStage MUST NOT run after request_halt"
    );
}

#[test]
fn subshell_end_exit_code_13_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(13)));
    match vm.run() {
        VMResult::Ok(Value::Status(13)) => {}
        other => panic!("exit code 13 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_after_first_pipeline_end_blocks_second_pipeline_end() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    PIPELINE_END_CALLS.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::PipelineEnd, 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::PipelineEnd, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(CountingPipelineHost));
    vm.register_builtin(100, builtin_request_halt);
    let _ = vm.run();
    assert_eq!(
        PIPELINE_END_CALLS.load(Ordering::SeqCst),
        1,
        "second PipelineEnd MUST NOT run after request_halt"
    );
}

#[test]
fn subshell_end_exit_code_14_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(14)));
    match vm.run() {
        VMResult::Ok(Value::Status(14)) => {}
        other => panic!("exit code 14 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_callee_with_two_args_halts_before_caller_loadint() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    let mut b = ChunkBuilder::new();
    let name = b.add_name("halt_callee");
    b.emit(Op::LoadInt(10), 1);
    b.emit(Op::LoadInt(20), 1);
    b.emit(Op::Call(name, 2), 1);
    b.emit(Op::LoadInt(999), 1);
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
            "halt in 2-arg callee MUST NOT run caller LoadInt(999), got {:?}",
            other
        ),
    }
}

#[test]
fn subshell_end_exit_code_15_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(15)));
    match vm.run() {
        VMResult::Ok(Value::Status(15)) => {}
        other => panic!("exit code 15 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_narrow_extension_blocks_following_wide_on_same_chunk() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    static NARROW_RAN: AtomicU32 = AtomicU32::new(0);
    static WIDE_RAN: AtomicU32 = AtomicU32::new(0);
    let mut b = ChunkBuilder::new();
    b.emit(Op::Extended(1, 0), 1);
    b.emit(Op::ExtendedWide(2, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_extension_handler(Box::new(|vm, _id, _arg| {
        NARROW_RAN.fetch_add(1, Ordering::SeqCst);
        vm.request_halt();
    }));
    vm.set_extension_wide_handler(Box::new(|_vm, _id, _payload| {
        WIDE_RAN.fetch_add(1, Ordering::SeqCst);
    }));
    let _ = vm.run();
    assert_eq!(NARROW_RAN.load(Ordering::SeqCst), 1);
    assert_eq!(WIDE_RAN.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_16_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(16)));
    match vm.run() {
        VMResult::Ok(Value::Status(16)) => {}
        other => panic!("exit code 16 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_wide_extension_blocks_following_narrow_on_same_chunk() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    static NARROW_RAN: AtomicU32 = AtomicU32::new(0);
    static WIDE_RAN: AtomicU32 = AtomicU32::new(0);
    let mut b = ChunkBuilder::new();
    b.emit(Op::ExtendedWide(3, 0), 1);
    b.emit(Op::Extended(4, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_extension_wide_handler(Box::new(|vm, _id, _payload| {
        WIDE_RAN.fetch_add(1, Ordering::SeqCst);
        vm.request_halt();
    }));
    vm.set_extension_handler(Box::new(|_vm, _id, _arg| {
        NARROW_RAN.fetch_add(1, Ordering::SeqCst);
    }));
    let _ = vm.run();
    assert_eq!(WIDE_RAN.load(Ordering::SeqCst), 1);
    assert_eq!(NARROW_RAN.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_17_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(17)));
    match vm.run() {
        VMResult::Ok(Value::Status(17)) => {}
        other => panic!("exit code 17 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_reset_after_halted_subshell_chunk_allows_fresh_getstatus_zero() {
    let mut run_b = ChunkBuilder::new();
    run_b.emit(Op::SubshellEnd, 1);
    run_b.emit(Op::CallBuiltin(100, 0), 1);
    let run_chunk = run_b.build();
    let mut status_b = ChunkBuilder::new();
    status_b.emit(Op::GetStatus, 1);
    let status_chunk = status_b.build();

    let mut vm = VM::new(run_chunk);
    vm.set_shell_host(Box::new(StatusReturningHost(52)));
    vm.register_builtin(100, builtin_request_halt);
    let _ = vm.run();
    assert_eq!(vm.last_status, 52);

    vm.reset(status_chunk);
    match vm.run() {
        VMResult::Ok(Value::Status(0)) => {}
        other => panic!("reset MUST clear last_status from halted subshell run, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_18_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(18)));
    match vm.run() {
        VMResult::Ok(Value::Status(18)) => {}
        other => panic!("exit code 18 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_pre_run_double_reset_then_subshell_end_runs() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    SUBSHELL_END_CALLS.store(0, Ordering::SeqCst);
    let noop = ChunkBuilder::new().build();
    let mut sub_b = ChunkBuilder::new();
    sub_b.emit(Op::SubshellEnd, 1);
    let sub_chunk = sub_b.build();

    let mut vm = VM::new(noop.clone());
    vm.set_shell_host(Box::new(CountingSubshellEndHost(64)));
    vm.request_halt();
    let _ = vm.run();
    vm.reset(noop);
    vm.reset(sub_chunk);
    let _ = vm.run();
    assert_eq!(SUBSHELL_END_CALLS.load(Ordering::SeqCst), 1);
    assert_eq!(vm.last_status, 64);
}

#[test]
fn subshell_end_exit_code_19_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(19)));
    match vm.run() {
        VMResult::Ok(Value::Status(19)) => {}
        other => panic!("exit code 19 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_top_level_return_on_nonempty_stack() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(44), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::Return, 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    match vm.run() {
        VMResult::Ok(Value::Int(0)) => {}
        other => panic!("Return MUST NOT run after halt, got {:?}", other),
    }
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_20_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(20)));
    match vm.run() {
        VMResult::Ok(Value::Status(20)) => {}
        other => panic!("exit code 20 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_with_two_ints_on_stack_returns_builtin_not_deeper_int() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(100), 1);
    b.emit(Op::LoadInt(200), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    match vm.run() {
        VMResult::Ok(Value::Int(0)) => {}
        other => panic!(
            "run() MUST pop only halt builtin result, not deeper Int(100), got {:?}",
            other
        ),
    }
}

#[test]
fn subshell_end_exit_code_21_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(21)));
    match vm.run() {
        VMResult::Ok(Value::Status(21)) => {}
        other => panic!("exit code 21 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_callbuiltin_three_args_reports_argc_before_stop() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::LoadInt(2), 1);
    b.emit(Op::LoadInt(3), 1);
    b.emit(Op::CallBuiltin(100, 3), 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_halt_report_argc);
    vm.register_builtin(200, builtin_count_after_halt);
    match vm.run() {
        VMResult::Ok(Value::Int(3)) => {}
        other => panic!("halt MUST push argc=3 before stopping, got {:?}", other),
    }
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_22_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(22)));
    match vm.run() {
        VMResult::Ok(Value::Status(22)) => {}
        other => panic!("exit code 22 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_second_subshell_begin_in_same_chunk_not_reached() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    SUBSHELL_BEGIN_CALLS.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::SubshellBegin, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(BeginEndCountingHost(0)));
    vm.register_builtin(100, builtin_request_halt);
    let _ = vm.run();
    assert_eq!(
        SUBSHELL_BEGIN_CALLS.load(Ordering::SeqCst),
        1,
        "second SubshellBegin MUST NOT run after halt"
    );
}

#[test]
fn subshell_end_exit_code_129_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(129)));
    match vm.run() {
        VMResult::Ok(Value::Status(129)) => {}
        other => panic!("exit code 129 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_between_subshell_end_and_getstatus_leaves_status_set() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(73)));
    vm.register_builtin(100, builtin_request_halt);
    match vm.run() {
        VMResult::Ok(Value::Int(0)) => {}
        other => panic!("GetStatus MUST NOT run after halt, got {:?}", other),
    }
    assert_eq!(vm.last_status, 73);
}

#[test]
fn subshell_end_exit_code_131_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(131)));
    match vm.run() {
        VMResult::Ok(Value::Status(131)) => {}
        other => panic!("exit code 131 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_five_consecutive_runs_never_advance_past_halt() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    for _ in 0..5 {
        let _ = vm.run();
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_132_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(132)));
    match vm.run() {
        VMResult::Ok(Value::Status(132)) => {}
        other => panic!("exit code 132 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_after_pipeline_begin_only_one_begin_call() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    PIPELINE_BEGIN_CALLS.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::PipelineBegin(2), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::PipelineBegin(3), 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(CountingPipelineBeginHost));
    vm.register_builtin(100, builtin_request_halt);
    let _ = vm.run();
    assert_eq!(
        PIPELINE_BEGIN_CALLS.load(Ordering::SeqCst),
        1,
        "second PipelineBegin MUST NOT run after halt"
    );
}

#[test]
fn subshell_end_exit_code_133_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(133)));
    match vm.run() {
        VMResult::Ok(Value::Status(133)) => {}
        other => panic!("exit code 133 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_blocks_loadint_between_two_subshell_ends() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    SUBSHELL_END_CALLS.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::LoadInt(555), 1);
    b.emit(Op::SubshellEnd, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(CountingSubshellEndHost(8)));
    vm.register_builtin(100, builtin_request_halt);
    let _ = vm.run();
    assert_eq!(
        SUBSHELL_END_CALLS.load(Ordering::SeqCst),
        1,
        "second SubshellEnd MUST NOT run after halt"
    );
    assert_eq!(vm.last_status, 8);
}

#[test]
fn subshell_end_exit_code_134_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(134)));
    match vm.run() {
        VMResult::Ok(Value::Status(134)) => {}
        other => panic!("exit code 134 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_extended_without_handler_still_blocks_following_loadint() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::Extended(9, 0), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::LoadInt(123), 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_status_minus_three_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(-3)));
    match vm.run() {
        VMResult::Ok(Value::Status(-3)) => {}
        other => panic!("status -3 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_jump_if_true_keep_with_true_condition_not_taken_before_halt() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(0), 1);
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::NumEq, 1);
    let jmp = b.emit(Op::JumpIfTrueKeep(0), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.patch_jump(jmp, b.current_pos());
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_status_minus_four_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(-4)));
    match vm.run() {
        VMResult::Ok(Value::Status(-4)) => {}
        other => panic!("status -4 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_jump_if_false_keep_with_false_condition_not_taken_before_halt() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::NumEq, 1);
    let jmp = b.emit(Op::JumpIfFalseKeep(0), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.patch_jump(jmp, b.current_pos());
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_eight_end_incrementing_from_thirty() {
    let mut b = ChunkBuilder::new();
    for _ in 0..8 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(IncrementingSubshellHost { next: 30 }));
    match vm.run() {
        VMResult::Ok(Value::Status(37)) => {}
        other => panic!("eighth subshell_end(Some(37)) MUST win, got {:?}", other),
    }
}

#[test]
fn request_halt_between_subshell_end_and_setstatus_preserves_subshell_status() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::LoadInt(99), 1);
    b.emit(Op::SetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(46)));
    vm.register_builtin(100, builtin_request_halt);
    let _ = vm.run();
    assert_eq!(
        vm.last_status, 46,
        "SetStatus MUST NOT run after halt — subshell status MUST stick"
    );
}

#[test]
fn subshell_nine_end_incrementing_from_forty() {
    let mut b = ChunkBuilder::new();
    for _ in 0..9 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(IncrementingSubshellHost { next: 40 }));
    match vm.run() {
        VMResult::Ok(Value::Status(48)) => {}
        other => panic!("ninth subshell_end(Some(48)) MUST win, got {:?}", other),
    }
}

#[test]
fn request_halt_reset_then_wide_handler_runs_on_fresh_chunk() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    static WIDE_TOUCHED: AtomicU32 = AtomicU32::new(0);
    let halt_chunk = {
        let mut b = ChunkBuilder::new();
        b.emit(Op::CallBuiltin(100, 0), 1);
        b.build()
    };
    let wide_chunk = {
        let mut b = ChunkBuilder::new();
        b.emit(Op::ExtendedWide(8, 0xBEEF), 1);
        b.build()
    };
    let mut vm = VM::new(halt_chunk);
    vm.set_extension_wide_handler(Box::new(|_vm, _id, _payload| {
        WIDE_TOUCHED.fetch_add(1, Ordering::SeqCst);
    }));
    vm.register_builtin(100, builtin_request_halt);
    let _ = vm.run();
    WIDE_TOUCHED.store(0, Ordering::SeqCst);
    vm.reset(wide_chunk);
    let _ = vm.run();
    assert_eq!(WIDE_TOUCHED.load(Ordering::SeqCst), 1);
}

#[test]
fn subshell_eleven_end_incrementing_from_two() {
    let mut b = ChunkBuilder::new();
    for _ in 0..11 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(IncrementingSubshellHost { next: 2 }));
    match vm.run() {
        VMResult::Ok(Value::Status(12)) => {}
        other => panic!("eleventh subshell_end(Some(12)) MUST win, got {:?}", other),
    }
}

#[test]
fn request_halt_subshell_begin_end_halt_reset_subshell_getstatus() {
    let mut sub_b = ChunkBuilder::new();
    sub_b.emit(Op::SubshellBegin, 1);
    sub_b.emit(Op::SubshellEnd, 1);
    sub_b.emit(Op::CallBuiltin(100, 0), 1);
    let sub_chunk = sub_b.build();
    let mut status_b = ChunkBuilder::new();
    status_b.emit(Op::GetStatus, 1);
    let status_chunk = status_b.build();

    let mut vm = VM::new(sub_chunk);
    vm.set_shell_host(Box::new(StatusReturningHost(58)));
    vm.register_builtin(100, builtin_request_halt);
    let _ = vm.run();
    assert_eq!(vm.last_status, 58);

    vm.reset(status_chunk);
    match vm.run() {
        VMResult::Ok(Value::Status(0)) => {}
        other => panic!("reset after subshell+halt MUST zero last_status, got {:?}", other),
    }
}

#[test]
fn subshell_thirteen_end_incrementing_from_seven() {
    let mut b = ChunkBuilder::new();
    for _ in 0..13 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(IncrementingSubshellHost { next: 7 }));
    match vm.run() {
        VMResult::Ok(Value::Status(19)) => {}
        other => panic!("thirteenth subshell_end(Some(19)) MUST win, got {:?}", other),
    }
}

#[test]
fn request_halt_callee_return_not_reached_when_halt_before_return_value() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    let mut b = ChunkBuilder::new();
    let name = b.add_name("inner_ret");
    b.emit(Op::Call(name, 0), 1);
    b.emit(Op::LoadInt(500), 1);
    let skip = b.emit(Op::Jump(0), 1);
    let entry = b.current_pos();
    b.add_sub_entry(name, entry);
    b.emit(Op::LoadInt(77), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::ReturnValue, 1);
    b.patch_jump(skip, b.current_pos());
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    match vm.run() {
        VMResult::Ok(Value::Int(0)) => {}
        other => panic!(
            "ReturnValue(77) MUST NOT run after halt in callee, got {:?}",
            other
        ),
    }
}

#[test]
fn subshell_fifteen_end_incrementing_from_twelve() {
    let mut b = ChunkBuilder::new();
    for _ in 0..15 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(IncrementingSubshellHost { next: 12 }));
    match vm.run() {
        VMResult::Ok(Value::Status(26)) => {}
        other => panic!("fifteenth subshell_end(Some(26)) MUST win, got {:?}", other),
    }
}

#[test]
fn request_halt_host_subshell_end_count_one_when_halt_after_first() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    SUBSHELL_END_CALLS.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::SubshellEnd, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(CountingSubshellEndHost(14)));
    vm.register_builtin(100, builtin_request_halt);
    let _ = vm.run();
    assert_eq!(SUBSHELL_END_CALLS.load(Ordering::SeqCst), 1);
}

#[test]
fn subshell_alternating_fourth_none_keeps_nine() {
    let mut b = ChunkBuilder::new();
    for _ in 0..4 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(AlternatingStatusHost { calls: 0 }));
    match vm.run() {
        VMResult::Ok(Value::Status(9)) => {}
        other => panic!(
            "AlternatingStatusHost fourth None MUST keep prior Some(9), got {:?}",
            other
        ),
    }
}

#[test]
fn request_halt_after_getstatus_leaves_prior_status_on_stack_unpopped() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(33), 1);
    b.emit(Op::SetStatus, 1);
    b.emit(Op::GetStatus, 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::LoadInt(44), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    match vm.run() {
        VMResult::Ok(Value::Int(0)) => {}
        other => panic!(
            "halt MUST return builtin Int(0), not LoadInt(44), got {:?}",
            other
        ),
    }
    assert_eq!(vm.last_status, 33);
}

#[test]
fn subshell_none_then_some_second_end_updates_from_eleven() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.last_status = 11;
    vm.set_shell_host(Box::new(NoneThenSomeHost { calls: 0 }));
    match vm.run() {
        VMResult::Ok(Value::Status(42)) => {}
        other => panic!(
            "NoneThenSomeHost second Some(42) MUST overwrite 11, got {:?}",
            other
        ),
    }
}

#[test]
fn request_halt_between_two_getstatus_ops_second_not_reached() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(25), 1);
    b.emit(Op::SetStatus, 1);
    b.emit(Op::GetStatus, 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    match vm.run() {
        VMResult::Ok(Value::Int(0)) => {}
        other => panic!(
            "second GetStatus MUST NOT run after halt, got {:?}",
            other
        ),
    }
}

#[test]
fn subshell_first_some_then_none_twice_preserves_eighty_eight() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::SubshellEnd, 1);
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
            "FirstSomeThenNoneHost MUST preserve Some(88) through later None, got {:?}",
            other
        ),
    }
}

#[test]
fn request_halt_pipeline_end_after_subshell_end_both_ran_status_from_subshell() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::PipelineEnd, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(PipelineThenSubshellHost));
    vm.register_builtin(100, builtin_request_halt);
    let _ = vm.run();
    assert_eq!(
        vm.last_status, 9,
        "subshell_end(Some(9)) MUST run before halt; PipelineEnd MUST NOT run after"
    );
}

#[test]
fn subshell_pipeline_end_before_subshell_end_getstatus_reads_subshell() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::PipelineEnd, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(PipelineThenSubshellHost));
    match vm.run() {
        VMResult::Ok(Value::Status(9)) => {}
        other => panic!(
            "GetStatus MUST read subshell_end(9) not pipeline_end(3), got {:?}",
            other
        ),
    }
}

#[test]
fn request_halt_after_open_pipeline_begin_blocks_pipeline_end() {
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
    assert_eq!(PIPELINE_END_CALLS.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_setstatus_between_two_ends_second_end_wins() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::LoadInt(50), 1);
    b.emit(Op::SetStatus, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(IncrementingSubshellHost { next: 100 }));
    match vm.run() {
        VMResult::Ok(Value::Status(101)) => {}
        other => panic!(
            "second subshell_end(Some(101)) MUST beat SetStatus(50), got {:?}",
            other
        ),
    }
}

#[test]
fn request_halt_three_level_nested_call_stops_at_innermost_halt() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    let outer = b.add_name("outer");
    let mid = b.add_name("mid");
    let inner = b.add_name("inner");
    b.emit(Op::Call(outer, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let skip = b.emit(Op::Jump(0), 1);
    let outer_e = b.current_pos();
    b.add_sub_entry(outer, outer_e);
    b.emit(Op::Call(mid, 0), 1);
    b.emit(Op::Return, 1);
    let mid_e = b.current_pos();
    b.add_sub_entry(mid, mid_e);
    b.emit(Op::Call(inner, 0), 1);
    b.emit(Op::Return, 1);
    let inner_e = b.current_pos();
    b.add_sub_entry(inner, inner_e);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::Return, 1);
    b.patch_jump(skip, b.current_pos());
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    let _ = vm.run();
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_three_begins_two_ends_last_end_status_wins() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(IncrementingSubshellHost { next: 5 }));
    match vm.run() {
        VMResult::Ok(Value::Status(6)) => {}
        other => panic!(
            "second SubshellEnd(Some(6)) MUST win over first Some(5), got {:?}",
            other
        ),
    }
}

#[test]
fn request_halt_as_only_op_on_empty_chunk_returns_ok_from_builtin() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    match vm.run() {
        VMResult::Ok(Value::Int(0)) => {}
        other => panic!(
            "sole halt builtin MUST return Ok(Int(0)) on empty pre-stack, got {:?}",
            other
        ),
    }
}

#[test]
fn subshell_host_swap_none_then_status_on_second_reset_chunk() {
    let chunk1 = {
        let mut b = ChunkBuilder::new();
        b.emit(Op::SubshellEnd, 1);
        b.build()
    };
    let chunk2 = {
        let mut b = ChunkBuilder::new();
        b.emit(Op::SubshellEnd, 1);
        b.emit(Op::GetStatus, 1);
        b.build()
    };
    let mut vm = VM::new(chunk1);
    vm.set_shell_host(Box::new(NoneHost));
    let _ = vm.run();
    assert_eq!(vm.last_status, 0);
    vm.set_shell_host(Box::new(StatusReturningHost(67)));
    vm.reset(chunk2);
    match vm.run() {
        VMResult::Ok(Value::Status(67)) => {}
        other => panic!("StatusReturningHost after swap MUST apply Some(67), got {:?}", other),
    }
}

#[test]
fn request_halt_after_subshell_begin_before_end_leaves_status_zero() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    SUBSHELL_END_CALLS.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::SubshellEnd, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(CountingSubshellEndHost(29)));
    vm.register_builtin(100, builtin_request_halt);
    let _ = vm.run();
    assert_eq!(SUBSHELL_END_CALLS.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 0);
}

#[test]
fn subshell_loadint_between_two_getstatus_second_on_top() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    b.emit(Op::LoadInt(77), 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(6)));
    match vm.run() {
        VMResult::Ok(Value::Status(6)) => {}
        other => panic!(
            "top GetStatus(6) MUST be returned; LoadInt(77) below it, got {:?}",
            other
        ),
    }
}

#[test]
fn request_halt_reset_clears_halt_allows_subshell_on_same_vm_instance() {
    let mut sub_b = ChunkBuilder::new();
    sub_b.emit(Op::SubshellEnd, 1);
    sub_b.emit(Op::CallBuiltin(100, 0), 1);
    let sub_chunk = sub_b.build();
    let mut vm = VM::new(sub_chunk.clone());
    vm.set_shell_host(Box::new(StatusReturningHost(84)));
    vm.register_builtin(100, builtin_request_halt);
    let _ = vm.run();
    vm.reset(sub_chunk);
    match vm.run() {
        VMResult::Ok(Value::Int(0)) => {}
        other => panic!("second run after reset MUST reach halt builtin again, got {:?}", other),
    }
    assert_eq!(vm.last_status, 84);
}

#[test]
fn subshell_custom_pipeline_negative_exit_on_stack() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::PipelineEnd, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(CustomPipelineHost(-7)));
    match vm.run() {
        VMResult::Ok(Value::Status(-7)) => {}
        other => panic!(
            "CustomPipelineHost(-7) MUST push negative pipeline status, got {:?}",
            other
        ),
    }
}

#[test]
fn request_halt_extension_handler_can_read_vm_last_status_before_halt() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    static SEEN: AtomicU32 = AtomicU32::new(0);
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(41), 1);
    b.emit(Op::SetStatus, 1);
    b.emit(Op::Extended(6, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_extension_handler(Box::new(|vm, _id, _arg| {
        SEEN.store(vm.last_status as u32, Ordering::SeqCst);
        vm.request_halt();
    }));
    let _ = vm.run();
    assert_eq!(SEEN.load(Ordering::SeqCst), 41);
}

#[test]
fn subshell_zero_after_max_on_single_vm_overwrites() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    struct MaxThenZeroHost {
        calls: u32,
    }
    impl ShellHost for MaxThenZeroHost {
        fn subshell_end(&mut self) -> Option<i32> {
            self.calls += 1;
            if self.calls == 1 {
                Some(i32::MAX)
            } else {
                Some(0)
            }
        }
    }
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(MaxThenZeroHost { calls: 0 }));
    match vm.run() {
        VMResult::Ok(Value::Status(0)) => {}
        other => panic!("Some(0) MUST overwrite i32::MAX, got {:?}", other),
    }
}

#[test]
fn request_halt_pre_run_nonempty_stack_still_halts_before_first_op() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(7, builtin_touched);
    vm.stack.push(Value::Int(1));
    vm.request_halt();
    match vm.run() {
        VMResult::Ok(Value::Int(1)) => {}
        other => panic!(
            "pre-run halt with pre-seeded stack MUST pop stack top without running ops, got {:?}",
            other
        ),
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_negative_after_zero_overwrites() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    struct ZeroThenNegativeHost {
        calls: u32,
    }
    impl ShellHost for ZeroThenNegativeHost {
        fn subshell_end(&mut self) -> Option<i32> {
            self.calls += 1;
            if self.calls == 1 {
                Some(0)
            } else {
                Some(-9)
            }
        }
    }
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(ZeroThenNegativeHost { calls: 0 }));
    match vm.run() {
        VMResult::Ok(Value::Status(-9)) => {}
        other => panic!("Some(-9) MUST overwrite Some(0), got {:?}", other),
    }
}

#[test]
fn request_halt_between_pushframe_and_popframe_pop_not_reached() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::PushFrame, 1);
    b.emit(Op::PushFrame, 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::PopFrame, 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_fourteen_end_incrementing_from_eighteen() {
    let mut b = ChunkBuilder::new();
    for _ in 0..14 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(IncrementingSubshellHost { next: 18 }));
    match vm.run() {
        VMResult::Ok(Value::Status(31)) => {}
        other => panic!("fourteenth subshell_end(Some(31)) MUST win, got {:?}", other),
    }
}

#[test]
fn request_halt_full_pipeline_begin_stage_end_then_halt_before_loadint() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    PIPELINE_END_CALLS.store(0, Ordering::SeqCst);
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::PipelineBegin(2), 1);
    b.emit(Op::PipelineStage, 1);
    b.emit(Op::PipelineEnd, 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(CountingPipelineHost));
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(PIPELINE_END_CALLS.load(Ordering::SeqCst), 1);
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_default_host_on_second_chunk_after_status_host_first() {
    let chunk1 = {
        let mut b = ChunkBuilder::new();
        b.emit(Op::SubshellEnd, 1);
        b.build()
    };
    let chunk2 = {
        let mut b = ChunkBuilder::new();
        b.emit(Op::SubshellEnd, 1);
        b.emit(Op::GetStatus, 1);
        b.build()
    };
    let mut vm = VM::new(chunk1);
    vm.set_shell_host(Box::new(StatusReturningHost(23)));
    let _ = vm.run();
    assert_eq!(vm.last_status, 23);
    vm.set_shell_host(Box::new(DefaultHost));
    vm.reset(chunk2);
    match vm.run() {
        VMResult::Ok(Value::Status(0)) => {}
        other => panic!(
            "DefaultHost subshell_end(None) after reset MUST leave last_status(0), got {:?}",
            other
        ),
    }
}

#[test]
fn request_halt_subshell_status_visible_to_extension_after_end_before_halt() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    static SEEN: AtomicU32 = AtomicU32::new(0);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::Extended(11, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(96)));
    vm.set_extension_handler(Box::new(|vm, _id, _arg| {
        SEEN.store(vm.last_status as u32, Ordering::SeqCst);
        vm.request_halt();
    }));
    let _ = vm.run();
    assert_eq!(SEEN.load(Ordering::SeqCst), 96);
}

#[test]
fn subshell_end_exit_code_135_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(135)));
    match vm.run() {
        VMResult::Ok(Value::Status(135)) => {}
        other => panic!("exit code 135 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_load_undef_after_halt_point() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::LoadUndef, 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_136_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(136)));
    match vm.run() {
        VMResult::Ok(Value::Status(136)) => {}
        other => panic!("exit code 136 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_pop_on_nonempty_stack() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(5), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::Pop, 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_sixteen_end_incrementing_from_twenty_two() {
    let mut b = ChunkBuilder::new();
    for _ in 0..16 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(IncrementingSubshellHost { next: 22 }));
    match vm.run() {
        VMResult::Ok(Value::Status(37)) => {}
        other => panic!("sixteenth subshell_end(Some(37)) MUST win, got {:?}", other),
    }
}

#[test]
fn request_halt_after_callee_returnvalue_before_second_builtin() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    let name = b.add_name("ret_one");
    b.emit(Op::Call(name, 0), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let skip = b.emit(Op::Jump(0), 1);
    let entry = b.current_pos();
    b.add_sub_entry(name, entry);
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::ReturnValue, 1);
    b.patch_jump(skip, b.current_pos());
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    match vm.run() {
        VMResult::Ok(Value::Int(0)) => {}
        other => panic!(
            "halt builtin result MUST be stack top over callee ReturnValue(1), got {:?}",
            other
        ),
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_third_some_host_on_fifth_end_applies_fifty() {
    let mut b = ChunkBuilder::new();
    for _ in 0..5 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    struct FifthSomeSubshellHost {
        calls: u32,
    }
    impl ShellHost for FifthSomeSubshellHost {
        fn subshell_end(&mut self) -> Option<i32> {
            self.calls += 1;
            if self.calls == 5 {
                Some(50)
            } else {
                None
            }
        }
    }
    let mut vm = VM::new(b.build());
    vm.last_status = 2;
    vm.set_shell_host(Box::new(FifthSomeSubshellHost { calls: 0 }));
    match vm.run() {
        VMResult::Ok(Value::Status(50)) => {}
        other => panic!("fifth subshell_end(Some(50)) MUST apply, got {:?}", other),
    }
}

#[test]
fn request_halt_between_two_extended_ops_second_not_reached() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    static FIRST: AtomicU32 = AtomicU32::new(0);
    static SECOND: AtomicU32 = AtomicU32::new(0);
    let mut b = ChunkBuilder::new();
    b.emit(Op::Extended(1, 0), 1);
    b.emit(Op::Extended(2, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_extension_handler(Box::new(|vm, id, _arg| {
        if id == 1 {
            FIRST.fetch_add(1, Ordering::SeqCst);
            vm.request_halt();
        } else {
            SECOND.fetch_add(1, Ordering::SeqCst);
        }
    }));
    let _ = vm.run();
    assert_eq!(FIRST.load(Ordering::SeqCst), 1);
    assert_eq!(SECOND.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_getstatus_after_begin_without_end_reads_prior_status() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.last_status = 38;
    vm.set_shell_host(Box::new(BeginEndCountingHost(99)));
    match vm.run() {
        VMResult::Ok(Value::Status(38)) => {}
        other => panic!(
            "GetStatus before SubshellEnd MUST read prior last_status(38), got {:?}",
            other
        ),
    }
}

#[test]
fn request_halt_reset_then_extension_handler_runs_extended_op() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    static EXT_RAN: AtomicU32 = AtomicU32::new(0);
    let halt = {
        let mut b = ChunkBuilder::new();
        b.emit(Op::CallBuiltin(100, 0), 1);
        b.build()
    };
    let ext = {
        let mut b = ChunkBuilder::new();
        b.emit(Op::Extended(20, 5), 1);
        b.build()
    };
    let mut vm = VM::new(halt);
    vm.set_extension_handler(Box::new(|_vm, id, arg| {
        assert_eq!(id, 20);
        assert_eq!(arg, 5);
        EXT_RAN.fetch_add(1, Ordering::SeqCst);
    }));
    vm.register_builtin(100, builtin_request_halt);
    let _ = vm.run();
    EXT_RAN.store(0, Ordering::SeqCst);
    vm.reset(ext);
    let _ = vm.run();
    assert_eq!(EXT_RAN.load(Ordering::SeqCst), 1);
}

#[test]
fn subshell_end_after_two_pipeline_stages_without_pipeline_end() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    PIPELINE_STAGE_CALLS.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::PipelineBegin(3), 1);
    b.emit(Op::PipelineStage, 1);
    b.emit(Op::PipelineStage, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    struct StageSubshellHost2;
    impl ShellHost for StageSubshellHost2 {
        fn pipeline_stage(&mut self) {
            PIPELINE_STAGE_CALLS.fetch_add(1, Ordering::SeqCst);
        }
        fn subshell_end(&mut self) -> Option<i32> {
            Some(62)
        }
    }
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StageSubshellHost2));
    match vm.run() {
        VMResult::Ok(Value::Status(62)) => {}
        other => panic!(
            "subshell_end(Some(62)) MUST apply mid-pipeline without PipelineEnd, got {:?}",
            other
        ),
    }
    assert_eq!(PIPELINE_STAGE_CALLS.load(Ordering::SeqCst), 2);
}

#[test]
fn request_halt_after_subshell_end_and_before_pipeline_begin() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    PIPELINE_BEGIN_CALLS.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::PipelineBegin(2), 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(CountingPipelineBeginHost));
    vm.register_builtin(100, builtin_request_halt);
    let _ = vm.run();
    assert_eq!(PIPELINE_BEGIN_CALLS.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 0);
}

#[test]
fn subshell_end_status_minus_five_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(-5)));
    match vm.run() {
        VMResult::Ok(Value::Status(-5)) => {}
        other => panic!("status -5 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_halt_flag_survives_ok_return_from_empty_halt_only_chunk() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(61, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(61, builtin_halt_only);
    vm.register_builtin(7, builtin_touched);
    assert!(matches!(vm.run(), VMResult::Ok(Value::Undef)));
    let _ = vm.run();
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_orphan_end_counting_host_invoked_without_begin() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    SUBSHELL_END_CALLS.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(CountingSubshellEndHost(16)));
    match vm.run() {
        VMResult::Ok(Value::Status(16)) => {}
        other => panic!("orphan SubshellEnd MUST invoke host and apply Some(16), got {:?}", other),
    }
    assert_eq!(SUBSHELL_END_CALLS.load(Ordering::SeqCst), 2);
}

#[test]
fn request_halt_between_subshell_begin_and_subshell_end_no_status_change() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    SUBSHELL_END_CALLS.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::SubshellEnd, 1);
    let mut vm = VM::new(b.build());
    vm.last_status = 14;
    vm.set_shell_host(Box::new(CountingSubshellEndHost(55)));
    vm.register_builtin(100, builtin_request_halt);
    let _ = vm.run();
    assert_eq!(SUBSHELL_END_CALLS.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 14);
}

// ─── Pin tests batch A (handwritten) ──────────────────────────────────────

#[test]
fn subshell_end_exit_code_23_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(23)));
    match vm.run() {
        VMResult::Ok(Value::Status(23)) => {}
        other => panic!("exit code 23 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_num_lt_conditional_jump() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::LoadInt(2), 1);
    b.emit(Op::LoadInt(5), 1);
    b.emit(Op::NumLt, 1);
    let jmp = b.emit(Op::JumpIfTrue(0), 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    b.patch_jump(jmp, b.current_pos());
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_24_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(24)));
    match vm.run() {
        VMResult::Ok(Value::Status(24)) => {}
        other => panic!("exit code 24 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_num_gt_conditional_jump() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::LoadInt(9), 1);
    b.emit(Op::LoadInt(3), 1);
    b.emit(Op::NumGt, 1);
    let jmp = b.emit(Op::JumpIfTrue(0), 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    b.patch_jump(jmp, b.current_pos());
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_25_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(25)));
    match vm.run() {
        VMResult::Ok(Value::Status(25)) => {}
        other => panic!("exit code 25 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_dup_on_stack() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(42), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::Dup, 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_26_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(26)));
    match vm.run() {
        VMResult::Ok(Value::Status(26)) => {}
        other => panic!("exit code 26 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_swap_on_stack() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::LoadInt(2), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::Swap, 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_27_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(27)));
    match vm.run() {
        VMResult::Ok(Value::Status(27)) => {}
        other => panic!("exit code 27 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_load_const() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    let c = b.add_constant(Value::Int(99));
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::LoadConst(c), 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_28_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(28)));
    match vm.run() {
        VMResult::Ok(Value::Status(28)) => {}
        other => panic!("exit code 28 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_after_subshell_in_open_pipeline_before_stage() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    PIPELINE_STAGE_CALLS.store(0, Ordering::SeqCst);
    struct SubshellStageHost39;
    impl ShellHost for SubshellStageHost39 {
        fn pipeline_stage(&mut self) {
            PIPELINE_STAGE_CALLS.fetch_add(1, Ordering::SeqCst);
        }
        fn subshell_end(&mut self) -> Option<i32> {
            Some(39)
        }
    }
    let mut b = ChunkBuilder::new();
    b.emit(Op::PipelineBegin(2), 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::PipelineStage, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(SubshellStageHost39));
    vm.register_builtin(100, builtin_request_halt);
    let _ = vm.run();
    assert_eq!(PIPELINE_STAGE_CALLS.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 39);
}

#[test]
fn subshell_end_exit_code_29_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(29)));
    match vm.run() {
        VMResult::Ok(Value::Status(29)) => {}
        other => panic!("exit code 29 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_quadruple_reset_then_touched_builtin_runs() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let noop = ChunkBuilder::new().build();
    let run = {
        let mut b = ChunkBuilder::new();
        b.emit(Op::CallBuiltin(7, 0), 1);
        b.build()
    };
    let mut vm = VM::new(noop.clone());
    vm.register_builtin(7, builtin_touched);
    vm.request_halt();
    let _ = vm.run();
    vm.reset(noop.clone());
    vm.reset(noop.clone());
    vm.reset(noop.clone());
    vm.reset(run);
    let _ = vm.run();
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 1);
}

#[test]
fn subshell_end_exit_code_30_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(30)));
    match vm.run() {
        VMResult::Ok(Value::Status(30)) => {}
        other => panic!("exit code 30 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_between_callbuiltin_and_loadint_blocks_loadint() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::LoadInt(808), 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_137_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(137)));
    match vm.run() {
        VMResult::Ok(Value::Status(137)) => {}
        other => panic!("exit code 137 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_top_level_returnvalue_after_halt() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(63), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::ReturnValue, 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    match vm.run() {
        VMResult::Ok(Value::Int(0)) => {}
        other => panic!("ReturnValue MUST NOT run after halt, got {:?}", other),
    }
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_138_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(138)));
    match vm.run() {
        VMResult::Ok(Value::Status(138)) => {}
        other => panic!("exit code 138 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_extension_wide_reads_subshell_status_in_handler() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    static SEEN: AtomicU32 = AtomicU32::new(0);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::ExtendedWide(10, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(77)));
    vm.set_extension_wide_handler(Box::new(|vm, _id, _payload| {
        SEEN.store(vm.last_status as u32, Ordering::SeqCst);
        vm.request_halt();
    }));
    let _ = vm.run();
    assert_eq!(SEEN.load(Ordering::SeqCst), 77);
}

#[test]
fn subshell_end_exit_code_139_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(139)));
    match vm.run() {
        VMResult::Ok(Value::Status(139)) => {}
        other => panic!("exit code 139 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_pre_run_halt_reset_pre_run_halt_reset_then_runs() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let chunk = {
        let mut b = ChunkBuilder::new();
        b.emit(Op::CallBuiltin(7, 0), 1);
        b.build()
    };
    let mut vm = VM::new(chunk.clone());
    vm.register_builtin(7, builtin_touched);
    vm.request_halt();
    let _ = vm.run();
    vm.reset(chunk.clone());
    vm.request_halt();
    let _ = vm.run();
    vm.reset(chunk);
    let _ = vm.run();
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 1);
}

#[test]
fn subshell_end_status_minus_six_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(-6)));
    match vm.run() {
        VMResult::Ok(Value::Status(-6)) => {}
        other => panic!("status -6 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_negate_on_stack() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(5), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::Negate, 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_status_minus_seven_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(-7)));
    match vm.run() {
        VMResult::Ok(Value::Status(-7)) => {}
        other => panic!("status -7 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_between_pipeline_end_and_getstatus() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::PipelineEnd, 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(CountingPipelineHost));
    vm.register_builtin(100, builtin_request_halt);
    match vm.run() {
        VMResult::Ok(Value::Int(0)) => {}
        other => panic!("GetStatus MUST NOT run after halt, got {:?}", other),
    }
    assert_eq!(vm.last_status, 0);
}

#[test]
fn subshell_seventeen_end_incrementing_from_nine() {
    let mut b = ChunkBuilder::new();
    for _ in 0..17 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(IncrementingSubshellHost { next: 9 }));
    match vm.run() {
        VMResult::Ok(Value::Status(25)) => {}
        other => panic!("seventeenth subshell_end(Some(25)) MUST win, got {:?}", other),
    }
}

#[test]
fn request_halt_three_stage_pipeline_halt_before_second_stage() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    PIPELINE_STAGE_CALLS.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::PipelineBegin(3), 1);
    b.emit(Op::PipelineStage, 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::PipelineStage, 1);
    b.emit(Op::PipelineStage, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(CountingPipelineStageHost));
    vm.register_builtin(100, builtin_request_halt);
    let _ = vm.run();
    assert_eq!(
        PIPELINE_STAGE_CALLS.load(Ordering::SeqCst),
        1,
        "only first PipelineStage MUST run before halt"
    );
}

#[test]
fn subshell_eighteen_end_incrementing_from_eleven() {
    let mut b = ChunkBuilder::new();
    for _ in 0..18 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(IncrementingSubshellHost { next: 11 }));
    match vm.run() {
        VMResult::Ok(Value::Status(28)) => {}
        other => panic!("eighteenth subshell_end(Some(28)) MUST win, got {:?}", other),
    }
}

#[test]
fn request_halt_nested_call_depth_four_halts_at_level_two() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    let l1 = b.add_name("l1");
    let l2 = b.add_name("l2");
    let l3 = b.add_name("l3");
    b.emit(Op::Call(l1, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let skip = b.emit(Op::Jump(0), 1);
    let e1 = b.current_pos();
    b.add_sub_entry(l1, e1);
    b.emit(Op::Call(l2, 0), 1);
    b.emit(Op::Return, 1);
    let e2 = b.current_pos();
    b.add_sub_entry(l2, e2);
    b.emit(Op::Call(l3, 0), 1);
    b.emit(Op::Return, 1);
    let e3 = b.current_pos();
    b.add_sub_entry(l3, e3);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::Return, 1);
    b.patch_jump(skip, b.current_pos());
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    let _ = vm.run();
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_nineteen_end_incrementing_from_four() {
    let mut b = ChunkBuilder::new();
    for _ in 0..19 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(IncrementingSubshellHost { next: 4 }));
    match vm.run() {
        VMResult::Ok(Value::Status(22)) => {}
        other => panic!("nineteenth subshell_end(Some(22)) MUST win, got {:?}", other),
    }
}

#[test]
fn request_halt_callbuiltin_argc_one_reports_one() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(9), 1);
    b.emit(Op::CallBuiltin(100, 1), 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_halt_report_argc);
    vm.register_builtin(200, builtin_count_after_halt);
    match vm.run() {
        VMResult::Ok(Value::Int(1)) => {}
        other => panic!("halt MUST report argc=1, got {:?}", other),
    }
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_getstatus_before_end_stays_on_stack_below_updated_getstatus() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::GetStatus, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.last_status = 10;
    vm.set_shell_host(Box::new(StatusReturningHost(32)));
    match vm.run() {
        VMResult::Ok(Value::Status(32)) => {}
        other => panic!("top GetStatus(32) MUST win, got {:?}", other),
    }
}

#[test]
fn request_halt_chain_of_three_extended_no_handler_then_halt_builtin() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::Extended(1, 0), 1);
    b.emit(Op::Extended(2, 0), 1);
    b.emit(Op::Extended(3, 0), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_loadint_deep_stack_unaffected_by_subshell_end() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(100), 1);
    b.emit(Op::LoadInt(200), 1);
    b.emit(Op::SubshellEnd, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(33)));
    match vm.run() {
        VMResult::Ok(Value::Int(200)) => {}
        other => panic!(
            "SubshellEnd MUST NOT pop stack — top MUST stay Int(200), got {:?}",
            other
        ),
    }
    assert_eq!(vm.last_status, 33);
}

#[test]
fn request_halt_after_setstatus_before_getstatus_on_stack() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(48), 1);
    b.emit(Op::SetStatus, 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    match vm.run() {
        VMResult::Ok(Value::Int(0)) => {}
        other => panic!("GetStatus MUST NOT run after halt, got {:?}", other),
    }
    assert_eq!(vm.last_status, 48);
}

#[test]
fn subshell_begin_end_begin_end_alternating_status_second_wins() {
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
        other => panic!("second Some(9) MUST win over first Some(3), got {:?}", other),
    }
}

#[test]
fn request_halt_subshell_end_then_pipeline_begin_blocked() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    PIPELINE_BEGIN_CALLS.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::PipelineBegin(2), 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(CountingPipelineBeginHost));
    vm.register_builtin(100, builtin_request_halt);
    let _ = vm.run();
    assert_eq!(PIPELINE_BEGIN_CALLS.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 0);
}

#[test]
fn subshell_setstatus_before_begin_end_getstatus_reads_subshell() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(55), 1);
    b.emit(Op::SetStatus, 1);
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(34)));
    match vm.run() {
        VMResult::Ok(Value::Status(34)) => {}
        other => panic!(
            "subshell_end(Some(34)) MUST beat prior SetStatus(55), got {:?}",
            other
        ),
    }
}

#[test]
fn request_halt_reset_after_halt_allows_second_subshell_end_on_new_chunk() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    SUBSHELL_END_CALLS.store(0, Ordering::SeqCst);
    let halt_chunk = {
        let mut b = ChunkBuilder::new();
        b.emit(Op::SubshellEnd, 1);
        b.emit(Op::CallBuiltin(100, 0), 1);
        b.build()
    };
    let end_chunk = {
        let mut b = ChunkBuilder::new();
        b.emit(Op::SubshellEnd, 1);
        b.build()
    };
    let mut vm = VM::new(halt_chunk);
    vm.set_shell_host(Box::new(CountingSubshellEndHost(35)));
    vm.register_builtin(100, builtin_request_halt);
    let _ = vm.run();
    assert_eq!(SUBSHELL_END_CALLS.load(Ordering::SeqCst), 1);
    SUBSHELL_END_CALLS.store(0, Ordering::SeqCst);
    vm.reset(end_chunk);
    let _ = vm.run();
    assert_eq!(SUBSHELL_END_CALLS.load(Ordering::SeqCst), 1);
    assert_eq!(vm.last_status, 35);
}

#[test]
fn subshell_end_after_loadtrue_stack_top_is_true() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::LoadTrue, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(36)));
    match vm.run() {
        VMResult::Ok(Value::Bool(true)) => {}
        other => panic!("LoadTrue MUST be stack top after subshell_end, got {:?}", other),
    }
    assert_eq!(vm.last_status, 36);
}

#[test]
fn request_halt_halt_between_two_return_ops_in_flat_chunk() {
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
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_after_loadfalse_stack_top_is_false() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::LoadFalse, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(37)));
    match vm.run() {
        VMResult::Ok(Value::Bool(false)) => {}
        other => panic!("LoadFalse MUST be stack top after subshell_end, got {:?}", other),
    }
}

#[test]
fn request_halt_six_consecutive_runs_without_reset_never_resume() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    for _ in 0..6 {
        let _ = vm.run();
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_two_setstatus_between_ends_last_end_wins() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::LoadInt(10), 1);
    b.emit(Op::SetStatus, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(IncrementingSubshellHost { next: 40 }));
    match vm.run() {
        VMResult::Ok(Value::Status(41)) => {}
        other => panic!(
            "second subshell_end(Some(41)) MUST beat SetStatus(10), got {:?}",
            other
        ),
    }
}

#[test]
fn request_halt_between_subshell_end_and_pipeline_stage_in_open_pipeline() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    PIPELINE_STAGE_CALLS.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::PipelineBegin(2), 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::PipelineStage, 1);
    struct SubshellStageHost;
    impl ShellHost for SubshellStageHost {
        fn pipeline_stage(&mut self) {
            PIPELINE_STAGE_CALLS.fetch_add(1, Ordering::SeqCst);
        }
        fn subshell_end(&mut self) -> Option<i32> {
            Some(43)
        }
    }
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(SubshellStageHost));
    vm.register_builtin(100, builtin_request_halt);
    let _ = vm.run();
    assert_eq!(PIPELINE_STAGE_CALLS.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 43);
}

// ─── Pin tests batch B (handwritten) ──────────────────────────────────────

#[test]
fn subshell_end_exit_code_140_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(140)));
    match vm.run() {
        VMResult::Ok(Value::Status(140)) => {}
        other => panic!("exit code 140 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_add_on_stack() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(3), 1);
    b.emit(Op::LoadInt(4), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::Add, 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_141_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(141)));
    match vm.run() {
        VMResult::Ok(Value::Status(141)) => {}
        other => panic!("exit code 141 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_sub_on_stack() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(10), 1);
    b.emit(Op::LoadInt(3), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::Sub, 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_142_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(142)));
    match vm.run() {
        VMResult::Ok(Value::Status(142)) => {}
        other => panic!("exit code 142 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_after_pipeline_stage_before_pipeline_end() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    PIPELINE_END_CALLS.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::PipelineBegin(2), 1);
    b.emit(Op::PipelineStage, 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::PipelineEnd, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(CountingPipelineHost));
    vm.register_builtin(100, builtin_request_halt);
    let _ = vm.run();
    assert_eq!(PIPELINE_END_CALLS.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_143_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(143)));
    match vm.run() {
        VMResult::Ok(Value::Status(143)) => {}
        other => panic!("exit code 143 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_mid_chunk_subshell_end_runs_before_halt_point() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    SUBSHELL_END_CALLS.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(CountingSubshellEndHost(44)));
    vm.register_builtin(100, builtin_request_halt);
    let _ = vm.run();
    assert_eq!(SUBSHELL_END_CALLS.load(Ordering::SeqCst), 1);
    assert_eq!(vm.last_status, 44);
}

#[test]
fn subshell_end_exit_code_144_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(144)));
    match vm.run() {
        VMResult::Ok(Value::Status(144)) => {}
        other => panic!("exit code 144 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_inc_on_stack() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(7), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::Inc, 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_145_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(145)));
    match vm.run() {
        VMResult::Ok(Value::Status(145)) => {}
        other => panic!("exit code 145 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_dec_on_stack() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(7), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::Dec, 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_twenty_one_end_incrementing_from_zero() {
    let mut b = ChunkBuilder::new();
    for _ in 0..21 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(IncrementingSubshellHost { next: 0 }));
    match vm.run() {
        VMResult::Ok(Value::Status(20)) => {}
        other => panic!("twenty-first subshell_end(Some(20)) MUST win, got {:?}", other),
    }
}

#[test]
fn request_halt_callee_returns_before_halt_in_caller_second_builtin_blocked() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    let name = b.add_name("add_one");
    b.emit(Op::Call(name, 0), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let skip = b.emit(Op::Jump(0), 1);
    let entry = b.current_pos();
    b.add_sub_entry(name, entry);
    b.emit(Op::LoadInt(2), 1);
    b.emit(Op::ReturnValue, 1);
    b.patch_jump(skip, b.current_pos());
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    match vm.run() {
        VMResult::Ok(Value::Int(0)) => {}
        other => panic!(
            "halt builtin MUST be stack top over callee ReturnValue(2), got {:?}",
            other
        ),
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_twenty_two_end_incrementing_from_six() {
    let mut b = ChunkBuilder::new();
    for _ in 0..22 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(IncrementingSubshellHost { next: 6 }));
    match vm.run() {
        VMResult::Ok(Value::Status(27)) => {}
        other => panic!("twenty-second subshell_end(Some(27)) MUST win, got {:?}", other),
    }
}

#[test]
fn request_halt_extension_handler_on_reset_chunk_sees_zero_last_status() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    static SEEN: AtomicU32 = AtomicU32::new(0);
    let halt = {
        let mut b = ChunkBuilder::new();
        b.emit(Op::LoadInt(99), 1);
        b.emit(Op::SetStatus, 1);
        b.emit(Op::CallBuiltin(100, 0), 1);
        b.build()
    };
    let ext = {
        let mut b = ChunkBuilder::new();
        b.emit(Op::Extended(15, 0), 1);
        b.build()
    };
    let mut vm = VM::new(halt);
    vm.set_extension_handler(Box::new(|vm, _id, _arg| {
        SEEN.store(vm.last_status as u32, Ordering::SeqCst);
    }));
    vm.register_builtin(100, builtin_request_halt);
    let _ = vm.run();
    vm.reset(ext);
    let _ = vm.run();
    assert_eq!(SEEN.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_twenty_three_end_incrementing_from_eight() {
    let mut b = ChunkBuilder::new();
    for _ in 0..23 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(IncrementingSubshellHost { next: 8 }));
    match vm.run() {
        VMResult::Ok(Value::Status(30)) => {}
        other => panic!("twenty-third subshell_end(Some(30)) MUST win, got {:?}", other),
    }
}

#[test]
fn request_halt_between_two_subshell_begins_without_end() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    SUBSHELL_BEGIN_CALLS.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellBegin, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(BeginEndCountingHost(0)));
    vm.register_builtin(100, builtin_request_halt);
    let _ = vm.run();
    assert_eq!(SUBSHELL_BEGIN_CALLS.load(Ordering::SeqCst), 1);
}

#[test]
fn subshell_twenty_four_end_incrementing_from_one() {
    let mut b = ChunkBuilder::new();
    for _ in 0..24 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(IncrementingSubshellHost { next: 1 }));
    match vm.run() {
        VMResult::Ok(Value::Status(24)) => {}
        other => panic!("twenty-fourth subshell_end(Some(24)) MUST win, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_forward_jump_over_halt_target() {
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
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_twenty_five_end_incrementing_from_two() {
    let mut b = ChunkBuilder::new();
    for _ in 0..25 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(IncrementingSubshellHost { next: 2 }));
    match vm.run() {
        VMResult::Ok(Value::Status(26)) => {}
        other => panic!("twenty-fifth subshell_end(Some(26)) MUST win, got {:?}", other),
    }
}

#[test]
fn request_halt_after_getstatus_leaves_status_on_stack_when_halt_follows() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(51), 1);
    b.emit(Op::SetStatus, 1);
    b.emit(Op::GetStatus, 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    match vm.run() {
        VMResult::Ok(Value::Int(0)) => {}
        other => panic!("halt MUST return builtin Int(0), not Status(51), got {:?}", other),
    }
}

#[test]
fn subshell_sixth_none_then_some_on_seventh_applies_sixty_three() {
    let mut b = ChunkBuilder::new();
    for _ in 0..7 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    struct SeventhSomeHost {
        calls: u32,
    }
    impl ShellHost for SeventhSomeHost {
        fn subshell_end(&mut self) -> Option<i32> {
            self.calls += 1;
            if self.calls == 7 {
                Some(63)
            } else {
                None
            }
        }
    }
    let mut vm = VM::new(b.build());
    vm.last_status = 5;
    vm.set_shell_host(Box::new(SeventhSomeHost { calls: 0 }));
    match vm.run() {
        VMResult::Ok(Value::Status(63)) => {}
        other => panic!("seventh subshell_end(Some(63)) MUST apply, got {:?}", other),
    }
}

#[test]
fn request_halt_two_subshell_ends_then_halt_third_end_blocked() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    SUBSHELL_END_CALLS.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::SubshellEnd, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(CountingSubshellEndHost(45)));
    vm.register_builtin(100, builtin_request_halt);
    let _ = vm.run();
    assert_eq!(SUBSHELL_END_CALLS.load(Ordering::SeqCst), 2);
    assert_eq!(vm.last_status, 45);
}

#[test]
fn subshell_pipeline_begin_without_end_subshell_end_status_persists() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    PIPELINE_BEGIN_CALLS.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::PipelineBegin(2), 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    struct BeginSubshellHost;
    impl ShellHost for BeginSubshellHost {
        fn pipeline_begin(&mut self, _n: u8) {
            PIPELINE_BEGIN_CALLS.fetch_add(1, Ordering::SeqCst);
        }
        fn subshell_end(&mut self) -> Option<i32> {
            Some(46)
        }
    }
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(BeginSubshellHost));
    match vm.run() {
        VMResult::Ok(Value::Status(46)) => {}
        other => panic!(
            "subshell_end(Some(46)) MUST apply without PipelineEnd, got {:?}",
            other
        ),
    }
    assert_eq!(PIPELINE_BEGIN_CALLS.load(Ordering::SeqCst), 1);
}

#[test]
fn request_halt_stops_before_mul_on_stack() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(6), 1);
    b.emit(Op::LoadInt(7), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::Mul, 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_status_minus_eight_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(-8)));
    match vm.run() {
        VMResult::Ok(Value::Status(-8)) => {}
        other => panic!("status -8 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_div_on_stack() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(20), 1);
    b.emit(Op::LoadInt(4), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::Div, 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_getstatus_with_zero_last_status_before_end() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::GetStatus, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.last_status = 0;
    vm.set_shell_host(Box::new(StatusReturningHost(47)));
    match vm.run() {
        VMResult::Ok(Value::Status(47)) => {}
        other => panic!("final GetStatus MUST read Some(47), got {:?}", other),
    }
}

#[test]
fn request_halt_after_subshell_end_preserves_stack_loadint_below_halt_result() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(88), 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(48)));
    vm.register_builtin(100, builtin_request_halt);
    match vm.run() {
        VMResult::Ok(Value::Int(0)) => {}
        other => panic!("halt MUST pop builtin Int(0) as top, got {:?}", other),
    }
    assert_eq!(vm.last_status, 48);
}

#[test]
fn subshell_custom_pipeline_zero_after_subshell_four_on_stack() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::PipelineEnd, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(CustomPipelineHost(0)));
    match vm.run() {
        VMResult::Ok(Value::Status(0)) => {}
        other => panic!(
            "PipelineEnd MUST push host.pipeline_end(0), not subshell last_status(4), got {:?}",
            other
        ),
    }
}

#[test]
fn request_halt_reset_clears_halt_three_subshell_ends_on_fresh_chunk() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    SUBSHELL_END_CALLS.store(0, Ordering::SeqCst);
    let halt = {
        let mut b = ChunkBuilder::new();
        b.emit(Op::CallBuiltin(100, 0), 1);
        b.build()
    };
    let ends = {
        let mut b = ChunkBuilder::new();
        b.emit(Op::SubshellEnd, 1);
        b.emit(Op::SubshellEnd, 1);
        b.emit(Op::SubshellEnd, 1);
        b.build()
    };
    let mut vm = VM::new(halt);
    vm.set_shell_host(Box::new(CountingSubshellEndHost(49)));
    vm.register_builtin(100, builtin_request_halt);
    let _ = vm.run();
    vm.reset(ends);
    let _ = vm.run();
    assert_eq!(SUBSHELL_END_CALLS.load(Ordering::SeqCst), 3);
    assert_eq!(vm.last_status, 49);
}

#[test]
fn subshell_end_after_manual_last_status_ninety_one() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.last_status = 91;
    vm.set_shell_host(Box::new(StatusReturningHost(52)));
    match vm.run() {
        VMResult::Ok(Value::Status(52)) => {}
        other => panic!("Some(52) MUST overwrite manual last_status(91), got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_mod_on_stack() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(10), 1);
    b.emit(Op::LoadInt(3), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::Mod, 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_none_host_first_chunk_status_host_second_after_reset() {
    let c1 = {
        let mut b = ChunkBuilder::new();
        b.emit(Op::SubshellEnd, 1);
        b.build()
    };
    let c2 = {
        let mut b = ChunkBuilder::new();
        b.emit(Op::SubshellEnd, 1);
        b.emit(Op::GetStatus, 1);
        b.build()
    };
    let mut vm = VM::new(c1);
    vm.last_status = 20;
    vm.set_shell_host(Box::new(NoneHost));
    let _ = vm.run();
    assert_eq!(vm.last_status, 20);
    vm.set_shell_host(Box::new(StatusReturningHost(53)));
    vm.reset(c2);
    match vm.run() {
        VMResult::Ok(Value::Status(53)) => {}
        other => panic!("StatusReturningHost MUST apply Some(53) on reset chunk, got {:?}", other),
    }
}

#[test]
fn request_halt_between_subshell_end_and_subshell_begin() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    SUBSHELL_BEGIN_CALLS.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::SubshellBegin, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(BeginEndCountingHost(54)));
    vm.register_builtin(100, builtin_request_halt);
    let _ = vm.run();
    assert_eq!(SUBSHELL_BEGIN_CALLS.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 54);
}

#[test]
fn subshell_halt_reset_subshell_end_twice_same_host() {
    let chunk = {
        let mut b = ChunkBuilder::new();
        b.emit(Op::SubshellEnd, 1);
        b.emit(Op::GetStatus, 1);
        b.build()
    };
    let mut vm = VM::new(chunk.clone());
    vm.set_shell_host(Box::new(StatusReturningHost(56)));
    match vm.run() {
        VMResult::Ok(Value::Status(56)) => {}
        other => panic!("first run MUST get Status(56), got {:?}", other),
    }
    vm.reset(chunk.clone());
    match vm.run() {
        VMResult::Ok(Value::Status(56)) => {}
        other => panic!("second run after reset MUST get Status(56), got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_num_le_conditional_jump() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::LoadInt(3), 1);
    b.emit(Op::LoadInt(3), 1);
    b.emit(Op::NumLe, 1);
    let jmp = b.emit(Op::JumpIfTrue(0), 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    b.patch_jump(jmp, b.current_pos());
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_150_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(150)));
    match vm.run() {
        VMResult::Ok(Value::Status(150)) => {}
        other => panic!("exit code 150 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_num_ge_conditional_jump() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::LoadInt(8), 1);
    b.emit(Op::LoadInt(5), 1);
    b.emit(Op::NumGe, 1);
    let jmp = b.emit(Op::JumpIfTrue(0), 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    b.patch_jump(jmp, b.current_pos());
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_begin_end_halt_reset_begin_end_runs_again() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    SUBSHELL_BEGIN_CALLS.store(0, Ordering::SeqCst);
    let chunk = {
        let mut b = ChunkBuilder::new();
        b.emit(Op::SubshellBegin, 1);
        b.emit(Op::SubshellEnd, 1);
        b.emit(Op::CallBuiltin(100, 0), 1);
        b.build()
    };
    let mut vm = VM::new(chunk.clone());
    vm.set_shell_host(Box::new(BeginEndCountingHost(57)));
    vm.register_builtin(100, builtin_request_halt);
    let _ = vm.run();
    assert_eq!(SUBSHELL_BEGIN_CALLS.load(Ordering::SeqCst), 1);
    SUBSHELL_BEGIN_CALLS.store(0, Ordering::SeqCst);
    vm.reset(chunk);
    let _ = vm.run();
    assert_eq!(SUBSHELL_BEGIN_CALLS.load(Ordering::SeqCst), 1);
    assert_eq!(vm.last_status, 57);
}

#[test]
fn request_halt_with_pipeline_host_and_builtin_in_same_chunk() {
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
fn subshell_max_alone_on_single_end_getstatus() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(i32::MAX)));
    match vm.run() {
        VMResult::Ok(Value::Status(i32::MAX)) => {}
        other => panic!("i32::MAX MUST propagate on orphan SubshellEnd, got {:?}", other),
    }
}

#[test]
fn request_halt_seventh_consecutive_run_still_blocked() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    for _ in 0..7 {
        let _ = vm.run();
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_min_alone_on_single_end_getstatus() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(i32::MIN)));
    match vm.run() {
        VMResult::Ok(Value::Status(i32::MIN)) => {}
        other => panic!("i32::MIN MUST propagate on orphan SubshellEnd, got {:?}", other),
    }
}

#[test]
fn request_halt_subshell_then_halt_then_reset_getstatus_zero() {
    let sub = {
        let mut b = ChunkBuilder::new();
        b.emit(Op::SubshellEnd, 1);
        b.emit(Op::CallBuiltin(100, 0), 1);
        b.build()
    };
    let status = {
        let mut b = ChunkBuilder::new();
        b.emit(Op::GetStatus, 1);
        b.build()
    };
    let mut vm = VM::new(sub);
    vm.set_shell_host(Box::new(StatusReturningHost(58)));
    vm.register_builtin(100, builtin_request_halt);
    let _ = vm.run();
    assert_eq!(vm.last_status, 58);
    vm.reset(status);
    match vm.run() {
        VMResult::Ok(Value::Status(0)) => {}
        other => panic!("reset MUST zero last_status before GetStatus, got {:?}", other),
    }
}

#[test]
fn subshell_end_between_two_loadints_top_is_second_int() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(11), 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::LoadInt(22), 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(59)));
    match vm.run() {
        VMResult::Ok(Value::Int(22)) => {}
        other => panic!("LoadInt(22) MUST be stack top, got {:?}", other),
    }
}

#[test]
fn request_halt_from_builtin_mid_chunk_leaves_earlier_loadint_on_stack() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(33), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    match vm.run() {
        VMResult::Ok(Value::Int(0)) => {}
        other => panic!("halt MUST return builtin on top, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_31_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(31)));
    match vm.run() {
        VMResult::Ok(Value::Status(31)) => {}
        other => panic!("exit code 31 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_load_undef_between_two_loadints() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::LoadInt(2), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::LoadUndef, 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_32_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(32)));
    match vm.run() {
        VMResult::Ok(Value::Status(32)) => {}
        other => panic!("exit code 32 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_after_two_push_frames_before_third_push() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::PushFrame, 1);
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
fn subshell_end_exit_code_33_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(33)));
    match vm.run() {
        VMResult::Ok(Value::Status(33)) => {}
        other => panic!("exit code 33 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_eight_consecutive_runs_without_reset_still_blocked() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    for _ in 0..8 {
        let _ = vm.run();
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

// ─── Pin tests batch C (handwritten) ──────────────────────────────────────

#[test]
fn subshell_end_exit_code_34_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(34)));
    match vm.run() {
        VMResult::Ok(Value::Status(34)) => {}
        other => panic!("exit code 34 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_dup2_on_stack() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(7), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::Dup2, 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_35_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(35)));
    match vm.run() {
        VMResult::Ok(Value::Status(35)) => {}
        other => panic!("exit code 35 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_rot_on_stack() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::LoadInt(2), 1);
    b.emit(Op::LoadInt(3), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::Rot, 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_36_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(36)));
    match vm.run() {
        VMResult::Ok(Value::Status(36)) => {}
        other => panic!("exit code 36 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_pow_on_stack() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(2), 1);
    b.emit(Op::LoadInt(3), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::Pow, 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_37_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(37)));
    match vm.run() {
        VMResult::Ok(Value::Status(37)) => {}
        other => panic!("exit code 37 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_spaceship_on_stack() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::LoadInt(2), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::Spaceship, 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_38_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(38)));
    match vm.run() {
        VMResult::Ok(Value::Status(38)) => {}
        other => panic!("exit code 38 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_load_float() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::LoadFloat(3.14), 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_39_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(39)));
    match vm.run() {
        VMResult::Ok(Value::Status(39)) => {}
        other => panic!("exit code 39 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_str_eq_on_string_consts() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    let a = b.add_constant(Value::str("a"));
    let bidx = b.add_constant(Value::str("b"));
    b.emit(Op::LoadConst(a), 1);
    b.emit(Op::LoadConst(bidx), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::StrEq, 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_40_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(40)));
    match vm.run() {
        VMResult::Ok(Value::Status(40)) => {}
        other => panic!("exit code 40 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_nine_consecutive_runs_without_reset_still_blocked() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    for _ in 0..9 {
        let _ = vm.run();
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_41_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(41)));
    match vm.run() {
        VMResult::Ok(Value::Status(41)) => {}
        other => panic!("exit code 41 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_ten_consecutive_runs_without_reset_still_blocked() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    for _ in 0..10 {
        let _ = vm.run();
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_42_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(42)));
    match vm.run() {
        VMResult::Ok(Value::Status(42)) => {}
        other => panic!("exit code 42 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_num_eq_jump_if_false_keep() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(5), 1);
    b.emit(Op::LoadInt(5), 1);
    b.emit(Op::NumEq, 1);
    let jmp = b.emit(Op::JumpIfFalseKeep(0), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.patch_jump(jmp, b.current_pos());
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_43_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(43)));
    match vm.run() {
        VMResult::Ok(Value::Status(43)) => {}
        other => panic!("exit code 43 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_num_eq_jump_if_true_keep() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::LoadInt(2), 1);
    b.emit(Op::NumEq, 1);
    let jmp = b.emit(Op::JumpIfTrueKeep(0), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.patch_jump(jmp, b.current_pos());
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_44_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(44)));
    match vm.run() {
        VMResult::Ok(Value::Status(44)) => {}
        other => panic!("exit code 44 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_after_subshell_end_before_second_getstatus() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(60)));
    vm.register_builtin(100, builtin_request_halt);
    match vm.run() {
        VMResult::Ok(Value::Int(0)) => {}
        other => panic!("second GetStatus MUST NOT run after halt, got {:?}", other),
    }
    assert_eq!(vm.last_status, 60);
}

#[test]
fn subshell_end_exit_code_45_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(45)));
    match vm.run() {
        VMResult::Ok(Value::Status(45)) => {}
        other => panic!("exit code 45 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_between_loadint_and_setstatus() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(70), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::SetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.last_status = 0;
    vm.register_builtin(100, builtin_request_halt);
    let _ = vm.run();
    assert_eq!(vm.last_status, 0, "SetStatus MUST NOT run after halt");
}

#[test]
fn subshell_end_exit_code_46_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(46)));
    match vm.run() {
        VMResult::Ok(Value::Status(46)) => {}
        other => panic!("exit code 46 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_reset_twice_then_subshell_end_on_third_chunk() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    SUBSHELL_END_CALLS.store(0, Ordering::SeqCst);
    let noop = ChunkBuilder::new().build();
    let sub = {
        let mut b = ChunkBuilder::new();
        b.emit(Op::SubshellEnd, 1);
        b.build()
    };
    let mut vm = VM::new(noop.clone());
    vm.set_shell_host(Box::new(CountingSubshellEndHost(61)));
    vm.request_halt();
    let _ = vm.run();
    vm.reset(noop.clone());
    vm.reset(sub);
    let _ = vm.run();
    assert_eq!(SUBSHELL_END_CALLS.load(Ordering::SeqCst), 1);
    assert_eq!(vm.last_status, 61);
}

#[test]
fn subshell_end_exit_code_47_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(47)));
    match vm.run() {
        VMResult::Ok(Value::Status(47)) => {}
        other => panic!("exit code 47 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_callee_halt_before_return_caller_gets_builtin_result() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    let mut b = ChunkBuilder::new();
    let name = b.add_name("halt_fn");
    b.emit(Op::Call(name, 0), 1);
    b.emit(Op::LoadInt(500), 1);
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
            "caller MUST get halt builtin Int(0), not LoadInt(500), got {:?}",
            other
        ),
    }
}

#[test]
fn subshell_end_exit_code_48_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(48)));
    match vm.run() {
        VMResult::Ok(Value::Status(48)) => {}
        other => panic!("exit code 48 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_subshell_begin_end_then_halt_before_getstatus() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(62)));
    vm.register_builtin(100, builtin_request_halt);
    let _ = vm.run();
    assert_eq!(vm.last_status, 62);
}

#[test]
fn subshell_end_exit_code_49_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(49)));
    match vm.run() {
        VMResult::Ok(Value::Status(49)) => {}
        other => panic!("exit code 49 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_three_extended_handlers_first_halts() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    static COUNTS: AtomicU32 = AtomicU32::new(0);
    let mut b = ChunkBuilder::new();
    b.emit(Op::Extended(1, 0), 1);
    b.emit(Op::Extended(2, 0), 1);
    b.emit(Op::Extended(3, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_extension_handler(Box::new(|vm, id, _arg| {
        COUNTS.fetch_add(1, Ordering::SeqCst);
        if id == 1 {
            vm.request_halt();
        }
    }));
    let _ = vm.run();
    assert_eq!(COUNTS.load(Ordering::SeqCst), 1);
}

#[test]
fn subshell_end_exit_code_50_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(50)));
    match vm.run() {
        VMResult::Ok(Value::Status(50)) => {}
        other => panic!("exit code 50 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_pipeline_begin_stage_halt_before_end() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    PIPELINE_END_CALLS.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::PipelineBegin(2), 1);
    b.emit(Op::PipelineStage, 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::PipelineEnd, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(CountingPipelineHost));
    vm.register_builtin(100, builtin_request_halt);
    let _ = vm.run();
    assert_eq!(PIPELINE_END_CALLS.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_146_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(146)));
    match vm.run() {
        VMResult::Ok(Value::Status(146)) => {}
        other => panic!("exit code 146 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_after_open_pipeline_subshell_end_before_stage() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    PIPELINE_STAGE_CALLS.store(0, Ordering::SeqCst);
    struct SubshellStageHost60;
    impl ShellHost for SubshellStageHost60 {
        fn pipeline_stage(&mut self) {
            PIPELINE_STAGE_CALLS.fetch_add(1, Ordering::SeqCst);
        }
        fn subshell_end(&mut self) -> Option<i32> {
            Some(60)
        }
    }
    let mut b = ChunkBuilder::new();
    b.emit(Op::PipelineBegin(2), 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::PipelineStage, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(SubshellStageHost60));
    vm.register_builtin(100, builtin_request_halt);
    let _ = vm.run();
    assert_eq!(PIPELINE_STAGE_CALLS.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 60);
}

#[test]
fn subshell_end_exit_code_147_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(147)));
    match vm.run() {
        VMResult::Ok(Value::Status(147)) => {}
        other => panic!("exit code 147 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_second_subshell_begin_after_first_end() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    SUBSHELL_BEGIN_CALLS.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::SubshellBegin, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(BeginEndCountingHost(63)));
    vm.register_builtin(100, builtin_request_halt);
    let _ = vm.run();
    assert_eq!(SUBSHELL_BEGIN_CALLS.load(Ordering::SeqCst), 1);
}

#[test]
fn subshell_end_exit_code_148_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(148)));
    match vm.run() {
        VMResult::Ok(Value::Status(148)) => {}
        other => panic!("exit code 148 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_wide_handler_halts_before_subsequent_subshell_end() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    SUBSHELL_END_CALLS.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::ExtendedWide(5, 0), 1);
    b.emit(Op::SubshellEnd, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(CountingSubshellEndHost(64)));
    vm.set_extension_wide_handler(Box::new(|vm, _id, _payload| {
        vm.request_halt();
    }));
    let _ = vm.run();
    assert_eq!(SUBSHELL_END_CALLS.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_149_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(149)));
    match vm.run() {
        VMResult::Ok(Value::Status(149)) => {}
        other => panic!("exit code 149 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_pre_run_then_reset_subshell_getstatus_lifecycle() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    let sub = {
        let mut b = ChunkBuilder::new();
        b.emit(Op::SubshellEnd, 1);
        b.emit(Op::GetStatus, 1);
        b.build()
    };
    let mut vm = VM::new(sub.clone());
    vm.set_shell_host(Box::new(StatusReturningHost(65)));
    vm.request_halt();
    match vm.run() {
        VMResult::Halted => {}
        other => panic!("pre-run halt MUST return Halted, got {:?}", other),
    }
    vm.reset(sub);
    match vm.run() {
        VMResult::Ok(Value::Status(65)) => {}
        other => panic!("after reset subshell MUST apply Some(65), got {:?}", other),
    }
}

#[test]
fn subshell_end_status_minus_nine_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(-9)));
    match vm.run() {
        VMResult::Ok(Value::Status(-9)) => {}
        other => panic!("status -9 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_third_push_frame() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::PushFrame, 1);
    b.emit(Op::PushFrame, 1);
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
fn subshell_end_status_minus_ten_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(-10)));
    match vm.run() {
        VMResult::Ok(Value::Status(-10)) => {}
        other => panic!("status -10 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_between_two_pipeline_end_ops_in_flat_chunk() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    PIPELINE_END_CALLS.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::PipelineEnd, 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::PipelineEnd, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(CountingPipelineHost));
    vm.register_builtin(100, builtin_request_halt);
    let _ = vm.run();
    assert_eq!(
        PIPELINE_END_CALLS.load(Ordering::SeqCst),
        1,
        "first PipelineEnd MUST run; second MUST NOT after halt"
    );
}

#[test]
fn subshell_twenty_six_end_incrementing_from_ten() {
    let mut b = ChunkBuilder::new();
    for _ in 0..26 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(IncrementingSubshellHost { next: 10 }));
    match vm.run() {
        VMResult::Ok(Value::Status(35)) => {}
        other => panic!("twenty-sixth subshell_end(Some(35)) MUST win, got {:?}", other),
    }
}

#[test]
fn request_halt_extension_reads_subshell_status_then_halts_before_loadint() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    static SEEN: AtomicU32 = AtomicU32::new(0);
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::Extended(8, 0), 1);
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(66)));
    vm.set_extension_handler(Box::new(|vm, _id, _arg| {
        SEEN.store(vm.last_status as u32, Ordering::SeqCst);
        vm.request_halt();
    }));
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(SEEN.load(Ordering::SeqCst), 66);
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_twenty_seven_end_incrementing_from_thirteen() {
    let mut b = ChunkBuilder::new();
    for _ in 0..27 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(IncrementingSubshellHost { next: 13 }));
    match vm.run() {
        VMResult::Ok(Value::Status(39)) => {}
        other => panic!("twenty-seventh subshell_end(Some(39)) MUST win, got {:?}", other),
    }
}

// ─── Pin tests batch D (handwritten) ──────────────────────────────────────

#[test]
fn subshell_end_exit_code_151_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(151)));
    match vm.run() {
        VMResult::Ok(Value::Status(151)) => {}
        other => panic!("exit code 151 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_num_ne_keep_conditional_jump() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(5), 1);
    b.emit(Op::LoadInt(5), 1);
    b.emit(Op::NumNe, 1);
    let jmp = b.emit(Op::JumpIfTrueKeep(0), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.patch_jump(jmp, b.current_pos());
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_152_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(152)));
    match vm.run() {
        VMResult::Ok(Value::Status(152)) => {}
        other => panic!("exit code 152 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_backward_jump_if_true() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    let fwd = b.emit(Op::Jump(0), 1);
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::NumEq, 1);
    let back = b.emit(Op::JumpIfTrue(0), 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    b.patch_jump(back, b.current_pos());
    b.patch_jump(fwd, back);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_153_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(153)));
    match vm.run() {
        VMResult::Ok(Value::Status(153)) => {}
        other => panic!("exit code 153 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_after_first_of_three_subshell_ends() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    SUBSHELL_END_CALLS.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::SubshellEnd, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(CountingSubshellEndHost(67)));
    vm.register_builtin(100, builtin_request_halt);
    let _ = vm.run();
    assert_eq!(SUBSHELL_END_CALLS.load(Ordering::SeqCst), 1);
    assert_eq!(vm.last_status, 67);
}

#[test]
fn subshell_end_exit_code_154_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(154)));
    match vm.run() {
        VMResult::Ok(Value::Status(154)) => {}
        other => panic!("exit code 154 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_subshell_status_persists_when_getstatus_on_stack_before_halt() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(68)));
    vm.register_builtin(100, builtin_request_halt);
    match vm.run() {
        VMResult::Ok(Value::Int(0)) => {}
        other => panic!("halt MUST return builtin, not Status(68), got {:?}", other),
    }
    assert_eq!(vm.last_status, 68);
}

#[test]
fn subshell_end_exit_code_155_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(155)));
    match vm.run() {
        VMResult::Ok(Value::Status(155)) => {}
        other => panic!("exit code 155 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_five_push_frames_halt_before_pop_frame() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    for _ in 0..5 {
        b.emit(Op::PushFrame, 1);
    }
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::PopFrame, 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_twenty_eight_end_incrementing_from_five() {
    let mut b = ChunkBuilder::new();
    for _ in 0..28 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(IncrementingSubshellHost { next: 5 }));
    match vm.run() {
        VMResult::Ok(Value::Status(32)) => {}
        other => panic!("twenty-eighth subshell_end(Some(32)) MUST win, got {:?}", other),
    }
}

#[test]
fn request_halt_call_op_in_caller_blocked_after_callee_halt() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    let inner = b.add_name("inner_h");
    let outer = b.add_name("outer_h");
    b.emit(Op::Call(outer, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let skip = b.emit(Op::Jump(0), 1);
    let oe = b.current_pos();
    b.add_sub_entry(outer, oe);
    b.emit(Op::Call(inner, 0), 1);
    b.emit(Op::Return, 1);
    let ie = b.current_pos();
    b.add_sub_entry(inner, ie);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::Return, 1);
    b.patch_jump(skip, b.current_pos());
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    let _ = vm.run();
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_twenty_nine_end_incrementing_from_seventeen() {
    let mut b = ChunkBuilder::new();
    for _ in 0..29 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(IncrementingSubshellHost { next: 17 }));
    match vm.run() {
        VMResult::Ok(Value::Status(45)) => {}
        other => panic!("twenty-ninth subshell_end(Some(45)) MUST win, got {:?}", other),
    }
}

#[test]
fn request_halt_reset_preserves_counting_subshell_host_across_three_chunks() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    SUBSHELL_END_CALLS.store(0, Ordering::SeqCst);
    let halt = {
        let mut b = ChunkBuilder::new();
        b.emit(Op::CallBuiltin(100, 0), 1);
        b.build()
    };
    let end = {
        let mut b = ChunkBuilder::new();
        b.emit(Op::SubshellEnd, 1);
        b.build()
    };
    let mut vm = VM::new(halt);
    vm.set_shell_host(Box::new(CountingSubshellEndHost(69)));
    vm.register_builtin(100, builtin_request_halt);
    let _ = vm.run();
    vm.reset(end.clone());
    let _ = vm.run();
    vm.reset(end);
    let _ = vm.run();
    assert_eq!(SUBSHELL_END_CALLS.load(Ordering::SeqCst), 2);
    assert_eq!(vm.last_status, 69);
}

#[test]
fn subshell_thirty_end_incrementing_from_nineteen() {
    let mut b = ChunkBuilder::new();
    for _ in 0..30 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(IncrementingSubshellHost { next: 19 }));
    match vm.run() {
        VMResult::Ok(Value::Status(48)) => {}
        other => panic!("thirtieth subshell_end(Some(48)) MUST win, got {:?}", other),
    }
}

#[test]
fn request_halt_between_getstatus_and_loadint_blocks_loadint() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(71), 1);
    b.emit(Op::SetStatus, 1);
    b.emit(Op::GetStatus, 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::LoadInt(72), 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_status_minus_eleven_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(-11)));
    match vm.run() {
        VMResult::Ok(Value::Status(-11)) => {}
        other => panic!("status -11 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_halt_only_after_loadint_returns_undef_not_int() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(99), 1);
    b.emit(Op::CallBuiltin(61, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(61, builtin_halt_only);
    match vm.run() {
        VMResult::Ok(Value::Undef) => {}
        other => panic!("halt_only MUST return Undef over LoadInt(99), got {:?}", other),
    }
}

#[test]
fn subshell_eighth_some_on_eighth_end_applies_seventy() {
    let mut b = ChunkBuilder::new();
    for _ in 0..8 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    struct EighthSomeHost {
        calls: u32,
    }
    impl ShellHost for EighthSomeHost {
        fn subshell_end(&mut self) -> Option<i32> {
            self.calls += 1;
            if self.calls == 8 {
                Some(70)
            } else {
                None
            }
        }
    }
    let mut vm = VM::new(b.build());
    vm.last_status = 3;
    vm.set_shell_host(Box::new(EighthSomeHost { calls: 0 }));
    match vm.run() {
        VMResult::Ok(Value::Status(70)) => {}
        other => panic!("eighth subshell_end(Some(70)) MUST apply, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_second_call_in_caller_after_callee_returns() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    static OUTER_CALLS: AtomicU32 = AtomicU32::new(0);
    fn outer_fn(_vm: &mut VM, _argc: u8) -> Value {
        OUTER_CALLS.fetch_add(1, Ordering::SeqCst);
        Value::Int(0)
    }
    let mut b = ChunkBuilder::new();
    let name = b.add_name("ret5");
    b.emit(Op::Call(name, 0), 1);
    b.emit(Op::CallBuiltin(300, 0), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    let skip = b.emit(Op::Jump(0), 1);
    let entry = b.current_pos();
    b.add_sub_entry(name, entry);
    b.emit(Op::LoadInt(5), 1);
    b.emit(Op::ReturnValue, 1);
    b.patch_jump(skip, b.current_pos());
    let mut vm = VM::new(b.build());
    vm.register_builtin(300, outer_fn);
    vm.register_builtin(100, builtin_request_halt);
    match vm.run() {
        VMResult::Ok(Value::Int(0)) => {}
        other => panic!("halt MUST be stack top after callee+outer builtin, got {:?}", other),
    }
    assert_eq!(OUTER_CALLS.load(Ordering::SeqCst), 1);
}

#[test]
fn subshell_pipeline_end_then_subshell_end_getstatus_reads_subshell() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::PipelineEnd, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(PipelineThenSubshellHost));
    match vm.run() {
        VMResult::Ok(Value::Status(9)) => {}
        other => panic!("GetStatus MUST read subshell(9) not pipeline(3), got {:?}", other),
    }
}

#[test]
fn request_halt_after_subshell_end_in_nested_begin_end_pair() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    SUBSHELL_END_CALLS.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::SubshellEnd, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(CountingSubshellEndHost(71)));
    vm.register_builtin(100, builtin_request_halt);
    let _ = vm.run();
    assert_eq!(SUBSHELL_END_CALLS.load(Ordering::SeqCst), 1);
    assert_eq!(vm.last_status, 71);
}

#[test]
fn subshell_setstatus_after_begin_before_end_overwritten_by_end() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::LoadInt(80), 1);
    b.emit(Op::SetStatus, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(72)));
    match vm.run() {
        VMResult::Ok(Value::Status(72)) => {}
        other => panic!(
            "subshell_end(Some(72)) MUST beat in-pair SetStatus(80), got {:?}",
            other
        ),
    }
}

#[test]
fn request_halt_stops_before_load_true_between_two_loadints() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(10), 1);
    b.emit(Op::LoadInt(20), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::LoadTrue, 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_two_orphan_ends_none_then_some_host() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.last_status = 15;
    vm.set_shell_host(Box::new(NoneThenSomeHost { calls: 0 }));
    match vm.run() {
        VMResult::Ok(Value::Status(42)) => {}
        other => panic!("NoneThenSomeHost second Some(42) MUST win, got {:?}", other),
    }
}

#[test]
fn request_halt_after_pipeline_end_subshell_end_both_complete_before_halt() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    PIPELINE_END_CALLS.store(0, Ordering::SeqCst);
    SUBSHELL_END_CALLS.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::PipelineEnd, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    struct BothCountingHost;
    impl ShellHost for BothCountingHost {
        fn pipeline_end(&mut self) -> i32 {
            PIPELINE_END_CALLS.fetch_add(1, Ordering::SeqCst);
            2
        }
        fn subshell_end(&mut self) -> Option<i32> {
            SUBSHELL_END_CALLS.fetch_add(1, Ordering::SeqCst);
            Some(73)
        }
    }
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(BothCountingHost));
    vm.register_builtin(100, builtin_request_halt);
    let _ = vm.run();
    assert_eq!(PIPELINE_END_CALLS.load(Ordering::SeqCst), 1);
    assert_eq!(SUBSHELL_END_CALLS.load(Ordering::SeqCst), 1);
    assert_eq!(vm.last_status, 73);
}

#[test]
fn subshell_getstatus_after_end_without_begin_on_stack() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(74)));
    match vm.run() {
        VMResult::Ok(Value::Status(74)) => {}
        other => panic!("orphan SubshellEnd MUST update GetStatus, got {:?}", other),
    }
}

#[test]
fn request_halt_mid_chunk_setstatus_before_halt_not_reached() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::LoadInt(75), 1);
    b.emit(Op::SetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.last_status = 1;
    vm.register_builtin(100, builtin_request_halt);
    let _ = vm.run();
    assert_eq!(vm.last_status, 1);
}

#[test]
fn subshell_four_begin_two_end_incrementing_status() {
    let mut b = ChunkBuilder::new();
    for _ in 0..4 {
        b.emit(Op::SubshellBegin, 1);
    }
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(IncrementingSubshellHost { next: 20 }));
    match vm.run() {
        VMResult::Ok(Value::Status(21)) => {}
        other => panic!("second end Some(21) MUST win, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_second_extended_wide() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    static WIDE1: AtomicU32 = AtomicU32::new(0);
    static WIDE2: AtomicU32 = AtomicU32::new(0);
    let mut b = ChunkBuilder::new();
    b.emit(Op::ExtendedWide(1, 0), 1);
    b.emit(Op::ExtendedWide(2, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_extension_wide_handler(Box::new(|vm, id, _payload| {
        if id == 1 {
            WIDE1.fetch_add(1, Ordering::SeqCst);
            vm.request_halt();
        } else {
            WIDE2.fetch_add(1, Ordering::SeqCst);
        }
    }));
    let _ = vm.run();
    assert_eq!(WIDE1.load(Ordering::SeqCst), 1);
    assert_eq!(WIDE2.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_halt_reset_halt_reset_subshell_end_third_run() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    SUBSHELL_END_CALLS.store(0, Ordering::SeqCst);
    let sub = {
        let mut b = ChunkBuilder::new();
        b.emit(Op::SubshellEnd, 1);
        b.emit(Op::CallBuiltin(100, 0), 1);
        b.build()
    };
    let end_only = {
        let mut b = ChunkBuilder::new();
        b.emit(Op::SubshellEnd, 1);
        b.build()
    };
    let mut vm = VM::new(sub.clone());
    vm.set_shell_host(Box::new(CountingSubshellEndHost(76)));
    vm.register_builtin(100, builtin_request_halt);
    let _ = vm.run();
    vm.reset(sub);
    let _ = vm.run();
    vm.reset(end_only);
    let _ = vm.run();
    assert_eq!(SUBSHELL_END_CALLS.load(Ordering::SeqCst), 3);
    assert_eq!(vm.last_status, 76);
}

#[test]
fn request_halt_empty_chunk_after_reset_on_pre_run_halt() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let empty = ChunkBuilder::new().build();
    let run = {
        let mut b = ChunkBuilder::new();
        b.emit(Op::CallBuiltin(7, 0), 1);
        b.build()
    };
    let mut vm = VM::new(empty.clone());
    vm.register_builtin(7, builtin_touched);
    vm.request_halt();
    match vm.run() {
        VMResult::Halted => {}
        other => panic!("pre-run halt on empty chunk MUST return Halted, got {:?}", other),
    }
    vm.reset(run);
    let _ = vm.run();
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 1);
}

#[test]
fn subshell_none_host_reset_then_status_host_on_same_chunk_type() {
    let chunk = {
        let mut b = ChunkBuilder::new();
        b.emit(Op::SubshellEnd, 1);
        b.emit(Op::GetStatus, 1);
        b.build()
    };
    let mut vm = VM::new(chunk.clone());
    vm.set_shell_host(Box::new(NoneHost));
    vm.last_status = 8;
    match vm.run() {
        VMResult::Ok(Value::Status(8)) => {}
        other => panic!("NoneHost MUST preserve last_status(8), got {:?}", other),
    }
    vm.reset(chunk);
    vm.set_shell_host(Box::new(StatusReturningHost(77)));
    match vm.run() {
        VMResult::Ok(Value::Status(77)) => {}
        other => panic!("StatusReturningHost MUST apply Some(77), got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_call_with_one_arg_in_caller() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    let name = b.add_name("noop");
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::Call(name, 1), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let skip = b.emit(Op::Jump(0), 1);
    let entry = b.current_pos();
    b.add_sub_entry(name, entry);
    b.emit(Op::Return, 1);
    b.patch_jump(skip, b.current_pos());
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    let _ = vm.run();
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_between_two_setstatus_second_end_wins() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(10), 1);
    b.emit(Op::SetStatus, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::LoadInt(20), 1);
    b.emit(Op::SetStatus, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(IncrementingSubshellHost { next: 30 }));
    match vm.run() {
        VMResult::Ok(Value::Status(31)) => {}
        other => panic!("second end Some(31) MUST beat SetStatus(20), got {:?}", other),
    }
}

#[test]
fn request_halt_subshell_begin_count_one_when_halt_after_begin() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    SUBSHELL_BEGIN_CALLS.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(BeginEndCountingHost(78)));
    vm.register_builtin(100, builtin_request_halt);
    let _ = vm.run();
    assert_eq!(SUBSHELL_BEGIN_CALLS.load(Ordering::SeqCst), 1);
}

#[test]
fn subshell_triple_getstatus_after_single_end_all_read_same() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    b.emit(Op::GetStatus, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(79)));
    match vm.run() {
        VMResult::Ok(Value::Status(79)) => {}
        other => panic!("top GetStatus(79) MUST win over deeper copies, got {:?}", other),
    }
}

#[test]
fn request_halt_eleven_consecutive_runs_without_reset_still_blocked() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    for _ in 0..11 {
        let _ = vm.run();
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_156_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(156)));
    match vm.run() {
        VMResult::Ok(Value::Status(156)) => {}
        other => panic!("exit code 156 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_twelve_consecutive_runs_without_reset_still_blocked() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    for _ in 0..12 {
        let _ = vm.run();
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_157_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(157)));
    match vm.run() {
        VMResult::Ok(Value::Status(157)) => {}
        other => panic!("exit code 157 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_after_subshell_end_preserves_deeper_loadint_on_stack() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(100), 1);
    b.emit(Op::LoadInt(200), 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(80)));
    vm.register_builtin(100, builtin_request_halt);
    match vm.run() {
        VMResult::Ok(Value::Int(0)) => {}
        other => panic!("halt MUST return builtin on top, got {:?}", other),
    }
    assert_eq!(vm.last_status, 80);
}

#[test]
fn subshell_end_exit_code_158_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(158)));
    match vm.run() {
        VMResult::Ok(Value::Status(158)) => {}
        other => panic!("exit code 158 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_full_lifecycle_host_swap_reset_getstatus() {
    let c1 = {
        let mut b = ChunkBuilder::new();
        b.emit(Op::SubshellEnd, 1);
        b.build()
    };
    let c2 = {
        let mut b = ChunkBuilder::new();
        b.emit(Op::GetStatus, 1);
        b.build()
    };
    let mut vm = VM::new(c1);
    vm.set_shell_host(Box::new(StatusReturningHost(81)));
    let _ = vm.run();
    vm.set_shell_host(Box::new(DefaultHost));
    vm.reset(c2);
    match vm.run() {
        VMResult::Ok(Value::Status(0)) => {}
        other => panic!("DefaultHost after reset MUST leave last_status(0), got {:?}", other),
    }
}

#[test]
fn subshell_end_after_halt_reset_on_same_vm_reads_new_host_status() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    let run = {
        let mut b = ChunkBuilder::new();
        b.emit(Op::CallBuiltin(100, 0), 1);
        b.build()
    };
    let sub = {
        let mut b = ChunkBuilder::new();
        b.emit(Op::SubshellEnd, 1);
        b.emit(Op::GetStatus, 1);
        b.build()
    };
    let mut vm = VM::new(run);
    vm.set_shell_host(Box::new(StatusReturningHost(82)));
    vm.register_builtin(100, builtin_request_halt);
    let _ = vm.run();
    vm.set_shell_host(Box::new(StatusReturningHost(83)));
    vm.reset(sub);
    match vm.run() {
        VMResult::Ok(Value::Status(83)) => {}
        other => panic!("swapped host MUST apply Some(83) after reset, got {:?}", other),
    }
}

#[test]
fn request_halt_between_subshell_end_and_pipeline_end() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    PIPELINE_END_CALLS.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::PipelineEnd, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(CountingPipelineHost));
    vm.register_builtin(100, builtin_request_halt);
    let _ = vm.run();
    assert_eq!(PIPELINE_END_CALLS.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 0);
}

#[test]
fn subshell_loadint_getstatus_subshell_end_final_getstatus_wins() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(44), 1);
    b.emit(Op::GetStatus, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.last_status = 5;
    vm.set_shell_host(Box::new(StatusReturningHost(84)));
    match vm.run() {
        VMResult::Ok(Value::Status(84)) => {}
        other => panic!("final GetStatus(84) MUST be stack top, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_second_subshell_end_after_first_updates_status() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    SUBSHELL_END_CALLS.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::SubshellEnd, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(CountingSubshellEndHost(85)));
    vm.register_builtin(100, builtin_request_halt);
    let _ = vm.run();
    assert_eq!(SUBSHELL_END_CALLS.load(Ordering::SeqCst), 1);
    assert_eq!(vm.last_status, 85);
}

#[test]
fn subshell_end_exit_code_159_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(159)));
    match vm.run() {
        VMResult::Ok(Value::Status(159)) => {}
        other => panic!("exit code 159 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_triple_halt_from_builtin_still_single_clean_stop() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(50, 0), 1);
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(50, builtin_triple_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_160_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(160)));
    match vm.run() {
        VMResult::Ok(Value::Status(160)) => {}
        other => panic!("exit code 160 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_reset_clears_halt_subshell_end_then_getstatus_on_fresh_chunk() {
    let halt = {
        let mut b = ChunkBuilder::new();
        b.emit(Op::CallBuiltin(100, 0), 1);
        b.build()
    };
    let sub = {
        let mut b = ChunkBuilder::new();
        b.emit(Op::SubshellEnd, 1);
        b.emit(Op::GetStatus, 1);
        b.build()
    };
    let mut vm = VM::new(halt);
    vm.set_shell_host(Box::new(StatusReturningHost(86)));
    vm.register_builtin(100, builtin_request_halt);
    let _ = vm.run();
    vm.reset(sub);
    match vm.run() {
        VMResult::Ok(Value::Status(86)) => {}
        other => panic!("subshell+GetStatus MUST work after reset, got {:?}", other),
    }
}

#[test]
fn subshell_first_some_then_none_on_second_end_preserves_first() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(FirstSomeThenNoneHost {
        calls: 0,
        status: 87,
    }));
    match vm.run() {
        VMResult::Ok(Value::Status(87)) => {}
        other => panic!("FirstSomeThenNoneHost MUST preserve Some(87), got {:?}", other),
    }
}

#[test]
fn request_halt_after_getstatus_does_not_pop_status_before_halt_builtin() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(88), 1);
    b.emit(Op::SetStatus, 1);
    b.emit(Op::GetStatus, 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    match vm.run() {
        VMResult::Ok(Value::Int(0)) => {}
        other => panic!("run MUST return halt builtin, not Status(88), got {:?}", other),
    }
}

// ─── Pin tests batch E (handwritten) ──────────────────────────────────────

#[test]
fn subshell_end_exit_code_51_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(51)));
    match vm.run() {
        VMResult::Ok(Value::Status(51)) => {}
        other => panic!("exit code 51 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_concat_on_string_consts() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    let a = b.add_constant(Value::str("foo"));
    let bidx = b.add_constant(Value::str("bar"));
    b.emit(Op::LoadConst(a), 1);
    b.emit(Op::LoadConst(bidx), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::Concat, 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_52_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(52)));
    match vm.run() {
        VMResult::Ok(Value::Status(52)) => {}
        other => panic!("exit code 52 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_string_len() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    let s = b.add_constant(Value::str("hello"));
    b.emit(Op::LoadConst(s), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::StringLen, 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_53_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(53)));
    match vm.run() {
        VMResult::Ok(Value::Status(53)) => {}
        other => panic!("exit code 53 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_string_repeat() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    let s = b.add_constant(Value::str("x"));
    b.emit(Op::LoadConst(s), 1);
    b.emit(Op::LoadInt(3), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::StringRepeat, 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_54_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(54)));
    match vm.run() {
        VMResult::Ok(Value::Status(54)) => {}
        other => panic!("exit code 54 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_thirteen_consecutive_runs_without_reset_still_blocked() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    for _ in 0..13 {
        let _ = vm.run();
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_55_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(55)));
    match vm.run() {
        VMResult::Ok(Value::Status(55)) => {}
        other => panic!("exit code 55 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_fourteen_consecutive_runs_without_reset_still_blocked() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    for _ in 0..14 {
        let _ = vm.run();
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_56_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(56)));
    match vm.run() {
        VMResult::Ok(Value::Status(56)) => {}
        other => panic!("exit code 56 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_str_ne_on_consts() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    let a = b.add_constant(Value::str("a"));
    let bidx = b.add_constant(Value::str("b"));
    b.emit(Op::LoadConst(a), 1);
    b.emit(Op::LoadConst(bidx), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::StrNe, 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_57_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(57)));
    match vm.run() {
        VMResult::Ok(Value::Status(57)) => {}
        other => panic!("exit code 57 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_after_subshell_end_before_pipeline_begin() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    PIPELINE_BEGIN_CALLS.store(0, Ordering::SeqCst);
    struct SubshellBeginHost88;
    impl ShellHost for SubshellBeginHost88 {
        fn pipeline_begin(&mut self, _n: u8) {
            PIPELINE_BEGIN_CALLS.fetch_add(1, Ordering::SeqCst);
        }
        fn subshell_end(&mut self) -> Option<i32> {
            Some(88)
        }
    }
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::PipelineBegin(2), 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(SubshellBeginHost88));
    vm.register_builtin(100, builtin_request_halt);
    let _ = vm.run();
    assert_eq!(PIPELINE_BEGIN_CALLS.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 88);
}

#[test]
fn subshell_end_exit_code_58_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(58)));
    match vm.run() {
        VMResult::Ok(Value::Status(58)) => {}
        other => panic!("exit code 58 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_59_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(59)));
    match vm.run() {
        VMResult::Ok(Value::Status(59)) => {}
        other => panic!("exit code 59 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_reset_then_subshell_begin_end_on_fresh_chunk() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    SUBSHELL_BEGIN_CALLS.store(0, Ordering::SeqCst);
    let halt = {
        let mut b = ChunkBuilder::new();
        b.emit(Op::CallBuiltin(100, 0), 1);
        b.build()
    };
    let pair = {
        let mut b = ChunkBuilder::new();
        b.emit(Op::SubshellBegin, 1);
        b.emit(Op::SubshellEnd, 1);
        b.build()
    };
    let mut vm = VM::new(halt);
    vm.set_shell_host(Box::new(BeginEndCountingHost(90)));
    vm.register_builtin(100, builtin_request_halt);
    let _ = vm.run();
    vm.reset(pair);
    let _ = vm.run();
    assert_eq!(SUBSHELL_BEGIN_CALLS.load(Ordering::SeqCst), 1);
    assert_eq!(vm.last_status, 90);
}

#[test]
fn subshell_end_exit_code_60_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(60)));
    match vm.run() {
        VMResult::Ok(Value::Status(60)) => {}
        other => panic!("exit code 60 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_between_two_load_const_ops() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    let c1 = b.add_constant(Value::Int(1));
    let c2 = b.add_constant(Value::Int(2));
    b.emit(Op::LoadConst(c1), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::LoadConst(c2), 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_61_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(61)));
    match vm.run() {
        VMResult::Ok(Value::Status(61)) => {}
        other => panic!("exit code 61 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_callee_returnvalue_before_halt_in_caller_stack_top_is_halt() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    let mut b = ChunkBuilder::new();
    let name = b.add_name("rv3");
    b.emit(Op::Call(name, 0), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    let skip = b.emit(Op::Jump(0), 1);
    let entry = b.current_pos();
    b.add_sub_entry(name, entry);
    b.emit(Op::LoadInt(3), 1);
    b.emit(Op::ReturnValue, 1);
    b.patch_jump(skip, b.current_pos());
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    match vm.run() {
        VMResult::Ok(Value::Int(0)) => {}
        other => panic!("halt MUST be stack top over ReturnValue(3), got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_62_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(62)));
    match vm.run() {
        VMResult::Ok(Value::Status(62)) => {}
        other => panic!("exit code 62 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_after_pipeline_stage_before_pipeline_end_in_open_pipeline() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    PIPELINE_END_CALLS.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::PipelineBegin(2), 1);
    b.emit(Op::PipelineStage, 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::PipelineEnd, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(CountingPipelineHost));
    vm.register_builtin(100, builtin_request_halt);
    let _ = vm.run();
    assert_eq!(PIPELINE_END_CALLS.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_63_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(63)));
    match vm.run() {
        VMResult::Ok(Value::Status(63)) => {}
        other => panic!("exit code 63 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_second_extended_after_first_halts() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    static FIRST: AtomicU32 = AtomicU32::new(0);
    static SECOND: AtomicU32 = AtomicU32::new(0);
    let mut b = ChunkBuilder::new();
    b.emit(Op::Extended(10, 0), 1);
    b.emit(Op::Extended(11, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_extension_handler(Box::new(|vm, id, _arg| {
        if id == 10 {
            FIRST.fetch_add(1, Ordering::SeqCst);
            vm.request_halt();
        } else {
            SECOND.fetch_add(1, Ordering::SeqCst);
        }
    }));
    let _ = vm.run();
    assert_eq!(FIRST.load(Ordering::SeqCst), 1);
    assert_eq!(SECOND.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_64_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(64)));
    match vm.run() {
        VMResult::Ok(Value::Status(64)) => {}
        other => panic!("exit code 64 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_pre_run_halt_reset_halt_reset_then_subshell_runs() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    SUBSHELL_END_CALLS.store(0, Ordering::SeqCst);
    let noop = ChunkBuilder::new().build();
    let sub = {
        let mut b = ChunkBuilder::new();
        b.emit(Op::SubshellEnd, 1);
        b.build()
    };
    let mut vm = VM::new(noop.clone());
    vm.set_shell_host(Box::new(CountingSubshellEndHost(91)));
    vm.request_halt();
    let _ = vm.run();
    vm.reset(noop.clone());
    vm.request_halt();
    let _ = vm.run();
    vm.reset(sub);
    let _ = vm.run();
    assert_eq!(SUBSHELL_END_CALLS.load(Ordering::SeqCst), 1);
    assert_eq!(vm.last_status, 91);
}

#[test]
fn subshell_end_exit_code_65_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(65)));
    match vm.run() {
        VMResult::Ok(Value::Status(65)) => {}
        other => panic!("exit code 65 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_load_false_after_load_true() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadTrue, 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::LoadFalse, 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_status_minus_twelve_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(-12)));
    match vm.run() {
        VMResult::Ok(Value::Status(-12)) => {}
        other => panic!("status -12 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_between_subshell_end_and_loadint() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::LoadInt(92), 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(92)));
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 92);
}

#[test]
fn subshell_end_status_minus_thirteen_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(-13)));
    match vm.run() {
        VMResult::Ok(Value::Status(-13)) => {}
        other => panic!("status -13 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_fifteen_consecutive_runs_without_reset_still_blocked() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    for _ in 0..15 {
        let _ = vm.run();
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_thirty_one_end_incrementing_from_twenty_one() {
    let mut b = ChunkBuilder::new();
    for _ in 0..31 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(IncrementingSubshellHost { next: 21 }));
    match vm.run() {
        VMResult::Ok(Value::Status(51)) => {}
        other => panic!("thirty-first subshell_end(Some(51)) MUST win, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_num_lt_keep_when_condition_true() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(2), 1);
    b.emit(Op::LoadInt(5), 1);
    b.emit(Op::NumLt, 1);
    let jmp = b.emit(Op::JumpIfFalseKeep(0), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.patch_jump(jmp, b.current_pos());
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_thirty_two_end_incrementing_from_fourteen() {
    let mut b = ChunkBuilder::new();
    for _ in 0..32 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(IncrementingSubshellHost { next: 14 }));
    match vm.run() {
        VMResult::Ok(Value::Status(45)) => {}
        other => panic!("thirty-second subshell_end(Some(45)) MUST win, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_num_gt_keep_when_condition_false() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(2), 1);
    b.emit(Op::LoadInt(5), 1);
    b.emit(Op::NumGt, 1);
    let jmp = b.emit(Op::JumpIfTrueKeep(0), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.patch_jump(jmp, b.current_pos());
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_ninth_some_on_ninth_end_applies_ninety_three() {
    let mut b = ChunkBuilder::new();
    for _ in 0..9 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    struct NinthSomeHost {
        calls: u32,
    }
    impl ShellHost for NinthSomeHost {
        fn subshell_end(&mut self) -> Option<i32> {
            self.calls += 1;
            if self.calls == 9 {
                Some(93)
            } else {
                None
            }
        }
    }
    let mut vm = VM::new(b.build());
    vm.last_status = 4;
    vm.set_shell_host(Box::new(NinthSomeHost { calls: 0 }));
    match vm.run() {
        VMResult::Ok(Value::Status(93)) => {}
        other => panic!("ninth subshell_end(Some(93)) MUST apply, got {:?}", other),
    }
}

#[test]
fn request_halt_subshell_end_then_setstatus_blocked() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::LoadInt(94), 1);
    b.emit(Op::SetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(94)));
    vm.register_builtin(100, builtin_request_halt);
    let _ = vm.run();
    assert_eq!(vm.last_status, 94);
}

#[test]
fn subshell_end_exit_code_66_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(66)));
    match vm.run() {
        VMResult::Ok(Value::Status(66)) => {}
        other => panic!("exit code 66 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_three_subshell_begins_halt_before_any_end() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    SUBSHELL_END_CALLS.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::SubshellEnd, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(CountingSubshellEndHost(95)));
    vm.register_builtin(100, builtin_request_halt);
    let _ = vm.run();
    assert_eq!(SUBSHELL_END_CALLS.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_67_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(67)));
    match vm.run() {
        VMResult::Ok(Value::Status(67)) => {}
        other => panic!("exit code 67 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_wide_then_narrow_extension_first_halts() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    static WIDE: AtomicU32 = AtomicU32::new(0);
    static NARROW: AtomicU32 = AtomicU32::new(0);
    let mut b = ChunkBuilder::new();
    b.emit(Op::ExtendedWide(1, 0), 1);
    b.emit(Op::Extended(2, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_extension_wide_handler(Box::new(|vm, _id, _payload| {
        WIDE.fetch_add(1, Ordering::SeqCst);
        vm.request_halt();
    }));
    vm.set_extension_handler(Box::new(|_vm, _id, _arg| {
        NARROW.fetch_add(1, Ordering::SeqCst);
    }));
    let _ = vm.run();
    assert_eq!(WIDE.load(Ordering::SeqCst), 1);
    assert_eq!(NARROW.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_68_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(68)));
    match vm.run() {
        VMResult::Ok(Value::Status(68)) => {}
        other => panic!("exit code 68 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_after_subshell_end_before_subshell_begin() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    SUBSHELL_BEGIN_CALLS.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::SubshellBegin, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(BeginEndCountingHost(96)));
    vm.register_builtin(100, builtin_request_halt);
    let _ = vm.run();
    assert_eq!(SUBSHELL_BEGIN_CALLS.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 96);
}

#[test]
fn subshell_end_exit_code_69_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(69)));
    match vm.run() {
        VMResult::Ok(Value::Status(69)) => {}
        other => panic!("exit code 69 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_reset_clears_halt_four_subshell_ends_on_two_resets() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    SUBSHELL_END_CALLS.store(0, Ordering::SeqCst);
    let halt = {
        let mut b = ChunkBuilder::new();
        b.emit(Op::CallBuiltin(100, 0), 1);
        b.build()
    };
    let two_ends = {
        let mut b = ChunkBuilder::new();
        b.emit(Op::SubshellEnd, 1);
        b.emit(Op::SubshellEnd, 1);
        b.build()
    };
    let mut vm = VM::new(halt);
    vm.set_shell_host(Box::new(CountingSubshellEndHost(97)));
    vm.register_builtin(100, builtin_request_halt);
    let _ = vm.run();
    vm.reset(two_ends.clone());
    let _ = vm.run();
    vm.reset(two_ends);
    let _ = vm.run();
    assert_eq!(SUBSHELL_END_CALLS.load(Ordering::SeqCst), 4);
    assert_eq!(vm.last_status, 97);
}

#[test]
fn subshell_end_exit_code_70_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(70)));
    match vm.run() {
        VMResult::Ok(Value::Status(70)) => {}
        other => panic!("exit code 70 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_pop_after_dup_on_stack() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(8), 1);
    b.emit(Op::Dup, 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::Pop, 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
}

// ─── Pin tests batch F (handwritten) ──────────────────────────────────────

#[test]
fn subshell_end_exit_code_161_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(161)));
    match vm.run() {
        VMResult::Ok(Value::Status(161)) => {}
        other => panic!("exit code 161 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_sixteen_consecutive_runs_without_reset_still_blocked() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    for _ in 0..16 {
        let _ = vm.run();
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_162_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(162)));
    match vm.run() {
        VMResult::Ok(Value::Status(162)) => {}
        other => panic!("exit code 162 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_num_le_keep_when_condition_false() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(5), 1);
    b.emit(Op::LoadInt(3), 1);
    b.emit(Op::NumLe, 1);
    let jmp = b.emit(Op::JumpIfTrueKeep(0), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.patch_jump(jmp, b.current_pos());
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_163_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(163)));
    match vm.run() {
        VMResult::Ok(Value::Status(163)) => {}
        other => panic!("exit code 163 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_num_ge_keep_when_condition_false() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(2), 1);
    b.emit(Op::LoadInt(5), 1);
    b.emit(Op::NumGe, 1);
    let jmp = b.emit(Op::JumpIfTrueKeep(0), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.patch_jump(jmp, b.current_pos());
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_164_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(164)));
    match vm.run() {
        VMResult::Ok(Value::Status(164)) => {}
        other => panic!("exit code 164 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_between_two_subshell_ends_with_loadint_between() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    SUBSHELL_END_CALLS.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::LoadInt(98), 1);
    b.emit(Op::SubshellEnd, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(CountingSubshellEndHost(98)));
    vm.register_builtin(100, builtin_request_halt);
    let _ = vm.run();
    assert_eq!(SUBSHELL_END_CALLS.load(Ordering::SeqCst), 1);
    assert_eq!(vm.last_status, 98);
}

#[test]
fn subshell_end_exit_code_165_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(165)));
    match vm.run() {
        VMResult::Ok(Value::Status(165)) => {}
        other => panic!("exit code 165 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_seventeen_consecutive_runs_without_reset_still_blocked() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    for _ in 0..17 {
        let _ = vm.run();
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_status_minus_fourteen_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(-14)));
    match vm.run() {
        VMResult::Ok(Value::Status(-14)) => {}
        other => panic!("status -14 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_swap_after_subshell_end() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::LoadInt(2), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::Swap, 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(99)));
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_status_minus_fifteen_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(-15)));
    match vm.run() {
        VMResult::Ok(Value::Status(-15)) => {}
        other => panic!("status -15 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_after_first_getstatus_before_subshell_end() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(100), 1);
    b.emit(Op::SetStatus, 1);
    b.emit(Op::GetStatus, 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::SubshellEnd, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(101)));
    vm.register_builtin(100, builtin_request_halt);
    match vm.run() {
        VMResult::Ok(Value::Int(0)) => {}
        other => panic!("SubshellEnd MUST NOT run after halt, got {:?}", other),
    }
    assert_eq!(vm.last_status, 100);
}

#[test]
fn subshell_thirty_three_end_incrementing_from_sixteen() {
    let mut b = ChunkBuilder::new();
    for _ in 0..33 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(IncrementingSubshellHost { next: 16 }));
    match vm.run() {
        VMResult::Ok(Value::Status(48)) => {}
        other => panic!("thirty-third subshell_end(Some(48)) MUST win, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_loadint_after_full_subshell_begin_end() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::LoadInt(102), 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(102)));
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 102);
}

#[test]
fn subshell_thirty_four_end_incrementing_from_nineteen() {
    let mut b = ChunkBuilder::new();
    for _ in 0..34 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(IncrementingSubshellHost { next: 19 }));
    match vm.run() {
        VMResult::Ok(Value::Status(52)) => {}
        other => panic!("thirty-fourth subshell_end(Some(52)) MUST win, got {:?}", other),
    }
}

#[test]
fn request_halt_nested_callee_halt_caller_never_reaches_loadint() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    let mut b = ChunkBuilder::new();
    let name = b.add_name("inner_h2");
    b.emit(Op::Call(name, 0), 1);
    b.emit(Op::LoadInt(777), 1);
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
        other => panic!("caller LoadInt MUST NOT run after callee halt, got {:?}", other),
    }
}

#[test]
fn subshell_thirty_five_end_incrementing_from_twenty_three() {
    let mut b = ChunkBuilder::new();
    for _ in 0..35 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(IncrementingSubshellHost { next: 23 }));
    match vm.run() {
        VMResult::Ok(Value::Status(57)) => {}
        other => panic!("thirty-fifth subshell_end(Some(57)) MUST win, got {:?}", other),
    }
}

#[test]
fn request_halt_extension_handler_halts_before_subshell_begin() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    SUBSHELL_BEGIN_CALLS.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::Extended(20, 0), 1);
    b.emit(Op::SubshellBegin, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(BeginEndCountingHost(103)));
    vm.set_extension_handler(Box::new(|vm, _id, _arg| {
        vm.request_halt();
    }));
    let _ = vm.run();
    assert_eq!(SUBSHELL_BEGIN_CALLS.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_166_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(166)));
    match vm.run() {
        VMResult::Ok(Value::Status(166)) => {}
        other => panic!("exit code 166 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_after_subshell_end_before_pipeline_stage_in_begin() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    PIPELINE_STAGE_CALLS.store(0, Ordering::SeqCst);
    struct StageAfterSubshellHost;
    impl ShellHost for StageAfterSubshellHost {
        fn pipeline_stage(&mut self) {
            PIPELINE_STAGE_CALLS.fetch_add(1, Ordering::SeqCst);
        }
        fn subshell_end(&mut self) -> Option<i32> {
            Some(104)
        }
    }
    let mut b = ChunkBuilder::new();
    b.emit(Op::PipelineBegin(2), 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::PipelineStage, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StageAfterSubshellHost));
    vm.register_builtin(100, builtin_request_halt);
    let _ = vm.run();
    assert_eq!(PIPELINE_STAGE_CALLS.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 104);
}

#[test]
fn subshell_end_exit_code_167_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(167)));
    match vm.run() {
        VMResult::Ok(Value::Status(167)) => {}
        other => panic!("exit code 167 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_halt_only_on_empty_stack_returns_halted() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(61, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(61, builtin_halt_only);
    match vm.run() {
        VMResult::Ok(Value::Undef) => {}
        other => panic!(
            "halt_only pushes Undef before run() pops stack top, got {:?}",
            other
        ),
    }
}

#[test]
fn subshell_end_exit_code_168_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(168)));
    match vm.run() {
        VMResult::Ok(Value::Status(168)) => {}
        other => panic!("exit code 168 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_double_subshell_end_then_halt_third_blocked() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    SUBSHELL_END_CALLS.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::SubshellEnd, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(CountingSubshellEndHost(105)));
    vm.register_builtin(100, builtin_request_halt);
    let _ = vm.run();
    assert_eq!(SUBSHELL_END_CALLS.load(Ordering::SeqCst), 2);
    assert_eq!(vm.last_status, 105);
}

#[test]
fn subshell_end_exit_code_169_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(169)));
    match vm.run() {
        VMResult::Ok(Value::Status(169)) => {}
        other => panic!("exit code 169 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_reset_after_subshell_halt_allows_getstatus_zero() {
    let sub = {
        let mut b = ChunkBuilder::new();
        b.emit(Op::SubshellEnd, 1);
        b.emit(Op::CallBuiltin(100, 0), 1);
        b.build()
    };
    let status = {
        let mut b = ChunkBuilder::new();
        b.emit(Op::GetStatus, 1);
        b.build()
    };
    let mut vm = VM::new(sub);
    vm.set_shell_host(Box::new(StatusReturningHost(106)));
    vm.register_builtin(100, builtin_request_halt);
    let _ = vm.run();
    assert_eq!(vm.last_status, 106);
    vm.reset(status);
    match vm.run() {
        VMResult::Ok(Value::Status(0)) => {}
        other => panic!("reset MUST clear last_status, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_170_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(170)));
    match vm.run() {
        VMResult::Ok(Value::Status(170)) => {}
        other => panic!("exit code 170 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_eighteen_consecutive_runs_without_reset_still_blocked() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    for _ in 0..18 {
        let _ = vm.run();
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_tenth_some_on_tenth_end_applies_one_seven() {
    let mut b = ChunkBuilder::new();
    for _ in 0..10 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    struct TenthSomeHost {
        calls: u32,
    }
    impl ShellHost for TenthSomeHost {
        fn subshell_end(&mut self) -> Option<i32> {
            self.calls += 1;
            if self.calls == 10 {
                Some(107)
            } else {
                None
            }
        }
    }
    let mut vm = VM::new(b.build());
    vm.last_status = 2;
    vm.set_shell_host(Box::new(TenthSomeHost { calls: 0 }));
    match vm.run() {
        VMResult::Ok(Value::Status(107)) => {}
        other => panic!("tenth subshell_end(Some(107)) MUST apply, got {:?}", other),
    }
}

#[test]
fn request_halt_between_pipeline_begin_and_subshell_end() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    SUBSHELL_END_CALLS.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::PipelineBegin(2), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::SubshellEnd, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(CountingSubshellEndHost(108)));
    vm.register_builtin(100, builtin_request_halt);
    let _ = vm.run();
    assert_eq!(SUBSHELL_END_CALLS.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_setstatus_after_end_before_getstatus_reads_setstatus() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::LoadInt(109), 1);
    b.emit(Op::SetStatus, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(50)));
    match vm.run() {
        VMResult::Ok(Value::Status(109)) => {}
        other => panic!("SetStatus(109) MUST beat subshell Some(50), got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_negate_after_subshell_end() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::LoadInt(4), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::Negate, 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(110)));
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_two_pipeline_ends_subshell_end_getstatus_reads_subshell() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::PipelineEnd, 1);
    b.emit(Op::PipelineEnd, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(PipelineThenSubshellHost));
    match vm.run() {
        VMResult::Ok(Value::Status(9)) => {}
        other => panic!("GetStatus MUST read subshell(9), got {:?}", other),
    }
}

#[test]
fn request_halt_callee_with_one_arg_before_halt_in_caller() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    let mut b = ChunkBuilder::new();
    let name = b.add_name("take1");
    b.emit(Op::LoadInt(7), 1);
    b.emit(Op::Call(name, 1), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    let skip = b.emit(Op::Jump(0), 1);
    let entry = b.current_pos();
    b.add_sub_entry(name, entry);
    b.emit(Op::ReturnValue, 1);
    b.patch_jump(skip, b.current_pos());
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    match vm.run() {
        VMResult::Ok(Value::Int(0)) => {}
        other => panic!("halt MUST be stack top after callee, got {:?}", other),
    }
}

#[test]
fn subshell_begin_end_reset_begin_end_same_host_twice() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    SUBSHELL_BEGIN_CALLS.store(0, Ordering::SeqCst);
    let chunk = {
        let mut b = ChunkBuilder::new();
        b.emit(Op::SubshellBegin, 1);
        b.emit(Op::SubshellEnd, 1);
        b.build()
    };
    let mut vm = VM::new(chunk.clone());
    vm.set_shell_host(Box::new(BeginEndCountingHost(111)));
    let _ = vm.run();
    vm.reset(chunk.clone());
    let _ = vm.run();
    assert_eq!(SUBSHELL_BEGIN_CALLS.load(Ordering::SeqCst), 2);
    assert_eq!(vm.last_status, 111);
}

#[test]
fn request_halt_stops_before_add_after_subshell_wrote_status() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::LoadInt(2), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::Add, 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(112)));
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_loadint_subshell_end_getstatus_order_final_status() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(55), 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(113)));
    match vm.run() {
        VMResult::Ok(Value::Status(113)) => {}
        other => panic!("GetStatus(113) MUST be stack top, got {:?}", other),
    }
}

#[test]
fn request_halt_nineteen_consecutive_runs_without_reset_still_blocked() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    for _ in 0..19 {
        let _ = vm.run();
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_alternating_host_fifth_end_none_keeps_nine() {
    let mut b = ChunkBuilder::new();
    for _ in 0..5 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(AlternatingStatusHost { calls: 0 }));
    match vm.run() {
        VMResult::Ok(Value::Status(9)) => {}
        other => panic!("AlternatingStatusHost fifth None MUST keep Some(9), got {:?}", other),
    }
}

#[test]
fn request_halt_twenty_consecutive_runs_without_reset_still_blocked() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    for _ in 0..20 {
        let _ = vm.run();
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_none_then_some_on_orphan_ends_only() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.last_status = 6;
    vm.set_shell_host(Box::new(NoneThenSomeHost { calls: 0 }));
    match vm.run() {
        VMResult::Ok(Value::Status(42)) => {}
        other => panic!("NoneThenSomeHost MUST apply Some(42), got {:?}", other),
    }
}

#[test]
fn request_halt_full_reset_lifecycle_halt_subshell_halt_getstatus() {
    let sub = {
        let mut b = ChunkBuilder::new();
        b.emit(Op::SubshellEnd, 1);
        b.emit(Op::CallBuiltin(100, 0), 1);
        b.build()
    };
    let status = {
        let mut b = ChunkBuilder::new();
        b.emit(Op::GetStatus, 1);
        b.build()
    };
    let mut vm = VM::new(sub.clone());
    vm.set_shell_host(Box::new(StatusReturningHost(114)));
    vm.register_builtin(100, builtin_request_halt);
    let _ = vm.run();
    assert_eq!(vm.last_status, 114);
    vm.reset(sub);
    let _ = vm.run();
    vm.reset(status);
    match vm.run() {
        VMResult::Ok(Value::Status(0)) => {}
        other => panic!("final GetStatus MUST read 0 after reset, got {:?}", other),
    }
}

#[test]
fn subshell_custom_pipeline_positive_after_subshell_on_stack() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::PipelineEnd, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(CustomPipelineHost(115)));
    match vm.run() {
        VMResult::Ok(Value::Status(115)) => {}
        other => panic!("PipelineEnd MUST push host 115, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_second_getstatus_after_subshell_end() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(116)));
    vm.register_builtin(100, builtin_request_halt);
    match vm.run() {
        VMResult::Ok(Value::Int(0)) => {}
        other => panic!("second GetStatus MUST NOT run, got {:?}", other),
    }
    assert_eq!(vm.last_status, 116);
}

#[test]
fn subshell_end_exit_code_171_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(171)));
    match vm.run() {
        VMResult::Ok(Value::Status(171)) => {}
        other => panic!("exit code 171 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_preserves_shell_host_after_reset_on_halt_chunk() {
    let halt = {
        let mut b = ChunkBuilder::new();
        b.emit(Op::CallBuiltin(100, 0), 1);
        b.build()
    };
    let sub = {
        let mut b = ChunkBuilder::new();
        b.emit(Op::SubshellEnd, 1);
        b.emit(Op::GetStatus, 1);
        b.build()
    };
    let mut vm = VM::new(halt);
    vm.set_shell_host(Box::new(StatusReturningHost(117)));
    vm.register_builtin(100, builtin_request_halt);
    let _ = vm.run();
    vm.reset(sub);
    match vm.run() {
        VMResult::Ok(Value::Status(117)) => {}
        other => panic!("preserved host MUST apply Some(117), got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_172_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(172)));
    match vm.run() {
        VMResult::Ok(Value::Status(172)) => {}
        other => panic!("exit code 172 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_pop_frame_after_three_push_frames() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::PushFrame, 1);
    b.emit(Op::PushFrame, 1);
    b.emit(Op::PushFrame, 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::PopFrame, 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_triple_begin_single_end_applies_status() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    SUBSHELL_BEGIN_CALLS.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(BeginEndCountingHost(118)));
    match vm.run() {
        VMResult::Ok(Value::Status(118)) => {}
        other => panic!("SubshellEnd MUST apply Some(118), got {:?}", other),
    }
    assert_eq!(SUBSHELL_BEGIN_CALLS.load(Ordering::SeqCst), 3);
}

#[test]
fn request_halt_after_subshell_end_extension_handler_not_reached() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    static EXT_RAN: AtomicU32 = AtomicU32::new(0);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::Extended(30, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(119)));
    vm.set_extension_handler(Box::new(|_vm, _id, _arg| {
        EXT_RAN.fetch_add(1, Ordering::SeqCst);
    }));
    vm.register_builtin(100, builtin_request_halt);
    let _ = vm.run();
    assert_eq!(EXT_RAN.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 119);
}

#[test]
fn subshell_end_exit_code_173_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(173)));
    match vm.run() {
        VMResult::Ok(Value::Status(173)) => {}
        other => panic!("exit code 173 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_subshell_and_halt_same_chunk_last_status_from_subshell() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(120)));
    vm.register_builtin(100, builtin_request_halt);
    let _ = vm.run();
    assert_eq!(vm.last_status, 120);
}
// ─── Pin tests batch G (handwritten) ──────────────────────────────────────

#[test]
fn subshell_end_exit_code_174_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(174)));
    match vm.run() {
        VMResult::Ok(Value::Status(174)) => {}
        other => panic!("exit code 174 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_str_lt_jump_if_false_keep() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    let a = b.add_constant(Value::str("a"));
    let bidx = b.add_constant(Value::str("z"));
    b.emit(Op::LoadConst(a), 1);
    b.emit(Op::LoadConst(bidx), 1);
    b.emit(Op::StrLt, 1);
    let jmp = b.emit(Op::JumpIfFalseKeep(0), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.patch_jump(jmp, b.current_pos());
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_175_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(175)));
    match vm.run() {
        VMResult::Ok(Value::Status(175)) => {}
        other => panic!("exit code 175 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_str_gt_jump_if_true_keep_when_false() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    let a = b.add_constant(Value::str("a"));
    let bidx = b.add_constant(Value::str("z"));
    b.emit(Op::LoadConst(a), 1);
    b.emit(Op::LoadConst(bidx), 1);
    b.emit(Op::StrGt, 1);
    let jmp = b.emit(Op::JumpIfTrueKeep(0), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.patch_jump(jmp, b.current_pos());
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_176_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(176)));
    match vm.run() {
        VMResult::Ok(Value::Status(176)) => {}
        other => panic!("exit code 176 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_str_le_jump_if_false_keep() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    let a = b.add_constant(Value::str("mid"));
    let bidx = b.add_constant(Value::str("mid"));
    b.emit(Op::LoadConst(a), 1);
    b.emit(Op::LoadConst(bidx), 1);
    b.emit(Op::StrLe, 1);
    let jmp = b.emit(Op::JumpIfFalseKeep(0), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.patch_jump(jmp, b.current_pos());
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_177_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(177)));
    match vm.run() {
        VMResult::Ok(Value::Status(177)) => {}
        other => panic!("exit code 177 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_str_ge_jump_if_false_keep() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    let a = b.add_constant(Value::str("z"));
    let bidx = b.add_constant(Value::str("a"));
    b.emit(Op::LoadConst(a), 1);
    b.emit(Op::LoadConst(bidx), 1);
    b.emit(Op::StrGe, 1);
    let jmp = b.emit(Op::JumpIfFalseKeep(0), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.patch_jump(jmp, b.current_pos());
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_178_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(178)));
    match vm.run() {
        VMResult::Ok(Value::Status(178)) => {}
        other => panic!("exit code 178 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_num_ne_jump_if_false_keep_when_true() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(4), 1);
    b.emit(Op::LoadInt(9), 1);
    b.emit(Op::NumNe, 1);
    let jmp = b.emit(Op::JumpIfFalseKeep(0), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.patch_jump(jmp, b.current_pos());
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_179_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(179)));
    match vm.run() {
        VMResult::Ok(Value::Status(179)) => {}
        other => panic!("exit code 179 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_twenty_one_consecutive_runs_without_reset_still_blocked() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    for _ in 0..21 {
        let _ = vm.run();
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_180_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(180)));
    match vm.run() {
        VMResult::Ok(Value::Status(180)) => {}
        other => panic!("exit code 180 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_twenty_two_consecutive_runs_without_reset_still_blocked() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    for _ in 0..22 {
        let _ = vm.run();
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_181_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(181)));
    match vm.run() {
        VMResult::Ok(Value::Status(181)) => {}
        other => panic!("exit code 181 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_twenty_three_consecutive_runs_without_reset_still_blocked() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    for _ in 0..23 {
        let _ = vm.run();
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_182_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(182)));
    match vm.run() {
        VMResult::Ok(Value::Status(182)) => {}
        other => panic!("exit code 182 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_twenty_four_consecutive_runs_without_reset_still_blocked() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    for _ in 0..24 {
        let _ = vm.run();
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_183_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(183)));
    match vm.run() {
        VMResult::Ok(Value::Status(183)) => {}
        other => panic!("exit code 183 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_twenty_five_consecutive_runs_without_reset_still_blocked() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    for _ in 0..25 {
        let _ = vm.run();
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_184_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(184)));
    match vm.run() {
        VMResult::Ok(Value::Status(184)) => {}
        other => panic!("exit code 184 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_thirty_six_end_incrementing_from_four() {
    let mut b = ChunkBuilder::new();
    for _ in 0..36 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(IncrementingSubshellHost { next: 4 }));
    match vm.run() {
        VMResult::Ok(Value::Status(39)) => {}
        other => panic!("thirty-sixth subshell_end(Some(39)) MUST win, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_185_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(185)));
    match vm.run() {
        VMResult::Ok(Value::Status(185)) => {}
        other => panic!("exit code 185 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_thirty_seven_end_incrementing_from_nine() {
    let mut b = ChunkBuilder::new();
    for _ in 0..37 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(IncrementingSubshellHost { next: 9 }));
    match vm.run() {
        VMResult::Ok(Value::Status(45)) => {}
        other => panic!("thirty-seventh subshell_end(Some(45)) MUST win, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_186_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(186)));
    match vm.run() {
        VMResult::Ok(Value::Status(186)) => {}
        other => panic!("exit code 186 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_thirty_eight_end_incrementing_from_twelve() {
    let mut b = ChunkBuilder::new();
    for _ in 0..38 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(IncrementingSubshellHost { next: 12 }));
    match vm.run() {
        VMResult::Ok(Value::Status(49)) => {}
        other => panic!("thirty-eighth subshell_end(Some(49)) MUST win, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_187_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(187)));
    match vm.run() {
        VMResult::Ok(Value::Status(187)) => {}
        other => panic!("exit code 187 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_thirty_nine_end_incrementing_from_fifteen() {
    let mut b = ChunkBuilder::new();
    for _ in 0..39 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(IncrementingSubshellHost { next: 15 }));
    match vm.run() {
        VMResult::Ok(Value::Status(53)) => {}
        other => panic!("thirty-ninth subshell_end(Some(53)) MUST win, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_188_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(188)));
    match vm.run() {
        VMResult::Ok(Value::Status(188)) => {}
        other => panic!("exit code 188 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_forty_end_incrementing_from_one() {
    let mut b = ChunkBuilder::new();
    for _ in 0..40 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(IncrementingSubshellHost { next: 1 }));
    match vm.run() {
        VMResult::Ok(Value::Status(40)) => {}
        other => panic!("fortieth subshell_end(Some(40)) MUST win, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_189_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(189)));
    match vm.run() {
        VMResult::Ok(Value::Status(189)) => {}
        other => panic!("exit code 189 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_status_minus_sixteen_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(-16)));
    match vm.run() {
        VMResult::Ok(Value::Status(-16)) => {}
        other => panic!("status -16 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_190_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(190)));
    match vm.run() {
        VMResult::Ok(Value::Status(190)) => {}
        other => panic!("exit code 190 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_status_minus_seventeen_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(-17)));
    match vm.run() {
        VMResult::Ok(Value::Status(-17)) => {}
        other => panic!("status -17 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_191_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(191)));
    match vm.run() {
        VMResult::Ok(Value::Status(191)) => {}
        other => panic!("exit code 191 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_status_minus_eighteen_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(-18)));
    match vm.run() {
        VMResult::Ok(Value::Status(-18)) => {}
        other => panic!("status -18 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_192_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(192)));
    match vm.run() {
        VMResult::Ok(Value::Status(192)) => {}
        other => panic!("exit code 192 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_status_minus_nineteen_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(-19)));
    match vm.run() {
        VMResult::Ok(Value::Status(-19)) => {}
        other => panic!("status -19 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_193_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(193)));
    match vm.run() {
        VMResult::Ok(Value::Status(193)) => {}
        other => panic!("exit code 193 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_status_minus_twenty_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(-20)));
    match vm.run() {
        VMResult::Ok(Value::Status(-20)) => {}
        other => panic!("status -20 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_194_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(194)));
    match vm.run() {
        VMResult::Ok(Value::Status(194)) => {}
        other => panic!("exit code 194 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_eleventh_some_host_applies_on_eleventh_end() {
    let mut b = ChunkBuilder::new();
    for _ in 0..11 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    struct EleventhSomeHost {
        calls: u32,
    }
    impl ShellHost for EleventhSomeHost {
        fn subshell_end(&mut self) -> Option<i32> {
            self.calls += 1;
            if self.calls == 11 {
                Some(121)
            } else {
                None
            }
        }
    }
    let mut vm = VM::new(b.build());
    vm.last_status = 3;
    vm.set_shell_host(Box::new(EleventhSomeHost { calls: 0 }));
    match vm.run() {
        VMResult::Ok(Value::Status(121)) => {}
        other => panic!("eleventh subshell_end(Some(121)) MUST apply, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_195_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(195)));
    match vm.run() {
        VMResult::Ok(Value::Status(195)) => {}
        other => panic!("exit code 195 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_load_float_after_subshell_wrote_status() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::LoadFloat(2.71), 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(196)));
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 196);
}

#[test]
fn subshell_end_exit_code_196_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(196)));
    match vm.run() {
        VMResult::Ok(Value::Status(196)) => {}
        other => panic!("exit code 196 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_after_subshell_end_before_str_eq_on_consts() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    let a = b.add_constant(Value::str("x"));
    let bidx = b.add_constant(Value::str("x"));
    b.emit(Op::LoadConst(a), 1);
    b.emit(Op::LoadConst(bidx), 1);
    b.emit(Op::StrEq, 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(197)));
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 197);
}

#[test]
fn subshell_end_exit_code_197_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(197)));
    match vm.run() {
        VMResult::Ok(Value::Status(197)) => {}
        other => panic!("exit code 197 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_halt_reset_halt_reset_subshell_end_fourth_run_applies() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    SUBSHELL_END_CALLS.store(0, Ordering::SeqCst);
    let sub = {
        let mut b = ChunkBuilder::new();
        b.emit(Op::SubshellEnd, 1);
        b.emit(Op::CallBuiltin(100, 0), 1);
        b.build()
    };
    let end_only = {
        let mut b = ChunkBuilder::new();
        b.emit(Op::SubshellEnd, 1);
        b.build()
    };
    let mut vm = VM::new(sub.clone());
    vm.set_shell_host(Box::new(CountingSubshellEndHost(198)));
    vm.register_builtin(100, builtin_request_halt);
    let _ = vm.run();
    vm.reset(sub);
    let _ = vm.run();
    vm.reset(end_only.clone());
    let _ = vm.run();
    vm.reset(end_only);
    let _ = vm.run();
    assert_eq!(SUBSHELL_END_CALLS.load(Ordering::SeqCst), 4);
    assert_eq!(vm.last_status, 198);
}

#[test]
fn subshell_end_exit_code_198_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(198)));
    match vm.run() {
        VMResult::Ok(Value::Status(198)) => {}
        other => panic!("exit code 198 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_between_subshell_end_and_second_subshell_begin() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    SUBSHELL_BEGIN_CALLS.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(BeginEndCountingHost(199)));
    vm.register_builtin(100, builtin_request_halt);
    let _ = vm.run();
    assert_eq!(SUBSHELL_BEGIN_CALLS.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 199);
}
// ─── Pin tests batch H (handwritten) ──────────────────────────────────────

#[test]
fn subshell_end_exit_code_199_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(199)));
    match vm.run() {
        VMResult::Ok(Value::Status(199)) => {}
        other => panic!("exit code 199 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_backward_jump_if_false_keep() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::NumEq, 1);
    let back = b.emit(Op::JumpIfFalseKeep(0), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.patch_jump(back, b.current_pos());
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_200_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(200)));
    match vm.run() {
        VMResult::Ok(Value::Status(200)) => {}
        other => panic!("exit code 200 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_sixth_some_host_applies_on_sixth_end() {
    let mut b = ChunkBuilder::new();
    for _ in 0..6 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    struct SixthSomeHost {
        calls: u32,
    }
    impl ShellHost for SixthSomeHost {
        fn subshell_end(&mut self) -> Option<i32> {
            self.calls += 1;
            if self.calls == 6 {
                Some(201)
            } else {
                None
            }
        }
    }
    let mut vm = VM::new(b.build());
    vm.last_status = 8;
    vm.set_shell_host(Box::new(SixthSomeHost { calls: 0 }));
    match vm.run() {
        VMResult::Ok(Value::Status(201)) => {}
        other => panic!("sixth subshell_end(Some(201)) MUST apply, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_201_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(201)));
    match vm.run() {
        VMResult::Ok(Value::Status(201)) => {}
        other => panic!("exit code 201 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_after_pipeline_begin_before_subshell_end_host() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    SUBSHELL_END_CALLS.store(0, Ordering::SeqCst);
    struct PipelineBeginSubshellEndHost202;
    impl ShellHost for PipelineBeginSubshellEndHost202 {
        fn pipeline_begin(&mut self, _n: u8) {
            PIPELINE_BEGIN_CALLS.fetch_add(1, Ordering::SeqCst);
        }
        fn subshell_end(&mut self) -> Option<i32> {
            SUBSHELL_END_CALLS.fetch_add(1, Ordering::SeqCst);
            Some(202)
        }
    }
    let mut b = ChunkBuilder::new();
    b.emit(Op::PipelineBegin(2), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::SubshellEnd, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(PipelineBeginSubshellEndHost202));
    vm.register_builtin(100, builtin_request_halt);
    let _ = vm.run();
    assert_eq!(SUBSHELL_END_CALLS.load(Ordering::SeqCst), 0);
    assert_eq!(PIPELINE_BEGIN_CALLS.load(Ordering::SeqCst), 1);
}

#[test]
fn subshell_end_exit_code_202_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(202)));
    match vm.run() {
        VMResult::Ok(Value::Status(202)) => {}
        other => panic!("exit code 202 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_setstatus_before_end_some_host_overwrites_setstatus() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(203), 1);
    b.emit(Op::SetStatus, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(203)));
    match vm.run() {
        VMResult::Ok(Value::Status(203)) => {}
        other => panic!("subshell Some(203) MUST beat SetStatus(203), got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_203_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(203)));
    match vm.run() {
        VMResult::Ok(Value::Status(203)) => {}
        other => panic!("exit code 203 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_four_begins_three_ends_last_end_status_wins() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    SUBSHELL_BEGIN_CALLS.store(0, Ordering::SeqCst);
    SUBSHELL_END_CALLS.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    struct CountingBeginEndHost204;
    impl ShellHost for CountingBeginEndHost204 {
        fn subshell_begin(&mut self) {
            SUBSHELL_BEGIN_CALLS.fetch_add(1, Ordering::SeqCst);
        }
        fn subshell_end(&mut self) -> Option<i32> {
            SUBSHELL_END_CALLS.fetch_add(1, Ordering::SeqCst);
            Some(204)
        }
    }
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(CountingBeginEndHost204));
    match vm.run() {
        VMResult::Ok(Value::Status(204)) => {}
        other => panic!("third subshell_end(Some(204)) MUST win, got {:?}", other),
    }
    assert_eq!(SUBSHELL_BEGIN_CALLS.load(Ordering::SeqCst), 4);
    assert_eq!(SUBSHELL_END_CALLS.load(Ordering::SeqCst), 3);
}

#[test]
fn subshell_end_exit_code_204_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(204)));
    match vm.run() {
        VMResult::Ok(Value::Status(204)) => {}
        other => panic!("exit code 204 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_pre_run_then_subshell_chunk_after_reset_runs_host() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    SUBSHELL_END_CALLS.store(0, Ordering::SeqCst);
    let empty = ChunkBuilder::new().build();
    let sub = {
        let mut b = ChunkBuilder::new();
        b.emit(Op::SubshellEnd, 1);
        b.build()
    };
    let mut vm = VM::new(empty);
    vm.set_shell_host(Box::new(CountingSubshellEndHost(205)));
    vm.request_halt();
    let _ = vm.run();
    vm.reset(sub);
    let _ = vm.run();
    assert_eq!(SUBSHELL_END_CALLS.load(Ordering::SeqCst), 1);
    assert_eq!(vm.last_status, 205);
}

#[test]
fn subshell_end_exit_code_205_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(205)));
    match vm.run() {
        VMResult::Ok(Value::Status(205)) => {}
        other => panic!("exit code 205 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_alternating_status_sixth_none_keeps_fifth_nine() {
    let mut b = ChunkBuilder::new();
    for _ in 0..6 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(AlternatingStatusHost { calls: 0 }));
    match vm.run() {
        VMResult::Ok(Value::Status(9)) => {}
        other => panic!("AlternatingStatusHost sixth None MUST keep Some(9), got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_206_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(206)));
    match vm.run() {
        VMResult::Ok(Value::Status(206)) => {}
        other => panic!("exit code 206 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_pop_after_two_push_frames() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::PushFrame, 1);
    b.emit(Op::PushFrame, 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::Pop, 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_207_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(207)));
    match vm.run() {
        VMResult::Ok(Value::Status(207)) => {}
        other => panic!("exit code 207 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_counting_end_host_two_ends_same_status() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    SUBSHELL_END_CALLS.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(CountingSubshellEndHost(208)));
    match vm.run() {
        VMResult::Ok(Value::Status(208)) => {}
        other => panic!("second subshell_end MUST still apply Some(208), got {:?}", other),
    }
    assert_eq!(SUBSHELL_END_CALLS.load(Ordering::SeqCst), 2);
}

#[test]
fn subshell_end_exit_code_208_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(208)));
    match vm.run() {
        VMResult::Ok(Value::Status(208)) => {}
        other => panic!("exit code 208 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_extension_no_handler_after_subshell_end_not_reached() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    static EXT_TOUCHED: AtomicU32 = AtomicU32::new(0);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::Extended(44, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(209)));
    vm.register_builtin(100, builtin_request_halt);
    let _ = vm.run();
    assert_eq!(EXT_TOUCHED.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 209);
}

#[test]
fn subshell_end_exit_code_209_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(209)));
    match vm.run() {
        VMResult::Ok(Value::Status(209)) => {}
        other => panic!("exit code 209 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_setstatus_zero_before_end_some_overwrites_zero() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(0), 1);
    b.emit(Op::SetStatus, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(210)));
    match vm.run() {
        VMResult::Ok(Value::Status(210)) => {}
        other => panic!("subshell Some(210) MUST overwrite SetStatus(0), got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_210_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(210)));
    match vm.run() {
        VMResult::Ok(Value::Status(210)) => {}
        other => panic!("exit code 210 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_callee_two_args_return_before_caller_halt() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    let name = b.add_name("pair");
    b.emit(Op::LoadInt(3), 1);
    b.emit(Op::LoadInt(4), 1);
    b.emit(Op::Call(name, 2), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let skip = b.emit(Op::Jump(0), 1);
    let entry = b.current_pos();
    b.add_sub_entry(name, entry);
    b.emit(Op::ReturnValue, 1);
    b.patch_jump(skip, b.current_pos());
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    match vm.run() {
        VMResult::Ok(Value::Int(0)) => {}
        other => panic!("halt MUST be stack top after callee return, got {:?}", other),
    }
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_211_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(211)));
    match vm.run() {
        VMResult::Ok(Value::Status(211)) => {}
        other => panic!("exit code 211 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_replace_host_between_runs_second_host_wins() {
    let chunk = {
        let mut b = ChunkBuilder::new();
        b.emit(Op::SubshellEnd, 1);
        b.emit(Op::GetStatus, 1);
        b.build()
    };
    let mut vm = VM::new(chunk.clone());
    vm.set_shell_host(Box::new(StatusReturningHost(212)));
    let _ = vm.run();
    vm.reset(chunk.clone());
    vm.set_shell_host(Box::new(StatusReturningHost(213)));
    match vm.run() {
        VMResult::Ok(Value::Status(213)) => {}
        other => panic!("replaced host MUST apply Some(213), got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_212_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(212)));
    match vm.run() {
        VMResult::Ok(Value::Status(212)) => {}
        other => panic!("exit code 212 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_load_int_after_getstatus_read() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(214), 1);
    b.emit(Op::SetStatus, 1);
    b.emit(Op::GetStatus, 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::LoadInt(999), 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    match vm.run() {
        VMResult::Ok(Value::Int(0)) => {}
        other => panic!("halt MUST return before LoadInt(999), got {:?}", other),
    }
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_213_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(213)));
    match vm.run() {
        VMResult::Ok(Value::Status(213)) => {}
        other => panic!("exit code 213 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_pipeline_end_subshell_end_getstatus_reads_subshell() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::PipelineEnd, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(PipelineThenSubshellHost));
    match vm.run() {
        VMResult::Ok(Value::Status(9)) => {}
        other => panic!("GetStatus MUST read subshell Some(9), got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_214_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(214)));
    match vm.run() {
        VMResult::Ok(Value::Status(214)) => {}
        other => panic!("exit code 214 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_between_two_extension_wide_handlers_second_not_reached() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    static WIDE_A: AtomicU32 = AtomicU32::new(0);
    static WIDE_B: AtomicU32 = AtomicU32::new(0);
    WIDE_A.store(0, Ordering::SeqCst);
    WIDE_B.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::ExtendedWide(1, 0), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::ExtendedWide(2, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_extension_wide_handler(Box::new(|_vm, id, _payload| {
        if id == 1 {
            WIDE_A.fetch_add(1, Ordering::SeqCst);
        } else if id == 2 {
            WIDE_B.fetch_add(1, Ordering::SeqCst);
        }
    }));
    vm.register_builtin(100, builtin_request_halt);
    let _ = vm.run();
    assert_eq!(WIDE_A.load(Ordering::SeqCst), 1);
    assert_eq!(WIDE_B.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_215_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(215)));
    match vm.run() {
        VMResult::Ok(Value::Status(215)) => {}
        other => panic!("exit code 215 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_double_getstatus_after_end_same_value() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(216)));
    match vm.run() {
        VMResult::Ok(Value::Status(216)) => {}
        other => panic!("second GetStatus MUST still read Status(216), got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_216_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(216)));
    match vm.run() {
        VMResult::Ok(Value::Status(216)) => {}
        other => panic!("exit code 216 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_num_eq_jump_if_true_when_false() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(6), 1);
    b.emit(Op::LoadInt(7), 1);
    b.emit(Op::NumEq, 1);
    let jmp = b.emit(Op::JumpIfTrueKeep(0), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.patch_jump(jmp, b.current_pos());
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_217_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(217)));
    match vm.run() {
        VMResult::Ok(Value::Status(217)) => {}
        other => panic!("exit code 217 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_orphan_after_none_host_leaves_prior_status() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.last_status = 218;
    vm.set_shell_host(Box::new(NoneHost));
    match vm.run() {
        VMResult::Ok(Value::Status(218)) => {}
        other => panic!("NoneHost orphan end MUST preserve last_status 218, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_218_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(218)));
    match vm.run() {
        VMResult::Ok(Value::Status(218)) => {}
        other => panic!("exit code 218 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_reset_twice_then_subshell_end_runs_host() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    SUBSHELL_END_CALLS.store(0, Ordering::SeqCst);
    let halt = {
        let mut b = ChunkBuilder::new();
        b.emit(Op::CallBuiltin(100, 0), 1);
        b.build()
    };
    let sub = {
        let mut b = ChunkBuilder::new();
        b.emit(Op::SubshellEnd, 1);
        b.build()
    };
    let mut vm = VM::new(halt.clone());
    vm.set_shell_host(Box::new(CountingSubshellEndHost(219)));
    vm.register_builtin(100, builtin_request_halt);
    let _ = vm.run();
    vm.reset(halt.clone());
    let _ = vm.run();
    vm.reset(sub);
    let _ = vm.run();
    assert_eq!(SUBSHELL_END_CALLS.load(Ordering::SeqCst), 1);
    assert_eq!(vm.last_status, 219);
}

#[test]
fn subshell_end_exit_code_219_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(219)));
    match vm.run() {
        VMResult::Ok(Value::Status(219)) => {}
        other => panic!("exit code 219 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_begin_counting_host_without_matching_end_leaves_status() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    SUBSHELL_BEGIN_CALLS.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.last_status = 220;
    vm.set_shell_host(Box::new(BeginEndCountingHost(220)));
    match vm.run() {
        VMResult::Ok(Value::Status(220)) => {}
        other => panic!("SubshellBegin alone MUST NOT change last_status, got {:?}", other),
    }
    assert_eq!(SUBSHELL_BEGIN_CALLS.load(Ordering::SeqCst), 2);
}

#[test]
fn subshell_end_exit_code_220_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(220)));
    match vm.run() {
        VMResult::Ok(Value::Status(220)) => {}
        other => panic!("exit code 220 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_after_first_of_three_callbuiltins_blocks_rest() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(7, 0), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(7, builtin_touched);
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 1);
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_221_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(221)));
    match vm.run() {
        VMResult::Ok(Value::Status(221)) => {}
        other => panic!("exit code 221 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_halt_same_vm_second_subshell_end_blocked() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    SUBSHELL_END_CALLS.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::SubshellEnd, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(CountingSubshellEndHost(222)));
    vm.register_builtin(100, builtin_request_halt);
    let _ = vm.run();
    assert_eq!(SUBSHELL_END_CALLS.load(Ordering::SeqCst), 1);
    assert_eq!(vm.last_status, 222);
}

#[test]
fn subshell_end_exit_code_222_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(222)));
    match vm.run() {
        VMResult::Ok(Value::Status(222)) => {}
        other => panic!("exit code 222 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_twenty_six_consecutive_runs_without_reset_still_blocked() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    for _ in 0..26 {
        let _ = vm.run();
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_223_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(223)));
    match vm.run() {
        VMResult::Ok(Value::Status(223)) => {}
        other => panic!("exit code 223 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_forty_one_end_incrementing_from_two() {
    let mut b = ChunkBuilder::new();
    for _ in 0..41 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(IncrementingSubshellHost { next: 2 }));
    match vm.run() {
        VMResult::Ok(Value::Status(42)) => {}
        other => panic!("forty-first subshell_end(Some(42)) MUST win, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_str_ne_jump_if_true_keep_when_false() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    let a = b.add_constant(Value::str("same"));
    let bidx = b.add_constant(Value::str("same"));
    b.emit(Op::LoadConst(a), 1);
    b.emit(Op::LoadConst(bidx), 1);
    b.emit(Op::StrNe, 1);
    let jmp = b.emit(Op::JumpIfTrueKeep(0), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.patch_jump(jmp, b.current_pos());
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_status_minus_twenty_one_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(-21)));
    match vm.run() {
        VMResult::Ok(Value::Status(-21)) => {}
        other => panic!("status -21 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_make_array_after_halt_point() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::LoadInt(2), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::MakeArray(2), 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_twelfth_some_host_applies_on_twelfth_end() {
    let mut b = ChunkBuilder::new();
    for _ in 0..12 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    struct TwelfthSomeHost {
        calls: u32,
    }
    impl ShellHost for TwelfthSomeHost {
        fn subshell_end(&mut self) -> Option<i32> {
            self.calls += 1;
            if self.calls == 12 {
                Some(224)
            } else {
                None
            }
        }
    }
    let mut vm = VM::new(b.build());
    vm.last_status = 1;
    vm.set_shell_host(Box::new(TwelfthSomeHost { calls: 0 }));
    match vm.run() {
        VMResult::Ok(Value::Status(224)) => {}
        other => panic!("twelfth subshell_end(Some(224)) MUST apply, got {:?}", other),
    }
}

#[test]
fn request_halt_between_subshell_end_and_getstatus_preserves_last_status() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(225)));
    vm.register_builtin(100, builtin_request_halt);
    let _ = vm.run();
    assert_eq!(vm.last_status, 225);
}
// ─── Pin tests batch I (handwritten) ──────────────────────────────────────

#[test]
fn subshell_end_exit_code_224_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(224)));
    match vm.run() {
        VMResult::Ok(Value::Status(224)) => {}
        other => panic!("exit code 224 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_twenty_seven_consecutive_runs_without_reset_still_blocked() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    for _ in 0..27 {
        let _ = vm.run();
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_225_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(225)));
    match vm.run() {
        VMResult::Ok(Value::Status(225)) => {}
        other => panic!("exit code 225 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_str_lt_on_consts() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    let a = b.add_constant(Value::str("alpha"));
    let bidx = b.add_constant(Value::str("omega"));
    b.emit(Op::LoadConst(a), 1);
    b.emit(Op::LoadConst(bidx), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::StrLt, 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_226_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(226)));
    match vm.run() {
        VMResult::Ok(Value::Status(226)) => {}
        other => panic!("exit code 226 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_twenty_eight_consecutive_runs_without_reset_still_blocked() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    for _ in 0..28 {
        let _ = vm.run();
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_227_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(227)));
    match vm.run() {
        VMResult::Ok(Value::Status(227)) => {}
        other => panic!("exit code 227 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_make_hash_after_halt_point() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    let k = b.add_constant(Value::str("k"));
    let v = b.add_constant(Value::str("v"));
    b.emit(Op::LoadConst(k), 1);
    b.emit(Op::LoadConst(v), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::MakeHash(1), 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_228_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(228)));
    match vm.run() {
        VMResult::Ok(Value::Status(228)) => {}
        other => panic!("exit code 228 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_twenty_nine_consecutive_runs_without_reset_still_blocked() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    for _ in 0..29 {
        let _ = vm.run();
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_229_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(229)));
    match vm.run() {
        VMResult::Ok(Value::Status(229)) => {}
        other => panic!("exit code 229 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_subshell_status230_blocks_following_pipeline_end() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    PIPELINE_END_CALLS.store(0, Ordering::SeqCst);
    struct SubshellThenPipelineEndHost230;
    impl ShellHost for SubshellThenPipelineEndHost230 {
        fn subshell_end(&mut self) -> Option<i32> {
            Some(230)
        }
        fn pipeline_end(&mut self) -> i32 {
            PIPELINE_END_CALLS.fetch_add(1, Ordering::SeqCst);
            0
        }
    }
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::PipelineEnd, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(SubshellThenPipelineEndHost230));
    vm.register_builtin(100, builtin_request_halt);
    let _ = vm.run();
    assert_eq!(PIPELINE_END_CALLS.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 230);
}

#[test]
fn subshell_end_exit_code_230_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(230)));
    match vm.run() {
        VMResult::Ok(Value::Status(230)) => {}
        other => panic!("exit code 230 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_thirty_consecutive_runs_without_reset_still_blocked() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    for _ in 0..30 {
        let _ = vm.run();
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_231_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(231)));
    match vm.run() {
        VMResult::Ok(Value::Status(231)) => {}
        other => panic!("exit code 231 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_forty_two_end_incrementing_from_three() {
    let mut b = ChunkBuilder::new();
    for _ in 0..42 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(IncrementingSubshellHost { next: 3 }));
    match vm.run() {
        VMResult::Ok(Value::Status(44)) => {}
        other => panic!("forty-second subshell_end(Some(44)) MUST win, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_232_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(232)));
    match vm.run() {
        VMResult::Ok(Value::Status(232)) => {}
        other => panic!("exit code 232 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_thirty_one_consecutive_runs_without_reset_still_blocked() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    for _ in 0..31 {
        let _ = vm.run();
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_233_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(233)));
    match vm.run() {
        VMResult::Ok(Value::Status(233)) => {}
        other => panic!("exit code 233 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_status_minus_twenty_two_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(-22)));
    match vm.run() {
        VMResult::Ok(Value::Status(-22)) => {}
        other => panic!("status -22 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_234_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(234)));
    match vm.run() {
        VMResult::Ok(Value::Status(234)) => {}
        other => panic!("exit code 234 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_forty_three_end_incrementing_from_five() {
    let mut b = ChunkBuilder::new();
    for _ in 0..43 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(IncrementingSubshellHost { next: 5 }));
    match vm.run() {
        VMResult::Ok(Value::Status(47)) => {}
        other => panic!("forty-third subshell_end(Some(47)) MUST win, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_235_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(235)));
    match vm.run() {
        VMResult::Ok(Value::Status(235)) => {}
        other => panic!("exit code 235 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_status_minus_twenty_three_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(-23)));
    match vm.run() {
        VMResult::Ok(Value::Status(-23)) => {}
        other => panic!("status -23 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_236_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(236)));
    match vm.run() {
        VMResult::Ok(Value::Status(236)) => {}
        other => panic!("exit code 236 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_thirteenth_some_host_applies_on_thirteenth_end() {
    let mut b = ChunkBuilder::new();
    for _ in 0..13 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    struct ThirteenthSomeHost {
        calls: u32,
    }
    impl ShellHost for ThirteenthSomeHost {
        fn subshell_end(&mut self) -> Option<i32> {
            self.calls += 1;
            if self.calls == 13 {
                Some(237)
            } else {
                None
            }
        }
    }
    let mut vm = VM::new(b.build());
    vm.last_status = 4;
    vm.set_shell_host(Box::new(ThirteenthSomeHost { calls: 0 }));
    match vm.run() {
        VMResult::Ok(Value::Status(237)) => {}
        other => panic!("thirteenth subshell_end(Some(237)) MUST apply, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_237_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(237)));
    match vm.run() {
        VMResult::Ok(Value::Status(237)) => {}
        other => panic!("exit code 237 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_status_minus_twenty_four_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(-24)));
    match vm.run() {
        VMResult::Ok(Value::Status(-24)) => {}
        other => panic!("status -24 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_238_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(238)));
    match vm.run() {
        VMResult::Ok(Value::Status(238)) => {}
        other => panic!("exit code 238 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_str_ge_on_consts() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    let a = b.add_constant(Value::str("zeta"));
    let bidx = b.add_constant(Value::str("alpha"));
    b.emit(Op::LoadConst(a), 1);
    b.emit(Op::LoadConst(bidx), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::StrGe, 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_239_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(239)));
    match vm.run() {
        VMResult::Ok(Value::Status(239)) => {}
        other => panic!("exit code 239 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_status_minus_twenty_five_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(-25)));
    match vm.run() {
        VMResult::Ok(Value::Status(-25)) => {}
        other => panic!("status -25 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_240_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(240)));
    match vm.run() {
        VMResult::Ok(Value::Status(240)) => {}
        other => panic!("exit code 240 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_forty_four_end_incrementing_from_eight() {
    let mut b = ChunkBuilder::new();
    for _ in 0..44 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(IncrementingSubshellHost { next: 8 }));
    match vm.run() {
        VMResult::Ok(Value::Status(51)) => {}
        other => panic!("forty-fourth subshell_end(Some(51)) MUST win, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_241_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(241)));
    match vm.run() {
        VMResult::Ok(Value::Status(241)) => {}
        other => panic!("exit code 241 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_status_minus_twenty_six_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(-26)));
    match vm.run() {
        VMResult::Ok(Value::Status(-26)) => {}
        other => panic!("status -26 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_242_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(242)));
    match vm.run() {
        VMResult::Ok(Value::Status(242)) => {}
        other => panic!("exit code 242 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_halt_reset_halt_reset_subshell_end_fifth_run_applies() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    SUBSHELL_END_CALLS.store(0, Ordering::SeqCst);
    let sub = {
        let mut b = ChunkBuilder::new();
        b.emit(Op::SubshellEnd, 1);
        b.emit(Op::CallBuiltin(100, 0), 1);
        b.build()
    };
    let end_only = {
        let mut b = ChunkBuilder::new();
        b.emit(Op::SubshellEnd, 1);
        b.build()
    };
    let mut vm = VM::new(sub.clone());
    vm.set_shell_host(Box::new(CountingSubshellEndHost(243)));
    vm.register_builtin(100, builtin_request_halt);
    let _ = vm.run();
    vm.reset(sub);
    let _ = vm.run();
    vm.reset(end_only.clone());
    let _ = vm.run();
    vm.reset(end_only.clone());
    let _ = vm.run();
    vm.reset(end_only);
    let _ = vm.run();
    assert_eq!(SUBSHELL_END_CALLS.load(Ordering::SeqCst), 5);
    assert_eq!(vm.last_status, 243);
}

#[test]
fn subshell_end_exit_code_243_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(243)));
    match vm.run() {
        VMResult::Ok(Value::Status(243)) => {}
        other => panic!("exit code 243 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_forty_five_end_incrementing_from_eleven() {
    let mut b = ChunkBuilder::new();
    for _ in 0..45 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(IncrementingSubshellHost { next: 11 }));
    match vm.run() {
        VMResult::Ok(Value::Status(55)) => {}
        other => panic!("forty-fifth subshell_end(Some(55)) MUST win, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_244_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(244)));
    match vm.run() {
        VMResult::Ok(Value::Status(244)) => {}
        other => panic!("exit code 244 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_triple_reset_same_host_same_status_each_run() {
    let chunk = {
        let mut b = ChunkBuilder::new();
        b.emit(Op::SubshellEnd, 1);
        b.emit(Op::GetStatus, 1);
        b.build()
    };
    let mut vm = VM::new(chunk.clone());
    vm.set_shell_host(Box::new(StatusReturningHost(245)));
    for _ in 0..3 {
        match vm.run() {
            VMResult::Ok(Value::Status(245)) => {}
            other => panic!("each reset run MUST return Status(245), got {:?}", other),
        }
        vm.reset(chunk.clone());
    }
}

#[test]
fn subshell_end_exit_code_245_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(245)));
    match vm.run() {
        VMResult::Ok(Value::Status(245)) => {}
        other => panic!("exit code 245 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_between_pipeline_stage_and_pipeline_end_after_subshell() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    PIPELINE_STAGE_CALLS.store(0, Ordering::SeqCst);
    PIPELINE_END_CALLS.store(0, Ordering::SeqCst);
    struct SubshellStagePipelineHost246;
    impl ShellHost for SubshellStagePipelineHost246 {
        fn subshell_end(&mut self) -> Option<i32> {
            Some(246)
        }
        fn pipeline_stage(&mut self) {
            PIPELINE_STAGE_CALLS.fetch_add(1, Ordering::SeqCst);
        }
        fn pipeline_end(&mut self) -> i32 {
            PIPELINE_END_CALLS.fetch_add(1, Ordering::SeqCst);
            0
        }
    }
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::PipelineBegin(2), 1);
    b.emit(Op::PipelineStage, 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::PipelineEnd, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(SubshellStagePipelineHost246));
    vm.register_builtin(100, builtin_request_halt);
    let _ = vm.run();
    assert_eq!(PIPELINE_STAGE_CALLS.load(Ordering::SeqCst), 1);
    assert_eq!(PIPELINE_END_CALLS.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 246);
}

#[test]
fn subshell_end_exit_code_246_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(246)));
    match vm.run() {
        VMResult::Ok(Value::Status(246)) => {}
        other => panic!("exit code 246 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_four_end_alternating_host_last_some_wins() {
    let mut b = ChunkBuilder::new();
    for _ in 0..4 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(AlternatingStatusHost { calls: 0 }));
    match vm.run() {
        VMResult::Ok(Value::Status(9)) => {}
        other => panic!("fourth AlternatingStatusHost end MUST keep Some(9), got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_247_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(247)));
    match vm.run() {
        VMResult::Ok(Value::Status(247)) => {}
        other => panic!("exit code 247 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_after_subshell_end_before_load_true() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::LoadTrue, 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(248)));
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 248);
}

#[test]
fn subshell_end_exit_code_248_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(248)));
    match vm.run() {
        VMResult::Ok(Value::Status(248)) => {}
        other => panic!("exit code 248 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_load_undef_after_subshell_end() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::LoadUndef, 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(249)));
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 249);
}
// ─── Pin tests batch J (handwritten) ──────────────────────────────────────

#[test]
fn subshell_end_exit_code_249_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(249)));
    match vm.run() {
        VMResult::Ok(Value::Status(249)) => {}
        other => panic!("exit code 249 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_thirty_two_consecutive_runs_without_reset_still_blocked() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    for _ in 0..32 {
        let _ = vm.run();
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_250_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(250)));
    match vm.run() {
        VMResult::Ok(Value::Status(250)) => {}
        other => panic!("exit code 250 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_thirty_three_consecutive_runs_without_reset_still_blocked() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    for _ in 0..33 {
        let _ = vm.run();
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_251_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(251)));
    match vm.run() {
        VMResult::Ok(Value::Status(251)) => {}
        other => panic!("exit code 251 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_thirty_four_consecutive_runs_without_reset_still_blocked() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    for _ in 0..34 {
        let _ = vm.run();
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_252_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(252)));
    match vm.run() {
        VMResult::Ok(Value::Status(252)) => {}
        other => panic!("exit code 252 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_thirty_five_consecutive_runs_without_reset_still_blocked() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    for _ in 0..35 {
        let _ = vm.run();
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_253_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(253)));
    match vm.run() {
        VMResult::Ok(Value::Status(253)) => {}
        other => panic!("exit code 253 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_forty_six_end_incrementing_from_two() {
    let mut b = ChunkBuilder::new();
    for _ in 0..46 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(IncrementingSubshellHost { next: 2 }));
    match vm.run() {
        VMResult::Ok(Value::Status(47)) => {}
        other => panic!("forty-sixth subshell_end(Some(47)) MUST win, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_254_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(254)));
    match vm.run() {
        VMResult::Ok(Value::Status(254)) => {}
        other => panic!("exit code 254 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_status_minus_twenty_seven_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(-27)));
    match vm.run() {
        VMResult::Ok(Value::Status(-27)) => {}
        other => panic!("status -27 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_256_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(256)));
    match vm.run() {
        VMResult::Ok(Value::Status(256)) => {}
        other => panic!("exit code 256 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_forty_seven_end_incrementing_from_six() {
    let mut b = ChunkBuilder::new();
    for _ in 0..47 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(IncrementingSubshellHost { next: 6 }));
    match vm.run() {
        VMResult::Ok(Value::Status(52)) => {}
        other => panic!("forty-seventh subshell_end(Some(52)) MUST win, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_257_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(257)));
    match vm.run() {
        VMResult::Ok(Value::Status(257)) => {}
        other => panic!("exit code 257 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_status_minus_twenty_eight_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(-28)));
    match vm.run() {
        VMResult::Ok(Value::Status(-28)) => {}
        other => panic!("status -28 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_258_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(258)));
    match vm.run() {
        VMResult::Ok(Value::Status(258)) => {}
        other => panic!("exit code 258 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_fourteenth_some_host_applies_on_fourteenth_end() {
    let mut b = ChunkBuilder::new();
    for _ in 0..14 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    struct FourteenthSomeHost {
        calls: u32,
    }
    impl ShellHost for FourteenthSomeHost {
        fn subshell_end(&mut self) -> Option<i32> {
            self.calls += 1;
            if self.calls == 14 {
                Some(259)
            } else {
                None
            }
        }
    }
    let mut vm = VM::new(b.build());
    vm.last_status = 6;
    vm.set_shell_host(Box::new(FourteenthSomeHost { calls: 0 }));
    match vm.run() {
        VMResult::Ok(Value::Status(259)) => {}
        other => panic!("fourteenth subshell_end(Some(259)) MUST apply, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_259_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(259)));
    match vm.run() {
        VMResult::Ok(Value::Status(259)) => {}
        other => panic!("exit code 259 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_status_minus_twenty_nine_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(-29)));
    match vm.run() {
        VMResult::Ok(Value::Status(-29)) => {}
        other => panic!("status -29 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_260_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(260)));
    match vm.run() {
        VMResult::Ok(Value::Status(260)) => {}
        other => panic!("exit code 260 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_forty_eight_end_incrementing_from_ten() {
    let mut b = ChunkBuilder::new();
    for _ in 0..48 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(IncrementingSubshellHost { next: 10 }));
    match vm.run() {
        VMResult::Ok(Value::Status(57)) => {}
        other => panic!("forty-eighth subshell_end(Some(57)) MUST win, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_261_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(261)));
    match vm.run() {
        VMResult::Ok(Value::Status(261)) => {}
        other => panic!("exit code 261 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_status_minus_thirty_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(-30)));
    match vm.run() {
        VMResult::Ok(Value::Status(-30)) => {}
        other => panic!("status -30 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_262_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(262)));
    match vm.run() {
        VMResult::Ok(Value::Status(262)) => {}
        other => panic!("exit code 262 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_callee_zero_args_return_before_caller_halt() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    let name = b.add_name("noop");
    b.emit(Op::Call(name, 0), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let skip = b.emit(Op::Jump(0), 1);
    let entry = b.current_pos();
    b.add_sub_entry(name, entry);
    b.emit(Op::ReturnValue, 1);
    b.patch_jump(skip, b.current_pos());
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    match vm.run() {
        VMResult::Ok(Value::Int(0)) => {}
        other => panic!("halt MUST be stack top after zero-arg callee, got {:?}", other),
    }
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_263_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(263)));
    match vm.run() {
        VMResult::Ok(Value::Status(263)) => {}
        other => panic!("exit code 263 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_after_first_getstatus_blocks_second_getstatus() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(264)));
    vm.register_builtin(100, builtin_request_halt);
    match vm.run() {
        VMResult::Ok(Value::Int(0)) => {}
        other => panic!("second GetStatus MUST NOT run after halt, got {:?}", other),
    }
    assert_eq!(vm.last_status, 264);
}

#[test]
fn subshell_end_exit_code_264_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(264)));
    match vm.run() {
        VMResult::Ok(Value::Status(264)) => {}
        other => panic!("exit code 264 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_pow_after_subshell_end() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::LoadInt(2), 1);
    b.emit(Op::LoadInt(3), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::Pow, 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(264)));
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 264);
}

#[test]
fn subshell_end_exit_code_265_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(265)));
    match vm.run() {
        VMResult::Ok(Value::Status(265)) => {}
        other => panic!("exit code 265 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_between_two_subshell_ends_second_host_not_called() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    SUBSHELL_END_CALLS.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::SubshellEnd, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(CountingSubshellEndHost(266)));
    vm.register_builtin(100, builtin_request_halt);
    let _ = vm.run();
    assert_eq!(SUBSHELL_END_CALLS.load(Ordering::SeqCst), 1);
    assert_eq!(vm.last_status, 266);
}

#[test]
fn subshell_end_exit_code_266_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(266)));
    match vm.run() {
        VMResult::Ok(Value::Status(266)) => {}
        other => panic!("exit code 266 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_str_le_on_consts() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    let a = b.add_constant(Value::str("abc"));
    let bidx = b.add_constant(Value::str("abd"));
    b.emit(Op::LoadConst(a), 1);
    b.emit(Op::LoadConst(bidx), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::StrLe, 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_267_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(267)));
    match vm.run() {
        VMResult::Ok(Value::Status(267)) => {}
        other => panic!("exit code 267 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_after_subshell_end_before_narrow_extension() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    static EXT_RAN: AtomicU32 = AtomicU32::new(0);
    EXT_RAN.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::Extended(55, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(268)));
    vm.set_extension_handler(Box::new(|_vm, _id, _arg| {
        EXT_RAN.fetch_add(1, Ordering::SeqCst);
    }));
    vm.register_builtin(100, builtin_request_halt);
    let _ = vm.run();
    assert_eq!(EXT_RAN.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 268);
}

#[test]
fn subshell_end_exit_code_268_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(268)));
    match vm.run() {
        VMResult::Ok(Value::Status(268)) => {}
        other => panic!("exit code 268 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_none_host_after_setstatus_before_getstatus_reads_setstatus() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(269), 1);
    b.emit(Op::SetStatus, 1);
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(NoneHost));
    match vm.run() {
        VMResult::Ok(Value::Status(269)) => {}
        other => panic!(
            "NoneHost after SubshellBegin MUST preserve SetStatus(269), got {:?}",
            other
        ),
    }
}

#[test]
fn subshell_end_exit_code_269_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(269)));
    match vm.run() {
        VMResult::Ok(Value::Status(269)) => {}
        other => panic!("exit code 269 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_subshell_status270_blocks_following_pipeline_begin() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    PIPELINE_BEGIN_CALLS.store(0, Ordering::SeqCst);
    struct SubshellPipelineBeginHost270;
    impl ShellHost for SubshellPipelineBeginHost270 {
        fn subshell_end(&mut self) -> Option<i32> {
            Some(270)
        }
        fn pipeline_begin(&mut self, _n: u8) {
            PIPELINE_BEGIN_CALLS.fetch_add(1, Ordering::SeqCst);
        }
    }
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::PipelineBegin(2), 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(SubshellPipelineBeginHost270));
    vm.register_builtin(100, builtin_request_halt);
    let _ = vm.run();
    assert_eq!(PIPELINE_BEGIN_CALLS.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 270);
}

#[test]
fn subshell_end_exit_code_270_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(270)));
    match vm.run() {
        VMResult::Ok(Value::Status(270)) => {}
        other => panic!("exit code 270 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_thirty_six_consecutive_runs_without_reset_still_blocked() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    for _ in 0..36 {
        let _ = vm.run();
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_271_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(271)));
    match vm.run() {
        VMResult::Ok(Value::Status(271)) => {}
        other => panic!("exit code 271 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_max_then_zero_second_end_overwrites() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    struct MaxThenZeroHost {
        calls: u32,
    }
    impl ShellHost for MaxThenZeroHost {
        fn subshell_end(&mut self) -> Option<i32> {
            self.calls += 1;
            if self.calls == 1 {
                Some(i32::MAX)
            } else {
                Some(0)
            }
        }
    }
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(MaxThenZeroHost { calls: 0 }));
    match vm.run() {
        VMResult::Ok(Value::Status(0)) => {}
        other => panic!("second subshell_end(Some(0)) MUST overwrite MAX, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_272_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(272)));
    match vm.run() {
        VMResult::Ok(Value::Status(272)) => {}
        other => panic!("exit code 272 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_between_two_loadints_after_subshell_end() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::LoadInt(11), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::LoadInt(22), 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(273)));
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 273);
}

#[test]
fn subshell_end_exit_code_273_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(273)));
    match vm.run() {
        VMResult::Ok(Value::Status(273)) => {}
        other => panic!("exit code 273 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_between_two_pipeline_ends_blocks_second() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    PIPELINE_END_CALLS.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::PipelineEnd, 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::PipelineEnd, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(CountingPipelineHost));
    vm.register_builtin(100, builtin_request_halt);
    let _ = vm.run();
    assert_eq!(PIPELINE_END_CALLS.load(Ordering::SeqCst), 1);
}

#[test]
fn subshell_end_exit_code_274_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(274)));
    match vm.run() {
        VMResult::Ok(Value::Status(274)) => {}
        other => panic!("exit code 274 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_fifteenth_some_host_applies_on_fifteenth_end() {
    let mut b = ChunkBuilder::new();
    for _ in 0..15 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    struct FifteenthSomeHost {
        calls: u32,
    }
    impl ShellHost for FifteenthSomeHost {
        fn subshell_end(&mut self) -> Option<i32> {
            self.calls += 1;
            if self.calls == 15 {
                Some(275)
            } else {
                None
            }
        }
    }
    let mut vm = VM::new(b.build());
    vm.last_status = 2;
    vm.set_shell_host(Box::new(FifteenthSomeHost { calls: 0 }));
    match vm.run() {
        VMResult::Ok(Value::Status(275)) => {}
        other => panic!("fifteenth subshell_end(Some(275)) MUST apply, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_str_gt_on_consts() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    let a = b.add_constant(Value::str("longer"));
    let bidx = b.add_constant(Value::str("short"));
    b.emit(Op::LoadConst(a), 1);
    b.emit(Op::LoadConst(bidx), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::StrGt, 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_276_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(276)));
    match vm.run() {
        VMResult::Ok(Value::Status(276)) => {}
        other => panic!("exit code 276 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_forty_nine_end_incrementing_from_thirteen() {
    let mut b = ChunkBuilder::new();
    for _ in 0..49 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(IncrementingSubshellHost { next: 13 }));
    match vm.run() {
        VMResult::Ok(Value::Status(61)) => {}
        other => panic!("forty-ninth subshell_end(Some(61)) MUST win, got {:?}", other),
    }
}

#[test]
fn request_halt_thirty_seven_consecutive_runs_without_reset_still_blocked() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    for _ in 0..37 {
        let _ = vm.run();
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_status_minus_thirty_one_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(-31)));
    match vm.run() {
        VMResult::Ok(Value::Status(-31)) => {}
        other => panic!("status -31 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_halt_reset_subshell_getstatus_on_fourth_reset_run() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    let halt = {
        let mut b = ChunkBuilder::new();
        b.emit(Op::CallBuiltin(100, 0), 1);
        b.build()
    };
    let sub = {
        let mut b = ChunkBuilder::new();
        b.emit(Op::SubshellEnd, 1);
        b.emit(Op::GetStatus, 1);
        b.build()
    };
    let mut vm = VM::new(halt.clone());
    vm.set_shell_host(Box::new(StatusReturningHost(277)));
    vm.register_builtin(100, builtin_request_halt);
    let _ = vm.run();
    vm.reset(halt);
    let _ = vm.run();
    vm.reset(sub.clone());
    let _ = vm.run();
    vm.reset(sub);
    match vm.run() {
        VMResult::Ok(Value::Status(277)) => {}
        other => panic!("fourth run MUST return Status(277), got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_277_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(277)));
    match vm.run() {
        VMResult::Ok(Value::Status(277)) => {}
        other => panic!("exit code 277 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_load_false_after_subshell_wrote_status() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::LoadFalse, 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(278)));
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 278);
}

#[test]
fn subshell_end_exit_code_278_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(278)));
    match vm.run() {
        VMResult::Ok(Value::Status(278)) => {}
        other => panic!("exit code 278 MUST propagate, got {:?}", other),
    }
}
// ─── Pin tests batch K (handwritten) ──────────────────────────────────────

#[test]
fn subshell_end_exit_code_279_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(279)));
    match vm.run() {
        VMResult::Ok(Value::Status(279)) => {}
        other => panic!("exit code 279 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_thirty_eight_consecutive_runs_without_reset_still_blocked() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    for _ in 0..38 {
        let _ = vm.run();
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_280_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(280)));
    match vm.run() {
        VMResult::Ok(Value::Status(280)) => {}
        other => panic!("exit code 280 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_thirty_nine_consecutive_runs_without_reset_still_blocked() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    for _ in 0..39 {
        let _ = vm.run();
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_281_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(281)));
    match vm.run() {
        VMResult::Ok(Value::Status(281)) => {}
        other => panic!("exit code 281 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_forty_consecutive_runs_without_reset_still_blocked() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    for _ in 0..40 {
        let _ = vm.run();
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_282_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(282)));
    match vm.run() {
        VMResult::Ok(Value::Status(282)) => {}
        other => panic!("exit code 282 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_fifty_end_incrementing_from_four() {
    let mut b = ChunkBuilder::new();
    for _ in 0..50 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(IncrementingSubshellHost { next: 4 }));
    match vm.run() {
        VMResult::Ok(Value::Status(53)) => {}
        other => panic!("fiftieth subshell_end(Some(53)) MUST win, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_283_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(283)));
    match vm.run() {
        VMResult::Ok(Value::Status(283)) => {}
        other => panic!("exit code 283 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_status_minus_thirty_two_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(-32)));
    match vm.run() {
        VMResult::Ok(Value::Status(-32)) => {}
        other => panic!("status -32 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_284_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(284)));
    match vm.run() {
        VMResult::Ok(Value::Status(284)) => {}
        other => panic!("exit code 284 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_status_minus_thirty_three_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(-33)));
    match vm.run() {
        VMResult::Ok(Value::Status(-33)) => {}
        other => panic!("status -33 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_285_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(285)));
    match vm.run() {
        VMResult::Ok(Value::Status(285)) => {}
        other => panic!("exit code 285 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_fifty_one_end_incrementing_from_seven() {
    let mut b = ChunkBuilder::new();
    for _ in 0..51 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(IncrementingSubshellHost { next: 7 }));
    match vm.run() {
        VMResult::Ok(Value::Status(57)) => {}
        other => panic!("fifty-first subshell_end(Some(57)) MUST win, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_286_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(286)));
    match vm.run() {
        VMResult::Ok(Value::Status(286)) => {}
        other => panic!("exit code 286 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_status_minus_thirty_four_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(-34)));
    match vm.run() {
        VMResult::Ok(Value::Status(-34)) => {}
        other => panic!("status -34 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_287_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(287)));
    match vm.run() {
        VMResult::Ok(Value::Status(287)) => {}
        other => panic!("exit code 287 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_status_minus_thirty_five_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(-35)));
    match vm.run() {
        VMResult::Ok(Value::Status(-35)) => {}
        other => panic!("status -35 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_288_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(288)));
    match vm.run() {
        VMResult::Ok(Value::Status(288)) => {}
        other => panic!("exit code 288 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_status_minus_thirty_six_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(-36)));
    match vm.run() {
        VMResult::Ok(Value::Status(-36)) => {}
        other => panic!("status -36 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_289_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(289)));
    match vm.run() {
        VMResult::Ok(Value::Status(289)) => {}
        other => panic!("exit code 289 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_sixteenth_some_host_applies_on_sixteenth_end() {
    let mut b = ChunkBuilder::new();
    for _ in 0..16 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    struct SixteenthSomeHost {
        calls: u32,
    }
    impl ShellHost for SixteenthSomeHost {
        fn subshell_end(&mut self) -> Option<i32> {
            self.calls += 1;
            if self.calls == 16 {
                Some(290)
            } else {
                None
            }
        }
    }
    let mut vm = VM::new(b.build());
    vm.last_status = 5;
    vm.set_shell_host(Box::new(SixteenthSomeHost { calls: 0 }));
    match vm.run() {
        VMResult::Ok(Value::Status(290)) => {}
        other => panic!("sixteenth subshell_end(Some(290)) MUST apply, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_290_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(290)));
    match vm.run() {
        VMResult::Ok(Value::Status(290)) => {}
        other => panic!("exit code 290 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_mod_after_subshell_end() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::LoadInt(10), 1);
    b.emit(Op::LoadInt(3), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::Mod, 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(291)));
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 291);
}

#[test]
fn subshell_end_exit_code_291_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(291)));
    match vm.run() {
        VMResult::Ok(Value::Status(291)) => {}
        other => panic!("exit code 291 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_div_after_subshell_end() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::LoadInt(20), 1);
    b.emit(Op::LoadInt(4), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::Div, 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(292)));
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 292);
}

#[test]
fn subshell_end_exit_code_292_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(292)));
    match vm.run() {
        VMResult::Ok(Value::Status(292)) => {}
        other => panic!("exit code 292 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_halt_reset_subshell_end_sixth_run_applies() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    SUBSHELL_END_CALLS.store(0, Ordering::SeqCst);
    let sub = {
        let mut b = ChunkBuilder::new();
        b.emit(Op::SubshellEnd, 1);
        b.emit(Op::CallBuiltin(100, 0), 1);
        b.build()
    };
    let end_only = {
        let mut b = ChunkBuilder::new();
        b.emit(Op::SubshellEnd, 1);
        b.build()
    };
    let mut vm = VM::new(sub.clone());
    vm.set_shell_host(Box::new(CountingSubshellEndHost(293)));
    vm.register_builtin(100, builtin_request_halt);
    let _ = vm.run();
    vm.reset(sub);
    let _ = vm.run();
    vm.reset(end_only.clone());
    let _ = vm.run();
    vm.reset(end_only.clone());
    let _ = vm.run();
    vm.reset(end_only.clone());
    let _ = vm.run();
    vm.reset(end_only);
    let _ = vm.run();
    assert_eq!(SUBSHELL_END_CALLS.load(Ordering::SeqCst), 6);
    assert_eq!(vm.last_status, 293);
}

#[test]
fn subshell_end_exit_code_293_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(293)));
    match vm.run() {
        VMResult::Ok(Value::Status(293)) => {}
        other => panic!("exit code 293 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_negate_after_subshell_wrote_status() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::LoadInt(8), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::Negate, 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(294)));
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 294);
}

#[test]
fn subshell_end_exit_code_294_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(294)));
    match vm.run() {
        VMResult::Ok(Value::Status(294)) => {}
        other => panic!("exit code 294 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_five_begins_two_ends_applies_last_end_status() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    SUBSHELL_BEGIN_CALLS.store(0, Ordering::SeqCst);
    SUBSHELL_END_CALLS.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    for _ in 0..5 {
        b.emit(Op::SubshellBegin, 1);
    }
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    struct CountingBeginEndHost295;
    impl ShellHost for CountingBeginEndHost295 {
        fn subshell_begin(&mut self) {
            SUBSHELL_BEGIN_CALLS.fetch_add(1, Ordering::SeqCst);
        }
        fn subshell_end(&mut self) -> Option<i32> {
            SUBSHELL_END_CALLS.fetch_add(1, Ordering::SeqCst);
            Some(295)
        }
    }
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(CountingBeginEndHost295));
    match vm.run() {
        VMResult::Ok(Value::Status(295)) => {}
        other => panic!("second subshell_end(Some(295)) MUST win, got {:?}", other),
    }
    assert_eq!(SUBSHELL_BEGIN_CALLS.load(Ordering::SeqCst), 5);
    assert_eq!(SUBSHELL_END_CALLS.load(Ordering::SeqCst), 2);
}

#[test]
fn subshell_end_exit_code_295_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(295)));
    match vm.run() {
        VMResult::Ok(Value::Status(295)) => {}
        other => panic!("exit code 295 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_between_second_subshell_begin_and_end() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    SUBSHELL_END_CALLS.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::SubshellEnd, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(CountingSubshellEndHost(296)));
    vm.register_builtin(100, builtin_request_halt);
    let _ = vm.run();
    assert_eq!(SUBSHELL_END_CALLS.load(Ordering::SeqCst), 1);
    assert_eq!(vm.last_status, 296);
}

#[test]
fn subshell_end_exit_code_296_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(296)));
    match vm.run() {
        VMResult::Ok(Value::Status(296)) => {}
        other => panic!("exit code 296 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_alternating_status_eighth_none_keeps_ninth() {
    let mut b = ChunkBuilder::new();
    for _ in 0..8 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(AlternatingStatusHost { calls: 0 }));
    match vm.run() {
        VMResult::Ok(Value::Status(9)) => {}
        other => panic!("AlternatingStatusHost eighth None MUST keep Some(9), got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_297_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(297)));
    match vm.run() {
        VMResult::Ok(Value::Status(297)) => {}
        other => panic!("exit code 297 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_pre_run_double_reset_then_subshell_end_runs_host298() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    SUBSHELL_END_CALLS.store(0, Ordering::SeqCst);
    let empty = ChunkBuilder::new().build();
    let sub = {
        let mut b = ChunkBuilder::new();
        b.emit(Op::SubshellEnd, 1);
        b.build()
    };
    let mut vm = VM::new(empty.clone());
    vm.set_shell_host(Box::new(CountingSubshellEndHost(298)));
    vm.request_halt();
    let _ = vm.run();
    vm.reset(empty);
    vm.request_halt();
    let _ = vm.run();
    vm.reset(sub);
    let _ = vm.run();
    assert_eq!(SUBSHELL_END_CALLS.load(Ordering::SeqCst), 1);
    assert_eq!(vm.last_status, 298);
}

#[test]
fn subshell_end_exit_code_298_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(298)));
    match vm.run() {
        VMResult::Ok(Value::Status(298)) => {}
        other => panic!("exit code 298 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_dup2_after_subshell_end() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::LoadInt(2), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::Dup2, 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(299)));
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 299);
}

#[test]
fn subshell_end_exit_code_299_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(299)));
    match vm.run() {
        VMResult::Ok(Value::Status(299)) => {}
        other => panic!("exit code 299 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_four_reset_runs_same_host_same_status() {
    let chunk = {
        let mut b = ChunkBuilder::new();
        b.emit(Op::SubshellEnd, 1);
        b.emit(Op::GetStatus, 1);
        b.build()
    };
    let mut vm = VM::new(chunk.clone());
    vm.set_shell_host(Box::new(StatusReturningHost(300)));
    for _ in 0..4 {
        match vm.run() {
            VMResult::Ok(Value::Status(300)) => {}
            other => panic!("reset run MUST return Status(300), got {:?}", other),
        }
        vm.reset(chunk.clone());
    }
}

#[test]
fn subshell_end_exit_code_300_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(300)));
    match vm.run() {
        VMResult::Ok(Value::Status(300)) => {}
        other => panic!("exit code 300 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_spaceship_after_subshell_end() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::LoadInt(3), 1);
    b.emit(Op::LoadInt(7), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::Spaceship, 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(301)));
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 301);
}

#[test]
fn subshell_end_exit_code_301_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(301)));
    match vm.run() {
        VMResult::Ok(Value::Status(301)) => {}
        other => panic!("exit code 301 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_pipeline_stage_subshell_end_getstatus_reads_subshell() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::PipelineBegin(2), 1);
    b.emit(Op::PipelineStage, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(PipelineThenSubshellHost));
    match vm.run() {
        VMResult::Ok(Value::Status(9)) => {}
        other => panic!("GetStatus MUST read subshell Some(9), got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_302_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(302)));
    match vm.run() {
        VMResult::Ok(Value::Status(302)) => {}
        other => panic!("exit code 302 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_make_array_after_subshell_end() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::LoadInt(2), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::MakeArray(2), 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(303)));
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 303);
}

#[test]
fn subshell_end_exit_code_303_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(303)));
    match vm.run() {
        VMResult::Ok(Value::Status(303)) => {}
        other => panic!("exit code 303 MUST propagate, got {:?}", other),
    }
}
// ─── Pin tests batch L (handwritten) ──────────────────────────────────────

#[test]
fn subshell_end_exit_code_304_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(304)));
    match vm.run() {
        VMResult::Ok(Value::Status(304)) => {}
        other => panic!("exit code 304 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_forty_one_consecutive_runs_without_reset_still_blocked() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    for _ in 0..41 {
        let _ = vm.run();
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_305_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(305)));
    match vm.run() {
        VMResult::Ok(Value::Status(305)) => {}
        other => panic!("exit code 305 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_forty_two_consecutive_runs_without_reset_still_blocked() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    for _ in 0..42 {
        let _ = vm.run();
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_306_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(306)));
    match vm.run() {
        VMResult::Ok(Value::Status(306)) => {}
        other => panic!("exit code 306 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_forty_three_consecutive_runs_without_reset_still_blocked() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    for _ in 0..43 {
        let _ = vm.run();
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_307_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(307)));
    match vm.run() {
        VMResult::Ok(Value::Status(307)) => {}
        other => panic!("exit code 307 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_fifty_two_end_incrementing_from_two() {
    let mut b = ChunkBuilder::new();
    for _ in 0..52 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(IncrementingSubshellHost { next: 2 }));
    match vm.run() {
        VMResult::Ok(Value::Status(53)) => {}
        other => panic!("fifty-second subshell_end(Some(53)) MUST win, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_308_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(308)));
    match vm.run() {
        VMResult::Ok(Value::Status(308)) => {}
        other => panic!("exit code 308 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_status_minus_thirty_seven_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(-37)));
    match vm.run() {
        VMResult::Ok(Value::Status(-37)) => {}
        other => panic!("status -37 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_309_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(309)));
    match vm.run() {
        VMResult::Ok(Value::Status(309)) => {}
        other => panic!("exit code 309 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_fifty_three_end_incrementing_from_nine() {
    let mut b = ChunkBuilder::new();
    for _ in 0..53 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(IncrementingSubshellHost { next: 9 }));
    match vm.run() {
        VMResult::Ok(Value::Status(61)) => {}
        other => panic!("fifty-third subshell_end(Some(61)) MUST win, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_310_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(310)));
    match vm.run() {
        VMResult::Ok(Value::Status(310)) => {}
        other => panic!("exit code 310 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_status_minus_thirty_eight_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(-38)));
    match vm.run() {
        VMResult::Ok(Value::Status(-38)) => {}
        other => panic!("status -38 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_311_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(311)));
    match vm.run() {
        VMResult::Ok(Value::Status(311)) => {}
        other => panic!("exit code 311 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_status_minus_thirty_nine_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(-39)));
    match vm.run() {
        VMResult::Ok(Value::Status(-39)) => {}
        other => panic!("status -39 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_312_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(312)));
    match vm.run() {
        VMResult::Ok(Value::Status(312)) => {}
        other => panic!("exit code 312 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_status_minus_forty_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(-40)));
    match vm.run() {
        VMResult::Ok(Value::Status(-40)) => {}
        other => panic!("status -40 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_313_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(313)));
    match vm.run() {
        VMResult::Ok(Value::Status(313)) => {}
        other => panic!("exit code 313 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_status_minus_forty_one_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(-41)));
    match vm.run() {
        VMResult::Ok(Value::Status(-41)) => {}
        other => panic!("status -41 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_314_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(314)));
    match vm.run() {
        VMResult::Ok(Value::Status(314)) => {}
        other => panic!("exit code 314 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_seventeenth_some_host_applies_on_seventeenth_end() {
    let mut b = ChunkBuilder::new();
    for _ in 0..17 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    struct SeventeenthSomeHost {
        calls: u32,
    }
    impl ShellHost for SeventeenthSomeHost {
        fn subshell_end(&mut self) -> Option<i32> {
            self.calls += 1;
            if self.calls == 17 {
                Some(315)
            } else {
                None
            }
        }
    }
    let mut vm = VM::new(b.build());
    vm.last_status = 1;
    vm.set_shell_host(Box::new(SeventeenthSomeHost { calls: 0 }));
    match vm.run() {
        VMResult::Ok(Value::Status(315)) => {}
        other => panic!("seventeenth subshell_end(Some(315)) MUST apply, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_315_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(315)));
    match vm.run() {
        VMResult::Ok(Value::Status(315)) => {}
        other => panic!("exit code 315 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_eighteenth_some_host_applies_on_eighteenth_end() {
    let mut b = ChunkBuilder::new();
    for _ in 0..18 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    struct EighteenthSomeHost {
        calls: u32,
    }
    impl ShellHost for EighteenthSomeHost {
        fn subshell_end(&mut self) -> Option<i32> {
            self.calls += 1;
            if self.calls == 18 {
                Some(316)
            } else {
                None
            }
        }
    }
    let mut vm = VM::new(b.build());
    vm.last_status = 3;
    vm.set_shell_host(Box::new(EighteenthSomeHost { calls: 0 }));
    match vm.run() {
        VMResult::Ok(Value::Status(316)) => {}
        other => panic!("eighteenth subshell_end(Some(316)) MUST apply, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_316_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(316)));
    match vm.run() {
        VMResult::Ok(Value::Status(316)) => {}
        other => panic!("exit code 316 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_mul_after_subshell_end() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::LoadInt(6), 1);
    b.emit(Op::LoadInt(7), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::Mul, 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(317)));
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 317);
}

#[test]
fn subshell_end_exit_code_317_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(317)));
    match vm.run() {
        VMResult::Ok(Value::Status(317)) => {}
        other => panic!("exit code 317 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_after_subshell_getstatus_blocks_second_getstatus() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(318)));
    vm.register_builtin(100, builtin_request_halt);
    match vm.run() {
        VMResult::Ok(Value::Int(0)) => {}
        other => panic!("second GetStatus MUST NOT run, got {:?}", other),
    }
    assert_eq!(vm.last_status, 318);
}

#[test]
fn subshell_end_exit_code_318_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(318)));
    match vm.run() {
        VMResult::Ok(Value::Status(318)) => {}
        other => panic!("exit code 318 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_halt_reset_subshell_end_seventh_run_applies() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    SUBSHELL_END_CALLS.store(0, Ordering::SeqCst);
    let sub = {
        let mut b = ChunkBuilder::new();
        b.emit(Op::SubshellEnd, 1);
        b.emit(Op::CallBuiltin(100, 0), 1);
        b.build()
    };
    let end_only = {
        let mut b = ChunkBuilder::new();
        b.emit(Op::SubshellEnd, 1);
        b.build()
    };
    let mut vm = VM::new(sub.clone());
    vm.set_shell_host(Box::new(CountingSubshellEndHost(319)));
    vm.register_builtin(100, builtin_request_halt);
    let _ = vm.run();
    vm.reset(sub.clone());
    let _ = vm.run();
    for _ in 0..4 {
        vm.reset(end_only.clone());
        let _ = vm.run();
    }
    vm.reset(end_only);
    let _ = vm.run();
    assert_eq!(SUBSHELL_END_CALLS.load(Ordering::SeqCst), 7);
    assert_eq!(vm.last_status, 319);
}

#[test]
fn subshell_end_exit_code_319_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(319)));
    match vm.run() {
        VMResult::Ok(Value::Status(319)) => {}
        other => panic!("exit code 319 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_callee_one_arg_before_caller_halt() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    let name = b.add_name("take1");
    b.emit(Op::LoadInt(9), 1);
    b.emit(Op::Call(name, 1), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let skip = b.emit(Op::Jump(0), 1);
    let entry = b.current_pos();
    b.add_sub_entry(name, entry);
    b.emit(Op::ReturnValue, 1);
    b.patch_jump(skip, b.current_pos());
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    match vm.run() {
        VMResult::Ok(Value::Int(0)) => {}
        other => panic!("halt MUST be stack top after one-arg callee, got {:?}", other),
    }
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_320_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(320)));
    match vm.run() {
        VMResult::Ok(Value::Status(320)) => {}
        other => panic!("exit code 320 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_none_none_some_none_some_host_pattern() {
    let mut b = ChunkBuilder::new();
    for _ in 0..5 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    struct NoneNoneSomeNoneSomeHost {
        calls: u32,
    }
    impl ShellHost for NoneNoneSomeNoneSomeHost {
        fn subshell_end(&mut self) -> Option<i32> {
            self.calls += 1;
            match self.calls {
                3 => Some(321),
                5 => Some(322),
                _ => None,
            }
        }
    }
    let mut vm = VM::new(b.build());
    vm.last_status = 10;
    vm.set_shell_host(Box::new(NoneNoneSomeNoneSomeHost { calls: 0 }));
    match vm.run() {
        VMResult::Ok(Value::Status(322)) => {}
        other => panic!("fifth end Some(322) MUST beat third Some(321), got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_321_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(321)));
    match vm.run() {
        VMResult::Ok(Value::Status(321)) => {}
        other => panic!("exit code 321 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_jump_if_false_after_subshell_end() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::LoadInt(4), 1);
    b.emit(Op::LoadInt(4), 1);
    b.emit(Op::NumEq, 1);
    let jmp = b.emit(Op::JumpIfFalse(0), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.patch_jump(jmp, b.current_pos());
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(322)));
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 322);
}

#[test]
fn subshell_end_exit_code_322_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(322)));
    match vm.run() {
        VMResult::Ok(Value::Status(322)) => {}
        other => panic!("exit code 322 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_subshell_end_then_pipeline_end_getstatus_reads_pipeline() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::PipelineEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(PipelineThenSubshellHost));
    match vm.run() {
        VMResult::Ok(Value::Status(3)) => {}
        other => panic!("GetStatus MUST read pipeline_end(3) after subshell Some(9), got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_323_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(323)));
    match vm.run() {
        VMResult::Ok(Value::Status(323)) => {}
        other => panic!("exit code 323 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_forty_four_consecutive_runs_without_reset_still_blocked() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    for _ in 0..44 {
        let _ = vm.run();
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_324_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(324)));
    match vm.run() {
        VMResult::Ok(Value::Status(324)) => {}
        other => panic!("exit code 324 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_min_then_zero_second_end_overwrites() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    struct MinThenZeroHost {
        calls: u32,
    }
    impl ShellHost for MinThenZeroHost {
        fn subshell_end(&mut self) -> Option<i32> {
            self.calls += 1;
            if self.calls == 1 {
                Some(i32::MIN)
            } else {
                Some(0)
            }
        }
    }
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(MinThenZeroHost { calls: 0 }));
    match vm.run() {
        VMResult::Ok(Value::Status(0)) => {}
        other => panic!("second subshell_end(Some(0)) MUST overwrite MIN, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_325_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(325)));
    match vm.run() {
        VMResult::Ok(Value::Status(325)) => {}
        other => panic!("exit code 325 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_load_int_pair_after_subshell_end() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::LoadInt(5), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::LoadInt(6), 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(326)));
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 326);
}

#[test]
fn subshell_end_exit_code_326_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(326)));
    match vm.run() {
        VMResult::Ok(Value::Status(326)) => {}
        other => panic!("exit code 326 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_sub_after_subshell_end() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::LoadInt(9), 1);
    b.emit(Op::LoadInt(4), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::Sub, 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(327)));
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 327);
}

#[test]
fn subshell_end_exit_code_327_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(327)));
    match vm.run() {
        VMResult::Ok(Value::Status(327)) => {}
        other => panic!("exit code 327 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_fifty_four_end_incrementing_from_five() {
    let mut b = ChunkBuilder::new();
    for _ in 0..54 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(IncrementingSubshellHost { next: 5 }));
    match vm.run() {
        VMResult::Ok(Value::Status(58)) => {}
        other => panic!("fifty-fourth subshell_end(Some(58)) MUST win, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_328_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(328)));
    match vm.run() {
        VMResult::Ok(Value::Status(328)) => {}
        other => panic!("exit code 328 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_rot_after_subshell_end() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::LoadInt(2), 1);
    b.emit(Op::LoadInt(3), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::Rot, 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(329)));
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 329);
}

#[test]
fn subshell_end_exit_code_329_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(329)));
    match vm.run() {
        VMResult::Ok(Value::Status(329)) => {}
        other => panic!("exit code 329 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_nineteenth_some_host_applies_on_nineteenth_end() {
    let mut b = ChunkBuilder::new();
    for _ in 0..19 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    struct NineteenthSomeHost {
        calls: u32,
    }
    impl ShellHost for NineteenthSomeHost {
        fn subshell_end(&mut self) -> Option<i32> {
            self.calls += 1;
            if self.calls == 19 {
                Some(330)
            } else {
                None
            }
        }
    }
    let mut vm = VM::new(b.build());
    vm.last_status = 2;
    vm.set_shell_host(Box::new(NineteenthSomeHost { calls: 0 }));
    match vm.run() {
        VMResult::Ok(Value::Status(330)) => {}
        other => panic!("nineteenth subshell_end(Some(330)) MUST apply, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_330_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(330)));
    match vm.run() {
        VMResult::Ok(Value::Status(330)) => {}
        other => panic!("exit code 330 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_inc_after_subshell_end() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::LoadInt(10), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::Inc, 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(331)));
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 331);
}

#[test]
fn subshell_end_exit_code_331_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(331)));
    match vm.run() {
        VMResult::Ok(Value::Status(331)) => {}
        other => panic!("exit code 331 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_forty_five_consecutive_runs_without_reset_still_blocked() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    for _ in 0..45 {
        let _ = vm.run();
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_332_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(332)));
    match vm.run() {
        VMResult::Ok(Value::Status(332)) => {}
        other => panic!("exit code 332 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_status_minus_forty_two_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(-42)));
    match vm.run() {
        VMResult::Ok(Value::Status(-42)) => {}
        other => panic!("status -42 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_dec_after_subshell_end() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::LoadInt(10), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::Dec, 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(333)));
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 333);
}

#[test]
fn subshell_end_exit_code_333_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(333)));
    match vm.run() {
        VMResult::Ok(Value::Status(333)) => {}
        other => panic!("exit code 333 MUST propagate, got {:?}", other),
    }
}
// ─── Pin tests batch M (handwritten) ──────────────────────────────────────

#[test]
fn subshell_end_exit_code_334_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(334)));
    match vm.run() {
        VMResult::Ok(Value::Status(334)) => {}
        other => panic!("exit code 334 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_forty_six_consecutive_runs_without_reset_still_blocked() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    for _ in 0..46 {
        let _ = vm.run();
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_335_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(335)));
    match vm.run() {
        VMResult::Ok(Value::Status(335)) => {}
        other => panic!("exit code 335 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_forty_seven_consecutive_runs_without_reset_still_blocked() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    for _ in 0..47 {
        let _ = vm.run();
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_336_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(336)));
    match vm.run() {
        VMResult::Ok(Value::Status(336)) => {}
        other => panic!("exit code 336 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_forty_eight_consecutive_runs_without_reset_still_blocked() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    for _ in 0..48 {
        let _ = vm.run();
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_337_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(337)));
    match vm.run() {
        VMResult::Ok(Value::Status(337)) => {}
        other => panic!("exit code 337 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_forty_nine_consecutive_runs_without_reset_still_blocked() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    for _ in 0..49 {
        let _ = vm.run();
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_338_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(338)));
    match vm.run() {
        VMResult::Ok(Value::Status(338)) => {}
        other => panic!("exit code 338 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_fifty_consecutive_runs_without_reset_still_blocked() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    for _ in 0..50 {
        let _ = vm.run();
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_339_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(339)));
    match vm.run() {
        VMResult::Ok(Value::Status(339)) => {}
        other => panic!("exit code 339 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_fifty_five_end_incrementing_from_six() {
    let mut b = ChunkBuilder::new();
    for _ in 0..55 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(IncrementingSubshellHost { next: 6 }));
    match vm.run() {
        VMResult::Ok(Value::Status(60)) => {}
        other => panic!("fifty-fifth subshell_end(Some(60)) MUST win, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_340_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(340)));
    match vm.run() {
        VMResult::Ok(Value::Status(340)) => {}
        other => panic!("exit code 340 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_status_minus_forty_three_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(-43)));
    match vm.run() {
        VMResult::Ok(Value::Status(-43)) => {}
        other => panic!("status -43 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_341_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(341)));
    match vm.run() {
        VMResult::Ok(Value::Status(341)) => {}
        other => panic!("exit code 341 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_fifty_six_end_incrementing_from_eleven() {
    let mut b = ChunkBuilder::new();
    for _ in 0..56 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(IncrementingSubshellHost { next: 11 }));
    match vm.run() {
        VMResult::Ok(Value::Status(66)) => {}
        other => panic!("fifty-sixth subshell_end(Some(66)) MUST win, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_342_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(342)));
    match vm.run() {
        VMResult::Ok(Value::Status(342)) => {}
        other => panic!("exit code 342 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_status_minus_forty_four_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(-44)));
    match vm.run() {
        VMResult::Ok(Value::Status(-44)) => {}
        other => panic!("status -44 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_343_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(343)));
    match vm.run() {
        VMResult::Ok(Value::Status(343)) => {}
        other => panic!("exit code 343 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_status_minus_forty_five_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(-45)));
    match vm.run() {
        VMResult::Ok(Value::Status(-45)) => {}
        other => panic!("status -45 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_344_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(344)));
    match vm.run() {
        VMResult::Ok(Value::Status(344)) => {}
        other => panic!("exit code 344 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_status_minus_forty_six_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(-46)));
    match vm.run() {
        VMResult::Ok(Value::Status(-46)) => {}
        other => panic!("status -46 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_345_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(345)));
    match vm.run() {
        VMResult::Ok(Value::Status(345)) => {}
        other => panic!("exit code 345 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_status_minus_forty_seven_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(-47)));
    match vm.run() {
        VMResult::Ok(Value::Status(-47)) => {}
        other => panic!("status -47 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_346_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(346)));
    match vm.run() {
        VMResult::Ok(Value::Status(346)) => {}
        other => panic!("exit code 346 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_twentieth_some_host_applies_on_twentieth_end() {
    let mut b = ChunkBuilder::new();
    for _ in 0..20 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    struct TwentiethSomeHost {
        calls: u32,
    }
    impl ShellHost for TwentiethSomeHost {
        fn subshell_end(&mut self) -> Option<i32> {
            self.calls += 1;
            if self.calls == 20 {
                Some(347)
            } else {
                None
            }
        }
    }
    let mut vm = VM::new(b.build());
    vm.last_status = 7;
    vm.set_shell_host(Box::new(TwentiethSomeHost { calls: 0 }));
    match vm.run() {
        VMResult::Ok(Value::Status(347)) => {}
        other => panic!("twentieth subshell_end(Some(347)) MUST apply, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_347_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(347)));
    match vm.run() {
        VMResult::Ok(Value::Status(347)) => {}
        other => panic!("exit code 347 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_concat_after_subshell_end() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    let a = b.add_constant(Value::str("foo"));
    let bidx = b.add_constant(Value::str("bar"));
    b.emit(Op::LoadConst(a), 1);
    b.emit(Op::LoadConst(bidx), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::Concat, 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(348)));
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 348);
}

#[test]
fn subshell_end_exit_code_348_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(348)));
    match vm.run() {
        VMResult::Ok(Value::Status(348)) => {}
        other => panic!("exit code 348 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_string_repeat_after_subshell_end() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    let s = b.add_constant(Value::str("x"));
    b.emit(Op::LoadConst(s), 1);
    b.emit(Op::LoadInt(2), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::StringRepeat, 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(349)));
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 349);
}

#[test]
fn subshell_end_exit_code_349_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(349)));
    match vm.run() {
        VMResult::Ok(Value::Status(349)) => {}
        other => panic!("exit code 349 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_halt_reset_subshell_end_eighth_run_applies() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    SUBSHELL_END_CALLS.store(0, Ordering::SeqCst);
    let sub = {
        let mut b = ChunkBuilder::new();
        b.emit(Op::SubshellEnd, 1);
        b.emit(Op::CallBuiltin(100, 0), 1);
        b.build()
    };
    let end_only = {
        let mut b = ChunkBuilder::new();
        b.emit(Op::SubshellEnd, 1);
        b.build()
    };
    let mut vm = VM::new(sub.clone());
    vm.set_shell_host(Box::new(CountingSubshellEndHost(350)));
    vm.register_builtin(100, builtin_request_halt);
    let _ = vm.run();
    vm.reset(sub);
    let _ = vm.run();
    for _ in 0..5 {
        vm.reset(end_only.clone());
        let _ = vm.run();
    }
    vm.reset(end_only);
    let _ = vm.run();
    assert_eq!(SUBSHELL_END_CALLS.load(Ordering::SeqCst), 8);
    assert_eq!(vm.last_status, 350);
}

#[test]
fn subshell_end_exit_code_350_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(350)));
    match vm.run() {
        VMResult::Ok(Value::Status(350)) => {}
        other => panic!("exit code 350 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_load_float_after_subshell_end_status351() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::LoadFloat(1.41), 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(351)));
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 351);
}

#[test]
fn subshell_end_exit_code_351_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(351)));
    match vm.run() {
        VMResult::Ok(Value::Status(351)) => {}
        other => panic!("exit code 351 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_six_begins_three_ends_last_end_status_wins() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    SUBSHELL_BEGIN_CALLS.store(0, Ordering::SeqCst);
    SUBSHELL_END_CALLS.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    for _ in 0..6 {
        b.emit(Op::SubshellBegin, 1);
    }
    for _ in 0..3 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    struct CountingBeginEndHost352;
    impl ShellHost for CountingBeginEndHost352 {
        fn subshell_begin(&mut self) {
            SUBSHELL_BEGIN_CALLS.fetch_add(1, Ordering::SeqCst);
        }
        fn subshell_end(&mut self) -> Option<i32> {
            SUBSHELL_END_CALLS.fetch_add(1, Ordering::SeqCst);
            Some(352)
        }
    }
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(CountingBeginEndHost352));
    match vm.run() {
        VMResult::Ok(Value::Status(352)) => {}
        other => panic!("third subshell_end(Some(352)) MUST win, got {:?}", other),
    }
    assert_eq!(SUBSHELL_BEGIN_CALLS.load(Ordering::SeqCst), 6);
    assert_eq!(SUBSHELL_END_CALLS.load(Ordering::SeqCst), 3);
}

#[test]
fn subshell_end_exit_code_352_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(352)));
    match vm.run() {
        VMResult::Ok(Value::Status(352)) => {}
        other => panic!("exit code 352 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_str_ne_on_consts_after_subshell_end() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    let a = b.add_constant(Value::str("left"));
    let bidx = b.add_constant(Value::str("right"));
    b.emit(Op::LoadConst(a), 1);
    b.emit(Op::LoadConst(bidx), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::StrNe, 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(353)));
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 353);
}

#[test]
fn subshell_end_exit_code_353_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(353)));
    match vm.run() {
        VMResult::Ok(Value::Status(353)) => {}
        other => panic!("exit code 353 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_five_reset_runs_same_host_same_status() {
    let chunk = {
        let mut b = ChunkBuilder::new();
        b.emit(Op::SubshellEnd, 1);
        b.emit(Op::GetStatus, 1);
        b.build()
    };
    let mut vm = VM::new(chunk.clone());
    vm.set_shell_host(Box::new(StatusReturningHost(354)));
    for _ in 0..5 {
        match vm.run() {
            VMResult::Ok(Value::Status(354)) => {}
            other => panic!("reset run MUST return Status(354), got {:?}", other),
        }
        vm.reset(chunk.clone());
    }
}

#[test]
fn subshell_end_exit_code_354_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(354)));
    match vm.run() {
        VMResult::Ok(Value::Status(354)) => {}
        other => panic!("exit code 354 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_add_after_subshell_end_status355() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::LoadInt(3), 1);
    b.emit(Op::LoadInt(4), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::Add, 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(355)));
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 355);
}

#[test]
fn subshell_end_exit_code_355_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(355)));
    match vm.run() {
        VMResult::Ok(Value::Status(355)) => {}
        other => panic!("exit code 355 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_alternating_status_tenth_none_keeps_ninth() {
    let mut b = ChunkBuilder::new();
    for _ in 0..10 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(AlternatingStatusHost { calls: 0 }));
    match vm.run() {
        VMResult::Ok(Value::Status(9)) => {}
        other => panic!("AlternatingStatusHost tenth None MUST keep Some(9), got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_356_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(356)));
    match vm.run() {
        VMResult::Ok(Value::Status(356)) => {}
        other => panic!("exit code 356 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_string_len_after_subshell_end() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    let s = b.add_constant(Value::str("halt"));
    b.emit(Op::LoadConst(s), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::StringLen, 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(357)));
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 357);
}

#[test]
fn subshell_end_exit_code_357_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(357)));
    match vm.run() {
        VMResult::Ok(Value::Status(357)) => {}
        other => panic!("exit code 357 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_fifty_seven_end_incrementing_from_three() {
    let mut b = ChunkBuilder::new();
    for _ in 0..57 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(IncrementingSubshellHost { next: 3 }));
    match vm.run() {
        VMResult::Ok(Value::Status(59)) => {}
        other => panic!("fifty-seventh subshell_end(Some(59)) MUST win, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_358_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(358)));
    match vm.run() {
        VMResult::Ok(Value::Status(358)) => {}
        other => panic!("exit code 358 MUST propagate, got {:?}", other),
    }
}
// ─── Pin tests batch N (handwritten) ──────────────────────────────────────

#[test]
fn subshell_end_exit_code_359_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(359)));
    match vm.run() {
        VMResult::Ok(Value::Status(359)) => {}
        other => panic!("exit code 359 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_fifty_one_consecutive_runs_without_reset_still_blocked() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    for _ in 0..51 {
        let _ = vm.run();
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_360_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(360)));
    match vm.run() {
        VMResult::Ok(Value::Status(360)) => {}
        other => panic!("exit code 360 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_fifty_two_consecutive_runs_without_reset_still_blocked() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    for _ in 0..52 {
        let _ = vm.run();
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_361_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(361)));
    match vm.run() {
        VMResult::Ok(Value::Status(361)) => {}
        other => panic!("exit code 361 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_fifty_three_consecutive_runs_without_reset_still_blocked() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    for _ in 0..53 {
        let _ = vm.run();
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_362_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(362)));
    match vm.run() {
        VMResult::Ok(Value::Status(362)) => {}
        other => panic!("exit code 362 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_fifty_eight_end_incrementing_from_eight() {
    let mut b = ChunkBuilder::new();
    for _ in 0..58 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(IncrementingSubshellHost { next: 8 }));
    match vm.run() {
        VMResult::Ok(Value::Status(65)) => {}
        other => panic!("fifty-eighth subshell_end(Some(65)) MUST win, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_363_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(363)));
    match vm.run() {
        VMResult::Ok(Value::Status(363)) => {}
        other => panic!("exit code 363 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_status_minus_forty_eight_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(-48)));
    match vm.run() {
        VMResult::Ok(Value::Status(-48)) => {}
        other => panic!("status -48 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_364_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(364)));
    match vm.run() {
        VMResult::Ok(Value::Status(364)) => {}
        other => panic!("exit code 364 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_status_minus_forty_nine_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(-49)));
    match vm.run() {
        VMResult::Ok(Value::Status(-49)) => {}
        other => panic!("status -49 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_365_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(365)));
    match vm.run() {
        VMResult::Ok(Value::Status(365)) => {}
        other => panic!("exit code 365 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_status_minus_fifty_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(-50)));
    match vm.run() {
        VMResult::Ok(Value::Status(-50)) => {}
        other => panic!("status -50 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_366_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(366)));
    match vm.run() {
        VMResult::Ok(Value::Status(366)) => {}
        other => panic!("exit code 366 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_twenty_first_some_host_applies_on_twenty_first_end() {
    let mut b = ChunkBuilder::new();
    for _ in 0..21 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    struct TwentyFirstSomeHost {
        calls: u32,
    }
    impl ShellHost for TwentyFirstSomeHost {
        fn subshell_end(&mut self) -> Option<i32> {
            self.calls += 1;
            if self.calls == 21 {
                Some(367)
            } else {
                None
            }
        }
    }
    let mut vm = VM::new(b.build());
    vm.last_status = 4;
    vm.set_shell_host(Box::new(TwentyFirstSomeHost { calls: 0 }));
    match vm.run() {
        VMResult::Ok(Value::Status(367)) => {}
        other => panic!("twenty-first subshell_end(Some(367)) MUST apply, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_367_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(367)));
    match vm.run() {
        VMResult::Ok(Value::Status(367)) => {}
        other => panic!("exit code 367 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_twenty_second_some_host_applies_on_twenty_second_end() {
    let mut b = ChunkBuilder::new();
    for _ in 0..22 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    struct TwentySecondSomeHost {
        calls: u32,
    }
    impl ShellHost for TwentySecondSomeHost {
        fn subshell_end(&mut self) -> Option<i32> {
            self.calls += 1;
            if self.calls == 22 {
                Some(368)
            } else {
                None
            }
        }
    }
    let mut vm = VM::new(b.build());
    vm.last_status = 1;
    vm.set_shell_host(Box::new(TwentySecondSomeHost { calls: 0 }));
    match vm.run() {
        VMResult::Ok(Value::Status(368)) => {}
        other => panic!("twenty-second subshell_end(Some(368)) MUST apply, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_368_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(368)));
    match vm.run() {
        VMResult::Ok(Value::Status(368)) => {}
        other => panic!("exit code 368 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_make_hash_after_subshell_end() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    let k = b.add_constant(Value::str("a"));
    let v = b.add_constant(Value::str("b"));
    b.emit(Op::LoadConst(k), 1);
    b.emit(Op::LoadConst(v), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::MakeHash(1), 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(369)));
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 369);
}

#[test]
fn subshell_end_exit_code_369_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(369)));
    match vm.run() {
        VMResult::Ok(Value::Status(369)) => {}
        other => panic!("exit code 369 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_spaceship_after_subshell_end_status370() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::LoadInt(5), 1);
    b.emit(Op::LoadInt(5), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::Spaceship, 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(370)));
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 370);
}

#[test]
fn subshell_end_exit_code_370_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(370)));
    match vm.run() {
        VMResult::Ok(Value::Status(370)) => {}
        other => panic!("exit code 370 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_halt_reset_subshell_end_ninth_run_applies() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    SUBSHELL_END_CALLS.store(0, Ordering::SeqCst);
    let sub = {
        let mut b = ChunkBuilder::new();
        b.emit(Op::SubshellEnd, 1);
        b.emit(Op::CallBuiltin(100, 0), 1);
        b.build()
    };
    let end_only = {
        let mut b = ChunkBuilder::new();
        b.emit(Op::SubshellEnd, 1);
        b.build()
    };
    let mut vm = VM::new(sub.clone());
    vm.set_shell_host(Box::new(CountingSubshellEndHost(371)));
    vm.register_builtin(100, builtin_request_halt);
    let _ = vm.run();
    vm.reset(sub);
    let _ = vm.run();
    for _ in 0..6 {
        vm.reset(end_only.clone());
        let _ = vm.run();
    }
    vm.reset(end_only);
    let _ = vm.run();
    assert_eq!(SUBSHELL_END_CALLS.load(Ordering::SeqCst), 9);
    assert_eq!(vm.last_status, 371);
}

#[test]
fn subshell_end_exit_code_371_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(371)));
    match vm.run() {
        VMResult::Ok(Value::Status(371)) => {}
        other => panic!("exit code 371 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_load_true_after_subshell_end() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::LoadTrue, 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(372)));
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 372);
}

#[test]
fn subshell_end_exit_code_372_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(372)));
    match vm.run() {
        VMResult::Ok(Value::Status(372)) => {}
        other => panic!("exit code 372 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_zero_then_negative_second_end_overwrites() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    struct ZeroThenNegativeHost {
        calls: u32,
    }
    impl ShellHost for ZeroThenNegativeHost {
        fn subshell_end(&mut self) -> Option<i32> {
            self.calls += 1;
            if self.calls == 1 {
                Some(0)
            } else {
                Some(-51)
            }
        }
    }
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(ZeroThenNegativeHost { calls: 0 }));
    match vm.run() {
        VMResult::Ok(Value::Status(-51)) => {}
        other => panic!("second subshell_end(Some(-51)) MUST overwrite zero, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_373_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(373)));
    match vm.run() {
        VMResult::Ok(Value::Status(373)) => {}
        other => panic!("exit code 373 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_fifty_four_consecutive_runs_without_reset_still_blocked() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    for _ in 0..54 {
        let _ = vm.run();
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_374_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(374)));
    match vm.run() {
        VMResult::Ok(Value::Status(374)) => {}
        other => panic!("exit code 374 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_fifty_nine_end_incrementing_from_twelve() {
    let mut b = ChunkBuilder::new();
    for _ in 0..59 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(IncrementingSubshellHost { next: 12 }));
    match vm.run() {
        VMResult::Ok(Value::Status(70)) => {}
        other => panic!("fifty-ninth subshell_end(Some(70)) MUST win, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_375_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(375)));
    match vm.run() {
        VMResult::Ok(Value::Status(375)) => {}
        other => panic!("exit code 375 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_between_third_subshell_end_and_fourth_subshell_begin() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    SUBSHELL_BEGIN_CALLS.store(0, Ordering::SeqCst);
    SUBSHELL_END_CALLS.store(0, Ordering::SeqCst);
    struct CountingBeginEndHost376;
    impl ShellHost for CountingBeginEndHost376 {
        fn subshell_begin(&mut self) {
            SUBSHELL_BEGIN_CALLS.fetch_add(1, Ordering::SeqCst);
        }
        fn subshell_end(&mut self) -> Option<i32> {
            SUBSHELL_END_CALLS.fetch_add(1, Ordering::SeqCst);
            Some(376)
        }
    }
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(CountingBeginEndHost376));
    vm.register_builtin(100, builtin_request_halt);
    let _ = vm.run();
    assert_eq!(SUBSHELL_END_CALLS.load(Ordering::SeqCst), 3);
    assert_eq!(SUBSHELL_BEGIN_CALLS.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 376);
}

#[test]
fn subshell_end_exit_code_376_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(376)));
    match vm.run() {
        VMResult::Ok(Value::Status(376)) => {}
        other => panic!("exit code 376 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_str_eq_on_consts_after_subshell_end() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    let a = b.add_constant(Value::str("eq"));
    let bidx = b.add_constant(Value::str("eq"));
    b.emit(Op::LoadConst(a), 1);
    b.emit(Op::LoadConst(bidx), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::StrEq, 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(377)));
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 377);
}

#[test]
fn subshell_end_exit_code_377_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(377)));
    match vm.run() {
        VMResult::Ok(Value::Status(377)) => {}
        other => panic!("exit code 377 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_six_reset_runs_same_host_same_status() {
    let chunk = {
        let mut b = ChunkBuilder::new();
        b.emit(Op::SubshellEnd, 1);
        b.emit(Op::GetStatus, 1);
        b.build()
    };
    let mut vm = VM::new(chunk.clone());
    vm.set_shell_host(Box::new(StatusReturningHost(378)));
    for _ in 0..6 {
        match vm.run() {
            VMResult::Ok(Value::Status(378)) => {}
            other => panic!("reset run MUST return Status(378), got {:?}", other),
        }
        vm.reset(chunk.clone());
    }
}

#[test]
fn subshell_end_exit_code_378_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(378)));
    match vm.run() {
        VMResult::Ok(Value::Status(378)) => {}
        other => panic!("exit code 378 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_pop_after_subshell_end() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::LoadInt(99), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::Pop, 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(379)));
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 379);
}

#[test]
fn subshell_end_exit_code_379_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(379)));
    match vm.run() {
        VMResult::Ok(Value::Status(379)) => {}
        other => panic!("exit code 379 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_status_minus_fifty_one_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(-51)));
    match vm.run() {
        VMResult::Ok(Value::Status(-51)) => {}
        other => panic!("status -51 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_380_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(380)));
    match vm.run() {
        VMResult::Ok(Value::Status(380)) => {}
        other => panic!("exit code 380 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_fifty_five_consecutive_runs_without_reset_still_blocked() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    for _ in 0..55 {
        let _ = vm.run();
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_381_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(381)));
    match vm.run() {
        VMResult::Ok(Value::Status(381)) => {}
        other => panic!("exit code 381 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_sixty_end_incrementing_from_one() {
    let mut b = ChunkBuilder::new();
    for _ in 0..60 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(IncrementingSubshellHost { next: 1 }));
    match vm.run() {
        VMResult::Ok(Value::Status(60)) => {}
        other => panic!("sixtieth subshell_end(Some(60)) MUST win, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_382_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(382)));
    match vm.run() {
        VMResult::Ok(Value::Status(382)) => {}
        other => panic!("exit code 382 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_swap_after_subshell_end_status383() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::LoadInt(2), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::Swap, 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(383)));
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 383);
}

#[test]
fn subshell_end_exit_code_383_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(383)));
    match vm.run() {
        VMResult::Ok(Value::Status(383)) => {}
        other => panic!("exit code 383 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_status_minus_fifty_two_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(-52)));
    match vm.run() {
        VMResult::Ok(Value::Status(-52)) => {}
        other => panic!("status -52 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_dup_after_subshell_end() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::LoadInt(42), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::Dup, 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(384)));
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 384);
}

#[test]
fn subshell_end_exit_code_384_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(384)));
    match vm.run() {
        VMResult::Ok(Value::Status(384)) => {}
        other => panic!("exit code 384 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_twenty_third_some_host_applies_on_twenty_third_end() {
    let mut b = ChunkBuilder::new();
    for _ in 0..23 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    struct TwentyThirdSomeHost {
        calls: u32,
    }
    impl ShellHost for TwentyThirdSomeHost {
        fn subshell_end(&mut self) -> Option<i32> {
            self.calls += 1;
            if self.calls == 23 {
                Some(385)
            } else {
                None
            }
        }
    }
    let mut vm = VM::new(b.build());
    vm.last_status = 9;
    vm.set_shell_host(Box::new(TwentyThirdSomeHost { calls: 0 }));
    match vm.run() {
        VMResult::Ok(Value::Status(385)) => {}
        other => panic!("twenty-third subshell_end(Some(385)) MUST apply, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_385_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(385)));
    match vm.run() {
        VMResult::Ok(Value::Status(385)) => {}
        other => panic!("exit code 385 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_fifty_six_consecutive_runs_without_reset_still_blocked() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    for _ in 0..56 {
        let _ = vm.run();
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_386_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(386)));
    match vm.run() {
        VMResult::Ok(Value::Status(386)) => {}
        other => panic!("exit code 386 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_halt_reset_getstatus_on_fifth_reset_run() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    let halt = {
        let mut b = ChunkBuilder::new();
        b.emit(Op::CallBuiltin(100, 0), 1);
        b.build()
    };
    let sub = {
        let mut b = ChunkBuilder::new();
        b.emit(Op::SubshellEnd, 1);
        b.emit(Op::GetStatus, 1);
        b.build()
    };
    let mut vm = VM::new(halt.clone());
    vm.set_shell_host(Box::new(StatusReturningHost(387)));
    vm.register_builtin(100, builtin_request_halt);
    let _ = vm.run();
    vm.reset(halt);
    let _ = vm.run();
    vm.reset(sub.clone());
    let _ = vm.run();
    vm.reset(sub.clone());
    let _ = vm.run();
    vm.reset(sub);
    match vm.run() {
        VMResult::Ok(Value::Status(387)) => {}
        other => panic!("fifth reset run MUST return Status(387), got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_387_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(387)));
    match vm.run() {
        VMResult::Ok(Value::Status(387)) => {}
        other => panic!("exit code 387 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_load_undef_after_subshell_end_status388() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::LoadUndef, 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(388)));
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 388);
}

#[test]
fn subshell_end_exit_code_388_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(388)));
    match vm.run() {
        VMResult::Ok(Value::Status(388)) => {}
        other => panic!("exit code 388 MUST propagate, got {:?}", other),
    }
}
// ─── Pin tests batch O (handwritten) ──────────────────────────────────────

#[test]
fn subshell_end_exit_code_389_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(389)));
    match vm.run() {
        VMResult::Ok(Value::Status(389)) => {}
        other => panic!("exit code 389 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_fifty_seven_consecutive_runs_without_reset_still_blocked() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    for _ in 0..57 {
        let _ = vm.run();
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_390_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(390)));
    match vm.run() {
        VMResult::Ok(Value::Status(390)) => {}
        other => panic!("exit code 390 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_fifty_eight_consecutive_runs_without_reset_still_blocked() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    for _ in 0..58 {
        let _ = vm.run();
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_391_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(391)));
    match vm.run() {
        VMResult::Ok(Value::Status(391)) => {}
        other => panic!("exit code 391 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_fifty_nine_consecutive_runs_without_reset_still_blocked() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    for _ in 0..59 {
        let _ = vm.run();
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_392_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(392)));
    match vm.run() {
        VMResult::Ok(Value::Status(392)) => {}
        other => panic!("exit code 392 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_sixty_consecutive_runs_without_reset_still_blocked() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    for _ in 0..60 {
        let _ = vm.run();
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_393_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(393)));
    match vm.run() {
        VMResult::Ok(Value::Status(393)) => {}
        other => panic!("exit code 393 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_sixty_one_consecutive_runs_without_reset_still_blocked() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    for _ in 0..61 {
        let _ = vm.run();
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_394_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(394)));
    match vm.run() {
        VMResult::Ok(Value::Status(394)) => {}
        other => panic!("exit code 394 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_sixty_one_end_incrementing_from_five() {
    let mut b = ChunkBuilder::new();
    for _ in 0..61 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(IncrementingSubshellHost { next: 5 }));
    match vm.run() {
        VMResult::Ok(Value::Status(65)) => {}
        other => panic!("sixty-first subshell_end(Some(65)) MUST win, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_395_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(395)));
    match vm.run() {
        VMResult::Ok(Value::Status(395)) => {}
        other => panic!("exit code 395 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_status_minus_fifty_three_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(-53)));
    match vm.run() {
        VMResult::Ok(Value::Status(-53)) => {}
        other => panic!("status -53 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_396_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(396)));
    match vm.run() {
        VMResult::Ok(Value::Status(396)) => {}
        other => panic!("exit code 396 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_sixty_two_end_incrementing_from_ten() {
    let mut b = ChunkBuilder::new();
    for _ in 0..62 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(IncrementingSubshellHost { next: 10 }));
    match vm.run() {
        VMResult::Ok(Value::Status(71)) => {}
        other => panic!("sixty-second subshell_end(Some(71)) MUST win, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_397_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(397)));
    match vm.run() {
        VMResult::Ok(Value::Status(397)) => {}
        other => panic!("exit code 397 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_status_minus_fifty_four_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(-54)));
    match vm.run() {
        VMResult::Ok(Value::Status(-54)) => {}
        other => panic!("status -54 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_398_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(398)));
    match vm.run() {
        VMResult::Ok(Value::Status(398)) => {}
        other => panic!("exit code 398 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_status_minus_fifty_five_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(-55)));
    match vm.run() {
        VMResult::Ok(Value::Status(-55)) => {}
        other => panic!("status -55 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_399_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(399)));
    match vm.run() {
        VMResult::Ok(Value::Status(399)) => {}
        other => panic!("exit code 399 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_status_minus_fifty_six_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(-56)));
    match vm.run() {
        VMResult::Ok(Value::Status(-56)) => {}
        other => panic!("status -56 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_400_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(400)));
    match vm.run() {
        VMResult::Ok(Value::Status(400)) => {}
        other => panic!("exit code 400 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_status_minus_fifty_seven_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(-57)));
    match vm.run() {
        VMResult::Ok(Value::Status(-57)) => {}
        other => panic!("status -57 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_401_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(401)));
    match vm.run() {
        VMResult::Ok(Value::Status(401)) => {}
        other => panic!("exit code 401 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_twenty_fourth_some_host_applies_on_twenty_fourth_end() {
    let mut b = ChunkBuilder::new();
    for _ in 0..24 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    struct TwentyFourthSomeHost {
        calls: u32,
    }
    impl ShellHost for TwentyFourthSomeHost {
        fn subshell_end(&mut self) -> Option<i32> {
            self.calls += 1;
            if self.calls == 24 {
                Some(402)
            } else {
                None
            }
        }
    }
    let mut vm = VM::new(b.build());
    vm.last_status = 6;
    vm.set_shell_host(Box::new(TwentyFourthSomeHost { calls: 0 }));
    match vm.run() {
        VMResult::Ok(Value::Status(402)) => {}
        other => panic!("twenty-fourth subshell_end(Some(402)) MUST apply, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_402_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(402)));
    match vm.run() {
        VMResult::Ok(Value::Status(402)) => {}
        other => panic!("exit code 402 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_halt_reset_subshell_end_tenth_run_applies() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    SUBSHELL_END_CALLS.store(0, Ordering::SeqCst);
    let sub = {
        let mut b = ChunkBuilder::new();
        b.emit(Op::SubshellEnd, 1);
        b.emit(Op::CallBuiltin(100, 0), 1);
        b.build()
    };
    let end_only = {
        let mut b = ChunkBuilder::new();
        b.emit(Op::SubshellEnd, 1);
        b.build()
    };
    let mut vm = VM::new(sub.clone());
    vm.set_shell_host(Box::new(CountingSubshellEndHost(403)));
    vm.register_builtin(100, builtin_request_halt);
    let _ = vm.run();
    vm.reset(sub);
    let _ = vm.run();
    for _ in 0..7 {
        vm.reset(end_only.clone());
        let _ = vm.run();
    }
    vm.reset(end_only);
    let _ = vm.run();
    assert_eq!(SUBSHELL_END_CALLS.load(Ordering::SeqCst), 10);
    assert_eq!(vm.last_status, 403);
}

#[test]
fn subshell_end_exit_code_403_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(403)));
    match vm.run() {
        VMResult::Ok(Value::Status(403)) => {}
        other => panic!("exit code 403 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_str_lt_on_consts_after_subshell_end_status404() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    let a = b.add_constant(Value::str("a"));
    let bidx = b.add_constant(Value::str("z"));
    b.emit(Op::LoadConst(a), 1);
    b.emit(Op::LoadConst(bidx), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::StrLt, 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(404)));
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 404);
}

#[test]
fn subshell_end_exit_code_404_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(404)));
    match vm.run() {
        VMResult::Ok(Value::Status(404)) => {}
        other => panic!("exit code 404 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_seven_begins_four_ends_last_end_status_wins() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    SUBSHELL_BEGIN_CALLS.store(0, Ordering::SeqCst);
    SUBSHELL_END_CALLS.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    for _ in 0..7 {
        b.emit(Op::SubshellBegin, 1);
    }
    for _ in 0..4 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    struct CountingBeginEndHost405;
    impl ShellHost for CountingBeginEndHost405 {
        fn subshell_begin(&mut self) {
            SUBSHELL_BEGIN_CALLS.fetch_add(1, Ordering::SeqCst);
        }
        fn subshell_end(&mut self) -> Option<i32> {
            SUBSHELL_END_CALLS.fetch_add(1, Ordering::SeqCst);
            Some(405)
        }
    }
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(CountingBeginEndHost405));
    match vm.run() {
        VMResult::Ok(Value::Status(405)) => {}
        other => panic!("fourth subshell_end(Some(405)) MUST win, got {:?}", other),
    }
    assert_eq!(SUBSHELL_BEGIN_CALLS.load(Ordering::SeqCst), 7);
    assert_eq!(SUBSHELL_END_CALLS.load(Ordering::SeqCst), 4);
}

#[test]
fn subshell_end_exit_code_405_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(405)));
    match vm.run() {
        VMResult::Ok(Value::Status(405)) => {}
        other => panic!("exit code 405 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_between_pipeline_end_and_subshell_begin() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    SUBSHELL_BEGIN_CALLS.store(0, Ordering::SeqCst);
    struct PipelineEndSubshellBeginHost406;
    impl ShellHost for PipelineEndSubshellBeginHost406 {
        fn pipeline_end(&mut self) -> i32 {
            0
        }
        fn subshell_begin(&mut self) {
            SUBSHELL_BEGIN_CALLS.fetch_add(1, Ordering::SeqCst);
        }
        fn subshell_end(&mut self) -> Option<i32> {
            Some(406)
        }
    }
    let mut b = ChunkBuilder::new();
    b.emit(Op::PipelineEnd, 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(PipelineEndSubshellBeginHost406));
    vm.register_builtin(100, builtin_request_halt);
    let _ = vm.run();
    assert_eq!(SUBSHELL_BEGIN_CALLS.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 0);
}

#[test]
fn subshell_end_exit_code_406_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(406)));
    match vm.run() {
        VMResult::Ok(Value::Status(406)) => {}
        other => panic!("exit code 406 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_seven_reset_runs_same_host_same_status() {
    let chunk = {
        let mut b = ChunkBuilder::new();
        b.emit(Op::SubshellEnd, 1);
        b.emit(Op::GetStatus, 1);
        b.build()
    };
    let mut vm = VM::new(chunk.clone());
    vm.set_shell_host(Box::new(StatusReturningHost(407)));
    for _ in 0..7 {
        match vm.run() {
            VMResult::Ok(Value::Status(407)) => {}
            other => panic!("reset run MUST return Status(407), got {:?}", other),
        }
        vm.reset(chunk.clone());
    }
}

#[test]
fn subshell_end_exit_code_407_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(407)));
    match vm.run() {
        VMResult::Ok(Value::Status(407)) => {}
        other => panic!("exit code 407 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_str_gt_on_consts_after_subshell_end_status408() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    let a = b.add_constant(Value::str("z"));
    let bidx = b.add_constant(Value::str("a"));
    b.emit(Op::LoadConst(a), 1);
    b.emit(Op::LoadConst(bidx), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::StrGt, 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(408)));
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 408);
}

#[test]
fn subshell_end_exit_code_408_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(408)));
    match vm.run() {
        VMResult::Ok(Value::Status(408)) => {}
        other => panic!("exit code 408 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_alternating_status_twelfth_none_keeps_ninth() {
    let mut b = ChunkBuilder::new();
    for _ in 0..12 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(AlternatingStatusHost { calls: 0 }));
    match vm.run() {
        VMResult::Ok(Value::Status(9)) => {}
        other => panic!("AlternatingStatusHost twelfth None MUST keep Some(9), got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_409_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(409)));
    match vm.run() {
        VMResult::Ok(Value::Status(409)) => {}
        other => panic!("exit code 409 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_pow_after_subshell_end_status410() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::LoadInt(2), 1);
    b.emit(Op::LoadInt(4), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::Pow, 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(410)));
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 410);
}

#[test]
fn subshell_end_exit_code_410_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(410)));
    match vm.run() {
        VMResult::Ok(Value::Status(410)) => {}
        other => panic!("exit code 410 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_sixty_three_end_incrementing_from_two() {
    let mut b = ChunkBuilder::new();
    for _ in 0..63 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(IncrementingSubshellHost { next: 2 }));
    match vm.run() {
        VMResult::Ok(Value::Status(64)) => {}
        other => panic!("sixty-third subshell_end(Some(64)) MUST win, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_411_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(411)));
    match vm.run() {
        VMResult::Ok(Value::Status(411)) => {}
        other => panic!("exit code 411 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_load_false_after_subshell_end_status412() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::LoadFalse, 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(412)));
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 412);
}

#[test]
fn subshell_end_exit_code_412_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(412)));
    match vm.run() {
        VMResult::Ok(Value::Status(412)) => {}
        other => panic!("exit code 412 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_rot_after_subshell_end_status413() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::LoadInt(2), 1);
    b.emit(Op::LoadInt(3), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::Rot, 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(413)));
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 413);
}

#[test]
fn subshell_end_exit_code_413_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(413)));
    match vm.run() {
        VMResult::Ok(Value::Status(413)) => {}
        other => panic!("exit code 413 MUST propagate, got {:?}", other),
    }
}
// ─── Pin tests batch P (handwritten) ──────────────────────────────────────

#[test]
fn subshell_end_exit_code_414_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(414)));
    match vm.run() {
        VMResult::Ok(Value::Status(414)) => {}
        other => panic!("exit code 414 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_sixty_two_consecutive_runs_without_reset_still_blocked() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    for _ in 0..62 {
        let _ = vm.run();
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_415_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(415)));
    match vm.run() {
        VMResult::Ok(Value::Status(415)) => {}
        other => panic!("exit code 415 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_sixty_three_consecutive_runs_without_reset_still_blocked() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    for _ in 0..63 {
        let _ = vm.run();
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_416_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(416)));
    match vm.run() {
        VMResult::Ok(Value::Status(416)) => {}
        other => panic!("exit code 416 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_sixty_four_consecutive_runs_without_reset_still_blocked() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    for _ in 0..64 {
        let _ = vm.run();
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_417_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(417)));
    match vm.run() {
        VMResult::Ok(Value::Status(417)) => {}
        other => panic!("exit code 417 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_sixty_five_consecutive_runs_without_reset_still_blocked() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    for _ in 0..65 {
        let _ = vm.run();
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_418_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(418)));
    match vm.run() {
        VMResult::Ok(Value::Status(418)) => {}
        other => panic!("exit code 418 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_sixty_six_consecutive_runs_without_reset_still_blocked() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    for _ in 0..66 {
        let _ = vm.run();
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_419_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(419)));
    match vm.run() {
        VMResult::Ok(Value::Status(419)) => {}
        other => panic!("exit code 419 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_sixty_four_end_incrementing_from_seven() {
    let mut b = ChunkBuilder::new();
    for _ in 0..64 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(IncrementingSubshellHost { next: 7 }));
    match vm.run() {
        VMResult::Ok(Value::Status(70)) => {}
        other => panic!("sixty-fourth subshell_end(Some(70)) MUST win, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_420_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(420)));
    match vm.run() {
        VMResult::Ok(Value::Status(420)) => {}
        other => panic!("exit code 420 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_status_minus_fifty_eight_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(-58)));
    match vm.run() {
        VMResult::Ok(Value::Status(-58)) => {}
        other => panic!("status -58 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_421_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(421)));
    match vm.run() {
        VMResult::Ok(Value::Status(421)) => {}
        other => panic!("exit code 421 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_status_minus_fifty_nine_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(-59)));
    match vm.run() {
        VMResult::Ok(Value::Status(-59)) => {}
        other => panic!("status -59 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_422_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(422)));
    match vm.run() {
        VMResult::Ok(Value::Status(422)) => {}
        other => panic!("exit code 422 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_status_minus_sixty_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(-60)));
    match vm.run() {
        VMResult::Ok(Value::Status(-60)) => {}
        other => panic!("status -60 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_423_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(423)));
    match vm.run() {
        VMResult::Ok(Value::Status(423)) => {}
        other => panic!("exit code 423 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_twenty_fifth_some_host_applies_on_twenty_fifth_end() {
    let mut b = ChunkBuilder::new();
    for _ in 0..25 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    struct TwentyFifthSomeHost {
        calls: u32,
    }
    impl ShellHost for TwentyFifthSomeHost {
        fn subshell_end(&mut self) -> Option<i32> {
            self.calls += 1;
            if self.calls == 25 {
                Some(424)
            } else {
                None
            }
        }
    }
    let mut vm = VM::new(b.build());
    vm.last_status = 3;
    vm.set_shell_host(Box::new(TwentyFifthSomeHost { calls: 0 }));
    match vm.run() {
        VMResult::Ok(Value::Status(424)) => {}
        other => panic!("twenty-fifth subshell_end(Some(424)) MUST apply, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_424_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(424)));
    match vm.run() {
        VMResult::Ok(Value::Status(424)) => {}
        other => panic!("exit code 424 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_twenty_sixth_some_host_applies_on_twenty_sixth_end() {
    let mut b = ChunkBuilder::new();
    for _ in 0..26 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    struct TwentySixthSomeHost {
        calls: u32,
    }
    impl ShellHost for TwentySixthSomeHost {
        fn subshell_end(&mut self) -> Option<i32> {
            self.calls += 1;
            if self.calls == 26 {
                Some(425)
            } else {
                None
            }
        }
    }
    let mut vm = VM::new(b.build());
    vm.last_status = 2;
    vm.set_shell_host(Box::new(TwentySixthSomeHost { calls: 0 }));
    match vm.run() {
        VMResult::Ok(Value::Status(425)) => {}
        other => panic!("twenty-sixth subshell_end(Some(425)) MUST apply, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_425_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(425)));
    match vm.run() {
        VMResult::Ok(Value::Status(425)) => {}
        other => panic!("exit code 425 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_halt_reset_subshell_end_eleventh_run_applies() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    SUBSHELL_END_CALLS.store(0, Ordering::SeqCst);
    let sub = {
        let mut b = ChunkBuilder::new();
        b.emit(Op::SubshellEnd, 1);
        b.emit(Op::CallBuiltin(100, 0), 1);
        b.build()
    };
    let end_only = {
        let mut b = ChunkBuilder::new();
        b.emit(Op::SubshellEnd, 1);
        b.build()
    };
    let mut vm = VM::new(sub.clone());
    vm.set_shell_host(Box::new(CountingSubshellEndHost(426)));
    vm.register_builtin(100, builtin_request_halt);
    let _ = vm.run();
    vm.reset(sub);
    let _ = vm.run();
    for _ in 0..8 {
        vm.reset(end_only.clone());
        let _ = vm.run();
    }
    vm.reset(end_only);
    let _ = vm.run();
    assert_eq!(SUBSHELL_END_CALLS.load(Ordering::SeqCst), 11);
    assert_eq!(vm.last_status, 426);
}

#[test]
fn subshell_end_exit_code_426_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(426)));
    match vm.run() {
        VMResult::Ok(Value::Status(426)) => {}
        other => panic!("exit code 426 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_jump_if_true_after_subshell_end() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::LoadInt(5), 1);
    b.emit(Op::LoadInt(2), 1);
    b.emit(Op::NumLt, 1);
    let jmp = b.emit(Op::JumpIfTrue(0), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.patch_jump(jmp, b.current_pos());
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(427)));
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 427);
}

#[test]
fn subshell_end_exit_code_427_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(427)));
    match vm.run() {
        VMResult::Ok(Value::Status(427)) => {}
        other => panic!("exit code 427 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_negative_then_zero_second_end_overwrites() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    struct NegativeThenZeroHost {
        calls: u32,
    }
    impl ShellHost for NegativeThenZeroHost {
        fn subshell_end(&mut self) -> Option<i32> {
            self.calls += 1;
            if self.calls == 1 {
                Some(-61)
            } else {
                Some(0)
            }
        }
    }
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(NegativeThenZeroHost { calls: 0 }));
    match vm.run() {
        VMResult::Ok(Value::Status(0)) => {}
        other => panic!("second subshell_end(Some(0)) MUST overwrite -61, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_428_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(428)));
    match vm.run() {
        VMResult::Ok(Value::Status(428)) => {}
        other => panic!("exit code 428 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_sixty_seven_consecutive_runs_without_reset_still_blocked() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    for _ in 0..67 {
        let _ = vm.run();
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_429_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(429)));
    match vm.run() {
        VMResult::Ok(Value::Status(429)) => {}
        other => panic!("exit code 429 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_sixty_five_end_incrementing_from_fourteen() {
    let mut b = ChunkBuilder::new();
    for _ in 0..65 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(IncrementingSubshellHost { next: 14 }));
    match vm.run() {
        VMResult::Ok(Value::Status(78)) => {}
        other => panic!("sixty-fifth subshell_end(Some(78)) MUST win, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_430_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(430)));
    match vm.run() {
        VMResult::Ok(Value::Status(430)) => {}
        other => panic!("exit code 430 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_between_fourth_subshell_end_and_fifth_subshell_begin() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    SUBSHELL_BEGIN_CALLS.store(0, Ordering::SeqCst);
    SUBSHELL_END_CALLS.store(0, Ordering::SeqCst);
    struct CountingBeginEndHost431;
    impl ShellHost for CountingBeginEndHost431 {
        fn subshell_begin(&mut self) {
            SUBSHELL_BEGIN_CALLS.fetch_add(1, Ordering::SeqCst);
        }
        fn subshell_end(&mut self) -> Option<i32> {
            SUBSHELL_END_CALLS.fetch_add(1, Ordering::SeqCst);
            Some(431)
        }
    }
    let mut b = ChunkBuilder::new();
    for _ in 0..4 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(CountingBeginEndHost431));
    vm.register_builtin(100, builtin_request_halt);
    let _ = vm.run();
    assert_eq!(SUBSHELL_END_CALLS.load(Ordering::SeqCst), 4);
    assert_eq!(SUBSHELL_BEGIN_CALLS.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 431);
}

#[test]
fn subshell_end_exit_code_431_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(431)));
    match vm.run() {
        VMResult::Ok(Value::Status(431)) => {}
        other => panic!("exit code 431 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_str_le_on_consts_after_subshell_end_status432() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    let a = b.add_constant(Value::str("abc"));
    let bidx = b.add_constant(Value::str("abd"));
    b.emit(Op::LoadConst(a), 1);
    b.emit(Op::LoadConst(bidx), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::StrLe, 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(432)));
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 432);
}

#[test]
fn subshell_end_exit_code_432_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(432)));
    match vm.run() {
        VMResult::Ok(Value::Status(432)) => {}
        other => panic!("exit code 432 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_eight_reset_runs_same_host_same_status() {
    let chunk = {
        let mut b = ChunkBuilder::new();
        b.emit(Op::SubshellEnd, 1);
        b.emit(Op::GetStatus, 1);
        b.build()
    };
    let mut vm = VM::new(chunk.clone());
    vm.set_shell_host(Box::new(StatusReturningHost(433)));
    for _ in 0..8 {
        match vm.run() {
            VMResult::Ok(Value::Status(433)) => {}
            other => panic!("reset run MUST return Status(433), got {:?}", other),
        }
        vm.reset(chunk.clone());
    }
}

#[test]
fn subshell_end_exit_code_433_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(433)));
    match vm.run() {
        VMResult::Ok(Value::Status(433)) => {}
        other => panic!("exit code 433 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_inc_after_subshell_end_status434() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::LoadInt(15), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::Inc, 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(434)));
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 434);
}

#[test]
fn subshell_end_exit_code_434_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(434)));
    match vm.run() {
        VMResult::Ok(Value::Status(434)) => {}
        other => panic!("exit code 434 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_status_minus_sixty_one_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(-61)));
    match vm.run() {
        VMResult::Ok(Value::Status(-61)) => {}
        other => panic!("status -61 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_435_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(435)));
    match vm.run() {
        VMResult::Ok(Value::Status(435)) => {}
        other => panic!("exit code 435 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_sixty_eight_consecutive_runs_without_reset_still_blocked() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    for _ in 0..68 {
        let _ = vm.run();
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_436_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(436)));
    match vm.run() {
        VMResult::Ok(Value::Status(436)) => {}
        other => panic!("exit code 436 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_sixty_six_end_incrementing_from_three() {
    let mut b = ChunkBuilder::new();
    for _ in 0..66 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(IncrementingSubshellHost { next: 3 }));
    match vm.run() {
        VMResult::Ok(Value::Status(68)) => {}
        other => panic!("sixty-sixth subshell_end(Some(68)) MUST win, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_437_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(437)));
    match vm.run() {
        VMResult::Ok(Value::Status(437)) => {}
        other => panic!("exit code 437 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_dec_after_subshell_end_status438() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::LoadInt(20), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::Dec, 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(438)));
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 438);
}

#[test]
fn subshell_end_exit_code_438_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(438)));
    match vm.run() {
        VMResult::Ok(Value::Status(438)) => {}
        other => panic!("exit code 438 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_twenty_seventh_some_host_applies_on_twenty_seventh_end() {
    let mut b = ChunkBuilder::new();
    for _ in 0..27 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    struct TwentySeventhSomeHost {
        calls: u32,
    }
    impl ShellHost for TwentySeventhSomeHost {
        fn subshell_end(&mut self) -> Option<i32> {
            self.calls += 1;
            if self.calls == 27 {
                Some(439)
            } else {
                None
            }
        }
    }
    let mut vm = VM::new(b.build());
    vm.last_status = 8;
    vm.set_shell_host(Box::new(TwentySeventhSomeHost { calls: 0 }));
    match vm.run() {
        VMResult::Ok(Value::Status(439)) => {}
        other => panic!("twenty-seventh subshell_end(Some(439)) MUST apply, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_439_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(439)));
    match vm.run() {
        VMResult::Ok(Value::Status(439)) => {}
        other => panic!("exit code 439 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_make_hash_after_subshell_end_status440() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    let k = b.add_constant(Value::str("x"));
    let v = b.add_constant(Value::str("y"));
    b.emit(Op::LoadConst(k), 1);
    b.emit(Op::LoadConst(v), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::MakeHash(1), 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(440)));
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 440);
}

#[test]
fn subshell_end_exit_code_440_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(440)));
    match vm.run() {
        VMResult::Ok(Value::Status(440)) => {}
        other => panic!("exit code 440 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_halt_reset_getstatus_on_sixth_reset_run() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    let halt = {
        let mut b = ChunkBuilder::new();
        b.emit(Op::CallBuiltin(100, 0), 1);
        b.build()
    };
    let sub = {
        let mut b = ChunkBuilder::new();
        b.emit(Op::SubshellEnd, 1);
        b.emit(Op::GetStatus, 1);
        b.build()
    };
    let mut vm = VM::new(halt.clone());
    vm.set_shell_host(Box::new(StatusReturningHost(441)));
    vm.register_builtin(100, builtin_request_halt);
    let _ = vm.run();
    vm.reset(halt);
    let _ = vm.run();
    vm.reset(sub.clone());
    let _ = vm.run();
    vm.reset(sub.clone());
    let _ = vm.run();
    vm.reset(sub.clone());
    let _ = vm.run();
    vm.reset(sub);
    match vm.run() {
        VMResult::Ok(Value::Status(441)) => {}
        other => panic!("sixth reset run MUST return Status(441), got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_441_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(441)));
    match vm.run() {
        VMResult::Ok(Value::Status(441)) => {}
        other => panic!("exit code 441 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_sixty_nine_consecutive_runs_without_reset_still_blocked() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    for _ in 0..69 {
        let _ = vm.run();
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_442_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(442)));
    match vm.run() {
        VMResult::Ok(Value::Status(442)) => {}
        other => panic!("exit code 442 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_status_minus_sixty_two_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(-62)));
    match vm.run() {
        VMResult::Ok(Value::Status(-62)) => {}
        other => panic!("status -62 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_dup2_after_subshell_end_status443() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::LoadInt(10), 1);
    b.emit(Op::LoadInt(20), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::Dup2, 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(443)));
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 443);
}

#[test]
fn subshell_end_exit_code_443_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(443)));
    match vm.run() {
        VMResult::Ok(Value::Status(443)) => {}
        other => panic!("exit code 443 MUST propagate, got {:?}", other),
    }
}

// ─── Pin tests batch Q (handwritten) ──────────────────────────────────────

#[test]
fn subshell_end_exit_code_444_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(444)));
    match vm.run() {
        VMResult::Ok(Value::Status(444)) => {}
        other => panic!("exit code 444 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_seventy_consecutive_runs_without_reset_still_blocked() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    for _ in 0..70 {
        let _ = vm.run();
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_445_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(445)));
    match vm.run() {
        VMResult::Ok(Value::Status(445)) => {}
        other => panic!("exit code 445 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_seventy_one_consecutive_runs_without_reset_still_blocked() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    for _ in 0..71 {
        let _ = vm.run();
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_446_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(446)));
    match vm.run() {
        VMResult::Ok(Value::Status(446)) => {}
        other => panic!("exit code 446 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_seventy_two_consecutive_runs_without_reset_still_blocked() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    for _ in 0..72 {
        let _ = vm.run();
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_447_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(447)));
    match vm.run() {
        VMResult::Ok(Value::Status(447)) => {}
        other => panic!("exit code 447 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_seventy_three_consecutive_runs_without_reset_still_blocked() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    for _ in 0..73 {
        let _ = vm.run();
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_448_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(448)));
    match vm.run() {
        VMResult::Ok(Value::Status(448)) => {}
        other => panic!("exit code 448 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_seventy_four_consecutive_runs_without_reset_still_blocked() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    for _ in 0..74 {
        let _ = vm.run();
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_449_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(449)));
    match vm.run() {
        VMResult::Ok(Value::Status(449)) => {}
        other => panic!("exit code 449 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_sixty_seven_end_incrementing_from_six() {
    let mut b = ChunkBuilder::new();
    for _ in 0..67 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(IncrementingSubshellHost { next: 6 }));
    match vm.run() {
        VMResult::Ok(Value::Status(72)) => {}
        other => panic!("sixty-seventh subshell_end(Some(72)) MUST win, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_450_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(450)));
    match vm.run() {
        VMResult::Ok(Value::Status(450)) => {}
        other => panic!("exit code 450 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_status_minus_sixty_three_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(-63)));
    match vm.run() {
        VMResult::Ok(Value::Status(-63)) => {}
        other => panic!("status -63 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_451_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(451)));
    match vm.run() {
        VMResult::Ok(Value::Status(451)) => {}
        other => panic!("exit code 451 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_sixty_eight_end_incrementing_from_eleven() {
    let mut b = ChunkBuilder::new();
    for _ in 0..68 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(IncrementingSubshellHost { next: 11 }));
    match vm.run() {
        VMResult::Ok(Value::Status(78)) => {}
        other => panic!("sixty-eighth subshell_end(Some(78)) MUST win, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_452_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(452)));
    match vm.run() {
        VMResult::Ok(Value::Status(452)) => {}
        other => panic!("exit code 452 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_status_minus_sixty_four_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(-64)));
    match vm.run() {
        VMResult::Ok(Value::Status(-64)) => {}
        other => panic!("status -64 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_453_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(453)));
    match vm.run() {
        VMResult::Ok(Value::Status(453)) => {}
        other => panic!("exit code 453 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_status_minus_sixty_five_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(-65)));
    match vm.run() {
        VMResult::Ok(Value::Status(-65)) => {}
        other => panic!("status -65 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_454_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(454)));
    match vm.run() {
        VMResult::Ok(Value::Status(454)) => {}
        other => panic!("exit code 454 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_status_minus_sixty_six_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(-66)));
    match vm.run() {
        VMResult::Ok(Value::Status(-66)) => {}
        other => panic!("status -66 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_455_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(455)));
    match vm.run() {
        VMResult::Ok(Value::Status(455)) => {}
        other => panic!("exit code 455 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_status_minus_sixty_seven_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(-67)));
    match vm.run() {
        VMResult::Ok(Value::Status(-67)) => {}
        other => panic!("status -67 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_456_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(456)));
    match vm.run() {
        VMResult::Ok(Value::Status(456)) => {}
        other => panic!("exit code 456 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_twenty_eighth_some_host_applies_on_twenty_eighth_end() {
    let mut b = ChunkBuilder::new();
    for _ in 0..28 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    struct TwentyEighthSomeHost {
        calls: u32,
    }
    impl ShellHost for TwentyEighthSomeHost {
        fn subshell_end(&mut self) -> Option<i32> {
            self.calls += 1;
            if self.calls == 28 {
                Some(457)
            } else {
                None
            }
        }
    }
    let mut vm = VM::new(b.build());
    vm.last_status = 5;
    vm.set_shell_host(Box::new(TwentyEighthSomeHost { calls: 0 }));
    match vm.run() {
        VMResult::Ok(Value::Status(457)) => {}
        other => panic!("twenty-eighth subshell_end(Some(457)) MUST apply, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_457_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(457)));
    match vm.run() {
        VMResult::Ok(Value::Status(457)) => {}
        other => panic!("exit code 457 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_halt_reset_subshell_end_twelfth_run_applies() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    SUBSHELL_END_CALLS.store(0, Ordering::SeqCst);
    let sub = {
        let mut b = ChunkBuilder::new();
        b.emit(Op::SubshellEnd, 1);
        b.emit(Op::CallBuiltin(100, 0), 1);
        b.build()
    };
    let end_only = {
        let mut b = ChunkBuilder::new();
        b.emit(Op::SubshellEnd, 1);
        b.build()
    };
    let mut vm = VM::new(sub.clone());
    vm.set_shell_host(Box::new(CountingSubshellEndHost(458)));
    vm.register_builtin(100, builtin_request_halt);
    let _ = vm.run();
    vm.reset(sub);
    let _ = vm.run();
    for _ in 0..9 {
        vm.reset(end_only.clone());
        let _ = vm.run();
    }
    vm.reset(end_only);
    let _ = vm.run();
    assert_eq!(SUBSHELL_END_CALLS.load(Ordering::SeqCst), 12);
    assert_eq!(vm.last_status, 458);
}

#[test]
fn subshell_end_exit_code_458_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(458)));
    match vm.run() {
        VMResult::Ok(Value::Status(458)) => {}
        other => panic!("exit code 458 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_mod_after_subshell_end_status459() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::LoadInt(17), 1);
    b.emit(Op::LoadInt(5), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::Mod, 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(459)));
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 459);
}

#[test]
fn subshell_end_exit_code_459_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(459)));
    match vm.run() {
        VMResult::Ok(Value::Status(459)) => {}
        other => panic!("exit code 459 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_eight_begins_five_ends_last_end_status_wins() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    SUBSHELL_BEGIN_CALLS.store(0, Ordering::SeqCst);
    SUBSHELL_END_CALLS.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    for _ in 0..8 {
        b.emit(Op::SubshellBegin, 1);
    }
    for _ in 0..5 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    struct CountingBeginEndHost460;
    impl ShellHost for CountingBeginEndHost460 {
        fn subshell_begin(&mut self) {
            SUBSHELL_BEGIN_CALLS.fetch_add(1, Ordering::SeqCst);
        }
        fn subshell_end(&mut self) -> Option<i32> {
            SUBSHELL_END_CALLS.fetch_add(1, Ordering::SeqCst);
            Some(460)
        }
    }
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(CountingBeginEndHost460));
    match vm.run() {
        VMResult::Ok(Value::Status(460)) => {}
        other => panic!("fifth subshell_end(Some(460)) MUST win, got {:?}", other),
    }
    assert_eq!(SUBSHELL_BEGIN_CALLS.load(Ordering::SeqCst), 8);
    assert_eq!(SUBSHELL_END_CALLS.load(Ordering::SeqCst), 5);
}

#[test]
fn subshell_end_exit_code_460_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(460)));
    match vm.run() {
        VMResult::Ok(Value::Status(460)) => {}
        other => panic!("exit code 460 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_div_after_subshell_end_status461() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::LoadInt(24), 1);
    b.emit(Op::LoadInt(6), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::Div, 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(461)));
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 461);
}

#[test]
fn subshell_end_exit_code_461_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(461)));
    match vm.run() {
        VMResult::Ok(Value::Status(461)) => {}
        other => panic!("exit code 461 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_nine_reset_runs_same_host_same_status() {
    let chunk = {
        let mut b = ChunkBuilder::new();
        b.emit(Op::SubshellEnd, 1);
        b.emit(Op::GetStatus, 1);
        b.build()
    };
    let mut vm = VM::new(chunk.clone());
    vm.set_shell_host(Box::new(StatusReturningHost(462)));
    for _ in 0..9 {
        match vm.run() {
            VMResult::Ok(Value::Status(462)) => {}
            other => panic!("reset run MUST return Status(462), got {:?}", other),
        }
        vm.reset(chunk.clone());
    }
}

#[test]
fn subshell_end_exit_code_462_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(462)));
    match vm.run() {
        VMResult::Ok(Value::Status(462)) => {}
        other => panic!("exit code 462 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_alternating_status_fourteenth_none_keeps_ninth() {
    let mut b = ChunkBuilder::new();
    for _ in 0..14 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(AlternatingStatusHost { calls: 0 }));
    match vm.run() {
        VMResult::Ok(Value::Status(9)) => {}
        other => panic!("AlternatingStatusHost fourteenth None MUST keep Some(9), got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_463_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(463)));
    match vm.run() {
        VMResult::Ok(Value::Status(463)) => {}
        other => panic!("exit code 463 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_mul_after_subshell_end_status464() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::LoadInt(8), 1);
    b.emit(Op::LoadInt(9), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::Mul, 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(464)));
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 464);
}

#[test]
fn subshell_end_exit_code_464_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(464)));
    match vm.run() {
        VMResult::Ok(Value::Status(464)) => {}
        other => panic!("exit code 464 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_sixty_nine_end_incrementing_from_four() {
    let mut b = ChunkBuilder::new();
    for _ in 0..69 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(IncrementingSubshellHost { next: 4 }));
    match vm.run() {
        VMResult::Ok(Value::Status(72)) => {}
        other => panic!("sixty-ninth subshell_end(Some(72)) MUST win, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_465_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(465)));
    match vm.run() {
        VMResult::Ok(Value::Status(465)) => {}
        other => panic!("exit code 465 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_sub_after_subshell_end_status466() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::LoadInt(11), 1);
    b.emit(Op::LoadInt(4), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::Sub, 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(466)));
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 466);
}

#[test]
fn subshell_end_exit_code_466_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(466)));
    match vm.run() {
        VMResult::Ok(Value::Status(466)) => {}
        other => panic!("exit code 466 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_make_array_after_subshell_end_status467() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::LoadInt(3), 1);
    b.emit(Op::LoadInt(4), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::MakeArray(2), 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(467)));
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 467);
}

#[test]
fn subshell_end_exit_code_467_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(467)));
    match vm.run() {
        VMResult::Ok(Value::Status(467)) => {}
        other => panic!("exit code 467 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_seventy_five_consecutive_runs_without_reset_still_blocked() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    for _ in 0..75 {
        let _ = vm.run();
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_468_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(468)));
    match vm.run() {
        VMResult::Ok(Value::Status(468)) => {}
        other => panic!("exit code 468 MUST propagate, got {:?}", other),
    }
}

// ─── Pin tests batch R (handwritten) ──────────────────────────────────────

#[test]
fn subshell_end_exit_code_469_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(469)));
    match vm.run() {
        VMResult::Ok(Value::Status(469)) => {}
        other => panic!("exit code 469 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_seventy_six_consecutive_runs_without_reset_still_blocked() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    for _ in 0..76 {
        let _ = vm.run();
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_470_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(470)));
    match vm.run() {
        VMResult::Ok(Value::Status(470)) => {}
        other => panic!("exit code 470 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_seventy_seven_consecutive_runs_without_reset_still_blocked() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    for _ in 0..77 {
        let _ = vm.run();
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_471_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(471)));
    match vm.run() {
        VMResult::Ok(Value::Status(471)) => {}
        other => panic!("exit code 471 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_seventy_eight_consecutive_runs_without_reset_still_blocked() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    for _ in 0..78 {
        let _ = vm.run();
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_472_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(472)));
    match vm.run() {
        VMResult::Ok(Value::Status(472)) => {}
        other => panic!("exit code 472 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_seventy_nine_consecutive_runs_without_reset_still_blocked() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    for _ in 0..79 {
        let _ = vm.run();
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_473_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(473)));
    match vm.run() {
        VMResult::Ok(Value::Status(473)) => {}
        other => panic!("exit code 473 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_eighty_consecutive_runs_without_reset_still_blocked() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    for _ in 0..80 {
        let _ = vm.run();
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_474_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(474)));
    match vm.run() {
        VMResult::Ok(Value::Status(474)) => {}
        other => panic!("exit code 474 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_seventy_end_incrementing_from_one() {
    let mut b = ChunkBuilder::new();
    for _ in 0..70 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(IncrementingSubshellHost { next: 1 }));
    match vm.run() {
        VMResult::Ok(Value::Status(70)) => {}
        other => panic!("seventieth subshell_end(Some(70)) MUST win, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_475_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(475)));
    match vm.run() {
        VMResult::Ok(Value::Status(475)) => {}
        other => panic!("exit code 475 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_status_minus_sixty_eight_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(-68)));
    match vm.run() {
        VMResult::Ok(Value::Status(-68)) => {}
        other => panic!("status -68 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_476_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(476)));
    match vm.run() {
        VMResult::Ok(Value::Status(476)) => {}
        other => panic!("exit code 476 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_status_minus_sixty_nine_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(-69)));
    match vm.run() {
        VMResult::Ok(Value::Status(-69)) => {}
        other => panic!("status -69 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_477_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(477)));
    match vm.run() {
        VMResult::Ok(Value::Status(477)) => {}
        other => panic!("exit code 477 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_status_minus_seventy_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(-70)));
    match vm.run() {
        VMResult::Ok(Value::Status(-70)) => {}
        other => panic!("status -70 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_478_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(478)));
    match vm.run() {
        VMResult::Ok(Value::Status(478)) => {}
        other => panic!("exit code 478 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_status_minus_seventy_one_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(-71)));
    match vm.run() {
        VMResult::Ok(Value::Status(-71)) => {}
        other => panic!("status -71 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_479_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(479)));
    match vm.run() {
        VMResult::Ok(Value::Status(479)) => {}
        other => panic!("exit code 479 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_status_minus_seventy_two_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(-72)));
    match vm.run() {
        VMResult::Ok(Value::Status(-72)) => {}
        other => panic!("status -72 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_twenty_ninth_some_host_applies_on_twenty_ninth_end() {
    let mut b = ChunkBuilder::new();
    for _ in 0..29 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    struct TwentyNinthSomeHost {
        calls: u32,
    }
    impl ShellHost for TwentyNinthSomeHost {
        fn subshell_end(&mut self) -> Option<i32> {
            self.calls += 1;
            if self.calls == 29 {
                Some(480)
            } else {
                None
            }
        }
    }
    let mut vm = VM::new(b.build());
    vm.last_status = 3;
    vm.set_shell_host(Box::new(TwentyNinthSomeHost { calls: 0 }));
    match vm.run() {
        VMResult::Ok(Value::Status(480)) => {}
        other => panic!("twenty-ninth subshell_end(Some(480)) MUST apply, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_480_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(480)));
    match vm.run() {
        VMResult::Ok(Value::Status(480)) => {}
        other => panic!("exit code 480 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_halt_reset_subshell_end_thirteenth_run_applies() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    SUBSHELL_END_CALLS.store(0, Ordering::SeqCst);
    let sub = {
        let mut b = ChunkBuilder::new();
        b.emit(Op::SubshellEnd, 1);
        b.emit(Op::CallBuiltin(100, 0), 1);
        b.build()
    };
    let end_only = {
        let mut b = ChunkBuilder::new();
        b.emit(Op::SubshellEnd, 1);
        b.build()
    };
    let mut vm = VM::new(sub.clone());
    vm.set_shell_host(Box::new(CountingSubshellEndHost(481)));
    vm.register_builtin(100, builtin_request_halt);
    let _ = vm.run();
    vm.reset(sub);
    let _ = vm.run();
    for _ in 0..10 {
        vm.reset(end_only.clone());
        let _ = vm.run();
    }
    vm.reset(end_only);
    let _ = vm.run();
    assert_eq!(SUBSHELL_END_CALLS.load(Ordering::SeqCst), 13);
    assert_eq!(vm.last_status, 481);
}

#[test]
fn subshell_end_exit_code_481_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(481)));
    match vm.run() {
        VMResult::Ok(Value::Status(481)) => {}
        other => panic!("exit code 481 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_add_after_subshell_end_status482() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::LoadInt(13), 1);
    b.emit(Op::LoadInt(7), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::Add, 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(482)));
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 482);
}

#[test]
fn subshell_end_exit_code_482_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(482)));
    match vm.run() {
        VMResult::Ok(Value::Status(482)) => {}
        other => panic!("exit code 482 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_nine_begins_six_ends_last_end_status_wins() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    SUBSHELL_BEGIN_CALLS.store(0, Ordering::SeqCst);
    SUBSHELL_END_CALLS.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    for _ in 0..9 {
        b.emit(Op::SubshellBegin, 1);
    }
    for _ in 0..6 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    struct CountingBeginEndHost483;
    impl ShellHost for CountingBeginEndHost483 {
        fn subshell_begin(&mut self) {
            SUBSHELL_BEGIN_CALLS.fetch_add(1, Ordering::SeqCst);
        }
        fn subshell_end(&mut self) -> Option<i32> {
            SUBSHELL_END_CALLS.fetch_add(1, Ordering::SeqCst);
            Some(483)
        }
    }
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(CountingBeginEndHost483));
    match vm.run() {
        VMResult::Ok(Value::Status(483)) => {}
        other => panic!("sixth subshell_end(Some(483)) MUST win, got {:?}", other),
    }
    assert_eq!(SUBSHELL_BEGIN_CALLS.load(Ordering::SeqCst), 9);
    assert_eq!(SUBSHELL_END_CALLS.load(Ordering::SeqCst), 6);
}

#[test]
fn subshell_end_exit_code_483_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(483)));
    match vm.run() {
        VMResult::Ok(Value::Status(483)) => {}
        other => panic!("exit code 483 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_neg_after_subshell_end_status484() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::LoadInt(5), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::Negate, 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(484)));
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 484);
}

#[test]
fn subshell_end_exit_code_484_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(484)));
    match vm.run() {
        VMResult::Ok(Value::Status(484)) => {}
        other => panic!("exit code 484 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_ten_reset_runs_same_host_same_status() {
    let chunk = {
        let mut b = ChunkBuilder::new();
        b.emit(Op::SubshellEnd, 1);
        b.emit(Op::GetStatus, 1);
        b.build()
    };
    let mut vm = VM::new(chunk.clone());
    vm.set_shell_host(Box::new(StatusReturningHost(485)));
    for _ in 0..10 {
        match vm.run() {
            VMResult::Ok(Value::Status(485)) => {}
            other => panic!("reset run MUST return Status(485), got {:?}", other),
        }
        vm.reset(chunk.clone());
    }
}

#[test]
fn subshell_end_exit_code_485_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(485)));
    match vm.run() {
        VMResult::Ok(Value::Status(485)) => {}
        other => panic!("exit code 485 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_first_some_then_none_host_fifteenth_none_keeps_first() {
    let mut b = ChunkBuilder::new();
    for _ in 0..15 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(FirstSomeThenNoneHost { calls: 0, status: 486 }));
    match vm.run() {
        VMResult::Ok(Value::Status(486)) => {}
        other => panic!("FirstSomeThenNoneHost fifteenth None MUST keep first Some(486), got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_486_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(486)));
    match vm.run() {
        VMResult::Ok(Value::Status(486)) => {}
        other => panic!("exit code 486 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_not_after_subshell_end_status487() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::LoadInt(0), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::LogNot, 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(487)));
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 487);
}

#[test]
fn subshell_end_exit_code_487_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(487)));
    match vm.run() {
        VMResult::Ok(Value::Status(487)) => {}
        other => panic!("exit code 487 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_seventy_one_end_incrementing_from_two() {
    let mut b = ChunkBuilder::new();
    for _ in 0..71 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(IncrementingSubshellHost { next: 2 }));
    match vm.run() {
        VMResult::Ok(Value::Status(72)) => {}
        other => panic!("seventy-first subshell_end(Some(72)) MUST win, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_488_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(488)));
    match vm.run() {
        VMResult::Ok(Value::Status(488)) => {}
        other => panic!("exit code 488 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_eq_after_subshell_end_status489() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::LoadInt(3), 1);
    b.emit(Op::LoadInt(3), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::NumEq, 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(489)));
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 489);
}

#[test]
fn subshell_end_exit_code_489_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(489)));
    match vm.run() {
        VMResult::Ok(Value::Status(489)) => {}
        other => panic!("exit code 489 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_lt_after_subshell_end_status490() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::LoadInt(2), 1);
    b.emit(Op::LoadInt(5), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::NumLt, 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(490)));
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 490);
}

#[test]
fn subshell_end_exit_code_490_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(490)));
    match vm.run() {
        VMResult::Ok(Value::Status(490)) => {}
        other => panic!("exit code 490 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_gt_after_subshell_end_status491() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::LoadInt(9), 1);
    b.emit(Op::LoadInt(2), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::NumGt, 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(491)));
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 491);
}

#[test]
fn subshell_end_exit_code_491_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(491)));
    match vm.run() {
        VMResult::Ok(Value::Status(491)) => {}
        other => panic!("exit code 491 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_eighty_one_consecutive_runs_without_reset_still_blocked() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    for _ in 0..81 {
        let _ = vm.run();
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_492_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(492)));
    match vm.run() {
        VMResult::Ok(Value::Status(492)) => {}
        other => panic!("exit code 492 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_thirtieth_some_host_applies_on_thirtieth_end() {
    let mut b = ChunkBuilder::new();
    for _ in 0..30 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    struct ThirtiethSomeHost {
        calls: u32,
    }
    impl ShellHost for ThirtiethSomeHost {
        fn subshell_end(&mut self) -> Option<i32> {
            self.calls += 1;
            if self.calls == 30 {
                Some(493)
            } else {
                None
            }
        }
    }
    let mut vm = VM::new(b.build());
    vm.last_status = 1;
    vm.set_shell_host(Box::new(ThirtiethSomeHost { calls: 0 }));
    match vm.run() {
        VMResult::Ok(Value::Status(493)) => {}
        other => panic!("thirtieth subshell_end(Some(493)) MUST apply, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_493_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(493)));
    match vm.run() {
        VMResult::Ok(Value::Status(493)) => {}
        other => panic!("exit code 493 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_pipeline_then_subshell_getstatus_reads_subshell_494() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::PipelineBegin(2), 1);
    b.emit(Op::PipelineEnd, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    struct PipelineSubshellHost494;
    impl ShellHost for PipelineSubshellHost494 {
        fn pipeline_end(&mut self) -> i32 {
            2
        }
        fn subshell_end(&mut self) -> Option<i32> {
            Some(494)
        }
    }
    vm.set_shell_host(Box::new(PipelineSubshellHost494));
    match vm.run() {
        VMResult::Ok(Value::Status(494)) => {}
        other => panic!("GetStatus after pipeline+subshell MUST read subshell 494, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_494_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(494)));
    match vm.run() {
        VMResult::Ok(Value::Status(494)) => {}
        other => panic!("exit code 494 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_none_host_fifteen_ends_leaves_initial_status() {
    let mut b = ChunkBuilder::new();
    for _ in 0..15 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.last_status = 495;
    vm.set_shell_host(Box::new(NoneHost));
    match vm.run() {
        VMResult::Ok(Value::Status(495)) => {}
        other => panic!("NoneHost fifteen ends MUST leave last_status 495, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_495_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(495)));
    match vm.run() {
        VMResult::Ok(Value::Status(495)) => {}
        other => panic!("exit code 495 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_pre_run_empty_stack_then_subshell_end_status496() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.request_halt();
    vm.set_shell_host(Box::new(StatusReturningHost(496)));
    match vm.run() {
        VMResult::Halted => {}
        other => panic!("pre-run halt empty stack MUST Halted before subshell_end, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_496_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(496)));
    match vm.run() {
        VMResult::Ok(Value::Status(496)) => {}
        other => panic!("exit code 496 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_negative_then_zero_host_last_zero_wins() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    struct NegativeThenZeroHost496 {
        calls: u32,
    }
    impl ShellHost for NegativeThenZeroHost496 {
        fn subshell_end(&mut self) -> Option<i32> {
            self.calls += 1;
            if self.calls == 1 {
                Some(-73)
            } else {
                Some(0)
            }
        }
    }
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(NegativeThenZeroHost496 { calls: 0 }));
    match vm.run() {
        VMResult::Ok(Value::Status(0)) => {}
        other => panic!("second subshell_end(Some(0)) MUST win over -73, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_497_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(497)));
    match vm.run() {
        VMResult::Ok(Value::Status(497)) => {}
        other => panic!("exit code 497 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_eighty_two_consecutive_runs_without_reset_still_blocked() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    for _ in 0..82 {
        let _ = vm.run();
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_498_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(498)));
    match vm.run() {
        VMResult::Ok(Value::Status(498)) => {}
        other => panic!("exit code 498 MUST propagate, got {:?}", other),
    }
}

// ─── Pin tests batch S (handwritten) ──────────────────────────────────────

#[test]
fn subshell_end_exit_code_499_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(499)));
    match vm.run() {
        VMResult::Ok(Value::Status(499)) => {}
        other => panic!("exit code 499 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_eighty_three_consecutive_runs_without_reset_still_blocked() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    for _ in 0..83 {
        let _ = vm.run();
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_500_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(500)));
    match vm.run() {
        VMResult::Ok(Value::Status(500)) => {}
        other => panic!("exit code 500 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_eighty_four_consecutive_runs_without_reset_still_blocked() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    for _ in 0..84 {
        let _ = vm.run();
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_501_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(501)));
    match vm.run() {
        VMResult::Ok(Value::Status(501)) => {}
        other => panic!("exit code 501 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_eighty_five_consecutive_runs_without_reset_still_blocked() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    for _ in 0..85 {
        let _ = vm.run();
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_502_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(502)));
    match vm.run() {
        VMResult::Ok(Value::Status(502)) => {}
        other => panic!("exit code 502 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_seventy_two_end_incrementing_from_three() {
    let mut b = ChunkBuilder::new();
    for _ in 0..72 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(IncrementingSubshellHost { next: 3 }));
    match vm.run() {
        VMResult::Ok(Value::Status(74)) => {}
        other => panic!("seventy-second subshell_end(Some(74)) MUST win, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_503_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(503)));
    match vm.run() {
        VMResult::Ok(Value::Status(503)) => {}
        other => panic!("exit code 503 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_status_minus_seventy_four_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(-74)));
    match vm.run() {
        VMResult::Ok(Value::Status(-74)) => {}
        other => panic!("status -74 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_504_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(504)));
    match vm.run() {
        VMResult::Ok(Value::Status(504)) => {}
        other => panic!("exit code 504 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_status_minus_seventy_five_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(-75)));
    match vm.run() {
        VMResult::Ok(Value::Status(-75)) => {}
        other => panic!("status -75 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_505_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(505)));
    match vm.run() {
        VMResult::Ok(Value::Status(505)) => {}
        other => panic!("exit code 505 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_thirty_first_some_host_applies_on_thirty_first_end() {
    let mut b = ChunkBuilder::new();
    for _ in 0..31 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    struct ThirtyFirstSomeHost {
        calls: u32,
    }
    impl ShellHost for ThirtyFirstSomeHost {
        fn subshell_end(&mut self) -> Option<i32> {
            self.calls += 1;
            if self.calls == 31 {
                Some(506)
            } else {
                None
            }
        }
    }
    let mut vm = VM::new(b.build());
    vm.last_status = 2;
    vm.set_shell_host(Box::new(ThirtyFirstSomeHost { calls: 0 }));
    match vm.run() {
        VMResult::Ok(Value::Status(506)) => {}
        other => panic!("thirty-first subshell_end(Some(506)) MUST apply, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_506_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(506)));
    match vm.run() {
        VMResult::Ok(Value::Status(506)) => {}
        other => panic!("exit code 506 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_halt_reset_subshell_end_fourteenth_run_applies() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    SUBSHELL_END_CALLS.store(0, Ordering::SeqCst);
    let sub = {
        let mut b = ChunkBuilder::new();
        b.emit(Op::SubshellEnd, 1);
        b.emit(Op::CallBuiltin(100, 0), 1);
        b.build()
    };
    let end_only = {
        let mut b = ChunkBuilder::new();
        b.emit(Op::SubshellEnd, 1);
        b.build()
    };
    let mut vm = VM::new(sub.clone());
    vm.set_shell_host(Box::new(CountingSubshellEndHost(507)));
    vm.register_builtin(100, builtin_request_halt);
    let _ = vm.run();
    vm.reset(sub);
    let _ = vm.run();
    for _ in 0..11 {
        vm.reset(end_only.clone());
        let _ = vm.run();
    }
    vm.reset(end_only);
    let _ = vm.run();
    assert_eq!(SUBSHELL_END_CALLS.load(Ordering::SeqCst), 14);
    assert_eq!(vm.last_status, 507);
}

#[test]
fn subshell_end_exit_code_507_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(507)));
    match vm.run() {
        VMResult::Ok(Value::Status(507)) => {}
        other => panic!("exit code 507 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_num_ne_after_subshell_end_status508() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::LoadInt(4), 1);
    b.emit(Op::LoadInt(9), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::NumNe, 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(508)));
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 508);
}

#[test]
fn subshell_end_exit_code_508_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(508)));
    match vm.run() {
        VMResult::Ok(Value::Status(508)) => {}
        other => panic!("exit code 508 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_ten_begins_seven_ends_last_end_status_wins() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    SUBSHELL_BEGIN_CALLS.store(0, Ordering::SeqCst);
    SUBSHELL_END_CALLS.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    for _ in 0..10 {
        b.emit(Op::SubshellBegin, 1);
    }
    for _ in 0..7 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    struct CountingBeginEndHost509;
    impl ShellHost for CountingBeginEndHost509 {
        fn subshell_begin(&mut self) {
            SUBSHELL_BEGIN_CALLS.fetch_add(1, Ordering::SeqCst);
        }
        fn subshell_end(&mut self) -> Option<i32> {
            SUBSHELL_END_CALLS.fetch_add(1, Ordering::SeqCst);
            Some(509)
        }
    }
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(CountingBeginEndHost509));
    match vm.run() {
        VMResult::Ok(Value::Status(509)) => {}
        other => panic!("seventh subshell_end(Some(509)) MUST win, got {:?}", other),
    }
    assert_eq!(SUBSHELL_BEGIN_CALLS.load(Ordering::SeqCst), 10);
    assert_eq!(SUBSHELL_END_CALLS.load(Ordering::SeqCst), 7);
}

#[test]
fn subshell_end_exit_code_509_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(509)));
    match vm.run() {
        VMResult::Ok(Value::Status(509)) => {}
        other => panic!("exit code 509 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_spaceship_after_subshell_end_status510() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::LoadInt(6), 1);
    b.emit(Op::LoadInt(2), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::Spaceship, 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(510)));
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 510);
}

#[test]
fn subshell_end_exit_code_510_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(510)));
    match vm.run() {
        VMResult::Ok(Value::Status(510)) => {}
        other => panic!("exit code 510 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_eleven_reset_runs_same_host_same_status() {
    let chunk = {
        let mut b = ChunkBuilder::new();
        b.emit(Op::SubshellEnd, 1);
        b.emit(Op::GetStatus, 1);
        b.build()
    };
    let mut vm = VM::new(chunk.clone());
    vm.set_shell_host(Box::new(StatusReturningHost(511)));
    for _ in 0..11 {
        match vm.run() {
            VMResult::Ok(Value::Status(511)) => {}
            other => panic!("reset run MUST return Status(511), got {:?}", other),
        }
        vm.reset(chunk.clone());
    }
}

#[test]
fn subshell_end_exit_code_511_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(511)));
    match vm.run() {
        VMResult::Ok(Value::Status(511)) => {}
        other => panic!("exit code 511 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_alternating_status_fifteenth_none_keeps_ninth() {
    let mut b = ChunkBuilder::new();
    for _ in 0..15 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(AlternatingStatusHost { calls: 0 }));
    match vm.run() {
        VMResult::Ok(Value::Status(9)) => {}
        other => panic!("AlternatingStatusHost fifteenth None MUST keep Some(9), got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_512_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(512)));
    match vm.run() {
        VMResult::Ok(Value::Status(512)) => {}
        other => panic!("exit code 512 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_bit_and_after_subshell_end_status513() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::LoadInt(5), 1);
    b.emit(Op::LoadInt(3), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::BitAnd, 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(513)));
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 513);
}

#[test]
fn subshell_end_exit_code_513_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(513)));
    match vm.run() {
        VMResult::Ok(Value::Status(513)) => {}
        other => panic!("exit code 513 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_seventy_three_end_incrementing_from_five() {
    let mut b = ChunkBuilder::new();
    for _ in 0..73 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(IncrementingSubshellHost { next: 5 }));
    match vm.run() {
        VMResult::Ok(Value::Status(77)) => {}
        other => panic!("seventy-third subshell_end(Some(77)) MUST win, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_514_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(514)));
    match vm.run() {
        VMResult::Ok(Value::Status(514)) => {}
        other => panic!("exit code 514 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_load_true_after_subshell_end_status515() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::LoadTrue, 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(515)));
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 515);
}

#[test]
fn subshell_end_exit_code_515_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(515)));
    match vm.run() {
        VMResult::Ok(Value::Status(515)) => {}
        other => panic!("exit code 515 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_load_false_after_subshell_end_status516() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::LoadFalse, 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(516)));
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 516);
}

#[test]
fn subshell_end_exit_code_516_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(516)));
    match vm.run() {
        VMResult::Ok(Value::Status(516)) => {}
        other => panic!("exit code 516 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_eighty_six_consecutive_runs_without_reset_still_blocked() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    for _ in 0..86 {
        let _ = vm.run();
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_517_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(517)));
    match vm.run() {
        VMResult::Ok(Value::Status(517)) => {}
        other => panic!("exit code 517 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_eighty_seven_consecutive_runs_without_reset_still_blocked() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    for _ in 0..87 {
        let _ = vm.run();
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_518_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(518)));
    match vm.run() {
        VMResult::Ok(Value::Status(518)) => {}
        other => panic!("exit code 518 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_dup_after_subshell_end_status519() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::LoadInt(12), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::Dup, 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(519)));
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 519);
}

#[test]
fn subshell_end_exit_code_519_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(519)));
    match vm.run() {
        VMResult::Ok(Value::Status(519)) => {}
        other => panic!("exit code 519 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_swap_after_subshell_end_status520() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::LoadInt(3), 1);
    b.emit(Op::LoadInt(8), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::Swap, 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(520)));
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 520);
}

#[test]
fn subshell_end_exit_code_520_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(520)));
    match vm.run() {
        VMResult::Ok(Value::Status(520)) => {}
        other => panic!("exit code 520 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_nine_begins_eight_ends_last_end_status_wins() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    SUBSHELL_BEGIN_CALLS.store(0, Ordering::SeqCst);
    SUBSHELL_END_CALLS.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    for _ in 0..9 {
        b.emit(Op::SubshellBegin, 1);
    }
    for _ in 0..8 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    struct CountingBeginEndHost521;
    impl ShellHost for CountingBeginEndHost521 {
        fn subshell_begin(&mut self) {
            SUBSHELL_BEGIN_CALLS.fetch_add(1, Ordering::SeqCst);
        }
        fn subshell_end(&mut self) -> Option<i32> {
            SUBSHELL_END_CALLS.fetch_add(1, Ordering::SeqCst);
            Some(521)
        }
    }
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(CountingBeginEndHost521));
    match vm.run() {
        VMResult::Ok(Value::Status(521)) => {}
        other => panic!("eighth subshell_end(Some(521)) MUST win, got {:?}", other),
    }
    assert_eq!(SUBSHELL_BEGIN_CALLS.load(Ordering::SeqCst), 9);
    assert_eq!(SUBSHELL_END_CALLS.load(Ordering::SeqCst), 8);
}

#[test]
fn subshell_end_exit_code_521_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(521)));
    match vm.run() {
        VMResult::Ok(Value::Status(521)) => {}
        other => panic!("exit code 521 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_seventy_four_end_incrementing_from_eight() {
    let mut b = ChunkBuilder::new();
    for _ in 0..74 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(IncrementingSubshellHost { next: 8 }));
    match vm.run() {
        VMResult::Ok(Value::Status(81)) => {}
        other => panic!("seventy-fourth subshell_end(Some(81)) MUST win, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_522_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(522)));
    match vm.run() {
        VMResult::Ok(Value::Status(522)) => {}
        other => panic!("exit code 522 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_load_undef_after_subshell_end_status523() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::LoadUndef, 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(523)));
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 523);
}

#[test]
fn subshell_end_exit_code_523_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(523)));
    match vm.run() {
        VMResult::Ok(Value::Status(523)) => {}
        other => panic!("exit code 523 MUST propagate, got {:?}", other),
    }
}

// ─── Pin tests batch T (handwritten) ──────────────────────────────────────

#[test]
fn subshell_end_exit_code_524_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(524)));
    match vm.run() {
        VMResult::Ok(Value::Status(524)) => {}
        other => panic!("exit code 524 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_eighty_eight_consecutive_runs_without_reset_still_blocked() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    for _ in 0..88 {
        let _ = vm.run();
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_525_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(525)));
    match vm.run() {
        VMResult::Ok(Value::Status(525)) => {}
        other => panic!("exit code 525 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_eighty_nine_consecutive_runs_without_reset_still_blocked() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    for _ in 0..89 {
        let _ = vm.run();
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_526_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(526)));
    match vm.run() {
        VMResult::Ok(Value::Status(526)) => {}
        other => panic!("exit code 526 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_ninety_consecutive_runs_without_reset_still_blocked() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    for _ in 0..90 {
        let _ = vm.run();
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_527_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(527)));
    match vm.run() {
        VMResult::Ok(Value::Status(527)) => {}
        other => panic!("exit code 527 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_seventy_five_end_incrementing_from_nine() {
    let mut b = ChunkBuilder::new();
    for _ in 0..75 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(IncrementingSubshellHost { next: 9 }));
    match vm.run() {
        VMResult::Ok(Value::Status(83)) => {}
        other => panic!("seventy-fifth subshell_end(Some(83)) MUST win, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_528_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(528)));
    match vm.run() {
        VMResult::Ok(Value::Status(528)) => {}
        other => panic!("exit code 528 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_status_minus_seventy_six_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(-76)));
    match vm.run() {
        VMResult::Ok(Value::Status(-76)) => {}
        other => panic!("status -76 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_529_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(529)));
    match vm.run() {
        VMResult::Ok(Value::Status(529)) => {}
        other => panic!("exit code 529 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_status_minus_seventy_seven_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(-77)));
    match vm.run() {
        VMResult::Ok(Value::Status(-77)) => {}
        other => panic!("status -77 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_530_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(530)));
    match vm.run() {
        VMResult::Ok(Value::Status(530)) => {}
        other => panic!("exit code 530 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_status_minus_seventy_eight_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(-78)));
    match vm.run() {
        VMResult::Ok(Value::Status(-78)) => {}
        other => panic!("status -78 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_531_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(531)));
    match vm.run() {
        VMResult::Ok(Value::Status(531)) => {}
        other => panic!("exit code 531 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_thirty_second_some_host_applies_on_thirty_second_end() {
    let mut b = ChunkBuilder::new();
    for _ in 0..32 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    struct ThirtySecondSomeHost {
        calls: u32,
    }
    impl ShellHost for ThirtySecondSomeHost {
        fn subshell_end(&mut self) -> Option<i32> {
            self.calls += 1;
            if self.calls == 32 {
                Some(532)
            } else {
                None
            }
        }
    }
    let mut vm = VM::new(b.build());
    vm.last_status = 4;
    vm.set_shell_host(Box::new(ThirtySecondSomeHost { calls: 0 }));
    match vm.run() {
        VMResult::Ok(Value::Status(532)) => {}
        other => panic!("thirty-second subshell_end(Some(532)) MUST apply, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_532_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(532)));
    match vm.run() {
        VMResult::Ok(Value::Status(532)) => {}
        other => panic!("exit code 532 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_halt_reset_subshell_end_fifteenth_run_applies() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    SUBSHELL_END_CALLS.store(0, Ordering::SeqCst);
    let sub = {
        let mut b = ChunkBuilder::new();
        b.emit(Op::SubshellEnd, 1);
        b.emit(Op::CallBuiltin(100, 0), 1);
        b.build()
    };
    let end_only = {
        let mut b = ChunkBuilder::new();
        b.emit(Op::SubshellEnd, 1);
        b.build()
    };
    let mut vm = VM::new(sub.clone());
    vm.set_shell_host(Box::new(CountingSubshellEndHost(533)));
    vm.register_builtin(100, builtin_request_halt);
    let _ = vm.run();
    vm.reset(sub);
    let _ = vm.run();
    for _ in 0..12 {
        vm.reset(end_only.clone());
        let _ = vm.run();
    }
    vm.reset(end_only);
    let _ = vm.run();
    assert_eq!(SUBSHELL_END_CALLS.load(Ordering::SeqCst), 15);
    assert_eq!(vm.last_status, 533);
}

#[test]
fn subshell_end_exit_code_533_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(533)));
    match vm.run() {
        VMResult::Ok(Value::Status(533)) => {}
        other => panic!("exit code 533 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_bit_or_after_subshell_end_status534() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::LoadInt(6), 1);
    b.emit(Op::LoadInt(10), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::BitOr, 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(534)));
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 534);
}

#[test]
fn subshell_end_exit_code_534_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(534)));
    match vm.run() {
        VMResult::Ok(Value::Status(534)) => {}
        other => panic!("exit code 534 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_eleven_begins_nine_ends_last_end_status_wins() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    SUBSHELL_BEGIN_CALLS.store(0, Ordering::SeqCst);
    SUBSHELL_END_CALLS.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    for _ in 0..11 {
        b.emit(Op::SubshellBegin, 1);
    }
    for _ in 0..9 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    struct CountingBeginEndHost535;
    impl ShellHost for CountingBeginEndHost535 {
        fn subshell_begin(&mut self) {
            SUBSHELL_BEGIN_CALLS.fetch_add(1, Ordering::SeqCst);
        }
        fn subshell_end(&mut self) -> Option<i32> {
            SUBSHELL_END_CALLS.fetch_add(1, Ordering::SeqCst);
            Some(535)
        }
    }
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(CountingBeginEndHost535));
    match vm.run() {
        VMResult::Ok(Value::Status(535)) => {}
        other => panic!("ninth subshell_end(Some(535)) MUST win, got {:?}", other),
    }
    assert_eq!(SUBSHELL_BEGIN_CALLS.load(Ordering::SeqCst), 11);
    assert_eq!(SUBSHELL_END_CALLS.load(Ordering::SeqCst), 9);
}

#[test]
fn subshell_end_exit_code_535_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(535)));
    match vm.run() {
        VMResult::Ok(Value::Status(535)) => {}
        other => panic!("exit code 535 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_num_le_after_subshell_end_status536() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::LoadInt(3), 1);
    b.emit(Op::LoadInt(5), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::NumLe, 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(536)));
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 536);
}

#[test]
fn subshell_end_exit_code_536_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(536)));
    match vm.run() {
        VMResult::Ok(Value::Status(536)) => {}
        other => panic!("exit code 536 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_twelve_reset_runs_same_host_same_status() {
    let chunk = {
        let mut b = ChunkBuilder::new();
        b.emit(Op::SubshellEnd, 1);
        b.emit(Op::GetStatus, 1);
        b.build()
    };
    let mut vm = VM::new(chunk.clone());
    vm.set_shell_host(Box::new(StatusReturningHost(537)));
    for _ in 0..12 {
        match vm.run() {
            VMResult::Ok(Value::Status(537)) => {}
            other => panic!("reset run MUST return Status(537), got {:?}", other),
        }
        vm.reset(chunk.clone());
    }
}

#[test]
fn subshell_end_exit_code_537_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(537)));
    match vm.run() {
        VMResult::Ok(Value::Status(537)) => {}
        other => panic!("exit code 537 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_num_ge_after_subshell_end_status538() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::LoadInt(7), 1);
    b.emit(Op::LoadInt(4), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::NumGe, 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(538)));
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 538);
}

#[test]
fn subshell_end_exit_code_538_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(538)));
    match vm.run() {
        VMResult::Ok(Value::Status(538)) => {}
        other => panic!("exit code 538 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_seventy_six_end_incrementing_from_twelve() {
    let mut b = ChunkBuilder::new();
    for _ in 0..76 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(IncrementingSubshellHost { next: 12 }));
    match vm.run() {
        VMResult::Ok(Value::Status(87)) => {}
        other => panic!("seventy-sixth subshell_end(Some(87)) MUST win, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_539_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(539)));
    match vm.run() {
        VMResult::Ok(Value::Status(539)) => {}
        other => panic!("exit code 539 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_inc_after_subshell_end_status540() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::LoadInt(14), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::Inc, 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(540)));
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 540);
}

#[test]
fn subshell_end_exit_code_540_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(540)));
    match vm.run() {
        VMResult::Ok(Value::Status(540)) => {}
        other => panic!("exit code 540 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_dec_after_subshell_end_status541() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::LoadInt(20), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::Dec, 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(541)));
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 541);
}

#[test]
fn subshell_end_exit_code_541_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(541)));
    match vm.run() {
        VMResult::Ok(Value::Status(541)) => {}
        other => panic!("exit code 541 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_ninety_one_consecutive_runs_without_reset_still_blocked() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    for _ in 0..91 {
        let _ = vm.run();
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_542_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(542)));
    match vm.run() {
        VMResult::Ok(Value::Status(542)) => {}
        other => panic!("exit code 542 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_thirty_third_some_host_applies_on_thirty_third_end() {
    let mut b = ChunkBuilder::new();
    for _ in 0..33 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    struct ThirtyThirdSomeHost {
        calls: u32,
    }
    impl ShellHost for ThirtyThirdSomeHost {
        fn subshell_end(&mut self) -> Option<i32> {
            self.calls += 1;
            if self.calls == 33 {
                Some(543)
            } else {
                None
            }
        }
    }
    let mut vm = VM::new(b.build());
    vm.last_status = 6;
    vm.set_shell_host(Box::new(ThirtyThirdSomeHost { calls: 0 }));
    match vm.run() {
        VMResult::Ok(Value::Status(543)) => {}
        other => panic!("thirty-third subshell_end(Some(543)) MUST apply, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_543_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(543)));
    match vm.run() {
        VMResult::Ok(Value::Status(543)) => {}
        other => panic!("exit code 543 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_first_some_then_none_host_sixteenth_none_keeps_first() {
    let mut b = ChunkBuilder::new();
    for _ in 0..16 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(FirstSomeThenNoneHost { calls: 0, status: 544 }));
    match vm.run() {
        VMResult::Ok(Value::Status(544)) => {}
        other => panic!("FirstSomeThenNoneHost sixteenth None MUST keep first Some(544), got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_544_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(544)));
    match vm.run() {
        VMResult::Ok(Value::Status(544)) => {}
        other => panic!("exit code 544 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_pow_after_subshell_end_status545() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::LoadInt(2), 1);
    b.emit(Op::LoadInt(4), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::Pow, 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(545)));
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 545);
}

#[test]
fn subshell_end_exit_code_545_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(545)));
    match vm.run() {
        VMResult::Ok(Value::Status(545)) => {}
        other => panic!("exit code 545 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_none_host_sixteen_ends_leaves_initial_status() {
    let mut b = ChunkBuilder::new();
    for _ in 0..16 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.last_status = 546;
    vm.set_shell_host(Box::new(NoneHost));
    match vm.run() {
        VMResult::Ok(Value::Status(546)) => {}
        other => panic!("NoneHost sixteen ends MUST leave last_status 546, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_546_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(546)));
    match vm.run() {
        VMResult::Ok(Value::Status(546)) => {}
        other => panic!("exit code 546 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_negative_then_zero_host_last_zero_wins_status547() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    struct NegativeThenZeroHost547 {
        calls: u32,
    }
    impl ShellHost for NegativeThenZeroHost547 {
        fn subshell_end(&mut self) -> Option<i32> {
            self.calls += 1;
            if self.calls == 1 {
                Some(-79)
            } else {
                Some(0)
            }
        }
    }
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(NegativeThenZeroHost547 { calls: 0 }));
    match vm.run() {
        VMResult::Ok(Value::Status(0)) => {}
        other => panic!("second subshell_end(Some(0)) MUST win over -79, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_547_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(547)));
    match vm.run() {
        VMResult::Ok(Value::Status(547)) => {}
        other => panic!("exit code 547 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_rot_after_subshell_end_status548() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::LoadInt(2), 1);
    b.emit(Op::LoadInt(3), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::Rot, 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(548)));
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 548);
}

#[test]
fn subshell_end_exit_code_548_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(548)));
    match vm.run() {
        VMResult::Ok(Value::Status(548)) => {}
        other => panic!("exit code 548 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_ninety_two_consecutive_runs_without_reset_still_blocked() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    for _ in 0..92 {
        let _ = vm.run();
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_subshell_then_pipeline_getstatus_reads_pipeline_549() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::PipelineBegin(2), 1);
    b.emit(Op::PipelineEnd, 1);
    b.emit(Op::GetStatus, 1);
    struct SubshellThenPipelineHost549;
    impl ShellHost for SubshellThenPipelineHost549 {
        fn subshell_end(&mut self) -> Option<i32> {
            Some(11)
        }
        fn pipeline_end(&mut self) -> i32 {
            549
        }
    }
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(SubshellThenPipelineHost549));
    match vm.run() {
        VMResult::Ok(Value::Status(549)) => {}
        other => panic!("GetStatus after subshell+pipeline MUST read pipeline 549, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_549_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(549)));
    match vm.run() {
        VMResult::Ok(Value::Status(549)) => {}
        other => panic!("exit code 549 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_seventy_seven_end_incrementing_from_one() {
    let mut b = ChunkBuilder::new();
    for _ in 0..77 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(IncrementingSubshellHost { next: 1 }));
    match vm.run() {
        VMResult::Ok(Value::Status(77)) => {}
        other => panic!("seventy-seventh subshell_end(Some(77)) MUST win, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_550_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(550)));
    match vm.run() {
        VMResult::Ok(Value::Status(550)) => {}
        other => panic!("exit code 550 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_dup2_after_subshell_end_status551() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::LoadInt(10), 1);
    b.emit(Op::LoadInt(20), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::Dup2, 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(551)));
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 551);
}

#[test]
fn subshell_end_exit_code_551_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(551)));
    match vm.run() {
        VMResult::Ok(Value::Status(551)) => {}
        other => panic!("exit code 551 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_ninety_three_consecutive_runs_without_reset_still_blocked() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    for _ in 0..93 {
        let _ = vm.run();
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_552_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(552)));
    match vm.run() {
        VMResult::Ok(Value::Status(552)) => {}
        other => panic!("exit code 552 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_alternating_status_sixteenth_none_keeps_ninth() {
    let mut b = ChunkBuilder::new();
    for _ in 0..16 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(AlternatingStatusHost { calls: 0 }));
    match vm.run() {
        VMResult::Ok(Value::Status(9)) => {}
        other => panic!("AlternatingStatusHost sixteenth None MUST keep Some(9), got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_553_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(553)));
    match vm.run() {
        VMResult::Ok(Value::Status(553)) => {}
        other => panic!("exit code 553 MUST propagate, got {:?}", other),
    }
}

// ─── Pin tests batch U (handwritten) ──────────────────────────────────────

#[test]
fn subshell_end_exit_code_554_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(554)));
    match vm.run() {
        VMResult::Ok(Value::Status(554)) => {}
        other => panic!("exit code 554 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_ninety_four_consecutive_runs_without_reset_still_blocked() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    for _ in 0..94 {
        let _ = vm.run();
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_555_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(555)));
    match vm.run() {
        VMResult::Ok(Value::Status(555)) => {}
        other => panic!("exit code 555 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_ninety_five_consecutive_runs_without_reset_still_blocked() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    for _ in 0..95 {
        let _ = vm.run();
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_556_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(556)));
    match vm.run() {
        VMResult::Ok(Value::Status(556)) => {}
        other => panic!("exit code 556 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_ninety_six_consecutive_runs_without_reset_still_blocked() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    for _ in 0..96 {
        let _ = vm.run();
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_557_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(557)));
    match vm.run() {
        VMResult::Ok(Value::Status(557)) => {}
        other => panic!("exit code 557 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_seventy_eight_end_incrementing_from_two() {
    let mut b = ChunkBuilder::new();
    for _ in 0..78 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(IncrementingSubshellHost { next: 2 }));
    match vm.run() {
        VMResult::Ok(Value::Status(79)) => {}
        other => panic!("seventy-eighth subshell_end(Some(79)) MUST win, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_558_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(558)));
    match vm.run() {
        VMResult::Ok(Value::Status(558)) => {}
        other => panic!("exit code 558 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_status_minus_eighty_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(-80)));
    match vm.run() {
        VMResult::Ok(Value::Status(-80)) => {}
        other => panic!("status -80 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_559_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(559)));
    match vm.run() {
        VMResult::Ok(Value::Status(559)) => {}
        other => panic!("exit code 559 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_status_minus_eighty_one_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(-81)));
    match vm.run() {
        VMResult::Ok(Value::Status(-81)) => {}
        other => panic!("status -81 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_560_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(560)));
    match vm.run() {
        VMResult::Ok(Value::Status(560)) => {}
        other => panic!("exit code 560 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_thirty_fourth_some_host_applies_on_thirty_fourth_end() {
    let mut b = ChunkBuilder::new();
    for _ in 0..34 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    struct ThirtyFourthSomeHost {
        calls: u32,
    }
    impl ShellHost for ThirtyFourthSomeHost {
        fn subshell_end(&mut self) -> Option<i32> {
            self.calls += 1;
            if self.calls == 34 {
                Some(561)
            } else {
                None
            }
        }
    }
    let mut vm = VM::new(b.build());
    vm.last_status = 7;
    vm.set_shell_host(Box::new(ThirtyFourthSomeHost { calls: 0 }));
    match vm.run() {
        VMResult::Ok(Value::Status(561)) => {}
        other => panic!("thirty-fourth subshell_end(Some(561)) MUST apply, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_561_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(561)));
    match vm.run() {
        VMResult::Ok(Value::Status(561)) => {}
        other => panic!("exit code 561 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_halt_reset_subshell_end_sixteenth_run_applies() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    SUBSHELL_END_CALLS.store(0, Ordering::SeqCst);
    let sub = {
        let mut b = ChunkBuilder::new();
        b.emit(Op::SubshellEnd, 1);
        b.emit(Op::CallBuiltin(100, 0), 1);
        b.build()
    };
    let end_only = {
        let mut b = ChunkBuilder::new();
        b.emit(Op::SubshellEnd, 1);
        b.build()
    };
    let mut vm = VM::new(sub.clone());
    vm.set_shell_host(Box::new(CountingSubshellEndHost(562)));
    vm.register_builtin(100, builtin_request_halt);
    let _ = vm.run();
    vm.reset(sub);
    let _ = vm.run();
    for _ in 0..13 {
        vm.reset(end_only.clone());
        let _ = vm.run();
    }
    vm.reset(end_only);
    let _ = vm.run();
    assert_eq!(SUBSHELL_END_CALLS.load(Ordering::SeqCst), 16);
    assert_eq!(vm.last_status, 562);
}

#[test]
fn subshell_end_exit_code_562_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(562)));
    match vm.run() {
        VMResult::Ok(Value::Status(562)) => {}
        other => panic!("exit code 562 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_bit_xor_after_subshell_end_status563() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::LoadInt(12), 1);
    b.emit(Op::LoadInt(5), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::BitXor, 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(563)));
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 563);
}

#[test]
fn subshell_end_exit_code_563_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(563)));
    match vm.run() {
        VMResult::Ok(Value::Status(563)) => {}
        other => panic!("exit code 563 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_twelve_begins_ten_ends_last_end_status_wins() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    SUBSHELL_BEGIN_CALLS.store(0, Ordering::SeqCst);
    SUBSHELL_END_CALLS.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    for _ in 0..12 {
        b.emit(Op::SubshellBegin, 1);
    }
    for _ in 0..10 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    struct CountingBeginEndHost564;
    impl ShellHost for CountingBeginEndHost564 {
        fn subshell_begin(&mut self) {
            SUBSHELL_BEGIN_CALLS.fetch_add(1, Ordering::SeqCst);
        }
        fn subshell_end(&mut self) -> Option<i32> {
            SUBSHELL_END_CALLS.fetch_add(1, Ordering::SeqCst);
            Some(564)
        }
    }
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(CountingBeginEndHost564));
    match vm.run() {
        VMResult::Ok(Value::Status(564)) => {}
        other => panic!("tenth subshell_end(Some(564)) MUST win, got {:?}", other),
    }
    assert_eq!(SUBSHELL_BEGIN_CALLS.load(Ordering::SeqCst), 12);
    assert_eq!(SUBSHELL_END_CALLS.load(Ordering::SeqCst), 10);
}

#[test]
fn subshell_end_exit_code_564_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(564)));
    match vm.run() {
        VMResult::Ok(Value::Status(564)) => {}
        other => panic!("exit code 564 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_bit_not_after_subshell_end_status565() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::LoadInt(7), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::BitNot, 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(565)));
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 565);
}

#[test]
fn subshell_end_exit_code_565_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(565)));
    match vm.run() {
        VMResult::Ok(Value::Status(565)) => {}
        other => panic!("exit code 565 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_thirteen_reset_runs_same_host_same_status() {
    let chunk = {
        let mut b = ChunkBuilder::new();
        b.emit(Op::SubshellEnd, 1);
        b.emit(Op::GetStatus, 1);
        b.build()
    };
    let mut vm = VM::new(chunk.clone());
    vm.set_shell_host(Box::new(StatusReturningHost(566)));
    for _ in 0..13 {
        match vm.run() {
            VMResult::Ok(Value::Status(566)) => {}
            other => panic!("reset run MUST return Status(566), got {:?}", other),
        }
        vm.reset(chunk.clone());
    }
}

#[test]
fn subshell_end_exit_code_566_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(566)));
    match vm.run() {
        VMResult::Ok(Value::Status(566)) => {}
        other => panic!("exit code 566 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_shl_after_subshell_end_status567() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::LoadInt(3), 1);
    b.emit(Op::LoadInt(2), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::Shl, 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(567)));
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 567);
}

#[test]
fn subshell_end_exit_code_567_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(567)));
    match vm.run() {
        VMResult::Ok(Value::Status(567)) => {}
        other => panic!("exit code 567 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_seventy_nine_end_incrementing_from_six() {
    let mut b = ChunkBuilder::new();
    for _ in 0..79 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(IncrementingSubshellHost { next: 6 }));
    match vm.run() {
        VMResult::Ok(Value::Status(84)) => {}
        other => panic!("seventy-ninth subshell_end(Some(84)) MUST win, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_568_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(568)));
    match vm.run() {
        VMResult::Ok(Value::Status(568)) => {}
        other => panic!("exit code 568 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_shr_after_subshell_end_status569() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::LoadInt(16), 1);
    b.emit(Op::LoadInt(2), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::Shr, 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(569)));
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 569);
}

#[test]
fn subshell_end_exit_code_569_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(569)));
    match vm.run() {
        VMResult::Ok(Value::Status(569)) => {}
        other => panic!("exit code 569 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_ninety_seven_consecutive_runs_without_reset_still_blocked() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    for _ in 0..97 {
        let _ = vm.run();
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_570_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(570)));
    match vm.run() {
        VMResult::Ok(Value::Status(570)) => {}
        other => panic!("exit code 570 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_ninety_eight_consecutive_runs_without_reset_still_blocked() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    for _ in 0..98 {
        let _ = vm.run();
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_571_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(571)));
    match vm.run() {
        VMResult::Ok(Value::Status(571)) => {}
        other => panic!("exit code 571 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_pop_after_subshell_end_status572() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::LoadInt(9), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::Pop, 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(572)));
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 572);
}

#[test]
fn subshell_end_exit_code_572_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(572)));
    match vm.run() {
        VMResult::Ok(Value::Status(572)) => {}
        other => panic!("exit code 572 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_eighty_end_incrementing_from_one() {
    let mut b = ChunkBuilder::new();
    for _ in 0..80 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(IncrementingSubshellHost { next: 1 }));
    match vm.run() {
        VMResult::Ok(Value::Status(80)) => {}
        other => panic!("eightieth subshell_end(Some(80)) MUST win, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_573_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(573)));
    match vm.run() {
        VMResult::Ok(Value::Status(573)) => {}
        other => panic!("exit code 573 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_log_and_after_subshell_end_status574() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::LoadTrue, 1);
    b.emit(Op::LoadFalse, 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::LogAnd, 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(574)));
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 574);
}

#[test]
fn subshell_end_exit_code_574_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(574)));
    match vm.run() {
        VMResult::Ok(Value::Status(574)) => {}
        other => panic!("exit code 574 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_status_minus_eighty_two_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(-82)));
    match vm.run() {
        VMResult::Ok(Value::Status(-82)) => {}
        other => panic!("status -82 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_575_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(575)));
    match vm.run() {
        VMResult::Ok(Value::Status(575)) => {}
        other => panic!("exit code 575 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_log_or_after_subshell_end_status576() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::LoadFalse, 1);
    b.emit(Op::LoadTrue, 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::LogOr, 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(576)));
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 576);
}

#[test]
fn subshell_end_exit_code_576_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(576)));
    match vm.run() {
        VMResult::Ok(Value::Status(576)) => {}
        other => panic!("exit code 576 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_none_host_seventeen_ends_leaves_initial_status() {
    let mut b = ChunkBuilder::new();
    for _ in 0..17 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.last_status = 577;
    vm.set_shell_host(Box::new(NoneHost));
    match vm.run() {
        VMResult::Ok(Value::Status(577)) => {}
        other => panic!("NoneHost seventeen ends MUST leave last_status 577, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_577_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(577)));
    match vm.run() {
        VMResult::Ok(Value::Status(577)) => {}
        other => panic!("exit code 577 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_first_some_then_none_host_seventeenth_none_keeps_first() {
    let mut b = ChunkBuilder::new();
    for _ in 0..17 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(FirstSomeThenNoneHost { calls: 0, status: 578 }));
    match vm.run() {
        VMResult::Ok(Value::Status(578)) => {}
        other => panic!("FirstSomeThenNoneHost seventeenth None MUST keep first Some(578), got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_578_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(578)));
    match vm.run() {
        VMResult::Ok(Value::Status(578)) => {}
        other => panic!("exit code 578 MUST propagate, got {:?}", other),
    }
}

// ─── Pin tests batch V (handwritten) ──────────────────────────────────────

#[test]
fn subshell_end_exit_code_579_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(579)));
    match vm.run() {
        VMResult::Ok(Value::Status(579)) => {}
        other => panic!("exit code 579 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_ninety_nine_consecutive_runs_without_reset_still_blocked() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    for _ in 0..99 {
        let _ = vm.run();
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_580_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(580)));
    match vm.run() {
        VMResult::Ok(Value::Status(580)) => {}
        other => panic!("exit code 580 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_one_hundred_consecutive_runs_without_reset_still_blocked() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    for _ in 0..100 {
        let _ = vm.run();
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_581_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(581)));
    match vm.run() {
        VMResult::Ok(Value::Status(581)) => {}
        other => panic!("exit code 581 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_one_hundred_one_consecutive_runs_without_reset_still_blocked() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    for _ in 0..101 {
        let _ = vm.run();
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_582_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(582)));
    match vm.run() {
        VMResult::Ok(Value::Status(582)) => {}
        other => panic!("exit code 582 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_eighty_one_end_incrementing_from_fifteen() {
    let mut b = ChunkBuilder::new();
    for _ in 0..81 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(IncrementingSubshellHost { next: 15 }));
    match vm.run() {
        VMResult::Ok(Value::Status(95)) => {}
        other => panic!("eighty-first subshell_end(Some(95)) MUST win, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_583_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(583)));
    match vm.run() {
        VMResult::Ok(Value::Status(583)) => {}
        other => panic!("exit code 583 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_status_minus_eighty_three_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(-83)));
    match vm.run() {
        VMResult::Ok(Value::Status(-83)) => {}
        other => panic!("status -83 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_584_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(584)));
    match vm.run() {
        VMResult::Ok(Value::Status(584)) => {}
        other => panic!("exit code 584 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_status_minus_eighty_four_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(-84)));
    match vm.run() {
        VMResult::Ok(Value::Status(-84)) => {}
        other => panic!("status -84 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_585_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(585)));
    match vm.run() {
        VMResult::Ok(Value::Status(585)) => {}
        other => panic!("exit code 585 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_thirty_fifth_some_host_applies_on_thirty_fifth_end() {
    let mut b = ChunkBuilder::new();
    for _ in 0..35 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    struct ThirtyFifthSomeHost {
        calls: u32,
    }
    impl ShellHost for ThirtyFifthSomeHost {
        fn subshell_end(&mut self) -> Option<i32> {
            self.calls += 1;
            if self.calls == 35 {
                Some(586)
            } else {
                None
            }
        }
    }
    let mut vm = VM::new(b.build());
    vm.last_status = 5;
    vm.set_shell_host(Box::new(ThirtyFifthSomeHost { calls: 0 }));
    match vm.run() {
        VMResult::Ok(Value::Status(586)) => {}
        other => panic!("thirty-fifth subshell_end(Some(586)) MUST apply, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_586_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(586)));
    match vm.run() {
        VMResult::Ok(Value::Status(586)) => {}
        other => panic!("exit code 586 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_halt_reset_subshell_end_seventeenth_run_applies() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    SUBSHELL_END_CALLS.store(0, Ordering::SeqCst);
    let sub = {
        let mut b = ChunkBuilder::new();
        b.emit(Op::SubshellEnd, 1);
        b.emit(Op::CallBuiltin(100, 0), 1);
        b.build()
    };
    let end_only = {
        let mut b = ChunkBuilder::new();
        b.emit(Op::SubshellEnd, 1);
        b.build()
    };
    let mut vm = VM::new(sub.clone());
    vm.set_shell_host(Box::new(CountingSubshellEndHost(587)));
    vm.register_builtin(100, builtin_request_halt);
    let _ = vm.run();
    vm.reset(sub);
    let _ = vm.run();
    for _ in 0..14 {
        vm.reset(end_only.clone());
        let _ = vm.run();
    }
    vm.reset(end_only);
    let _ = vm.run();
    assert_eq!(SUBSHELL_END_CALLS.load(Ordering::SeqCst), 17);
    assert_eq!(vm.last_status, 587);
}

#[test]
fn subshell_end_exit_code_587_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(587)));
    match vm.run() {
        VMResult::Ok(Value::Status(587)) => {}
        other => panic!("exit code 587 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_load_float_after_subshell_end_status588() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::LoadFloat(3.14), 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(588)));
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 588);
}

#[test]
fn subshell_end_exit_code_588_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(588)));
    match vm.run() {
        VMResult::Ok(Value::Status(588)) => {}
        other => panic!("exit code 588 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_thirteen_begins_eleven_ends_last_end_status_wins() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    SUBSHELL_BEGIN_CALLS.store(0, Ordering::SeqCst);
    SUBSHELL_END_CALLS.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    for _ in 0..13 {
        b.emit(Op::SubshellBegin, 1);
    }
    for _ in 0..11 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    struct CountingBeginEndHost589;
    impl ShellHost for CountingBeginEndHost589 {
        fn subshell_begin(&mut self) {
            SUBSHELL_BEGIN_CALLS.fetch_add(1, Ordering::SeqCst);
        }
        fn subshell_end(&mut self) -> Option<i32> {
            SUBSHELL_END_CALLS.fetch_add(1, Ordering::SeqCst);
            Some(589)
        }
    }
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(CountingBeginEndHost589));
    match vm.run() {
        VMResult::Ok(Value::Status(589)) => {}
        other => panic!("eleventh subshell_end(Some(589)) MUST win, got {:?}", other),
    }
    assert_eq!(SUBSHELL_BEGIN_CALLS.load(Ordering::SeqCst), 13);
    assert_eq!(SUBSHELL_END_CALLS.load(Ordering::SeqCst), 11);
}

#[test]
fn subshell_end_exit_code_589_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(589)));
    match vm.run() {
        VMResult::Ok(Value::Status(589)) => {}
        other => panic!("exit code 589 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_concat_after_subshell_end_status590() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let chunk = {
        let mut cb = ChunkBuilder::new();
        let a = cb.add_constant(Value::str("a"));
        let bidx = cb.add_constant(Value::str("b"));
        cb.emit(Op::SubshellEnd, 1);
        cb.emit(Op::LoadConst(a), 1);
        cb.emit(Op::LoadConst(bidx), 1);
        cb.emit(Op::CallBuiltin(100, 0), 1);
        cb.emit(Op::Concat, 1);
        cb.emit(Op::CallBuiltin(200, 0), 1);
        cb.build()
    };
    let mut vm = VM::new(chunk);
    vm.set_shell_host(Box::new(StatusReturningHost(590)));
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 590);
}

#[test]
fn subshell_end_exit_code_590_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(590)));
    match vm.run() {
        VMResult::Ok(Value::Status(590)) => {}
        other => panic!("exit code 590 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_fourteen_reset_runs_same_host_same_status() {
    let chunk = {
        let mut b = ChunkBuilder::new();
        b.emit(Op::SubshellEnd, 1);
        b.emit(Op::GetStatus, 1);
        b.build()
    };
    let mut vm = VM::new(chunk.clone());
    vm.set_shell_host(Box::new(StatusReturningHost(591)));
    for _ in 0..14 {
        match vm.run() {
            VMResult::Ok(Value::Status(591)) => {}
            other => panic!("reset run MUST return Status(591), got {:?}", other),
        }
        vm.reset(chunk.clone());
    }
}

#[test]
fn subshell_end_exit_code_591_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(591)));
    match vm.run() {
        VMResult::Ok(Value::Status(591)) => {}
        other => panic!("exit code 591 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_str_eq_after_subshell_end_status592() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let chunk = {
        let mut cb = ChunkBuilder::new();
        let a = cb.add_constant(Value::str("x"));
        let bidx = cb.add_constant(Value::str("x"));
        cb.emit(Op::SubshellEnd, 1);
        cb.emit(Op::LoadConst(a), 1);
        cb.emit(Op::LoadConst(bidx), 1);
        cb.emit(Op::CallBuiltin(100, 0), 1);
        cb.emit(Op::StrEq, 1);
        cb.emit(Op::CallBuiltin(200, 0), 1);
        cb.build()
    };
    let mut vm = VM::new(chunk);
    vm.set_shell_host(Box::new(StatusReturningHost(592)));
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 592);
}

#[test]
fn subshell_end_exit_code_592_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(592)));
    match vm.run() {
        VMResult::Ok(Value::Status(592)) => {}
        other => panic!("exit code 592 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_eighty_two_end_incrementing_from_four() {
    let mut b = ChunkBuilder::new();
    for _ in 0..82 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(IncrementingSubshellHost { next: 4 }));
    match vm.run() {
        VMResult::Ok(Value::Status(85)) => {}
        other => panic!("eighty-second subshell_end(Some(85)) MUST win, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_593_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(593)));
    match vm.run() {
        VMResult::Ok(Value::Status(593)) => {}
        other => panic!("exit code 593 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_str_ne_after_subshell_end_status594() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let chunk = {
        let mut cb = ChunkBuilder::new();
        let a = cb.add_constant(Value::str("a"));
        let bidx = cb.add_constant(Value::str("b"));
        cb.emit(Op::SubshellEnd, 1);
        cb.emit(Op::LoadConst(a), 1);
        cb.emit(Op::LoadConst(bidx), 1);
        cb.emit(Op::CallBuiltin(100, 0), 1);
        cb.emit(Op::StrNe, 1);
        cb.emit(Op::CallBuiltin(200, 0), 1);
        cb.build()
    };
    let mut vm = VM::new(chunk);
    vm.set_shell_host(Box::new(StatusReturningHost(594)));
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 594);
}

#[test]
fn subshell_end_exit_code_594_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(594)));
    match vm.run() {
        VMResult::Ok(Value::Status(594)) => {}
        other => panic!("exit code 594 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_negative_then_zero_host_last_zero_wins_status595() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    struct NegativeThenZeroHost595 {
        calls: u32,
    }
    impl ShellHost for NegativeThenZeroHost595 {
        fn subshell_end(&mut self) -> Option<i32> {
            self.calls += 1;
            if self.calls == 1 {
                Some(-85)
            } else {
                Some(0)
            }
        }
    }
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(NegativeThenZeroHost595 { calls: 0 }));
    match vm.run() {
        VMResult::Ok(Value::Status(0)) => {}
        other => panic!("second subshell_end(Some(0)) MUST win over -85, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_595_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(595)));
    match vm.run() {
        VMResult::Ok(Value::Status(595)) => {}
        other => panic!("exit code 595 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_one_hundred_two_consecutive_runs_without_reset_still_blocked() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    for _ in 0..102 {
        let _ = vm.run();
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_596_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(596)));
    match vm.run() {
        VMResult::Ok(Value::Status(596)) => {}
        other => panic!("exit code 596 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_thirty_sixth_some_host_applies_on_thirty_sixth_end() {
    let mut b = ChunkBuilder::new();
    for _ in 0..36 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    struct ThirtySixthSomeHost {
        calls: u32,
    }
    impl ShellHost for ThirtySixthSomeHost {
        fn subshell_end(&mut self) -> Option<i32> {
            self.calls += 1;
            if self.calls == 36 {
                Some(597)
            } else {
                None
            }
        }
    }
    let mut vm = VM::new(b.build());
    vm.last_status = 8;
    vm.set_shell_host(Box::new(ThirtySixthSomeHost { calls: 0 }));
    match vm.run() {
        VMResult::Ok(Value::Status(597)) => {}
        other => panic!("thirty-sixth subshell_end(Some(597)) MUST apply, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_597_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(597)));
    match vm.run() {
        VMResult::Ok(Value::Status(597)) => {}
        other => panic!("exit code 597 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_alternating_status_seventeenth_none_keeps_ninth() {
    let mut b = ChunkBuilder::new();
    for _ in 0..17 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(AlternatingStatusHost { calls: 0 }));
    match vm.run() {
        VMResult::Ok(Value::Status(9)) => {}
        other => panic!("AlternatingStatusHost seventeenth None MUST keep Some(9), got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_598_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(598)));
    match vm.run() {
        VMResult::Ok(Value::Status(598)) => {}
        other => panic!("exit code 598 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_str_lt_after_subshell_end_status599() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let chunk = {
        let mut cb = ChunkBuilder::new();
        let a = cb.add_constant(Value::str("a"));
        let bidx = cb.add_constant(Value::str("z"));
        cb.emit(Op::SubshellEnd, 1);
        cb.emit(Op::LoadConst(a), 1);
        cb.emit(Op::LoadConst(bidx), 1);
        cb.emit(Op::CallBuiltin(100, 0), 1);
        cb.emit(Op::StrLt, 1);
        cb.emit(Op::CallBuiltin(200, 0), 1);
        cb.build()
    };
    let mut vm = VM::new(chunk);
    vm.set_shell_host(Box::new(StatusReturningHost(599)));
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 599);
}

#[test]
fn subshell_end_exit_code_599_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(599)));
    match vm.run() {
        VMResult::Ok(Value::Status(599)) => {}
        other => panic!("exit code 599 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_pipeline_then_subshell_getstatus_reads_subshell_600() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::PipelineBegin(2), 1);
    b.emit(Op::PipelineEnd, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    struct PipelineSubshellHost600;
    impl ShellHost for PipelineSubshellHost600 {
        fn pipeline_end(&mut self) -> i32 {
            4
        }
        fn subshell_end(&mut self) -> Option<i32> {
            Some(600)
        }
    }
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(PipelineSubshellHost600));
    match vm.run() {
        VMResult::Ok(Value::Status(600)) => {}
        other => panic!("GetStatus after pipeline+subshell MUST read subshell 600, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_600_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(600)));
    match vm.run() {
        VMResult::Ok(Value::Status(600)) => {}
        other => panic!("exit code 600 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_str_gt_after_subshell_end_status601() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let chunk = {
        let mut cb = ChunkBuilder::new();
        let a = cb.add_constant(Value::str("z"));
        let bidx = cb.add_constant(Value::str("a"));
        cb.emit(Op::SubshellEnd, 1);
        cb.emit(Op::LoadConst(a), 1);
        cb.emit(Op::LoadConst(bidx), 1);
        cb.emit(Op::CallBuiltin(100, 0), 1);
        cb.emit(Op::StrGt, 1);
        cb.emit(Op::CallBuiltin(200, 0), 1);
        cb.build()
    };
    let mut vm = VM::new(chunk);
    vm.set_shell_host(Box::new(StatusReturningHost(601)));
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 601);
}

#[test]
fn subshell_end_exit_code_601_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(601)));
    match vm.run() {
        VMResult::Ok(Value::Status(601)) => {}
        other => panic!("exit code 601 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_eighty_three_end_incrementing_from_seven() {
    let mut b = ChunkBuilder::new();
    for _ in 0..83 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(IncrementingSubshellHost { next: 7 }));
    match vm.run() {
        VMResult::Ok(Value::Status(89)) => {}
        other => panic!("eighty-third subshell_end(Some(89)) MUST win, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_602_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(602)));
    match vm.run() {
        VMResult::Ok(Value::Status(602)) => {}
        other => panic!("exit code 602 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_str_le_after_subshell_end_status603() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let chunk = {
        let mut cb = ChunkBuilder::new();
        let a = cb.add_constant(Value::str("a"));
        let bidx = cb.add_constant(Value::str("a"));
        cb.emit(Op::SubshellEnd, 1);
        cb.emit(Op::LoadConst(a), 1);
        cb.emit(Op::LoadConst(bidx), 1);
        cb.emit(Op::CallBuiltin(100, 0), 1);
        cb.emit(Op::StrLe, 1);
        cb.emit(Op::CallBuiltin(200, 0), 1);
        cb.build()
    };
    let mut vm = VM::new(chunk);
    vm.set_shell_host(Box::new(StatusReturningHost(603)));
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 603);
}

#[test]
fn subshell_end_exit_code_603_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(603)));
    match vm.run() {
        VMResult::Ok(Value::Status(603)) => {}
        other => panic!("exit code 603 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_one_hundred_three_consecutive_runs_without_reset_still_blocked() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    for _ in 0..103 {
        let _ = vm.run();
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_604_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(604)));
    match vm.run() {
        VMResult::Ok(Value::Status(604)) => {}
        other => panic!("exit code 604 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_str_ge_after_subshell_end_status605() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let chunk = {
        let mut cb = ChunkBuilder::new();
        let a = cb.add_constant(Value::str("z"));
        let bidx = cb.add_constant(Value::str("a"));
        cb.emit(Op::SubshellEnd, 1);
        cb.emit(Op::LoadConst(a), 1);
        cb.emit(Op::LoadConst(bidx), 1);
        cb.emit(Op::CallBuiltin(100, 0), 1);
        cb.emit(Op::StrGe, 1);
        cb.emit(Op::CallBuiltin(200, 0), 1);
        cb.build()
    };
    let mut vm = VM::new(chunk);
    vm.set_shell_host(Box::new(StatusReturningHost(605)));
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 605);
}

#[test]
fn subshell_end_exit_code_605_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(605)));
    match vm.run() {
        VMResult::Ok(Value::Status(605)) => {}
        other => panic!("exit code 605 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_status_minus_eighty_six_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(-86)));
    match vm.run() {
        VMResult::Ok(Value::Status(-86)) => {}
        other => panic!("status -86 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_606_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(606)));
    match vm.run() {
        VMResult::Ok(Value::Status(606)) => {}
        other => panic!("exit code 606 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_subshell_then_pipeline_getstatus_reads_pipeline_607() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::PipelineBegin(2), 1);
    b.emit(Op::PipelineEnd, 1);
    b.emit(Op::GetStatus, 1);
    struct SubshellThenPipelineHost607;
    impl ShellHost for SubshellThenPipelineHost607 {
        fn subshell_end(&mut self) -> Option<i32> {
            Some(13)
        }
        fn pipeline_end(&mut self) -> i32 {
            607
        }
    }
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(SubshellThenPipelineHost607));
    match vm.run() {
        VMResult::Ok(Value::Status(607)) => {}
        other => panic!("GetStatus after subshell+pipeline MUST read pipeline 607, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_607_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(607)));
    match vm.run() {
        VMResult::Ok(Value::Status(607)) => {}
        other => panic!("exit code 607 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_eighty_four_end_incrementing_from_ten() {
    let mut b = ChunkBuilder::new();
    for _ in 0..84 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(IncrementingSubshellHost { next: 10 }));
    match vm.run() {
        VMResult::Ok(Value::Status(93)) => {}
        other => panic!("eighty-fourth subshell_end(Some(93)) MUST win, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_608_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(608)));
    match vm.run() {
        VMResult::Ok(Value::Status(608)) => {}
        other => panic!("exit code 608 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_one_hundred_four_consecutive_runs_without_reset_still_blocked() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    for _ in 0..104 {
        let _ = vm.run();
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_609_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(609)));
    match vm.run() {
        VMResult::Ok(Value::Status(609)) => {}
        other => panic!("exit code 609 MUST propagate, got {:?}", other),
    }
}

// ─── Pin tests batch W (handwritten) ──────────────────────────────────────

#[test]
fn subshell_end_exit_code_610_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(610)));
    match vm.run() {
        VMResult::Ok(Value::Status(610)) => {}
        other => panic!("exit code 610 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_one_hundred_five_consecutive_runs_without_reset_still_blocked() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    for _ in 0..105 {
        let _ = vm.run();
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_611_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(611)));
    match vm.run() {
        VMResult::Ok(Value::Status(611)) => {}
        other => panic!("exit code 611 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_one_hundred_six_consecutive_runs_without_reset_still_blocked() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    for _ in 0..106 {
        let _ = vm.run();
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_612_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(612)));
    match vm.run() {
        VMResult::Ok(Value::Status(612)) => {}
        other => panic!("exit code 612 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_one_hundred_seven_consecutive_runs_without_reset_still_blocked() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    for _ in 0..107 {
        let _ = vm.run();
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_613_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(613)));
    match vm.run() {
        VMResult::Ok(Value::Status(613)) => {}
        other => panic!("exit code 613 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_eighty_five_end_incrementing_from_two() {
    let mut b = ChunkBuilder::new();
    for _ in 0..85 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(IncrementingSubshellHost { next: 2 }));
    match vm.run() {
        VMResult::Ok(Value::Status(86)) => {}
        other => panic!("eighty-fifth subshell_end(Some(86)) MUST win, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_614_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(614)));
    match vm.run() {
        VMResult::Ok(Value::Status(614)) => {}
        other => panic!("exit code 614 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_status_minus_eighty_seven_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(-87)));
    match vm.run() {
        VMResult::Ok(Value::Status(-87)) => {}
        other => panic!("status -87 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_615_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(615)));
    match vm.run() {
        VMResult::Ok(Value::Status(615)) => {}
        other => panic!("exit code 615 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_status_minus_eighty_eight_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(-88)));
    match vm.run() {
        VMResult::Ok(Value::Status(-88)) => {}
        other => panic!("status -88 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_616_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(616)));
    match vm.run() {
        VMResult::Ok(Value::Status(616)) => {}
        other => panic!("exit code 616 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_thirty_seventh_some_host_applies_on_thirty_seventh_end() {
    let mut b = ChunkBuilder::new();
    for _ in 0..37 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    struct ThirtySeventhSomeHost {
        calls: u32,
    }
    impl ShellHost for ThirtySeventhSomeHost {
        fn subshell_end(&mut self) -> Option<i32> {
            self.calls += 1;
            if self.calls == 37 {
                Some(617)
            } else {
                None
            }
        }
    }
    let mut vm = VM::new(b.build());
    vm.last_status = 3;
    vm.set_shell_host(Box::new(ThirtySeventhSomeHost { calls: 0 }));
    match vm.run() {
        VMResult::Ok(Value::Status(617)) => {}
        other => panic!("thirty-seventh subshell_end(Some(617)) MUST apply, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_617_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(617)));
    match vm.run() {
        VMResult::Ok(Value::Status(617)) => {}
        other => panic!("exit code 617 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_halt_reset_subshell_end_eighteenth_run_applies() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    SUBSHELL_END_CALLS.store(0, Ordering::SeqCst);
    let sub = {
        let mut b = ChunkBuilder::new();
        b.emit(Op::SubshellEnd, 1);
        b.emit(Op::CallBuiltin(100, 0), 1);
        b.build()
    };
    let end_only = {
        let mut b = ChunkBuilder::new();
        b.emit(Op::SubshellEnd, 1);
        b.build()
    };
    let mut vm = VM::new(sub.clone());
    vm.set_shell_host(Box::new(CountingSubshellEndHost(618)));
    vm.register_builtin(100, builtin_request_halt);
    let _ = vm.run();
    vm.reset(sub);
    let _ = vm.run();
    for _ in 0..15 {
        vm.reset(end_only.clone());
        let _ = vm.run();
    }
    vm.reset(end_only);
    let _ = vm.run();
    assert_eq!(SUBSHELL_END_CALLS.load(Ordering::SeqCst), 18);
    assert_eq!(vm.last_status, 618);
}

#[test]
fn subshell_end_exit_code_618_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(618)));
    match vm.run() {
        VMResult::Ok(Value::Status(618)) => {}
        other => panic!("exit code 618 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_string_len_after_subshell_end_status619() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let chunk = {
        let mut cb = ChunkBuilder::new();
        let s = cb.add_constant(Value::str("abc"));
        cb.emit(Op::SubshellEnd, 1);
        cb.emit(Op::LoadConst(s), 1);
        cb.emit(Op::CallBuiltin(100, 0), 1);
        cb.emit(Op::StringLen, 1);
        cb.emit(Op::CallBuiltin(200, 0), 1);
        cb.build()
    };
    let mut vm = VM::new(chunk);
    vm.set_shell_host(Box::new(StatusReturningHost(619)));
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 619);
}

#[test]
fn subshell_end_exit_code_619_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(619)));
    match vm.run() {
        VMResult::Ok(Value::Status(619)) => {}
        other => panic!("exit code 619 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_fourteen_begins_twelve_ends_last_end_status_wins() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    SUBSHELL_BEGIN_CALLS.store(0, Ordering::SeqCst);
    SUBSHELL_END_CALLS.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    for _ in 0..14 {
        b.emit(Op::SubshellBegin, 1);
    }
    for _ in 0..12 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    struct CountingBeginEndHost620;
    impl ShellHost for CountingBeginEndHost620 {
        fn subshell_begin(&mut self) {
            SUBSHELL_BEGIN_CALLS.fetch_add(1, Ordering::SeqCst);
        }
        fn subshell_end(&mut self) -> Option<i32> {
            SUBSHELL_END_CALLS.fetch_add(1, Ordering::SeqCst);
            Some(620)
        }
    }
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(CountingBeginEndHost620));
    match vm.run() {
        VMResult::Ok(Value::Status(620)) => {}
        other => panic!("twelfth subshell_end(Some(620)) MUST win, got {:?}", other),
    }
    assert_eq!(SUBSHELL_BEGIN_CALLS.load(Ordering::SeqCst), 14);
    assert_eq!(SUBSHELL_END_CALLS.load(Ordering::SeqCst), 12);
}

#[test]
fn subshell_end_exit_code_620_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(620)));
    match vm.run() {
        VMResult::Ok(Value::Status(620)) => {}
        other => panic!("exit code 620 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_string_repeat_after_subshell_end_status621() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let chunk = {
        let mut cb = ChunkBuilder::new();
        let s = cb.add_constant(Value::str("x"));
        cb.emit(Op::SubshellEnd, 1);
        cb.emit(Op::LoadConst(s), 1);
        cb.emit(Op::LoadInt(3), 1);
        cb.emit(Op::CallBuiltin(100, 0), 1);
        cb.emit(Op::StringRepeat, 1);
        cb.emit(Op::CallBuiltin(200, 0), 1);
        cb.build()
    };
    let mut vm = VM::new(chunk);
    vm.set_shell_host(Box::new(StatusReturningHost(621)));
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 621);
}

#[test]
fn subshell_end_exit_code_621_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(621)));
    match vm.run() {
        VMResult::Ok(Value::Status(621)) => {}
        other => panic!("exit code 621 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_fifteen_reset_runs_same_host_same_status() {
    let chunk = {
        let mut b = ChunkBuilder::new();
        b.emit(Op::SubshellEnd, 1);
        b.emit(Op::GetStatus, 1);
        b.build()
    };
    let mut vm = VM::new(chunk.clone());
    vm.set_shell_host(Box::new(StatusReturningHost(622)));
    for _ in 0..15 {
        match vm.run() {
            VMResult::Ok(Value::Status(622)) => {}
            other => panic!("reset run MUST return Status(622), got {:?}", other),
        }
        vm.reset(chunk.clone());
    }
}

#[test]
fn subshell_end_exit_code_622_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(622)));
    match vm.run() {
        VMResult::Ok(Value::Status(622)) => {}
        other => panic!("exit code 622 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_range_after_subshell_end_status623() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::LoadInt(4), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::Range, 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(623)));
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 623);
}

#[test]
fn subshell_end_exit_code_623_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(623)));
    match vm.run() {
        VMResult::Ok(Value::Status(623)) => {}
        other => panic!("exit code 623 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_eighty_six_end_incrementing_from_nine() {
    let mut b = ChunkBuilder::new();
    for _ in 0..86 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(IncrementingSubshellHost { next: 9 }));
    match vm.run() {
        VMResult::Ok(Value::Status(94)) => {}
        other => panic!("eighty-sixth subshell_end(Some(94)) MUST win, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_624_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(624)));
    match vm.run() {
        VMResult::Ok(Value::Status(624)) => {}
        other => panic!("exit code 624 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_range_step_after_subshell_end_status625() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::LoadInt(0), 1);
    b.emit(Op::LoadInt(6), 1);
    b.emit(Op::LoadInt(2), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::RangeStep, 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(625)));
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 625);
}

#[test]
fn subshell_end_exit_code_625_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(625)));
    match vm.run() {
        VMResult::Ok(Value::Status(625)) => {}
        other => panic!("exit code 625 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_one_hundred_eight_consecutive_runs_without_reset_still_blocked() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    for _ in 0..108 {
        let _ = vm.run();
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_626_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(626)));
    match vm.run() {
        VMResult::Ok(Value::Status(626)) => {}
        other => panic!("exit code 626 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_one_hundred_nine_consecutive_runs_without_reset_still_blocked() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    for _ in 0..109 {
        let _ = vm.run();
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_627_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(627)));
    match vm.run() {
        VMResult::Ok(Value::Status(627)) => {}
        other => panic!("exit code 627 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_make_hash_after_subshell_end_status628() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let chunk = {
        let mut cb = ChunkBuilder::new();
        let k = cb.add_constant(Value::str("a"));
        let v = cb.add_constant(Value::str("b"));
        cb.emit(Op::SubshellEnd, 1);
        cb.emit(Op::LoadConst(k), 1);
        cb.emit(Op::LoadConst(v), 1);
        cb.emit(Op::CallBuiltin(100, 0), 1);
        cb.emit(Op::MakeHash(1), 1);
        cb.emit(Op::CallBuiltin(200, 0), 1);
        cb.build()
    };
    let mut vm = VM::new(chunk);
    vm.set_shell_host(Box::new(StatusReturningHost(628)));
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 628);
}

#[test]
fn subshell_end_exit_code_628_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(628)));
    match vm.run() {
        VMResult::Ok(Value::Status(628)) => {}
        other => panic!("exit code 628 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_alternating_status_eighteenth_none_keeps_ninth() {
    let mut b = ChunkBuilder::new();
    for _ in 0..18 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(AlternatingStatusHost { calls: 0 }));
    match vm.run() {
        VMResult::Ok(Value::Status(9)) => {}
        other => panic!("AlternatingStatusHost eighteenth None MUST keep Some(9), got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_629_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(629)));
    match vm.run() {
        VMResult::Ok(Value::Status(629)) => {}
        other => panic!("exit code 629 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_none_host_eighteen_ends_leaves_initial_status() {
    let mut b = ChunkBuilder::new();
    for _ in 0..18 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.last_status = 630;
    vm.set_shell_host(Box::new(NoneHost));
    match vm.run() {
        VMResult::Ok(Value::Status(630)) => {}
        other => panic!("NoneHost eighteen ends MUST leave last_status 630, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_630_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(630)));
    match vm.run() {
        VMResult::Ok(Value::Status(630)) => {}
        other => panic!("exit code 630 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_first_some_then_none_host_eighteenth_none_keeps_first() {
    let mut b = ChunkBuilder::new();
    for _ in 0..18 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(FirstSomeThenNoneHost { calls: 0, status: 631 }));
    match vm.run() {
        VMResult::Ok(Value::Status(631)) => {}
        other => panic!("FirstSomeThenNoneHost eighteenth None MUST keep first Some(631), got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_631_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(631)));
    match vm.run() {
        VMResult::Ok(Value::Status(631)) => {}
        other => panic!("exit code 631 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_status_minus_eighty_nine_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(-89)));
    match vm.run() {
        VMResult::Ok(Value::Status(-89)) => {}
        other => panic!("status -89 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_632_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(632)));
    match vm.run() {
        VMResult::Ok(Value::Status(632)) => {}
        other => panic!("exit code 632 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_setstatus_after_subshell_end_status633() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::LoadInt(99), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::SetStatus, 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(633)));
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 633);
}

#[test]
fn subshell_end_exit_code_633_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(633)));
    match vm.run() {
        VMResult::Ok(Value::Status(633)) => {}
        other => panic!("exit code 633 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_eighty_seven_end_incrementing_from_four() {
    let mut b = ChunkBuilder::new();
    for _ in 0..87 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(IncrementingSubshellHost { next: 4 }));
    match vm.run() {
        VMResult::Ok(Value::Status(90)) => {}
        other => panic!("eighty-seventh subshell_end(Some(90)) MUST win, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_634_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(634)));
    match vm.run() {
        VMResult::Ok(Value::Status(634)) => {}
        other => panic!("exit code 634 MUST propagate, got {:?}", other),
    }
}

// ─── Pin tests batch X (handwritten) ──────────────────────────────────────

#[test]
fn subshell_end_exit_code_635_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(635)));
    match vm.run() {
        VMResult::Ok(Value::Status(635)) => {}
        other => panic!("exit code 635 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_one_hundred_ten_consecutive_runs_without_reset_still_blocked() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    for _ in 0..110 {
        let _ = vm.run();
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_636_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(636)));
    match vm.run() {
        VMResult::Ok(Value::Status(636)) => {}
        other => panic!("exit code 636 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_one_hundred_eleven_consecutive_runs_without_reset_still_blocked() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    for _ in 0..111 {
        let _ = vm.run();
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_637_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(637)));
    match vm.run() {
        VMResult::Ok(Value::Status(637)) => {}
        other => panic!("exit code 637 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_one_hundred_twelve_consecutive_runs_without_reset_still_blocked() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    for _ in 0..112 {
        let _ = vm.run();
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_638_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(638)));
    match vm.run() {
        VMResult::Ok(Value::Status(638)) => {}
        other => panic!("exit code 638 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_eighty_eight_end_incrementing_from_eleven() {
    let mut b = ChunkBuilder::new();
    for _ in 0..88 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(IncrementingSubshellHost { next: 11 }));
    match vm.run() {
        VMResult::Ok(Value::Status(98)) => {}
        other => panic!("eighty-eighth subshell_end(Some(98)) MUST win, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_639_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(639)));
    match vm.run() {
        VMResult::Ok(Value::Status(639)) => {}
        other => panic!("exit code 639 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_status_minus_ninety_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(-90)));
    match vm.run() {
        VMResult::Ok(Value::Status(-90)) => {}
        other => panic!("status -90 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_640_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(640)));
    match vm.run() {
        VMResult::Ok(Value::Status(640)) => {}
        other => panic!("exit code 640 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_status_minus_ninety_one_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(-91)));
    match vm.run() {
        VMResult::Ok(Value::Status(-91)) => {}
        other => panic!("status -91 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_641_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(641)));
    match vm.run() {
        VMResult::Ok(Value::Status(641)) => {}
        other => panic!("exit code 641 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_thirty_eighth_some_host_applies_on_thirty_eighth_end() {
    let mut b = ChunkBuilder::new();
    for _ in 0..38 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    struct ThirtyEighthSomeHost {
        calls: u32,
    }
    impl ShellHost for ThirtyEighthSomeHost {
        fn subshell_end(&mut self) -> Option<i32> {
            self.calls += 1;
            if self.calls == 38 {
                Some(642)
            } else {
                None
            }
        }
    }
    let mut vm = VM::new(b.build());
    vm.last_status = 1;
    vm.set_shell_host(Box::new(ThirtyEighthSomeHost { calls: 0 }));
    match vm.run() {
        VMResult::Ok(Value::Status(642)) => {}
        other => panic!("thirty-eighth subshell_end(Some(642)) MUST apply, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_642_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(642)));
    match vm.run() {
        VMResult::Ok(Value::Status(642)) => {}
        other => panic!("exit code 642 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_halt_reset_subshell_end_nineteenth_run_applies() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    SUBSHELL_END_CALLS.store(0, Ordering::SeqCst);
    let sub = {
        let mut b = ChunkBuilder::new();
        b.emit(Op::SubshellEnd, 1);
        b.emit(Op::CallBuiltin(100, 0), 1);
        b.build()
    };
    let end_only = {
        let mut b = ChunkBuilder::new();
        b.emit(Op::SubshellEnd, 1);
        b.build()
    };
    let mut vm = VM::new(sub.clone());
    vm.set_shell_host(Box::new(CountingSubshellEndHost(643)));
    vm.register_builtin(100, builtin_request_halt);
    let _ = vm.run();
    vm.reset(sub);
    let _ = vm.run();
    for _ in 0..16 {
        vm.reset(end_only.clone());
        let _ = vm.run();
    }
    vm.reset(end_only);
    let _ = vm.run();
    assert_eq!(SUBSHELL_END_CALLS.load(Ordering::SeqCst), 19);
    assert_eq!(vm.last_status, 643);
}

#[test]
fn subshell_end_exit_code_643_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(643)));
    match vm.run() {
        VMResult::Ok(Value::Status(643)) => {}
        other => panic!("exit code 643 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_push_frame_after_subshell_end_status644() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::PushFrame, 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(644)));
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 644);
}

#[test]
fn subshell_end_exit_code_644_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(644)));
    match vm.run() {
        VMResult::Ok(Value::Status(644)) => {}
        other => panic!("exit code 644 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_fifteen_begins_thirteen_ends_last_end_status_wins() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    SUBSHELL_BEGIN_CALLS.store(0, Ordering::SeqCst);
    SUBSHELL_END_CALLS.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    for _ in 0..15 {
        b.emit(Op::SubshellBegin, 1);
    }
    for _ in 0..13 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    struct CountingBeginEndHost645;
    impl ShellHost for CountingBeginEndHost645 {
        fn subshell_begin(&mut self) {
            SUBSHELL_BEGIN_CALLS.fetch_add(1, Ordering::SeqCst);
        }
        fn subshell_end(&mut self) -> Option<i32> {
            SUBSHELL_END_CALLS.fetch_add(1, Ordering::SeqCst);
            Some(645)
        }
    }
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(CountingBeginEndHost645));
    match vm.run() {
        VMResult::Ok(Value::Status(645)) => {}
        other => panic!("thirteenth subshell_end(Some(645)) MUST win, got {:?}", other),
    }
    assert_eq!(SUBSHELL_BEGIN_CALLS.load(Ordering::SeqCst), 15);
    assert_eq!(SUBSHELL_END_CALLS.load(Ordering::SeqCst), 13);
}

#[test]
fn subshell_end_exit_code_645_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(645)));
    match vm.run() {
        VMResult::Ok(Value::Status(645)) => {}
        other => panic!("exit code 645 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_pop_frame_after_subshell_end_status646() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::PushFrame, 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::PopFrame, 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(646)));
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 646);
}

#[test]
fn subshell_end_exit_code_646_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(646)));
    match vm.run() {
        VMResult::Ok(Value::Status(646)) => {}
        other => panic!("exit code 646 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_sixteen_reset_runs_same_host_same_status() {
    let chunk = {
        let mut b = ChunkBuilder::new();
        b.emit(Op::SubshellEnd, 1);
        b.emit(Op::GetStatus, 1);
        b.build()
    };
    let mut vm = VM::new(chunk.clone());
    vm.set_shell_host(Box::new(StatusReturningHost(647)));
    for _ in 0..16 {
        match vm.run() {
            VMResult::Ok(Value::Status(647)) => {}
            other => panic!("reset run MUST return Status(647), got {:?}", other),
        }
        vm.reset(chunk.clone());
    }
}

#[test]
fn subshell_end_exit_code_647_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(647)));
    match vm.run() {
        VMResult::Ok(Value::Status(647)) => {}
        other => panic!("exit code 647 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_return_after_subshell_end_status648() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::Return, 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(648)));
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 648);
}

#[test]
fn subshell_end_exit_code_648_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(648)));
    match vm.run() {
        VMResult::Ok(Value::Status(648)) => {}
        other => panic!("exit code 648 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_eighty_nine_end_incrementing_from_three() {
    let mut b = ChunkBuilder::new();
    for _ in 0..89 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(IncrementingSubshellHost { next: 3 }));
    match vm.run() {
        VMResult::Ok(Value::Status(91)) => {}
        other => panic!("eighty-ninth subshell_end(Some(91)) MUST win, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_649_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(649)));
    match vm.run() {
        VMResult::Ok(Value::Status(649)) => {}
        other => panic!("exit code 649 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_return_value_after_subshell_end_status650() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::LoadInt(42), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::ReturnValue, 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(650)));
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 650);
}

#[test]
fn subshell_end_exit_code_650_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(650)));
    match vm.run() {
        VMResult::Ok(Value::Status(650)) => {}
        other => panic!("exit code 650 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_negative_then_zero_host_last_zero_wins_status651() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    struct NegativeThenZeroHost651 {
        calls: u32,
    }
    impl ShellHost for NegativeThenZeroHost651 {
        fn subshell_end(&mut self) -> Option<i32> {
            self.calls += 1;
            if self.calls == 1 {
                Some(-92)
            } else {
                Some(0)
            }
        }
    }
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(NegativeThenZeroHost651 { calls: 0 }));
    match vm.run() {
        VMResult::Ok(Value::Status(0)) => {}
        other => panic!("second subshell_end(Some(0)) MUST win over -92, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_651_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(651)));
    match vm.run() {
        VMResult::Ok(Value::Status(651)) => {}
        other => panic!("exit code 651 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_one_hundred_thirteen_consecutive_runs_without_reset_still_blocked() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    for _ in 0..113 {
        let _ = vm.run();
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_652_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(652)));
    match vm.run() {
        VMResult::Ok(Value::Status(652)) => {}
        other => panic!("exit code 652 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_thirty_ninth_some_host_applies_on_thirty_ninth_end() {
    let mut b = ChunkBuilder::new();
    for _ in 0..39 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    struct ThirtyNinthSomeHost {
        calls: u32,
    }
    impl ShellHost for ThirtyNinthSomeHost {
        fn subshell_end(&mut self) -> Option<i32> {
            self.calls += 1;
            if self.calls == 39 {
                Some(653)
            } else {
                None
            }
        }
    }
    let mut vm = VM::new(b.build());
    vm.last_status = 9;
    vm.set_shell_host(Box::new(ThirtyNinthSomeHost { calls: 0 }));
    match vm.run() {
        VMResult::Ok(Value::Status(653)) => {}
        other => panic!("thirty-ninth subshell_end(Some(653)) MUST apply, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_653_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(653)));
    match vm.run() {
        VMResult::Ok(Value::Status(653)) => {}
        other => panic!("exit code 653 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_alternating_status_nineteenth_none_keeps_ninth() {
    let mut b = ChunkBuilder::new();
    for _ in 0..19 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(AlternatingStatusHost { calls: 0 }));
    match vm.run() {
        VMResult::Ok(Value::Status(9)) => {}
        other => panic!("AlternatingStatusHost nineteenth None MUST keep Some(9), got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_654_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(654)));
    match vm.run() {
        VMResult::Ok(Value::Status(654)) => {}
        other => panic!("exit code 654 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_pipeline_begin_after_subshell_end_status655() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::PipelineBegin(2), 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(655)));
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 655);
}

#[test]
fn subshell_end_exit_code_655_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(655)));
    match vm.run() {
        VMResult::Ok(Value::Status(655)) => {}
        other => panic!("exit code 655 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_pipeline_stage_after_subshell_end_status656() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::PipelineBegin(2), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::PipelineStage, 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(656)));
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 656);
}

#[test]
fn subshell_end_exit_code_656_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(656)));
    match vm.run() {
        VMResult::Ok(Value::Status(656)) => {}
        other => panic!("exit code 656 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_ninety_end_incrementing_from_five() {
    let mut b = ChunkBuilder::new();
    for _ in 0..90 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(IncrementingSubshellHost { next: 5 }));
    match vm.run() {
        VMResult::Ok(Value::Status(94)) => {}
        other => panic!("ninetieth subshell_end(Some(94)) MUST win, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_657_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(657)));
    match vm.run() {
        VMResult::Ok(Value::Status(657)) => {}
        other => panic!("exit code 657 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_pipeline_end_after_subshell_end_status658() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::PipelineBegin(2), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::PipelineEnd, 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(658)));
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 658);
}

#[test]
fn subshell_end_exit_code_658_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(658)));
    match vm.run() {
        VMResult::Ok(Value::Status(658)) => {}
        other => panic!("exit code 658 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_one_hundred_fourteen_consecutive_runs_without_reset_still_blocked() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    for _ in 0..114 {
        let _ = vm.run();
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_659_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(659)));
    match vm.run() {
        VMResult::Ok(Value::Status(659)) => {}
        other => panic!("exit code 659 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_fortieth_some_host_applies_on_fortieth_end() {
    let mut b = ChunkBuilder::new();
    for _ in 0..40 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    struct FortiethSomeHost {
        calls: u32,
    }
    impl ShellHost for FortiethSomeHost {
        fn subshell_end(&mut self) -> Option<i32> {
            self.calls += 1;
            if self.calls == 40 {
                Some(660)
            } else {
                None
            }
        }
    }
    let mut vm = VM::new(b.build());
    vm.last_status = 2;
    vm.set_shell_host(Box::new(FortiethSomeHost { calls: 0 }));
    match vm.run() {
        VMResult::Ok(Value::Status(660)) => {}
        other => panic!("fortieth subshell_end(Some(660)) MUST apply, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_660_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(660)));
    match vm.run() {
        VMResult::Ok(Value::Status(660)) => {}
        other => panic!("exit code 660 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_pipeline_then_subshell_getstatus_reads_subshell_661() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::PipelineBegin(2), 1);
    b.emit(Op::PipelineEnd, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    struct PipelineSubshellHost661;
    impl ShellHost for PipelineSubshellHost661 {
        fn pipeline_end(&mut self) -> i32 {
            5
        }
        fn subshell_end(&mut self) -> Option<i32> {
            Some(661)
        }
    }
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(PipelineSubshellHost661));
    match vm.run() {
        VMResult::Ok(Value::Status(661)) => {}
        other => panic!("GetStatus after pipeline+subshell MUST read subshell 661, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_661_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(661)));
    match vm.run() {
        VMResult::Ok(Value::Status(661)) => {}
        other => panic!("exit code 661 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_subshell_then_pipeline_getstatus_reads_pipeline_662() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::PipelineBegin(2), 1);
    b.emit(Op::PipelineEnd, 1);
    b.emit(Op::GetStatus, 1);
    struct SubshellThenPipelineHost662;
    impl ShellHost for SubshellThenPipelineHost662 {
        fn subshell_end(&mut self) -> Option<i32> {
            Some(14)
        }
        fn pipeline_end(&mut self) -> i32 {
            662
        }
    }
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(SubshellThenPipelineHost662));
    match vm.run() {
        VMResult::Ok(Value::Status(662)) => {}
        other => panic!("GetStatus after subshell+pipeline MUST read pipeline 662, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_662_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(662)));
    match vm.run() {
        VMResult::Ok(Value::Status(662)) => {}
        other => panic!("exit code 662 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_status_minus_ninety_three_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(-93)));
    match vm.run() {
        VMResult::Ok(Value::Status(-93)) => {}
        other => panic!("status -93 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_663_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(663)));
    match vm.run() {
        VMResult::Ok(Value::Status(663)) => {}
        other => panic!("exit code 663 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_ninety_one_end_incrementing_from_six() {
    let mut b = ChunkBuilder::new();
    for _ in 0..91 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(IncrementingSubshellHost { next: 6 }));
    match vm.run() {
        VMResult::Ok(Value::Status(96)) => {}
        other => panic!("ninety-first subshell_end(Some(96)) MUST win, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_664_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(664)));
    match vm.run() {
        VMResult::Ok(Value::Status(664)) => {}
        other => panic!("exit code 664 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_one_hundred_fifteen_consecutive_runs_without_reset_still_blocked() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    for _ in 0..115 {
        let _ = vm.run();
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_665_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(665)));
    match vm.run() {
        VMResult::Ok(Value::Status(665)) => {}
        other => panic!("exit code 665 MUST propagate, got {:?}", other),
    }
}

// ─── Pin tests batch Y (handwritten) ──────────────────────────────────────

#[test]
fn subshell_end_exit_code_666_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(666)));
    match vm.run() {
        VMResult::Ok(Value::Status(666)) => {}
        other => panic!("exit code 666 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_one_hundred_sixteen_consecutive_runs_without_reset_still_blocked() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    for _ in 0..116 {
        let _ = vm.run();
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_667_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(667)));
    match vm.run() {
        VMResult::Ok(Value::Status(667)) => {}
        other => panic!("exit code 667 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_one_hundred_seventeen_consecutive_runs_without_reset_still_blocked() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    for _ in 0..117 {
        let _ = vm.run();
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_668_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(668)));
    match vm.run() {
        VMResult::Ok(Value::Status(668)) => {}
        other => panic!("exit code 668 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_one_hundred_eighteen_consecutive_runs_without_reset_still_blocked() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    for _ in 0..118 {
        let _ = vm.run();
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_669_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(669)));
    match vm.run() {
        VMResult::Ok(Value::Status(669)) => {}
        other => panic!("exit code 669 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_ninety_two_end_incrementing_from_seven() {
    let mut b = ChunkBuilder::new();
    for _ in 0..92 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(IncrementingSubshellHost { next: 7 }));
    match vm.run() {
        VMResult::Ok(Value::Status(98)) => {}
        other => panic!("ninety-second subshell_end(Some(98)) MUST win, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_670_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(670)));
    match vm.run() {
        VMResult::Ok(Value::Status(670)) => {}
        other => panic!("exit code 670 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_status_minus_ninety_four_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(-94)));
    match vm.run() {
        VMResult::Ok(Value::Status(-94)) => {}
        other => panic!("status -94 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_671_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(671)));
    match vm.run() {
        VMResult::Ok(Value::Status(671)) => {}
        other => panic!("exit code 671 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_status_minus_ninety_five_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(-95)));
    match vm.run() {
        VMResult::Ok(Value::Status(-95)) => {}
        other => panic!("status -95 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_672_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(672)));
    match vm.run() {
        VMResult::Ok(Value::Status(672)) => {}
        other => panic!("exit code 672 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_forty_first_some_host_applies_on_forty_first_end() {
    let mut b = ChunkBuilder::new();
    for _ in 0..41 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    struct FortyFirstSomeHost {
        calls: u32,
    }
    impl ShellHost for FortyFirstSomeHost {
        fn subshell_end(&mut self) -> Option<i32> {
            self.calls += 1;
            if self.calls == 41 {
                Some(673)
            } else {
                None
            }
        }
    }
    let mut vm = VM::new(b.build());
    vm.last_status = 4;
    vm.set_shell_host(Box::new(FortyFirstSomeHost { calls: 0 }));
    match vm.run() {
        VMResult::Ok(Value::Status(673)) => {}
        other => panic!("forty-first subshell_end(Some(673)) MUST apply, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_673_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(673)));
    match vm.run() {
        VMResult::Ok(Value::Status(673)) => {}
        other => panic!("exit code 673 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_halt_reset_subshell_end_twentieth_run_applies() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    SUBSHELL_END_CALLS.store(0, Ordering::SeqCst);
    let sub = {
        let mut b = ChunkBuilder::new();
        b.emit(Op::SubshellEnd, 1);
        b.emit(Op::CallBuiltin(100, 0), 1);
        b.build()
    };
    let end_only = {
        let mut b = ChunkBuilder::new();
        b.emit(Op::SubshellEnd, 1);
        b.build()
    };
    let mut vm = VM::new(sub.clone());
    vm.set_shell_host(Box::new(CountingSubshellEndHost(674)));
    vm.register_builtin(100, builtin_request_halt);
    let _ = vm.run();
    vm.reset(sub);
    let _ = vm.run();
    for _ in 0..17 {
        vm.reset(end_only.clone());
        let _ = vm.run();
    }
    vm.reset(end_only);
    let _ = vm.run();
    assert_eq!(SUBSHELL_END_CALLS.load(Ordering::SeqCst), 20);
    assert_eq!(vm.last_status, 674);
}

#[test]
fn subshell_end_exit_code_674_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(674)));
    match vm.run() {
        VMResult::Ok(Value::Status(674)) => {}
        other => panic!("exit code 674 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_subshell_begin_after_subshell_end_status675() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(675)));
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 675);
}

#[test]
fn subshell_end_exit_code_675_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(675)));
    match vm.run() {
        VMResult::Ok(Value::Status(675)) => {}
        other => panic!("exit code 675 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_sixteen_begins_fourteen_ends_last_end_status_wins() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    SUBSHELL_BEGIN_CALLS.store(0, Ordering::SeqCst);
    SUBSHELL_END_CALLS.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    for _ in 0..16 {
        b.emit(Op::SubshellBegin, 1);
    }
    for _ in 0..14 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    struct CountingBeginEndHost676;
    impl ShellHost for CountingBeginEndHost676 {
        fn subshell_begin(&mut self) {
            SUBSHELL_BEGIN_CALLS.fetch_add(1, Ordering::SeqCst);
        }
        fn subshell_end(&mut self) -> Option<i32> {
            SUBSHELL_END_CALLS.fetch_add(1, Ordering::SeqCst);
            Some(676)
        }
    }
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(CountingBeginEndHost676));
    match vm.run() {
        VMResult::Ok(Value::Status(676)) => {}
        other => panic!("fourteenth subshell_end(Some(676)) MUST win, got {:?}", other),
    }
    assert_eq!(SUBSHELL_BEGIN_CALLS.load(Ordering::SeqCst), 16);
    assert_eq!(SUBSHELL_END_CALLS.load(Ordering::SeqCst), 14);
}

#[test]
fn subshell_end_exit_code_676_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(676)));
    match vm.run() {
        VMResult::Ok(Value::Status(676)) => {}
        other => panic!("exit code 676 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_nop_after_subshell_end_status677() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::Nop, 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(677)));
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 677);
}

#[test]
fn subshell_end_exit_code_677_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(677)));
    match vm.run() {
        VMResult::Ok(Value::Status(677)) => {}
        other => panic!("exit code 677 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_seventeen_reset_runs_same_host_same_status() {
    let chunk = {
        let mut b = ChunkBuilder::new();
        b.emit(Op::SubshellEnd, 1);
        b.emit(Op::GetStatus, 1);
        b.build()
    };
    let mut vm = VM::new(chunk.clone());
    vm.set_shell_host(Box::new(StatusReturningHost(678)));
    for _ in 0..17 {
        match vm.run() {
            VMResult::Ok(Value::Status(678)) => {}
            other => panic!("reset run MUST return Status(678), got {:?}", other),
        }
        vm.reset(chunk.clone());
    }
}

#[test]
fn subshell_end_exit_code_678_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(678)));
    match vm.run() {
        VMResult::Ok(Value::Status(678)) => {}
        other => panic!("exit code 678 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_str_cmp_after_subshell_end_status679() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let chunk = {
        let mut cb = ChunkBuilder::new();
        let a = cb.add_constant(Value::str("cat"));
        let bidx = cb.add_constant(Value::str("dog"));
        cb.emit(Op::SubshellEnd, 1);
        cb.emit(Op::LoadConst(a), 1);
        cb.emit(Op::LoadConst(bidx), 1);
        cb.emit(Op::CallBuiltin(100, 0), 1);
        cb.emit(Op::StrCmp, 1);
        cb.emit(Op::CallBuiltin(200, 0), 1);
        cb.build()
    };
    let mut vm = VM::new(chunk);
    vm.set_shell_host(Box::new(StatusReturningHost(679)));
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 679);
}

#[test]
fn subshell_end_exit_code_679_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(679)));
    match vm.run() {
        VMResult::Ok(Value::Status(679)) => {}
        other => panic!("exit code 679 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_ninety_three_end_incrementing_from_eight() {
    let mut b = ChunkBuilder::new();
    for _ in 0..93 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(IncrementingSubshellHost { next: 8 }));
    match vm.run() {
        VMResult::Ok(Value::Status(100)) => {}
        other => panic!("ninety-third subshell_end(Some(100)) MUST win, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_680_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(680)));
    match vm.run() {
        VMResult::Ok(Value::Status(680)) => {}
        other => panic!("exit code 680 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_second_getstatus_after_subshell_end_status681() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::GetStatus, 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(681)));
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 681);
}

#[test]
fn subshell_end_exit_code_681_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(681)));
    match vm.run() {
        VMResult::Ok(Value::Status(681)) => {}
        other => panic!("exit code 681 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_one_hundred_nineteen_consecutive_runs_without_reset_still_blocked() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    for _ in 0..119 {
        let _ = vm.run();
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_682_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(682)));
    match vm.run() {
        VMResult::Ok(Value::Status(682)) => {}
        other => panic!("exit code 682 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_one_hundred_twenty_consecutive_runs_without_reset_still_blocked() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    for _ in 0..120 {
        let _ = vm.run();
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_683_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(683)));
    match vm.run() {
        VMResult::Ok(Value::Status(683)) => {}
        other => panic!("exit code 683 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_alternating_status_twentieth_none_keeps_ninth() {
    let mut b = ChunkBuilder::new();
    for _ in 0..20 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(AlternatingStatusHost { calls: 0 }));
    match vm.run() {
        VMResult::Ok(Value::Status(9)) => {}
        other => panic!("AlternatingStatusHost twentieth None MUST keep Some(9), got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_684_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(684)));
    match vm.run() {
        VMResult::Ok(Value::Status(684)) => {}
        other => panic!("exit code 684 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_none_host_nineteen_ends_leaves_initial_status() {
    let mut b = ChunkBuilder::new();
    for _ in 0..19 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.last_status = 685;
    vm.set_shell_host(Box::new(NoneHost));
    match vm.run() {
        VMResult::Ok(Value::Status(685)) => {}
        other => panic!("NoneHost nineteen ends MUST leave last_status 685, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_685_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(685)));
    match vm.run() {
        VMResult::Ok(Value::Status(685)) => {}
        other => panic!("exit code 685 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_first_some_then_none_host_nineteenth_none_keeps_first() {
    let mut b = ChunkBuilder::new();
    for _ in 0..19 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(FirstSomeThenNoneHost { calls: 0, status: 686 }));
    match vm.run() {
        VMResult::Ok(Value::Status(686)) => {}
        other => panic!("FirstSomeThenNoneHost nineteenth None MUST keep first Some(686), got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_686_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(686)));
    match vm.run() {
        VMResult::Ok(Value::Status(686)) => {}
        other => panic!("exit code 686 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_status_minus_ninety_six_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(-96)));
    match vm.run() {
        VMResult::Ok(Value::Status(-96)) => {}
        other => panic!("status -96 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_687_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(687)));
    match vm.run() {
        VMResult::Ok(Value::Status(687)) => {}
        other => panic!("exit code 687 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_ninety_four_end_incrementing_from_one() {
    let mut b = ChunkBuilder::new();
    for _ in 0..94 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(IncrementingSubshellHost { next: 1 }));
    match vm.run() {
        VMResult::Ok(Value::Status(94)) => {}
        other => panic!("ninety-fourth subshell_end(Some(94)) MUST win, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_688_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(688)));
    match vm.run() {
        VMResult::Ok(Value::Status(688)) => {}
        other => panic!("exit code 688 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_load_const_after_subshell_end_status689() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let chunk = {
        let mut cb = ChunkBuilder::new();
        let c = cb.add_constant(Value::Int(55));
        cb.emit(Op::SubshellEnd, 1);
        cb.emit(Op::CallBuiltin(100, 0), 1);
        cb.emit(Op::LoadConst(c), 1);
        cb.emit(Op::CallBuiltin(200, 0), 1);
        cb.build()
    };
    let mut vm = VM::new(chunk);
    vm.set_shell_host(Box::new(StatusReturningHost(689)));
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 689);
}

#[test]
fn subshell_end_exit_code_689_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(689)));
    match vm.run() {
        VMResult::Ok(Value::Status(689)) => {}
        other => panic!("exit code 689 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_negative_then_zero_host_last_zero_wins_status690() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    struct NegativeThenZeroHost690 {
        calls: u32,
    }
    impl ShellHost for NegativeThenZeroHost690 {
        fn subshell_end(&mut self) -> Option<i32> {
            self.calls += 1;
            if self.calls == 1 {
                Some(-97)
            } else {
                Some(0)
            }
        }
    }
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(NegativeThenZeroHost690 { calls: 0 }));
    match vm.run() {
        VMResult::Ok(Value::Status(0)) => {}
        other => panic!("second subshell_end(Some(0)) MUST win over -97, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_690_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(690)));
    match vm.run() {
        VMResult::Ok(Value::Status(690)) => {}
        other => panic!("exit code 690 MUST propagate, got {:?}", other),
    }
}

// ─── Pin tests batch Z (handwritten) ──────────────────────────────────────

#[test]
fn subshell_end_exit_code_691_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(691)));
    match vm.run() {
        VMResult::Ok(Value::Status(691)) => {}
        other => panic!("exit code 691 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_one_hundred_twenty_one_consecutive_runs_without_reset_still_blocked() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    for _ in 0..121 {
        let _ = vm.run();
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_692_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(692)));
    match vm.run() {
        VMResult::Ok(Value::Status(692)) => {}
        other => panic!("exit code 692 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_one_hundred_twenty_two_consecutive_runs_without_reset_still_blocked() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    for _ in 0..122 {
        let _ = vm.run();
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_693_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(693)));
    match vm.run() {
        VMResult::Ok(Value::Status(693)) => {}
        other => panic!("exit code 693 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_one_hundred_twenty_three_consecutive_runs_without_reset_still_blocked() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    for _ in 0..123 {
        let _ = vm.run();
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_694_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(694)));
    match vm.run() {
        VMResult::Ok(Value::Status(694)) => {}
        other => panic!("exit code 694 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_ninety_five_end_incrementing_from_twelve() {
    let mut b = ChunkBuilder::new();
    for _ in 0..95 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(IncrementingSubshellHost { next: 12 }));
    match vm.run() {
        VMResult::Ok(Value::Status(106)) => {}
        other => panic!("ninety-fifth subshell_end(Some(106)) MUST win, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_695_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(695)));
    match vm.run() {
        VMResult::Ok(Value::Status(695)) => {}
        other => panic!("exit code 695 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_status_minus_ninety_eight_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(-98)));
    match vm.run() {
        VMResult::Ok(Value::Status(-98)) => {}
        other => panic!("status -98 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_696_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(696)));
    match vm.run() {
        VMResult::Ok(Value::Status(696)) => {}
        other => panic!("exit code 696 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_status_minus_ninety_nine_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(-99)));
    match vm.run() {
        VMResult::Ok(Value::Status(-99)) => {}
        other => panic!("status -99 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_697_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(697)));
    match vm.run() {
        VMResult::Ok(Value::Status(697)) => {}
        other => panic!("exit code 697 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_forty_second_some_host_applies_on_forty_second_end() {
    let mut b = ChunkBuilder::new();
    for _ in 0..42 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    struct FortySecondSomeHost {
        calls: u32,
    }
    impl ShellHost for FortySecondSomeHost {
        fn subshell_end(&mut self) -> Option<i32> {
            self.calls += 1;
            if self.calls == 42 {
                Some(698)
            } else {
                None
            }
        }
    }
    let mut vm = VM::new(b.build());
    vm.last_status = 6;
    vm.set_shell_host(Box::new(FortySecondSomeHost { calls: 0 }));
    match vm.run() {
        VMResult::Ok(Value::Status(698)) => {}
        other => panic!("forty-second subshell_end(Some(698)) MUST apply, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_698_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(698)));
    match vm.run() {
        VMResult::Ok(Value::Status(698)) => {}
        other => panic!("exit code 698 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_halt_reset_subshell_end_twenty_first_run_applies() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    SUBSHELL_END_CALLS.store(0, Ordering::SeqCst);
    let sub = {
        let mut b = ChunkBuilder::new();
        b.emit(Op::SubshellEnd, 1);
        b.emit(Op::CallBuiltin(100, 0), 1);
        b.build()
    };
    let end_only = {
        let mut b = ChunkBuilder::new();
        b.emit(Op::SubshellEnd, 1);
        b.build()
    };
    let mut vm = VM::new(sub.clone());
    vm.set_shell_host(Box::new(CountingSubshellEndHost(699)));
    vm.register_builtin(100, builtin_request_halt);
    let _ = vm.run();
    vm.reset(sub);
    let _ = vm.run();
    for _ in 0..18 {
        vm.reset(end_only.clone());
        let _ = vm.run();
    }
    vm.reset(end_only);
    let _ = vm.run();
    assert_eq!(SUBSHELL_END_CALLS.load(Ordering::SeqCst), 21);
    assert_eq!(vm.last_status, 699);
}

#[test]
fn subshell_end_exit_code_699_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(699)));
    match vm.run() {
        VMResult::Ok(Value::Status(699)) => {}
        other => panic!("exit code 699 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_jump_if_true_after_subshell_end_status700() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::LoadInt(5), 1);
    b.emit(Op::LoadInt(2), 1);
    b.emit(Op::NumLt, 1);
    let jmp = b.emit(Op::JumpIfTrue(0), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.patch_jump(jmp, b.current_pos());
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(700)));
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 700);
}

#[test]
fn subshell_end_exit_code_700_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(700)));
    match vm.run() {
        VMResult::Ok(Value::Status(700)) => {}
        other => panic!("exit code 700 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_seventeen_begins_fifteen_ends_last_end_status_wins() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    SUBSHELL_BEGIN_CALLS.store(0, Ordering::SeqCst);
    SUBSHELL_END_CALLS.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    for _ in 0..17 {
        b.emit(Op::SubshellBegin, 1);
    }
    for _ in 0..15 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    struct CountingBeginEndHost701;
    impl ShellHost for CountingBeginEndHost701 {
        fn subshell_begin(&mut self) {
            SUBSHELL_BEGIN_CALLS.fetch_add(1, Ordering::SeqCst);
        }
        fn subshell_end(&mut self) -> Option<i32> {
            SUBSHELL_END_CALLS.fetch_add(1, Ordering::SeqCst);
            Some(701)
        }
    }
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(CountingBeginEndHost701));
    match vm.run() {
        VMResult::Ok(Value::Status(701)) => {}
        other => panic!("fifteenth subshell_end(Some(701)) MUST win, got {:?}", other),
    }
    assert_eq!(SUBSHELL_BEGIN_CALLS.load(Ordering::SeqCst), 17);
    assert_eq!(SUBSHELL_END_CALLS.load(Ordering::SeqCst), 15);
}

#[test]
fn subshell_end_exit_code_701_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(701)));
    match vm.run() {
        VMResult::Ok(Value::Status(701)) => {}
        other => panic!("exit code 701 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_jump_if_false_after_subshell_end_status702() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::LoadInt(4), 1);
    b.emit(Op::LoadInt(4), 1);
    b.emit(Op::NumEq, 1);
    let jmp = b.emit(Op::JumpIfFalse(0), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.patch_jump(jmp, b.current_pos());
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(702)));
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 702);
}

#[test]
fn subshell_end_exit_code_702_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(702)));
    match vm.run() {
        VMResult::Ok(Value::Status(702)) => {}
        other => panic!("exit code 702 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_eighteen_reset_runs_same_host_same_status() {
    let chunk = {
        let mut b = ChunkBuilder::new();
        b.emit(Op::SubshellEnd, 1);
        b.emit(Op::GetStatus, 1);
        b.build()
    };
    let mut vm = VM::new(chunk.clone());
    vm.set_shell_host(Box::new(StatusReturningHost(703)));
    for _ in 0..18 {
        match vm.run() {
            VMResult::Ok(Value::Status(703)) => {}
            other => panic!("reset run MUST return Status(703), got {:?}", other),
        }
        vm.reset(chunk.clone());
    }
}

#[test]
fn subshell_end_exit_code_703_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(703)));
    match vm.run() {
        VMResult::Ok(Value::Status(703)) => {}
        other => panic!("exit code 703 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_jump_if_true_keep_after_subshell_end_status704() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::LoadFalse, 1);
    let jmp = b.emit(Op::JumpIfTrueKeep(0), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.patch_jump(jmp, b.current_pos());
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(704)));
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 704);
}

#[test]
fn subshell_end_exit_code_704_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(704)));
    match vm.run() {
        VMResult::Ok(Value::Status(704)) => {}
        other => panic!("exit code 704 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_ninety_six_end_incrementing_from_nine() {
    let mut b = ChunkBuilder::new();
    for _ in 0..96 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(IncrementingSubshellHost { next: 9 }));
    match vm.run() {
        VMResult::Ok(Value::Status(104)) => {}
        other => panic!("ninety-sixth subshell_end(Some(104)) MUST win, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_705_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(705)));
    match vm.run() {
        VMResult::Ok(Value::Status(705)) => {}
        other => panic!("exit code 705 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_jump_if_false_keep_after_subshell_end_status706() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::LoadTrue, 1);
    let jmp = b.emit(Op::JumpIfFalseKeep(0), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.patch_jump(jmp, b.current_pos());
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(706)));
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 706);
}

#[test]
fn subshell_end_exit_code_706_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(706)));
    match vm.run() {
        VMResult::Ok(Value::Status(706)) => {}
        other => panic!("exit code 706 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_one_hundred_twenty_four_consecutive_runs_without_reset_still_blocked() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    for _ in 0..124 {
        let _ = vm.run();
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_707_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(707)));
    match vm.run() {
        VMResult::Ok(Value::Status(707)) => {}
        other => panic!("exit code 707 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_forty_third_some_host_applies_on_forty_third_end() {
    let mut b = ChunkBuilder::new();
    for _ in 0..43 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    struct FortyThirdSomeHost {
        calls: u32,
    }
    impl ShellHost for FortyThirdSomeHost {
        fn subshell_end(&mut self) -> Option<i32> {
            self.calls += 1;
            if self.calls == 43 {
                Some(708)
            } else {
                None
            }
        }
    }
    let mut vm = VM::new(b.build());
    vm.last_status = 3;
    vm.set_shell_host(Box::new(FortyThirdSomeHost { calls: 0 }));
    match vm.run() {
        VMResult::Ok(Value::Status(708)) => {}
        other => panic!("forty-third subshell_end(Some(708)) MUST apply, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_708_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(708)));
    match vm.run() {
        VMResult::Ok(Value::Status(708)) => {}
        other => panic!("exit code 708 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_alternating_status_twenty_first_none_keeps_ninth() {
    let mut b = ChunkBuilder::new();
    for _ in 0..21 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(AlternatingStatusHost { calls: 0 }));
    match vm.run() {
        VMResult::Ok(Value::Status(9)) => {}
        other => panic!("AlternatingStatusHost twenty-first None MUST keep Some(9), got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_709_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(709)));
    match vm.run() {
        VMResult::Ok(Value::Status(709)) => {}
        other => panic!("exit code 709 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_second_subshell_end_after_subshell_end_status710() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    SUBSHELL_END_CALLS.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(CountingSubshellEndHost(710)));
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
    assert_eq!(SUBSHELL_END_CALLS.load(Ordering::SeqCst), 1);
    assert_eq!(vm.last_status, 710);
}

#[test]
fn subshell_end_exit_code_710_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(710)));
    match vm.run() {
        VMResult::Ok(Value::Status(710)) => {}
        other => panic!("exit code 710 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_ninety_seven_end_incrementing_from_four() {
    let mut b = ChunkBuilder::new();
    for _ in 0..97 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(IncrementingSubshellHost { next: 4 }));
    match vm.run() {
        VMResult::Ok(Value::Status(100)) => {}
        other => panic!("ninety-seventh subshell_end(Some(100)) MUST win, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_711_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(711)));
    match vm.run() {
        VMResult::Ok(Value::Status(711)) => {}
        other => panic!("exit code 711 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_spaceship_after_subshell_end_status712() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::LoadInt(8), 1);
    b.emit(Op::LoadInt(3), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::Spaceship, 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(712)));
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 712);
}

#[test]
fn subshell_end_exit_code_712_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(712)));
    match vm.run() {
        VMResult::Ok(Value::Status(712)) => {}
        other => panic!("exit code 712 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_none_host_twenty_ends_leaves_initial_status() {
    let mut b = ChunkBuilder::new();
    for _ in 0..20 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.last_status = 713;
    vm.set_shell_host(Box::new(NoneHost));
    match vm.run() {
        VMResult::Ok(Value::Status(713)) => {}
        other => panic!("NoneHost twenty ends MUST leave last_status 713, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_713_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(713)));
    match vm.run() {
        VMResult::Ok(Value::Status(713)) => {}
        other => panic!("exit code 713 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_first_some_then_none_host_twentieth_none_keeps_first() {
    let mut b = ChunkBuilder::new();
    for _ in 0..20 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(FirstSomeThenNoneHost { calls: 0, status: 714 }));
    match vm.run() {
        VMResult::Ok(Value::Status(714)) => {}
        other => panic!("FirstSomeThenNoneHost twentieth None MUST keep first Some(714), got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_714_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(714)));
    match vm.run() {
        VMResult::Ok(Value::Status(714)) => {}
        other => panic!("exit code 714 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_one_hundred_twenty_five_consecutive_runs_without_reset_still_blocked() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    for _ in 0..125 {
        let _ = vm.run();
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_715_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(715)));
    match vm.run() {
        VMResult::Ok(Value::Status(715)) => {}
        other => panic!("exit code 715 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_forty_fourth_some_host_applies_on_forty_fourth_end() {
    let mut b = ChunkBuilder::new();
    for _ in 0..44 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    struct FortyFourthSomeHost {
        calls: u32,
    }
    impl ShellHost for FortyFourthSomeHost {
        fn subshell_end(&mut self) -> Option<i32> {
            self.calls += 1;
            if self.calls == 44 {
                Some(716)
            } else {
                None
            }
        }
    }
    let mut vm = VM::new(b.build());
    vm.last_status = 10;
    vm.set_shell_host(Box::new(FortyFourthSomeHost { calls: 0 }));
    match vm.run() {
        VMResult::Ok(Value::Status(716)) => {}
        other => panic!("forty-fourth subshell_end(Some(716)) MUST apply, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_716_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(716)));
    match vm.run() {
        VMResult::Ok(Value::Status(716)) => {}
        other => panic!("exit code 716 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_pipeline_then_subshell_getstatus_reads_subshell_717() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::PipelineBegin(2), 1);
    b.emit(Op::PipelineEnd, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    struct PipelineSubshellHost717;
    impl ShellHost for PipelineSubshellHost717 {
        fn pipeline_end(&mut self) -> i32 {
            6
        }
        fn subshell_end(&mut self) -> Option<i32> {
            Some(717)
        }
    }
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(PipelineSubshellHost717));
    match vm.run() {
        VMResult::Ok(Value::Status(717)) => {}
        other => panic!("GetStatus after pipeline+subshell MUST read subshell 717, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_717_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(717)));
    match vm.run() {
        VMResult::Ok(Value::Status(717)) => {}
        other => panic!("exit code 717 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_subshell_then_pipeline_getstatus_reads_pipeline_718() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::PipelineBegin(2), 1);
    b.emit(Op::PipelineEnd, 1);
    b.emit(Op::GetStatus, 1);
    struct SubshellThenPipelineHost718;
    impl ShellHost for SubshellThenPipelineHost718 {
        fn subshell_end(&mut self) -> Option<i32> {
            Some(15)
        }
        fn pipeline_end(&mut self) -> i32 {
            718
        }
    }
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(SubshellThenPipelineHost718));
    match vm.run() {
        VMResult::Ok(Value::Status(718)) => {}
        other => panic!("GetStatus after subshell+pipeline MUST read pipeline 718, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_718_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(718)));
    match vm.run() {
        VMResult::Ok(Value::Status(718)) => {}
        other => panic!("exit code 718 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_status_minus_one_hundred_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(-100)));
    match vm.run() {
        VMResult::Ok(Value::Status(-100)) => {}
        other => panic!("status -100 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_719_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(719)));
    match vm.run() {
        VMResult::Ok(Value::Status(719)) => {}
        other => panic!("exit code 719 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_ninety_eight_end_incrementing_from_two() {
    let mut b = ChunkBuilder::new();
    for _ in 0..98 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(IncrementingSubshellHost { next: 2 }));
    match vm.run() {
        VMResult::Ok(Value::Status(99)) => {}
        other => panic!("ninety-eighth subshell_end(Some(99)) MUST win, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_720_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(720)));
    match vm.run() {
        VMResult::Ok(Value::Status(720)) => {}
        other => panic!("exit code 720 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_one_hundred_twenty_six_consecutive_runs_without_reset_still_blocked() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    for _ in 0..126 {
        let _ = vm.run();
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_721_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(721)));
    match vm.run() {
        VMResult::Ok(Value::Status(721)) => {}
        other => panic!("exit code 721 MUST propagate, got {:?}", other),
    }
}

// ─── Pin tests batch AA (handwritten) ─────────────────────────────────────

#[test]
fn subshell_end_exit_code_722_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(722)));
    match vm.run() {
        VMResult::Ok(Value::Status(722)) => {}
        other => panic!("exit code 722 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_one_hundred_twenty_seven_consecutive_runs_without_reset_still_blocked() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    for _ in 0..127 {
        let _ = vm.run();
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_723_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(723)));
    match vm.run() {
        VMResult::Ok(Value::Status(723)) => {}
        other => panic!("exit code 723 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_one_hundred_twenty_eight_consecutive_runs_without_reset_still_blocked() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    for _ in 0..128 {
        let _ = vm.run();
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_724_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(724)));
    match vm.run() {
        VMResult::Ok(Value::Status(724)) => {}
        other => panic!("exit code 724 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_one_hundred_twenty_nine_consecutive_runs_without_reset_still_blocked() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    for _ in 0..129 {
        let _ = vm.run();
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_725_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(725)));
    match vm.run() {
        VMResult::Ok(Value::Status(725)) => {}
        other => panic!("exit code 725 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_ninety_nine_end_incrementing_from_ten() {
    let mut b = ChunkBuilder::new();
    for _ in 0..99 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(IncrementingSubshellHost { next: 10 }));
    match vm.run() {
        VMResult::Ok(Value::Status(108)) => {}
        other => panic!("ninety-ninth subshell_end(Some(108)) MUST win, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_726_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(726)));
    match vm.run() {
        VMResult::Ok(Value::Status(726)) => {}
        other => panic!("exit code 726 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_status_minus_one_hundred_one_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(-101)));
    match vm.run() {
        VMResult::Ok(Value::Status(-101)) => {}
        other => panic!("status -101 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_727_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(727)));
    match vm.run() {
        VMResult::Ok(Value::Status(727)) => {}
        other => panic!("exit code 727 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_status_minus_one_hundred_two_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(-102)));
    match vm.run() {
        VMResult::Ok(Value::Status(-102)) => {}
        other => panic!("status -102 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_728_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(728)));
    match vm.run() {
        VMResult::Ok(Value::Status(728)) => {}
        other => panic!("exit code 728 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_forty_fifth_some_host_applies_on_forty_fifth_end() {
    let mut b = ChunkBuilder::new();
    for _ in 0..45 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    struct FortyFifthSomeHost {
        calls: u32,
    }
    impl ShellHost for FortyFifthSomeHost {
        fn subshell_end(&mut self) -> Option<i32> {
            self.calls += 1;
            if self.calls == 45 {
                Some(729)
            } else {
                None
            }
        }
    }
    let mut vm = VM::new(b.build());
    vm.last_status = 5;
    vm.set_shell_host(Box::new(FortyFifthSomeHost { calls: 0 }));
    match vm.run() {
        VMResult::Ok(Value::Status(729)) => {}
        other => panic!("forty-fifth subshell_end(Some(729)) MUST apply, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_729_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(729)));
    match vm.run() {
        VMResult::Ok(Value::Status(729)) => {}
        other => panic!("exit code 729 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_halt_reset_subshell_end_twenty_second_run_applies() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    SUBSHELL_END_CALLS.store(0, Ordering::SeqCst);
    let sub = {
        let mut b = ChunkBuilder::new();
        b.emit(Op::SubshellEnd, 1);
        b.emit(Op::CallBuiltin(100, 0), 1);
        b.build()
    };
    let end_only = {
        let mut b = ChunkBuilder::new();
        b.emit(Op::SubshellEnd, 1);
        b.build()
    };
    let mut vm = VM::new(sub.clone());
    vm.set_shell_host(Box::new(CountingSubshellEndHost(730)));
    vm.register_builtin(100, builtin_request_halt);
    let _ = vm.run();
    vm.reset(sub);
    let _ = vm.run();
    for _ in 0..19 {
        vm.reset(end_only.clone());
        let _ = vm.run();
    }
    vm.reset(end_only);
    let _ = vm.run();
    assert_eq!(SUBSHELL_END_CALLS.load(Ordering::SeqCst), 22);
    assert_eq!(vm.last_status, 730);
}

#[test]
fn subshell_end_exit_code_730_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(730)));
    match vm.run() {
        VMResult::Ok(Value::Status(730)) => {}
        other => panic!("exit code 730 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_load_int_pair_after_subshell_end_status731() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::LoadInt(11), 1);
    b.emit(Op::LoadInt(22), 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(731)));
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 731);
}

#[test]
fn subshell_end_exit_code_731_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(731)));
    match vm.run() {
        VMResult::Ok(Value::Status(731)) => {}
        other => panic!("exit code 731 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_eighteen_begins_sixteen_ends_last_end_status_wins() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    SUBSHELL_BEGIN_CALLS.store(0, Ordering::SeqCst);
    SUBSHELL_END_CALLS.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    for _ in 0..18 {
        b.emit(Op::SubshellBegin, 1);
    }
    for _ in 0..16 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    struct CountingBeginEndHost732;
    impl ShellHost for CountingBeginEndHost732 {
        fn subshell_begin(&mut self) {
            SUBSHELL_BEGIN_CALLS.fetch_add(1, Ordering::SeqCst);
        }
        fn subshell_end(&mut self) -> Option<i32> {
            SUBSHELL_END_CALLS.fetch_add(1, Ordering::SeqCst);
            Some(732)
        }
    }
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(CountingBeginEndHost732));
    match vm.run() {
        VMResult::Ok(Value::Status(732)) => {}
        other => panic!("sixteenth subshell_end(Some(732)) MUST win, got {:?}", other),
    }
    assert_eq!(SUBSHELL_BEGIN_CALLS.load(Ordering::SeqCst), 18);
    assert_eq!(SUBSHELL_END_CALLS.load(Ordering::SeqCst), 16);
}

#[test]
fn subshell_end_exit_code_732_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(732)));
    match vm.run() {
        VMResult::Ok(Value::Status(732)) => {}
        other => panic!("exit code 732 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_second_subshell_begin_after_subshell_end_status733() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    SUBSHELL_BEGIN_CALLS.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    struct CountingBeginHost733;
    impl ShellHost for CountingBeginHost733 {
        fn subshell_begin(&mut self) {
            SUBSHELL_BEGIN_CALLS.fetch_add(1, Ordering::SeqCst);
        }
        fn subshell_end(&mut self) -> Option<i32> {
            Some(733)
        }
    }
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(CountingBeginHost733));
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
    assert_eq!(SUBSHELL_BEGIN_CALLS.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 733);
}

#[test]
fn subshell_end_exit_code_733_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(733)));
    match vm.run() {
        VMResult::Ok(Value::Status(733)) => {}
        other => panic!("exit code 733 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_nineteen_reset_runs_same_host_same_status() {
    let chunk = {
        let mut b = ChunkBuilder::new();
        b.emit(Op::SubshellEnd, 1);
        b.emit(Op::GetStatus, 1);
        b.build()
    };
    let mut vm = VM::new(chunk.clone());
    vm.set_shell_host(Box::new(StatusReturningHost(734)));
    for _ in 0..19 {
        match vm.run() {
            VMResult::Ok(Value::Status(734)) => {}
            other => panic!("reset run MUST return Status(734), got {:?}", other),
        }
        vm.reset(chunk.clone());
    }
}

#[test]
fn subshell_end_exit_code_734_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(734)));
    match vm.run() {
        VMResult::Ok(Value::Status(734)) => {}
        other => panic!("exit code 734 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_num_le_after_subshell_end_status735() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::LoadInt(2), 1);
    b.emit(Op::LoadInt(8), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::NumLe, 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(735)));
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 735);
}

#[test]
fn subshell_end_exit_code_735_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(735)));
    match vm.run() {
        VMResult::Ok(Value::Status(735)) => {}
        other => panic!("exit code 735 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_one_hundred_end_incrementing_from_three() {
    let mut b = ChunkBuilder::new();
    for _ in 0..100 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(IncrementingSubshellHost { next: 3 }));
    match vm.run() {
        VMResult::Ok(Value::Status(102)) => {}
        other => panic!("one-hundredth subshell_end(Some(102)) MUST win, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_736_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(736)));
    match vm.run() {
        VMResult::Ok(Value::Status(736)) => {}
        other => panic!("exit code 736 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_num_ge_after_subshell_end_status737() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::LoadInt(9), 1);
    b.emit(Op::LoadInt(3), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::NumGe, 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(737)));
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 737);
}

#[test]
fn subshell_end_exit_code_737_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(737)));
    match vm.run() {
        VMResult::Ok(Value::Status(737)) => {}
        other => panic!("exit code 737 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_one_hundred_thirty_consecutive_runs_without_reset_still_blocked() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    for _ in 0..130 {
        let _ = vm.run();
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_738_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(738)));
    match vm.run() {
        VMResult::Ok(Value::Status(738)) => {}
        other => panic!("exit code 738 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_one_hundred_thirty_one_consecutive_runs_without_reset_still_blocked() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    for _ in 0..131 {
        let _ = vm.run();
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_739_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(739)));
    match vm.run() {
        VMResult::Ok(Value::Status(739)) => {}
        other => panic!("exit code 739 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_alternating_status_twenty_second_none_keeps_ninth() {
    let mut b = ChunkBuilder::new();
    for _ in 0..22 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(AlternatingStatusHost { calls: 0 }));
    match vm.run() {
        VMResult::Ok(Value::Status(9)) => {}
        other => panic!("AlternatingStatusHost twenty-second None MUST keep Some(9), got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_740_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(740)));
    match vm.run() {
        VMResult::Ok(Value::Status(740)) => {}
        other => panic!("exit code 740 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_none_host_twenty_one_ends_leaves_initial_status() {
    let mut b = ChunkBuilder::new();
    for _ in 0..21 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.last_status = 741;
    vm.set_shell_host(Box::new(NoneHost));
    match vm.run() {
        VMResult::Ok(Value::Status(741)) => {}
        other => panic!("NoneHost twenty-one ends MUST leave last_status 741, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_741_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(741)));
    match vm.run() {
        VMResult::Ok(Value::Status(741)) => {}
        other => panic!("exit code 741 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_first_some_then_none_host_twenty_first_none_keeps_first() {
    let mut b = ChunkBuilder::new();
    for _ in 0..21 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(FirstSomeThenNoneHost { calls: 0, status: 742 }));
    match vm.run() {
        VMResult::Ok(Value::Status(742)) => {}
        other => panic!("FirstSomeThenNoneHost twenty-first None MUST keep first Some(742), got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_742_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(742)));
    match vm.run() {
        VMResult::Ok(Value::Status(742)) => {}
        other => panic!("exit code 742 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_status_minus_one_hundred_three_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(-103)));
    match vm.run() {
        VMResult::Ok(Value::Status(-103)) => {}
        other => panic!("status -103 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_743_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(743)));
    match vm.run() {
        VMResult::Ok(Value::Status(743)) => {}
        other => panic!("exit code 743 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_one_hundred_one_end_incrementing_from_five() {
    let mut b = ChunkBuilder::new();
    for _ in 0..101 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(IncrementingSubshellHost { next: 5 }));
    match vm.run() {
        VMResult::Ok(Value::Status(105)) => {}
        other => panic!("one-hundred-first subshell_end(Some(105)) MUST win, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_744_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(744)));
    match vm.run() {
        VMResult::Ok(Value::Status(744)) => {}
        other => panic!("exit code 744 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_negative_then_zero_host_last_zero_wins_status745() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    struct NegativeThenZeroHost745 {
        calls: u32,
    }
    impl ShellHost for NegativeThenZeroHost745 {
        fn subshell_end(&mut self) -> Option<i32> {
            self.calls += 1;
            if self.calls == 1 {
                Some(-104)
            } else {
                Some(0)
            }
        }
    }
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(NegativeThenZeroHost745 { calls: 0 }));
    match vm.run() {
        VMResult::Ok(Value::Status(0)) => {}
        other => panic!("second subshell_end(Some(0)) MUST win over -104, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_745_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(745)));
    match vm.run() {
        VMResult::Ok(Value::Status(745)) => {}
        other => panic!("exit code 745 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_bit_xor_after_subshell_end_status746() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::LoadInt(15), 1);
    b.emit(Op::LoadInt(7), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::BitXor, 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(746)));
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 746);
}

#[test]
fn subshell_end_exit_code_746_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(746)));
    match vm.run() {
        VMResult::Ok(Value::Status(746)) => {}
        other => panic!("exit code 746 MUST propagate, got {:?}", other),
    }
}

// ─── Pin tests batch AB (handwritten) ─────────────────────────────────────

#[test]
fn subshell_end_exit_code_747_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(747)));
    match vm.run() {
        VMResult::Ok(Value::Status(747)) => {}
        other => panic!("exit code 747 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_one_hundred_thirty_two_consecutive_runs_without_reset_still_blocked() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    for _ in 0..132 {
        let _ = vm.run();
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_748_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(748)));
    match vm.run() {
        VMResult::Ok(Value::Status(748)) => {}
        other => panic!("exit code 748 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_one_hundred_thirty_three_consecutive_runs_without_reset_still_blocked() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    for _ in 0..133 {
        let _ = vm.run();
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_749_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(749)));
    match vm.run() {
        VMResult::Ok(Value::Status(749)) => {}
        other => panic!("exit code 749 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_one_hundred_thirty_four_consecutive_runs_without_reset_still_blocked() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    for _ in 0..134 {
        let _ = vm.run();
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_750_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(750)));
    match vm.run() {
        VMResult::Ok(Value::Status(750)) => {}
        other => panic!("exit code 750 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_one_hundred_two_end_incrementing_from_six() {
    let mut b = ChunkBuilder::new();
    for _ in 0..102 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(IncrementingSubshellHost { next: 6 }));
    match vm.run() {
        VMResult::Ok(Value::Status(107)) => {}
        other => panic!("one-hundred-second subshell_end(Some(107)) MUST win, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_751_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(751)));
    match vm.run() {
        VMResult::Ok(Value::Status(751)) => {}
        other => panic!("exit code 751 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_status_minus_one_hundred_four_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(-104)));
    match vm.run() {
        VMResult::Ok(Value::Status(-104)) => {}
        other => panic!("status -104 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_752_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(752)));
    match vm.run() {
        VMResult::Ok(Value::Status(752)) => {}
        other => panic!("exit code 752 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_status_minus_one_hundred_five_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(-105)));
    match vm.run() {
        VMResult::Ok(Value::Status(-105)) => {}
        other => panic!("status -105 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_753_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(753)));
    match vm.run() {
        VMResult::Ok(Value::Status(753)) => {}
        other => panic!("exit code 753 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_forty_sixth_some_host_applies_on_forty_sixth_end() {
    let mut b = ChunkBuilder::new();
    for _ in 0..46 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    struct FortySixthSomeHost {
        calls: u32,
    }
    impl ShellHost for FortySixthSomeHost {
        fn subshell_end(&mut self) -> Option<i32> {
            self.calls += 1;
            if self.calls == 46 {
                Some(754)
            } else {
                None
            }
        }
    }
    let mut vm = VM::new(b.build());
    vm.last_status = 8;
    vm.set_shell_host(Box::new(FortySixthSomeHost { calls: 0 }));
    match vm.run() {
        VMResult::Ok(Value::Status(754)) => {}
        other => panic!("forty-sixth subshell_end(Some(754)) MUST apply, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_754_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(754)));
    match vm.run() {
        VMResult::Ok(Value::Status(754)) => {}
        other => panic!("exit code 754 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_halt_reset_subshell_end_twenty_third_run_applies() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    SUBSHELL_END_CALLS.store(0, Ordering::SeqCst);
    let sub = {
        let mut b = ChunkBuilder::new();
        b.emit(Op::SubshellEnd, 1);
        b.emit(Op::CallBuiltin(100, 0), 1);
        b.build()
    };
    let end_only = {
        let mut b = ChunkBuilder::new();
        b.emit(Op::SubshellEnd, 1);
        b.build()
    };
    let mut vm = VM::new(sub.clone());
    vm.set_shell_host(Box::new(CountingSubshellEndHost(755)));
    vm.register_builtin(100, builtin_request_halt);
    let _ = vm.run();
    vm.reset(sub);
    let _ = vm.run();
    for _ in 0..20 {
        vm.reset(end_only.clone());
        let _ = vm.run();
    }
    vm.reset(end_only);
    let _ = vm.run();
    assert_eq!(SUBSHELL_END_CALLS.load(Ordering::SeqCst), 23);
    assert_eq!(vm.last_status, 755);
}

#[test]
fn subshell_end_exit_code_755_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(755)));
    match vm.run() {
        VMResult::Ok(Value::Status(755)) => {}
        other => panic!("exit code 755 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_log_not_after_subshell_end_status756() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::LoadFalse, 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::LogNot, 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(756)));
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 756);
}

#[test]
fn subshell_end_exit_code_756_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(756)));
    match vm.run() {
        VMResult::Ok(Value::Status(756)) => {}
        other => panic!("exit code 756 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_nineteen_begins_seventeen_ends_last_end_status_wins() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    SUBSHELL_BEGIN_CALLS.store(0, Ordering::SeqCst);
    SUBSHELL_END_CALLS.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    for _ in 0..19 {
        b.emit(Op::SubshellBegin, 1);
    }
    for _ in 0..17 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    struct CountingBeginEndHost757;
    impl ShellHost for CountingBeginEndHost757 {
        fn subshell_begin(&mut self) {
            SUBSHELL_BEGIN_CALLS.fetch_add(1, Ordering::SeqCst);
        }
        fn subshell_end(&mut self) -> Option<i32> {
            SUBSHELL_END_CALLS.fetch_add(1, Ordering::SeqCst);
            Some(757)
        }
    }
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(CountingBeginEndHost757));
    match vm.run() {
        VMResult::Ok(Value::Status(757)) => {}
        other => panic!("seventeenth subshell_end(Some(757)) MUST win, got {:?}", other),
    }
    assert_eq!(SUBSHELL_BEGIN_CALLS.load(Ordering::SeqCst), 19);
    assert_eq!(SUBSHELL_END_CALLS.load(Ordering::SeqCst), 17);
}

#[test]
fn subshell_end_exit_code_757_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(757)));
    match vm.run() {
        VMResult::Ok(Value::Status(757)) => {}
        other => panic!("exit code 757 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_negate_after_subshell_end_status758() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::LoadInt(6), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::Negate, 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(758)));
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 758);
}

#[test]
fn subshell_end_exit_code_758_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(758)));
    match vm.run() {
        VMResult::Ok(Value::Status(758)) => {}
        other => panic!("exit code 758 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_twenty_reset_runs_same_host_same_status() {
    let chunk = {
        let mut b = ChunkBuilder::new();
        b.emit(Op::SubshellEnd, 1);
        b.emit(Op::GetStatus, 1);
        b.build()
    };
    let mut vm = VM::new(chunk.clone());
    vm.set_shell_host(Box::new(StatusReturningHost(759)));
    for _ in 0..20 {
        match vm.run() {
            VMResult::Ok(Value::Status(759)) => {}
            other => panic!("reset run MUST return Status(759), got {:?}", other),
        }
        vm.reset(chunk.clone());
    }
}

#[test]
fn subshell_end_exit_code_759_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(759)));
    match vm.run() {
        VMResult::Ok(Value::Status(759)) => {}
        other => panic!("exit code 759 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_make_array_after_subshell_end_status760() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::LoadInt(2), 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::MakeArray(2), 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(760)));
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 760);
}

#[test]
fn subshell_end_exit_code_760_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(760)));
    match vm.run() {
        VMResult::Ok(Value::Status(760)) => {}
        other => panic!("exit code 760 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_one_hundred_three_end_incrementing_from_seven() {
    let mut b = ChunkBuilder::new();
    for _ in 0..103 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(IncrementingSubshellHost { next: 7 }));
    match vm.run() {
        VMResult::Ok(Value::Status(109)) => {}
        other => panic!("one-hundred-third subshell_end(Some(109)) MUST win, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_761_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(761)));
    match vm.run() {
        VMResult::Ok(Value::Status(761)) => {}
        other => panic!("exit code 761 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_concat_after_subshell_end_status762() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let chunk = {
        let mut cb = ChunkBuilder::new();
        let a = cb.add_constant(Value::str("p"));
        let bidx = cb.add_constant(Value::str("q"));
        cb.emit(Op::SubshellEnd, 1);
        cb.emit(Op::LoadConst(a), 1);
        cb.emit(Op::LoadConst(bidx), 1);
        cb.emit(Op::CallBuiltin(100, 0), 1);
        cb.emit(Op::Concat, 1);
        cb.emit(Op::CallBuiltin(200, 0), 1);
        cb.build()
    };
    let mut vm = VM::new(chunk);
    vm.set_shell_host(Box::new(StatusReturningHost(762)));
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 762);
}

#[test]
fn subshell_end_exit_code_762_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(762)));
    match vm.run() {
        VMResult::Ok(Value::Status(762)) => {}
        other => panic!("exit code 762 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_one_hundred_thirty_five_consecutive_runs_without_reset_still_blocked() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    for _ in 0..135 {
        let _ = vm.run();
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_763_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(763)));
    match vm.run() {
        VMResult::Ok(Value::Status(763)) => {}
        other => panic!("exit code 763 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_forty_seventh_some_host_applies_on_forty_seventh_end() {
    let mut b = ChunkBuilder::new();
    for _ in 0..47 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    struct FortySeventhSomeHost {
        calls: u32,
    }
    impl ShellHost for FortySeventhSomeHost {
        fn subshell_end(&mut self) -> Option<i32> {
            self.calls += 1;
            if self.calls == 47 {
                Some(764)
            } else {
                None
            }
        }
    }
    let mut vm = VM::new(b.build());
    vm.last_status = 11;
    vm.set_shell_host(Box::new(FortySeventhSomeHost { calls: 0 }));
    match vm.run() {
        VMResult::Ok(Value::Status(764)) => {}
        other => panic!("forty-seventh subshell_end(Some(764)) MUST apply, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_764_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(764)));
    match vm.run() {
        VMResult::Ok(Value::Status(764)) => {}
        other => panic!("exit code 764 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_alternating_status_twenty_third_none_keeps_ninth() {
    let mut b = ChunkBuilder::new();
    for _ in 0..23 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(AlternatingStatusHost { calls: 0 }));
    match vm.run() {
        VMResult::Ok(Value::Status(9)) => {}
        other => panic!("AlternatingStatusHost twenty-third None MUST keep Some(9), got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_765_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(765)));
    match vm.run() {
        VMResult::Ok(Value::Status(765)) => {}
        other => panic!("exit code 765 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_stops_before_load_float_after_subshell_end_status766() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    AFTER_HALT_COUNT.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::LoadFloat(2.5), 1);
    b.emit(Op::CallBuiltin(200, 0), 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(766)));
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(200, builtin_count_after_halt);
    let _ = vm.run();
    assert_eq!(AFTER_HALT_COUNT.load(Ordering::SeqCst), 0);
    assert_eq!(vm.last_status, 766);
}

#[test]
fn subshell_end_exit_code_766_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(766)));
    match vm.run() {
        VMResult::Ok(Value::Status(766)) => {}
        other => panic!("exit code 766 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_one_hundred_four_end_incrementing_from_one() {
    let mut b = ChunkBuilder::new();
    for _ in 0..104 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(IncrementingSubshellHost { next: 1 }));
    match vm.run() {
        VMResult::Ok(Value::Status(104)) => {}
        other => panic!("one-hundred-fourth subshell_end(Some(104)) MUST win, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_767_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(767)));
    match vm.run() {
        VMResult::Ok(Value::Status(767)) => {}
        other => panic!("exit code 767 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_none_host_twenty_two_ends_leaves_initial_status() {
    let mut b = ChunkBuilder::new();
    for _ in 0..22 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.last_status = 768;
    vm.set_shell_host(Box::new(NoneHost));
    match vm.run() {
        VMResult::Ok(Value::Status(768)) => {}
        other => panic!("NoneHost twenty-two ends MUST leave last_status 768, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_768_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(768)));
    match vm.run() {
        VMResult::Ok(Value::Status(768)) => {}
        other => panic!("exit code 768 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_first_some_then_none_host_twenty_second_none_keeps_first() {
    let mut b = ChunkBuilder::new();
    for _ in 0..22 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(FirstSomeThenNoneHost { calls: 0, status: 769 }));
    match vm.run() {
        VMResult::Ok(Value::Status(769)) => {}
        other => panic!("FirstSomeThenNoneHost twenty-second None MUST keep first Some(769), got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_769_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(769)));
    match vm.run() {
        VMResult::Ok(Value::Status(769)) => {}
        other => panic!("exit code 769 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_one_hundred_thirty_six_consecutive_runs_without_reset_still_blocked() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    for _ in 0..136 {
        let _ = vm.run();
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_770_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(770)));
    match vm.run() {
        VMResult::Ok(Value::Status(770)) => {}
        other => panic!("exit code 770 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_forty_eighth_some_host_applies_on_forty_eighth_end() {
    let mut b = ChunkBuilder::new();
    for _ in 0..48 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    struct FortyEighthSomeHost {
        calls: u32,
    }
    impl ShellHost for FortyEighthSomeHost {
        fn subshell_end(&mut self) -> Option<i32> {
            self.calls += 1;
            if self.calls == 48 {
                Some(771)
            } else {
                None
            }
        }
    }
    let mut vm = VM::new(b.build());
    vm.last_status = 2;
    vm.set_shell_host(Box::new(FortyEighthSomeHost { calls: 0 }));
    match vm.run() {
        VMResult::Ok(Value::Status(771)) => {}
        other => panic!("forty-eighth subshell_end(Some(771)) MUST apply, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_771_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(771)));
    match vm.run() {
        VMResult::Ok(Value::Status(771)) => {}
        other => panic!("exit code 771 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_pipeline_then_subshell_getstatus_reads_subshell_772() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::PipelineBegin(2), 1);
    b.emit(Op::PipelineEnd, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    struct PipelineSubshellHost772;
    impl ShellHost for PipelineSubshellHost772 {
        fn pipeline_end(&mut self) -> i32 {
            7
        }
        fn subshell_end(&mut self) -> Option<i32> {
            Some(772)
        }
    }
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(PipelineSubshellHost772));
    match vm.run() {
        VMResult::Ok(Value::Status(772)) => {}
        other => panic!("GetStatus after pipeline+subshell MUST read subshell 772, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_772_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(772)));
    match vm.run() {
        VMResult::Ok(Value::Status(772)) => {}
        other => panic!("exit code 772 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_subshell_then_pipeline_getstatus_reads_pipeline_773() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::PipelineBegin(2), 1);
    b.emit(Op::PipelineEnd, 1);
    b.emit(Op::GetStatus, 1);
    struct SubshellThenPipelineHost773;
    impl ShellHost for SubshellThenPipelineHost773 {
        fn subshell_end(&mut self) -> Option<i32> {
            Some(16)
        }
        fn pipeline_end(&mut self) -> i32 {
            773
        }
    }
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(SubshellThenPipelineHost773));
    match vm.run() {
        VMResult::Ok(Value::Status(773)) => {}
        other => panic!("GetStatus after subshell+pipeline MUST read pipeline 773, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_773_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(773)));
    match vm.run() {
        VMResult::Ok(Value::Status(773)) => {}
        other => panic!("exit code 773 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_status_minus_one_hundred_six_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(-106)));
    match vm.run() {
        VMResult::Ok(Value::Status(-106)) => {}
        other => panic!("status -106 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_774_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(774)));
    match vm.run() {
        VMResult::Ok(Value::Status(774)) => {}
        other => panic!("exit code 774 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_one_hundred_five_end_incrementing_from_eight() {
    let mut b = ChunkBuilder::new();
    for _ in 0..105 {
        b.emit(Op::SubshellEnd, 1);
    }
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(IncrementingSubshellHost { next: 8 }));
    match vm.run() {
        VMResult::Ok(Value::Status(112)) => {}
        other => panic!("one-hundred-fifth subshell_end(Some(112)) MUST win, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_775_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(775)));
    match vm.run() {
        VMResult::Ok(Value::Status(775)) => {}
        other => panic!("exit code 775 MUST propagate, got {:?}", other),
    }
}

#[test]
fn request_halt_one_hundred_thirty_seven_consecutive_runs_without_reset_still_blocked() {
    let _guard = PIN_TEST_LOCK.lock().unwrap();
    TOUCHED.store(0, Ordering::SeqCst);
    let mut b = ChunkBuilder::new();
    b.emit(Op::CallBuiltin(100, 0), 1);
    b.emit(Op::CallBuiltin(7, 0), 1);
    let mut vm = VM::new(b.build());
    vm.register_builtin(100, builtin_request_halt);
    vm.register_builtin(7, builtin_touched);
    for _ in 0..137 {
        let _ = vm.run();
    }
    assert_eq!(TOUCHED.load(Ordering::SeqCst), 0);
}

#[test]
fn subshell_end_exit_code_776_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellBegin, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(776)));
    match vm.run() {
        VMResult::Ok(Value::Status(776)) => {}
        other => panic!("exit code 776 MUST propagate, got {:?}", other),
    }
}

#[test]
fn subshell_negative_then_zero_host_last_zero_wins_status777() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    struct NegativeThenZeroHost777 {
        calls: u32,
    }
    impl ShellHost for NegativeThenZeroHost777 {
        fn subshell_end(&mut self) -> Option<i32> {
            self.calls += 1;
            if self.calls == 1 {
                Some(-107)
            } else {
                Some(0)
            }
        }
    }
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(NegativeThenZeroHost777 { calls: 0 }));
    match vm.run() {
        VMResult::Ok(Value::Status(0)) => {}
        other => panic!("second subshell_end(Some(0)) MUST win over -107, got {:?}", other),
    }
}

#[test]
fn subshell_end_exit_code_777_propagates() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::SubshellEnd, 1);
    b.emit(Op::GetStatus, 1);
    let mut vm = VM::new(b.build());
    vm.set_shell_host(Box::new(StatusReturningHost(777)));
    match vm.run() {
        VMResult::Ok(Value::Status(777)) => {}
        other => panic!("exit code 777 MUST propagate, got {:?}", other),
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
