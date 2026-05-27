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
