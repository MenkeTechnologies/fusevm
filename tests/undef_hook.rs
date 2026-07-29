//! Strict undef mode (`VM::set_undef_hook`): a read of a variable that was
//! never assigned asks the host instead of pushing `Value::Undef`.
//!
//! fusevm's default is the shell/awk one — an unset variable reads as the empty
//! string and counts as zero — which is right for zshrs, awkrs and stryke and
//! wrong for a language that distinguishes absence from emptiness. Tcl answers
//! `can't read "x": no such variable`; a Lisp signals `void-variable`.
//!
//! What these pin, in the order that matters:
//!
//! 1. the mode is **off by default**, so every frontend that installs no hook is
//!    byte-for-byte unchanged;
//! 2. a global read hands the host the chunk's interned *name*, which is what a
//!    frontend needs to word its own diagnostic;
//! 3. a slot read hands it `name: None` and `from_slot: true`, because a slot is
//!    addressed by index and the chunk carries no name for it;
//! 4. `Ok(v)` substitutes a value and `Ok(Value::Undef)` is exactly the default
//!    reading, so a host can refuse some reads and not others;
//! 5. a read carries its **op index**, so a frontend can make one site refuse
//!    and another tolerate absence — Tcl's `$x` errors where its `incr x`
//!    initialises the same variable, and both are `GetVar`;
//! 6. **the JIT cannot bypass it.** A loop that reads an unset global must raise
//!    on every iteration count, including one well past the tracing threshold —
//!    a check the interpreter makes and native code skips would be worse than no
//!    check at all.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use fusevm::{Chunk, ChunkBuilder, Op, UndefRead, VMResult, Value, VM};

/// Every read the hook saw, as `(name, from_slot, ip)`.
type Seen = Arc<std::sync::Mutex<Vec<(Option<String>, bool, usize)>>>;

fn recording_hook(seen: &Seen, answer: Result<Value, String>) -> fusevm::UndefHook {
    let seen = Arc::clone(seen);
    Arc::new(move |read: UndefRead<'_>| {
        seen.lock()
            .expect("seen lock")
            .push((read.name.map(str::to_string), read.from_slot, read.ip));
        answer.clone()
    })
}

fn run(chunk: Chunk, hook: Option<fusevm::UndefHook>) -> Result<Value, String> {
    let mut vm = VM::new(chunk);
    #[cfg(feature = "jit")]
    vm.enable_tracing_jit();
    if let Some(h) = hook {
        vm.set_undef_hook(h);
    }
    match vm.run() {
        VMResult::Ok(v) => Ok(v),
        VMResult::Error(e) => Err(e),
        VMResult::Halted => Err("halted".to_string()),
    }
}

/// `x` was never assigned; the chunk's value is whatever reading it produced.
fn read_unset_global() -> Chunk {
    let mut b = ChunkBuilder::new();
    let x = b.add_name("x");
    b.emit(Op::GetVar(x), 1);
    b.build()
}

#[test]
fn without_a_hook_an_unset_global_reads_as_undef() {
    assert_eq!(run(read_unset_global(), None), Ok(Value::Undef));
}

#[test]
fn the_hook_receives_the_globals_interned_name() {
    let seen: Seen = Arc::default();
    let err = run(
        read_unset_global(),
        Some(recording_hook(
            &seen,
            Err("can't read \"x\": no such variable".to_string()),
        )),
    );
    assert_eq!(err, Err("can't read \"x\": no such variable".to_string()));
    assert_eq!(
        *seen.lock().expect("seen lock"),
        vec![(Some("x".to_string()), false, 0)],
        "a global read carries its name and is not flagged as a slot"
    );
}

#[test]
fn the_hook_may_substitute_a_value_instead_of_refusing() {
    let seen: Seen = Arc::default();
    assert_eq!(
        run(
            read_unset_global(),
            Some(recording_hook(&seen, Ok(Value::Int(7))))
        ),
        Ok(Value::Int(7))
    );
}

#[test]
fn answering_undef_is_the_default_reading() {
    let seen: Seen = Arc::default();
    assert_eq!(
        run(
            read_unset_global(),
            Some(recording_hook(&seen, Ok(Value::Undef)))
        ),
        Ok(Value::Undef),
        "a host that declines to refuse gets exactly the hookless behaviour"
    );
}

#[test]
fn a_slot_read_is_flagged_and_carries_no_name() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::GetSlot(3), 1);
    let seen: Seen = Arc::default();
    let _ = run(b.build(), Some(recording_hook(&seen, Ok(Value::Undef))));
    assert_eq!(
        *seen.lock().expect("seen lock"),
        vec![(None, true, 0)],
        "a slot has no interned name; the host is told so rather than guessing"
    );
}

#[test]
fn an_assigned_variable_never_reaches_the_hook() {
    let mut b = ChunkBuilder::new();
    let x = b.add_name("x");
    // The empty string is a *value*: assigning it must not look like absence.
    b.emit(Op::LoadConst(0), 1);
    b.emit(Op::SetVar(x), 1);
    b.emit(Op::GetVar(x), 1);
    let mut chunk = b.build();
    chunk.constants.push(Value::Str(Arc::new(String::new())));
    let seen: Seen = Arc::default();
    let got = run(
        chunk,
        Some(recording_hook(&seen, Err("must not fire".to_string()))),
    );
    assert_eq!(got, Ok(Value::Str(Arc::new(String::new()))));
    assert!(
        seen.lock().expect("seen lock").is_empty(),
        "an empty string is not an unset variable"
    );
}

/// A counted loop that reads an unset global on every iteration, run past the
/// tracing threshold. The interpreter refuses each read; native code, which
/// reads globals out of a flat `i64` buffer where `Undef` has no encoding, would
/// see the integer 0 and loop silently instead.
#[test]
#[cfg(feature = "jit")]
fn a_hot_loop_cannot_read_an_unset_global_natively() {
    let calls = Arc::new(AtomicUsize::new(0));
    let hook: fusevm::UndefHook = {
        let calls = Arc::clone(&calls);
        Arc::new(move |_read: UndefRead<'_>| {
            calls.fetch_add(1, Ordering::Relaxed);
            Err("unset".to_string())
        })
    };

    // i = 0; while (i < 100000) { i = i + unset }   — rotated, so the tracing
    // tier would take it if the read were not refused.
    let mut b = ChunkBuilder::new();
    let i = b.add_name("i");
    let never = b.add_name("never");
    b.emit(Op::LoadInt(0), 1);
    b.emit(Op::SetVar(i), 1);
    let enter = b.emit(Op::Jump(usize::MAX), 1);
    let body = b.current_pos();
    b.emit(Op::GetVar(i), 1);
    b.emit(Op::GetVar(never), 1);
    b.emit(Op::Add, 1);
    b.emit(Op::SetVar(i), 1);
    let cond = b.current_pos();
    b.patch_jump(enter, cond);
    b.emit(Op::GetVar(i), 1);
    b.emit(Op::LoadInt(100_000), 1);
    b.emit(Op::NumLt, 1);
    b.emit(Op::JumpIfTrue(body), 1);
    b.emit(Op::GetVar(i), 1);

    assert_eq!(run(b.build(), Some(hook)), Err("unset".to_string()));
    assert_eq!(
        calls.load(Ordering::Relaxed),
        1,
        "the first read refuses and ends the run; none is answered natively"
    );
}

/// Two reads of the same unset variable, one refused and one tolerated, told
/// apart only by where they are. This is the shape a frontend needs for Tcl's
/// `$x` (an error) and `incr x` (initialises to zero), which compile to the
/// same op on the same name.
#[test]
fn a_frontend_can_refuse_one_read_site_and_tolerate_another() {
    let mut b = ChunkBuilder::new();
    let x = b.add_name("x");
    let tolerant = b.emit(Op::GetVar(x), 1); // the `incr`-like read
    b.emit(Op::Pop, 1);
    b.emit(Op::GetVar(x), 1); // the `$x`-like read
    let chunk = b.build();

    let hook: fusevm::UndefHook = Arc::new(move |read: UndefRead<'_>| {
        if read.ip == tolerant {
            Ok(Value::Int(0))
        } else {
            Err(format!(
                "can't read \"{}\": no such variable",
                read.name.unwrap_or("?")
            ))
        }
    });

    assert_eq!(
        run(chunk, Some(hook)),
        Err("can't read \"x\": no such variable".to_string()),
        "the tolerant site passed and the refusing one raised"
    );
}
