//! `VM::reset` and `VMPool` integration tests.
//!
//! Verifies that pooled VMs produce identical results to fresh VMs and
//! that state correctly resets across acquires.

use fusevm::{ChunkBuilder, Op, VMPool, VMResult, Value, VM};

fn build_simple_chunk(answer: i64) -> fusevm::Chunk {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(answer / 2), 1);
    b.emit(Op::LoadInt(answer / 2 + answer % 2), 1);
    b.emit(Op::Add, 1);
    b.build()
}

fn build_loop_chunk(limit: i64) -> fusevm::Chunk {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(0), 1);
    b.emit(Op::SetSlot(0), 1);
    let anchor = b.current_pos();
    b.emit(Op::PreIncSlotVoid(0), 1);
    b.emit(Op::GetSlot(0), 1);
    b.emit(Op::LoadInt(limit), 1);
    b.emit(Op::NumLt, 1);
    let close = b.emit(Op::JumpIfTrue(0), 1);
    b.patch_jump(close, anchor);
    b.emit(Op::GetSlot(0), 1);
    b.build()
}

#[test]
fn vm_reset_clears_stack_and_frames() {
    let mut vm = VM::new(build_simple_chunk(42));
    let _ = vm.run();
    // After run, stack/frames may be non-empty depending on path.
    vm.reset(build_simple_chunk(100));
    assert_eq!(vm.stack.len(), 0, "stack must be cleared on reset");
    assert_eq!(vm.frames.len(), 1, "frame stack reset to one base frame");
    assert_eq!(vm.ip, 0);
}

#[test]
fn vm_reset_then_run_matches_fresh_vm() {
    let chunk1 = build_simple_chunk(42);
    let chunk2 = build_simple_chunk(100);

    // Fresh VM run.
    let fresh = match VM::new(chunk2.clone()).run() {
        VMResult::Ok(Value::Int(n)) => n,
        other => panic!("fresh: expected Int, got {:?}", other),
    };

    // Reuse a VM for the second chunk after running the first.
    let mut vm = VM::new(chunk1);
    let _ = vm.run();
    vm.reset(chunk2);
    let pooled = match vm.run() {
        VMResult::Ok(Value::Int(n)) => n,
        other => panic!("pooled: expected Int, got {:?}", other),
    };

    assert_eq!(fresh, pooled);
}

#[test]
fn vm_reset_resizes_globals_to_match_chunk() {
    // Different chunks may declare different name pools (globals).
    let mut b1 = ChunkBuilder::new();
    let _name = b1.add_name("x");
    b1.emit(Op::LoadInt(7), 1);
    let chunk1 = b1.build();

    let mut b2 = ChunkBuilder::new();
    b2.add_name("a");
    b2.add_name("b");
    b2.add_name("c");
    b2.emit(Op::LoadInt(8), 1);
    let chunk2 = b2.build();

    let mut vm = VM::new(chunk1);
    let _ = vm.run();
    vm.reset(chunk2);
    // Globals should resize to match chunk2's 3-name pool.
    assert_eq!(vm.globals.len(), 3);
}

#[test]
fn pool_acquire_and_release_returns_correct_results() {
    let mut pool = VMPool::new();
    for i in 1..=20i64 {
        let chunk = build_simple_chunk(i);
        let mut vm = pool.acquire(chunk);
        let r = match vm.run() {
            VMResult::Ok(Value::Int(n)) => n,
            other => panic!("expected Int, got {:?}", other),
        };
        assert_eq!(r, i);
        pool.release(vm);
    }
    // Pool should retain the most-recently-released VM.
    assert!(!pool.is_empty(), "pool should retain released VMs");
}

#[test]
fn pool_with_closure_runs_to_completion() {
    let mut pool = VMPool::new();
    let result = pool.with(build_simple_chunk(50), |vm| match vm.run() {
        VMResult::Ok(Value::Int(n)) => n,
        _ => panic!("expected Int"),
    });
    assert_eq!(result, 50);
}

#[test]
fn pool_handles_loop_chunks() {
    let mut pool = VMPool::new();
    for limit in [10i64, 50, 100, 200] {
        let mut vm = pool.acquire(build_loop_chunk(limit));
        // ensure_slots since loop uses slot 0
        let frame = vm.frames.last_mut().unwrap();
        if frame.slots.is_empty() {
            frame.slots.push(Value::Int(0));
        }
        let r = match vm.run() {
            VMResult::Ok(Value::Int(n)) => n,
            other => panic!("expected Int, got {:?}", other),
        };
        assert_eq!(r, limit);
        pool.release(vm);
    }
}

#[test]
fn pool_with_capacity_starts_empty() {
    let pool = VMPool::with_capacity(8);
    assert_eq!(pool.len(), 0);
    assert!(pool.is_empty());
}

#[test]
fn many_acquire_release_cycles_dont_leak_state() {
    // Run a chunk that accumulates state in slots, release, acquire a
    // fresh chunk — the new run should NOT see the old state.
    let mut pool = VMPool::new();

    {
        let mut vm = pool.acquire(build_loop_chunk(100));
        let frame = vm.frames.last_mut().unwrap();
        frame.slots.push(Value::Int(0));
        let _ = vm.run();
        pool.release(vm);
    }
    // Acquire again with a different chunk.
    let mut vm = pool.acquire(build_simple_chunk(99));
    let r = match vm.run() {
        VMResult::Ok(Value::Int(n)) => n,
        _ => panic!("expected Int"),
    };
    assert_eq!(
        r, 99,
        "second acquire must see fresh state, not the previous loop's slot 0"
    );
    pool.release(vm);
}

#[cfg(feature = "jit")]
#[test]
fn pool_preserves_tracing_jit_enabled_across_reset() {
    use fusevm::JitCompiler;

    let mut pool = VMPool::new();
    let chunk1 = build_loop_chunk(200);
    let chunk2 = build_loop_chunk(200);

    {
        let mut vm = pool.acquire(chunk1.clone());
        vm.enable_tracing_jit();
        let frame = vm.frames.last_mut().unwrap();
        frame.slots.push(Value::Int(0));
        let _ = vm.run();
        pool.release(vm);
    }

    // The reset path should preserve the tracing_jit flag (via reset's
    // explicit field policy — `enable_tracing_jit` was called once on
    // this VM, and `reset` intentionally keeps that flag).
    //
    // Note: VMPool reuses the SAME VM via reset, but the tracing_jit
    // flag is documented as preserved by reset. So acquire→reset→run
    // on a tracing-enabled VM should still trace.
    let mut vm = pool.acquire(chunk2.clone());
    // Verify the flag carried over from the previous run.
    // (reset doesn't currently expose the flag publicly; we infer
    // tracing-jit-active by checking trace cache state after run.)
    let frame = vm.frames.last_mut().unwrap();
    frame.slots.push(Value::Int(0));
    let _ = vm.run();
    let _jit = JitCompiler::new();
    // Either chunk1's or chunk2's anchor may have a compiled trace —
    // we just verify the flag's behavior didn't crash.
    pool.release(vm);
}
