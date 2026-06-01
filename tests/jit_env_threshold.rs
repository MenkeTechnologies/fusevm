//! Verifies the `FUSEVM_JIT_BLOCK_THRESHOLD` env override reaches the per-thread
//! `TraceJitConfig`. Kept in its own test binary (a single test) so the
//! process-global env var can't race parallel sibling tests.

#![cfg(feature = "jit")]

use fusevm::{ChunkBuilder, JitCompiler, Op};

/// A block-eligible sum loop; result is sum 0..(limit-1).
fn sum_loop(limit: i32) -> fusevm::Chunk {
    let mut b = ChunkBuilder::new();
    b.emit(Op::PushFrame, 1);
    b.emit(Op::LoadInt(0), 1);
    b.emit(Op::SetSlot(0), 1);
    b.emit(Op::LoadInt(0), 1);
    b.emit(Op::SetSlot(1), 1);
    b.emit(Op::GetSlot(0), 1);
    b.emit(Op::GetSlot(1), 1);
    b.emit(Op::Add, 1);
    b.emit(Op::SetSlot(0), 1);
    b.emit(Op::PreIncSlotVoid(1), 1);
    b.emit(Op::SlotLtIntJumpIfFalse(1, limit, 12), 1);
    b.emit(Op::Jump(5), 1);
    b.emit(Op::GetSlot(0), 1);
    b.build()
}

#[test]
fn env_block_threshold_zero_compiles_on_first_call() {
    // Set the override BEFORE the worker thread initializes its TLS config,
    // so `config_from_env` picks it up. Edition 2021: set_var is safe.
    std::env::set_var("FUSEVM_JIT_BLOCK_THRESHOLD", "0");

    // A fresh thread reads the env var when it first touches the JIT.
    let handle = std::thread::spawn(|| {
        let jit = JitCompiler::new();
        let chunk = sum_loop(101); // sum 0..100 = 5050
        let mut slots = vec![0i64; 4];
        // With block_threshold = 0, the very first invocation compiles+runs
        // (hot_count 1 > threshold 0) instead of returning None for warmup.
        jit.try_run_block(&chunk, &mut slots)
    });

    let first_call = handle.join().expect("worker thread panicked");
    std::env::remove_var("FUSEVM_JIT_BLOCK_THRESHOLD");

    assert_eq!(
        first_call,
        Some(5050),
        "FUSEVM_JIT_BLOCK_THRESHOLD=0 should make the block JIT compile on the first invocation"
    );
}
