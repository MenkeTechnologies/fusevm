#![cfg(feature = "jit")]

//! `fusevm::jit::stats()` / `reset_stats()`: the process-global counters move
//! at the real compile, cache-lookup, disk, and fallback sites.
//!
//! The counters and the disk-cache directory are process-global, so every test
//! holds `serial()` and asserts on a `reset_stats()` baseline. Each test runs
//! on its own thread, so the per-thread tier caches start empty.

use std::sync::{Mutex, MutexGuard, OnceLock};

use fusevm::jit::{reset_stats, stats, JitStats};
use fusevm::{ChunkBuilder, JitCompiler, Op, SlotKind, VMResult, Value, VM};

fn serial() -> MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|e| e.into_inner())
}

/// Point the disk cache (when compiled in) at a fresh directory so a test never
/// touches the user's cache and never sees blobs from an earlier run.
#[cfg(feature = "jit-disk-cache")]
fn fresh_cache_dir(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "fusevm_jit_stats_{tag}_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    JitCompiler::new().set_jit_cache_dir(Some(dir.clone()));
    dir
}

#[cfg(not(feature = "jit-disk-cache"))]
fn fresh_cache_dir(_tag: &str) {}

fn release_cache_dir() {
    #[cfg(feature = "jit-disk-cache")]
    JitCompiler::new().set_jit_cache_dir(None);
}

/// `(a + b) * 3`, distinct per `(a, b)` so each test gets its own op hash.
fn arith_chunk(a: i64, b: i64) -> fusevm::Chunk {
    let mut c = ChunkBuilder::new();
    c.emit(Op::LoadInt(a), 1);
    c.emit(Op::LoadInt(b), 1);
    c.emit(Op::Add, 1);
    c.emit(Op::LoadInt(3), 1);
    c.emit(Op::Mul, 1);
    c.build()
}

/// A counting loop the block tier accepts.
fn loop_chunk(limit: i64) -> fusevm::Chunk {
    let mut c = ChunkBuilder::new();
    c.emit(Op::LoadInt(0), 1);
    c.emit(Op::SetSlot(0), 1);
    let top = c.current_pos();
    c.emit(Op::PreIncSlotVoid(0), 1);
    c.emit(Op::GetSlot(0), 1);
    c.emit(Op::LoadInt(limit), 1);
    c.emit(Op::NumLt, 1);
    let j = c.emit(Op::JumpIfTrue(0), 1);
    c.patch_jump(j, top);
    c.emit(Op::GetSlot(0), 1);
    c.build()
}

#[test]
fn default_snapshot_is_all_zero_after_reset() {
    let _g = serial();
    let _dir = fresh_cache_dir("reset");
    let jit = JitCompiler::new();
    assert!(jit.try_run_linear(&arith_chunk(1, 2), &[]).is_some());
    assert_ne!(
        stats(),
        JitStats::default(),
        "a compile must move a counter"
    );
    reset_stats();
    assert_eq!(stats(), JitStats::default());
    release_cache_dir();
}

#[test]
fn linear_miss_then_hit() {
    let _g = serial();
    let _dir = fresh_cache_dir("linear");
    let jit = JitCompiler::new();
    let chunk = arith_chunk(10, 20);
    reset_stats();

    assert!(matches!(
        jit.try_run_linear(&chunk, &[]),
        Some(Value::Int(90))
    ));
    let s = stats();
    assert_eq!(s.linear_cache_misses, 1);
    assert_eq!(s.linear_cache_hits, 0);
    // With the disk cache the miss builds a native blob and stores it; without
    // it the in-memory JIT compiles directly.
    #[cfg(feature = "jit-disk-cache")]
    {
        assert_eq!((s.disk_misses, s.native_builds, s.disk_stores), (1, 1, 1));
        assert_eq!(s.linear_compiles, 0);
    }
    #[cfg(not(feature = "jit-disk-cache"))]
    {
        assert_eq!(s.linear_compiles, 1);
        assert_eq!(s.native_builds, 0);
    }

    for _ in 0..3 {
        assert!(matches!(
            jit.try_run_linear(&chunk, &[]),
            Some(Value::Int(90))
        ));
    }
    let s = stats();
    assert_eq!(s.linear_cache_hits, 3);
    assert_eq!(s.linear_cache_misses, 1, "hits must not count as misses");
    assert_eq!(s.linear_fallbacks, 0);
    release_cache_dir();
}

#[test]
fn linear_ineligible_chunk_counts_a_fallback() {
    let _g = serial();
    let _dir = fresh_cache_dir("fallback");
    let jit = JitCompiler::new();
    let mut c = ChunkBuilder::new();
    let k = c.add_constant(Value::str("not a number"));
    c.emit(Op::LoadConst(k), 1);
    let chunk = c.build();
    reset_stats();

    assert!(jit.try_run_linear(&chunk, &[]).is_none());
    let s = stats();
    assert_eq!(s.linear_fallbacks, 1);
    assert_eq!(s.linear_compiles, 0);
    assert_eq!(s.native_builds, 0);
    assert_eq!(s.linear_cache_hits, 0);
    release_cache_dir();
}

#[test]
fn block_warmup_miss_and_hit() {
    let _g = serial();
    let _dir = fresh_cache_dir("block");
    let jit = JitCompiler::new();
    let chunk = loop_chunk(50);
    let kinds = [SlotKind::Int];
    reset_stats();

    // Eager entry compiles on first call regardless of the warmup threshold.
    let mut slots = [0i64];
    assert_eq!(
        jit.try_run_block_eager_kinded(&chunk, &mut slots, &kinds),
        Some(50)
    );
    let s = stats();
    assert_eq!(s.block_cache_misses, 1);
    assert_eq!(s.block_cache_hits, 0);
    assert_eq!(s.block_warmups, 0);
    #[cfg(feature = "jit-disk-cache")]
    assert_eq!((s.native_builds, s.disk_stores), (1, 1));
    #[cfg(not(feature = "jit-disk-cache"))]
    assert_eq!(s.block_compiles, 1);

    let mut slots = [0i64];
    assert_eq!(
        jit.try_run_block_eager_kinded(&chunk, &mut slots, &kinds),
        Some(50)
    );
    let s = stats();
    assert_eq!(s.block_cache_hits, 1);
    assert_eq!(s.block_cache_misses, 1);

    // A frame shorter than the chunk's highest slot is declined, not run.
    assert_eq!(
        jit.try_run_block_eager_kinded(&chunk, &mut [], &kinds),
        None
    );
    assert_eq!(stats().block_fallbacks, 1);
    release_cache_dir();
}

#[test]
fn block_below_threshold_counts_warmups() {
    let _g = serial();
    let _dir = fresh_cache_dir("warm");
    let jit = JitCompiler::new();
    let chunk = loop_chunk(7);
    reset_stats();

    let mut slots = [0i64];
    // First call of a fresh chunk: either still warming (default threshold) or
    // compiled immediately (FUSEVM_JIT_BLOCK_THRESHOLD=0). Exactly one of the
    // two counters records it.
    let ran = jit.try_run_block(&chunk, &mut slots);
    let s = stats();
    assert_eq!(s.block_warmups + s.block_cache_misses, 1);
    assert_eq!(ran.is_none(), s.block_warmups == 1);
    release_cache_dir();
}

/// A second thread has empty in-memory caches, so a chunk the first thread
/// built is served from the disk cache with no new codegen.
#[cfg(feature = "jit-disk-cache")]
#[test]
fn disk_cache_serves_a_second_thread_without_codegen() {
    let _g = serial();
    let dir = fresh_cache_dir("disk");
    let chunk = arith_chunk(100, 200);
    reset_stats();

    {
        let chunk = chunk.clone();
        std::thread::spawn(move || {
            assert!(matches!(
                JitCompiler::new().try_run_linear(&chunk, &[]),
                Some(Value::Int(900))
            ));
        })
        .join()
        .unwrap();
    }
    let s = stats();
    assert_eq!((s.disk_misses, s.native_builds, s.disk_stores), (1, 1, 1));
    assert_eq!(s.disk_loads, 0);

    reset_stats();
    std::thread::spawn(move || {
        assert!(matches!(
            JitCompiler::new().try_run_linear(&chunk, &[]),
            Some(Value::Int(900))
        ));
    })
    .join()
    .unwrap();
    let s = stats();
    assert_eq!(s.disk_loads, 1);
    assert_eq!(s.native_builds, 0, "a disk load must not run Cranelift");
    assert_eq!(s.disk_stores, 0);
    assert_eq!(s.linear_cache_misses, 1, "per-thread cache was cold");
    release_cache_dir();
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn trace_hits_move_when_a_hot_loop_runs_native() {
    let _g = serial();
    let _dir = fresh_cache_dir("trace");
    let mut c = ChunkBuilder::new();
    c.emit(Op::LoadInt(0), 1);
    c.emit(Op::SetSlot(0), 1);
    let anchor = c.current_pos();
    c.emit(Op::PreIncSlotVoid(0), 1);
    c.emit(Op::GetSlot(0), 1);
    c.emit(Op::LoadInt(5000), 1);
    c.emit(Op::NumLt, 1);
    let j = c.emit(Op::JumpIfTrue(0), 1);
    c.patch_jump(j, anchor);
    c.emit(Op::GetSlot(0), 1);
    let chunk = c.build();

    let mut vm = VM::new(chunk);
    vm.enable_tracing_jit();
    vm.frames.last_mut().unwrap().slots.push(Value::Int(0));
    reset_stats();

    match vm.run() {
        VMResult::Ok(Value::Int(5000)) => {}
        other => panic!("expected Int(5000), got {other:?}"),
    }
    let s = stats();
    // Whichever tier took the loop, native code ran and was counted; the
    // tiers are mutually exclusive per chunk.
    let native_runs = s.trace_hits + s.block_cache_hits + s.block_cache_misses;
    assert!(native_runs >= 1, "no native tier ran: {s:?}");
    let codegen = s.trace_compiles + s.block_compiles + s.native_builds + s.disk_loads;
    assert!(codegen >= 1, "no compile or disk load recorded: {s:?}");
    release_cache_dir();
}
