#![allow(clippy::approx_constant)]
#![cfg(feature = "jit-disk-cache")]

//! Integration tests for the persistent on-disk native-JIT cache.
//!
//! These exercise the `jit-disk-cache` feature end-to-end: native compilation,
//! atomic disk persistence, mmap-based loading with relocation patching, and
//! the fingerprint/op_hash/corruption rejection paths.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard, OnceLock};

use fusevm::{ChunkBuilder, JitCompiler, Op, Value, VMResult, VM};

/// The cache directory is a process-global override, so tests that configure it
/// must not run concurrently. Each such test holds this lock for its duration.
fn serial() -> MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|e| e.into_inner())
}

/// A unique temp directory per test, so concurrent test threads never collide.
fn fresh_dir(tag: &str) -> PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "fusevm_jit_cache_{}_{}_{}_{}",
        tag,
        std::process::id(),
        n,
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

fn build(ops: &[(Op, u32)]) -> fusevm::Chunk {
    let mut b = ChunkBuilder::new();
    for (op, line) in ops {
        b.emit(op.clone(), *line);
    }
    b.build()
}

/// Run a chunk with the disk cache pointed at `dir`, returning the result.
/// A fresh `JitCompiler` is used each call so the per-thread in-memory cache
/// does not mask the disk path (note: TLS is per-thread, not per-compiler, so
/// distinct op_hashes are still required to force a disk hit — see the
/// dedicated roundtrip test which uses a subprocess-style fresh op set).
fn run_with_cache(chunk: &fusevm::Chunk, dir: &PathBuf, slots: &[i64]) -> Option<Value> {
    let jit = JitCompiler::new();
    jit.set_jit_cache_dir(Some(dir.clone()));
    let r = jit.try_run_linear(chunk, slots);
    jit.set_jit_cache_dir(None);
    r
}

#[test]
fn disk_cache_leaf_matches_interp() {
    let _g = serial();
    let dir = fresh_dir("leaf");
    // (2 + 3) * 4 = 20
    let chunk = build(&[
        (Op::LoadInt(2), 1),
        (Op::LoadInt(3), 1),
        (Op::Add, 1),
        (Op::LoadInt(4), 1),
        (Op::Mul, 1),
    ]);
    assert_eq!(run_with_cache(&chunk, &dir, &[]), Some(Value::Int(20)));
    // A file should have been written to the cache dir.
    let entries: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map_or(false, |x| x == "fjit"))
        .collect();
    assert_eq!(entries.len(), 1, "expected exactly one cached .fjit file");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn disk_cache_pow_uses_host_reloc() {
    let _g = serial();
    let dir = fresh_dir("pow");
    // 2 ** 10 = 1024 — Pow emits a call to a host helper, exercising the
    // Abs8 relocation patching path on load.
    let chunk = build(&[(Op::LoadInt(2), 1), (Op::LoadInt(10), 1), (Op::Pow, 1)]);
    assert_eq!(run_with_cache(&chunk, &dir, &[]), Some(Value::Int(1024)));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn disk_cache_float() {
    let _g = serial();
    let dir = fresh_dir("float");
    let chunk = build(&[
        (Op::LoadFloat(1.5), 1),
        (Op::LoadFloat(2.5), 1),
        (Op::Add, 1),
    ]);
    match run_with_cache(&chunk, &dir, &[]) {
        Some(Value::Float(f)) => assert!((f - 4.0).abs() < 1e-10),
        other => panic!("expected Float(4.0), got {other:?}"),
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn disk_cache_slots() {
    let _g = serial();
    let dir = fresh_dir("slots");
    let chunk = build(&[(Op::GetSlot(0), 1), (Op::GetSlot(1), 1), (Op::Add, 1)]);
    assert_eq!(
        run_with_cache(&chunk, &dir, &[100, 200]),
        Some(Value::Int(300))
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn disk_cache_roundtrip_second_load_from_disk() {
    let _g = serial();
    // First compiler builds + persists the blob. A *second* compiler in a
    // separate thread (fresh TLS in-memory cache) must load it back from disk
    // and produce the identical result, proving the file path works without
    // any in-memory state.
    let dir = fresh_dir("roundtrip");
    let chunk = build(&[
        (Op::LoadInt(7), 1),
        (Op::LoadInt(6), 1),
        (Op::Mul, 1),
    ]);

    let first = run_with_cache(&chunk, &dir, &[]);
    assert_eq!(first, Some(Value::Int(42)));

    let dir2 = dir.clone();
    let chunk2 = chunk.clone();
    let second = std::thread::spawn(move || run_with_cache(&chunk2, &dir2, &[]))
        .join()
        .unwrap();
    assert_eq!(second, Some(Value::Int(42)));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn disk_cache_corrupted_file_rejected() {
    let _g = serial();
    // A garbage cache file must be ignored (bad magic / truncated), the chunk
    // recompiled, and the correct result still returned — no crash.
    let dir = fresh_dir("corrupt");
    let chunk = build(&[(Op::LoadInt(40), 1), (Op::LoadInt(2), 1), (Op::Add, 1)]);

    // Pre-populate the would-be cache path with junk. The filename scheme is
    // "{op_hash:016x}.fjit"; we don't know the hash here, so instead seed a
    // wrong file AND verify a good run still works, then corrupt the real file.
    std::fs::create_dir_all(&dir).unwrap();

    // First good run writes the real file.
    assert_eq!(run_with_cache(&chunk, &dir, &[]), Some(Value::Int(42)));

    // Corrupt every .fjit file in place.
    for e in std::fs::read_dir(&dir).unwrap().flatten() {
        if e.path().extension().map_or(false, |x| x == "fjit") {
            std::fs::write(e.path(), b"not a real blob").unwrap();
        }
    }

    // A fresh thread (no TLS hit) must reject the corrupt file, recompile, and
    // overwrite it — still returning the correct value.
    let dir2 = dir.clone();
    let chunk2 = chunk.clone();
    let r = std::thread::spawn(move || run_with_cache(&chunk2, &dir2, &[]))
        .join()
        .unwrap();
    assert_eq!(r, Some(Value::Int(42)));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn disk_cache_on_by_default_and_opt_out() {
    let _g = serial();
    let jit = JitCompiler::new();
    // Clear any programmatic override left by another test.
    jit.set_jit_cache_dir(None);

    // Default-on: with no override and no env opt-out, a default dir resolves.
    let saved = std::env::var_os("FUSEVM_JIT_CACHE_DIR");
    std::env::remove_var("FUSEVM_JIT_CACHE_DIR");
    let def = jit.jit_cache_dir();
    assert!(
        def.as_ref()
            .map_or(false, |p| p.ends_with("fusevm-jit")),
        "expected default dir ending in fusevm-jit, got {def:?}"
    );

    // Env opt-out disables caching entirely.
    std::env::set_var("FUSEVM_JIT_CACHE_DIR", "off");
    assert_eq!(jit.jit_cache_dir(), None);

    // Restore prior env state for other tests.
    match saved {
        Some(v) => std::env::set_var("FUSEVM_JIT_CACHE_DIR", v),
        None => std::env::remove_var("FUSEVM_JIT_CACHE_DIR"),
    }
}

#[test]
fn disk_cache_in_memory_still_correct_when_disabled() {
    let _g = serial();
    let jit = JitCompiler::new();
    let saved = std::env::var_os("FUSEVM_JIT_CACHE_DIR");
    std::env::set_var("FUSEVM_JIT_CACHE_DIR", "off");
    jit.set_jit_cache_dir(None);
    assert_eq!(jit.jit_cache_dir(), None);
    let chunk = build(&[(Op::LoadInt(21), 1), (Op::Dup, 1), (Op::Add, 1)]);
    assert_eq!(jit.try_run_linear(&chunk, &[]), Some(Value::Int(42)));
    match saved {
        Some(v) => std::env::set_var("FUSEVM_JIT_CACHE_DIR", v),
        None => std::env::remove_var("FUSEVM_JIT_CACHE_DIR"),
    }
}

#[test]
fn disk_cache_set_and_get_dir() {
    let _g = serial();
    let jit = JitCompiler::new();
    let dir = fresh_dir("api");
    jit.set_jit_cache_dir(Some(dir.clone()));
    assert_eq!(jit.jit_cache_dir(), Some(dir));
    jit.set_jit_cache_dir(None);
}

// ── Block tier ──

/// `sum = 0; i = 0; while (i < 100) { sum += i; i++ }` → 4950. Block-JIT
/// eligible (slot ops + a fused loop branch), so it exercises the block-tier
/// native compile + disk persist + mmap-load path.
fn block_sum_loop() -> fusevm::Chunk {
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
    b.emit(Op::SlotLtIntJumpIfFalse(1, 100, 12), 1);
    b.emit(Op::Jump(5), 1);
    b.emit(Op::GetSlot(0), 1);
    b.build()
}

/// True if `dir` contains at least one finalized cache file whose name carries
/// the given tier tag (e.g. `.blk.` or `.trc.`).
fn has_tagged_file(dir: &PathBuf, tag: &str) -> bool {
    let needle = format!(".{tag}.fjit");
    std::fs::read_dir(dir)
        .map(|rd| {
            rd.flatten().any(|e| {
                e.file_name()
                    .to_string_lossy()
                    .ends_with(needle.as_str())
            })
        })
        .unwrap_or(false)
}

fn run_block_with_cache(chunk: &fusevm::Chunk, dir: &PathBuf) -> Option<i64> {
    let jit = JitCompiler::new();
    jit.set_jit_cache_dir(Some(dir.clone()));
    let mut slots = vec![0i64; 4];
    let r = jit.try_run_block_eager(chunk, &mut slots);
    jit.set_jit_cache_dir(None);
    r
}

#[test]
fn disk_cache_block_matches_and_persists() {
    let _g = serial();
    let dir = fresh_dir("block");
    let chunk = block_sum_loop();

    assert_eq!(run_block_with_cache(&chunk, &dir), Some(4950));
    assert!(
        has_tagged_file(&dir, "blk"),
        "expected a .blk.fjit block cache file to be written"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn disk_cache_block_roundtrip_second_thread() {
    let _g = serial();
    // First thread compiles + persists the block blob; a second thread (fresh
    // per-thread cache) must load it from disk and produce the same result.
    let dir = fresh_dir("block_rt");
    let chunk = block_sum_loop();
    assert_eq!(run_block_with_cache(&chunk, &dir), Some(4950));

    let dir2 = dir.clone();
    let chunk2 = chunk.clone();
    let second = std::thread::spawn(move || run_block_with_cache(&chunk2, &dir2))
        .join()
        .unwrap();
    assert_eq!(second, Some(4950));
    let _ = std::fs::remove_dir_all(&dir);
}

/// Tight do-while counter loop; returns (chunk, anchor_ip).
fn counter_loop(limit: i64) -> (fusevm::Chunk, usize) {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(0), 1);
    b.emit(Op::SetSlot(0), 1);
    let anchor = b.current_pos();
    b.emit(Op::PreIncSlotVoid(0), 1);
    b.emit(Op::GetSlot(0), 1);
    b.emit(Op::LoadInt(limit), 1);
    b.emit(Op::NumLt, 1);
    let jmp = b.emit(Op::JumpIfTrue(0), 1);
    b.patch_jump(jmp, anchor);
    b.emit(Op::GetSlot(0), 1);
    (b.build(), anchor)
}

fn run_trace_vm(chunk: &fusevm::Chunk) -> i64 {
    let mut vm = VM::new(chunk.clone());
    vm.enable_tracing_jit();
    {
        let frame = vm.frames.last_mut().unwrap();
        while frame.slots.len() < 1 {
            frame.slots.push(Value::Int(0));
        }
    }
    match vm.run() {
        VMResult::Ok(Value::Int(n)) => n,
        other => panic!("expected Int result, got {other:?}"),
    }
}

#[test]
fn disk_cache_trace_matches_and_persists() {
    let _g = serial();
    let dir = fresh_dir("trace");
    let jit = JitCompiler::new();
    jit.set_jit_cache_dir(Some(dir.clone()));

    let (chunk, _anchor) = counter_loop(200);
    // First run records + compiles the trace; with caching on this builds the
    // native blob, persists it, and runs the freshly-loaded native trace.
    assert_eq!(run_trace_vm(&chunk), 200);
    assert!(
        has_tagged_file(&dir, "trc"),
        "expected a .trc.fjit trace cache file to be written"
    );

    jit.set_jit_cache_dir(None);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn disk_cache_trace_roundtrip_second_thread() {
    let _g = serial();
    let dir = fresh_dir("trace_rt");
    let (chunk, _anchor) = counter_loop(200);

    let dir_a = dir.clone();
    let chunk_a = chunk.clone();
    let first = std::thread::spawn(move || {
        let jit = JitCompiler::new();
        jit.set_jit_cache_dir(Some(dir_a.clone()));
        let r = run_trace_vm(&chunk_a);
        jit.set_jit_cache_dir(None);
        r
    })
    .join()
    .unwrap();
    assert_eq!(first, 200);
    assert!(has_tagged_file(&dir, "trc"));

    // Second thread: empty trace TLS forces a fresh install, which must load
    // the native trace from disk and still produce the correct result.
    let dir_b = dir.clone();
    let chunk_b = chunk.clone();
    let second = std::thread::spawn(move || {
        let jit = JitCompiler::new();
        jit.set_jit_cache_dir(Some(dir_b.clone()));
        let r = run_trace_vm(&chunk_b);
        jit.set_jit_cache_dir(None);
        r
    })
    .join()
    .unwrap();
    assert_eq!(second, 200);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn cache_size_clear_and_max_bytes_api() {
    let _g = serial();
    let dir = fresh_dir("size_api");
    let jit = JitCompiler::new();
    jit.set_jit_cache_dir(Some(dir.clone()));

    // Populate the cache with several distinct linear chunks.
    for k in 0..6 {
        let chunk = build(&[(Op::LoadInt(k), 1), (Op::LoadInt(1), 1), (Op::Add, 1)]);
        let _ = jit.try_run_linear(&chunk, &[]);
    }
    let size = jit.jit_cache_size_bytes().expect("caching enabled");
    assert!(size > 0, "cache should have grown after compiling chunks");

    // Force a tiny cap and prune: size must drop to at most the cap.
    jit.set_jit_cache_max_bytes(Some(200));
    let freed = jit.prune_jit_cache();
    assert!(freed > 0, "prune should have evicted blobs to meet the 200B cap");
    let after = jit.jit_cache_size_bytes().unwrap();
    assert!(after <= 200, "expected ≤200B after prune, got {after}");

    // clear() removes everything.
    let removed = jit.clear_jit_cache();
    assert!(removed > 0);
    assert_eq!(jit.jit_cache_size_bytes().unwrap(), 0);

    // Restore default cap resolution and detach the dir override.
    jit.set_jit_cache_max_bytes(None);
    jit.set_jit_cache_dir(None);
    let _ = std::fs::remove_dir_all(&dir);
}

/// New slot ops (PreDecSlot / PostIncSlot / PostDecSlot): block JIT must match
/// the interpreter. Builds a counted loop that exercises each op and compares
/// the native (eager block JIT) result against a pure-interpreter run.
#[test]
fn new_slot_ops_block_jit_matches_interp() {
    use fusevm::Op::*;
    // slots: 0=i (counter), 1=acc, 2=scratch
    // Loop: while i < 5 { acc += PostIncSlot(scratch-ish)... } — keep it simple:
    // acc = 0; i = 0; do { tmp = i++ (PostIncSlot i); acc = acc + tmp; } while i<5
    // Then j = 5; --j (PreDecSlot); k = j--; (PostDecSlot) push results to acc.
    let ops = vec![
        (PushFrame, 1),
        (LoadInt(0), 1),
        (SetSlot(1), 1), // acc = 0
        (LoadInt(0), 1),
        (SetSlot(0), 1), // i = 0
        // loop header @ ip 5
        (PostIncSlot(0), 1), // push old i, i++
        (GetSlot(1), 1),
        (Add, 1),
        (SetSlot(1), 1), // acc += old_i
        (SlotLtIntJumpIfFalse(0, 5, 14), 1), // if i<5 fallthrough else jump exit(14)
        (Jump(5), 1),
        (Nop, 1),
        (Nop, 1),
        (Nop, 1),
        // exit @ 14: j=10; --j; pre-dec pushes 9; acc+=9
        (LoadInt(10), 1),
        (SetSlot(2), 1),
        (PreDecSlot(2), 1), // push 9
        (GetSlot(1), 1),
        (Add, 1),
        (SetSlot(1), 1),
        // post-dec j(=9) -> push 9, j=8; acc+=9
        (PostDecSlot(2), 1),
        (GetSlot(1), 1),
        (Add, 1),
        (SetSlot(1), 1),
        (GetSlot(1), 1), // result = acc
    ];
    let chunk = build(&ops);

    // Interpreter result.
    let mut vm = VM::new(chunk.clone());
    let interp = match vm.run() {
        VMResult::Ok(v) => v.to_int(),
        other => panic!("interp failed: {:?}", other),
    };

    // Block JIT (eager) result via slot buffer.
    let jit = JitCompiler::new();
    let mut slots = vec![0i64; 4];
    let native = jit
        .try_run_block(&chunk, &mut slots)
        .or_else(|| jit.try_run_block(&chunk, &mut slots))
        .expect("block jit should compile new slot ops");
    // sum of 0..5 = 10, plus 9 + 9 = 28
    assert_eq!(interp, 28, "interp value");
    assert_eq!(native, interp, "block jit must match interpreter");
}
