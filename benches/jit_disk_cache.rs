//! On-disk native-JIT cache: cold codegen vs cached native load.
//!
//! Measures the per-chunk cost the disk cache eliminates on a fresh process:
//! a full Cranelift compile of a linear chunk versus loading the previously
//! persisted native code (mmap + relocation patch + W^X flip).
//!
//! Run: cargo bench --features jit-disk-cache --bench jit_disk_cache
//!
//! Note: the per-thread in-memory cache would mask repeated compiles, so each
//! "cold compile" sample runs on a freshly spawned thread (empty TLS) and uses
//! a unique chunk so neither the TLS nor the on-disk cache is hit. The "cached
//! load" arm pre-populates the on-disk cache, then times loading it back on a
//! fresh thread.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use fusevm::{Chunk, ChunkBuilder, JitCompiler, Op};

static UNIQ: AtomicU64 = AtomicU64::new(1);

/// A non-constant-foldable slot workload, made unique by `salt` so each chunk
/// has a distinct op_hash (defeating both caches when we want a cold compile).
fn slot_chunk(salt: i64, n: usize) -> Chunk {
    let mut b = ChunkBuilder::new();
    b.emit(Op::GetSlot(0), 1);
    b.emit(Op::LoadInt(salt), 1);
    b.emit(Op::Add, 1);
    for _ in 0..n {
        b.emit(Op::LoadInt(1), 1);
        b.emit(Op::Add, 1);
        b.emit(Op::LoadInt(2), 1);
        b.emit(Op::Mul, 1);
        b.emit(Op::LoadInt(2), 1);
        b.emit(Op::Div, 1);
    }
    b.build()
}

/// A unique block-JIT-eligible loop chunk (sum 0..limit), salted via `limit`
/// so each has a distinct op_hash.
fn block_chunk(limit: i32) -> Chunk {
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

fn bench_dir() -> PathBuf {
    let dir = std::env::temp_dir().join(format!("fusevm_bench_jit_cache_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn bench(c: &mut Criterion) {
    let dir = bench_dir();
    let slots = [7i64];
    let n = 400;

    // Cold compile: fresh thread (empty TLS) + unique chunk + no cache dir.
    c.bench_function("cold_jit_compile", |b| {
        b.iter(|| {
            let salt = UNIQ.fetch_add(1, Ordering::Relaxed) as i64;
            let chunk = slot_chunk(salt, n);
            // Fresh thread => empty linear TLS cache => real codegen every time.
            std::thread::spawn(move || {
                let jit = JitCompiler::new();
                jit.set_jit_cache_dir(None);
                black_box(jit.try_run_linear(black_box(&chunk), black_box(&slots)))
            })
            .join()
            .unwrap()
        })
    });

    // Cached load: pre-build & persist a single chunk, then time loading it
    // back from disk on a fresh thread (empty TLS, on-disk hit).
    let cached_chunk = slot_chunk(-999, n);
    {
        let jit = JitCompiler::new();
        jit.set_jit_cache_dir(Some(dir.clone()));
        let _ = jit.try_run_linear(&cached_chunk, &slots); // write the blob
        jit.set_jit_cache_dir(None);
    }
    c.bench_function("cached_native_load", |b| {
        b.iter(|| {
            let chunk = cached_chunk.clone();
            let dir = dir.clone();
            std::thread::spawn(move || {
                let jit = JitCompiler::new();
                jit.set_jit_cache_dir(Some(dir));
                let r = black_box(jit.try_run_linear(black_box(&chunk), black_box(&slots)));
                jit.set_jit_cache_dir(None);
                r
            })
            .join()
            .unwrap()
        })
    });

    // ── Block tier: cold codegen vs cached native load ──
    c.bench_function("cold_block_compile", |b| {
        b.iter(|| {
            let salt = (UNIQ.fetch_add(1, Ordering::Relaxed) as i32) % 100_000 + 1;
            let chunk = block_chunk(salt);
            std::thread::spawn(move || {
                let jit = JitCompiler::new();
                jit.set_jit_cache_dir(None);
                let mut s = vec![0i64; 4];
                black_box(jit.try_run_block_eager(black_box(&chunk), &mut s))
            })
            .join()
            .unwrap()
        })
    });

    let cached_block = block_chunk(1234);
    {
        let jit = JitCompiler::new();
        jit.set_jit_cache_dir(Some(dir.clone()));
        let mut s = vec![0i64; 4];
        let _ = jit.try_run_block_eager(&cached_block, &mut s); // write the blob
        jit.set_jit_cache_dir(None);
    }
    c.bench_function("cached_block_load", |b| {
        b.iter(|| {
            let chunk = cached_block.clone();
            let dir = dir.clone();
            std::thread::spawn(move || {
                let jit = JitCompiler::new();
                jit.set_jit_cache_dir(Some(dir));
                let mut s = vec![0i64; 4];
                let r = black_box(jit.try_run_block_eager(black_box(&chunk), &mut s));
                jit.set_jit_cache_dir(None);
                r
            })
            .join()
            .unwrap()
        })
    });

    let _ = std::fs::remove_dir_all(&dir);
}

criterion_group!(benches, bench);
criterion_main!(benches);
