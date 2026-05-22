//! JitCompiler public API smoke tests. These compile/run with or without
//! the `jit` feature — when the feature is off, the stubs return false/None
//! for `is_eligible`, `try_run_*`, etc.

use fusevm::jit::{JitCompiler, JitExtension};
use fusevm::{ChunkBuilder, Op, Value};

struct FakeExt {
    op_ids: Vec<u16>,
    label: String,
}

impl JitExtension for FakeExt {
    fn can_jit(&self, ext_id: u16) -> bool {
        self.op_ids.contains(&ext_id)
    }
    fn op_count(&self) -> usize {
        self.op_ids.len()
    }
    fn name(&self) -> &str {
        &self.label
    }
}

#[test]
fn new_and_default_yield_usable_compiler() {
    let _c1 = JitCompiler::new();
    let _c2 = JitCompiler::default();
}

#[test]
fn register_extension_does_not_panic() {
    let mut c = JitCompiler::new();
    c.register_extension(Box::new(FakeExt {
        op_ids: vec![1, 2, 3],
        label: "fake".into(),
    }));
    // Just verifying call path runs without panic; the extension list is
    // private and consulted internally during JIT compilation.
}

#[test]
fn register_multiple_extensions() {
    let mut c = JitCompiler::new();
    for i in 0..5 {
        c.register_extension(Box::new(FakeExt {
            op_ids: vec![i as u16],
            label: format!("ext-{}", i),
        }));
    }
}

#[test]
fn empty_chunk_is_jit_eligible_or_not_without_panic() {
    // The eligibility predicate must not panic on any well-formed chunk,
    // including the empty one. Outcome (true/false) depends on feature flag.
    let c = JitCompiler::new();
    let chunk = ChunkBuilder::new().build();
    let _ = c.is_eligible(&chunk);
}

#[test]
fn simple_arithmetic_chunk_does_not_panic_eligibility_check() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::LoadInt(2), 1);
    b.emit(Op::Add, 1);
    let c = JitCompiler::new();
    let _ = c.is_eligible(&b.build());
}

#[test]
fn chunk_with_extended_op_does_not_panic_eligibility_check() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::Extended(7, 42), 1);
    let c = JitCompiler::new();
    let _ = c.is_eligible(&b.build());
}

#[test]
fn try_run_linear_does_not_panic() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(2), 1);
    b.emit(Op::LoadInt(3), 1);
    b.emit(Op::Mul, 1);
    let c = JitCompiler::new();
    let slots = [0i64; 4];
    let _ = c.try_run_linear(&b.build(), &slots);
}

#[test]
fn try_run_block_does_not_panic() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(2), 1);
    b.emit(Op::LoadInt(3), 1);
    b.emit(Op::Add, 1);
    let c = JitCompiler::new();
    let mut slots = [0i64; 4];
    let _ = c.try_run_block(&b.build(), &mut slots);
}

#[test]
fn try_run_block_eager_does_not_panic() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(7), 1);
    let c = JitCompiler::new();
    let mut slots = [0i64; 4];
    let _ = c.try_run_block_eager(&b.build(), &mut slots);
}

#[test]
fn find_jit_region_callable_on_empty_chunk() {
    let c = JitCompiler::new();
    let chunk = ChunkBuilder::new().build();
    let _ = c.find_jit_region(&chunk);
}

#[test]
fn extract_region_returns_sub_chunk() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::LoadInt(2), 1);
    b.emit(Op::Add, 1);
    b.emit(Op::LoadInt(99), 1);
    let chunk = b.build();
    let c = JitCompiler::new();
    let extracted = c.extract_region(&chunk, 0, 3);
    // Extracted chunk should be a valid (possibly empty) Chunk.
    assert!(extracted.ops.len() <= chunk.ops.len());
}

#[test]
fn block_jit_is_compiled_starts_false_for_uncompiled_chunk() {
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(1), 1);
    let chunk = b.build();
    let c = JitCompiler::new();
    assert!(!c.block_jit_is_compiled(&chunk));
}

#[test]
fn is_linear_eligible_does_not_panic_on_various_chunks() {
    let c = JitCompiler::new();
    let empty = ChunkBuilder::new().build();
    let _ = c.is_linear_eligible(&empty);
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(5), 1);
    b.emit(Op::LoadInt(7), 1);
    b.emit(Op::Add, 1);
    let _ = c.is_linear_eligible(&b.build());
}

#[test]
fn is_block_eligible_does_not_panic() {
    let c = JitCompiler::new();
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(1), 1);
    let _ = c.is_block_eligible(&b.build());
}

#[test]
fn fake_extension_can_jit_only_registered_ids() {
    let e = FakeExt {
        op_ids: vec![10, 20, 30],
        label: "ids".into(),
    };
    assert!(e.can_jit(10));
    assert!(e.can_jit(20));
    assert!(e.can_jit(30));
    assert!(!e.can_jit(0));
    assert!(!e.can_jit(11));
    assert_eq!(e.op_count(), 3);
    assert_eq!(e.name(), "ids");
}

#[test]
fn fake_extension_with_empty_ids() {
    let e = FakeExt {
        op_ids: vec![],
        label: "none".into(),
    };
    assert!(!e.can_jit(0));
    assert!(!e.can_jit(u16::MAX));
    assert_eq!(e.op_count(), 0);
    assert_eq!(e.name(), "none");
}

#[test]
fn jit_compiler_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    // JitCompiler should be Send + Sync — it's used across threads in
    // some frontends. Compile-time assertion via marker traits.
    // (Empty body: succeeds if the bound holds.)
    let _ = assert_send_sync::<JitCompiler>;
    // Smoke construction across thread:
    let handle = std::thread::spawn(|| {
        let _ = JitCompiler::new();
        let _ = Value::int(7);
    });
    handle.join().unwrap();
}
