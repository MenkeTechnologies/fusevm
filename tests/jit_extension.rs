//! End-to-end test for the `JitExtension::emit_extended` hook: a frontend
//! registers a language-specific op (here Perl/strykelang-style *floored*
//! modulo) and the block JIT lowers it to native code purely through the
//! `ExtJitCtx` integer helpers — no direct Cranelift dependency in the
//! frontend, and no host-helper relocations, so the result is also cacheable.

#![cfg(feature = "jit")]

use fusevm::jit::{register_global_extension, ExtJitCtx, JitCompiler, JitExtension, JitValue};
use fusevm::{ChunkBuilder, Op};
use std::sync::Arc;

/// Extension op id for floored modulo (frontend-owned id space).
const FLOORED_MOD: u16 = 0xF00D;

struct FlooredModExt;

impl JitExtension for FlooredModExt {
    fn can_jit(&self, id: u16) -> bool {
        id == FLOORED_MOD
    }
    fn op_count(&self) -> usize {
        1
    }
    fn name(&self) -> &str {
        "test-floored-mod"
    }

    /// floored(a, b) = truncated(a % b), adjusted by +b when the remainder is
    /// non-zero and its sign differs from the divisor. Guards b ∈ {0, -1} so the
    /// hardware divide never traps (both cases yield 0 mathematically/by
    /// definition for the degenerate b==0 path).
    fn emit_extended(&self, id: u16, _arg: u8, cx: &mut ExtJitCtx) -> bool {
        if id != FLOORED_MOD {
            return false;
        }
        let (Some(b), Some(a)) = (cx.pop_i64(), cx.pop_i64()) else {
            return false;
        };
        let zero = cx.iconst(0);
        let one = cx.iconst(1);
        let neg1 = cx.iconst(-1);

        let is_zero = cx.icmp_eq(b, zero);
        let is_neg1 = cx.icmp_eq(b, neg1);
        let special: JitValue = cx.bor(is_zero, is_neg1);
        let safe_b = cx.select(special, one, b);

        let t = cx.srem(a, safe_b);
        let xor = cx.bxor(t, safe_b);
        let signs_differ = cx.icmp_slt(xor, zero);
        let t_nonzero = cx.icmp_ne(t, zero);
        let need = cx.band(signs_differ, t_nonzero);
        let adj = cx.select(need, safe_b, zero);
        let floored = cx.iadd(t, adj);

        let result = cx.select(special, zero, floored);
        cx.push_i64(result);
        true
    }
}

fn floored_mod_chunk(a: i64, b: i64) -> fusevm::Chunk {
    let mut bld = ChunkBuilder::new();
    bld.emit(Op::PushFrame, 0);
    bld.emit(Op::LoadInt(a), 0);
    bld.emit(Op::LoadInt(b), 0);
    bld.emit(Op::Extended(FLOORED_MOD, 0), 0);
    bld.build()
}

fn run(a: i64, b: i64) -> i64 {
    let jit = JitCompiler::new();
    let chunk = floored_mod_chunk(a, b);
    let mut slots: [i64; 0] = [];
    jit.try_run_block_eager(&chunk, &mut slots)
        .expect("extension chunk must JIT-compile and run")
}

/// Reference floored modulo (Perl `%` / strykelang semantics).
fn perl_mod(a: i64, b: i64) -> i64 {
    if b == 0 || b == -1 {
        return 0;
    }
    let r = a % b;
    if r != 0 && ((r ^ b) < 0) {
        r + b
    } else {
        r
    }
}

#[test]
fn registered_extension_makes_chunk_block_eligible() {
    register_global_extension(Arc::new(FlooredModExt));
    let jit = JitCompiler::new();
    let chunk = floored_mod_chunk(-7, 3);
    assert!(
        jit.is_block_eligible(&chunk),
        "a chunk using a registered extension op must be block-JIT eligible"
    );
}

#[test]
fn unregistered_extension_id_is_not_eligible() {
    register_global_extension(Arc::new(FlooredModExt));
    let jit = JitCompiler::new();
    let mut bld = ChunkBuilder::new();
    bld.emit(Op::PushFrame, 0);
    bld.emit(Op::LoadInt(1), 0);
    bld.emit(Op::LoadInt(2), 0);
    bld.emit(Op::Extended(0x0BAD, 0), 0); // no extension handles this id
    let chunk = bld.build();
    assert!(!jit.is_block_eligible(&chunk));
}

#[test]
fn floored_mod_matches_perl_semantics() {
    register_global_extension(Arc::new(FlooredModExt));
    let cases = [
        (7, 3),
        (-7, 3),
        (7, -3),
        (-7, -3),
        (6, 3),
        (-6, 3),
        (10, 4),
        (-10, 4),
        (1, 1000),
        (-1, 1000),
        (i64::MIN, -1), // would trap a raw idiv; guard returns 0
        (5, 0),         // degenerate; guard returns 0 instead of trapping
        (123_456, 789),
        (-123_456, 789),
    ];
    for (a, b) in cases {
        assert_eq!(
            run(a, b),
            perl_mod(a, b),
            "floored_mod({a}, {b}) mismatch vs reference"
        );
    }
}
