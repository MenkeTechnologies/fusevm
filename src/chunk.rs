//! Bytecode container — a compiled unit of execution.
//!
//! A `Chunk` holds the bytecodes, constant pool, name pool, and metadata
//! for one compilation unit (script, function, block). Language frontends
//! build Chunks via the `ChunkBuilder`.

use crate::op::Op;
use crate::value::Value;
use serde::{Deserialize, Serialize};

/// A compiled bytecode unit.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Chunk {
    /// Bytecode instructions
    pub ops: Vec<Op>,
    /// Constant pool: literals, patterns, format strings
    pub constants: Vec<Value>,
    /// Name pool: variable names, function names (interned/deduped)
    pub names: Vec<String>,
    /// Source line for each op (parallel array for error reporting)
    pub lines: Vec<u32>,
    /// Compiled subroutine entry points: (name_index, op_index)
    pub sub_entries: Vec<(u16, usize)>,
    /// Block regions for map/grep/sort/foreach: (start_ip, end_ip)
    pub block_ranges: Vec<(usize, usize)>,
    /// Sub-chunks for nested execution: `$(cmd)` bodies, `<(cmd)` /
    /// `>(cmd)` bodies, trap handlers, with-redirects bodies, function bodies
    /// when they're stored as separate chunks. Indexed by `Op::CmdSubst(u16)`,
    /// `Op::ProcessSubIn(u16)`, `Op::ProcessSubOut(u16)`, `Op::TrapSet(u16)`.
    pub sub_chunks: Vec<Chunk>,
    /// Source file name (for error messages)
    pub source: String,
    /// Cached hash of ops + constants (computed once at build time for O(1) JIT cache lookup)
    #[serde(skip)]
    pub op_hash: u64,
}

impl Chunk {
    pub fn new() -> Self {
        Self::default()
    }

    /// Find a subroutine entry by name pool index.
    pub fn find_sub(&self, name_idx: u16) -> Option<usize> {
        self.sub_entries
            .iter()
            .find(|(n, _)| *n == name_idx)
            .map(|(_, ip)| *ip)
    }
}

/// Builder for constructing Chunks incrementally.
pub struct ChunkBuilder {
    chunk: Chunk,
    name_map: std::collections::HashMap<String, u16>,
}

impl ChunkBuilder {
    pub fn new() -> Self {
        Self {
            chunk: Chunk::new(),
            name_map: std::collections::HashMap::new(),
        }
    }

    /// Emit an op at the current position.
    pub fn emit(&mut self, op: Op, line: u32) -> usize {
        let idx = self.chunk.ops.len();
        self.chunk.ops.push(op);
        self.chunk.lines.push(line);
        idx
    }

    /// Add a constant to the pool, return its index.
    pub fn add_constant(&mut self, val: Value) -> u16 {
        let idx = self.chunk.constants.len();
        self.chunk.constants.push(val);
        idx as u16
    }

    /// Intern a name, return its pool index.
    pub fn add_name(&mut self, name: &str) -> u16 {
        if let Some(&idx) = self.name_map.get(name) {
            return idx;
        }
        let idx = self.chunk.names.len() as u16;
        self.chunk.names.push(name.to_string());
        self.name_map.insert(name.to_string(), idx);
        idx
    }

    /// Current bytecode position (for jump targets).
    pub fn current_pos(&self) -> usize {
        self.chunk.ops.len()
    }

    /// Patch a jump target at the given op index.
    pub fn patch_jump(&mut self, op_idx: usize, target: usize) {
        match &mut self.chunk.ops[op_idx] {
            Op::Jump(t)
            | Op::JumpIfTrue(t)
            | Op::JumpIfFalse(t)
            | Op::JumpIfTrueKeep(t)
            | Op::JumpIfFalseKeep(t) => *t = target,
            _ => panic!("patch_jump on non-jump op at {}", op_idx),
        }
    }

    /// Register a subroutine entry point.
    pub fn add_sub_entry(&mut self, name_idx: u16, ip: usize) {
        self.chunk.sub_entries.push((name_idx, ip));
    }

    /// Register a block region (for map/grep/sort).
    pub fn add_block_range(&mut self, start: usize, end: usize) -> u16 {
        let idx = self.chunk.block_ranges.len();
        self.chunk.block_ranges.push((start, end));
        idx as u16
    }

    /// Add a nested sub-chunk (for cmd subst, process subst, trap handlers,
    /// function bodies). Returns the index used by `Op::CmdSubst`,
    /// `Op::ProcessSubIn`/`Out`, `Op::TrapSet`.
    pub fn add_sub_chunk(&mut self, sub: Chunk) -> u16 {
        let idx = self.chunk.sub_chunks.len();
        self.chunk.sub_chunks.push(sub);
        idx as u16
    }

    /// Set source file name.
    pub fn set_source(&mut self, source: impl Into<String>) {
        self.chunk.source = source.into();
    }

    /// Finalize and return the chunk with precomputed op hash.
    pub fn build(mut self) -> Chunk {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut h = DefaultHasher::new();
        self.chunk.ops.hash(&mut h);
        self.chunk.constants.hash(&mut h);
        self.chunk.op_hash = h.finish();
        self.chunk
    }
}

impl Default for ChunkBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::op::Op;
    use crate::value::Value;

    #[test]
    fn new_and_default_are_equivalent() {
        let a = ChunkBuilder::new().build();
        let b = ChunkBuilder::default().build();
        assert_eq!(a.ops, b.ops);
        assert_eq!(a.names, b.names);
        assert_eq!(a.constants.len(), b.constants.len());
        assert_eq!(a.op_hash, b.op_hash);
    }

    #[test]
    fn find_sub_returns_first_match_when_duplicate_names_registered() {
        // Builder does not prevent duplicate sub_entries; lookup returns the first.
        let mut b = ChunkBuilder::new();
        let n = b.add_name("foo");
        b.add_sub_entry(n, 10);
        b.add_sub_entry(n, 20);
        let chunk = b.build();
        assert_eq!(chunk.find_sub(n), Some(10));
    }

    #[test]
    fn find_sub_distinguishes_multiple_subs() {
        let mut b = ChunkBuilder::new();
        let f = b.add_name("f");
        let g = b.add_name("g");
        b.add_sub_entry(f, 7);
        b.add_sub_entry(g, 13);
        let chunk = b.build();
        assert_eq!(chunk.find_sub(f), Some(7));
        assert_eq!(chunk.find_sub(g), Some(13));
    }

    #[test]
    fn add_sub_chunk_returns_sequential_indices() {
        let mut b = ChunkBuilder::new();
        let i0 = b.add_sub_chunk(Chunk::new());
        let i1 = b.add_sub_chunk(Chunk::new());
        let i2 = b.add_sub_chunk(Chunk::new());
        assert_eq!((i0, i1, i2), (0, 1, 2));
        assert_eq!(b.build().sub_chunks.len(), 3);
    }

    #[test]
    fn sub_chunks_preserve_inner_content() {
        let inner = {
            let mut ib = ChunkBuilder::new();
            ib.emit(Op::LoadInt(7), 1);
            ib.build()
        };
        let mut b = ChunkBuilder::new();
        let idx = b.add_sub_chunk(inner);
        let outer = b.build();
        assert_eq!(idx, 0);
        assert_eq!(outer.sub_chunks[0].ops, vec![Op::LoadInt(7)]);
    }

    #[test]
    fn add_constant_returns_monotonic_indices() {
        let mut b = ChunkBuilder::new();
        for i in 0..5u16 {
            assert_eq!(b.add_constant(Value::Int(i as i64)), i);
        }
    }

    #[test]
    fn add_name_first_index_is_zero() {
        let mut b = ChunkBuilder::new();
        assert_eq!(b.add_name("first"), 0);
        assert_eq!(b.add_name("second"), 1);
    }

    #[test]
    fn build_computes_nonzero_hash_for_nonempty_chunk() {
        let mut b = ChunkBuilder::new();
        b.emit(Op::LoadInt(1), 1);
        let c = b.build();
        // Extremely unlikely to be exactly 0 for a non-empty chunk.
        assert_ne!(c.op_hash, 0);
    }

    #[test]
    fn op_hash_ignores_line_and_name_pool() {
        // op_hash is derived from ops + constants only.
        let a = {
            let mut b = ChunkBuilder::new();
            b.add_name("alpha");
            b.emit(Op::LoadInt(1), 5);
            b.build()
        };
        let b = {
            let mut b = ChunkBuilder::new();
            b.add_name("beta");
            b.emit(Op::LoadInt(1), 99);
            b.build()
        };
        assert_eq!(a.op_hash, b.op_hash);
    }

    #[test]
    fn set_source_overwrites_previous_value() {
        let mut b = ChunkBuilder::new();
        b.set_source("first.fuse");
        b.set_source("second.fuse");
        assert_eq!(b.build().source, "second.fuse");
    }
}
