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
    /// Source file name (for error messages)
    pub source: String,
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

    /// Set source file name.
    pub fn set_source(&mut self, source: impl Into<String>) {
        self.chunk.source = source.into();
    }

    /// Finalize and return the chunk.
    pub fn build(self) -> Chunk {
        self.chunk
    }
}

impl Default for ChunkBuilder {
    fn default() -> Self {
        Self::new()
    }
}
