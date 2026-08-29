//! Round-trip of a `Chunk` through an rkyv archive, under the optional
//! `rkyv-archive` feature.
//!
//! This guards the three things that made the derive non-trivial:
//!   * `Chunk.sub_chunks: Vec<Chunk>` and `Value::Ref(Box<Value>)` are
//!     RECURSIVE, so they carry `#[omit_bounds]`. rkyv reads that attribute
//!     from FIELD attrs only — a variant-level attribute is silently ignored,
//!     which compiles for structs and fails for enums.
//!   * `Value::Str`/`Array` hold `Arc`, a SHARED pointer. Serializing needs a
//!     `SharedSerializeRegistry` and validating needs a `SharedContext`; the
//!     plain `rkyv::check_archived_root` path exercises both.
//!   * `Chunk.op_hash` is `#[serde(skip)]` but has NO rkyv equivalent, so it
//!     IS archived. The test pins that difference deliberately: bincode drops
//!     it and rkyv preserves it.
#![cfg(feature = "rkyv-archive")]

use fusevm::chunk::Chunk;
use fusevm::op::Op;
use fusevm::value::Value;
use std::collections::HashMap;
use std::sync::Arc;

/// A chunk that hits every awkward field: recursion, both `Arc` variants, a
/// map, a boxed ref, and a non-empty name/line pool.
fn nasty_chunk() -> Chunk {
    let mut hash = HashMap::new();
    hash.insert("k".to_string(), Value::Int(7));
    hash.insert("nested".to_string(), Value::Ref(Box::new(Value::Float(2.5))));

    let inner = Chunk {
        ops: vec![Op::Return],
        constants: vec![Value::Str(Arc::new("inner".into()))],
        names: vec!["sub".into()],
        lines: vec![1],
        source: "inner.zsh".into(),
        ..Default::default()
    };

    Chunk {
        ops: vec![Op::Nop, Op::Return],
        constants: vec![
            Value::Undef,
            Value::Bool(true),
            Value::Int(-9),
            Value::Float(1.5),
            Value::Str(Arc::new("hello".into())),
            Value::Array(Arc::new(vec![Value::Int(1), Value::Str(Arc::new("two".into()))])),
            Value::Hash(hash),
            Value::Ref(Box::new(Value::Int(3))),
            Value::Status(0),
            Value::NativeFn(4),
            Value::Obj(11),
        ],
        names: vec!["a".into(), "b".into()],
        lines: vec![10, 20],
        sub_entries: vec![(0, 1)],
        block_ranges: vec![(0, 2)],
        sub_chunks: vec![inner],
        source: "outer.zsh".into(),
        op_hash: 0xDEAD_BEEF,
        ..Default::default()
    }
}

#[test]
fn chunk_survives_archive_validate_and_deserialize() {
    let original = nasty_chunk();
    let bytes = rkyv::to_bytes::<_, 4096>(&original).expect("serialize");

    // Validation is the half most likely to break on the Arc shared pointers.
    let archived = rkyv::check_archived_root::<Chunk>(&bytes[..])
        .expect("check_archived_root must validate a chunk containing Arc values");

    // Zero-copy reads straight off the archive — no deserialize yet.
    assert_eq!(archived.source, "outer.zsh");
    assert_eq!(archived.ops.len(), 2);
    assert_eq!(archived.names.len(), 2);
    assert_eq!(archived.sub_chunks.len(), 1);
    assert_eq!(archived.sub_chunks[0].source, "inner.zsh");

    let restored: Chunk = {
        use rkyv::Deserialize;
        archived
            .deserialize(&mut rkyv::de::deserializers::SharedDeserializeMap::new())
            .expect("deserialize")
    };

    assert_eq!(restored.ops, original.ops);
    assert_eq!(restored.constants, original.constants);
    assert_eq!(restored.names, original.names);
    assert_eq!(restored.lines, original.lines);
    assert_eq!(restored.sub_entries, original.sub_entries);
    assert_eq!(restored.block_ranges, original.block_ranges);
    assert_eq!(restored.source, original.source);
    assert_eq!(restored.sub_chunks.len(), 1);
    assert_eq!(restored.sub_chunks[0].constants, original.sub_chunks[0].constants);
}

/// `op_hash` is `#[serde(skip)]`, so the two codecs DISAGREE on it by
/// construction. Pinned so nobody "fixes" one side without noticing.
#[test]
fn op_hash_is_archived_by_rkyv_but_dropped_by_serde() {
    let original = nasty_chunk();
    assert_eq!(original.op_hash, 0xDEAD_BEEF);

    let bytes = rkyv::to_bytes::<_, 4096>(&original).unwrap();
    let archived = rkyv::check_archived_root::<Chunk>(&bytes[..]).unwrap();
    assert_eq!(archived.op_hash, 0xDEAD_BEEF, "rkyv archives op_hash");

    let round: Chunk = bincode::deserialize(&bincode::serialize(&original).unwrap()).unwrap();
    assert_eq!(round.op_hash, 0, "serde skips op_hash, so it comes back zero");
}
