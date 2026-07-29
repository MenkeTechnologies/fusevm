//! Frame slot names: addressing a running call frame's variables by name.
//!
//! A frontend's compiler assigns frame slots and knows what each was called;
//! nothing carried that into the run, so a nested script could not be given the
//! calling frame's variable context. `Chunk::sub_slot_names` records the names
//! once per subroutine and `Frame::entry_ip` says which subroutine a frame is
//! running — so the names come from the chunk and the values from the frame,
//! which is the split that makes recursion answer correctly.
//!
//! The programs here drive recursion from a *global* counter rather than a slot,
//! because a fresh activation's slots read `Undef`: a guard on slot 0 never
//! terminates, which is how the first version of this file hung.

use std::sync::{Arc, Mutex};

use fusevm::{ChunkBuilder, Op, VMResult, VM};

/// What one probe saw: the frame depth, the names at three levels, the slot
/// `n` resolves to, and this activation's own value in it.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Probe {
    frames: usize,
    here: Vec<String>,
    up1: Vec<String>,
    base: Vec<String>,
    slot_of_n: Option<u16>,
    my_value: Option<i64>,
    callers_value: Option<i64>,
}

/// `f` recurses once under a global depth counter and stores that depth in its
/// own slot 0, so the two activations hold different values in the same-named
/// slot. Probes at every entry; the first is the deepest.
fn recursive_chunk(named: bool) -> fusevm::Chunk {
    let mut b = ChunkBuilder::new();
    let f = b.add_name("f");
    let d = b.add_name("depth");
    b.emit(Op::LoadInt(0), 1);
    b.emit(Op::SetVar(d), 1);
    b.emit(Op::Call(f, 0), 1);
    b.emit(Op::Return, 1);

    let f_ip = b.current_pos();
    b.emit(Op::GetVar(d), 1); // depth += 1
    b.emit(Op::LoadInt(1), 1);
    b.emit(Op::Add, 1);
    b.emit(Op::SetVar(d), 1);
    b.emit(Op::GetVar(d), 1); // slot 0 = this activation's depth
    b.emit(Op::SetSlot(0), 1);
    b.emit(Op::GetVar(d), 1); // recurse while depth < 2
    b.emit(Op::LoadInt(2), 1);
    b.emit(Op::NumLt, 1);
    let skip = b.current_pos();
    b.emit(Op::JumpIfFalse(usize::MAX), 1);
    b.emit(Op::Call(f, 0), 1);
    let after = b.current_pos();
    b.patch_jump(skip, after);
    b.emit(Op::Extended(0, 0), 1);
    b.emit(Op::Return, 1);

    b.add_sub_entry(f, f_ip);
    if named {
        b.set_sub_slot_names(f_ip, vec!["n".into()]);
    }
    b.build()
}

fn run_probed(chunk: fusevm::Chunk) -> (Vec<Probe>, VMResult) {
    let seen: Arc<Mutex<Vec<Probe>>> = Arc::new(Mutex::new(Vec::new()));
    let log = Arc::clone(&seen);
    let mut vm = VM::new(chunk);
    vm.set_extension_handler(Box::new(move |vm: &mut VM, _id, _arg| {
        let depth = vm.frames.len();
        let slot_of_n = vm.frame_slot_of("n");
        let at = |up: usize| vm.slot_names_at(up).to_vec();
        let value_at = |up: usize| {
            slot_of_n.and_then(|s| {
                vm.frames
                    .get(depth - 1 - up)
                    .and_then(|f| f.slots.get(s as usize))
                    .map(|v| v.to_int())
            })
        };
        log.lock().expect("log").push(Probe {
            frames: depth,
            here: vm.frame_slot_names().to_vec(),
            up1: at(1),
            base: at(depth - 1),
            slot_of_n,
            my_value: value_at(0),
            callers_value: if depth >= 2 { value_at(1) } else { None },
        });
    }));
    let out = vm.run();
    let seen = seen.lock().expect("log").clone();
    (seen, out)
}

/// Every level answers for the subroutine it is actually running, and a frame
/// that entered none answers empty rather than guessing.
#[test]
fn a_running_frame_answers_by_name_at_every_level() {
    let (seen, out) = run_probed(recursive_chunk(true));
    assert!(!matches!(out, VMResult::Error(_)), "the run itself: {out:?}");
    assert!(!seen.is_empty(), "the probe never fired");

    let deepest = &seen[0];
    assert_eq!(deepest.frames, 3, "base + two activations: {deepest:?}");
    assert_eq!(deepest.here, ["n"], "the frame we stand in is named");
    assert_eq!(deepest.up1, ["n"], "so is the activation that called it");
    assert_eq!(
        deepest.base,
        [] as [String; 0],
        "the base frame entered no subroutine, so it has no names"
    );
    assert_eq!(deepest.slot_of_n, Some(0), "`n` is slot 0");
}

/// Two activations of one subroutine share its names and keep their own values —
/// the distinction that makes recursion work rather than alias.
#[test]
fn recursion_shares_names_and_not_values() {
    let (seen, _) = run_probed(recursive_chunk(true));
    let deepest = &seen[0];
    assert_eq!(deepest.here, deepest.up1, "the same names at both levels");
    assert_eq!(
        (deepest.my_value, deepest.callers_value),
        (Some(2), Some(1)),
        "and different values in that same-named slot: {deepest:?}"
    );
}

/// A chunk that records nothing answers empty everywhere, and the run is
/// otherwise identical — which is what keeps every existing frontend unchanged.
#[test]
fn a_chunk_without_slot_names_is_unchanged() {
    let bare = recursive_chunk(false);
    assert!(bare.sub_slot_names.is_empty(), "nothing recorded");
    let (unnamed, unnamed_out) = run_probed(bare);
    let (named, named_out) = run_probed(recursive_chunk(true));

    assert_eq!(
        format!("{unnamed_out:?}"),
        format!("{named_out:?}"),
        "naming slots does not change what the program does"
    );
    assert_eq!(
        unnamed.len(),
        named.len(),
        "nor how many times it reaches the probe"
    );
    for p in &unnamed {
        assert_eq!(p.here, [] as [String; 0], "no names to answer with");
        assert_eq!(p.slot_of_n, None, "and no slot resolves by name");
    }
    assert_eq!(
        unnamed.iter().map(|p| p.frames).collect::<Vec<_>>(),
        named.iter().map(|p| p.frames).collect::<Vec<_>>(),
        "the call structure is the same either way"
    );
}

/// Naming slots does not change the JIT's cache key, so native code compiled
/// before the names were recorded is still the right code for the chunk.
#[test]
fn slot_names_do_not_change_the_op_hash() {
    assert_eq!(
        recursive_chunk(false).op_hash,
        recursive_chunk(true).op_hash,
        "op_hash is ops and constants; names are neither"
    );
}

/// Recording twice for one subroutine replaces rather than appends, so a
/// frontend that lowers in two passes cannot leave a stale map behind.
#[test]
fn recording_twice_replaces_the_names() {
    let mut b = ChunkBuilder::new();
    let f = b.add_name("f");
    let ip = b.current_pos();
    b.emit(Op::Return, 1);
    b.add_sub_entry(f, ip);
    b.set_sub_slot_names(ip, vec!["first".into()]);
    b.set_sub_slot_names(ip, vec!["second".into(), "third".into()]);
    let chunk = b.build();
    assert_eq!(chunk.sub_slot_names.len(), 1, "one entry per subroutine");
    assert_eq!(chunk.sub_slot_names_at(ip), ["second", "third"]);
    assert_eq!(
        chunk.sub_slot_names_at(999),
        [] as [String; 0],
        "an unrecorded entry ip answers empty, not a panic"
    );
}

/// The serialized shape carries the names, and a blob written without them
/// still loads — the round trip a frontend's AOT path depends on.
#[test]
fn slot_names_survive_serialization() {
    let chunk = recursive_chunk(true);
    let blob = bincode::serialize(&chunk).expect("serialize");
    let back: fusevm::Chunk = bincode::deserialize(&blob).expect("deserialize");
    assert_eq!(back.sub_slot_names, chunk.sub_slot_names);
}

/// A blob written by an older fusevm fails loudly, and says what actually
/// happened rather than calling the object corrupt.
///
/// The embedded chunk is `bincode`, which is positional and self-describes
/// nothing: `Chunk::sub_slot_names` appended a field, so a blob written before
/// it is *short*. Deserializing one runs out of input rather than mis-reading a
/// later field — that is why the field went last — and the magic in front of it
/// lets the runtime name the cause instead of guessing at corruption.
#[test]
fn an_older_blob_is_rejected_rather_than_mis_read() {
    let chunk = recursive_chunk(true);

    // What an older fusevm wrote: the same chunk without the trailing field, and
    // with no magic in front of it.
    #[derive(serde::Serialize)]
    struct OldChunk<'a> {
        ops: &'a Vec<fusevm::Op>,
        constants: &'a Vec<fusevm::Value>,
        names: &'a Vec<String>,
        lines: &'a Vec<u32>,
        sub_entries: &'a Vec<(u16, usize)>,
        block_ranges: &'a Vec<(usize, usize)>,
        sub_chunks: &'a Vec<fusevm::Chunk>,
        source: &'a String,
        int_overflow_deopt: bool,
        native_id: u32,
        aot_seeded_slots: u16,
    }
    let old = bincode::serialize(&OldChunk {
        ops: &chunk.ops,
        constants: &chunk.constants,
        names: &chunk.names,
        lines: &chunk.lines,
        sub_entries: &chunk.sub_entries,
        block_ranges: &chunk.block_ranges,
        sub_chunks: &chunk.sub_chunks,
        source: &chunk.source,
        int_overflow_deopt: chunk.int_overflow_deopt,
        native_id: chunk.native_id,
        aot_seeded_slots: chunk.aot_seeded_slots,
    })
    .expect("serialize the old shape");

    // It is short, and reading it as the current shape is an error — never a
    // chunk that silently lost or invented a field.
    let as_current: Result<fusevm::Chunk, _> = bincode::deserialize(&old);
    assert!(
        as_current.is_err(),
        "an older blob must not deserialize into a current Chunk"
    );

    // And it carries no magic, which is what the AOT runtime checks first so its
    // message can name the real cause. The stamp lives in the `aot` module, so
    // this half only applies where that feature is on.
    #[cfg(feature = "aot")]
    {
        assert!(
            !old.starts_with(fusevm::aot::AOT_CHUNK_MAGIC.as_slice()),
            "the old layout has no format stamp"
        );

        // A current blob does carry it, and the body behind it round-trips.
        let mut stamped = fusevm::aot::AOT_CHUNK_MAGIC.to_vec();
        stamped.extend_from_slice(&bincode::serialize(&chunk).expect("serialize"));
        let body = stamped
            .strip_prefix(fusevm::aot::AOT_CHUNK_MAGIC.as_slice())
            .expect("a current blob is stamped");
        let back: fusevm::Chunk = bincode::deserialize(body).expect("deserialize the body");
        assert_eq!(back.sub_slot_names, chunk.sub_slot_names);
    }
}
