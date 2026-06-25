//! Emit a closed-world AOT object for a tiny chunk, to validate the
//! object→link→run pipeline. Usage: `emit_aot <out.o>`. The chunk computes
//! `6 * 7` and leaves 42 on the stack, so the linked binary exits with code 42.
//!
//! Run: `cargo run --example emit_aot --features aot -- /tmp/x.o`

#[cfg(feature = "aot")]
fn main() {
    use fusevm::aot;
    use fusevm::chunk::ChunkBuilder;
    use fusevm::op::Op;

    let out = std::env::args().nth(1).expect("usage: emit_aot <out.o>");
    let mut b = ChunkBuilder::new();
    b.emit(Op::LoadInt(6), 1);
    b.emit(Op::LoadInt(7), 1);
    b.emit(Op::Mul, 1);
    let chunk = b.build();
    aot::compile_object(&chunk, std::path::Path::new(&out)).expect("emit object");
    eprintln!("wrote {out}");
}

#[cfg(not(feature = "aot"))]
fn main() {
    eprintln!("build with --features aot");
}
