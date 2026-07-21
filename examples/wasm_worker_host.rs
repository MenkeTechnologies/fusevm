//! Web-worker I/O bridging pattern for fusevm frontends.
//!
//! A frontend (stryke/zshrs/awkrs) running inside a browser web worker compiles
//! its source to a `Chunk` and runs it on a `VM` — but wasm has no real stdout,
//! so `Op::Print` output must be captured and forwarded to the JS host. This
//! example shows the reusable pattern: install an [`OutputSink`] backed by an
//! `Arc<Mutex<String>>`, run the VM, then drain the buffer. In a real worker the
//! drained string is handed to `postMessage`; here it is written to stdout so
//! the example is runnable and self-checking on native targets.
//!
//! The same code compiles for `wasm32-unknown-unknown` (the sink stays `Send`
//! because it captures an `Arc<Mutex<_>>`, not a JS handle):
//!
//! ```text
//! cargo build --example wasm_worker_host --target wasm32-unknown-unknown
//! cargo run   --example wasm_worker_host          # native, prints "hello 42"
//! ```

use std::sync::{Arc, Mutex};

use fusevm::{ChunkBuilder, Op, VMResult, Value, VM};

fn main() {
    // A frontend would produce this chunk from source. Here we hand-build one
    // that prints `hello ` then the number 42 with a trailing newline.
    let mut b = ChunkBuilder::new();
    let greeting = b.add_constant(Value::str("hello "));
    b.emit(Op::LoadConst(greeting), 1);
    b.emit(Op::Print(1), 1);
    b.emit(Op::LoadInt(42), 1);
    b.emit(Op::PrintLn(1), 1);
    let chunk = b.build();

    // The output buffer the worker drains. `Arc<Mutex<String>>` keeps the sink
    // closure `Send`, so `VM` stays `Send` (usable from `VMPool` on native).
    let captured = Arc::new(Mutex::new(String::new()));
    let sink_buf = Arc::clone(&captured);

    let mut vm = VM::new(chunk);
    vm.set_output_sink(Box::new(move |s: &str| {
        sink_buf.lock().unwrap().push_str(s);
    }));

    // Frontends that read input install a source too. This one feeds two lines
    // then signals end-of-input; `Op::ReadLine` pushes `Undef` past the end.
    let mut pending = vec!["line two".to_string(), "line one".to_string()];
    vm.set_input_source(Box::new(move || pending.pop()));

    match vm.run() {
        VMResult::Ok(_) | VMResult::Halted => {}
        VMResult::Error(e) => {
            eprintln!("vm error: {e}");
            std::process::exit(1);
        }
    }

    // In a browser worker: `postMessage(captured)`. Here: assert + print.
    let out = captured.lock().unwrap().clone();
    assert_eq!(out, "hello 42\n", "sink did not capture VM output");
    print!("{out}");
}
