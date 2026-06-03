//! End-to-end test for extension JIT *host helpers*: a frontend registers a
//! host function via `register_jit_helper` and an extension calls it from
//! block-JIT codegen through `ExtJitCtx::call_host`. This is the mechanism that
//! lets non-numeric ops (here string equality / comparison, with operands
//! passed as `i64` pointer handles) be JIT-compiled *and* persisted to the
//! on-disk native cache — the relocation re-resolves the helper by a stable,
//! name-derived id at load time.

#![cfg(feature = "jit")]

use fusevm::jit::{
    jit_helper_id, register_global_extension, register_jit_helper, ExtJitCtx, JitCompiler,
    JitExtension,
};
use fusevm::{ChunkBuilder, Op};
use std::ffi::CStr;
use std::os::raw::c_char;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

// ── Host helpers (operands are NUL-terminated C-string pointers as i64) ──

extern "C" fn host_str_eq(a: *const c_char, b: *const c_char) -> i64 {
    let (sa, sb) = unsafe { (CStr::from_ptr(a), CStr::from_ptr(b)) };
    i64::from(sa == sb)
}

extern "C" fn host_str_cmp(a: *const c_char, b: *const c_char) -> i64 {
    let (sa, sb) = unsafe { (CStr::from_ptr(a).to_bytes(), CStr::from_ptr(b).to_bytes()) };
    match sa.cmp(sb) {
        std::cmp::Ordering::Less => -1,
        std::cmp::Ordering::Equal => 0,
        std::cmp::Ordering::Greater => 1,
    }
}

const STR_EQ: u16 = 0xC0DE;
const STR_CMP: u16 = 0xC0DF;
const H_EQ_NAME: &str = "fusevm_test_host_str_eq";
const H_CMP_NAME: &str = "fusevm_test_host_str_cmp";

struct StrOpsExt;

impl JitExtension for StrOpsExt {
    fn can_jit(&self, id: u16) -> bool {
        id == STR_EQ || id == STR_CMP
    }
    fn op_count(&self) -> usize {
        2
    }
    fn name(&self) -> &str {
        "test-str-ops"
    }
    fn emit_extended(&self, id: u16, _arg: u8, cx: &mut ExtJitCtx) -> bool {
        let helper = match id {
            STR_EQ => jit_helper_id(H_EQ_NAME),
            STR_CMP => jit_helper_id(H_CMP_NAME),
            _ => return false,
        };
        // Stack: [a, b] with b on top → pop b then a.
        let (Some(b), Some(a)) = (cx.pop_i64(), cx.pop_i64()) else {
            return false;
        };
        match cx.call_host(helper, &[a, b]) {
            Some(r) => {
                cx.push_i64(r);
                true
            }
            None => false,
        }
    }
}

/// Register the extension + host helpers exactly once, process-wide.
fn setup() {
    static ONCE: OnceLock<()> = OnceLock::new();
    ONCE.get_or_init(|| {
        unsafe {
            register_jit_helper(H_EQ_NAME, host_str_eq as *const u8, 2, false);
            register_jit_helper(H_CMP_NAME, host_str_cmp as *const u8, 2, false);
        }
        register_global_extension(Arc::new(StrOpsExt));
    });
}

fn serial() -> MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|p| p.into_inner())
}

/// A chunk that loads two string-pointer slots and applies the given str op.
fn str_op_chunk(op_id: u16) -> fusevm::Chunk {
    let mut b = ChunkBuilder::new();
    b.emit(Op::PushFrame, 0);
    b.emit(Op::GetSlot(0), 0);
    b.emit(Op::GetSlot(1), 0);
    b.emit(Op::Extended(op_id, 0), 0);
    b.build()
}

/// NUL-terminated owned C strings whose stable pointers we feed as slots.
struct CPair {
    _a: std::ffi::CString,
    _b: std::ffi::CString,
    slots: [i64; 2],
}
fn cpair(a: &str, b: &str) -> CPair {
    let ca = std::ffi::CString::new(a).unwrap();
    let cb = std::ffi::CString::new(b).unwrap();
    let slots = [ca.as_ptr() as i64, cb.as_ptr() as i64];
    CPair {
        _a: ca,
        _b: cb,
        slots,
    }
}

fn run_eq(a: &str, b: &str) -> i64 {
    setup();
    let jit = JitCompiler::new();
    jit.set_jit_cache_dir(None); // in-memory only
    let chunk = str_op_chunk(STR_EQ);
    let mut p = cpair(a, b);
    let r = jit
        .try_run_block_eager(&chunk, &mut p.slots)
        .expect("str-eq chunk must JIT-compile via host helper");
    r
}

fn run_cmp(a: &str, b: &str) -> i64 {
    setup();
    let jit = JitCompiler::new();
    jit.set_jit_cache_dir(None);
    let chunk = str_op_chunk(STR_CMP);
    let mut p = cpair(a, b);
    jit.try_run_block_eager(&chunk, &mut p.slots)
        .expect("str-cmp chunk must JIT-compile via host helper")
}

#[test]
fn host_helper_str_eq_matches() {
    assert_eq!(run_eq("hello", "hello"), 1);
    assert_eq!(run_eq("hello", "world"), 0);
    assert_eq!(run_eq("", ""), 1);
    assert_eq!(run_eq("a", "ab"), 0);
}

#[test]
fn host_helper_str_cmp_matches() {
    assert_eq!(run_cmp("abc", "abc"), 0);
    assert_eq!(run_cmp("abc", "abd"), -1);
    assert_eq!(run_cmp("abd", "abc"), 1);
    assert_eq!(run_cmp("ab", "abc"), -1);
}

fn fresh_dir(tag: &str) -> PathBuf {
    let mut d = std::env::temp_dir();
    let n = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    d.push(format!("fusevm-host-helper-{tag}-{n}"));
    let _ = std::fs::create_dir_all(&d);
    d
}

fn has_blk_file(dir: &PathBuf) -> bool {
    std::fs::read_dir(dir)
        .map(|rd| {
            rd.flatten()
                .any(|e| e.file_name().to_string_lossy().ends_with(".blk.fjit"))
        })
        .unwrap_or(false)
}

fn run_eq_with_cache(dir: &PathBuf, a: &str, b: &str) -> Option<i64> {
    setup();
    let jit = JitCompiler::new();
    jit.set_jit_cache_dir(Some(dir.clone()));
    let chunk = str_op_chunk(STR_EQ);
    let mut p = cpair(a, b);
    let r = jit.try_run_block_eager(&chunk, &mut p.slots);
    jit.set_jit_cache_dir(None);
    r
}

#[test]
fn host_helper_chunk_is_disk_cached_and_reloads() {
    let _g = serial();
    let dir = fresh_dir("eq");

    // First run: native-compile a chunk that calls a host helper, persist it.
    assert_eq!(run_eq_with_cache(&dir, "needle", "needle"), Some(1));
    assert!(
        has_blk_file(&dir),
        "a host-helper chunk must still be persisted as a .blk.fjit native blob"
    );

    // Second thread: a fresh per-thread in-memory cache forces a load of the
    // persisted blob from disk. The helper relocation must re-resolve by its
    // stable id, so the reloaded native code produces the same answers.
    let dir2 = dir.clone();
    let handle = std::thread::spawn(move || {
        (
            run_eq_with_cache(&dir2, "needle", "needle"),
            run_eq_with_cache(&dir2, "needle", "haystack"),
        )
    });
    let (same, diff) = handle.join().unwrap();
    assert_eq!(
        same,
        Some(1),
        "reloaded cached host-helper code: equal strings"
    );
    assert_eq!(
        diff,
        Some(0),
        "reloaded cached host-helper code: unequal strings"
    );

    let _ = std::fs::remove_dir_all(&dir);
}
