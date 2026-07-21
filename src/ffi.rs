//! Inline Rust FFI runtime — shared across every fusevm frontend.
//!
//! A frontend surfaces a `rust { ... }` block (see [`crate::rust_sugar`]) which
//! is desugared, at the source level, into a builtin call carrying the block
//! body base64-encoded. That builtin calls [`crate::ffi::compile_and_register`];
//! every bareword call the frontend cannot otherwise resolve falls through to
//! [`crate::ffi::try_call`]. The body is compiled to a `cdylib` on first run,
//! cached,
//! `dlopen`ed, and its `pub extern "C" fn` exports are registered as callable
//! functions marshalled against [`crate::Value`].
//!
//! Flow (driven by [`crate::ffi::compile_and_register`]):
//! 1. Base64-decode the block body.
//! 2. SHA-256(salt ++ body) → short hex cache key.
//! 3. If `~/.cache/fusevm/ffi/lib<hash>.<ext>` exists, `dlopen` and register.
//! 4. Otherwise: write the body wrapped in a minimal `cdylib` crate stub,
//!    invoke `rustc --edition=2021 -O`, then `dlopen`.
//! 5. Scan the body for `pub extern "C" fn NAME(args) -> ret`, match each
//!    signature against the v1 table, `dlsym`, and register one entry each in
//!    the per-process `FFI_REGISTRY`.
//!
//! ## Supported signatures (v1)
//!
//! | rust signature                          | args → return    |
//! |-----------------------------------------|------------------|
//! | `fn([i64; 0..=4]) -> i64`               | ints    → int    |
//! | `fn([f64; 0..=3]) -> f64`               | floats  → float  |
//! | `fn(*const c_char) -> i64`              | string  → int    |
//! | `fn(*const c_char) -> *const c_char`    | string  → string |
//!
//! Deliberately tiny — it covers the crc32 / hashing / numeric-kernel cases
//! that motivate inline FFI. Extending it is one function-pointer type plus one
//! match arm per entry; a future revision can drop in libffi if the set grows
//! past what hand-enumeration can handle.
//!
//! ## Requirements at runtime
//!
//! `rustc` must be on `PATH` (override with `$RUSTC`). First-run compilation
//! costs ~1-2 s; subsequent runs hit the cache and pay only `dlopen`. A clear
//! error is returned when `rustc` is missing — the frontend should not silently
//! fall back.
//!
//! ## Limitations
//!
//! - The cdylib runs with the calling process's privileges.
//! - `*const c_char` returns are copied into a fusevm string immediately; when
//!   the lib exports `fusevm_free_cstring(*mut c_char)` it is freed after the
//!   copy, otherwise the allocation leaks (inline blocks may omit it).
//! - The body must be self-contained Rust — `std` only, no external crates.

use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::fs;
use std::os::raw::c_char;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

use sha2::{Digest, Sha256};

use crate::value::Value;

/// One registered FFI entry: a signature kind + a raw symbol pointer. The dylib
/// handle is kept alive in [`Registry::libs`] so `sym` stays valid.
#[derive(Clone)]
struct FfiEntry {
    sig: FfiSig,
    sym: usize, // raw `*const ()` boxed as usize for Send+Sync
    /// Optional `fusevm_free_cstring(*mut c_char)` export from the same lib.
    free_sym: Option<usize>,
}

// SAFETY: the pointer is a function pointer in a shared library kept alive by
// [`Registry::libs`] for the process lifetime; it is never freed, and calling
// it is already `unsafe`. This is the usual dlopen `Send + Sync` pattern.
unsafe impl Send for FfiEntry {}
unsafe impl Sync for FfiEntry {}

#[derive(Clone, Copy, Debug)]
enum FfiSig {
    I0,       // fn() -> i64
    I1,       // fn(i64) -> i64
    I2,       // fn(i64, i64) -> i64
    I3,       // fn(i64, i64, i64) -> i64
    I4,       // fn(i64, i64, i64, i64) -> i64
    F0,       // fn() -> f64
    F1,       // fn(f64) -> f64
    F2,       // fn(f64, f64) -> f64
    F3,       // fn(f64, f64, f64) -> f64
    StrToInt, // fn(*const c_char) -> i64
    StrToStr, // fn(*const c_char) -> *const c_char
}

impl FfiSig {
    fn arity(self) -> usize {
        match self {
            FfiSig::I0 | FfiSig::F0 => 0,
            FfiSig::I1 | FfiSig::F1 | FfiSig::StrToInt | FfiSig::StrToStr => 1,
            FfiSig::I2 | FfiSig::F2 => 2,
            FfiSig::I3 | FfiSig::F3 => 3,
            FfiSig::I4 => 4,
        }
    }
}

/// Global registry: function name → entry. Populated by [`compile_and_register`],
/// looked up by [`try_call`].
struct Registry {
    entries: HashMap<String, FfiEntry>,
    /// Loaded shared libraries, kept alive for the process lifetime.
    libs: Vec<LoadedLib>,
}

struct LoadedLib {
    path: PathBuf,
    // Raw dlopen handle; retained solely so the symbol tables stay mapped for
    // the process lifetime. We never `dlclose`.
    #[allow(dead_code)]
    handle: usize,
    #[allow(dead_code)]
    free_sym: Option<usize>,
}

// SAFETY: dlopen handles are opaque and only closed on process exit here; all
// access is through read-only lookups.
unsafe impl Send for LoadedLib {}
unsafe impl Sync for LoadedLib {}

static FFI_REGISTRY: OnceLock<Arc<Mutex<Registry>>> = OnceLock::new();

fn registry() -> &'static Arc<Mutex<Registry>> {
    FFI_REGISTRY.get_or_init(|| {
        Arc::new(Mutex::new(Registry {
            entries: HashMap::new(),
            libs: Vec::new(),
        }))
    })
}

/// Whether `name` resolves to a registered FFI function. A cheap membership
/// check so a frontend can route a call to [`try_call`] only when it will hit,
/// without cloning args on the hot path.
pub fn is_registered(name: &str) -> bool {
    registry()
        .lock()
        .map(|g| g.entries.contains_key(name))
        .unwrap_or(false)
}

/// Lookup hook: `Some(Ok(result))` when `name` is a registered FFI function,
/// `None` otherwise. Frontends call this from their builtin-dispatch fallback.
pub fn try_call(name: &str, args: &[Value]) -> Option<Result<Value, String>> {
    let entry = {
        let guard = registry().lock().ok()?;
        guard.entries.get(name).cloned()?
    };
    Some(invoke(name, &entry, args))
}

/// Compile (if needed), `dlopen`, and register every exported function from a
/// base64-encoded `rust { ... }` block body. Idempotent per body hash. The
/// frontend's `__rust_compile` builtin calls this; on `Err` it should attach
/// its own source-line context.
pub fn compile_and_register(body_b64: &str) -> Result<(), String> {
    use base64::Engine as _;
    let body = base64::engine::general_purpose::STANDARD
        .decode(body_b64)
        .map_err(|e| format!("rust FFI: invalid base64 body: {e}"))?;
    let body =
        String::from_utf8(body).map_err(|e| format!("rust FFI: non-utf8 body: {e}"))?;

    // Hash the body (not the wrapped crate source): same body → same dylib
    // unless the wrapper template (and thus the salt) changes.
    let mut hasher = Sha256::new();
    hasher.update(WRAPPER_SALT);
    hasher.update(body.as_bytes());
    let hash = hex_short(&hasher.finalize());

    let cache_dir = ffi_cache_dir()?;
    let lib_path = cache_dir.join(format!("lib{}{}", hash, dylib_ext()));

    if !lib_path.exists() {
        let src_path = cache_dir.join(format!("{}.rs", hash));
        let wrapped = wrap_crate_source(&body);
        fs::write(&src_path, &wrapped).map_err(|e| format!("rust FFI: write source: {e}"))?;
        invoke_rustc(&src_path, &lib_path)?;
    }

    let handle = dlopen_lib(&lib_path)?;

    let decls = parse_extern_fns(&body);
    if decls.is_empty() {
        return Err("rust FFI: no `pub extern \"C\" fn ...` declarations found in block — \
                    v1 requires at least one exported function"
            .to_string());
    }

    let free_sym = dlsym_optional(handle, "fusevm_free_cstring");

    let mut reg = registry()
        .lock()
        .map_err(|_| "rust FFI: registry lock poisoned".to_string())?;
    if !reg.libs.iter().any(|l| l.path == lib_path) {
        reg.libs.push(LoadedLib {
            path: lib_path.clone(),
            handle,
            free_sym,
        });
    }
    for (name, sig) in decls {
        let sym = dlsym_lookup(handle, &name)?;
        reg.entries.insert(
            name.clone(),
            FfiEntry {
                sig,
                sym: sym as usize,
                free_sym,
            },
        );
    }
    Ok(())
}

/// `dlsym` that returns `None` for missing symbols instead of erroring.
#[cfg(unix)]
fn dlsym_optional(handle: usize, name: &str) -> Option<usize> {
    let cname = CString::new(name).ok()?;
    // SAFETY: handle came from a successful dlopen.
    let sym = unsafe { libc::dlsym(handle as *mut libc::c_void, cname.as_ptr()) };
    if sym.is_null() {
        None
    } else {
        Some(sym as usize)
    }
}

#[cfg(not(unix))]
fn dlsym_optional(_handle: usize, _name: &str) -> Option<usize> {
    None
}

fn ffi_cache_dir() -> Result<PathBuf, String> {
    // Shared cache root: `~/.cache/fusevm/ffi/` (sibling of the JIT disk cache
    // `~/.cache/fusevm-jit`). Honors `$FUSEVM_FFI_DIR` for callers that want to
    // relocate or isolate the cache (tests, sandboxes).
    let dir = if let Ok(explicit) = std::env::var("FUSEVM_FFI_DIR") {
        PathBuf::from(explicit)
    } else if let Ok(cache) = std::env::var("XDG_CACHE_HOME") {
        PathBuf::from(cache).join("fusevm").join("ffi")
    } else if let Ok(home) = std::env::var("HOME") {
        PathBuf::from(home).join(".cache").join("fusevm").join("ffi")
    } else {
        return Err("rust FFI: cannot locate cache dir (no $HOME / $XDG_CACHE_HOME)".to_string());
    };
    fs::create_dir_all(&dir)
        .map_err(|e| format!("rust FFI: create cache dir {}: {}", dir.display(), e))?;
    Ok(dir)
}

fn hex_short(bytes: &[u8]) -> String {
    // 20 hex chars (10 bytes) = 80 bits of namespace — plenty for a per-user cache.
    let mut s = String::with_capacity(20);
    for b in bytes.iter().take(10) {
        s.push_str(&format!("{:02x}", b));
    }
    s
}

fn dylib_ext() -> &'static str {
    if cfg!(target_os = "macos") {
        ".dylib"
    } else if cfg!(target_os = "windows") {
        ".dll"
    } else {
        ".so"
    }
}

/// Wrapper crate template. Keep synchronized with [`WRAPPER_SALT`] — any change
/// to the template must bump the salt so stale cached dylibs are rebuilt.
const WRAPPER_SALT: &[u8] = b"fusevm-rust-ffi-v1";

fn wrap_crate_source(body: &str) -> String {
    // Users write `pub extern "C" fn ...` directly; we auto-insert `#[no_mangle]`
    // before each so `dlsym(name)` resolves (without it rustc hash-mangles the
    // symbol). The `#![crate_type]` attribute lets us call `rustc` without
    // `--crate-type=cdylib` on the CLI.
    let body = auto_no_mangle(body);
    format!(
        "// auto-generated by fusevm rust FFI\n\
         #![crate_type = \"cdylib\"]\n\
         #![allow(unused)]\n\
         #![allow(unused_imports)]\n\
         use std::os::raw::c_char;\n\
         use std::ffi::{{CStr, CString}};\n\
         \n\
         {body}\n"
    )
}

/// Insert `#[no_mangle]` before every `pub extern "C" fn` not already carrying
/// it. Single left-to-right pass; preserves whitespace / indentation.
fn auto_no_mangle(body: &str) -> String {
    let needle = "pub extern \"C\" fn ";
    let mut out = String::with_capacity(body.len() + 32);
    let mut cursor = 0usize;
    while let Some(rel) = body[cursor..].find(needle) {
        let pos = cursor + rel;
        out.push_str(&body[cursor..pos]);
        let line_start = body[..pos].rfind('\n').map(|p| p + 1).unwrap_or(0);
        let indent = &body[line_start..pos];
        let already_marked = body[..line_start].trim_end().ends_with("#[no_mangle]");
        if !already_marked {
            out.push_str("#[no_mangle]\n");
            out.push_str(indent);
        }
        out.push_str(needle);
        cursor = pos + needle.len();
    }
    out.push_str(&body[cursor..]);
    out
}

fn invoke_rustc(src: &Path, out: &Path) -> Result<(), String> {
    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".to_string());
    let res = std::process::Command::new(&rustc)
        .arg("--edition=2021")
        .arg("-O")
        .arg("-o")
        .arg(out)
        .arg(src)
        .output();
    let out_res = match res {
        Ok(o) => o,
        Err(e) => {
            return Err(format!(
                "rust FFI: failed to invoke `{rustc}`: {e}. Install Rust to use rust {{}} blocks."
            ))
        }
    };
    if !out_res.status.success() {
        let stderr = String::from_utf8_lossy(&out_res.stderr);
        return Err(format!(
            "rust FFI: rustc failed compiling {}:\n{}",
            src.display(),
            stderr
        ));
    }
    Ok(())
}

#[cfg(unix)]
fn dlopen_lib(path: &Path) -> Result<usize, String> {
    let cpath = CString::new(path.to_string_lossy().as_bytes())
        .map_err(|e| format!("rust FFI: dlopen path nul: {e}"))?;
    // SAFETY: dlopen with RTLD_NOW|RTLD_LOCAL is the standard portable load path.
    let handle = unsafe { libc::dlopen(cpath.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL) };
    if handle.is_null() {
        // SAFETY: dlerror returns a static thread-local C string.
        let err = unsafe {
            let e = libc::dlerror();
            if e.is_null() {
                "unknown dlopen error".to_string()
            } else {
                CStr::from_ptr(e).to_string_lossy().into_owned()
            }
        };
        return Err(format!("rust FFI: dlopen {}: {}", path.display(), err));
    }
    Ok(handle as usize)
}

#[cfg(not(unix))]
fn dlopen_lib(_path: &Path) -> Result<usize, String> {
    Err("rust FFI: only unix (Linux/macOS) is supported in v1".to_string())
}

#[cfg(unix)]
fn dlsym_lookup(handle: usize, name: &str) -> Result<*const (), String> {
    let cname = CString::new(name).map_err(|e| format!("rust FFI: symbol nul: {e}"))?;
    // SAFETY: handle came from a successful dlopen; dlsym returns a fn ptr or NULL.
    let sym = unsafe { libc::dlsym(handle as *mut libc::c_void, cname.as_ptr()) };
    if sym.is_null() {
        return Err(format!(
            "rust FFI: symbol `{name}` not found in compiled cdylib"
        ));
    }
    Ok(sym as *const ())
}

#[cfg(not(unix))]
fn dlsym_lookup(_h: usize, _n: &str) -> Result<*const (), String> {
    Err("rust FFI: only unix supported in v1".to_string())
}

/// Parse a Rust body for `pub extern "C" fn NAME(ARGS) -> RET` declarations
/// matching a v1 signature. Non-matching declarations are silently ignored
/// (they stay in the cdylib but are not callable) so private helpers are fine.
fn parse_extern_fns(body: &str) -> Vec<(String, FfiSig)> {
    let mut out = Vec::new();
    let needle = "pub extern \"C\" fn ";
    let bytes = body.as_bytes();
    let mut start = 0usize;
    while let Some(rel) = body[start..].find(needle) {
        let pos = start + rel;
        let after = pos + needle.len();
        let mut j = after;
        while j < bytes.len() && (bytes[j].is_ascii_alphanumeric() || bytes[j] == b'_') {
            j += 1;
        }
        if j == after {
            start = after;
            continue;
        }
        let name = body[after..j].to_string();
        while j < bytes.len() && (bytes[j] as char).is_whitespace() {
            j += 1;
        }
        if j >= bytes.len() || bytes[j] != b'(' {
            start = after;
            continue;
        }
        let args_start = j + 1;
        let mut depth = 1i32;
        j += 1;
        while j < bytes.len() && depth > 0 {
            match bytes[j] {
                b'(' => depth += 1,
                b')' => depth -= 1,
                _ => {}
            }
            if depth == 0 {
                break;
            }
            j += 1;
        }
        if j >= bytes.len() {
            break;
        }
        let args_text = body[args_start..j].trim().to_string();
        j += 1; // past `)`
        while j < bytes.len() && (bytes[j] as char).is_whitespace() {
            j += 1;
        }
        let mut ret = String::new();
        if j + 1 < bytes.len() && bytes[j] == b'-' && bytes[j + 1] == b'>' {
            j += 2;
            while j < bytes.len() && (bytes[j] as char).is_whitespace() {
                j += 1;
            }
            let rstart = j;
            while j < bytes.len()
                && bytes[j] != b'{'
                && bytes[j] != b';'
                && !(bytes[j] == b'w' && body[j..].starts_with("where"))
            {
                j += 1;
            }
            ret = body[rstart..j].trim().to_string();
        }
        if let Some(sig) = match_signature(&args_text, &ret) {
            out.push((name, sig));
        }
        start = j;
    }
    out
}

/// Match `(args)` + `-> ret` text against the v1 table. Each arg element is
/// expected as `_name: TYPE`; the name is ignored.
fn match_signature(args_text: &str, ret: &str) -> Option<FfiSig> {
    let ret_norm: String = ret.split_whitespace().collect();
    let types: Vec<String> = if args_text.trim().is_empty() {
        Vec::new()
    } else {
        args_text
            .split(',')
            .map(|seg| {
                let seg = seg.trim();
                if let Some(colon) = seg.find(':') {
                    seg[colon + 1..].split_whitespace().collect::<String>()
                } else {
                    seg.split_whitespace().collect::<String>()
                }
            })
            .collect()
    };

    let all_i64 = !types.is_empty() && types.iter().all(|t| t == "i64");
    let all_f64 = !types.is_empty() && types.iter().all(|t| t == "f64");

    match (types.as_slice(), ret_norm.as_str()) {
        ([], "i64") => Some(FfiSig::I0),
        (_, "i64") if all_i64 && types.len() == 1 => Some(FfiSig::I1),
        (_, "i64") if all_i64 && types.len() == 2 => Some(FfiSig::I2),
        (_, "i64") if all_i64 && types.len() == 3 => Some(FfiSig::I3),
        (_, "i64") if all_i64 && types.len() == 4 => Some(FfiSig::I4),
        ([], "f64") => Some(FfiSig::F0),
        (_, "f64") if all_f64 && types.len() == 1 => Some(FfiSig::F1),
        (_, "f64") if all_f64 && types.len() == 2 => Some(FfiSig::F2),
        (_, "f64") if all_f64 && types.len() == 3 => Some(FfiSig::F3),
        _ => {
            if types.len() == 1 && is_c_str_ptr(&types[0]) {
                if ret_norm == "i64" {
                    return Some(FfiSig::StrToInt);
                }
                if is_c_str_ptr(&ret_norm) {
                    return Some(FfiSig::StrToStr);
                }
            }
            None
        }
    }
}

fn is_c_str_ptr(t: &str) -> bool {
    t == "*constc_char" || t == "*mutc_char"
}

fn invoke(name: &str, entry: &FfiEntry, args: &[Value]) -> Result<Value, String> {
    let expected = entry.sig.arity();
    if args.len() != expected {
        return Err(format!(
            "rust FFI: {name} expects {expected} args, got {}",
            args.len()
        ));
    }
    // Each arm transmutes the raw sym to the exact fn-pointer type, then calls.
    // SAFETY: `sig` came from [`parse_extern_fns`], which only yields entries
    // whose body signature matches the arm type; `sym` is a valid function
    // pointer into a dlopened cdylib alive for the process lifetime.
    unsafe {
        match entry.sig {
            FfiSig::I0 => {
                let f: extern "C" fn() -> i64 = std::mem::transmute(entry.sym);
                Ok(Value::int(f()))
            }
            FfiSig::I1 => {
                let f: extern "C" fn(i64) -> i64 = std::mem::transmute(entry.sym);
                Ok(Value::int(f(args[0].to_int())))
            }
            FfiSig::I2 => {
                let f: extern "C" fn(i64, i64) -> i64 = std::mem::transmute(entry.sym);
                Ok(Value::int(f(args[0].to_int(), args[1].to_int())))
            }
            FfiSig::I3 => {
                let f: extern "C" fn(i64, i64, i64) -> i64 = std::mem::transmute(entry.sym);
                Ok(Value::int(f(
                    args[0].to_int(),
                    args[1].to_int(),
                    args[2].to_int(),
                )))
            }
            FfiSig::I4 => {
                let f: extern "C" fn(i64, i64, i64, i64) -> i64 = std::mem::transmute(entry.sym);
                Ok(Value::int(f(
                    args[0].to_int(),
                    args[1].to_int(),
                    args[2].to_int(),
                    args[3].to_int(),
                )))
            }
            FfiSig::F0 => {
                let f: extern "C" fn() -> f64 = std::mem::transmute(entry.sym);
                Ok(Value::float(f()))
            }
            FfiSig::F1 => {
                let f: extern "C" fn(f64) -> f64 = std::mem::transmute(entry.sym);
                Ok(Value::float(f(args[0].to_float())))
            }
            FfiSig::F2 => {
                let f: extern "C" fn(f64, f64) -> f64 = std::mem::transmute(entry.sym);
                Ok(Value::float(f(args[0].to_float(), args[1].to_float())))
            }
            FfiSig::F3 => {
                let f: extern "C" fn(f64, f64, f64) -> f64 = std::mem::transmute(entry.sym);
                Ok(Value::float(f(
                    args[0].to_float(),
                    args[1].to_float(),
                    args[2].to_float(),
                )))
            }
            FfiSig::StrToInt => {
                let s = args[0].to_str();
                let c = CString::new(s).map_err(|e| format!("rust FFI: arg nul: {e}"))?;
                let f: extern "C" fn(*const c_char) -> i64 = std::mem::transmute(entry.sym);
                Ok(Value::int(f(c.as_ptr())))
            }
            FfiSig::StrToStr => {
                let s = args[0].to_str();
                let c = CString::new(s).map_err(|e| format!("rust FFI: arg nul: {e}"))?;
                let f: extern "C" fn(*const c_char) -> *const c_char =
                    std::mem::transmute(entry.sym);
                let ret = f(c.as_ptr());
                if ret.is_null() {
                    return Ok(Value::Undef);
                }
                let cs = CStr::from_ptr(ret);
                let owned = cs.to_string_lossy().into_owned();
                // Free the cdylib's allocation via `fusevm_free_cstring` when the
                // lib exports it; otherwise the returned buffer leaks.
                if let Some(free_addr) = entry.free_sym {
                    let free_fn: extern "C" fn(*mut c_char) = std::mem::transmute(free_addr);
                    free_fn(ret as *mut c_char);
                }
                Ok(Value::str(owned))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signature_match_i2() {
        assert!(matches!(
            match_signature("a: i64, b: i64", "i64"),
            Some(FfiSig::I2)
        ));
    }

    #[test]
    fn signature_match_i0() {
        assert!(matches!(match_signature("", "i64"), Some(FfiSig::I0)));
    }

    #[test]
    fn signature_match_f3() {
        assert!(matches!(
            match_signature("a: f64, b: f64, c: f64", "f64"),
            Some(FfiSig::F3)
        ));
    }

    #[test]
    fn signature_mixed_types_rejected() {
        assert!(match_signature("a: i64, b: f64", "i64").is_none());
    }

    #[test]
    fn signature_str_to_str() {
        assert!(matches!(
            match_signature("s: *const c_char", "*const c_char"),
            Some(FfiSig::StrToStr)
        ));
    }

    #[test]
    fn signature_str_to_int() {
        assert!(matches!(
            match_signature("s: *const c_char", "i64"),
            Some(FfiSig::StrToInt)
        ));
    }

    #[test]
    fn parse_extern_fns_picks_up_simple_add() {
        let body = "pub extern \"C\" fn add(a: i64, b: i64) -> i64 { a + b }";
        let decls = parse_extern_fns(body);
        assert_eq!(decls.len(), 1);
        assert_eq!(decls[0].0, "add");
        assert!(matches!(decls[0].1, FfiSig::I2));
    }

    #[test]
    fn parse_extern_fns_ignores_unsupported_signatures() {
        let body = "pub extern \"C\" fn mixed(a: i64, b: f64) -> i64 { 0 }";
        assert_eq!(parse_extern_fns(body).len(), 0);
    }

    #[test]
    fn parse_extern_fns_picks_up_multiple() {
        let body = "\
            pub extern \"C\" fn a1() -> i64 { 1 }\n\
            pub extern \"C\" fn a2(x: f64, y: f64) -> f64 { x + y }\n\
            fn private_helper() {}\n\
        ";
        let decls = parse_extern_fns(body);
        assert_eq!(decls.len(), 2);
        assert_eq!(decls[0].0, "a1");
        assert_eq!(decls[1].0, "a2");
    }

    #[test]
    fn auto_no_mangle_inserts_marker() {
        let body = "pub extern \"C\" fn f() -> i64 { 1 }";
        let out = auto_no_mangle(body);
        assert!(out.contains("#[no_mangle]"));
        // Idempotent — an already-marked fn is not double-marked.
        assert_eq!(auto_no_mangle(&out).matches("#[no_mangle]").count(), 1);
    }

    #[test]
    fn try_call_unknown_name_is_none() {
        assert!(try_call("definitely_not_registered_ffi_fn", &[]).is_none());
    }

    #[test]
    fn is_registered_false_for_unknown() {
        assert!(!is_registered("definitely_not_registered_ffi_fn"));
    }
}
