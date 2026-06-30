// File: build.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// The `_ppf_cts_py` cdylib is built by plain `cargo build` (this crate is a
// workspace default-member) and loaded directly from `target/<profile>/` by
// `frontend/__init__.py`; there is no maturin wheel step anymore.
//
// On macOS the extension must NOT link libpython: the symbols are resolved
// at load time by whichever interpreter dlopen's the module. Without the
// `-undefined dynamic_lookup` linker flag the cdylib link fails with
// `Undefined symbol: __Py_TrueStruct` (and friends). maturin used to inject
// this flag; now we add it ourselves. `rustc-cdylib-link-arg` scopes it to
// the cdylib only, so the crate's rlib and (test = false) test harness are
// unaffected. Linux resolves undefined symbols in a shared object lazily and
// Windows links the Python import library via pyo3's own build script, so
// neither needs anything here.

fn main() {
    #[cfg(target_os = "macos")]
    {
        println!("cargo:rustc-cdylib-link-arg=-undefined");
        println!("cargo:rustc-cdylib-link-arg=dynamic_lookup");
    }
}
