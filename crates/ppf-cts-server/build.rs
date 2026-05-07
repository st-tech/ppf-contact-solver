// File: crates/ppf-cts-server/build.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// Bake the docstring-harvested log channel map into a static array so
// the response builder has a complete fallback when the runtime
// harvester in `main.rs` can't see the source tree. Both code paths
// call the same `ppf_cts_core::parsers::get_logging_docstrings`
// function, so the runtime live harvest and this build-time snapshot
// can never drift in shape — only in freshness, which the runtime
// harvest wins by virtue of running later.

use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::PathBuf;

fn main() {
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    // crates/ppf-cts-server -> ../.. = repo root.
    let workspace = manifest
        .parent()
        .and_then(|p| p.parent())
        .map(|p| p.to_path_buf())
        .expect("workspace root above CARGO_MANIFEST_DIR");

    // Mirror the runtime harvest in main.rs: walk every directory
    // that holds `// Name:` / `// Map:` annotated logging sites.
    let roots = [
        workspace.join("src"),
        workspace.join("crates").join("ppf-cts-solver").join("src"),
        workspace.join("crates").join("ppf-cts-core").join("src"),
    ];

    let mut merged: BTreeMap<String, String> = BTreeMap::new();
    for root in &roots {
        if !root.is_dir() {
            continue;
        }
        let docs = ppf_cts_core::parsers::get_logging_docstrings(root);
        for (name, entry) in docs {
            merged.insert(name, entry.filename);
        }
    }

    // Emit a Rust array literal that the consumer `include!`s. Wrap
    // the body with `&[ ... ]` on the consumer side; we only emit the
    // array contents so the file remains an expression whose top
    // level is a slice.
    let mut out = String::from("[\n");
    for (name, filename) in &merged {
        out.push_str("    (");
        out.push_str(&escape(name));
        out.push_str(", ");
        out.push_str(&escape(filename));
        out.push_str("),\n");
    }
    out.push(']');

    let out_dir = env::var("OUT_DIR").expect("OUT_DIR");
    let path = PathBuf::from(out_dir).join("log_channels_baked.rs");
    fs::write(&path, out).expect("write log_channels_baked.rs");

    // Re-run the build script when any logging-bearing source changes.
    // The walker only looks at `.cu` and `.rs` files, so directory
    // mtime tracking is sufficient (cargo treats `rerun-if-changed`
    // pointed at a directory as "rerun if any descendant changes").
    for root in &roots {
        if root.is_dir() {
            println!("cargo:rerun-if-changed={}", root.display());
        }
    }
    println!("cargo:rerun-if-changed=build.rs");
}

fn escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\x{:02x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}
