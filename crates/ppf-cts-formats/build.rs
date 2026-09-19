// File: crates/ppf-cts-formats/build.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

//! Digests the sources every producer and consumer of this crate's formats must
//! agree on into `PPF_SOURCE_STAMP`, which `SOURCE_STAMP` publishes.
//!
//! WHY A STAMP AND NOT A VERSION. A distribution carries one directory per
//! backend, and a run can take its solver from a different directory than the
//! one the Python extension was loaded from. Two builds of different source
//! then meet in one run: the extension writes a session the solver reads, and
//! a parameter one side knows and the other does not surfaces far from its
//! cause. `SCHEMA_VERSION` changes only when someone bumps it; this changes
//! whenever the sources that define those files change, committed or not.
//!
//! WHAT IS DIGESTED: this crate's `src` (the envelopes and status records) and
//! `ppf-cts-core/src` (the parameter registry and the session files the solver
//! reads). Each file contributes its path relative to `crates/` and its
//! contents with carriage returns dropped, in sorted path order, so the stamp
//! is the same on a checkout that converted line endings and on one that did
//! not. FNV-1a over those bytes, spelled out here, so the value does not
//! depend on which toolchain built the binary.

use std::fs;
use std::path::{Path, PathBuf};

fn collect_files(dir: &Path, files: &mut Vec<PathBuf>) {
    let entries = fs::read_dir(dir)
        .unwrap_or_else(|error| panic!("cannot list {}: {error}", dir.display()));
    for entry in entries {
        let path = entry
            .unwrap_or_else(|error| panic!("cannot read an entry of {}: {error}", dir.display()))
            .path();
        if path.is_dir() {
            collect_files(&path, files);
        } else {
            files.push(path);
        }
    }
}

fn main() {
    let manifest = PathBuf::from(
        std::env::var("CARGO_MANIFEST_DIR").expect("cargo sets CARGO_MANIFEST_DIR"),
    );
    let crates = manifest
        .parent()
        .expect("this crate lives under crates/")
        .to_path_buf();
    let roots = [manifest.join("src"), crates.join("ppf-cts-core").join("src")];

    let mut files = Vec::new();
    for root in &roots {
        // A directory here makes cargo rerun this script when anything under it
        // changes, which is the whole contract of the stamp.
        println!("cargo:rerun-if-changed={}", root.display());
        collect_files(root, &mut files);
    }
    let mut named: Vec<(String, PathBuf)> = files
        .into_iter()
        .map(|path| {
            let relative = path
                .strip_prefix(&crates)
                .expect("every digested file is under crates/")
                .components()
                .map(|part| part.as_os_str().to_string_lossy().into_owned())
                .collect::<Vec<_>>()
                .join("/");
            (relative, path)
        })
        .collect();
    named.sort();

    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut hash = FNV_OFFSET;
    let mut feed = |bytes: &[u8]| {
        for &byte in bytes {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(FNV_PRIME);
        }
    };
    for (relative, path) in &named {
        let contents = fs::read(path)
            .unwrap_or_else(|error| panic!("cannot read {}: {error}", path.display()));
        feed(relative.as_bytes());
        feed(&[0]);
        let without_cr: Vec<u8> = contents.into_iter().filter(|&byte| byte != b'\r').collect();
        feed(&without_cr);
        feed(&[0]);
    }
    println!("cargo:rustc-env=PPF_SOURCE_STAMP={hash:016x}");
}
