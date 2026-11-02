// File: crates/ppf-cts-compute/src/build/kernelgen.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
//! Running the transcompiler over many sources: in parallel, and only where an
//! input changed.
//!
//! One rendering costs about a second of Python, and a build asks for a few
//! hundred of them, so a serial run that re-renders everything is most of the
//! wall clock of any build that touched the kernel tree. This module runs every
//! rendering a caller asks for, on all cores, and reuses a rendering from an
//! earlier build when nothing it was rendered from has changed.
//!
//! # What "nothing it was rendered from" means, and who decides it
//!
//! The GENERATOR decides, by being watched rather than predicted. Each run goes
//! through `seam/kernelgen_deps.py`, which records every file the generator
//! opened with its SHA-256, every existence check with its answer, and every
//! whole-tree scan by the value it returned. A cached rendering is reused only
//! while every recorded fact still holds, and a scan is re-checked by calling
//! the generator's own function again. Nothing here spells which files a
//! rendering includes, so there is no second copy of that logic to go stale,
//! and a stale rendering is the worst outcome this module could have: a build
//! that compiles clean from the previous physics.
//!
//! What the record cannot account for makes a rendering uncacheable, and an
//! uncacheable rendering is produced by the generator on every build. Slower,
//! never stale.
//!
//! # What a caller gets
//!
//! One outcome per job, IN JOB ORDER, whatever order the work finished in. A
//! caller that concatenates renderings, or reports the first failure, therefore
//! behaves exactly as a serial loop over the same list would.
//!
//! An output file is written only when its text changed, which is also what the
//! generator does when it writes, so a rendering that came out the same keeps its
//! mtime and an mtime-driven recompile downstream skips it.

use std::collections::{HashMap, HashSet};
use std::ffi::OsString;
use std::io::{self, Write as _};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

/// One transcompiler invocation.
///
/// It runs as `kernelgen.py <args...> --out <out> <source>`, so `args` holds
/// the target and emit flags and nothing else.
pub struct Job {
    pub args: Vec<OsString>,
    pub out: PathBuf,
    pub source: PathBuf,
}

/// What one invocation produced. A reused rendering has empty output streams,
/// because the generator prints nothing on success.
pub struct Rendered {
    pub success: bool,
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
}

/// The transcompiler, resolved against this crate's own tree.
///
/// A caller does not name it: rendering a kernel for a target is this crate's
/// work, and `CARGO_MANIFEST_DIR` is expanded when this crate is compiled, so it
/// points at this source tree however the crate is consumed.
#[must_use]
pub fn transcompiler() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("seam/kernelgen.py")
}

fn recorder() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("seam/kernelgen_deps.py")
}

/// Renders every job and returns one outcome per job, in job order.
///
/// `cache_dir` is where reusable renderings are kept. It must be inside the
/// caller's `OUT_DIR`: the cache belongs to one build configuration, and a
/// `cargo clean` must take it with everything else.
///
/// An `Err` means python3 could not be started at all; whether that is fatal is
/// the caller's decision, because one caller treats its renderings as optional.
#[must_use]
pub fn render(jobs: &[Job], cache_dir: &Path) -> Vec<io::Result<Rendered>> {
    let generator = transcompiler();
    let recorder = recorder();
    println!("cargo:rerun-if-changed={}", generator.display());
    println!("cargo:rerun-if-changed={}", recorder.display());

    // THE KEY NAMES THE INVOCATION, AND THE MANIFEST NAMES ITS INPUTS. The
    // generator and the recorder are hashed whole, because a changed generator
    // changes every rendering; the interpreter's version is in it too, because
    // what a regular expression matches is the interpreter's business. The
    // source's own content is in the key so that an edit and its revert keep
    // both renderings rather than replacing one with the other.
    let version = match Command::new("python3")
        .arg("-B")
        .arg("-c")
        .arg("import sys; print(sys.version)")
        .output()
    {
        Ok(out) if out.status.success() => out.stdout,
        Ok(_) => b"unknown".to_vec(),
        Err(e) => return jobs.iter().map(|_| Err(io::Error::new(e.kind(), e.to_string()))).collect(),
    };
    let mut base = Sha256::new();
    for part in [file_bytes(&generator), file_bytes(&recorder), Some(version)] {
        base.field(part.as_deref().unwrap_or(b"<missing>"));
    }
    let base = base.finish();

    let keys: Vec<String> = jobs
        .iter()
        .map(|job| {
            let mut key = Sha256::new();
            key.field(base.as_bytes());
            for arg in invocation(job) {
                key.field(arg.to_string_lossy().as_bytes());
            }
            key.field(file_bytes(&job.source).as_deref().unwrap_or(b"<missing>"));
            key.finish()
        })
        .collect();

    let hits = find_hits(&keys, cache_dir, &recorder);

    // THE MISSES RUN ON EVERY CORE, each into its own output file. The pool is
    // a counter over the job list rather than a split of it, because one
    // rendering can take ten times another and a static split would leave cores
    // idle behind the slowest share.
    let misses: Vec<usize> = (0..jobs.len()).filter(|i| hits[*i].is_none()).collect();
    let slots: Vec<Mutex<Option<io::Result<Rendered>>>> =
        (0..jobs.len()).map(|_| Mutex::new(None)).collect();
    let next = AtomicUsize::new(0);
    let threads = std::thread::available_parallelism()
        .map_or(1, std::num::NonZeroUsize::get)
        .min(misses.len().max(1));
    std::thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(|| loop {
                let at = next.fetch_add(1, Ordering::Relaxed);
                let Some(&i) = misses.get(at) else { break };
                let outcome = run_and_store(&jobs[i], &keys[i], cache_dir, &recorder);
                *slots[i].lock().unwrap() = Some(outcome);
            });
        }
    });

    slots
        .into_iter()
        .zip(hits)
        .zip(jobs)
        .map(|((slot, hit), job)| match hit {
            Some(text) => {
                write_if_changed(&job.out, &text)?;
                Ok(Rendered { success: true, stdout: Vec::new(), stderr: Vec::new() })
            }
            None => slot.into_inner().unwrap().expect("every miss was run"),
        })
        .collect()
}

/// The generator's own argument list for a job, after the script path.
fn invocation(job: &Job) -> Vec<OsString> {
    let mut argv = job.args.clone();
    argv.push("--out".into());
    argv.push(job.out.clone().into_os_string());
    argv.push(job.source.clone().into_os_string());
    argv
}

/// Runs one job through the recorder and keeps what it produced when the
/// record is complete.
fn run_and_store(job: &Job, key: &str, cache_dir: &Path, recorder: &Path) -> io::Result<Rendered> {
    let entry = cache_dir.join(key);
    let manifest = entry.join("manifest");
    let pending = entry.join("manifest.new");
    std::fs::create_dir_all(&entry)?;
    // The manifest is what makes an entry valid, so it goes first: a build
    // interrupted between here and the rename below leaves no entry rather than
    // an old manifest beside a new output.
    remove_if_present(&manifest)?;
    remove_if_present(&pending)?;
    let out = Command::new("python3")
        .arg("-B")
        .arg(recorder)
        .arg("run")
        .arg(&pending)
        .args(invocation(job))
        .output()?;
    let success = out.status.success();
    if success && pending.is_file() {
        let text = std::fs::read(&job.out)?;
        let staged = entry.join("output.new");
        std::fs::write(&staged, &text)?;
        std::fs::rename(&staged, entry.join("output"))?;
        std::fs::rename(&pending, &manifest)?;
    }
    Ok(Rendered { success, stdout: out.stdout, stderr: out.stderr })
}

/// For each key, the cached rendering if every fact its manifest records still
/// holds.
fn find_hits(keys: &[String], cache_dir: &Path, recorder: &Path) -> Vec<Option<Vec<u8>>> {
    let mut digests: HashMap<PathBuf, Option<String>> = HashMap::new();
    let mut candidates: Vec<Option<(Vec<u8>, Vec<String>)>> = Vec::with_capacity(keys.len());
    for key in keys {
        let entry = cache_dir.join(key);
        let (Ok(manifest), Ok(output)) = (
            std::fs::read_to_string(entry.join("manifest")),
            std::fs::read(entry.join("output")),
        ) else {
            candidates.push(None);
            continue;
        };
        let mut calls = Vec::new();
        let mut valid = true;
        for line in manifest.lines() {
            let mut parts = line.splitn(3, ' ');
            let (Some(kind), Some(fact), Some(rest)) = (parts.next(), parts.next(), parts.next())
            else {
                valid = false;
                break;
            };
            valid = match kind {
                "read" => {
                    let path = PathBuf::from(rest);
                    let digest = digests
                        .entry(path)
                        .or_insert_with_key(|p| file_bytes(p).map(|b| sha256_hex(&b)));
                    digest.as_deref() == Some(fact)
                }
                "probe" => classify(Path::new(rest)) == fact,
                "call" => {
                    calls.push(line.to_string());
                    true
                }
                _ => false,
            };
            if !valid {
                break;
            }
        }
        candidates.push(valid.then_some((output, calls)));
    }

    // THE WHOLE-TREE SCANS ARE RE-RUN BY THE GENERATOR'S OWN FUNCTIONS, once
    // per distinct call across every candidate, in one process.
    let mut specs: Vec<&str> = Vec::new();
    let mut seen = HashSet::new();
    for (_, calls) in candidates.iter().flatten() {
        for line in calls {
            let spec = line.splitn(3, ' ').nth(2).unwrap_or("");
            if seen.insert(spec) {
                specs.push(spec);
            }
        }
    }
    let answers = validate_calls(&specs, recorder);
    candidates
        .into_iter()
        .map(|candidate| {
            let (output, calls) = candidate?;
            let answers = answers.as_ref()?;
            calls
                .iter()
                .all(|line| {
                    let mut parts = line.splitn(3, ' ');
                    let digest = parts.nth(1);
                    let spec = parts.next().unwrap_or("");
                    digest.is_some() && answers.get(spec).map(String::as_str) == digest
                })
                .then_some(output)
        })
        .collect()
}

/// `spec -> digest` for every call spec, or `None` if the check could not run,
/// which turns every candidate into a miss.
fn validate_calls(specs: &[&str], recorder: &Path) -> Option<HashMap<String, String>> {
    if specs.is_empty() {
        return Some(HashMap::new());
    }
    let mut child = Command::new("python3")
        .arg("-B")
        .arg(recorder)
        .arg("validate")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .ok()?;
    let input: String = specs.iter().map(|s| format!("{s}\n")).collect();
    let mut stdin = child.stdin.take()?;
    // Written from a thread so a large answer cannot deadlock against a large
    // question on the two pipes.
    let writer = std::thread::spawn(move || stdin.write_all(input.as_bytes()));
    let out = child.wait_with_output().ok()?;
    writer.join().ok()?.ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8(out.stdout).ok()?;
    let lines: Vec<&str> = text.lines().collect();
    if lines.len() != specs.len() {
        return None;
    }
    Some(
        specs
            .iter()
            .zip(lines)
            .filter(|(_, digest)| *digest != "error")
            .map(|(spec, digest)| ((*spec).to_string(), digest.to_string()))
            .collect(),
    )
}

/// The recorder's classification of a path: a regular file, a directory,
/// something else, or nothing. Follows symlinks, as `os.stat` does.
fn classify(path: &Path) -> &'static str {
    match std::fs::metadata(path) {
        Ok(meta) if meta.is_file() => "f",
        Ok(meta) if meta.is_dir() => "d",
        Ok(_) => "o",
        Err(_) => "n",
    }
}

fn file_bytes(path: &Path) -> Option<Vec<u8>> {
    std::fs::read(path).ok()
}

fn remove_if_present(path: &Path) -> io::Result<()> {
    match std::fs::remove_file(path) {
        Err(e) if e.kind() != io::ErrorKind::NotFound => Err(e),
        _ => Ok(()),
    }
}

/// Writes `text` to `path` unless the file already holds exactly that.
fn write_if_changed(path: &Path, text: &[u8]) -> io::Result<()> {
    if std::fs::read(path).is_ok_and(|current| current == text) {
        return Ok(());
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, text)
}

fn sha256_hex(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    hasher.finish()
}

/// SHA-256 (FIPS 180-4), written out because a build script here takes no
/// crate it does not already have: the boxes build offline, and a new
/// dependency is a failed build there. It must agree with Python's
/// `hashlib.sha256`, since the recorder writes the digests this compares.
struct Sha256 {
    state: [u32; 8],
    block: Vec<u8>,
    length: u64,
}

const K: [u32; 64] = [
    0x428a_2f98, 0x7137_4491, 0xb5c0_fbcf, 0xe9b5_dba5, 0x3956_c25b, 0x59f1_11f1, 0x923f_82a4,
    0xab1c_5ed5, 0xd807_aa98, 0x1283_5b01, 0x2431_85be, 0x550c_7dc3, 0x72be_5d74, 0x80de_b1fe,
    0x9bdc_06a7, 0xc19b_f174, 0xe49b_69c1, 0xefbe_4786, 0x0fc1_9dc6, 0x240c_a1cc, 0x2de9_2c6f,
    0x4a74_84aa, 0x5cb0_a9dc, 0x76f9_88da, 0x983e_5152, 0xa831_c66d, 0xb003_27c8, 0xbf59_7fc7,
    0xc6e0_0bf3, 0xd5a7_9147, 0x06ca_6351, 0x1429_2967, 0x27b7_0a85, 0x2e1b_2138, 0x4d2c_6dfc,
    0x5338_0d13, 0x650a_7354, 0x766a_0abb, 0x81c2_c92e, 0x9272_2c85, 0xa2bf_e8a1, 0xa81a_664b,
    0xc24b_8b70, 0xc76c_51a3, 0xd192_e819, 0xd699_0624, 0xf40e_3585, 0x106a_a070, 0x19a4_c116,
    0x1e37_6c08, 0x2748_774c, 0x34b0_bcb5, 0x391c_0cb3, 0x4ed8_aa4a, 0x5b9c_ca4f, 0x682e_6ff3,
    0x748f_82ee, 0x78a5_636f, 0x84c8_7814, 0x8cc7_0208, 0x90be_fffa, 0xa450_6ceb, 0xbef9_a3f7,
    0xc671_78f2,
];

impl Sha256 {
    fn new() -> Self {
        Self {
            state: [
                0x6a09_e667, 0xbb67_ae85, 0x3c6e_f372, 0xa54f_f53a, 0x510e_527f, 0x9b05_688c,
                0x1f83_d9ab, 0x5be0_cd19,
            ],
            block: Vec::with_capacity(64),
            length: 0,
        }
    }

    /// A length-prefixed field, so that two field lists hash differently
    /// whenever they differ, wherever the boundaries fall.
    fn field(&mut self, data: &[u8]) {
        self.update(&(data.len() as u64).to_le_bytes());
        self.update(data);
    }

    fn update(&mut self, mut data: &[u8]) {
        self.length = self.length.wrapping_add(data.len() as u64);
        if !self.block.is_empty() {
            let take = (64 - self.block.len()).min(data.len());
            self.block.extend_from_slice(&data[..take]);
            data = &data[take..];
            if self.block.len() == 64 {
                let block: [u8; 64] = self.block[..].try_into().unwrap();
                self.compress(&block);
                self.block.clear();
            }
        }
        while data.len() >= 64 {
            self.compress(data[..64].try_into().unwrap());
            data = &data[64..];
        }
        self.block.extend_from_slice(data);
    }

    fn compress(&mut self, block: &[u8; 64]) {
        let mut w = [0u32; 64];
        for (i, word) in block.chunks_exact(4).enumerate() {
            w[i] = u32::from_be_bytes(word.try_into().unwrap());
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16].wrapping_add(s0).wrapping_add(w[i - 7]).wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = self.state;
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ (!e & g);
            let t1 = h.wrapping_add(s1).wrapping_add(ch).wrapping_add(K[i]).wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        for (slot, value) in self.state.iter_mut().zip([a, b, c, d, e, f, g, h]) {
            *slot = slot.wrapping_add(value);
        }
    }

    fn finish(mut self) -> String {
        let bits = self.length.wrapping_mul(8);
        self.update(&[0x80]);
        while self.block.len() != 56 {
            self.update(&[0]);
        }
        self.update(&bits.to_be_bytes());
        self.state.iter().map(|word| format!("{word:08x}")).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::sha256_hex;

    #[test]
    fn sha256_matches_the_standard_vectors() {
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        let million = vec![b'a'; 1_000_000];
        assert_eq!(
            sha256_hex(&million),
            "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"
        );
    }
}
