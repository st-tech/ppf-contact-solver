// File: crates/ppf-cts-core/src/parsers.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// Source-text parsers used by the frontend session layer. Today
// there's exactly one: the C++/Rust log-docstring scanner that
// produces the `name -> {Description, filename, ...}` table the
// session-output reader uses to annotate channels.
//
// Direct port of frontend/_parse_.py:CppRustDocStringParser.
//
// The walker visits every `.cu` and `.rs` file under `root` (except
// `args.rs`) and runs a small state machine on each line:
//
//   * `// Name: foo` style labels accumulate into the "current"
//     docstring.
//   * `// Description: ...` enters multi-line description mode; the
//     following `//` lines append to a single space-joined string.
//   * Empty lines reset the current docstring.
//   * `SimpleLog logging("name")` opens a parent group (subsequent
//     entries get a `parent.name.out` filename). It also flushes any
//     pending docstring.
//   * `/*== push` / `logging.push("name")` / `logging.mark("name")`
//     each emit one entry with the named filename.
//   * If the docstring contains a `Map: <other_name>` field, the
//     entry is keyed under `<other_name>` instead of the original.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LogDocEntry {
    /// `Label -> value` pairs gathered from the `//` comment block
    /// (e.g. `Name`, `Author` is filtered out, `Map` is consumed
    /// into the entry key).
    pub fields: BTreeMap<String, String>,
    /// Free-form description joined from `// Description:` and the
    /// continuation lines that follow.
    pub description: String,
    /// Output filename for this log channel: `<parent>.<name>.out`
    /// when nested under a `SimpleLog logging("parent")`, else
    /// `<name>.out`.
    pub filename: String,
}

/// Walk `root` recursively for `.cu` / `.rs` files, parse log
/// docstrings, return the `name -> entry` map sorted by key. The
/// walker is best-effort: unreadable files are skipped silently
/// (matches the Python source's behavior under the `for dirpath,
/// _, filenames in os.walk(root)` loop).
pub fn get_logging_docstrings(root: impl AsRef<Path>) -> BTreeMap<String, LogDocEntry> {
    let mut out: BTreeMap<String, LogDocEntry> = BTreeMap::new();
    walk(root.as_ref(), &mut out);
    out
}

fn walk(dir: &Path, out: &mut BTreeMap<String, LogDocEntry>) {
    let entries = match fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            walk(&path, out);
            continue;
        }
        let name = match path.file_name().and_then(|n| n.to_str()) {
            Some(n) => n,
            None => continue,
        };
        if name == "args.rs" {
            continue;
        }
        // Skip this file: it contains raw-string-literal test fixtures
        // (`SimpleLog logging("solver");`, `logging.push("pcg_iter");`,
        // ...) that the line-based parser cannot distinguish from real
        // logging sites. Letting them through registered bogus entries
        // (e.g. `pcg-iter -> solver.pcg_iter.out`) that overwrote the
        // legitimate ones harvested from `.cu` sources, leaving the
        // panel's Realtime Statistics rows showing "N/A".
        if name == "parsers.rs" {
            continue;
        }
        if !name.ends_with(".cu") && !name.ends_with(".rs") {
            continue;
        }
        if let Ok(text) = fs::read_to_string(&path) {
            parse_file(&text, out);
        }
    }
}

fn parse_file(text: &str, out: &mut BTreeMap<String, LogDocEntry>) {
    let mut state = ParseState::default();
    for line in text.lines() {
        let line = line.trim();
        if line.contains("#include") {
            continue;
        }
        state.parse_line(line, out);
    }
}

#[derive(Debug, Default)]
struct ParseState {
    /// Current `Label -> value` map from the in-progress `//` block.
    fields: BTreeMap<String, String>,
    /// Accumulated description string (space-joined `// Description:`
    /// continuation lines).
    description: String,
    /// True once we've seen `Description:`; subsequent `//` lines
    /// append to `description` until a non-`//` line ends the block.
    description_mode: bool,
    /// Most recently opened `SimpleLog logging("...")` parent name.
    /// Empty string means "no parent".
    parent_name: String,
}

impl ParseState {
    fn clear(&mut self) {
        self.fields.clear();
        self.description.clear();
        self.description_mode = false;
    }

    fn register(&mut self, mut name: String, out: &mut BTreeMap<String, LogDocEntry>) {
        if !self.fields.contains_key("Name") {
            self.clear();
            return;
        }
        let mut entry = LogDocEntry {
            fields: std::mem::take(&mut self.fields),
            description: std::mem::take(&mut self.description),
            filename: if !self.parent_name.is_empty() {
                format!("{}.{}.out", self.parent_name, name)
            } else {
                format!("{name}.out")
            },
        };
        // The `Map` field overrides the entry key (and is consumed).
        if let Some(map) = entry.fields.remove("Map") {
            name = map;
        }
        let key = name.replace('_', "-");
        out.insert(key, entry);
        self.clear();
    }

    fn parse_line(&mut self, line: &str, out: &mut BTreeMap<String, LogDocEntry>) {
        if line.is_empty() {
            self.clear();
            return;
        }

        if let Some(rest) = line.strip_prefix("//") {
            let content = rest.trim();
            if self.description_mode {
                if !self.description.is_empty() {
                    self.description.push(' ');
                }
                self.description.push_str(content);
            } else if content.starts_with("Description:") {
                self.description_mode = true;
            } else if let Some(colon) = content.find(':') {
                let label = content[..colon].trim().to_string();
                if matches!(label.as_str(), "File" | "Author" | "License" | "https") {
                    return;
                }
                let value = content[colon + 1..].trim().to_string();
                self.fields.insert(label, value);
            }
            return;
        }

        if line.starts_with("SimpleLog logging") {
            self.parent_name.clear();
            let name = extract_name(line);
            self.register(name.clone(), out);
            self.parent_name = name;
            return;
        }

        if line.starts_with("/*== push") || line.contains("logging.push(") || line.contains("logging.mark(") {
            let name = extract_name(line);
            self.register(name, out);
        }
    }
}

/// Extract the first quoted string from a line and replace internal
/// spaces with underscores.
fn extract_name(line: &str) -> String {
    let start = match line.find('"') {
        Some(i) => i + 1,
        None => return String::new(),
    };
    let rest = &line[start..];
    let end = match rest.find('"') {
        Some(i) => i,
        None => return String::new(),
    };
    rest[..end].replace(' ', "_")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_file(dir: &Path, name: &str, body: &str) {
        let path = dir.join(name);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(path, body).unwrap();
    }

    #[test]
    fn parses_simple_logging_entry() {
        // Description: is a *mode toggle*; the content goes on
        // subsequent `//` lines (matches the Python parser).
        let dir = tempfile::tempdir().unwrap();
        write_file(
            dir.path(),
            "metric.cu",
            r#"
// Name: time-per-frame
// Description:
// average wall-clock time per video frame.
logging.push("time_per_frame");
"#,
        );
        let out = get_logging_docstrings(dir.path());
        let entry = out.get("time-per-frame").expect("time-per-frame should be present");
        assert_eq!(entry.fields.get("Name").unwrap(), "time-per-frame");
        assert!(entry.description.contains("average wall-clock"));
        assert_eq!(entry.filename, "time_per_frame.out");
    }

    #[test]
    fn parent_logger_propagates_to_filename() {
        let dir = tempfile::tempdir().unwrap();
        write_file(
            dir.path(),
            "module.rs",
            r#"
SimpleLog logging("solver");

// Name: pcg_iter
// Description: PCG iteration count.
logging.push("pcg_iter");
"#,
        );
        let out = get_logging_docstrings(dir.path());
        let entry = out.get("pcg-iter").expect("pcg-iter present");
        assert_eq!(entry.filename, "solver.pcg_iter.out");
    }

    #[test]
    fn map_field_overrides_entry_key() {
        let dir = tempfile::tempdir().unwrap();
        write_file(
            dir.path(),
            "alias.rs",
            r#"
// Name: original_name
// Map: override_key
logging.mark("original_name");
"#,
        );
        let out = get_logging_docstrings(dir.path());
        // Key should be the Map value (with underscores hyphenated),
        // not the Name.
        assert!(out.contains_key("override-key"));
        assert!(!out.contains_key("original-name"));
        let entry = &out["override-key"];
        assert!(!entry.fields.contains_key("Map"), "Map field should be consumed");
    }

    #[test]
    fn skips_args_rs() {
        let dir = tempfile::tempdir().unwrap();
        write_file(
            dir.path(),
            "args.rs",
            r#"
// Name: ignored
logging.push("ignored");
"#,
        );
        write_file(
            dir.path(),
            "main.rs",
            r#"
// Name: kept
logging.push("kept");
"#,
        );
        let out = get_logging_docstrings(dir.path());
        assert!(out.contains_key("kept"));
        assert!(!out.contains_key("ignored"));
    }

    #[test]
    fn skips_meta_labels() {
        let dir = tempfile::tempdir().unwrap();
        write_file(
            dir.path(),
            "header.rs",
            r#"
// File: header.rs
// Author: someone
// License: Apache v2.0
// https://example.com
// Name: real
logging.push("real");
"#,
        );
        let out = get_logging_docstrings(dir.path());
        let entry = &out["real"];
        // Skip-labels never enter the field map.
        assert!(!entry.fields.contains_key("File"));
        assert!(!entry.fields.contains_key("Author"));
        assert!(!entry.fields.contains_key("License"));
        assert!(!entry.fields.contains_key("https"));
        assert_eq!(entry.fields.get("Name").unwrap(), "real");
    }

    #[test]
    fn description_continuation_concatenates_with_space() {
        let dir = tempfile::tempdir().unwrap();
        write_file(
            dir.path(),
            "x.rs",
            r#"
// Name: foo
// Description: line one
// continued on line two
// continued on line three
logging.push("foo");
"#,
        );
        let out = get_logging_docstrings(dir.path());
        let entry = &out["foo"];
        // Space-joined; the leading `Description:` token does NOT
        // appear because we enter description mode after seeing it.
        assert!(entry.description.contains("continued on line two"));
        assert!(entry.description.contains("continued on line three"));
        // Single-space joining.
        assert!(!entry.description.contains("  "));
    }

    #[test]
    fn empty_line_resets_pending_block() {
        let dir = tempfile::tempdir().unwrap();
        write_file(
            dir.path(),
            "x.rs",
            r#"
// Name: orphan_one

// Name: orphan_two
logging.push("orphan_two");
"#,
        );
        let out = get_logging_docstrings(dir.path());
        // `orphan_one` had no logging.push; cleared by the blank
        // line. Only `orphan_two` survives.
        assert!(!out.contains_key("orphan-one"));
        assert!(out.contains_key("orphan-two"));
    }

    #[test]
    fn ignores_include_lines() {
        let dir = tempfile::tempdir().unwrap();
        write_file(
            dir.path(),
            "x.cu",
            r#"
#include "logging.push(\"sneaky\");"
// Name: real
logging.push("real");
"#,
        );
        let out = get_logging_docstrings(dir.path());
        assert!(out.contains_key("real"));
        assert!(!out.contains_key("sneaky"));
    }

    #[test]
    fn empty_root_returns_empty_map() {
        let dir = tempfile::tempdir().unwrap();
        let out = get_logging_docstrings(dir.path());
        assert!(out.is_empty());
    }

    #[test]
    fn missing_root_returns_empty_map() {
        let out = get_logging_docstrings("/tmp/no-such-dir-{xyz}");
        assert!(out.is_empty());
    }

    #[test]
    fn extract_name_handles_quotes() {
        assert_eq!(extract_name(r#"logging.push("abc def");"#), "abc_def");
        assert_eq!(extract_name("no quotes"), "");
        assert_eq!(extract_name(r#"only_open_quote"#), "");
    }

    #[test]
    fn nested_directories_recurse() {
        let dir = tempfile::tempdir().unwrap();
        let sub = dir.path().join("subsystem");
        std::fs::create_dir(&sub).unwrap();
        write_file(
            &sub,
            "deep.rs",
            r#"
// Name: deep
logging.push("deep");
"#,
        );
        let out = get_logging_docstrings(dir.path());
        assert!(out.contains_key("deep"));
    }
}
