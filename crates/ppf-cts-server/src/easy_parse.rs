// File: crates/ppf-cts-server/src/easy_parse.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// TCMD command-line splitter. The Blender addon emits lines like
//
//     --request build --name demo
//
// and the only callers (`wire::handle_tcmd`) look up `name` and
// `request`. Repeated keys are last-write-wins; `--name` is
// single-valued by construction and the seven `--request` values are
// enumerated in `wire/mod.rs`. Bare `--flag` tokens with no
// following value are skipped: every flag the addon emits today
// carries a value.

use std::collections::HashMap;

/// Parsed TCMD args: last-write-wins on repeated keys.
pub type Args = HashMap<String, String>;

/// Split a whitespace-delimited TCMD line into `--key value` pairs.
/// A bare `--key` with no following value is silently dropped (no
/// production caller checks for valueless flags). Tokens before the
/// first `--key` are ignored.
pub fn parse(line: &str) -> Args {
    let mut args: Args = HashMap::new();
    let mut current_key: Option<String> = None;

    for token in line.split_whitespace() {
        if let Some(stripped) = token.strip_prefix("--") {
            current_key = Some(stripped.to_string());
        } else if let Some(key) = current_key.take() {
            args.insert(key, token.to_string());
        }
    }
    args
}

/// Convenience accessor matching the previous API.
pub fn get<'a>(args: &'a Args, key: &str) -> Option<&'a str> {
    args.get(key).map(String::as_str)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_request_and_name() {
        let a = parse("--request build --name demo");
        assert_eq!(get(&a, "request"), Some("build"));
        assert_eq!(get(&a, "name"), Some("demo"));
    }

    #[test]
    fn ping_with_no_request() {
        let a = parse("--name demo");
        assert!(get(&a, "request").is_none());
        assert_eq!(get(&a, "name"), Some("demo"));
    }

    #[test]
    fn flag_without_value_is_dropped() {
        // Production callers never look for valueless flags, and the
        // simplified parser drops them rather than carrying a None
        // sentinel through the map.
        let a = parse("--verbose");
        assert!(a.get("verbose").is_none());
    }

    #[test]
    fn repeated_key_keeps_last_value() {
        let a = parse("--tag x --tag y --tag z");
        assert_eq!(get(&a, "tag"), Some("z"));
    }

    #[test]
    fn empty_line() {
        let a = parse("");
        assert!(a.is_empty());
    }

    #[test]
    fn tokens_before_first_key_are_ignored() {
        let a = parse("stray tokens --name demo");
        assert_eq!(get(&a, "name"), Some("demo"));
    }
}
