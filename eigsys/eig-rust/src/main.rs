// File: main.rs
// Code: Claude Code and Codex
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

mod eigsolve2x2;
mod eigsolve3x3;
mod eigsys_2;
mod eigsys_3;

extern crate nalgebra as na;
use clap::Parser;

#[derive(Parser, Debug)]
struct Args {
    #[arg(short, long)]
    dimension: i32,
}

fn main() {
    let args = Args::parse();
    if args.dimension == 2 {
        eigsys_2::run();
    } else {
        eigsys_3::run();
    }
}
