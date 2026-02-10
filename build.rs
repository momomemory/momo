use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=frontend/src");
    println!("cargo:rerun-if-changed=frontend/index.html");
    println!("cargo:rerun-if-changed=frontend/package.json");
    println!("cargo:rerun-if-changed=frontend/bun.lock");
    println!("cargo:rerun-if-changed=frontend/tsconfig.json");
    println!("cargo:rerun-if-changed=frontend/vite.config.ts");

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let frontend_dir = manifest_dir.join("frontend");
    let dist_dir = frontend_dir.join("dist");

    if has_built_frontend(&dist_dir) {
        return;
    }

    ensure_bun_available();
    run_bun(&frontend_dir, &["install"]);
    run_bun(&frontend_dir, &["run", "build"]);

    if !has_built_frontend(&dist_dir) {
        panic!(
            "Frontend build did not produce dist assets at {}",
            dist_dir.display()
        );
    }
}

fn has_built_frontend(dist_dir: &Path) -> bool {
    dist_dir.join("index.html").exists()
}

fn ensure_bun_available() {
    let status = Command::new("bun")
        .arg("--version")
        .status()
        .expect("Failed to execute `bun --version`");
    assert!(
        status.success(),
        "bun is required to build the embedded frontend assets"
    );
}

fn run_bun(frontend_dir: &Path, args: &[&str]) {
    let status = Command::new("bun")
        .args(args)
        .current_dir(frontend_dir)
        .status()
        .unwrap_or_else(|error| panic!("Failed to execute bun {args:?}: {error}"));

    assert!(
        status.success(),
        "bun {:?} failed in {}",
        args,
        frontend_dir.display()
    );
}
