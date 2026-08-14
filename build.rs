use std::error::Error;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() -> Result<(), Box<dyn Error>> {
    println!("cargo:rerun-if-changed=Cargo.toml");
    println!("cargo:rerun-if-changed=Cargo.lock");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src");
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    emit_git_rerun_paths(root);
    let manifest = std::fs::read_to_string("Cargo.toml")?;
    let dependency = manifest
        .lines()
        .find(|line| line.trim_start().starts_with("a3s-power ="))
        .ok_or("Cargo.toml does not declare a3s-power")?;
    let revision = dependency
        .split("rev = \"")
        .nth(1)
        .and_then(|suffix| suffix.split('"').next())
        .ok_or("a3s-power must use an exact git revision")?;
    if !(7..=40).contains(&revision.len()) || !revision.bytes().all(|byte| byte.is_ascii_hexdigit())
    {
        return Err("a3s-power revision must be a 7-to-40-character git hash".into());
    }
    println!("cargo:rustc-env=A3S_POWER_REVISION={revision}");
    println!("cargo:rustc-env=A3S_MOE_REVISION={}", git_revision(root));
    Ok(())
}

fn emit_git_rerun_paths(root: &Path) {
    for arguments in [
        ["rev-parse", "--git-path", "HEAD"],
        ["rev-parse", "--git-path", "index"],
    ] {
        if let Some(path) = git_output(root, &arguments) {
            emit_rerun_path(root, &path);
        }
    }
    if let Some(reference) = git_output(root, &["symbolic-ref", "-q", "HEAD"]) {
        if let Some(path) = git_output(root, &["rev-parse", "--git-path", &reference]) {
            emit_rerun_path(root, &path);
        }
    }
}

fn emit_rerun_path(root: &Path, path: &str) {
    let path = PathBuf::from(path);
    let path = if path.is_absolute() {
        path
    } else {
        root.join(path)
    };
    println!("cargo:rerun-if-changed={}", path.display());
}

fn git_revision(root: &Path) -> String {
    let revision = git_output(root, &["rev-parse", "HEAD"])
        .filter(|value| value.len() == 40 && value.bytes().all(|byte| byte.is_ascii_hexdigit()));
    let Some(revision) = revision else {
        return "unknown".to_string();
    };
    match git_dirty(root) {
        Some(true) => format!("{revision}-dirty"),
        Some(false) => revision,
        None => "unknown".to_string(),
    }
}

fn git_dirty(root: &Path) -> Option<bool> {
    let output = Command::new("git")
        .current_dir(root)
        .args([
            "status",
            "--porcelain",
            "--untracked-files=normal",
            "--",
            "Cargo.toml",
            "Cargo.lock",
            "build.rs",
            "src",
        ])
        .output()
        .ok()?;
    output.status.success().then_some(!output.stdout.is_empty())
}

fn git_output(root: &Path, arguments: &[&str]) -> Option<String> {
    let output = Command::new("git")
        .current_dir(root)
        .args(arguments)
        .output()
        .ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_string())
}
