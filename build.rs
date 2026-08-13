use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    println!("cargo:rerun-if-changed=Cargo.toml");
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
    Ok(())
}
