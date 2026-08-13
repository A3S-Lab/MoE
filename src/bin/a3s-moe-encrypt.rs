use std::path::PathBuf;
use std::process::ExitCode;

use a3s_moe::olmoe::OlmoePackedCheckpoint;
use a3s_moe::{MoeError, Result};
use a3s_power::inference::{
    DevicePreference, EmbeddedRuntime, InferenceLimits, SeekableWeightKey,
    DEFAULT_ENCRYPTED_CHUNK_BYTES,
};

const MIB: u32 = 1024 * 1024;

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("a3s-moe-encrypt: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let Some(source) = args.next() else {
        print_usage();
        return Err(MoeError::InvalidConfig(
            "source packed checkpoint path is required".to_string(),
        ));
    };
    if source == "--help" || source == "-h" {
        print_usage();
        return Ok(());
    }
    let destination = args.next().ok_or_else(|| {
        MoeError::InvalidConfig("destination checkpoint path is required".to_string())
    })?;
    let mut key_environment = None;
    let mut chunk_bytes = DEFAULT_ENCRYPTED_CHUNK_BYTES;
    while let Some(flag) = args.next() {
        let value = args
            .next()
            .ok_or_else(|| MoeError::InvalidConfig(format!("option '{flag}' requires a value")))?;
        match flag.as_str() {
            "--key-env" => key_environment = Some(value),
            "--chunk-mib" => {
                let mib = parse_positive_u32(&value, &flag)?;
                chunk_bytes = mib.checked_mul(MIB).ok_or_else(|| {
                    MoeError::InvalidConfig(format!("option '{flag}' byte count overflowed"))
                })?;
            }
            _ => return Err(MoeError::InvalidConfig(format!("unknown option '{flag}'"))),
        }
    }
    let key_environment = key_environment
        .ok_or_else(|| MoeError::InvalidConfig("option '--key-env' is required".to_string()))?;
    let key = SeekableWeightKey::from_env(&key_environment)?;
    let limits = InferenceLimits::default();
    let runtime = EmbeddedRuntime::new(DevicePreference::Cpu, limits.clone())?;
    let checkpoint = OlmoePackedCheckpoint::open(PathBuf::from(source), runtime)?;
    let report =
        checkpoint.encrypt_to_seekable(PathBuf::from(destination), &key, chunk_bytes, &limits)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

fn parse_positive_u32(value: &str, flag: &str) -> Result<u32> {
    let parsed = value.parse::<u32>().map_err(|error| {
        MoeError::InvalidConfig(format!("option '{flag}' must be an integer: {error}"))
    })?;
    if parsed == 0 {
        return Err(MoeError::InvalidConfig(format!(
            "option '{flag}' must be greater than zero"
        )));
    }
    Ok(parsed)
}

fn print_usage() {
    eprintln!(
        "Usage: a3s-moe-encrypt <packed-source> <destination> --key-env NAME [--chunk-mib N]"
    );
}
