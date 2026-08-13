use std::path::PathBuf;
use std::process::ExitCode;

use a3s_moe::olmoe::{OlmoeCheckpoint, OlmoeConversionOptions};
use a3s_moe::qwen3_moe::Qwen3MoeCheckpoint;
use a3s_moe::{MoeError, Result};
use a3s_power::inference::InferenceLimits;
use serde::Deserialize;

const MIB: u64 = 1024 * 1024;

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("a3s-moe-pack: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let Some(source) = args.next() else {
        print_usage();
        return Err(MoeError::InvalidConfig(
            "source checkpoint path is required".to_string(),
        ));
    };
    if source == "--help" || source == "-h" {
        print_usage();
        return Ok(());
    }
    let destination = args.next().ok_or_else(|| {
        MoeError::InvalidConfig("destination checkpoint path is required".to_string())
    })?;
    let mut options = OlmoeConversionOptions::default();
    while let Some(flag) = args.next() {
        let value = args
            .next()
            .ok_or_else(|| MoeError::InvalidConfig(format!("option '{flag}' requires a value")))?;
        match flag.as_str() {
            "--experts-per-file" => {
                options.experts_per_file = parse_positive(&value, &flag)?;
            }
            "--max-buffer-mib" => {
                let mib = u64::try_from(parse_positive(&value, &flag)?).map_err(|_| {
                    MoeError::InvalidConfig(format!("option '{flag}' is too large"))
                })?;
                options.max_buffer_bytes = mib.checked_mul(MIB).ok_or_else(|| {
                    MoeError::InvalidConfig(format!("option '{flag}' byte count overflowed"))
                })?;
            }
            _ => {
                return Err(MoeError::InvalidConfig(format!("unknown option '{flag}'")));
            }
        }
    }

    let source = PathBuf::from(source);
    let destination = PathBuf::from(destination);
    let report = match architecture(&source)?.as_str() {
        "olmoe" => OlmoeCheckpoint::open(&source)?.convert_to_packed(
            &destination,
            &InferenceLimits::default(),
            options,
        )?,
        "qwen3_moe" => Qwen3MoeCheckpoint::open(&source)?.convert_to_packed(
            &destination,
            &InferenceLimits::default(),
            options,
        )?,
        model_type => {
            return Err(MoeError::InvalidConfig(format!(
                "unsupported checkpoint model_type '{model_type}'"
            )))
        }
    };
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

#[derive(Deserialize)]
struct ArchitectureProbe {
    model_type: String,
}

fn architecture(source: &std::path::Path) -> Result<String> {
    let path = source.join("config.json");
    let metadata = path.symlink_metadata()?;
    if !metadata.is_file() || metadata.file_type().is_symlink() || metadata.len() > 1_048_576 {
        return Err(MoeError::InvalidConfig(format!(
            "checkpoint config '{}' must be a regular non-symlink file no larger than 1048576 bytes",
            path.display()
        )));
    }
    Ok(serde_json::from_slice::<ArchitectureProbe>(&std::fs::read(path)?)?.model_type)
}

fn parse_positive(value: &str, flag: &str) -> Result<usize> {
    let parsed = value.parse::<usize>().map_err(|error| {
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
        "Usage: a3s-moe-pack <source> <destination> [--experts-per-file N] [--max-buffer-mib N]\n\
         Source model_type must be olmoe or qwen3_moe."
    );
}
