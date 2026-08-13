use std::path::PathBuf;
use std::process::ExitCode;

use a3s_moe::olmoe::{OlmoeCheckpoint, OlmoeConversionOptions};
use a3s_moe::qwen3_moe::Qwen3MoeCheckpoint;
use a3s_moe::{MoeArchitecture, MoeError, Result};

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
    let architecture = MoeArchitecture::detect(&source)?;
    let limits = architecture.inference_limits();
    let report = match architecture {
        MoeArchitecture::Olmoe => {
            OlmoeCheckpoint::open(&source)?.convert_to_packed(&destination, &limits, options)?
        }
        MoeArchitecture::Qwen3Moe => {
            Qwen3MoeCheckpoint::open(&source)?.convert_to_packed(&destination, &limits, options)?
        }
    };
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
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
