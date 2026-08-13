use std::path::PathBuf;
use std::process::ExitCode;

use a3s_moe::olmoe::{
    validate_public_checkpoint, OlmoeValidationStatus, OlmoeValidationTolerances,
};
use a3s_moe::{MoeError, Result};

fn main() -> ExitCode {
    match run() {
        Ok(status) => status,
        Err(error) => {
            eprintln!("a3s-moe-validate: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<ExitCode> {
    let mut args = std::env::args().skip(1);
    let Some(checkpoint) = args.next() else {
        print_usage();
        return Err(MoeError::InvalidConfig(
            "checkpoint path is required".to_string(),
        ));
    };
    if checkpoint == "--help" || checkpoint == "-h" {
        print_usage();
        return Ok(ExitCode::SUCCESS);
    }
    let oracle = args
        .next()
        .ok_or_else(|| MoeError::InvalidConfig("public oracle path is required".to_string()))?;
    let mut tolerances = OlmoeValidationTolerances::default();
    while let Some(flag) = args.next() {
        let value = args
            .next()
            .ok_or_else(|| MoeError::InvalidConfig(format!("option '{flag}' requires a value")))?;
        let parsed = value.parse::<f32>().map_err(|error| {
            MoeError::InvalidConfig(format!("option '{flag}' must be a number: {error}"))
        })?;
        match flag.as_str() {
            "--logit-atol" => tolerances.logits_abs = parsed,
            "--router-atol" => tolerances.router_logits_abs = parsed,
            "--route-weight-atol" => tolerances.route_weights_abs = parsed,
            _ => return Err(MoeError::InvalidConfig(format!("unknown option '{flag}'"))),
        }
    }

    let report =
        validate_public_checkpoint(PathBuf::from(checkpoint), PathBuf::from(oracle), tolerances)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(if report.status == OlmoeValidationStatus::Passed {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    })
}

fn print_usage() {
    eprintln!(
        "Usage: a3s-moe-validate <checkpoint> <oracle> [--logit-atol F] [--router-atol F] [--route-weight-atol F]"
    );
}
