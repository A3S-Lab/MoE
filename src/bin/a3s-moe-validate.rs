use std::path::PathBuf;
use std::process::ExitCode;

use a3s_moe::olmoe::{
    validate_public_checkpoint as validate_olmoe, OlmoeValidationStatus, OlmoeValidationTolerances,
};
use a3s_moe::qwen3_moe::{
    validate_public_checkpoint as validate_qwen3_moe, Qwen3MoeValidationOptions,
    Qwen3MoeValidationStatus,
};
use a3s_moe::{MoeArchitecture, MoeError, Result};

const MIB: u64 = 1024 * 1024;

#[tokio::main]
async fn main() -> ExitCode {
    match run().await {
        Ok(status) => status,
        Err(error) => {
            eprintln!("a3s-moe-validate: {error}");
            ExitCode::FAILURE
        }
    }
}

async fn run() -> Result<ExitCode> {
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
    let mut parsed = ValidationArguments::default();
    while let Some(flag) = args.next() {
        let value = args
            .next()
            .ok_or_else(|| MoeError::InvalidConfig(format!("option '{flag}' requires a value")))?;
        match flag.as_str() {
            "--packed-checkpoint" => parsed.packed_checkpoint = Some(PathBuf::from(value)),
            "--host-cache-mib" => parsed.host_cache_mib = parse_u64(&value, &flag)?,
            "--logit-atol" => parsed.logits_abs = Some(parse_f32(&value, &flag)?),
            "--router-atol" => parsed.router_logits_abs = Some(parse_f32(&value, &flag)?),
            "--route-weight-atol" => parsed.route_weights_abs = Some(parse_f32(&value, &flag)?),
            _ => return Err(MoeError::InvalidConfig(format!("unknown option '{flag}'"))),
        }
    }

    let checkpoint = PathBuf::from(checkpoint);
    match MoeArchitecture::detect(&checkpoint)? {
        MoeArchitecture::Olmoe => {
            if parsed.packed_checkpoint.is_some() {
                return Err(MoeError::InvalidConfig(
                    "--packed-checkpoint applies only to Qwen3-MoE validation".to_string(),
                ));
            }
            let mut tolerances = OlmoeValidationTolerances::default();
            parsed.apply_tolerances(
                &mut tolerances.logits_abs,
                &mut tolerances.router_logits_abs,
                &mut tolerances.route_weights_abs,
            );
            let report = validate_olmoe(checkpoint, PathBuf::from(oracle), tolerances)?;
            println!("{}", serde_json::to_string_pretty(&report)?);
            Ok(if report.status == OlmoeValidationStatus::Passed {
                ExitCode::SUCCESS
            } else {
                ExitCode::FAILURE
            })
        }
        MoeArchitecture::Qwen3Moe => {
            let packed = parsed.packed_checkpoint.clone().ok_or_else(|| {
                MoeError::InvalidConfig(
                    "Qwen3-MoE validation requires --packed-checkpoint".to_string(),
                )
            })?;
            let mut options = Qwen3MoeValidationOptions::default();
            parsed.apply_tolerances(
                &mut options.tolerances.logits_abs,
                &mut options.tolerances.router_logits_abs,
                &mut options.tolerances.route_weights_abs,
            );
            options.residency_policy.host_cache_bytes =
                parsed.host_cache_mib.checked_mul(MIB).ok_or_else(|| {
                    MoeError::InvalidConfig("--host-cache-mib byte count overflowed".to_string())
                })?;
            let report =
                validate_qwen3_moe(checkpoint, packed, PathBuf::from(oracle), options).await?;
            println!("{}", serde_json::to_string_pretty(&report)?);
            Ok(if report.status == Qwen3MoeValidationStatus::Passed {
                ExitCode::SUCCESS
            } else {
                ExitCode::FAILURE
            })
        }
        MoeArchitecture::Qwen35Moe => Err(MoeError::InvalidConfig(
            "Qwen3.6 validation requires the qwen3.6 public-oracle validator".to_string(),
        )),
    }
}

struct ValidationArguments {
    packed_checkpoint: Option<PathBuf>,
    host_cache_mib: u64,
    logits_abs: Option<f32>,
    router_logits_abs: Option<f32>,
    route_weights_abs: Option<f32>,
}

impl Default for ValidationArguments {
    fn default() -> Self {
        Self {
            packed_checkpoint: None,
            host_cache_mib: 4096,
            logits_abs: None,
            router_logits_abs: None,
            route_weights_abs: None,
        }
    }
}

impl ValidationArguments {
    fn apply_tolerances(&self, logits: &mut f32, router_logits: &mut f32, routes: &mut f32) {
        if let Some(value) = self.logits_abs {
            *logits = value;
        }
        if let Some(value) = self.router_logits_abs {
            *router_logits = value;
        }
        if let Some(value) = self.route_weights_abs {
            *routes = value;
        }
    }
}

fn parse_f32(value: &str, flag: &str) -> Result<f32> {
    value.parse::<f32>().map_err(|error| {
        MoeError::InvalidConfig(format!("option '{flag}' must be a number: {error}"))
    })
}

fn parse_u64(value: &str, flag: &str) -> Result<u64> {
    value.parse::<u64>().map_err(|error| {
        MoeError::InvalidConfig(format!(
            "option '{flag}' must be a non-negative integer: {error}"
        ))
    })
}

fn print_usage() {
    eprintln!(
        "Usage: a3s-moe-validate <source-checkpoint> <oracle> \
         [--packed-checkpoint PATH] [--host-cache-mib N] \
         [--logit-atol F] [--router-atol F] [--route-weight-atol F]"
    );
}
