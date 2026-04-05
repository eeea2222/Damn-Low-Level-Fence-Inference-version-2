use common::{CliConfig, FenceError, chat_format_name};
use fence_model::{Qwen3Model, apply_cli_to_model_config};
use std::env;
use tokenizer::Tokenizer;

fn print_usage(prog: &str) {
    println!("Usage: {prog} --model PATH [options]\n");
    println!("Required:");
    println!("  --model PATH       GGUF model file\n");
    println!("Options:");
    println!("  --prompt TEXT      Single prompt (non-interactive mode)");
    println!("  --system TEXT      System prompt (default: 'You are a helpful assistant.')");
    println!("  --max-tokens N     Max new tokens to generate (default: 512)");
    println!("  --max-ctx N        Max context length in tokens (default: 4096)");
    println!("  --temp F           Sampling temperature (default: 0.7)");
    println!("  --top-p F          Top-p nucleus cutoff (default: 0.8)");
    println!("  --top-k N          Top-k candidates (default: 50)");
    println!("  --rep-penalty F    Repetition penalty (default: 1.15)");
    println!("  --debug            Print per-layer activation statistics");
    println!("  --help             Show this help\n");
    println!("Supported architectures: Qwen2, Qwen3, LLaMA-3, LLaMA-2, Mistral, Mixtral");
}

fn parse_cli_args() -> Result<CliConfig, FenceError> {
    let mut cfg = CliConfig::default();
    let args: Vec<String> = env::args().collect();

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" if i + 1 < args.len() => {
                cfg.model_path = args[i + 1].clone();
                i += 1;
            }
            "--prompt" if i + 1 < args.len() => {
                cfg.prompt = Some(args[i + 1].clone());
                i += 1;
            }
            "--system" if i + 1 < args.len() => {
                cfg.system_prompt = args[i + 1].clone();
                i += 1;
            }
            "--max-tokens" if i + 1 < args.len() => {
                cfg.max_tokens = args[i + 1].parse().unwrap_or(cfg.max_tokens);
                i += 1;
            }
            "--max-ctx" if i + 1 < args.len() => {
                cfg.max_ctx = args[i + 1].parse().unwrap_or(cfg.max_ctx);
                i += 1;
            }
            "--temp" if i + 1 < args.len() => {
                cfg.sampling.temperature = args[i + 1].parse().unwrap_or(cfg.sampling.temperature);
                i += 1;
            }
            "--top-p" if i + 1 < args.len() => {
                cfg.sampling.top_p = args[i + 1].parse().unwrap_or(cfg.sampling.top_p);
                i += 1;
            }
            "--top-k" if i + 1 < args.len() => {
                cfg.sampling.top_k = args[i + 1].parse().unwrap_or(cfg.sampling.top_k);
                i += 1;
            }
            "--rep-penalty" if i + 1 < args.len() => {
                cfg.sampling.repetition_penalty = args[i + 1]
                    .parse()
                    .unwrap_or(cfg.sampling.repetition_penalty);
                i += 1;
            }
            "--debug" => {
                cfg.debug = true;
            }
            "--help" => {
                print_usage(args.first().map_or("fence", String::as_str));
                std::process::exit(0);
            }
            other => {
                return Err(FenceError::Parse(format!("unknown option: {other}")));
            }
        }
        i += 1;
    }

    if cfg.model_path.is_empty() {
        return Err(FenceError::Parse("--model PATH is required".into()));
    }

    Ok(cfg)
}

fn run() -> Result<(), FenceError> {
    println!("╔═══════════════════════════════════════════════════╗");
    println!("║   Fence Inference Engine v0.2                     ║");
    println!("║   Multi-architecture GGUF — Rust migration        ║");
    println!("╚═══════════════════════════════════════════════════╝\n");

    let cfg = parse_cli_args()?;

    let mut tokenizer = Tokenizer::default();
    tokenizer.load_from_gguf(&cfg.model_path)?;
    println!(
        "Chat format : {}",
        chat_format_name(tokenizer.chat_format())
    );

    let mut model = Qwen3Model::new();
    apply_cli_to_model_config(&cfg, &mut model);
    model.load(&cfg.model_path)?;

    if let Some(prompt) = &cfg.prompt {
        let mut prompt_tokens = tokenizer.format_chat(&cfg.system_prompt, prompt);
        println!("Prompt tokens: {}", prompt_tokens.len());
        let output = model.generate(&prompt_tokens, cfg.max_tokens, true)?;
        prompt_tokens = output;
        println!("Generated token count: {}", prompt_tokens.len());
        model.unload();
        return Ok(());
    }

    println!("Interactive mode is not fully ported yet in Rust build.");
    model.unload();
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {e}");
        if let Some(prog) = env::args().next() {
            print_usage(&prog);
        }
        std::process::exit(1);
    }
}
