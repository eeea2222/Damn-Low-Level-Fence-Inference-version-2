use common::{CliConfig, FenceError, SamplingConfig};
use gguf::GgufFile;

#[derive(Debug, Clone)]
pub struct Qwen3Config {
    pub n_layers: i32,
    pub embed_dim: i32,
    pub ff_dim: i32,
    pub n_heads: i32,
    pub n_kv_heads: i32,
    pub head_dim: i32,
    pub vocab_size: i32,
    pub max_ctx: i32,
    pub rope_freq_base: f32,
    pub rms_eps: f32,
    pub sampling: SamplingConfig,
}

impl Default for Qwen3Config {
    fn default() -> Self {
        Self {
            n_layers: 36,
            embed_dim: 2560,
            ff_dim: 9728,
            n_heads: 32,
            n_kv_heads: 8,
            head_dim: 128,
            vocab_size: 151_936,
            max_ctx: 4096,
            rope_freq_base: 5_000_000.0,
            rms_eps: 1e-6,
            sampling: SamplingConfig::default(),
        }
    }
}

impl Qwen3Config {
    pub fn q_dim(&self) -> i32 {
        self.n_heads * self.head_dim
    }
    pub fn kv_dim(&self) -> i32 {
        self.n_kv_heads * self.head_dim
    }
    pub fn heads_per_group(&self) -> i32 {
        self.n_heads / self.n_kv_heads
    }
}

pub struct Qwen3Model {
    pub config: Qwen3Config,
    gguf: Option<GgufFile>,
}

impl Default for Qwen3Model {
    fn default() -> Self {
        Self::new()
    }
}

impl Qwen3Model {
    pub fn new() -> Self {
        Self {
            config: Qwen3Config::default(),
            gguf: None,
        }
    }

    pub fn load(&mut self, gguf_path: &str) -> Result<(), FenceError> {
        let file = GgufFile::open(gguf_path)?;
        self.gguf = Some(file);
        Ok(())
    }

    pub fn unload(&mut self) {
        self.gguf = None;
    }

    pub fn generate(
        &mut self,
        prompt_tokens: &[i32],
        _max_new_tokens: i32,
        _print_tokens: bool,
    ) -> Result<Vec<i32>, FenceError> {
        if self.gguf.is_none() {
            return Err(FenceError::Unsupported("model not loaded".into()));
        }
        Ok(prompt_tokens.to_vec())
    }
}

pub fn apply_cli_to_model_config(cfg: &CliConfig, model: &mut Qwen3Model) {
    model.config.max_ctx = cfg.max_ctx;
    model.config.sampling = cfg.sampling.clone();
}
