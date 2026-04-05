#[derive(Debug, Clone)]
pub struct SamplingConfig {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: i32,
    pub repetition_penalty: f32,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_p: 0.8,
            top_k: 50,
            repetition_penalty: 1.15,
        }
    }
}

#[derive(Debug, Clone)]
pub struct CliConfig {
    pub model_path: String,
    pub prompt: Option<String>,
    pub system_prompt: String,
    pub max_tokens: i32,
    pub max_ctx: i32,
    pub debug: bool,
    pub sampling: SamplingConfig,
}

impl Default for CliConfig {
    fn default() -> Self {
        Self {
            model_path: String::new(),
            prompt: None,
            system_prompt: "You are a helpful assistant.".to_string(),
            max_tokens: 512,
            max_ctx: 4096,
            debug: false,
            sampling: SamplingConfig::default(),
        }
    }
}
