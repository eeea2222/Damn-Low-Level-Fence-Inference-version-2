#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatFormat {
    ChatMl,
    Llama3,
    Llama2,
    Mistral,
    Raw,
}

pub fn detect_chat_format(arch: &str, chat_template: &str) -> ChatFormat {
    if !chat_template.is_empty() {
        if chat_template.contains("im_start") {
            return ChatFormat::ChatMl;
        }
        if chat_template.contains("begin_of_text") || chat_template.contains("start_header_id") {
            return ChatFormat::Llama3;
        }
        if chat_template.contains("<<SYS>>") {
            return ChatFormat::Llama2;
        }
        if chat_template.contains("[INST]") {
            return ChatFormat::Mistral;
        }
    }

    match arch {
        "qwen2" | "qwen3" | "internlm2" | "deepseek2" => ChatFormat::ChatMl,
        "llama" => ChatFormat::Llama3,
        "mistral" | "mixtral" => ChatFormat::Mistral,
        _ => ChatFormat::Raw,
    }
}

pub fn chat_format_name(fmt: ChatFormat) -> &'static str {
    match fmt {
        ChatFormat::ChatMl => "ChatML",
        ChatFormat::Llama3 => "LLaMA-3",
        ChatFormat::Llama2 => "LLaMA-2",
        ChatFormat::Mistral => "Mistral",
        ChatFormat::Raw => "Raw",
    }
}
