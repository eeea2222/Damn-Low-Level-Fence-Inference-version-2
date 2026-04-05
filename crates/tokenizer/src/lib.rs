use common::{ChatFormat, FenceError, detect_chat_format};
use gguf::GgufFile;
use std::collections::HashMap;

pub const IM_START: i32 = 151644;
pub const IM_END: i32 = 151645;
pub const BOS: i32 = 151643;
pub const EOS: i32 = 151645;

pub struct Tokenizer {
    id_to_token: Vec<String>,
    token_to_id: HashMap<String, i32>,
    _merges: HashMap<String, i32>,
    chat_format: ChatFormat,
    bos_id: i32,
    eos_id: i32,
    llama3_begin_text_id: i32,
    llama3_start_header_id: i32,
    llama3_end_header_id: i32,
    llama3_eot_id: i32,
    mistral_bos_id: i32,
    mistral_eos_id: i32,
    inst_start_id: i32,
    inst_end_id: i32,
    pending_system: String,
    system_used: bool,
}

impl Default for Tokenizer {
    fn default() -> Self {
        Self {
            id_to_token: Vec::new(),
            token_to_id: HashMap::new(),
            _merges: HashMap::new(),
            chat_format: ChatFormat::ChatMl,
            bos_id: BOS,
            eos_id: EOS,
            llama3_begin_text_id: 0,
            llama3_start_header_id: 0,
            llama3_end_header_id: 0,
            llama3_eot_id: 0,
            mistral_bos_id: 0,
            mistral_eos_id: 0,
            inst_start_id: 0,
            inst_end_id: 0,
            pending_system: String::new(),
            system_used: false,
        }
    }
}

impl Tokenizer {
    pub fn load_from_gguf(&mut self, path: &str) -> Result<(), FenceError> {
        let gguf = GgufFile::open(path)?;
        let chat_template_raw = gguf
            .get_string("tokenizer.chat_template")
            .unwrap_or_default();
        let arch_hint = gguf.get_string("general.architecture").unwrap_or_default();
        self.detect_format_and_special_tokens(&chat_template_raw, &arch_hint);
        Ok(())
    }

    pub fn chat_format(&self) -> ChatFormat {
        self.chat_format
    }

    pub fn eos_id(&self) -> i32 {
        self.eos_id
    }

    pub fn bos_id(&self) -> i32 {
        self.bos_id
    }

    pub fn vocab(&self) -> &[String] {
        &self.id_to_token
    }

    pub fn encode(&self, text: &str) -> Vec<i32> {
        let encoded = bytes_to_gpt2_str(text.as_bytes());
        encoded
            .chars()
            .filter_map(|c| self.token_to_id.get(&c.to_string()).copied())
            .collect()
    }

    pub fn decode_one(&self, token_id: i32) -> String {
        self.id_to_token
            .get(token_id as usize)
            .map_or_else(String::new, |t| gpt2_str_to_bytes(t))
    }

    pub fn decode(&self, tokens: &[i32]) -> String {
        tokens
            .iter()
            .map(|&t| self.decode_one(t))
            .collect::<String>()
    }

    pub fn format_chat(&mut self, system_prompt: &str, user_message: &str) -> Vec<i32> {
        let mut tokens = Vec::new();
        self.system_used = false;
        self.pending_system.clear();

        self.begin_sequence(&mut tokens);
        if !system_prompt.is_empty() {
            self.append_system_turn(&mut tokens, system_prompt);
        }
        self.append_user_turn(&mut tokens, user_message);
        self.append_assistant_header(&mut tokens);
        tokens
    }

    pub fn begin_sequence(&mut self, tokens: &mut Vec<i32>) {
        self.system_used = false;
        self.pending_system.clear();
        match self.chat_format {
            ChatFormat::Llama3 => {
                if self.llama3_begin_text_id > 0 {
                    tokens.push(self.llama3_begin_text_id);
                }
            }
            ChatFormat::Llama2 | ChatFormat::Mistral => {
                if self.mistral_bos_id > 0 {
                    tokens.push(self.mistral_bos_id);
                }
            }
            ChatFormat::ChatMl | ChatFormat::Raw => {}
        }
    }

    pub fn append_system_turn(&mut self, tokens: &mut Vec<i32>, content: &str) {
        match self.chat_format {
            ChatFormat::ChatMl => {
                tokens.push(IM_START);
                self.push_encoded(tokens, &format!("system\n{content}"));
                tokens.push(IM_END);
                self.push_encoded(tokens, "\n");
            }
            ChatFormat::Llama3 => {
                if self.llama3_start_header_id > 0 {
                    tokens.push(self.llama3_start_header_id);
                }
                self.push_encoded(tokens, "system");
                if self.llama3_end_header_id > 0 {
                    tokens.push(self.llama3_end_header_id);
                }
                self.push_encoded(tokens, &format!("\n\n{content}"));
                if self.llama3_eot_id > 0 {
                    tokens.push(self.llama3_eot_id);
                }
            }
            ChatFormat::Llama2 | ChatFormat::Mistral => self.pending_system = content.to_string(),
            ChatFormat::Raw => self.push_encoded(tokens, &format!("{content}\n")),
        }
    }

    pub fn append_user_turn(&mut self, tokens: &mut Vec<i32>, content: &str) {
        match self.chat_format {
            ChatFormat::ChatMl => {
                tokens.push(IM_START);
                self.push_encoded(tokens, &format!("user\n{content}"));
                tokens.push(IM_END);
                self.push_encoded(tokens, "\n");
            }
            ChatFormat::Llama3 => {
                if self.llama3_start_header_id > 0 {
                    tokens.push(self.llama3_start_header_id);
                }
                self.push_encoded(tokens, "user");
                if self.llama3_end_header_id > 0 {
                    tokens.push(self.llama3_end_header_id);
                }
                self.push_encoded(tokens, &format!("\n\n{content}"));
                if self.llama3_eot_id > 0 {
                    tokens.push(self.llama3_eot_id);
                }
            }
            ChatFormat::Llama2 => {
                if self.inst_start_id > 0 {
                    tokens.push(self.inst_start_id);
                }
                if !self.pending_system.is_empty() && !self.system_used {
                    self.push_encoded(
                        tokens,
                        &format!(" <<SYS>>\n{}\n<</SYS>>\n\n", self.pending_system),
                    );
                    self.system_used = true;
                } else {
                    self.push_encoded(tokens, " ");
                }
                self.push_encoded(tokens, &format!("{content} "));
                if self.inst_end_id > 0 {
                    tokens.push(self.inst_end_id);
                }
            }
            ChatFormat::Mistral => {
                if self.inst_start_id > 0 {
                    tokens.push(self.inst_start_id);
                }
                if !self.pending_system.is_empty() && !self.system_used {
                    self.push_encoded(tokens, &format!(" {}\n\n", self.pending_system));
                    self.system_used = true;
                } else {
                    self.push_encoded(tokens, " ");
                }
                self.push_encoded(tokens, &format!("{content} "));
                if self.inst_end_id > 0 {
                    tokens.push(self.inst_end_id);
                }
            }
            ChatFormat::Raw => self.push_encoded(tokens, &format!("{content}\n")),
        }
    }

    pub fn append_assistant_header(&mut self, tokens: &mut Vec<i32>) {
        match self.chat_format {
            ChatFormat::ChatMl => {
                tokens.push(IM_START);
                self.push_encoded(tokens, "assistant\n");
            }
            ChatFormat::Llama3 => {
                if self.llama3_start_header_id > 0 {
                    tokens.push(self.llama3_start_header_id);
                }
                self.push_encoded(tokens, "assistant");
                if self.llama3_end_header_id > 0 {
                    tokens.push(self.llama3_end_header_id);
                }
                self.push_encoded(tokens, "\n\n");
            }
            ChatFormat::Llama2 | ChatFormat::Mistral => self.push_encoded(tokens, " "),
            ChatFormat::Raw => {}
        }
    }

    pub fn append_turn_end(&mut self, tokens: &mut Vec<i32>) {
        match self.chat_format {
            ChatFormat::ChatMl => {
                tokens.push(IM_END);
                self.push_encoded(tokens, "\n");
            }
            ChatFormat::Llama3 => {
                if self.llama3_eot_id > 0 {
                    tokens.push(self.llama3_eot_id);
                }
            }
            ChatFormat::Llama2 | ChatFormat::Mistral => {
                if self.mistral_eos_id > 0 {
                    tokens.push(self.mistral_eos_id);
                }
            }
            ChatFormat::Raw => {}
        }
    }

    fn push_encoded(&self, tokens: &mut Vec<i32>, text: &str) {
        tokens.extend(self.encode(text));
    }

    fn lookup_token(&self, token: &str) -> i32 {
        self.token_to_id.get(token).copied().unwrap_or(-1)
    }

    fn detect_format_and_special_tokens(&mut self, chat_template_raw: &str, arch_hint: &str) {
        self.chat_format = detect_chat_format(arch_hint, chat_template_raw);

        let lookup = |s: &str, me: &Tokenizer| me.lookup_token(s);
        self.llama3_begin_text_id = lookup("<|begin_of_text|>", self);
        self.llama3_start_header_id = lookup("<|start_header_id|>", self);
        self.llama3_end_header_id = lookup("<|end_header_id|>", self);
        self.llama3_eot_id = lookup("<|eot_id|>", self);
        self.mistral_bos_id = lookup("<s>", self);
        self.mistral_eos_id = lookup("</s>", self);
        self.inst_start_id = lookup("[INST]", self);
        self.inst_end_id = lookup("[/INST]", self);

        if self.chat_format == ChatFormat::Llama3 && self.llama3_begin_text_id > 0 {
            self.bos_id = self.llama3_begin_text_id;
        }
        if self.chat_format == ChatFormat::Llama3 && self.llama3_eot_id > 0 {
            self.eos_id = self.llama3_eot_id;
        }
        if (self.chat_format == ChatFormat::Mistral || self.chat_format == ChatFormat::Llama2)
            && self.mistral_eos_id > 0
        {
            self.eos_id = self.mistral_eos_id;
        }
    }
}

fn bytes_to_gpt2_str(raw: &[u8]) -> String {
    let mut byte_to_unicode = [0_u32; 256];
    let mut n = 0;
    for (b, slot) in byte_to_unicode.iter_mut().enumerate() {
        let b = b as u32;
        *slot = if (0x21..=0x7E).contains(&b)
            || (0xA1..=0xAC).contains(&b)
            || (0xAE..=0xFF).contains(&b)
        {
            b
        } else {
            let v = 256 + n;
            n += 1;
            v
        };
    }
    raw.iter()
        .filter_map(|&b| char::from_u32(byte_to_unicode[b as usize]))
        .collect()
}

fn gpt2_str_to_bytes(s: &str) -> String {
    let mut byte_to_unicode = [0_u32; 256];
    let mut unicode_to_byte = HashMap::<u32, u8>::new();
    let mut n = 0;
    for (b, slot) in byte_to_unicode.iter_mut().enumerate() {
        let b32 = b as u32;
        *slot = if (0x21..=0x7E).contains(&b32)
            || (0xA1..=0xAC).contains(&b32)
            || (0xAE..=0xFF).contains(&b32)
        {
            b32
        } else {
            let v = 256 + n;
            n += 1;
            v
        };
        unicode_to_byte.insert(*slot, b as u8);
    }

    let bytes: Vec<u8> = s
        .chars()
        .map(|ch| unicode_to_byte.get(&(ch as u32)).copied().unwrap_or(b'?'))
        .collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use common::{ChatFormat, chat_format_name};

    #[test]
    fn chat_format_detection_matches_cpp_intent() {
        assert_eq!(detect_chat_format("qwen3", ""), ChatFormat::ChatMl);
        assert_eq!(
            detect_chat_format("llama", "contains begin_of_text"),
            ChatFormat::Llama3
        );
        assert_eq!(detect_chat_format("", "[INST] ..."), ChatFormat::Mistral);
        assert_eq!(chat_format_name(ChatFormat::Raw), "Raw");
    }

    #[test]
    fn gpt2_roundtrip_simple_ascii() {
        let s = "hello world";
        let enc = bytes_to_gpt2_str(s.as_bytes());
        let dec = gpt2_str_to_bytes(&enc);
        assert_eq!(dec, s);
    }

    #[test]
    fn chatml_format_inserts_assistant_header() {
        let mut t = Tokenizer {
            chat_format: ChatFormat::ChatMl,
            ..Tokenizer::default()
        };
        let v = t.format_chat("sys", "usr");
        assert_eq!(v.first().copied(), Some(IM_START));
    }
}
