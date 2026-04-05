use thiserror::Error;

#[derive(Debug, Error)]
pub enum FenceError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("parse error: {0}")]
    Parse(String),
    #[error("missing key: {0}")]
    MissingKey(String),
    #[error("type mismatch: {0}")]
    TypeMismatch(String),
    #[error("unsupported: {0}")]
    Unsupported(String),
}
