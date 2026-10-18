use serde::Serialize;
pub type AppResult<T> = Result<T, AppError>;

#[derive(Debug, Serialize)]
pub struct AppError {
    pub key: &'static str,
    pub message: String,
}

impl std::fmt::Display for AppError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.key, self.message)
    }
}

impl std::error::Error for AppError {}

impl AppError {
    pub fn new(key: &'static str, message: impl Into<String>) -> Self {
        Self {
            key,
            message: message.into(),
        }
    }

    pub fn config_missing() -> Self {
        Self::new("errors.config.missingWorkDir", "workDir is missing in config.json")
    }

    pub fn not_found(entity: &'static str) -> Self {
        Self::new("errors.common.notFound", format!("{entity} not found"))
    }

    pub fn validation(key: &'static str, message: impl Into<String>) -> Self {
        Self::new(key, message)
    }

    pub fn ai_service(message: impl Into<String>) -> Self {
        Self::new("errors.ai.service", message)
    }
}

impl From<std::io::Error> for AppError {
    fn from(value: std::io::Error) -> Self {
        Self::new("errors.fs.io", value.to_string())
    }
}

impl From<rusqlite::Error> for AppError {
    fn from(value: rusqlite::Error) -> Self {
        Self::new("errors.db.queryFailed", value.to_string())
    }
}

impl From<serde_json::Error> for AppError {
    fn from(value: serde_json::Error) -> Self {
        Self::new("errors.serialization.invalidPayload", value.to_string())
    }
}
