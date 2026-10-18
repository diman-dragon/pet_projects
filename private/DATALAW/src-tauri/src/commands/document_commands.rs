use std::path::Path;
use std::fs;
use zip::ZipArchive;
use std::io::Read;
use regex::Regex;
use crate::error::{AppResult, AppError};

#[tauri::command]
pub async fn extract_document_text(file_path: String) -> AppResult<String> {
    extract_text_from_path(Path::new(&file_path))
}

fn extract_text(path: &Path) -> AppResult<String> {
    let extension = path.extension().and_then(|value| value.to_str()).unwrap_or_default().to_lowercase();
    match extension.as_str() {
        "txt" | "md" | "markdown" => fs::read_to_string(path).map_err(AppError::from),
        "docx" => extract_docx(path),
        "pdf" => pdf_extract::extract_text(path).map_err(|err| AppError::new("errors.document.extractFailed", err.to_string())),
        _ => Ok(String::new()),
    }
}

fn extract_docx(path: &Path) -> AppResult<String> {
    let file = fs::File::open(path)?;
    let mut archive = ZipArchive::new(file)
        .map_err(|err: zip::result::ZipError| AppError::new("errors.document.extractFailed", err.to_string()))?;
    let mut xml = String::new();
    archive
        .by_name("word/document.xml")
        .map_err(|err: zip::result::ZipError| AppError::new("errors.document.extractFailed", err.to_string()))?
        .read_to_string(&mut xml)
        .map_err(|err: std::io::Error| AppError::new("errors.document.extractFailed", err.to_string()))?;
    
    let text = Regex::new(r"<[^>]+>")
        .map_err(|err: regex::Error| AppError::new("errors.regex.invalid", err.to_string()))?
        .replace_all(&xml, " ")
        .to_string();
    Ok(text)
}

fn extract_text_from_path(path: &Path) -> AppResult<String> {
    extract_text(path)
}
