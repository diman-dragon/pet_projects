// src-tauri/src/commands/ai_pull_progress.rs
//
// FIX: Структура события приведена в соответствие с ожиданиями фронтенда.
//
// Ранее Rust эмитировал { status, completed, total, percent, digest }
// но MultiModelPullStep.tsx ожидал { stage, percent, total_mb, done_mb }.
// Из-за несовместимости структур прогресс никогда не отображался в UI.
//
// Теперь:
// - Rust маппит Ollama-статусы на stage: "manifest"|"download"|"verify"|"complete"
// - Размеры передаются в MB (не байтах) как total_mb / done_mb
// - percent всегда число 0–100 (не Option)

use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter};

use crate::error::{AppError, AppResult};

const OLLAMA_API_URL: &str = "http://127.0.0.1:11434";

/// Фронтенд-совместимое событие прогресса скачивания модели.
/// Эмитируется в канал "ai-pull-progress".
///
/// Поля совпадают с интерфейсом PullProgress в MultiModelPullStep.tsx:
///   stage:    'manifest' | 'download' | 'verify' | 'complete'
///   percent:  0–100 (всегда число)
///   total_mb: общий размер в MB (опционально)
///   done_mb:  скачано в MB (опционально)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPullProgress {
    pub stage: String,
    pub percent: f32,
    pub total_mb: Option<f64>,
    pub done_mb: Option<f64>,
}

/// Финальное событие — успех или ошибка.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPullDone {
    pub model: String,
    pub success: bool,
    pub error: Option<String>,
}

/// Маппит статус Ollama API на stage для UI.
///
/// Ollama отдаёт следующие статусы в процессе pull:
///   "pulling manifest"          → manifest (получение метаданных)
///   "pulling fs layer"          → download (начало слоя)
///   "downloading"               → download (идёт скачивание байт)
///   "verifying sha256 digest"   → verify   (проверка хеша)
///   "writing manifest"          → verify   (запись метаданных)
///   "removing any unused layers"→ complete (финальная уборка)
///   "success"                   → complete (готово)
fn map_status_to_stage(status: &str) -> &'static str {
    let s = status.to_lowercase();
    if s.contains("manifest") && s.contains("pulling") {
        "manifest"
    } else if s.contains("pulling fs layer") || s.contains("downloading") || s.contains("pulling") {
        "download"
    } else if s.contains("verifying") || s.contains("writing manifest") {
        "verify"
    } else if s.contains("removing") || s.contains("success") {
        "complete"
    } else {
        "manifest"
    }
}

/// Вычисляет процент из completed/total.
/// Возвращает 0.0 если данных нет.
fn compute_percent(completed: Option<u64>, total: Option<u64>) -> f32 {
    match (completed, total) {
        (Some(c), Some(t)) if t > 0 => ((c as f64 / t as f64) * 100.0) as f32,
        _ => 0.0,
    }
}

/// Конвертирует байты в MB с точностью до 1 знака.
fn bytes_to_mb(bytes: u64) -> f64 {
    (bytes as f64) / 1_048_576.0
}

/// [5.1] Tauri-команда: скачивает модель через Ollama /api/pull со стримом прогресса.
///
/// Фронтенд слушает события:
///   "ai-pull-progress" → ModelPullProgress (каждый значимый chunk)
///   "ai-pull-done"     → ModelPullDone (в конце)
#[tauri::command]
pub async fn ai_pull_model_with_progress(
    app: AppHandle,
    model: String,
) -> AppResult<()> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(7200)) // 2 часа — большие модели
        .build()
        .map_err(|e| AppError::ai_service(format!("HTTP client error: {}", e)))?;

    let url = format!("{}/api/pull", OLLAMA_API_URL);

    let body = serde_json::json!({
        "model": model,
        "stream": true,
    });

    let response = client
        .post(&url)
        .json(&body)
        .send()
        .await
        .map_err(|e| AppError::ai_service(format!("Ollama unreachable: {}", e)))?;

    if !response.status().is_success() {
        let status = response.status();
        let text = response.text().await.unwrap_or_default();
        let err = format!("Ollama pull error {}: {}", status, text);
        let _ = app.emit(
            "ai-pull-done",
            ModelPullDone {
                model: model.clone(),
                success: false,
                error: Some(err.clone()),
            },
        );
        return Err(AppError::ai_service(err));
    }

    let mut stream = response.bytes_stream();
    let mut last_status = String::new();
    // Накапливаем суммарный прогресс по всем слоям (резерв для будущей статистики)
    let mut _total_bytes_sum: u64 = 0;
    let mut done_bytes_sum: u64 = 0;
    // Трекер слоёв: digest → total_bytes (чтобы суммировать)
    let mut layer_totals: std::collections::HashMap<String, u64> = std::collections::HashMap::new();

    while let Some(chunk) = stream.next().await {
        let chunk = match chunk {
            Ok(c) => c,
            Err(e) => {
                let err = format!("Stream error: {}", e);
                let _ = app.emit(
                    "ai-pull-done",
                    ModelPullDone {
                        model: model.clone(),
                        success: false,
                        error: Some(err.clone()),
                    },
                );
                return Err(AppError::ai_service(err));
            }
        };

        // Ollama шлёт newline-delimited JSON
        let text = String::from_utf8_lossy(&chunk);
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            if let Ok(val) = serde_json::from_str::<serde_json::Value>(line) {
                let status = val
                    .get("status")
                    .and_then(|s| s.as_str())
                    .unwrap_or("")
                    .to_string();

                let completed = val.get("completed").and_then(|v| v.as_u64());
                let total = val.get("total").and_then(|v| v.as_u64());
                let digest = val
                    .get("digest")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());

                last_status = status.clone();
                let stage = map_status_to_stage(&status);

                // Накапливаем суммарный прогресс по слоям
                if let Some(ref dgst) = digest {
                    if let Some(t) = total {
                        layer_totals.insert(dgst.clone(), t);
                        _total_bytes_sum = layer_totals.values().sum();
                    }
                    if let Some(c) = completed {
                        done_bytes_sum = done_bytes_sum.max(c); // упрощённо для текущего слоя
                    }
                }

                let percent = if stage == "complete" || status == "success" {
                    100.0
                } else {
                    compute_percent(completed, total)
                };

                let total_mb = total.map(bytes_to_mb);
                let done_mb = completed.map(bytes_to_mb);

                let _ = app.emit(
                    "ai-pull-progress",
                    ModelPullProgress {
                        stage: stage.to_string(),
                        percent,
                        total_mb,
                        done_mb,
                    },
                );
            }
        }
    }

    let success = last_status == "success" || last_status.contains("success");
    let _ = app.emit(
        "ai-pull-done",
        ModelPullDone {
            model: model.clone(),
            success,
            error: if success {
                None
            } else {
                Some(format!("Unexpected final status: {}", last_status))
            },
        },
    );

    if success {
        Ok(())
    } else {
        Err(AppError::ai_service(format!(
            "Pull ended with unexpected status: {}",
            last_status
        )))
    }
}
