use std::time::Duration;

use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tauri::AppHandle;

use crate::{
    error::{AppError, AppResult},
    models::payloads::SummaryGenerationOptions,
    repository::db,
};

const OLLAMA_API_URL: &str = "http://127.0.0.1:11434";

#[derive(Debug, Serialize, Deserialize)]
struct OllamaChatMessage {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct OllamaChatResponse {
    message: OllamaChatMessage,
}

#[derive(Debug, Deserialize)]
struct OllamaModelList {
    models: Vec<OllamaModelInfo>,
}

#[derive(Debug, Deserialize)]
struct OllamaModelInfo {
    name: String,
}

pub async fn summarize_document(
    app: AppHandle,
    file_name: String,
    document_text: String,
    options: Option<SummaryGenerationOptions>,
) -> AppResult<(String, Vec<String>)> {
    let system_instruction = "Ты — профессиональный юридический аналитик. \
        Твоя задача — проанализировать текст юридического документа и составить краткую, но содержательную сводку. \
        Выдели основные факты, требования, стороны и риски. \
        Ответ должен быть строго в формате JSON: {\"summary\": \"...\", \"key_points\": [\"...\", \"...\"]}. \
        Используй профессиональный юридический язык.";

    let prompt = format!(
        "{}\n{}\n\nДокумент: {}\nТекст:\n{}",
        system_instruction,
        language_instruction(&app)?,
        file_name,
        limit_text(&document_text, 15000)
    );
    let raw = send_prompt(&app, &prompt, options).await?;
    let parsed = parse_json(&raw);
    let summary = parsed
        .get("summary")
        .and_then(Value::as_str)
        .unwrap_or("Не удалось сформировать сводку")
        .to_string();
    let key_points = parsed
        .get("key_points")
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(Value::as_str)
                .map(ToOwned::to_owned)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    Ok((summary, key_points))
}

pub async fn summarize_chain(
    app: AppHandle,
    scope: String,
    title: String,
    items: Vec<String>,
    options: Option<SummaryGenerationOptions>,
) -> AppResult<String> {
    if items.is_empty() {
        return Ok("Нет данных для анализа".to_string());
    }

    let body = items
        .iter()
        .enumerate()
        .map(|(index, item)| format!("{}. {}", index + 1, item))
        .collect::<Vec<_>>()
        .join("\n");

    let system_instruction = format!(
        "Ты — профессиональный юридический консультант и аналитик. Твоя задача — составить сводный отчет для объекта типа {scope} под названием \"{title}\". \
        Входные данные включают выписки из официальных реестров (ЕГРЮЛ), данные судебных дел (КАД Арбитр) и аналитику по внутренним документам. \
        Сгруппируй информацию логически: сначала общие сведения и статус, затем ключевые юридические факты и активность, в конце — выводы или риски. \
        Сводка должна быть профессиональной, структурированной и легко читаемой. \
        Ответ должен быть строго в формате JSON: {{\"summary\": \"...\"}}."
    );

    let prompt = format!(
        "{}\n{}\n\nЧасти анализа:\n{}",
        system_instruction,
        language_instruction(&app)?,
        limit_text(&body, 20000)
    );

    let raw = send_prompt(&app, &prompt, options).await?;
    let parsed = parse_json(&raw);
    Ok(parsed
        .get("summary")
        .and_then(Value::as_str)
        .unwrap_or("Не удалось сформировать сводку")
        .to_string())
}

async fn send_prompt(
    app: &AppHandle,
    prompt: &str,
    options: Option<SummaryGenerationOptions>,
) -> AppResult<String> {
    send_prompt_via_ollama(app, prompt, options).await
}

fn find_best_available_model(model_list: &OllamaModelList, preferred: Option<&str>) -> Option<String> {
    if let Some(preferred_name) = preferred {
        if let Some(exact) = model_list
            .models
            .iter()
            .find(|item| item.name == preferred_name || item.name.starts_with(preferred_name))
        {
            return Some(exact.name.clone());
        }
    }

    for family in ["gemma3", "gemma2"] {
        if let Some(candidate) = model_list
            .models
            .iter()
            .find(|item| item.name.to_lowercase().starts_with(family))
        {
            return Some(candidate.name.clone());
        }
    }

    model_list.models.first().map(|item| item.name.clone())
}

async fn send_prompt_via_ollama(
    _app: &AppHandle,
    prompt: &str,
    options: Option<SummaryGenerationOptions>,
) -> AppResult<String> {
    let client = Client::builder()
        .timeout(Duration::from_secs(120))
        .build()
        .map_err(|err| AppError::new("errors.ai.transport", err.to_string()))?;

    // Try to find an available model
    let models_res = client
        .get(format!("{}/api/tags", OLLAMA_API_URL))
        .send()
        .await
        .map_err(|err| AppError::new("errors.ai.unavailable", format!("Ollama not running: {}", err)))?;

    let model_list: OllamaModelList = models_res
        .json()
        .await
        .map_err(|err| AppError::new("errors.ai.invalidResponse", err.to_string()))?;

    let preferred_model = options
        .as_ref()
        .and_then(|item| item.model_name.as_deref());
    let num_ctx = options
        .as_ref()
        .and_then(|item| item.num_ctx)
        .unwrap_or(4096);

    let model = find_best_available_model(&model_list, preferred_model)
        .ok_or_else(|| AppError::ai_service("No models found in Ollama. Please pull a model first."))?;

    let payload = json!({
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "stream": false,
        "options": {
            "temperature": 0.1,
            "num_predict": 2048,
            "num_ctx": num_ctx
        }
    });

    let response = client
        .post(format!("{}/api/chat", OLLAMA_API_URL))
        .json(&payload)
        .send()
        .await
        .map_err(|err| AppError::new("errors.ai.unavailable", err.to_string()))?;

    if !response.status().is_success() {
        let status = response.status();
        let err_body = response.text().await.unwrap_or_default();
        return Err(AppError::new(
            "errors.ai.requestFailed",
            format!("Ollama status {}: {}", status, err_body),
        ));
    }

    let json: OllamaChatResponse = response
        .json()
        .await
        .map_err(|err| AppError::new("errors.ai.invalidResponse", err.to_string()))?;
    Ok(json.message.content)
}

fn parse_json(raw: &str) -> Value {
    let raw = raw.trim();
    let start = raw.find('{').unwrap_or(0);
    let end = raw.rfind('}').map(|idx| idx + 1).unwrap_or(raw.len());
    if start >= end {
        return json!({});
    }
    serde_json::from_str(&raw[start..end]).unwrap_or_else(|_| json!({}))
}

fn language_instruction(app: &AppHandle) -> AppResult<String> {
    let conn = db::open_user_db(app)?;
    let value = conn
        .query_row(
            "SELECT value FROM app_settings WHERE key = 'ui_language' LIMIT 1",
            [],
            |row| row.get::<_, String>(0),
        )
        .unwrap_or_else(|_| "ru".to_string());

    let instruction = match value.as_str() {
        "en" => "Respond in English only.",
        "sr" => "Odgovaraj samo na srpskom (latinica).",
        _ => "Отвечай только на русском языке.",
    };

    Ok(instruction.to_string())
}

fn limit_text(text: &str, max_chars: usize) -> String {
    text.chars().take(max_chars).collect()
}
