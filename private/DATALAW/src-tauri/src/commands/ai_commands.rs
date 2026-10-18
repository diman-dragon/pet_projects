use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::process::Command;
use sysinfo::System;
use tauri::{AppHandle, Emitter};

use crate::error::{AppError, AppResult};

const OLLAMA_API_URL: &str = "http://127.0.0.1:11434";

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
    pub images: Option<Vec<String>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct OllamaOptions {
    pub num_ctx:     Option<u32>,
    pub num_predict: Option<i32>,
    pub temperature: Option<f32>,
    pub top_p:       Option<f32>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ChatCompletionRequest {
    pub model:    String,
    pub messages: Vec<ChatMessage>,
    pub stream:   Option<bool>,
    pub options:  Option<OllamaOptions>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ChatResponse {
    pub message: Option<ChatMessage>,
    pub response: Option<String>,
    pub done: bool,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct OllamaModel {
    pub name: String,
    pub size: u64,
    pub details: Option<serde_json::Value>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct OllamaListResponse {
    pub models: Vec<OllamaModel>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SidecarStatus {
    pub running: bool,
    pub version: Option<String>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRecommendation {
    pub name: String,
    pub label: String,
    pub profile: String,
    pub min_memory_gb: u64,
    pub recommended_memory_gb: u64,
    pub rationale: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareInfo {
    pub os: String,
    pub arch: String,
    pub cpu_brand: String,
    pub cpu_cores: usize,
    pub total_memory_gb: u64,
    pub available_memory_gb: u64,
    pub ollama_installed: bool,
    pub ollama_running: bool,
    pub ollama_version: Option<String>,
    pub install_supported: bool,
    pub recommended_model: Option<String>,
    pub recommended_models: Vec<ModelRecommendation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaInstallResult {
    pub installed: bool,
    pub method: String,
    pub message: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct OllamaPullRequest {
    pub model: String,
    pub stream: Option<bool>,
}

fn to_gb(bytes: u64) -> u64 {
    ((bytes as f64) / 1024f64 / 1024f64 / 1024f64).floor() as u64
}

fn detect_ollama_binary() -> Option<String> {
    // Try common locations first
    let candidates = [
        "C:\\Program Files\\Ollama\\ollama.exe",
        "C:\\Program Files (x86)\\Ollama\\ollama.exe",
        "ollama", // If in PATH
    ];
    
    for path in &candidates {
        if Path::new(path).exists() {
            return Some(path.to_string());
        }
    }
    
    // Try to find in PATH using where command
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x08000000;
        if let Ok(output) = Command::new("where")
            .arg("ollama")
            .creation_flags(CREATE_NO_WINDOW)
            .output() 
        {
            if output.status.success() {
                let stdout = String::from_utf8_lossy(&output.stdout);
                let path = stdout.lines().next()?;
                return Some(path.trim().to_string());
            }
        }
    }
    
    #[cfg(not(windows))]
    {
        if let Ok(output) = Command::new("which").arg("ollama").output() {
            if output.status.success() {
                let stdout = String::from_utf8_lossy(&output.stdout);
                let path = stdout.lines().next()?;
                return Some(path.trim().to_string());
            }
        }
    }
    
    None
}

fn read_local_ollama_version(binary: Option<&str>) -> Option<String> {
    let executable = binary.unwrap_or("ollama");
    
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x08000000;
        let output = Command::new(executable)
            .arg("--version")
            .creation_flags(CREATE_NO_WINDOW)
            .output()
            .ok()?;
        if !output.status.success() {
            return None;
        }
        let stdout = String::from_utf8_lossy(&output.stdout);
        let version = stdout.trim();
        if version.is_empty() {
            None
        } else {
            Some(version.to_string())
        }
    }
    
    #[cfg(not(windows))]
    {
        let output = Command::new(executable).arg("--version").output().ok()?;
        if !output.status.success() {
            return None;
        }
        let stdout = String::from_utf8_lossy(&output.stdout);
        let version = stdout.trim();
        if version.is_empty() {
            None
        } else {
            Some(version.to_string())
        }
    }
}

fn build_recommendations(total_memory_gb: u64) -> Vec<ModelRecommendation> {
    let all = vec![
        ModelRecommendation {
            name: "mxbai-embed-large".to_string(),
            label: "mxbai-embed-large (эмбеддинги, 274 MB)".to_string(),
            profile: "embed".to_string(),
            min_memory_gb: 6,
            recommended_memory_gb: 6,
            rationale: "Embed-only модель 1024D. Заменяет nomic-embed-text для семантического RAG-поиска. Лёгкая, работает на любом профиле.".to_string(),
        },
        ModelRecommendation {
            name: "qwen2.5:7b".to_string(),
            label: "Qwen 2.5 7B".to_string(),
            profile: "quality".to_string(),
            min_memory_gb: 16,
            recommended_memory_gb: 24,
            rationale: "Высокое качество рассуждений для сложных правовых задач и длинных текстов.".to_string(),
        },
        ModelRecommendation {
            name: "gemma3:4b".to_string(),
            label: "Gemma 3 4B".to_string(),
            profile: "balanced".to_string(),
            min_memory_gb: 10,
            recommended_memory_gb: 12,
            rationale: "Приоритетная модель для юридических текстов на русском языке. Рекомендуется для систем 10–16 GB RAM.".to_string(),
        },
        ModelRecommendation {
            name: "phi3:mini".to_string(),
            label: "Phi-3 Mini".to_string(),
            profile: "balanced".to_string(),
            min_memory_gb: 12,
            recommended_memory_gb: 16,
            rationale: "Сбалансированная модель для составления документов и анализа.".to_string(),
        },
        ModelRecommendation {
            name: "qwen2.5:3b".to_string(),
            label: "Qwen 2.5 3B".to_string(),
            profile: "balanced".to_string(),
            min_memory_gb: 8,
            recommended_memory_gb: 10,
            rationale: "Альтернатива для русскоязычных задач при 8+ GB RAM.".to_string(),
        },
        ModelRecommendation {
            name: "gemma2:2b".to_string(),
            label: "Gemma 2 2B".to_string(),
            profile: "light".to_string(),
            min_memory_gb: 8,
            recommended_memory_gb: 10,
            rationale: "Компактная и быстрая, подходит для систем с 8–10 GB RAM.".to_string(),
        },
        ModelRecommendation {
            name: "gemma3:1b".to_string(),
            label: "Gemma 3 1B".to_string(),
            profile: "minimal".to_string(),
            min_memory_gb: 6,
            recommended_memory_gb: 8,
            rationale: "Минимальная модель для машин 6–8 GB RAM. Быстрые ответы, базовые задачи.".to_string(),
        },
    ];

    let embed_models: Vec<_> = all.iter().filter(|m| m.profile == "embed").cloned().collect();
    let mut llm_models: Vec<_> = all
        .iter()
        .filter(|m| m.profile != "embed" && total_memory_gb >= m.min_memory_gb)
        .cloned()
        .collect();
    llm_models.sort_by(|a, b| b.recommended_memory_gb.cmp(&a.recommended_memory_gb));

    let mut filtered = embed_models;
    filtered.extend(llm_models);

    if filtered.is_empty() {
        let embed = all.iter().find(|m| m.profile == "embed").cloned();
        let minimal = all.iter().find(|m| m.profile == "minimal").cloned();
        let mut fallback = Vec::new();
        if let Some(e) = embed { fallback.push(e); }
        if let Some(m) = minimal { fallback.push(m); }
        fallback
    } else {
        filtered
    }
}

async fn make_ollama_request_with_timeout<T: serde::Serialize, R: serde::de::DeserializeOwned>(
    endpoint: &str,
    method: &str,
    body: Option<T>,
    timeout_secs: u64,
) -> AppResult<R> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(timeout_secs))
        .build()
        .map_err(|e| AppError::ai_service(format!("Failed to build HTTP client: {}", e)))?;

    let url = format!("{}{}", OLLAMA_API_URL, endpoint);

    let request = match method {
        "GET" => client.get(&url),
        "POST" if body.is_some() => client.post(&url).json(&body.unwrap()),
        "POST" => client.post(&url),
        _ => return Err(AppError::ai_service("Unsupported method")),
    };

    let response = request.send().await.map_err(|e| {
        AppError::ai_service(format!(
            "Ollama not reachable: {}. Make sure Ollama is running.",
            e
        ))
    })?;

    if response.status().is_success() {
        response
            .json::<R>()
            .await
            .map_err(|e| AppError::ai_service(format!("Response parsing failed: {}", e)))
    } else {
        let status = response.status();
        let body_text = response.text().await.unwrap_or_default();
        Err(AppError::ai_service(format!(
            "Ollama error: {} {}",
            status, body_text
        )))
    }
}

async fn make_ollama_request<T: serde::Serialize, R: serde::de::DeserializeOwned>(
    endpoint: &str,
    method: &str,
    body: Option<T>,
) -> AppResult<R> {
    make_ollama_request_with_timeout(endpoint, method, body, 120).await
}

#[tauri::command]
pub async fn ai_get_status() -> AppResult<SidecarStatus> {
    match make_ollama_request::<(), serde_json::Value>("/api/version", "GET", None).await {
        Ok(payload) => Ok(SidecarStatus {
            running: true,
            version: payload
                .get("version")
                .and_then(|value| value.as_str())
                .map(|value| value.to_string()),
            error: None,
        }),
        Err(error) => {
            let binary = detect_ollama_binary();
            Ok(SidecarStatus {
                running: false,
                version: read_local_ollama_version(binary.as_deref()),
                error: Some(error.message),
            })
        }
    }
}

#[tauri::command]
pub async fn ai_list_models() -> AppResult<Vec<OllamaModel>> {
    let response: OllamaListResponse = make_ollama_request("/api/tags", "GET", None::<()>).await?;
    Ok(response.models)
}

#[tauri::command]
pub async fn ai_chat_completion(
    app: AppHandle,
    request: ChatCompletionRequest,
) -> AppResult<ChatMessage> {
    let is_stream = request.stream.unwrap_or(false);

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(120))
        .build()
        .map_err(|e| AppError::ai_service(format!("Failed to build HTTP client: {}", e)))?;

    let url = format!("{}/api/chat", OLLAMA_API_URL);

    if !is_stream {
        let response: ChatResponse = make_ollama_request("/api/chat", "POST", Some(request)).await?;
        return response
            .message
            .ok_or_else(|| AppError::ai_service("Empty response from Ollama"));
    }

    let response = client
        .post(&url)
        .json(&request)
        .send()
        .await
        .map_err(|e| AppError::ai_service(format!("Ollama request failed: {}", e)))?;

    let mut full_content = String::new();
    let mut stream = response.bytes_stream();

    while let Some(item) = stream.next().await {
        let chunk = item.map_err(|e| AppError::ai_service(format!("Stream error: {}", e)))?;
        if let Ok(res) = serde_json::from_slice::<ChatResponse>(&chunk) {
            if let Some(msg) = res.message {
                full_content.push_str(&msg.content);
                let _ = app.emit("ai-chat-token", msg.content);
            }
            if res.done {
                break;
            }
        }
    }

    Ok(ChatMessage {
        role: "assistant".to_string(),
        content: full_content,
        images: None,
    })
}

#[tauri::command]
pub async fn ai_pull_model(model: String) -> AppResult<()> {
    let request = OllamaPullRequest {
        model,
        stream: Some(false),
    };

    let _: serde_json::Value =
        make_ollama_request_with_timeout("/api/pull", "POST", Some(request), 3600).await?;
    Ok(())
}

#[tauri::command]
pub async fn ai_get_hardware_info() -> AppResult<HardwareInfo> {
    let mut system = System::new_all();
    system.refresh_all();

    let binary = detect_ollama_binary();
    let status = ai_get_status().await?;
    let total_memory_gb = to_gb(system.total_memory());
    let available_memory_gb = to_gb(system.available_memory());
    let recommended_models = build_recommendations(total_memory_gb);

    Ok(HardwareInfo {
        os: System::name().unwrap_or_else(|| "Unknown".to_string()),
        arch: std::env::consts::ARCH.to_string(),
        cpu_brand: system
            .cpus()
            .first()
            .map(|cpu| cpu.brand().to_string())
            .filter(|brand| !brand.is_empty())
            .unwrap_or_else(|| "Unknown CPU".to_string()),
        cpu_cores: system.cpus().len(),
        total_memory_gb,
        available_memory_gb,
        ollama_installed: binary.is_some(),
        ollama_running: status.running,
        ollama_version: status.version.or_else(|| read_local_ollama_version(binary.as_deref())),
        install_supported: cfg!(target_os = "windows"),
        recommended_model: recommended_models
            .iter()
            .find(|item| item.profile != "embed")
            .map(|item| item.name.clone()),
        recommended_models,
    })
}

#[tauri::command]
pub async fn ai_install_ollama() -> AppResult<OllamaInstallResult> {
    #[cfg(target_os = "windows")]
    {
        // FIX: CREATE_NO_WINDOW — winget не должен мигать терминальным окном
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x08000000;

        let output = Command::new("winget")
            .args([
                "install",
                "--id",
                "Ollama.Ollama",
                "-e",
                "--accept-package-agreements",
                "--accept-source-agreements",
            ])
            .creation_flags(CREATE_NO_WINDOW)
            .output()
            .map_err(|e| {
                AppError::ai_service(format!(
                    "Unable to launch winget for Ollama installation: {}",
                    e
                ))
            })?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(AppError::ai_service(format!(
                "Ollama installation failed: {}",
                stderr.trim()
            )));
        }

        return Ok(OllamaInstallResult {
            installed: true,
            method: "winget".to_string(),
            message: "Ollama installed successfully via winget.".to_string(),
        });
    }

    #[cfg(not(target_os = "windows"))]
    {
        Err(AppError::ai_service(
            "Automatic Ollama installation is currently supported only on Windows.",
        ))
    }
}

/// FIX: ai_start_sidecar — запускаем Ollama БЕЗ видимого консольного окна.
///
/// Старый вариант: `cmd /C start "" ollama serve`
///   → открывает CMD-окно → мигание терминала в EXE-сборке.
///
/// Новый вариант (Windows): `ollama serve` с флагом CREATE_NO_WINDOW.
///   Процесс запускается в фоне, без какого-либо GUI-окна.
///
/// Unix: spawn без терминала (стандартное поведение).
#[tauri::command]
pub async fn ai_start_sidecar() -> AppResult<()> {
    // Если уже запущен — ничего не делаем
    let status = ai_get_status().await?;
    if status.running {
        return Ok(());
    }

    let executable = detect_ollama_binary().unwrap_or_else(|| "ollama".to_string());

    #[cfg(target_os = "windows")]
    {
        use std::os::windows::process::CommandExt;
        // CREATE_NO_WINDOW (0x08000000): процесс не создаёт консольное окно.
        // DETACHED_PROCESS (0x00000008): процесс не привязан к родительской консоли.
        // Оба флага вместе гарантируют полностью скрытый фоновый процесс.
        const CREATE_NO_WINDOW: u32 = 0x08000000;
        const DETACHED_PROCESS: u32 = 0x00000008;

        Command::new(&executable)
            .arg("serve")
            .creation_flags(CREATE_NO_WINDOW | DETACHED_PROCESS)
            .spawn()
            .map_err(|e| AppError::ai_service(format!("Failed to start Ollama: {}", e)))?;

        return Ok(());
    }

    #[cfg(not(target_os = "windows"))]
    {
        Command::new(executable)
            .arg("serve")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .map_err(|e| AppError::ai_service(format!("Failed to start Ollama: {}", e)))?;
        Ok(())
    }
}

#[tauri::command]
pub async fn ai_stop_sidecar() -> AppResult<()> {
    #[cfg(target_os = "windows")]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x08000000;

        let output = Command::new("taskkill")
            .args(["/IM", "ollama.exe", "/F"])
            .creation_flags(CREATE_NO_WINDOW)
            .output()
            .map_err(|e| AppError::ai_service(format!("Failed to stop Ollama: {}", e)))?;

        if output.status.success() {
            return Ok(());
        }

        let stderr = String::from_utf8_lossy(&output.stderr);
        if stderr.to_lowercase().contains("not found") || stderr.contains("не найден") {
            return Ok(());
        }

        return Err(AppError::ai_service(format!(
            "Failed to stop Ollama: {}",
            stderr.trim()
        )));
    }

    #[cfg(not(target_os = "windows"))]
    {
        let output = Command::new("pkill")
            .args(["-f", "ollama"])
            .output()
            .map_err(|e| AppError::ai_service(format!("Failed to stop Ollama: {}", e)))?;

        if output.status.success() {
            return Ok(());
        }

        let stderr = String::from_utf8_lossy(&output.stderr);
        if stderr.trim().is_empty() {
            return Ok(());
        }

        Err(AppError::ai_service(format!(
            "Failed to stop Ollama: {}",
            stderr.trim()
        )))
    }
}

#[tauri::command]
pub async fn ai_get_model_library() -> AppResult<serde_json::Value> {
    let info = ai_get_hardware_info().await?;
    Ok(serde_json::json!({
        "models": info.recommended_models,
        "diagnostics": info,
    }))
}

#[tauri::command]
pub async fn ai_scan_models() -> AppResult<serde_json::Value> {
    let models = ai_list_models().await.unwrap_or_default();
    Ok(serde_json::json!({ "models": models }))
}

#[tauri::command]
pub async fn ai_load_model(_path: String) -> AppResult<()> {
    Ok(())
}

#[tauri::command]
pub async fn ai_unload_model() -> AppResult<()> {
    Ok(())
}

#[tauri::command]
pub async fn ai_send_heartbeat() -> AppResult<()> {
    Ok(())
}
