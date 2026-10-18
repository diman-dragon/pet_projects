use std::{fs, thread, time::Duration};

use chrono::Local;
use reqwest::blocking::Client;
use reqwest::header::{HeaderMap, HeaderValue, ACCEPT, ACCEPT_LANGUAGE, CONTENT_TYPE, ORIGIN, REFERER, USER_AGENT};
use rusqlite::{params, OptionalExtension};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tauri::{AppHandle, Emitter};

use crate::{
    error::{AppError, AppResult},
    models::payloads::{
        ClientDossierOverviewPayload, ClientDossierProvider, ClientDossierSnapshot, RefreshClientDossierPayload,
        RefreshClientDossierRequest,
    },
    repository::{client_dossier_repository, client_repository, case_repository, line_repository, db},
    services::{path, naming},
};

#[derive(Debug, Deserialize)]
struct EgrulSearchResponse {
    rows: Option<Vec<EgrulRow>>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
struct EgrulOkved {
    #[serde(default)]
    kd: String,
    #[serde(default)]
    nm: String,
    #[serde(default, rename = "pr")]
    primary: String,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
struct EgrulRow {
    #[serde(default)]
    c: String,
    #[serde(default)]
    g: String,
    #[serde(default)]
    cnt: String,
    #[serde(default)]
    i: String,
    #[serde(default)]
    k: String,
    #[serde(default)]
    n: String,
    #[serde(default)]
    o: String,
    #[serde(default)]
    p: String,
    #[serde(default)]
    r: String,
    #[serde(default)]
    t: String,
    #[serde(default)]
    pg: String,
    #[serde(default)]
    tot: String,
    #[serde(default)]
    rn: String,
    #[serde(default)]
    e: String,
    #[serde(default)]
    a: String,
    #[serde(default)]
    d: String,
    #[serde(default)]
    f: String,
    #[serde(default)]
    ul: Vec<EgrulOkved>,
}

struct EgrulPackage {
    row: EgrulRow,
    raw_payload: Value,
    pdf_bytes: Option<Vec<u8>>,
    pdf_error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct KadCaseSummary {
    case_number: String,
    court: String,
    date: String,
    case_type: String,
    plaintiffs: Vec<String>,
    respondents: Vec<String>,
    card_url: String,
}

struct KadPackage {
    cases: Vec<KadCaseSummary>,
    raw_payload: Value,
    error: Option<String>,
}

pub fn get_client_dossier(app: &AppHandle, client_id: i64) -> AppResult<ClientDossierOverviewPayload> {
    let conn = db::open_clients_db(app)?;
    let snapshots = client_dossier_repository::list_by_client(&conn, client_id)?
        .into_iter()
        .map(map_snapshot)
        .collect::<Vec<_>>();

    let has_snapshots = !snapshots.is_empty();
    Ok(ClientDossierOverviewPayload {
        providers: build_provider_catalog(
            has_snapshots,
            if has_snapshots { "ready" } else { "configured" },
            if has_snapshots { "ready" } else { "configured" },
        ),
        snapshots,
    })
}

pub fn refresh_client_dossier(app: &AppHandle, request: RefreshClientDossierRequest) -> AppResult<RefreshClientDossierPayload> {
    let conn = db::open_clients_db(app)?;
    let client = client_repository::get_by_id(&conn, request.client_id)?
        .ok_or_else(|| AppError::not_found("client"))?;

    let tax_id = request
        .tax_id
        .clone()
        .or(client.tax_id.clone())
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .ok_or_else(|| AppError::validation("client.error.taxIdRequired", "tax_id is required"))?;

    let app_clone = app.clone();
    let client_id = request.client_id;
    let tax_id_clone = tax_id.clone();

    std::thread::spawn(move || {
        let res = (|| -> AppResult<()> {
            let conn = db::open_clients_db(&app_clone)?;
            let client = client_repository::get_by_id(&conn, client_id)?
                .ok_or_else(|| AppError::not_found("client"))?;

            let dossier_root = db::work_dir(&app_clone)?
                .join(path::CLIENTS_DIR)
                .join(&client.folder_name)
                .join(path::CLIENT_BASE_DIR)
                .join("Client_Dossier");

            if dossier_root.exists() {
                let _ = fs::remove_dir_all(&dossier_root);
            }
            fs::create_dir_all(&dossier_root)?;

            let egrul_result = fetch_egrul_package(&tax_id_clone)?;

            let kad_search_id = if !egrul_result.row.i.is_empty() {
                egrul_result.row.i.clone()
            } else {
                tax_id_clone.clone()
            };

            let ogrn = if egrul_result.row.o.is_empty() { None } else { Some(egrul_result.row.o.as_str()) };
            let kad_result = fetch_kad_package(&kad_search_id, ogrn).unwrap_or_else(|err| KadPackage {
                cases: Vec::new(),
                raw_payload: json!({ "error": err.to_string() }),
                error: Some(err.to_string()),
            });

            let date_str = Local::now().format("%y%m%d").to_string();
            let extract_filename = format!("{}_{}.pdf", date_str, tax_id_clone);
            let extract_pdf_path = dossier_root.join(&extract_filename);
            let search_json_path = dossier_root.join("egrul-search.json");
            let kad_json_path = dossier_root.join("kad-arbitr-search.json");
            let manifest_path = dossier_root.join("dossier-manifest.json");
            let summary_path = dossier_root.join("README.md");

            fs::write(&search_json_path, serde_json::to_vec_pretty(&egrul_result.raw_payload)?)?;
            if let Some(pdf_bytes) = &egrul_result.pdf_bytes {
                fs::write(&extract_pdf_path, pdf_bytes)?;
            }
            fs::write(&kad_json_path, serde_json::to_vec_pretty(&kad_result.raw_payload)?)?;

            let summary = build_summary(
                &client.full_name,
                &tax_id_clone,
                &egrul_result.row,
                egrul_result.pdf_bytes.as_ref().map(|_| extract_pdf_path.to_string_lossy().to_string()).as_deref(),
                egrul_result.pdf_error.as_deref(),
                &kad_result,
            );
            let client_data = build_client_data_text(
                &egrul_result.row,
                egrul_result.pdf_bytes.as_ref().map(|_| extract_pdf_path.to_string_lossy().to_string()).as_deref(),
                egrul_result.pdf_error.as_deref(),
                &kad_result,
            );

            let manifest_json = json!({
                "client": {
                    "id": client.id,
                    "full_name": client.full_name,
                    "folder_name": client.folder_name,
                    "tax_id": tax_id_clone,
                },
                "generated_at": Local::now().to_rfc3339(),
                "provider": "combined",
                "providers": {
                    "egrul": {
                        "short_name": egrul_result.row.c,
                        "full_name": egrul_result.row.n,
                        "inn": if egrul_result.row.i.is_empty() { json!(null) } else { json!(egrul_result.row.i) },
                        "director": egrul_result.row.g,
                        "ogrn": egrul_result.row.o,
                        "kpp": egrul_result.row.p,
                        "registration_date": if egrul_result.row.r.is_empty() { json!(null) } else { json!(normalize_egrul_date(&egrul_result.row.r)) },
                        "registration_date_pre2002": if egrul_result.row.d.is_empty() { json!(null) } else { json!(normalize_egrul_date(&egrul_result.row.d)) },
                        "region": egrul_result.row.rn,
                        "entity_kind": egrul_result.row.k,
                        "status": egrul_result.row.f,
                        "email": if egrul_result.row.e.is_empty() { json!(null) } else { json!(egrul_result.row.e) },
                        "registration_address": if egrul_result.row.a.is_empty() { json!(null) } else { json!(egrul_result.row.a) },
                        "okved": egrul_result.row.ul.iter().map(|v| json!({
                            "code": v.kd,
                            "description": v.nm,
                            "primary": v.primary == "1",
                        })).collect::<Vec<_>>(),
                        "pdf_path": egrul_result.pdf_bytes.as_ref().map(|_| extract_pdf_path.to_string_lossy().to_string()),
                        "pdf_error": egrul_result.pdf_error,
                        "search_json_path": search_json_path.to_string_lossy(),
                    },
                    "kad_arbitr": {
                        "search_json_path": kad_json_path.to_string_lossy(),
                        "case_count": kad_result.cases.len(),
                        "cases": kad_result.cases,
                        "error": kad_result.error.as_ref().map(|err| {
                            if err.contains("Cloudflare") || err.contains("cloudflare") || err.contains("403") {
                                json!({
                                    "key": "errors.kad.cloudflare",
                                    "message": err,
                                    "securityText": "КАД Арбитр временно недоступен (защита Cloudflare)"
                                })
                            } else {
                                json!(err)
                            }
                        }),
                    }
                }
            });

            fs::write(&manifest_path, serde_json::to_vec_pretty(&manifest_json)?)?;
            fs::write(&summary_path, summary.as_bytes())?;

            client_repository::save_dossier_enrichment(
                &conn,
                client.id,
                &tax_id_clone,
                &client_data,
                &manifest_json.to_string(),
            )?;

            let egrul_status = if egrul_result.pdf_error.is_some() { "degraded" } else if egrul_result.pdf_bytes.is_none() { "degraded" } else { "ready" };
            let kad_status = if kad_result.error.is_some() { "degraded" } else if kad_result.cases.is_empty() { "degraded" } else { "ready" };
            let providers = build_provider_catalog(true, egrul_status, kad_status);
            let providers_json = serde_json::to_string(&providers)?;
            let snapshot_id = client_dossier_repository::insert_snapshot(
                &conn,
                client.id,
                &tax_id_clone,
                "ready",
                &providers_json,
                &summary,
                &dossier_root.to_string_lossy(),
                &manifest_path.to_string_lossy(),
                &manifest_json.to_string(),
            )?;

            sync_kad_cases_to_lines(&app_clone, client.id, &kad_result.cases)?;

            let _ = app_clone.emit("dossier-refreshed", json!({ "client_id": client_id, "snapshot_id": snapshot_id }));
            Ok(())
        })();

        if let Err(err) = res {
            eprintln!("[dossier] Background refresh failed: {err:?}");
            let _ = app_clone.emit("dossier-refresh-error", json!({ "client_id": client_id, "error": err.to_string() }));
        }
    });

    let has_snapshots = !client_dossier_repository::list_by_client(&conn, client_id)?.is_empty();
    Ok(RefreshClientDossierPayload {
        snapshot: ClientDossierSnapshot {
            id: 0,
            client_id,
            tax_id,
            status: "pending".to_string(),
            providers_json: "[]".to_string(),
            summary_text: Some("Загрузка данных...".to_string()),
            folder_path: "".to_string(),
            manifest_path: "".to_string(),
            dossier_json: Some("{}".to_string()),
            created_at: Some(Local::now().to_rfc3339()),
            updated_at: Some(Local::now().to_rfc3339()),
        },
        providers: build_provider_catalog(has_snapshots, "pending", "pending"),
    })
}

fn sync_kad_cases_to_lines(app: &AppHandle, client_id: i64, kad_cases: &[KadCaseSummary]) -> AppResult<()> {
    if kad_cases.is_empty() {
        return Ok(());
    }

    let conn = db::open_clients_db(app)?;

    let mut stmt = conn.prepare("SELECT id, folder_name FROM cases WHERE client_id = ?1 AND title = 'Арбитражные дела' AND deleted_at IS NULL LIMIT 1")?;
    let case_info: Option<(i64, String)> = stmt.query_row(params![client_id], |row| Ok((row.get(0)?, row.get(1)?))).optional()?;

    let (case_id, case_folder) = match case_info {
        Some(info) => info,
        None => {
            let folder_name = "Arbitration_Cases";
            let id = case_repository::insert(
                &conn,
                client_id,
                "Арбитражные дела",
                "ACTIVE",
                Some("litigation"),
                None,
                None,
                None,
                None,
                folder_name,
                None,
                None,
                None,
                None,
                None,
            )?;

            let client_folder = client_repository::get_folder_name(&conn, client_id)?
                .ok_or_else(|| AppError::not_found("client"))?;
            fs::create_dir_all(path::case_path(db::work_dir(app)?, &client_folder, folder_name))?;

            (id, folder_name.to_string())
        }
    };

    for (idx, kad_case) in kad_cases.iter().enumerate() {
        let title = format!("{} | {}", kad_case.case_number, kad_case.court);

        let exists: bool = conn.query_row(
            "SELECT EXISTS(SELECT 1 FROM lines WHERE case_id = ?1 AND title = ?2 AND deleted_at IS NULL)",
            params![case_id, title],
            |row| row.get(0),
        )?;

        if !exists {
            let folder_name = naming::line_folder_name(&title, (idx + 1) as i64);
            line_repository::insert(
                &conn,
                case_id,
                &title,
                None,
                &folder_name,
                Some(&kad_case.card_url),
                None,
                None,
                None,
                None,
            )?;

            let client_folder = client_repository::get_folder_name(&conn, client_id)?
                .ok_or_else(|| AppError::not_found("client"))?;
            fs::create_dir_all(path::line_path(
                db::work_dir(app)?,
                &client_folder,
                &case_folder,
                &folder_name,
                Some("litigation"),
            ))?;
        }
    }

    Ok(())
}

fn build_egrul_client() -> AppResult<Client> {
    let mut headers = HeaderMap::new();
    headers.insert(USER_AGENT, HeaderValue::from_static("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"));
    headers.insert(ACCEPT, HeaderValue::from_static("application/json, text/plain, */*"));
    headers.insert(ACCEPT_LANGUAGE, HeaderValue::from_static("ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7"));
    headers.insert(ORIGIN, HeaderValue::from_static("https://egrul.nalog.ru"));
    headers.insert(REFERER, HeaderValue::from_static("https://egrul.nalog.ru/index.html"));

    Client::builder()
        .default_headers(headers)
        .cookie_store(true)
        .timeout(Duration::from_secs(30))
        .build()
        .map_err(|err| AppError::new("errors.ai.transport", err.to_string()))
}

fn fetch_egrul_package(tax_id: &str) -> AppResult<EgrulPackage> {
    let client = build_egrul_client()?;

    let _ = client.get("https://egrul.nalog.ru/index.html").send();
    thread::sleep(Duration::from_millis(500));

    let token_response = client
        .post("https://egrul.nalog.ru/")
        .form(&[
            ("vyp3CaptchaToken", ""),
            ("page", ""),
            ("query", tax_id),
            ("region", ""),
            ("PreventChromeAutocomplete", ""),
        ])
        .send()
        .map_err(|err| AppError::new("errors.ai.unavailable", err.to_string()))?;

    if !token_response.status().is_success() {
        return Err(AppError::new(
            "errors.ai.requestFailed",
            format!("ФНС поиск вернул {}", token_response.status()),
        ));
    }

    let token_payload: Value = token_response
        .json()
        .map_err(|err| AppError::new("errors.ai.invalidResponse", err.to_string()))?;

    let token = token_payload
        .get("t")
        .and_then(Value::as_str)
        .ok_or_else(|| AppError::new("errors.ai.invalidResponse", "ФНС не вернул токен поиска"))?;

    let (raw_payload, row) = poll_search_result(&client, token)?;

    let _ = client.get(format!("https://egrul.nalog.ru/vyp-request/{}", row.t)).send();
    thread::sleep(Duration::from_millis(1500));

    let (pdf_bytes, pdf_error) = match download_extract_pdf(&client, &row.t) {
        Ok(bytes) => (Some(bytes), None),
        Err(err) => (None, Some(err.to_string())),
    };

    Ok(EgrulPackage {
        row,
        raw_payload,
        pdf_bytes,
        pdf_error,
    })
}

fn poll_search_result(client: &Client, token: &str) -> AppResult<(Value, EgrulRow)> {
    let mut last_payload = None;

    for attempt in 0..15 {
        let response = client
            .get(format!("https://egrul.nalog.ru/search-result/{token}"))
            .send()
            .map_err(|err| AppError::new("errors.ai.unavailable", err.to_string()))?;

        if !response.status().is_success() {
            return Err(AppError::new(
                "errors.ai.requestFailed",
                format!("ФНС search-result вернул {}", response.status()),
            ));
        }

        let raw_payload: Value = response
            .json()
            .map_err(|err| AppError::new("errors.ai.invalidResponse", err.to_string()))?;
        let parsed: EgrulSearchResponse = serde_json::from_value(raw_payload.clone())
            .map_err(|err| AppError::new("errors.ai.invalidResponse", err.to_string()))?;

        if let Some(row) = parsed.rows.and_then(|rows| rows.into_iter().next()) {
            return Ok((raw_payload, row));
        }

        last_payload = Some(raw_payload);
        if attempt < 14 {
            thread::sleep(Duration::from_millis(1000));
        }
    }

    let details = last_payload
        .map(|value| value.to_string())
        .unwrap_or_else(|| "search-result returned no payload".to_string());
    Err(AppError::new(
        "errors.common.notFound",
        format!("По этому ИНН ничего не найдено в ЕГРЮЛ. Последний ответ: {details}"),
    ))
}

fn download_extract_pdf(client: &Client, token: &str) -> AppResult<Vec<u8>> {
    let mut last_error = None;

    for attempt in 0..5 {
        let response = client
            .get(format!("https://egrul.nalog.ru/vyp-download/{token}"))
            .send();

        match response {
            Ok(response) => {
                if !response.status().is_success() {
                    last_error = Some(format!("ФНС выписка вернула {}", response.status()));
                } else {
                    match response.bytes() {
                        Ok(buffer) if !buffer.is_empty() => return Ok(buffer.to_vec()),
                        Ok(_) => last_error = Some("ФНС вернула пустую PDF-выписку".to_string()),
                        Err(err) => last_error = Some(err.to_string()),
                    }
                }
            }
            Err(err) => {
                last_error = Some(err.to_string());
            }
        }

        if attempt < 4 {
            thread::sleep(Duration::from_millis(1200));
        }
    }

    Err(AppError::new(
        "errors.ai.requestFailed",
        format!(
            "Не удалось скачать PDF-выписку ФНС: {}",
            last_error.unwrap_or_else(|| "unknown error".to_string())
        ),
    ))
}

fn extract_kad_sides(row: &Value, field: &str) -> Vec<String> {
    row.get(field)
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(|side| side.get("Name").and_then(Value::as_str))
                .filter(|s| !s.is_empty())
                .map(|s| s.to_string())
                .collect()
        })
        .unwrap_or_default()
}

fn normalize_egrul_date(raw: &str) -> String {
    let parts: Vec<&str> = raw.trim().split('.').collect();
    if parts.len() == 3 && parts[0].len() == 2 && parts[1].len() == 2 && parts[2].len() == 4 {
        return format!("{}-{}-{}", parts[2], parts[1], parts[0]);
    }
    raw.to_string()
}

fn parse_dotnet_date(raw: &str) -> String {
    let inner = raw.trim_start_matches("/Date(").trim_end_matches(")/");
    let ms_str = inner.split('+').next().unwrap_or(inner).split('-').next().unwrap_or(inner);
    if let Ok(ms) = ms_str.parse::<i64>() {
        let secs = ms / 1000;
        if let Some(dt) = chrono::DateTime::<chrono::Utc>::from_timestamp_secs(secs) {
            return dt.format("%Y-%m-%d").to_string();
        }
    }
    raw.to_string()
}

fn build_kad_client() -> AppResult<Client> {
    let mut headers = HeaderMap::new();
    headers.insert(USER_AGENT, HeaderValue::from_static("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"));
    headers.insert(ACCEPT, HeaderValue::from_static("application/json, text/javascript, */*; q=0.01"));
    headers.insert(ACCEPT_LANGUAGE, HeaderValue::from_static("ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7"));
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json; charset=UTF-8"));
    headers.insert(ORIGIN, HeaderValue::from_static("https://kad.arbitr.ru"));
    headers.insert(REFERER, HeaderValue::from_static("https://kad.arbitr.ru/"));
    // FIX: обязательный хедер для API КАД Арбитр
    headers.insert("X-Requested-With", HeaderValue::from_static("XMLHttpRequest"));
    // FIX: требуется для корректной сериализации дат в ISO формате
    headers.insert("x-date-format", HeaderValue::from_static("iso"));
    headers.insert("sec-ch-ua", HeaderValue::from_static("\"Not_A Brand\";v=\"8\", \"Chromium\";v=\"120\", \"Google Chrome\";v=\"120\""));
    headers.insert("sec-ch-ua-mobile", HeaderValue::from_static("?0"));
    headers.insert("sec-ch-ua-platform", HeaderValue::from_static("\"Windows\""));
    headers.insert("sec-fetch-site", HeaderValue::from_static("same-origin"));
    headers.insert("sec-fetch-mode", HeaderValue::from_static("cors"));
    headers.insert("sec-fetch-dest", HeaderValue::from_static("empty"));

    Client::builder()
        .default_headers(headers)
        .cookie_store(true)
        .timeout(Duration::from_secs(30))
        .build()
        .map_err(|err| AppError::new("errors.ai.transport", err.to_string()))
}

fn fetch_kad_package(tax_id: &str, ogrn: Option<&str>) -> AppResult<KadPackage> {
    let client = build_kad_client()?;

    let normalized_tax_id = tax_id.trim().replace(" ", "").to_string();

    // Первый GET для инициализации cookies (может не дать cf_clearance, но помогает с сессионными)
    let _ = client.get("https://kad.arbitr.ru/").send();
    thread::sleep(Duration::from_millis(800));

    // FIX БАГ 1: правильный endpoint — /Kad/SearchInstances (не /Result/Search)
    // FIX БАГ 1: ИНН передаётся в поле "Name", не "Inn"
    // FIX БАГ 1: убран несуществующий ключ "WithDocuments", добавлен "WithVKSInstances"
    // FIX БАГ 1: Type = -1 (любая сторона), ExactMatch = false
    let payload = json!({
        "Page": 1,
        "Count": 50,
        "Courts":   [],
        "DateFrom": null,
        "DateTo":   null,
        "Sides": [
            {
                "Name":       normalized_tax_id,
                "Type":       -1,
                "ExactMatch": false
            }
        ],
        "Judges":           [],
        "CaseNumbers":      [],
        "WithVKSInstances": false
    });

    let max_retries = 3;
    for attempt in 0..max_retries {
        // FIX БАГ 1: правильный endpoint
        let response = client
            .post("https://kad.arbitr.ru/Kad/SearchInstances")
            .json(&payload)
            .send();

        match response {
            Ok(res) => {
                match res.status().as_u16() {
                    200..=299 => {
                        let raw: Value = res.json().unwrap_or(json!({}));
                        let mut cases = Vec::new();

                        if let Some(rows) = raw.get("Result")
                            .and_then(|r| r.get("Items"))
                            .and_then(Value::as_array)
                        {
                            for row in rows {
                                let case_number = row.get("CaseNumber").and_then(Value::as_str);
                                let id = row.get("Id").and_then(Value::as_str);

                                let court = row.get("Court").and_then(|c| {
                                    c.get("ShortName").and_then(Value::as_str)
                                        .or_else(|| c.get("Name").and_then(Value::as_str))
                                        .or_else(|| c.as_str())
                                });

                                if let (Some(num), Some(id)) = (case_number, id) {
                                    let court_name = court.unwrap_or("Арбитражный суд").to_string();
                                    let date = parse_dotnet_date(
                                        row.get("RegistrationDate").and_then(Value::as_str).unwrap_or("")
                                    );
                                    let case_type = row.get("CaseType")
                                        .and_then(Value::as_str)
                                        .unwrap_or("")
                                        .to_string();
                                    let plaintiffs  = extract_kad_sides(row, "Plaintiffs");
                                    let respondents = extract_kad_sides(row, "Respondents");

                                    cases.push(KadCaseSummary {
                                        case_number: num.to_string(),
                                        court: court_name,
                                        date,
                                        case_type,
                                        plaintiffs,
                                        respondents,
                                        card_url: format!("https://kad.arbitr.ru/Card/{}", id),
                                    });
                                }
                            }
                        }

                        // Если ИНН не дал результатов — повтор по ОГРН
                        if cases.is_empty() {
                            if let Some(ogrn_val) = ogrn {
                                let ogrn_clean = ogrn_val.trim().replace(" ", "");
                                if !ogrn_clean.is_empty() && ogrn_clean != normalized_tax_id {
                                    eprintln!("[kad] ИНН поиск пустой, повтор по ОГРН: {}", ogrn_clean);
                                    thread::sleep(Duration::from_millis(600));
                                    // FIX: тот же правильный endpoint и формат
                                    let ogrn_payload = json!({
                                        "Page": 1,
                                        "Count": 50,
                                        "Courts":   [],
                                        "DateFrom": null,
                                        "DateTo":   null,
                                        "Sides": [{ "Name": ogrn_clean, "Type": -1, "ExactMatch": false }],
                                        "Judges":           [],
                                        "CaseNumbers":      [],
                                        "WithVKSInstances": false
                                    });
                                    // FIX БАГ 1: правильный endpoint в retry
                                    if let Ok(ogrn_res) = client
                                        .post("https://kad.arbitr.ru/Kad/SearchInstances")
                                        .json(&ogrn_payload)
                                        .send()
                                    {
                                        if ogrn_res.status().is_success() {
                                            if let Ok(ogrn_raw) = ogrn_res.json::<Value>() {
                                                let mut ogrn_cases = Vec::new();
                                                if let Some(rows) = ogrn_raw.get("Result")
                                                    .and_then(|r| r.get("Items"))
                                                    .and_then(Value::as_array)
                                                {
                                                    for row in rows {
                                                        let case_number = row.get("CaseNumber").and_then(Value::as_str);
                                                        let id = row.get("Id").and_then(Value::as_str);
                                                        let court = row.get("Court").and_then(|c| {
                                                            c.get("ShortName").and_then(Value::as_str)
                                                                .or_else(|| c.get("Name").and_then(Value::as_str))
                                                                .or_else(|| c.as_str())
                                                        });
                                                        if let (Some(num), Some(id)) = (case_number, id) {
                                                            ogrn_cases.push(KadCaseSummary {
                                                                case_number: num.to_string(),
                                                                court: court.unwrap_or("Арбитражный суд").to_string(),
                                                                date: parse_dotnet_date(row.get("RegistrationDate").and_then(Value::as_str).unwrap_or("")),
                                                                case_type: row.get("CaseType").and_then(Value::as_str).unwrap_or("").to_string(),
                                                                plaintiffs:  extract_kad_sides(row, "Plaintiffs"),
                                                                respondents: extract_kad_sides(row, "Respondents"),
                                                                card_url: format!("https://kad.arbitr.ru/Card/{}", id),
                                                            });
                                                        }
                                                    }
                                                }
                                                if !ogrn_cases.is_empty() {
                                                    return Ok(KadPackage { cases: ogrn_cases, raw_payload: ogrn_raw, error: None });
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        return Ok(KadPackage { cases, raw_payload: raw, error: None });
                    }

                    403 | 429 => {
                        if attempt < max_retries - 1 {
                            let backoff_ms = 1000 * (1 << attempt);
                            eprintln!("[kad] статус {}, повтор через {}мс ({}/{})",
                                res.status(), backoff_ms, attempt + 1, max_retries);
                            thread::sleep(Duration::from_millis(backoff_ms));
                            continue;
                        } else {
                            let status = res.status();
                            return Ok(KadPackage {
                                cases: Vec::new(),
                                raw_payload: json!({}),
                                error: Some(format!(
                                    "КАД Арбитр заблокировал запрос ({status}). \
                                    Cloudflare требует выполнения JavaScript (cookie pr_fp/wasm). \
                                    reqwest не может обойти эту защиту автоматически. \
                                    Используйте кнопку «Загрузить вручную» или откройте КАД в браузере."
                                )),
                            });
                        }
                    }

                    other => {
                        let body_text = res.text().unwrap_or_default();
                        return Ok(KadPackage {
                            cases: Vec::new(),
                            raw_payload: json!({}),
                            error: Some(format!(
                                "КАД Арбитр вернул {} при поиске по ИНН {}. Ответ: {}",
                                other, normalized_tax_id,
                                if body_text.len() > 200 { &body_text[..200] } else { &body_text }
                            )),
                        });
                    }
                }
            }

            Err(err) => {
                if attempt < max_retries - 1 {
                    let backoff_ms = 1000 * (1 << attempt);
                    eprintln!("[kad] сетевая ошибка, повтор через {}мс: {}", backoff_ms, err);
                    thread::sleep(Duration::from_millis(backoff_ms));
                    continue;
                } else {
                    return Ok(KadPackage {
                        cases: Vec::new(),
                        raw_payload: json!({}),
                        error: Some(format!(
                            "КАД Арбитр или сеть недоступны: {}. Проверьте интернет-соединение.",
                            err
                        )),
                    });
                }
            }
        }
    }

    Ok(KadPackage {
        cases: Vec::new(),
        raw_payload: json!({}),
        error: Some("КАД Арбитр недоступен после нескольких попыток.".to_string()),
    })
}

fn build_provider_catalog(has_snapshot: bool, egrul_status: &str, kad_status: &str) -> Vec<ClientDossierProvider> {
    vec![
        ClientDossierProvider {
            id: "egrul".to_string(),
            name: "ЕГРЮЛ / ЕГРИП".to_string(),
            description: "Официальный поиск по ИНН, извлечение реквизитов и сохранение PDF-выписки ФНС в папку клиента.".to_string(),
            status: if has_snapshot { egrul_status.to_string() } else { "configured".to_string() },
            enabled: true,
        },
        ClientDossierProvider {
            id: "kad_arbitr".to_string(),
            name: "КАД Арбитр".to_string(),
            description: "Поиск арбитражных дел по ИНН/ОГРН с сохранением JSON-ответа и карточек дел в автодосье.".to_string(),
            status: kad_status.to_string(),
            enabled: true,
        },
    ]
}

fn build_summary(
    client_name: &str,
    tax_id: &str,
    row: &EgrulRow,
    pdf_path: Option<&str>,
    pdf_error: Option<&str>,
    kad_result: &KadPackage,
) -> String {
    let egrul_pdf_block = if let Some(path) = pdf_path {
        format!("PDF-выписка сохранена: {}\n", path)
    } else if let Some(error) = pdf_error {
        format!("PDF-выписка ФНС не получена: {}\n", error)
    } else {
        "PDF-выписка ФНС не получена.\n".to_string()
    };
    let kad_block = if let Some(error) = &kad_result.error {
        format!("\nКАД Арбитр: временно недоступен или вернул нестандартный ответ.\nПричина: {error}\n")
    } else if kad_result.cases.is_empty() {
        "\nКАД Арбитр: арбитражные дела по текущему запросу не найдены.\n".to_string()
    } else {
        let preview = kad_result
            .cases
            .iter()
            .take(5)
            .map(|c| format!("- {} | {} | {}", c.case_number, c.court, c.card_url))
            .collect::<Vec<_>>()
            .join("\n");
        format!("\nКАД Арбитр: найдено дел — {}.\n{}\n", kad_result.cases.len(), preview)
    };

    format!(
        "# Автодосье клиента\n\nКлиент: {client_name}\nИНН: {tax_id}\n\nИсточник 1: официальный сервис ФНС ЕГРЮЛ/ЕГРИП\nПолное наименование: {}\nКраткое наименование: {}\nОГРН: {}\nКПП: {}\nРегион: {}\nДата регистрации: {}\nРуководитель: {}\n\n{}{}",
        row.n, row.c, row.o, row.p, row.rn, normalize_egrul_date(&row.r), row.g, egrul_pdf_block, kad_block
    )
}

fn build_client_data_text(row: &EgrulRow, pdf_path: Option<&str>, pdf_error: Option<&str>, kad_result: &KadPackage) -> String {
    let kad_line = if kad_result.error.is_some() {
        "КАД Арбитр: недоступен при последней проверке.".to_string()
    } else {
        format!("КАД Арбитр: найдено дел {}", kad_result.cases.len())
    };
    let pdf_line = if let Some(path) = pdf_path {
        format!("Выписка PDF: {}", path)
    } else if let Some(error) = pdf_error {
        format!("Выписка PDF: ошибка получения ({})", error)
    } else {
        "Выписка PDF: не получена".to_string()
    };

    format!(
        "ЕГРЮЛ\n\nПолное наименование: {}\nКраткое наименование: {}\nИНН: {}\nОГРН: {}\nКПП: {}\nРегион: {}\nДата регистрации: {}\nРуководитель: {}\n{}\n{}\n",
        row.n, row.c, row.i, row.o, row.p, row.rn, row.r, row.g, pdf_line, kad_line
    )
}

fn map_snapshot(record: client_dossier_repository::ClientDossierSnapshotRecord) -> ClientDossierSnapshot {
    ClientDossierSnapshot {
        id: record.id,
        client_id: record.client_id,
        tax_id: record.tax_id,
        status: record.status,
        providers_json: record.providers_json,
        summary_text: record.summary_text,
        folder_path: record.folder_path,
        manifest_path: record.manifest_path,
        dossier_json: record.dossier_json,
        created_at: record.created_at,
        updated_at: record.updated_at,
    }
}
