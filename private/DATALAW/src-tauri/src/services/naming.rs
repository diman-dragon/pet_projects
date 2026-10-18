
use chrono::{Local, NaiveDate};
use crate::services::path;

pub fn to_standard_name(text: &str) -> String {
    let normalized = text.chars().map(map_char).collect::<String>();
    normalized
        .split('_')
        .filter(|segment| !segment.is_empty())
        .collect::<Vec<_>>()
        .join("_")
        .to_uppercase()
}

pub fn to_standard_name_preserved(text: &str) -> String {
    let normalized = text.chars().map(map_char_preserved).collect::<String>();
    normalized
        .split('_')
        .filter(|segment| !segment.is_empty())
        .collect::<Vec<_>>()
        .join("_")
}

pub fn normalize_status(status: &str) -> String {
    status
        .split([' ', '_'])
        .next()
        .unwrap_or("ACTIVE")
        .trim()
        .to_uppercase()
}

pub fn case_status_code(status: &str) -> String {
    match status.to_uppercase().as_str() {
        "LEAD" => "LEAD".to_string(),
        "ACTIVE" | "ACT" => "ACT".to_string(),
        "PAUSED" | "PAUSE" => "PAUSE".to_string(),
        "ARCHIVE" | "ARC" => "ARC".to_string(),
        _ => "ACT".to_string(),
    }
}

pub fn client_status_code(status: &str) -> String {
    case_status_code(status)
}

pub fn client_folder_name(full_name: &str, client_type: &str, status: &str, sequence: i64) -> String {
    let type_code = match client_type.to_uppercase().as_str() {
        "IP" | "INDIVIDUAL" => "IP",
        "LE" | "LEGAL" => "LE",
        "NO" | "NGO" => "NO",
        _ => "LE",
    };
    
    let yymm = chrono::Local::now().format("%y%m").to_string();
    let nn = format!("{:02}", sequence + 1);
    
    let name = to_standard_name_preserved(full_name);
    let status_code = client_status_code(status);
    
    format!("{}_{}_{}_{}_{}", type_code, yymm, nn, name, status_code)
}

pub fn rename_client_folder(current: &str, status: &str) -> String {
    let mut parts = current.split('_').map(str::to_string).collect::<Vec<_>>();
    if parts.len() >= 5 {
        // TYPE_YYMM_NN_Name_STATUS
        // STATUS is always the last part.
        let last_index = parts.len() - 1;
        parts[last_index] = client_status_code(status);
        return parts.join("_");
    }
    current.to_string()
}

pub fn case_folder_name(title: &str, status: &str) -> String {
    format!("{}_{}", to_standard_name(title), normalize_status(status))
}

pub fn other_folder_name(date_str: &str, sequence: i64, status: &str) -> String {
    let status_code = case_status_code(status);
    format!("{}_{:02}_{}", date_str, sequence + 1, status_code)
}

pub fn consulting_folder_name(date_str: &str, sequence: i64, status: &str) -> String {
    other_folder_name(date_str, sequence, status)
}

pub fn contract_folder_name(counterparty: &str, date_str: &str, num_str: &str, status: &str) -> String {
    let kont = to_standard_name(counterparty);
    let kont_clean = if kont.is_empty() { "BEZ_KONT".to_string() } else { kont };
    let status_code = case_status_code(status);
    
    if num_str.is_empty() {
        format!("{}/{}/{}_{}", path::CONTRACTS_DIR, kont_clean, date_str, status_code)
    } else {
        format!("{}/{}/{}_{}_{}", path::CONTRACTS_DIR, kont_clean, date_str, to_standard_name(num_str), status_code)
    }
}

pub fn line_folder_name(title: &str, position_index: i64) -> String {
    format!("{:02}_{}", position_index, to_standard_name(title))
}

pub fn appeal_folder_name(
    title: &str,
    agency_code: Option<&str>,
    status: &str,
    _sequence: i64,
    case_type: Option<&str>,
    appeal_date: Option<&str>,
    incoming_number: Option<&str>,
) -> String {
    let status_norm = normalize_status(status);

    if case_type == Some("other") {
        let parsed_date = appeal_date
            .and_then(|date| NaiveDate::parse_from_str(date, "%Y-%m-%d").ok())
            .unwrap_or_else(|| Local::now().date_naive());
        let date_block = parsed_date.format("%d%m%y").to_string();
        return format!("{}_{}", date_block, status_norm);
    }

    // For litigation cases: ГГММ_КОД_НОМЕРВХОДЯЩИЙ_ОПИСАНИЕ_СТАТУС
    if case_type == Some("litigation") {
        let parsed_date = appeal_date
            .and_then(|date| NaiveDate::parse_from_str(date, "%Y-%m-%d").ok())
            .unwrap_or_else(|| Local::now().date_naive());
        let yymm = parsed_date.format("%y%m").to_string();
        // agency_code is the agency short code (e.g., "СКРу", "ПРОК")
        let agency_short_clean = agency_code
            .map(|value| to_standard_name(value).chars().take(4).collect::<String>())
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| "UNK".to_string());
        // incoming_number is the incoming number from the agency
        let incoming_num = incoming_number.unwrap_or("PENDING");
        let incoming_num_clean = if incoming_num.is_empty() { "PENDING" } else { incoming_num };
        return format!("{}_{}_{}_{}_{}", yymm, agency_short_clean, incoming_num_clean, to_standard_name(title), status_norm);
    }

    let agency = to_standard_name(agency_code.unwrap_or(""));
    if agency.is_empty() {
        format!(
            "{}_{}",
            to_standard_name(title),
            status_norm
        )
    } else {
        format!(
            "{}_{}_{}",
            agency,
            to_standard_name(title),
            status_norm
        )
    }
}

fn map_char(ch: char) -> String {
    map_char_preserved(ch).to_uppercase()
}

fn map_char_preserved(ch: char) -> String {
    let mapped = match ch {
        'а' | 'А' => "A",
        'б' | 'Б' => "B",
        'в' | 'В' => "V",
        'г' | 'Г' => "G",
        'д' | 'Д' => "D",
        'е' | 'Е' => "E",
        'ё' | 'Ё' => "YO",
        'ж' | 'Ж' => "ZH",
        'з' | 'З' => "Z",
        'и' | 'И' => "I",
        'й' | 'Й' => "I",
        'к' | 'К' => "K",
        'л' | 'Л' => "L",
        'м' | 'М' => "M",
        'н' | 'Н' => "N",
        'о' | 'О' => "O",
        'п' | 'П' => "P",
        'р' | 'Р' => "R",
        'с' | 'С' => "S",
        'т' | 'Т' => "T",
        'у' | 'У' => "U",
        'ф' | 'Ф' => "F",
        'х' | 'Х' => "KH",
        'ц' | 'Ц' => "TS",
        'ч' | 'Ч' => "CH",
        'ш' | 'Ш' => "SH",
        'щ' | 'Щ' => "SHCH",
        'ы' | 'Ы' => "Y",
        'э' | 'Э' => "E",
        'ю' | 'Ю' => "YU",
        'я' | 'Я' => "YA",
        _ if ch.is_ascii_alphanumeric() => return ch.to_string(),
        _ if ch.is_whitespace() || matches!(ch, '_' | '-' | '/' | '\\' | '.') => "_",
        _ => "_",
    };
    mapped.to_string()
}
