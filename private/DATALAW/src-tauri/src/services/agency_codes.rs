use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use tauri::AppHandle;

use crate::{
    error::{AppError, AppResult},
    repository::db,
    services::naming,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgencyDictionaryEntry {
    pub name: String,
    pub code: String,
    pub source: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgencyCodeSuggestion {
    pub name: String,
    pub code: String,
}

const DEFAULT_AGENCIES: &[(&str, &str)] = &[
    ("Следственный комитет РФ", "SKRF"),
    ("Прокуратура", "PROK"),
    ("Суд", "SUD"),
    ("Министерство внутренних дел", "MVD"),
    ("Федеральная служба безопасности", "FSB"),
    ("Федеральная налоговая служба", "FNS"),
    ("Федеральная миграционная служба", "FMS"),
    ("Федеральная служба судебных приставов", "FSSP"),
    ("Роспотребнадзор", "RPN"),
    ("Роспатент", "RPAT"),
    ("Росреестр", "RRS"),
    ("Росимущество", "RIMU"),
    ("Росприроднадзор", "RPRN"),
    ("Роструд", "RTRD"),
    ("Росздравнадзор", "RZDN"),
    ("Роскомнадзор", "RKN"),
    ("Росаккредитация", "RAKR"),
    ("Росалкогольрегулирование", "RAR"),
    ("Росрыболовство", "RRYB"),
    ("Росводоресурсы", "RVDR"),
    ("Рослесхоз", "RLSH"),
    ("Роснедра", "RNDR"),
    ("Ростехнадзор", "RTKH"),
    ("Росгидромет", "RGMT"),
    ("Россельхознадзор", "RSHN"),
    ("Росавиация", "RAVI"),
    ("Росжелдор", "RZHD"),
    ("Росморречфлот", "RMRF"),
    ("Ространснадзор", "RTRN"),
    ("Росатом", "RATM"),
    ("Роскосмос", "RKSM"),
    ("Роснаука", "RNAU"),
    ("Росмолодежь", "RMLD"),
    ("Росстат", "RSTA"),
    ("Росархив", "RARH"),
    ("Росфиннадзор", "RFND"),
    ("Федеральная антимонопольная служба", "FAS"),
    ("Федеральная служба по техническому и экспортному контролю", "FSTK"),
    ("Федеральная служба по тарифам", "FST"),
    ("Федеральная служба по финансовым рынкам", "FSFR"),
    ("Федеральная служба военно-технического сотрудничества", "FSVK"),
    ("Фонд социального страхования", "FSS"),
    ("Пенсионный фонд РФ", "PFR"),
    ("Фонд обязательного медицинского страхования", "FOMS"),
    ("Уполномоченный по правам человека", "UPCH"),
];

fn normalize_name(name: &str) -> String {
    naming::to_standard_name(name)
}

fn normalize_code(code: &str) -> String {
    naming::to_standard_name(code).chars().take(4).collect()
}

fn generate_base_code(name: &str) -> String {
    let normalized = normalize_name(name);
    if normalized.is_empty() {
        return "ORG".to_string();
    }

    let tokens = normalized
        .split('_')
        .filter(|token| !token.is_empty())
        .collect::<Vec<_>>();

    if tokens.len() >= 2 {
        let initials = tokens
            .iter()
            .filter_map(|token| token.chars().next())
            .take(4)
            .collect::<String>();
        if !initials.is_empty() {
            return initials;
        }
    }

    normalized.chars().take(4).collect()
}

fn ensure_unique_code(conn: &Connection, base: &str, exclude_name: Option<&str>) -> AppResult<String> {
    let normalized_base = normalize_code(base);
    let base_code = if normalized_base.is_empty() {
        "ORG".to_string()
    } else {
        normalized_base
    };

    let mut candidate = base_code.clone();
    let mut suffix = 2;

    loop {
        let conflict: Option<String> = conn
            .query_row(
                "SELECT name FROM agency_dictionary WHERE code = ?1",
                [candidate.as_str()],
                |row| row.get(0),
            )
            .ok();

        match conflict {
            None => return Ok(candidate),
            Some(existing_name) if exclude_name.is_some() && exclude_name == Some(existing_name.as_str()) => {
                return Ok(candidate);
            }
            Some(_) => {
                let stem = base_code.chars().take(3).collect::<String>();
                candidate = format!("{}{}", stem, suffix);
                candidate = candidate.chars().take(4).collect();
                suffix += 1;
            }
        }
    }
}

fn ensure_schema(conn: &Connection) -> AppResult<()> {
    conn.execute(
        "CREATE TABLE IF NOT EXISTS agency_dictionary (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            normalized_name TEXT NOT NULL UNIQUE,
            code TEXT NOT NULL UNIQUE,
            source TEXT NOT NULL DEFAULT 'system',
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )",
        [],
    )?;
    Ok(())
}

fn seed_defaults(conn: &Connection) -> AppResult<()> {
    ensure_schema(conn)?;

    for (name, code) in DEFAULT_AGENCIES {
        conn.execute(
            "INSERT OR IGNORE INTO agency_dictionary (name, normalized_name, code, source, created_at, updated_at)
             VALUES (?1, ?2, ?3, 'system', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)",
            params![*name, normalize_name(name), normalize_code(code)],
        )?;
    }

    Ok(())
}

fn list_entries(conn: &Connection) -> AppResult<Vec<AgencyDictionaryEntry>> {
    seed_defaults(conn)?;

    let mut stmt = conn.prepare(
        "SELECT name, code, source
         FROM agency_dictionary
         ORDER BY source DESC, name ASC",
    )?;

    let rows = stmt.query_map([], |row| {
        Ok(AgencyDictionaryEntry {
            name: row.get(0)?,
            code: row.get(1)?,
            source: row.get(2)?,
        })
    })?;

    rows.collect::<Result<Vec<_>, _>>().map_err(AppError::from)
}

pub fn list_agency_codes(app: &AppHandle) -> AppResult<Vec<AgencyDictionaryEntry>> {
    let conn = db::open_user_db(app)?;
    list_entries(&conn)
}

pub fn suggest_agency_code(app: &AppHandle, name: &str) -> AppResult<AgencyCodeSuggestion> {
    let conn = db::open_user_db(app)?;
    seed_defaults(&conn)?;

    let trimmed = name.trim();
    if trimmed.is_empty() {
        return Err(AppError::validation("appeal.error.agencyRequired", "agency name is required"));
    }

    let normalized = normalize_name(trimmed);
    if let Ok(existing) = conn.query_row(
        "SELECT code FROM agency_dictionary WHERE normalized_name = ?1",
        [normalized.as_str()],
        |row| row.get::<_, String>(0),
    ) {
        return Ok(AgencyCodeSuggestion {
            name: trimmed.to_string(),
            code: existing,
        });
    }

    let base = generate_base_code(trimmed);
    let code = ensure_unique_code(&conn, &base, None)?;

    Ok(AgencyCodeSuggestion {
        name: trimmed.to_string(),
        code,
    })
}

pub fn save_agency_code(app: &AppHandle, name: &str, code: &str) -> AppResult<AgencyDictionaryEntry> {
    let conn = db::open_user_db(app)?;
    seed_defaults(&conn)?;

    let trimmed_name = name.trim();
    if trimmed_name.is_empty() {
        return Err(AppError::validation("appeal.error.agencyRequired", "agency name is required"));
    }

    let normalized_name = normalize_name(trimmed_name);
    let normalized_code = normalize_code(code);

    if normalized_code.is_empty() {
        return Err(AppError::validation("appeal.error.agencyCodeInvalid", "agency code is required"));
    }

    let unique_code = ensure_unique_code(&conn, &normalized_code, Some(trimmed_name))?;

    conn.execute(
        "INSERT INTO agency_dictionary (name, normalized_name, code, source, created_at, updated_at)
         VALUES (?1, ?2, ?3, 'custom', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
         ON CONFLICT(name) DO UPDATE SET
           normalized_name = excluded.normalized_name,
           code = excluded.code,
           source = 'custom',
           updated_at = CURRENT_TIMESTAMP",
        params![trimmed_name, normalized_name, unique_code],
    )?;

    Ok(AgencyDictionaryEntry {
        name: trimmed_name.to_string(),
        code: unique_code,
        source: "custom".to_string(),
    })
}

pub fn resolve_agency_code(app: &AppHandle, value: &str) -> AppResult<String> {
    let conn = db::open_user_db(app)?;
    seed_defaults(&conn)?;

    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Ok(String::new());
    }

    let normalized_name = normalize_name(trimmed);
    if let Ok(code) = conn.query_row(
        "SELECT code FROM agency_dictionary WHERE normalized_name = ?1 OR code = ?2",
        params![normalized_name, normalize_code(trimmed)],
        |row| row.get::<_, String>(0),
    ) {
        return Ok(code);
    }

    let base = generate_base_code(trimmed);
    ensure_unique_code(&conn, &base, None)
}
