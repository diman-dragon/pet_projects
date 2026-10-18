use std::{
    fs,
    path::{Path, PathBuf},
    sync::Mutex,
};

use rusqlite::{params, Connection, Result as SqlResult};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::geometry::{DEFAULT_MARRIAGE_GAP, PERSON_CARD_WIDTH};

const INITIAL_SCHEMA: &str = include_str!("../migrations/001_initial.sql");
const CARD_DIMENSIONS_MIGRATION: &str =
    include_str!("../migrations/002_canvas_card_dimensions.sql");

pub struct DatabaseState {
    pub path: Mutex<PathBuf>,
    pub config_path: PathBuf,
}

#[derive(Debug, Deserialize, Serialize)]
struct WorkspaceConfig {
    database_path: String,
}

pub fn saved_database_path(config_path: &Path) -> Option<PathBuf> {
    let content = fs::read_to_string(config_path).ok()?;
    let config = serde_json::from_str::<WorkspaceConfig>(&content).ok()?;
    let path = PathBuf::from(config.database_path);
    path.is_file().then_some(path)
}

#[derive(Debug, Serialize)]
pub struct WorkspaceInfo {
    pub root_dir: String,
    pub tree_name: String,
    pub database_dir: String,
    pub media_dir: String,
}

#[derive(Debug, Deserialize)]
pub struct CreateCanvasCardInput {
    pub x: f64,
    pub y: f64,
    pub generation: i32,
}

#[derive(Debug, Deserialize)]
pub struct CanvasCardPositionInput {
    pub id: String,
    pub x: f64,
    pub y: f64,
    pub generation: i32,
}

#[derive(Debug, Deserialize)]
pub struct PersonInput {
    pub id: String,
    pub sex: Option<String>,
    pub first_name: Option<String>,
    pub patronymic: Option<String>,
    pub last_name: Option<String>,
    pub previous_last_name: Option<String>,
    pub birth_date: Option<String>,
    pub death_date: Option<String>,
    pub photo_url: Option<String>,
    pub photo_zoom: Option<f64>,
    pub photo_offset_x: Option<f64>,
    pub photo_offset_y: Option<f64>,
    pub gender_identity: Option<String>,
    pub nationality: Option<String>,
    pub ethnicity: Option<String>,
    pub religion: Option<String>,
    pub occupation: Option<String>,
    pub education: Option<String>,
    pub description: Option<String>,
    pub biography: Option<String>,
    pub research_status: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct CanvasCardRecord {
    pub id: String,
    pub x: f64,
    pub y: f64,
    pub generation: i32,
    pub locked: bool,
    pub first_name: Option<String>,
    pub patronymic: Option<String>,
    pub last_name: Option<String>,
    pub previous_last_name: Option<String>,
    pub birth_date: Option<String>,
    pub death_date: Option<String>,
    pub photo_url: Option<String>,
    pub photo_zoom: f64,
    pub photo_offset_x: f64,
    pub photo_offset_y: f64,
    pub sex: Option<String>,
    pub gender_identity: Option<String>,
    pub nationality: Option<String>,
    pub ethnicity: Option<String>,
    pub religion: Option<String>,
    pub occupation: Option<String>,
    pub education: Option<String>,
    pub description: Option<String>,
    pub biography: Option<String>,
    pub research_status: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct MarriageRecord {
    pub id: String,
    pub union_type: String,
    pub person_a_id: String,
    pub person_b_id: String,
    pub details_json: Option<String>,
}

fn validate_new_marriage(
    connection: &Connection,
    person_a_id: &str,
    person_b_id: &str,
    exclude_id: Option<&str>,
    details_json: Option<&str>,
    allow_existing: bool,
) -> Result<(), String> {
    let is_dissolved = details_json
        .and_then(|details| serde_json::from_str::<serde_json::Value>(details).ok())
        .is_some_and(|details| {
            details.get("status").and_then(serde_json::Value::as_str) == Some("dissolved")
        });
    if allow_existing || is_dissolved {
        return Ok(());
    }
    for person_id in [person_a_id, person_b_id] {
        let query = "SELECT CASE WHEN a.person_id=?1 THEN b.person_id ELSE a.person_id END, COALESCE((SELECT TRIM(COALESCE(pn.given_name,'') || ' ' || COALESCE(pn.patronymic,'') || ' ' || COALESCE(pn.surname,'')) FROM person_names pn WHERE pn.person_id = CASE WHEN a.person_id=?1 THEN b.person_id ELSE a.person_id END AND pn.is_preferred=1 LIMIT 1), 'Без имени') FROM unions u JOIN union_members a ON a.union_id=u.id JOIN union_members b ON b.union_id=u.id AND a.person_id <> b.person_id JOIN persons p ON p.id = CASE WHEN a.person_id=?1 THEN b.person_id ELSE a.person_id END WHERE u.union_type='marriage' AND (a.person_id=?1 OR b.person_id=?1) AND p.deleted_at IS NULL AND (u.details_json IS NULL OR json_extract(u.details_json, '$.status') IS NULL OR json_extract(u.details_json, '$.status') <> 'dissolved') AND (?2 IS NULL OR u.id <> ?2)";
        let mut statement = connection
            .prepare(query)
            .map_err(|error| error.to_string())?;
        let rows = statement
            .query_map(params![person_id, exclude_id], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
            })
            .map_err(|error| error.to_string())?;
        for row in rows {
            let (partner_id, partner_name) = row.map_err(|error| error.to_string())?;
            let died: i64 = connection.query_row("SELECT COUNT(*) FROM events e JOIN event_participants ep ON ep.event_id=e.id WHERE ep.person_id=?1 AND e.event_type='death'", params![partner_id], |result| result.get(0)).map_err(|error| error.to_string())?;
            if died == 0 {
                return Err(format!("Нельзя зарегистрировать новый брак: у человека уже есть действующий брак с «{partner_name}». Сначала укажите расторжение предыдущего брака или данные о смерти супруга."));
            }
        }
    }
    Ok(())
}

pub fn initialize_database(database_path: &Path) -> SqlResult<()> {
    if let Some(parent) = database_path.parent() {
        fs::create_dir_all(parent)
            .map_err(|error| rusqlite::Error::ToSqlConversionFailure(Box::new(error)))?;
    }
    let mut connection = Connection::open(database_path)?;
    connection.execute_batch("PRAGMA foreign_keys = ON;")?;
    connection.execute_batch("PRAGMA busy_timeout = 5000;")?;
    connection.execute_batch(INITIAL_SCHEMA)?;
    let transaction = connection.transaction()?;
    let has_photo_url: i64 = transaction.query_row(
        "SELECT COUNT(*) FROM pragma_table_info('persons') WHERE name = 'photo_url'",
        [],
        |row| row.get(0),
    )?;
    if has_photo_url == 0 {
        transaction.execute("ALTER TABLE persons ADD COLUMN photo_url TEXT", [])?;
    }
    for (column, definition) in [
        ("photo_zoom", "REAL NOT NULL DEFAULT 1"),
        ("photo_offset_x", "REAL NOT NULL DEFAULT 0"),
        ("photo_offset_y", "REAL NOT NULL DEFAULT 0"),
    ] {
        let exists: i64 = transaction.query_row(
            &format!("SELECT COUNT(*) FROM pragma_table_info('persons') WHERE name = '{column}'"),
            [],
            |row| row.get(0),
        )?;
        if exists == 0 {
            transaction.execute(
                &format!("ALTER TABLE persons ADD COLUMN {column} {definition}"),
                [],
            )?;
        }
    }
    let has_union_details: i64 = transaction.query_row(
        "SELECT COUNT(*) FROM pragma_table_info('unions') WHERE name = 'details_json'",
        [],
        |row| row.get(0),
    )?;
    if has_union_details == 0 {
        transaction.execute("ALTER TABLE unions ADD COLUMN details_json TEXT", [])?;
    }
    let has_dimensions_migration: i64 = transaction.query_row(
        "SELECT COUNT(*) FROM schema_migrations WHERE version = 2",
        [],
        |row| row.get(0),
    )?;
    if has_dimensions_migration == 0 {
        transaction.execute_batch(CARD_DIMENSIONS_MIGRATION)?;
        transaction.execute("INSERT INTO schema_migrations (version) VALUES (2)", [])?;
    }
    transaction.commit()?;
    Ok(())
}

fn open_database(state: &DatabaseState) -> Result<Connection, String> {
    let path = state
        .path
        .lock()
        .map_err(|_| "Не удалось получить путь базы".to_owned())?
        .clone();
    let connection = Connection::open(path).map_err(|error| error.to_string())?;
    connection
        .execute_batch("PRAGMA foreign_keys = ON;")
        .map_err(|error| error.to_string())?;
    connection
        .execute_batch("PRAGMA busy_timeout = 5000;")
        .map_err(|error| error.to_string())?;
    Ok(connection)
}

fn person_media_root(state: &DatabaseState, person_id: &str) -> Result<PathBuf, String> {
    let database_path = state
        .path
        .lock()
        .map_err(|_| "Не удалось получить путь базы".to_owned())?
        .clone();
    let tree_root = database_path
        .parent()
        .and_then(Path::parent)
        .ok_or_else(|| "Не удалось определить каталог дерева".to_owned())?;
    Ok(tree_root.join("media").join(person_id))
}

#[tauri::command]
pub fn read_local_file(path: String) -> Result<Vec<u8>, String> {
    let target = PathBuf::from(path.trim());
    if !target.is_file() {
        return Err("Файл изображения не найден".to_owned());
    }
    let metadata = fs::metadata(&target).map_err(|error| error.to_string())?;
    if metadata.len() > 25 * 1024 * 1024 {
        return Err("Файл слишком большой (больше 25 МБ)".to_owned());
    }
    fs::read(&target).map_err(|error| error.to_string())
}

fn ensure_person_media_dirs(state: &DatabaseState, person_id: &str) -> Result<PathBuf, String> {
    let root = person_media_root(state, person_id)?;
    for directory in [
        "profile",
        "documents/birth",
        "documents/death",
        "documents/marriage",
        "documents/residence",
        "documents/military",
        "documents/other",
    ] {
        fs::create_dir_all(root.join(directory)).map_err(|error| error.to_string())?;
    }
    Ok(root)
}

fn person_media_paths(connection: &Connection, person_id: &str) -> Result<Vec<PathBuf>, String> {
    let mut statement = connection
        .prepare("SELECT m.file_path FROM media m JOIN media_links ml ON ml.media_id=m.id WHERE ml.person_id=?1")
        .map_err(|error| error.to_string())?;
    let result = statement
        .query_map(params![person_id], |row| row.get::<_, String>(0))
        .map_err(|error| error.to_string())?
        .map(|path| path.map(PathBuf::from).map_err(|error| error.to_string()))
        .collect();
    result
}

fn remove_person_media_directory(state: &DatabaseState, person_id: &str) {
    if let Ok(root) = person_media_root(state, person_id) {
        let _ = fs::remove_dir_all(root);
    }
}

fn clean_file_stem(value: &str) -> String {
    let cleaned = value
        .chars()
        .filter(|character| {
            character.is_ascii_alphanumeric() || *character == '-' || *character == '_'
        })
        .collect::<String>();
    if cleaned.is_empty() {
        "file".to_owned()
    } else {
        cleaned
    }
}

fn workspace_info(root: &Path, tree_name: &str) -> WorkspaceInfo {
    WorkspaceInfo {
        root_dir: root.to_string_lossy().into_owned(),
        tree_name: tree_name.to_owned(),
        database_dir: root.join("db").to_string_lossy().into_owned(),
        media_dir: root.join("media").to_string_lossy().into_owned(),
    }
}

fn copy_directory(source: &Path, target: &Path) -> Result<(), String> {
    fs::create_dir_all(target).map_err(|error| error.to_string())?;
    for entry in fs::read_dir(source).map_err(|error| error.to_string())? {
        let entry = entry.map_err(|error| error.to_string())?;
        let source_path = entry.path();
        let target_path = target.join(entry.file_name());
        if source_path.is_dir() {
            copy_directory(&source_path, &target_path)?;
        } else {
            fs::copy(&source_path, &target_path).map_err(|error| error.to_string())?;
        }
    }
    Ok(())
}

fn temporary_sibling(path: &Path, label: &str) -> PathBuf {
    let suffix = Uuid::new_v4().simple().to_string();
    path.with_file_name(format!(
        ".{}.{}-{}",
        path.file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("family-tree"),
        label,
        suffix
    ))
}

fn write_atomic(path: &Path, content: &str) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    let temporary = temporary_sibling(path, "config");
    fs::write(&temporary, content).map_err(|error| error.to_string())?;
    let backup = temporary_sibling(path, "config-backup");
    let had_existing = path.exists();
    if had_existing {
        if let Err(error) = fs::rename(path, &backup) {
            let _ = fs::remove_file(&temporary);
            return Err(error.to_string());
        }
    }
    if let Err(error) = fs::rename(&temporary, path) {
        if had_existing {
            let _ = fs::rename(&backup, path);
        }
        let _ = fs::remove_file(&temporary);
        return Err(error.to_string());
    }
    // If cleanup is denied, the new configuration is still valid. Keep the
    // backup so the user can recover it instead of risking data loss.
    let _ = fs::remove_file(&backup);
    Ok(())
}

fn relocate_workspace(source: &Path, target: &Path) -> Result<PathBuf, String> {
    if target.starts_with(source) || source.starts_with(target) {
        return Err(
            "Каталог дерева нельзя переносить внутрь самого себя или его подкаталога".to_owned(),
        );
    }
    if target.exists() {
        return Err(format!(
            "Каталог назначения уже существует: {}",
            target.display()
        ));
    }
    if let Some(parent) = target.parent() {
        fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }

    // Prepare a complete copy first. The current workspace remains untouched
    // until the copy has been created and its database has been opened.
    let staging = temporary_sibling(target, "staging");
    if let Err(error) = copy_directory(source, &staging) {
        let _ = fs::remove_dir_all(&staging);
        return Err(format!("Не удалось подготовить перенос дерева: {error}"));
    }
    let staged_database = staging.join("db").join("family-tree.sqlite3");
    if let Err(error) = initialize_database(&staged_database) {
        let _ = fs::remove_dir_all(&staging);
        return Err(format!(
            "Не удалось проверить базу в новом каталоге: {error}"
        ));
    }

    let backup = temporary_sibling(source, "backup");
    if let Err(error) = fs::rename(source, &backup) {
        let _ = fs::remove_dir_all(&staging);
        return Err(format!(
            "Не удалось подготовить старый каталог к переносу: {error}"
        ));
    }
    if let Err(error) = fs::rename(&staging, target) {
        let _ = fs::rename(&backup, source);
        let _ = fs::remove_dir_all(&staging);
        return Err(format!("Не удалось завершить перенос дерева: {error}"));
    }
    Ok(backup)
}

fn safe_tree_name(value: &str) -> Result<String, String> {
    let name = value.trim();
    let reserved = [
        "CON", "PRN", "AUX", "NUL", "COM1", "COM2", "COM3", "COM4", "LPT1", "LPT2", "LPT3",
    ];
    if name.is_empty()
        || name == "."
        || name == ".."
        || name.ends_with('.')
        || name.ends_with(' ')
        || name.contains(['/', '\\', ':', '*', '?', '"', '<', '>', '|'])
        || name.chars().any(char::is_control)
        || reserved.iter().any(|item| item.eq_ignore_ascii_case(name))
    {
        return Err("Название дерева должно быть непустым именем папки".to_owned());
    }
    Ok(name.to_owned())
}

#[tauri::command]
pub fn get_workspace(state: tauri::State<'_, DatabaseState>) -> Result<WorkspaceInfo, String> {
    let path = state
        .path
        .lock()
        .map_err(|_| "Не удалось получить путь базы".to_owned())?
        .clone();
    let root = path
        .parent()
        .and_then(Path::parent)
        .ok_or_else(|| "Не удалось определить каталог дерева".to_owned())?;
    let tree_name = root
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| "Не удалось определить название дерева".to_owned())?;
    Ok(workspace_info(root, tree_name))
}

#[tauri::command]
pub fn set_workspace(
    parent_dir: String,
    tree_name: String,
    state: tauri::State<'_, DatabaseState>,
) -> Result<WorkspaceInfo, String> {
    let tree_name = safe_tree_name(&tree_name)?;
    let current_database_path = state
        .path
        .lock()
        .map_err(|_| "Не удалось получить путь базы".to_owned())?
        .clone();
    let current_root = current_database_path
        .parent()
        .and_then(Path::parent)
        .ok_or_else(|| "Не удалось определить текущий каталог дерева".to_owned())?
        .to_path_buf();
    let parent_dir = parent_dir.trim();
    if parent_dir.is_empty() {
        return Err("Не выбран родительский каталог дерева".to_owned());
    }
    let parent = PathBuf::from(parent_dir);
    if !parent.is_absolute() {
        return Err("Родительский каталог должен быть абсолютным путём".to_owned());
    }
    let root = parent.join("FamilyTree").join(&tree_name);
    let backup = if root != current_root {
        Some(relocate_workspace(&current_root, &root)?)
    } else {
        None
    };
    let database_dir = root.join("db");
    let media_dir = root.join("media");
    if let Err(error) = fs::create_dir_all(&database_dir)
        .map_err(|error| error.to_string())
        .and_then(|_| fs::create_dir_all(&media_dir).map_err(|error| error.to_string()))
    {
        if let Some(backup) = &backup {
            let _ = fs::remove_dir_all(&root);
            let _ = fs::rename(backup, &current_root);
        }
        return Err(error);
    }
    let database_path = database_dir.join("family-tree.sqlite3");
    if let Err(error) = initialize_database(&database_path).map_err(|error| error.to_string()) {
        if let Some(backup) = &backup {
            let _ = fs::remove_dir_all(&root);
            let _ = fs::rename(backup, &current_root);
        }
        return Err(error);
    }
    if root != current_root {
        let mut connection = match Connection::open(&database_path) {
            Ok(connection) => connection,
            Err(error) => {
                if let Some(backup) = &backup {
                    let _ = fs::remove_dir_all(&root);
                    let _ = fs::rename(backup, &current_root);
                }
                return Err(error.to_string());
            }
        };
        let transaction = connection
            .transaction()
            .map_err(|error| error.to_string())?;
        if let Err(error) = transaction.execute(
            "UPDATE persons SET photo_url = replace(photo_url, ?1, ?2) WHERE photo_url IS NOT NULL",
            params![current_root.to_string_lossy(), root.to_string_lossy()],
        ).map_err(|error| error.to_string()).and_then(|_| transaction.commit().map_err(|error| error.to_string())) {
            if let Some(backup) = &backup { let _ = fs::remove_dir_all(&root); let _ = fs::rename(backup, &current_root); }
            return Err(error);
        }
    }
    let config = WorkspaceConfig {
        database_path: root
            .join("db")
            .join("family-tree.sqlite3")
            .to_string_lossy()
            .into_owned(),
    };
    let mut active_path = state
        .path
        .lock()
        .map_err(|_| "Не удалось изменить путь базы".to_owned())?;
    if let Err(error) = write_atomic(
        &state.config_path,
        &serde_json::to_string_pretty(&config).map_err(|error| error.to_string())?,
    ) {
        if let Some(backup) = &backup {
            let _ = fs::remove_dir_all(&root);
            let _ = fs::rename(backup, &current_root);
        }
        return Err(error);
    }
    *active_path = database_path;
    if let Some(backup) = backup {
        let _ = fs::remove_dir_all(backup);
    }
    Ok(workspace_info(&root, &tree_name))
}

#[tauri::command]
pub fn import_person_media(
    person_id: String,
    source_path: String,
    media_type: String,
    state: tauri::State<'_, DatabaseState>,
) -> Result<String, String> {
    let person_root = ensure_person_media_dirs(&state, &person_id)?;
    let source = PathBuf::from(&source_path);
    let extension = source
        .extension()
        .and_then(|extension| extension.to_str())
        .map(|value| format!(".{value}"))
        .unwrap_or_default();
    let (directory, stem): (PathBuf, String) = match media_type.as_str() {
        "profile" => (person_root.join("profile"), "profile".to_owned()),
        "birth" => (
            person_root.join("documents/birth"),
            "birth-certificate".to_owned(),
        ),
        "death" => (
            person_root.join("documents/death"),
            "death-certificate".to_owned(),
        ),
        "marriage" => (
            person_root.join("documents/marriage"),
            "marriage".to_owned(),
        ),
        "marriage_photo" => (
            person_root.join("documents/marriage"),
            "marriage-photo".to_owned(),
        ),
        "residence" => (
            person_root.join("documents/residence"),
            "residence".to_owned(),
        ),
        "military" => (
            person_root.join("documents/military"),
            "military".to_owned(),
        ),
        _ => (
            person_root.join("documents/other"),
            clean_file_stem(
                source
                    .file_stem()
                    .and_then(|value| value.to_str())
                    .unwrap_or("document"),
            ),
        ),
    };
    let stem = clean_file_stem(&stem);
    let mut target = directory.join(format!("{stem}{extension}"));
    let mut suffix = 2;
    while target.exists() {
        target = directory.join(format!("{stem}-{suffix}{extension}"));
        suffix += 1;
    }
    fs::copy(&source, &target).map_err(|error| error.to_string())?;
    let media_id = uuid::Uuid::new_v4().to_string();
    let file_name = target
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("media");
    let mime_type = source
        .extension()
        .and_then(|extension| extension.to_str())
        .map(|extension| format!("image/{extension}"))
        .unwrap_or_else(|| "application/octet-stream".to_owned());
    let connection = open_database(&state)?;
    if let Err(error) = connection.execute(
        "INSERT INTO media (id,file_name,file_path,mime_type,media_type) VALUES (?1,?2,?3,?4,?5)",
        rusqlite::params![
            media_id,
            file_name,
            target.to_string_lossy(),
            mime_type,
            media_type
        ],
    ) {
        let _ = fs::remove_file(&target);
        return Err(error.to_string());
    }
    if let Err(error) = connection.execute(
        "INSERT INTO media_links (media_id,person_id) VALUES (?1,?2)",
        rusqlite::params![media_id, person_id],
    ) {
        let _ = connection.execute("DELETE FROM media WHERE id=?1", params![media_id]);
        let _ = fs::remove_file(&target);
        return Err(error.to_string());
    }
    Ok(target.to_string_lossy().into_owned())
}

#[tauri::command]
pub fn save_person_photo_crop(
    person_id: String,
    image_data: Vec<u8>,
    zoom: f64,
    offset_x: f64,
    offset_y: f64,
    state: tauri::State<'_, DatabaseState>,
) -> Result<String, String> {
    if person_id.trim().is_empty() {
        return Err("Сначала создайте карточку человека".to_owned());
    }
    if image_data.is_empty() {
        return Err("Кадр фотографии пустой".to_owned());
    }
    let person_root = ensure_person_media_dirs(&state, &person_id)?;
    let directory = person_root.join("profile");
    let mut target = directory.join("profile-cropped.png");
    let mut suffix = 2;
    while target.exists() {
        target = directory.join(format!("profile-cropped-{suffix}.png"));
        suffix += 1;
    }
    fs::write(&target, image_data).map_err(|error| error.to_string())?;
    let media_id = uuid::Uuid::new_v4().to_string();
    let file_name = target
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("profile-cropped.png");
    let connection = open_database(&state)?;
    if let Err(error) = connection.execute(
            "INSERT INTO media (id,file_name,file_path,mime_type,media_type) VALUES (?1,?2,?3,'image/png','profile')",
            params![media_id, file_name, target.to_string_lossy()],
        ) {
        let _ = fs::remove_file(&target);
        return Err(error.to_string());
    }
    if let Err(error) = connection.execute(
        "INSERT INTO media_links (media_id,person_id) VALUES (?1,?2)",
        params![media_id, person_id],
    ) {
        let _ = connection.execute("DELETE FROM media WHERE id=?1", params![media_id]);
        let _ = fs::remove_file(&target);
        return Err(error.to_string());
    }
    connection
        .execute(
            "UPDATE persons SET photo_url = ?1, photo_zoom = ?2, photo_offset_x = ?3, photo_offset_y = ?4 WHERE id = ?5",
            params![target.to_string_lossy(), zoom, offset_x, offset_y, person_id],
        )
        .map_err(|error| error.to_string())?;
    Ok(target.to_string_lossy().into_owned())
}

#[tauri::command]
pub fn delete_person_photo(
    person_id: String,
    state: tauri::State<'_, DatabaseState>,
) -> Result<(), String> {
    let connection = open_database(&state)?;
    let mut statement = connection
        .prepare("SELECT m.id, m.file_path FROM media m JOIN media_links ml ON ml.media_id=m.id WHERE m.media_type='profile' AND ml.person_id=?1")
        .map_err(|error| error.to_string())?;
    let records = statement
        .query_map(params![&person_id], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })
        .map_err(|error| error.to_string())?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| error.to_string())?;
    drop(statement);
    for (media_id, file_path) in records {
        let _ = file_path;
        connection
            .execute(
                "DELETE FROM media_links WHERE media_id=?1",
                params![media_id],
            )
            .map_err(|error| error.to_string())?;
    }
    connection
        .execute(
            "UPDATE persons SET photo_url = NULL, photo_zoom = 1, photo_offset_x = 0, photo_offset_y = 0 WHERE id = ?1",
            params![person_id],
        )
        .map_err(|error| error.to_string())?;
    Ok(())
}

#[tauri::command]
pub fn save_relation_photo_crop(
    relation_id: String,
    person_id: String,
    image_data: Vec<u8>,
    state: tauri::State<'_, DatabaseState>,
) -> Result<String, String> {
    if image_data.is_empty() {
        return Err("Кадр фотографии пары пустой".to_owned());
    }
    let person_root = ensure_person_media_dirs(&state, &person_id)?;
    let directory = person_root.join("documents/marriage");
    let mut target = directory.join(format!("relation-photo-{relation_id}.png"));
    let mut suffix = 2;
    while target.exists() {
        target = directory.join(format!("relation-photo-{relation_id}-{suffix}.png"));
        suffix += 1;
    }
    fs::write(&target, image_data).map_err(|error| error.to_string())?;
    let media_id = uuid::Uuid::new_v4().to_string();
    let file_name = target
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("relation-photo.png");
    let connection = open_database(&state)?;
    if let Err(error) = connection.execute(
            "INSERT INTO media (id,file_name,file_path,mime_type,media_type) VALUES (?1,?2,?3,'image/png','marriage_photo')",
            params![media_id, file_name, target.to_string_lossy()],
        ) {
        let _ = fs::remove_file(&target);
        return Err(error.to_string());
    }
    if let Err(error) = connection.execute(
        "INSERT INTO media_links (media_id,person_id) VALUES (?1,?2)",
        params![media_id, person_id],
    ) {
        let _ = connection.execute("DELETE FROM media WHERE id=?1", params![media_id]);
        let _ = fs::remove_file(&target);
        return Err(error.to_string());
    }
    Ok(target.to_string_lossy().into_owned())
}

#[tauri::command]
pub fn delete_relation_photo(
    relation_id: String,
    state: tauri::State<'_, DatabaseState>,
) -> Result<(), String> {
    let connection = open_database(&state)?;
    let mut statement = connection
        .prepare("SELECT m.id, m.file_path FROM media m JOIN media_links ml ON ml.media_id=m.id WHERE m.media_type='marriage_photo' AND m.file_name LIKE ?1")
        .map_err(|error| error.to_string())?;
    let records = statement
        .query_map(params![format!("relation-photo-{relation_id}%")], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })
        .map_err(|error| error.to_string())?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| error.to_string())?;
    drop(statement);
    for (media_id, file_path) in records {
        // Файл оставляем на диске: undo может вернуть прежнюю ссылку без потери медиа.
        let _ = file_path;
        connection
            .execute(
                "DELETE FROM media_links WHERE media_id=?1",
                params![media_id],
            )
            .map_err(|error| error.to_string())?;
    }
    Ok(())
}

#[tauri::command]
pub fn update_relation_details(
    relation_id: String,
    details_json: Option<String>,
    state: tauri::State<'_, DatabaseState>,
) -> Result<(), String> {
    let connection = open_database(&state)?;
    connection
        .execute(
            "UPDATE unions SET details_json=?2 WHERE id=?1",
            params![relation_id, details_json],
        )
        .map_err(|error| error.to_string())?;
    Ok(())
}

#[tauri::command]
pub fn update_parenthood_auto_align_family(
    relation_ids: Vec<String>,
    auto_align: bool,
    state: tauri::State<'_, DatabaseState>,
) -> Result<usize, String> {
    if relation_ids.is_empty() {
        return Ok(0);
    }
    let mut connection = open_database(&state)?;
    let transaction = connection
        .transaction()
        .map_err(|error| error.to_string())?;
    let mut updated = 0;
    for relation_id in relation_ids {
        let details_json: Option<String> = transaction
            .query_row(
                "SELECT details_json FROM unions WHERE id=?1 AND union_type='parenthood'",
                params![relation_id],
                |row| row.get(0),
            )
            .map_err(|error| error.to_string())?;
        let mut details = details_json
            .as_deref()
            .and_then(|value| serde_json::from_str::<serde_json::Value>(value).ok())
            .filter(serde_json::Value::is_object)
            .unwrap_or_else(|| serde_json::json!({}));
        details["autoAlign"] = serde_json::Value::Bool(auto_align);
        transaction
            .execute(
                "UPDATE unions SET details_json=?1 WHERE id=?2",
                params![details.to_string(), relation_id],
            )
            .map_err(|error| error.to_string())?;
        updated += 1;
    }
    transaction.commit().map_err(|error| error.to_string())?;
    Ok(updated)
}

#[derive(Debug, Serialize)]
pub struct PersonMediaRecord {
    pub media_type: String,
    pub file_path: String,
}

#[tauri::command]
pub fn load_person_media(
    person_id: String,
    state: tauri::State<'_, DatabaseState>,
) -> Result<Vec<PersonMediaRecord>, String> {
    let connection = open_database(&state)?;
    let mut statement = connection
        .prepare(
            "SELECT m.media_type, m.file_path
             FROM media m
             JOIN media_links ml ON ml.media_id = m.id
             WHERE ml.person_id = ?1
             ORDER BY m.created_at",
        )
        .map_err(|error| error.to_string())?;
    let rows = statement
        .query_map(params![person_id], |row| {
            Ok(PersonMediaRecord {
                media_type: row.get(0)?,
                file_path: row.get(1)?,
            })
        })
        .map_err(|error| error.to_string())?;
    rows.collect::<SqlResult<Vec<_>>>()
        .map_err(|error| error.to_string())
}

fn clean(value: Option<String>) -> Option<String> {
    value.and_then(|value| {
        let value = value.trim().to_owned();
        (!value.is_empty()).then_some(value)
    })
}

fn arrange_relation_cards(
    connection: &Connection,
    anchor_person: &str,
    left_person: &str,
    right_person: &str,
    details_json: Option<&str>,
) -> Result<(), String> {
    let anchor: (f64, f64, i32) = connection
        .query_row(
            "SELECT x, y, generation FROM canvas_cards WHERE person_id=?1",
            params![anchor_person],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )
        .map_err(|error| error.to_string())?;
    let spacing = details_json
        .and_then(|value| serde_json::from_str::<serde_json::Value>(value).ok())
        .and_then(|value| value.get("spacing").and_then(serde_json::Value::as_f64))
        .unwrap_or(DEFAULT_MARRIAGE_GAP)
        .max(10.0);
    let card_width = PERSON_CARD_WIDTH;
    let left_x = if anchor_person == left_person {
        anchor.0
    } else {
        anchor.0 - card_width - spacing
    };
    let right_x = left_x + card_width + spacing;
    connection
        .execute(
            "UPDATE canvas_cards SET x=?1, y=?2, generation=?3 WHERE person_id=?4",
            params![left_x, anchor.1, anchor.2, left_person],
        )
        .map_err(|error| error.to_string())?;
    connection
        .execute(
            "UPDATE canvas_cards SET x=?1, y=?2, generation=?3 WHERE person_id=?4",
            params![right_x, anchor.1, anchor.2, right_person],
        )
        .map_err(|error| error.to_string())?;
    Ok(())
}

fn arrange_parenthood_child(
    connection: &Connection,
    child_id: &str,
    details_json: Option<&str>,
) -> Result<(), String> {
    let auto_align = details_json
        .and_then(|value| serde_json::from_str::<serde_json::Value>(value).ok())
        .and_then(|details| {
            details
                .get("autoAlign")
                .and_then(serde_json::Value::as_bool)
        })
        .unwrap_or(true);
    if !auto_align {
        return Ok(());
    }
    let parents: Vec<f64> = {
        let mut statement = connection.prepare("SELECT c.x FROM union_members um JOIN canvas_cards c ON c.person_id=um.person_id JOIN unions u ON u.id=um.union_id WHERE u.union_type='parenthood' AND um.role='parent' AND um.person_id<>?1 AND EXISTS (SELECT 1 FROM union_members child WHERE child.union_id=u.id AND child.person_id=?1)").map_err(|error| error.to_string())?;
        let rows = statement
            .query_map(params![child_id], |row| row.get(0))
            .map_err(|error| error.to_string())?;
        rows.collect::<SqlResult<Vec<f64>>>()
            .map_err(|error| error.to_string())?
    };
    if parents.len() >= 2 {
        let center = parents.iter().map(|x| x + 125.0).sum::<f64>() / parents.len() as f64;
        connection
            .execute(
                "UPDATE canvas_cards SET x=?1-125.0 WHERE person_id=?2",
                params![center, child_id],
            )
            .map_err(|error| error.to_string())?;
    }
    Ok(())
}

fn date_metadata(value: &str) -> (&'static str, &'static str, Option<String>, Option<String>) {
    let value_lower = value.to_lowercase();
    let qualifier = if value_lower.starts_with("около")
        || value_lower.starts_with("примерно")
        || value_lower.starts_with("about")
    {
        Some("about".to_owned())
    } else if value_lower.starts_with("до") || value_lower.starts_with("before") {
        Some("before".to_owned())
    } else if value_lower.starts_with("после") || value_lower.starts_with("after") {
        Some("after".to_owned())
    } else {
        None
    };
    let year = value
        .split(|character: char| !character.is_ascii_digit())
        .find(|part| part.len() == 4)
        .map(str::to_owned);
    let kind = if qualifier.is_some() {
        "approximate"
    } else if value.contains('-') {
        "range"
    } else if year.is_some() {
        "partial"
    } else {
        "phrase"
    };
    let precision = if value.len() == 4 && year.is_some() {
        "year"
    } else {
        "unknown"
    };
    (kind, precision, year, qualifier)
}

fn insert_date(connection: &Connection, value: Option<&str>) -> Result<Option<String>, String> {
    let Some(value) = value.and_then(|value| clean(Some(value.to_owned()))) else {
        return Ok(None);
    };
    let (kind, precision, year, qualifier) = date_metadata(&value);
    let id = Uuid::new_v4().to_string();
    connection.execute(
        "INSERT INTO date_values (id, original_text, date_kind, precision, year, qualifier, normalized_text) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?2)",
        params![id, value, kind, precision, year, qualifier],
    ).map_err(|error| error.to_string())?;
    Ok(Some(id))
}

fn insert_profile_event(
    connection: &Connection,
    person_id: &str,
    event_type: &str,
    date: Option<&str>,
) -> Result<(), String> {
    let Some(date_id) = insert_date(connection, date)? else {
        return Ok(());
    };
    let event_id = Uuid::new_v4().to_string();
    connection
        .execute(
            "INSERT INTO events (id, event_type, date_id, is_primary) VALUES (?1, ?2, ?3, 1)",
            params![event_id, event_type, date_id],
        )
        .map_err(|error| error.to_string())?;
    connection.execute("INSERT INTO event_participants (event_id, person_id, role) VALUES (?1, ?2, 'principal')", params![event_id, person_id]).map_err(|error| error.to_string())?;
    Ok(())
}

#[tauri::command]
pub fn load_canvas_cards(
    state: tauri::State<'_, DatabaseState>,
) -> Result<Vec<CanvasCardRecord>, String> {
    let connection = open_database(&state)?;
    let mut statement = connection.prepare(
        "SELECT c.id, c.x, c.y, c.generation, p.photo_url, p.photo_zoom, p.photo_offset_x, p.photo_offset_y, p.sex,
            p.gender_identity, p.nationality, p.ethnicity, p.religion, p.occupation, p.education, p.description, p.biography, p.research_status,
            (SELECT pn.given_name FROM person_names pn WHERE pn.person_id = p.id AND pn.is_preferred = 1 LIMIT 1),
            (SELECT pn.patronymic FROM person_names pn WHERE pn.person_id = p.id AND pn.is_preferred = 1 LIMIT 1),
            (SELECT pn.surname FROM person_names pn WHERE pn.person_id = p.id AND pn.is_preferred = 1 LIMIT 1),
            (SELECT pn.maiden_name FROM person_names pn WHERE pn.person_id = p.id AND pn.is_preferred = 1 LIMIT 1),
            (SELECT dv.original_text FROM events e JOIN event_participants ep ON ep.event_id = e.id JOIN date_values dv ON dv.id = e.date_id WHERE ep.person_id = p.id AND e.event_type = 'birth' LIMIT 1),
            (SELECT dv.original_text FROM events e JOIN event_participants ep ON ep.event_id = e.id JOIN date_values dv ON dv.id = e.date_id WHERE ep.person_id = p.id AND e.event_type = 'death' LIMIT 1), c.is_pinned
         FROM canvas_cards c JOIN persons p ON p.id = c.person_id WHERE p.deleted_at IS NULL ORDER BY c.created_at",
    ).map_err(|error| error.to_string())?;
    let rows = statement
        .query_map([], |row| {
            Ok(CanvasCardRecord {
                id: row.get(0)?,
                x: row.get(1)?,
                y: row.get(2)?,
                generation: row.get(3)?,
                locked: row.get::<_, i64>(24)? != 0,
                photo_url: row.get(4)?,
                photo_zoom: row.get(5)?,
                photo_offset_x: row.get(6)?,
                photo_offset_y: row.get(7)?,
                sex: row.get(8)?,
                gender_identity: row.get(9)?,
                nationality: row.get(10)?,
                ethnicity: row.get(11)?,
                religion: row.get(12)?,
                occupation: row.get(13)?,
                education: row.get(14)?,
                description: row.get(15)?,
                biography: row.get(16)?,
                research_status: row.get(17)?,
                first_name: row.get(18)?,
                patronymic: row.get(19)?,
                last_name: row.get(20)?,
                previous_last_name: row.get(21)?,
                birth_date: row.get(22)?,
                death_date: row.get(23)?,
            })
        })
        .map_err(|error| error.to_string())?;
    rows.collect::<SqlResult<Vec<_>>>()
        .map_err(|error| error.to_string())
}

#[tauri::command]
pub fn load_relations(
    state: tauri::State<'_, DatabaseState>,
) -> Result<Vec<MarriageRecord>, String> {
    let connection = open_database(&state)?;
    let mut statement = connection.prepare("SELECT u.id, CASE WHEN u.union_type='marriage' AND json_extract(u.details_json, '$.role') IN ('father','mother') THEN 'parenthood' ELSE u.union_type END, a.person_id, b.person_id, u.details_json FROM unions u JOIN union_members a ON a.union_id = u.id JOIN union_members b ON b.union_id = u.id AND a.sequence < b.sequence JOIN persons pa ON pa.id = a.person_id AND pa.deleted_at IS NULL JOIN persons pb ON pb.id = b.person_id AND pb.deleted_at IS NULL WHERE u.union_type IN ('marriage', 'parenthood') AND (SELECT COUNT(*) FROM union_members members WHERE members.union_id = u.id) = 2 ORDER BY u.id").map_err(|error| error.to_string())?;
    let rows = statement
        .query_map([], |row| {
            Ok(MarriageRecord {
                id: row.get(0)?,
                union_type: row.get(1)?,
                person_a_id: row.get(2)?,
                person_b_id: row.get(3)?,
                details_json: row.get(4)?,
            })
        })
        .map_err(|error| error.to_string())?;
    rows.collect::<SqlResult<Vec<_>>>()
        .map_err(|error| error.to_string())
}

#[tauri::command]
pub fn create_relation(
    relation_type: String,
    person_a_id: String,
    person_b_id: String,
    details_json: Option<String>,
    allow_existing_marriage: Option<bool>,
    state: tauri::State<'_, DatabaseState>,
) -> Result<String, String> {
    if person_a_id == person_b_id {
        return Err("Нельзя создать связь человека с самим собой".to_owned());
    }
    let connection = open_database(&state)?;
    let exists: i64 = connection
        .query_row(
            "SELECT COUNT(*) FROM persons WHERE id IN (?1, ?2) AND deleted_at IS NULL",
            params![person_a_id, person_b_id],
            |row| row.get(0),
        )
        .map_err(|error| error.to_string())?;
    if exists != 2 {
        return Err("Оба человека должны существовать и быть доступны".to_owned());
    }
    if relation_type == "parenthood" {
        let is_cycle: i64 = connection.query_row(
            "WITH RECURSIVE ancestor_tree(person_id, current_id) AS (SELECT ?1, ?1 UNION ALL SELECT ut.person_id, um.person_id FROM union_members um JOIN unions ut ON ut.id=um.union_id WHERE ut.union_type='parenthood' AND um.role='child' AND um.person_id=ancestor_tree.current_id) SELECT COUNT(*) FROM ancestor_tree WHERE person_id=?2",
            params![person_b_id, person_a_id],
            |row| row.get(0),
        ).map_err(|error| error.to_string())?;
        if is_cycle > 0 {
            return Err("Нельзя создать родительскую связь: это создаст цикл в генеалогическом дереве (предок станет потомком самого себя).".to_owned());
        }
    }
    let allowed_types = ["marriage", "parenthood"];
    if !allowed_types.contains(&relation_type.as_str()) {
        return Err("Неизвестный тип связи".to_owned());
    }
    let already_exists: i64 = connection.query_row("SELECT COUNT(*) FROM unions u JOIN union_members a ON a.union_id=u.id JOIN union_members b ON b.union_id=u.id WHERE u.union_type=?3 AND ((a.person_id=?1 AND b.person_id=?2) OR (a.person_id=?2 AND b.person_id=?1))", params![person_a_id, person_b_id, relation_type], |row| row.get(0)).map_err(|error| error.to_string())?;
    if already_exists > 0 {
        return Err(if relation_type == "parenthood" {
            "Такая родительская связь уже существует. Дубликат не создан; можно открыть существующую связь и отредактировать её.".to_owned()
        } else {
            "Этот брак уже есть в базе. Создание дубликата не требуется; можно открыть его и отредактировать данные.".to_owned()
        });
    }
    if relation_type == "marriage" {
        validate_new_marriage(
            &connection,
            &person_a_id,
            &person_b_id,
            None,
            details_json.as_deref(),
            allow_existing_marriage.unwrap_or(false),
        )?;
    }
    let union_id = Uuid::new_v4().to_string();
    let sex_a: Option<String> = connection
        .query_row(
            "SELECT sex FROM persons WHERE id=?1",
            params![person_a_id],
            |row| row.get(0),
        )
        .map_err(|error| error.to_string())?;
    let sex_b: Option<String> = connection
        .query_row(
            "SELECT sex FROM persons WHERE id=?1",
            params![person_b_id],
            |row| row.get(0),
        )
        .map_err(|error| error.to_string())?;
    let (left_person, right_person) = if relation_type == "parenthood"
        || (sex_a.as_deref() == Some("male") && sex_b.as_deref() == Some("female"))
    {
        (person_a_id.clone(), person_b_id.clone())
    } else if sex_a.as_deref() == Some("female") && sex_b.as_deref() == Some("male") {
        (person_b_id.clone(), person_a_id.clone())
    } else {
        (person_a_id.clone(), person_b_id.clone())
    };
    connection
        .execute(
            "INSERT INTO unions (id, union_type, details_json) VALUES (?1, ?2, ?3)",
            params![union_id, relation_type, details_json],
        )
        .map_err(|error| error.to_string())?;
    let (left_role, right_role) = if relation_type == "parenthood" {
        let role = details_json
            .as_deref()
            .and_then(|value| serde_json::from_str::<serde_json::Value>(value).ok())
            .and_then(|value| {
                value
                    .get("role")
                    .and_then(serde_json::Value::as_str)
                    .map(str::to_owned)
            })
            .unwrap_or_else(|| "parent".to_owned());
        (role, "child".to_owned())
    } else {
        ("spouse".to_owned(), "spouse".to_owned())
    };
    connection.execute("INSERT INTO union_members (union_id, person_id, role, sequence) VALUES (?1, ?2, ?4, 1), (?1, ?3, ?5, 2)", params![union_id, left_person, right_person, left_role, right_role]).map_err(|error| error.to_string())?;
    if relation_type == "marriage" {
        arrange_relation_cards(
            &connection,
            &person_a_id,
            &left_person,
            &right_person,
            details_json.as_deref(),
        )?;
    }
    if relation_type == "parenthood" {
        arrange_parenthood_child(&connection, &person_b_id, details_json.as_deref())?;
    }
    Ok(union_id)
}

#[tauri::command]
pub fn update_relation(
    relation_id: String,
    relation_type: String,
    person_a_id: String,
    person_b_id: String,
    details_json: Option<String>,
    allow_existing_marriage: Option<bool>,
    state: tauri::State<'_, DatabaseState>,
) -> Result<(), String> {
    if person_a_id == person_b_id {
        return Err("Нельзя создать связь человека с самим собой".to_owned());
    }
    if !["marriage", "parenthood"].contains(&relation_type.as_str()) {
        return Err("Неизвестный тип связи".to_owned());
    }
    let connection = open_database(&state)?;
    let people: i64 = connection
        .query_row(
            "SELECT COUNT(*) FROM persons WHERE id IN (?1, ?2) AND deleted_at IS NULL",
            params![person_a_id, person_b_id],
            |row| row.get(0),
        )
        .map_err(|error| error.to_string())?;
    if people != 2 {
        return Err("Оба человека должны существовать и быть доступны".to_owned());
    }
    let duplicate: i64 = connection.query_row("SELECT COUNT(*) FROM unions u JOIN union_members a ON a.union_id=u.id JOIN union_members b ON b.union_id=u.id WHERE u.id <> ?4 AND u.union_type=?3 AND ((a.person_id=?1 AND b.person_id=?2) OR (a.person_id=?2 AND b.person_id=?1))", params![person_a_id, person_b_id, relation_type, relation_id], |row| row.get(0)).map_err(|error| error.to_string())?;
    if duplicate > 0 {
        return Err(
            "Такой брак уже есть в базе. Измените существующую запись вместо создания дубликата."
                .to_owned(),
        );
    }
    if relation_type == "marriage" {
        validate_new_marriage(
            &connection,
            &person_a_id,
            &person_b_id,
            Some(&relation_id),
            details_json.as_deref(),
            allow_existing_marriage.unwrap_or(false),
        )?;
    }
    let sex_a: Option<String> = connection
        .query_row(
            "SELECT sex FROM persons WHERE id=?1",
            params![person_a_id],
            |row| row.get(0),
        )
        .map_err(|error| error.to_string())?;
    let sex_b: Option<String> = connection
        .query_row(
            "SELECT sex FROM persons WHERE id=?1",
            params![person_b_id],
            |row| row.get(0),
        )
        .map_err(|error| error.to_string())?;
    let (left_person, right_person) = if relation_type == "parenthood"
        || (sex_a.as_deref() == Some("male") && sex_b.as_deref() == Some("female"))
    {
        (person_a_id.clone(), person_b_id.clone())
    } else if sex_a.as_deref() == Some("female") && sex_b.as_deref() == Some("male") {
        (person_b_id.clone(), person_a_id.clone())
    } else {
        (person_a_id.clone(), person_b_id.clone())
    };
    connection
        .execute(
            "UPDATE unions SET union_type=?2, details_json=?3 WHERE id=?1",
            params![relation_id, relation_type, details_json],
        )
        .map_err(|error| error.to_string())?;
    connection
        .execute(
            "DELETE FROM union_members WHERE union_id=?1",
            params![relation_id],
        )
        .map_err(|error| error.to_string())?;
    let (left_role, right_role) = if relation_type == "parenthood" {
        let role = details_json
            .as_deref()
            .and_then(|value| serde_json::from_str::<serde_json::Value>(value).ok())
            .and_then(|value| {
                value
                    .get("role")
                    .and_then(serde_json::Value::as_str)
                    .map(str::to_owned)
            })
            .unwrap_or_else(|| "parent".to_owned());
        (role, "child".to_owned())
    } else {
        ("spouse".to_owned(), "spouse".to_owned())
    };
    connection.execute("INSERT INTO union_members (union_id, person_id, role, sequence) VALUES (?1, ?2, ?4, 1), (?1, ?3, ?5, 2)", params![relation_id, left_person, right_person, left_role, right_role]).map_err(|error| error.to_string())?;
    if relation_type == "parenthood" {
        arrange_parenthood_child(&connection, &person_b_id, details_json.as_deref())?;
    }
    Ok(())
}

#[tauri::command]
pub fn delete_relation(
    relation_id: String,
    state: tauri::State<'_, DatabaseState>,
) -> Result<(), String> {
    let mut connection = open_database(&state)?;
    let transaction = connection
        .transaction()
        .map_err(|error| error.to_string())?;
    transaction
        .execute(
            "DELETE FROM union_members WHERE union_id=?1",
            params![relation_id],
        )
        .map_err(|error| error.to_string())?;
    transaction
        .execute("DELETE FROM unions WHERE id=?1", params![relation_id])
        .map_err(|error| error.to_string())?;
    transaction.commit().map_err(|error| error.to_string())
}

#[tauri::command]
pub fn swap_relation_members(
    relation_id: String,
    state: tauri::State<'_, DatabaseState>,
) -> Result<(), String> {
    let connection = open_database(&state)?;
    let transaction = connection
        .unchecked_transaction()
        .map_err(|error| error.to_string())?;
    let members: Vec<(String, f64)> = {
        let mut statement = transaction
            .prepare("SELECT um.person_id, c.x FROM union_members um JOIN canvas_cards c ON c.person_id = um.person_id WHERE um.union_id=?1 ORDER BY um.sequence")
            .map_err(|error| error.to_string())?;
        let rows = statement
            .query_map(params![relation_id], |row| Ok((row.get(0)?, row.get(1)?)))
            .map_err(|error| error.to_string())?;
        rows.collect::<SqlResult<Vec<_>>>()
            .map_err(|error| error.to_string())?
    };
    if members.len() != 2 {
        return Err("Нельзя поменять местами: у связи должны быть ровно две карточки".to_owned());
    }
    transaction
        .execute(
            "UPDATE canvas_cards SET x=?1 WHERE person_id=?2",
            params![members[1].1, members[0].0],
        )
        .map_err(|error| error.to_string())?;
    transaction
        .execute(
            "UPDATE canvas_cards SET x=?1 WHERE person_id=?2",
            params![members[0].1, members[1].0],
        )
        .map_err(|error| error.to_string())?;
    transaction.execute("UPDATE union_members SET sequence = CASE sequence WHEN 1 THEN 2 WHEN 2 THEN 1 ELSE sequence END WHERE union_id=?1", params![relation_id]).map_err(|error| error.to_string())?;
    transaction.commit().map_err(|error| error.to_string())
}

#[tauri::command]
pub fn create_canvas_card(
    input: CreateCanvasCardInput,
    state: tauri::State<'_, DatabaseState>,
) -> Result<String, String> {
    let mut connection = open_database(&state)?;
    let transaction = connection
        .transaction()
        .map_err(|error| error.to_string())?;
    let next_value: i64 = transaction
        .query_row(
            "SELECT next_value FROM id_sequences WHERE name = 'person'",
            [],
            |row| row.get(0),
        )
        .map_err(|error| error.to_string())?;
    if next_value > 9999 {
        return Err("Достигнут предел идентификаторов людей P9999".to_owned());
    }
    let id = format!("P{next_value:04}");
    ensure_person_media_dirs(&state, &id)?;
    transaction
        .execute(
            "UPDATE id_sequences SET next_value = next_value + 1 WHERE name = 'person'",
            [],
        )
        .map_err(|error| error.to_string())?;
    transaction
        .execute("INSERT INTO persons (id) VALUES (?1)", params![id])
        .map_err(|error| error.to_string())?;
    transaction.execute("INSERT INTO canvas_cards (id, person_id, x, y, generation) VALUES (?1, ?1, ?2, ?3, ?4)", params![id, input.x, input.y, input.generation]).map_err(|error| error.to_string())?;
    transaction.commit().map_err(|error| error.to_string())?;
    Ok(id)
}

#[tauri::command]
pub fn update_canvas_card_position(
    id: String,
    x: f64,
    y: f64,
    generation: i32,
    state: tauri::State<'_, DatabaseState>,
) -> Result<(), String> {
    let connection = open_database(&state)?;
    connection.execute("UPDATE canvas_cards SET x = ?1, y = ?2, generation = ?3, updated_at = CURRENT_TIMESTAMP WHERE id = ?4", params![x, y, generation, id]).map_err(|error| error.to_string())?;
    Ok(())
}

#[tauri::command]
pub fn update_canvas_cards_positions(
    positions: Vec<CanvasCardPositionInput>,
    state: tauri::State<'_, DatabaseState>,
) -> Result<(), String> {
    if positions
        .iter()
        .any(|position| !position.x.is_finite() || !position.y.is_finite())
    {
        return Err("Координаты карточки должны быть конечными числами".to_owned());
    }
    let mut connection = open_database(&state)?;
    let transaction = connection
        .transaction()
        .map_err(|error| error.to_string())?;
    for position in positions {
        transaction.execute(
            "UPDATE canvas_cards SET x=?1,y=?2,generation=?3,updated_at=CURRENT_TIMESTAMP WHERE person_id=?4",
            params![position.x, position.y, position.generation, position.id],
        ).map_err(|error| error.to_string())?;
    }
    transaction.commit().map_err(|error| error.to_string())
}

#[tauri::command]
pub fn set_canvas_cards_locked(
    ids: Vec<String>,
    locked: bool,
    state: tauri::State<'_, DatabaseState>,
) -> Result<(), String> {
    let connection = open_database(&state)?;
    if ids.is_empty() {
        connection
            .execute(
                "UPDATE canvas_cards SET is_pinned=?1, updated_at=CURRENT_TIMESTAMP",
                params![locked as i64],
            )
            .map_err(|error| error.to_string())?;
        return Ok(());
    }
    let transaction = connection
        .unchecked_transaction()
        .map_err(|error| error.to_string())?;
    for id in ids {
        transaction
            .execute(
                "UPDATE canvas_cards SET is_pinned=?1, updated_at=CURRENT_TIMESTAMP WHERE id=?2",
                params![locked as i64, id],
            )
            .map_err(|error| error.to_string())?;
    }
    transaction.commit().map_err(|error| error.to_string())
}

#[tauri::command]
pub fn update_person(
    input: PersonInput,
    state: tauri::State<'_, DatabaseState>,
) -> Result<(), String> {
    let mut connection = open_database(&state)?;
    let transaction = connection
        .transaction()
        .map_err(|error| error.to_string())?;
    transaction.execute("UPDATE persons SET sex = ?1, photo_url = ?2, photo_zoom = ?3, photo_offset_x = ?4, photo_offset_y = ?5, gender_identity = ?6, nationality = ?7, ethnicity = ?8, religion = ?9, occupation = ?10, education = ?11, description = ?12, biography = ?13, research_status = ?14, updated_at = CURRENT_TIMESTAMP WHERE id = ?15", params![clean(input.sex), clean(input.photo_url.clone()), input.photo_zoom.unwrap_or(1.0), input.photo_offset_x.unwrap_or(0.0), input.photo_offset_y.unwrap_or(0.0), clean(input.gender_identity), clean(input.nationality), clean(input.ethnicity), clean(input.religion), clean(input.occupation), clean(input.education), clean(input.description), clean(input.biography), clean(input.research_status), input.id]).map_err(|error| error.to_string())?;
    transaction
        .execute(
            "DELETE FROM person_names WHERE person_id = ?1",
            params![input.id],
        )
        .map_err(|error| error.to_string())?;
    let full_name = [
        input.first_name.clone(),
        input.patronymic.clone(),
        input.last_name.clone(),
    ]
    .into_iter()
    .flatten()
    .collect::<Vec<_>>()
    .join(" ");
    if !full_name.is_empty() || input.previous_last_name.is_some() {
        transaction.execute("INSERT INTO person_names (id, person_id, full_text, given_name, patronymic, surname, maiden_name, is_preferred) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, 1)", params![Uuid::new_v4().to_string(), input.id, full_name, clean(input.first_name), clean(input.patronymic), clean(input.last_name), clean(input.previous_last_name)]).map_err(|error| error.to_string())?;
    }
    transaction.execute("DELETE FROM events WHERE id IN (SELECT ep.event_id FROM event_participants ep JOIN events e ON e.id = ep.event_id WHERE ep.person_id = ?1 AND e.event_type IN ('birth', 'death'))", params![input.id]).map_err(|error| error.to_string())?;
    insert_profile_event(
        &transaction,
        &input.id,
        "birth",
        input.birth_date.as_deref(),
    )?;
    insert_profile_event(
        &transaction,
        &input.id,
        "death",
        input.death_date.as_deref(),
    )?;
    transaction.commit().map_err(|error| error.to_string())
}

#[tauri::command]
pub fn delete_canvas_card(
    id: String,
    state: tauri::State<'_, DatabaseState>,
) -> Result<(), String> {
    let mut connection = open_database(&state)?;
    let transaction = connection
        .transaction()
        .map_err(|error| error.to_string())?;
    transaction
        .execute(
            "UPDATE persons SET deleted_at = CURRENT_TIMESTAMP WHERE id = ?1",
            params![id],
        )
        .map_err(|error| error.to_string())?;
    transaction.commit().map_err(|error| error.to_string())?;
    Ok(())
}

#[tauri::command]
pub fn clear_tree(state: tauri::State<'_, DatabaseState>) -> Result<(), String> {
    let connection = open_database(&state)?;
    connection
        .execute(
            "UPDATE persons SET deleted_at = CURRENT_TIMESTAMP WHERE deleted_at IS NULL",
            [],
        )
        .map_err(|error| error.to_string())?;
    Ok(())
}

#[tauri::command]
pub fn clear_relations(state: tauri::State<'_, DatabaseState>) -> Result<(), String> {
    let mut connection = open_database(&state)?;
    let transaction = connection
        .transaction()
        .map_err(|error| error.to_string())?;
    transaction
        .execute("DELETE FROM union_members", [])
        .map_err(|error| error.to_string())?;
    transaction
        .execute("DELETE FROM unions", [])
        .map_err(|error| error.to_string())?;
    transaction.commit().map_err(|error| error.to_string())
}

#[tauri::command]
pub fn restore_canvas_card(
    id: String,
    state: tauri::State<'_, DatabaseState>,
) -> Result<(), String> {
    let connection = open_database(&state)?;
    connection
        .execute(
            "UPDATE persons SET deleted_at = NULL WHERE id = ?1",
            params![id],
        )
        .map_err(|error| error.to_string())?;
    Ok(())
}

#[tauri::command]
pub fn purge_canvas_card(id: String, state: tauri::State<'_, DatabaseState>) -> Result<(), String> {
    let mut connection = open_database(&state)?;
    let media_paths = person_media_paths(&connection, &id)?;
    let transaction = connection
        .transaction()
        .map_err(|error| error.to_string())?;
    transaction
        .execute("DELETE FROM persons WHERE id = ?1", params![id])
        .map_err(|error| error.to_string())?;
    transaction.commit().map_err(|error| error.to_string())?;
    for path in media_paths {
        let _ = fs::remove_file(path);
    }
    remove_person_media_directory(&state, &id);
    Ok(())
}

#[tauri::command]
pub fn purge_canvas_cards(
    ids: Vec<String>,
    state: tauri::State<'_, DatabaseState>,
) -> Result<(), String> {
    let mut connection = open_database(&state)?;
    let mut media_paths = Vec::new();
    for id in &ids {
        media_paths.extend(person_media_paths(&connection, id)?);
    }
    let transaction = connection
        .transaction()
        .map_err(|error| error.to_string())?;
    for id in &ids {
        transaction
            .execute("DELETE FROM persons WHERE id = ?1", params![id])
            .map_err(|error| error.to_string())?;
    }
    transaction.commit().map_err(|error| error.to_string())?;
    for path in media_paths {
        let _ = fs::remove_file(path);
    }
    for id in ids {
        remove_person_media_directory(&state, &id);
    }
    Ok(())
}

#[tauri::command]
pub fn load_trash_cards(
    state: tauri::State<'_, DatabaseState>,
) -> Result<Vec<CanvasCardRecord>, String> {
    let connection = open_database(&state)?;
    let mut statement = connection.prepare("SELECT c.id,c.x,c.y,c.generation,p.photo_url,p.photo_zoom,p.photo_offset_x,p.photo_offset_y,p.sex,(SELECT pn.given_name FROM person_names pn WHERE pn.person_id=p.id AND pn.is_preferred=1 LIMIT 1),(SELECT pn.patronymic FROM person_names pn WHERE pn.person_id=p.id AND pn.is_preferred=1 LIMIT 1),(SELECT pn.surname FROM person_names pn WHERE pn.person_id=p.id AND pn.is_preferred=1 LIMIT 1),(SELECT pn.maiden_name FROM person_names pn WHERE pn.person_id=p.id AND pn.is_preferred=1 LIMIT 1),(SELECT dv.original_text FROM events e JOIN event_participants ep ON ep.event_id=e.id JOIN date_values dv ON dv.id=e.date_id WHERE ep.person_id=p.id AND e.event_type='birth' LIMIT 1),(SELECT dv.original_text FROM events e JOIN event_participants ep ON ep.event_id=e.id JOIN events e2 ON e2.id=ep.event_id JOIN date_values dv ON dv.id=e.date_id WHERE ep.person_id=p.id AND e.event_type='death' LIMIT 1),c.is_pinned FROM canvas_cards c JOIN persons p ON p.id=c.person_id WHERE p.deleted_at IS NOT NULL ORDER BY p.deleted_at DESC").map_err(|error| error.to_string())?;
    let rows = statement
        .query_map([], |row| {
            Ok(CanvasCardRecord {
                id: row.get(0)?,
                x: row.get(1)?,
                y: row.get(2)?,
                generation: row.get(3)?,
                locked: row.get::<_, i64>(15)? != 0,
                photo_url: row.get(4)?,
                photo_zoom: row.get(5)?,
                photo_offset_x: row.get(6)?,
                photo_offset_y: row.get(7)?,
                sex: row.get(8)?,
                gender_identity: None,
                nationality: None,
                ethnicity: None,
                religion: None,
                occupation: None,
                education: None,
                description: None,
                biography: None,
                research_status: None,
                first_name: row.get(9)?,
                patronymic: row.get(10)?,
                last_name: row.get(11)?,
                previous_last_name: row.get(12)?,
                birth_date: row.get(13)?,
                death_date: row.get(14)?,
            })
        })
        .map_err(|error| error.to_string())?;
    rows.collect::<SqlResult<Vec<_>>>()
        .map_err(|error| error.to_string())
}
