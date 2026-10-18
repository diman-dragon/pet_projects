use rusqlite::{params, Connection, OptionalExtension};

use crate::models::{
    domain::{Appeal, AppealDocument, AppealSummary},
    payloads::AppealChainContext,
};

pub fn list_by_line(conn: &Connection, line_id: i64) -> rusqlite::Result<Vec<Appeal>> {
    let mut stmt = conn.prepare(
        "SELECT id, line_id, title, status, description, file_path, folder_name, agency_code, appeal_date, recipient, incoming_number, outgoing_number, summary, created_at, updated_at
         FROM appeals WHERE line_id = ?1 AND deleted_at IS NULL ORDER BY created_at DESC",
    )?;
    let rows = stmt.query_map(params![line_id], map_appeal)?;
    rows.collect()
}

pub fn get_by_id(conn: &Connection, appeal_id: i64) -> rusqlite::Result<Option<Appeal>> {
    conn.query_row(
        "SELECT id, line_id, title, status, description, file_path, folder_name, agency_code, appeal_date, recipient, incoming_number, outgoing_number, summary, created_at, updated_at
         FROM appeals WHERE id = ?1 AND deleted_at IS NULL",
        params![appeal_id],
        map_appeal,
    )
    .optional()
}

pub fn get_summary(conn: &Connection, appeal_id: i64) -> rusqlite::Result<Option<AppealSummary>> {
    conn.query_row(
        "SELECT id, appeal_id, text, key_points, generated_by, updated_at FROM appeal_summaries WHERE appeal_id = ?1",
        params![appeal_id],
        |row| {
            let raw: Option<String> = row.get(3)?;
            let key_points = raw.and_then(|value| serde_json::from_str(&value).ok());
            Ok(AppealSummary {
                id: row.get(0)?,
                appeal_id: row.get(1)?,
                text: row.get(2)?,
                key_points,
                generated_by: row.get(4)?,
                updated_at: row.get(5)?,
            })
        },
    )
    .optional()
}

pub fn get_documents(conn: &Connection, appeal_id: i64) -> rusqlite::Result<Vec<AppealDocument>> {
    let mut stmt = conn.prepare(
        "SELECT id, appeal_id, file_name, file_path, file_hash, summary, key_points, duplicate_of_path, duplicate_appeal_id, created_at
         FROM appeal_documents WHERE appeal_id = ?1 ORDER BY created_at DESC",
    )?;
    let rows = stmt.query_map(params![appeal_id], |row| {
        let doc = AppealDocument {
            id: row.get(0)?,
            appeal_id: row.get(1)?,
            file_name: row.get(2)?,
            file_path: row.get(3)?,
            file_hash: row.get(4)?,
            summary: row.get(5)?,
            key_points: row.get(6)?,
            duplicate_of_path: row.get(7)?,
            duplicate_appeal_id: row.get(8)?,
            created_at: row.get(9)?,
        };
        // eprintln!("[appeal_repository] get_documents: file_path={}, has_summary={}, summary_len={}", doc.file_path, doc.summary.is_some(), doc.summary.as_ref().map_or(0, |s| s.len()));
        Ok(doc)
    })?;
    rows.collect()
}

pub fn get_chain_context(conn: &Connection, appeal_id: i64) -> rusqlite::Result<Option<AppealChainContext>> {
    conn.query_row(
        "SELECT l.id, c.id, l.title, c.title
         FROM appeals a
         JOIN lines l ON a.line_id = l.id
         JOIN cases c ON l.case_id = c.id
         WHERE a.id = ?1 AND a.deleted_at IS NULL AND l.deleted_at IS NULL AND c.deleted_at IS NULL",
        params![appeal_id],
        |row| {
            Ok(AppealChainContext {
                line_id: row.get(0)?,
                case_id: row.get(1)?,
                line_title: row.get(2)?,
                case_title: row.get(3)?,
            })
        },
    )
    .optional()
}

pub fn get_path_data(conn: &Connection, appeal_id: i64) -> rusqlite::Result<Option<(String, String, String, Option<String>)>> {
    conn.query_row(
        "SELECT cl.folder_name, c.folder_name, l.folder_name, c.case_type
         FROM appeals a
         JOIN lines l ON a.line_id = l.id
         JOIN cases c ON l.case_id = c.id
         JOIN clients cl ON c.client_id = cl.id
         WHERE a.id = ?1",
        params![appeal_id],
        |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
    )
    .optional()
}

pub fn insert(
    conn: &Connection,
    line_id: i64,
    title: &str,
    status: &str,
    agency_code: Option<&str>,
    appeal_date: Option<&str>,
    incoming_number: Option<&str>,
    outgoing_number: Option<&str>,
    folder_name: &str,
) -> rusqlite::Result<i64> {
    conn.execute(
        "INSERT INTO appeals (line_id, title, status, agency_code, appeal_date, incoming_number, outgoing_number, folder_name, created_at, updated_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)",
        params![line_id, title, status, agency_code, appeal_date, incoming_number, outgoing_number, folder_name],
    )?;
    Ok(conn.last_insert_rowid())
}

pub fn update(
    conn: &Connection,
    id: i64,
    title: &str,
    status: &str,
    agency_code: Option<&str>,
    appeal_date: Option<&str>,
    incoming_number: Option<&str>,
    outgoing_number: Option<&str>,
    folder_name: &str,
) -> rusqlite::Result<()> {
    conn.execute(
        "UPDATE appeals
         SET title = ?1, status = ?2, agency_code = ?3, appeal_date = ?4, incoming_number = ?5, outgoing_number = ?6, folder_name = ?7, updated_at = CURRENT_TIMESTAMP
         WHERE id = ?8",
        params![title, status, agency_code, appeal_date, incoming_number, outgoing_number, folder_name, id],
    )?;
    Ok(())
}

pub fn find_duplicate(
    conn: &Connection,
    file_hash: &str,
    current_appeal_id: i64,
) -> rusqlite::Result<Option<(String, i64, String)>> {
    conn.query_row(
        "SELECT d.file_path, d.appeal_id, a.title
         FROM appeal_documents d
         JOIN appeals a ON d.appeal_id = a.id
         WHERE d.file_hash = ?1 AND d.appeal_id != ?2
         LIMIT 1",
        params![file_hash, current_appeal_id],
        |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
    )
    .optional()
}

/// Returns true if a document with this exact file_path already exists for the appeal.
pub fn exists_by_path(conn: &Connection, appeal_id: i64, file_path: &str) -> rusqlite::Result<bool> {
    let count: i64 = conn.query_row(
        "SELECT COUNT(*) FROM appeal_documents WHERE appeal_id = ?1 AND file_path = ?2",
        params![appeal_id, file_path],
        |row| row.get(0),
    )?;
    Ok(count > 0)
}

pub fn insert_document(
    conn: &Connection,
    appeal_id: i64,
    file_name: &str,
    file_path: &str,
    file_hash: &str,
    summary: Option<&str>,
    key_points: Option<&str>,
    duplicate_of_path: Option<&str>,
    duplicate_appeal_id: Option<i64>,
) -> rusqlite::Result<i64> {
    conn.execute(
        "INSERT INTO appeal_documents
         (appeal_id, file_name, file_path, file_hash, summary, key_points, duplicate_of_path, duplicate_appeal_id)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
        params![
            appeal_id,
            file_name,
            file_path,
            file_hash,
            summary,
            key_points,
            duplicate_of_path,
            duplicate_appeal_id
        ],
    )?;
    Ok(conn.last_insert_rowid())
}

pub fn update_appeal_summary(conn: &Connection, appeal_id: i64, summary: &str) -> rusqlite::Result<()> {
    conn.execute(
        "UPDATE appeals SET summary = ?1 WHERE id = ?2",
        params![summary, appeal_id],
    )?;
    Ok(())
}

pub fn update_document_analysis(
    conn: &Connection,
    appeal_id: i64,
    file_path: &str,
    summary: &str,
    key_points: Option<&str>,
) -> rusqlite::Result<()> {
    conn.execute(
        "UPDATE appeal_documents SET summary = ?1, key_points = ?2 WHERE appeal_id = ?3 AND file_path = ?4",
        params![summary, key_points, appeal_id, file_path],
    )?;
    Ok(())
}

fn map_appeal(row: &rusqlite::Row<'_>) -> rusqlite::Result<Appeal> {
    Ok(Appeal {
        id: row.get(0)?,
        line_id: row.get(1)?,
        title: row.get(2)?,
        status: row.get(3)?,
        description: row.get(4)?,
        file_path: row.get(5)?,
        folder_name: row.get(6)?,
        agency_code: row.get(7)?,
        appeal_date: row.get(8)?,
        recipient: row.get(9)?,
        incoming_number: row.get(10)?,
        outgoing_number: row.get(11)?,
        summary: row.get(12)?,
        created_at: row.get(13)?,
        updated_at: row.get(14)?,
    })
}
