use rusqlite::{params, Connection, OptionalExtension};

use crate::models::domain::{Line, LineDetails};

pub fn list_by_case(conn: &Connection, case_id: i64) -> rusqlite::Result<Vec<Line>> {
    let mut stmt = conn.prepare(
        "SELECT id, case_id, title, status, folder_name, description, created_at, updated_at, consultation_text, consultation_topic, contract_analysis, contract_risks
         FROM lines WHERE case_id = ?1 AND deleted_at IS NULL ORDER BY id ASC",
    )?;
    let rows = stmt.query_map(params![case_id], map_line)?;
    rows.collect()
}

pub fn get_by_id(conn: &Connection, line_id: i64) -> rusqlite::Result<Option<LineDetails>> {
    conn.query_row(
        "SELECT l.id, l.case_id, l.title, l.status, l.folder_name, l.description, l.created_at, l.updated_at,
                c.folder_name, cl.folder_name, l.consultation_text, l.consultation_topic, l.contract_analysis, l.contract_risks
         FROM lines l
         JOIN cases c ON l.case_id = c.id
         JOIN clients cl ON c.client_id = cl.id
         WHERE l.id = ?1 AND l.deleted_at IS NULL",
        params![line_id],
        |row| {
            Ok(LineDetails {
                id: row.get(0)?,
                case_id: row.get(1)?,
                title: row.get(2)?,
                status: row.get(3)?,
                folder_name: row.get(4)?,
                description: row.get(5)?,
                created_at: row.get(6)?,
                updated_at: row.get(7)?,
                case_folder: row.get(8)?,
                client_folder: row.get(9)?,
                consultation_text: row.get(10)?,
                consultation_topic: row.get(11)?,
                contract_analysis: row.get(12)?,
                contract_risks: row.get(13)?,
            })
        },
    )
    .optional()
}

/// Check if a line with the given title already exists for this case (to prevent duplicates)
pub fn get_by_case_and_title(conn: &Connection, case_id: i64, title: &str) -> rusqlite::Result<Option<Line>> {
    let mut stmt = conn.prepare(
        "SELECT id, case_id, title, status, folder_name, description, created_at, updated_at, consultation_text, consultation_topic, contract_analysis, contract_risks
         FROM lines WHERE case_id = ?1 AND title = ?2 AND deleted_at IS NULL LIMIT 1",
    )?;
    let mut rows = stmt.query_map(params![case_id, title], map_line)?;
    rows.next().transpose()
}

pub fn insert(
    conn: &Connection,
    case_id: i64,
    title: &str,
    status: Option<&str>,
    folder_name: &str,
    description: Option<&str>,
    consultation_text: Option<&str>,
    consultation_topic: Option<&str>,
    contract_analysis: Option<&str>,
    contract_risks: Option<&str>,
) -> rusqlite::Result<i64> {
    conn.execute(
        "INSERT INTO lines (case_id, title, status, folder_name, description, consultation_text, consultation_topic, contract_analysis, contract_risks, created_at, updated_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)",
        params![case_id, title, status, folder_name, description, consultation_text, consultation_topic, contract_analysis, contract_risks],
    )?;
    Ok(conn.last_insert_rowid())
}

pub fn update(
    conn: &Connection,
    id: i64,
    title: &str,
    status: Option<&str>,
    description: Option<&str>,
    consultation_text: Option<&str>,
    consultation_topic: Option<&str>,
    contract_analysis: Option<&str>,
    contract_risks: Option<&str>,
) -> rusqlite::Result<()> {
    conn.execute(
        "UPDATE lines SET title = ?1, status = ?2, description = ?3, consultation_text = ?4, consultation_topic = ?5, contract_analysis = ?6, contract_risks = ?7, updated_at = CURRENT_TIMESTAMP WHERE id = ?8",
        params![title, status, description, consultation_text, consultation_topic, contract_analysis, contract_risks, id],
    )?;
    Ok(())
}

pub fn get_appeal_summaries(conn: &Connection, line_id: i64) -> rusqlite::Result<Vec<(String, Option<String>)>> {
    let mut stmt = conn.prepare(
        "SELECT title, summary
         FROM appeals
         WHERE line_id = ?1 AND deleted_at IS NULL
         ORDER BY appeal_date ASC",
    )?;
    let rows = stmt.query_map(params![line_id], |row| Ok((row.get(0)?, row.get(1)?)))?;
    rows.collect()
}

pub fn save_summary(conn: &Connection, line_id: i64, summary: &str) -> rusqlite::Result<()> {
    conn.execute(
        "UPDATE lines SET summary = ?1 WHERE id = ?2",
        params![summary, line_id],
    )?;
    Ok(())
}

fn map_line(row: &rusqlite::Row<'_>) -> rusqlite::Result<Line> {
    Ok(Line {
        id: row.get(0)?,
        case_id: row.get(1)?,
        title: row.get(2)?,
        status: row.get(3)?,
        folder_name: row.get(4)?,
        description: row.get(5)?,
        created_at: row.get(6)?,
        updated_at: row.get(7)?,
        consultation_text: row.get(8)?,
        consultation_topic: row.get(9)?,
        contract_analysis: row.get(10)?,
        contract_risks: row.get(11)?,
    })
}
