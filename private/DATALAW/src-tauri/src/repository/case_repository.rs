use rusqlite::{params, Connection, OptionalExtension};

use crate::models::domain::Case;

pub fn count_by_type_in_month(
    conn: &Connection,
    client_id: Option<i64>,
    case_type_folder: &str,
    year_month: &str, // YYMM
) -> rusqlite::Result<i64> {
    // We check folder_name starts with type/YYMM
    // The format is type/YYMMDD_NN_STATUS
    let pattern = format!("{}/{}%", case_type_folder, year_month);
    match client_id {
        Some(id) => conn.query_row(
            "SELECT COUNT(*) FROM cases WHERE client_id = ?1 AND folder_name LIKE ?2 AND deleted_at IS NULL",
            params![id, pattern],
            |row| row.get(0),
        ),
        None => conn.query_row(
            "SELECT COUNT(*) FROM cases WHERE folder_name LIKE ?1 AND deleted_at IS NULL",
            params![pattern],
            |row| row.get(0),
        ),
    }
}

pub fn list_by_client(conn: &Connection, client_id: i64) -> rusqlite::Result<Vec<Case>> {
    let mut stmt = conn.prepare(
        "SELECT id, client_id, title, status, folder_name, case_number, case_type, litigation_subtype, counterparty, metadata, summary, contract_date, contract_type, contract_amount, metadata_description, created_at, updated_at
         FROM cases WHERE client_id = ?1 AND deleted_at IS NULL ORDER BY id DESC",
    )?;
    let rows = stmt.query_map(params![client_id], map_case)?;
    rows.collect()
}

pub fn get_by_id(conn: &Connection, case_id: i64) -> rusqlite::Result<Option<Case>> {
    conn.query_row(
        "SELECT id, client_id, title, status, folder_name, case_number, case_type, litigation_subtype, counterparty, metadata, summary, contract_date, contract_type, contract_amount, metadata_description, created_at, updated_at
         FROM cases WHERE id = ?1 AND deleted_at IS NULL",
        params![case_id],
        map_case,
    )
    .optional()
}

pub fn insert(
    conn: &Connection,
    client_id: i64,
    title: &str,
    status: &str,
    case_type: Option<&str>,
    litigation_subtype: Option<&str>,
    counterparty: Option<&str>,
    case_number: Option<&str>,
    metadata: Option<&str>,
    folder_name: &str,
    summary: Option<&str>,
    contract_date: Option<&str>,
    contract_type: Option<&str>,
    contract_amount: Option<&str>,
    metadata_description: Option<&str>,
) -> rusqlite::Result<i64> {
    conn.execute(
        "INSERT INTO cases (client_id, title, status, case_type, litigation_subtype, counterparty, case_number, folder_name, metadata, summary, contract_date, contract_type, contract_amount, metadata_description, created_at, updated_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)",
        params![client_id, title, status, case_type, litigation_subtype, counterparty, case_number, folder_name, metadata, summary, contract_date, contract_type, contract_amount, metadata_description],
    )?;
    Ok(conn.last_insert_rowid())
}

pub fn update(
    conn: &Connection,
    id: i64,
    title: &str,
    status: &str,
    case_type: Option<&str>,
    litigation_subtype: Option<&str>,
    counterparty: Option<&str>,
    case_number: Option<&str>,
    metadata: Option<&str>,
    folder_name: &str,
    summary: Option<&str>,
    contract_date: Option<&str>,
    contract_type: Option<&str>,
    contract_amount: Option<&str>,
    metadata_description: Option<&str>,
) -> rusqlite::Result<()> {
    conn.execute(
        "UPDATE cases
         SET title = ?1, status = ?2, case_type = ?3, litigation_subtype = ?4, counterparty = ?5, case_number = ?6, metadata = ?7, folder_name = ?8, summary = ?9, contract_date = ?10, contract_type = ?11, contract_amount = ?12, metadata_description = ?13, updated_at = CURRENT_TIMESTAMP
         WHERE id = ?14",
        params![title, status, case_type, litigation_subtype, counterparty, case_number, metadata, folder_name, summary, contract_date, contract_type, contract_amount, metadata_description, id],
    )?;
    Ok(())
}

pub fn get_line_summaries(conn: &Connection, case_id: i64) -> rusqlite::Result<Vec<(String, Option<String>)>> {
    let mut stmt = conn.prepare(
        "SELECT title, summary
         FROM lines
         WHERE case_id = ?1 AND deleted_at IS NULL
         ORDER BY position_index ASC",
    )?;
    let rows = stmt.query_map(params![case_id], |row| Ok((row.get(0)?, row.get(1)?)))?;
    rows.collect()
}

pub fn save_summary(conn: &Connection, case_id: i64, summary: &str) -> rusqlite::Result<()> {
    conn.execute(
        "UPDATE cases SET summary = ?1 WHERE id = ?2",
        params![summary, case_id],
    )?;
    Ok(())
}

fn map_case(row: &rusqlite::Row<'_>) -> rusqlite::Result<Case> {
    Ok(Case {
        id: row.get(0)?,
        client_id: row.get(1)?,
        title: row.get(2)?,
        status: row.get(3)?,
        folder_name: row.get(4)?,
        case_number: row.get(5)?,
        case_type: row.get(6)?,
        litigation_subtype: row.get(7)?,
        counterparty: row.get(8)?,
        metadata: row.get(9)?,
        summary: row.get(10)?,
        contract_date: row.get(11)?,
        contract_type: row.get(12)?,
        contract_amount: row.get(13)?,
        metadata_description: row.get(14)?,
        created_at: row.get(15)?,
        updated_at: row.get(16)?,
    })
}
