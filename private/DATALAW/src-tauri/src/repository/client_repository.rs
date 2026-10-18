use rusqlite::{params, Connection, OptionalExtension};

use crate::models::domain::Client;

pub fn count_all(conn: &Connection) -> rusqlite::Result<i64> {
    conn.query_row(
        "SELECT COUNT(*) FROM clients",
        [],
        |row| row.get(0),
    )
}

pub fn list_all(conn: &Connection) -> rusqlite::Result<Vec<Client>> {
    let mut stmt = conn.prepare(
        "SELECT id, full_name, folder_name, email, phone, tax_id, type, status, metadata, client_data, summary, created_at, updated_at
         FROM clients
         WHERE deleted_at IS NULL
         ORDER BY full_name ASC",
    )?;
    let rows = stmt.query_map([], map_client)?;
    rows.collect()
}

pub fn get_by_id(conn: &Connection, client_id: i64) -> rusqlite::Result<Option<Client>> {
    conn.query_row(
        "SELECT id, full_name, folder_name, email, phone, tax_id, type, status, metadata, client_data, summary, created_at, updated_at
         FROM clients WHERE id = ?1 AND deleted_at IS NULL",
        params![client_id],
        map_client,
    )
    .optional()
}

pub fn insert(
    conn: &Connection,
    full_name: &str,
    tax_id: Option<&str>,
    email: Option<&str>,
    client_type: &str,
    status: &str,
    folder_name: &str,
    metadata: Option<&str>,
) -> rusqlite::Result<i64> {
    conn.execute(
        "INSERT INTO clients (full_name, tax_id, email, type, status, folder_name, metadata, created_at, updated_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)",
        params![full_name, tax_id, email, client_type, status, folder_name, metadata],
    )?;
    Ok(conn.last_insert_rowid())
}

pub fn update(
    conn: &Connection,
    client_id: i64,
    full_name: &str,
    tax_id: Option<&str>,
    email: Option<&str>,
    client_type: &str,
    status: &str,
    folder_name: &str,
    metadata: Option<&str>,
) -> rusqlite::Result<()> {
    conn.execute(
        "UPDATE clients
         SET full_name = ?1, tax_id = ?2, email = ?3, type = ?4, status = ?5, folder_name = ?6, metadata = ?7, updated_at = CURRENT_TIMESTAMP
         WHERE id = ?8",
        params![full_name, tax_id, email, client_type, status, folder_name, metadata, client_id],
    )?;
    Ok(())
}

pub fn get_folder_name(conn: &Connection, client_id: i64) -> rusqlite::Result<Option<String>> {
    conn.query_row(
        "SELECT folder_name FROM clients WHERE id = ?1 AND deleted_at IS NULL",
        params![client_id],
        |row| row.get(0),
    )
    .optional()
}

pub fn get_cases_for_summary(
    conn: &Connection,
    client_id: i64,
) -> rusqlite::Result<Vec<(String, String, Option<String>)>> {
    let mut stmt = conn.prepare(
        "SELECT title, status, summary
         FROM cases
         WHERE client_id = ?1 AND deleted_at IS NULL
         ORDER BY created_at ASC",
    )?;
    let rows = stmt.query_map(params![client_id], |row| {
        Ok((row.get(0)?, row.get(1)?, row.get(2)?))
    })?;
    rows.collect()
}

pub fn save_summary(conn: &Connection, client_id: i64, summary: &str) -> rusqlite::Result<()> {
    conn.execute(
        "UPDATE clients SET summary = ?1 WHERE id = ?2",
        params![summary, client_id],
    )?;
    Ok(())
}

pub fn save_dossier_enrichment(
    conn: &Connection,
    client_id: i64,
    tax_id: &str,
    client_data: &str,
    metadata: &str,
) -> rusqlite::Result<()> {
    conn.execute(
        "UPDATE clients
         SET tax_id = COALESCE(NULLIF(tax_id, ''), ?1),
             client_data = ?2,
             metadata = ?3,
             updated_at = CURRENT_TIMESTAMP
         WHERE id = ?4",
        params![tax_id, client_data, metadata, client_id],
    )?;
    Ok(())
}

fn map_client(row: &rusqlite::Row<'_>) -> rusqlite::Result<Client> {
    Ok(Client {
        id: row.get(0)?,
        full_name: row.get(1)?,
        folder_name: row.get(2)?,
        email: row.get(3)?,
        phone: row.get(4)?,
        tax_id: row.get(5)?,
        client_type: row.get(6)?,
        status: row.get(7)?,
        metadata: row.get(8)?,
        client_data: row.get(9)?,
        summary: row.get(10)?,
        created_at: row.get(11)?,
        updated_at: row.get(12)?,
    })
}
