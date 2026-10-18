use rusqlite::{params, Connection, OptionalExtension};

use crate::models::domain::TrashItem;

pub fn soft_delete(conn: &Connection, entity_type: &str, entity_id: i64) -> rusqlite::Result<()> {
    conn.execute(
        &format!(
            "UPDATE {} SET deleted_at = CURRENT_TIMESTAMP WHERE id = ?1 AND deleted_at IS NULL",
            table_name(entity_type)
        ),
        params![entity_id],
    )?;
    Ok(())
}

pub fn restore(conn: &Connection, entity_type: &str, entity_id: i64) -> rusqlite::Result<()> {
    conn.execute(
        &format!("UPDATE {} SET deleted_at = NULL WHERE id = ?1", table_name(entity_type)),
        params![entity_id],
    )?;
    Ok(())
}

pub fn purge(conn: &Connection, entity_type: &str, entity_id: i64) -> rusqlite::Result<()> {
    conn.execute(
        &format!("DELETE FROM {} WHERE id = ?1", table_name(entity_type)),
        params![entity_id],
    )?;
    Ok(())
}

pub fn list_all(conn: &Connection) -> rusqlite::Result<Vec<TrashItem>> {
    let mut items = Vec::new();
    items.extend(list_for_table(conn, "client", "clients", "full_name")?);
    items.extend(list_for_table(conn, "case", "cases", "title")?);
    items.extend(list_for_table(conn, "line", "lines", "title")?);
    items.extend(list_for_table(conn, "appeal", "appeals", "title")?);
    items.sort_by(|left, right| right.deleted_at.cmp(&left.deleted_at));
    Ok(items)
}

pub fn is_deleted(conn: &Connection, entity_type: &str, entity_id: i64) -> rusqlite::Result<bool> {
    let value = conn
        .query_row(
            &format!("SELECT deleted_at FROM {} WHERE id = ?1", table_name(entity_type)),
            params![entity_id],
            |row| row.get::<_, Option<String>>(0),
        )
        .optional()?;
    Ok(value.flatten().is_some())
}

pub fn parent_deleted(conn: &Connection, entity_type: &str, entity_id: i64) -> rusqlite::Result<bool> {
    let query = match entity_type {
        "case" => {
            "SELECT cl.deleted_at
             FROM clients cl
             JOIN cases ca ON ca.client_id = cl.id
             WHERE ca.id = ?1"
        }
        "line" => {
            "SELECT ca.deleted_at
             FROM cases ca
             JOIN lines l ON l.case_id = ca.id
             WHERE l.id = ?1"
        }
        "appeal" => {
            "SELECT l.deleted_at
             FROM lines l
             JOIN appeals a ON a.line_id = l.id
             WHERE a.id = ?1"
        }
        _ => return Ok(false),
    };
    let value = conn.query_row(query, params![entity_id], |row| row.get::<_, Option<String>>(0)).optional()?;
    Ok(value.flatten().is_some())
}

pub fn client_path_parts(conn: &Connection, entity_id: i64) -> rusqlite::Result<Option<(String,)>> {
    conn.query_row(
        "SELECT folder_name FROM clients WHERE id = ?1",
        params![entity_id],
        |row| Ok((row.get(0)?,)),
    )
    .optional()
}

pub fn case_path_parts(conn: &Connection, entity_id: i64) -> rusqlite::Result<Option<(String, String)>> {
    conn.query_row(
        "SELECT cl.folder_name, ca.folder_name
         FROM cases ca
         JOIN clients cl ON cl.id = ca.client_id
         WHERE ca.id = ?1",
        params![entity_id],
        |row| Ok((row.get(0)?, row.get(1)?)),
    )
    .optional()
}

pub fn line_path_parts(conn: &Connection, entity_id: i64) -> rusqlite::Result<Option<(String, String, String, Option<String>)>> {
    conn.query_row(
        "SELECT cl.folder_name, ca.folder_name, l.folder_name, ca.case_type
         FROM lines l
         JOIN cases ca ON ca.id = l.case_id
         JOIN clients cl ON cl.id = ca.client_id
         WHERE l.id = ?1",
        params![entity_id],
        |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
    )
    .optional()
}

pub fn appeal_path_parts(conn: &Connection, entity_id: i64) -> rusqlite::Result<Option<(String, String, String, String, Option<String>)>> {
    conn.query_row(
        "SELECT cl.folder_name, ca.folder_name, l.folder_name, a.folder_name, ca.case_type
         FROM appeals a
         JOIN lines l ON l.id = a.line_id
         JOIN cases ca ON ca.id = l.case_id
         JOIN clients cl ON cl.id = ca.client_id
         WHERE a.id = ?1",
        params![entity_id],
        |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?, row.get(4)?)),
    )
    .optional()
}

fn list_for_table(
    conn: &Connection,
    entity_type: &str,
    table_name: &str,
    label_column: &str,
) -> rusqlite::Result<Vec<TrashItem>> {
    let mut stmt = conn.prepare(&format!(
        "SELECT id, {label_column}, deleted_at, folder_name
         FROM {table_name}
         WHERE deleted_at IS NOT NULL
         ORDER BY deleted_at DESC"
    ))?;
    let rows = stmt.query_map([], |row| {
        Ok(TrashItem {
            entity_type: entity_type.to_string(),
            entity_id: row.get(0)?,
            entity_label: row.get::<_, Option<String>>(1)?.unwrap_or_default(),
            deleted_at: row.get(2)?,
            folder_name: row.get(3)?,
        })
    })?;
    rows.collect()
}

fn table_name(entity_type: &str) -> &'static str {
    match entity_type {
        "client" => "clients",
        "case" => "cases",
        "line" => "lines",
        "appeal" => "appeals",
        _ => "clients",
    }
}
