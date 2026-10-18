use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientDossierSnapshotRecord {
    pub id: i64,
    pub client_id: i64,
    pub tax_id: String,
    pub status: String,
    pub providers_json: String,
    pub summary_text: Option<String>,
    pub folder_path: String,
    pub manifest_path: String,
    pub dossier_json: Option<String>,
    pub created_at: Option<String>,
    pub updated_at: Option<String>,
}

pub fn list_by_client(conn: &Connection, client_id: i64) -> rusqlite::Result<Vec<ClientDossierSnapshotRecord>> {
    let mut stmt = conn.prepare(
        "SELECT id, client_id, tax_id, status, providers_json, summary_text, folder_path, manifest_path, dossier_json, created_at, updated_at
         FROM client_dossier_snapshots
         WHERE client_id = ?1
         ORDER BY created_at DESC, id DESC",
    )?;

    let rows = stmt.query_map(params![client_id], map_snapshot)?;
    rows.collect()
}

pub fn insert_snapshot(
    conn: &Connection,
    client_id: i64,
    tax_id: &str,
    status: &str,
    providers_json: &str,
    summary_text: &str,
    folder_path: &str,
    manifest_path: &str,
    dossier_json: &str,
) -> rusqlite::Result<i64> {
    conn.execute(
        "INSERT INTO client_dossier_snapshots
         (client_id, tax_id, status, providers_json, summary_text, folder_path, manifest_path, dossier_json, created_at, updated_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)",
        params![client_id, tax_id, status, providers_json, summary_text, folder_path, manifest_path, dossier_json],
    )?;

    Ok(conn.last_insert_rowid())
}

fn map_snapshot(row: &rusqlite::Row<'_>) -> rusqlite::Result<ClientDossierSnapshotRecord> {
    Ok(ClientDossierSnapshotRecord {
        id: row.get(0)?,
        client_id: row.get(1)?,
        tax_id: row.get(2)?,
        status: row.get(3)?,
        providers_json: row.get(4)?,
        summary_text: row.get(5)?,
        folder_path: row.get(6)?,
        manifest_path: row.get(7)?,
        dossier_json: row.get(8)?,
        created_at: row.get(9)?,
        updated_at: row.get(10)?,
    })
}
