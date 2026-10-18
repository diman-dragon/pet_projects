use std::path::{Path, PathBuf};

pub const CLIENTS_DIR: &str = "Clients";
pub const CLIENT_BASE_DIR: &str = "00_BASE";
pub const CONSULTATIONS_DIR: &str = "CONSULT";
pub const OTHER_DIR: &str = "OTHER";
pub const CONTRACTS_DIR: &str = "CONTRACTS";

pub fn case_path(work_dir: PathBuf, client_folder: &str, case_folder: &str) -> PathBuf {
    // Standard case folder under Clients/<ClientFolder>/
    work_dir.join(CLIENTS_DIR).join(client_folder).join(case_folder)
}

pub fn line_path(
    work_dir: PathBuf,
    client_folder: &str,
    case_folder: &str,
    line_folder: &str,
    case_type: Option<&str>,
) -> PathBuf {
    let base = case_path(work_dir, client_folder, case_folder);
    match case_type {
        Some("litigation") | None => base.join(line_folder),
        // For consulting and other, we used to use dedicated directories.
        // But since case_folder already contains the type-based prefix (e.g. CONSULT/2404_01),
        // we can just return the base case path.
        Some("consulting") | Some("contract") | Some("other") => base,
        _ => base.join(line_folder),
    }
}

pub fn appeal_path(
    work_dir: PathBuf,
    client_folder: &str,
    case_folder: &str,
    line_folder: &str,
    appeal_folder: &str,
    case_type: Option<&str>,
) -> PathBuf {
    match case_type {
        // For litigation: appeals go directly under case folder (no line folder)
        Some("litigation") => case_path(work_dir, client_folder, case_folder).join(appeal_folder),
        None => line_path(work_dir, client_folder, case_folder, line_folder, case_type).join(appeal_folder),
        // For consulting and other, we used to return just the base folder.
        // But for consistency with upload_appeal_document, we SHOULD have subfolders.
        // Actually, let's keep it flattened BUT make sure upload_appeal_document handles it.
        Some("consulting") | Some("contract") | Some("other") => line_path(work_dir, client_folder, case_folder, line_folder, case_type),
        _ => line_path(work_dir, client_folder, case_folder, line_folder, case_type).join(appeal_folder),
    }
}

/// Validates that a path is within the work directory or allowed system directories.
/// Used to enforce path security after user selects a work directory.
#[allow(dead_code)]
pub fn is_path_allowed(path: &Path, work_dir: &Path, allowed_system_dirs: &[PathBuf]) -> bool {
    let canonical_path = match path.canonicalize() {
        Ok(p) => p,
        Err(_) => path.to_path_buf(), // Allow non-existent paths for file creation
    };

    let canonical_work_dir = match work_dir.canonicalize() {
        Ok(p) => p,
        Err(_) => return false, // Work dir must exist
    };

    // Check if path is under work_dir
    if canonical_path.starts_with(&canonical_work_dir) {
        return true;
    }

    // Check if path is under allowed system directories (e.g., APPDATA for logs)
    for allowed_dir in allowed_system_dirs {
        if let Ok(canonical_allowed) = allowed_dir.canonicalize() {
            if canonical_path.starts_with(&canonical_allowed) {
                return true;
            }
        }
    }

    false
}
