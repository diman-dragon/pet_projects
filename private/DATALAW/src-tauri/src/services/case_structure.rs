use std::fs;
use std::path::PathBuf;

use crate::error::AppResult;
use crate::services::path;

pub fn create_case_structure(
    case_path: PathBuf,
    case_type: Option<&str>,
    counterparty: Option<&str>,
    litigation_subtype: Option<&str>,
    metadata: Option<&str>, // YYMM or YYMMDD
) -> AppResult<()> {
    match case_type {
        Some("contract") => create_contract_case_structure(case_path, counterparty, metadata),
        Some("consulting") => create_consulting_case_structure(case_path, metadata),
        Some("litigation") => {
            create_litigation_case_structure(case_path, litigation_subtype)
        }
        Some("other") => create_other_case_structure(case_path),
        _ => create_default_case_structure(case_path),
    }
}

fn create_contract_case_structure(
    case_path: PathBuf,
    _counterparty: Option<&str>,
    _metadata: Option<&str>,
) -> AppResult<()> {
    // Structure: Client -> contracts -> YYMM_NN
    fs::create_dir_all(case_path)?;
    Ok(())
}

fn create_consulting_case_structure(case_path: PathBuf, _metadata: Option<&str>) -> AppResult<()> {
    // Structure: Client -> consultations -> YYMM_NN
    fs::create_dir_all(case_path)?;
    Ok(())
}

fn create_other_case_structure(case_path: PathBuf) -> AppResult<()> {
    // Structure: Client -> other -> YYMM_NN
    fs::create_dir_all(case_path)?;
    Ok(())
}

fn create_litigation_case_structure(_case_path: PathBuf, _litigation_subtype: Option<&str>) -> AppResult<()> {
    // Litigation: no subdirectories, clean folder
    Ok(())
}

fn create_default_case_structure(case_path: PathBuf) -> AppResult<()> {
    // Fallback for other case types: base directory only
    fs::create_dir_all(case_path.join(path::CLIENT_BASE_DIR))?;
    Ok(())
}
