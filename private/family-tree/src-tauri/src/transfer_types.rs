use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize)]
pub struct ImportResult {
    pub added: usize,
    pub skipped_duplicates: usize,
    pub replaced: bool,
    pub relations_added: usize,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TransferPerson {
    pub id: String,
    pub first_name: Option<String>,
    pub patronymic: Option<String>,
    pub last_name: Option<String>,
    pub previous_last_name: Option<String>,
    pub sex: Option<String>,
    pub birth_date: Option<String>,
    pub death_date: Option<String>,
    pub photo_url: Option<String>,
    #[serde(default)]
    pub x: Option<f64>,
    #[serde(default)]
    pub y: Option<f64>,
    #[serde(default)]
    pub generation: Option<i32>,
    #[serde(default)]
    pub gender_identity: Option<String>,
    #[serde(default)]
    pub nationality: Option<String>,
    #[serde(default)]
    pub ethnicity: Option<String>,
    #[serde(default)]
    pub religion: Option<String>,
    #[serde(default)]
    pub occupation: Option<String>,
    #[serde(default)]
    pub education: Option<String>,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub biography: Option<String>,
    #[serde(default)]
    pub research_status: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TransferRelation {
    pub id: Option<String>,
    pub relation_type: String,
    pub person_a_id: String,
    pub person_b_id: String,
    pub details_json: Option<String>,
}
