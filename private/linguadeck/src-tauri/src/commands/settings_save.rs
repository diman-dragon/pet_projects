use rusqlite::params;
use serde_json::Value;
use crate::db::connection::open_connection;
use crate::db::models::SettingsInput;
use crate::db::settings::settings_value;

#[tauri::command]
pub fn settings_save(db_path: String, settings: SettingsInput) -> Result<Value, String> {
    let connection = open_connection(&db_path)?;
    let entries = [
        ("learningLanguage", settings.learning_language.clone()),
        ("nativeLanguage", settings.native_language.clone()),
        ("mediaRoot", settings.media_root.clone().unwrap_or_default()),
        ("studyAlgorithm", settings.study_algorithm.clone()),
        ("learningSteps", settings.learning_steps.clone()),
        ("graduatingInterval", settings.graduating_interval.to_string()),
        ("easyInterval", settings.easy_interval.to_string()),
        ("maxReviewsPerSession", settings.max_reviews_per_session.to_string()),
    ];
    for (key, value) in entries {
        connection
            .execute(
                "INSERT OR REPLACE INTO app_settings (key, value) VALUES (?1, ?2)",
                params![key, value],
            )
            .map_err(|err| err.to_string())?;
    }
    Ok(settings_value(&settings))
}
