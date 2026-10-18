use rusqlite::Connection;
use serde_json::{json, Value};
use std::collections::HashMap;
use super::models::SettingsInput;

pub fn load_settings(connection: &Connection) -> Result<SettingsInput, String> {
    let mut stmt = connection
        .prepare("SELECT key, value FROM app_settings")
        .map_err(|err| err.to_string())?;
    let rows = stmt
        .query_map([], |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)))
        .map_err(|err| err.to_string())?;
    
    let mut map = HashMap::new();
    for row in rows {
        let (key, value) = row.map_err(|err| err.to_string())?;
        map.insert(key, value);
    }

    Ok(SettingsInput {
        learning_language: map.remove("learningLanguage").unwrap_or_else(|| "English".into()),
        native_language: map.remove("nativeLanguage").unwrap_or_else(|| "Russian".into()),
        media_root: map.remove("mediaRoot").filter(|value| !value.is_empty()),
        study_algorithm: map.remove("studyAlgorithm").unwrap_or_else(|| "sm2".into()),
        learning_steps: map.remove("learningSteps").unwrap_or_else(|| "1,10".into()),
        graduating_interval: map
            .remove("graduatingInterval")
            .and_then(|value| value.parse().ok())
            .unwrap_or(3),
        easy_interval: map
            .remove("easyInterval")
            .and_then(|value| value.parse().ok())
            .unwrap_or(5),
        max_reviews_per_session: map
            .remove("maxReviewsPerSession")
            .and_then(|value| value.parse().ok())
            .unwrap_or(50),
    })
}

pub fn settings_value(settings: &SettingsInput) -> Value {
    json!({
        "learningLanguage": settings.learning_language,
        "nativeLanguage": settings.native_language,
        "mediaRoot": settings.media_root,
        "studyAlgorithm": settings.study_algorithm,
        "learningSteps": settings.learning_steps,
        "graduatingInterval": settings.graduating_interval,
        "easyInterval": settings.easy_interval,
        "maxReviewsPerSession": settings.max_reviews_per_session,
    })
}
