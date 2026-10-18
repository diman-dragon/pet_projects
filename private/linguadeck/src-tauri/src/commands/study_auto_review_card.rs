use serde_json::Value;
use crate::db::connection::open_connection;
use crate::db::cards::apply_card_review;
use crate::db::utils::infer_behavior_rating;
use crate::db::models::StudyBehaviorInput;

#[tauri::command]
pub fn study_auto_review_card(
    db_path: String,
    card_id: i64,
    metrics: StudyBehaviorInput,
) -> Result<Value, String> {
    let connection = open_connection(&db_path)?;
    let rating = infer_behavior_rating(&metrics);
    apply_card_review(&connection, card_id, rating, Some(&metrics))
}
