use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct CardInput {
    pub front: String,
    pub back: String,
    pub context: Option<String>,
    pub transcription: Option<String>,
    pub etymology: Option<String>,
    pub mnemonic: Option<String>,
    pub notes: Option<String>,
    pub example_1: Option<String>,
    pub example_2: Option<String>,
    pub example_3: Option<String>,
    pub audio_path: Option<String>,
    pub image_path: Option<String>,
    pub video_path: Option<String>,
    pub source: Option<String>,
    pub tags: Option<String>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct SettingsInput {
    #[serde(rename = "learningLanguage")]
    pub learning_language: String,
    #[serde(rename = "nativeLanguage")]
    pub native_language: String,
    #[serde(rename = "mediaRoot")]
    pub media_root: Option<String>,
    #[serde(rename = "studyAlgorithm")]
    pub study_algorithm: String,
    #[serde(rename = "learningSteps")]
    pub learning_steps: String,
    #[serde(rename = "graduatingInterval")]
    pub graduating_interval: i64,
    #[serde(rename = "easyInterval")]
    pub easy_interval: i64,
    #[serde(rename = "maxReviewsPerSession")]
    pub max_reviews_per_session: i64,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct StudyBehaviorInput {
    #[serde(rename = "frontTimeMs")]
    pub front_time_ms: i64,
    #[serde(rename = "totalTimeMs")]
    pub total_time_ms: i64,
    #[serde(rename = "flipsCount")]
    pub flips_count: i64,
    #[serde(rename = "backtracksCount")]
    pub backtracks_count: i64,
    #[serde(rename = "manualWrong")]
    pub manual_wrong: bool,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct GrammarUnitInput {
    pub title: String,
    pub language: String,
    pub level: String,
    pub section: String,
    pub tags: Option<String>,
    pub theory: String,
    pub examples: String,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct GrammarExerciseInput {
    #[serde(rename = "unitId")]
    pub unit_id: i64,
    #[serde(rename = "type")]
    pub exercise_type: String,
    pub prompt: String,
    pub payload: String,
    pub answer: String,
    pub explanation: String,
}
