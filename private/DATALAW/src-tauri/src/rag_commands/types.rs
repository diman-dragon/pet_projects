use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ChunkTextRequest {
    pub text: String,
    pub chunk_size: Option<usize>,
    pub overlap_size: Option<usize>,
    pub min_chunk_size: Option<usize>,
    pub max_chunk_chars: Option<usize>,
    pub child_size: Option<usize>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TextChunkDto {
    pub text: String,
    pub chunk_index: usize,
    pub char_start: usize,
    pub char_end: usize,
    pub parent_index: Option<usize>,
    pub article_ref: Option<String>,
}

#[derive(Deserialize, Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct SearchResultDto {
    pub chunk_id: i64,
    pub library_id: i64,
    pub title: String,
    pub category: String,
    pub file_path: String,
    pub chunk_text: String,
    pub chunk_index: i64,
    pub score: f32,
    pub match_type: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MergeHybridRequest {
    pub vector_results: Vec<SearchResultDto>,
    pub text_results: Vec<SearchResultDto>,
    pub top_k: usize,
    pub cosine_weight: Option<f32>,
    pub keyword_weight: Option<f32>,
}
