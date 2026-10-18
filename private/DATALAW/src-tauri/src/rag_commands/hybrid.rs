use std::cmp::Ordering;
use std::collections::BTreeMap;

use super::math::clamp_relevance;
use super::types::{MergeHybridRequest, SearchResultDto};

const DEFAULT_COSINE_WEIGHT: f32 = 1.0;
const DEFAULT_KEYWORD_WEIGHT: f32 = 1.0;

const RRF_K: f32 = 60.0; // Reciprocal Rank Fusion constant

#[tauri::command]
pub fn rag_merge_hybrid(request: MergeHybridRequest) -> Vec<SearchResultDto> {
    let vector_weight = request.cosine_weight.unwrap_or(DEFAULT_COSINE_WEIGHT);
    let text_weight = request.keyword_weight.unwrap_or(DEFAULT_KEYWORD_WEIGHT);
    let mut grouped = BTreeMap::<i64, (Option<SearchResultDto>, Option<SearchResultDto>)>::new();

    // Group results by chunk_id
    for result in &request.vector_results {
        grouped.insert(result.chunk_id, (Some(result.clone()), None));
    }

    for result in &request.text_results {
        grouped
            .entry(result.chunk_id)
            .and_modify(|entry| entry.1 = Some(result.clone()))
            .or_insert((None, Some(result.clone())));
    }

    let mut merged = grouped
        .into_values()
        .map(|(vector_result, text_result)| {
            // Calculate ranks (1-based)
            let vector_rank = if vector_result.is_some() {
                // Find rank in original vector results
                request.vector_results.iter().position(|r| r.chunk_id == vector_result.as_ref().unwrap().chunk_id)
                    .map(|pos| pos as f32 + 1.0).unwrap_or(request.vector_results.len() as f32 + 1.0)
            } else {
                request.vector_results.len() as f32 + 1.0
            };

            let text_rank = if text_result.is_some() {
                // Find rank in original text results
                request.text_results.iter().position(|r| r.chunk_id == text_result.as_ref().unwrap().chunk_id)
                    .map(|pos| pos as f32 + 1.0).unwrap_or(request.text_results.len() as f32 + 1.0)
            } else {
                request.text_results.len() as f32 + 1.0
            };

            // RRF score calculation
            let vector_rrf = if vector_rank <= request.vector_results.len() as f32 {
                vector_weight * (1.0 / (RRF_K + vector_rank))
            } else {
                0.0
            };
            let text_rrf = if text_rank <= request.text_results.len() as f32 {
                text_weight * (1.0 / (RRF_K + text_rank))
            } else {
                0.0
            };

            let rrf_score = vector_rrf + text_rrf;

            // Use the available result as base, update score to RRF score
            let is_hybrid = vector_result.is_some() && text_result.is_some();
            let mut base = vector_result.or(text_result).expect("hybrid result missing base");
            base.score = clamp_relevance(rrf_score);

            if is_hybrid {
                base.match_type = "hybrid".to_string();
            }

            base
        })
        .collect::<Vec<_>>();

    // Sort by RRF score descending
    merged.sort_by(|left, right| right.score.partial_cmp(&left.score).unwrap_or(Ordering::Equal));
    merged.truncate(request.top_k);
    merged
}
