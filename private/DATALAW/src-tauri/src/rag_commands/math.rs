fn normalize_vector(vector: &[f32]) -> Vec<f32> {
    let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm < 1e-12 {
        return vec![0.0; vector.len()];
    }
    vector.iter().map(|value| value / norm).collect()
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.is_empty() || b.is_empty() || a.len() != b.len() {
        return 0.0;
    }

    a.iter()
        .zip(b.iter())
        .map(|(x, y)| x * y)
        .sum::<f32>()
        .clamp(-1.0, 1.0)
}

#[tauri::command]
pub fn rag_normalize(vector: Vec<f32>) -> Vec<f32> {
    normalize_vector(&vector)
}

#[tauri::command]
pub fn rag_cosine_similarity(a: Vec<f32>, b: Vec<f32>) -> f32 {
    cosine_similarity(&a, &b)
}

#[tauri::command]
pub fn rag_score_vectors(query_vector: Vec<f32>, candidate_vectors: Vec<Vec<f32>>) -> Vec<f32> {
    let normalized_query = normalize_vector(&query_vector);

    candidate_vectors
        .iter()
        .map(|candidate| cosine_similarity(&normalized_query, &normalize_vector(candidate)))
        .collect()
}

pub fn clamp_relevance(score: f32) -> f32 {
    score.clamp(0.0, 1.0)
}
