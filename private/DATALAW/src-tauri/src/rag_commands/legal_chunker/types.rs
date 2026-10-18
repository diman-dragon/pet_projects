use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq)]
pub enum DocType {
    Codex,           // НПА, кодекс: «Статья N.»
    Contract,        // договор: «1. РАЗДЕЛ», «1.1. пункт»
    ArbitrationCourt,// арбитражный суд: «Арбитражный суд», дело «А41-»
    Court,           // общая юрисдикция: РЕШИЛ/ОПРЕДЕЛИЛ без арбитражных маркеров
    Procedural,      // процессуальный: УСТАНОВИЛ без резолюции
    Generic,         // не распознан → fallback на rag_chunk_text
}

#[derive(Debug, Clone)]
pub struct RawChunk {
    pub text: String,
    pub char_start: usize,
    pub char_end: usize,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LegalChunkRequest {
    pub text: String,
    pub doc_title: String,
    pub max_chunk_chars: Option<usize>,
    pub min_chunk_chars: Option<usize>,
    pub child_size: Option<usize>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct LegalChunkDto {
    pub text: String,
    pub chunk_index: usize,
    pub char_start: usize,
    pub char_end: usize,
    pub doc_type: String,
    pub parent_index: Option<usize>,
    pub article_ref: Option<String>,
}

pub const DEFAULT_MAX_CHARS: usize = 900;
pub const DEFAULT_MIN_CHARS: usize = 80;
pub const OVERLAP_CHARS: usize = 120;
pub const MAX_HEADER_LEN: usize = 100;
