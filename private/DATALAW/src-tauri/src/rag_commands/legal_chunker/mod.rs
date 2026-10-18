pub mod arbitration;
pub mod arbitration_context;
pub mod arbitration_helpers;
pub mod breadcrumb;
pub mod codex;
pub mod codex_context;
pub mod contract;
pub mod contract_parser;
pub mod court;
pub mod detector;
pub mod detector_helpers;
pub mod splitter;
pub mod splitter_utils;
pub mod types;

use types::{DocType, LegalChunkDto, LegalChunkRequest, DEFAULT_MAX_CHARS, DEFAULT_MIN_CHARS};
use detector::detect_doc_type;
use arbitration::chunk_arbitration;
use codex::chunk_codex;
use contract::chunk_contract;
use court::chunk_court;
use splitter_utils::{extract_article_ref, split_into_children};

#[tauri::command]
pub fn rag_chunk_legal(request: LegalChunkRequest) -> Vec<LegalChunkDto> {
    let text = request.text.replace("\r\n", "\n");
    let text = text.trim();
    if text.is_empty() { return vec![]; }
    let max = request.max_chunk_chars.unwrap_or(DEFAULT_MAX_CHARS);
    let min = request.min_chunk_chars.unwrap_or(DEFAULT_MIN_CHARS);
    let title = &request.doc_title;
    let doc_type = detect_doc_type(text);
    
    let raw = match &doc_type {
        DocType::ArbitrationCourt => chunk_arbitration(text, title, max, min),
        DocType::Codex            => chunk_codex(text, title, max, min),
        DocType::Contract         => chunk_contract(text, title, max, min),
        DocType::Court | DocType::Procedural => chunk_court(text, title, max, min),
        DocType::Generic          => return vec![],
    };

    let type_str = match doc_type {
        DocType::ArbitrationCourt => "arbitration",
        DocType::Codex            => "codex",
        DocType::Contract         => "contract",
        DocType::Court | DocType::Procedural => "court",
        DocType::Generic          => "generic",
    };

    let mut result = Vec::new();
    let child_size = request.child_size;

    for (_i, c) in raw.into_iter().enumerate() {
        let parent_text = c.text.clone();
        let p_start = c.char_start;
        let p_end = c.char_end;
        
        let p_idx = result.len();
        result.push(LegalChunkDto {
            text: parent_text.clone(),
            chunk_index: p_idx,
            char_start: p_start,
            char_end: p_end,
            doc_type: type_str.to_string(),
            parent_index: None,
            article_ref: extract_article_ref(&parent_text),
        });

        if let Some(c_size) = child_size {
            let children = split_into_children(&parent_text, c_size);
            for (cs, ce, ct) in children {
                let c_idx = result.len();
                result.push(LegalChunkDto {
                    text: ct.clone(),
                    chunk_index: c_idx,
                    char_start: p_start + cs,
                    char_end: p_start + ce,
                    doc_type: type_str.to_string(),
                    parent_index: Some(p_idx),
                    article_ref: extract_article_ref(&ct),
                });
            }
        }
    }

    result
}
