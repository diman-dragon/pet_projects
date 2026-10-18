use super::arbitration_context::ArbContext;
use super::arbitration_helpers::*;
use super::breadcrumb::compose_chunk;
use super::splitter::{split_long, finalize};
use super::types::{RawChunk, DEFAULT_MAX_CHARS, DEFAULT_MIN_CHARS};

pub fn chunk_arbitration(text: &str, doc_title: &str, max_chars: usize, min_chars: usize) -> Vec<RawChunk> {
    let max = max_chars.max(DEFAULT_MAX_CHARS);
    let min = min_chars.max(DEFAULT_MIN_CHARS);
    let ctx = ArbContext::new(doc_title, &extract_court_name(text),
                               extract_case_number(text).unwrap_or_default());

    let flush = |label: &str, party: Option<&str>, body: &str, start: usize, res: &mut Vec<RawChunk>| {
        if body.trim().is_empty() { return; }
        let crumb = ctx.crumb(label, party);
        let full = compose_chunk(&crumb, "", body);
        if full.chars().count() <= max {
            res.extend(finalize(vec![full], start, min));
        } else {
            for part in split_long(body, max.saturating_sub(crumb.len()), min) {
                res.extend(finalize(vec![compose_chunk(&crumb, "", &part)], start, min));
            }
        }
    };

    let mut result = Vec::new();
    let (mut label, mut body, mut party) = ("ШАПКА".to_string(), String::new(), None::<String>);
    let (mut in_party, mut char_pos, mut block_start) = (false, 0usize, 0usize);

    for line in text.lines() {
        let ll = line.chars().count() + 1;
        if let Some(m) = is_operative_marker(line) {
            flush(&label, party.as_deref(), &body, block_start, &mut result);
            label = m; body.clear(); block_start = char_pos; in_party = false; party = None;
        } else if let Some(p) = is_party_instruction(line) {
            if in_party { flush(&label, party.as_deref(), &body, block_start, &mut result); body.clear(); block_start = char_pos; }
            else { flush(&label, None, &body, block_start, &mut result); label = "Подготовка к заседанию".to_string(); body.clear(); block_start = char_pos; in_party = true; }
            party = Some(p);
        }
        body.push_str(line); body.push('\n'); char_pos += ll;
    }
    flush(&label, party.as_deref(), &body, block_start, &mut result);
    result
}
