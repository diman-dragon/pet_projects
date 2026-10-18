// Чанкер для судов общей юрисдикции (районные, городские, областные).
// Разбивает по процессуальным блокам: УСТАНОВИЛ / РЕШИЛ / РЕЗОЛЮТИВНАЯ ЧАСТЬ.
use super::breadcrumb::{build_breadcrumb, compose_chunk, trim_header};
use super::splitter::{split_long, finalize};
use super::types::{RawChunk, DEFAULT_MAX_CHARS, DEFAULT_MIN_CHARS};

const MARKERS: &[&str] = &[
    "УСТАНОВИЛ:", "РЕШИЛ:", "ОПРЕДЕЛИЛ:", "ПОСТАНОВИЛ:",
    "МОТИВИРОВОЧНАЯ ЧАСТЬ", "РЕЗОЛЮТИВНАЯ ЧАСТЬ", "ОПИСАТЕЛЬНАЯ ЧАСТЬ",
];

fn detect_marker(line: &str) -> Option<String> {
    let t = line.trim().to_uppercase();
    MARKERS.iter()
        .find(|&&m| t.starts_with(m.trim_end_matches(':')))
        .map(|m| m.trim_end_matches(':').to_string())
}

pub fn chunk_court(text: &str, doc_title: &str, max_chars: usize, min_chars: usize) -> Vec<RawChunk> {
    let max = max_chars.max(DEFAULT_MAX_CHARS);
    let min = min_chars.max(DEFAULT_MIN_CHARS);
    let mut result = Vec::new();
    let mut label = String::new();
    let mut body = String::new();
    let mut char_pos = 0usize;
    let mut block_start = 0usize;

    let flush = |lbl: &str, body: &str, start: usize, res: &mut Vec<RawChunk>| {
        if body.trim().is_empty() { return; }
        let crumb = build_breadcrumb(&[
            Some(trim_header(doc_title)),
            if lbl.is_empty() { None } else { Some(lbl.to_string()) },
        ]);
        let full = compose_chunk(&crumb, "", body);
        if full.chars().count() <= max {
            res.extend(finalize(vec![full], start, min));
        } else {
            for part in split_long(body, max.saturating_sub(crumb.len()), min) {
                res.extend(finalize(vec![compose_chunk(&crumb, "", &part)], start, min));
            }
        }
    };

    for line in text.lines() {
        let line_len = line.chars().count() + 1;
        if let Some(marker) = detect_marker(line) {
            flush(&label, &body, block_start, &mut result);
            label = marker; body.clear(); block_start = char_pos;
        } else {
            body.push_str(line); body.push('\n');
        }
        char_pos += line_len;
    }
    flush(&label, &body, block_start, &mut result);
    result
}
