use super::breadcrumb::{build_breadcrumb, compose_chunk, trim_header};
use super::contract_parser::parse_section_header;
use super::splitter::{split_long, finalize};
use super::types::{RawChunk, DEFAULT_MAX_CHARS, DEFAULT_MIN_CHARS};

pub fn chunk_contract(text: &str, doc_title: &str, max_chars: usize, min_chars: usize) -> Vec<RawChunk> {
    let max = max_chars.max(DEFAULT_MAX_CHARS);
    let min = min_chars.max(DEFAULT_MIN_CHARS);
    let mut result = Vec::new();
    let mut current = String::new();
    let mut current_section = String::new();
    let mut char_pos: usize = 0;
    let mut current_start: usize = 0;

    let flush = |cur: &str, sec: &str, start: usize, res: &mut Vec<RawChunk>| {
        if cur.trim().is_empty() { return; }
        let crumb = build_breadcrumb(&[Some(trim_header(doc_title)), Some(trim_header(sec))]);
        let full = compose_chunk(&crumb, "", cur);
        if full.chars().count() <= max {
            res.extend(finalize(vec![full], start, min));
        } else {
            for part in split_long(cur, max.saturating_sub(crumb.len()), min) {
                res.extend(finalize(vec![compose_chunk(&crumb, "", &part)], start, min));
            }
        }
    };

    for line in text.lines() {
        let line_len = line.chars().count() + 1;
        if let Some((num, depth)) = parse_section_header(line) {
            if depth == 1 {
                flush(&current, &current_section, current_start, &mut result);
                current.clear(); current_start = char_pos; current_section = num;
            }
        }
        if current.chars().count() + line.chars().count() + 1 > max && !current.is_empty() {
            flush(&current, &current_section, current_start, &mut result);
            current.clear(); current_start = char_pos;
        }
        current.push_str(line); current.push('\n');
        char_pos += line_len;
    }
    flush(&current, &current_section, current_start, &mut result);
    result
}
