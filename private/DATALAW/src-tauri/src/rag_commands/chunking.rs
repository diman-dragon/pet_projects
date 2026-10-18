use regex::Regex;

use super::types::{ChunkTextRequest, TextChunkDto};

const DEFAULT_CHUNK_SIZE: usize = 1200;
const DEFAULT_OVERLAP_SIZE: usize = 150;
const DEFAULT_MIN_CHUNK_SIZE: usize = 300;
const DEFAULT_MAX_CHUNK_CHARS: usize = 1600;

struct Section {
    text: String,
    char_start: usize,
    char_end: usize,
}

fn collapse_blank_lines(text: &str) -> String {
    let mut output = String::with_capacity(text.len());
    let mut newline_streak = 0usize;

    for ch in text.chars() {
        if ch == '\n' {
            newline_streak += 1;
            if newline_streak <= 2 {
                output.push(ch);
            }
        } else {
            newline_streak = 0;
            output.push(ch);
        }
    }

    output
}

#[inline]
fn char_len(s: &str) -> usize {
    s.chars().count()
}

fn char_slice_prefix(s: &str, n: usize) -> &str {
    match s.char_indices().nth(n) {
        Some((byte_pos, _)) => &s[..byte_pos],
        None => s,
    }
}

fn char_slice_suffix(s: &str, n: usize) -> String {
    let total = char_len(s);
    if n >= total {
        return s.to_string();
    }

    match s.char_indices().nth(total - n) {
        Some((byte_pos, _)) => s[byte_pos..].to_string(),
        None => s.to_string(),
    }
}

/// Smart overlap that finds sentence boundary within the overlap window.
/// Looks for the last sentence-ending punctuation (., ?, !) followed by space or newline,
/// or a paragraph break (double newline). Returns the suffix starting from that boundary.
fn smart_overlap(s: &str, max_overlap: usize) -> String {
    let total = char_len(s);
    if max_overlap >= total {
        return s.to_string();
    }

    // Get the overlap window (last max_overlap chars)
    let window_start = total.saturating_sub(max_overlap);
    let window = match s.char_indices().nth(window_start) {
        Some((byte_pos, _)) => &s[byte_pos..],
        None => return s.to_string(),
    };

    // Find the best sentence boundary within the window
    // Priority: double newline > sentence-ending punctuation + space/newline
    let bytes = window.as_bytes();
    let mut best_pos = 0;

    for i in 0..bytes.len().saturating_sub(1) {
        // Check for sentence-ending punctuation followed by space or newline
        let is_sentence_end = matches!(bytes[i], b'.' | b'?' | b'!');
        let is_boundary = is_sentence_end && (matches!(bytes[i + 1], b' ' | b'\n') || i + 1 == bytes.len() - 1);

        // Also check for paragraph break (double newline)
        let is_paragraph = i + 1 < bytes.len() && bytes[i] == b'\n' && bytes[i + 1] == b'\n';

        if is_boundary || is_paragraph {
            // Skip the punctuation and following whitespace
            let mut skip = 1;
            while i + skip < bytes.len() && matches!(bytes[i + skip], b' ' | b'\n') {
                skip += 1;
            }
            best_pos = i + skip;
        }
    }

    if best_pos > 0 {
        // Convert byte position to char position and return suffix from that point
        let prefix_bytes = &window.as_bytes()[..best_pos];
        let prefix_str = std::str::from_utf8(prefix_bytes).unwrap_or("");
        let chars_to_skip = prefix_str.chars().count();
        match window.char_indices().nth(chars_to_skip) {
            Some((byte_pos, _)) => window[byte_pos..].to_string(),
            None => window.to_string(),
        }
    } else {
        // No good boundary found, fall back to simple suffix
        window.to_string()
    }
}

fn build_heading_regexes() -> Vec<Regex> {
    [
        r"(?i)^(статья|article)\s+\d+([.-]\d+)?",
        r"(?i)^(раздел|section)\s+[ivxlcdm\d]+",
        r"(?i)^(глава|chapter)\s+\d+",
        r"(?i)^(пункт|подпункт|paragraph|clause)\s+\d+([.-]\d+)?",
        r"(?i)^§\s*\d+([.-]\d+)?",
        r"(?i)^\d+\.\d+(\.\d+)*\.?\s+\S+",
        r"(?i)^([ivxlcdm]+\.?)\s+\S+",
        r"(?i)^\d{1,2}\.\s+[А-ЯЁA-Z]\S*",
    ]
    .iter()
    .filter_map(|pattern| Regex::new(pattern).ok())
    .collect()
}

fn is_structural_heading(line: &str, regexes: &[Regex]) -> bool {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return false;
    }

    regexes.iter().any(|regex| regex.is_match(trimmed))
}

fn split_structural_sections(text: &str) -> Vec<Section> {
    let regexes = build_heading_regexes();
    let mut sections: Vec<Section> = Vec::new();
    let mut current_text = String::new();
    let mut current_start = 0usize;
    let mut current_end = 0usize;
    let mut char_offset = 0usize;

    for segment in text.split_inclusive('\n') {
        let segment_len = char_len(segment);
        let trimmed = segment.trim();
        let is_heading = is_structural_heading(trimmed, &regexes);

        if is_heading && !current_text.trim().is_empty() {
            sections.push(Section {
                text: current_text.trim().to_string(),
                char_start: current_start,
                char_end: current_end,
            });
            current_text.clear();
            current_start = char_offset;
        } else if current_text.is_empty() {
            current_start = char_offset;
        }

        current_text.push_str(segment);
        current_end = char_offset + segment_len;
        char_offset += segment_len;
    }

    if !current_text.trim().is_empty() {
        sections.push(Section {
            text: current_text.trim().to_string(),
            char_start: current_start,
            char_end: current_end,
        });
    }

    if sections.is_empty() {
        sections.push(Section {
            text: text.trim().to_string(),
            char_start: 0,
            char_end: char_len(text),
        });
    }

    sections
}

fn split_large_section(section: &Section, chunk_size: usize, max_chunk_chars: usize) -> Vec<Section> {
    let paragraphs: Vec<&str> = section
        .text
        .split("\n\n")
        .map(|part| part.trim())
        .filter(|part| !part.is_empty())
        .collect();

    if paragraphs.is_empty() {
        return vec![Section {
            text: char_slice_prefix(&section.text, max_chunk_chars).trim().to_string(),
            char_start: section.char_start,
            char_end: section.char_start + max_chunk_chars.min(char_len(&section.text)),
        }];
    }

    let mut result: Vec<Section> = Vec::new();
    let mut current = String::new();
    let mut current_start = section.char_start;
    let mut consumed = 0usize;

    for paragraph in paragraphs {
        let paragraph_len = char_len(paragraph);
        let separator_len = if current.is_empty() { 0 } else { 2 };

        if !current.is_empty() && char_len(&current) + separator_len + paragraph_len > chunk_size {
            let current_len = char_len(&current);
            result.push(Section {
                text: current.trim().to_string(),
                char_start: current_start,
                char_end: current_start + current_len,
            });
            consumed += current_len;
            current.clear();
            current_start = section.char_start + consumed;
        }

        if !current.is_empty() {
            current.push_str("\n\n");
        }
        current.push_str(paragraph);

        while char_len(&current) > max_chunk_chars {
            let part = char_slice_prefix(&current, max_chunk_chars).trim().to_string();
            let part_len = char_len(&part);
            result.push(Section {
                text: part,
                char_start: current_start,
                char_end: current_start + part_len,
            });
            consumed += part_len;
            current = char_slice_suffix(&current, char_len(&current).saturating_sub(part_len)).trim().to_string();
            current_start = section.char_start + consumed;
        }
    }

    if !current.trim().is_empty() {
        let current_len = char_len(&current);
        result.push(Section {
            text: current.trim().to_string(),
            char_start: current_start,
            char_end: current_start + current_len,
        });
    }

    result
}

fn prepare_sections(text: &str, chunk_size: usize, max_chunk_chars: usize) -> Vec<Section> {
    split_structural_sections(text)
        .into_iter()
        .flat_map(|section| {
            if char_len(&section.text) > max_chunk_chars {
                split_large_section(&section, chunk_size, max_chunk_chars)
            } else {
                vec![section]
            }
        })
        .collect()
}

fn extract_article_ref(text: &str) -> Option<String> {
    let patterns = [
        r"(?i)(?:ст(?:атья)?|art(?:icle)?|§)\s*\.?\s*(\d+(?:\.\d+)?(?:\s*[а-яёa-z])?(?:\s*п\.\s*\d+)?)",
        r"(?i)Rule\s+(\d+)",
        r"(?i)(?:Правило|п(?:ункт)?)\s*(\d+(?:\.\d+)?)",
    ];

    for pattern in patterns {
        if let Ok(re) = Regex::new(pattern) {
            if let Some(cap) = re.captures(text) {
                return Some(cap[0].chars().take(80).collect());
            }
        }
    }
    None
}

fn push_chunk(
    chunks: &mut Vec<TextChunkDto>,
    text: &str,
    char_start: usize,
    char_end: usize,
    min_chunk_size: usize,
    parent_index: Option<usize>,
) -> Option<usize> {
    let trimmed = text.trim();
    if char_len(trimmed) < min_chunk_size {
        return None;
    }

    let index = chunks.len();
    chunks.push(TextChunkDto {
        text: trimmed.to_string(),
        chunk_index: index,
        char_start,
        char_end,
        parent_index,
        article_ref: extract_article_ref(trimmed),
    });
    Some(index)
}

fn pack_sections_into_chunks(
    sections: Vec<Section>,
    chunk_size: usize,
    overlap_size: usize,
    min_chunk_size: usize,
    max_chunk_chars: usize,
) -> Vec<TextChunkDto> {
    let mut chunks: Vec<TextChunkDto> = Vec::new();
    let mut current_text = String::new();
    let mut current_start = 0usize;
    let mut current_end = 0usize;

    for section in sections {
        let section_len = char_len(&section.text);

        if current_text.is_empty() {
            current_start = section.char_start;
        }

        if !current_text.is_empty() && char_len(&current_text) + 2 + section_len > chunk_size {
            push_chunk(&mut chunks, &current_text, current_start, current_end, min_chunk_size, None);

            let overlap = smart_overlap(&current_text, overlap_size);
            current_text = if overlap.trim().is_empty() { String::new() } else { overlap };
            current_start = current_end.saturating_sub(char_len(&current_text));
        }

        if !current_text.is_empty() {
            current_text.push_str("\n\n");
        }
        current_text.push_str(&section.text);
        current_end = section.char_end;

        while char_len(&current_text) > max_chunk_chars {
            let part = char_slice_prefix(&current_text, max_chunk_chars).trim().to_string();
            let part_len = char_len(&part);
            push_chunk(
                &mut chunks,
                &part,
                current_start,
                current_start + part_len,
                min_chunk_size,
                None,
            );
            let overlap = smart_overlap(&part, overlap_size);
            current_text = format!("{}\n\n{}", overlap, char_slice_suffix(&current_text, char_len(&current_text).saturating_sub(part_len)));
            current_text = current_text.trim().to_string();
            current_start = current_start + part_len.saturating_sub(char_len(&overlap));
        }
    }

    push_chunk(&mut chunks, &current_text, current_start, current_end, min_chunk_size, None);
    chunks
}

fn split_into_children(text: &str, child_size: usize) -> Vec<(usize, usize, String)> {
    let mut children = Vec::new();
    let sentences: Vec<&str> = text.split_inclusive(|c| c == '.' || c == '?' || c == '!')
        .collect();
    
    let mut current = String::new();
    let mut current_start = 0;
    let mut offset = 0;

    for sentence in sentences {
        let s_len = char_len(sentence);
        if !current.is_empty() && char_len(&current) + s_len > child_size {
            children.push((current_start, offset, current.trim().to_string()));
            current.clear();
            current_start = offset;
        }
        current.push_str(sentence);
        offset += s_len;
    }

    if !current.trim().is_empty() {
        children.push((current_start, offset, current.trim().to_string()));
    }

    children
}

#[tauri::command]
pub fn rag_chunk_text(request: ChunkTextRequest) -> Vec<TextChunkDto> {
    let normalized = request.text.replace("\r\n", "\n");
    let compact = collapse_blank_lines(normalized.trim());
    if compact.is_empty() {
        return Vec::new();
    }

    let chunk_size = request.chunk_size.unwrap_or(DEFAULT_CHUNK_SIZE);
    let overlap_size = request.overlap_size.unwrap_or(DEFAULT_OVERLAP_SIZE);
    let min_chunk_size = request.min_chunk_size.unwrap_or(DEFAULT_MIN_CHUNK_SIZE);
    let max_chunk_chars = request.max_chunk_chars.unwrap_or(DEFAULT_MAX_CHUNK_CHARS);

    #[cfg(debug_assertions)]
    eprintln!("[CHUNK] Input chars: {}, chunk_size: {}, max_chars: {}", 
        compact.chars().count(), chunk_size, max_chunk_chars);

    let sections = prepare_sections(&compact, chunk_size, max_chunk_chars);
    let parents = pack_sections_into_chunks(
        sections,
        chunk_size,
        overlap_size,
        min_chunk_size,
        max_chunk_chars,
    );
    
    if let Some(child_size) = request.child_size {
        let mut all_chunks = Vec::with_capacity(parents.len() * 4);
        for parent in parents {
            let parent_text = parent.text.clone();
            let p_start = parent.char_start;
            let p_end = parent.char_end;
            
            // Push parent
            let parent_idx = push_chunk(&mut all_chunks, &parent_text, p_start, p_end, min_chunk_size, None);
            
            if let Some(p_idx) = parent_idx {
                let children = split_into_children(&parent_text, child_size);
                for (c_start, c_end, c_text) in children {
                    push_chunk(&mut all_chunks, &c_text, p_start + c_start, p_start + c_end, min_chunk_size / 2, Some(p_idx));
                }
            }
        }
        return all_chunks;
    }
    
    parents
}
