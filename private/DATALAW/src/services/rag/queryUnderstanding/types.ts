export interface QueryUnderstandingResult {
  rewritten_query: string;
  expansions: string[];
  intent: 'factual' | 'legal' | 'procedural' | 'biographical' | 'conversational';
  lang: string;
  hyde_snippet?: string;
}

export interface PreprocessedQuery {
  normalized: string;
  intentHint: string;
}
