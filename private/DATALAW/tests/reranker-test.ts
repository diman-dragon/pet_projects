// tests/reranker-test.ts
//
// Simple test for the Rust reranker to verify it works without Tauri GUI.

import { llmRerank } from '../src/services/rag/llmReranker';

async function testReranker() {
    console.log('Testing Rust reranker...');

    const testCandidates = [
        {
            chunkId: 1,
            libraryId: 1,
            title: 'Test Document 1',
            category: 'legal',
            filePath: '/test/doc1.pdf',
            chunkText: 'This is a test document about legal matters and contracts.',
            chunkIndex: 0,
            rrfScore: 0.8,
            matchType: 'hybrid' as const,
            fastRerankScore: 0.7,
        },
        {
            chunkId: 2,
            libraryId: 1,
            title: 'Test Document 2',
            category: 'legal',
            filePath: '/test/doc2.pdf',
            chunkText: 'Another document discussing financial regulations and compliance.',
            chunkIndex: 1,
            rrfScore: 0.6,
            matchType: 'hybrid' as const,
            fastRerankScore: 0.5,
        },
    ];

    try {
        const results = await llmRerank({
            query: 'legal contracts',
            candidates: testCandidates,
            maxCandidates: 5,
        });

        console.log('Reranker test PASSED');
        console.log('Results:', results.map(r => ({
            chunkId: r.chunkId,
            score: r.llmRerankScore?.toFixed(4),
        })));
    } catch (error) {
        console.error('Reranker test FAILED:', error);
        throw error;
    }
}

// Run test if this file is executed directly
if (require.main === module) {
    testReranker().catch(console.error);
}

export { testReranker };