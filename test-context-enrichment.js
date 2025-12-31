#!/usr/bin/env node

/**
 * Test script for Context Enrichment functionality
 * Tests the enrich_context tool and ContextEnrichment module
 */

import { ContextEnrichment } from './src/core/context-enrichment.js';
import { initializeDatabase, getDatabase, closeDatabase } from './src/config/database.js';
import { fileURLToPath } from 'url';
import { dirname } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

class ContextEnrichmentTester {
  constructor() {
    this.testResults = {
      passed: 0,
      failed: 0,
      total: 0,
    };
    this.db = null;
  }

  assert(condition, message) {
    this.testResults.total++;
    if (condition) {
      this.testResults.passed++;
      console.log(`  ✅ ${message}`);
      return true;
    } else {
      this.testResults.failed++;
      console.log(`  ❌ ${message}`);
      return false;
    }
  }

  async initialize() {
    console.log('🔧 Initializing test environment...');
    try {
      await initializeDatabase();
      this.db = getDatabase();
      console.log('✅ Database initialized');
    } catch (error) {
      console.error('❌ Database initialization failed:', error);
      throw error;
    }
  }

  async cleanup() {
    if (this.db) {
      await closeDatabase();
    }
  }

  async setupTestCases() {
    console.log('📝 Setting up test cases...');
    const testCases = [
      {
        case_id: 'test-auth-jwt',
        task: 'How to implement JWT authentication in Node.js?',
        tags: ['javascript', 'nodejs', 'authentication', 'jwt'],
        solution: {
          description: 'Complete guide to JWT authentication. First, install jsonwebtoken package. Then generate tokens using jwt.sign(). Verify tokens with jwt.verify(). Store tokens securely in HTTP-only cookies. Handle token expiration properly. Refresh tokens when they expire.',
        }
      },
      {
        case_id: 'test-database-query',
        task: 'How to optimize database queries in PostgreSQL?',
        tags: ['database', 'postgresql', 'performance'],
        solution: {
          description: 'Database optimization techniques. Use indexes on frequently queried columns. Analyze query execution plans with EXPLAIN. Normalize your database schema. Use connection pooling to reduce overhead. Implement pagination for large result sets. Cache frequently accessed data.',
        }
      },
      {
        case_id: 'test-api-design',
        task: 'Best practices for REST API design',
        tags: ['api', 'rest', 'design'],
        solution: {
          description: 'REST API design guidelines. Use proper HTTP methods: GET for retrieval, POST for creation, PUT for updates, DELETE for removal. Return appropriate status codes. Version your API. Implement pagination. Use consistent naming conventions. Document your API endpoints clearly.',
        }
      }
    ];

    for (const testCase of testCases) {
      try {
        // Check if case already exists
        const existing = await this.db.query(
          'SELECT case_id FROM cases WHERE case_id = $1',
          [testCase.case_id]
        );

        if (existing.rows.length === 0) {
          await this.db.query(
            'INSERT INTO cases (case_id, task, tags, solution) VALUES ($1, $2, $3, $4)',
            [
              testCase.case_id,
              testCase.task,
              JSON.stringify(testCase.tags),
              JSON.stringify(testCase.solution)
            ]
          );
          console.log(`  ✓ Added test case: ${testCase.case_id}`);
        } else {
          console.log(`  ⊙ Test case already exists: ${testCase.case_id}`);
        }
      } catch (error) {
        console.error(`  ✗ Failed to add test case ${testCase.case_id}:`, error.message);
      }
    }
  }

  async testContextEnrichmentClass() {
    console.log('\n🧪 Testing ContextEnrichment class...');

    const enrichment = new ContextEnrichment();

    // Test sentence splitting
    const testText = 'This is sentence one. This is sentence two! This is sentence three?';
    const sentences = enrichment.splitIntoSentences(testText);
    this.assert(sentences.length >= 3, `Should split text into sentences (got ${sentences.length})`);

    // Test relevance calculation
    const relevance = enrichment.calculateRelevance('JWT authentication', 'Implement JWT tokens for secure authentication');
    this.assert(relevance > 0, `Should calculate relevance score (got ${relevance})`);

    // Test solution text extraction
    const solutionObj = { description: 'Test solution', answer: 'Test answer' };
    const extractedText = enrichment.extractSolutionText(solutionObj);
    this.assert(extractedText.includes('Test solution'), 'Should extract solution text from object');

    // Test relevant content extraction
    const solutionText = 'Authentication is important. JWT tokens provide secure authentication. Use jwt.sign() to create tokens. Verify with jwt.verify(). Store tokens securely.';
    const extraction = await enrichment.extractRelevantContent(
      solutionText,
      'JWT token authentication',
      { maxSentences: 2 }
    );

    this.assert(extraction.relevantContent.length > 0, 'Should extract relevant content');
    this.assert(extraction.extractedLength <= 2, `Should respect maxSentences limit (got ${extraction.extractedLength})`);
    this.assert(extraction.relevantContent.includes('JWT'), 'Should include relevant sentences containing query keywords');

    return true;
  }

  async testEnrichPromptContext() {
    console.log('\n🧪 Testing enrichPromptContext method...');

    const enrichment = new ContextEnrichment();

    const mockCases = [
      {
        case_id: 'test-1',
        task: 'JWT authentication',
        solution: {
          description: 'Use jsonwebtoken package. Generate tokens with jwt.sign(). Verify tokens with jwt.verify(). Store securely in cookies.',
        },
        tags: ['auth', 'jwt'],
        q_value: 0.5
      },
      {
        case_id: 'test-2',
        task: 'Database optimization',
        solution: {
          description: 'Use indexes on columns. Analyze queries with EXPLAIN. Normalize schema. Use connection pooling.',
        },
        tags: ['database'],
        q_value: 0.3
      }
    ];

    const enrichedContext = await enrichment.enrichPromptContext(
      'How to use JWT tokens?',
      mockCases,
      {
        maxCases: 2,
        maxSentencesPerCase: 2,
        minRelevanceScore: 0.1
      }
    );

    this.assert(enrichedContext.query === 'How to use JWT tokens?', 'Should preserve query');
    this.assert(enrichedContext.enrichedCases.length === 2, 'Should process all cases');
    this.assert(enrichedContext.enrichedCases[0].case_id === 'test-1', 'Should preserve case IDs');
    this.assert(enrichedContext.enrichedCases[0].relevant_content.length > 0, 'Should extract relevant content');

    // First case should have higher relevance to JWT query
    const firstCaseContent = enrichedContext.enrichedCases[0].relevant_content.toLowerCase();
    const hasJWTContent = firstCaseContent.includes('jwt') || firstCaseContent.includes('token');
    this.assert(hasJWTContent, 'Should prioritize JWT-relevant content');

    // Test formatting
    const formatted = enrichment.formatContextForPrompt(enrichedContext, 'text');
    this.assert(formatted.includes('JWT'), 'Formatted context should include query-relevant content');
    this.assert(formatted.includes('test-1'), 'Formatted context should include case IDs');

    return true;
  }

  async testEnrichContextTool() {
    console.log('\n🧪 Testing enrich_context tool integration...');

    // Create a mock similarity calculator (simulating AdvancedAI)
    const mockAI = {
      calculateSemanticSimilarity: (query, text) => {
        const queryWords = query.toLowerCase().split(/\W+/).filter(w => w.length > 2);
        const textWords = text.toLowerCase().split(/\W+/).filter(w => w.length > 2);
        const intersection = queryWords.filter(w => textWords.includes(w));
        const union = [...new Set([...queryWords, ...textWords])];
        return intersection.length / Math.max(union.length, 1);
      }
    };
    const enrichmentWithAI = new ContextEnrichment(mockAI);

    // Get test cases from database
    const result = await this.db.query(
      "SELECT * FROM cases WHERE case_id LIKE 'test-%' ORDER BY case_id"
    );
    const testCases = result.rows.map(row => ({
      case_id: row.case_id,
      task: row.task,
      solution: JSON.parse(row.solution),
      tags: JSON.parse(row.tags || '[]'),
      q_value: row.q_value || 0
    }));

    this.assert(testCases.length >= 2, `Should have test cases in database (got ${testCases.length})`);

    // Test enrichment with specific query
    const query = 'JWT authentication implementation';
    const enrichedContext = await enrichmentWithAI.enrichPromptContext(query, testCases, {
      maxCases: 3,
      maxSentencesPerCase: 2,
      minRelevanceScore: 0.1
    });

    this.assert(enrichedContext.enrichedCases.length > 0, 'Should enrich at least one case');

    // Check that JWT-related case was prioritized
    const jwtCase = enrichedContext.enrichedCases.find(c => c.case_id === 'test-auth-jwt');
    this.assert(!!jwtCase, 'Should include JWT-related case for JWT query');

    if (jwtCase) {
      this.assert(jwtCase.relevant_content.length > 0, 'JWT case should have relevant content extracted');
      console.log(`    📄 Sample enriched content: "${jwtCase.relevant_content.substring(0, 100)}..."`);
    }

    // Test JSON format
    const jsonFormat = enrichmentWithAI.formatContextForPrompt(enrichedContext, 'json');
    this.assert(jsonFormat.includes('"query"'), 'JSON format should include query field');

    return true;
  }

  async runAllTests() {
    console.log('🚀 Starting Context Enrichment Tests\n');
    console.log('='.repeat(60));

    try {
      await this.initialize();
      await this.setupTestCases();

      await this.testContextEnrichmentClass();
      await this.testEnrichPromptContext();
      await this.testEnrichContextTool();

      console.log('\n' + '='.repeat(60));
      console.log('\n📊 Test Results:');
      console.log(`   Total: ${this.testResults.total}`);
      console.log(`   ✅ Passed: ${this.testResults.passed}`);
      console.log(`   ❌ Failed: ${this.testResults.failed}`);

      const successRate = ((this.testResults.passed / this.testResults.total) * 100).toFixed(1);
      console.log(`   Success Rate: ${successRate}%`);

      if (this.testResults.failed === 0) {
        console.log('\n🎉 All tests passed!');
        process.exit(0);
      } else {
        console.log('\n⚠️  Some tests failed. Please review the output above.');
        process.exit(1);
      }
    } catch (error) {
      console.error('\n💥 Test suite crashed:', error);
      process.exit(1);
    } finally {
      await this.cleanup();
    }
  }
}

// Run the tests
const tester = new ContextEnrichmentTester();
tester.runAllTests().catch(error => {
  console.error('\n💥 Test execution failed:', error);
  process.exit(1);
});

export default ContextEnrichmentTester;
