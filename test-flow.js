/**
 * Comprehensive Flow Test for SolvedBook MCP Server
 * Tests the complete workflow: storing solutions → fetching solutions → Q-learning
 */

import { SolvedBookAlgorithms } from './src/core/algorithms.js';
import { initializeDatabase, getDatabase, closeDatabase } from './src/config/database.js';

// Handle ES module imports
import { fileURLToPath } from 'url';
import { dirname } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

class FlowTester {
  constructor() {
    this.algorithms = new SolvedBookAlgorithms({
      learningRate: 0.1,
      qualityThreshold: 0.3,
      maxMemorySize: 100
    });
    this.testResults = {
      storage: [],
      retrieval: [],
      qlearning: [],
      algorithms: []
    };
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
    await closeDatabase();
  }

  // =================== STORAGE FLOW TESTS ===================

  async testStorageFlow() {
    console.log('\n📦 Testing Storage Flow...');
    
    const testCases = [
      {
        case_id: 'flow-test-1',
        task: 'How to implement a REST API with Express.js',
        tags: ['api', 'express', 'backend'],
        solution: {
          description: 'Create a REST API using Express.js framework',
          steps: [
            'Install Express: npm install express',
            'Create app.js file',
            'Define routes with app.get(), app.post(), etc.',
            'Start server with app.listen()'
          ],
          code: `
const express = require('express');
const app = express();

app.get('/api/users', (req, res) => {
  res.json({ users: [] });
});

app.listen(3000, () => {
  console.log('Server running on port 3000');
});
          `.trim(),
          references: ['https://expressjs.com/'],
          difficulty: 'intermediate'
        }
      },
      {
        case_id: 'flow-test-2',
        task: 'How to handle database connections in Node.js',
        tags: ['database', 'nodejs', 'connection'],
        solution: {
          description: 'Manage database connections using connection pooling',
          steps: [
            'Install database driver (e.g., pg for PostgreSQL)',
            'Create connection pool',
            'Use pool.query() for database operations',
            'Handle connection errors properly'
          ],
          code: `
const { Pool } = require('pg');
const pool = new Pool({
  user: 'username',
  host: 'localhost',
  database: 'mydb',
  password: 'password',
  port: 5432,
});

module.exports = pool;
          `.trim(),
          references: ['https://node-postgres.com/'],
          difficulty: 'advanced'
        }
      },
      {
        case_id: 'flow-test-3',
        task: 'React component state management',
        tags: ['react', 'state', 'frontend'],
        solution: {
          description: 'Manage state in React components using hooks',
          steps: [
            'Import useState hook',
            'Initialize state with useState()',
            'Update state with setter function',
            'Use useEffect for side effects'
          ],
          code: `
import React, { useState, useEffect } from 'react';

function MyComponent() {
  const [count, setCount] = useState(0);
  
  useEffect(() => {
    document.title = \`Count: \${count}\`;
  }, [count]);

  return (
    <div>
      <p>Count: {count}</p>
      <button onClick={() => setCount(count + 1)}>
        Increment
      </button>
    </div>
  );
}
          `.trim(),
          references: ['https://reactjs.org/docs/hooks-state.html'],
          difficulty: 'beginner'
        }
      }
    ];

    for (const testCase of testCases) {
      try {
        const sql = `INSERT INTO cases (case_id, task, tags, solution) VALUES ($1, $2, $3, $4)`;
        await this.db.query(sql, [
          testCase.case_id,
          testCase.task,
          JSON.stringify(testCase.tags),
          JSON.stringify(testCase.solution)
        ]);
        
        console.log(`✅ Stored case: ${testCase.case_id}`);
        this.testResults.storage.push({
          case_id: testCase.case_id,
          status: 'success',
          message: 'Successfully stored'
        });
      } catch (error) {
        console.log(`❌ Failed to store case: ${testCase.case_id} - ${error.message}`);
        this.testResults.storage.push({
          case_id: testCase.case_id,
          status: 'failed',
          error: error.message
        });
      }
    }
  }

  // =================== RETRIEVAL FLOW TESTS ===================

  async testRetrievalFlow() {
    console.log('\n🔍 Testing Retrieval Flow...');

    const retrievalTests = [
      {
        name: 'Get specific case by ID',
        test: async () => {
          const result = await this.db.query('SELECT * FROM cases WHERE case_id = $1', ['flow-test-1']);
          return {
            success: result.rows.length > 0,
            data: result.rows[0],
            message: result.rows.length > 0 ? 'Case found' : 'Case not found'
          };
        }
      },
      {
        name: 'Search by tags',
        test: async () => {
          const result = await this.db.query('SELECT * FROM cases WHERE tags LIKE $1', ['%"api"%']);
          return {
            success: result.rows.length > 0,
            data: result.rows,
            message: `Found ${result.rows.length} cases with API tag`
          };
        }
      },
      {
        name: 'Search by keyword',
        test: async () => {
          const result = await this.db.query('SELECT * FROM cases WHERE task ILIKE $1', ['%express%']);
          return {
            success: result.rows.length > 0,
            data: result.rows,
            message: `Found ${result.rows.length} cases with Express keyword`
          };
        }
      },
      {
        name: 'Get top cases by Q-value',
        test: async () => {
          const result = await this.db.query('SELECT * FROM cases ORDER BY q_value DESC LIMIT 3');
          return {
            success: result.rows.length > 0,
            data: result.rows,
            message: `Retrieved ${result.rows.length} top cases`
          };
        }
      }
    ];

    for (const retrievalTest of retrievalTests) {
      try {
        const result = await retrievalTest.test();
        console.log(`✅ ${retrievalTest.name}: ${result.message}`);
        this.testResults.retrieval.push({
          name: retrievalTest.name,
          status: 'success',
          ...result
        });
      } catch (error) {
        console.log(`❌ ${retrievalTest.name}: ${error.message}`);
        this.testResults.retrieval.push({
          name: retrievalTest.name,
          status: 'failed',
          error: error.message
        });
      }
    }
  }

  // =================== SEMANTIC SIMILARITY TESTS ===================

  async testSemanticSimilarity() {
    console.log('\n🧠 Testing Semantic Similarity...');

    const query = 'How to create a web API server';
    const testCases = [
      'How to implement a REST API with Express.js',
      'React component state management',
      'How to handle database connections in Node.js'
    ];

    console.log(`Query: "${query}"`);

    for (const testCase of testCases) {
      const similarity = this.algorithms.calculateSemanticSimilarity(query, testCase);
      console.log(`📊 Similarity with "${testCase}": ${(similarity * 100).toFixed(1)}%`);
      
      this.testResults.algorithms.push({
        type: 'semantic_similarity',
        query,
        case: testCase,
        score: similarity,
        status: 'success'
      });
    }
  }

  // =================== Q-LEARNING FLOW TESTS ===================

  async testQLearningFlow() {
    console.log('\n🎯 Testing Q-Learning Flow...');

    const qLearningTests = [
      {
        case_id: 'flow-test-1',
        reward: 0.8,
        userConfidence: 0.9,
        context: { tags: ['api', 'express'], currentTask: 'Building REST API' }
      },
      {
        case_id: 'flow-test-2',
        reward: 0.6,
        userConfidence: 0.7,
        context: { tags: ['database', 'nodejs'], currentTask: 'Database connection pooling' }
      },
      {
        case_id: 'flow-test-3',
        reward: 0.9,
        userConfidence: 0.95,
        context: { tags: ['react', 'state'], currentTask: 'React state management' }
      }
    ];

    for (const test of qLearningTests) {
      try {
        // Test Q-value update using algorithms
        const initialQ = this.algorithms.getQValue(test.case_id);
        const newQ = this.algorithms.updateQValueMemento(
          test.case_id,
          test.reward,
          test.userConfidence,
          test.context
        );

        // Update database
        await this.db.query(
          'UPDATE cases SET q_value = $1, reward = $2, reward_count = reward_count + 1 WHERE case_id = $3',
          [newQ, test.reward, test.case_id]
        );

        const improvement = newQ - initialQ;
        console.log(`✅ Q-learning update for ${test.case_id}:`);
        console.log(`   Initial Q: ${initialQ.toFixed(3)}`);
        console.log(`   New Q: ${newQ.toFixed(3)} (${improvement >= 0 ? '+' : ''}${improvement.toFixed(3)})`);
        console.log(`   Reward: ${test.reward}`);

        this.testResults.qlearning.push({
          case_id: test.case_id,
          status: 'success',
          initialQ,
          newQ,
          improvement,
          reward: test.reward
        });
      } catch (error) {
        console.log(`❌ Q-learning update failed for ${test.case_id}: ${error.message}`);
        this.testResults.qlearning.push({
          case_id: test.case_id,
          status: 'failed',
          error: error.message
        });
      }
    }
  }

  // =================== EXPERIENCE REPLAY TEST ===================

  async testExperienceReplay() {
    console.log('\n🔄 Testing Experience Replay...');

    try {
      const insights = this.algorithms.performExperienceReplay(10);
      
      console.log(`✅ Experience replay completed:`);
      console.log(`   Average reward: ${insights.avgReward.toFixed(3)}`);
      console.log(`   Success patterns: ${insights.successPatterns.length}`);
      console.log(`   Failure patterns: ${insights.failurePatterns.length}`);
      console.log(`   Q-value updates: ${insights.qValueUpdates}`);

      this.testResults.algorithms.push({
        type: 'experience_replay',
        status: 'success',
        insights
      });
    } catch (error) {
      console.log(`❌ Experience replay failed: ${error.message}`);
      this.testResults.algorithms.push({
        type: 'experience_replay',
        status: 'failed',
        error: error.message
      });
    }
  }

  // =================== ADVANCED MEMENTO SCORING TEST ===================

  async testAdvancedMementoScoring() {
    console.log('\n🚀 Testing Advanced Memento Scoring...');

    const query = 'How to build a scalable web API';
    
    try {
      // Get a case from database
      const result = await this.db.query('SELECT * FROM cases WHERE case_id = $1', ['flow-test-1']);
      if (result.rows.length === 0) {
        throw new Error('Test case not found');
      }

      const caseData = {
        id: result.rows[0].case_id,
        task: result.rows[0].task,
        solution: JSON.parse(result.rows[0].solution),
        tags: JSON.parse(result.rows[0].tags),
        q_value: result.rows[0].q_value
      };

      const scoring = await this.algorithms.calculateAdvancedMementoScore(
        query,
        caseData,
        { history: ['api', 'backend'] }
      );

      console.log(`✅ Advanced scoring for "${caseData.task}":`);
      console.log(`   Final Score: ${(scoring.finalScore * 100).toFixed(1)}%`);
      console.log(`   Semantic: ${(scoring.breakdown.semantic * 100).toFixed(1)}%`);
      console.log(`   Context: ${(scoring.breakdown.context * 100).toFixed(1)}%`);
      console.log(`   Q-Value: ${(scoring.breakdown.qValue * 100).toFixed(1)}%`);
      console.log(`   Confidence: ${(scoring.breakdown.confidence * 100).toFixed(1)}%`);

      this.testResults.algorithms.push({
        type: 'advanced_memento_scoring',
        status: 'success',
        query,
        caseData: caseData.task,
        scoring
      });
    } catch (error) {
      console.log(`❌ Advanced Memento scoring failed: ${error.message}`);
      this.testResults.algorithms.push({
        type: 'advanced_memento_scoring',
        status: 'failed',
        error: error.message
      });
    }
  }

  // =================== DATA INTEGRITY TESTS ===================

  async testDataIntegrity() {
    console.log('\n🔒 Testing Data Integrity...');

    const integrityTests = [
      {
        name: 'JSON parsing of stored solutions',
        test: async () => {
          const result = await this.db.query('SELECT case_id, solution FROM cases LIMIT 3');
          let parsed = 0;
          let errors = 0;

          for (const row of result.rows) {
            try {
              const solution = JSON.parse(row.solution);
              if (solution.description && solution.steps) {
                parsed++;
              }
            } catch (e) {
              errors++;
            }
          }

          return {
            success: errors === 0,
            message: `Parsed ${parsed} solutions successfully, ${errors} errors`
          };
        }
      },
      {
        name: 'Tags array parsing',
        test: async () => {
          const result = await this.db.query('SELECT case_id, tags FROM cases LIMIT 3');
          let parsed = 0;
          let errors = 0;

          for (const row of result.rows) {
            try {
              const tags = JSON.parse(row.tags);
              if (Array.isArray(tags)) {
                parsed++;
              }
            } catch (e) {
              errors++;
            }
          }

          return {
            success: errors === 0,
            message: `Parsed ${parsed} tag arrays successfully, ${errors} errors`
          };
        }
      },
      {
        name: 'Q-value ranges',
        test: async () => {
          const result = await this.db.query('SELECT case_id, q_value FROM cases WHERE q_value IS NOT NULL');
          let validRanges = 0;
          let invalidRanges = 0;

          for (const row of result.rows) {
            if (row.q_value >= 0 && row.q_value <= 1) {
              validRanges++;
            } else {
              invalidRanges++;
            }
          }

          return {
            success: invalidRanges === 0,
            message: `${validRanges} Q-values in valid range, ${invalidRanges} invalid`
          };
        }
      }
    ];

    for (const test of integrityTests) {
      try {
        const result = await test.test();
        console.log(`${result.success ? '✅' : '❌'} ${test.name}: ${result.message}`);
        this.testResults.retrieval.push({
          name: test.name,
          status: result.success ? 'success' : 'failed',
          message: result.message
        });
      } catch (error) {
        console.log(`❌ ${test.name}: ${error.message}`);
        this.testResults.retrieval.push({
          name: test.name,
          status: 'failed',
          error: error.message
        });
      }
    }
  }

  // =================== MAIN TEST RUNNER ===================

  async runAllTests() {
    console.log('🧪 Starting Comprehensive Flow Test for SolvedBook MCP Server');
    console.log('='.repeat(70));

    try {
      await this.initialize();

      // Run all test suites
      await this.testStorageFlow();
      await this.testRetrievalFlow();
      await this.testSemanticSimilarity();
      await this.testQLearningFlow();
      await this.testExperienceReplay();
      await this.testAdvancedMementoScoring();
      await this.testDataIntegrity();

      // Generate summary report
      this.generateSummaryReport();

    } catch (error) {
      console.error('❌ Test execution failed:', error);
    } finally {
      await this.cleanup();
    }
  }

  generateSummaryReport() {
    console.log('\n📊 COMPREHENSIVE TEST REPORT');
    console.log('='.repeat(70));

    const categories = ['storage', 'retrieval', 'qlearning', 'algorithms'];
    let totalTests = 0;
    let passedTests = 0;

    for (const category of categories) {
      const results = this.testResults[category];
      const passed = results.filter(r => r.status === 'success').length;
      const failed = results.length - passed;
      
      console.log(`\n${category.toUpperCase()} TESTS:`);
      console.log(`  ✅ Passed: ${passed}`);
      console.log(`  ❌ Failed: ${failed}`);
      console.log(`  📊 Success Rate: ${((passed / results.length) * 100).toFixed(1)}%`);

      totalTests += results.length;
      passedTests += passed;
    }

    console.log('\n' + '='.repeat(70));
    console.log(`OVERALL RESULTS:`);
    console.log(`  Total Tests: ${totalTests}`);
    console.log(`  Passed: ${passedTests}`);
    console.log(`  Failed: ${totalTests - passedTests}`);
    console.log(`  Success Rate: ${((passedTests / totalTests) * 100).toFixed(1)}%`);

    // Detailed failure analysis
    const failures = [];
    for (const category of categories) {
      this.testResults[category]
        .filter(r => r.status === 'failed')
        .forEach(r => failures.push({ category, ...r }));
    }

    if (failures.length > 0) {
      console.log('\n❌ FAILURES ANALYSIS:');
      failures.forEach(f => {
        console.log(`  ${f.category}: ${f.name || f.case_id || f.type} - ${f.error || f.message}`);
      });
    }

    console.log('\n🏁 Test execution completed!');
  }
}

// Run the tests
const tester = new FlowTester();
tester.runAllTests().catch(console.error);
