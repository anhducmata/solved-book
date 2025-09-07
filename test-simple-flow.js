/**
 * Simple Flow Test using direct MCP calls
 * Tests the application flow through the MCP interface
 */

import { spawn } from 'child_process';

class SimpleFlowTest {
  constructor() {
    this.mcpProcess = null;
    this.testResults = [];
  }

  async startMCPServer() {
    console.log('🚀 Starting MCP Server...');
    return new Promise((resolve, reject) => {
      this.mcpProcess = spawn('node', ['src/mcp-server.js'], {
        stdio: ['pipe', 'pipe', 'pipe'],
        cwd: process.cwd()
      });

      this.mcpProcess.stderr.on('data', (data) => {
        const output = data.toString();
        if (output.includes('SolvedBook MCP Server running on stdio')) {
          console.log('✅ Server started successfully');
          resolve();
        }
      });

      this.mcpProcess.on('error', reject);
      
      setTimeout(() => reject(new Error('Server startup timeout')), 5000);
    });
  }

  async sendMCPRequest(request) {
    return new Promise((resolve, reject) => {
      let responseBuffer = '';
      
      const timeout = setTimeout(() => {
        reject(new Error('Request timeout'));
      }, 5000);

      const dataHandler = (data) => {
        responseBuffer += data.toString();
        try {
          // Try to parse complete JSON response
          const lines = responseBuffer.split('\n');
          for (const line of lines) {
            if (line.trim()) {
              const response = JSON.parse(line);
              if (response.id === request.id) {
                clearTimeout(timeout);
                this.mcpProcess.stdout.removeListener('data', dataHandler);
                resolve(response);
                return;
              }
            }
          }
        } catch (e) {
          // Continue waiting for complete response
        }
      };

      this.mcpProcess.stdout.on('data', dataHandler);
      this.mcpProcess.stdin.write(JSON.stringify(request) + '\n');
    });
  }

  async testStoreCase() {
    console.log('\n📦 Testing Case Storage...');
    
    const testCases = [
      {
        case_id: 'flow-test-express',
        task: 'How to create a REST API with Express.js',
        tags: ['express', 'api', 'javascript'],
        solution: {
          description: 'Create a REST API using Express.js framework',
          steps: [
            'Install Express: npm install express',
            'Create app.js with basic routes',
            'Configure middleware',
            'Start server'
          ],
          code: 'const express = require("express"); const app = express(); app.listen(3000);',
          difficulty: 'intermediate'
        }
      },
      {
        case_id: 'flow-test-react',
        task: 'React component state management with hooks',
        tags: ['react', 'hooks', 'state'],
        solution: {
          description: 'Manage component state using React hooks',
          steps: [
            'Import useState hook',
            'Initialize state variable',
            'Create update handlers',
            'Use state in JSX'
          ],
          code: 'const [state, setState] = useState(initialValue);',
          difficulty: 'beginner'
        }
      }
    ];

    for (const testCase of testCases) {
      try {
        const request = {
          jsonrpc: '2.0',
          id: Date.now(),
          method: 'tools/call',
          params: {
            name: 'add_case',
            arguments: testCase
          }
        };

        const response = await this.sendMCPRequest(request);
        
        if (response.result) {
          console.log(`✅ Successfully stored: ${testCase.case_id}`);
          this.testResults.push({
            test: 'storage',
            case_id: testCase.case_id,
            status: 'success'
          });
        } else {
          console.log(`❌ Failed to store: ${testCase.case_id}`);
          this.testResults.push({
            test: 'storage',
            case_id: testCase.case_id,
            status: 'failed',
            error: response.error
          });
        }
      } catch (error) {
        console.log(`❌ Error storing ${testCase.case_id}: ${error.message}`);
        this.testResults.push({
          test: 'storage',
          case_id: testCase.case_id,
          status: 'failed',
          error: error.message
        });
      }
    }
  }

  async testRetrieveCase() {
    console.log('\n🔍 Testing Case Retrieval...');

    const retrievalTests = [
      {
        name: 'Get specific case',
        request: {
          name: 'get_case',
          arguments: { case_id: 'flow-test-express' }
        }
      },
      {
        name: 'Search by tags',
        request: {
          name: 'search_cases',
          arguments: { 
            tags: ['express'],
            limit: 5
          }
        }
      },
      {
        name: 'Retrieve with similarity',
        request: {
          name: 'retrieve_cases',
          arguments: {
            query: 'How to build a web API',
            mode: 'non-parametric',
            top_k: 3
          }
        }
      },
      {
        name: 'Get top cases',
        request: {
          name: 'get_top_cases',
          arguments: { limit: 3 }
        }
      }
    ];

    for (const test of retrievalTests) {
      try {
        const request = {
          jsonrpc: '2.0',
          id: Date.now(),
          method: 'tools/call',
          params: test.request
        };

        const response = await this.sendMCPRequest(request);
        
        if (response.result) {
          console.log(`✅ ${test.name}: Success`);
          this.testResults.push({
            test: 'retrieval',
            name: test.name,
            status: 'success'
          });
        } else {
          console.log(`❌ ${test.name}: Failed`);
          this.testResults.push({
            test: 'retrieval',
            name: test.name,
            status: 'failed',
            error: response.error
          });
        }
      } catch (error) {
        console.log(`❌ ${test.name}: Error - ${error.message}`);
        this.testResults.push({
          test: 'retrieval',
          name: test.name,
          status: 'failed',
          error: error.message
        });
      }
    }
  }

  async testQLearning() {
    console.log('\n🎯 Testing Q-Learning...');

    const qTests = [
      { case_id: 'flow-test-express', reward: 0.8 },
      { case_id: 'flow-test-react', reward: 0.9 }
    ];

    for (const test of qTests) {
      try {
        const request = {
          jsonrpc: '2.0',
          id: Date.now(),
          method: 'tools/call',
          params: {
            name: 'reward_case',
            arguments: test
          }
        };

        const response = await this.sendMCPRequest(request);
        
        if (response.result) {
          console.log(`✅ Q-learning update for ${test.case_id}: Success`);
          this.testResults.push({
            test: 'qlearning',
            case_id: test.case_id,
            status: 'success'
          });
        } else {
          console.log(`❌ Q-learning update for ${test.case_id}: Failed`);
          this.testResults.push({
            test: 'qlearning',
            case_id: test.case_id,
            status: 'failed',
            error: response.error
          });
        }
      } catch (error) {
        console.log(`❌ Q-learning error for ${test.case_id}: ${error.message}`);
        this.testResults.push({
          test: 'qlearning',
          case_id: test.case_id,
          status: 'failed',
          error: error.message
        });
      }
    }
  }

  async testFeedback() {
    console.log('\n💬 Testing Feedback System...');

    try {
      const request = {
        jsonrpc: '2.0',
        id: Date.now(),
        method: 'tools/call',
        params: {
          name: 'submit_feedback',
          arguments: {
            case_id: 'flow-test-express',
            feedback_type: 'helpful',
            feedback_data: { quality: 'high', relevance: 'very relevant' }
          }
        }
      };

      const response = await this.sendMCPRequest(request);
      
      if (response.result) {
        console.log('✅ Feedback submission: Success');
        this.testResults.push({
          test: 'feedback',
          status: 'success'
        });
      } else {
        console.log('❌ Feedback submission: Failed');
        this.testResults.push({
          test: 'feedback',
          status: 'failed',
          error: response.error
        });
      }
    } catch (error) {
      console.log(`❌ Feedback error: ${error.message}`);
      this.testResults.push({
        test: 'feedback',
        status: 'failed',
        error: error.message
      });
    }
  }

  generateReport() {
    console.log('\n📊 FLOW TEST REPORT');
    console.log('='.repeat(50));

    const testTypes = ['storage', 'retrieval', 'qlearning', 'feedback'];
    let totalTests = 0;
    let passedTests = 0;

    for (const testType of testTypes) {
      const results = this.testResults.filter(r => r.test === testType);
      const passed = results.filter(r => r.status === 'success').length;
      
      console.log(`\n${testType.toUpperCase()}:`);
      console.log(`  ✅ Passed: ${passed}/${results.length}`);
      
      if (results.length > 0) {
        console.log(`  📊 Success Rate: ${((passed / results.length) * 100).toFixed(1)}%`);
      }
      
      totalTests += results.length;
      passedTests += passed;
    }

    console.log('\n' + '='.repeat(50));
    console.log(`OVERALL: ${passedTests}/${totalTests} tests passed`);
    console.log(`Success Rate: ${totalTests > 0 ? ((passedTests / totalTests) * 100).toFixed(1) : 0}%`);

    // Show failures
    const failures = this.testResults.filter(r => r.status === 'failed');
    if (failures.length > 0) {
      console.log('\n❌ FAILURES:');
      failures.forEach(f => {
        console.log(`  ${f.test}: ${f.case_id || f.name} - ${f.error || 'Unknown error'}`);
      });
    }
  }

  async cleanup() {
    if (this.mcpProcess) {
      this.mcpProcess.kill();
    }
  }

  async runFlowTest() {
    console.log('🧪 Starting Application Flow Test');
    console.log('='.repeat(50));

    try {
      await this.startMCPServer();
      
      // Wait a bit for server to fully initialize
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      await this.testStoreCase();
      await this.testRetrieveCase();
      await this.testQLearning();
      await this.testFeedback();
      
      this.generateReport();
      
    } catch (error) {
      console.error('❌ Flow test failed:', error.message);
    } finally {
      await this.cleanup();
    }
  }
}

// Run the test
const tester = new SimpleFlowTest();
tester.runFlowTest().catch(console.error);
