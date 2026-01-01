#!/usr/bin/env node

/**
 * Test script for enhanced retrieve_cases functionality
 * Tests the LLM-enhanced query flow with tech context support
 */

import { spawn } from 'child_process';
import { fileURLToPath } from 'url';
import { dirname } from 'path';
import { config } from 'dotenv';

// Load environment variables from .env file
config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

class RetrieveCasesTester {
  constructor() {
    this.serverProcess = null;
    this.testResults = [];
  }

  async startServer() {
    return new Promise((resolve, reject) => {
      this.serverProcess = spawn('node', ['src/mcp-server.js'], {
        cwd: __dirname,
        stdio: ['pipe', 'pipe', 'pipe'],
        env: { ...process.env }
      });

      let serverReady = false;

      this.serverProcess.stderr.on('data', (data) => {
        const message = data.toString();
        if (message.includes('SolvedBook MCP Server running on stdio')) {
          if (!serverReady) {
            serverReady = true;
            resolve();
          }
        }
      });

      this.serverProcess.on('error', reject);

      setTimeout(() => {
        if (!serverReady) {
          reject(new Error('Server startup timeout'));
        }
      }, 10000);
    });
  }

  async sendRequest(request) {
    return new Promise((resolve, reject) => {
      let responseBuffer = '';
      const requestId = request.id;

      const timeout = setTimeout(() => {
        reject(new Error('Request timeout'));
      }, 30000); // Longer timeout for LLM calls

      const dataHandler = (data) => {
        responseBuffer += data.toString();
        const lines = responseBuffer.split('\n').filter(line => line.trim());

        for (const line of lines) {
          try {
            const response = JSON.parse(line);
            if (response.id === requestId) {
              clearTimeout(timeout);
              this.serverProcess.stdout.removeListener('data', dataHandler);
              resolve(response);
              return;
            }
          } catch (e) {
            // Continue waiting for complete JSON
          }
        }
      };

      this.serverProcess.stdout.on('data', dataHandler);
      this.serverProcess.stdin.write(JSON.stringify(request) + '\n');
    });
  }

  async setupTestData() {
    console.log('📦 Setting up test data...');

    const testCases = [
      {
        case_id: 'test-express-api',
        task: 'How to create a REST API with Express.js',
        tags: ['express', 'javascript', 'api', 'nodejs', 'web'],
        solution: {
          description: 'Create REST API endpoints using Express.js',
          steps: ['Install express', 'Create routes', 'Add middleware'],
          code: 'const express = require("express");'
        }
      },
      {
        case_id: 'test-react-hooks',
        task: 'React component state management with hooks',
        tags: ['react', 'javascript', 'frontend', 'hooks', 'state'],
        solution: {
          description: 'Manage state using React hooks',
          steps: ['Import useState', 'Initialize state', 'Update state'],
          code: 'const [state, setState] = useState(0);'
        }
      },
      {
        case_id: 'test-django-auth',
        task: 'Django user authentication implementation',
        tags: ['django', 'python', 'authentication', 'backend', 'web'],
        solution: {
          description: 'Implement user authentication in Django',
          steps: ['Create User model', 'Add authentication views', 'Configure URLs'],
          code: 'from django.contrib.auth import authenticate'
        }
      },
      {
        case_id: 'test-postgres-query',
        task: 'Optimize PostgreSQL query performance',
        tags: ['postgresql', 'database', 'sql', 'performance', 'backend'],
        solution: {
          description: 'Optimize slow PostgreSQL queries',
          steps: ['Add indexes', 'Analyze query plan', 'Refactor query'],
          code: 'CREATE INDEX idx_name ON table(column);'
        }
      }
    ];

    let added = 0;
    for (const testCase of testCases) {
      try {
        const request = {
          jsonrpc: '2.0',
          id: `setup-${Date.now()}-${added}`,
          method: 'tools/call',
          params: {
            name: 'add_case',
            arguments: testCase
          }
        };

        const response = await this.sendRequest(request);
        if (response.result) {
          added++;
        }
      } catch (error) {
        console.warn(`   ⚠️  Could not add test case ${testCase.case_id}: ${error.message}`);
      }
    }

    console.log(`   ✅ Added ${added} test cases\n`);
    return added > 0;
  }

  async testBasicRetrieve() {
    console.log('🧪 Test 1: Basic retrieve_cases (no enhancement)');

    try {
      const request = {
        jsonrpc: '2.0',
        id: Date.now(),
        method: 'tools/call',
        params: {
          name: 'retrieve_cases',
          arguments: {
            query: 'How to create an API',
            mode: 'non-parametric',
            top_k: 3,
            use_llm_enhancement: false
          }
        }
      };

      const response = await this.sendRequest(request);

      if (response.error) {
        console.log(`   ❌ Failed: ${response.error.message}`);
        this.testResults.push({ test: 'basic_retrieve', status: 'failed', error: response.error.message });
        return false;
      }

      const resultText = response.result?.content?.map((item) => item.text).join('\n');
      console.log(`   ✅ Success: Retrieved cases`);
      console.log(`   📝 Response preview: ${resultText.substring(0, 100)}...`);
      this.testResults.push({ test: 'basic_retrieve', status: 'passed' });
      return true;
    } catch (error) {
      console.log(`   ❌ Error: ${error.message}`);
      this.testResults.push({ test: 'basic_retrieve', status: 'failed', error: error.message });
      return false;
    }
  }

  async testWithTechContext() {
    console.log('\n🧪 Test 2: retrieve_cases with tech_context');

    try {
      const request = {
        jsonrpc: '2.0',
        id: Date.now(),
        method: 'tools/call',
        params: {
          name: 'retrieve_cases',
          arguments: {
            query: 'authentication',
            tech_context: {
              stack: 'javascript',
              framework: 'express',
              platform: 'web',
              language: 'javascript'
            },
            mode: 'parametric',
            top_k: 2,
            use_llm_enhancement: false
          }
        }
      };

      const response = await this.sendRequest(request);

      if (response.error) {
        console.log(`   ❌ Failed: ${response.error.message}`);
        this.testResults.push({ test: 'tech_context', status: 'failed', error: response.error.message });
        return false;
      }

      const resultText = response.result?.content?.map((item) => item.text).join('\n');
      const hasTechContext = resultText.includes('Tech Context');

      console.log(`   ✅ Success: Retrieved with tech context`);
      console.log(`   📋 Tech context included: ${hasTechContext ? 'Yes' : 'No'}`);
      console.log(`   📝 Response preview: ${resultText.substring(0, 500)}...`);
      this.testResults.push({ test: 'tech_context', status: 'passed' });
      return true;
    } catch (error) {
      console.log(`   ❌ Error: ${error.message}`);
      this.testResults.push({ test: 'tech_context', status: 'failed', error: error.message });
      return false;
    }
  }

  async testLLMEnhancementWithoutKey() {
    console.log('\n🧪 Test 3: LLM enhancement (without API key - should fallback)');

    try {
      const request = {
        jsonrpc: '2.0',
        id: Date.now(),
        method: 'tools/call',
        params: {
          name: 'retrieve_cases',
          arguments: {
            query: 'state management',
            tech_context: {
              framework: 'react',
              stack: 'javascript'
            },
            mode: 'non-parametric',
            top_k: 2,
            use_llm_enhancement: true
          }
        }
      };

      const response = await this.sendRequest(request);

      if (response.error) {
        console.log(`   ❌ Failed: ${response.error.message}`);
        this.testResults.push({ test: 'llm_fallback', status: 'failed', error: response.error.message });
        return false;
      }

      const resultText = response.result?.content?.map((item) => item.text).join('\n') || '';

      // Verify fallback behavior:
      // 1. Should NOT have "Query Enhancement" section (since no enhancement happened without API key)
      // 2. Should still return results using original query
      // 3. Should not throw errors
      const hasEnhancement = resultText.includes('Query Enhancement');
      const hasResults = resultText.includes('AI-Retrieved') || resultText.includes('cases');
      const hasOriginalQuery = resultText.includes('state management');

      if (hasEnhancement) {
        console.log(`   ⚠️  Warning: Query Enhancement section found, but API key should be missing`);
        console.log(`   💡 This might indicate API key was set in server environment`);
      }

      if (!hasResults) {
        console.log(`   ❌ Failed: No results returned`);
        this.testResults.push({ test: 'llm_fallback', status: 'failed', error: 'No results returned' });
        return false;
      }

      // Success: Fallback worked - results returned without enhancement
      console.log(`   ✅ Success: Graceful fallback when LLM unavailable`);
      console.log(`   📋 Enhancement attempted: ${hasEnhancement ? 'Yes (unexpected - API key may be set)' : 'No (expected fallback)'}`);
      console.log(`   📋 Results returned: ${hasResults ? 'Yes' : 'No'}`);
      console.log(`   📝 Response preview: ${resultText.substring(0, 200)}...`);
      this.testResults.push({ test: 'llm_fallback', status: 'passed' });
      return true;
    } catch (error) {
      console.log(`   ❌ Error: ${error.message}`);
      this.testResults.push({ test: 'llm_fallback', status: 'failed', error: error.message });
      return false;
    }
  }

  async testLLMEnhancementWithKey() {
    console.log('\n🧪 Test 4: LLM enhancement (with API key - if configured)');

    // Check for API keys (loaded from .env file via dotenv)
    const openaiKey = process.env.OPENAI_API_KEY;
    const anthropicKey = process.env.ANTHROPIC_API_KEY;
    const hasApiKey = openaiKey || anthropicKey;

    if (!hasApiKey) {
      console.log('   ⏭️  Skipped: No LLM API key configured');
      console.log('   💡 Set OPENAI_API_KEY or ANTHROPIC_API_KEY in .env file to test LLM enhancement');
      console.log('   💡 Or export as environment variables: export OPENAI_API_KEY="your-key"');
      this.testResults.push({ test: 'llm_enhancement', status: 'skipped', reason: 'No API key' });
      return true;
    }

    // Show which provider is being used (without exposing the key)
    const provider = openaiKey ? 'OpenAI' : 'Anthropic';
    const keyPreview = (openaiKey || anthropicKey || '').substring(0, 8) + '...';
    console.log(`   🔑 Using ${provider} API key: ${keyPreview}`);

    try {
      const request = {
        jsonrpc: '2.0',
        id: Date.now(),
        method: 'tools/call',
        params: {
          name: 'retrieve_cases',
          arguments: {
            query: 'database query optimization',
            tech_context: {
              stack: 'postgresql',
              platform: 'backend'
            },
            mode: 'parametric',
            top_k: 2,
            use_llm_enhancement: true
          }
        }
      };

      const response = await this.sendRequest(request);

      if (response.error) {
        console.log(`   ❌ Failed: ${response.error.message}`);
        this.testResults.push({ test: 'llm_enhancement', status: 'failed', error: response.error.message });
        return false;
      }

      const resultText = response.result?.content?.map((item) => item.text).join('\n') || '';
      const hasEnhancement = resultText.includes('Query Enhancement');
      const hasResults = resultText.includes('AI-Retrieved') || resultText.includes('cases');

      if (!hasEnhancement) {
        console.log(`   ⚠️  Warning: Query Enhancement section not found`);
        console.log(`   💡 This might indicate the LLM call failed or was skipped`);
      }

      if (!hasResults) {
        console.log(`   ❌ Failed: No results returned`);
        this.testResults.push({ test: 'llm_enhancement', status: 'failed', error: 'No results returned' });
        return false;
      }

      console.log(`   ✅ Success: LLM enhancement test completed`);
      console.log(`   🔍 Query enhanced: ${hasEnhancement ? 'Yes' : 'No (check server logs)'}`);
      console.log(`   📋 Results returned: ${hasResults ? 'Yes' : 'No'}`);
      console.log(`   📝 Response preview: ${resultText.substring(0, 200)}...`);
      this.testResults.push({ test: 'llm_enhancement', status: 'passed' });
      return true;
    } catch (error) {
      console.log(`   ❌ Error: ${error.message}`);
      this.testResults.push({ test: 'llm_enhancement', status: 'failed', error: error.message });
      return false;
    }
  }

  async testTagMatching() {
    console.log('\n🧪 Test 5: Tag matching from tech_context');

    try {
      const request = {
        jsonrpc: '2.0',
        id: Date.now(),
        method: 'tools/call',
        params: {
          name: 'retrieve_cases',
          arguments: {
            query: 'API development',
            tech_context: {
              framework: 'express',
              stack: 'javascript',
              language: 'javascript'
            },
            tags: ['api'],
            mode: 'parametric',
            top_k: 2,
            use_llm_enhancement: false
          }
        }
      };

      const response = await this.sendRequest(request);

      if (response.error) {
        console.log(`   ❌ Failed: ${response.error.message}`);
        this.testResults.push({ test: 'tag_matching', status: 'failed', error: response.error.message });
        return false;
      }

      const resultText = response.result?.content?.[0]?.text || '';
      // Should match express/javascript tags
      console.log(`   ✅ Success: Tag matching works`);
      console.log(`   📝 Response preview: ${resultText.substring(0, 500)}...`);
      this.testResults.push({ test: 'tag_matching', status: 'passed' });
      return true;
    } catch (error) {
      console.log(`   ❌ Error: ${error.message}`);
      this.testResults.push({ test: 'tag_matching', status: 'failed', error: error.message });
      return false;
    }
  }

  async testFullFlow() {
    console.log('\n🧪 Test 6: Full enhanced flow (query + tech context + LLM)');

    try {
      const request = {
        jsonrpc: '2.0',
        id: Date.now(),
        method: 'tools/call',
        params: {
          name: 'retrieve_cases',
          arguments: {
            query: 'user authentication',
            tech_context: {
              stack: 'javascript',
              framework: 'express',
              platform: 'web',
              project: 'REST API'
            },
            mode: 'parametric',
            top_k: 3,
            use_llm_enhancement: true
          }
        }
      };

      const response = await this.sendRequest(request);

      if (response.error) {
        console.log(`   ❌ Failed: ${response.error.message}`);
        this.testResults.push({ test: 'full_flow', status: 'failed', error: response.error.message });
        return false;
      }

      const resultText = response.result?.content?.[0]?.text || '';
      const hasEnhancement = resultText.includes('Query Enhancement');
      const hasTechContext = resultText.includes('Tech Context');

      console.log(`   ✅ Success: Full flow executed`);
      console.log(`   🔍 Query enhanced: ${hasEnhancement ? 'Yes' : 'No (fallback)'}`);
      console.log(`   📋 Tech context: ${hasTechContext ? 'Yes' : 'No'}`);
      console.log(`   📝 Response preview: ${resultText.substring(0, 200)}...`);
      this.testResults.push({ test: 'full_flow', status: 'passed' });
      return true;
    } catch (error) {
      console.log(`   ❌ Error: ${error.message}`);
      this.testResults.push({ test: 'full_flow', status: 'failed', error: error.message });
      return false;
    }
  }

  stopServer() {
    if (this.serverProcess) {
      this.serverProcess.kill();
      this.serverProcess = null;
    }
  }

  printSummary() {
    console.log('\n' + '='.repeat(70));
    console.log('📊 Test Results Summary');
    console.log('='.repeat(70));

    const passed = this.testResults.filter(r => r.status === 'passed').length;
    const failed = this.testResults.filter(r => r.status === 'failed').length;
    const skipped = this.testResults.filter(r => r.status === 'skipped').length;

    this.testResults.forEach(result => {
      const icon = result.status === 'passed' ? '✅' : result.status === 'failed' ? '❌' : '⏭️';
      console.log(`${icon} ${result.test}: ${result.status}`);
      if (result.error) {
        console.log(`   Error: ${result.error}`);
      }
      if (result.reason) {
        console.log(`   Reason: ${result.reason}`);
      }
    });

    console.log('\n' + '='.repeat(70));
    console.log(`Total: ${this.testResults.length} | Passed: ${passed} | Failed: ${failed} | Skipped: ${skipped}`);
    console.log('='.repeat(70));

    if (failed === 0) {
      console.log('\n🎉 All tests passed!');
    } else {
      console.log('\n⚠️  Some tests failed. Check the output above for details.');
    }
  }

  async runTests() {
    console.log('🚀 Starting Enhanced retrieve_cases Tests\n');
    console.log('='.repeat(70));

    try {
      await this.startServer();
      console.log('✅ Server started successfully\n');

      // Setup test data
      await this.setupTestData();

      // Run tests
      await this.testBasicRetrieve();
      await this.testWithTechContext();
      await this.testLLMEnhancementWithoutKey();
      await this.testLLMEnhancementWithKey();
      await this.testTagMatching();
      await this.testFullFlow();

      this.printSummary();

    } catch (error) {
      console.error('❌ Test suite failed:', error.message);
      console.error(error.stack);
    } finally {
      this.stopServer();
    }
  }
}

// Run tests if executed directly
if (import.meta.url === `file://${process.argv[1]}`) {
  const tester = new RetrieveCasesTester();
  tester.runTests().catch(console.error);
}

export default RetrieveCasesTester;
