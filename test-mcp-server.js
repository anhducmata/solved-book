#!/usr/bin/env node

/**
 * Simple test script to verify MCP server functionality
 */

import { spawn } from 'child_process';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

class MCPTester {
  constructor() {
    this.serverProcess = null;
  }

  async startServer() {
    return new Promise((resolve, reject) => {
      this.serverProcess = spawn('node', ['src/mcp-server.js'], {
        cwd: __dirname,
        stdio: ['pipe', 'pipe', 'pipe']
      });

      this.serverProcess.stderr.on('data', (data) => {
        const message = data.toString();
        console.log('Server stderr:', message);
        if (message.includes('SolvedBook MCP Server running on stdio')) {
          resolve();
        }
      });

      this.serverProcess.on('error', reject);

      setTimeout(() => {
        reject(new Error('Server startup timeout'));
      }, 10000);
    });
  }

  async sendRequest(request) {
    return new Promise((resolve, reject) => {
      let responseData = '';

      const timeout = setTimeout(() => {
        reject(new Error('Request timeout'));
      }, 5000);

      this.serverProcess.stdout.once('data', (data) => {
        clearTimeout(timeout);
        responseData = data.toString();
        try {
          const response = JSON.parse(responseData);
          resolve(response);
        } catch (e) {
          reject(new Error(`Invalid JSON response: ${responseData}`));
        }
      });

      this.serverProcess.stdin.write(JSON.stringify(request) + '\n');
    });
  }

  async testListTools() {
    console.log('🧪 Testing list tools...');
    const request = {
      jsonrpc: '2.0',
      id: 1,
      method: 'tools/list'
    };

    try {
      const response = await this.sendRequest(request);
      console.log('✅ Tools listed successfully');
      console.log(`   Found ${response.result?.tools?.length || 0} tools`);
      return true;
    } catch (error) {
      console.log('❌ List tools failed:', error.message);
      return false;
    }
  }

  async testListResources() {
    console.log('🧪 Testing list resources...');
    const request = {
      jsonrpc: '2.0',
      id: 2,
      method: 'resources/list'
    };

    try {
      const response = await this.sendRequest(request);
      console.log('✅ Resources listed successfully');
      console.log(`   Found ${response.result?.resources?.length || 0} resources`);
      return true;
    } catch (error) {
      console.log('❌ List resources failed:', error.message);
      return false;
    }
  }

  async testAddCase() {
    console.log('🧪 Testing add case...');
    const request = {
      jsonrpc: '2.0',
      id: 3,
      method: 'tools/call',
      params: {
        name: 'add_case',
        arguments: {
          case_id: 'test-case-' + Date.now(),
          task: 'Test case for MCP server verification',
          tags: ['test', 'mcp'],
          solution: {
            description: 'This is a test case',
            steps: ['Step 1', 'Step 2'],
            code: 'console.log("Hello MCP");'
          }
        }
      }
    };

    try {
      const response = await this.sendRequest(request);
      console.log('✅ Add case succeeded');
      return true;
    } catch (error) {
      console.log('❌ Add case failed:', error.message);
      return false;
    }
  }

  async testEnrichContextTool() {
    console.log('🧪 Testing enrich_context tool...');

    // First check if tool is in the list
    const listRequest = {
      jsonrpc: '2.0',
      id: 10,
      method: 'tools/list'
    };

    try {
      const listResponse = await this.sendRequest(listRequest);
      const tools = listResponse.result?.tools || [];
      const enrichContextTool = tools.find(t => t.name === 'enrich_context');

      if (!enrichContextTool) {
        console.log('❌ enrich_context tool not found in tools list');
        return false;
      }

      console.log('✅ enrich_context tool found in tools list');
      console.log(`   Description: ${enrichContextTool.description?.substring(0, 60)}...`);
      return true;
    } catch (error) {
      console.log('❌ Failed to check enrich_context tool:', error.message);
      return false;
    }
  }

  stopServer() {
    if (this.serverProcess) {
      this.serverProcess.kill();
      this.serverProcess = null;
    }
  }

  async runTests() {
    console.log('🚀 Starting MCP Server Tests\n');

    try {
      await this.startServer();
      console.log('✅ Server started successfully\n');

      const results = [];
      results.push(await this.testListTools());
      results.push(await this.testListResources());
      results.push(await this.testAddCase());
      results.push(await this.testEnrichContextTool());

      const passed = results.filter(Boolean).length;
      const total = results.length;

      console.log(`\n📊 Test Results: ${passed}/${total} tests passed`);

      if (passed === total) {
        console.log('🎉 All tests passed! MCP server is working correctly.');
      } else {
        console.log('⚠️  Some tests failed. Check the logs above.');
      }

    } catch (error) {
      console.error('❌ Test suite failed:', error.message);
    } finally {
      this.stopServer();
    }
  }
}

// Run tests if this script is executed directly
if (import.meta.url === `file://${process.argv[1]}`) {
  const tester = new MCPTester();
  tester.runTests().catch(console.error);
}

export default MCPTester;
