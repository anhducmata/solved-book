#!/usr/bin/env node

/**
 * Test script for SolvedBook MCP Server Docker setup
 * This script verifies that the application is running correctly in Docker
 */

import http from 'http';

const BASE_URL = 'http://localhost:3000';

// Test functions
async function testHealthEndpoint() {
  console.log('🏥 Testing health endpoint...');
  
  return new Promise((resolve, reject) => {
    http.get(`${BASE_URL}/health`, (res) => {
      let data = '';
      
      res.on('data', chunk => {
        data += chunk;
      });
      
      res.on('end', () => {
        try {
          const health = JSON.parse(data);
          if (health.status === 'healthy') {
            console.log('✅ Health check passed');
            console.log(`   Server: ${health.server}`);
            console.log(`   Version: ${health.version}`);
            console.log(`   Timestamp: ${health.timestamp}`);
            resolve(true);
          } else {
            console.log('❌ Health check failed - status not healthy');
            resolve(false);
          }
        } catch (error) {
          console.log('❌ Health check failed - invalid JSON response');
          console.log('   Response:', data);
          resolve(false);
        }
      });
      
    }).on('error', (error) => {
      console.log('❌ Health check failed - connection error');
      console.log('   Error:', error.message);
      resolve(false);
    });
  });
}

async function testDatabaseConnection() {
  console.log('\n🗄️  Testing database connection...');
  
  try {
    const { execSync } = await import('child_process');
    const result = execSync('docker-compose exec -T postgres pg_isready -U solvedbook', {
      encoding: 'utf8',
      cwd: process.cwd()
    });
    
    if (result.includes('accepting connections')) {
      console.log('✅ Database connection successful');
      return true;
    } else {
      console.log('❌ Database connection failed');
      console.log('   Response:', result);
      return false;
    }
  } catch (error) {
    console.log('❌ Database connection test failed');
    console.log('   Error:', error.message);
    return false;
  }
}

async function testDatabaseSchema() {
  console.log('\n📋 Testing database schema...');
  
  try {
    const { execSync } = await import('child_process');
    const result = execSync('docker-compose exec -T postgres psql -U solvedbook -d solvedbook -c "\\dt"', {
      encoding: 'utf8',
      cwd: process.cwd()
    });
    
    if (result.includes('cases') && result.includes('table')) {
      console.log('✅ Database schema verified - cases table exists');
      return true;
    } else {
      console.log('❌ Database schema verification failed');
      console.log('   Response:', result);
      return false;
    }
  } catch (error) {
    console.log('❌ Database schema test failed');
    console.log('   Error:', error.message);
    return false;
  }
}

async function testContainerStatus() {
  console.log('\n🐳 Testing container status...');
  
  try {
    const { execSync } = await import('child_process');
    const result = execSync('docker-compose ps --format json', {
      encoding: 'utf8',
      cwd: process.cwd()
    });
    
    const containers = result.trim().split('\n').map(line => JSON.parse(line));
    let allHealthy = true;
    
    for (const container of containers) {
      const serviceName = container.Service;
      const status = container.State;
      const health = container.Health || 'N/A';
      
      console.log(`   ${serviceName}: ${status} (Health: ${health})`);
      
      if (status !== 'running') {
        allHealthy = false;
      }
      
      if (health !== 'N/A' && health !== 'healthy' && health !== 'starting') {
        allHealthy = false;
      }
    }
    
    if (allHealthy) {
      console.log('✅ All containers are running properly');
      return true;
    } else {
      console.log('❌ Some containers have issues');
      return false;
    }
  } catch (error) {
    console.log('❌ Container status test failed');
    console.log('   Error:', error.message);
    return false;
  }
}

async function runAllTests() {
  console.log('🚀 Starting SolvedBook MCP Server Docker Tests\n');
  console.log('=' .repeat(50));
  
  const tests = [
    { name: 'Health Endpoint', fn: testHealthEndpoint },
    { name: 'Database Connection', fn: testDatabaseConnection },
    { name: 'Database Schema', fn: testDatabaseSchema },
    { name: 'Container Status', fn: testContainerStatus }
  ];
  
  let passed = 0;
  let failed = 0;
  
  for (const test of tests) {
    try {
      const result = await test.fn();
      if (result) {
        passed++;
      } else {
        failed++;
      }
    } catch (error) {
      console.log(`❌ ${test.name} test threw an error:`, error.message);
      failed++;
    }
  }
  
  console.log('\n' + '=' .repeat(50));
  console.log('📊 Test Results Summary:');
  console.log(`   ✅ Passed: ${passed}`);
  console.log(`   ❌ Failed: ${failed}`);
  console.log(`   📈 Success Rate: ${((passed / (passed + failed)) * 100).toFixed(1)}%`);
  
  if (failed === 0) {
    console.log('\n🎉 All tests passed! SolvedBook MCP Server is running correctly in Docker.');
    process.exit(0);
  } else {
    console.log('\n⚠️  Some tests failed. Please check the Docker setup.');
    process.exit(1);
  }
}

// Handle command line arguments
if (process.argv.includes('--help') || process.argv.includes('-h')) {
  console.log('SolvedBook MCP Server Docker Test Suite');
  console.log('');
  console.log('Usage: node docker-test.js [options]');
  console.log('');
  console.log('Options:');
  console.log('  --help, -h     Show this help message');
  console.log('  --health       Test health endpoint only');
  console.log('  --db           Test database only');
  console.log('  --status       Test container status only');
  console.log('');
  console.log('Examples:');
  console.log('  node docker-test.js           # Run all tests');
  console.log('  node docker-test.js --health  # Test health endpoint only');
  process.exit(0);
}

if (process.argv.includes('--health')) {
  testHealthEndpoint().then(result => {
    process.exit(result ? 0 : 1);
  });
} else if (process.argv.includes('--db')) {
  testDatabaseConnection().then(result => {
    process.exit(result ? 0 : 1);
  });
} else if (process.argv.includes('--status')) {
  testContainerStatus().then(result => {
    process.exit(result ? 0 : 1);
  });
} else {
  runAllTests();
}
