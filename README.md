# SolvedBook Smart MCP Server

A **Model Context Protocol (MCP)** server with **advanced AI capabilities** that provides intelligent case management with sophisticated Q-learning, semantic analysis, and predictive recommendations.

## 🚀 Quick Start

### Docker (Recommended)
```bash
# Clone and start
git clone <repository-url>
cd solvedbook-mcp-server
make up

# Check status
make health
```

### Local Development
```bash
npm install
npm run dev
```

## 🔌 Client Integration

### VS Code with Claude/Copilot
Add to your VS Code `settings.json`:
```json
{
  "mcp.servers": {
    "solvedbook": {
      "command": "node",
      "args": ["src/mcp-server.js"],
      "cwd": "/path/to/solvedbook-mcp-server"
    }
  }
}
```

### Claude Desktop App
Edit `~/Library/Application Support/Claude/claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "solvedbook": {
      "command": "node",
      "args": ["src/mcp-server.js"],
      "cwd": "/path/to/solvedbook-mcp-server"
    }
  }
# SolvedBook Smart MCP Server

A **Model Context Protocol (MCP)** server with **advanced AI capabilities** that provides intelligent case management with sophisticated Q-learning, semantic analysis, and predictive recommendations.

## 🚀 Quick Start

### Docker (Recommended)
```bash
# Clone and start
git clone <repository-url>
cd solvedbook-mcp-server
make up

# Check status
make health
```

### Local Development
```bash
npm install
npm run dev
```

## 🔌 Client Integration

### VS Code with Claude/Copilot
Add to your VS Code `settings.json`:
```json
{
  "mcp.servers": {
    "solvedbook": {
      "command": "node",
      "args": ["src/mcp-server.js"],
      "cwd": "/path/to/solvedbook-mcp-server"
    }
  }
}
```

### Claude Desktop App
Edit `~/Library/Application Support/Claude/claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "solvedbook": {
      "command": "node",
      "args": ["src/mcp-server.js"],
      "cwd": "/path/to/solvedbook-mcp-server"
    }
  }
}
```

### Docker-based Connection
```json
{
  "mcp.servers": {
    "solvedbook": {
      "command": "docker",
      "args": ["exec", "-i", "solvedbook-app", "node", "src/mcp-server.js"],
      "cwd": "/path/to/solvedbook-mcp-server"
    }
  }
}
```

## 🧠 AI Features

### **Advanced Q-Learning System**
- **Adaptive Learning Rate**: Adjusts based on user confidence and feedback history
- **Context-Aware Scoring**: Considers case relevance and user patterns
- **Exploration Bonus**: Encourages discovery of new solutions
- **Decay Factor**: Prevents overfitting to old feedback
- **Confidence Tracking**: Measures system certainty about recommendations

### **Semantic Intelligence**
- **Keyword Extraction**: Intelligent text analysis with stop-word filtering
- **Jaccard Similarity**: Advanced text matching algorithms
- **Context Relevance**: Tag matching and user pattern analysis
- **Multi-Factor Scoring**: Combines semantic, context, and Q-value scores

### **Predictive AI**
- **User Pattern Recognition**: Learns from individual user behavior
- **Feedback Sentiment Analysis**: Analyzes feedback patterns over time
- **Confidence Thresholds**: Only shows high-confidence recommendations
- **Success Rate Tracking**: Monitors case effectiveness per user

## 🎯 Intelligence Components

### Smart Q-Learning Formula
```javascript
newQ = currentQ * decayFactor + 
       adaptiveLearningRate * (contextAdjustedReward - currentQ) + 
       explorationBonus
```

### Multi-Factor Case Scoring
```javascript
finalScore = (semanticScore * 0.6) + 
            (contextScore * 0.3) + 
            (qValueScore * 0.1)
```

### Confidence Calculation
```javascript
confidence = (feedbackConfidence * 0.4) + (scoreConfidence * 0.6)
```

## 🛠️ Available Tools

| Tool | Intelligence Features |
|------|----------------------|
| `add_case` | Automatic keyword extraction, semantic preprocessing |
| `search_cases` | Smart ranking by Q-values and relevance |
| `retrieve_cases` | AI-powered semantic matching + Q-learning |
| `reward_case` | Adaptive learning with context awareness |
| `get_case` | Enhanced with confidence and success metrics |
| `get_top_cases` | Multi-factor ranking with AI insights |
| `submit_feedback` | Sentiment analysis and pattern recognition |

## 💡 Example Usage

### Intelligent Case Retrieval
```javascript
await callTool('retrieve_cases', {
  query: 'authentication with JWT tokens',
  mode: 'parametric',  // Uses AI + Q-learning
  top_k: 5
});

// Returns AI-enhanced results with confidence scores
```

### Smart Feedback Learning
```javascript
await callTool('reward_case', {
  case_id: 'jwt-auth-api',
  reward: 0.8
});

// Updates Q-values with adaptive learning
```

## 🐳 Docker Operations

### Development
```bash
make up              # Start development environment
make down            # Stop all services
make logs            # View logs
make health          # Check health status
make shell           # Access app container
make test            # Run tests
```

### Production
```bash
make prod-up         # Start production environment
make prod-down       # Stop production environment
```

### Database
```bash
make db-shell        # Access PostgreSQL shell
make backup          # Backup database
make restore         # Restore from backup
```

## 🔧 Configuration

### Environment Variables
```bash
# Database
DB_HOST=postgres
DB_PORT=5432
DB_NAME=solvedbook
DB_USER=solvedbook
DB_PASSWORD=password

# Application
NODE_ENV=development
PORT=3000
```

### AI Configuration
```javascript
const AI_CONFIG = {
  qlearning: {
    learningRate: 0.1,          // Base learning rate
    decayFactor: 0.95,          // Q-value decay
    explorationRate: 0.1,       // Exploration bonus
    confidenceThreshold: 0.7    // Minimum confidence
  },
  similarity: {
    semanticWeight: 0.6,        // Semantic similarity weight
    contextWeight: 0.3,         // Context relevance weight
    qvalueWeight: 0.1          // Q-value weight
  }
}
```

## 📁 Project Structure

```
solvedbook-mcp-server/
├── src/
│   ├── mcp-server.js          # Main MCP server implementation
│   ├── config/
│   │   └── database.js        # Database configuration
│   └── core/
│       └── algorithms.js      # AI algorithms (Q-learning, semantic analysis)
├── database/
│   └── schema.sql            # PostgreSQL database schema
├── docker-compose.yml        # Main orchestration
├── docker-compose.override.yml # Development overrides
├── docker-compose.prod.yml   # Production configuration
├── Dockerfile                # Multi-stage container build
├── docker-entrypoint.sh      # Container startup script
├── Makefile                 # Docker operation commands
├── package.json              # Node.js dependencies and scripts
├── deploy.sh                # Deployment automation script
├── docker-test.js           # Integration testing suite
└── test-mcp-server.js       # MCP server testing
```

## 🧪 Testing

### Run Tests
```bash
# All tests
make test

# Integration tests
node docker-test.js

# MCP server tests
node test-mcp-server.js
```

### Monitor AI Learning
```bash
# Check Q-learning progress
docker-compose exec postgres psql -U solvedbook -d solvedbook -c "
  SELECT case_id, q_value, reward_count, 
         ROUND(q_value::numeric, 3) as smart_score
  FROM cases 
  ORDER BY q_value DESC 
  LIMIT 10;"
```

## 🔒 Security & Performance

- **Input Validation**: All parameters validated before AI processing
- **SQL Injection Prevention**: Parameterized queries
- **Memory Efficiency**: Smart caching of AI patterns
- **Rate Limiting Ready**: Can implement per-user AI usage limits
- **Privacy**: User patterns stored temporarily, not permanently

## 🚨 Troubleshooting

### Common Issues
1. **Port conflicts**: Check `lsof -i :3000` and `lsof -i :5432`
2. **Database connection**: Run `make health` to verify
3. **Container issues**: Try `make clean && make up`
4. **MCP client**: Verify paths in client configuration

### Debug Information
- Health endpoint: `http://localhost:3000/health`
- Application logs: `make logs app`
- Database logs: `make logs postgres`
- Integration tests: `node docker-test.js`

## 📈 Intelligence Metrics

The system tracks and provides:
- **Confidence Scores**: How certain the AI is about recommendations
- **Success Rates**: Historical performance per case and user
- **Semantic Similarity**: Text-based relevance matching
- **Context Relevance**: Tag and pattern-based scoring
- **Learning Trends**: Q-value improvement over time
- **Feedback Sentiment**: Positive/negative feedback patterns

## 🎯 Intelligence Advantages

1. **Learns from Feedback**: Gets smarter with every interaction
2. **Context Aware**: Understands user patterns and preferences  
3. **Confidence Driven**: Only suggests high-confidence recommendations
4. **Multi-Modal**: Combines Q-learning, semantics, and context
5. **Adaptive**: Learning rate adjusts based on user behavior
6. **Predictive**: Anticipates which cases will be most helpful

## 📚 Further Reading

- **MCP Protocol**: [Model Context Protocol Specification](https://modelcontextprotocol.io/introduction)
- **Q-Learning**: See `src/core/algorithms.js` for implementation
- **Docker Best Practices**: Review `Dockerfile` and compose files
- **PostgreSQL**: Check `database/schema.sql` for data structure

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with `make test`
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

---

**🧠 Powered by Advanced AI • Built for Smart Agents**

*An intelligent MCP server that learns from your interactions and provides context-aware recommendations using cutting-edge Q-learning and semantic analysis.*