#!/usr/bin/env node

/**
 * SolvedBook Model Context Protocol (MCP) Server
 * 
 * This server provides AI agents with access to a case bank with Q-learning capabilities.
 * It implements the Model Context Protocol specification for seamless integration
 * with AI clients like Claude, GPT, and other language models.
 */

import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import {
  ListResourcesRequestSchema,
  ReadResourceRequestSchema,
  ListToolsRequestSchema,
  CallToolRequestSchema,
  ErrorCode,
  McpError,
} from '@modelcontextprotocol/sdk/types.js';

import { config } from 'dotenv';
import { getDatabase, initializeDatabase } from './config/database.js';
import http from 'http';
import { ContextEnrichment } from './core/context-enrichment.js';

// Load environment variables
// LLM Configuration
const LLM_CONFIG = {
  provider: process.env.LLM_PROVIDER || 'openai', // 'openai' or 'anthropic'
  apiKey: process.env.OPENAI_API_KEY || process.env.ANTHROPIC_API_KEY || null,
  model: process.env.LLM_MODEL || 'gpt-4o-mini',
  baseURL: process.env.LLM_BASE_URL || null,
};

// Advanced AI Configuration
const AI_CONFIG = {
  qlearning: {
    learningRate: 0.1,
    decayFactor: 0.95,
    explorationRate: 0.1,
    confidenceThreshold: 0.7
  },
  similarity: {
    semanticWeight: 0.6,
    contextWeight: 0.3,
    qvalueWeight: 0.1
  },
  adaptation: {
    userPatternTracking: true,
    contextualLearning: true,
    feedbackAnalysis: true
  }
};

class AdvancedAI {
  constructor() {
    this.userPatterns = new Map();
    this.contextHistory = [];
    this.feedbackPatterns = new Map();
    this.llmEnhancer = new LLMQueryEnhancer(LLM_CONFIG);
  }

  // Smart Q-Learning with decay and exploration
  calculateSmartQValue(currentQ, reward, contextScore = 1.0, userConfidence = 1.0) {
    const { learningRate, decayFactor, explorationRate } = AI_CONFIG.qlearning;
    
    // Adaptive learning rate based on confidence
    const adaptiveLearningRate = learningRate * userConfidence;
    
    // Q-learning with exploration bonus
    const explorationBonus = Math.random() < explorationRate ? 0.1 : 0;
    
    // Context-aware Q-value update
    const contextAdjustedReward = reward * contextScore;
    
    // Advanced Q-learning formula
    const newQ = currentQ * decayFactor + 
                 adaptiveLearningRate * (contextAdjustedReward - currentQ) + 
                 explorationBonus;
    
    return Math.max(-1.0, Math.min(1.0, newQ));
  }

  // Semantic similarity using simple text analysis
  calculateSemanticSimilarity(query, caseText) {
    const queryWords = this.extractKeywords(query.toLowerCase());
    const caseWords = this.extractKeywords(caseText.toLowerCase());
    
    if (queryWords.length === 0 || caseWords.length === 0) return 0;
    
    const intersection = queryWords.filter(word => caseWords.includes(word));
    const union = [...new Set([...queryWords, ...caseWords])];
    
    // Jaccard similarity with length normalization
    const jaccardSim = intersection.length / union.length;
    const lengthSim = 1 - Math.abs(queryWords.length - caseWords.length) / Math.max(queryWords.length, caseWords.length);
    
    return (jaccardSim * 0.7 + lengthSim * 0.3);
  }

  extractKeywords(text) {
    const stopWords = new Set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should']);
    
    return text.split(/\W+/)
               .filter(word => word.length > 2 && !stopWords.has(word))
               .slice(0, 20); // Limit to top 20 keywords
  }

  // Context-aware case scoring
  scoreCase(caseItem, query, userContext = {}) {
    const { semanticWeight, contextWeight, qvalueWeight } = AI_CONFIG.similarity;
    
    // Semantic similarity
    const semanticScore = this.calculateSemanticSimilarity(query, caseItem.task);
    
    // Q-value normalized score
    const qScore = (caseItem.q_value + 1) / 2; // Normalize from [-1,1] to [0,1]
    
    // Context relevance (tags, recent usage, user patterns)
    const contextScore = this.calculateContextRelevance(caseItem, userContext);
    
    // Weighted combined score
    const finalScore = (semanticScore * semanticWeight) + 
                      (contextScore * contextWeight) + 
                      (qScore * qvalueWeight);
    
    return {
      score: finalScore,
      breakdown: {
        semantic: semanticScore,
        context: contextScore,
        qvalue: qScore
      }
    };
  }

  calculateContextRelevance(caseItem, userContext) {
    let relevance = 0.5; // Base relevance
    
    // Tag matching bonus
    if (userContext.preferredTags && caseItem.tags) {
      const tagMatch = userContext.preferredTags.filter(tag => 
        caseItem.tags.includes(tag)
      ).length;
      relevance += (tagMatch / Math.max(userContext.preferredTags.length, 1)) * 0.3;
    }
    
    // Recent success pattern
    if (this.userPatterns.has(caseItem.case_id)) {
      const pattern = this.userPatterns.get(caseItem.case_id);
      relevance += pattern.successRate * 0.2;
    }
    
    return Math.min(1.0, relevance);
  }

  // Learning from user feedback patterns
  analyzeFeedbackPattern(caseId, feedbackType, feedbackData) {
    const key = `${caseId}_${feedbackType}`;
    
    if (!this.feedbackPatterns.has(key)) {
      this.feedbackPatterns.set(key, { count: 0, sentiment: 0 });
    }
    
    const pattern = this.feedbackPatterns.get(key);
    pattern.count++;
    
    // Simple sentiment analysis
    if (feedbackType.includes('helpful') || feedbackType.includes('good')) {
      pattern.sentiment += 0.5;
    } else if (feedbackType.includes('unclear') || feedbackType.includes('bad')) {
      pattern.sentiment -= 0.3;
    }
    
    return pattern;
  }

  // Predictive recommendations
  predictCaseRelevance(cases, query, userHistory = []) {
    return cases.map(caseItem => {
      const score = this.scoreCase(caseItem, query, { preferredTags: userHistory });
      return {
        ...caseItem,
        aiScore: score.score,
        aiBreakdown: score.breakdown,
        confidence: this.calculateConfidence(score.score, caseItem.reward_count || 0)
      };
    }).sort((a, b) => b.aiScore - a.aiScore);
  }

  calculateConfidence(score, rewardCount) {
    // Confidence increases with more feedback and higher scores
    const feedbackConfidence = Math.min(1.0, rewardCount / 10);
    const scoreConfidence = score;
    return (feedbackConfidence * 0.4 + scoreConfidence * 0.6);
  }
}

class SolvedBookMCPServer {
  constructor() {
    this.server = new Server(
      {
        name: 'solvedbook-mcp-server',
        version: '1.0.0',
      },
      {
        capabilities: {
          resources: {},
          tools: {},
        },
      }
    );

    // Initialize AI engine
    this.ai = new AdvancedAI();
    this.setupHandlers();
  }

  setupHandlers() {
    // List available resources
    this.server.setRequestHandler(ListResourcesRequestSchema, async () => {
      return {
        resources: [
          {
            uri: 'solvedbook://cases',
            name: 'All Cases',
            description: 'Access to the complete case database with Q-learning values',
            mimeType: 'application/json',
          },
          {
            uri: 'solvedbook://qvalues',
            name: 'Q-Values',
            description: 'Current Q-learning values for all cases',
            mimeType: 'application/json',
          },
          {
            uri: 'solvedbook://top-cases',
            name: 'Top Performing Cases',
            description: 'Cases with highest Q-values (best performing)',
            mimeType: 'application/json',
          },
        ],
      };
    });

    // Read specific resources
    this.server.setRequestHandler(ReadResourceRequestSchema, async (request) => {
      const { uri } = request.params;

      try {
        const db = getDatabase();
        let data, description;

        switch (uri) {
          case 'solvedbook://cases':
            const allCases = await db.query('SELECT * FROM cases ORDER BY created_at DESC');
            data = allCases.rows.map(this.formatCase);
            description = `Retrieved ${data.length} cases from the database`;
            break;

          case 'solvedbook://qvalues':
            const qValues = await db.query('SELECT case_id, q_value, reward_count FROM cases ORDER BY q_value DESC');
            data = qValues.rows;
            description = `Retrieved Q-values for ${data.length} cases`;
            break;

          case 'solvedbook://top-cases':
            const topCases = await db.query('SELECT * FROM cases ORDER BY q_value DESC LIMIT 10');
            data = topCases.rows.map(this.formatCase);
            description = `Retrieved top 10 performing cases by Q-value`;
            break;

          default:
            throw new McpError(ErrorCode.InvalidRequest, `Unknown resource: ${uri}`);
        }

        return {
          contents: [
            {
              uri,
              mimeType: 'application/json',
              text: JSON.stringify({ data, description }, null, 2),
            },
          ],
        };
      } catch (error) {
        throw new McpError(ErrorCode.InternalError, `Failed to read resource: ${error.message}`);
      }
    });

    // List available tools
    this.server.setRequestHandler(ListToolsRequestSchema, async () => {
      return {
        tools: [
          {
            name: 'add_case',
            description: 'Add a new case to the knowledge base',
            inputSchema: {
              type: 'object',
              properties: {
                case_id: { type: 'string', description: 'Unique identifier for the case' },
                task: { type: 'string', description: 'Description of the task or problem' },
                tags: { type: 'array', items: { type: 'string' }, description: 'Tags for categorization' },
                solution: { type: 'object', description: 'Solution details including description, steps, code, etc.' },
                embedding: { type: 'string', description: 'Optional: Base64 encoded embedding vector' },
              },
              required: ['case_id', 'task', 'solution'],
            },
          },
          {
            name: 'search_cases',
            description: 'Search for cases using tags or keywords',
            inputSchema: {
              type: 'object',
              properties: {
                tags: { type: 'array', items: { type: 'string' }, description: 'Tags to search for' },
                keyword: { type: 'string', description: 'Keyword to search in task descriptions' },
                limit: { type: 'number', description: 'Maximum number of results (default: 10)' },
              },
            },
          },
          {
            name: 'retrieve_cases',
            description: 'Retrieve cases using Q-learning (parametric) or similarity (non-parametric). Enhanced with LLM query optimization based on available solutions and tech context.',
            inputSchema: {
              type: 'object',
              properties: {
                query: { type: 'string', description: 'Query string for case retrieval' },
                mode: {
                  type: 'string',
                  enum: ['parametric', 'non-parametric'],
                  description: 'Retrieval mode: parametric (Q-value based) or non-parametric (similarity based)'
                },
                top_k: { type: 'number', description: 'Number of cases to retrieve (default: 5)' },
                tags: { type: 'array', items: { type: 'string' }, description: 'Optional: Filter by tags' },
                tech_context: {
                  type: 'object',
                  description: 'Optional: Technical context (stack, framework, project type, etc.) that can match against solution tags',
                  properties: {
                    stack: { type: 'string', description: 'Technology stack (e.g., "javascript", "python")' },
                    framework: { type: 'string', description: 'Framework (e.g., "react", "express", "django")' },
                    project: { type: 'string', description: 'Project type or domain' },
                    language: { type: 'string', description: 'Programming language' },
                    platform: { type: 'string', description: 'Platform (e.g., "web", "mobile", "backend")' },
                  },
                },
                use_llm_enhancement: {
                  type: 'boolean',
                  description: 'Whether to use LLM to enhance the query based on available solutions (default: true)',
                },
              },
              required: ['query'],
            },
          },
          {
            name: 'reward_case',
            description: 'Provide feedback to improve Q-learning (updates Q-values)',
            inputSchema: {
              type: 'object',
              properties: {
                case_id: { type: 'string', description: 'ID of the case to reward' },
                reward: { type: 'number', description: 'Reward value (-1.0 to 1.0)' },
              },
              required: ['case_id', 'reward'],
            },
          },
          {
            name: 'get_case',
            description: 'Get a specific case by ID',
            inputSchema: {
              type: 'object',
              properties: {
                case_id: { type: 'string', description: 'ID of the case to retrieve' },
              },
              required: ['case_id'],
            },
          },
          {
            name: 'get_top_cases',
            description: 'Get top performing cases by Q-value',
            inputSchema: {
              type: 'object',
              properties: {
                limit: { type: 'number', description: 'Number of top cases to retrieve (default: 10)' },
              },
            },
          },
          {
            name: 'submit_feedback',
            description: 'Submit detailed feedback for a case',
            inputSchema: {
              type: 'object',
              properties: {
                case_id: { type: 'string', description: 'ID of the case' },
                feedback_type: { type: 'string', description: 'Type of feedback (e.g., helpful, unclear, incorrect)' },
                feedback_data: { type: 'object', description: 'Additional feedback details' },
              },
              required: ['case_id', 'feedback_type'],
            },
          },
          {
            name: 'add_case_from_conversation',
            description: 'Add a case to the knowledge base from a question and answer pair (captures previous conversation context)',
            inputSchema: {
              type: 'object',
              properties: {
                question: { type: 'string', description: 'The question that was asked' },
                answer: { type: 'string', description: 'The answer provided' },
                tags: { type: 'array', items: { type: 'string' }, description: 'Optional tags for categorization (will be auto-extracted if not provided)' },
                case_id: { type: 'string', description: 'Optional: Custom case ID (will be auto-generated if not provided)' },
                category: { type: 'string', description: 'Optional category (e.g., "performance", "optimization", "computer-science")' },
              },
              required: ['question', 'answer'],
            },
          },
          {
            name: 'enrich_context',
            description: 'Extract query-relevant content from cases to create focused prompt context. Uses similarity-based attribution to identify and include only the most relevant parts of case solutions.',
            inputSchema: {
              type: 'object',
              properties: {
                query: { type: 'string', description: 'The query/question for which to enrich context' },
                case_ids: { type: 'array', items: { type: 'string' }, description: 'Optional: Specific case IDs to enrich. If not provided, will retrieve relevant cases automatically.' },
                mode: { type: 'string', enum: ['parametric', 'non-parametric'], description: 'Retrieval mode if case_ids not provided: parametric (Q-value based) or non-parametric (similarity based) (default: parametric)' },
                top_k: { type: 'number', description: 'Number of cases to enrich (default: 5)' },
                max_sentences_per_case: { type: 'number', description: 'Maximum sentences to extract from each case solution (default: 3)' },
                min_relevance_score: { type: 'number', description: 'Minimum relevance score for sentence inclusion (default: 0.1)' },
                format: { type: 'string', enum: ['text', 'json'], description: 'Output format: text (for prompt inclusion) or json (structured data) (default: text)' },
                include_full_solution: { type: 'boolean', description: 'Include full solution in addition to relevant content (default: false)' },
                tags: { type: 'array', items: { type: 'string' }, description: 'Optional: Filter cases by tags (only used if case_ids not provided)' },
              },
              required: ['query'],
            },
          },
        ],
      };
    });

    // Handle tool calls
    this.server.setRequestHandler(CallToolRequestSchema, async (request) => {
      const { name, arguments: args } = request.params;

      try {
        const db = getDatabase();

        switch (name) {
          case 'add_case':
            return await this.handleAddCase(db, args);
          
          case 'search_cases':
            return await this.handleSearchCases(db, args);
          
          case 'retrieve_cases':
            return await this.handleRetrieveCases(db, args);
          
          case 'reward_case':
            return await this.handleRewardCase(db, args);
          
          case 'get_case':
            return await this.handleGetCase(db, args);
          
          case 'get_top_cases':
            return await this.handleGetTopCases(db, args);
          
          case 'submit_feedback':
            return await this.handleSubmitFeedback(db, args);


          case 'add_case_from_conversation':
            return await this.handleAddCaseFromConversation(db, args);

          case 'enrich_context':
            return await this.handleEnrichContext(db, args);

          default:
            throw new McpError(ErrorCode.MethodNotFound, `Unknown tool: ${name}`);
        }
      } catch (error) {
        throw new McpError(ErrorCode.InternalError, `Tool execution failed: ${error.message}`);
      }
    });
  }

  // Tool handlers
  async handleAddCase(db, args) {
    const { case_id, task, tags = [], solution, embedding = null } = args;

    // Validate required fields
    if (!case_id || !task || !solution) {
      throw new McpError(ErrorCode.InvalidParams, 'case_id, task, and solution are required');
    }

    // Validate tags uniqueness
    if (tags.length !== new Set(tags).size) {
      throw new McpError(ErrorCode.InvalidParams, 'Tags must be unique');
    }

    const sql = `INSERT INTO cases (case_id, task, tags, solution, embedding) VALUES ($1, $2, $3, $4, $5)`;
    await db.query(sql, [case_id, task, JSON.stringify(tags), JSON.stringify(solution), embedding]);

    return {
      content: [
        {
          type: 'text',
          text: `✅ Case '${case_id}' added successfully to the knowledge base`,
        },
      ],
    };
  }

  async handleSearchCases(db, args) {
    const { tags = [], keyword, limit = 10 } = args;
    
    let sql = 'SELECT * FROM cases WHERE 1=1';
    const params = [];
    let paramCount = 0;

    if (tags.length > 0) {
      paramCount++;
      sql += ` AND tags LIKE $${paramCount}`;
      params.push(`%${JSON.stringify(tags)}%`);
    }

    if (keyword) {
      paramCount++;
      sql += ` AND task ILIKE $${paramCount}`;
      params.push(`%${keyword}%`);
    }

    sql += ` ORDER BY q_value DESC LIMIT $${paramCount + 1}`;
    params.push(limit);

    const result = await db.query(sql, params);
    const cases = result.rows.map(this.formatCase);

    return {
      content: [
        {
          type: 'text',
          text: `🔍 Found ${cases.length} cases matching your search criteria:\n\n${this.formatSearchResults(cases)}`,
        },
      ],
    };
  }

  async handleRetrieveCases(db, args) {
    const {
      query,
      mode = 'parametric',
      top_k = 5,
      tags = [],
      tech_context = {},
      use_llm_enhancement = true,
    } = args;

    // Step 1: User sends init query request (already received)
    const originalQuery = query;

    // Step 2: Server lists existing solution titles
    const solutionTitles = await this.listSolutionTitles(db, 10);
    console.error({ solutionTitles });

    // Step 3 & 4: Server sends user query with solution titles to LLM, LLM summarizes a better vector query
    let enhancedQuery = originalQuery;
    if (use_llm_enhancement && solutionTitles.length > 0) {
      try {
        enhancedQuery = await this.ai.llmEnhancer.enhanceQuery(
          originalQuery,
          solutionTitles,
          tech_context
        );
        console.error({ enhancedQuery, originalQuery });
      } catch (error) {
        console.warn('LLM enhancement failed, using original query:', error.message);
        enhancedQuery = originalQuery;
      }
    }

    // Step 5: Server makes a query to vector DB (or similarity search)
    // Combine tags from user input and tech context
    const techContextTags = this.extractTagsFromTechContext(tech_context);
    const allTags = [...new Set([...tags, ...techContextTags])];

    // Get all relevant cases (broader search first)
    let sql = 'SELECT * FROM cases';
    let params = [];

    if (allTags.length > 0) {
      // Match tags more effectively
      const tagConditions = allTags.map((tag, idx) => {
        params.push(`%${tag}%`);
        return `tags::text ILIKE $${params.length}`;
      }).join(' OR ');
      sql += ` WHERE ${tagConditions}`;
    }

    const result = await db.query(sql, params);
    let cases = result.rows.map(this.formatCase);

    // Use enhanced query for retrieval
    const searchQuery = enhancedQuery;

    if (mode === 'parametric') {
      // AI-enhanced Q-learning retrieval
      cases = this.ai.predictCaseRelevance(cases, query, tags);
      
      // Filter by confidence threshold
      const confidentCases = cases.filter(c => c.confidence > AI_CONFIG.qlearning.confidenceThreshold);
      const finalCases = confidentCases.length >= top_k ? confidentCases : cases;
      
      cases = finalCases.slice(0, top_k);
    } else {
      // Advanced semantic similarity retrieval
      cases = cases
        .map(caseItem => ({
          ...caseItem,
          semanticScore: this.ai.calculateSemanticSimilarity(query, caseItem.task)
        }))
        .filter(c => c.semanticScore > 0.1) // Filter out very low similarity
        .sort((a, b) => b.semanticScore - a.semanticScore)
        .slice(0, top_k);
    }

    // Build response with enhancement info
    const enhancementInfo = enhancedQuery !== originalQuery
      ? `\n\n🔍 Query Enhancement:\nOriginal: "${originalQuery}"\nEnhanced: "${enhancedQuery}"`
      : '';

    const techContextInfo = Object.keys(tech_context).length > 0
      ? `\n\n📋 Tech Context: ${JSON.stringify(tech_context)}`
      : '';

    return {
      content: [
        {
          type: 'text',
          text: `🧠 AI-Retrieved ${cases.length} cases using ${mode} mode:${enhancementInfo}${techContextInfo}\n\n${this.formatSmartRetrievalResults(cases, searchQuery, mode)}`,
        },
      ],
    };
  }

  async handleRewardCase(db, args) {
    const { case_id, reward } = args;

    if (reward < -1.0 || reward > 1.0) {
      throw new McpError(ErrorCode.InvalidParams, 'Reward must be between -1.0 and 1.0');
    }

    // Get current Q-value and case data for context
    const currentResult = await db.query('SELECT * FROM cases WHERE case_id = $1', [case_id]);
    
    if (currentResult.rows.length === 0) {
      throw new McpError(ErrorCode.InvalidParams, `Case '${case_id}' not found`);
    }

    const caseData = currentResult.rows[0];
    const { q_value: currentQ = 0, reward_count = 0 } = caseData;
    
    // AI-enhanced Q-learning calculation
    const contextScore = this.ai.calculateContextRelevance(this.formatCase(caseData), {});
    const userConfidence = Math.min(1.0, (reward_count + 1) / 5); // Confidence grows with feedback
    
    const newQ = this.ai.calculateSmartQValue(currentQ, reward, contextScore, userConfidence);
    const newRewardCount = reward_count + 1;

    // Update user patterns for learning
    if (!this.ai.userPatterns.has(case_id)) {
      this.ai.userPatterns.set(case_id, { successRate: 0, totalFeedback: 0 });
    }
    
    const pattern = this.ai.userPatterns.get(case_id);
    pattern.totalFeedback++;
    pattern.successRate = ((pattern.successRate * (pattern.totalFeedback - 1)) + (reward > 0 ? 1 : 0)) / pattern.totalFeedback;

    await db.query(
      'UPDATE cases SET q_value = $1, reward = $2, reward_count = $3 WHERE case_id = $4',
      [newQ, reward, newRewardCount, case_id]
    );

    // Calculate learning insights
    const improvement = newQ - currentQ;
    const confidence = this.ai.calculateConfidence(newQ, newRewardCount);

    return {
      content: [
        {
          type: 'text',
          text: `🧠 Smart Q-learning update applied to case '${case_id}':\n` +
                `• Previous Q-value: ${currentQ.toFixed(3)}\n` +
                `• New Q-value: ${newQ.toFixed(3)} (${improvement >= 0 ? '+' : ''}${improvement.toFixed(3)})\n` +
                `• AI Confidence: ${(confidence * 100).toFixed(1)}%\n` +
                `• Context Score: ${(contextScore * 100).toFixed(1)}%\n` +
                `• Success Rate: ${(pattern.successRate * 100).toFixed(1)}%\n` +
                `• Reward given: ${reward}\n` +
                `• Total feedback: ${newRewardCount}`,
        },
      ],
    };
  }

  async handleGetCase(db, args) {
    const { case_id } = args;

    const result = await db.query('SELECT * FROM cases WHERE case_id = $1', [case_id]);
    
    if (result.rows.length === 0) {
      throw new McpError(ErrorCode.InvalidParams, `Case '${case_id}' not found`);
    }

    const caseData = this.formatCase(result.rows[0]);

    return {
      content: [
        {
          type: 'text',
          text: `📄 Case Details:\n\n${this.formatCaseDetails(caseData)}`,
        },
      ],
    };
  }

  async handleGetTopCases(db, args) {
    const { limit = 10 } = args;

    const result = await db.query('SELECT * FROM cases ORDER BY q_value DESC LIMIT $1', [limit]);
    const cases = result.rows.map(this.formatCase);

    return {
      content: [
        {
          type: 'text',
          text: `🏆 Top ${cases.length} performing cases by Q-value:\n\n${this.formatTopCases(cases)}`,
        },
      ],
    };
  }

  async handleSubmitFeedback(db, args) {
    const { case_id, feedback_type, feedback_data = {} } = args;

    // AI analysis of feedback pattern
    const pattern = this.ai.analyzeFeedbackPattern(case_id, feedback_type, feedback_data);
    
    // Log feedback with AI insights
    console.log('Smart feedback received:', { 
      case_id, 
      feedback_type, 
      feedback_data, 
      pattern,
      timestamp: new Date() 
    });

    // Extract insights from feedback
    const sentiment = pattern.sentiment / pattern.count;
    const frequency = pattern.count;
    
    let insights = '';
    if (frequency > 3) {
      if (sentiment > 0.3) {
        insights = `\n• 🔍 AI Insight: This case is consistently helpful (${frequency} feedback items, positive sentiment)`;
      } else if (sentiment < -0.2) {
        insights = `\n• ⚠️ AI Insight: This case may need improvement (${frequency} feedback items, negative trend)`;
      }
    }

    return {
      content: [
        {
          type: 'text',
          text: `💬 Smart feedback analysis for case '${case_id}':\n` +
                `• Type: ${feedback_type}\n` +
                `• Sentiment Score: ${sentiment.toFixed(2)}\n` +
                `• Feedback Frequency: ${frequency}\n` +
                `• Timestamp: ${new Date().toISOString()}${insights}\n` +
                `• Thank you for helping train our AI system!`,
        },
      ],
    };
  }
  async handleAddCaseFromConversation(db, args) {
    const { question, answer, tags = [], case_id = null, category = null } = args;

    // Validate required fields
    if (!question || !answer) {
      throw new McpError(ErrorCode.InvalidParams, 'question and answer are required');
    }

    try {
      // Generate case_id if not provided
      let finalCaseId = case_id;
      if (!finalCaseId) {
        // Generate a readable case ID from the question
        const slug = question
          .toLowerCase()
          .replace(/[^a-z0-9]+/g, '-')
          .replace(/^-+|-+$/g, '')
          .substring(0, 50);
        finalCaseId = `conversation-${slug}-${Date.now()}`;
      }

      // Check if case already exists
      const existing = await db.query('SELECT case_id FROM cases WHERE case_id = $1', [finalCaseId]);
      if (existing.rows.length > 0) {
        // Update existing case
        finalCaseId = `${finalCaseId}-${Date.now()}`;
      }

      // Auto-extract tags if not provided
      let finalTags = tags.length > 0 ? tags : this.extractTagsFromQuestion(question);

      // Build solution object
      const solution = {
        description: `Knowledge captured from conversation: ${question}`,
        question: question,
        answer: answer,
        source: 'conversation',
        category: category || 'general',
        capturedAt: new Date().toISOString(),
      };

      // Insert case into database
      const sql = `INSERT INTO cases (case_id, task, tags, solution) VALUES ($1, $2, $3, $4)`;
      await db.query(sql, [
        finalCaseId,
        question,
        JSON.stringify(finalTags),
        JSON.stringify(solution),
      ]);

      return {
        content: [
          {
            type: 'text',
            text: `✅ Case added to knowledge base:\n` +
                  `• Case ID: ${finalCaseId}\n` +
                  `• Question: ${question.substring(0, 100)}${question.length > 100 ? '...' : ''}\n` +
                  `• Tags: ${finalTags.join(', ')}\n` +
                  `• Category: ${category || 'general'}`,
          },
        ],
      };
    } catch (error) {
      throw new McpError(ErrorCode.InternalError, `Failed to add case from conversation: ${error.message}`);
    }
  }

  async handleEnrichContext(db, args) {
    const {
      query,
      case_ids = null,
      mode = 'parametric',
      top_k = 5,
      max_sentences_per_case = 3,
      min_relevance_score = 0.1,
      format = 'text',
      include_full_solution = false,
      tags = []
    } = args;

    try {
      let cases = [];

      // If specific case IDs provided, fetch those cases
      if (case_ids && case_ids.length > 0) {
        const placeholders = case_ids.map((_, i) => `$${i + 1}`).join(',');
        const result = await db.query(
          `SELECT * FROM cases WHERE case_id IN (${placeholders})`,
          case_ids
        );
        cases = result.rows.map(this.formatCase);
      } else {
        // Otherwise, retrieve relevant cases using existing retrieval logic
        let sql = 'SELECT * FROM cases';
        let params = [];

        if (tags.length > 0) {
          sql += ' WHERE tags LIKE $1';
          params.push(`%${JSON.stringify(tags)}%`);
        }

        const result = await db.query(sql, params);
        cases = result.rows.map(this.formatCase);

        // Apply retrieval ranking
        if (mode === 'parametric') {
          cases = this.ai.predictCaseRelevance(cases, query, tags);
          const confidentCases = cases.filter(c => c.confidence > AI_CONFIG.qlearning.confidenceThreshold);
          cases = confidentCases.length >= top_k ? confidentCases : cases;
          cases = cases.slice(0, top_k);
        } else {
          cases = cases
            .map(caseItem => ({
              ...caseItem,
              semanticScore: this.ai.calculateSemanticSimilarity(query, caseItem.task)
            }))
            .filter(c => c.semanticScore > 0.1)
            .sort((a, b) => b.semanticScore - a.semanticScore)
            .slice(0, top_k);
        }
      }

      if (cases.length === 0) {
        return {
          content: [
            {
              type: 'text',
              text: `⚠️ No cases found to enrich context for query: "${query}"`,
            },
          ],
        };
      }

      // Initialize context enrichment with similarity calculator
      const enrichment = new ContextEnrichment(this.ai);

      // Enrich context with query-relevant content
      const enrichedContext = await enrichment.enrichPromptContext(query, cases, {
        maxCases: top_k,
        maxSentencesPerCase: max_sentences_per_case,
        minRelevanceScore: min_relevance_score,
        includeFullSolution: include_full_solution
      });

      // Format output
      let outputText;
      if (format === 'json') {
        outputText = JSON.stringify(enrichedContext, null, 2);
      } else {
        // Use the formatting method
        outputText = enrichment.formatContextForPrompt(enrichedContext, 'text');

        // Add metadata summary
        outputText += `\n---\n`;
        outputText += `Summary: Enriched ${enrichedContext.totalCases} cases with average compression ratio of ${enrichedContext.averageCompressionRatio}\n`;
        outputText += `Each case was compressed to show only the most query-relevant content.\n`;
      }

      return {
        content: [
          {
            type: 'text',
            text: `✨ Context enriched for query: "${query}"\n\n${outputText}`,
          },
        ],
      };
    } catch (error) {
      throw new McpError(ErrorCode.InternalError, `Failed to enrich context: ${error.message}`);
    }
  }

  // Extract relevant tags from a question (helper method)
  extractTagsFromQuestion(question) {
    const commonTags = {
      'performance': ['performance', 'optimization', 'speed', 'fast', 'slow', 'efficient'],
      'cpu': ['cpu', 'processor', 'branch', 'prediction', 'pipeline'],
      'memory': ['memory', 'cache', 'heap', 'stack', 'allocation'],
      'algorithm': ['algorithm', 'sort', 'search', 'complexity', 'big-o'],
      'array': ['array', 'list', 'vector', 'sorted', 'unsorted'],
      'javascript': ['javascript', 'js', 'node', 'typescript'],
      'python': ['python', 'django', 'flask'],
      'database': ['database', 'sql', 'query', 'postgresql', 'mysql'],
      'web': ['web', 'http', 'api', 'rest', 'frontend', 'backend'],
      'computer-science': ['computer-science', 'cs', 'theory'],
    };

    const questionLower = question.toLowerCase();
    const extractedTags = [];

    for (const [tag, keywords] of Object.entries(commonTags)) {
      if (keywords.some(keyword => questionLower.includes(keyword))) {
        extractedTags.push(tag);
      }
    }

    // Always add 'conversation' tag for cases added from conversation
    if (!extractedTags.includes('conversation')) {
      extractedTags.push('conversation');
    }

    return extractedTags.length > 0 ? extractedTags : ['conversation', 'general'];
  }

  // Convert Stack Overflow question to case format (helper method)
  convertQuestionToCase(question) {
    return {
      case_id: `stackoverflow-${question.id}`,
      task: question.title,
      tags: question.tags || [],
      solution: {
        description: `Stack Overflow question: ${question.title}`,
        questionBody: question.body,
        link: question.link,
        score: question.score,
        viewCount: question.viewCount,
        answerCount: question.answerCount,
        source: 'stackoverflow',
        sourceId: question.id.toString(),
        ...(question.acceptedAnswer && {
          acceptedAnswer: {
            body: question.acceptedAnswer.body,
            score: question.acceptedAnswer.score,
            owner: question.acceptedAnswer.owner,
          },
        }),
      },
    };
  }

  // Utility methods
  formatCase(row) {
    return {
      case_id: row.case_id,
      task: row.task,
      tags: row.tags ? JSON.parse(row.tags) : [],
      solution: row.solution ? JSON.parse(row.solution) : {},
      q_value: row.q_value || 0,
      reward_count: row.reward_count || 0,
      created_at: row.created_at,
    };
  }

  formatSearchResults(cases) {
    return cases.map((c, i) => 
      `${i + 1}. **${c.case_id}** (Q: ${c.q_value?.toFixed(3) || '0.000'})\n` +
      `   Task: ${c.task}\n` +
      `   Tags: ${c.tags.join(', ')}\n`
    ).join('\n');
  }

  formatRetrievalResults(cases, query) {
    return `Query: "${query}"\n\n` + cases.map((c, i) => 
      `${i + 1}. **${c.case_id}** (Q: ${c.q_value?.toFixed(3) || '0.000'})\n` +
      `   ${c.task}\n` +
      `   Tags: ${c.tags.join(', ')}\n`
    ).join('\n');
  }

  formatSmartRetrievalResults(cases, query, mode) {
    const header = `🔍 Query: "${query}" | Mode: ${mode}\n\n`;
    
    return header + cases.map((c, i) => {
      let scoreInfo = '';
      if (c.aiScore !== undefined) {
        scoreInfo = `AI: ${(c.aiScore * 100).toFixed(0)}% | Conf: ${(c.confidence * 100).toFixed(0)}% | `;
      } else if (c.semanticScore !== undefined) {
        scoreInfo = `Semantic: ${(c.semanticScore * 100).toFixed(0)}% | `;
      }
      
      return `${i + 1}. **${c.case_id}** (${scoreInfo}Q: ${c.q_value?.toFixed(3) || '0.000'})\n` +
             `   ${c.task}\n` +
             `   Tags: ${c.tags.join(', ')}\n`;
    }).join('\n');
  }

  formatCaseDetails(caseData) {
    return `**ID:** ${caseData.case_id}\n` +
           `**Task:** ${caseData.task}\n` +
           `**Tags:** ${caseData.tags.join(', ')}\n` +
           `**Q-value:** ${caseData.q_value?.toFixed(3) || '0.000'}\n` +
           `**Reward Count:** ${caseData.reward_count}\n` +
           `**Created:** ${caseData.created_at}\n\n` +
           `**Solution:**\n${JSON.stringify(caseData.solution, null, 2)}`;
  }

  formatTopCases(cases) {
    return cases.map((c, i) => 
      `${i + 1}. **${c.case_id}** - Q: ${c.q_value?.toFixed(3) || '0.000'} (${c.reward_count} rewards)\n` +
      `   ${c.task.substring(0, 80)}${c.task.length > 80 ? '...' : ''}\n`
    ).join('\n');
  }

  async run() {
    // Initialize database
    await initializeDatabase();
    
    // Create HTTP server for health checks
    const httpServer = http.createServer((req, res) => {
      if (req.url === '/health') {
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ 
          status: 'healthy', 
          timestamp: new Date().toISOString(),
          server: 'solvedbook-mcp-server',
          version: '1.0.0'
        }));
      } else {
        res.writeHead(404, { 'Content-Type': 'text/plain' });
        res.end('Not Found');
      }
    });
    
    const port = process.env.PORT || 3000;
    httpServer.listen(port, () => {
      console.error(`Health check server running on port ${port}`);
    });
    
    // Create transport
    const transport = new StdioServerTransport();
    
    // Connect server to transport
    await this.server.connect(transport);
    
    console.error('SolvedBook MCP Server running on stdio');
  }
}

// Start the server
const server = new SolvedBookMCPServer();
server.run().catch(console.error);
