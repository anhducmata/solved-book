/**
 * Advanced AI Algorithms for SolvedBook MCP Server
 * Implements Memento-style agent fine-tuning without LLM modification
 *
 * Based on concepts from "Memento: Fine-tuning LLM Agents without Fine-tuning LLMs"
 */

export class SolvedBookAlgorithms {
  constructor(config = {}) {
    this.config = {
      learningRate: 0.1,
      decayFactor: 0.95,
      explorationRate: 0.05,
      confidenceThreshold: 0.7,
      maxMemorySize: 10000,
      semanticWeight: 0.6,
      contextWeight: 0.3,
      qValueWeight: 0.1,
      diversityPenalty: 0.1,
      qualityThreshold: 0.3,
      memoryDecayDays: 30,
      // Hierarchical memory settings
      maxTraceLength: 50,
      compressionThreshold: 20,
      // Vector embedding settings
      embeddingWeight: 0.7,
      traditionalWeight: 0.3,
      // Adaptive decay settings
      usageBasedDecay: true,
      popularityStabilizationFactor: 0.8,
      // Clustering settings
      maxClusters: 10,
      clusterSamplingRate: 0.3,
      // Meta-feedback settings
      feedbackWeight: 0.2,
      ...config,
    };

    // Memory buffers for different types of learning
    this.episodicMemory = new Map(); // Recent experiences with quality scores (short solutions)
    this.semanticMemory = new Map(); // Conceptual knowledge with diversity tracking
    this.proceduralMemory = new Map(); // How-to knowledge (long traces)
    this.visitCounts = new Map(); // Proper visit tracking
    this.domainConcepts = this.initializeDomainConcepts(); // Enhanced semantic understanding

    // Advanced memory structures
    this.hierarchicalMemory = new Map(); // Maps caseId -> {short, long, compressed}
    this.usageStats = new Map(); // Usage-based decay tracking
    this.solutionClusters = new Map(); // Clustering for diversity
    this.metaFeedback = new Map(); // User feedback for reinforcement
    this.embeddingCache = new Map(); // Cache for vector embeddings
  }

  /**
   * Initialize domain-specific concept mappings for better semantic understanding
   */
  initializeDomainConcepts() {
    return {
      algorithm: [
        "method",
        "approach",
        "technique",
        "strategy",
        "procedure",
        "process",
      ],
      database: [
        "storage",
        "persistence",
        "data",
        "repository",
        "sql",
        "nosql",
        "query",
      ],
      frontend: [
        "ui",
        "interface",
        "client",
        "browser",
        "react",
        "vue",
        "angular",
      ],
      backend: [
        "server",
        "api",
        "service",
        "endpoint",
        "microservice",
        "nodejs",
      ],
      security: [
        "auth",
        "authentication",
        "authorization",
        "encryption",
        "ssl",
        "jwt",
      ],
      performance: [
        "optimization",
        "speed",
        "cache",
        "benchmark",
        "profiling",
        "memory",
      ],
      testing: ["unittest", "integration", "e2e", "mock", "stub", "coverage"],
      deployment: ["docker", "kubernetes", "ci", "cd", "pipeline", "release"],
    };
  }

  /**
   * Advanced Q-Learning with quality control and Thompson sampling for cold start
   */
  updateQValueMemento(caseId, reward, userConfidence = 1.0, context = {}) {
    const currentQ = this.getQValue(caseId);
    const adaptiveLearningRate = this.config.learningRate * userConfidence;
    const contextScore = this.calculateContextRelevance(context);
    const explorationBonus = this.calculateExplorationBonus(caseId);

    // Memento-style Q-learning: external memory enhancement
    const contextAdjustedReward = reward * contextScore;
    const newQ =
      currentQ * this.config.decayFactor +
      adaptiveLearningRate * (contextAdjustedReward - currentQ) +
      explorationBonus;

    // Calculate experience quality before storing
    const experience = {
      reward,
      confidence: userConfidence,
      context,
      timestamp: Date.now(),
      qValue: Math.max(0, Math.min(1, newQ)),
    };

    const qualityScore = this.calculateExperienceQuality(experience);

    // Only store high-quality experiences
    if (qualityScore >= this.config.qualityThreshold) {
      experience.qualityScore = qualityScore;
      this.addToEpisodicMemoryWithQuality(caseId, experience);
      this.updateVisitCount(caseId);
    }

    return experience.qValue;
  }

  /**
   * Enhanced semantic similarity with multiple algorithms and domain concepts
   */
  calculateSemanticSimilarity(query, caseText) {
    const queryTerms = this.extractKeywords(query);
    const caseTerms = this.extractKeywords(caseText);

    if (queryTerms.length === 0 || caseTerms.length === 0) return 0;

    // Multiple similarity measures
    const jaccardSim = this.calculateJaccardSimilarity(queryTerms, caseTerms);
    const cosineSim = this.calculateCosineSimilarity(queryTerms, caseTerms);
    const conceptSim = this.calculateConceptualSimilarity(
      queryTerms,
      caseTerms
    );
    const editSim = this.calculateNormalizedEditDistance(query, caseText);

    // Weighted combination with improved weights
    return (
      jaccardSim * 0.25 + cosineSim * 0.35 + conceptSim * 0.3 + editSim * 0.1
    );
  }

  /**
   * Jaccard similarity with proper null handling
   */
  calculateJaccardSimilarity(terms1, terms2) {
    const set1 = new Set(terms1);
    const set2 = new Set(terms2);
    const intersection = new Set([...set1].filter((x) => set2.has(x)));
    const union = new Set([...set1, ...set2]);
    return union.size === 0 ? 0 : intersection.size / union.size;
  }

  /**
   * Cosine similarity using term frequency
   */
  calculateCosineSimilarity(terms1, terms2) {
    const freq1 = this.getTermFrequency(terms1);
    const freq2 = this.getTermFrequency(terms2);

    const allTerms = new Set([...Object.keys(freq1), ...Object.keys(freq2)]);

    let dotProduct = 0;
    let magnitude1 = 0;
    let magnitude2 = 0;

    for (const term of allTerms) {
      const f1 = freq1[term] || 0;
      const f2 = freq2[term] || 0;

      dotProduct += f1 * f2;
      magnitude1 += f1 * f1;
      magnitude2 += f2 * f2;
    }

    const magnitude = Math.sqrt(magnitude1) * Math.sqrt(magnitude2);
    return magnitude === 0 ? 0 : dotProduct / magnitude;
  }

  cosineSimilarity(vecA, vecB) {
    if (!vecA || !vecB || vecA.length !== vecB.length) {
        throw new Error("Invalid vectors for cosine similarity");
    }

    let dot = 0.0;
    let normA = 0.0;
    let normB = 0.0;

    for (let i = 0; i < vecA.length; i++) {
        dot += vecA[i] * vecB[i];
        normA += vecA[i] * vecA[i];
        normB += vecB[i] * vecB[i];
    }

    if (normA === 0 || normB === 0) return 0.0;
    return dot / (Math.sqrt(normA) + 1e-10) / (Math.sqrt(normB) + 1e-10);
}

  /**
   * Domain-aware conceptual similarity
   */
  calculateConceptualSimilarity(terms1, terms2) {
    let conceptMatches = 0;
    let totalComparisons = 0;

    for (const term1 of terms1) {
      for (const term2 of terms2) {
        totalComparisons++;
        if (this.areConceptsRelated(term1, term2)) {
          conceptMatches++;
        }
      }
    }

    return totalComparisons === 0 ? 0 : conceptMatches / totalComparisons;
  }

  /**
   * Check if two terms are conceptually related using domain knowledge
   */
  areConceptsRelated(term1, term2) {
    if (term1 === term2) return true;

    for (const [concept, related] of Object.entries(this.domainConcepts)) {
      if (
        (related.includes(term1) || concept === term1) &&
        (related.includes(term2) || concept === term2)
      ) {
        return true;
      }
    }

    // Check for common stems (improved version)
    return this.haveSimilarStems(term1, term2);
  }

  /**
   * Improved stemming comparison
   */
  haveSimilarStems(term1, term2) {
    if (term1.length < 4 || term2.length < 4) return false;

    // Check prefixes of different lengths
    const minLength = Math.min(term1.length, term2.length);
    const stemLength = Math.max(4, Math.floor(minLength * 0.7));

    return term1.substring(0, stemLength) === term2.substring(0, stemLength);
  }

  /**
   * Normalized edit distance for string similarity
   */
  calculateNormalizedEditDistance(str1, str2) {
    const distance = this.levenshteinDistance(str1, str2);
    const maxLength = Math.max(str1.length, str2.length);
    return maxLength === 0 ? 1 : 1 - distance / maxLength;
  }

  /**
   * Levenshtein distance calculation
   */
  levenshteinDistance(str1, str2) {
    const matrix = Array(str2.length + 1)
      .fill(null)
      .map(() => Array(str1.length + 1).fill(null));

    for (let i = 0; i <= str1.length; i++) matrix[0][i] = i;
    for (let j = 0; j <= str2.length; j++) matrix[j][0] = j;

    for (let j = 1; j <= str2.length; j++) {
      for (let i = 1; i <= str1.length; i++) {
        const cost = str1[i - 1] === str2[j - 1] ? 0 : 1;
        matrix[j][i] = Math.min(
          matrix[j][i - 1] + 1, // insertion
          matrix[j - 1][i] + 1, // deletion
          matrix[j - 1][i - 1] + cost // substitution
        );
      }
    }

    return matrix[str2.length][str1.length];
  }

  /**
   * Calculate term frequency for cosine similarity
   */
  getTermFrequency(terms) {
    const freq = {};
    for (const term of terms) {
      freq[term] = (freq[term] || 0) + 1;
    }
    return freq;
  }

  /**
   * Enhanced Multi-factor case scoring with advanced features
   * This is the core of the Memento approach - external intelligence
   */
  async calculateAdvancedMementoScore(
    query,
    caseData,
    userContext = {},
    embeddingFunction = null
  ) {
    // Validate required case data properties
    const caseText = (caseData.task || "") + " " + (caseData.solution || "");
    if (!caseText.trim()) {
      console.warn("Case data missing task or solution fields:", caseData);
      return {
        finalScore: 0,
        breakdown: {
          semantic: 0,
          context: 0,
          qValue: 0,
          confidence: 0,
          diversity: 1.0,
          error: "Missing task or solution data",
        },
      };
    }

    // Use hybrid similarity if embeddings available
    const semanticScore = embeddingFunction
      ? await this.calculateHybridSimilarity(query, caseText, embeddingFunction)
      : this.calculateSemanticSimilarity(query, caseText);

    const contextScore = this.calculateContextRelevance({
      tags: caseData.tags,
      userHistory: userContext.history,
      currentTask: query,
    });

    // Apply adaptive decay to Q-values
    const baseQValue = this.normalizeQValue(caseData.q_value || 0);
    const adaptiveQValue = this.applyAdaptiveDecay(caseData.id, baseQValue);

    // Enhanced confidence calculation
    const confidence = this.calculateConfidence(caseData);
    const confidenceMultiplier =
      confidence > this.config.confidenceThreshold ? 1.0 : 0.5;

    // Apply diversity penalty with clustering
    const contextKey = this.generateContextKey({
      currentTask: query,
      tags: caseData.tags,
    });
    const pattern = this.semanticMemory.get(contextKey);
    const diversityMultiplier = pattern ? pattern.diversity : 1.0;

    // Meta-feedback influence
    const feedbackData = this.metaFeedback.get(caseData.id);
    const feedbackMultiplier = feedbackData
      ? 0.5 + feedbackData.avgHelpfulness * this.config.feedbackWeight
      : 1.0;

    // Update usage statistics
    if (caseData.id) {
      this.updateUsageStats(caseData.id);
    }

    const finalScore =
      (semanticScore * this.config.semanticWeight +
        contextScore * this.config.contextWeight +
        adaptiveQValue * this.config.qValueWeight) *
      confidenceMultiplier *
      diversityMultiplier *
      feedbackMultiplier;

    return {
      finalScore,
      breakdown: {
        semantic: semanticScore,
        context: contextScore,
        qValue: adaptiveQValue,
        confidence: confidence,
        diversity: diversityMultiplier,
        feedback: feedbackMultiplier,
        hybrid: embeddingFunction ? true : false,
      },
    };
  }

  /**
   * Experience replay mechanism inspired by Memento paper Algorithm 1
   * Implements batch learning from stored trajectories (lines 9-10)
   */
  performExperienceReplay(batchSize = 32) {
    const experiences = Array.from(this.episodicMemory.values())
      .sort((a, b) => b.timestamp - a.timestamp)
      .slice(0, batchSize);

    const insights = {
      avgReward: 0,
      successPatterns: [],
      failurePatterns: [],
      optimalActions: new Map(),
      qValueUpdates: 0,
    };

    if (experiences.length === 0) return insights;

    // Batch Q-learning updates from replay buffer (Memento Eq. 10)
    experiences.forEach((exp) => {
      insights.avgReward += exp.reward;

      if (exp.reward > 0.7) {
        insights.successPatterns.push(exp.context);
        // Positive reinforcement for successful patterns
        this.reinforceSuccessPattern(exp);
      } else if (exp.reward < 0.3) {
        insights.failurePatterns.push(exp.context);
        // Learn from failures (critical for Memento approach)
        this.analyzeFailurePattern(exp);
      }

      insights.qValueUpdates++;
    });

    insights.avgReward /= experiences.length;
    return insights;
  }

  /**
   * Reinforce successful patterns (from Memento's success analysis)
   */
  reinforceSuccessPattern(experience) {
    // Store successful context patterns in semantic memory
    const contextKey = this.generateContextKey(experience.context);
    const existing = this.semanticMemory.get(contextKey) || {
      count: 0,
      avgReward: 0,
      lastSeen: 0,
      diversity: 1.0,
    };

    // Calculate days since last seen BEFORE updating lastSeen
    const daysSinceLastSeen =
      existing.lastSeen > 0
        ? (Date.now() - existing.lastSeen) / (1000 * 60 * 60 * 24)
        : 0;

    existing.count++;
    existing.avgReward =
      (existing.avgReward * (existing.count - 1) + experience.reward) /
      existing.count;
    existing.lastSeen = Date.now(); // Update after calculating days

    // Apply diversity penalty for overused patterns
    if (existing.count > 10) {
      existing.diversity = Math.max(0.3, 1.0 - existing.count / 100);
    }

    // Time decay for old patterns
    if (daysSinceLastSeen > this.config.memoryDecayDays) {
      existing.diversity *= 0.9; // Decay old patterns
    }

    this.semanticMemory.set(contextKey, existing);
  }

  /**
   * Analyze failure patterns to avoid repeating mistakes
   */
  analyzeFailurePattern(experience) {
    // Store failure patterns in procedural memory for avoidance
    const contextKey = this.generateContextKey(experience.context);
    const existing = this.proceduralMemory.get(contextKey) || {
      failures: 0,
      lessons: [],
    };

    existing.failures++;
    existing.lessons.push({
      reason: "low_reward",
      reward: experience.reward,
      timestamp: experience.timestamp,
    });
    this.proceduralMemory.set(contextKey, existing);
  }

  /**
   * Adaptive exploration strategy
   * Balances exploitation of known good solutions with exploration
   */
  calculateExplorationBonus(caseId) {
    const visitCount = this.getVisitCount(caseId);
    const totalVisits = this.getTotalVisits();

    // UCB1-style exploration bonus with cap to prevent instability
    if (visitCount === 0) return this.config.explorationRate;
    if (totalVisits === 0) return this.config.explorationRate;

    const bonus =
      this.config.explorationRate *
      Math.sqrt(Math.log(totalVisits) / visitCount);

    // Cap the exploration bonus to prevent unrealistically high values
    return Math.min(bonus, this.config.explorationRate * 5);
  }

  /**
   * Context relevance calculation
   * Measures how well a case fits the current context
   */
  calculateContextRelevance(context = {}) {
    let relevanceScore = 0.5; // Base relevance

    // Tag matching
    if (context.tags && context.currentTask) {
      const queryKeywords = this.extractKeywords(context.currentTask);
      const tagKeywords = this.extractKeywords(context.tags);
      const tagMatch = this.calculateSemanticSimilarity(
        queryKeywords.join(" "),
        tagKeywords.join(" ")
      );
      relevanceScore += tagMatch * 0.3;
    }

    // User history patterns
    if (context.userHistory) {
      const historyRelevance = this.calculateHistoryRelevance(context);
      relevanceScore += historyRelevance * 0.2;
    }

    return Math.max(0, Math.min(1, relevanceScore));
  }

  /**
   * Keyword extraction with stop-word filtering
   * Essential for semantic analysis
   */
  extractKeywords(text) {
    const stopWords = new Set([
      "the",
      "a",
      "an",
      "and",
      "or",
      "but",
      "in",
      "on",
      "at",
      "to",
      "for",
      "of",
      "with",
      "by",
      "is",
      "are",
      "was",
      "were",
      "be",
      "been",
      "have",
      "has",
      "had",
      "do",
      "does",
      "did",
      "will",
      "would",
      "could",
      "should",
    ]);

    return text
      .toLowerCase()
      .replace(/[^\w\s]/g, " ")
      .split(/\s+/)
      .filter((word) => word.length > 2 && !stopWords.has(word))
      .filter(Boolean);
  }

  /**
   * Calculate confidence with Thompson sampling for cold start
   * Higher confidence for cases with more positive feedback
   */
  calculateConfidence(caseData) {
    const rewardCount = caseData.reward_count || 0;
    const avgReward = caseData.reward || 0;

    if (rewardCount === 0) {
      // Thompson sampling for exploration of new cases
      return 0.5 + (Math.random() - 0.5) * 0.4; // Range: 0.3 to 0.7
    }

    // Beta distribution approximation for uncertainty
    const alpha = avgReward * rewardCount + 1;
    const beta = (1 - avgReward) * rewardCount + 1;

    // Uncertainty decreases with more samples
    const uncertainty = Math.sqrt(
      (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))
    );
    const baseConfidence = avgReward;

    // Add uncertainty bonus for exploration
    return Math.max(0.1, Math.min(0.9, baseConfidence + uncertainty * 0.2));
  }

  /**
   * Normalize Q-values for scoring
   */
  normalizeQValue(qValue) {
    return Math.max(0, Math.min(1, qValue));
  }

  // Helper methods for memory management
  addToEpisodicMemory(caseId, experience) {
    this.episodicMemory.set(caseId, experience);

    // Limit memory size
    if (this.episodicMemory.size > this.config.maxMemorySize) {
      const oldestKey = this.episodicMemory.keys().next().value;
      this.episodicMemory.delete(oldestKey);
    }
  }

  /**
   * Quality-based memory management
   */
  addToEpisodicMemoryWithQuality(caseId, experience) {
    this.episodicMemory.set(caseId, experience);

    // Remove lowest quality experiences when at capacity
    if (this.episodicMemory.size > this.config.maxMemorySize) {
      const sortedEntries = Array.from(this.episodicMemory.entries()).sort(
        (a, b) => a[1].qualityScore - b[1].qualityScore
      );

      // Remove only enough entries to get back under the limit
      const toRemove = this.episodicMemory.size - this.config.maxMemorySize;
      for (let i = 0; i < toRemove; i++) {
        this.episodicMemory.delete(sortedEntries[i][0]);
      }
    }
  }

  /**
   * Calculate experience quality based on multiple factors
   */
  calculateExperienceQuality(experience) {
    const ageDays = (Date.now() - experience.timestamp) / (1000 * 60 * 60 * 24);
    const factors = {
      rewardReliability: experience.confidence,
      contextRichness: Math.min(1, Object.keys(experience.context).length / 10),
      recency: Math.exp(-ageDays / this.config.memoryDecayDays), // Fixed: proper normalization in days
      consistency: this.checkConsistencyWithHistory(experience),
    };

    return (
      factors.rewardReliability * 0.4 +
      factors.contextRichness * 0.2 +
      factors.recency * 0.2 +
      factors.consistency * 0.2
    );
  }

  /**
   * Check consistency of experience with historical patterns
   */
  checkConsistencyWithHistory(experience) {
    const contextKey = this.generateContextKey(experience.context);
    const historicalPattern = this.semanticMemory.get(contextKey);

    if (!historicalPattern || historicalPattern.avgReward === undefined) {
      return 0.5; // Neutral for new patterns or patterns without reward data
    }

    const rewardDiff = Math.abs(
      experience.reward - historicalPattern.avgReward
    );
    return Math.max(0, 1 - rewardDiff); // Higher consistency = lower difference
  }

  /**
   * Proper visit count tracking
   */
  updateVisitCount(caseId) {
    const current = this.visitCounts.get(caseId) || 0;
    this.visitCounts.set(caseId, current + 1);
  }

  getQValue(caseId) {
    const memory = this.episodicMemory.get(caseId);
    return memory ? memory.qValue : 0.5; // Default Q-value
  }

  getVisitCount(caseId) {
    return this.visitCounts.get(caseId) || 0;
  }

  getTotalVisits() {
    return Array.from(this.visitCounts.values()).reduce(
      (sum, count) => sum + count,
      0
    );
  }

  calculateExpandedSimilarity(terms1, terms2) {
    // Simplified semantic expansion - in production, use word embeddings
    const commonRoots = this.findCommonRoots(terms1, terms2);
    return commonRoots.length / Math.max(terms1.length, terms2.length);
  }

  findCommonRoots(terms1, terms2) {
    // Simple stemming-like comparison
    return terms1.filter((term1) =>
      terms2.some(
        (term2) =>
          term1.slice(0, 4) === term2.slice(0, 4) ||
          term2.slice(0, 4) === term1.slice(0, 4)
      )
    );
  }

  calculateHistoryRelevance(context) {
    // TODO: Implement based on user history data
    // This is a placeholder - actual implementation should analyze user's
    // historical success with similar cases based on context.userHistory
    return 0.5; // Placeholder - always returns neutral relevance
  }

  /**
   * Generate a context key for memory storage (helper for Memento approach)
   */
  generateContextKey(context) {
    const tags = context.tags || "";
    const taskType = context.currentTask
      ? this.extractKeywords(context.currentTask).slice(0, 3).join("_")
      : "";
    return `${tags}_${taskType}`;
  }

  /**
   * Soft Q-Learning update based on Memento paper Equation (8)
   * Implements the exact formula from the research
   */
  updateQValueSoftMemento(caseId, reward, nextStateCases = [], alpha = 0.1) {
    const currentQ = this.getQValue(caseId);
    const gamma = 0.95; // Discount factor

    // Calculate the soft value function V(s') = α * log(Σ exp(Q(s',c')/α))
    let nextStateValue = 0;
    if (nextStateCases.length > 0) {
      const expQValues = nextStateCases.map((c) =>
        Math.exp(this.getQValue(c) / alpha)
      );
      const sumExpQ = expQValues.reduce((sum, val) => sum + val, 0);
      nextStateValue = alpha * Math.log(sumExpQ);
    }

    // Memento Equation (8): Q(s,c) ← Q(s,c) + η[r + γV(s') - Q(s,c)]
    const learningRate = this.config.learningRate;
    const tdTarget = reward + gamma * nextStateValue;
    const newQ = currentQ + learningRate * (tdTarget - currentQ);

    return Math.max(0, Math.min(1, newQ));
  }

  /**
   * Retrieval policy based on Memento Equation (7)
   * μ*(c|s,M) = exp(Q*(s,M,c)/α) / Σ exp(Q*(s,M,c')/α)
   */
  calculateRetrievalPolicy(currentState, caseBank, alpha = 0.1) {
    const policies = new Map();
    let totalExpQ = 0;

    // Calculate exp(Q/α) for each case
    caseBank.forEach((caseData, caseId) => {
      const qValue = this.getQValue(caseId);
      const expQ = Math.exp(qValue / alpha);
      policies.set(caseId, expQ);
      totalExpQ += expQ;
    });

    // Normalize to get probability distribution
    policies.forEach((expQ, caseId) => {
      policies.set(caseId, expQ / totalExpQ);
    });

    return policies;
  }

  /**
   * Advanced Case-Based Reasoning Agent with all enhancement features
   */
  async getAdvancedCBRAgentPolicy(
    currentState,
    caseBank,
    embeddingFunction = null,
    useDiversification = true
  ) {
    // Use diversified retrieval if enabled
    const candidateCases = useDiversification
      ? this.getDiversifiedCases(currentState, caseBank)
      : Array.from(caseBank.entries()).map(([caseId, caseData]) => ({
          caseId,
          caseData,
        }));

    const retrievalPolicies = this.calculateRetrievalPolicy(
      currentState,
      caseBank
    );
    const weightedScores = new Map();

    // Process each case with advanced scoring
    for (const { caseId, caseData } of candidateCases) {
      const retrievalProb = retrievalPolicies.get(caseId) || 0;
      const caseScore = await this.calculateAdvancedMementoScore(
        currentState,
        { ...caseData, id: caseId },
        {},
        embeddingFunction
      );

      // Prepare hierarchical context if available
      const hierarchicalCase = this.retrieveHierarchicalCase(caseId, false);
      const metaData = this.metaFeedback.get(caseId);

      weightedScores.set(caseId, {
        finalScore: caseScore.finalScore * retrievalProb,
        retrievalProb,
        hierarchical: hierarchicalCase ? true : false,
        feedbackScore: metaData?.avgHelpfulness || 0.5,
        ...caseScore.breakdown,
      });
    }

    return weightedScores;
  }

  /**
   * Main retrieval method that combines all advanced features
   */
  async retrieveOptimalCases(query, caseBank, options = {}) {
    const {
      maxResults = 5,
      embeddingFunction = null, // optional vector embedding scorer
      useDiversification = true, // cluster-based diversification
      includeHierarchical = false, // include long traces if true
      prepareFewShot = false, // return cases formatted as few-shot examples
      fewShotCount = 3, // how many few-shot examples to prepare
    } = options;

    // 1. Compute base similarity scores
    let scored = this.computeCaseSimilarities(query, caseBank);

    // 2. Optional embedding integration
    if (embeddingFunction) {
      const queryEmbedding = await embeddingFunction(query);
      for (let s of scored) {
        const caseEmbedding = s.case.embedding;
        if (caseEmbedding) {
          const embedSim = cosineSimilarity(queryEmbedding, caseEmbedding);
          // Blend embedding score with existing score
          s.score = s.score * 0.7 + embedSim * 0.3;
        }
      }
    }

    // 3. Apply diversity via clustering if requested
    if (useDiversification && scored.length > maxResults) {
      scored = this.diversifyCases(scored, maxResults);
    } else {
      scored = scored.sort((a, b) => b.score - a.score).slice(0, maxResults);
    }

    // 4. Expand hierarchical traces if requested
    if (includeHierarchical) {
      scored = scored.map((s) => ({
        ...s,
        hierarchical: {
          short: s.case.solution.short,
          long: s.case.solution.long || null,
        },
      }));
    }

    // 5. Prepare few-shot style examples if requested
    if (prepareFewShot) {
      scored = scored.slice(0, fewShotCount).map((s) => ({
        role: "user",
        content: `Problem: ${s.case.problem.description}`,
        score: s.score,
        solution: s.case.solution.short.summary,
      }));
    }

    return scored;
  }

  // =================== ADVANCED FEATURES ===================

  /**
   * 1. HIERARCHICAL MEMORY REPRESENTATION
   * Store short canonical fixes and long detailed traces
   */
  storeHierarchicalCase(
    caseId,
    shortSolution,
    longTrace = null,
    attempts = []
  ) {
    const hierarchicalData = {
      short: {
        solution: shortSolution,
        timestamp: Date.now(),
        canonical: true,
      },
      long: longTrace
        ? {
            trace: longTrace,
            attempts: attempts,
            retryCount: attempts.length,
            timestamp: Date.now(),
          }
        : null,
      compressed: null, // Will be generated if needed
    };

    this.hierarchicalMemory.set(caseId, hierarchicalData);

    // Auto-compress if trace is too long
    if (longTrace && attempts.length > this.config.compressionThreshold) {
      this.compressAndSummarize(caseId);
    }
  }

  /**
   * Retrieve hierarchical case with expansion option
   */
  retrieveHierarchicalCase(caseId, expandToLong = false) {
    const hierarchicalData = this.hierarchicalMemory.get(caseId);
    if (!hierarchicalData) return null;

    if (expandToLong && hierarchicalData.long) {
      return {
        ...hierarchicalData.short,
        ...hierarchicalData.long,
        expanded: true,
      };
    }

    return hierarchicalData.short;
  }

  /**
   * 2. VECTOR EMBEDDINGS INTEGRATION
   * Hybrid search combining embeddings with traditional similarity
   */
  async calculateHybridSimilarity(query, caseText, embeddingFunction = null) {
    // Traditional multi-factor similarity
    const traditionalScore = this.calculateSemanticSimilarity(query, caseText);

    // Vector embedding similarity (if available)
    let embeddingScore = 0;
    if (embeddingFunction) {
      try {
        embeddingScore = await this.calculateEmbeddingSimilarity(
          query,
          caseText,
          embeddingFunction
        );
      } catch (error) {
        console.warn(
          "Embedding calculation failed, falling back to traditional:",
          error
        );
        embeddingScore = traditionalScore;
      }
    } else {
      embeddingScore = traditionalScore; // Fallback
    }

    // Hybrid combination
    return (
      embeddingScore * this.config.embeddingWeight +
      traditionalScore * this.config.traditionalWeight
    );
  }

  /**
   * Calculate cosine similarity between embeddings
   */
  async calculateEmbeddingSimilarity(query, caseText, embeddingFunction) {
    // Check cache first
    const queryKey = `query_${this.hashString(query)}`;
    const caseKey = `case_${this.hashString(caseText)}`;

    let queryEmbedding = this.embeddingCache.get(queryKey);
    let caseEmbedding = this.embeddingCache.get(caseKey);

    // Generate embeddings if not cached
    if (!queryEmbedding) {
      queryEmbedding = await embeddingFunction(query);
      this.embeddingCache.set(queryKey, queryEmbedding);
    }

    if (!caseEmbedding) {
      caseEmbedding = await embeddingFunction(caseText);
      this.embeddingCache.set(caseKey, caseEmbedding);
    }

    // Calculate cosine similarity
    return this.cosineSimilarityVectors(queryEmbedding, caseEmbedding);
  }

  /**
   * Cosine similarity for vectors
   */
  cosineSimilarityVectors(vecA, vecB) {
    if (vecA.length !== vecB.length) return 0;

    let dotProduct = 0;
    let magnitudeA = 0;
    let magnitudeB = 0;

    for (let i = 0; i < vecA.length; i++) {
      dotProduct += vecA[i] * vecB[i];
      magnitudeA += vecA[i] * vecA[i];
      magnitudeB += vecB[i] * vecB[i];
    }

    const magnitude = Math.sqrt(magnitudeA) * Math.sqrt(magnitudeB);
    return magnitude === 0 ? 0 : dotProduct / magnitude;
  }

  /**
   * 3. ADAPTIVE MEMORY DECAY
   * Usage-based decay with popularity stabilization
   */
  updateUsageStats(caseId) {
    const current = this.usageStats.get(caseId) || {
      accessCount: 0,
      lastAccessed: Date.now(),
      decayRate: 1.0,
      popularity: 0,
    };

    current.accessCount++;
    current.lastAccessed = Date.now();

    // Calculate popularity (recent access frequency)
    const daysSinceCreation =
      (Date.now() - (current.createdAt || Date.now())) / (1000 * 60 * 60 * 24);
    current.popularity = current.accessCount / Math.max(1, daysSinceCreation);

    // Adaptive decay rate based on usage
    if (current.popularity > 1.0) {
      // Popular cases decay slower
      current.decayRate = Math.max(
        0.1,
        1.0 - current.popularity * this.config.popularityStabilizationFactor
      );
    } else {
      // Unused cases decay faster
      const daysSinceLastAccess =
        (Date.now() - current.lastAccessed) / (1000 * 60 * 60 * 24);
      current.decayRate = Math.min(
        2.0,
        1.0 + daysSinceLastAccess / this.config.memoryDecayDays
      );
    }

    this.usageStats.set(caseId, current);
  }

  /**
   * Apply adaptive decay to Q-values
   */
  applyAdaptiveDecay(caseId, baseQValue) {
    if (!this.config.usageBasedDecay) return baseQValue;

    const usageStats = this.usageStats.get(caseId);
    if (!usageStats) return baseQValue;

    const decayFactor = Math.exp(-usageStats.decayRate);
    return baseQValue * decayFactor;
  }

  /**
   * 4. CASE COMPRESSION & SUMMARIZATION
   * Compress long traces into structured summaries
   */
  compressAndSummarize(caseId) {
    const hierarchicalData = this.hierarchicalMemory.get(caseId);
    if (!hierarchicalData?.long) return;

    const { attempts, trace } = hierarchicalData.long;

    // Extract key patterns from attempts
    const patterns = this.extractPatterns(attempts);
    const summary = this.generateSummary(trace, patterns);

    hierarchicalData.compressed = {
      summary,
      patterns,
      originalLength: attempts.length,
      compressionRatio: summary.length / trace.length,
      timestamp: Date.now(),
    };

    this.hierarchicalMemory.set(caseId, hierarchicalData);
  }

  /**
   * Extract common patterns from failed attempts
   */
  extractPatterns(attempts) {
    const patterns = {
      commonErrors: {},
      retryReasons: {},
      timeToSuccess: 0,
      successfulApproach: null,
    };

    attempts.forEach((attempt, index) => {
      if (attempt.error) {
        const errorType = attempt.error.type || "unknown";
        patterns.commonErrors[errorType] =
          (patterns.commonErrors[errorType] || 0) + 1;
      }

      if (attempt.retryReason) {
        patterns.retryReasons[attempt.retryReason] =
          (patterns.retryReasons[attempt.retryReason] || 0) + 1;
      }

      if (attempt.success && !patterns.successfulApproach) {
        patterns.successfulApproach = attempt.approach;
        patterns.timeToSuccess = index + 1;
      }
    });

    return patterns;
  }

  /**
   * Generate structured summary
   */
  generateSummary(trace, patterns) {
    return {
      overview: `Solution found after ${patterns.timeToSuccess} attempts`,
      mainErrors: Object.keys(patterns.commonErrors).slice(0, 3),
      successfulStrategy: patterns.successfulApproach,
      keyLearnings: this.extractKeyLearnings(patterns),
      originalTrace: trace.substring(0, 200) + "...", // Truncated original
    };
  }

  /**
   * 5. CLUSTERING FOR DIVERSITY
   * Cluster similar solutions and sample across clusters
   */
  updateSolutionClusters(caseId, solutionVector) {
    // Simple k-means style clustering
    let bestCluster = null;
    let bestDistance = Infinity;

    // Find closest existing cluster
    for (const [clusterId, cluster] of this.solutionClusters.entries()) {
      const distance = this.calculateClusterDistance(
        solutionVector,
        cluster.centroid
      );
      if (distance < bestDistance) {
        bestDistance = distance;
        bestCluster = clusterId;
      }
    }

    // Create new cluster if no good match found
    if (!bestCluster || this.solutionClusters.size < this.config.maxClusters) {
      if (bestDistance > 0.7) {
        // Threshold for new cluster
        const newClusterId = `cluster_${this.solutionClusters.size}`;
        this.solutionClusters.set(newClusterId, {
          centroid: [...solutionVector],
          members: [caseId],
          lastUpdated: Date.now(),
        });
        return newClusterId;
      }
    }

    // Add to existing cluster and update centroid
    if (bestCluster) {
      const cluster = this.solutionClusters.get(bestCluster);
      cluster.members.push(caseId);
      this.updateClusterCentroid(bestCluster, solutionVector);
      return bestCluster;
    }

    return null;
  }

  /**
   * Diversified retrieval across clusters
   */
  getDiversifiedCases(query, caseBank, maxResults = 10) {
    const clusteredCases = new Map();

    // Group cases by cluster
    caseBank.forEach((caseData, caseId) => {
      const clusterId = caseData.clusterId || "unclustered";
      if (!clusteredCases.has(clusterId)) {
        clusteredCases.set(clusterId, []);
      }
      clusteredCases.get(clusterId).push({ caseId, caseData });
    });

    // Sample from each cluster
    const samplesPerCluster = Math.max(
      1,
      Math.floor(maxResults / clusteredCases.size)
    );
    const diversifiedResults = [];

    clusteredCases.forEach((cases, clusterId) => {
      const sortedCases = cases
        .map(({ caseId, caseData }) => ({
          caseId,
          caseData,
          score: this.calculateMementoScore(query, caseData).finalScore,
        }))
        .sort((a, b) => b.score - a.score)
        .slice(0, samplesPerCluster);

      diversifiedResults.push(...sortedCases);
    });

    return diversifiedResults
      .sort((a, b) => b.score - a.score)
      .slice(0, maxResults);
  }

  /**
   * 6. META-FEEDBACK LOOPS
   * Collect and process user feedback for reinforcement
   */
  recordMetaFeedback(caseId, helpful, confidence = 1.0, details = {}) {
    const feedback = {
      helpful: helpful, // boolean
      confidence: confidence, // 0-1
      details: details, // additional context
      timestamp: Date.now(),
    };

    const existing = this.metaFeedback.get(caseId) || {
      feedbacks: [],
      avgHelpfulness: 0.5,
      totalFeedbacks: 0,
    };

    existing.feedbacks.push(feedback);
    existing.totalFeedbacks++;

    // Calculate weighted average helpfulness
    const weights = existing.feedbacks.map((f) => f.confidence);
    const weightedSum = existing.feedbacks.reduce(
      (sum, f, i) => sum + (f.helpful ? 1 : 0) * weights[i],
      0
    );
    const totalWeight = weights.reduce((sum, w) => sum + w, 0);

    existing.avgHelpfulness = totalWeight > 0 ? weightedSum / totalWeight : 0.5;

    this.metaFeedback.set(caseId, existing);

    // Update Q-value based on feedback
    this.updateQValueFromFeedback(caseId, feedback);
  }

  /**
   * Update Q-values based on meta-feedback
   */
  updateQValueFromFeedback(caseId, feedback) {
    const currentQ = this.getQValue(caseId);
    const feedbackReward = feedback.helpful ? 1.0 : 0.0;
    const learningRate = this.config.learningRate * this.config.feedbackWeight;

    const experience = this.episodicMemory.get(caseId);
    if (experience) {
      experience.qValue = currentQ + learningRate * (feedbackReward - currentQ);
      experience.qValue = Math.max(0, Math.min(1, experience.qValue));
      this.episodicMemory.set(caseId, experience);
    }
  }

  /**
   * 7. FEW-SHOT FINE-TUNING HOOKS
   * Prepare cases for LLM conditioning
   */
  prepareFewShotContext(selectedCases, currentQuery, maxExamples = 3) {
    const fewShotExamples = selectedCases
      .slice(0, maxExamples)
      .map(({ caseId, caseData }) => {
        const hierarchicalCase = this.retrieveHierarchicalCase(caseId, false);
        const metaData = this.metaFeedback.get(caseId);

        return {
          input: caseData.task || caseData.problem,
          output: hierarchicalCase?.solution || caseData.solution,
          context: {
            tags: caseData.tags,
            confidence: this.calculateConfidence(caseData),
            helpfulness: metaData?.avgHelpfulness || 0.5,
            clusterId: caseData.clusterId,
          },
        };
      });

    return {
      examples: fewShotExamples,
      currentQuery: currentQuery,
      contextualHints: this.generateContextualHints(selectedCases),
      adaptationInstructions:
        this.generateAdaptationInstructions(selectedCases),
    };
  }

  /**
   * Generate contextual hints for LLM conditioning
   */
  generateContextualHints(selectedCases) {
    const patterns = selectedCases.map(({ caseData }) => {
      const clusterId = caseData.clusterId;
      const cluster = this.solutionClusters.get(clusterId);
      return {
        approach: this.extractApproachFromSolution(caseData.solution),
        domain: this.identifyDomain(caseData.tags),
        complexity: this.estimateComplexity(caseData),
      };
    });

    return {
      commonApproaches: this.findCommonPatterns(
        patterns.map((p) => p.approach)
      ),
      primaryDomains: this.findCommonPatterns(patterns.map((p) => p.domain)),
      complexityRange: {
        min: Math.min(...patterns.map((p) => p.complexity)),
        max: Math.max(...patterns.map((p) => p.complexity)),
      },
    };
  }

  // =================== HELPER METHODS ===================

  hashString(str) {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i);
      hash = (hash << 5) - hash + char;
      hash = hash & hash; // Convert to 32-bit integer
    }
    return hash.toString();
  }

  calculateClusterDistance(vector1, vector2) {
    if (vector1.length !== vector2.length) return Infinity;

    let sum = 0;
    for (let i = 0; i < vector1.length; i++) {
      sum += Math.pow(vector1[i] - vector2[i], 2);
    }
    return Math.sqrt(sum);
  }

  updateClusterCentroid(clusterId, newVector) {
    const cluster = this.solutionClusters.get(clusterId);
    if (!cluster) return;

    const alpha = 0.1; // Learning rate for centroid update
    for (let i = 0; i < cluster.centroid.length; i++) {
      cluster.centroid[i] =
        cluster.centroid[i] * (1 - alpha) + newVector[i] * alpha;
    }
    cluster.lastUpdated = Date.now();
  }

  extractKeyLearnings(patterns) {
    const learnings = [];

    if (patterns.commonErrors) {
      const topError = Object.keys(patterns.commonErrors)[0];
      if (topError) learnings.push(`Avoid ${topError} errors`);
    }

    if (patterns.successfulApproach) {
      learnings.push(`Successful approach: ${patterns.successfulApproach}`);
    }

    return learnings;
  }

  extractApproachFromSolution(solution) {
    // Simple heuristic - in production, use more sophisticated analysis
    const keywords = this.extractKeywords(solution);
    const approaches = [
      "algorithmic",
      "database",
      "api",
      "frontend",
      "backend",
    ];

    for (const approach of approaches) {
      if (keywords.some((k) => this.domainConcepts[approach]?.includes(k))) {
        return approach;
      }
    }
    return "general";
  }

  identifyDomain(tags) {
    if (!tags) return "general";
    const tagWords = this.extractKeywords(tags);

    for (const [domain, concepts] of Object.entries(this.domainConcepts)) {
      if (tagWords.some((tag) => concepts.includes(tag) || domain === tag)) {
        return domain;
      }
    }
    return "general";
  }

  estimateComplexity(caseData) {
    // Simple complexity estimation based on solution length and attempts
    const solutionLength = (caseData.solution || "").length;
    const attemptCount = caseData.attempts?.length || 1;

    return Math.min(10, Math.max(1, Math.log10(solutionLength) + attemptCount));
  }

  findCommonPatterns(items) {
    const frequency = {};
    items.forEach((item) => {
      frequency[item] = (frequency[item] || 0) + 1;
    });

    return Object.entries(frequency)
      .sort(([, a], [, b]) => b - a)
      .slice(0, 3)
      .map(([item]) => item);
  }

  generateAdaptationInstructions(selectedCases) {
    const instructions = [];

    if (selectedCases.length > 0) {
      instructions.push("Consider the patterns from similar successful cases");
      instructions.push("Adapt the approach based on the specific context");
      instructions.push(
        "Leverage proven strategies while avoiding known pitfalls"
      );
    }

    return instructions;
  }
}

export default SolvedBookAlgorithms;
