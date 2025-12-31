/**
 * Context Enrichment Module
 * Uses similarity-based attribution to extract query-relevant content from cases
 * Based on: https://towardsdatascience.com/explaining-llms-for-rag-and-summarization-067e486020b4/
 *
 * This module helps solidify context by identifying and extracting the most relevant
 * parts of case solutions for a given query, making prompt context more focused and efficient.
 */

export class ContextEnrichment {
  constructor(similarityCalculator = null) {
    // Optional: can use existing similarity calculator from AdvancedAI
    this.similarityCalculator = similarityCalculator;
  }

  /**
   * Split text into sentences
   * Simple sentence splitting - handles common punctuation
   */
  splitIntoSentences(text) {
    if (!text || typeof text !== 'string') return [];

    // Clean HTML tags if present (from Stack Overflow content)
    const cleanText = text.replace(/<[^>]*>/g, ' ').replace(/\s+/g, ' ').trim();

    // Split on sentence-ending punctuation followed by space or end of string
    const sentences = cleanText
      .replace(/([.!?])\s+/g, '$1|SPLIT|')
      .replace(/([.!?])$/g, '$1|SPLIT|')
      .split('|SPLIT|')
      .map(s => s.trim())
      .filter(s => s.length > 10); // Filter out very short fragments

    return sentences.length > 0 ? sentences : [cleanText]; // Fallback to whole text if no sentences found
  }

  /**
   * Calculate similarity between two texts using keyword overlap
   * Enhanced version that works at sentence level
   */
  calculateRelevance(query, text) {
    if (this.similarityCalculator) {
      return this.similarityCalculator.calculateSemanticSimilarity(query, text);
    }

    // Fallback: simple keyword-based similarity
    const queryWords = this.extractKeywords(query.toLowerCase());
    const textWords = this.extractKeywords(text.toLowerCase());

    if (queryWords.length === 0 || textWords.length === 0) return 0;

    const intersection = queryWords.filter(word => textWords.includes(word));
    const union = [...new Set([...queryWords, ...textWords])];

    const jaccardSim = intersection.length / union.length;
    const lengthSim = 1 - Math.abs(queryWords.length - textWords.length) / Math.max(queryWords.length, textWords.length, 1);

    return (jaccardSim * 0.7 + lengthSim * 0.3);
  }

  /**
   * Extract keywords from text (similar to AdvancedAI implementation)
   */
  extractKeywords(text) {
    const stopWords = new Set([
      'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
      'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did',
      'will', 'would', 'could', 'should', 'this', 'that', 'these', 'those', 'it', 'its'
    ]);

    return text.split(/\W+/)
      .filter(word => word.length > 2 && !stopWords.has(word.toLowerCase()))
      .slice(0, 20);
  }

  /**
   * Extract solution text from case object (handles various formats)
   * Combines description, steps, code examples, and other relevant fields
   */
  extractSolutionText(caseSolution) {
    if (typeof caseSolution === 'string') {
      return caseSolution;
    }

    if (typeof caseSolution === 'object' && caseSolution !== null) {
      const parts = [];

      // Add description
      if (caseSolution.description) {
        parts.push(caseSolution.description);
      }

      // Add answer if available
      if (caseSolution.answer) {
        parts.push(caseSolution.answer);
      }

      // Add question body if available
      if (caseSolution.questionBody) {
        parts.push(caseSolution.questionBody);
      }

      // Add accepted answer body if available
      if (caseSolution.acceptedAnswer?.body) {
        parts.push(caseSolution.acceptedAnswer.body);
      }

      // Add steps if available (array of strings)
      if (Array.isArray(caseSolution.steps)) {
        parts.push(caseSolution.steps.join('. '));
      }

      // Add code examples if available (object with multiple code blocks)
      if (caseSolution.code && typeof caseSolution.code === 'object') {
        Object.entries(caseSolution.code).forEach(([key, value]) => {
          if (typeof value === 'string' && value.trim().length > 0) {
            parts.push(`Code example (${key}): ${value.substring(0, 500)}`);
          }
        });
      } else if (typeof caseSolution.code === 'string') {
        parts.push(`Code: ${caseSolution.code.substring(0, 500)}`);
      }

      // Add security best practices if available
      if (Array.isArray(caseSolution.security_best_practices)) {
        parts.push('Security best practices: ' + caseSolution.security_best_practices.join('. '));
      }

      // Add common mistakes if available
      if (Array.isArray(caseSolution.common_mistakes)) {
        parts.push('Common mistakes to avoid: ' + caseSolution.common_mistakes.join('. '));
      }

      // If we have parts, combine them
      if (parts.length > 0) {
        return parts.join(' ');
      }

      // Fallback to JSON string
      return JSON.stringify(caseSolution);
    }

    return String(caseSolution || '');
  }

  /**
   * Extract the most relevant parts of a case solution for a given query
   * Returns focused content with relevance scores
   */
  async extractRelevantContent(caseSolution, query, options = {}) {
    const {
      maxSentences = 5,
      minRelevanceScore = 0.1,
      includeScores = false
    } = options;

    const solutionText = this.extractSolutionText(caseSolution);
    const sentences = this.splitIntoSentences(solutionText);

    if (sentences.length === 0) {
      return {
        relevantContent: solutionText,
        relevanceScores: [],
        originalLength: 0,
        extractedLength: 0
      };
    }

    // Calculate relevance score for each sentence
    const sentenceRelevance = sentences.map((sentence, index) => {
      const relevance = this.calculateRelevance(query, sentence);
      return {
        sentence,
        relevance,
        index
      };
    });

    // Sort by relevance (highest first)
    sentenceRelevance.sort((a, b) => b.relevance - a.relevance);

    // Take top N sentences that meet minimum relevance threshold
    const topSentences = sentenceRelevance
      .filter(item => item.relevance >= minRelevanceScore)
      .slice(0, maxSentences);

    // Re-order top sentences back to original order for readability
    topSentences.sort((a, b) => a.index - b.index);

    const relevantContent = topSentences.map(item => item.sentence).join(' ');

    // If no sentences met threshold, take at least the top one
    const finalContent = relevantContent || sentenceRelevance[0]?.sentence || solutionText;
    const finalScores = includeScores ? sentenceRelevance : [];

    return {
      relevantContent: finalContent,
      relevanceScores: finalScores,
      topSentences: topSentences.map(item => ({
        sentence: item.sentence,
        score: item.relevance,
        index: item.index
      })),
      originalLength: sentences.length,
      extractedLength: topSentences.length || 1,
      compressionRatio: (topSentences.length || 1) / sentences.length
    };
  }

  /**
   * Enrich prompt context with focused case content
   * Takes multiple cases and extracts query-relevant parts from each
   */
  async enrichPromptContext(query, cases, options = {}) {
    const {
      maxCases = 5,
      maxSentencesPerCase = 3,
      minRelevanceScore = 0.1,
      includeFullSolution = false
    } = options;

    const enrichedCases = await Promise.all(
      cases.slice(0, maxCases).map(async (caseItem) => {
        const extraction = await this.extractRelevantContent(
          caseItem.solution,
          query,
          {
            maxSentences: maxSentencesPerCase,
            minRelevanceScore,
            includeScores: true
          }
        );

        const result = {
          case_id: caseItem.case_id,
          task: caseItem.task,
          relevant_content: extraction.relevantContent,
          relevance_summary: {
            original_sentences: extraction.originalLength,
            extracted_sentences: extraction.extractedLength,
            compression_ratio: extraction.compressionRatio.toFixed(2),
            top_sentences: extraction.topSentences.slice(0, 3) // Top 3 sentences with scores
          },
          tags: caseItem.tags || [],
          q_value: caseItem.q_value || 0
        };

        if (includeFullSolution) {
          result.full_solution = caseItem.solution;
        }

        return result;
      })
    );

    return {
      query,
      enrichedCases,
      totalCases: enrichedCases.length,
      averageCompressionRatio: enrichedCases.length > 0
        ? (enrichedCases.reduce((sum, c) => sum + parseFloat(c.relevance_summary.compression_ratio), 0) / enrichedCases.length).toFixed(2)
        : '0.00'
    };
  }

  /**
   * Format enriched context for prompt inclusion
   * Creates a structured format suitable for LLM context
   */
  formatContextForPrompt(enrichedContext, format = 'text') {
    if (format === 'json') {
      return JSON.stringify(enrichedContext, null, 2);
    }

    // Text format optimized for prompt inclusion
    let formatted = `Relevant Context for Query: "${enrichedContext.query}"\n\n`;

    enrichedContext.enrichedCases.forEach((enrichedCase, index) => {
      formatted += `[Case ${index + 1}] ${enrichedCase.case_id}\n`;
      formatted += `Task: ${enrichedCase.task}\n`;
      formatted += `Relevant Content: ${enrichedCase.relevant_content}\n`;
      if (enrichedCase.tags.length > 0) {
        formatted += `Tags: ${enrichedCase.tags.join(', ')}\n`;
      }
      formatted += `\n`;
    });

    return formatted;
  }
}
