/**
 * LLM Query Enhancer
 *
 * Enhances user queries using LLM APIs (OpenAI or Anthropic) based on
 * available solution titles and technical context to improve retrieval results.
 */

export class LLMQueryEnhancer {
  constructor(config) {
    this.provider = config.provider;
    this.apiKey = config.apiKey;
    this.model = config.model;
    this.baseURL = config.baseURL;
  }

  /**
   * Enhance query using LLM based on user query, solution titles, and tech context
   */
  async enhanceQuery(userQuery, solutionTitles, techContext = {}) {
    if (!this.apiKey) {
      console.warn('LLM API key not configured, skipping query enhancement');
      return userQuery;
    }

    try {
      const solutionTitlesText = solutionTitles.length > 0
        ? solutionTitles.slice(0, 50).map((title, idx) => `${idx + 1}. ${title}`).join('\n')
        : 'No existing solutions found.';

      const techContextText = Object.keys(techContext).length > 0
        ? `\n\nTechnical Context:\n${JSON.stringify(techContext, null, 2)}`
        : '';

      const prompt = `You are a query enhancement assistant for a technical case retrieval system.

User Query: "${userQuery}"

Available Solution Titles:
${solutionTitlesText}${techContextText}

Your task: Based on the user query, available solution titles, and technical context, generate an enhanced, more specific query that will help retrieve the most relevant solutions. The enhanced query should:
1. Preserve the user's original intent
2. Incorporate relevant technical terms from the context (stack, framework, project type)
3. Consider semantic relationships with the available solution titles
4. Be concise but comprehensive

Return ONLY the enhanced query text, without any explanation or additional text.`;

      if (this.provider === 'openai') {
        return await this.enhanceWithOpenAI(prompt);
      } else if (this.provider === 'anthropic') {
        return await this.enhanceWithAnthropic(prompt);
      } else {
        console.warn(`Unknown LLM provider: ${this.provider}, skipping enhancement`);
        return userQuery;
      }
    } catch (error) {
      console.error('Error enhancing query with LLM:', error.message);
      return userQuery; // Fallback to original query
    }
  }

  async enhanceWithOpenAI(prompt) {
    const url = this.baseURL || 'https://api.openai.com/v1/chat/completions';
    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${this.apiKey}`,
      },
      body: JSON.stringify({
        model: this.model,
        messages: [
          {
            role: 'system',
            content: 'You are a helpful assistant that enhances search queries for better retrieval results.',
          },
          {
            role: 'user',
            content: prompt,
          },
        ],
        temperature: 0.3,
        max_tokens: 200,
      }),
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`OpenAI API error: ${response.status} - ${error}`);
    }

    const data = await response.json();
    return data.choices[0]?.message?.content?.trim() || '';
  }

  async enhanceWithAnthropic(prompt) {
    const url = this.baseURL || 'https://api.anthropic.com/v1/messages';
    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-api-key': this.apiKey,
        'anthropic-version': '2023-06-01',
      },
      body: JSON.stringify({
        model: this.model,
        max_tokens: 200,
        messages: [
          {
            role: 'user',
            content: prompt,
          },
        ],
      }),
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`Anthropic API error: ${response.status} - ${error}`);
    }

    const data = await response.json();
    return data.content[0]?.text?.trim() || '';
  }
}
