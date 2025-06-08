import axios from 'axios';
import { EmbeddingService, type EmbeddingModelInfo } from './EmbeddingService.js';
import { logger } from '../utils/logger.js';

/**
 * Configuration for Ollama embedding service
 */
export interface OllamaEmbeddingConfig {
  /**
   * Ollama API endpoint
   */
  apiEndpoint?: string;

  /**
   * Optional model name to use
   */
  model?: string;

  /**
   * Optional dimensions override
   */
  dimensions?: number;

  /**
   * Optional version string
   */
  version?: string;
}

/**
 * Ollama API response structure
 */
interface OllamaEmbeddingResponse {
  embedding: number[];
}

/**
 * Service implementation that generates embeddings using Ollama's API
 */
export class OllamaEmbeddingService extends EmbeddingService {
  private model: string;
  private dimensions: number;
  private version: string;
  private apiEndpoint: string;

  /**
   * Create a new Ollama embedding service
   *
   * @param config - Configuration for the service
   */
  constructor(config: OllamaEmbeddingConfig = {}) {
    super();

    this.model = config.model || 'mxbai-embed-large';
    this.dimensions = config.dimensions || 1024;
    this.version = config.version || '1.0.0';
    this.apiEndpoint = config.apiEndpoint || 'http://localhost:11434/api/embeddings';
  }

  /**
   * Generate an embedding for a single text
   *
   * @param text - Text to generate embedding for
   * @returns Promise resolving to embedding vector
   */
  override async generateEmbedding(text: string): Promise<number[]> {
    logger.debug('Generating embedding', {
      text: text.substring(0, 50) + '...',
      model: this.model,
      apiEndpoint: this.apiEndpoint,
    });

    try {
      const response = await axios.post<OllamaEmbeddingResponse>(
        this.apiEndpoint,
        {
          prompt: text,
          model: this.model,
        },
        {
          headers: {
            'Content-Type': 'application/json',
          },
          timeout: 10000, // 10 second timeout
        }
      );

      logger.debug('Received response from Ollama API');

      if (!response.data || !response.data.embedding) {
        logger.error('Invalid response from Ollama API', { response: response.data });
        throw new Error('Invalid response from Ollama API - missing embedding data');
      }

      const embedding = response.data.embedding;

      if (!embedding || !Array.isArray(embedding) || embedding.length === 0) {
        logger.error('Invalid embedding returned', { embedding });
        throw new Error('Invalid embedding returned from Ollama API');
      }

      logger.debug('Generated embedding', {
        length: embedding.length,
        sample: embedding.slice(0, 5),
        isArray: Array.isArray(embedding),
      });

      // Normalize the embedding vector
      this._normalizeVector(embedding);
      logger.debug('Normalized embedding', {
        length: embedding.length,
        sample: embedding.slice(0, 5),
      });

      return embedding;
    } catch (error: unknown) {
      // Handle axios errors specifically
      const axiosError = error as {
        isAxiosError?: boolean;
        response?: {
          status?: number;
          data?: unknown;
        };
        message?: string;
      };
      if (axiosError.isAxiosError) {
        const statusCode = axiosError.response?.status;
        const responseData = axiosError.response?.data;

        logger.error('Ollama API error', {
          status: statusCode,
          data: responseData,
          message: axiosError.message,
        });
        
        const errorDetails = responseData
          ? `: ${JSON.stringify(responseData).substring(0, 200)}`
          : '';

        throw new Error(`Ollama API error (${statusCode || 'unknown'})${errorDetails}`);
      }

      // Handle other errors
      const errorMessage = this._getErrorMessage(error);
      logger.error('Failed to generate embedding', { error: errorMessage });
      throw new Error(`Error generating embedding: ${errorMessage}`);
    }
  }

  /**
   * Generate embeddings for multiple texts
   *
   * @param texts - Array of texts to generate embeddings for
   * @returns Promise resolving to array of embedding vectors
   */
  override async generateEmbeddings(texts: string[]): Promise<number[][]> {
    const embeddings: number[][] = [];
    for(const text of texts) {
        embeddings.push(await this.generateEmbedding(text));
    }
    return embeddings;
  }

  /**
   * Get information about the embedding model
   *
   * @returns Model information
   */
  override getModelInfo(): EmbeddingModelInfo {
    return {
      name: this.model,
      dimensions: this.dimensions,
      version: this.version,
    };
  }

  /**
   * Extract error message from error object
   *
   * @private
   * @param error - Error object
   * @returns Error message string
   */
  private _getErrorMessage(error: unknown): string {
    if (error instanceof Error) {
      return error.message;
    }
    return String(error);
  }

  /**
   * Normalize a vector to unit length (L2 norm)
   *
   * @private
   * @param vector - Vector to normalize in-place
   */
  private _normalizeVector(vector: number[]): void {
    // Calculate magnitude (Euclidean norm / L2 norm)
    let magnitude = 0;
    for (let i = 0; i < vector.length; i++) {
      magnitude += vector[i] * vector[i];
    }
    magnitude = Math.sqrt(magnitude);

    // Avoid division by zero
    if (magnitude > 0) {
      // Normalize each component
      for (let i = 0; i < vector.length; i++) {
        vector[i] /= magnitude;
      }
    } else {
      // If magnitude is 0, set first element to 1 for a valid unit vector
      vector[0] = 1;
    }
  }
}
