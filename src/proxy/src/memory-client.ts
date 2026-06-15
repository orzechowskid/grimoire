// SPDX-License-Identifier: MIT

/**
 * HTTP client for the Python memory library.
 *
 * This module provides a typed HTTP client that communicates with
 * the memory library's REST API for:
 * - Memory injection (getting context for user messages)
 * - Response observation (feeding agent output for memory processing)
 * - Semantic search (querying stored memory)
 * - Health checks and graceful shutdown
 *
 * Features:
 * - Retry with exponential backoff
 * - Circuit breaker pattern
 * - Health-aware routing
 * - Structured error types
 */

import { HttpError, httpGet, httpPost } from './http';
import {
  InjectRequest,
  InjectResponse,
  ObserveRequest,
  ObserveResponse,
  SearchRequest,
  SearchResponse,
  MemoryHealthResponse,
} from './types/models';

// ── Error Classes ──────────────────────────────────────────────────────

/**
 * Base error class for all memory client errors.
 */
export class MemoryClientError extends Error {
  constructor(
    message: string,
    public readonly code: string,
    public readonly cause?: Error,
  ) {
    super(message);
    this.name = 'MemoryClientError';
    if (cause) {
      this.cause = cause;
    }
  }
}

/**
 * Error thrown when the circuit breaker is open.
 */
export class CircuitBreakerError extends MemoryClientError {
  constructor(message: string, public readonly circuitState: string) {
    super(message, 'CIRCUIT_BREAKER_OPEN');
    this.name = 'CircuitBreakerError';
  }
}

// ── Retry Configuration ────────────────────────────────────────────────

export interface RetryOptions {
  maxRetries: number;
  initialDelayMs: number;
  maxDelayMs: number;
  backoffMultiplier: number;
}

/**
 * Default retry options.
 */
const DEFAULT_RETRY_OPTIONS: RetryOptions = {
  maxRetries: 3,
  initialDelayMs: 100,
  maxDelayMs: 2000,
  backoffMultiplier: 2,
};

/**
 * Network error codes that are safe to retry.
 */
const RETRYABLE_NETWORK_CODES = new Set([
  'ECONNREFUSED',
  'ECONNRESET',
  'EPIPE',
  'ECONNABORTED',
]);

/**
 * Delay in milliseconds between retries (exponential backoff).
 */
export function calculateDelay(
  attempt: number,
  options: RetryOptions,
): number {
  const delay = options.initialDelayMs * Math.pow(options.backoffMultiplier, attempt);
  return Math.min(delay, options.maxDelayMs);
}

/**
 * Check whether an error is retryable.
 */
export function isRetryableError(error: unknown): boolean {
  if (error instanceof HttpError) {
    // HTTP 503 is retryable
    if (error.status === 503) {
      return true;
    }
    // Network-level errors
    if (error.code && RETRYABLE_NETWORK_CODES.has(error.code)) {
      return true;
    }
    // Timeout errors
    if (error.code === 'ABORTED') {
      return true;
    }
    return false;
  }
  // Generic network errors with error codes
  if (error && typeof error === 'object' && 'code' in error) {
    const err = error as { code?: string };
    if (err.code && RETRYABLE_NETWORK_CODES.has(err.code)) {
      return true;
    }
  }
  return false;
}

/**
 * Execute an async function with retry logic and exponential backoff.
 *
 * @param fn - Async function to execute.
 * @param options - Retry configuration.
 * @returns Result of the async function.
 * @throws MemoryClientError on failure after all retries exhausted.
 */
export async function withRetry<T>(
  fn: () => Promise<T>,
  options: RetryOptions = DEFAULT_RETRY_OPTIONS,
): Promise<T> {
  let lastError: Error | undefined;

  for (let attempt = 0; attempt <= options.maxRetries; attempt++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error instanceof Error ? error : new Error(String(error));

      // Check if this error is retryable
      if (!isRetryableError(error)) {
        throw new MemoryClientError(
          `Non-retryable error: ${lastError.message}`,
          'NON_RETRYABLE_ERROR',
          lastError,
        );
      }

      // If we've exhausted retries, throw
      if (attempt >= options.maxRetries) {
        break;
      }

      const delay = calculateDelay(attempt, options);
      console.warn(
        `[MemoryClient] Retry attempt ${attempt + 1}/${options.maxRetries}: ` +
          `${lastError.message}. Retrying in ${delay}ms.`,
      );

      await new Promise((resolve) => setTimeout(resolve, delay));
    }
  }

  throw new MemoryClientError(
    `Memory client request failed after ${options.maxRetries} retries: ${lastError?.message}`,
    'MAX_RETRIES_EXCEEDED',
    lastError,
  );
}

// ── Circuit Breaker ────────────────────────────────────────────────────

export type CircuitBreakerState = 'CLOSED' | 'OPEN' | 'HALF_OPEN';

export interface CircuitBreakerOptions {
  failureThreshold: number;
  resetTimeoutMs: number;
  successThreshold: number;
}

/**
 * Default circuit breaker options.
 */
const DEFAULT_CIRCUIT_BREAKER_OPTIONS: CircuitBreakerOptions = {
  failureThreshold: 5,
  resetTimeoutMs: 30000,
  successThreshold: 2,
};

/**
 * Circuit breaker for protecting against cascading failures.
 *
 * States:
 * - CLOSED: Normal operation, counting failures. Trips to OPEN after
 *   failureThreshold consecutive failures.
 * - OPEN: All requests rejected immediately. After resetTimeoutMs,
 *   transitions to HALF_OPEN.
 * - HALF_OPEN: Allow requests through. After successThreshold consecutive
 *   successes, closes (returns to CLOSED). On any failure, trips back to OPEN.
 */
export class CircuitBreaker {
  private state: CircuitBreakerState = 'CLOSED';
  private failureCount = 0;
  private successCount = 0;
  private lastFailureTime: number | null = null;
  private lastError: string | null = null;
  private lastErrorTime: number | null = null;

  constructor(private options: CircuitBreakerOptions = DEFAULT_CIRCUIT_BREAKER_OPTIONS) {}

  /**
   * Get the current circuit state for observability.
   */
  getState(): CircuitBreakerState {
    // Check if we should transition from OPEN to HALF_OPEN
    if (this.state === 'OPEN' && this.lastFailureTime !== null) {
      const elapsed = Date.now() - this.lastFailureTime;
      if (elapsed >= this.options.resetTimeoutMs) {
        this.transitionTo('HALF_OPEN');
      }
    }
    return this.state;
  }

  /**
   * Execute a function through the circuit breaker.
   *
   * @param fn - Async function to execute.
   * @returns Result of the async function.
   * @throws CircuitBreakerError if the circuit is open.
   */
  async execute<T>(fn: () => Promise<T>): Promise<T> {
    // Check and potentially transition state
    const currentState = this.getState();

    if (currentState === 'OPEN') {
      const remaining = this.lastFailureTime
        ? Math.max(0, this.options.resetTimeoutMs - (Date.now() - this.lastFailureTime))
        : this.options.resetTimeoutMs;
      throw new CircuitBreakerError(
        `Circuit breaker is OPEN. ${Math.ceil(remaining / 1000)}s until half-open.`,
        'OPEN',
      );
    }

    try {
      const result = await fn();

      // Record success
      if (currentState === 'HALF_OPEN') {
        this.successCount++;
        if (this.successCount >= this.options.successThreshold) {
          this.transitionTo('CLOSED');
        }
      } else {
        // CLOSED state: reset failure count on success
        this.failureCount = 0;
      }

      return result;
    } catch (error) {
      // Record failure
      const errorMessage = error instanceof Error ? error.message : String(error);
      this.lastError = errorMessage;
      this.lastErrorTime = Date.now();
      this.lastFailureTime = Date.now();

      if (currentState === 'HALF_OPEN') {
        this.transitionTo('OPEN');
      } else {
        // CLOSED state: count failures
        this.failureCount++;
        if (this.failureCount >= this.options.failureThreshold) {
          this.transitionTo('OPEN');
        }
      }

      throw new MemoryClientError(
        `Circuit breaker detected failure: ${errorMessage}`,
        'CIRCUIT_BREAKER_FAILURE',
        error instanceof Error ? error : undefined,
      );
    }
  }

  /**
   * Get the last error message, if any.
   */
  getLastError(): string | null {
    return this.lastError;
  }

  /**
   * Get the timestamp of the last error.
   */
  getLastErrorTime(): number | null {
    return this.lastErrorTime;
  }

  /**
   * Manually reset the circuit breaker to CLOSED state.
   */
  reset(): void {
    const newState = this.transitionTo('CLOSED');
    if (newState === 'CLOSED') {
      console.log('[MemoryClient] Circuit breaker manually reset to CLOSED.');
    }
  }

  /**
   * Internal state transition (for internal use and testing).
   */
  private transitionTo(newState: CircuitBreakerState): CircuitBreakerState {
    const oldState = this.state;
    if (oldState !== newState) {
      console.log(
        `[MemoryClient] Circuit breaker: ${oldState} → ${newState}`,
      );
      this.state = newState;

      // Reset counters on state transition
      if (newState === 'CLOSED') {
        this.failureCount = 0;
        this.successCount = 0;
      } else if (newState === 'HALF_OPEN') {
        this.successCount = 0;
      }
    }
    return newState;
  }
}

// ── Client Configuration ───────────────────────────────────────────────

export interface MemoryClientOptions {
  timeout?: number;
  retryOptions?: RetryOptions;
  circuitBreakerOptions?: CircuitBreakerOptions;
  enableCircuitBreaker?: boolean;
}

// ── Memory Client ──────────────────────────────────────────────────────

/**
 * HTTP client for the Python memory library.
 *
 * Provides a typed interface to the memory library's REST API
 * with retry logic, circuit breaking, and health monitoring.
 */
export class MemoryClient {
  private baseUrl: string;
  private circuitBreaker: CircuitBreaker;
  private enableCircuitBreaker: boolean;
  private retryOptions: RetryOptions;
  private timeout: number;

  constructor(baseUrl: string, options: MemoryClientOptions = {}) {
    this.baseUrl = baseUrl;
    this.timeout = options.timeout ?? 10000;
    this.retryOptions = options.retryOptions ?? DEFAULT_RETRY_OPTIONS;
    this.enableCircuitBreaker = options.enableCircuitBreaker ?? true;
    this.circuitBreaker = new CircuitBreaker(
      options.circuitBreakerOptions ?? DEFAULT_CIRCUIT_BREAKER_OPTIONS,
    );
  }

  /**
   * Get memory context for a user message.
   *
   * Uses retry + circuit breaker.
   *
   * @param request - Inject request with text and optional parameters.
   * @returns Inject response with relevant memory sessions.
   */
  async inject(request: InjectRequest): Promise<InjectResponse> {
    return this.withProtected(
      async () => {
        const result = await httpPost<InjectResponse>(`${this.baseUrl}/inject`, request, { 'Content-Type': 'application/json' }, this.timeout);
        return result.data;
      },
      'inject',
    );
  }

  /**
   * Feed agent response into the memory pipeline.
   *
   * Uses retry only (circuit breaker disabled — observation is best-effort).
   *
   * @param request - Observe request with text and optional session_id.
   * @returns Observe response with processing results.
   */
  async observe(request: ObserveRequest): Promise<ObserveResponse> {
    return this.withRetryOnly(
      async () => {
        const result = await httpPost<ObserveResponse>(`${this.baseUrl}/observe`, request, { 'Content-Type': 'application/json' }, this.timeout);
        return result.data;
      },
      'observe',
    );
  }

  /**
   * Perform semantic search over stored memory.
   *
   * Uses retry + circuit breaker.
   *
   * @param request - Search request with query and optional parameters.
   * @returns Search response with matching sessions.
   */
  async search(request: SearchRequest): Promise<SearchResponse> {
    return this.withProtected(
      async () => {
        const result = await httpPost<SearchResponse>(`${this.baseUrl}/search`, request, { 'Content-Type': 'application/json' }, this.timeout);
        return result.data;
      },
      'search',
    );
  }

  /**
   * Check the health status of the memory library.
   *
   * Uses retry + circuit breaker.
   *
   * @returns Health response with system status.
   */
  async health(): Promise<MemoryHealthResponse> {
    return this.withProtected(
      async () => {
        const result = await httpGet<MemoryHealthResponse>(`${this.baseUrl}/health`, { 'Content-Type': 'application/json' }, this.timeout);
        return result.data;
      },
      'health',
    );
  }

  /**
   * Trigger graceful shutdown of the memory library.
   *
   * No retry — single attempt only.
   */
  async shutdown(): Promise<void> {
    try {
      await httpPost('/shutdown', {}, { 'Content-Type': 'application/json' }, this.timeout);
    } catch (error) {
      throw new MemoryClientError(
        `Shutdown failed: ${error instanceof Error ? error.message : String(error)}`,
        'SHUTDOWN_FAILED',
        error instanceof Error ? error : undefined,
      );
    }
  }

  /**
   * Check if the memory library is healthy.
   *
   * @returns true if the health endpoint reports status "ok".
   */
  async isHealthy(): Promise<boolean> {
    try {
      const health = await this.health();
      return health.status === 'ok';
    } catch {
      return false;
    }
  }

  /**
   * Get the current status of the client for observability.
   *
   * @returns Status object with circuit breaker state and error info.
   */
  status(): {
    circuitState: string;
    baseUrl: string;
    lastError: string | null;
    lastErrorTime: number | null;
  } {
    return {
      circuitState: this.circuitBreaker.getState(),
      baseUrl: this.baseUrl,
      lastError: this.circuitBreaker.getLastError(),
      lastErrorTime: this.circuitBreaker.getLastErrorTime(),
    };
  }

  /**
   * Manually reset the circuit breaker.
   */
  resetCircuitBreaker(): void {
    this.circuitBreaker.reset();
  }

  /**
   * Execute with both retry and circuit breaker protection.
   */
  private async withProtected<T>(fn: () => Promise<T>, operationName: string): Promise<T> {
    if (this.enableCircuitBreaker) {
      return this.circuitBreaker.execute(fn);
    } else {
      return withRetry(fn, this.retryOptions);
    }
  }

  /**
   * Execute with retry only (no circuit breaker).
   */
  private async withRetryOnly<T>(fn: () => Promise<T>, _operationName: string): Promise<T> {
    return withRetry(fn, this.retryOptions);
  }
}

/**
 * Create a new memory library client.
 *
 * @param baseUrl - Base URL of the memory library.
 * @param options - Optional configuration.
 * @returns A configured MemoryClient instance.
 */
export function createMemoryClient(
  baseUrl: string,
  options?: MemoryClientOptions,
): MemoryClient {
  return new MemoryClient(baseUrl, options);
}
