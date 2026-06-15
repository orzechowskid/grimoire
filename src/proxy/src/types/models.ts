// SPDX-License-Identifier: MIT

/**
 * TypeScript type definitions for API contracts.
 *
 * Mirrors the Python memory library's Pydantic models and defines
 * additional types needed for the proxy's own request/response flow,
 * inference server communication, and configuration.
 */

// ── Memory Library API Types ──────────────────────────────────────────

/** Request body for the memory injection endpoint. */
export interface InjectRequest {
  text: string;
  top_k?: number;
  time_weighted?: boolean;
  min_similarity?: number;
}

/** Response from the memory injection endpoint. */
export interface InjectResult {
  session_id: string;
  brief: string;
  tags: string[];
  importance: string;
  score: number;
  key_facts?: string[];
}

/** Individual intuition signal from the experience layer. */
export interface IntuitionSignal {
  type: string;
  tag: string;
  message: string;
}

/** Response from the memory injection endpoint (wrapper). */
export interface InjectResponse {
  results: InjectResult[];
  total_candidates: number;
  warning?: string;
  intuition_signals?: IntuitionSignal[];
}

/** Request body for the memory observation endpoint. */
export interface ObserveRequest {
  text: string;
  session_id?: string;
  source?: "user" | "agent";  // NEW: distinguish user vs agent messages
}

/** Response from the memory observation endpoint. */
export interface ObserveResponse {
  session_id: string;
  brief: string;
  tags: string[];
  importance: string;
  score: number;
  discarded: boolean;
}

/** Request body for the memory search endpoint. */
export interface SearchRequest {
  query: string;
  top_k?: number;
  min_similarity?: number;
  time_weighted?: boolean;
}

/** Individual result from a memory search. */
export interface SearchResult {
  session_id: string;
  brief: string;
  tags: string[];
  importance: string;
  score: number;
  created_at: number;
  similarity: number;
}

/** Response from the memory search endpoint (wrapper). */
export interface SearchResponse {
  results: SearchResult[];
  total_candidates: number;
}

/** Health check response from the memory library. */
export interface MemoryHealthResponse {
  status: string;
  sessions_in_ram: number;
  anchors_in_ram: number;
  experience_clusters: number;
  embedder: string;
  ner: string;
  reranker: string;
}

// ── Proxy Configuration Types ─────────────────────────────────────────

/** Proxy server configuration loaded from environment variables. */
export interface ProxyConfig {
  memoryLibUrl: string;
  inferenceServerUrl: string;
  inferenceServerKey?: string;
  port: number;
  corsOrigin: string;
  memoryInjection: {
    enabled: boolean;
    topK: number;
    timeWeighted: boolean;
  };
  observation: {
    enabled: boolean;
    batchSize: number;
  };
}

// ── Inference Server Types ────────────────────────────────────────────

/** Generic inference request (OpenAI-compatible chat completion). */
export interface InferenceRequest {
  model: string;
  messages: Array<{ role: string; content: string }>;
  stream?: boolean;
  temperature?: number;
  max_tokens?: number;
  top_p?: number;
  frequency_penalty?: number;
  presence_penalty?: number;
  [key: string]: unknown;
}

/** Individual chat completion choice. */
export interface ChatChoice {
  index: number;
  message: { role: string; content: string };
  finish_reason: string | null;
}

/** Usage statistics from an inference response. */
export interface Usage {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
}

/** Generic inference response (OpenAI-compatible chat completion). */
export interface InferenceResponse {
  id: string;
  object: string;
  created: number;
  model: string;
  choices: ChatChoice[];
  usage?: Usage;
}

/** Chunk from a streaming inference response (SSE). */
export interface StreamingChunk {
  id: string;
  object: string;
  created: number;
  model: string;
  choices: Array<{
    index: number;
    delta: { role?: string; content?: string };
    finish_reason: string | null;
  }>;
}

// ── Proxy Internal Types ──────────────────────────────────────────────

/** Health check response from the proxy server. */
export interface HealthResponse {
  status: string;
  timestamp?: string;
  memory_library?: string;
}

export interface ErrorResponse {
  error: string;
  code?: number;
  details?: unknown;
}

// ── Admin API Types ─────────────────────────────────────────────────────

/** Enhanced health check response from the proxy admin endpoint. */
export interface AdminHealthResponse {
  status: string;
  timestamp: string;
  version: string;
  memory_library: {
    status: 'connected' | 'unavailable' | 'degraded';
    details?: MemoryHealthResponse;
    circuitState: string;
  };
  memory_injection: { enabled: boolean };
  observation: { enabled: boolean };
}

/** Memory statistics response from the proxy admin endpoint. */
export interface AdminStatsResponse {
  circuitState: string;
  lastError: string | null;
  lastErrorTime: number | null;
  memoryLibUrl: string;
}

/** Graceful shutdown response. */
export interface AdminShutdownResponse {
  status: string;
}

/** Circuit breaker reset response. */
export interface AdminCircuitResetResponse {
  status: string;
  circuitState: string;
}

