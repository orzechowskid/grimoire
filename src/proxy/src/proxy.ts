// SPDX-License-Identifier: MIT

/**
 * Core proxy logic.
 *
 * This module contains the main proxy engine that:
 * - Receives requests from the agent harness
 * - Forwards enhanced requests to the inference server
 * - Handles both streaming (SSE) and non-streaming responses
 * - Observes non-streaming responses to build memory
 */

import { HttpError, httpGet, httpPost, httpPostStream } from './http';
import express, { Request, Response, NextFunction } from 'express';
import {
  ErrorResponse,
  ProxyConfig,
  InjectResult,
  InjectResponse,
  StreamingChunk,
} from './types/models';
import { MemoryClient } from './memory-client';
import { randomUUID } from 'crypto';

/**
 * Create and configure the core proxy engine.
 *
 * @param config - Proxy configuration.
 * @param memoryClient - Optional memory client for injecting context.
 * @returns Express application configured as a proxy.
 */
export function createProxyApp(config: ProxyConfig, memoryClient: MemoryClient | null = null): express.Express {
  const app = express();

  // ── Non-streaming: POST /v1/chat/completions ──────────────────────────
  app.post('/v1/chat/completions', async (req: Request, res: Response) => {
    const startTime = Date.now();
    const model = (req.body?.model as string) || 'unknown';
    const isStreaming = req.body?.stream === true;

    console.log(
      `[proxy] POST /v1/chat/completions  model=${model}  stream=${isStreaming}`
    );

    // Validate that the body exists and is an object
    if (!req.body || typeof req.body !== 'object') {
      const errResp: ErrorResponse = {
        error: 'Request body must be a JSON object',
        code: 400,
      };
      console.log(
        `[proxy] 400 Bad Request — invalid body (${(Date.now() - startTime)}ms)`
      );
      return res.status(400).json(errResp);
    }

    if (isStreaming) {
      return handleStreaming(req, res, config, startTime, memoryClient);
    }

    // ── Non-streaming flow ──────────────────────────────────────────────
    // Inject memory context if client is available
    await injectMemoryContext(req.body, memoryClient, config);

    try {
      const inferenceUrl = `${config.inferenceServerUrl}/chat/completions`;

      const headers: Record<string, string> = {
        'Content-Type': 'application/json',
      };
      if (config.inferenceServerKey) {
        headers['Authorization'] = `Bearer ${config.inferenceServerKey}`;
      }

      const result = await httpPost(inferenceUrl, req.body, headers, 300_000);

      const latency = Date.now() - startTime;
      console.log(
        `[proxy] → ${result.status} (${latency}ms)  model=${model}  id=${(result.data as Record<string, unknown>)?.id}`
      );

      // Observe the response for memory building (fire-and-forget, non-blocking)
      observeAgentResponse(result.data, memoryClient, config).catch(console.warn);

      return res.status(result.status).json(result.data);
    } catch (err) {
      return handleInferenceError(err, res, config, startTime, model);
    }
  });

  // ── GET /v1/models ────────────────────────────────────────────────────
  app.get('/v1/models', async (req: Request, res: Response) => {
    const startTime = Date.now();

    console.log(`[proxy] GET /v1/models`);

    try {
      const inferenceUrl = `${config.inferenceServerUrl}/models`;

      const headers: Record<string, string> = {};
      if (config.inferenceServerKey) {
        headers['Authorization'] = `Bearer ${config.inferenceServerKey}`;
      }

      const result = await httpGet(inferenceUrl, headers, 10_000);

      const latency = Date.now() - startTime;
      console.log(`[proxy] → ${result.status} (${latency}ms)`);

      return res.status(result.status).json(result.data);
    } catch (err) {
      return handleInferenceError(err, res, config, startTime, 'models');
    }
  });

  // ── 404 handler ───────────────────────────────────────────────────────
  app.use((_req: Request, res: Response) => {
    res.status(404).json({
      error: 'Not found',
      code: 404,
      details: 'The requested endpoint does not exist on this proxy.',
    });
  });

  // ── Global error handler ──────────────────────────────────────────────
  app.use((err: Error, _req: Request, res: Response, _next: NextFunction) => {
    console.error(`[proxy] Unhandled error: ${err.message}`);
    const errResp: ErrorResponse = {
      error: 'Internal server error',
      code: 500,
    };
    res.status(500).json(errResp);
  });

  return app;
}

// ── Memory injection ──────────────────────────────────────────────────────

/**
 * Inject memory context into the request body.
 *
 * Extracts the last user message, queries the memory library,
 * and prepends the retrieved context to the system message.
 *
 * @param body - The request body containing messages.
 * @param memoryClient - The memory client to query.
 * @param config - Proxy configuration (for topK, timeWeighted settings).
 */
async function injectMemoryContext(
  body: any,
  memoryClient: MemoryClient | null,
  config: ProxyConfig
): Promise<void> {
  if (!memoryClient || config.memoryInjection?.enabled === false) {
    console.log('[proxy] Memory injection skipped — disabled or no client available');
    return;
  }

  // Find the last user message
  const messages = body?.messages as Array<{ role: string; content: string }> | undefined;
  if (!Array.isArray(messages) || messages.length === 0) {
    console.log('[proxy] Memory injection skipped — no messages found');
    return;
  }

  let userMessage: string | undefined;
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i].role === 'user') {
      userMessage = messages[i].content;
      break;
    }
  }

  if (!userMessage) {
    console.log('[proxy] Memory injection skipped — no user message found');
    return;
  }

  // Observe user message for memory building (fire-and-forget, non-blocking)
  observeUserMessage(userMessage, memoryClient, config);

  // Query the memory library for context
  const topK = config.memoryInjection?.topK ?? 10;
  try {
    const result: InjectResponse = await memoryClient.inject({
      text: userMessage,
      top_k: topK,
      time_weighted: config.memoryInjection?.timeWeighted ?? false,
    });

    if (!result.results || result.results.length === 0) {
      console.log('[proxy] Memory injection — no context retrieved');
      return;
    }

    // Format the memory context block
    const memoryBlock = formatMemoryContext(result);

    // Find existing system message or create one
    let systemMessageIndex = -1;
    for (let i = 0; i < messages.length; i++) {
      if (messages[i].role === 'system') {
        systemMessageIndex = i;
        break;
      }
    }

    if (systemMessageIndex >= 0) {
      // Prepend to existing system message
      const existingContent = messages[systemMessageIndex].content || '';
      messages[systemMessageIndex].content = memoryBlock + '\n\n' + existingContent;
    } else {
      // Create a new system message at the beginning
      messages.unshift({
        role: 'system',
        content: memoryBlock,
      });
    }

    console.log(
      `[proxy] Memory injection — injected ${result.results.length} context(s) into request`
    );
  } catch (err) {
    // Fail-open: log error and continue without memory context
    const errMessage = err instanceof Error ? err.message : String(err);
    console.warn(`[proxy] Memory injection failed: ${errMessage}`);
  }
}

/**
 * Format memory injection results into a grouped narrative context block.
 *
 * Groups results by importance level, renders each as a bullet point with
 * brief and tags, and includes a summary of related tags.
 *
 * @param result - Inject response from the memory library.
 * @returns Formatted memory context string.
 */
function formatMemoryContext(result: InjectResponse): string {
  if (!result.results || result.results.length === 0) {
    return '### MEMORY CONTEXT\nNo relevant memories found.';
  }

  const lines: string[] = [];

  // Build intuition alerts block if signals present
  if (result.intuition_signals && result.intuition_signals.length > 0) {
    const emojiMap: Record<string, string> = {
      TENSION: '⚠ TENSION',
      DO_THIS: '💡 RECOMMENDATION',
      AVOID_THIS: '🚫 AVOID',
      ATTRACT: '❤ ATTRACT',
      REPEL: '💔 REPEL',
      AMBIVALENT: '⚖ AMBIVALENT',
    };
    lines.push('### COGNITIVE INTUITION ALERTS');
    for (const sig of result.intuition_signals) {
      const prefix = emojiMap[sig.type] || sig.type;
      lines.push(`${prefix}: ${sig.message}`);
    }
    lines.push(''); // blank line separator
  }

  // Importance display order
  const importanceOrder: string[] = ['CRITICAL', 'PRINCIPLE', 'IMPORTANT', 'BACKGROUND'];

  // Group results by importance (uppercased)
  const groups: Map<string, InjectResult[]> = new Map();
  for (const r of result.results) {
    const key = (r.importance || 'BACKGROUND').toUpperCase();
    if (!groups.has(key)) {
      groups.set(key, []);
    }
    groups.get(key)!.push(r);
  }

  lines.push('### MEMORY CONTEXT');
  lines.push(`Retrieved ${result.results.length} relevant memor${result.results.length > 1 ? 'ies' : 'y'}:`);

  // Render groups in priority order
  for (const importance of importanceOrder) {
    const group = groups.get(importance);
    if (!group || group.length === 0) {
      continue;
    }

    lines.push('');
    lines.push(`**${importance} (${group.length})**`);

    for (const r of group) {
      let bullet = `• ${r.brief}`;
      if (r.tags && r.tags.length > 0) {
        bullet += ` [${r.tags.join(', ')}]`;
      }
      lines.push(bullet);
      // Render key facts as sub-bullets
      if (r.key_facts && r.key_facts.length > 0) {
        for (const fact of r.key_facts) {
          lines.push(`  ↳ Fact: ${fact}`);
        }
      }
    }
  }

  // Collect unique tags for summary
  const allTags = new Set<string>();
  for (const r of result.results) {
    if (r.tags) {
      for (const tag of r.tags) {
        allTags.add(tag);
      }
    }
  }

  if (allTags.size > 0) {
    lines.push('');
    lines.push(`Related tags: ${Array.from(allTags).join(', ')}`);
  }

  return lines.join('\n');
}

// ── User message observation ─────────────────────────────────────────────

/**
 * Observe a user message for memory building (fire-and-forget).
 *
 * @param text - The user's message text.
 * @param memoryClient - The memory client, or null if unavailable.
 * @param config - Proxy configuration.
 */
async function observeUserMessage(
  text: string,
  memoryClient: MemoryClient | null,
  config: ProxyConfig,
): Promise<void> {
  if (config.observation?.enabled === false) {
    return;
  }
  if (!memoryClient) {
    return;
  }
  if (!text || text.trim().length === 0) {
    return;
  }

  try {
    await memoryClient.observe({
      text: text,
      session_id: `usr_${randomUUID().replace(/-/g, '').slice(0, 12)}`,
      source: "user",
    });
  } catch (err) {
    const errMessage = err instanceof Error ? err.message : String(err);
    console.warn(`[proxy] User message observation failed: ${errMessage}`);
  }
}

// ── Response observation ────────────────────────────────────────────────

/**
 * Observe an inference response for memory building.
 *
 * Extracts the agent's response text from the inference response
 * and feeds it to the memory library's observer pipeline.
 *
 * Best-effort: if the memory client throws (circuit open, unavailable,
 * timeout), the error is logged and swallowed so observation never
 * blocks or affects the response to the client.
 *
 * @param responseBody - The inference server response body.
 * @param memoryClient - The memory client, or null if unavailable.
 * @param config - Proxy configuration (for observation settings).
 */
async function observeAgentResponse(
  responseBody: any,
  memoryClient: MemoryClient | null,
  config: ProxyConfig
): Promise<void> {
  // Skip if observation is disabled
  if (config.observation?.enabled === false) {
    console.log('[proxy] Response observation skipped — observation disabled in config');
    return;
  }

  if (!memoryClient) {
    console.log('[proxy] Response observation skipped — no memory client available');
    return;
  }

  // Extract the agent's response text
  const content = responseBody?.choices?.[0]?.message?.content;

  if (!content || typeof content !== 'string' || content.trim().length === 0) {
    console.log('[proxy] Response observation skipped — no response content found');
    return;
  }

  // Fire-and-forget observation
  try {
    await memoryClient.observe({ text: content });
    console.log('[proxy] Response observation succeeded');
  } catch (err) {
    const errMessage = err instanceof Error ? err.message : String(err);
    console.warn(`[proxy] Response observation failed: ${errMessage}`);
  }
}

// ── Streaming helper ──────────────────────────────────────────────────────

/**
 * Collects streaming response chunks and accumulates full response text.
 */
class StreamCollector {
  private contentBuffer: string[] = [];

  /**
   * Collect a content delta from a streaming chunk.
   * Skips undefined or empty strings.
   */
  collect(deltaContent: string | undefined): void {
    if (deltaContent && deltaContent.length > 0) {
      this.contentBuffer.push(deltaContent);
    }
  }

  /**
   * Return the full collected text.
   */
  getFullText(): string {
    return this.contentBuffer.join('');
  }
}

async function handleStreaming(
  req: Request,
  res: Response,
  config: ProxyConfig,
  startTime: number,
  memoryClient: MemoryClient | null
): Promise<void> {
  const model = (req.body?.model as string) || 'unknown';
  const inferenceUrl = `${config.inferenceServerUrl}/chat/completions`;

  // Inject memory context if client is available (await until complete)
  await injectMemoryContext(req.body, memoryClient, config);

  // Set SSE headers
  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');
  res.flushHeaders?.();

  // Create a stream collector to accumulate response content
  const collector = new StreamCollector();
  let observedOnce = false;

  // Helper: observe the collected text once the stream ends
  const observeOnStreamEnd = (): void => {
    if (observedOnce) return;
    observedOnce = true;

    const fullText = collector.getFullText();
    if (!fullText || fullText.trim().length === 0) {
      console.log('[proxy] Streaming observation skipped — no content collected');
      return;
    }

    const syntheticResponse = {
      choices: [{ message: { content: fullText } }],
    } as { choices: Array<{ message: { content: string } }> };

    observeAgentResponse(syntheticResponse, memoryClient, config).catch(console.warn);
  };

  try {
    const streamHeaders: Record<string, string> = {
      'Content-Type': 'application/json',
    };
    if (config.inferenceServerKey) {
      streamHeaders['Authorization'] = `Bearer ${config.inferenceServerKey}`;
    }

    const streamResult = await httpPostStream(inferenceUrl, req.body, streamHeaders, 300_000);

    console.log(`[proxy] → streaming started (${streamResult.status})  model=${model}`);

    const reader = streamResult.body.getReader();
    const decoder = new TextDecoder();

    try {
      let buffer = '';
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        const parts = buffer.split('\n\n');
        buffer = parts.pop() || '';

        for (const event of parts) {
          const lines = event.split('\n');
          for (const line of lines) {
            const trimmed = line.trimEnd();
            if (trimmed === '') continue;

            if (trimmed.startsWith('data: ')) {
              const data = trimmed.slice(6);
              if (data === '[DONE]') {
                const latency = Date.now() - startTime;
                console.log(`[proxy] → streaming done (${latency}ms)  model=${model}`);
                res.write('data: [DONE]\n\n');
                observeOnStreamEnd();
                res.end();
                return;
              }

              try {
                const parsed = JSON.parse(data) as StreamingChunk;
                const content = parsed.choices?.[0]?.delta?.content;
                collector.collect(content);
              } catch (parseErr) {
                console.warn(`[proxy] Failed to parse SSE data chunk: ${parseErr instanceof Error ? parseErr.message : String(parseErr)}`);
              }

              res.write(`data: ${data}\n\n`);
            } else {
              res.write(trimmed + '\n');
            }
          }
        }
      }

      // Process any remaining buffer after stream ends
      if (buffer.trim().length > 0) {
        const lines = buffer.split('\n');
        for (const line of lines) {
          const trimmed = line.trimEnd();
          if (trimmed === '') continue;

          if (trimmed.startsWith('data: ')) {
            const data = trimmed.slice(6);
            if (data !== '[DONE]') {
              try {
                const parsed = JSON.parse(data) as StreamingChunk;
                const content = parsed.choices?.[0]?.delta?.content;
                collector.collect(content);
              } catch (parseErr) {
                console.warn(`[proxy] Failed to parse SSE data chunk: ${parseErr instanceof Error ? parseErr.message : String(parseErr)}`);
              }
              res.write(`data: ${data}\n\n`);
            }
          } else {
            res.write(trimmed + '\n');
          }
        }
      }
    } finally {
      reader.releaseLock();
    }

    // Stream ended without [DONE]
    if (!res.writableEnded) {
      res.write('data: [DONE]\n\n');
      observeOnStreamEnd();
      res.end();
    }
  } catch (err) {
    if (res.writableEnded) {
      const latency = Date.now() - startTime;
      console.warn(`[proxy] Suppressed error after response ended (${latency}ms) model=${model}: ${err instanceof Error ? err.message : String(err)}`);
      observeOnStreamEnd();
      return;
    }
    const latency = Date.now() - startTime;
    handleInferenceError(err, res, config, latency, model, true);
  }
}

// ── Error handling helper ─────────────────────────────────────────────────

export function handleInferenceError(
  err: unknown,
  res: Response,
  config: ProxyConfig,
  elapsed: number,
  model: string,
  isStreaming = false
): void {
  const httpErr = err as HttpError | null;
  const errorMessage = httpErr?.message ?? String(err ?? '');

  if (httpErr && httpErr.isServerResponse) {
    // The inference server returned an error response (4xx / 5xx)
    const status = httpErr.status!;
    const data = httpErr.data as Record<string, unknown> | undefined;
    console.log(
      `[proxy] ← ${status} (${elapsed}ms)  model=${model}`
    );

    if (isStreaming) {
      const errMsg = data?.error ?? httpErr.message ?? String(data ?? '');
      res.write(`data: ${JSON.stringify({ error: errMsg })}\n\n`);
      res.end();
      return;
    }

    res.status(status).json(data);
    return;
  }

  if (httpErr && httpErr.code === 'ABORTED') {
    console.log(`[proxy] ← 504 timeout (${elapsed}ms)  model=${model}`);
    const errResp: ErrorResponse = {
      error: `Inference server request timed out`,
      code: 504,
      details: `The request to ${config.inferenceServerUrl} did not complete in time.`,
    };
    if (isStreaming) {
      res.write(`data: ${JSON.stringify(errResp)}\n\n`);
      res.end();
      return;
    }
    res.status(504).json(errResp);
    return;
  }

  // Connection refused, DNS failure, etc. → 502
  console.log(
    `[proxy] ← 502 Bad Gateway (${elapsed}ms)  model=${model}  reason=${errorMessage}`
  );
  const errResp: ErrorResponse = {
    error: `Bad gateway — could not connect to inference server (${errorMessage})`,
    code: 502,
    details: {
      inferenceServerUrl: config.inferenceServerUrl,
      originalError: errorMessage,
    },
  };
  if (isStreaming) {
    res.write(`data: ${JSON.stringify(errResp)}\n\n`);
    res.end();
    return;
  }
  res.status(502).json(errResp);
}

/**
 * Forward a request to the inference server.
 *
 * This is a convenience wrapper that performs a non-streaming forward.
 * For streaming or more control, use the proxy app created by createProxyApp.
 *
 * @param req - Incoming Express request.
 * @param config - Proxy configuration.
 * @returns Promise resolving to the inference server response data.
 */
export async function forwardToInferenceServer(
  req: Request,
  config: Pick<ProxyConfig, 'inferenceServerUrl' | 'inferenceServerKey'>
): Promise<unknown> {
  const inferenceUrl = `${config.inferenceServerUrl}/chat/completions`;

  const fwdHeaders: Record<string, string> = {
    'Content-Type': 'application/json',
  };
  if (config.inferenceServerKey) {
    fwdHeaders['Authorization'] = `Bearer ${config.inferenceServerKey}`;
  }

  const result = await httpPost(inferenceUrl, req.body, fwdHeaders, 300_000);

  return result.data;
}
