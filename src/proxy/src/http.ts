// SPDX-License-Identifier: MIT

/**
 * Fetch-based HTTP client module.
 *
 * Replaces axios with native fetch, AbortController for timeouts,
 * and structured error handling.
 *
 * This module is self-contained with no dependencies on other proxy source files.
 */

import { Readable } from 'stream';

// ── HttpError ──────────────────────────────────────────────────────────

export class HttpError extends Error {
  readonly status: number | undefined;
  readonly data: unknown;
  readonly code: string | undefined;
  readonly isServerResponse: boolean;

  constructor({
    status,
    data,
    code,
    message,
  }: {
    status?: number;
    data?: unknown;
    code?: string;
    message: string;
  }) {
    super(message);
    this.name = 'HttpError';
    this.status = status;
    this.data = data;
    this.code = code;
    this.isServerResponse = status !== undefined;
  }
}

// ── Helpers ────────────────────────────────────────────────────────────

/**
 * Parse JSON from a response body. Throws HttpError if parsing fails.
 */
async function parseJson(response: Response): Promise<unknown> {
  const text = await response.text();
  try {
    return JSON.parse(text);
  } catch {
    throw new HttpError({
      status: response.status,
      data: text,
      message: `Failed to parse JSON response (${response.status} ${response.statusText}): ${text.slice(0, 200)}`,
    });
  }
}

// ── httpGet ────────────────────────────────────────────────────────────

export async function httpGet<T>(
  url: string,
  headers: Record<string, string>,
  timeoutMs: number,
): Promise<{ status: number; data: T }> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(url, {
      method: 'GET',
      headers,
      signal: controller.signal,
    });

    // Non-2xx → server error response
    if (!response.ok) {
      const data = await parseJson(response);
      throw new HttpError({
        status: response.status,
        data,
        message: `HTTP ${response.status} ${response.statusText}`,
      });
    }

    const data = (await response.json()) as T;
    return { status: response.status, data };
  } catch (err) {
    if (err instanceof HttpError) {
      throw err;
    }
    if (err instanceof DOMException && err.name === 'AbortError') {
      throw new HttpError({
        code: 'ABORTED',
        message: `Request to ${url} timed out after ${timeoutMs}ms`,
      });
    }
    // TypeError → network error (connection refused, DNS failure, etc.)
    throw new HttpError({
      code: 'NETWORK_ERROR',
      message: `Network error fetching ${url}: ${err instanceof Error ? err.message : String(err)}`,
    });
  } finally {
    clearTimeout(timeoutId);
  }
}

// ── httpPost ───────────────────────────────────────────────────────────

export async function httpPost<T>(
  url: string,
  body: unknown,
  headers: Record<string, string>,
  timeoutMs: number,
): Promise<{ status: number; data: T }> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  // Ensure Content-Type header
  const contentHeaders: Record<string, string> = {
    ...headers,
    'Content-Type': headers['Content-Type'] ?? 'application/json',
  };

  try {
    const response = await fetch(url, {
      method: 'POST',
      headers: contentHeaders,
      body: JSON.stringify(body),
      signal: controller.signal,
    });

    // Non-2xx → server error response
    if (!response.ok) {
      const data = await parseJson(response);
      throw new HttpError({
        status: response.status,
        data,
        message: `HTTP ${response.status} ${response.statusText}`,
      });
    }

    const data = (await response.json()) as T;
    return { status: response.status, data };
  } catch (err) {
    if (err instanceof HttpError) {
      throw err;
    }
    if (err instanceof DOMException && err.name === 'AbortError') {
      throw new HttpError({
        code: 'ABORTED',
        message: `Request to ${url} timed out after ${timeoutMs}ms`,
      });
    }
    throw new HttpError({
      code: 'NETWORK_ERROR',
      message: `Network error fetching ${url}: ${err instanceof Error ? err.message : String(err)}`,
    });
  } finally {
    clearTimeout(timeoutId);
  }
}

// ── httpPostStream ─────────────────────────────────────────────────────

export async function httpPostStream(
  url: string,
  body: unknown,
  headers: Record<string, string>,
  timeoutMs: number,
): Promise<{ status: number; body: ReadableStream<Uint8Array> }> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  // Ensure Content-Type header
  const contentHeaders: Record<string, string> = {
    ...headers,
    'Content-Type': headers['Content-Type'] ?? 'application/json',
  };

  try {
    const response = await fetch(url, {
      method: 'POST',
      headers: contentHeaders,
      body: JSON.stringify(body),
      signal: controller.signal,
    });

    // Non-2xx → try to parse error body as JSON; fallback to string
    if (!response.ok) {
      let data: unknown;
      try {
        const text = await response.text();
        data = JSON.parse(text);
      } catch {
        // If parsing fails, throw with the raw text as data
        const text = await response.text();
        throw new HttpError({
          status: response.status,
          data: text,
          message: `HTTP ${response.status} ${response.statusText}`,
        });
      }
      throw new HttpError({
        status: response.status,
        data,
        message: `HTTP ${response.status} ${response.statusText}`,
      });
    }

    if (!response.body) {
      throw new HttpError({
        message: `Response from ${url} has no body stream`,
      });
    }

    return { status: response.status, body: response.body };
  } catch (err) {
    if (err instanceof HttpError) {
      throw err;
    }
    if (err instanceof DOMException && err.name === 'AbortError') {
      throw new HttpError({
        code: 'ABORTED',
        message: `Request to ${url} timed out after ${timeoutMs}ms`,
      });
    }
    throw new HttpError({
      code: 'NETWORK_ERROR',
      message: `Network error fetching ${url}: ${err instanceof Error ? err.message : String(err)}`,
    });
  } finally {
    clearTimeout(timeoutId);
  }
}

// ── httpPostRawStream ────────────────────────────────────────────────────

export async function httpPostRawStream(
  url: string,
  bodyStream: Readable,
  headers: Record<string, string>,
  timeoutMs: number,
): Promise<{ status: number; body: ReadableStream<Uint8Array> }> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  // Convert Node.js Readable to Web ReadableStream
  const webStream = new ReadableStream({
    async pull(controller) {
      for await (const chunk of bodyStream) {
        controller.enqueue(Buffer.from(chunk));
      }
      controller.close();
    },
  });

  try {
    const response = await fetch(url, {
      method: 'POST',
      headers,
      body: webStream,
      signal: controller.signal,
    });

    // Non-2xx → try to parse error body as JSON; fallback to string
    if (!response.ok) {
      let data: unknown;
      try {
        const text = await response.text();
        data = JSON.parse(text);
      } catch {
        // If parsing fails, throw with the raw text as data
        const text = await response.text();
        throw new HttpError({
          status: response.status,
          data: text,
          message: `HTTP ${response.status} ${response.statusText}`,
        });
      }
      throw new HttpError({
        status: response.status,
        data,
        message: `HTTP ${response.status} ${response.statusText}`,
      });
    }

    if (!response.body) {
      throw new HttpError({
        message: `Response from ${url} has no body stream`,
      });
    }

    return { status: response.status, body: response.body };
  } catch (err) {
    if (err instanceof HttpError) {
      throw err;
    }
    if (err instanceof DOMException && err.name === 'AbortError') {
      throw new HttpError({
        code: 'ABORTED',
        message: `Request to ${url} timed out after ${timeoutMs}ms`,
      });
    }
    throw new HttpError({
      code: 'NETWORK_ERROR',
      message: `Network error fetching ${url}: ${err instanceof Error ? err.message : String(err)}`,
    });
  } finally {
    clearTimeout(timeoutId);
  }
}