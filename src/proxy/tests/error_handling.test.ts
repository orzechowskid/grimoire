// SPDX-License-Identifier: MIT

import { describe, it, expect, vi } from 'vitest';
import { handleInferenceError } from '../src/proxy';
import { HttpError } from '../src/http';
import type { ProxyConfig } from '../src/types/models';
import type { Response as ExpressResponse } from 'express';

// ── Helpers ────────────────────────────────────────────────────────────────

/** Mutable container for end state. */
const endState = { ended: false };

/**
 * Build a mock Express Response that captures calls to .status(), .json(), .write(), .end().
 */
function buildMockRes(): {
  res: Partial<ExpressResponse>;
  calls: Array<{ status?: number; jsonArg?: unknown }>;
  written: string[];
} {
  const calls: Array<{ status?: number; jsonArg?: unknown }> = [];
  const written: string[] = [];

  const res: Partial<ExpressResponse> = {
    status: vi.fn(function (this: any, status: number) {
      calls.push({ status });
      return this;
    }),
    json: vi.fn(function (this: any, arg: unknown) {
      const last = calls[calls.length - 1];
      if (last) last.jsonArg = arg;
      return this;
    }),
    write: vi.fn(function (this: any, chunk: string) {
      written.push(chunk);
      return true;
    }),
    end: vi.fn(function (this: any) {
      endState.ended = true;
      return this;
    }),
  };

  return { res, calls, written };
}

/** Minimal proxy config for the tests. */
const config: ProxyConfig = {
  memoryLibUrl: 'http://localhost:8081',
  inferenceServerUrl: 'http://localhost:8080',
  inferenceServerKey: 'test-key',
  port: 8080,
  corsOrigin: '*',
  memoryInjection: {
    enabled: false,
    topK: 10,
    timeWeighted: false,
  },
  observation: {
    enabled: false,
    batchSize: 10,
  },
};

// ── Test suites ────────────────────────────────────────────────────────────

describe('handleInferenceError — non-streaming', () => {
  const elapsed = 123;
  const model = 'test-model';

  beforeEach(() => {
    endState.ended = false;
    vi.clearAllMocks();
  });

  // ── 400 Bad Request ──────────────────────────────────────────────────────
  describe('400 status', () => {
    it('passes through the error response body from the inference server', () => {
      const { res, calls } = buildMockRes();
      const serverError = { error: 'Invalid request', code: 'invalid_param' };
      const httpErr = new HttpError({ message: 'Bad Request', status: 400, data: serverError });

      handleInferenceError(httpErr, res as ExpressResponse, config, elapsed, model, false);

      expect(calls).toEqual([{ status: 400, jsonArg: serverError }]);
    });

    it('passes through minimal error body with only "error" field', () => {
      const { res, calls } = buildMockRes();
      const serverError = { error: 'Invalid body' };
      const httpErr = new HttpError({ message: 'Bad Request', status: 400, data: serverError });

      handleInferenceError(httpErr, res as ExpressResponse, config, elapsed, model, false);

      expect(calls).toEqual([{ status: 400, jsonArg: serverError }]);
    });

    it('handles undefined response data', () => {
      const { res, calls } = buildMockRes();
      const httpErr = new HttpError({ message: 'Bad Request', status: 400, data: undefined });

      handleInferenceError(httpErr, res as ExpressResponse, config, elapsed, model, false);

      expect(calls).toEqual([{ status: 400, jsonArg: undefined }]);
    });
  });

  // ── 401 Unauthorized ─────────────────────────────────────────────────────
  describe('401 status — authentication error', () => {
    it('passes through the error response body from the inference server', () => {
      const { res, calls } = buildMockRes();
      const serverError = { error: 'Invalid or missing API key' };
      const httpErr = new HttpError({ message: 'Unauthorized', status: 401, data: serverError });

      handleInferenceError(httpErr, res as ExpressResponse, config, elapsed, model, false);

      expect(calls).toEqual([{ status: 401, jsonArg: serverError }]);
    });

    it('passes through a response with code field', () => {
      const { res, calls } = buildMockRes();
      const serverError = { error: 'Invalid or missing API key', code: 401 };
      const httpErr = new HttpError({ message: 'Unauthorized', status: 401, data: serverError });

      handleInferenceError(httpErr, res as ExpressResponse, config, elapsed, model, false);

      expect(calls).toEqual([{ status: 401, jsonArg: serverError }]);
    });
  });

  // ── 429 Rate Limit ───────────────────────────────────────────────────────
  describe('429 status — rate limit', () => {
    it('passes through the rate-limit error response', () => {
      const { res, calls } = buildMockRes();
      const serverError = { error: 'Rate limit exceeded' };
      const httpErr = new HttpError({ message: 'Too Many Requests', status: 429, data: serverError });

      handleInferenceError(httpErr, res as ExpressResponse, config, elapsed, model, false);

      expect(calls).toEqual([{ status: 429, jsonArg: serverError }]);
    });

    it('passes through detailed rate-limit error response', () => {
      const { res, calls } = buildMockRes();
      const serverError = {
        error: 'Rate limit exceeded',
        retry_after: 5,
        code: 'rate_limited',
      };
      const httpErr = new HttpError({ message: 'Too Many Requests', status: 429, data: serverError });

      handleInferenceError(httpErr, res as ExpressResponse, config, elapsed, model, false);

      expect(calls).toEqual([{ status: 429, jsonArg: serverError }]);
    });
  });

  // ── 500 Internal Server Error ────────────────────────────────────────────
  describe('500 status — server error', () => {
    it('passes through the 500 error response body', () => {
      const { res, calls } = buildMockRes();
      const serverError = { error: 'Internal server error', details: 'crash' };
      const httpErr = new HttpError({ message: 'Internal Server Error', status: 500, data: serverError });

      handleInferenceError(httpErr, res as ExpressResponse, config, elapsed, model, false);

      expect(calls).toEqual([{ status: 500, jsonArg: serverError }]);
    });
  });

  // ── Other 4xx / 5xx status codes ─────────────────────────────────────────
  describe('other 4xx / 5xx status codes', () => {
    it('passes through 404 Not Found', () => {
      const { res, calls } = buildMockRes();
      const serverError = { error: 'Model not found' };
      const httpErr = new HttpError({ message: 'Not Found', status: 404, data: serverError });

      handleInferenceError(httpErr, res as ExpressResponse, config, elapsed, model, false);

      expect(calls).toEqual([{ status: 404, jsonArg: serverError }]);
    });

    it('passes through 502 Bad Gateway from inference server', () => {
      const { res, calls } = buildMockRes();
      const serverError = { error: 'Bad gateway from upstream' };
      const httpErr = new HttpError({ message: 'Bad Gateway', status: 502, data: serverError });

      handleInferenceError(httpErr, res as ExpressResponse, config, elapsed, model, false);

      expect(calls).toEqual([{ status: 502, jsonArg: serverError }]);
    });
  });

  // ── 504 timeout (ECONNABORTED / ETIMEDOUT) ──────────────────────────────
  describe('504 timeout (ECONNABORTED / ETIMEDOUT)', () => {
    it('returns a 504 response when error code is ECONNABORTED', () => {
      const { res, calls } = buildMockRes();
      const httpErr = new HttpError({ message: 'timeout', code: 'ABORTED' });

      handleInferenceError(httpErr, res as ExpressResponse, config, elapsed, model, false);

      expect(calls).toEqual([
        {
          status: 504,
          jsonArg: {
            error: 'Inference server request timed out',
            code: 504,
            details: `The request to ${config.inferenceServerUrl} did not complete in time.`,
          },
        },
      ]);
    });

    it('returns a 504 response when error code is ETIMEDOUT', () => {
      const { res, calls } = buildMockRes();
      const httpErr = new HttpError({ message: 'timeout', code: 'ABORTED' });

      handleInferenceError(httpErr, res as ExpressResponse, config, elapsed, model, false);

      expect(calls).toEqual([
        {
          status: 504,
          jsonArg: {
            error: 'Inference server request timed out',
            code: 504,
            details: `The request to ${config.inferenceServerUrl} did not complete in time.`,
          },
        },
      ]);
    });
  });

  // ── 502 fallback (generic errors — no HttpError.response) ───────────────
  describe('502 fallback for generic / non-HttpError errors', () => {
    it('handles a generic Error object → 502', () => {
      const { res, calls } = buildMockRes();
      const genericError = new Error('connection refused');

      handleInferenceError(genericError, res as ExpressResponse, config, elapsed, model, false);

      expect(calls).toEqual([
        {
          status: 502,
          jsonArg: expect.objectContaining({
            error: expect.stringContaining('Bad gateway'),
            code: 502,
          }),
        },
      ]);
    });

    it('handles a string error → 502', () => {
      const { res, calls } = buildMockRes();
      const strError = 'network unreachable' as unknown;

      handleInferenceError(strError, res as ExpressResponse, config, elapsed, model, false);

      expect(calls).toEqual([
        {
          status: 502,
          jsonArg: expect.objectContaining({
            error: expect.stringContaining('Bad gateway'),
            code: 502,
          }),
        },
      ]);
    });

    it('handles null → 502', () => {
      const { res, calls } = buildMockRes();

      handleInferenceError(null, res as ExpressResponse, config, elapsed, model, false);

      expect(calls).toEqual([
        {
          status: 502,
          jsonArg: expect.objectContaining({
            error: expect.stringContaining('Bad gateway'),
            code: 502,
          }),
        },
      ]);
    });

    it('handles undefined → 502', () => {
      const { res, calls } = buildMockRes();

      handleInferenceError(undefined, res as ExpressResponse, config, elapsed, model, false);

      expect(calls).toEqual([
        {
          status: 502,
          jsonArg: expect.objectContaining({
            error: expect.stringContaining('Bad gateway'),
            code: 502,
          }),
        },
      ]);
    });

    it('handles empty HttpError with no response → 502', () => {
      const { res, calls } = buildMockRes();
      const emptyHttpErr = new HttpError({ message: '' });

      handleInferenceError(emptyHttpErr, res as ExpressResponse, config, elapsed, model, false);

      expect(calls).toEqual([
        {
          status: 502,
          jsonArg: expect.objectContaining({
            error: expect.stringContaining('Bad gateway'),
            code: 502,
          }),
        },
      ]);
    });
  });

  // ── Error response structure ─────────────────────────────────────────────
  describe('error response structure', () => {
    it('400 response body contains expected keys from inference server', () => {
      const { res, calls } = buildMockRes();
      const serverError = { error: 'Bad request', code: 'invalid_param' };
      const httpErr = new HttpError({ message: 'Bad Request', status: 400, data: serverError });

      handleInferenceError(httpErr, res as ExpressResponse, config, elapsed, model, false);

      const [call] = calls;
      expect(call.status).toBe(400);
      expect(call.jsonArg).toEqual(serverError);
    });

    it('504 response has error, code, and details fields', () => {
      const { res, calls } = buildMockRes();
      const httpErr = new HttpError({ message: 'timeout', code: 'ABORTED' });

      handleInferenceError(httpErr, res as ExpressResponse, config, elapsed, model, false);

      const [call] = calls;
      expect(call.status).toBe(504);
      const body = call.jsonArg as Record<string, unknown>;
      expect(typeof body.error).toBe('string');
      expect(body.code).toBe(504);
      expect(typeof body.details).toBe('string');
    });

    it('502 response has error, code, and details fields', () => {
      const { res, calls } = buildMockRes();
      const genericError = new Error('connection refused');

      handleInferenceError(genericError, res as ExpressResponse, config, elapsed, model, false);

      const [call] = calls;
      expect(call.status).toBe(502);
      const body = call.jsonArg as Record<string, unknown>;
      expect(typeof body.error).toBe('string');
      expect(body.code).toBe(502);
      expect(typeof body.details).toBe('object');
    });

    it('429 response passes through server error body', () => {
      const { res, calls } = buildMockRes();
      const serverError = { error: 'Rate limit exceeded', retry_after: 5 };
      const httpErr = new HttpError({ message: 'Too Many Requests', status: 429, data: serverError });

      handleInferenceError(httpErr, res as ExpressResponse, config, elapsed, model, false);

      const [call] = calls;
      expect(call.status).toBe(429);
      expect(call.jsonArg).toEqual(serverError);
    });

    it('401 response passes through server error body', () => {
      const { res, calls } = buildMockRes();
      const serverError = { error: 'Invalid API key' };
      const httpErr = new HttpError({ message: 'Unauthorized', status: 401, data: serverError });

      handleInferenceError(httpErr, res as ExpressResponse, config, elapsed, model, false);

      const [call] = calls;
      expect(call.status).toBe(401);
      expect(call.jsonArg).toEqual(serverError);
    });

    it('500 response passes through server error body', () => {
      const { res, calls } = buildMockRes();
      const serverError = { error: 'Internal server error', details: 'crash' };
      const httpErr = new HttpError({ message: 'Internal Server Error', status: 500, data: serverError });

      handleInferenceError(httpErr, res as ExpressResponse, config, elapsed, model, false);

      const [call] = calls;
      expect(call.status).toBe(500);
      expect(call.jsonArg).toEqual(serverError);
    });
  });
});

describe('handleInferenceError — streaming mode', () => {
  const elapsed = 123;
  const model = 'test-model';

  beforeEach(() => {
    endState.ended = false;
    vi.restoreAllMocks();
  });

  describe('streaming error responses', () => {
    it('writes SSE data for 400 errors in streaming mode', () => {
      const { res, written } = buildMockRes();
      const serverError = { error: 'Invalid request' };
      const httpErr = new HttpError({ message: 'Bad Request', status: 400, data: serverError });

      handleInferenceError(httpErr, res as ExpressResponse, config, elapsed, model, true);

      expect(endState.ended).toBe(true);
      expect(written.join('')).toContain(`data: ${JSON.stringify(serverError)}\n\n`);
    });

    it('writes SSE data for 401 errors in streaming mode', () => {
      const { res, written } = buildMockRes();
      const serverError = { error: 'Invalid API key' };
      const httpErr = new HttpError({ message: 'Unauthorized', status: 401, data: serverError });

      handleInferenceError(httpErr, res as ExpressResponse, config, elapsed, model, true);

      expect(endState.ended).toBe(true);
      expect(written).toContain(`data: ${JSON.stringify(serverError)}\n\n`);
    });

    it('writes SSE data for 429 errors in streaming mode', () => {
      const { res, written } = buildMockRes();
      const serverError = { error: 'Rate limit exceeded' };
      const httpErr = new HttpError({ message: 'Too Many Requests', status: 429, data: serverError });

      handleInferenceError(httpErr, res as ExpressResponse, config, elapsed, model, true);

      expect(endState.ended).toBe(true);
      expect(written).toContain(`data: ${JSON.stringify(serverError)}\n\n`);
    });

    it('writes SSE data for 500 errors in streaming mode', () => {
      const { res, written } = buildMockRes();
      const serverError = { error: 'Internal server error' };
      const httpErr = new HttpError({ message: 'Internal Server Error', status: 500, data: serverError });

      handleInferenceError(httpErr, res as ExpressResponse, config, elapsed, model, true);

      expect(endState.ended).toBe(true);
      expect(written).toContain(`data: ${JSON.stringify(serverError)}\n\n`);
    });

    it('writes SSE data for 504 timeout errors in streaming mode', () => {
      const { res, written } = buildMockRes();
      const httpErr = new HttpError({ message: 'timeout', code: 'ABORTED' });

      handleInferenceError(httpErr, res as ExpressResponse, config, elapsed, model, true);

      expect(endState.ended).toBe(true);
      expect(written.join('')).toContain('data:');
      expect(written.join('')).toContain('504');
    });

    it('writes SSE data for 502 fallback errors in streaming mode', () => {
      const { res, written } = buildMockRes();
      const genericError = new Error('connection refused');

      handleInferenceError(genericError, res as ExpressResponse, config, elapsed, model, true);

      expect(endState.ended).toBe(true);
      expect(written.join('')).toContain('data:');
      expect(written.join('')).toContain('502');
    });

    it('uses httpErr.message when response data has no "error" field (streaming)', () => {
      const { res, written } = buildMockRes();
      const httpErr = new HttpError({ message: 'custom server message', status: 400, data: { foo: 'bar' } });

      handleInferenceError(httpErr, res as ExpressResponse, config, elapsed, model, true);

      expect(endState.ended).toBe(true);
      expect(written.join('')).toContain('custom server message');
    });

    it('uses data string when no error field and no message (streaming)', () => {
      const { res, written } = buildMockRes();
      const httpErr = new HttpError({ message: '', status: 400, data: 'plain error text' });

      handleInferenceError(httpErr, res as ExpressResponse, config, elapsed, model, true);

      expect(endState.ended).toBe(true);
      expect(written.join('')).toContain('data:');
    });
  });
});
