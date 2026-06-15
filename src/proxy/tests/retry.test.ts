// SPDX-License-Identifier: MIT

import { describe, it, expect } from 'vitest';
import { calculateDelay, isRetryableError } from '../src/memory-client';
import { HttpError } from '../src/http';

// ── calculateDelay ───────────────────────────────────────────────────────

describe('calculateDelay', () => {
  const defaultOptions = {
    maxRetries: 3,
    initialDelayMs: 100,
    maxDelayMs: 2000,
    backoffMultiplier: 2,
  };

  it('attempt 0 returns initialDelayMs', () => {
    expect(calculateDelay(0, defaultOptions)).toBe(100);
  });

  it('attempt 1 returns initialDelayMs * backoffMultiplier', () => {
    expect(calculateDelay(1, defaultOptions)).toBe(200);
  });

  it('attempt 2 returns initialDelayMs * backoffMultiplier²', () => {
    expect(calculateDelay(2, defaultOptions)).toBe(400);
  });

  it('attempt 3 returns initialDelayMs * backoffMultiplier³', () => {
    expect(calculateDelay(3, defaultOptions)).toBe(800);
  });

  it('attempt 4 returns initialDelayMs * backoffMultiplier⁴', () => {
    expect(calculateDelay(4, defaultOptions)).toBe(1600);
  });

  it('attempt 5 is capped at maxDelayMs', () => {
    // 100 * 2^5 = 3200, but maxDelayMs is 2000
    expect(calculateDelay(5, defaultOptions)).toBe(2000);
  });

  it('attempt 10 is capped at maxDelayMs', () => {
    expect(calculateDelay(10, defaultOptions)).toBe(2000);
  });

  it('uses custom backoffMultiplier of 1 (linear, no growth)', () => {
    const linearOptions = { ...defaultOptions, backoffMultiplier: 1 };
    expect(calculateDelay(0, linearOptions)).toBe(100);
    expect(calculateDelay(1, linearOptions)).toBe(100);
    expect(calculateDelay(2, linearOptions)).toBe(100);
    expect(calculateDelay(3, linearOptions)).toBe(100);
  });

  it('uses custom backoffMultiplier of 3 (faster growth)', () => {
    const fastOptions = { ...defaultOptions, backoffMultiplier: 3 };
    expect(calculateDelay(0, fastOptions)).toBe(100);
    expect(calculateDelay(1, fastOptions)).toBe(300);
    expect(calculateDelay(2, fastOptions)).toBe(900);
    expect(calculateDelay(3, fastOptions)).toBe(2000); // capped at maxDelayMs
  });

  it('uses custom backoffMultiplier of 1.5 (moderate growth)', () => {
    const moderateOptions = { ...defaultOptions, backoffMultiplier: 1.5 };
    expect(calculateDelay(0, moderateOptions)).toBe(100);
    expect(calculateDelay(1, moderateOptions)).toBe(150);
    expect(calculateDelay(2, moderateOptions)).toBe(225);
    expect(calculateDelay(3, moderateOptions)).toBe(337.5);
  });

  it('uses custom initialDelayMs of 500', () => {
    const slowOptions = { ...defaultOptions, initialDelayMs: 500 };
    expect(calculateDelay(0, slowOptions)).toBe(500);
    expect(calculateDelay(1, slowOptions)).toBe(1000);
    expect(calculateDelay(2, slowOptions)).toBe(2000); // capped at maxDelayMs
  });

  it('uses custom maxDelayMs of 100', () => {
    const cappedOptions = { ...defaultOptions, maxDelayMs: 100 };
    // Even attempt 0 should be capped since initialDelayMs (100) <= maxDelayMs (100)
    expect(calculateDelay(0, cappedOptions)).toBe(100);
    expect(calculateDelay(1, cappedOptions)).toBe(100);
  });

  it('uses custom maxDelayMs of 1500', () => {
    const cappedOptions = { ...defaultOptions, maxDelayMs: 1500 };
    expect(calculateDelay(0, cappedOptions)).toBe(100);
    expect(calculateDelay(1, cappedOptions)).toBe(200);
    expect(calculateDelay(2, cappedOptions)).toBe(400);
    expect(calculateDelay(3, cappedOptions)).toBe(800);
    expect(calculateDelay(4, cappedOptions)).toBe(1500); // capped
    expect(calculateDelay(5, cappedOptions)).toBe(1500); // capped
  });
});

// ── isRetryableError ─────────────────────────────────────────────────────

describe('isRetryableError', () => {
  // --- Network errors (non-Axios) with error codes ---
  describe('network error codes', () => {
    it('ECONNREFUSED is retryable', () => {
      const err = { code: 'ECONNREFUSED' };
      expect(isRetryableError(err)).toBe(true);
    });

    it('ETIMEDOUT is not retryable (not in RETRYABLE_NETWORK_CODES)', () => {
      const err = { code: 'ETIMEDOUT' };
      expect(isRetryableError(err)).toBe(false);
    });

    it('ECONNRESET is retryable', () => {
      const err = { code: 'ECONNRESET' };
      expect(isRetryableError(err)).toBe(true);
    });

    it('EPIPE is retryable', () => {
      const err = { code: 'EPIPE' };
      expect(isRetryableError(err)).toBe(true);
    });

    it('ECONNABORTED is retryable', () => {
      const err = { code: 'ECONNABORTED' };
      expect(isRetryableError(err)).toBe(true);
    });

    it('unknown error code is not retryable', () => {
      const err = { code: 'UNKNOWN_CODE' };
      expect(isRetryableError(err)).toBe(false);
    });
  });

  // --- Axios errors with retryable HTTP status codes ---
  describe('HttpError — HTTP status 503', () => {
    it('HttpError with 503 status is retryable', () => {
      const error = new HttpError({ message: 'Service Unavailable', status: 503 });
      expect(isRetryableError(error)).toBe(true);
    });
  });

  // --- Non-retryable errors ---
  describe('non-retryable errors', () => {
    it('generic Error object is not retryable', () => {
      const err = new Error('something went wrong');
      expect(isRetryableError(err)).toBe(false);
    });

    it('string error is not retryable', () => {
      expect(isRetryableError('something went wrong')).toBe(false);
    });

    it('null is not retryable', () => {
      expect(isRetryableError(null)).toBe(false);
    });

    it('undefined is not retryable', () => {
      expect(isRetryableError(undefined)).toBe(false);
    });

    it('number is not retryable', () => {
      expect(isRetryableError(42)).toBe(false);
    });

    it('boolean is not retryable', () => {
      expect(isRetryableError(true)).toBe(false);
    });

    it('empty object is not retryable', () => {
      expect(isRetryableError({})).toBe(false);
    });

    it('HttpError with non-retryable status 400 is not retryable', () => {
      const error = new HttpError({ message: 'Bad Request', status: 400 });
      expect(isRetryableError(error)).toBe(false);
    });

    it('HttpError with non-retryable status 401 is not retryable', () => {
      const error = new HttpError({ message: 'Unauthorized', status: 401 });
      expect(isRetryableError(error)).toBe(false);
    });

    it('HttpError with non-retryable status 403 is not retryable', () => {
      const error = new HttpError({ message: 'Forbidden', status: 403 });
      expect(isRetryableError(error)).toBe(false);
    });

    it('HttpError with non-retryable status 404 is not retryable', () => {
      const error = new HttpError({ message: 'Not Found', status: 404 });
      expect(isRetryableError(error)).toBe(false);
    });

    it('HttpError with no response is not retryable', () => {
      const error = new HttpError({ message: 'Network error' });
      expect(isRetryableError(error)).toBe(false);
    });

    it('HttpError with ABORTED code (timeout) is retryable', () => {
      const error = new HttpError({ message: 'Request to http://example.com timed out after 5000ms', code: 'ABORTED' });
      expect(isRetryableError(error)).toBe(true);
    });
  });

  // --- Axios errors with retryable network-level codes ---
  describe('HttpError with retryable network codes', () => {
    it('HttpError with ECONNREFUSED code is retryable', () => {
      const error = new HttpError({ message: 'Connection refused', code: 'ECONNREFUSED' });
      expect(isRetryableError(error)).toBe(true);
    });

    it('HttpError with ECONNRESET code is retryable', () => {
      const error = new HttpError({ message: 'Connection reset', code: 'ECONNRESET' });
      expect(isRetryableError(error)).toBe(true);
    });

    it('HttpError with EPIPE code is retryable', () => {
      const error = new HttpError({ message: 'Broken pipe', code: 'EPIPE' });
      expect(isRetryableError(error)).toBe(true);
    });

    it('HttpError with ECONNABORTED code is retryable', () => {
      const error = new HttpError({ message: 'Request aborted', code: 'ECONNABORTED' });
      expect(isRetryableError(error)).toBe(true);
    });
  });
});
