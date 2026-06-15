// SPDX-License-Identifier: MIT

/**
 * Admin route handlers.
 *
 * This module exposes administrative endpoints for:
 * - Health checks of the proxy and memory library
 * - Configuration introspection
 * - Graceful shutdown
 * - Memory statistics and management
 */

import { Router, Request, Response } from 'express';
import {
  MemoryClient,
  CircuitBreakerError,
} from '../memory-client';
import { ProxyConfig, MemoryHealthResponse } from '../types/models';

/**
 * Create and configure the admin routes.
 *
 * @param memoryClient - The memory client instance (may be null).
 * @param config - Proxy configuration.
 * @returns Express router with admin endpoints.
 */
export function createAdminRoutes(
  memoryClient: MemoryClient | null,
  config: ProxyConfig,
): Router {
  const router = Router();

  // ── GET /admin/health ────────────────────────────────────────────────

  router.get('/health', async (_req: Request, res: Response) => {
    const version = '0.1.0';

    let memoryStatus: 'connected' | 'unavailable' | 'degraded' = 'unavailable';
    let memoryDetails: MemoryHealthResponse | undefined;
    let circuitState = 'unknown';

    if (memoryClient) {
      circuitState = memoryClient.status().circuitState;

      try {
        const health = await memoryClient.health();
        memoryDetails = health;
        memoryStatus = health.status === 'ok' ? 'connected' : 'degraded';
      } catch (err) {
        // Memory is unavailable or degraded
        const circuit = memoryClient.status();
        memoryStatus = circuit.circuitState === 'OPEN' ? 'degraded' : 'unavailable';
      }
    }

    const response = {
      status: 'ok',
      timestamp: new Date().toISOString(),
      version,
      memory_library: {
        status: memoryStatus,
        ...(memoryDetails && { details: memoryDetails }),
        circuitState,
      },
      memory_injection: { enabled: config.memoryInjection.enabled },
      observation: { enabled: config.observation.enabled },
    };

    res.json(response);
  });

  // ── GET /admin/stats ─────────────────────────────────────────────────

  router.get('/stats', (_req: Request, res: Response) => {
    let circuitState = 'unknown';
    let lastError: string | null = null;
    let lastErrorTime: number | null = null;

    if (memoryClient) {
      const status = memoryClient.status();
      circuitState = status.circuitState;
      lastError = status.lastError;
      lastErrorTime = status.lastErrorTime;
    }

    const response = {
      circuitState,
      lastError,
      lastErrorTime,
      memoryLibUrl: config.memoryLibUrl,
    };

    res.json(response);
  });

  // ── POST /admin/shutdown ─────────────────────────────────────────────

  router.post('/shutdown', async (_req: Request, res: Response) => {
    // Best-effort shutdown of memory client
    if (memoryClient) {
      try {
        await memoryClient.shutdown();
      } catch {
        // Ignore errors — we're shutting down anyway
      }
    }

    res.json({ status: 'shutting_down' });
  });

  // ── POST /admin/memory/reset-circuit ─────────────────────────────────

  router.post('/memory/reset-circuit', (_req: Request, res: Response) => {
    if (!memoryClient) {
      return res.status(503).json({
        status: 'error',
        message: 'Memory client is not available',
      });
    }

    memoryClient.resetCircuitBreaker();

    const circuitState = memoryClient.status().circuitState;

    return res.json({
      status: 'circuit_reset',
      circuitState,
    });
  });

  return router;
}
