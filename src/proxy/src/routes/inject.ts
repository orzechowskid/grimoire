// SPDX-License-Identifier: MIT

/**
 * Memory injection route handlers.
 *
 * This module exposes routes for:
 * - Direct memory injection: POST /memory/inject
 * - Observing agent responses: POST /memory/observe
 * - Memory search: POST /memory/search
 */

import { Router, Request, Response } from 'express';
import { MemoryClient } from '../memory-client';
import { InjectRequest, ObserveRequest, SearchRequest } from '../types/models';

/**
 * Create and configure the injection routes.
 *
 * @param memoryClient - HTTP client for the memory library.
 * @returns Express router with injection endpoints.
 */
export function createInjectionRoutes(memoryClient: MemoryClient | null): Router {
  const router = Router();

  /**
   * POST /memory/inject
   * Inject memory context for a given text.
   */
  router.post('/inject', async (req: Request, res: Response) => {
    if (!memoryClient) {
      return res.status(503).json({
        error: 'Memory client is not available',
        code: 503,
      });
    }

    const body: InjectRequest = req.body;

    if (!body?.text || typeof body.text !== 'string') {
      return res.status(400).json({
        error: 'Request body must contain a "text" field (string)',
        code: 400,
      });
    }

    try {
      const result = await memoryClient.inject({
        text: body.text,
        top_k: body.top_k,
        time_weighted: body.time_weighted,
      });
      return res.json(result);
    } catch (err) {
      const errMessage = err instanceof Error ? err.message : String(err);
      return res.status(503).json({
        error: 'Memory injection failed',
        code: 503,
        details: errMessage,
      });
    }
  });

  /**
   * POST /memory/observe
   * Feed an agent response into the memory pipeline.
   */
  router.post('/observe', async (req: Request, res: Response) => {
    if (!memoryClient) {
      return res.status(503).json({
        error: 'Memory client is not available',
        code: 503,
      });
    }

    const body: ObserveRequest = req.body;

    if (!body?.text || typeof body.text !== 'string') {
      return res.status(400).json({
        error: 'Request body must contain a "text" field (string)',
        code: 400,
      });
    }

    try {
      const result = await memoryClient.observe({
        text: body.text,
        session_id: body.session_id,
      });
      return res.json(result);
    } catch (err) {
      const errMessage = err instanceof Error ? err.message : String(err);
      return res.status(503).json({
        error: 'Memory observation failed',
        code: 503,
        details: errMessage,
      });
    }
  });

  /**
   * POST /memory/search
   * Perform semantic search over stored memory.
   */
  router.post('/search', async (req: Request, res: Response) => {
    if (!memoryClient) {
      return res.status(503).json({
        error: 'Memory client is not available',
        code: 503,
      });
    }

    const body: SearchRequest = req.body;

    if (!body?.query || typeof body.query !== 'string') {
      return res.status(400).json({
        error: 'Request body must contain a "query" field (string)',
        code: 400,
      });
    }

    try {
      const result = await memoryClient.search({
        query: body.query,
        top_k: body.top_k,
        min_similarity: body.min_similarity,
        time_weighted: body.time_weighted,
      });
      return res.json(result);
    } catch (err) {
      const errMessage = err instanceof Error ? err.message : String(err);
      return res.status(503).json({
        error: 'Memory search failed',
        code: 503,
        details: errMessage,
      });
    }
  });

  return router;
}
