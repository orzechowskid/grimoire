// SPDX-License-Identifier: MIT

/**
 * Grimoire Proxy Server - Entry Point
 *
 * Memory-aware proxy server for AI agent inference.
 * Sits between agent harness and inference server,
 * enhancing requests with memory context and observing
 * responses to build memory.
 */

import express, { Request, Response } from 'express';
import { loadConfig } from './config';
import { createCorsMiddleware } from './middleware/cors';
import { createLoggingMiddleware } from './middleware/logging';
import { createProxyApp } from './proxy';
import { createInjectionRoutes } from './routes/inject';
import { createAdminRoutes } from './routes/admin';
import { createMemoryClient } from './memory-client';
import { HealthResponse } from './types/models';

/**
 * Load configuration.
 */
const config = loadConfig();

/**
 * Create and configure the Express application.
 */
const app = express();

// Configure middleware
app.use(createCorsMiddleware(config.corsOrigin));
app.use(createLoggingMiddleware());
// Parse JSON bodies for smaller endpoints (health, admin, memory routes).
// Chat completions (/v1/chat/completions) uses streaming metadata extraction
// to avoid buffering massive conversation histories into memory.
const smallBodyParser = express.json({ limit: '1mb' });
app.use('/health', smallBodyParser);
app.use('/memory', smallBodyParser);
app.use('/admin', smallBodyParser);

// Health check endpoint
app.get('/health', (_req: Request, res: Response) => {
  const healthResponse: HealthResponse = {
    status: 'ok',
    timestamp: new Date().toISOString(),
  };
  res.json(healthResponse);
});

// Create the memory client (may be null if unavailable)
let memoryClient = null;
try {
  memoryClient = createMemoryClient(config.memoryLibUrl);
  console.log(`[index] Memory client created: ${config.memoryLibUrl}`);
} catch (err) {
  const errMessage = err instanceof Error ? err.message : String(err);
  console.warn(`[index] Warning: Failed to create memory client: ${errMessage}. Continuing without memory injection.`);
}

// Create the proxy app with the memory client
const proxyApp = createProxyApp(config, memoryClient);

// Mount the proxy routes
app.use('/', proxyApp);

// Mount memory routes — createInjectionRoutes provides /inject, /observe, and /search
app.use('/memory', createInjectionRoutes(memoryClient));

// Mount admin routes
app.use('/admin', createAdminRoutes(memoryClient, config));

// Start the server
const PORT = config.port;
app.listen(PORT, () => {
  console.log(`Grimoire proxy server listening on port ${PORT}`);
  console.log(`Memory library URL: ${config.memoryLibUrl}`);
  console.log(`Inference server URL: ${config.inferenceServerUrl}`);
  console.log(`CORS origin: ${config.corsOrigin}`);
  console.log(`Memory injection enabled: ${config.memoryInjection.enabled}`);
  console.log(`Memory client available: ${memoryClient ? 'yes' : 'no'}`);
});

export { app };
