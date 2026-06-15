// SPDX-License-Identifier: MIT

/**
 * Request logging middleware.
 *
 * This module provides request logging for the proxy server.
 * It will log incoming requests with method, URL, and status code.
 *
 * The actual logging is delegated to the `morgan` package,
 * but this module provides a centralized configuration point.
 */

import { Request, Response, NextFunction } from 'express';

/**
 * Create request logging middleware.
 *
 * @returns Express middleware function.
 */
export function createLoggingMiddleware() {
  // TODO: Use the 'morgan' package for request logging
  // morgan('combined') or morgan('dev') will be applied here
  return (_req: Request, _res: Response, next: NextFunction) => {
    next();
  };
}
