// SPDX-License-Identifier: MIT

/**
 * CORS middleware.
 *
 * This module configures CORS (Cross-Origin Resource Sharing)
 * for the proxy server based on the configured CORS origin.
 *
 * The actual CORS handling is delegated to the `cors` package,
 * but this module provides a centralized configuration point.
 */

import { Request, Response, NextFunction } from 'express';

/**
 * Create CORS configuration middleware.
 *
 * @param origin - Allowed CORS origin (e.g., '*', 'http://localhost:3000').
 * @returns Express middleware function.
 */
export function createCorsMiddleware(origin: string) {
  // TODO: Use the 'cors' package to create middleware
  // For now, return a stub that does nothing
  return (_req: Request, _res: Response, next: NextFunction) => {
    // cors({ origin }) will be applied here
    next();
  };
}
