// SPDX-License-Identifier: MIT

/**
 * Configuration loader for the proxy server.
 *
 * Loads configuration from `config.json` (with environment variable
 * overrides) and provides sensible hardcoded defaults for development.
 *
 * Priority order (highest → lowest):
 *   1. Environment variable
 *   2. config.json value
 *   3. Hardcoded default
 */

import * as fs from 'fs';
import * as path from 'path';
import { ProxyConfig } from './types/models';

// ── Hardcoded defaults ──────────────────────────────────────────────────
// These are the baseline values used when neither config.json nor env vars
// provide a value for a given setting.

const DEFAULTS: ProxyConfig = {
  inferenceServerUrl: '',               // required — must be provided
  inferenceServerKey: '',
  port: 8767,
  memoryLibUrl: 'http://127.0.0.1:8766',
  corsOrigin: '*',
  memoryInjection: {
    enabled: true,
    topK: 10,
    timeWeighted: false,
  },
  observation: {
    enabled: true,
    batchSize: 1,
  },
};

// ── Config path resolution ──────────────────────────────────────────────

/**
 * Resolve the path to the project-wide config.json file.
 *
 * 1. Check `GRIMOIRE_CONFIG` env var (set by the launch script).
 * 2. Fall back to `repo_root/config.json` computed relative to this file.
 */
function resolveConfigPath(): string {
  const envPath = process.env.GRIMOIRE_CONFIG;
  if (envPath) {
    return envPath;
  }
  // __dirname is src/proxy/src/; go up 4 levels to reach repo root.
  return path.resolve(__dirname, '..', '..', '..', '..', 'config.json');
}

// ── JSON → ProxyConfig mapping ──────────────────────────────────────────

/** Raw JSON shape for the proxy section of config.json (snake_case). */
interface ProxyConfigJson {
  inference_server_url?: string;
  inference_server_key?: string;
  port?: number;
  memory_lib_url?: string;
  cors_origin?: string;
  memory_injection?: {
    enabled?: boolean;
    top_k?: number;
    time_weighted?: boolean;
  };
  observation?: {
    enabled?: boolean;
    batch_size?: number;
  };
}

/**
 * Read config.json and return the proxy section merged with defaults.
 *
 * Converts snake_case JSON keys to camelCase properties on ProxyConfig.
 * If the file cannot be read or the proxy section is absent, returns the
 * defaults.
 */
function loadProxyConfigFromJson(): ProxyConfig {
  const configPath = resolveConfigPath();

  // If the file does not exist, fall back to defaults.
  if (!fs.existsSync(configPath)) {
    return DEFAULTS;
  }

  let parsed: unknown;
  try {
    const raw = fs.readFileSync(configPath, 'utf-8');
    parsed = JSON.parse(raw);
  } catch {
    // If parsing fails, fall back to defaults.
    return DEFAULTS;
  }

  // Extract the "proxy" section (or use empty object).
  const proxySection: ProxyConfigJson =
    typeof parsed === 'object' && parsed !== null && 'proxy' in parsed
      ? (parsed as Record<string, unknown>).proxy as ProxyConfigJson || {}
      : {};

  // Map snake_case → camelCase, falling back to defaults for missing keys.
  return {
    inferenceServerUrl: proxySection.inference_server_url ?? DEFAULTS.inferenceServerUrl,
    inferenceServerKey: proxySection.inference_server_key ?? DEFAULTS.inferenceServerKey,
    port: proxySection.port ?? DEFAULTS.port,
    memoryLibUrl: proxySection.memory_lib_url ?? DEFAULTS.memoryLibUrl,
    corsOrigin: proxySection.cors_origin ?? DEFAULTS.corsOrigin,
    memoryInjection: {
      enabled: proxySection.memory_injection?.enabled ?? DEFAULTS.memoryInjection.enabled,
      topK: proxySection.memory_injection?.top_k ?? DEFAULTS.memoryInjection.topK,
      timeWeighted: proxySection.memory_injection?.time_weighted ?? DEFAULTS.memoryInjection.timeWeighted,
    },
    observation: {
      enabled: proxySection.observation?.enabled ?? DEFAULTS.observation.enabled,
      batchSize: proxySection.observation?.batch_size ?? DEFAULTS.observation.batchSize,
    },
  };
}

// ── Env-var override helpers ────────────────────────────────────────────

/** Coerce a string env var to a number, or return a fallback. */
function envNum(env: string | undefined, fallback: number): number {
  if (env === undefined || env === '') return fallback;
  const n = parseInt(env, 10);
  return isNaN(n) ? fallback : n;
}

/** Coerce a string env var to a boolean (default `true`). */
function envBoolTrue(env: string | undefined): boolean {
  if (env === undefined || env === '') return true;
  return env !== 'false';
}

/** Coerce a string env var to a boolean (default `false`). */
function envBoolFalse(env: string | undefined): boolean {
  if (env === undefined || env === '') return false;
  return env === 'true';
}

/**
 * Layer environment variable overrides on top of a base config.
 *
 * Every env var that is set (and non-empty) overrides the corresponding
 * property on the base config.
 */
function applyEnvOverrides(base: ProxyConfig): ProxyConfig {
  const out: ProxyConfig = { ...base };

  // Top-level properties
  if (process.env.INFERENCE_SERVER_URL !== undefined && process.env.INFERENCE_SERVER_URL !== '') {
    out.inferenceServerUrl = process.env.INFERENCE_SERVER_URL;
  }
  if (process.env.INFERENCE_SERVER_KEY !== undefined && process.env.INFERENCE_SERVER_KEY !== '') {
    out.inferenceServerKey = process.env.INFERENCE_SERVER_KEY;
  }
  const portVal = envNum(process.env.PORT, base.port);
  if (process.env.PORT !== undefined && process.env.PORT !== '') {
    out.port = portVal;
  }
  if (process.env.MEMORY_LIB_URL !== undefined && process.env.MEMORY_LIB_URL !== '') {
    out.memoryLibUrl = process.env.MEMORY_LIB_URL;
  }
  if (process.env.CORS_ORIGIN !== undefined && process.env.CORS_ORIGIN !== '') {
    out.corsOrigin = process.env.CORS_ORIGIN;
  }

  // memoryInjection overrides
  if (process.env.MEMORY_INJECTION_ENABLED !== undefined && process.env.MEMORY_INJECTION_ENABLED !== '') {
    out.memoryInjection = {
      ...out.memoryInjection,
      enabled: envBoolTrue(process.env.MEMORY_INJECTION_ENABLED),
    };
  }
  if (process.env.MEMORY_INJECTION_TOP_K !== undefined && process.env.MEMORY_INJECTION_TOP_K !== '') {
    out.memoryInjection = {
      ...out.memoryInjection,
      topK: envNum(process.env.MEMORY_INJECTION_TOP_K, out.memoryInjection.topK),
    };
  }
  if (process.env.MEMORY_INJECTION_TIME_WEIGHTED !== undefined && process.env.MEMORY_INJECTION_TIME_WEIGHTED !== '') {
    out.memoryInjection = {
      ...out.memoryInjection,
      timeWeighted: envBoolFalse(process.env.MEMORY_INJECTION_TIME_WEIGHTED),
    };
  }

  // observation overrides
  if (process.env.OBSERVATION_ENABLED !== undefined && process.env.OBSERVATION_ENABLED !== '') {
    out.observation = {
      ...out.observation,
      enabled: envBoolTrue(process.env.OBSERVATION_ENABLED),
    };
  }
  if (process.env.OBSERVATION_BATCH_SIZE !== undefined && process.env.OBSERVATION_BATCH_SIZE !== '') {
    out.observation = {
      ...out.observation,
      batchSize: envNum(process.env.OBSERVATION_BATCH_SIZE, out.observation.batchSize),
    };
  }

  return out;
}

// ── Public API ──────────────────────────────────────────────────────────

/**
 * Load and validate proxy configuration.
 *
 * Configuration is resolved in the following priority (highest → lowest):
 *   1. Environment variables (e.g. INFERENCE_SERVER_URL)
 *   2. Values from config.json (proxy section)
 *   3. Hardcoded defaults
 *
 * @returns ProxyConfig with all values resolved.
 * @throws Error if `inferenceServerUrl` is not provided by any source.
 */
export function loadConfig(): ProxyConfig {
  // Start with JSON + defaults.
  const config = loadProxyConfigFromJson();

  // Layer environment variables on top.
  const resolved = applyEnvOverrides(config);

  // Validate required field.
  if (!resolved.inferenceServerUrl) {
    throw new Error(
      'INFERENCE_SERVER_URL is required. ' +
      'Set it via the INFERENCE_SERVER_URL environment variable ' +
      'or the "inference_server_url" field in config.json ' +
      '(e.g. http://localhost:11434/v1).'
    );
  }

  return resolved;
}
