// SPDX-License-Identifier: MIT

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import * as fs from 'fs';

vi.mock('fs', async () => {
  const actualFs = await vi.importActual<typeof import('fs')>('fs');
  return {
    ...actualFs,
    existsSync: vi.fn(),
    readFileSync: vi.fn(),
  };
});

import { loadConfig } from '../src/config';

const configEnvVars = [
  'INFERENCE_SERVER_KEY',
  'INFERENCE_SERVER_URL',
  'MEMORY_LIB_URL',
  'PORT',
  'CORS_ORIGIN',
  'MEMORY_INJECTION_ENABLED',
  'MEMORY_INJECTION_TOP_K',
  'MEMORY_INJECTION_TIME_WEIGHTED',
  'OBSERVATION_ENABLED',
  'OBSERVATION_BATCH_SIZE',
  'GRIMOIRE_CONFIG',
] as const;

const originalEnv: Record<string, string | undefined> = {};

function clearAllEnv() {
  for (const key of configEnvVars) {
    delete process.env[key];
  }
}

beforeEach(() => {
  for (const key of configEnvVars) {
    originalEnv[key] = process.env[key];
  }
  clearAllEnv();
  vi.clearAllMocks();
});

afterEach(() => {
  for (const key of configEnvVars) {
    if (originalEnv[key] === undefined) {
      delete process.env[key];
    } else {
      process.env[key] = originalEnv[key];
    }
  }
});

function mockJsonFile(content: Record<string, unknown> | null) {
  if (content === null) {
    (fs.existsSync as vi.Mock).mockReturnValue(false);
  } else {
    (fs.existsSync as vi.Mock).mockReturnValue(true);
    (fs.readFileSync as vi.Mock).mockReturnValue(JSON.stringify(content));
  }
}

function setEnv(key: string, value: string) {
  process.env[key] = value;
}

function fullProxyMock(): Record<string, unknown> {
  return {
    proxy: {
      inference_server_url: 'http://json-server:11434/v1',
      inference_server_key: 'json-key',
      port: 8767,
      memory_lib_url: 'http://127.0.0.1:8766',
      cors_origin: '*',
      memory_injection: { enabled: true, top_k: 10, time_weighted: false },
      observation: { enabled: true, batch_size: 1 },
    },
  };
}

// ── 1. JSON Loading ────────────────────────────────────────────────────

describe('JSON loading', () => {
  it('loads full proxy section from config.json', () => {
    mockJsonFile(fullProxyMock());
    const config = loadConfig();
    expect(config.inferenceServerUrl).toBe('http://json-server:11434/v1');
    expect(config.inferenceServerKey).toBe('json-key');
    expect(config.port).toBe(8767);
    expect(config.memoryLibUrl).toBe('http://127.0.0.1:8766');
    expect(config.corsOrigin).toBe('*');
    expect(config.memoryInjection.enabled).toBe(true);
    expect(config.memoryInjection.topK).toBe(10);
    expect(config.memoryInjection.timeWeighted).toBe(false);
    expect(config.observation.enabled).toBe(true);
    expect(config.observation.batchSize).toBe(1);
  });

  it('loads partial proxy section, fills missing keys from defaults', () => {
    mockJsonFile({
      proxy: {
        port: 9999,
        cors_origin: 'http://localhost:3000',
      },
    });
    setEnv('INFERENCE_SERVER_URL', 'http://default-server:11434/v1');
    const config = loadConfig();
    expect(config.port).toBe(9999);
    expect(config.corsOrigin).toBe('http://localhost:3000');
    // Missing keys should come from hardcoded defaults
    expect(config.inferenceServerKey).toBe('');
    expect(config.memoryLibUrl).toBe('http://127.0.0.1:8766');
    expect(config.memoryInjection.enabled).toBe(true);
    expect(config.memoryInjection.topK).toBe(10);
    expect(config.memoryInjection.timeWeighted).toBe(false);
    expect(config.observation.enabled).toBe(true);
    expect(config.observation.batchSize).toBe(1);
  });

  it('falls back to defaults when proxy section is missing', () => {
    mockJsonFile({ other_section: { some_key: 'some_value' } });
    setEnv('INFERENCE_SERVER_URL', 'http://default-server:11434/v1');
    const config = loadConfig();
    expect(config.inferenceServerUrl).toBe('http://default-server:11434/v1');
    expect(config.port).toBe(8767);
    expect(config.corsOrigin).toBe('*');
    expect(config.memoryLibUrl).toBe('http://127.0.0.1:8766');
    expect(config.memoryInjection.enabled).toBe(true);
    expect(config.memoryInjection.topK).toBe(10);
    expect(config.memoryInjection.timeWeighted).toBe(false);
    expect(config.observation.enabled).toBe(true);
    expect(config.observation.batchSize).toBe(1);
  });

  it('falls back to defaults when config.json doesn\'t exist', () => {
    mockJsonFile(null);
    setEnv('INFERENCE_SERVER_URL', 'http://default-server:11434/v1');
    const config = loadConfig();
    expect(config.inferenceServerUrl).toBe('http://default-server:11434/v1');
    expect(config.port).toBe(8767);
    expect(config.corsOrigin).toBe('*');
    expect(config.memoryLibUrl).toBe('http://127.0.0.1:8766');
    expect(config.memoryInjection.enabled).toBe(true);
    expect(config.memoryInjection.topK).toBe(10);
    expect(config.memoryInjection.timeWeighted).toBe(false);
    expect(config.observation.enabled).toBe(true);
    expect(config.observation.batchSize).toBe(1);
  });

  it('falls back to defaults when config.json has invalid JSON', () => {
    (fs.existsSync as vi.Mock).mockReturnValue(true);
    (fs.readFileSync as vi.Mock).mockReturnValue('not valid json{{{');
    setEnv('INFERENCE_SERVER_URL', 'http://default-server:11434/v1');
    const config = loadConfig();
    expect(config.port).toBe(8767);
    expect(config.corsOrigin).toBe('*');
    expect(config.memoryLibUrl).toBe('http://127.0.0.1:8766');
    expect(config.memoryInjection.enabled).toBe(true);
    expect(config.memoryInjection.topK).toBe(10);
    expect(config.memoryInjection.timeWeighted).toBe(false);
    expect(config.observation.enabled).toBe(true);
    expect(config.observation.batchSize).toBe(1);
  });

  it('respects GRIMOIRE_CONFIG env var for path resolution', () => {
    mockJsonFile({
      proxy: {
        port: 4444,
        inference_server_url: 'http://custom-path-server:11434/v1',
      },
    });
    setEnv('INFERENCE_SERVER_URL', 'http://default-server:11434/v1');
    setEnv('GRIMOIRE_CONFIG', '/some/custom/path/config.json');
    const config = loadConfig();
    expect(config.port).toBe(4444);
    expect(config.inferenceServerUrl).toBe('http://default-server:11434/v1');
  });
});

// ── 2. Three-layer priority ────────────────────────────────────────────

describe('Three-layer priority', () => {
  it('env var overrides JSON value', () => {
    mockJsonFile({
      proxy: {
        port: 8767,
        inference_server_url: 'http://json-server:11434/v1',
        inference_server_key: 'json-key',
        memory_lib_url: 'http://127.0.0.1:8766',
        cors_origin: '*',
        memory_injection: { enabled: true, top_k: 10, time_weighted: false },
        observation: { enabled: true, batch_size: 1 },
      },
    });
    process.env.PORT = '9999';
    const config = loadConfig();
    expect(config.port).toBe(9999);
  });

  it('JSON value overrides hardcoded default', () => {
    mockJsonFile({
      proxy: {
        port: 9999,
        inference_server_url: 'http://json-server:11434/v1',
        inference_server_key: 'json-key',
        memory_lib_url: 'http://127.0.0.1:8766',
        cors_origin: '*',
        memory_injection: { enabled: true, top_k: 10, time_weighted: false },
        observation: { enabled: true, batch_size: 1 },
      },
    });
    const config = loadConfig();
    expect(config.port).toBe(9999);
  });

  it('env var overrides everything in full chain', () => {
    mockJsonFile({
      proxy: {
        port: 8767,
        inference_server_url: 'http://json-server:11434/v1',
        inference_server_key: 'json-key',
        memory_lib_url: 'http://127.0.0.1:8766',
        cors_origin: '*',
        memory_injection: { enabled: true, top_k: 10, time_weighted: false },
        observation: { enabled: true, batch_size: 1 },
      },
    });
    process.env.PORT = '9999';
    const config = loadConfig();
    expect(config.port).toBe(9999);
  });
});

// ── 3. Defaults ────────────────────────────────────────────────────────

describe('Defaults', () => {
  it('returns correct defaults when no JSON file and no env vars', () => {
    mockJsonFile(null);
    setEnv('INFERENCE_SERVER_URL', 'https://custom.api.com/v1');
    const config = loadConfig();

    expect(config.inferenceServerKey).toBe('');
    expect(config.memoryLibUrl).toBe('http://127.0.0.1:8766');
    expect(config.inferenceServerUrl).toBe('https://custom.api.com/v1');
    expect(config.port).toBe(8767);
    expect(config.corsOrigin).toBe('*');

    expect(config.memoryInjection.enabled).toBe(true);
    expect(config.memoryInjection.topK).toBe(10);
    expect(config.memoryInjection.timeWeighted).toBe(false);

    expect(config.observation.enabled).toBe(true);
    expect(config.observation.batchSize).toBe(1);
  });

  it('throws when no inference server URL from any source', () => {
    mockJsonFile(null);
    expect(() => loadConfig()).toThrow();
  });
});

// ── 4. Environment variable overrides ─────────────────────────────────

describe('Environment variable overrides', () => {
  it('INFERENCE_SERVER_URL env var overrides JSON value', () => {
    mockJsonFile({
      proxy: {
        inference_server_url: 'http://json-server:11434/v1',
        inference_server_key: 'json-key',
        port: 8767,
        memory_lib_url: 'http://127.0.0.1:8766',
        cors_origin: '*',
        memory_injection: { enabled: true, top_k: 10, time_weighted: false },
        observation: { enabled: true, batch_size: 1 },
      },
    });
    setEnv('INFERENCE_SERVER_URL', 'http://env-server:11434/v1');
    const config = loadConfig();
    expect(config.inferenceServerUrl).toBe('http://env-server:11434/v1');
  });

  it('MEMORY_LIB_URL env var overrides JSON value', () => {
    mockJsonFile({
      proxy: {
        inference_server_url: 'http://json-server:11434/v1',
        inference_server_key: 'json-key',
        port: 8767,
        memory_lib_url: 'http://127.0.0.1:8766',
        cors_origin: '*',
        memory_injection: { enabled: true, top_k: 10, time_weighted: false },
        observation: { enabled: true, batch_size: 1 },
      },
    });
    setEnv('INFERENCE_SERVER_URL', 'http://json-server:11434/v1');
    setEnv('MEMORY_LIB_URL', 'http://env-memory:9999');
    const config = loadConfig();
    expect(config.memoryLibUrl).toBe('http://env-memory:9999');
  });

  it('PORT env var overrides JSON value', () => {
    mockJsonFile({
      proxy: {
        inference_server_url: 'http://json-server:11434/v1',
        inference_server_key: 'json-key',
        port: 8767,
        memory_lib_url: 'http://127.0.0.1:8766',
        cors_origin: '*',
        memory_injection: { enabled: true, top_k: 10, time_weighted: false },
        observation: { enabled: true, batch_size: 1 },
      },
    });
    setEnv('INFERENCE_SERVER_URL', 'http://json-server:11434/v1');
    setEnv('PORT', '9999');
    const config = loadConfig();
    expect(config.port).toBe(9999);
  });

  it('CORS_ORIGIN env var overrides JSON value', () => {
    mockJsonFile({
      proxy: {
        inference_server_url: 'http://json-server:11434/v1',
        inference_server_key: 'json-key',
        port: 8767,
        memory_lib_url: 'http://127.0.0.1:8766',
        cors_origin: '*',
        memory_injection: { enabled: true, top_k: 10, time_weighted: false },
        observation: { enabled: true, batch_size: 1 },
      },
    });
    setEnv('INFERENCE_SERVER_URL', 'http://json-server:11434/v1');
    setEnv('CORS_ORIGIN', 'http://localhost:3000');
    const config = loadConfig();
    expect(config.corsOrigin).toBe('http://localhost:3000');
  });

  it('INFERENCE_SERVER_KEY env var is picked up', () => {
    mockJsonFile({
      proxy: {
        inference_server_url: 'http://json-server:11434/v1',
        inference_server_key: 'json-key',
        port: 8767,
        memory_lib_url: 'http://127.0.0.1:8766',
        cors_origin: '*',
        memory_injection: { enabled: true, top_k: 10, time_weighted: false },
        observation: { enabled: true, batch_size: 1 },
      },
    });
    setEnv('INFERENCE_SERVER_URL', 'http://json-server:11434/v1');
    setEnv('INFERENCE_SERVER_KEY', 'env-api-key');
    const config = loadConfig();
    expect(config.inferenceServerKey).toBe('env-api-key');
  });

  it('MEMORY_INJECTION_ENABLED=false overrides JSON value', () => {
    mockJsonFile({
      proxy: {
        inference_server_url: 'http://json-server:11434/v1',
        inference_server_key: 'json-key',
        port: 8767,
        memory_lib_url: 'http://127.0.0.1:8766',
        cors_origin: '*',
        memory_injection: { enabled: true, top_k: 10, time_weighted: false },
        observation: { enabled: true, batch_size: 1 },
      },
    });
    setEnv('INFERENCE_SERVER_URL', 'http://json-server:11434/v1');
    setEnv('MEMORY_INJECTION_ENABLED', 'false');
    const config = loadConfig();
    expect(config.memoryInjection.enabled).toBe(false);
  });

  it('MEMORY_INJECTION_TOP_K env var overrides JSON value', () => {
    mockJsonFile({
      proxy: {
        inference_server_url: 'http://json-server:11434/v1',
        inference_server_key: 'json-key',
        port: 8767,
        memory_lib_url: 'http://127.0.0.1:8766',
        cors_origin: '*',
        memory_injection: { enabled: true, top_k: 10, time_weighted: false },
        observation: { enabled: true, batch_size: 1 },
      },
    });
    setEnv('INFERENCE_SERVER_URL', 'http://json-server:11434/v1');
    setEnv('MEMORY_INJECTION_TOP_K', '42');
    const config = loadConfig();
    expect(config.memoryInjection.topK).toBe(42);
  });

  it('MEMORY_INJECTION_TIME_WEIGHTED=true overrides JSON value', () => {
    mockJsonFile({
      proxy: {
        inference_server_url: 'http://json-server:11434/v1',
        inference_server_key: 'json-key',
        port: 8767,
        memory_lib_url: 'http://127.0.0.1:8766',
        cors_origin: '*',
        memory_injection: { enabled: true, top_k: 10, time_weighted: false },
        observation: { enabled: true, batch_size: 1 },
      },
    });
    setEnv('INFERENCE_SERVER_URL', 'http://json-server:11434/v1');
    setEnv('MEMORY_INJECTION_TIME_WEIGHTED', 'true');
    const config = loadConfig();
    expect(config.memoryInjection.timeWeighted).toBe(true);
  });

  it('OBSERVATION_ENABLED=false overrides JSON value', () => {
    mockJsonFile({
      proxy: {
        inference_server_url: 'http://json-server:11434/v1',
        inference_server_key: 'json-key',
        port: 8767,
        memory_lib_url: 'http://127.0.0.1:8766',
        cors_origin: '*',
        memory_injection: { enabled: true, top_k: 10, time_weighted: false },
        observation: { enabled: true, batch_size: 1 },
      },
    });
    setEnv('INFERENCE_SERVER_URL', 'http://json-server:11434/v1');
    setEnv('OBSERVATION_ENABLED', 'false');
    const config = loadConfig();
    expect(config.observation.enabled).toBe(false);
  });

  it('OBSERVATION_BATCH_SIZE env var overrides JSON value', () => {
    mockJsonFile({
      proxy: {
        inference_server_url: 'http://json-server:11434/v1',
        inference_server_key: 'json-key',
        port: 8767,
        memory_lib_url: 'http://127.0.0.1:8766',
        cors_origin: '*',
        memory_injection: { enabled: true, top_k: 10, time_weighted: false },
        observation: { enabled: true, batch_size: 1 },
      },
    });
    setEnv('INFERENCE_SERVER_URL', 'http://json-server:11434/v1');
    setEnv('OBSERVATION_BATCH_SIZE', '8');
    const config = loadConfig();
    expect(config.observation.batchSize).toBe(8);
  });
});

// ── 5. Edge cases ─────────────────────────────────────────────────────

describe('Edge cases', () => {
  it('falls back to JSON value when PORT is non-numeric', () => {
    mockJsonFile({
      proxy: {
        inference_server_url: 'http://json-server:11434/v1',
        inference_server_key: 'json-key',
        port: 7777,
        memory_lib_url: 'http://127.0.0.1:8766',
        cors_origin: '*',
        memory_injection: { enabled: true, top_k: 10, time_weighted: false },
        observation: { enabled: true, batch_size: 1 },
      },
    });
    setEnv('INFERENCE_SERVER_URL', 'http://json-server:11434/v1');
    setEnv('PORT', 'not-a-number');
    const config = loadConfig();
    expect(config.port).toBe(7777);
  });

  it('falls back to default when PORT is empty string', () => {
    mockJsonFile(null);
    setEnv('INFERENCE_SERVER_URL', 'http://json-server:11434/v1');
    setEnv('PORT', '');
    const config = loadConfig();
    expect(config.port).toBe(8767);
  });

  it('MEMORY_INJECTION_TIME_WEIGHTED with non-true value stays false', () => {
    mockJsonFile({
      proxy: {
        inference_server_url: 'http://json-server:11434/v1',
        inference_server_key: 'json-key',
        port: 8767,
        memory_lib_url: 'http://127.0.0.1:8766',
        cors_origin: '*',
        memory_injection: { enabled: true, top_k: 10, time_weighted: false },
        observation: { enabled: true, batch_size: 1 },
      },
    });
    setEnv('INFERENCE_SERVER_URL', 'http://json-server:11434/v1');
    setEnv('MEMORY_INJECTION_TIME_WEIGHTED', 'yes');
    const config = loadConfig();
    expect(config.memoryInjection.timeWeighted).toBe(false);
  });
});
