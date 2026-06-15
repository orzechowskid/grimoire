// SPDX-License-Identifier: MIT

import { describe, it, expect } from 'vitest';

// ── Minimal type definitions for InjectResponse ──────────────────────────

interface InjectResult {
  session_id: string;
  brief: string;
  tags: string[];
  importance: string;
  score: number;
  similarity: number;
  key_facts?: string[];
}

interface IntuitionSignal {
  type: string;
  tag: string;
  message: string;
}

interface InjectResponse {
  results: InjectResult[];
  total_candidates: number;
  warning?: string;
  intuition_signals?: IntuitionSignal[];
}

// ── The function under test (copied from proxy.ts for unit-test isolation) ─

function formatMemoryContext(result: InjectResponse): string {
  if (!result.results || result.results.length === 0) {
    return '### MEMORY CONTEXT\nNo relevant memories found.';
  }

  const lines: string[] = [];

  // Build intuition alerts block if signals present
  if (result.intuition_signals && result.intuition_signals.length > 0) {
    const emojiMap: Record<string, string> = {
      TENSION: '⚠ TENSION',
      DO_THIS: '💡 RECOMMENDATION',
      AVOID_THIS: '🚫 AVOID',
      ATTRACT: '❤ ATTRACT',
      REPEL: '💔 REPEL',
      AMBIVALENT: '⚖ AMBIVALENT',
    };
    lines.push('### COGNITIVE INTUITION ALERTS');
    for (const sig of result.intuition_signals) {
      const prefix = emojiMap[sig.type] || sig.type;
      lines.push(`${prefix}: ${sig.message}`);
    }
    lines.push(''); // blank line separator
  }

  // Importance display order
  const importanceOrder: string[] = ['CRITICAL', 'PRINCIPLE', 'IMPORTANT', 'BACKGROUND'];

  // Group results by importance (uppercased)
  const groups: Map<string, InjectResult[]> = new Map();
  for (const r of result.results) {
    const key = (r.importance || 'BACKGROUND').toUpperCase();
    if (!groups.has(key)) {
      groups.set(key, []);
    }
    groups.get(key)!.push(r);
  }

  lines.push('### MEMORY CONTEXT');
  lines.push(`Retrieved ${result.results.length} relevant memor${result.results.length > 1 ? 'ies' : 'y'}:`);

  // Render groups in priority order
  for (const importance of importanceOrder) {
    const group = groups.get(importance);
    if (!group || group.length === 0) {
      continue;
    }

    lines.push('');
    lines.push(`**${importance} (${group.length})**`);

    for (const r of group) {
      let bullet = `• ${r.brief}`;
      if (r.tags && r.tags.length > 0) {
        bullet += ` [${r.tags.join(', ')}]`;
      }
      lines.push(bullet);
      // Render key facts as sub-bullets
      if (r.key_facts && r.key_facts.length > 0) {
        for (const fact of r.key_facts) {
          lines.push(`  ↳ Fact: ${fact}`);
        }
      }
    }
  }

  // Collect unique tags for summary
  const allTags = new Set<string>();
  for (const r of result.results) {
    if (r.tags) {
      for (const tag of r.tags) {
        allTags.add(tag);
      }
    }
  }

  if (allTags.size > 0) {
    lines.push('');
    lines.push(`Related tags: ${Array.from(allTags).join(', ')}`);
  }

  return lines.join('\n');
}

// ── Tests ────────────────────────────────────────────────────────────────

describe('formatMemoryContext', () => {

  it('returns header with "No relevant memories found" for empty results', () => {
    const input: InjectResponse = { results: [], total_candidates: 0 };
    const output = formatMemoryContext(input);
    expect(output).toBe('### MEMORY CONTEXT\nNo relevant memories found.');
  });

  it('formats a single result as a bullet point with tags', () => {
    const input: InjectResponse = {
      results: [
        {
          session_id: 'sess-001',
          brief: 'Replaced axios with native fetch in http.ts',
          tags: ['http', 'proxy', 'refactoring'],
          importance: 'critical',
          score: 0.95,
          similarity: 0.98,
        },
      ],
      total_candidates: 1,
    };
    const output = formatMemoryContext(input);

    expect(output).toContain('### MEMORY CONTEXT');
    expect(output).toContain('Retrieved 1 relevant memory:');
    expect(output).toContain('**CRITICAL (1)**');
    expect(output).toContain('• Replaced axios with native fetch in http.ts');
    expect(output).toContain('[http, proxy, refactoring]');
    // Session IDs should NOT appear
    expect(output).not.toContain('sess-001');
    // Similarity should NOT appear
    expect(output).not.toContain('similarity');
  });

  it('groups results by importance level', () => {
    const input: InjectResponse = {
      results: [
        { session_id: 's1', brief: 'Critical security fix', tags: ['security'], importance: 'critical', score: 0.9, similarity: 0.9 },
        { session_id: 's2', brief: 'Important design change', tags: ['design'], importance: 'important', score: 0.7, similarity: 0.7 },
        { session_id: 's3', brief: 'Background noise', tags: ['chitchat'], importance: 'background', score: 0.3, similarity: 0.3 },
        { session_id: 's4', brief: 'Another important item', tags: ['code'], importance: 'important', score: 0.6, similarity: 0.6 },
      ],
      total_candidates: 4,
    };
    const output = formatMemoryContext(input);

    expect(output).toContain('**CRITICAL (1)**');
    expect(output).toContain('**IMPORTANT (2)**');
    expect(output).toContain('**BACKGROUND (1)**');
    // CRITICAL should appear before IMPORTANT
    expect(output.indexOf('**CRITICAL')).toBeLessThan(output.indexOf('**IMPORTANT'));
    expect(output.indexOf('**IMPORTANT')).toBeLessThan(output.indexOf('**BACKGROUND'));
  });

  it('includes tags in bullet points', () => {
    const input: InjectResponse = {
      results: [
        { session_id: 's1', brief: 'Test result', tags: ['tag1', 'tag2'], importance: 'important', score: 0.5, similarity: 0.5 },
      ],
      total_candidates: 1,
    };
    const output = formatMemoryContext(input);
    expect(output).toContain('[tag1, tag2]');
  });

  it('handles results with no tags gracefully', () => {
    const input: InjectResponse = {
      results: [
        { session_id: 's1', brief: 'Test result', tags: [], importance: 'important', score: 0.5, similarity: 0.5 },
      ],
      total_candidates: 1,
    };
    const output = formatMemoryContext(input);
    expect(output).toContain('• Test result');
    expect(output).not.toContain('[]');
  });

  it('omits session ids from output', () => {
    const input: InjectResponse = {
      results: [
        { session_id: 'session-abc-123', brief: 'Some memory', tags: [], importance: 'important', score: 0.5, similarity: 0.5 },
      ],
      total_candidates: 1,
    };
    const output = formatMemoryContext(input);
    expect(output).not.toContain('session-abc-123');
  });

  it('generates related tags summary at bottom', () => {
    const input: InjectResponse = {
      results: [
        { session_id: 's1', brief: 'Memory one', tags: ['http', 'proxy'], importance: 'critical', score: 0.9, similarity: 0.9 },
        { session_id: 's2', brief: 'Memory two', tags: ['config', 'proxy'], importance: 'important', score: 0.7, similarity: 0.7 },
      ],
      total_candidates: 2,
    };
    const output = formatMemoryContext(input);
    expect(output).toContain('Related tags:');
    expect(output).toContain('http');
    expect(output).toContain('proxy');
    expect(output).toContain('config');
  });

  it('uses singular "memory" for single result', () => {
    const input: InjectResponse = {
      results: [{ session_id: 's1', brief: 'Test', tags: [], importance: 'important', score: 0.5, similarity: 0.5 }],
      total_candidates: 1,
    };
    const output = formatMemoryContext(input);
    expect(output).toContain('Retrieved 1 relevant memory:');
  });

  it('uses plural "memories" for multiple results', () => {
    const input: InjectResponse = {
      results: [
        { session_id: 's1', brief: 'One', tags: [], importance: 'important', score: 0.5, similarity: 0.5 },
        { session_id: 's2', brief: 'Two', tags: [], importance: 'important', score: 0.5, similarity: 0.5 },
      ],
      total_candidates: 2,
    };
    const output = formatMemoryContext(input);
    expect(output).toContain('Retrieved 2 relevant memories:');
  });

  it('handles empty importance by defaulting to BACKGROUND', () => {
    const input: InjectResponse = {
      results: [
        { session_id: 's1', brief: 'No importance', tags: [], importance: '', score: 0.3, similarity: 0.3 },
      ],
      total_candidates: 1,
    };
    const output = formatMemoryContext(input);
    expect(output).toContain('**BACKGROUND (1)**');
  });

  it('does not include related tags line when no tags exist', () => {
    const input: InjectResponse = {
      results: [
        { session_id: 's1', brief: 'Test', tags: [], importance: 'important', score: 0.5, similarity: 0.5 },
        { session_id: 's2', brief: 'Test2', tags: [], importance: 'important', score: 0.5, similarity: 0.5 },
      ],
      total_candidates: 2,
    };
    const output = formatMemoryContext(input);
    expect(output).not.toContain('Related tags:');
  });

  it('handles missing results array', () => {
    const input = { results: null as unknown as InjectResult[], total_candidates: 0 };
    const output = formatMemoryContext(input);
    expect(output).toBe('### MEMORY CONTEXT\nNo relevant memories found.');
  });

  it('renders COGNITIVE INTUITION ALERTS block when intuition_signals present', () => {
    const input: InjectResponse = {
      results: [
        { session_id: 's1', brief: 'Test memory', tags: ['coding'], importance: 'important', score: 0.5, similarity: 0.5 },
      ],
      total_candidates: 1,
      intuition_signals: [
        { type: 'TENSION', tag: 'coding', message: "Topic 'coding' has unresolved contradictions (3 conflicts)." },
      ],
    };
    const output = formatMemoryContext(input);
    expect(output).toContain('### COGNITIVE INTUITION ALERTS');
    expect(output).toContain('⚠ TENSION:');
    expect(output).toContain("Topic 'coding' has unresolved contradictions (3 conflicts).");
    // Alerts should appear before MEMORY CONTEXT
    expect(output.indexOf('### COGNITIVE INTUITION ALERTS')).toBeLessThan(output.indexOf('### MEMORY CONTEXT'));
  });

  it('renders key_facts as sub-bullets under brief', () => {
    const input: InjectResponse = {
      results: [
        {
          session_id: 's1',
          brief: 'Refactored the database schema',
          tags: ['database', 'sqlite'],
          importance: 'critical',
          score: 0.95,
          similarity: 0.98,
          key_facts: [
            'Replaced standard sync with aiosqlite batched queues.',
            'Set journal mode to WAL for concurrent read support.',
          ],
        },
      ],
      total_candidates: 1,
    };
    const output = formatMemoryContext(input);
    expect(output).toContain('• Refactored the database schema');
    expect(output).toContain('  ↳ Fact: Replaced standard sync with aiosqlite batched queues.');
    expect(output).toContain('  ↳ Fact: Set journal mode to WAL for concurrent read support.');
  });

  it('renders both intuition alerts and key facts together', () => {
    const input: InjectResponse = {
      results: [
        {
          session_id: 's1',
          brief: 'Refactored the database schema',
          tags: ['database'],
          importance: 'critical',
          score: 0.95,
          similarity: 0.98,
          key_facts: ['Replaced standard sync with aiosqlite.'],
        },
      ],
      total_candidates: 1,
      intuition_signals: [
        { type: 'DO_THIS', tag: 'cli-tools', message: "'cli-tools' is a verified pattern (6 sessions, score 0.88)." },
      ],
    };
    const output = formatMemoryContext(input);
    expect(output).toContain('### COGNITIVE INTUITION ALERTS');
    expect(output).toContain('💡 RECOMMENDATION:');
    expect(output).toContain('### MEMORY CONTEXT');
    expect(output).toContain('↳ Fact: Replaced standard sync with aiosqlite.');
  });

  it('omits intuition block when no signals present', () => {
    const input: InjectResponse = {
      results: [
        { session_id: 's1', brief: 'Test memory', tags: ['coding'], importance: 'important', score: 0.5, similarity: 0.5 },
      ],
      total_candidates: 1,
      intuition_signals: [],
    };
    const output = formatMemoryContext(input);
    expect(output).not.toContain('### COGNITIVE INTUITION ALERTS');
    expect(output).toContain('### MEMORY CONTEXT');
  });

  it('handles multiple signal types with correct emoji prefixes', () => {
    const input: InjectResponse = {
      results: [
        { session_id: 's0', brief: 'Placeholder', tags: [], importance: 'background', score: 0, similarity: 0 },
      ],
      total_candidates: 0,
      intuition_signals: [
        { type: 'TENSION', tag: 'a', message: 'tension msg' },
        { type: 'DO_THIS', tag: 'b', message: 'do this msg' },
        { type: 'AVOID_THIS', tag: 'c', message: 'avoid msg' },
        { type: 'ATTRACT', tag: 'd', message: 'attract msg' },
        { type: 'REPEL', tag: 'e', message: 'repel msg' },
        { type: 'AMBIVALENT', tag: 'f', message: 'ambivalent msg' },
      ],
    };
    const output = formatMemoryContext(input);
    expect(output).toContain('⚠ TENSION: tension msg');
    expect(output).toContain('💡 RECOMMENDATION: do this msg');
    expect(output).toContain('🚫 AVOID: avoid msg');
    expect(output).toContain('❤ ATTRACT: attract msg');
    expect(output).toContain('💔 REPEL: repel msg');
    expect(output).toContain('⚖ AMBIVALENT: ambivalent msg');
  });
});
