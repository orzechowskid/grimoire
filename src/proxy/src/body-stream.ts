// SPDX-License-Identifier: MIT

/**
 * Streaming JSON metadata extractor for chat completion requests.
 *
 * Extracts `model`, `stream`, and the last user message content from
 * a chat completion request body *without* fully buffering the body.
 *
 * This module uses a state-machine JSON tokenizer that reads chunk-by-chunk
 * from a Node.js `Readable` stream, teeing data into a `PassThrough` stream
 * so the original body remains available for downstream consumers.
 */

import { Readable, PassThrough } from 'stream';

// ── Types ────────────────────────────────────────────────────────────────

/**
 * Extracted metadata from a chat completion request body.
 */
export interface ChatCompletionMetadata {
  /** The model identifier (e.g. "llama3.1:latest"). */
  model: string;
  /** Whether the request is a streaming request. */
  stream: boolean;
  /** The content of the last message with role === 'user'. */
  lastUserMessage: string | undefined;
}

// ── Sync fallback ────────────────────────────────────────────────────────

/**
 * Extract metadata from an already-parsed JSON body.
 *
 * This is a synchronous fallback for cases where the body has already been
 * parsed (e.g., Express's `req.body` in non-streaming routes).
 *
 * Handles both string content and array-of-content-parts formats used by
 * newer API variants.
 *
 * @param body - The parsed JSON body (typically an object).
 * @returns Extracted metadata, with defaults for missing fields.
 */
export function extractMetadataFromBody(body: unknown): ChatCompletionMetadata {
  if (body == null || typeof body !== 'object' || Array.isArray(body)) {
    return { model: '', stream: false, lastUserMessage: undefined };
  }

  const obj = body as Record<string, unknown>;

  const model = typeof obj.model === 'string' ? obj.model : '';
  const stream = obj.stream === true;

  // Find the last user message
  const messages = obj.messages as unknown[] | undefined;
  let lastUserMessage: string | undefined = undefined;

  if (Array.isArray(messages)) {
    for (let i = messages.length - 1; i >= 0; i--) {
      const msg = messages[i];
      if (msg != null && typeof msg === 'object' && !Array.isArray(msg)) {
        const record = msg as Record<string, unknown>;
        if (record.role === 'user') {
          lastUserMessage = extractUserMessageContent(record.content);
          break;
        }
      }
    }
  }

  return { model, stream, lastUserMessage };
}

/**
 * Extract user message content from either a string or an array of parts.
 *
 * @param content - The content field from a message object.
 * @returns The extracted content string, or undefined.
 */
function extractUserMessageContent(content: unknown): string | undefined {
  if (typeof content === 'string') {
    return content;
  }
  if (Array.isArray(content)) {
    const parts: string[] = [];
    for (const part of content) {
      if (part != null && typeof part === 'object' && !Array.isArray(part)) {
        const record = part as Record<string, unknown>;
        if (typeof record.text === 'string' && record.text.length > 0) {
          parts.push(record.text);
        }
      }
    }
    return parts.length > 0 ? parts.join('') : undefined;
  }
  return undefined;
}

// ── Streaming tokenizer constants ────────────────────────────────────────

/**
 * Byte constants for JSON token characters.
 */
const C_QUOTE = 34;       // "
const C_BS = 92;          // \
const C_LBRACE = 123;     // {
const C_RBRACE = 125;     // }
const C_LBRACKET = 91;    // [
const C_RBRACKET = 93;    // ]
const C_COLON = 58;       // :
const C_COMMA = 44;       // ,
const C_N = 110;          // n (null)
const C_T = 116;          // t (true)
const C_F = 102;          // f (false)
const C_0 = 48;           // 0
const C_9 = 57;           // 9
const C_DASH = 45;        // -

// ── Tokenizer state ──────────────────────────────────────────────────────

/**
 * State for the streaming JSON tokenizer.
 *
 * The tokenizer uses a "key path" stack model:
 * - `keyPath` is an array where keyPath[D] holds the key name that opened
 *   depth D. For the root object, keyPath[0] = null.
 * - When we encounter a string token at depth D:
 *   - If keyPath[D] is null → this string is a KEY. Set keyPath[D] = string.
 *   - If keyPath[D] is non-null → this string is a VALUE. Use keyPath[D]
 *     to determine which metadata to extract, then set keyPath[D] = null.
 * - `{` pushes the current string (as the key) and initializes a new depth.
 * - `}` pops the stack.
 */
interface TokenizerState {
  /** Stack of keys at each nesting depth. Null entries mean "expecting a key". */
  keyPath: (string | null)[];
  /** Buffer for accumulating key or value string characters. */
  buf: string;
  /** Whether we are currently inside a JSON string literal. */
  inString: boolean;
  /** Whether the previous character inside a string was a backslash. */
  escapeNext: boolean;
  /** Whether we just saw the "messages" key and are expecting the array bracket. */
  sawMessagesKey: boolean;
  /** Nesting depth where the messages array begins. -1 means not in messages array. */
  messagesDepth: number;
  /** Content tracking for the current message in the messages array. */
  /** Role of the current message object. */
  msgRole: string | null;
  /** Whether we are currently processing the content value of a user message. */
  inContentValue: boolean;
  /** Content format: 'none', 'string' (direct string), or 'array' (array of parts). */
  contentKind: 'none' | 'string' | 'array';
  /** For string content: the accumulated content. */
  contentString: string;
  /** For array content: whether we're currently inside a "text" key. */
  inPartText: boolean;
  /** For array content: accumulated text from part.text values. */
  contentArray: string;
  /** Whether we're currently inside an array of content parts. */
  inContentArray: boolean;
  /** Depth at which the content array begins. */
  contentArrayDepth: number;
}

function initState(): TokenizerState {
  return {
    keyPath: [null],
    buf: '',
    inString: false,
    escapeNext: false,
    sawMessagesKey: false,
    messagesDepth: -1,
    msgRole: null,
    inContentValue: false,
    contentKind: 'none',
    contentString: '',
    inPartText: false,
    contentArray: '',
    inContentArray: false,
    contentArrayDepth: 0,
  };
}

/**
 * Accumulator for metadata extracted during streaming.
 */
interface MetadataAcc {
  model: string | null;
  stream: boolean | null;
  lastUserMessage: string | null;
}

function createAcc(): MetadataAcc {
  return { model: null, stream: null, lastUserMessage: null };
}

function toMetadata(acc: MetadataAcc): ChatCompletionMetadata {
  return {
    model: acc.model ?? '',
    stream: acc.stream ?? false,
    lastUserMessage: acc.lastUserMessage ?? undefined,
  };
}

/**
 * Flush accumulated content for the current user message into the accumulator.
 */
function flushUserContent(state: TokenizerState, acc: MetadataAcc): void {
  let content: string;
  if (state.contentKind === 'string') {
    content = state.contentString;
  } else if (state.contentKind === 'array') {
    content = state.contentArray;
  } else {
    return;
  }

  if (content.length === 0) {
    return;
  }

  if (acc.lastUserMessage === null) {
    acc.lastUserMessage = content;
  } else {
    acc.lastUserMessage += content;
  }
}

/**
 * Process a byte chunk through the state-machine JSON tokenizer.
 */
function tokenizeChunk(state: TokenizerState, acc: MetadataAcc, chunk: Uint8Array): void {
  let i = 0;
  const len = chunk.length;

  while (i < len) {
    const c = chunk[i];
    i++;

    // ── Inside a JSON string ───────────────────────────────────────
    if (state.inString) {
      if (state.escapeNext) {
        state.escapeNext = false;
        continue;
      }
      if (c === C_BS) {
        state.escapeNext = true;
        continue;
      }
      if (c === C_QUOTE) {
        // String finished
        state.inString = false;
        const str = state.buf;
        state.buf = '';
        const depth = state.keyPath.length;

        // Determine if this string is a key or value
        const prevKey = depth > 0 ? state.keyPath[depth - 1] : null;

        if (prevKey === null) {
          // ── This is a KEY ───────────────────────────────────────
          state.keyPath[depth - 1] = str;

          // Special: check if this is the "messages" key at top level
          if (depth === 1 && str === 'messages') {
            state.sawMessagesKey = true;
          }
        } else {
          // ── This is a VALUE ─────────────────────────────────────
          handleStringValue(state, acc, str, prevKey, depth);
          // Reset key position for this depth
          state.keyPath[depth - 1] = null;
        }
        continue;
      }

      state.buf += String.fromCharCode(c);
      continue;
    }

    // ── Not inside a string ────────────────────────────────────────

    // Whitespace — skip
    if (c === 0x20 || c === 0x09 || c === 0x0A || c === 0x0D) {
      continue;
    }

    // ── Opening brace { ───────────────────────────────────────────
    if (c === C_LBRACE) {
      // Push the current buf as the key for the new depth
      const openKey = state.buf.length > 0 ? state.buf : null;
      state.buf = '';
      state.keyPath.push(openKey);
      continue;
    }

    // ── Closing brace } ───────────────────────────────────────────
    if (c === C_RBRACE) {
      if (state.keyPath.length > 0) {
        state.keyPath.pop();
      }

      // If we just finished a message object inside the messages array
      if (
        state.messagesDepth >= 0 &&
        state.keyPath.length === state.messagesDepth + 1
      ) {
        // We've exited a message object. Flush content if needed.
        if (state.msgRole === 'user' && state.inContentValue) {
          flushUserContent(state, acc);
        }
        // Reset message state
        state.msgRole = null;
        state.inContentValue = false;
        state.contentKind = 'none';
        state.contentString = '';
        state.inPartText = false;
        state.contentArray = '';
        state.inContentArray = false;
      }
      continue;
    }

    // ── Opening bracket [ ─────────────────────────────────────────
    if (c === C_LBRACKET) {
      // Check if this is the messages array
      if (state.sawMessagesKey) {
        state.sawMessagesKey = false;
        state.messagesDepth = state.keyPath.length;
      } else if (state.inContentValue && state.contentKind === 'none') {
        // We're inside a user message's content, and it's an array
        state.contentKind = 'array';
        state.inContentArray = true;
        state.contentArrayDepth = state.keyPath.length;
        state.contentArray = '';
      }
      continue;
    }

    // ── Closing bracket ] ─────────────────────────────────────────
    if (c === C_RBRACKET) {
      // Check if this closes the content array
      if (
        state.inContentArray &&
        state.keyPath.length - 1 === state.contentArrayDepth
      ) {
        state.inContentArray = false;
        state.inPartText = false;
      }
      // Check if this closes the messages array
      else if (
        state.messagesDepth >= 0 &&
        state.keyPath.length - 1 === state.messagesDepth
      ) {
        state.messagesDepth = -1;
        state.msgRole = null;
        state.inContentValue = false;
        state.contentKind = 'none';
        state.contentString = '';
        state.inPartText = false;
        state.contentArray = '';
        state.inContentArray = false;
      }
      continue;
    }

    // ── Colon : ───────────────────────────────────────────────────
    if (c === C_COLON) {
      // Check if the next value after this colon is "content" for a user message.
      // We can't know yet; the detection happens when we see the next string or [.
      continue;
    }

    // ── Comma , ───────────────────────────────────────────────────
    if (c === C_COMMA) {
      // Reset partial content text tracking
      if (state.inPartText && state.inContentValue && state.contentKind === 'array') {
        state.inPartText = false;
        state.buf = '';
      }
      continue;
    }

    // ── String " ──────────────────────────────────────────────────
    if (c === C_QUOTE) {
      // Handle content array text tracking
      if (state.inContentValue && state.contentKind === 'array' && state.inContentArray) {
        // If we're inside the content array and just finished a key like "text"
        // The value will be accumulated in buf, then processed in the string handler
      }
      state.inString = true;
      state.buf = '';
      state.escapeNext = false;
      continue;
    }

    // ── Number ────────────────────────────────────────────────────
    if ((c >= C_0 && c <= C_9) || c === C_DASH) {
      while (i < len) {
        const nc = chunk[i];
        if (
          (nc >= C_0 && nc <= C_9) ||
          nc === C_DASH ||
          nc === 43 ||
          nc === 46 ||
          nc === 101 ||
          nc === 69
        ) {
          i++;
        } else {
          break;
        }
      }
      continue;
    }

    // ── null literal ──────────────────────────────────────────────
    if (c === C_N) {
      i += 3;
      continue;
    }

    // ── true literal ──────────────────────────────────────────────
    if (c === C_T) {
      i += 3;
      continue;
    }

    // ── false literal ─────────────────────────────────────────────
    if (c === C_F) {
      i += 4;
      continue;
    }

    // Unknown character — skip
    continue;
  }
}

/**
 * Handle a completed string value.
 */
function handleStringValue(
  state: TokenizerState,
  acc: MetadataAcc,
  str: string,
  parentKey: string,
  depth: number
): void {
  // ── Top-level values ────────────────────────────────────────────
  if (depth === 1) {
    if (parentKey === 'model') {
      acc.model = str;
    } else if (parentKey === 'stream') {
      acc.stream = str === 'true';
    }
  }

  // ── Inside messages array ───────────────────────────────────────
  if (state.messagesDepth >= 0 && depth === state.messagesDepth + 1) {
    if (parentKey === 'role') {
      // Update current message role
      if (state.msgRole === 'user' && state.inContentValue) {
        // Previous message was user — flush its content
        flushUserContent(state, acc);
      }
      state.msgRole = str;
      if (str === 'user') {
        // Start tracking content for this user message
        state.inContentValue = true;
        state.contentKind = 'none'; // will be determined by next token
        state.contentString = '';
        state.contentArray = '';
        state.inPartText = false;
        state.inContentArray = false;
      } else {
        state.inContentValue = false;
        state.contentKind = 'none';
      }
    } else if (parentKey === 'content') {
      if (state.msgRole === 'user') {
        state.inContentValue = true;
        // We're about to see the content value. It could be a string
        // or an array. We'll determine the kind from the next tokens.
        // For now, assume it's a string — if we later see a [ we'll switch.
        // Actually, the current token IS the content value. If it's a string,
        // kind = string. We'll handle array detection in tokenizeChunk.
        state.contentKind = 'none';
        state.contentString = '';
        state.contentArray = '';
      }
    }
  }

  // ── Content string accumulation ─────────────────────────────────
  if (state.inContentValue && state.contentKind === 'string' && parentKey === 'content') {
    state.contentString = str;
  }

  // ── Content array part text accumulation ────────────────────────
  if (
    state.inContentValue &&
    state.contentKind === 'array' &&
    parentKey === 'text' &&
    state.inPartText
  ) {
    if (state.contentArray.length === 0) {
      state.contentArray = str;
    } else {
      state.contentArray += str;
    }
  }
}

/**
 * Post-process: handle content array detection after seeing a colon.
 * This is called internally when we detect content: and need to determine
 * whether the next token starts a string or an array.
 */
function checkContentBeforeToken(state: TokenizerState, tokenType: 'string' | 'bracket'): void {
  if (state.inContentValue && state.contentKind === 'none') {
    if (tokenType === 'bracket') {
      state.contentKind = 'array';
      state.inContentArray = true;
      state.contentArrayDepth = state.keyPath.length;
      state.contentArray = '';
    } else {
      state.contentKind = 'string';
      state.contentString = '';
    }
  }
}

// ── End-of-stream finalization ──────────────────────────────────────────

/**
 * Process the end-of-stream signal. Finalizes any pending content.
 */
function finalizeState(state: TokenizerState, acc: MetadataAcc): void {
  if (state.inContentValue && state.contentKind !== 'none') {
    flushUserContent(state, acc);
  }
}

// ── Public API ───────────────────────────────────────────────────────────

/**
 * Streaming JSON metadata extractor for chat completion requests.
 *
 * Reads from the input `reqStream` chunk-by-chunk, using an incremental
 * JSON tokenizer to extract `model`, `stream`, and the last user message
 * content without buffering the entire body.
 *
 * Data is teed through a `PassThrough` stream so the original body
 * remains available for downstream consumers.
 *
 * @param reqStream - The readable stream containing the request body.
 * @returns A promise resolving to an object containing the extracted
 *          `metadata` and a `bodyStream` that replays the original body.
 */
export async function extractChatCompletionMetadata(
  reqStream: Readable
): Promise<{ metadata: ChatCompletionMetadata; bodyStream: Readable }> {
  const passthrough = new PassThrough();
  const state = initState();
  const acc = createAcc();

  let resolved = false;

  return new Promise((resolve) => {
    reqStream.on('data', (chunk: Uint8Array | string) => {
      let buf: Uint8Array;
      if (typeof chunk === 'string') {
        buf = new TextEncoder().encode(chunk);
      } else {
        buf = chunk;
      }

      passthrough.write(buf);
      tokenizeChunk(state, acc, buf);
    });

    reqStream.on('end', () => {
      finalizeState(state, acc);
      passthrough.end();

      if (!resolved) {
        resolved = true;
        resolve({
          metadata: toMetadata(acc),
          bodyStream: passthrough,
        });
      }
    });

    reqStream.on('error', (err: Error) => {
      passthrough.destroy(err);

      if (!resolved) {
        resolved = true;
        resolve({
          metadata: { model: '', stream: false, lastUserMessage: undefined },
          bodyStream: passthrough,
        });
      }
    });
  });
}
