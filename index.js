import { readFileSync, existsSync } from 'fs';
import http from 'http';
import crypto from 'crypto';
import https from 'https';

function loadEnv() {
  const envPath = new URL('.env', import.meta.url).pathname;
  if (!existsSync(envPath)) return;
  let lines;
  try {
    lines = readFileSync(envPath, 'utf-8').split('\n');
  } catch {
    return;
  }
  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith('#')) continue;
    const eq = trimmed.indexOf('=');
    if (eq === -1) continue;
    const key = trimmed.slice(0, eq).trim();
    let value = trimmed.slice(eq + 1).trim();
    // Strip inline comments (hash outside quotes)
    const hashIdx = value.indexOf('#');
    if (hashIdx >= 0) {
      value = value.slice(0, hashIdx).trim();
    }
    if ((value.startsWith('"') && value.endsWith('"')) || (value.startsWith("'") && value.endsWith("'"))) {
      value = value.slice(1, -1);
    }
    if (!(key in process.env)) process.env[key] = value;
  }
}
loadEnv();

const config = {
  port: parseInt(process.env.PORT, 10) || 6767,
  host: process.env.HOST || '0.0.0.0',
  apiKey: process.env.API_KEY || '',
  chatjimmyUrl: process.env.CHATJIMMY_URL || 'https://chatjimmy.ai',
  chatjimmyTimeout: parseInt(process.env.CHATJIMMY_TIMEOUT, 10) || 30000,
  maxContextTokens: parseInt(process.env.MAX_CONTEXT_TOKENS, 10) || 5000,
  maxBodySize: parseInt(process.env.MAX_BODY_SIZE, 10) || 1048576,
};

const CORS_HEADERS = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type, Authorization',
};

const REASONING_SYSTEM_PROMPT = `You MUST follow this format exactly for every response:

<thinking>
Your step-by-step reasoning here. Think through the problem before answering.
</thinking>
<answer>
Your final answer here. This is what the user will see. Be direct and concise.
</answer>

Both tags are required. Never skip the <answer> tag.`;

function injectXMLReasoning(messages) {
  if (!messages || messages.length === 0) return [];
  const hasSystem = messages[0].role === 'system';
  if (hasSystem) {
    const [first, ...rest] = messages;
    return [{ role: 'system', content: first.content + '\n\n' + REASONING_SYSTEM_PROMPT }, ...rest];
  }
  return [{ role: 'system', content: REASONING_SYSTEM_PROMPT }, ...messages];
}

function charCount(obj) {
  if (typeof obj === 'string') return obj.length;
  if (Array.isArray(obj)) {
    if (obj.length === 0) return 0;
    if (obj[0] && obj[0].type === 'text') return obj.map(p => (p.text || '').length).reduce((a, b) => a + b, 0);
    return JSON.stringify(obj).length;
  }
  return JSON.stringify(obj).length;
}

function estimateTokens(messages) {
  let total = 0;
  for (const msg of messages) {
    total += charCount(msg.content);
  }
  return Math.ceil(total / 4);
}

function trimMessages(messages, maxTokens) {
  const systemMsgs = messages.filter(m => m.role === 'system');
  const otherMsgs = messages.filter(m => m.role !== 'system');

  let total = estimateTokens(messages);
  if (total <= maxTokens) return;

  while (otherMsgs.length > 2) {
    total = estimateTokens([...systemMsgs, ...otherMsgs]);
    if (total <= maxTokens) break;
    otherMsgs.shift();
  }

  if (otherMsgs.length < messages.length - systemMsgs.length) {
    const lastSystem = systemMsgs.length > 0 ? systemMsgs[systemMsgs.length - 1] : null;
    if (lastSystem) {
      const marker = '\n\n[Earlier messages were trimmed to fit context window.]';
      systemMsgs[systemMsgs.length - 1] = { role: 'system', content: lastSystem.content + marker };
    } else {
      systemMsgs.push({ role: 'system', content: '[Earlier messages were trimmed to fit context window.]' });
    }
  }

  messages.length = 0;
  messages.push(...systemMsgs, ...otherMsgs);
}

function callChatjimmy(messages) {
  const systemMsg = messages.find(m => m.role === 'system');
  const systemPrompt = systemMsg ? (typeof systemMsg.content === 'string' ? systemMsg.content : JSON.stringify(systemMsg.content)) : '';

  const payload = JSON.stringify({
    messages,
    chatOptions: {
      selectedModel: 'llama3.1-8B',
      systemPrompt: systemPrompt,
      topK: 8,
    },
    attachment: null,
  });

  const url = new URL('/api/chat', config.chatjimmyUrl);
  const options = {
    hostname: url.hostname,
    port: url.port || 443,
    path: url.pathname,
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Content-Length': Buffer.byteLength(payload),
    },
    timeout: config.chatjimmyTimeout,
  };

  return new Promise((resolve) => {
    const req = https.request(options, (res) => {
      const chunks = [];
      res.on('data', c => chunks.push(c));
      res.on('end', () => {
        const body = Buffer.concat(chunks).toString('utf-8');
        if (res.statusCode >= 400) {
          resolve({ error: true, status: res.statusCode, body });
        } else {
          resolve({ error: false, status: res.statusCode, body });
        }
      });
    });

    req.on('error', (err) => {
      const status = err.code === 'ENOTFOUND' || err.code === 'ECONNREFUSED' ? 502 : 504;
      resolve({ error: true, status, body: err.message });
    });
    req.on('timeout', () => {
      req.destroy();
      resolve({ error: true, status: 504, body: 'Upstream timeout' });
    });

    req.write(payload);
    req.end();
  });
}

function parseResponse(text) {
  const statsMatch = text.match(/<\|stats\|>([\s\S]*?)<\|\/stats\|>/s);
  let cleaned = text;
  let stats = null;

  if (statsMatch) {
    cleaned = text.slice(0, text.indexOf('<|stats|>')).trim();
    try {
      stats = JSON.parse(statsMatch[1]);
    } catch (_) { /* ignore malformed stats */ }
  }

  const thinkingTag = cleaned.match(/<thinking>([\s\S]*?)<\/thinking>/i);
  const answerTag = cleaned.match(/<answer>([\s\S]*?)(?:<\/answer>|$)/i);

  let reasoningContent = null;
  let content = cleaned;

  if (answerTag) {
    content = answerTag[1].trim();
    const beforeAnswer = cleaned.slice(0, cleaned.indexOf(answerTag[0]));
    reasoningContent = beforeAnswer.replace(/<thinking>[\s\S]*?<\/thinking>/gi, '').trim();
    if (!reasoningContent && thinkingTag) {
      reasoningContent = thinkingTag[1].trim();
    }
  } else if (thinkingTag) {
    reasoningContent = thinkingTag[1].trim();
    content = cleaned.replace(thinkingTag[0], '').trim();
  }

  return { reasoningContent, content, stats };
}

function timingSafeStringCompare(a, b) {
  const maxLen = Math.max(a.length, b.length);
  const aBuf = Buffer.from(a.padEnd(maxLen, '\0'), 'utf-8');
  const bBuf = Buffer.from(b.padEnd(maxLen, '\0'), 'utf-8');
  if (aBuf.length !== bBuf.length) return false;
  return crypto.timingSafeEqual(aBuf, bBuf);
}

function json(res, status, body) {
  const data = JSON.stringify(body);
  res.writeHead(status, {
    'Content-Type': 'application/json',
    ...CORS_HEADERS,
    'Cache-Control': 'no-store',
  });
  res.end(data);
}

function readBody(req) {
  return new Promise((resolve) => {
    const chunks = [];
    let size = 0;
    let settled = false;
    const timeout = setTimeout(() => {
      if (!settled) { settled = true; req.destroy(); resolve({ ok: false, error: 'timeout' }); }
    }, 10000);
    const cleanup = () => { clearTimeout(timeout); };
    req.on('data', c => {
      size += c.length;
      if (size > config.maxBodySize) {
        if (!settled) { settled = true; cleanup(); req.destroy(); resolve({ ok: false, error: 'too_large' }); }
        return;
      }
      chunks.push(c);
    });
    req.on('end', () => {
      if (!settled) { settled = true; cleanup(); resolve({ ok: true, body: Buffer.concat(chunks).toString('utf-8') }); }
    });
     req.on('error', (_err) => {
      if (!settled) { settled = true; cleanup(); resolve({ ok: false, error: 'read_error' }); }
    });
  });
}

async function parseJsonBody(req) {
  const result = await readBody(req);
  if (!result.ok) {
    if (result.error === 'too_large') throw Object.assign(new Error('Request body too large'), { code: 'too_large' });
    throw new Error('Failed to read request body');
  }
  const body = result.body;
  let depth = 0, maxDepth = 0, inString = false, escaped = false;
  for (const ch of body) {
    if (inString) { if (escaped) { escaped = false; continue; } if (ch === '\\') { escaped = true; continue; } if (ch === '"') { inString = false; } continue; }
    if (ch === '"') { inString = true; continue; }
    if (ch === '{' || ch === '[') { depth++; if (depth > maxDepth) maxDepth = depth; }
    else if (ch === '}' || ch === ']') depth--;
  }
  if (maxDepth > 50) throw new Error('Request body too deeply nested');
  return JSON.parse(body);
}

function extractToken(req) {
  const auth = String(req.headers['authorization'] || '').trim();
  const m = auth.match(/^Bearer\s+(.+)$/i);
  if (m) return m[1].trim();
  return '';
}

function validateApiKey(token) {
  if (!config.apiKey) return true;
  return timingSafeStringCompare(String(token || ''), String(config.apiKey));
}

async function handleRequest(req, res) {
  const { method } = req;
  const parsedUrl = new URL(req.url, 'http://localhost');
  const path = parsedUrl.pathname;

  if (method === 'OPTIONS') {
    res.writeHead(204, { ...CORS_HEADERS });
    return res.end();
  }

  if (path === '/health') {
    return json(res, 200, { status: 'ok', uptime: Math.round(process.uptime()), model: 'jimmy/llama3.1-8B' });
  }

  if (path.startsWith('/v1/')) {
    if (!validateApiKey(extractToken(req))) {
      return json(res, 401, { error: { message: 'Invalid API key', type: 'auth_error', code: 'invalid_api_key' } });
    }
  }

  if (path === '/v1/models' && method === 'GET') {
    return json(res, 200, {
      object: 'list',
      data: [{
        id: 'jimmy/llama3.1-8B',
        object: 'model',
        created: Math.floor(Date.now() / 1000),
        owned_by: 'jimmy',
      }],
    });
  }

  if (path === '/v1/chat/completions' && method === 'POST') {
    let body;
    try {
      body = await parseJsonBody(req);
    } catch (err) {
      if (err.code === 'too_large') {
        return json(res, 413, { error: { message: 'Request body too large', type: 'invalid_request', code: 'body_too_large' } });
      }
      return json(res, 400, { error: { message: 'Invalid JSON body', type: 'invalid_request', code: 'invalid_json' } });
    }
    if (!body) {
      return json(res, 400, { error: { message: 'Invalid JSON body', type: 'invalid_request', code: 'invalid_json' } });
    }
    const messages = body.messages || [];
    if (messages.length === 0) {
      return json(res, 400, { error: { message: 'messages array is required', type: 'invalid_request', code: 'missing_messages' } });
    }

    const injected = injectXMLReasoning(messages);
    const trimmed = [...injected];
    trimMessages(trimmed, config.maxContextTokens);

    const result = await callChatjimmy(trimmed);

    if (result.error) {
      const status = result.status || 502;
      const errType = status === 504 ? 'timeout' : 'upstream_error';
      let errBody = result.body;
      try { errBody = JSON.parse(result.body); } catch (_) { errBody = { message: String(result.body) }; }
      return json(res, status, { error: { message: errBody.message || errBody, type: errType, code: 'upstream_error' } });
    }

    const parsed = parseResponse(result.body);
    const id = 'chatcmpl-' + crypto.randomUUID().replace(/-/g, '').slice(0, 29);
    const finishReason = parsed.stats?.done_reason === 'stop' ? 'stop' :
      parsed.stats?.done_reason === 'length' ? 'length' : 'stop';

    const isStream = !!body.stream;

    if (isStream) {
      res.writeHead(200, {
        'Content-Type': 'text/event-stream',
        ...CORS_HEADERS,
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
      });
      const created = Math.floor(Date.now() / 1000);
      const chunk = (delta, finish) => {
        const obj = {
          id,
          object: 'chat.completion.chunk',
          created,
          model: 'jimmy/llama3.1-8B',
          choices: [{
            index: 0,
            delta: delta,
            finish_reason: finish,
          }],
          ...(finish && parsed.stats ? {
            usage: {
              prompt_tokens: parsed.stats.prefill_tokens || 0,
              completion_tokens: parsed.stats.decode_tokens || 0,
              total_tokens: parsed.stats.total_tokens || 0,
            },
          } : {}),
        };
        res.write(`data: ${JSON.stringify(obj)}\n\n`);
      };
      if (parsed.reasoningContent) {
        chunk({ role: 'assistant', reasoning_content: parsed.reasoningContent }, null);
      }
      chunk({ role: 'assistant', content: parsed.content || '' }, null);
      chunk({}, finishReason);
      res.write('data: [DONE]\n\n');
      return res.end();
    }

    const responseBody = {
      id,
      object: 'chat.completion',
      created: Math.floor(Date.now() / 1000),
      model: 'jimmy/llama3.1-8B',
      choices: [{
        index: 0,
        message: {
          role: 'assistant',
          content: parsed.content || '',
          ...(parsed.reasoningContent ? { reasoning_content: parsed.reasoningContent } : {}),
        },
        finish_reason: finishReason,
      }],
      usage: parsed.stats ? {
        prompt_tokens: parsed.stats.prefill_tokens || 0,
        completion_tokens: parsed.stats.decode_tokens || 0,
        total_tokens: parsed.stats.total_tokens || 0,
      } : {
        prompt_tokens: 0,
        completion_tokens: 0,
        total_tokens: 0,
      },
    };

    return json(res, 200, responseBody);
  }

  json(res, 404, { error: { message: 'Not found', type: 'invalid_request' } });
}

const server = http.createServer(handleRequest);

server.listen(config.port, config.host, () => {
  console.log(`jimmychat-proxy listening on http://${config.host}:${config.port}`);
  console.log(`Auth: ${config.apiKey ? 'enabled' : 'disabled (no API_KEY set)'}`);
});
