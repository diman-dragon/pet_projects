import http from 'http';
import fs from 'fs/promises';
import path from 'path';
import crypto from 'crypto';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const port = process.env.PORT ? parseInt(process.env.PORT, 10) : 3000;
const dataDir = path.join(__dirname, '.data');
const visitorStatsPath = path.join(dataDir, 'visitor-stats.json');
const downloadLogPath  = path.join(dataDir, 'download-log.json');

const mimeTypes = {
  '.html': 'text/html; charset=utf-8',
  '.css':  'text/css; charset=utf-8',
  '.js':   'application/javascript; charset=utf-8',
  '.mjs':  'application/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.svg':  'image/svg+xml',
  '.pdf':  'application/pdf',
  '.doc':  'application/msword',
  '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
  '.png':  'image/png',
  '.jpg':  'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.webp': 'image/webp',
};

// Cache-Control by file type
const CACHE_CONTROL = {
  '.html': 'no-cache',
  '.json': 'no-cache',
  '.js':   'no-cache',
  '.mjs':  'no-cache',
  '.css':  'no-cache',
  '.pdf':  'public, max-age=86400',
  '.png':  'public, max-age=86400',
  '.jpg':  'public, max-age=86400',
  '.jpeg': 'public, max-age=86400',
  '.webp': 'public, max-age=86400',
};

let visitorStatsCache = null;

function createEmptyVisitorStats() {
  return {
    visits: 0,
    visitors: 0,
    visitorIds: {},
    updatedAt: null,
  };
}

async function loadVisitorStats() {
  if (visitorStatsCache) return visitorStatsCache;
  try {
    const raw = await fs.readFile(visitorStatsPath, 'utf8');
    const parsed = JSON.parse(raw);
    visitorStatsCache = {
      visits: Number(parsed.visits) || 0,
      visitors: Number(parsed.visitors) || 0,
      visitorIds: parsed.visitorIds && typeof parsed.visitorIds === 'object' ? parsed.visitorIds : {},
      updatedAt: parsed.updatedAt || null,
    };
  } catch {
    visitorStatsCache = createEmptyVisitorStats();
  }
  return visitorStatsCache;
}

async function saveVisitorStats() {
  if (!visitorStatsCache) return;
  await fs.mkdir(dataDir, { recursive: true });
  await fs.writeFile(visitorStatsPath, JSON.stringify(visitorStatsCache, null, 2));
}

function getPublicVisitorStats(stats) {
  return {
    visits: stats.visits || 0,
    visitors: stats.visitors || 0,
    updatedAt: stats.updatedAt,
  };
}

async function logDownload(req, filePath) {
  const ext = path.extname(filePath).toLowerCase();
  if (ext !== '.pdf' && ext !== '.docx' && ext !== '.doc') return;

  const entry = {
    ts: new Date().toISOString(),
    file: filePath.replace(process.cwd() + path.sep, '').replace(/\\/g, '/'),
    ip: (req.headers['x-forwarded-for'] || req.socket?.remoteAddress || '').split(',')[0].trim(),
    ua: (req.headers['user-agent'] || '').slice(0, 200),
    referer: (req.headers['referer'] || req.headers['referrer'] || '').slice(0, 200),
  };

  let log = [];
  try {
    const raw = await fs.readFile(downloadLogPath, 'utf8');
    log = JSON.parse(raw);
    if (!Array.isArray(log)) log = [];
  } catch { /* first run */ }

  log.push(entry);
  // Keep last 5 000 entries
  if (log.length > 5000) log = log.slice(log.length - 5000);

  await fs.mkdir(dataDir, { recursive: true });
  await fs.writeFile(downloadLogPath, JSON.stringify(log, null, 2));
}

function sendJson(res, statusCode, payload) {
  res.writeHead(statusCode, {
    'Content-Type': 'application/json; charset=utf-8',
    'Cache-Control': 'no-store',
    'X-Content-Type-Options': 'nosniff',
  });
  res.end(JSON.stringify(payload));
}

function normaliseVisitorId(input) {
  if (typeof input !== 'string') return '';
  const trimmed = input.trim();
  if (!/^[a-zA-Z0-9_-]{8,128}$/.test(trimmed)) return '';
  return trimmed;
}

async function readJsonBody(req) {
  const chunks = [];
  for await (const chunk of req) {
    chunks.push(chunk);
  }
  if (!chunks.length) return {};
  return JSON.parse(Buffer.concat(chunks).toString('utf8'));
}

async function handleVisitorStats(req, res) {
  const stats = await loadVisitorStats();

  if (req.method === 'GET' || req.method === 'HEAD') {
    if (req.method === 'HEAD') {
      res.writeHead(200, {
        'Content-Type': 'application/json; charset=utf-8',
        'Cache-Control': 'no-store',
        'X-Content-Type-Options': 'nosniff',
      });
      res.end();
      return;
    }
    sendJson(res, 200, getPublicVisitorStats(stats));
    return;
  }

  if (req.method !== 'POST') {
    res.writeHead(405, {
      'Content-Type': 'text/plain; charset=utf-8',
      Allow: 'GET, HEAD, POST',
    });
    res.end('Method Not Allowed');
    return;
  }

  try {
    const payload = await readJsonBody(req);
    const visitorId = normaliseVisitorId(payload.visitorId);
    const pagePath = typeof payload.path === 'string' ? payload.path.slice(0, 200) : '/';
    const now = new Date().toISOString();

    stats.visits = (stats.visits || 0) + 1;

    if (visitorId) {
      const existing = stats.visitorIds[visitorId];
      const userAgentHash = crypto
        .createHash('sha256')
        .update(req.headers['user-agent'] || '')
        .digest('hex')
        .slice(0, 16);

      if (!existing) {
        stats.visitorIds[visitorId] = {
          firstSeenAt: now,
          lastSeenAt: now,
          lastPath: pagePath,
          userAgentHash,
        };
        stats.visitors = Object.keys(stats.visitorIds).length;
      } else {
        existing.lastSeenAt = now;
        existing.lastPath = pagePath;
        existing.userAgentHash = userAgentHash;
      }
    }

    stats.updatedAt = now;
    await saveVisitorStats();
    sendJson(res, 200, getPublicVisitorStats(stats));
  } catch (error) {
    sendJson(res, 400, { error: 'Invalid request body' });
  }
}

async function serveFile(filePath, res, req) {
  try {
    const data = await fs.readFile(filePath);
    const ext = path.extname(filePath).toLowerCase();
    const contentType = mimeTypes[ext] || 'application/octet-stream';
    const headers = {
      'Content-Type': contentType,
      'Cache-Control': CACHE_CONTROL[ext] || 'no-cache',
      'X-Content-Type-Options': 'nosniff',
      'X-Frame-Options': 'SAMEORIGIN',
    };
    if (ext === '.pdf' || ext === '.png' || ext === '.jpg' || ext === '.jpeg' || ext === '.svg' || ext === '.webp') {
      headers['Content-Disposition'] = 'inline';
    }
    await logDownload(req, filePath);
    res.writeHead(200, headers);
    res.end(data);
  } catch (error) {
    res.writeHead(404, { 'Content-Type': 'text/plain; charset=utf-8' });
    res.end('Not found');
  }
}

const server = http.createServer(async (req, res) => {
  let requestUrl;
  try {
    requestUrl = new URL(req.url, `http://${req.headers.host}`);
  } catch {
    res.writeHead(400, { 'Content-Type': 'text/plain; charset=utf-8' });
    res.end('Bad Request');
    return;
  }

  let pathname;
  try {
    pathname = decodeURIComponent(requestUrl.pathname);
  } catch {
    res.writeHead(400, { 'Content-Type': 'text/plain; charset=utf-8' });
    res.end('Bad Request');
    return;
  }
  if (pathname === '/') pathname = '/index.html';

  if (pathname === '/api/visits') {
    await handleVisitorStats(req, res);
    return;
  }

  if (pathname === '/api/downloads') {
    if (req.method !== 'GET') {
      res.writeHead(405, { 'Content-Type': 'text/plain; charset=utf-8', Allow: 'GET' });
      res.end('Method Not Allowed');
      return;
    }
    try {
      const raw = await fs.readFile(downloadLogPath, 'utf8');
      const log = JSON.parse(raw);
      sendJson(res, 200, { count: log.length, entries: log.slice(-200) });
    } catch {
      sendJson(res, 200, { count: 0, entries: [] });
    }
    return;
  }

  // Only allow GET and HEAD for static files
  if (req.method !== 'GET' && req.method !== 'HEAD') {
    res.writeHead(405, { 'Content-Type': 'text/plain; charset=utf-8', Allow: 'GET, HEAD' });
    res.end('Method Not Allowed');
    return;
  }

  // Normalise to prevent path traversal (e.g. /../../../etc/passwd)
  const normalised = path.normalize(pathname);
  const filePath = path.join(__dirname, normalised);

  // Guard: resolved path must still start with project root
  const root = __dirname.endsWith(path.sep) ? __dirname : __dirname + path.sep;
  if (!filePath.startsWith(root) && filePath !== __dirname) {
    res.writeHead(403, { 'Content-Type': 'text/plain; charset=utf-8' });
    res.end('Forbidden');
    return;
  }

  await serveFile(filePath, res, req);
});

server.on('error', (err) => {
  if (err.code === 'EADDRINUSE') {
    console.error(
      `\n[server] ERROR: Port ${port} is already in use.\n` +
      `  Stop the process using that port, or set a different port:\n` +
      `    PORT=3001 node server.mjs\n`
    );
  } else {
    console.error('[server] Unexpected error:', err);
  }
  process.exit(1);
});

server.listen(port, () => {
  console.log(`Server started at http://localhost:${port}`);
});
