#!/usr/bin/env npx tsx

import * as crypto from 'crypto';
import * as fs from 'fs';
import * as path from 'path';

type LicenseTier = 'basic' | 'pro' | 'enterprise';

const GENERATED_KEY_PREFIX = 'DL1-';
const EPOCH_DATE = 1776038400000; // 2026-04-10 00:00:00 UTC

// Формат ключа v2: data(2) + nonce(3) + hmac(8) = 13 байт → 21 символ base32 → pad до 24
// Итоговый ключ: DL1-XXXX-XXXX-XXXX-XXXX-XXXX-XXXX (24 символа, 6 групп по 4)

const TIER_TO_VAL: Record<LicenseTier, number> = {
  basic: 1,
  pro: 2,
  enterprise: 3,
};

// 32 characters for 5 bits per char
const CHARSET = '23456789ABCDEFGHJKLMNPQRSTUVWXYZ';

function loadMasterKey(): string {
  const envValue = process.env.Dorlane_LICENSE_MASTER_KEY?.trim();
  if (envValue) return envValue;

  const secretPath = path.resolve('.secrets/license-master.key');
  if (!fs.existsSync(secretPath)) {
    throw new Error(
      'Missing master key. Set Dorlane_LICENSE_MASTER_KEY or create .secrets/license-master.key',
    );
  }

  const fileValue = fs.readFileSync(secretPath, 'utf8').trim();
  if (!fileValue) {
    throw new Error(`Master key file is empty: ${secretPath}`);
  }

  return fileValue;
}

function encodeBase32(buffer: Buffer): string {
  let bits = 0;
  let value = 0;
  let output = '';

  for (let i = 0; i < buffer.length; i++) {
    value = (value << 8) | buffer[i];
    bits += 8;

    while (bits >= 5) {
      output += CHARSET[(value >>> (bits - 5)) & 31];
      bits -= 5;
    }
  }

  if (bits > 0) {
    output += CHARSET[(value << (5 - bits)) & 31];
  }

  return output;
}

function main() {
  const masterKey = loadMasterKey();
  const args = parseArgs();
  const tier = (args.tier ?? 'basic').toLowerCase() as LicenseTier;
  const days = Number.parseInt(args.days ?? '365', 10);
  const outputFile = args.output;

  if (!(tier in TIER_TO_VAL)) {
    throw new Error(`Unknown tier: ${tier}`);
  }

  const tierVal = TIER_TO_VAL[tier];
  const issuedAtMs = Date.now();
  const expiresAtMs = days > 0 ? issuedAtMs + days * 86_400_000 : 0;

  // 12 bits for expiry: days since EPOCH_DATE
  const expiryDays =
    expiresAtMs > 0
      ? Math.max(0, Math.min(4095, Math.floor((expiresAtMs - EPOCH_DATE) / 86400000)))
      : 0;

  // 1. Pack data into 2 bytes: [4 bits: Tier] [12 bits: Expiry Days]
  const data = Buffer.alloc(2);
  data[0] = (tierVal << 4) | (expiryDays >> 8);
  data[1] = expiryDays & 0xff;

  // 2. Generate 3 random nonce bytes — makes every key unique
  const nonce = crypto.randomBytes(3);

  // 3. Sign data + nonce together (8 bytes HMAC)
  const hmac = crypto.createHmac('sha256', masterKey);
  hmac.update(data);
  hmac.update(nonce);
  const signature = hmac.digest().subarray(0, 8);

  // 4. Combine: data(2) + nonce(3) + signature(8) = 13 bytes
  const total = Buffer.concat([data, nonce, signature]);

  // 5. Encode as base32 → 21 chars, pad to 24
  const rawKey = encodeBase32(total).padEnd(24, '2');

  // 6. Format as 6 groups of 4: DL1-XXXX-XXXX-XXXX-XXXX-XXXX-XXXX
  const groups = rawKey.match(/.{1,4}/g)!.join('-');
  const formattedKey = `${GENERATED_KEY_PREFIX}${groups}`;

  const result = {
    activationKey: formattedKey,
    key: formattedKey,
    tier,
    expiresAt: expiresAtMs > 0 ? new Date(expiresAtMs).toISOString() : null,
    issuedAt: new Date(issuedAtMs).toISOString(),
    isMaster: false,
  };

  if (outputFile) {
    const outPath = path.resolve(outputFile);
    fs.mkdirSync(path.dirname(outPath), { recursive: true });
    fs.writeFileSync(outPath, JSON.stringify(result, null, 2), 'utf8');
    console.log(`Saved license key to ${outPath}`);
    return;
  }

  console.log(JSON.stringify(result, null, 2));
}

function parseArgs(): Record<string, string> {
  const args: Record<string, string> = {};
  const argv = process.argv.slice(2);
  for (let i = 0; i < argv.length; i++) {
    if (argv[i].startsWith('--')) {
      const key = argv[i].slice(2);
      const val = argv[i + 1];
      if (val && !val.startsWith('--')) {
        args[key] = val;
        i++;
      } else {
        args[key] = 'true';
      }
    }
  }
  return args;
}

main();
