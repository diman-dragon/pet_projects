import { ConnectionManager } from '../db/ConnectionManager';
import { AuditService } from '../audit/AuditService';
import type { UserProfile } from '../../types/entities';

const encoder = new TextEncoder();
const decoder = new TextDecoder();

function bytesToBase64(bytes: Uint8Array): string {
  let binary = '';
  for (const byte of bytes) binary += String.fromCharCode(byte);
  return btoa(binary);
}

function base64ToBytes(value: string): Uint8Array {
  const binary = atob(value);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return bytes;
}

async function sha256Bytes(value: string): Promise<Uint8Array> {
  const digest = await crypto.subtle.digest('SHA-256', encoder.encode(value));
  return new Uint8Array(digest);
}

async function importAesKey(raw: Uint8Array): Promise<CryptoKey> {
  return crypto.subtle.importKey('raw', raw, 'AES-GCM', false, ['encrypt', 'decrypt']);
}

function randomBytes(length: number): Uint8Array {
  const bytes = new Uint8Array(length);
  crypto.getRandomValues(bytes);
  return bytes;
}

async function encryptJson(payload: unknown, keyBytes: Uint8Array, aad?: string) {
  const key = await importAesKey(keyBytes);
  const nonce = randomBytes(12);
  const cipher = await crypto.subtle.encrypt(
    { name: 'AES-GCM', iv: nonce, additionalData: aad ? encoder.encode(aad) : undefined },
    key,
    encoder.encode(JSON.stringify(payload)),
  );
  return { cipher: new Uint8Array(cipher), nonce };
}

async function decryptJson<T>(cipher: Uint8Array, nonce: Uint8Array, keyBytes: Uint8Array, aad?: string): Promise<T> {
  const key = await importAesKey(keyBytes);
  const plain = await crypto.subtle.decrypt(
    { name: 'AES-GCM', iv: nonce, additionalData: aad ? encoder.encode(aad) : undefined },
    key,
    cipher,
  );
  return JSON.parse(decoder.decode(plain)) as T;
}

async function wrapDek(dek: Uint8Array, userId: number): Promise<string> {
  const kek = await importAesKey(await sha256Bytes(`user-kek:v1:${userId}`));
  const nonce = randomBytes(12);
  const cipher = await crypto.subtle.encrypt({ name: 'AES-GCM', iv: nonce }, kek, dek);
  return JSON.stringify({ nonce: bytesToBase64(nonce), cipher: bytesToBase64(new Uint8Array(cipher)) });
}

async function unwrapDek(wrappedDek: string, userId: number): Promise<Uint8Array> {
  const parsed = JSON.parse(wrappedDek) as { nonce: string; cipher: string };
  const kek = await importAesKey(await sha256Bytes(`user-kek:v1:${userId}`));
  const dek = await crypto.subtle.decrypt(
    { name: 'AES-GCM', iv: base64ToBytes(parsed.nonce) },
    kek,
    base64ToBytes(parsed.cipher),
  );
  return new Uint8Array(dek);
}

export interface CreatePiiSubjectInput {
  displayCode: string;
  piiPayload: Record<string, unknown>;
  actor: UserProfile;
}

export const PiiEnvelopeService = {
  async createSubject(input: CreatePiiSubjectInput): Promise<{ subjectId: number; piiRef: string }> {
    const clientsDb = ConnectionManager.get('clients_main');
    const piiDb = ConnectionManager.get('pii_vault');
    const piiRef = `pii:${crypto.randomUUID()}`;
    const dek = randomBytes(32);
    const aad = input.displayCode;
    const encrypted = await encryptJson(input.piiPayload, dek, aad);

    await piiDb.execute(
      `INSERT INTO pii_records (pii_ref, subject_code, payload_cipher, nonce, aad, algo)
       VALUES ($1, $2, $3, $4, $5, $6)`,
      [piiRef, input.displayCode, bytesToBase64(encrypted.cipher), bytesToBase64(encrypted.nonce), aad, 'AES-256-GCM'],
    );

    await clientsDb.execute(
      `INSERT INTO subjects (display_code, pii_ref) VALUES ($1, $2)`,
      [input.displayCode, piiRef],
    );

    const [subject] = await clientsDb.select<{ id: number }[]>(
      `SELECT id FROM subjects WHERE pii_ref = $1`,
      [piiRef],
    );

    const subjectId = subject?.id;
    if (!subjectId) throw new Error('Subject was not created');

    await AuditService.logForUser(input.actor, {
      action: 'subject:create',
      entityType: 'subject',
      entityId: subjectId,
      entityLabel: input.displayCode,
    });

    return { subjectId, piiRef };
  },

  async grantDek(subjectId: number, userId: number, caseAccessId: number, dek: Uint8Array): Promise<void> {
    const clientsDb = ConnectionManager.get('clients_main');
    const wrapped = await wrapDek(dek, userId);
    await clientsDb.execute(
      `INSERT OR IGNORE INTO pii_key_grants (subject_id, user_id, wrapped_dek, granted_via_case_access_id)
       VALUES ($1, $2, $3, $4)`,
      [subjectId, userId, wrapped, caseAccessId],
    );
  },

  async decryptForUser<T = Record<string, unknown>>(
    actor: UserProfile,
    subjectId: number,
    reason: string,
  ): Promise<T> {
    const clientsDb = ConnectionManager.get('clients_main');
    const piiDb = ConnectionManager.get('pii_vault');

    const [subject] = await clientsDb.select<{ id: number; display_code: string; pii_ref: string }[]>(
      `SELECT id, display_code, pii_ref FROM subjects WHERE id = $1`,
      [subjectId],
    );
    if (!subject) throw new Error('Subject not found');

    const [grant] = await clientsDb.select<{ wrapped_dek: string }[]>(
      `SELECT wrapped_dek
       FROM pii_key_grants
       WHERE subject_id = $1 AND user_id = $2
       ORDER BY created_at DESC
       LIMIT 1`,
      [subjectId, actor.id],
    );
    if (!grant) throw new Error('No PII key grant for this subject');

    const [record] = await piiDb.select<{
      payload_cipher: string;
      nonce: string;
      aad: string | null;
    }[]>(
      `SELECT payload_cipher, nonce, aad FROM pii_records WHERE pii_ref = $1`,
      [subject.pii_ref],
    );
    if (!record) throw new Error('PII record not found');

    const dek = await unwrapDek(grant.wrapped_dek, actor.id);
    const payload = await decryptJson<T>(
      base64ToBytes(record.payload_cipher),
      base64ToBytes(record.nonce),
      dek,
      record.aad ?? undefined,
    );

    await piiDb.execute(
      `INSERT INTO pii_access_events (pii_ref, subject_id, actor_id, action, reason)
       VALUES ($1, $2, $3, $4, $5)`,
      [subject.pii_ref, subjectId, actor.id, 'pii:decrypt', reason],
    );
    await AuditService.logForUser(actor, {
      action: 'pii:decrypt',
      entityType: 'subject',
      entityId: subjectId,
      entityLabel: subject.display_code,
      details: reason,
    });

    return payload;
  },
};
