// Loads a FictionBook: bytes (plain or zipped) -> text in the declared encoding -> XML document + book info.
import { firstEntry } from './zip.js';
const NS_XLINK = 'http://www.w3.org/1999/xlink';
function decode(bytes) {
  if (bytes[0] === 0xff && bytes[1] === 0xfe) return new TextDecoder('utf-16le').decode(bytes);
  if (bytes[0] === 0xfe && bytes[1] === 0xff) return new TextDecoder('utf-16be').decode(bytes);
  const head = new TextDecoder('latin1').decode(bytes.subarray(0, 200));
  const label = bytes[0] === 0xef ? 'utf-8' : (/encoding=["']([\w-]+)["']/i.exec(head)?.[1] ?? 'utf-8');
  try { return new TextDecoder(label).decode(bytes); } catch { return new TextDecoder('utf-8').decode(bytes); }
}
function parseXml(text) {
  const parse = (s) => new DOMParser().parseFromString(s, 'application/xml');
  let doc = parse(text);
  if (doc.querySelector('parsererror')) doc = parse(text.replace(/&(?!(?:amp|lt|gt|quot|apos|#\d+|#x[0-9a-f]+);)/gi, '&amp;'));
  if (doc.querySelector('parsererror')) throw new Error('bad FB2 xml');
  return doc;
}
export const hrefOf = (el) => (el.getAttributeNS(NS_XLINK, 'href') ?? [...el.attributes].find((a) => a.localName === 'href')?.value ?? '').replace(/^#/, '');
const nameOf = (a) => ['first-name', 'middle-name', 'last-name'].map((k) => a.querySelector(k)?.textContent.trim()).filter(Boolean).join(' ') || a.querySelector('nickname')?.textContent.trim() || '';
export async function loadBook(blob) {
  let buf = await blob.arrayBuffer();
  const head = new Uint8Array(buf, 0, 2);
  if (head[0] === 0x50 && head[1] === 0x4b) { const e = await firstEntry(buf, (n) => /\.fb2$/i.test(n)); buf = e.buffer.slice(e.byteOffset, e.byteOffset + e.byteLength); }
  const doc = parseXml(decode(new Uint8Array(buf)));
  const info = doc.querySelector('description > title-info');
  const bodies = [...doc.querySelectorAll('FictionBook > body')];
  const length = bodies.filter((b) => b.getAttribute('name') !== 'notes').reduce((n, b) => n + b.textContent.length, 0);
  const binaries = new Map([...doc.querySelectorAll('FictionBook > binary')].map((b) => [b.getAttribute('id'), { type: b.getAttribute('content-type') || 'image/jpeg', b64: b.textContent.replace(/\s+/g, '') }]));
  const cover = info?.querySelector('coverpage > image');
  return {
    doc, bodies, binaries,
    title: info?.querySelector('book-title')?.textContent.trim() ?? '',
    authors: [...(info?.querySelectorAll('author') ?? [])].map(nameOf).filter(Boolean),
    lang: info?.querySelector('lang')?.textContent.trim() || 'ru',
    annotation: info?.querySelector('annotation') ?? null,
    coverId: cover ? hrefOf(cover) : '',
    pages: Math.max(1, Math.ceil(length / 1800)),
  };
}
