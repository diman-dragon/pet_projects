import { convertFileSrc } from "@tauri-apps/api/core";

export function getLocalFilePath(value?: string) {
  const source = value?.trim();
  if (!source) return undefined;
  if (/^(https?:|asset:|data:|blob:)/i.test(source)) return undefined;

  if (/^file:\/\//i.test(source)) {
    const withoutProtocol = decodeURIComponent(source.replace(/^file:\/\//i, ""));
    return withoutProtocol.replace(/^\/([A-Za-z]:)/, "$1").replace(/\//g, "\\");
  }

  return source;
}

export function getPersonPhotoSource(value?: string) {
  const source = value?.trim();
  if (!source) return undefined;
  if (/^(https?:|asset:|data:|blob:)/i.test(source)) return source;

  const filePath = getLocalFilePath(source) ?? source;
  try {
    return convertFileSrc(filePath.replace(/\\/g, "/"));
  } catch {
    return undefined;
  }
}

export function toUint8Array(data: unknown) {
  if (data instanceof Uint8Array) return data;
  if (data instanceof ArrayBuffer) return new Uint8Array(data);
  if (ArrayBuffer.isView(data)) return new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
  if (Array.isArray(data)) return Uint8Array.from(data);
  throw new Error("Некорректные данные изображения");
}
