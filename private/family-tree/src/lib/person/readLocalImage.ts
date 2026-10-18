import { invoke } from "@tauri-apps/api/core";
import { getLocalFilePath, toUint8Array } from "@/lib/person/getPersonPhotoSource";

export async function readLocalImageObjectUrl(pathOrUrl?: string) {
  if (!pathOrUrl?.trim()) return "";
  if (/^(data:|blob:)/i.test(pathOrUrl)) return pathOrUrl;

  const path = getLocalFilePath(pathOrUrl) ?? pathOrUrl.replace(/^file:\/\//i, "");
  const data = await invoke<unknown>("read_local_file", { path });
  return URL.createObjectURL(new Blob([toUint8Array(data)], { type: "image/png" }));
}
