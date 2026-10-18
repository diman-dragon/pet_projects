import { useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { getLocalFilePath, getPersonPhotoSource, toUint8Array } from "@/lib/person/getPersonPhotoSource";

export function usePhotoSrc(value?: string) {
  const [src, setSrc] = useState<string | undefined>(() => getPersonPhotoSource(value));

  useEffect(() => {
    let objectUrl: string | undefined;
    let cancelled = false;
    const immediate = getPersonPhotoSource(value);
    const localPath = getLocalFilePath(value);

    if (!value?.trim()) {
      setSrc(undefined);
      return;
    }

    setSrc(immediate);

    if (!localPath) {
      return;
    }

    void invoke<unknown>("read_local_file", { path: localPath })
      .then((data) => {
        if (cancelled) return;
        objectUrl = URL.createObjectURL(new Blob([toUint8Array(data)], { type: "image/png" }));
        setSrc(objectUrl);
      })
      .catch(() => {
        if (!cancelled) setSrc(immediate);
      });

    return () => {
      cancelled = true;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [value]);

  return src;
}
