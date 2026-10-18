import { useState } from 'react';
import type { ScannedFile } from '../../types';
import type { TranslationKey } from '../../../../i18n/translations';

export const STATUSES = ['ACT', 'CLS', 'PEND', 'DRAFT'] as const;
export type Status = (typeof STATUSES)[number];

export const FILE_STATUS_HINT: Record<Status, TranslationKey> = {
  ACT: 'documentPreview.fileStatus.ACT',
  CLS: 'documentPreview.fileStatus.CLS',
  PEND: 'documentPreview.fileStatus.PEND',
  DRAFT: 'documentPreview.fileStatus.DRAFT',
};

function pad2(n: number) {
  return String(n).padStart(2, '0');
}

function toYYMMDD(d: Date): string {
  const yy = String(d.getFullYear()).slice(2);
  return `${yy}${pad2(d.getMonth() + 1)}${pad2(d.getDate())}`;
}

function isValidDescription(s: string): boolean {
  return /^[A-Za-z][A-Za-z0-9_]*$/.test(s);
}

export const useDocumentPreview = (
  file: ScannedFile,
  onRename: (file: ScannedFile, newStem: string) => Promise<boolean>,
) => {
  const today = new Date();
  const [dateStr, setDateStr] = useState<string>(file.dateStamp || toYYMMDD(today));
  const [description, setDescription] = useState<string>(
    file.description ? file.description.replace(/\s+/g, '_') : '',
  );
  const [version, setVersion] = useState<string>(file.version ?? 'v01');
  const [status, setStatus] = useState<Status>((file.status as Status) ?? 'ACT');
  const [isSaving, setIsSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);

  const newStem = `${dateStr}_${description || 'Description'}_${version}_${status}`;
  const newName = `${newStem}${file.ext}`;
  const isValid =
    /^\d{6}$/.test(dateStr) && isValidDescription(description) && /^v\d{2,}$/.test(version);

  const handleRename = async () => {
    if (!isValid) return;
    setIsSaving(true);
    setSaveError(null);
    try {
      const ok = await onRename(file, newStem);
      if (ok) {
        setSaved(true);
        setTimeout(() => setSaved(false), 2000);
      } else {
        setSaveError('documentPreview.renameError');
      }
    } finally {
      setIsSaving(false);
    }
  };

  const incrementVersion = () => {
    const num = parseInt(version.slice(1)) || 1;
    setVersion(`v${pad2(num + 1)}`);
  };

  return {
    dateStr,
    setDateStr,
    description,
    setDescription,
    version,
    setVersion,
    status,
    setStatus,
    isSaving,
    saved,
    saveError,
    newStem,
    newName,
    isValid,
    handleRename,
    incrementVersion,
    isValidDescription,
  };
};
