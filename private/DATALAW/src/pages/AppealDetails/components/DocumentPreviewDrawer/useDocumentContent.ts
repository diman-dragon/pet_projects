import { useState, useEffect } from 'react';
import { readFile } from '@tauri-apps/plugin-fs';
import * as mammoth from 'mammoth';
import * as XLSX from 'xlsx';

export type PreviewType = 'image' | 'pdf' | 'docx' | 'xlsx' | 'text' | 'unsupported';

export function useDocumentContent(filePath: string, ext: string) {
  const [content, setContent] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [type, setType] = useState<PreviewType>('unsupported');

  useEffect(() => {
    let active = true;
    const normalizedExt = ext.toLowerCase();

    const loadContent = async () => {
      setLoading(true);
      setError(null);
      setContent(null);

      try {
        if (['.jpg', '.jpeg', '.png', '.webp', '.gif'].includes(normalizedExt)) {
          setType('image');
        } else if (normalizedExt === '.pdf') {
          setType('pdf');
        } else if (normalizedExt === '.docx') {
          setType('docx');
          const bytes = await readFile(filePath);
          const result = await mammoth.convertToHtml({ arrayBuffer: bytes.buffer });
          if (active) setContent(result.value);
        } else if (['.xlsx', '.xls', '.csv'].includes(normalizedExt)) {
          setType('xlsx');
          const bytes = await readFile(filePath);
          const workbook = XLSX.read(bytes.buffer, { type: 'array' });
          const firstSheetName = workbook.SheetNames[0];
          const worksheet = workbook.Sheets[firstSheetName];
          const html = XLSX.utils.sheet_to_html(worksheet);
          if (active) setContent(html);
        } else if (['.txt', '.md', '.json', '.log'].includes(normalizedExt)) {
          setType('text');
          const bytes = await readFile(filePath);
          const text = new TextDecoder().decode(bytes);
          if (active) setContent(text);
        } else {
          setType('unsupported');
        }
      } catch (err) {
        console.error('Failed to load document preview:', err);
        if (active) setError(String(err));
      } finally {
        if (active) setLoading(false);
      }
    };

    loadContent();
    return () => {
      active = false;
    };
  }, [filePath, ext]);

  return { content, loading, error, type };
}
