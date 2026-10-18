import { useMemo, useState } from 'react';
import type { HierarchyClientNode } from '../../../services/backend/businessApi';
import type { SystematizerJob } from '../../../store/systematizerStore';
import { NamingService } from '../../../services/utils/NamingService';

export interface JobEditValues {
  client: string;
  client_type: 'LEGAL' | 'INDIVIDUAL';
  case: string;
  line: string;
  appeal: string;
  date?: string;
  filename?: string;
}

export function useJobEditing(hierarchy: HierarchyClientNode[], jobs: SystematizerJob[]) {
  const [editingId, setEditingId] = useState<string | null>(null);
  const [values, setValues] = useState<JobEditValues | null>(null);

  const startEditing = (job: SystematizerJob) => {
    setEditingId(job.id);
    setValues({
      client: job.result?.client || '',
      client_type: job.result?.client_type || 'LEGAL',
      case: job.result?.case || '',
      line: job.result?.line || '',
      appeal: job.result?.appeal || '',
      date: job.result?.metadata?.date || '',
      filename: job.result?.metadata?.suggested_filename || job.fileName,
    });
  };

  const cancelEditing = () => {
    setEditingId(null);
    setValues(null);
  };

  const options = useMemo(() => {
    const clientOptions = hierarchy.map((c) => ({ label: c.full_name, value: c.folder_name }));
    const selectedClient = hierarchy.find((c) => c.folder_name === values?.client);
    const caseOptions = selectedClient?.cases?.map((c) => ({ label: c.title, value: c.folder_name })) || [];
    const selectedCase = selectedClient?.cases?.find((c) => c.folder_name === values?.case);
    const lineOptions = selectedCase?.lines?.map((l) => ({ label: l.title, value: l.folder_name })) || [];
    const selectedLine = selectedCase?.lines?.find((l) => l.folder_name === values?.line);
    const appealOptions = selectedLine?.appeals?.map((a) => ({ label: a.title, value: a.folder_name })) || [];
    return { clientOptions, caseOptions, lineOptions, appealOptions };
  }, [hierarchy, values?.appeal, values?.case, values?.client, values?.line]);

  const saveEditing = (id: string, updateJobResult: (id: string, result: any) => void) => {
    if (!values) return;
    const job = jobs.find((item) => item.id === id);
    if (!job?.result) return cancelEditing();
    const rawName = values.filename || '';
    const ext = rawName.includes('.') ? rawName.split('.').pop() : '';
    const stem = rawName.includes('.') ? rawName.slice(0, rawName.lastIndexOf('.')) : rawName;
    const normalizedName = /[а-яА-Я]/.test(rawName) ? `${NamingService.toStandardName(stem)}${ext ? `.${ext.toLowerCase()}` : ''}` : rawName;
    updateJobResult(id, {
      ...job.result,
      client: values.client.slice(0, 40),
      client_type: values.client_type,
      case: values.case.slice(0, 35),
      line: values.line.slice(0, 35),
      appeal: values.appeal.slice(0, 35),
      metadata: { ...(job.result.metadata || {}), date: values.date, suggested_filename: normalizedName.slice(0, 50) },
    });
    cancelEditing();
  };

  return { editingId, values, setValues, options, startEditing, cancelEditing, saveEditing };
}
