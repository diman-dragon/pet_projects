import { join, tempDir } from '@tauri-apps/api/path';
import { readFile, writeFile } from '@tauri-apps/plugin-fs';
import { businessApi } from '../../services/backend/businessApi';
import { StorageService } from '../../services/fs/StorageService';
import { CaseNamingService } from '../../services/naming/CaseNamingService';
import { createLogger } from '../../services/logger/Logger';
import type { SystematizerStoreState } from './types';

const log = createLogger('SystematizerEditActions');

export function createEditActions(set: any, get: () => SystematizerStoreState) {
  return {
    approveAll: () => set((state: SystematizerStoreState) => ({ queue: state.queue.map((job) => job.status === 'completed' ? { ...job, userStatus: 'approved' } : job) })),
    updateUserStatus: (id: string, status: any) => set((state: SystematizerStoreState) => ({ queue: state.queue.map((job) => job.id === id ? { ...job, userStatus: status } : job) })),
    updateJobResult: (id: string, result: any) => set((state: SystematizerStoreState) => ({ queue: state.queue.map((job) => job.id === id ? { ...job, result } : job) })),
    updateJobStatus: (id: string, status: any, result?: any, error?: string) => set((state: SystematizerStoreState) => ({ queue: state.queue.map((job) => job.id === id ? { ...job, status, result, error } : job) })),
    applyAll: async () => {
      const approved = get().queue.filter((job) => job.userStatus === 'approved' && job.status === 'completed' && job.result);
      if (!approved.length || get().isApplying) return;
      set({ isApplying: true });
      let tree = await businessApi.getHierarchyTree();
      const normalize = (value: string) => CaseNamingService.toCaseName(value || '').toLowerCase().replace(/[^a-z0-9]/g, '');

      for (let index = 0; index < approved.length; index++) {
        const job = approved[index];
        const result = job.result!;
        set({ currentFile: job.fileName, progress: Math.round((index / approved.length) * 100) });
        get().updateJobStatus(job.id, 'applying');
        try {
          let client = tree.find((item) => normalize(item.full_name) === normalize(result.client) || normalize(item.folder_name).includes(normalize(result.client)));
          if (!client) {
            await businessApi.saveClient({ full_name: result.client, client_type: result.client_type || 'LEGAL', status: 'ACTIVE' });
            tree = await businessApi.getHierarchyTree();
            client = tree.find((item) => normalize(item.full_name) === normalize(result.client) || normalize(item.folder_name).includes(normalize(result.client)));
          }
          if (!client) throw new Error(`Failed to find/create client: ${result.client}`);

          let caseItem = client.cases?.find((item) => normalize(item.title) === normalize(result.case) || normalize(item.folder_name).includes(normalize(result.case)));
          if (!caseItem) {
            await businessApi.saveCase({ client_id: client.id, client_folder_name: client.folder_name, title: result.case, status: 'ACTIVE' });
            tree = await businessApi.getHierarchyTree();
            client = tree.find((item) => item.id === client!.id)!;
            caseItem = client.cases?.find((item) => normalize(item.title) === normalize(result.case) || normalize(item.folder_name).includes(normalize(result.case)));
          }
          if (!caseItem) throw new Error(`Failed to find/create case: ${result.case}`);

          let line = caseItem.lines?.find((item) => normalize(item.title) === normalize(result.line) || normalize(item.folder_name).includes(normalize(result.line)));
          if (!line) {
            await businessApi.saveLine({ case_id: caseItem.id, case_folder_name: caseItem.folder_name, client_folder_name: client.folder_name, title: result.line });
            tree = await businessApi.getHierarchyTree();
            client = tree.find((item) => item.id === client!.id)!;
            caseItem = client.cases.find((item) => item.id === caseItem!.id)!;
            line = caseItem.lines?.find((item) => normalize(item.title) === normalize(result.line) || normalize(item.folder_name).includes(normalize(result.line)));
          }
          if (!line) throw new Error(`Failed to find/create line: ${result.line}`);

          let appeal = line.appeals?.find((item) => normalize(item.title) === normalize(result.appeal) || normalize(item.folder_name).includes(normalize(result.appeal)));
          if (!appeal) {
            await businessApi.saveAppeal({ line_id: line.id, client_folder_name: client.folder_name, case_folder_name: caseItem.folder_name, line_folder_name: line.folder_name, title: result.appeal, status: 'OPEN', appeal_date: result.metadata?.date || undefined, recipient: result.metadata?.recipient || undefined, outgoing_number: result.metadata?.outgoing_number || undefined });
            tree = await businessApi.getHierarchyTree();
            client = tree.find((item) => item.id === client!.id)!;
            caseItem = client.cases.find((item) => item.id === caseItem!.id)!;
            line = caseItem.lines.find((item) => item.id === line!.id)!;
            appeal = line.appeals?.find((item) => normalize(item.title) === normalize(result.appeal) || normalize(item.folder_name).includes(normalize(result.appeal)));
          }
          if (!appeal) throw new Error('Failed to find/create appeal');

          const finalFileName = result.metadata?.suggested_filename?.trim() || job.fileName;
          const tempPath = await join(await tempDir(), finalFileName);
          await writeFile(tempPath, await readFile(job.filePath));
          await businessApi.uploadAppealDocument(appeal.id, tempPath);
          get().updateJobStatus(job.id, 'applied');
          try { await StorageService.deleteFile(tempPath); } catch {}
        } catch (err: any) {
          log.error('Job apply failed', { job, err });
          get().updateJobStatus(job.id, 'error', undefined, err?.message || 'Apply failed');
        }
      }
      set({ isApplying: false, currentFile: '', progress: 100 });
    },
  };
}
