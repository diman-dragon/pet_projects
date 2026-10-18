import { useState, useCallback, useEffect } from 'react';
import { open } from '@tauri-apps/plugin-dialog';
import { readDir } from '@tauri-apps/plugin-fs';
import { join } from '@tauri-apps/api/path';
import { listen } from '@tauri-apps/api/event';
import { useSystematizerStore } from '../../store/systematizerStore';
import { createLogger } from '../../services/logger/Logger';
import { businessApi, HierarchyClientNode } from '../../services/backend/businessApi';
import { useI18n } from '../../i18n/AppLanguageProvider';

const log = createLogger('SystematizerLogic');

export function useSystematizerLogic() {
  const { t } = useI18n();
  const [isDragging, setIsDragging] = useState(false);
  const [hierarchy, setHierarchy] = useState<HierarchyClientNode[]>([]);
  const [alertMessage, setAlertMessage] = useState<string | null>(null);

  const {
    queue,
    isProcessing,
    isApplying,
    currentFile,
    progress,
    addToQueue,
    approveAll,
    applyAll,
    clearQueue,
    updateUserStatus,
    updateJobResult,
    removeJob,
  } = useSystematizerStore();

  const refreshHierarchy = useCallback(async () => {
    try {
      const tree = await businessApi.getHierarchyTree();
      setHierarchy(tree);
    } catch (err) {
      log.error('Failed to fetch hierarchy', err);
    }
  }, []);

  useEffect(() => {
    refreshHierarchy();
  }, [refreshHierarchy]);

  useEffect(() => {
    let unlisten: () => void;

    listen<string[]>('tauri://file-drop', (event) => {
      if (event.payload.length > 0) {
        addToQueue(event.payload);
      }
    }).then((u) => (unlisten = u));

    return () => {
      if (unlisten) unlisten();
    };
  }, [addToQueue]);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    // Drop is handled via tauri://file-drop event for full paths
  }, []);

  const pickFolder = useCallback(async () => {
    try {
      const selected = await open({
        directory: true,
        multiple: false,
      });

      if (selected && typeof selected === 'string') {
        const files: string[] = [];
        const scan = async (dir: string) => {
          const entries = await readDir(dir);
          for (const entry of entries) {
            const fullPath = await join(dir, entry.name);
            if (entry.isDirectory) {
              await scan(fullPath);
            } else {
              files.push(fullPath);
            }
          }
        };
        await scan(selected);
        if (files.length > 0) addToQueue(files);
      }
    } catch (err) {
      log.error('Folder picker error', err);
    }
  }, [addToQueue]);

  const pickFiles = useCallback(async () => {
    try {
      const selected = await open({
        multiple: true,
        directory: false,
      });

      if (selected && Array.isArray(selected)) {
        addToQueue(selected);
      } else if (selected && typeof selected === 'string') {
        addToQueue([selected]);
      }
    } catch (err) {
      log.error('File picker error', err);
    }
  }, [addToQueue]);

  const approve = useCallback(
    (id: string) => {
      updateUserStatus(id, 'approved');
    },
    [updateUserStatus],
  );

  const reject = useCallback(
    (id: string) => {
      updateUserStatus(id, 'rejected');
    },
    [updateUserStatus],
  );

  const handleApplyAll = useCallback(async () => {
    const approvedCount = queue.filter((j) => j.userStatus === 'approved').length;
    if (approvedCount === 0) {
      return {
        error: 'Нет одобренных документов. Пожалуйста, одобрите документы (кнопка с галочкой) перед сохранением.',
      };
    }
    await applyAll();
    return { error: null };
  }, [queue, applyAll]);

  return {
    title: t('systematizer.dropZone.title'),
    subtitle: t('systematizer.dropZone.subtitle'),
    pickFilesLabel: t('systematizer.btn.pickFiles'),
    pickFolderLabel: t('systematizer.btn.pickFolder'),
    processingLabel: t('systematizer.status.processing'),
    scanningLabel: t('systematizer.status.scanning'),
    resultsLabel: t('systematizer.results.title'),
    clearLabel: t('common.clear'),
    approveAllLabel: t('systematizer.results.approveAll'),
    applyAllLabel: t('systematizer.btn.applyAll'),
    isDragging,
    isProcessing: isProcessing || isApplying,
    currentFile,
    progress,
    suggestions: queue,
    handleDragOver,
    handleDragLeave,
    handleDrop,
    pickFolder,
    pickFiles,
    approve,
    approveAll,
    reject,
    clearAll: clearQueue,
    applyAll: async () => {
      const result = await handleApplyAll();
      if (result.error) setAlertMessage(result.error);
    },
    removeJob,
    hierarchy,
    refreshHierarchy,
    updateJobResult,
    updateUserStatus,
    alertMessage,
    clearAlert: () => setAlertMessage(null),
  };
}
