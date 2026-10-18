// src/pages/audit/components/FsTab.tsx

import { useMemo, useState } from 'react';
import { ChevronRight, FileText, Folder, ShieldCheck } from 'lucide-react';
import { useI18n } from '../../../i18n/AppLanguageProvider';
import type { FileNode } from '../../../services/fs/ScannerService';

export function FileNodeItem({ node, depth = 0 }: { node: FileNode; depth?: number }) {
  const { t } = useI18n();
  const [collapsed, setCollapsed] = useState(depth > 1);

  const filesCount = useMemo(() => {
    if (node.isFile) return 0;

    const countAll = (current: FileNode): number => {
      let count = current.children?.filter((child) => child.isFile).length ?? 0;
      current.children?.forEach((child) => {
        if (!child.isFile) count += countAll(child);
      });
      return count;
    };

    return countAll(node);
  }, [node]);

  const hasChildren = !node.isFile && (node.children?.length ?? 0) > 0;

  return (
    <div>
      <div
        role={hasChildren ? 'button' : undefined}
        tabIndex={hasChildren ? 0 : undefined}
        className={`flex h-7 items-center rounded px-2 transition-colors hover:bg-[color-mix(in_srgb,var(--app-text)_4%,transparent)] ${hasChildren ? 'cursor-pointer' : ''}`}
        style={{ paddingLeft: `${8 + depth * 14}px` }}
        onClick={() => hasChildren && setCollapsed((value) => !value)}
        onKeyDown={(e) => {
          if (hasChildren && (e.key === 'Enter' || e.key === ' ')) {
            e.preventDefault();
            setCollapsed((value) => !value);
          }
        }}
      >
        {hasChildren && (
          <ChevronRight
            size={12}
            className={`mr-1 shrink-0 text-[var(--app-text-muted)] transition-transform ${collapsed ? '' : 'rotate-90'}`}
          />
        )}
        {!hasChildren && !node.isFile && <div className="w-4 shrink-0" />}

        {node.isFile ? (
          <FileText className="mr-1.5 h-3.5 w-3.5 shrink-0 text-[color-mix(in_srgb,#3b82f6_60%,var(--app-text-muted))]" />
        ) : (
          <Folder className="mr-1.5 h-3.5 w-3.5 shrink-0 text-[color-mix(in_srgb,#f59e0b_70%,var(--app-text-muted))]" />
        )}

        <span
          className={`truncate text-[length:var(--text-xs)] leading-[var(--leading-body)] ${
            node.isFile ? 'text-[var(--app-text-secondary)]' : 'font-medium text-[var(--app-text)]'
          }`}
        >
          {node.name}
        </span>

        {!node.isFile && filesCount > 0 && (
          <span className="ml-auto shrink-0 text-[length:var(--text-xs)] font-bold leading-[var(--leading-heading)] text-[var(--app-text-muted)]">
            {t('audit.fs.filesShort', { count: filesCount })}
          </span>
        )}
      </div>

      {!collapsed &&
        node.children?.map((child) => (
          <FileNodeItem key={child.path} node={child} depth={depth + 1} />
        ))}
    </div>
  );
}

export function FsTab({ fsTree, configPath }: { fsTree: FileNode | null; configPath: string }) {
  const { t } = useI18n();

  return (
    <div className="space-y-4">
      {configPath && (
        <div className="flex items-center gap-2 rounded-lg border border-[var(--app-border)] bg-[var(--app-surface-raised)] px-4 py-2.5">
          <ShieldCheck
            size={14}
            strokeWidth={2}
            className="shrink-0 text-[var(--app-text-muted)]"
          />
          <span className="shrink-0 text-[length:var(--text-xs)] font-bold uppercase leading-[var(--leading-heading)] tracking-wider text-[var(--app-text-muted)]">
            {t('audit.fs.configDir')}
          </span>
          <code className="ml-1 truncate font-mono text-[length:var(--text-xs)] leading-[var(--leading-body)] text-[var(--app-text-secondary)]">
            {configPath}
          </code>
        </div>
      )}

      <div className="overflow-hidden rounded-xl border border-[var(--app-border)] bg-[var(--app-surface-raised)] shadow-[var(--app-shadow-sm)]">
        <div className="flex items-center gap-2 border-b border-[var(--app-border)] bg-[color-mix(in_srgb,var(--app-text)_2%,transparent)] px-4 py-3">
          <Folder
            size={14}
            strokeWidth={2}
            className="text-[color-mix(in_srgb,#f59e0b_70%,var(--app-text-muted))]"
          />
          <span className="text-[length:var(--text-xs)] font-semibold leading-[var(--leading-body)] text-[var(--app-text-secondary)]">
            {t('audit.fs.workdir')}
          </span>
        </div>
        <div className="custom-scrollbar max-h-[600px] overflow-y-auto p-2">
          {fsTree ? (
            <FileNodeItem node={fsTree} />
          ) : (
            <div className="py-12 text-center text-[length:var(--text-xs)] italic leading-[var(--leading-body)] text-[var(--app-text-muted)]">
              {t('audit.fs.empty')}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
