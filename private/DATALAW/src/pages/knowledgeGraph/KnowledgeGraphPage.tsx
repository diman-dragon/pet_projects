import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  Building2,
  Info,
  Network,
  RefreshCw,
  Search,
  Users,
  ZoomIn,
  ZoomOut,
} from 'lucide-react';
import { SegmentedControl } from '../../components/ui/layout/SegmentedControl';
import { KnowledgeGraphService } from '../../services/knowledgeGraph/KnowledgeGraphService';
import type {
  KgEntityRow,
  KgEntityTableRow,
  KgGraphData,
  KgRelationRow,
} from '../../services/knowledgeGraph/KnowledgeGraphService';
import { createLogger } from '../../services/logger/Logger';

const log = createLogger('KnowledgeGraphPage');

const ENTITY_COLORS: Record<string, string> = {
  PERSON: '#2563eb',
  ORGANIZATION: '#0f766e',
};

const ENTITY_LABELS: Record<string, string> = {
  PERSON: 'Лицо',
  ORGANIZATION: 'Организация',
};

const VIEW_SEGMENTS = [
  { id: 'graph', label: 'Граф', icon: <Network size={14} /> },
  { id: 'table', label: 'Сущности', icon: <Users size={14} /> },
] as const;

const TYPE_SEGMENTS = [
  { id: 'ALL', label: 'Все' },
  { id: 'PERSON', label: 'Лица' },
  { id: 'ORGANIZATION', label: 'Организации' },
] as const;

type ViewMode = 'graph' | 'table';

interface NodePos {
  id: number;
  x: number;
  y: number;
  vx: number;
  vy: number;
}

function formatPercent(value: number): string {
  return `${Math.round(value * 100)}%`;
}

function relationCountForEntity(relations: KgRelationRow[], entityId: number): number {
  return relations.filter((rel) => rel.source_id === entityId || rel.target_id === entityId).length;
}

function runLayout(
  entities: KgEntityRow[],
  relations: KgRelationRow[],
  width: number,
  height: number,
): Map<number, NodePos> {
  const nodes = new Map<number, NodePos>();
  const centerX = width / 2;
  const centerY = height / 2;
  const radius = Math.min(width, height) * 0.32;

  entities.forEach((entity, index) => {
    const angle = (2 * Math.PI * index) / Math.max(entities.length, 1);
    nodes.set(entity.id, {
      id: entity.id,
      x: centerX + radius * Math.cos(angle) + (Math.random() - 0.5) * 30,
      y: centerY + radius * Math.sin(angle) + (Math.random() - 0.5) * 30,
      vx: 0,
      vy: 0,
    });
  });

  const springLength = 100;
  const repulse = 4200;
  const damp = 0.86;

  for (let iteration = 0; iteration < 120; iteration += 1) {
    const ids = [...nodes.keys()];
    for (let i = 0; i < ids.length; i += 1) {
      for (let j = i + 1; j < ids.length; j += 1) {
        const a = nodes.get(ids[i]);
        const b = nodes.get(ids[j]);
        if (!a || !b) continue;
        const dx = a.x - b.x;
        const dy = a.y - b.y;
        const distance = Math.max(Math.sqrt(dx * dx + dy * dy), 1);
        const force = repulse / (distance * distance);
        a.vx += (dx / distance) * force;
        a.vy += (dy / distance) * force;
        b.vx -= (dx / distance) * force;
        b.vy -= (dy / distance) * force;
      }
    }

    for (const relation of relations) {
      const source = nodes.get(relation.source_id);
      const target = nodes.get(relation.target_id);
      if (!source || !target) continue;
      const dx = target.x - source.x;
      const dy = target.y - source.y;
      const distance = Math.max(Math.sqrt(dx * dx + dy * dy), 1);
      const force = (distance - springLength) * 0.04 * Math.max(relation.weight, 0.3);
      source.vx += (dx / distance) * force;
      source.vy += (dy / distance) * force;
      target.vx -= (dx / distance) * force;
      target.vy -= (dy / distance) * force;
    }

    for (const node of nodes.values()) {
      node.vx += (centerX - node.x) * 0.002;
      node.vy += (centerY - node.y) * 0.002;
      node.vx *= damp;
      node.vy *= damp;
      node.x += node.vx;
      node.y += node.vy;
    }
  }

  return nodes;
}

export default function KnowledgeGraphPage() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const graphViewportRef = useRef<HTMLDivElement>(null);
  const positionsRef = useRef<Map<number, NodePos>>(new Map());
  const animationFrameRef = useRef<number>(0);
  const dragStartRef = useRef({ x: 0, y: 0, panX: 0, panY: 0 });

  const [viewMode, setViewMode] = useState<ViewMode>('graph');
  const [graph, setGraph] = useState<KgGraphData>({ entities: [], relations: [] });
  const [tableRows, setTableRows] = useState<KgEntityTableRow[]>([]);
  const [stats, setStats] = useState<{
    totalEntities: number;
    totalRelations: number;
    byType: Record<string, number>;
    pendingJobs: number;
    completedJobs: number;
    failedJobs: number;
    totalJobs: number;
    progressPercent: number;
  } | null>(null);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState('');
  const [filterType, setFilterType] = useState<string>('ALL');
  const [selectedEntity, setSelectedEntity] = useState<KgEntityRow | null>(null);
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [canvasSize, setCanvasSize] = useState({ width: 1100, height: 720 });

  const loadData = useCallback(async () => {
    setLoading(true);
    try {
      const entityType = filterType === 'ALL' ? undefined : filterType;
      const entityPromise =
        search.trim().length >= 2
          ? KnowledgeGraphService.searchEntities(search, entityType, 220)
          : KnowledgeGraphService.getAllEntities(entityType, 220);

      const [nextStats, entities, nextTableRows] = await Promise.all([
        KnowledgeGraphService.getStats(),
        entityPromise,
        KnowledgeGraphService.getEntityTable(entityType, search, 400),
      ]);

      setStats(nextStats);
      setTableRows(nextTableRows);

      if (entities.length === 0) {
        setGraph({ entities: [], relations: [] });
        setSelectedEntity(null);
        return;
      }

      const relations = await KnowledgeGraphService.getRelationsForEntities(
        entities.map((entity) => entity.id),
        700,
      );

      setGraph({ entities, relations });
      setSelectedEntity((current) =>
        current ? entities.find((entity) => entity.id === current.id) ?? null : null,
      );
    } catch (error) {
      log.error('KnowledgeGraphPage load failed', error);
    } finally {
      setLoading(false);
    }
  }, [filterType, search]);

  useEffect(() => {
    void loadData();
  }, [loadData]);

  useEffect(() => {
    const container = graphViewportRef.current;
    if (!container) return;

    const updateSize = () => {
      const width = Math.max(640, Math.floor(container.clientWidth));
      const height = Math.max(500, Math.floor(container.clientHeight));
      setCanvasSize({ width, height });
    };

    updateSize();
    const observer = new ResizeObserver(updateSize);
    observer.observe(container);
    return () => observer.disconnect();
  }, []);

  const drawGraph = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.save();
    ctx.translate(pan.x, pan.y);
    ctx.scale(zoom, zoom);

    for (const relation of graph.relations) {
      const source = positionsRef.current.get(relation.source_id);
      const target = positionsRef.current.get(relation.target_id);
      if (!source || !target) continue;

      ctx.beginPath();
      ctx.moveTo(source.x, source.y);
      ctx.lineTo(target.x, target.y);
      ctx.strokeStyle = 'rgba(37, 99, 235, 0.22)';
      ctx.lineWidth = 1 + Math.min(relation.weight, 1.5);
      ctx.stroke();

      const angle = Math.atan2(target.y - source.y, target.x - source.x);
      const arrowX = target.x - Math.cos(angle) * 16;
      const arrowY = target.y - Math.sin(angle) * 16;
      ctx.beginPath();
      ctx.moveTo(arrowX, arrowY);
      ctx.lineTo(arrowX - 7 * Math.cos(angle - 0.42), arrowY - 7 * Math.sin(angle - 0.42));
      ctx.lineTo(arrowX - 7 * Math.cos(angle + 0.42), arrowY - 7 * Math.sin(angle + 0.42));
      ctx.closePath();
      ctx.fillStyle = 'rgba(37, 99, 235, 0.34)';
      ctx.fill();

      if (zoom > 0.8 && relation.label) {
        ctx.font = '10px sans-serif';
        ctx.fillStyle = 'rgba(71, 85, 105, 0.95)';
        ctx.textAlign = 'center';
        ctx.fillText(
          relation.label.slice(0, 24),
          (source.x + target.x) / 2,
          (source.y + target.y) / 2 - 5,
        );
      }
    }

    for (const entity of graph.entities) {
      const pos = positionsRef.current.get(entity.id);
      if (!pos) continue;

      const selected = selectedEntity?.id === entity.id;
      const color = ENTITY_COLORS[entity.entity_type] ?? '#64748b';
      const radius = selected ? 15 : 10;

      ctx.shadowColor = color;
      ctx.shadowBlur = selected ? 18 : 8;
      ctx.beginPath();
      ctx.arc(pos.x, pos.y, radius, 0, 2 * Math.PI);
      ctx.fillStyle = color;
      ctx.fill();

      if (selected) {
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 2.5;
        ctx.stroke();
      }

      ctx.shadowBlur = 0;
      if (zoom > 0.55) {
        ctx.font = `${selected ? '600 ' : ''}11px sans-serif`;
        ctx.fillStyle = 'rgba(15, 23, 42, 0.96)';
        ctx.textAlign = 'center';
        const label = entity.name.length > 28 ? `${entity.name.slice(0, 26)}...` : entity.name;
        ctx.fillText(label, pos.x, pos.y + radius + 14);
      }
    }

    ctx.restore();
  }, [graph.entities, graph.relations, pan.x, pan.y, selectedEntity, zoom]);

  useEffect(() => {
    if (graph.entities.length === 0) return;
    positionsRef.current = runLayout(graph.entities, graph.relations, canvasSize.width, canvasSize.height);
    cancelAnimationFrame(animationFrameRef.current);
    animationFrameRef.current = requestAnimationFrame(drawGraph);
  }, [canvasSize.height, canvasSize.width, drawGraph, graph.entities, graph.relations]);

  useEffect(() => {
    cancelAnimationFrame(animationFrameRef.current);
    animationFrameRef.current = requestAnimationFrame(drawGraph);
  }, [drawGraph]);

  const selectedRelations = useMemo(
    () =>
      selectedEntity
        ? graph.relations.filter(
            (relation) =>
              relation.source_id === selectedEntity.id || relation.target_id === selectedEntity.id,
          )
        : [],
    [graph.relations, selectedEntity],
  );

  const neighborEntities = useMemo(() => {
    if (!selectedEntity) return [];
    const ids = new Set<number>();
    for (const relation of selectedRelations) {
      ids.add(relation.source_id);
      ids.add(relation.target_id);
    }
    return graph.entities.filter((entity) => ids.has(entity.id) && entity.id !== selectedEntity.id);
  }, [graph.entities, selectedEntity, selectedRelations]);

  const selectedTableRow = useMemo(() => {
    if (!selectedEntity) return null;
    return tableRows.find((row) => row.id === selectedEntity.id) ?? null;
  }, [selectedEntity, tableRows]);

  const handleCanvasClick = (event: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const mouseX = (event.clientX - rect.left - pan.x) / zoom;
    const mouseY = (event.clientY - rect.top - pan.y) / zoom;

    let hit: KgEntityRow | null = null;
    for (const entity of graph.entities) {
      const pos = positionsRef.current.get(entity.id);
      if (!pos) continue;
      const distance = Math.sqrt((pos.x - mouseX) ** 2 + (pos.y - mouseY) ** 2);
      if (distance <= 16) {
        hit = entity;
        break;
      }
    }

    setSelectedEntity(hit);
  };

  const resetView = () => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  };

  const isEmpty = !loading && graph.entities.length === 0 && tableRows.length === 0;

  return (
    <div className="flex h-full flex-col gap-4 p-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h1 className="text-lg font-semibold text-[var(--text-primary)]">Граф связей участников</h1>
          {stats ? (
            <div className="mt-1 space-y-2">
              <p className="text-xs text-[var(--text-muted)]">
                {stats.totalEntities} сущностей, {stats.totalRelations} связей
                {stats.pendingJobs > 0 ? `, в обработке ${stats.pendingJobs} чанков` : ''}
              </p>
              {stats.pendingJobs > 0 ? (
                <div className="max-w-[360px]">
                  <div className="mb-1 flex items-center justify-between text-[11px] text-[var(--app-text-muted)]">
                    <span>Анализ продолжается</span>
                    <span>
                      {stats.totalJobs > 0 ? `${stats.completedJobs}/${stats.totalJobs}` : `${stats.pendingJobs} active`}
                    </span>
                  </div>
                  <div className="h-1.5 overflow-hidden rounded-full bg-black/[0.06]">
                    <div
                      className="h-full rounded-full bg-[var(--app-accent)] transition-[width] duration-500 ease-out"
                      style={{ width: `${Math.max(stats.progressPercent, 8)}%` }}
                    />
                  </div>
                </div>
              ) : null}
            </div>
          ) : null}
        </div>

        <div className="flex items-center gap-2">
          <SegmentedControl
            active={viewMode}
            onChange={(id) => setViewMode(id as ViewMode)}
            segments={[...VIEW_SEGMENTS]}
          />
          <button
            type="button"
            onClick={() => void loadData()}
            className="inline-flex h-8 items-center gap-1.5 rounded-[8px] border border-[var(--app-border)] bg-[var(--app-surface)] px-3 text-[length:var(--text-xs)] font-medium text-[var(--app-text-secondary)] transition-colors hover:bg-[var(--app-surface-soft)]"
          >
            <RefreshCw size={13} />
            Обновить
          </button>
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-3">
        <div className="relative min-w-[260px] flex-1">
          <Search
            size={14}
            className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-[var(--app-text-muted)]"
          />
          <input
            value={search}
            onChange={(event) => setSearch(event.target.value)}
            placeholder="Поиск участников..."
            className="h-9 w-full rounded-[10px] border border-[var(--app-border)] bg-[var(--app-surface)] py-2 pl-9 pr-3 text-sm text-[var(--app-text)] outline-none transition-colors focus:border-[var(--app-accent)]"
          />
        </div>

        <SegmentedControl
          active={filterType}
          onChange={setFilterType}
          segments={TYPE_SEGMENTS.map((segment) => ({
            ...segment,
            label:
              segment.id === 'ALL'
                ? `Все (${stats?.totalEntities ?? 0})`
                : `${segment.label} (${stats?.byType[segment.id] ?? 0})`,
          }))}
        />

        {viewMode === 'graph' ? (
          <div className="ml-auto flex items-center gap-1">
            <button
              type="button"
              onClick={() => setZoom((value) => Math.min(3, value + 0.2))}
              className="rounded-[8px] border border-[var(--app-border)] p-2 transition-colors hover:bg-[var(--app-surface-soft)]"
            >
              <ZoomIn size={13} />
            </button>
            <button
              type="button"
              onClick={() => setZoom((value) => Math.max(0.2, value - 0.2))}
              className="rounded-[8px] border border-[var(--app-border)] p-2 transition-colors hover:bg-[var(--app-surface-soft)]"
            >
              <ZoomOut size={13} />
            </button>
            <button
              type="button"
              onClick={resetView}
              className="rounded-[8px] border border-[var(--app-border)] px-3 py-2 text-[length:var(--text-xs)] font-medium transition-colors hover:bg-[var(--app-surface-soft)]"
            >
              Сбросить
            </button>
          </div>
        ) : null}
      </div>

      <div className="flex flex-wrap gap-4 text-[11px] text-[var(--app-text-secondary)]">
        <span className="flex items-center gap-1.5">
          <span className="inline-block h-2.5 w-2.5 rounded-full bg-[#2563eb]" />
          Лица
        </span>
        <span className="flex items-center gap-1.5">
          <span className="inline-block h-2.5 w-2.5 rounded-full bg-[#0f766e]" />
          Организации
        </span>
      </div>

      {viewMode === 'graph' ? (
        <div className="flex min-h-0 flex-1 gap-4">
          <div
            ref={graphViewportRef}
            className="relative min-h-[520px] flex-1 overflow-hidden rounded-[20px] border border-[var(--app-border)] bg-[var(--app-surface)]"
          >
            {loading ? (
              <div className="absolute inset-0 z-10 flex items-center justify-center bg-[var(--app-surface)]/80">
                <div className="flex flex-col items-center gap-2">
                  <div className="h-7 w-7 animate-spin rounded-full border-2 border-[var(--app-accent)] border-t-transparent" />
                  <span className="text-xs text-[var(--app-text-muted)]">Загрузка графа...</span>
                </div>
              </div>
            ) : null}

            {isEmpty ? (
              <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 px-6 text-center">
                <Info size={30} className="text-[var(--app-text-muted)]" strokeWidth={1.6} />
                <div>
                  <p className="text-sm font-medium text-[var(--app-text-secondary)]">
                    Пока нет участников для отображения
                  </p>
                  <p className="mt-1 text-xs text-[var(--app-text-muted)]">
                    После индексации документов здесь появятся люди и организации, найденные в чанках.
                  </p>
                </div>
              </div>
            ) : null}

            <canvas
              ref={canvasRef}
              width={canvasSize.width}
              height={canvasSize.height}
              className="h-full w-full"
              style={{ cursor: isDragging ? 'grabbing' : 'grab' }}
              onClick={handleCanvasClick}
              onMouseDown={(event) => {
                setIsDragging(true);
                dragStartRef.current = {
                  x: event.clientX,
                  y: event.clientY,
                  panX: pan.x,
                  panY: pan.y,
                };
              }}
              onMouseMove={(event) => {
                if (!isDragging) return;
                setPan({
                  x: dragStartRef.current.panX + (event.clientX - dragStartRef.current.x),
                  y: dragStartRef.current.panY + (event.clientY - dragStartRef.current.y),
                });
              }}
              onMouseUp={() => setIsDragging(false)}
              onMouseLeave={() => setIsDragging(false)}
              onWheel={(event) => {
                event.preventDefault();
                setZoom((value) => Math.min(3, Math.max(0.2, value - event.deltaY * 0.001)));
              }}
            />
          </div>

          <aside className="flex w-[330px] shrink-0 flex-col gap-3 overflow-y-auto">
            {selectedEntity ? (
              <>
                <section className="rounded-[20px] border border-[var(--app-border)] bg-[var(--app-surface)] p-4">
                  <div className="flex items-start gap-3">
                    <span
                      className="mt-1 inline-flex h-8 w-8 items-center justify-center rounded-full text-white"
                      style={{ backgroundColor: ENTITY_COLORS[selectedEntity.entity_type] ?? '#64748b' }}
                    >
                      {selectedEntity.entity_type === 'PERSON' ? <Users size={16} /> : <Building2 size={16} />}
                    </span>
                    <div className="min-w-0">
                      <p className="break-words text-sm font-semibold text-[var(--app-text)]">{selectedEntity.name}</p>
                      <p className="mt-0.5 text-[11px] text-[var(--app-text-muted)]">
                        {ENTITY_LABELS[selectedEntity.entity_type] ?? selectedEntity.entity_type}
                      </p>
                    </div>
                  </div>

                  {selectedEntity.description ? (
                    <p className="mt-3 border-t border-[var(--app-border)] pt-3 text-xs text-[var(--app-text-secondary)]">
                      {selectedEntity.description}
                    </p>
                  ) : null}

                  <div className="mt-4 grid grid-cols-2 gap-2 text-[11px]">
                    <div className="rounded-[14px] bg-black/[0.03] p-2.5">
                      <p className="text-[var(--app-text-muted)]">Уверенность</p>
                      <p className="mt-1 font-medium text-[var(--app-text)]">{formatPercent(selectedEntity.confidence)}</p>
                    </div>
                    <div className="rounded-[14px] bg-black/[0.03] p-2.5">
                      <p className="text-[var(--app-text-muted)]">Связи</p>
                      <p className="mt-1 font-medium text-[var(--app-text)]">
                        {selectedTableRow?.relation_count ?? relationCountForEntity(graph.relations, selectedEntity.id)}
                      </p>
                    </div>
                    <div className="rounded-[14px] bg-black/[0.03] p-2.5">
                      <p className="text-[var(--app-text-muted)]">Документы</p>
                      <p className="mt-1 font-medium text-[var(--app-text)]">{selectedTableRow?.document_count ?? 0}</p>
                    </div>
                    <div className="rounded-[14px] bg-black/[0.03] p-2.5">
                      <p className="text-[var(--app-text-muted)]">Чанки</p>
                      <p className="mt-1 font-medium text-[var(--app-text)]">{selectedTableRow?.chunk_count ?? 0}</p>
                    </div>
                  </div>
                </section>

                <section className="rounded-[20px] border border-[var(--app-border)] bg-[var(--app-surface)] p-4">
                  <p className="mb-2 text-[11px] font-semibold uppercase tracking-wider text-[var(--app-text-muted)]">
                    Связи ({selectedRelations.length})
                  </p>
                  <div className="space-y-2">
                    {selectedRelations.length === 0 ? (
                      <p className="text-xs text-[var(--app-text-muted)]">Для этой сущности связи пока не найдены.</p>
                    ) : (
                      selectedRelations.slice(0, 12).map((relation) => {
                        const outgoing = relation.source_id === selectedEntity.id;
                        const otherName = outgoing ? relation.target_name : relation.source_name;
                        return (
                          <div
                            key={relation.id}
                            className="rounded-[14px] border border-[var(--app-border)] px-3 py-2 text-xs"
                          >
                            <div className="flex items-center gap-2">
                              <span className="font-mono text-[var(--app-accent)]">{outgoing ? '->' : '<-'}</span>
                              <span className="min-w-0 flex-1 truncate text-[var(--app-text-secondary)]">{otherName}</span>
                            </div>
                            <p className="mt-1 text-[11px] text-[var(--app-text-muted)]">
                              {relation.label || relation.relation_type}
                            </p>
                          </div>
                        );
                      })
                    )}
                  </div>
                </section>

                <section className="rounded-[20px] border border-[var(--app-border)] bg-[var(--app-surface)] p-4">
                  <p className="mb-2 text-[11px] font-semibold uppercase tracking-wider text-[var(--app-text-muted)]">
                    Соседи
                  </p>
                  <div className="space-y-1.5">
                    {neighborEntities.length === 0 ? (
                      <p className="text-xs text-[var(--app-text-muted)]">Связанные участники не найдены.</p>
                    ) : (
                      neighborEntities.slice(0, 10).map((entity) => (
                        <button
                          key={entity.id}
                          type="button"
                          onClick={() => setSelectedEntity(entity)}
                          className="flex w-full items-center gap-2 rounded-[12px] px-2 py-1.5 text-left transition-colors hover:bg-[var(--app-surface-soft)]"
                        >
                          <span
                            className="inline-block h-2.5 w-2.5 rounded-full"
                            style={{ backgroundColor: ENTITY_COLORS[entity.entity_type] ?? '#64748b' }}
                          />
                          <span className="truncate text-xs text-[var(--app-text-secondary)]">{entity.name}</span>
                        </button>
                      ))
                    )}
                  </div>
                </section>
              </>
            ) : (
              <section className="rounded-[20px] border border-[var(--app-border)] bg-[var(--app-surface)] p-4 text-center">
                <p className="text-xs text-[var(--app-text-muted)]">
                  Выберите узел графа, чтобы посмотреть его роль, покрытие по документам и связи с другими участниками.
                </p>
              </section>
            )}
          </aside>
        </div>
      ) : (
        <div className="min-h-0 flex-1 overflow-hidden rounded-[20px] border border-[var(--app-border)] bg-[var(--app-surface)]">
          <div className="h-full overflow-auto">
            <table className="w-full min-w-[900px] text-left text-sm">
              <thead className="sticky top-0 z-10 border-b border-[var(--app-border)] bg-[var(--app-surface)]">
                <tr>
                  <th className="px-4 py-3 text-[11px] font-semibold uppercase tracking-wider text-[var(--app-text-muted)]">Сущность</th>
                  <th className="px-4 py-3 text-[11px] font-semibold uppercase tracking-wider text-[var(--app-text-muted)]">Тип</th>
                  <th className="px-4 py-3 text-[11px] font-semibold uppercase tracking-wider text-[var(--app-text-muted)]">Уверенность</th>
                  <th className="px-4 py-3 text-[11px] font-semibold uppercase tracking-wider text-[var(--app-text-muted)]">Документы</th>
                  <th className="px-4 py-3 text-[11px] font-semibold uppercase tracking-wider text-[var(--app-text-muted)]">Чанки</th>
                  <th className="px-4 py-3 text-[11px] font-semibold uppercase tracking-wider text-[var(--app-text-muted)]">Связи</th>
                  <th className="px-4 py-3 text-[11px] font-semibold uppercase tracking-wider text-[var(--app-text-muted)]">Описание</th>
                  <th className="px-4 py-3 text-[11px] font-semibold uppercase tracking-wider text-[var(--app-text-muted)]">Действие</th>
                </tr>
              </thead>
              <tbody>
                {loading ? (
                  <tr>
                    <td colSpan={8} className="px-4 py-12 text-center text-sm text-[var(--app-text-muted)]">
                      Загрузка сущностей...
                    </td>
                  </tr>
                ) : tableRows.length === 0 ? (
                  <tr>
                    <td colSpan={8} className="px-4 py-12 text-center text-sm text-[var(--app-text-muted)]">
                      По текущему фильтру участники не найдены.
                    </td>
                  </tr>
                ) : (
                  tableRows.map((row) => (
                    <tr
                      key={row.id}
                      className="border-b border-[var(--app-border)]/70 transition-colors hover:bg-black/[0.02]"
                    >
                      <td className="px-4 py-3">
                        <div className="flex items-start gap-2">
                          <span
                            className="mt-1 inline-block h-2.5 w-2.5 rounded-full"
                            style={{ backgroundColor: ENTITY_COLORS[row.entity_type] ?? '#64748b' }}
                          />
                          <div>
                            <p className="font-medium text-[var(--app-text)]">{row.name}</p>
                            <p className="text-[11px] text-[var(--app-text-muted)]">{row.canonical_name}</p>
                          </div>
                        </div>
                      </td>
                      <td className="px-4 py-3 text-sm text-[var(--app-text-secondary)]">
                        {ENTITY_LABELS[row.entity_type] ?? row.entity_type}
                      </td>
                      <td className="px-4 py-3 text-sm text-[var(--app-text-secondary)]">{formatPercent(row.confidence)}</td>
                      <td className="px-4 py-3 text-sm text-[var(--app-text-secondary)]">{row.document_count}</td>
                      <td className="px-4 py-3 text-sm text-[var(--app-text-secondary)]">{row.chunk_count}</td>
                      <td className="px-4 py-3 text-sm text-[var(--app-text-secondary)]">{row.relation_count}</td>
                      <td className="max-w-[340px] px-4 py-3 text-sm text-[var(--app-text-secondary)]">
                        <div className="line-clamp-2">{row.description || '—'}</div>
                      </td>
                      <td className="px-4 py-3">
                        <button
                          type="button"
                          onClick={() => {
                            setSelectedEntity(row);
                            setViewMode('graph');
                          }}
                          className="rounded-[8px] border border-[var(--app-border)] px-3 py-1.5 text-xs font-medium text-[var(--app-text-secondary)] transition-colors hover:bg-[var(--app-surface-soft)]"
                        >
                          Открыть в графе
                        </button>
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
