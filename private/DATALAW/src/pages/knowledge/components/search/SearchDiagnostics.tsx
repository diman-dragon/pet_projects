// src/pages/knowledge/components/search/SearchDiagnostics.tsx
// Диагностика — убери когда семантика заработает стабильно.

import { useEffect, useState } from 'react';
import { ConnectionManager } from '../../../../services/db/ConnectionManager';
import { EmbeddingModelService } from '../../../../services/ai/EmbeddingModelService';
import { EmbeddingService } from '../../../../services/rag/EmbeddingService';

interface DiagState {
  embeddingModel: string;
  modelAvailable: boolean;
  embeddingsCount: number;
  modelInDB: string;
  vecTableDim: number | null;
  embeddingDim: number | null;
  capability: string;
  knnTestCount: number;
  knnTestError: string | null;
  knnTestModels: string[];
}

export function SearchDiagnostics() {
  const [d, setD] = useState<DiagState | null>(null);
  const [open, setOpen] = useState(false);

  const run = async () => {
    setD(null);
    try {
      const model = await EmbeddingModelService.get();
      const modelAvailable = await EmbeddingService.isModelAvailable(model);
      const db = ConnectionManager.get('kb_main');

      const cntRows = await db.select(`SELECT COUNT(*) AS cnt FROM embeddings`) as any[];
      const embeddingsCount = cntRows?.[0]?.cnt ?? 0;

      const modelRows = await db.select(
        `SELECT model FROM embeddings GROUP BY model ORDER BY COUNT(*) DESC LIMIT 1`
      ) as any[];
      const modelInDB = modelRows?.[0]?.model ?? '(нет)';

      let vecTableDim: number | null = null;
      try {
        const sqlRows = await db.select(
          `SELECT sql FROM sqlite_master WHERE name='vec_embeddings' LIMIT 1`
        ) as any[];
        const sql = sqlRows?.[0]?.sql ?? '';
        const m = sql.match(/FLOAT\[(\d+)\]/i);
        vecTableDim = m ? Number(m[1]) : null;
      } catch { /* vec0 not available */ }

      let embeddingDim: number | null = null;
      if (modelAvailable) {
        try {
          const res = await EmbeddingService.embed('тест запроса', model, true);
          embeddingDim = res.dim;
        } catch { /* ignore */ }
      }

      let capability = 'unknown';
      try {
        await db.select(`SELECT vec_version()`);
        try { await db.select(`SELECT count(*) FROM vec_embeddings LIMIT 1`); capability = 'vec0'; }
        catch { capability = 'vec_scalar'; }
      } catch { capability = 'js_fallback'; }

      // Тест реального KNN запроса (как в исправленном vec0.ts)
      let knnTestCount = 0;
      let knnTestError: string | null = null;
      let knnTestModels: string[] = [];
      if (modelAvailable && embeddingDim) {
        try {
          const testVec = await EmbeddingService.embed('договор аренда', model, true);
          const vecJson = JSON.stringify(Array.from(testVec.vector));
          const knnRows = await db.select(
            `SELECT chunk_id, model, distance FROM vec_embeddings WHERE embedding MATCH $1 AND k = $2`,
            [vecJson, 20],
          ) as any[];
          knnTestCount = knnRows?.length ?? 0;
          knnTestModels = [...new Set((knnRows ?? []).map((r: any) => r.model as string))];
          // Показываем диапазон distance чтобы понять score
          if (knnRows?.length) {
            const distances = knnRows.map((r: any) => Number(r.distance));
            const minD = Math.min(...distances).toFixed(4);
            const maxD = Math.max(...distances).toFixed(4);
            const scoreMin = (1 - (Number(maxD)**2)/2).toFixed(3);
            const scoreMax = (1 - (Number(minD)**2)/2).toFixed(3);
            knnTestError = `L2: ${minD}..${maxD} → score: ${scoreMin}..${scoreMax}`;
          }
        } catch (e) {
          knnTestError = String(e);
        }
      }

      setD({
        embeddingModel: model, modelAvailable, embeddingsCount, modelInDB,
        vecTableDim, embeddingDim, capability,
        knnTestCount, knnTestError, knnTestModels,
      });
    } catch (e) {
      console.error('[SearchDiagnostics]', e);
    }
  };

  useEffect(() => { if (open) run(); }, [open]);

  if (!open) return (
    <button type="button" onClick={() => setOpen(true)}
      className="w-full text-left px-3 py-1.5 text-[10px] text-black/30 hover:text-black/50 border-t border-black/[0.05] transition-colors">
      🔍 диагностика поиска
    </button>
  );

  const row = (label: string, value: string, warn = false) => (
    <div key={label} className="flex justify-between gap-2 py-0.5">
      <span className="text-black/40 shrink-0">{label}</span>
      <span className={`font-mono text-right text-[9px] ${warn ? 'text-red-500 font-bold' : 'text-black/70'}`}>{value}</span>
    </div>
  );

  return (
    <div className="mx-3 mb-3 rounded-lg border border-black/[0.08] bg-white p-3 text-[10px]">
      <div className="flex items-center justify-between mb-2">
        <span className="font-bold text-black/50 uppercase tracking-wide text-[9px]">Диагностика</span>
        <button onClick={run} className="text-black/30 hover:text-black/60 text-[9px]">↻ обновить</button>
      </div>
      {!d ? <div className="text-black/30 py-1">Загрузка...</div> : (
        <div className="flex flex-col">
          {row('Модель', d.embeddingModel)}
          {row('Ollama', d.modelAvailable ? '✅ доступна' : '❌ недоступна', !d.modelAvailable)}
          {row('Эмбеддингов в БД', String(d.embeddingsCount), d.embeddingsCount === 0)}
          {row('Модель в БД', d.modelInDB, d.modelInDB !== d.embeddingModel)}
          {d.modelInDB !== d.embeddingModel && row('⚠️', 'РАЗНЫЕ — нужен реиндекс!', true)}
          {row('Capability', d.capability)}
          {d.vecTableDim !== null && row('Vec dim', String(d.vecTableDim), d.vecTableDim !== d.embeddingDim)}
          {d.embeddingDim !== null && row('Модель dim', String(d.embeddingDim))}
          <div className="mt-2 pt-2 border-t border-black/[0.06]">
            <div className="text-[9px] font-bold text-black/40 uppercase mb-1">KNN тест (договор аренда)</div>
            {d.knnTestError && d.knnTestCount === 0
              ? row('Ошибка KNN', d.knnTestError, true)
              : row('KNN результатов', String(d.knnTestCount), d.knnTestCount === 0)}
            {d.knnTestModels.length > 0 && row('Модели в vec', d.knnTestModels.join(', '))}
            {d.knnTestCount > 0 && d.knnTestError && row('Диапазон', d.knnTestError)}
          </div>
        </div>
      )}
    </div>
  );
}
