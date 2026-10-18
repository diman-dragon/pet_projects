import { useEffect, useState } from "react";
import { Clock, Eye, RefreshCw, TrendingUp, X, Zap } from "lucide-react";
import { StudyService, type DayStats } from "../../../services/db/StudyService";
import { Chip, RATING_META, fmtInterval, fmtMs, formatDateRU } from "./forecastUtils";

export function DayPanel({ date, onClose }: { date: string; onClose: () => void }) {
  const [data, setData] = useState<DayStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState("");

  useEffect(() => {
    setLoading(true);
    setSearch("");
    StudyService.getDayStats(date)
      .then(setData)
      .catch(() => setData(null))
      .finally(() => setLoading(false));
  }, [date]);

  const s = data?.summary;
  const rv = data?.reviews;

  const totalRatings = rv ? rv.rating1 + rv.rating2 + rv.rating3 + rv.rating4 : 0;
  const successRate =
    totalRatings > 0
      ? Math.round(((rv!.rating3 + rv!.rating4) / totalRatings) * 100)
      : null;

  const filteredCards = (data?.cards ?? []).filter(
    (c) =>
      search === "" ||
      c.front.toLowerCase().includes(search.toLowerCase()) ||
      c.back.toLowerCase().includes(search.toLowerCase()) ||
      c.deckName.toLowerCase().includes(search.toLowerCase())
  );

  const byDeck: Record<string, typeof filteredCards> = {};
  for (const c of filteredCards) {
    if (!byDeck[c.deckName]) byDeck[c.deckName] = [];
    byDeck[c.deckName].push(c);
  }

  return (
    <div className="flex flex-col h-full bg-white">
      {/* Header */}
      <div className="shrink-0 px-5 pt-4 pb-3 border-b border-gray-100">
        <div className="flex items-start justify-between mb-1">
          <div>
            <div className="text-sm font-bold text-gray-900">{formatDateRU(date)}</div>
            <div className="text-[11px] text-gray-400">
              {loading
                ? "загрузка..."
                : data?.summary.uniqueCards
                ? `${data.summary.uniqueCards} уникальных карточек`
                : "нет данных"}
            </div>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="flex h-7 w-7 items-center justify-center rounded-lg text-gray-400 hover:bg-gray-100 hover:text-gray-600 transition-all"
          >
            <X size={14} />
          </button>
        </div>
      </div>

      {loading ? (
        <div className="flex flex-1 items-center justify-center">
          <div className="h-6 w-6 animate-spin rounded-full border-2 border-[#0078D4] border-t-transparent" />
        </div>
      ) : !data || data.summary.uniqueCards === 0 ? (
        <div className="flex flex-1 flex-col items-center justify-center gap-2 text-center px-6">
          <div className="text-3xl">😴</div>
          <p className="text-sm text-gray-400">В этот день занятий не было</p>
        </div>
      ) : (
        <div className="flex-1 overflow-y-auto">
          {/* Summary stats */}
          <div className="px-4 pt-3 pb-2 grid grid-cols-2 gap-2">
            <Chip icon={<Eye size={13} />}        value={s!.uniqueCards}                          label="карточек (уник.)"    color="text-slate-700"   bg="bg-slate-50"   />
            <Chip icon={<RefreshCw size={13} />}   value={s!.totalFlips}                           label="переворотов"         color="text-indigo-700"  bg="bg-indigo-50"  />
            <Chip icon={<Clock size={13} />}       value={fmtMs(s!.avgFrontMs)}                    label="среднее до пер-та"   color="text-amber-700"   bg="bg-amber-50"   />
            <Chip icon={<Clock size={13} />}       value={fmtMs(s!.avgBackMs)}                     label="среднее с пер-том"   color="text-cyan-700"    bg="bg-cyan-50"    />
            <Chip icon={<Zap size={13} />}         value={rv!.totalReviews}                        label="повторений"          color="text-blue-700"    bg="bg-blue-50"    />
            <Chip icon={<TrendingUp size={13} />}  value={successRate != null ? `${successRate}%` : "—"} label="успешно (3+4)"  color="text-emerald-700" bg="bg-emerald-50" />
          </div>

          {/* Rating bar */}
          {totalRatings > 0 && (
            <div className="px-4 pb-3">
              <div className="text-[10px] font-semibold text-gray-400 uppercase tracking-wide mb-1.5">Оценки</div>
              <div className="h-3 w-full flex rounded-full overflow-hidden gap-px">
                {[4, 3, 2, 1].map((r) => {
                  const rm = RATING_META[r as keyof typeof RATING_META];
                  const n = rv![`rating${r}` as keyof typeof rv] as number;
                  const pct = totalRatings > 0 ? (n / totalRatings) * 100 : 0;
                  if (pct < 1) return null;
                  return <div key={r} className={`h-full ${rm.bar} transition-all`} style={{ width: `${pct}%` }} title={`${rm.label}: ${n}`} />;
                })}
              </div>
              <div className="mt-1.5 flex flex-wrap gap-x-3 gap-y-0.5">
                {[4, 3, 2, 1].map((r) => {
                  const rm = RATING_META[r as keyof typeof RATING_META];
                  const n = rv![`rating${r}` as keyof typeof rv] as number;
                  if (n === 0) return null;
                  return (
                    <span key={r} className={`text-[11px] font-medium ${rm.color}`}>
                      {rm.label}: {n}
                    </span>
                  );
                })}
              </div>
            </div>
          )}

          {/* Search */}
          <div className="px-4 pb-2">
            <input
              type="text"
              placeholder="Поиск карточки..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="w-full rounded-xl border border-gray-200 bg-gray-50 px-3 py-1.5 text-[12px] outline-none focus:border-[#0078D4]/50 focus:ring-1 focus:ring-[#0078D4]/20 transition-all placeholder:text-gray-300"
            />
          </div>

          {/* Cards by deck */}
          {Object.entries(byDeck).map(([deckName, cards]) => (
            <div key={deckName} className="px-4 pb-3">
              <div className="flex items-center gap-2 mb-2 sticky top-0 bg-white/90 backdrop-blur-sm py-1">
                <div className="h-2 w-2 rounded-full shrink-0" style={{ backgroundColor: cards[0]?.deckColor || "#6b7280" }} />
                <span className="text-[11px] font-bold uppercase tracking-wide text-gray-500">{deckName}</span>
                <span className="text-[10px] text-gray-400">({cards.length})</span>
              </div>
              <div className="space-y-1.5">
                {cards.map((card) => {
                  const rm = card.lastRating ? RATING_META[card.lastRating as keyof typeof RATING_META] : null;
                  return (
                    <div
                      key={card.cardId}
                      className="flex items-start gap-2 rounded-xl border border-gray-100 bg-white px-3 py-2 hover:border-gray-200 hover:shadow-sm transition-all"
                    >
                      <div className="shrink-0 mt-0.5">
                        {rm ? (
                          <span className={`inline-flex h-5 w-5 items-center justify-center rounded-full text-[10px] font-black ${rm.bg} ${rm.color}`}>
                            {rm.short}
                          </span>
                        ) : (
                          <span className="inline-flex h-5 w-5 items-center justify-center rounded-full bg-gray-100 text-[10px] text-gray-400">—</span>
                        )}
                      </div>
                      <div className="min-w-0 flex-1">
                        <div className="text-[12px] font-semibold text-gray-900 leading-tight truncate">{card.front}</div>
                        <div className="text-[11px] text-gray-400 leading-tight truncate">{card.back}</div>
                      </div>
                      <div className="shrink-0 text-right space-y-0.5">
                        <div className="flex items-center justify-end gap-1 text-[10px] text-gray-400">
                          <Eye size={9} /><span>{card.shows}</span>
                          <RefreshCw size={9} /><span>{card.flips}</span>
                        </div>
                        <div className="text-[10px] text-gray-400">{fmtMs(card.totalViewMs)}</div>
                        {card.intervalAfter != null && (
                          <div className="text-[10px] text-blue-400">→ {fmtInterval(card.intervalAfter)}</div>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
