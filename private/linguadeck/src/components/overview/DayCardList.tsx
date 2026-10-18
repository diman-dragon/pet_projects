import { useState, useEffect } from "react";
import { ChevronDown, ChevronRight, Eye, Clock } from "lucide-react";
import { StudyService } from "../../services/db/StudyService";

interface DayCardListProps {
  date: string;
}

// Matches what the backend actually returns from views_for_day
interface RawViewEntry {
  id: number;
  card_id: number;
  viewed_at: string;
  time_spent_ms: number;
  front: string;
  back: string;
  deck_id: number;
  deck_name: string;
}

interface GroupedCard {
  card_id: number;
  front: string;
  back: string;
  views: number;
  totalTimeMs: number;
}

interface GroupedDeck {
  deckName: string;
  cards: GroupedCard[];
  totalViews: number;
}

export function DayCardList({ date }: DayCardListProps) {
  const [grouped, setGrouped] = useState<GroupedDeck[]>([]);
  const [totalViews, setTotalViews] = useState(0);
  const [loading, setLoading] = useState(true);
  const [expandedDecks, setExpandedDecks] = useState<Set<string>>(new Set());

  useEffect(() => {
    setLoading(true);
    StudyService.getViewsForDay(date).then((rawData) => {
      const rows = rawData as unknown as RawViewEntry[];

      // Group by deck, then by card
      const deckMap = new Map<string, Map<number, GroupedCard>>();
      for (const row of rows) {
        if (!deckMap.has(row.deck_name)) {
          deckMap.set(row.deck_name, new Map());
        }
        const cardMap = deckMap.get(row.deck_name)!;
        if (!cardMap.has(row.card_id)) {
          cardMap.set(row.card_id, {
            card_id: row.card_id,
            front: row.front,
            back: row.back,
            views: 0,
            totalTimeMs: 0,
          });
        }
        const card = cardMap.get(row.card_id)!;
        card.views += 1;
        card.totalTimeMs += row.time_spent_ms || 0;
      }

      const result: GroupedDeck[] = [];
      for (const [deckName, cardMap] of deckMap.entries()) {
        const cards = Array.from(cardMap.values());
        result.push({
          deckName,
          cards,
          totalViews: cards.reduce((s, c) => s + c.views, 0),
        });
      }

      setGrouped(result);
      setTotalViews(rows.length);
      setLoading(false);
    }).catch(() => setLoading(false));
  }, [date]);

  const toggleDeck = (deckName: string) => {
    setExpandedDecks((prev) => {
      const next = new Set(prev);
      if (next.has(deckName)) next.delete(deckName);
      else next.add(deckName);
      return next;
    });
  };

  if (loading) {
    return (
      <div className="rounded-xl bg-white p-4 shadow-sm ring-1 ring-gray-200">
        <div className="h-4 w-32 animate-pulse rounded bg-gray-100" />
        <div className="mt-3 space-y-2">
          {[...Array(3)].map((_, i) => (
            <div key={i} className="h-10 animate-pulse rounded-lg bg-gray-50" />
          ))}
        </div>
      </div>
    );
  }

  if (grouped.length === 0) {
    return (
      <div className="rounded-xl bg-white p-4 shadow-sm ring-1 ring-gray-200">
        <p className="text-sm text-gray-500">Нет показанных карточек за этот день</p>
      </div>
    );
  }

  return (
    <div className="rounded-xl bg-white p-3 shadow-sm ring-1 ring-gray-200">
      <div className="mb-2 flex items-center justify-between">
        <span className="text-xs font-medium text-gray-500">
          {totalViews} показов в {grouped.length} {grouped.length === 1 ? "колоде" : "колодах"}
        </span>
      </div>

      <div className="space-y-1">
        {grouped.map((group) => {
          const isExpanded = expandedDecks.has(group.deckName);
          return (
            <div key={group.deckName} className="rounded-lg border border-gray-100 overflow-hidden">
              <button
                onClick={() => toggleDeck(group.deckName)}
                className="flex w-full items-center justify-between px-3 py-2 hover:bg-gray-50"
                type="button"
              >
                <div className="flex items-center gap-2">
                  <div className="h-2 w-2 rounded-full bg-blue-400" />
                  <span className="text-sm font-medium text-gray-700">{group.deckName}</span>
                  <span className="text-xs text-gray-400">({group.cards.length} карт, {group.totalViews} показов)</span>
                </div>
                {isExpanded ? (
                  <ChevronDown size={14} className="text-gray-400" />
                ) : (
                  <ChevronRight size={14} className="text-gray-400" />
                )}
              </button>

              {isExpanded && (
                <div className="border-t border-gray-100 bg-gray-50/50">
                  {group.cards.map((card) => (
                    <div
                      key={card.card_id}
                      className="flex items-center justify-between px-3 py-2 text-sm border-b border-gray-50 last:border-0"
                    >
                      <div className="flex-1 min-w-0">
                        <p className="truncate text-gray-700 font-medium">{card.front}</p>
                        <p className="truncate text-xs text-gray-400">{card.back}</p>
                      </div>
                      <div className="ml-2 flex items-center gap-3 shrink-0">
                        <span className="flex items-center gap-1 text-[10px] text-gray-400">
                          <Eye size={10} />
                          {card.views}
                        </span>
                        {card.totalTimeMs > 0 && (
                          <span className="flex items-center gap-1 text-[10px] text-gray-400">
                            <Clock size={10} />
                            {Math.round(card.totalTimeMs / 1000)}с
                          </span>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
