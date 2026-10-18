import { BookOpenText, Brain, Heart, Layers3, LibraryBig, ListChecks, Settings2, Sparkles } from "lucide-react";
import { useTranslation } from "react-i18next";

export type ViewId = "dashboard" | "decks" | "cards" | "study" | "grammar" | "settings" | "donate";

interface SidebarNavProps {
  activeView: ViewId;
  onChangeView: (view: ViewId) => void;
}

export function SidebarNav(props: SidebarNavProps) {
  const { t } = useTranslation();

  const navItems: Array<{ id: ViewId; label: string; icon: typeof LibraryBig }> = [
    { id: "dashboard", label: t("sidebar.progress"), icon: LibraryBig },
    { id: "decks", label: t("sidebar.decks"), icon: Layers3 },
    { id: "cards", label: t("sidebar.cards"), icon: ListChecks },
    { id: "study", label: t("sidebar.study"), icon: Brain },
    { id: "grammar", label: t("sidebar.grammar"), icon: BookOpenText },
  ];

  return (
    <aside className="fixed left-0 top-0 z-50 flex h-screen w-60 flex-col overflow-hidden bg-gradient-to-b from-white to-slate-50 p-4 shadow-2xl shadow-gray-200/50">
      <div className="mb-5">
        <div className="relative overflow-hidden rounded-2xl bg-gradient-to-br from-[#0078D4] via-[#106EBE] to-[#005A9E] p-5 text-white shadow-xl shadow-blue-500/30">
          <div className="absolute -right-4 -top-4 h-24 w-24 rounded-full bg-white/10 blur-2xl" />
          <div className="absolute -bottom-4 -left-4 h-16 w-16 rounded-full bg-white/10 blur-xl" />

          <div className="relative flex items-center gap-3">
            <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-gradient-to-br from-white/30 to-white/10 shadow-inner ring-1 ring-white/20 backdrop-blur-sm">
              <Sparkles size={22} className="text-white" />
            </div>

            <div>
              <div className="flex items-baseline gap-1">
                <span className="text-xl font-bold tracking-tight">Lingua</span>
                <span className="text-xl font-bold tracking-tight text-blue-200">Deck</span>
              </div>
              <div className="flex items-center gap-1.5 text-xs font-medium text-blue-100">
                <Brain size={12} />
                <span>{t("sidebar.tagline")}</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      <nav className="flex flex-col gap-2">
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = props.activeView === item.id;
          return (
            <button
              key={item.id}
              onClick={() => props.onChangeView(item.id)}
              className={`group flex items-center gap-3 rounded-xl px-4 py-3 text-sm font-medium transition-all duration-200 ${isActive ? "bg-gradient-to-r from-[#0078D4] to-[#106EBE] text-white shadow-lg shadow-blue-500/25" : "bg-white/80 text-gray-600 shadow-sm hover:bg-white hover:shadow-md"}`}
              type="button"
            >
              <div className={`flex h-8 w-8 items-center justify-center rounded-lg transition-all ${isActive ? "bg-white/20" : "bg-gray-100 group-hover:bg-gray-200"}`}>
                <Icon size={18} />
              </div>
              <span>{item.label}</span>
            </button>
          );
        })}
      </nav>

      <div className="mt-auto flex flex-col gap-2 pt-4">
        <button
          onClick={() => props.onChangeView("donate")}
          className={`group flex w-full items-center gap-3 rounded-xl px-4 py-3 text-sm font-medium transition-all duration-200 ${props.activeView === "donate" ? "bg-gradient-to-r from-rose-500 to-pink-500 text-white shadow-lg shadow-rose-400/30" : "bg-gradient-to-r from-rose-50 to-pink-50 text-rose-600 shadow-sm hover:from-rose-100 hover:to-pink-100 hover:shadow-md"}`}
          type="button"
        >
          <div className={`flex h-8 w-8 items-center justify-center rounded-lg transition-all ${props.activeView === "donate" ? "bg-white/20" : "bg-rose-100 group-hover:bg-rose-200"}`}>
            <Heart size={18} />
          </div>
          <span>{t("sidebar.donate")}</span>
        </button>

        <div className="rounded-2xl bg-gradient-to-b from-gray-50 to-white p-1 shadow-inner">
          <button
            onClick={() => props.onChangeView("settings")}
            className={`flex w-full items-center gap-3 rounded-xl px-4 py-3 text-sm font-medium transition-all ${props.activeView === "settings" ? "bg-gradient-to-r from-[#0078D4] to-[#106EBE] text-white shadow-lg shadow-blue-500/25" : "text-gray-600 hover:bg-gray-100"}`}
            type="button"
          >
            <div className={`flex h-8 w-8 items-center justify-center rounded-lg ${props.activeView === "settings" ? "bg-white/20" : "bg-gray-100"}`}>
              <Settings2 size={18} />
            </div>
            <span>{t("sidebar.settings")}</span>
          </button>
        </div>
      </div>
    </aside>
  );
}

