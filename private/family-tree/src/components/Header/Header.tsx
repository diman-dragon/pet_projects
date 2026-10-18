import { useEffect, useRef, useState } from "react";
import { useClickOutside } from "@/hooks/useClickOutside";
import { HeaderLegend } from "@/components/Header/HeaderLegend";
import { HeaderMenu } from "@/components/Header/HeaderMenu";
import { WorkspaceSettingsModal } from "@/components/WorkspaceSettingsModal/WorkspaceSettingsModal";
import { TrashModal } from "@/components/TrashModal/TrashModal";
import { PeopleListModal } from "@/components/PeopleListModal/PeopleListModal";
import { useCanvasCardsStore } from "@/store/useCanvasCardsStore";
import type { AppTheme } from "@/types/appTheme";
import "@/components/Header/Header.css";

type OpenPanel = "menu" | "legend" | "undo" | null;
type HeaderProps = {
  theme: AppTheme;
  onThemeChange: (theme: AppTheme) => void;
  showAuxiliaryLines: boolean;
  onAuxiliaryLinesChange: (show: boolean) => void;
};

export function Header({ theme, onThemeChange, showAuxiliaryLines, onAuxiliaryLinesChange }: HeaderProps) {
  const headerRef = useRef<HTMLElement>(null);
  const [openPanel, setOpenPanel] = useState<OpenPanel>(null);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [isTrashOpen, setIsTrashOpen] = useState(false);
  const [isPeopleListOpen, setIsPeopleListOpen] = useState(false);
  const canUndo = useCanvasCardsStore((state) => state.canUndo);
  const undoCount = useCanvasCardsStore((state) => state.undoCount);
  const undoHistory = useCanvasCardsStore((state) => state.undoHistory);
  const undoLastAction = useCanvasCardsStore((state) => state.undoLastAction);
  const undoActions = useCanvasCardsStore((state) => state.undoActions);

  useClickOutside(headerRef, () => setOpenPanel(null));

  useEffect(() => {
    function handleEscape(event: KeyboardEvent) {
      if (event.key === "Escape") { setOpenPanel(null); setIsSettingsOpen(false); setIsTrashOpen(false); }
    }

    document.addEventListener("keydown", handleEscape);
    return () => document.removeEventListener("keydown", handleEscape);
  }, []);

  function togglePanel(panel: Exclude<OpenPanel, null>) {
    setOpenPanel((current) => (current === panel ? null : panel));
  }

  return (
    <header ref={headerRef} className="app-header">
      <div className="app-header__side app-header__side--left">
        <HeaderMenu
          isOpen={openPanel === "menu"}
          onToggle={() => togglePanel("menu")}
          onSettings={() => {
            setIsSettingsOpen(true);
            setOpenPanel(null);
          }}
          onTrash={() => { setIsTrashOpen(true); setOpenPanel(null); }}
          onPeopleList={() => setIsPeopleListOpen(true)}
        />
        <div className="header-undo">
          <button type="button" className="header-trigger header-undo__button" disabled={!canUndo} aria-label="Отменить последнее изменение" onClick={() => void undoLastAction()}><span aria-hidden="true">↶</span><span>{undoCount || ""}</span></button>
          <button type="button" className="header-trigger header-undo__toggle" disabled={!canUndo} aria-label="Показать изменения" onClick={() => setOpenPanel((current) => current === "undo" ? null : "undo")}>⌄</button>
          {openPanel === "undo" && <div className="header-dropdown header-undo__dropdown" role="menu"><button type="button" className="header-undo__all" onClick={() => { setOpenPanel(null); void undoActions(undoHistory.length); }}>Отменить всё ({undoHistory.length})</button>{[...undoHistory].reverse().map((label, index) => <button key={`${label}-${index}`} type="button" className="header-undo__item" onClick={() => { setOpenPanel(null); void undoActions(index + 1); }}><span>↶</span><span>{label}</span><small>Отменить {index + 1}</small></button>)}</div>}
        </div>
      </div>
      <div className="app-header__title">
        <span className="app-header__eyebrow">История семьи</span>
        <h1>Древо поколений</h1>
      </div>
      <div className="app-header__side app-header__side--right">
        <HeaderLegend
          isOpen={openPanel === "legend"}
          onToggle={() => togglePanel("legend")}
        />
      </div>
      {isSettingsOpen && <WorkspaceSettingsModal theme={theme} onThemeChange={onThemeChange} showAuxiliaryLines={showAuxiliaryLines} onAuxiliaryLinesChange={onAuxiliaryLinesChange} onClose={() => setIsSettingsOpen(false)} />}
      {isTrashOpen && <TrashModal onClose={() => setIsTrashOpen(false)} />}
      {isPeopleListOpen && <PeopleListModal onClose={() => setIsPeopleListOpen(false)} />}
    </header>
  );
}
