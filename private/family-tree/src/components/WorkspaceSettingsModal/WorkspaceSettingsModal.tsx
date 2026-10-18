import { useState } from "react";
import { Modal } from "@/components/Modal/Modal";
import { DataTransferModal } from "@/components/DataTransferModal/DataTransferModal";
import { DangerousActionsModal } from "@/components/WorkspaceSettingsModal/DangerousActionsModal";
import { WorkspaceFolderModal } from "@/components/WorkspaceSettingsModal/WorkspaceFolderModal";
import { LockedAreasModal } from "@/components/WorkspaceSettingsModal/LockedAreasModal";
import type { AppTheme } from "@/types/appTheme";
import "@/components/WorkspaceSettingsModal/WorkspaceSettingsModal.css";

type SettingsPanel = "folder" | "transfer" | "locks" | "danger" | null;
type WorkspaceSettingsModalProps = { theme: AppTheme; onThemeChange: (theme: AppTheme) => void; showAuxiliaryLines: boolean; onAuxiliaryLinesChange: (show: boolean) => void; onClose: () => void };

export function WorkspaceSettingsModal({ theme, onThemeChange, showAuxiliaryLines, onAuxiliaryLinesChange, onClose }: WorkspaceSettingsModalProps) {
  const [panel, setPanel] = useState<SettingsPanel>(null);

  if (panel === "locks") {
    return <LockedAreasModal onClose={onClose} onCompleted={onClose} />;
  }

  return <Modal onClose={panel ? () => undefined : onClose} onBack={panel ? () => setPanel(null) : onClose} backdropClassName="workspace-settings-backdrop" className="workspace-settings-modal workspace-settings-modal--index" labelledBy="workspace-settings-title">
      <div className="workspace-settings-modal__header"><div><span className="workspace-settings-modal__eyebrow">Настройки</span><h2 id="workspace-settings-title">Параметры приложения</h2></div><button type="button" className="workspace-settings-modal__close" onClick={onClose} aria-label="Закрыть">×</button></div>
      <section className="workspace-settings-modal__theme-top" aria-labelledby="theme-title"><div><span id="theme-title" className="workspace-settings-modal__section-title">Тема интерфейса</span><p>Выберите оформление приложения.</p></div><div className="workspace-settings-modal__theme-buttons"><button type="button" className={theme === "light" ? "workspace-settings-modal__theme-button workspace-settings-modal__theme-button--active" : "workspace-settings-modal__theme-button"} aria-pressed={theme === "light"} onClick={() => onThemeChange("light")}>☀ Светлая</button><button type="button" className={theme === "dark" ? "workspace-settings-modal__theme-button workspace-settings-modal__theme-button--active" : "workspace-settings-modal__theme-button"} aria-pressed={theme === "dark"} onClick={() => onThemeChange("dark")}>☾ Тёмная</button></div></section>
      <section className="workspace-settings-modal__theme-top" aria-labelledby="auxiliary-lines-title"><div><span id="auxiliary-lines-title" className="workspace-settings-modal__section-title">Вспомогательные линии</span><p>Линии поколений и границы межпоколенческих полос.</p></div><div className="workspace-settings-modal__theme-buttons"><button type="button" className={showAuxiliaryLines ? "workspace-settings-modal__theme-button workspace-settings-modal__theme-button--active" : "workspace-settings-modal__theme-button"} aria-pressed={showAuxiliaryLines} onClick={() => onAuxiliaryLinesChange(true)}>Показать</button><button type="button" className={!showAuxiliaryLines ? "workspace-settings-modal__theme-button workspace-settings-modal__theme-button--active" : "workspace-settings-modal__theme-button"} aria-pressed={!showAuxiliaryLines} onClick={() => onAuxiliaryLinesChange(false)}>Спрятать</button></div></section>
      <div className="workspace-settings-modal__links" role="menu" aria-label="Разделы настроек"><button type="button" className="workspace-settings-modal__link" onClick={() => setPanel("folder")}><span className="workspace-settings-modal__link-icon" aria-hidden="true">⌂</span><span><strong>Каталог семейного дерева</strong><small>Расположение базы, медиа и название дерева</small></span><span aria-hidden="true">›</span></button><button type="button" className="workspace-settings-modal__link" onClick={() => setPanel("transfer")}><span className="workspace-settings-modal__link-icon" aria-hidden="true">⇄</span><span><strong>Импорт и экспорт</strong><small>Форматы, режим объединения и замены</small></span><span aria-hidden="true">›</span></button></div>
      <section className="workspace-settings-modal__safety-link" aria-labelledby="canvas-lock-title"><button type="button" className="workspace-settings-modal__link" onClick={() => setPanel("locks")}><span className="workspace-settings-modal__link-icon" aria-hidden="true">⌑</span><span><strong id="canvas-lock-title">Разблокировать области</strong><small>Показать зафиксированные области и выбрать нужную</small></span><span aria-hidden="true">›</span></button></section>
      <section className="workspace-settings-modal__danger-link" aria-labelledby="danger-title"><button type="button" className="workspace-settings-modal__link workspace-settings-modal__link--danger" onClick={() => setPanel("danger")}><span className="workspace-settings-modal__link-icon" aria-hidden="true">⚠</span><span><strong id="danger-title">Опасные действия</strong><small>Очистка дерева и удаление связей</small></span><span aria-hidden="true">›</span></button></section>
      <div className="workspace-settings-modal__actions"><button type="button" className="workspace-settings-modal__secondary" onClick={onClose}>Закрыть</button></div>
    {panel === "folder" && <WorkspaceFolderModal onClose={() => setPanel(null)} onSaved={onClose} />}
    {panel === "transfer" && <DataTransferModal onClose={() => setPanel(null)} />}
    {panel === "danger" && <DangerousActionsModal onClose={() => setPanel(null)} onCompleted={onClose} />}
  </Modal>;
}
