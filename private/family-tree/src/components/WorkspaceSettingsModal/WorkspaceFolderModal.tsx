import { useEffect, useState } from "react";
import { open } from "@tauri-apps/plugin-dialog";
import { invoke } from "@tauri-apps/api/core";
import { Modal } from "@/components/Modal/Modal";

type WorkspaceInfo = { root_dir: string; tree_name: string; database_dir: string; media_dir: string };
type WorkspaceFolderModalProps = { onClose: () => void; onSaved: () => void };

export function WorkspaceFolderModal({ onClose, onSaved }: WorkspaceFolderModalProps) {
  const [parentDir, setParentDir] = useState(""); const [treeName, setTreeName] = useState(""); const [workspace, setWorkspace] = useState<WorkspaceInfo | null>(null); const [error, setError] = useState(""); const [isSaving, setIsSaving] = useState(false);
  useEffect(() => { void invoke<WorkspaceInfo>("get_workspace").then((current) => { setWorkspace(current); setTreeName(current.tree_name); setParentDir(current.root_dir); }).catch(() => setError("Не удалось загрузить текущий каталог дерева.")); }, []);
  async function chooseParentDirectory() { const selected = await open({ title: "Выберите каталог для FamilyTree", directory: true, multiple: false }); if (typeof selected === "string") setParentDir(selected); }
  async function saveWorkspace() { setError(""); setIsSaving(true); try { await invoke<WorkspaceInfo>("set_workspace", { parentDir, treeName }); window.dispatchEvent(new Event("family-tree:workspace-changed")); onSaved(); } catch (saveError) { setError(String(saveError)); } finally { setIsSaving(false); } }
  return <Modal onClose={onClose} onBack={onClose} backdropClassName="workspace-settings-backdrop workspace-settings-backdrop--submodal" className="workspace-settings-modal" labelledBy="workspace-folder-title">
    <div className="workspace-settings-modal__header"><div><span className="workspace-settings-modal__eyebrow">Настройки · каталог</span><h2 id="workspace-folder-title">Каталог дерева</h2></div><button type="button" className="workspace-settings-modal__close" onClick={onClose} aria-label="Закрыть">×</button></div><p className="workspace-settings-modal__hint">Внутри каталога находятся база данных и папка медиафайлов.</p>
    <label>Родительский каталог<div className="workspace-settings-modal__path"><input value={parentDir} readOnly placeholder="Выберите папку" /><button type="button" onClick={() => void chooseParentDirectory()}>Выбрать</button></div></label><label>Название дерева<input value={treeName} onChange={(event) => setTreeName(event.target.value)} placeholder="Например: Ивановы" /></label>
    {workspace && <div className="workspace-settings-modal__current">База: <strong>{workspace.database_dir}</strong><br />Медиа: <strong>{workspace.media_dir}</strong></div>}{error && <div className="workspace-settings-modal__error" role="alert">{error}</div>}
    <div className="workspace-settings-modal__actions"><button type="button" className="workspace-settings-modal__secondary" onClick={onClose}>Назад</button><button type="button" className="workspace-settings-modal__primary" disabled={!parentDir || !treeName.trim() || isSaving} onClick={() => void saveWorkspace()}>{isSaving ? "Сохранение…" : "Сохранить"}</button></div>
  </Modal>;
}
