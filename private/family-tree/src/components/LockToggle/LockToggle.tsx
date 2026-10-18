import "@/components/LockToggle/LockToggle.css";

type Props = { locked: boolean; onToggle: () => void };
export function LockToggle({ locked, onToggle }: Props) {
  return <button type="button" role="switch" aria-checked={locked} className={`lock-toggle${locked ? " lock-toggle--on" : ""}`} onClick={onToggle}>
    <span className="lock-toggle__track"><span className="lock-toggle__thumb">{locked ? "🔒" : "🔓"}</span></span>
    <span className="lock-toggle__text"><strong>{locked ? "Изменения заблокированы" : "Изменения разрешены"}</strong><small>{locked ? "Карточка защищена от изменений" : "Можно редактировать данные и положение"}</small></span>
  </button>;
}
