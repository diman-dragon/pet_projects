import { getAppVersion } from "@/lib/app/getAppVersion";
import { useState } from "react";
import "@/components/Header/HeaderMenu.css";

type HeaderMenuProps = {
  isOpen: boolean;
  onToggle: () => void;
  onSettings: () => void;
  onTrash: () => void;
  onPeopleList: () => void;
};

export function HeaderMenu({ isOpen, onToggle, onSettings, onTrash, onPeopleList }: HeaderMenuProps) {
  const [openGroups, setOpenGroups] = useState({ tree: false, view: false, data: false });

  function toggleGroup(group: keyof typeof openGroups) {
    setOpenGroups((current) => ({ ...current, [group]: !current[group] }));
  }

  function GroupButton({ group, icon, title }: { group: keyof typeof openGroups; icon: string; title: string }) {
    return (
      <button
        type="button"
        className="header-menu__group-toggle"
        aria-expanded={openGroups[group]}
        onClick={() => toggleGroup(group)}
      >
        <span className="header-menu__group-icon" aria-hidden="true">{icon}</span>
        <span>{title}</span>
        <span className="header-menu__group-chevron" aria-hidden="true">{openGroups[group] ? "⌃" : "⌄"}</span>
      </button>
    );
  }

  return (
    <div className="header-menu">
      <button
        type="button"
        className="header-trigger"
        aria-expanded={isOpen}
        aria-controls="main-menu"
        aria-haspopup="menu"
        onClick={onToggle}
      >
        <span className="header-trigger__icon" aria-hidden="true">☰</span>
        <span>Меню</span>
        <span className="header-trigger__chevron" aria-hidden="true">⌄</span>
      </button>
      {isOpen && (
        <div
          id="main-menu"
          className="header-dropdown header-menu__dropdown"
          role="menu"
        >
          <div className="header-menu__roadmap" aria-label="Разделы меню">
            <button type="button" className="header-menu__people" role="menuitem" onClick={() => { onPeopleList(); onToggle(); }}><span className="header-menu__future-icon" aria-hidden="true">♙</span><span>Список всех людей</span><span aria-hidden="true">›</span></button>
            <section className="header-menu__group">
              <GroupButton group="tree" icon="♧" title="Дерево" />
              {openGroups.tree && <div className="header-menu__group-content">
                <button type="button" className="header-menu__future" role="menuitem" disabled title="Функция находится в разработке">
                  <span className="header-menu__future-icon" aria-hidden="true">✣</span><span className="header-menu__future-name">Фокус на ветви</span><span className="header-menu__badge header-menu__badge--dev">В разработке</span>
                </button>
                <button type="button" className="header-menu__future" role="menuitem" disabled title="Функция находится в разработке">
                  <span className="header-menu__future-icon" aria-hidden="true">⌗</span><span className="header-menu__future-name">Автораскладка дерева</span><span className="header-menu__badge header-menu__badge--dev">В разработке</span>
                </button>
              </div>}
            </section>
            <section className="header-menu__group">
              <GroupButton group="view" icon="◈" title="Представление" />
              {openGroups.view && <div className="header-menu__group-content">
                <button type="button" className="header-menu__future" role="menuitem" disabled title="Функция скоро появится">
                  <span className="header-menu__future-icon" aria-hidden="true">◈</span><span className="header-menu__future-name">Мини-карта дерева</span><span className="header-menu__badge header-menu__badge--soon">Скоро</span>
                </button>
                <button type="button" className="header-menu__future" role="menuitem" disabled title="Функция скоро появится">
                  <span className="header-menu__future-icon" aria-hidden="true">↕</span><span className="header-menu__future-name">Лента поколений</span><span className="header-menu__badge header-menu__badge--soon">Скоро</span>
                </button>
                <div className="header-menu__future-submenu" aria-label="Изменить представление дерева">
                  <div className="header-menu__future-submenu-title"><span>Изменить представление дерева</span><span aria-hidden="true">›</span></div>
              <button type="button" className="header-menu__future header-menu__future--nested" role="menuitem" disabled title="Функция находится в разработке">
                <span className="header-menu__future-icon" aria-hidden="true">↓</span>
                <span className="header-menu__future-name">Вертикальное направление</span>
                <span className="header-menu__badge header-menu__badge--dev">В разработке</span>
              </button>
              <button type="button" className="header-menu__future header-menu__future--nested" role="menuitem" disabled title="Функция находится в разработке">
                <span className="header-menu__future-icon" aria-hidden="true">→</span>
                <span className="header-menu__future-name">Горизонтальное направление</span>
                <span className="header-menu__badge header-menu__badge--dev">В разработке</span>
              </button>
              <button type="button" className="header-menu__future header-menu__future--nested" role="menuitem" disabled title="Функция находится в разработке">
                <span className="header-menu__future-icon" aria-hidden="true">◉</span>
                <span className="header-menu__future-name">Концентрическое</span>
                <span className="header-menu__badge header-menu__badge--dev">В разработке</span>
              </button>
              <button type="button" className="header-menu__future header-menu__future--nested" role="menuitem" disabled title="Функция находится в разработке">
                <span className="header-menu__future-icon" aria-hidden="true">✣</span>
                <span className="header-menu__future-name">Радиальное</span>
                <span className="header-menu__badge header-menu__badge--dev">В разработке</span>
              </button>
              <button type="button" className="header-menu__future header-menu__future--nested" role="menuitem" disabled title="Функция находится в разработке">
                <span className="header-menu__future-icon" aria-hidden="true">▦</span>
                <span className="header-menu__future-name">Компактное по поколениям</span>
                <span className="header-menu__badge header-menu__badge--dev">В разработке</span>
              </button>
                </div>
              </div>}
            </section>
            <section className="header-menu__group">
              <GroupButton group="data" icon="⚙" title="Данные и обслуживание" />
              {openGroups.data && <div className="header-menu__group-content">
                <button type="button" className="header-menu__future" role="menuitem" disabled title="Функция скоро появится">
                  <span className="header-menu__future-icon" aria-hidden="true">⌁</span><span className="header-menu__future-name">Поиск дубликатов</span><span className="header-menu__badge header-menu__badge--soon">Скоро</span>
                </button>
                <button type="button" className="header-menu__future" role="menuitem" disabled title="Функция скоро появится">
                  <span className="header-menu__future-icon" aria-hidden="true">▤</span><span className="header-menu__future-name">Источники и доказательства</span><span className="header-menu__badge header-menu__badge--soon">Скоро</span>
                </button>
                <button type="button" className="header-menu__settings" role="menuitem" onClick={onSettings}><span aria-hidden="true">⚙</span><span>Настройки</span></button>
                <button type="button" className="header-menu__settings" role="menuitem" onClick={onTrash}><span aria-hidden="true">♜</span><span>Корзина</span></button>
              </div>}
            </section>
          </div>
          <div className="header-menu__footer">
            <span className="header-menu__version">v{getAppVersion()}</span>
          </div>
        </div>
      )}
    </div>
  );
}
