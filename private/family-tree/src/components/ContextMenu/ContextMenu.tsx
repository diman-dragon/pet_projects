import { useEffect, useLayoutEffect, useRef, useState } from "react";
import type { KeyboardEvent } from "react";
import { useClickOutside } from "@/hooks/useClickOutside";
import type { ContextMenuPosition } from "@/types/contextMenu";
import "@/components/ContextMenu/ContextMenu.css";

type ContextMenuItem = {
  label: string;
  onSelect: () => void;
};

type ContextMenuProps = {
  position: ContextMenuPosition;
  items: ContextMenuItem[];
  onClose: () => void;
};

const MENU_MARGIN = 8;

export function ContextMenu({ position, items, onClose }: ContextMenuProps) {
  const menuRef = useRef<HTMLDivElement>(null);
  const [safePosition, setSafePosition] = useState(position);

  useClickOutside(menuRef, onClose);

  useLayoutEffect(() => {
    const menu = menuRef.current;
    if (!menu) return;

    setSafePosition({
      x: Math.min(
        position.x,
        window.innerWidth - menu.offsetWidth - MENU_MARGIN,
      ),
      y: Math.min(
        position.y,
        window.innerHeight - menu.offsetHeight - MENU_MARGIN,
      ),
    });
  }, [position]);

  useEffect(() => {
    menuRef.current?.querySelector<HTMLButtonElement>("button")?.focus();
  }, []);

  function handleKeyDown(event: KeyboardEvent<HTMLDivElement>) {
    const menuItems = Array.from(
      menuRef.current?.querySelectorAll<HTMLButtonElement>("button") ?? [],
    );
    const currentIndex = menuItems.indexOf(document.activeElement as HTMLButtonElement);

    if (event.key === "Escape") {
      event.preventDefault();
      onClose();
      return;
    }

    if (!menuItems.length) return;

    if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      event.preventDefault();
      const direction = event.key === "ArrowDown" ? 1 : -1;
      const nextIndex =
        (currentIndex + direction + menuItems.length) % menuItems.length;
      menuItems[nextIndex].focus();
    }

    if (event.key === "Home" || event.key === "End") {
      event.preventDefault();
      menuItems[event.key === "Home" ? 0 : menuItems.length - 1].focus();
    }
  }

  return (
    <div
      ref={menuRef}
      className="context-menu"
      role="menu"
      style={{ left: safePosition.x, top: safePosition.y }}
      onKeyDown={handleKeyDown}
      onPointerDown={(event) => event.stopPropagation()}
      onContextMenu={(event) => event.preventDefault()}
    >
      {items.map((item) => (
        <button
          key={item.label}
          type="button"
          className="context-menu__item"
          role="menuitem"
          onClick={item.onSelect}
        >
          {item.label}
        </button>
      ))}
    </div>
  );
}
