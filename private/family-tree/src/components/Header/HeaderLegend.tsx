import "@/components/Header/HeaderLegend.css";
import { RelationHeart } from "@/components/RelationHeart/RelationHeart";

type HeaderLegendProps = {
  isOpen: boolean;
  onToggle: () => void;
};

export function HeaderLegend({ isOpen, onToggle }: HeaderLegendProps) {
  return (
    <div className="header-legend">
      <button
        type="button"
        className="header-trigger"
        aria-expanded={isOpen}
        aria-controls="legend-menu"
        aria-haspopup="menu"
        onClick={onToggle}
      >
        <span>Легенда</span>
        <span className="header-trigger__chevron" aria-hidden="true">⌄</span>
      </button>
      {isOpen && (
        <div
          id="legend-menu"
          className="header-dropdown header-legend__dropdown"
          role="menu"
        >
          <div className="header-legend__heading">Обозначения на холсте</div>
          <div className="header-legend__item">
            <span className="header-legend__relation-line header-legend__relation-line--parenthood" />
            <span>Родительская связь</span>
          </div>
          <div className="header-legend__item">
            <span className="header-legend__marriage"><i /> <RelationHeart size={18} /> <i /></span>
            <span>Действующий брак</span>
          </div>
          <div className="header-legend__item">
            <span className="header-legend__marriage is-dissolved"><i /> <RelationHeart variant="broken" size={18} /> <i /></span>
            <span>Расторгнутый брак</span>
          </div>
          <div className="header-legend__item">
            <span className="header-legend__marriage is-older"><i /> <RelationHeart size={18} /> <i /></span>
            <span>Предыдущий брак</span>
          </div>
        </div>
      )}
    </div>
  );
}
