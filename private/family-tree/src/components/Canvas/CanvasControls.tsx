import "@/components/Canvas/CanvasControls.css";

type CanvasControlsProps = {
  zoom: number;
  onZoomIn: () => void;
  onZoomOut: () => void;
  onZoomTo100: () => void;
  onFitTree: () => void;
};

export function CanvasControls({
  zoom,
  onZoomIn,
  onZoomOut,
  onZoomTo100,
  onFitTree,
}: CanvasControlsProps) {
  return (
    <div
      className="canvas-controls"
      onPointerDown={(event) => event.stopPropagation()}
    >
      <button
        type="button"
        onClick={onZoomOut}
        aria-label="Уменьшить масштаб"
      >
        −
      </button>
      <span className="canvas-controls__zoom-value" aria-label={`Масштаб ${Math.round(zoom * 100)} процентов`}>
        {Math.round(zoom * 100)}%
      </span>
      <button type="button" className="canvas-controls__zoom-100" onClick={onZoomTo100} aria-label="Масштаб 100 процентов" title="Масштаб 100 процентов">100%</button>
      <button type="button" onClick={onZoomIn} aria-label="Увеличить масштаб">
        +
      </button>
      <button type="button" className="canvas-controls__fit-tree" onClick={onFitTree} aria-label="Показать всё дерево" title="Показать всё дерево">
        ⛶
      </button>
    </div>
  );
}
