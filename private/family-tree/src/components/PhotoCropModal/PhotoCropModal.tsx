import { useState, useRef, useEffect, type PointerEvent, type WheelEvent } from "react";
import { createPortal } from "react-dom";
import { open } from "@tauri-apps/plugin-dialog";
import { Modal } from "@/components/Modal/Modal";
import { readLocalImageObjectUrl } from "@/lib/person/readLocalImage";
import { toUint8Array } from "@/lib/person/getPersonPhotoSource";
import "@/components/PhotoCropModal/PhotoCropModal.css";

const VIEWPORT_SIZE = 320;
const CROP_SIZE = 240;
const OUTPUT_SIZE = 500;

type Props = {
  title: string;
  currentUrl?: string;
  initialZoom?: number;
  initialOffsetX?: number;
  initialOffsetY?: number;
  onSave: (imageData: number[], zoom: number, offsetX: number, offsetY: number) => Promise<void>;
  onClose: () => void;
};

export function PhotoCropModal({
  title,
  currentUrl,
  initialZoom = 1,
  initialOffsetX = 0,
  initialOffsetY = 0,
  onSave,
  onClose,
}: Props) {
  const [imageSrc, setImageSrc] = useState("");
  const [zoom, setZoom] = useState(initialZoom);
  const [offsetX, setOffsetX] = useState(initialOffsetX);
  const [offsetY, setOffsetY] = useState(initialOffsetY);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");

  const isDragging = useRef(false);
  const dragStart = useRef({ x: 0, y: 0 });
  const imageElementRef = useRef<HTMLImageElement | null>(null);
  const objectUrlRef = useRef<string | undefined>(undefined);

  useEffect(() => {
    let cancelled = false;
    async function loadCurrent() {
      if (!currentUrl) {
        setImageSrc("");
        return;
      }
      try {
        const src = await readLocalImageObjectUrl(currentUrl);
        if (cancelled) {
          if (src.startsWith("blob:")) URL.revokeObjectURL(src);
          return;
        }
        if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);
        objectUrlRef.current = src.startsWith("blob:") ? src : undefined;
        setImageSrc(src);
      } catch (reason) {
        if (!cancelled) setError(reason instanceof Error ? reason.message : "Не удалось открыть текущее фото.");
      }
    }
    void loadCurrent();
    return () => {
      cancelled = true;
    };
  }, [currentUrl]);

  useEffect(() => {
    return () => {
      if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);
    };
  }, []);

  async function choosePhoto() {
    setError("");
    try {
      const selected = await open({
        multiple: false,
        directory: false,
        filters: [{ name: "Изображения", extensions: ["png", "jpg", "jpeg", "webp", "gif", "bmp"] }],
      });
      if (typeof selected !== "string") return;
      const src = await readLocalImageObjectUrl(selected);
      if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);
      objectUrlRef.current = src.startsWith("blob:") ? src : undefined;
      setImageSrc(src);
      setZoom(1);
      setOffsetX(0);
      setOffsetY(0);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Не удалось загрузить изображение.");
    }
  }

  function handlePointerDown(event: PointerEvent<HTMLDivElement>) {
    if (!imageSrc) return;
    isDragging.current = true;
    dragStart.current = { x: event.clientX - offsetX, y: event.clientY - offsetY };
    event.currentTarget.setPointerCapture(event.pointerId);
  }

  function handlePointerMove(event: PointerEvent<HTMLDivElement>) {
    if (!isDragging.current) return;
    setOffsetX(event.clientX - dragStart.current.x);
    setOffsetY(event.clientY - dragStart.current.y);
  }

  function handlePointerUp(event: PointerEvent<HTMLDivElement>) {
    isDragging.current = false;
    try {
      event.currentTarget.releasePointerCapture(event.pointerId);
    } catch {
      // ignore
    }
  }

  function handleWheel(event: WheelEvent<HTMLDivElement>) {
    if (!imageSrc) return;
    event.preventDefault();
    const delta = event.deltaY < 0 ? 0.1 : -0.1;
    setZoom((current) => Math.max(0.2, Math.min(5, Number((current + delta).toFixed(2)))));
  }

  function handleReset() {
    setZoom(1);
    setOffsetX(0);
    setOffsetY(0);
  }

  async function handleSave() {
    if (!imageElementRef.current || !imageSrc) return;
    setSaving(true);
    setError("");
    try {
      const img = imageElementRef.current;
      await img.decode();
      if (!img.naturalWidth || !img.naturalHeight) throw new Error("Изображение ещё не загрузилось");

      const canvas = document.createElement("canvas");
      canvas.width = OUTPUT_SIZE;
      canvas.height = OUTPUT_SIZE;
      const ctx = canvas.getContext("2d");
      if (!ctx) throw new Error("Не удалось инициализировать холст");

      const scale = OUTPUT_SIZE / CROP_SIZE;
      ctx.clearRect(0, 0, OUTPUT_SIZE, OUTPUT_SIZE);
      ctx.save();
      ctx.beginPath();
      ctx.arc(OUTPUT_SIZE / 2, OUTPUT_SIZE / 2, OUTPUT_SIZE / 2, 0, Math.PI * 2);
      ctx.clip();
      ctx.translate(OUTPUT_SIZE / 2 + offsetX * scale, OUTPUT_SIZE / 2 + offsetY * scale);
      ctx.scale(zoom * scale, zoom * scale);
      ctx.drawImage(img, -img.naturalWidth / 2, -img.naturalHeight / 2);
      ctx.restore();

      const blob = await new Promise<Blob>((resolve, reject) => {
        canvas.toBlob((result) => {
          if (result) resolve(result);
          else reject(new Error("Не удалось создать изображение"));
        }, "image/png");
      });

      const bytes = Array.from(toUint8Array(await blob.arrayBuffer()));
      await onSave(bytes, 1, 0, 0);
      onClose();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Не удалось сохранить фотографию.");
    } finally {
      setSaving(false);
    }
  }

  return createPortal(
    <Modal onClose={onClose} onBack={onClose} backdropClassName="app-modal-backdrop--nested" className="photo-crop-modal" labelledBy="photo-crop-title">
      <div className="photo-crop-modal__header">
        <h2 id="photo-crop-title">{title}</h2>
        <button type="button" onClick={onClose} aria-label="Закрыть">
          ×
        </button>
      </div>

      {error && (
        <p className="photo-crop-modal__error" role="alert">
          {error}
        </p>
      )}

      {!imageSrc ? (
        <div style={{ textAlign: "center", padding: "30px 0" }}>
          <p style={{ marginBottom: "16px", color: "#6c4f2b" }}>Фотография не выбрана.</p>
          <button type="button" className="photo-crop-modal__primary" onClick={() => void choosePhoto()}>
            Выбрать фото
          </button>
        </div>
      ) : (
        <>
          <div
            className="photo-crop-modal__viewport"
            style={{ width: VIEWPORT_SIZE, height: VIEWPORT_SIZE }}
            onPointerDown={handlePointerDown}
            onPointerMove={handlePointerMove}
            onPointerUp={handlePointerUp}
            onWheel={handleWheel}
          >
            <div className="photo-crop-modal__image-container">
              <img
                ref={imageElementRef}
                src={imageSrc}
                alt="Предпросмотр обрезки"
                className="photo-crop-modal__image"
                style={{
                  transform: `translate(-50%, -50%) translate(${offsetX}px, ${offsetY}px) scale(${zoom})`,
                }}
                draggable={false}
              />
            </div>
            <div className="photo-crop-modal__overlay" />
          </div>

          <div className="photo-crop-modal__controls">
            <div className="photo-crop-modal__slider-row">
              <span>Зум</span>
              <input
                type="range"
                min="0.2"
                max="4"
                step="0.05"
                value={zoom}
                onChange={(event) => setZoom(parseFloat(event.target.value))}
              />
              <span>{Math.round(zoom * 100)}%</span>
            </div>

            <div className="photo-crop-modal__buttons-row">
              <button type="button" onClick={handleReset}>
                По центру
              </button>
              <button type="button" onClick={() => void choosePhoto()}>
                Другое фото
              </button>
            </div>
          </div>
        </>
      )}

      <div className="photo-crop-modal__footer">
        <button type="button" onClick={onClose}>
          Отмена
        </button>
        {imageSrc && (
          <button
            type="button"
            className="photo-crop-modal__primary"
            disabled={saving}
            onClick={() => void handleSave()}
          >
            {saving ? "Сохранение…" : "Сохранить"}
          </button>
        )}
      </div>
    </Modal>,
    document.body,
  );
}
