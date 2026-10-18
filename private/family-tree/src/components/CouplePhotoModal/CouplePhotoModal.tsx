import { invoke } from "@tauri-apps/api/core";
import { PhotoCropModal } from "@/components/PhotoCropModal/PhotoCropModal";

type Props = { relationId: string; personId: string; currentUrl?: string; onSave: (url: string) => Promise<void>; onClose: () => void };

export function CouplePhotoModal({ relationId, personId, currentUrl, onSave, onClose }: Props) {
  async function handleSave(imageData: number[]) {
    const result = await invoke<string>("save_relation_photo_crop", { relationId, personId, imageData });
    await onSave(result);
  }

  return (
    <PhotoCropModal
      title="Фото пары"
      currentUrl={currentUrl}
      onSave={async (imageData) => {
        await handleSave(imageData);
      }}
      onClose={onClose}
    />
  );
}
