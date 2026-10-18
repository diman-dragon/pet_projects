import { useState } from 'react';
import { SearchInput } from '../../ui/form/SearchInput';
import { useI18n } from '../../../i18n/AppLanguageProvider';
import type { UserProfile } from '../../../types/entities';
import { GlobalSearchModal } from './GlobalSearchModal';

interface HeaderSearchProps {
  user: UserProfile;
  onNavigate: (route: string, params?: Record<string, unknown>) => void;
}

export function HeaderSearch({ user, onNavigate }: HeaderSearchProps) {
  const { t } = useI18n();
  const [previewValue, setPreviewValue] = useState('');
  const [open, setOpen] = useState(false);

  return (
    <>
      <div className="flex flex-1 items-center justify-center px-6">
        <button
          type="button"
          className="w-full max-w-md text-left"
          aria-label={t('header.searchOpenAria')}
          onClick={() => setOpen(true)}
        >
          <SearchInput
            value={previewValue}
            onChange={setPreviewValue}
            onFocus={() => setOpen(true)}
            placeholder={t('header.searchOpenPlaceholder')}
            className="pointer-events-none max-w-md"
          />
        </button>
      </div>

      <GlobalSearchModal
        open={open}
        onClose={() => setOpen(false)}
        user={user}
        onNavigate={onNavigate}
      />
    </>
  );
}
