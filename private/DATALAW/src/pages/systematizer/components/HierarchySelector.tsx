import { useState, useMemo } from 'react';
import { Plus, Check, X } from 'lucide-react';
import { Select, AppInput } from '../../../components/ui';
import { NamingService } from '../../../services/utils/NamingService';
import { useI18n } from '../../../i18n/AppLanguageProvider';

interface HierarchySelectorProps {
  value: string;
  onChange: (value: string) => void;
  options: { label: string; value: string }[];
  placeholder?: string;
  type: 'client' | 'case' | 'line' | 'appeal';
  clientType?: 'LEGAL' | 'INDIVIDUAL';
}

export function HierarchySelector({
  value,
  onChange,
  options,
  placeholder,
  type,
  clientType = 'LEGAL',
}: HierarchySelectorProps) {
  const { t } = useI18n();
  const [isAddingNew, setIsAddingNew] = useState(false);
  const [newValue, setNewValue] = useState('');

  const transliteratedValue = useMemo(() => {
    const text = String(newValue || '').trim();
    if (!text) return '';

    if (type === 'client') {
      return clientType === 'LEGAL'
        ? NamingService.toLegalName(text)
        : NamingService.toShortName(text);
    }

    return NamingService.toStandardName(text);
  }, [newValue, type, clientType]);

  const optionExists = (options || []).some((o) => o.value === value);

  const selectOptions = useMemo(() => {
    const base = [...(options || [])];
    if (value && !optionExists) {
      base.unshift({ label: value, value });
    }
    if (!value) {
      base.unshift({ label: placeholder || t('systematizer.hierarchy.selectPlaceholder'), value: '' });
    }
    return base;
  }, [options, value, optionExists, placeholder, t]);

  const handleAddNew = () => {
    if (transliteratedValue) {
      onChange(transliteratedValue);
      setIsAddingNew(false);
      setNewValue('');
    }
  };

  const handleCancel = () => {
    setIsAddingNew(false);
    setNewValue('');
  };

  if (isAddingNew) {
    return (
      <div className="flex flex-col gap-1 w-full">
        <div className="flex items-center gap-1">
          <div className="flex-1 relative">
            <AppInput
              autoFocus
              placeholder={t('systematizer.hierarchy.inputPlaceholder')}
              value={newValue}
              onChange={(e) => setNewValue(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') handleAddNew();
                if (e.key === 'Escape') handleCancel();
              }}
            />
          </div>
          <button
            onClick={handleAddNew}
            disabled={!transliteratedValue}
            className="p-1.5 bg-[var(--app-success)] text-white rounded-md hover:bg-[color-mix(in_srgb,var(--app-success)_88%,black)] disabled:opacity-50 transition-colors"
            title={t('systematizer.hierarchy.applyTitle')}
          >
            <Check size={16} />
          </button>
          <button
            onClick={handleCancel}
            className="p-1.5 bg-[var(--app-surface-soft)] text-[var(--app-text-secondary)] rounded-md hover:bg-[color-mix(in_srgb,var(--app-surface-soft)_88%,black)] transition-colors"
            title={t('systematizer.hierarchy.cancelTitle')}
          >
            <X size={16} />
          </button>
        </div>
        {transliteratedValue && (
          <div className="px-2 py-1 bg-blue-50 text-[10px] font-mono text-blue-700 rounded border border-blue-100 truncate">
            {transliteratedValue}
          </div>
        )}
      </div>
    );
  }

  return (
    <div className="flex items-center gap-1 w-full">
      <div className="flex-1 min-w-0">
        <Select
          value={value}
          onChange={onChange}
          options={selectOptions}
          className="h-8 py-0 text-[length:var(--text-xs)]"
        />
      </div>
      <button
        onClick={() => setIsAddingNew(true)}
        className="p-1.5 bg-[var(--app-info-soft)] text-[var(--app-accent)] rounded-md hover:bg-[color-mix(in_srgb,var(--app-info)_10%,transparent)] transition-colors flex-shrink-0"
        title={t('systematizer.hierarchy.addNewTitle')}
      >
        <Plus size={16} />
      </button>
    </div>
  );
}
