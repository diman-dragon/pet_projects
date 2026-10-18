import { useEffect } from 'react';
import { Cpu } from 'lucide-react';
import { Select } from '../ui';
import type { SelectOption } from '../ui';
import { useOllamaStore } from '../../store/ollamaStore';

interface OllamaModelSelectorProps {
  className?: string;
}

export function OllamaModelSelector({ className = '' }: OllamaModelSelectorProps) {
  const { models, selectedModel, setSelectedModel, refreshModels, isLoading } = useOllamaStore();

  useEffect(() => {
    refreshModels().catch(() => undefined);
  }, [refreshModels]);

  const handleChange = (nextValue: string) => {
    setSelectedModel(nextValue);
  };

  const options: SelectOption[] = models.map((model) => ({
    value: model.name,
    label: model.name,
  }));

  return (
    <div className={`flex min-w-[240px] items-center gap-2 ${className}`}>
      <Cpu size={14} className="text-black/40" />
      <div className="min-w-0 flex-1">
        <Select
          value={selectedModel || ''}
          onChange={handleChange}
          options={options}
          disabled={isLoading || options.length === 0}
          className="min-w-0"
        />
      </div>
    </div>
  );
}
