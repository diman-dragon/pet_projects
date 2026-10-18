// src/pages/knowledge/components/ai/AITextBody.tsx

import { MarkdownMessage } from '../../../laila/components/shared/MarkdownMessage';
import { StreamingCursor } from '../../../../components/ui/feedback/StreamingCursor';

interface AITextBodyProps {
  text: string;
  streaming: boolean;
}

/** Стриминговый текст AI-ответа (краткая справка над результатами поиска) */
export function AITextBody({ text, streaming }: AITextBodyProps) {
  return (
    <div className="select-text cursor-text">
      <MarkdownMessage content={text} />
      {streaming && <StreamingCursor />}
    </div>
  );
}
