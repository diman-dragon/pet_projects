// components/shared/MarkdownMessage.tsx

import type { MouseEvent } from 'react';
import ReactMarkdown from 'react-markdown';
import type { Components } from 'react-markdown';

interface MarkdownMessageProps {
  content: string;
  isUser?: boolean;
  onLinkClick?: (href: string) => boolean | void;
  /** Перекрывает/дополняет дефолтные markdown-компоненты */
  overrideComponents?: Partial<Components>;
}

const bodyLead = 'leading-[var(--leading-body)]';

function createMarkdownComponents(onLinkClick?: (href: string) => boolean | void): Components {
  return {
    p: ({ children }) => <p className={`mb-2 last:mb-0 ${bodyLead}`}>{children}</p>,
    strong: ({ children }) => <strong className="font-bold text-black/90">{children}</strong>,
    em: ({ children }) => <em className="italic text-black/70">{children}</em>,
    ul: ({ children }) => <ul className="my-2 space-y-1 pl-1 list-none">{children}</ul>,
    ol: ({ children }) => <ol className="my-2 space-y-1 pl-1 list-none">{children}</ol>,
    li: ({ children }) => (
      <li className={`${bodyLead} inline-block w-full`}>
        <span className="text-violet-400 mr-2">•</span>
        <span>{children}</span>
      </li>
    ),
    h1: ({ children }) => (
      <h1
        className={`font-black text-[length:var(--text-sm)] text-black/90 mb-2 mt-3 first:mt-0 uppercase tracking-tight ${bodyLead}`}
      >
        {children}
      </h1>
    ),
    h2: ({ children }) => (
      <h2
        className={`font-bold text-[length:var(--text-sm)] text-black/85 mb-1.5 mt-3 first:mt-0 ${bodyLead}`}
      >
        {children}
      </h2>
    ),
    h3: ({ children }) => (
      <h3
        className={`font-semibold text-[length:var(--text-xs)] text-black/80 mb-1 mt-2 first:mt-0 ${bodyLead}`}
      >
        {children}
      </h3>
    ),
    pre: ({ children }) => <pre className="my-2 bg-transparent p-0">{children}</pre>,
    hr: () => <hr className="my-3 border-black/[0.08]" />,
    blockquote: ({ children }) => (
      <blockquote
        className={`border-l-2 border-violet-300 pl-3 my-2 text-black/60 italic bg-violet-50/30 py-1 rounded-r-lg ${bodyLead}`}
      >
        {children}
      </blockquote>
    ),
    code: ({ className, children }) => {
      const isBlock = !!className;
      return isBlock ? (
        <code
          className={`block bg-black/[0.06] rounded-lg px-3 py-2 text-[length:var(--text-xs)] font-mono text-black/70 my-2 whitespace-pre-wrap overflow-x-auto border border-black/[0.03] ${bodyLead}`}
        >
          {children}
        </code>
      ) : (
        <code className="bg-violet-50 text-violet-700 rounded px-1.5 py-0.5 text-[length:var(--text-xs)] font-mono font-medium">
          {children}
        </code>
      );
    },
    a: ({ href, children, ...props }) => {
      const handleClick = (event: MouseEvent<HTMLAnchorElement>) => {
        if (!href || !onLinkClick) return;
        const handled = onLinkClick(href);
        if (handled) event.preventDefault();
      };

      const isExternal = typeof href === 'string' && /^https?:\/\//.test(href);

      return (
        <a
          href={href}
          onClick={handleClick}
          className="text-[var(--app-accent)] underline decoration-[var(--app-accent)]/40 hover:text-[var(--app-accent)]"
          target={isExternal ? '_blank' : undefined}
          rel={isExternal ? 'noreferrer noopener' : undefined}
          {...props}
        >
          {children}
        </a>
      );
    },
  };
}

/**
 * Рендерит текст как markdown (для Laila) или plain text (для пользователя).
 */
export function MarkdownMessage({ content, isUser = false, onLinkClick, overrideComponents }: MarkdownMessageProps) {
  if (isUser) {
    return (
      <span className={`whitespace-pre-wrap text-[length:var(--text-sm)] ${bodyLead}`}>
        {content}
      </span>
    );
  }

  return (
    <div className={`text-[length:var(--text-sm)] text-black/75 ${bodyLead} overflow-hidden`}>
      <ReactMarkdown components={{ ...createMarkdownComponents(onLinkClick), ...overrideComponents }}>{content}</ReactMarkdown>
    </div>
  );
}
