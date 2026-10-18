import React from 'react';
import i18n from '../../i18n/i18n';
import type { TranslationKey } from '../../i18n/translations';
import { ErrorReporter } from '../../services/errors/ErrorReporter';

interface Props {
  children: React.ReactNode;
  fallback?: React.ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

export class ErrorBoundary extends React.Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, info: React.ErrorInfo) {
    ErrorReporter.capture('ErrorBoundary', error, {
      componentStack: info.componentStack,
    });
  }

  handleReset = () => {
    this.setState({ hasError: false, error: null });
  };

  render() {
    if (this.state.hasError) {
      if (this.props.fallback) return this.props.fallback;

      const t = (key: TranslationKey) => (i18n.t as (k: string) => string)(key);

      return (
        <div className="flex h-full flex-col items-center justify-center px-8 py-12 text-center select-none">
          <div className="mb-6 flex size-16 items-center justify-center rounded-[20px] border border-[var(--app-border-strong)] bg-[var(--app-surface-raised)] shadow-[var(--app-shadow-md)]">
            <span className="text-[length:var(--text-hero)] leading-[var(--leading-heading)] text-[var(--app-danger)]">
              !
            </span>
          </div>
          <h2 className="mb-2 text-[length:var(--text-xl)] font-semibold leading-[var(--leading-heading)] text-[var(--app-text)]">
            {t('errorBoundary.title')}
          </h2>
          <p className="mb-3 max-w-xl text-[length:var(--text-sm)] leading-[var(--leading-body)] text-[var(--app-text-muted)]">
            {t('errorBoundary.body')}
          </p>
          <p className="mb-6 max-w-xl rounded-[14px] border border-[var(--app-border)] bg-[var(--app-surface)] px-4 py-3 font-mono text-[length:var(--text-xs)] leading-[var(--leading-body)] text-[var(--app-text-muted)]">
            {this.state.error?.message ?? t('errorBoundary.unknown')}
          </p>
          <button
            type="button"
            onClick={this.handleReset}
            className="inline-flex h-10 items-center justify-center rounded-[var(--app-radius-sm)] border border-[var(--app-accent)] bg-[var(--app-accent)] px-5 text-[length:var(--text-sm)] font-semibold leading-[var(--leading-body)] text-[var(--app-surface-raised)] shadow-[var(--app-shadow-sm)] transition-all hover:brightness-105 active:scale-[0.99]"
          >
            {t('errorBoundary.reset')}
          </button>
        </div>
      );
    }

    return this.props.children;
  }
}
