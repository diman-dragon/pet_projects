import { Component, useState, type ErrorInfo, type ReactNode } from "react";
import { motion } from "framer-motion";
import { AlertTriangle, RefreshCw, RotateCcw, Terminal, Copy } from "lucide-react";

interface ErrorBoundaryProps {
  children: ReactNode;
  fallback?: ReactNode;
}

interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
  errorInfo: ErrorInfo | null;
  errorLog: string;
}

export class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null,
      errorLog: "",
    };
  }

  static getDerivedStateFromError(error: Error): Partial<ErrorBoundaryState> {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo): void {
    const timestamp = new Date().toISOString();
    const errorLog = `
[${timestamp}] ERROR:
${error.name}: ${error.message}

Stack Trace:
${error.stack || "No stack trace available"}

Component Stack:
${errorInfo.componentStack}
    `.trim();

    // Log to console
    console.error("ErrorBoundary caught an error:", error, errorInfo);

    // Save to state
    this.setState({ errorInfo, errorLog });

    // Optional: Send to error tracking service
    this.sendErrorToService(error, errorInfo);
  }

  private sendErrorToService(_error: Error, _errorInfo: ErrorInfo): void {
    // Placeholder for error tracking integration
    // Example: Sentry, LogRocket, etc.
    if (typeof window !== "undefined" && (window as Window & { __TAURI__?: boolean }).__TAURI__) {
      // In Tauri, we could send to a local log file
      console.log("Error would be logged to local file in production");
    }
  }

  private handleReload = (): void => {
    window.location.reload();
  };

  private handleReset = (): void => {
    this.setState({
      hasError: false,
      error: null,
      errorInfo: null,
      errorLog: "",
    });
  };

  private handleCopyError = async (): Promise<void> => {
    try {
      await navigator.clipboard.writeText(this.state.errorLog);
      // Could show a toast notification here
    } catch (err) {
      console.error("Failed to copy error to clipboard:", err);
    }
  };

  render(): ReactNode {
    if (this.state.hasError) {
      // Custom fallback UI
      if (this.props.fallback) {
        return this.props.fallback;
      }

      return <DefaultErrorFallback
        error={this.state.error}
        errorLog={this.state.errorLog}
        onReload={this.handleReload}
        onReset={this.handleReset}
        onCopyError={this.handleCopyError}
      />;
    }

    return this.props.children;
  }
}

interface DefaultErrorFallbackProps {
  error: Error | null;
  errorLog: string;
  onReload: () => void;
  onReset: () => void;
  onCopyError: () => void;
}

function DefaultErrorFallback({
  error,
  errorLog,
  onReload,
  onReset,
  onCopyError,
}: DefaultErrorFallbackProps) {
  const [showDetails, setShowDetails] = useState(false);

  return (
    <div className="flex min-h-screen items-center justify-center bg-gradient-to-br from-slate-50 to-slate-100 p-4">
      <motion.div
        initial={{ opacity: 0, y: 20, scale: 0.95 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        transition={{ duration: 0.3 }}
        className="w-full max-w-lg rounded-[30px] border border-white/70 bg-white/90 p-8 shadow-2xl shadow-gray-200/50 backdrop-blur-xl"
      >
        {/* Icon */}
        <div className="mb-6 flex justify-center">
          <motion.div
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{ delay: 0.1, type: "spring", stiffness: 200 }}
            className="flex h-20 w-20 items-center justify-center rounded-2xl bg-rose-100"
          >
            <AlertTriangle size={40} className="text-rose-500" />
          </motion.div>
        </div>

        {/* Title */}
        <h2 className="mb-2 text-center text-2xl font-bold text-gray-900">
          Что-то пошло не так
        </h2>
        <p className="mb-6 text-center text-gray-500">
          Произошла непредвиденная ошибка. Попробуйте перезагрузить приложение или вернуться назад.
        </p>

        {/* Error Message */}
        {error && (
          <div className="mb-6 rounded-xl bg-rose-50 p-4">
            <p className="font-mono text-sm text-rose-700">
              {error.name}: {error.message}
            </p>
          </div>
        )}

        {/* Action Buttons */}
        <div className="mb-4 grid grid-cols-2 gap-3">
          <button
            onClick={onReload}
            className="flex items-center justify-center gap-2 rounded-xl bg-[#0078D4] px-4 py-3 font-medium text-white transition-all hover:bg-[#006CBD] active:scale-95"
          >
            <RotateCcw size={18} />
            Перезагрузить
          </button>
          <button
            onClick={onReset}
            className="flex items-center justify-center gap-2 rounded-xl border-2 border-gray-200 bg-white px-4 py-3 font-medium text-gray-700 transition-all hover:bg-gray-50 active:scale-95"
          >
            <RefreshCw size={18} />
            Попробовать снова
          </button>
        </div>

        {/* Error Details Toggle */}
        <button
          onClick={() => setShowDetails(!showDetails)}
          className="flex w-full items-center justify-center gap-2 py-2 text-sm text-gray-500 transition-colors hover:text-gray-700"
        >
          <Terminal size={16} />
          {showDetails ? "Скрыть детали" : "Показать детали ошибки"}
        </button>

        {/* Error Details */}
        {showDetails && errorLog && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            className="mt-4 overflow-hidden"
          >
            <div className="relative rounded-xl bg-gray-900 p-4">
              <button
                onClick={onCopyError}
                className="absolute right-2 top-2 flex items-center gap-1 rounded-lg bg-gray-800 px-2 py-1 text-xs text-gray-300 transition-colors hover:bg-gray-700"
              >
                <Copy size={12} />
                Копировать
              </button>
              <pre className="max-h-48 overflow-auto font-mono text-xs leading-relaxed text-gray-300">
                {errorLog}
              </pre>
            </div>
          </motion.div>
        )}

        {/* Footer */}
        <p className="mt-6 text-center text-xs text-gray-400">
          Если ошибка повторяется, пожалуйста, сообщите о ней разработчикам.
        </p>
      </motion.div>
    </div>
  );
}
