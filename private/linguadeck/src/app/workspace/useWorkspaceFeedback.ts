import { useEffect, useState } from "react";
import { LogService } from "../../services/db/LogService";

export function useWorkspaceFeedback() {
  const [message, setMessage] = useState<string | null>(null);
  const [isError, setIsError] = useState(false);

  // Auto-dismiss after 3s
  useEffect(() => {
    if (!message) return;
    const t = setTimeout(() => setMessage(null), 3000);
    return () => clearTimeout(t);
  }, [message]);

  async function withFeedback<T>(
    scope: string,
    action: () => Promise<T>,
    successMessage?: string
  ) {
    try {
      const result = await action();
      if (successMessage) {
        setIsError(false);
        setMessage(successMessage);
        await LogService.info(scope, successMessage);
      }
      return result;
    } catch (error) {
      const text = error instanceof Error ? error.message : "Неизвестная ошибка";
      setIsError(true);
      setMessage(text);
      await LogService.error(scope, text, error);
      throw error;
    }
  }

  return {
    message,
    isError,
    setMessage,
    withFeedback
  };
}
