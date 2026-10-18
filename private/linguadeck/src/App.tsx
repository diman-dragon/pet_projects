import { useEffect, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { BootService } from "./services/db/BootService";
import { LinguadeckWorkspace } from "./app/LinguadeckWorkspace";
import { ErrorBoundary } from "./components/error/ErrorBoundary";
import { SetupScreen } from "./components/setup/SetupScreen";
import { useAppInitialization } from "./hooks/useAppInitialization";

export default function App() {
  const [libraryPath, setLibraryPath] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const setup = useAppInitialization((path) => setLibraryPath(path));

  useEffect(() => {
    BootService.bootstrap()
      .then((path) => setLibraryPath(path))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return <div className="min-h-screen bg-slate-100" />;
  }

  return (
    <AnimatePresence mode="wait">
      {libraryPath ? (
        <motion.div
          key="workspace"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
        >
          <ErrorBoundary>
            <LinguadeckWorkspace libraryPath={libraryPath} />
          </ErrorBoundary>
        </motion.div>
      ) : (
        <motion.div
          key="setup"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
        >
          <SetupScreen
            selectedPath={setup.selectedPath}
            busy={setup.busy}
            error={setup.error}
            onSelectFolder={setup.selectFolder}
            onInitialize={setup.initialize}
          />
        </motion.div>
      )}
    </AnimatePresence>
  );
}
