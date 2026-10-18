import { useState } from "react";
import "@/App.css";
import { Canvas } from "@/components/Canvas/Canvas";
import { Header } from "@/components/Header/Header";
import type { AppTheme } from "@/types/appTheme";

export function App() {
  const [theme, setTheme] = useState<AppTheme>(() => {
    const savedTheme = localStorage.getItem("family-tree-theme");
    return savedTheme === "dark" ? "dark" : "light";
  });
  const [showAuxiliaryLines, setShowAuxiliaryLines] = useState(() => localStorage.getItem("family-tree-auxiliary-lines") !== "hidden");

  function handleThemeChange(nextTheme: AppTheme) {
    setTheme(nextTheme);
    localStorage.setItem("family-tree-theme", nextTheme);
  }

  function handleAuxiliaryLinesChange(show: boolean) {
    setShowAuxiliaryLines(show);
    localStorage.setItem("family-tree-auxiliary-lines", show ? "shown" : "hidden");
  }

  return (
    <div className={`app-shell app-shell--${theme}`}>
      <Header theme={theme} onThemeChange={handleThemeChange} showAuxiliaryLines={showAuxiliaryLines} onAuxiliaryLinesChange={handleAuxiliaryLinesChange} />
      <main className="app-shell__content">
      <Canvas showAuxiliaryLines={showAuxiliaryLines} />
      </main>
    </div>
  );
}
