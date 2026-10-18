import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

const host = process.env.TAURI_DEV_HOST;

export default defineConfig({
  plugins: [
    react(),
    tailwindcss(),
  ],
  clearScreen: false,
  server: {
    port: 3005,
    strictPort: false,
    host: host || false,
    hmr: host
      ? {
        protocol: "ws",
        host,
        port: 3006,
      }
      : undefined,
  },
});