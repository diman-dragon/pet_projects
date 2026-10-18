import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';
import path from 'node:path';

export default defineConfig({
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  plugins: [react(), tailwindcss()],
  build: {
    chunkSizeWarningLimit: 1000,
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (id.includes('node_modules/recharts') || id.includes('node_modules/d3')) {
            return 'charts';
          }

          if (id.includes('node_modules/react-calendar')) {
            return 'calendar';
          }

          if (id.includes('node_modules/lucide-react')) {
            return 'icons';
          }

          if (id.includes('@tauri-apps')) {
            return 'tauri';
          }

          if (
            id.includes('node_modules/pdfjs-dist') ||
            id.includes('node_modules/pdf-parse') ||
            id.includes('node_modules/xlsx')
          ) {
            return 'parsers';
          }

          if (id.includes('node_modules/i18next') || id.includes('node_modules/react-i18next')) {
            return 'i18n';
          }
        },
      },
    },
  },
  server: {
    port: 1420,
    strictPort: true,
    host: true,
  },
});
