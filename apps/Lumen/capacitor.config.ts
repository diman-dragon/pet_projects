import type { CapacitorConfig } from '@capacitor/cli';

const config: CapacitorConfig = {
  appId: 'com.lumen.gallery',
  appName: 'Lumen',
  webDir: 'dist',
  backgroundColor: '#0b0b0d',
  android: {
    allowMixedContent: false,
  },
  plugins: {
    SystemBars: { insetsHandling: 'css' },
  },
};

export default config;
