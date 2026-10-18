import './styles.css';
import { App as CapApp } from '@capacitor/app';
import { api, isNative } from './native';
import { store } from './store';
import { App } from './views';
import { handleBack } from './ui';

async function boot(): Promise<void> {
  await store.init();
  const app = new App(document.getElementById('app')!);

  // аппаратная «назад»: оверлеи → экран → выход
  if (isNative) {
    void CapApp.addListener('backButton', () => {
      if (handleBack()) return;
      if (app.back()) return;
      void CapApp.minimizeApp();
    });
    // вернулись в приложение — подхватим новые снимки
    void CapApp.addListener('resume', () => {
      if (store.state === 'ready') void store.rescan();
    });
  } else {
    window.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && !handleBack()) app.back();
    });
  }

  // MediaStore изменился (новый снимок, удаление в другом приложении)
  void api.addListener('changed', () => {
    if (store.state === 'ready') void store.rescan();
  });

  await store.load();
}

void boot();
