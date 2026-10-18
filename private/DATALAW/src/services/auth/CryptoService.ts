// src/services/auth/CryptoService.ts

/**
 * Сравнивает два Uint8Array за постоянное время (constant-time).
 *
 * Web Crypto API не предоставляет timingSafeEqual — это Node.js API.
 * Реализуем вручную: аккумулируем XOR всех байт через побитовый OR.
 * Результат ненулевой тогда и только тогда, когда хотя бы один байт отличается.
 * Время выполнения не зависит от позиции первого несовпадения.
 *
 * Длины должны совпадать до вызова — проверяй снаружи.
 */
function constantTimeEqual(a: Uint8Array, b: Uint8Array): boolean {
  let diff = 0;
  for (let i = 0; i < a.length; i++) {
    diff |= a[i] ^ b[i];
  }
  return diff === 0;
}

export const CryptoService = {
  /**
   * Генерирует случайную соль
   */
  generateSalt: (): string => {
    const array = new Uint8Array(16);
    window.crypto.getRandomValues(array);
    return Array.from(array, (byte) => byte.toString(16).padStart(2, '0')).join('');
  },

  /**
   * Создает безопасный хэш пароля с использованием PBKDF2
   */
  hashPassword: async (password: string, salt: string): Promise<string> => {
    const encoder = new TextEncoder();
    const passwordKey = await window.crypto.subtle.importKey(
      'raw',
      encoder.encode(password),
      { name: 'PBKDF2' },
      false,
      ['deriveBits', 'deriveKey'],
    );

    const hashBuffer = await window.crypto.subtle.deriveBits(
      {
        name: 'PBKDF2',
        salt: encoder.encode(salt),
        iterations: 100000,
        hash: 'SHA-256',
      },
      passwordKey,
      256,
    );

    return Array.from(new Uint8Array(hashBuffer), (byte) =>
      byte.toString(16).padStart(2, '0'),
    ).join('');
  },

  /**
   * Сверяет введённый пароль с хэшем из базы.
   *
   * Использует constant-time сравнение через побитовый XOR/OR —
   * время выполнения не зависит от того, в каком байте хэши расходятся.
   *
   * crypto.subtle.timingSafeEqual не используется: это Node.js API,
   * недоступный в браузерном окружении Tauri WebView.
   */
  verify: async (password: string, salt: string, originalHash: string): Promise<boolean> => {
    const newHash = await CryptoService.hashPassword(password, salt);

    const encoder = new TextEncoder();
    const newHashBytes = encoder.encode(newHash);
    const originalHashBytes = encoder.encode(originalHash);

    // PBKDF2-SHA256 всегда даёт 64 hex-символа — длины совпадают в норме.
    // Если вдруг нет (повреждённые данные в БД) — возвращаем false.
    if (newHashBytes.byteLength !== originalHashBytes.byteLength) {
      return false;
    }

    return constantTimeEqual(newHashBytes, originalHashBytes);
  },

  sha256: async (value: string): Promise<string> => {
    const hashBuffer = await window.crypto.subtle.digest(
      'SHA-256',
      new TextEncoder().encode(value),
    );
    return Array.from(new Uint8Array(hashBuffer), (byte) =>
      byte.toString(16).padStart(2, '0'),
    ).join('');
  },
};
