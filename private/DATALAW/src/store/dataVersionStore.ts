// src/store/dataVersionStore.ts
//
// P1-3 FIX: Zustand-хранилище версионных сигналов.
//
// Паттерн «version counter»:
//   - domain-сервис мутировал данные → вызывает bumpVersion('clients')
//   - компонент/хук читает счётчик → добавляет его в deps useEffect
//   - счётчик изменился → useEffect перезапускается → данные обновляются
//
// Преимущества перед window.dispatchEvent:
//   1. Нет риска утечки памяти: Zustand сам управляет подписками
//   2. Автоматическая отписка при размонтировании компонента
//   3. Типобезопасность: DomainKey ограничен конкретными значениями
//   4. Тестируемость: стор можно сбросить между тестами
//
// Пример использования в хуке:
//
//   import { useDataVersionStore } from '../../store/dataVersionStore';
//
//   function useMyData() {
//     const clientsVersion = useDataVersionStore((s) => s.clientsVersion);
//
//     useEffect(() => {
//       fetchClients();
//     }, [fetchClients, clientsVersion]); // ← перезагрузка при изменении
//   }

import { create } from 'zustand';

export type DomainKey = 'clients' | 'cases' | 'lines' | 'appeals';

interface DataVersionStore {
  clientsVersion: number;
  casesVersion: number;
  linesVersion: number;
  appealsVersion: number;
  bump: (key: DomainKey) => void;
}

export const useDataVersionStore = create<DataVersionStore>((set) => ({
  clientsVersion: 0,
  casesVersion: 0,
  linesVersion: 0,
  appealsVersion: 0,

  bump: (key: DomainKey) =>
    set((state) => ({
      // Приводим key к имени поля и инкрементируем
      [`${key}Version`]: (state[`${key}Version` as keyof DataVersionStore] as number) + 1,
    })),
}));

/**
 * Вспомогательная функция для вызова из domain-сервисов
 * (вне React-дерева, без useStore).
 *
 * Использование:
 *   import { bumpVersion } from '../../store/dataVersionStore';
 *   bumpVersion('clients');
 */
export const bumpVersion = (key: DomainKey): void => useDataVersionStore.getState().bump(key);
