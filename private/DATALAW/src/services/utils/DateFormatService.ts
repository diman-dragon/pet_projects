// src/services/utils/DateFormatService.ts
// Ответственность: форматирование дат для имён папок и ID-блоков.
// Без бизнес-логики именования.

export const DateFormatService = {
  /**
   * Дата → 'YYMMDD'
   * @example toShortDate(new Date('2025-03-15')) → '250315'
   */
  toShortDate(d?: Date | string): string {
    const date = d && !isNaN(new Date(d as string).getTime()) ? new Date(d as string) : new Date();
    const yy = date.getFullYear().toString().slice(-2);
    const mm = String(date.getMonth() + 1).padStart(2, '0');
    const dd = String(date.getDate()).padStart(2, '0');
    return `${yy}${mm}${dd}`;
  },

  /**
   * Порядковый номер + дата → 'YYMM_NN'
   * Используется как ID-блок в именах папок.
   * @example toIdBlock(2) → '2503_03'  (март 2025, третий элемент)
   */
  toIdBlock(sequence = 0, date?: Date): string {
    const d = date instanceof Date ? date : new Date();
    const yy = d.getFullYear().toString().slice(-2);
    const mm = (d.getMonth() + 1).toString().padStart(2, '0');
    const seq = Math.max(1, sequence + 1)
      .toString()
      .padStart(2, '0');
    return `${yy}${mm}_${seq}`;
  },
};
