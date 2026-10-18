// src/services/init/pipeline.ts
// Ответственность: запустить список шагов по порядку.
// Не знает ничего про конфиг, папки или базы — только цикл и контекст.

import { InitStep, InitContext } from './types';
import { createLogger } from '../logger/Logger';

const log = createLogger('Pipeline');

/**
 * Запускает шаги один за другим, передавая контекст по цепочке.
 * Останавливается если статус вышел из 'running' (setup / error).
 * Ошибки шага перехватываются и кладутся в контекст — не бросаются наружу.
 */
export async function runPipeline(
  steps: InitStep[],
  initialCtx: Partial<InitContext> = {},
): Promise<InitContext> {
  let ctx: InitContext = { status: 'running', ...initialCtx };

  for (const step of steps) {
    if (ctx.status !== 'running') {
      log.warn(`Шаг "${step.name}" пропущен — статус: ${ctx.status}`);
      break;
    }

    try {
      log.info(`→ ${step.name}`);
      ctx = await step.run(ctx);
    } catch (e: any) {
      const message = e?.message ?? String(e);
      log.error(`❌ ${step.name}: ${message}`);
      ctx = { ...ctx, status: 'error', error: `[${step.name}] ${message}` };
    }
  }

  return ctx;
}
