import { Coffee, Github, Globe, Heart, Sparkles, Zap } from "lucide-react";

export function DonatePage() {
  return (
    <div className="mx-auto max-w-2xl px-4 py-8">
      {/* Hero */}
      <div className="relative overflow-hidden rounded-3xl bg-gradient-to-br from-[#0078D4] via-[#106EBE] to-[#005A9E] p-8 text-white shadow-2xl shadow-blue-500/30 mb-6">
        <div className="absolute -right-8 -top-8 h-40 w-40 rounded-full bg-white/10 blur-3xl" />
        <div className="absolute -bottom-6 -left-6 h-28 w-28 rounded-full bg-white/10 blur-2xl" />
        <div className="relative">
          <div className="mb-4 flex items-center gap-3">
            <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-white/20 backdrop-blur-sm ring-1 ring-white/30">
              <Heart size={24} className="text-white" />
            </div>
            <div>
              <h1 className="text-2xl font-bold tracking-tight">Поддержать проект</h1>
              <p className="text-sm text-blue-100">LinguaDeck — бесплатно и навсегда</p>
            </div>
          </div>
          <p className="text-base leading-relaxed text-blue-50">
            Мы верим, что качественный инструмент для изучения языков должен быть доступен каждому —
            без подписок, без ограничений, без рекламы.
          </p>
        </div>
      </div>

      {/* Why free */}
      <div className="rounded-3xl border border-slate-200/80 bg-white p-6 shadow-sm mb-4">
        <h2 className="mb-4 flex items-center gap-2 text-base font-bold text-slate-900">
          <Sparkles size={18} className="text-[#0078D4]" />
          Почему это бесплатно?
        </h2>
        <div className="space-y-3 text-sm leading-relaxed text-slate-600">
          <p>
            Платные приложения для изучения языков зарабатывают на твоей вовлечённости, а не на твоём прогрессе.
            Они добавляют стрики, уведомления и искусственные ограничения — лишь бы ты платил снова.
          </p>
          <p>
            LinguaDeck работает иначе: локальная база данных, твои карточки, твой темп. 
            Никакой телеметрии, никаких серверов с твоими данными. Приложение принадлежит тебе.
          </p>
          <p>
            Мы распространяем его бесплатно, потому что знание языка — это свобода.
            Доступ к нему не должен зависеть от толщины кошелька.
          </p>
        </div>
      </div>

      {/* Why we need help */}
      <div className="rounded-3xl border border-slate-200/80 bg-white p-6 shadow-sm mb-4">
        <h2 className="mb-4 flex items-center gap-2 text-base font-bold text-slate-900">
          <Zap size={18} className="text-amber-500" />
          Почему нам нужна помощь?
        </h2>
        <div className="space-y-3 text-sm leading-relaxed text-slate-600">
          <p>
            Разработка, тестирование, поддержка — всё это требует времени. Время — ресурс,
            который у людей с работой и жизнью ограничен.
          </p>
          <p>
            Каждая поддержка — это сигнал: <span className="font-semibold text-slate-800">«продолжай»</span>.
            Она помогает находить время на новые функции, исправление багов и развитие проекта
            вместо того, чтобы бросить всё на полпути.
          </p>
          <p>
            Нам не нужны миллионы. Даже маленький жест от людей, которым это полезно, меняет очень много.
          </p>
        </div>
      </div>

      {/* Donate buttons */}
      <div className="rounded-3xl border border-slate-200/80 bg-gradient-to-b from-slate-50 to-white p-6 shadow-sm mb-4">
        <h2 className="mb-4 flex items-center gap-2 text-base font-bold text-slate-900">
          <Coffee size={18} className="text-orange-500" />
          Как поддержать
        </h2>
        <div className="flex flex-col gap-3">
          <a
            href="https://buymeacoffee.com"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-3 rounded-2xl bg-[#FFDD00] px-5 py-3.5 font-semibold text-[#0D0C22] shadow-sm transition-all hover:brightness-95 hover:shadow-md active:scale-[0.98]"
          >
            <Coffee size={20} />
            <span>Купить кофе на Buy Me a Coffee</span>
          </a>
          <a
            href="https://ko-fi.com"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-3 rounded-2xl bg-[#FF5E5B] px-5 py-3.5 font-semibold text-white shadow-sm transition-all hover:brightness-95 hover:shadow-md active:scale-[0.98]"
          >
            <Heart size={20} />
            <span>Поддержать на Ko-fi</span>
          </a>
          <a
            href="https://github.com"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-3 rounded-2xl border border-slate-200 bg-white px-5 py-3.5 font-semibold text-slate-800 shadow-sm transition-all hover:bg-slate-50 hover:shadow-md active:scale-[0.98]"
          >
            <Github size={20} />
            <span>Поставить ⭐ на GitHub</span>
          </a>
        </div>
      </div>

      {/* Footer note */}
      <div className="rounded-2xl bg-slate-100/80 px-5 py-4 text-center text-xs leading-relaxed text-slate-500">
        <Globe size={14} className="inline-block mr-1.5 mb-0.5" />
        Независимо от того, поддержишь ты нас финансово или нет — LinguaDeck останется бесплатным.
        Спасибо, что пользуешься им.
      </div>
    </div>
  );
}
