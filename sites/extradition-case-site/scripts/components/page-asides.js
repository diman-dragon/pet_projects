const ASIDES = {
  timeline: {
    ru: {
      label: 'Маршрут дела',
      text: 'Раздел собирает последовательность решений в одну проверяемую линию: от корпоративного конфликта до экстрадиционных и международных процедур.',
    },
    en: {
      label: 'Case route',
      text: 'This section turns scattered decisions into one verifiable line, from the corporate conflict to extradition and international review.',
    },
    sr: {
      label: 'Tok predmeta',
      text: 'Ovaj deo sklapa rasute odluke u jednu proverljivu liniju, od korporativnog sukoba do ekstradicionih i međunarodnih postupaka.',
    },
  },
  legal: {
    ru: {
      label: 'Опорный вопрос',
      text: 'Раздел показывает, где уголовная версия расходится с обязательными актами арбитража и почему подмена гражданского спора уголовным преследованием не выдерживает проверки документами.',
    },
    en: {
      label: 'Core question',
      text: 'This section shows where the criminal narrative diverges from binding arbitrazh acts and why reframing a civil dispute as theft fails against the record.',
    },
    sr: {
      label: 'Ključno pitanje',
      text: 'Ovaj deo pokazuje gde se krivična verzija razilazi sa obaveznim arbitražnim aktima i zašto pretvaranje građanskog spora u krađu ne izdržava proveru dokumentima.',
    },
  },
  persons: {
    ru: {
      label: 'Фокус раздела',
      text: 'Этот раздел фиксирует персональный вклад каждого участника в создание «правового оксюморона». Мы показываем, как через замалчивание фактов и подмену смыслов создается версия, противоречащая документальной реальности.',
    },
    en: {
      label: 'Section focus',
      text: 'This section captures each participant’s personal contribution to the legal oxymoron. It shows how omissions and substitutions produce a version that contradicts the documentary record.',
    },
    sr: {
      label: 'Fokus odeljka',
      text: 'Ovaj deo beleži lični doprinos svakog učesnika stvaranju pravnog oksimorona. Pokazujemo kako se prećutkivanjem činjenica i zamenom značenja gradi verzija koja protivreči dokumentovanoj stvarnosti.',
    },
  },
  docs: {
    ru: {
      label: 'Как читать архив',
      text: 'Архив собран как рабочая доказательная база: сначала тематический раздел, затем конкретный документ, затем связанная версия или перевод. Так проще проверять аргументы по первоисточникам.',
    },
    en: {
      label: 'How to read the archive',
      text: 'The archive is structured as a working evidence base: start with the topic, then the specific document, then the related version or translation.',
    },
    sr: {
      label: 'Kako čitati arhivu',
      text: 'Arhiva je složena kao radna baza dokaza: prvo tematska celina, zatim konkretan dokument, pa povezana verzija ili prevod.',
    },
  },
  intl: {
    ru: {
      label: 'Зачем это важно',
      text: 'Здесь собран внешний контур проверки дела: разные международные механизмы оценивают разные риски, но вместе они показывают системную проблему экстрадиционного запроса.',
    },
    en: {
      label: 'Why it matters',
      text: 'This section collects the case’s external review track: each mechanism checks a different risk, but together they reveal the systemic problem behind the extradition request.',
    },
    sr: {
      label: 'Zašto je važno',
      text: 'Ovde je okupljen spoljašnji krug provere predmeta: različiti međunarodni mehanizmi procenjuju različite rizike, ali zajedno otkrivaju sistemski problem zahteva za izručenje.',
    },
  },
  media: {
    ru: {
      label: 'Внешняя фиксация',
      text: 'Раздел показывает, как дело выглядит со стороны независимых редакций и какие факты уже вышли за пределы внутренних процессуальных документов.',
    },
    en: {
      label: 'External record',
      text: 'This section shows how the case appears from independent newsrooms and which facts have already moved beyond internal procedural papers.',
    },
    sr: {
      label: 'Spoljašnja potvrda',
      text: 'Ovaj deo pokazuje kako predmet izgleda iz ugla nezavisnih redakcija i koje su činjenice već izašle izvan internih procesnih akata.',
    },
  },
  flagrant: {
    ru: {
      label: 'Что доказывает раздел',
      text: 'Здесь собран повторяющийся шаблон формальных отказов. В совокупности он показывает не частную ошибку, а системный отказ в доступе к правосудию по существу.',
    },
    en: {
      label: 'What this proves',
      text: 'This section collects the recurring pattern of formal refusals. Taken together, it shows not an isolated mistake but a flagrant denial of justice on the merits.',
    },
    sr: {
      label: 'Šta ovaj deo dokazuje',
      text: 'Ovde je prikupljen obrazac ponovljenih formalnih odbijanja. Zajedno, on pokazuje ne pojedinačnu grešku već flagrantno uskraćivanje pravde po suštini.',
    },
  },
  article8: {
    ru: {
      label: 'Что здесь доказано',
      text: 'Раздел фиксирует полное отсутствие анализа по ст. 8 ЕКПЧ во всех решениях сербских органов — и показывает, почему это обязательная, а не факультативная проверка при экстрадиции.',
    },
    en: {
      label: 'What this section establishes',
      text: 'This section documents the complete absence of Article 8 ECHR analysis in all Serbian decisions — and shows why that assessment is mandatory, not optional, in extradition cases.',
    },
    sr: {
      label: 'Šta ovaj deo utvrđuje',
      text: 'Ovaj deo dokumentuje potpuno odsustvo analize po članu 8 EKLJP u svim srpskim odlukama — i pokazuje zašto je ta procena obavezna, a ne fakultativna u predmetima ekstradicije.',
    },
  },
};

export function getPageAside(pageId, lang) {
  const entry = ASIDES[pageId];
  if (!entry) return { label: '', text: '' };
  return entry[lang] || entry.en;
}
