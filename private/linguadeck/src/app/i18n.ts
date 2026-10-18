import i18n from "i18next";
import { initReactI18next } from "react-i18next";
import LanguageDetector from "i18next-browser-languagedetector";

const resources = {
  ru: {
    translation: {
      sidebar: {
        progress: "Прогресс",
        decks: "Колоды",
        cards: "Карточки",
        study: "Изучение",
        grammar: "Грамматика",
        donate: "Поддержать",
        settings: "Настройки",
        tagline: "Умное изучение"
      },
      dashboard: {
        header: "История и прогноз",
        subheader: "последние 30 дней · {{days}} дней вперёд",
        active_days: "{{count}} активных дней",
        future_due: "{{count}} впереди",
        today_summary: {
          reviewed: "Сегодня: {{reviewed}} повторений, еще {{remaining}} осталось",
          due: "Сегодня: {{count}} карточек к повторению"
        },
        open_details: "открыть",
        stats: {
          unique_cards: "карточек (уник.)",
          total_flips: "переворотов",
          avg_front: "среднее до пер-та",
          avg_back: "среднее с пер-том",
          reviews: "повторений",
          success: "успешно (3+4)"
        },
        legend: {
          none: "Нет",
          low: "Мало",
          high: "Много",
          today: "Сегодня",
          planned: "Запланир.",
          planned_high: "Много (план)"
        }
      },
      study: {
        header: "Изучение",
        finished: "Сессия завершена!",
        streak: {
          day_1: "день подряд",
          day_2_4: "дня подряд",
          day_many: "дней подряд"
        },
        controls: {
          flip: "Показать перевод",
          hide: "Скрыть",
          auto_rating: "автооценка",
          next_hint: "→ перейти к следующей карточке",
          flip_hint: "пробел или кнопка — показать перевод"
        },
        stats: {
          interval: "Интервал",
          ease: "Ease",
          repetitions: "Повторений",
          lapses: "Ошибок",
          last: "Последний",
          next: "Следующий",
          views: "Просмотрено",
          flips: "Перевёрнуто",
          reviews: "Оценено",
          recent_ratings: "Последние оценки"
        }
      },
      settings: {
        title: "Настройки",
        description: "Языки, алгоритм обучения и системные логи",
        tabs: {
          language: "Языки и ",
          algorithm: "Алгоритм",
          logs: "Логи"
        }
      },
      common: {
        loading: "Загрузка...",
        save: "Сохранить",
        cancel: "Отмена",
        delete: "Удалить",
        edit: "Редактировать",
        close: "Закрыть",
        no_data: "Нет данных"
      }
    }
  },
  en: {
    translation: {
      sidebar: {
        progress: "Progress",
        decks: "Decks",
        cards: "Cards",
        study: "Study",
        grammar: "Grammar",
        donate: "Support",
        settings: "Settings",
        tagline: "Smart Learning"
      },
      dashboard: {
        header: "History & Forecast",
        subheader: "last 30 days В· {{days}} days ahead",
        active_days: "{{count}} active days",
        future_due: "{{count}} ahead",
        today_summary: {
          reviewed: "Today: {{reviewed}} reviewed, {{remaining}} remaining",
          due: "Today: {{count}} cards to review"
        },
        open_details: "open",
        stats: {
          unique_cards: "cards (unique)",
          total_flips: "total flips",
          avg_front: "avg time (front)",
          avg_back: "avg time (back)",
          reviews: "reviews",
          success: "success (3+4)"
        },
        legend: {
          none: "None",
          low: "Low",
          high: "High",
          today: "Today",
          planned: "Planned",
          planned_high: "Planned High"
        }
      },
      study: {
        header: "Study",
        finished: "Session finished!",
        streak: {
          day_1: "day streak",
          day_2_4: "day streak",
          day_many: "day streak"
        },
        controls: {
          flip: "Show answer",
          hide: "Hide",
          auto_rating: "auto rating",
          next_hint: "в†’ next card",
          flip_hint: "space or button to flip"
        },
        stats: {
          interval: "Interval",
          ease: "Ease",
          repetitions: "Repetitions",
          lapses: "Lapses",
          last: "Last reviewed",
          next: "Due next",
          views: "Total views",
          flips: "Total flips",
          reviews: "Total reviews",
          recent_ratings: "Recent ratings"
        }
      },
      settings: {
        title: "Settings",
        description: "Language, algorithm and system logs",
        tabs: {
          language: "Languages",
          algorithm: "Algorithm",
          logs: "Logs"
        }
      },
      common: {
        loading: "Loading...",
        save: "Save",
        cancel: "Cancel",
        delete: "Delete",
        edit: "Edit",
        close: "Close",
        no_data: "No data"
      }
    }
  }
};

i18n
  .use(LanguageDetector)
  .use(initReactI18next)
  .init({
    resources,
    fallbackLng: "ru",
    interpolation: {
      escapeValue: false
    }
  });

export default i18n;
