export interface LanguageOption {
  value: string;
  label: string;
  flag?: string;
  color?: string;
}

export const LANGUAGE_OPTIONS: LanguageOption[] = [
  { value: "", label: "— не указан —", color: "#6b7280" },
  { value: "en", label: "English", flag: "🇬🇧", color: "#3b82f6" },
  { value: "de", label: "Deutsch", flag: "🇩🇪", color: "#ef4444" },
  { value: "fr", label: "Français", flag: "🇫🇷", color: "#3b82f6" },
  { value: "es", label: "Español", flag: "🇪🇸", color: "#f59e0b" },
  { value: "it", label: "Italiano", flag: "🇮🇹", color: "#10b981" },
  { value: "pt", label: "Português", flag: "🇵🇹", color: "#10b981" },
  { value: "ru", label: "Русский", flag: "🇷🇺", color: "#8b5cf6" },
  { value: "zh", label: "中文", flag: "🇨🇳", color: "#f43f5e" },
  { value: "ja", label: "日本語", flag: "🇯🇵", color: "#f43f5e" },
  { value: "ko", label: "한국어", flag: "🇰🇷", color: "#f43f5e" },
  { value: "ar", label: "العربية", flag: "🇸🇦", color: "#8b5cf6" },
  { value: "tr", label: "Türkçe", flag: "🇹🇷", color: "#f59e0b" },
  { value: "pl", label: "Polski", flag: "🇵🇱", color: "#ef4444" },
  { value: "nl", label: "Nederlands", flag: "🇳🇱", color: "#f59e0b" },
  { value: "sv", label: "Svenska", flag: "🇸🇪", color: "#3b82f6" },
  { value: "sr", label: "Српски", flag: "🇷🇸", color: "#ef4444" },
  { value: "other", label: "Другой…", color: "#6b7280" },
];

export function getLanguageOption(value: string): LanguageOption {
  return LANGUAGE_OPTIONS.find((o) => o.value === value) || { value, label: value, color: "#6b7280" };
}
