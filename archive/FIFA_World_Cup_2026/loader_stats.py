import requests
import json
import os
from datetime import datetime, timedelta

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Словарь для нормализации и перевода
TEAM_NAME_MAP = {
    "United States": "США",
    "Bosnia-Herzegovina": "Босния и Герцеговина",
    "Curaçao": "Кюрасао",
    "Curacao": "Кюрасао",
    "Czech Republic": "Чехия",
    "Czechia": "Чехия",
    "DR Congo": "ДР Конго",
    "Congo DR": "ДР Конго",  # ESPN отдаёт именно "Congo DR", а не "DR Congo"
    "Ivory Coast": "Кот-д'Ивуар",
    "South Korea": "Южная Корея",
    "Mexico": "Мексика",
    "South Africa": "ЮАР",
    "Canada": "Канада",
    "Paraguay": "Парагвай",
    "Qatar": "Катар",
    "Switzerland": "Швейцария",
    "Brazil": "Бразилия",
    "Morocco": "Марокко",
    "Haiti": "Гаити",
    "Scotland": "Шотландия",
    "Australia": "Австралия",
    "Türkiye": "Турция",
    "Germany": "Германия",
    "Netherlands": "Нидерланды",
    "Japan": "Япония",
    "Ecuador": "Эквадор",
    "Sweden": "Швеция",
    "Tunisia": "Тунис",
    "Spain": "Испания",
    "Cape Verde": "Кабо-Верде",
    "Belgium": "Бельгия",
    "Egypt": "Египет",
    "Saudi Arabia": "Саудовская Аравия",
    "Uruguay": "Уругвай",
    "Iran": "Иран",
    "New Zealand": "Новая Зеландия",
    "France": "Франция",
    "Senegal": "Сенегал",
    "Iraq": "Ирак",
    "Norway": "Норвегия",
    "Argentina": "Аргентина",
    "Algeria": "Алжир",
    "Austria": "Австрия",
    "Jordan": "Иордания",
    "Portugal": "Португалия",
    "England": "Англия",
    "Croatia": "Хорватия",
    "Ghana": "Гана",
    "Panama": "Панама",
    "Uzbekistan": "Узбекистан",
    "Colombia": "Колумбия"
}

def normalize_team(name):
    return TEAM_NAME_MAP.get(name, name)

FULL_STATS_PATH = os.path.join(BASE_DIR, "js", "data", "full_match_stats.js")
MATCH_STATS_PATH = os.path.join(BASE_DIR, "js", "data", "match_stats.js")


def load_js_object(path, var_name):
    """Читает уже сохранённый JS-файл вида `window.VAR = {...};` и возвращает
    словарь. Используется как локальный кэш, чтобы не скачивать заново то,
    что уже есть в системе. Если файла нет/он повреждён — возвращает {}."""
    if not os.path.exists(path):
        return {}
    try:
        text = open(path, encoding="utf-8").read()
        marker = f"window.{var_name} ="
        idx = text.index(marker)
        payload = text[idx + len(marker):].strip().rstrip(";").strip()
        return json.loads(payload)
    except Exception:
        return {}


def collect_stats_with_file_logs():
    base_url = "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/scoreboard"
    start_date = datetime(2026, 6, 11)
    today = datetime.now().date()
    end_date = datetime.now() + timedelta(days=1)

    # Кэш с предыдущего запуска: то, что уже скачано, не запрашиваем заново.
    all_stats_dump = load_js_object(FULL_STATS_PATH, "FULL_MATCH_STATS")
    cached_match_stats = load_js_object(MATCH_STATS_PATH, "MATCH_STATS")
    all_scores_dump = dict(cached_match_stats.get("stats", {}))

    # Даты, для которых уже есть хотя бы один кэшированный матч — раз день
    # полностью в прошлом, новых матчей на нём появиться не может, поэтому
    # повторно скачивать статистику за такой день не нужно.
    cached_dates = {info.get("date") for info in all_scores_dump.values() if info.get("date")}

    print("🚀 Сбор детальной статистики...")

    fetched_days = 0
    skipped_days = 0
    current_date = start_date
    while current_date <= end_date:
        date_str = current_date.strftime("%Y%m%d")
        iso_date = current_date.strftime("%Y-%m-%d")

        if current_date.date() < today and iso_date in cached_dates:
            # Этот день уже завершён и его статистика уже есть в системе —
            # пропускаем запрос к ESPN.
            skipped_days += 1
            current_date += timedelta(days=1)
            continue

        try:
            r = requests.get(base_url, params={"dates": date_str}, timeout=10)
            data = r.json()
            fetched_days += 1
            for event in data.get("events", []):
                comp = event.get("competitions", [{}])[0]
                if comp.get("status", {}).get("type", {}).get("completed"):
                    competitors = comp["competitors"]

                    # Названия на русском для ключа
                    home_raw = next(t for t in competitors if t["homeAway"] == "home")['team']['displayName']
                    away_raw = next(t for t in competitors if t["homeAway"] == "away")['team']['displayName']

                    home_name = normalize_team(home_raw)
                    away_name = normalize_team(away_raw)
                    match_key = f"{home_name} vs {away_name}"

                    stats_data = {}
                    score_parts = []
                    for competitor in competitors:
                        orig_name = competitor["team"]["displayName"]
                        norm_name = normalize_team(orig_name)
                        stats_data[norm_name] = {s["name"]: s["displayValue"] for s in competitor.get("statistics", [])}
                        score_parts.append((competitor["homeAway"], competitor.get("score", "0")))

                    all_stats_dump[match_key] = stats_data

                    home_score = next((s for ha, s in score_parts if ha == "home"), "0")
                    away_score = next((s for ha, s in score_parts if ha == "away"), "0")
                    all_scores_dump[match_key] = {
                        "score": f"{home_score} - {away_score}",
                        "date": iso_date
                    }
        except Exception:
            pass
        current_date += timedelta(days=1)

    os.makedirs(os.path.dirname(FULL_STATS_PATH), exist_ok=True)

    with open(FULL_STATS_PATH, "w", encoding="utf-8") as f:
        f.write("// Автоматически сгенерированный файл\n")
        f.write("window.FULL_MATCH_STATS = ")
        json.dump(all_stats_dump, f, indent=4, ensure_ascii=False)
        f.write(";")

    with open(MATCH_STATS_PATH, "w", encoding="utf-8") as f:
        f.write("// Автоматически сгенерированный файл\n")
        f.write("window.MATCH_STATS = ")
        json.dump({"stats": all_scores_dump}, f, indent=4, ensure_ascii=False)
        f.write(";")

    print(f"📅 Дней запрошено у ESPN: {fetched_days}, пропущено (уже в кэше): {skipped_days}")
    print(f"📊 Деталей собрано: {len(all_stats_dump)}")
    print(f"📊 Счетов/дат собрано: {len(all_scores_dump)}")
    print(f"✅ Файлы '{os.path.basename(FULL_STATS_PATH)}' и '{os.path.basename(MATCH_STATS_PATH)}' обновлены.")

if __name__ == "__main__":
    collect_stats_with_file_logs()
