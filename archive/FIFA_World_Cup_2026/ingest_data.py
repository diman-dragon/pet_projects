import requests
import json
import os
import time
from datetime import datetime, timedelta

# Конфигурация
BASE_URL = "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "js", "data", "final")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Турнир стартовал 11 июня 2026 — тот же диапазон дат, что и в loader_stats.py
TOURNAMENT_START = datetime(2026, 6, 11)

TEAM_NAME_MAP = {
    "United States": "США", "Bosnia-Herzegovina": "Босния и Герцеговина", "Curaçao": "Кюрасао", "Curacao": "Кюрасао",
    "Czech Republic": "Чехия", "Czechia": "Чехия", "DR Congo": "ДР Конго",
    "Congo DR": "ДР Конго",  # ESPN отдаёт именно "Congo DR", а не "DR Congo"
    "Ivory Coast": "Кот-д'Ивуар",
    "South Korea": "Южная Корея", "Mexico": "Мексика", "South Africa": "ЮАР", "Canada": "Канада",
    "Paraguay": "Парагвай", "Qatar": "Катар", "Switzerland": "Швейцария", "Brazil": "Бразилия",
    "Morocco": "Марокко", "Haiti": "Гаити", "Scotland": "Шотландия", "Australia": "Австралия",
    "Türkiye": "Турция", "Germany": "Германия", "Netherlands": "Нидерланды", "Japan": "Япония",
    "Ecuador": "Эквадор", "Sweden": "Швеция", "Tunisia": "Тунис", "Spain": "Испания",
    "Cape Verde": "Кабо-Верде", "Belgium": "Бельгия", "Egypt": "Египет", "Saudi Arabia": "Саудовская Аравия",
    "Uruguay": "Уругвай", "Iran": "Иран", "New Zealand": "Новая Зеландия", "France": "Франция",
    "Senegal": "Сенегал", "Iraq": "Ирак", "Norway": "Норвегия", "Argentina": "Аргентина",
    "Algeria": "Алжир", "Austria": "Австрия", "Jordan": "Иордания", "Portugal": "Португалия",
    "England": "Англия", "Croatia": "Хорватия", "Ghana": "Гана", "Panama": "Панама",
    "Uzbekistan": "Узбекистан", "Colombia": "Колумбия"
}

def normalize_team(name):
    return TEAM_NAME_MAP.get(name, name)

OUTPUT_PATH = os.path.join(OUTPUT_DIR, "tournament_data_pro.js")


def load_cached_data():
    """Подгружает уже сохранённый tournament_data_pro.js, чтобы не скачивать
    заново статистику матчей, которая уже есть в системе. Если файла нет/он
    повреждён — возвращает пустую структуру."""
    if not os.path.exists(OUTPUT_PATH):
        return {"matches": {}, "teams": {}}
    try:
        text = open(OUTPUT_PATH, encoding="utf-8").read()
        marker = "window.PRO_TOURNAMENT_DATA ="
        idx = text.index(marker)
        payload = text[idx + len(marker):].strip().rstrip(";").strip()
        data = json.loads(payload)
        data.setdefault("matches", {})
        data.setdefault("teams", {})
        return data
    except Exception:
        return {"matches": {}, "teams": {}}


def fetch_data():
    print("🚀 Старт сбора данных...")

    scoreboard_url = f"{BASE_URL}/scoreboard"

    # ВАЖНО: запрос без параметра "dates" отдаёт только СЕГОДНЯШНИЕ матчи ESPN.
    # Раньше скрипт делал именно так, из-за чего PRO_TOURNAMENT_DATA почти всегда
    # содержал лишь несколько матчей одного дня. Собираем события по всем дням
    # турнира, как это уже делает loader_stats.py.
    events_by_id = {}
    current_date = TOURNAMENT_START
    end_date = datetime.now() + timedelta(days=1)

    while current_date <= end_date:
        date_str = current_date.strftime("%Y%m%d")
        try:
            r = requests.get(scoreboard_url, params={"dates": date_str}, timeout=15)
            r.raise_for_status()
            day_data = r.json()
            for event in day_data.get("events", []):
                events_by_id[event["id"]] = event
        except Exception as e:
            print(f"⚠️ Ошибка при загрузке расписания на {date_str}: {e}")
        current_date += timedelta(days=1)

    # Кэш с предыдущего запуска: матчи, у которых статистика уже скачана,
    # не запрашиваем у ESPN повторно.
    cached_data = load_cached_data()
    cached_by_id = {}
    for key, match in cached_data["matches"].items():
        if match.get("statistics") and match.get("id"):
            cached_by_id[str(match["id"])] = (key, match)

    full_tournament_data = {
        "matches": {},
        "teams": {}
    }

    events = list(events_by_id.values())
    total_events = len(events)
    print(f"📦 Найдено матчей за весь турнир: {total_events}. Начинаю глубокий сбор (может занять время)...")

    processed = 0
    reused = 0
    for i, event in enumerate(events):
        match_id = event["id"]

        # Пропускаем матчи, которые ещё не начались/не завершились — у них нет
        # реального boxscore, а ESPN вместо него отдаёт предматчевую "превью"
        # статистику (totalGoals/goalDifference и т.п.) с фиктивным счётом 0:0.
        # Раньше это приводило к тому, что несыгранным матчам ошибочно
        # засчитывалась "ничья 0:0" в общей статистике.
        comp = event.get("competitions", [{}])[0]
        is_completed = comp.get("status", {}).get("type", {}).get("completed")
        if not is_completed:
            print(f"[{i+1}/{total_events}] Матч {match_id} ещё не завершён — пропускаю.")
            continue

        if match_id in cached_by_id:
            # Статистика этого матча уже есть в системе — переиспользуем её
            # без обращения к ESPN, чтобы не скачивать данные заново.
            cached_key, cached_match = cached_by_id[match_id]
            print(f"[{i+1}/{total_events}] Матч {match_id} уже есть в системе — пропускаю скачивание.")
            full_tournament_data["matches"][cached_key] = cached_match
            for comp_team in comp.get("competitors", []):
                team = comp_team.get("team", {})
                norm_name = normalize_team(team.get("displayName", ""))
                cached_team = cached_data["teams"].get(norm_name)
                if cached_team:
                    full_tournament_data["teams"][norm_name] = cached_team
                elif norm_name:
                    full_tournament_data["teams"][norm_name] = {
                        "id": team.get("id"),
                        "abbreviation": team.get("abbreviation")
                    }
            reused += 1
            continue

        print(f"[{i+1}/{total_events}] Обработка матча {match_id}...")

        summary = fetch_json(f"{BASE_URL}/summary?event={match_id}")

        if not summary:
            continue

        process_match(match_id, summary, full_tournament_data)
        processed += 1
        time.sleep(1.5)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write("window.PRO_TOURNAMENT_DATA = ")
        json.dump(full_tournament_data, f, indent=4, ensure_ascii=False)
        f.write(";")

    print(f"♻️ Взято из кэша: {reused}")
    print(f"✅ Скачано новых завершённых матчей: {processed}/{total_events}")
    print(f"✅ Данные успешно собраны и сохранены в {OUTPUT_PATH}")

def fetch_json(url):
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"⚠️ Ошибка запроса к {url}: {e}")
        return None

def process_match(match_id, summary, storage):
    header = summary.get("header", {})
    game_info = header.get("competitions", [{}])[0]
    
    home_raw = game_info['competitors'][0]['team']['displayName']
    away_raw = game_info['competitors'][1]['team']['displayName']
    
    match_key = f"{normalize_team(home_raw)} vs {normalize_team(away_raw)}"
    
    boxscore = summary.get("boxscore", {})
    team_stats = {}
    for team in boxscore.get("teams", []):
        team_stats[normalize_team(team['team']['displayName'])] = team.get("statistics", [])
    
    storage["matches"][match_key] = {
        "id": match_id,
        "score": f"{game_info['competitors'][0].get('score', '0')} - {game_info['competitors'][1].get('score', '0')}",
        "venue": game_info.get("venue", {}).get("fullName", "Неизвестно"),
        "statistics": team_stats
    }
    
    for comp in game_info['competitors']:
        team = comp['team']
        storage["teams"][normalize_team(team['displayName'])] = {
            "id": team['id'],
            "abbreviation": team.get('abbreviation')
        }

if __name__ == "__main__":
    fetch_data()
