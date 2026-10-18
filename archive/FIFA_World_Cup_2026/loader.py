import requests
import json
import os
import time
from datetime import datetime, timedelta

TEAM_NAME_MAP = {
    "United States": "США", "Bosnia-Herzegovina": "Босния и Герцеговина", "Curaçao": "Кюрасао", "Curacao": "Кюрасао",
    "Czech Republic": "Чехия", "Czechia": "Чехия", "DR Congo": "ДР Конго",
    "Congo DR": "ДР Конго",
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

TOURNAMENT_START = datetime(2026, 6, 11)

def normalize_team(name):
    return TEAM_NAME_MAP.get(name, name)

def fetch_tournament_data():
    base_url = "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world"
    scoreboard_url = f"{base_url}/scoreboard"

    print("🔄 Получение списка матчей...")

    events_by_id = {}
    # Расширяем диапазон до 30 дней назад, чтобы захватить весь турнир
    current_date = datetime.now() - timedelta(days=30)
    end_date = datetime.now() + timedelta(days=1)

    while current_date <= end_date:
        date_str = current_date.strftime("%Y%m%d")
        print(f"  За {date_str}...")
        try:
            r = requests.get(scoreboard_url, params={"dates": date_str}, timeout=10)
            r.raise_for_status()
            day_data = r.json()
            for event in day_data.get("events", []):
                events_by_id[event["id"]] = event
        except Exception as e:
            print(f"⚠️ Ошибка получения списка матчей на {date_str}: {e}")
        current_date += timedelta(days=1)

    events = list(events_by_id.values())
    if not events:
        print("⚠️ Не удалось получить ни одного матча.")
        return

    tournament_stats = {
        "matches": {},
        "teams": {}
    }

    total_events = len(events)
    for i, event in enumerate(events):
        match_id = event["id"]

        # Убираем жесткую проверку is_completed, чтобы включить предматчевые игры в расписание,
        # но статистика будет собираться только для завершенных.
        summary_url = f"{base_url}/summary?event={match_id}"

        print(f"🔍 [{i+1}/{total_events}] Загрузка деталей матча {match_id}...")
        try:
            time.sleep(0.5)
            r_sum = requests.get(summary_url, timeout=10)
            summary = r_sum.json()
        except Exception as e:
            print(f"⚠️ Ошибка получения деталей матча {match_id}: {e}")
            continue

        header = summary.get("header", {})
        game_info = header.get("competitions", [{}])[0]
        
        home_raw = game_info['competitors'][0]['team']['displayName']
        away_raw = game_info['competitors'][1]['team']['displayName']
        
        match_key = f"{normalize_team(home_raw)} vs {normalize_team(away_raw)}"
        
        match_data = {
            "id": match_id,
            "score": f"{game_info['competitors'][0].get('score', '0')} - {game_info['competitors'][1].get('score', '0')}",
            "venue": game_info.get("venue", {}).get("fullName", "Неизвестно"),
            "status": header.get("status", {}).get("type", {}).get("description"),
            "statistics": summary.get("statistics", []),
            "officials": summary.get("officials", []),
            "scoringPlays": summary.get("scoringPlays", []),
            "cards": summary.get("cards", [])
        }
        
        for comp in game_info['competitors']:
            team_name = normalize_team(comp['team']['displayName'])
            tournament_stats["teams"][team_name] = {
                "coach": comp['team'].get('coaches', []),
                "abbreviation": comp['team'].get('abbreviation')
            }
        
        tournament_stats["matches"][match_key] = match_data

    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, "js", "data", "tournament_data.js")
    
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("// Автоматически сгенерированный файл с глубокой статистикой\n")
        f.write("window.TOURNAMENT_STATS = ")
        json.dump(tournament_stats, f, indent=4, ensure_ascii=False)
        f.write(";")
        
    print(f"✅ Файл '{os.path.basename(file_path)}' успешно обновлен с данными за последние 4 дня.")

if __name__ == "__main__":
    fetch_tournament_data()
