import requests
import json

GAME_ID = "760449"
BASE_API = "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world"

summary_url = f"{BASE_API}/summary?event={GAME_ID}"

r_sum = requests.get(summary_url)
data = r_sum.json()

# Посмотрим, что внутри data['boxscore']['teams']
if 'boxscore' in data and 'teams' in data['boxscore']:
    teams = data['boxscore']['teams']
    for team in teams:
        print(f"Команда: {team['team']['displayName']}")
        if 'statistics' in team:
            print("Статистика найдена!")
            print(json.dumps(team['statistics'], indent=2))
        else:
            print("Статистика не найдена в этой команде.")
else:
    print("Структура не та.")
