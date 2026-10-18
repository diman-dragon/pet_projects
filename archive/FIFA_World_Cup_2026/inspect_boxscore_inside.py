import requests
import json

GAME_ID = "760449"
BASE_API = "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world"

summary_url = f"{BASE_API}/summary?event={GAME_ID}"

r_sum = requests.get(summary_url)
data = r_sum.json()

# Посмотрим, что внутри data['boxscore']
if 'boxscore' in data:
    print("Boxscore найден в summary!")
    boxscore = data['boxscore']
    # Проверим, есть ли там statistics
    if 'statistics' in boxscore:
        print("Статистика найдена внутри boxscore!")
        print(json.dumps(boxscore['statistics'], indent=2))
    else:
        print("Статистика не найдена внутри boxscore.")
        print(boxscore.keys())
else:
    print("Boxscore не найден в summary.")
