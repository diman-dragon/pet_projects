import requests
import json

# Попробуем правильный API эндпоинт для деталей матча (Soccer)
# Пример структуры для других видов спорта: https://site.api.espn.com/apis/site/v2/sports/soccer/eng.1/summary?event=GAME_ID
# Попробуем для FIFA World Cup:

GAME_ID = "760449"
BASE_API = "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world"

summary_url = f"{BASE_API}/summary?event={GAME_ID}"
boxscore_url = f"{BASE_API}/boxscore?event={GAME_ID}"

print(f"Testing: {summary_url}")
r_sum = requests.get(summary_url)
print(f"Status Code Summary: {r_sum.status_code}")

print(f"Testing: {boxscore_url}")
r_box = requests.get(boxscore_url)
print(f"Status Code Boxscore: {r_box.status_code}")
