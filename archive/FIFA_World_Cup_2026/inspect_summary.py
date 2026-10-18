import requests
import json

GAME_ID = "760449"
BASE_API = "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world"

summary_url = f"{BASE_API}/summary?event={GAME_ID}"

r_sum = requests.get(summary_url)
data = r_sum.json()

# Проверяем, есть ли в summary статистика
if "statistics" in data:
    print("Статистика найдена в summary!")
else:
    # Может быть в другом месте?
    print("Статистика не найдена в корне summary.")
    # Посмотрим ключи
    print(data.keys())
