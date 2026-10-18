import requests
import json

# Попробуем получить один матч из scoreboard и посмотреть на структуру ссылок, которые он предлагает
scoreboard_url = "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/scoreboard"
try:
    response = requests.get(scoreboard_url, timeout=10)
    data = response.json()
    events = data.get("events", [])
    if events:
        first_event = events[0]
        print(json.dumps(first_event.get("links", []), indent=2))
    else:
        print("Нет событий.")
except Exception as e:
    print(e)
