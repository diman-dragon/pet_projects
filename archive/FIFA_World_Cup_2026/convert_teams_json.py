import json
import os

# Пути
TEAMS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "js", "data", "info", "teams")

def convert_json_to_js():
    for filename in os.listdir(TEAMS_DIR):
        if filename.endswith(".json"):
            js_filename = filename.replace(".json", ".js")
            
            with open(os.path.join(TEAMS_DIR, filename), 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Создаем имя переменной на основе имени файла
            var_name = "TEAM_DATA_" + os.path.splitext(filename)[0].replace("-", "_").replace(" ", "_")
            
            with open(os.path.join(TEAMS_DIR, js_filename), 'w', encoding='utf-8') as f:
                f.write(f"window.{var_name} = ")
                json.dump(data, f, indent=4, ensure_ascii=False)
                f.write(";")
            print(f"Converted {filename} to {js_filename}")

if __name__ == "__main__":
    convert_json_to_js()
