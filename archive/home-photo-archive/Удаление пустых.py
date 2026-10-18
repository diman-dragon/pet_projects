import os
import logging
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("cleanup_log.txt", encoding='utf-8'),
        logging.StreamHandler()
    ]
)

class EmptyFolderRemover:
    def __init__(self, root):
        self.root = root
        self.root.title("Удаление пустых папок")
        self.root.geometry("600x450")

        # Элементы интерфейса
        self.label = tk.Label(root, text="Выберите папку для очистки от пустых подпапок:", font=("Arial", 10))
        self.label.pack(pady=10)

        self.btn_select = tk.Button(root, text="Выбрать папку и начать", command=self.start_process, 
                                    bg="#4CAF50", fg="white", font=("Arial", 10, "bold"), padx=20, pady=10)
        self.btn_select.pack(pady=5)

        self.log_area = scrolledtext.ScrolledText(root, width=70, height=15, font=("Consolas", 9))
        self.log_area.pack(pady=10, padx=10)
        
        self.status_var = tk.StringVar(value="Ожидание выбора...")
        self.status_label = tk.Label(root, textvariable=self.status_var, fg="blue")
        self.status_label.pack()

    def log_message(self, message, level="info"):
        """Вывод сообщения в GUI и в файл лога"""
        self.log_area.insert(tk.END, f"{message}\n")
        self.log_area.see(tk.END)
        if level == "info":
            logging.info(message)
        else:
            logging.error(message)

    def remove_empty_folders(self, path):
        folders_removed = 0
        
        # walk с topdown=False критически важен: идем от самых глубоких папок к корню
        for root_dir, dirs, files in os.walk(path, topdown=False):
            for name in dirs:
                full_path = os.path.join(root_dir, name)
                try:
                    # Проверяем содержимое папки
                    if not os.listdir(full_path):
                        os.rmdir(full_path)
                        self.log_message(f"Удалена папка: {full_path}")
                        folders_removed += 1
                except Exception as e:
                    self.log_message(f"Ошибка при удалении {full_path}: {e}", "error")

        # Финальная проверка самой выбранной папки
        if os.path.exists(path) and not os.listdir(path):
            try:
                os.rmdir(path)
                self.log_message(f"Удалена корневая выбранная папка (была пуста): {path}")
                folders_removed += 1
            except Exception as e:
                self.log_message(f"Не удалось удалить корневую папку: {e}", "error")

        return folders_removed

    def start_process(self):
        target_dir = filedialog.askdirectory()
        
        if not target_dir:
            return

        self.log_area.delete(1.0, tk.END)
        self.log_message(f"--- Начало анализа папки: {target_dir} ---")
        self.status_var.set("В процессе удаления...")
        self.root.update()

        count = self.remove_empty_folders(target_dir)

        self.log_message(f"--- Завершено. Удалено папок: {count} ---")
        self.status_var.set(f"Готово. Удалено: {count}")
        messagebox.showinfo("Результат", f"Очистка завершена.\nУдалено папок: {count}\nПодробности в cleanup_log.txt")

if __name__ == "__main__":
    root = tk.Tk()
    app = EmptyFolderRemover(root)
    root.mainloop()