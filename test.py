import sqlite3
import json
import requests
from PIL import Image
from io import BytesIO
from server import gen_answer_with_image

# Загрузка данных из файла
with open("animals_data.json", "r", encoding="utf-8") as file:
    animal_data = json.load(file)

# Создание или подключение к базе данных SQLite
conn = sqlite3.connect("animal_results.db")
cursor = conn.cursor()

# Создание таблицы для хранения результатов
cursor.execute("""
CREATE TABLE IF NOT EXISTS animal_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    animal_name TEXT NOT NULL,
    generated_answer TEXT NOT NULL,
    correct_answer TEXT NOT NULL,
    is_correct BOOLEAN NOT NULL,
    source_model TEXT NOT NULL
)
""")
conn.commit()

# Функция для проверки, обработано ли уже животное
def is_animal_processed(animal_name):
    cursor.execute("SELECT COUNT(*) FROM animal_results WHERE animal_name = ?", (animal_name,))
    return cursor.fetchone()[0] > 0

# Функция для загрузки изображения
def load_image_from_url(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return img


question = f"название одного животного на фото, без дополнительного описания, выбери из этого списка: " + ", ".join([a['name'] for a in animal_data])

print(question)

# Генерация ответов и сохранение в базу данных
for animal in animal_data:
    animal_name = animal["name"]
    image_url = animal["image"]

    # Пропускаем обработку, если животное уже есть в базе
    if is_animal_processed(animal_name):
        print(f"Skipping already processed animal: {animal_name}")
        continue

    # Загрузка изображения
    img = load_image_from_url(image_url)

    # Генерация ответа
    generated_answer = gen_answer_with_image(
        question,
        img
    )

    # Сравнение ответа
    is_correct = animal_name.lower() in generated_answer.lower()

    # Сохранение результата в базу данных
    cursor.execute("""
    INSERT INTO animal_results (animal_name, generated_answer, correct_answer, is_correct, source_model)
    VALUES (?, ?, ?, ?, ?)
    """, (animal_name, generated_answer, animal_name, is_correct, "Omni"))
    conn.commit()

    print(f"Animal: {animal_name}, Answer: {generated_answer}, Correct: {is_correct}")

# Закрытие соединения с базой данных
conn.close()
print("Processing completed. Results saved in SQLite database.")
