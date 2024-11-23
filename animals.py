import requests
from bs4 import BeautifulSoup
import json

BASE_URL = "https://зоопарк.екатеринбург.рф/"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
}

def fetch_page(page_number):
    params = {"AJAX": "Y", "PAGEN_1": page_number}
    response = requests.get(BASE_URL + 'animals_infos/', params=params, headers=HEADERS)
    response.raise_for_status()
    return response.text

def parse_animals(page_content):
    soup = BeautifulSoup(page_content, 'html.parser')
    animal_list = []
    
    # Находим контейнер с животными
    animal_container = soup.find('ul', class_='elem-list custom list-3 type-2')
    if not animal_container:
        print("Контейнер с животными не найден.")
        return []
    
    # Находим все элементы списка
    animal_items = animal_container.find_all('li')
    for li in animal_items:
        animal_id = li.get('id', 'unknown')
        name = li.find('span', class_='df-name').get_text(strip=True)
        image = li.find('span', class_='df-img').find('img')['src']
        if not image.startswith('http'):
            image = "https://xn--80ankoagi.xn--80acgfbsl1azdqr.xn--p1ai" + image
        info = li.find('span', class_='df-ot').get_text(strip=True)
        details_link = li.find('a', class_='link')['href']
        if not details_link.startswith('http'):
            details_link = "https://xn--80ankoagi.xn--80acgfbsl1azdqr.xn--p1ai" + details_link
        
        animal_list.append({
            "id": animal_id,
            "name": name,
            "image": image,
            "info": info,
            "details_link": details_link
        })
    
    return animal_list


def main():
    all_animals = []
    seen_ids = set()
    page_number = 1
    
    while True:
        print(f"Скачиваю страницу {page_number}...")
        page_content = fetch_page(page_number)
        animals = parse_animals(page_content)
        print(animals)
        
        # Проверка на повторяющиеся данные
        current_ids = {animal['id'] for animal in animals}
        if not animals or current_ids & seen_ids:
            print("Данные начали повторяться, завершение.")
            break
        
        all_animals.extend(animals)
        seen_ids.update(current_ids)
        page_number += 1
    
    # Сохраняем результат
    output_file = "animals_data.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_animals, f, ensure_ascii=False, indent=4)
    
    print(f"Данные о животных сохранены в {output_file}.")

if __name__ == "__main__":
    main()
