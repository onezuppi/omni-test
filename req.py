import requests
from io import BytesIO

# URL API сервера
url = "https://34be-34-16-176-65.ngrok-free.app/process"

# URL изображения
file_url = "https://xn--80ankoagi.xn--80acgfbsl1azdqr.xn--p1ai/upload/iblock/583/29t6v9ubjnn2tin0nfm218wrkmds40i3/1.-Seven_0914.jpg"
question = "Что за животное на изображении?"

# Загрузка изображения из сети
response_image = requests.get(file_url)
if response_image.status_code == 200:
    # Преобразуем содержимое в байтовый поток
    image_file = BytesIO(response_image.content)
    # Отправка POST-запроса на сервер
    response = requests.post(
        url,
        files={"file": ("image.jpg", image_file, "image/jpeg")},
        data={"question": question}
    )
    # Вывод ответа от сервера
    print(response.text)
else:
    print(f"Ошибка при загрузке изображения: {response_image.status_code}")
