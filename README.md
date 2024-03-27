# Хакатон кафедры ВТ РТУ МИРЭА
Команда __Йцукен слышит ZoV__.
## Установка
Для корректной работы проекта требуется Python 3.11.5+
### Клонирование репозитория
```bash
git clone https://github.com/zibestr/Competition.git
cd Competition
```
### Установка необходимых пакетов и фреймворков
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
## Обработка датасета

### Расчет расстояний между координатами

Формула:\
$\Delta \sigma = arccos(sin \phi_1 sin \phi_2 + cos \phi_1 cos \phi_2 cos \Delta \lambda)$\
$d=r \Delta \sigma$\
$\phi_1$, $\phi_2$ - долгота, $\lambda_1$, $\lambda_2$ - ширина

### Нормализация
Числовые данные нормализуются по формуле:\
$x_{norm}=\frac{x-mean}{std}$

### Обработка изображений
Изображения разбираются на 3 цветовых канала и сжимаются до размера __400px на 400px__

## Нейронная сеть

### Кандидаты
Для решения были протестированы следующие архитектуры:
* AlexNet
* ResNet18
* ResNet34
* InceptionV4

### Итоговая архитектура
За основу для работы модели взята архитектура [__ResNet18__](https://arxiv.org/abs/1512.03385), в линейном слое добавлен один вход для учета расстояния между камерой и зданием.
![Где картинка??](/img/architeture.png)

## Конечный продукт
Доступ к модели осуществляется через [__Telegram Bot__](https://t.me/HeightChecker_bot).\
Все расчеты представлены в ___calculation.ipynb___, а исходный код в пакете ___src___.\
Обученная модель лежит в [___model/resnet18.pth___](model)