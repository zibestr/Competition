import asyncio
import netModule
import os
import logging
from aiogram import Bot, Dispatcher, types
from aiogram.filters.command import Command
from aiogram import F
from aiogram.fsm.context import FSMContext
from aiogram.types import Message
from aiogram.fsm.state import State, StatesGroup
# Включаем логирование, чтобы не пропустить важные сообщения
logging.basicConfig(level=logging.INFO)
# Объект бота
API_TOKEN = '7112301780:AAEeHCfAsGJoBcg545D8uIVNFKDjfD5Tmpk'
bot = Bot(token=API_TOKEN)
# Диспетчер
dp = Dispatcher()

class Form(StatesGroup):
    photo = State()
    coords_building = State()
    coords_camera = State()




# Хэндлер на команду /start
@dp.message(Command("start"))
async def cmd_start(message: types.Message,  state: FSMContext) -> None:
    await message.answer("Здравствуйте! Это бот, предназначенный для определения высоты здания по фотографии, пришлите фотографию, если хотите получить высоту")
    await state.set_state(Form.photo)


# Хэндлер на обработку текстовых сообщений
@dp.message(F.text)
async def echo(message: Message,  state: FSMContext) -> None:
    st = await state.get_state()
    if st==Form.coords_building: # Состояние ожидания координат здания
        await message.answer("Введены координаты здания. Теперь пришлите координаты камеры")
        await state.update_data(coords_building=message.text) # Запоминание данных в поле состояния
        await state.set_state(Form.coords_camera) # Перевод в состояние ожидания координат камеры
    elif st==Form.coords_camera: # Состояние ожидания координат камеры
        await message.answer("Введены координаты камеры. Происходит расчёт")
        await state.update_data(coords_camera=message.text)
        await calculate(message, state)
    else:
        await message.answer("Произошла ошибка. Попробуйте позже")



async def calculate(message: Message, state: FSMContext) -> None:
        data = await state.get_data()
        photo_path = data["photo"]
        folder = str(message.chat.id)
        coords = (data["coords_building"], data["coords_camera"])
        model_path = 'model.pth'
        result = netModule.model_calculate(photo_path, folder, coords, model_path)
        os.remove(str(message.chat.id)+"/"+photo_path)
        os.rmdir(str(message.chat.id))
        await message.answer("{:.3f} - высота дома.\n Для повторных измерений пришлите ещё фотографию здания".format(result[0][0].item()))
        await state.set_state(Form.photo)

# Хэндлер для обработки сообщений типа фото
@dp.message(F.photo)
async def get_photo(message: types.Message,  state: FSMContext) -> None:
    await state.set_state(Form.coords_building)
    await state.update_data(photo=str(message.chat.id)+"To"+str(message.photo[-1].file_id)+".jpg")
    os.makedirs(str(message.chat.id), exist_ok=True)
    await message.bot.download(file=message.photo[-1].file_id, destination=str(message.chat.id)+"/"+str(message.chat.id)+"To"+str(message.photo[-1].file_id)+".jpg")
    await message.answer("Фотография получена, теперь нужно прислать координаты здания")


# Запуск процесса поллинга новых апдейтов
async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
