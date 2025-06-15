from telegram import InputMediaPhoto, InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, CallbackQueryHandler, CommandHandler, filters
import os
from dotenv import load_dotenv
from neuralNetworkCarsSystem.carsFacade import createAutoAssistantInstance
from utils import setup_logger
import asyncio
from neuralNetworkCarsSystem.models import ActionType

logger = setup_logger("tg_bot")

load_dotenv()

API_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
user_assistants = {}


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    chat_id = update.effective_chat.id

    try:
        if user_id not in user_assistants:
            assistant = createAutoAssistantInstance()
            user_assistants[user_id] = assistant
        else:
            assistant = user_assistants[user_id]

        user_message = update.message.text
        logger.info(f"Получен запрос от пользователя {user_id}: {user_message}")

        try:
            response = assistant.process_message(user_message)
            
            if response.action == ActionType.ASK_QUESTION:
                # Если нужно задать вопрос
                message_text = response.message
                if response.question and response.question.options:
                    # Создаем клавиатуру с вариантами ответов
                    keyboard = []
                    for option in response.question.options:
                        keyboard.append([InlineKeyboardButton(option, callback_data=f"answer_{option}")])
                    reply_markup = InlineKeyboardMarkup(keyboard)
                    await context.bot.send_message(
                        chat_id=chat_id,
                        text=message_text,
                        reply_markup=reply_markup
                    )
                else:
                    await context.bot.send_message(chat_id=chat_id, text=message_text)
                    
            elif response.action == ActionType.SHOW_CARS:
                # Если нужно показать машины
                pre_message = "🔍 Начинаю поиск автомобилей по вашим критериям...\n" \
                             "Я подберу 3 наиболее подходящих варианта."
                await context.bot.send_message(chat_id=chat_id, text=pre_message)

                docs = response.docs
                logger.info(f"Найдено {len(docs)} автомобилей для запроса: {user_message}")
                
                if len(docs) == 0:
                    message_text = "❌ К сожалению, не удалось найти подходящих автомобилей по вашему запросу.\n\n" \
                                 "Попробуйте изменить критерии поиска или начать поиск заново."
                    await context.bot.send_message(chat_id=chat_id, text=message_text)
                    return

                for doc in docs:
                    metadata = doc.metadata
                    logger.debug(f"Обработка автомобиля: {metadata.get('brand')} {metadata.get('model')}")

                    brand = metadata.get('brand', '').upper()
                    model = metadata.get('model', '').upper()
                    price = metadata.get('price', 0)
                    desc_summarization = metadata.get('desc_summarization', '')
                    desc_plus = metadata.get('desc_plus', '')
                    desc_minus = metadata.get('desc_minus', '')

                    formatted_price = f"{int(price):,}".replace(',', ' ')

                    message_text = (
                        f"🚗 <b>{brand} {model}</b>\n\n"
                        f"💰 <b>Средняя цена:</b> {formatted_price} ₽\n\n"
                        f"📝 <b>Описание:</b>\n{desc_summarization}\n\n"
                        f"✅ <b>Плюсы:</b>\n{desc_plus}\n\n"
                        f"❌ <b>Минусы:</b>\n{desc_minus}\n\n"
                    )

                    brand_link = metadata.get('brand', '').lower().replace(' ', '_')
                    model_link = metadata.get('model', '').lower().replace(' ', '_')
                    link = f"https://auto.drom.ru/{brand_link}/{model_link}/"

                    message_text += f'🔗 <a href="{link}">Посмотреть объявления на Drom.ru</a>'

                    images = metadata.get('images', [])[:5]
                
                    if images:
                        message_text += f'\n\n📸 Галерея автомобиля ⬇️'

                    retry_count = 3
                    for attempt in range(retry_count):
                        try:
                            await context.bot.send_message(
                                chat_id=chat_id,
                                text=message_text,
                                parse_mode='HTML',
                                disable_web_page_preview=True
                            )
                            break
                        except Exception as e:
                            logger.error(f"Попытка {attempt + 1}/{retry_count}: Ошибка при отправке сообщения для {brand} {model}: {e}")
                            if attempt == retry_count - 1:
                                raise

                    if images:
                        try:
                            media_group = [InputMediaPhoto(media=image_url) for image_url in images]
                            await context.bot.send_media_group(chat_id=chat_id, media=media_group)
                        except Exception as e:
                            logger.error(f"Ошибка при отправке изображений для {brand} {model}: {e}")

                    await asyncio.sleep(1)

            elif response.action == ActionType.CLARIFY:
                # Если нужно уточнить
                await context.bot.send_message(chat_id=chat_id, text=response.message)

        except Exception as e:
            logger.error(f"Ошибка при обработке запроса: {e}", exc_info=True)
            await context.bot.send_message(
                chat_id=chat_id,
                text="❌ Произошла ошибка при обработке вашего запроса. Пожалуйста, попробуйте еще раз."
            )

    except Exception as e:
        logger.error(f"Ошибка при обработке сообщения от пользователя {user_id}: {e}", exc_info=True)
        await context.bot.send_message(
            chat_id=chat_id,
            text="❌ Произошла ошибка при обработке вашего запроса. Пожалуйста, попробуйте еще раз."
        )


async def reset_context(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    user_id = update.callback_query.from_user.id
    data = query.data

    if data == f"reset_{user_id}":
        assistant = user_assistants.get(user_id)
        if assistant:
            assistant.reset()
            await query.edit_message_text(text="Все забыл! Готов к новому поиску")


async def reset_context_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    assistant = user_assistants.get(user_id)
    assistant.reset()
    await context.bot.send_message(chat_id=update.effective_chat.id, text="Все забыл! Готов к новому поиску")


async def start_context_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    welcome_text = (
        "👋 Привет! Я помогу вам подобрать автомобиль.\n\n"
        "Доступные команды:\n"
        "/start - Начать диалог\n"
        "/reset - Сбросить текущий поиск\n"
        "/filter - Показать текущие фильтры\n\n"
        "Просто напишите мне ваши предпочтения, и я помогу найти подходящий автомобиль!"
    )
    await context.bot.send_message(chat_id=update.effective_chat.id, text=welcome_text)


async def handle_filter_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    chat_id = update.effective_chat.id

    try:
        if user_id not in user_assistants:
            assistant = createAutoAssistantInstance()
            user_assistants[user_id] = assistant
        else:
            assistant = user_assistants[user_id]

        filter_data = assistant.db.filter.messages[-1]['content'] if len(assistant.db.filter.messages) > 1 else "Нет активных фильтров"
        
        # Форматируем сообщение
        message = "🔍 *Текущие фильтры поиска*\n\n"
        
        # Разбиваем на секции
        sections = filter_data.split('\n\n')
        has_filters = False
        
        # Группируем фильтры по категориям
        filters = {
            "💰 Цена": [],
            "🏎 Марка и модель": [],
            "🌍 Происхождение": [],
            "🚗 Характеристики": [],
            "⚙️ Технические параметры": []
        }
        
        for section in sections:
            if section.strip():
                lines = section.split('\n')
                for line in lines:
                    if line.strip() and "NaN" not in line:
                        parts = line.split(" - ")
                        if len(parts) == 2:
                            param = parts[0].strip()
                            value = parts[1].strip()
                            
                            # Определяем категорию фильтра
                            if "цена" in param.lower():
                                filters["💰 Цена"].append((param, value))
                            elif "марка" in param.lower() or "модель" in param.lower():
                                filters["🏎 Марка и модель"].append((param, value))
                            elif "страна" in param.lower():
                                filters["🌍 Происхождение"].append((param, value))
                            elif any(x in param.lower() for x in ["тип кузова", "привод", "тип двигателя", "коробка"]):
                                filters["🚗 Характеристики"].append((param, value))
                            else:
                                filters["⚙️ Технические параметры"].append((param, value))

        # Формируем сообщение
        for category, values in filters.items():
            if values:
                has_filters = True
                message += f"{category}:\n"
                for param, value in values:
                    # Форматируем значение
                    if "от" in value and "до" in value:
                        start, end = value.replace("от", "").replace("до", "").split(",")
                        if start.strip() and end.strip():
                            message += f"• {param}: {start.strip()} - {end.strip()}\n"
                    else:
                        message += f"• {param}: {value}\n"
                message += "\n"

        # Если нет активных фильтров
        if not has_filters:
            message += "❌ Нет активных фильтров"

        # Разбиваем на части, если сообщение слишком длинное
        parts = []
        current_part = ""
        lines = message.split('\n')
        
        for line in lines:
            if len(current_part) + len(line) + 1 > 4000:
                parts.append(current_part)
                current_part = line
            else:
                current_part += line + '\n'
        
        if current_part:
            parts.append(current_part)

        # Отправляем сообщения
        if len(parts) == 1:
            await context.bot.send_message(
                chat_id=chat_id,
                text=parts[0],
                parse_mode='Markdown'
            )
        else:
            for i, part in enumerate(parts, 1):
                if i == 1:
                    await context.bot.send_message(
                        chat_id=chat_id,
                        text=f"{part}\n\n*Часть {i}/{len(parts)}*",
                        parse_mode='Markdown'
                    )
                else:
                    await context.bot.send_message(
                        chat_id=chat_id,
                        text=f"{part}\n\n*Часть {i}/{len(parts)}*",
                        parse_mode='Markdown'
                    )
                await asyncio.sleep(0.5)

    except Exception as e:
        logger.error(f"Ошибка при обработке команды /filter для пользователя {user_id}: {e}", exc_info=True)
        await context.bot.send_message(
            chat_id=chat_id,
            text="❌ Произошла ошибка при получении фильтров. Пожалуйста, попробуйте еще раз."
        )


async def handle_answer(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    user_id = update.callback_query.from_user.id
    chat_id = update.effective_chat.id
    answer = query.data.replace('answer_', '')

    try:
        if user_id not in user_assistants:
            assistant = createAutoAssistantInstance()
            user_assistants[user_id] = assistant
        else:
            assistant = user_assistants[user_id]

        # Обрабатываем ответ пользователя
        response = assistant.process_message(answer)
        
        # Удаляем сообщение с вопросом
        await query.message.delete()
        
        # Отправляем ответ пользователя как новое сообщение
        await context.bot.send_message(chat_id=chat_id, text=answer)
        
        # Обрабатываем ответ
        if response.action == ActionType.ASK_QUESTION:
            # Если нужно задать вопрос
            message_text = response.message
            if response.question and response.question.options:
                # Создаем клавиатуру с вариантами ответов
                keyboard = []
                for option in response.question.options:
                    keyboard.append([InlineKeyboardButton(option, callback_data=f"answer_{option}")])
                reply_markup = InlineKeyboardMarkup(keyboard)
                await context.bot.send_message(
                    chat_id=chat_id,
                    text=message_text,
                    reply_markup=reply_markup
                )
            else:
                await context.bot.send_message(chat_id=chat_id, text=message_text)
                
        elif response.action == ActionType.SHOW_CARS:
            # Если нужно показать машины
            pre_message = "🔍 Начинаю поиск автомобилей по вашим критериям...\n" \
                         "Я подберу 3 наиболее подходящих варианта."
            await context.bot.send_message(chat_id=chat_id, text=pre_message)

            docs = response.docs
            logger.info(f"Найдено {len(docs)} автомобилей для запроса: {answer}")
            
            if len(docs) == 0:
                message_text = "❌ К сожалению, не удалось найти подходящих автомобилей по вашему запросу.\n\n" \
                             "Попробуйте изменить критерии поиска или начать поиск заново."
                await context.bot.send_message(chat_id=chat_id, text=message_text)
                return

            for car in docs:
                metadata = car.metadata
                logger.debug(f"Обработка автомобиля: {metadata.get('brand')} {metadata.get('model')}")

                brand = metadata.get('brand', '').upper()
                model = metadata.get('model', '').upper()
                price = metadata.get('price', 0)
                desc_summarization = metadata.get('desc_summarization', '')
                desc_plus = metadata.get('desc_plus', '')
                desc_minus = metadata.get('desc_minus', '')

                formatted_price = f"{int(price):,}".replace(',', ' ')

                message_text = (
                    f"🚗 <b>{brand} {model}</b>\n\n"
                    f"💰 <b>Средняя цена:</b> {formatted_price} ₽\n\n"
                    f"📝 <b>Описание:</b>\n{desc_summarization}\n\n"
                    f"✅ <b>Плюсы:</b>\n{desc_plus}\n\n"
                    f"❌ <b>Минусы:</b>\n{desc_minus}\n\n"
                )

                brand_link = metadata.get('brand', '').lower().replace(' ', '_')
                model_link = metadata.get('model', '').lower().replace(' ', '_')
                link = f"https://auto.drom.ru/{brand_link}/{model_link}/"

                message_text += f'🔗 <a href="{link}">Посмотреть объявления на Drom.ru</a>'

                images = metadata.get('images', [])[:5]
            
                if images:
                    message_text += f'\n\n📸 Галерея автомобиля ⬇️'

                retry_count = 3
                for attempt in range(retry_count):
                    try:
                        await context.bot.send_message(
                            chat_id=chat_id,
                            text=message_text,
                            parse_mode='HTML',
                            disable_web_page_preview=True
                        )
                        break
                    except Exception as e:
                        logger.error(f"Попытка {attempt + 1}/{retry_count}: Ошибка при отправке сообщения для {brand} {model}: {e}")
                        if attempt == retry_count - 1:
                            raise

                if images:
                    try:
                        media_group = [InputMediaPhoto(media=image_url) for image_url in images]
                        await context.bot.send_media_group(chat_id=chat_id, media=media_group)
                    except Exception as e:
                        logger.error(f"Ошибка при отправке изображений для {brand} {model}: {e}")

                await asyncio.sleep(1)

        elif response.action == ActionType.CLARIFY:
            # Если нужно уточнить
            await context.bot.send_message(chat_id=chat_id, text=response.message)

    except Exception as e:
        logger.error(f"Ошибка при обработке ответа от пользователя {user_id}: {e}", exc_info=True)
        await context.bot.send_message(
            chat_id=chat_id,
            text="❌ Произошла ошибка при обработке вашего ответа. Пожалуйста, попробуйте еще раз."
        )


def main():
    application = ApplicationBuilder().token(API_TOKEN).build()
    
    # Регистрируем обработчики
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_handler(CallbackQueryHandler(handle_answer, pattern=r'^answer_.+$'))
    application.add_handler(CommandHandler('filter', handle_filter_command))
    application.add_handler(CommandHandler('reset', reset_context_command))
    application.add_handler(CommandHandler('start', start_context_command))

    logger.info("--------------------------------Бот запущен!--------------------------------")
    application.run_polling()


if __name__ == '__main__':
    main()
