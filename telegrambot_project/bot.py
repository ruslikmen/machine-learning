import logging
import os
from dotenv import load_dotenv
from telegram import Update, ReplyKeyboardMarkup, KeyboardButton, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, ContextTypes, MessageHandler, 
    filters, CallbackQueryHandler, ConversationHandler
)
import requests
from gliclass import GLiClassModel, ZeroShotClassificationPipeline
from transformers import AutoTokenizer

MENU, MOVIE_INPUT, SENTIMENT_INPUT, ASK_INPUT = range(4)

load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN")
MOVIE_API_KEY = os.getenv("MOVIE_API_KEY")
GEN_API_KEY = os.getenv("GEN_API_KEY")

MOVIE_API_URL = "https://api.poiskkino.dev/v1.4/movie/search"
GEN_API_URL = "https://api.gen-api.ru/api/v1/networks/glm-5"

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

logger.info(f"BOT_TOKEN загружен: {'да' if BOT_TOKEN else 'нет'}")
logger.info(f"MOVIE_API_KEY загружен: {'да' if MOVIE_API_KEY else 'нет'}")
logger.info(f"GEN_API_KEY загружен: {'да' if GEN_API_KEY else 'нет'}")

# Загрузка модели GLiClass
try:
    logger.info("Загрузка модели GLiClass...")
    model = GLiClassModel.from_pretrained("knowledgator/gliclass-instruct-large-v1.0")
    tokenizer = AutoTokenizer.from_pretrained("knowledgator/gliclass-instruct-large-v1.0")
    classifier = ZeroShotClassificationPipeline(model, tokenizer, device='cpu')
    logger.info("Модель GLiClass успешно загружена!")
except Exception as e:
    logger.error(f"КРИТИЧЕСКАЯ ОШИБКА: Не удалось загрузить модель GLiClass: {e}")
    classifier = None

def get_main_reply_keyboard():
    """Создает основную клавиатуру с кнопками команд"""
    keyboard = [
        [KeyboardButton("/help"), KeyboardButton("/cancel")],
        [KeyboardButton("/movie"), KeyboardButton("/sentiment")],
        [KeyboardButton("/ask"), KeyboardButton("/start")]
    ]
    return ReplyKeyboardMarkup(keyboard, resize_keyboard=True, one_time_keyboard=False)

def get_inline_keyboard():
    """Создает инлайн-клавиатуру для дополнительных действий"""
    keyboard = [
        [InlineKeyboardButton("🎬 Поиск фильмов", callback_data="movie")],
        [InlineKeyboardButton("😊 Анализ тональности", callback_data="sentiment")],
        [InlineKeyboardButton("🤖 Задать вопрос", callback_data="ask")],
        [InlineKeyboardButton("❓ Помощь", callback_data="help")],
        [InlineKeyboardButton("🚪 Выход", callback_data="cancel")]
    ]
    return InlineKeyboardMarkup(keyboard)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /start"""
    welcome_text = (
        "🌟 Привет! Я обновленный многофункциональный бот.\n"
        "Теперь у меня есть кнопки для быстрого доступа к командам!\n\n"
        "📋 *Доступные команды:*\n"
        "/help - показать справку\n"
        "/cancel - отмена и возврат в меню\n"
        "/movie - поиск фильмов\n"
        "/sentiment - анализ тональности текста\n"
        "/ask - задать вопрос AI\n\n"
        "👇 Нажми на кнопку с командой или выбери действие в меню ниже:"
    )
    
    await update.message.reply_text(
        welcome_text,
        parse_mode='Markdown',
        reply_markup=get_main_reply_keyboard()
    )
    
    # Дополнительно показываем инлайн-кнопки для более удобного выбора
    await update.message.reply_text(
        "Или выбери действие здесь:",
        reply_markup=get_inline_keyboard()
    )
    return MENU

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /help"""
    help_text = (
        "📚 *Справка по командам:*\n\n"
        "🎬 */movie* - поиск фильмов\n"
        "   *Пример:* `/movie Аватар`\n\n"
        "😊 */sentiment* - анализ тональности текста\n"
        "   *Пример:* `/sentiment Отличный фильм!`\n\n"
        "🤖 */ask* - задать вопрос AI\n"
        "   *Пример:* `/ask Когда был основан Рим?`\n\n"
        "❓ */help* - показать эту справку\n"
        "🚪 */cancel* - отмена и возврат в меню\n"
        "🏠 */start* - главное меню"
    )
    
    await update.message.reply_text(
        help_text,
        parse_mode='Markdown',
        reply_markup=get_main_reply_keyboard()
    )
    return MENU

async def movie_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /movie"""
    if context.args:
        # Если есть аргументы, сразу ищем фильм
        query = " ".join(context.args)
        await update.message.reply_text(f"🔍 Ищу фильм: {query}")
        return await search_movie(update, context, query)
    else:
        # Если нет аргументов, запрашиваем название
        await update.message.reply_text(
            "🎬 Введите название фильма для поиска:",
            reply_markup=ReplyKeyboardMarkup(
                [[KeyboardButton("🔙 Отмена")]], 
                resize_keyboard=True, 
                one_time_keyboard=True
            )
        )
        return MOVIE_INPUT

async def sentiment_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /sentiment"""
    if classifier is None:
        await update.message.reply_text(
            "❌ Модель анализа тональности не загружена. Попробуйте позже.",
            reply_markup=get_main_reply_keyboard()
        )
        return MENU
    
    if context.args:
        # Если есть аргументы, сразу анализируем
        text = " ".join(context.args)
        return await analyze_sentiment(update, context, text)
    else:
        # Если нет аргументов, запрашиваем текст
        await update.message.reply_text(
            "😊 Введите текст для анализа тональности:",
            reply_markup=ReplyKeyboardMarkup(
                [[KeyboardButton("🔙 Отмена")]], 
                resize_keyboard=True, 
                one_time_keyboard=True
            )
        )
        return SENTIMENT_INPUT

async def ask_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /ask"""
    if not GEN_API_KEY:
        await update.message.reply_text(
            "❌ API ключ Gen API не настроен.",
            reply_markup=get_main_reply_keyboard()
        )
        return MENU
    
    if context.args:
        # Если есть аргументы, сразу задаем вопрос
        question = " ".join(context.args)
        return await ask_question(update, context, question)
    else:
        # Если нет аргументов, запрашиваем вопрос
        await update.message.reply_text(
            "🤖 Введите ваш вопрос:",
            reply_markup=ReplyKeyboardMarkup(
                [[KeyboardButton("🔙 Отмена")]], 
                resize_keyboard=True, 
                one_time_keyboard=True
            )
        )
        return ASK_INPUT

async def search_movie(update: Update, context: ContextTypes.DEFAULT_TYPE, query=None):
    """Поиск фильмов"""
    if query is None:
        query = " ".join(context.args) if context.args else update.message.text
    
    if query.lower() == '🔙 отмена':
        await update.message.reply_text(
            "Поиск отменен. Возвращаюсь в меню.",
            reply_markup=get_main_reply_keyboard()
        )
        return MENU
    
    await update.message.chat.send_action(action="typing")
    
    try:
        headers = {"X-API-KEY": MOVIE_API_KEY}
        params = {"query": query}
        response = requests.get(MOVIE_API_URL, headers=headers, params=params, timeout=10)

        if response.status_code != 200:
            logger.error(f"Movie API error: {response.status_code} - {response.text}")
            await update.message.reply_text(
                "❌ Сервис поиска фильмов временно недоступен.",
                reply_markup=get_main_reply_keyboard()
            )
            return MENU

        data = response.json()
        movies = data.get("docs", [])
        
        if not movies:
            await update.message.reply_text(
                f"❌ По запросу «{query}» ничего не найдено.",
                reply_markup=get_main_reply_keyboard()
            )
            return MENU

        # Показываем первые 3 фильма
        reply_lines = [f"📽 *Результаты поиска по запросу:* «{query}»"]
        for i, movie in enumerate(movies[:3], 1):
            name = movie.get("name") or movie.get("alternativeName") or "Без названия"
            year = movie.get("year", "неизвестно")
            rating = movie.get("rating", {}).get("kp", "—")
            description = movie.get("shortDescription") or movie.get("description", "")
            if len(description) > 100:
                description = description[:100] + "…"
            
            reply_lines.append(
                f"\n*{i}. {name}* ({year})\n"
                f"⭐️ Рейтинг: {rating}\n"
                f"📝 {description}"
            )

        await update.message.reply_text(
            "\n".join(reply_lines),
            parse_mode='Markdown',
            reply_markup=get_main_reply_keyboard()
        )

    except requests.exceptions.Timeout:
        await update.message.reply_text(
            "⏳ Сервер не ответил вовремя. Попробуйте позже.",
            reply_markup=get_main_reply_keyboard()
        )
    except Exception as e:
        logger.error(f"Ошибка поиска фильмов: {e}")
        await update.message.reply_text(
            "❌ Не удалось выполнить поиск. Попробуйте позже.",
            reply_markup=get_main_reply_keyboard()
        )
    
    return MENU

async def analyze_sentiment(update: Update, context: ContextTypes.DEFAULT_TYPE, text=None):
    """Анализ тональности текста"""
    if text is None:
        text = " ".join(context.args) if context.args else update.message.text
    
    if text.lower() == '🔙 отмена':
        await update.message.reply_text(
            "Анализ отменен. Возвращаюсь в меню.",
            reply_markup=get_main_reply_keyboard()
        )
        return MENU
    
    await update.message.chat.send_action(action="typing")
    
    try:
        labels = ["positive", "negative", "neutral"]
        prompt = "Analyze the sentiment of this text:"
        
        logger.info(f"Анализ тональности для текста: {text[:50]}...")
        
        results = classifier(text, labels, prompt=prompt, threshold=0.0)[0]
        best_result = max(results, key=lambda x: x['score'])
        label = best_result['label']
        score = best_result['score']

        if label == "positive":
            emotion = "😊 положительная"
        elif label == "negative":
            emotion = "😞 отрицательная"
        else:
            emotion = "😐 нейтральная"

        await update.message.reply_text(
            f"📊 *Результат анализа:*\n\n"
            f"Текст: _{text[:100]}{'...' if len(text) > 100 else ''}_\n"
            f"Тональность: {emotion}\n"
            f"Уверенность: {score:.2%}",
            parse_mode='Markdown',
            reply_markup=get_main_reply_keyboard()
        )

    except Exception as e:
        logger.error(f"Ошибка анализа с GLiClass: {e}", exc_info=True)
        await update.message.reply_text(
            "❌ Не удалось выполнить анализ тональности.",
            reply_markup=get_main_reply_keyboard()
        )
    
    return MENU

async def ask_question(update: Update, context: ContextTypes.DEFAULT_TYPE, question=None):
    """Задать вопрос к Gen API"""
    if question is None:
        question = " ".join(context.args) if context.args else update.message.text
    
    if question.lower() == '🔙 отмена':
        await update.message.reply_text(
            "Вопрос отменен. Возвращаюсь в меню.",
            reply_markup=get_main_reply_keyboard()
        )
        return MENU
    
    await update.message.chat.send_action(action="typing")
    
    payload = {
        "is_sync": True,
        "messages": [{"role": "user", "content": question}]
    }

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {GEN_API_KEY}"
    }

    try:
        logger.info("Отправка запроса к Gen API")
        response = requests.post(GEN_API_URL, json=payload, headers=headers, timeout=60)
        response.raise_for_status()

        data = response.json()
        logger.info("Ответ от Gen API получен")
        
        answer = None

        if isinstance(data, dict):
            if data.get("response") and isinstance(data["response"], list) and len(data["response"]) > 0:
                first_response = data["response"][0]
                if isinstance(first_response, dict):
                    if first_response.get("choices") and isinstance(first_response["choices"], list) and len(first_response["choices"]) > 0:
                        choice = first_response["choices"][0]
                        if isinstance(choice, dict):
                            if choice.get("message") and isinstance(choice["message"], dict) and choice["message"].get("content"):
                                answer = choice["message"]["content"]
                            elif choice.get("text"):
                                answer = choice["text"]

            if answer is None and data.get("choices") and isinstance(data["choices"], list) and len(data["choices"]) > 0:
                choice = data["choices"][0]
                if isinstance(choice, dict):
                    if choice.get("message") and isinstance(choice["message"], dict) and choice["message"].get("content"):
                        answer = choice["message"]["content"]
                    elif choice.get("text"):
                        answer = choice["text"]

            if answer is None and data.get("content"):
                answer = data["content"]

        if answer is None:
            answer = "❌ Не удалось получить ответ от API."

        MAX_LEN = 4000
        if len(answer) > MAX_LEN:
            answer = answer[:MAX_LEN] + "\n\n... (сообщение обрезано)"

        await update.message.reply_text(
            f"🤖 *Вопрос:* {question}\n\n*Ответ:*\n{answer}",
            parse_mode='Markdown',
            reply_markup=get_main_reply_keyboard()
        )

    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code
        error_body = e.response.text
        logger.error(f"HTTP ошибка {status_code}: {error_body}")
        
        error_message = "❌ Ошибка при обращении к API."
        if status_code == 402:
            error_message = "❌ Недостаточно средств на счету Gen API."
        elif status_code == 401:
            error_message = "❌ Ошибка аутентификации. Проверьте API ключ."
        elif status_code == 429:
            error_message = "❌ Превышен лимит запросов. Попробуйте позже."
            
        await update.message.reply_text(
            error_message,
            reply_markup=get_main_reply_keyboard()
        )
    except Exception as e:
        logger.error(f"Неизвестная ошибка: {e}", exc_info=True)
        await update.message.reply_text(
            "❌ Не удалось получить ответ. Попробуйте позже.",
            reply_markup=get_main_reply_keyboard()
        )
    
    return MENU

async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик нажатий на инлайн-кнопки"""
    query = update.callback_query
    await query.answer()
    
    if query.data == "movie":
        await query.edit_message_text(
            "🎬 Введите название фильма для поиска:",
            reply_markup=InlineKeyboardMarkup([[
                InlineKeyboardButton("🔙 Вернуться в меню", callback_data="back_to_menu")
            ]])
        )
        return MOVIE_INPUT
    
    elif query.data == "sentiment":
        if classifier is None:
            await query.edit_message_text(
                "❌ Модель анализа тональности не загружена.",
                reply_markup=InlineKeyboardMarkup([[
                    InlineKeyboardButton("🔙 Вернуться в меню", callback_data="back_to_menu")
                ]])
            )
            return MENU
        
        await query.edit_message_text(
            "😊 Введите текст для анализа тональности:",
            reply_markup=InlineKeyboardMarkup([[
                InlineKeyboardButton("🔙 Вернуться в меню", callback_data="back_to_menu")
            ]])
        )
        return SENTIMENT_INPUT
    
    elif query.data == "ask":
        if not GEN_API_KEY:
            await query.edit_message_text(
                "❌ API ключ Gen API не настроен.",
                reply_markup=InlineKeyboardMarkup([[
                    InlineKeyboardButton("🔙 Вернуться в меню", callback_data="back_to_menu")
                ]])
            )
            return MENU
        
        await query.edit_message_text(
            "🤖 Введите ваш вопрос:",
            reply_markup=InlineKeyboardMarkup([[
                InlineKeyboardButton("🔙 Вернуться в меню", callback_data="back_to_menu")
            ]])
        )
        return ASK_INPUT
    
    elif query.data == "help":
        help_text = (
            "📚 *Доступные команды:*\n\n"
            "🎬 */movie* - поиск фильмов\n"
            "😊 */sentiment* - анализ тональности\n"
            "🤖 */ask* - задать вопрос AI\n"
            "❓ */help* - справка\n"
            "🚪 */cancel* - отмена"
        )
        await query.edit_message_text(
            help_text,
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([[
                InlineKeyboardButton("🔙 Вернуться в меню", callback_data="back_to_menu")
            ]])
        )
        return MENU
    
    elif query.data == "cancel" or query.data == "back_to_menu":
        await query.edit_message_text(
            "Главное меню. Выберите действие:",
            reply_markup=get_inline_keyboard()
        )
        return MENU

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /cancel"""
    await update.message.reply_text(
        "🚪 Возвращаюсь в главное меню.",
        reply_markup=get_main_reply_keyboard()
    )
    return MENU

async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик текстовых сообщений вне диалога"""
    await update.message.reply_text(
        "Я не понимаю эту команду. Используйте кнопки или /help для списка команд.",
        reply_markup=get_main_reply_keyboard()
    )
    return MENU

def main():
    if not BOT_TOKEN:
        logger.error("BOT_TOKEN не найден! Бот не может запуститься.")
        return

    application = Application.builder().token(BOT_TOKEN).build()

    conv_handler = ConversationHandler(
        entry_points=[
            CommandHandler("start", start),
            CommandHandler("help", help_command),
            CommandHandler("movie", movie_command),
            CommandHandler("sentiment", sentiment_command),
            CommandHandler("ask", ask_command),
            CommandHandler("cancel", cancel),
            CallbackQueryHandler(button_handler)
        ],
        states={
            MENU: [
                CommandHandler("help", help_command),
                CommandHandler("movie", movie_command),
                CommandHandler("sentiment", sentiment_command),
                CommandHandler("ask", ask_command),
                CommandHandler("cancel", cancel),
                CommandHandler("start", start),
                CallbackQueryHandler(button_handler),
                MessageHandler(filters.TEXT & ~filters.COMMAND, echo)
            ],
            MOVIE_INPUT: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, search_movie),
                CommandHandler("cancel", cancel),
                CallbackQueryHandler(button_handler)
            ],
            SENTIMENT_INPUT: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, analyze_sentiment),
                CommandHandler("cancel", cancel),
                CallbackQueryHandler(button_handler)
            ],
            ASK_INPUT: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, ask_question),
                CommandHandler("cancel", cancel),
                CallbackQueryHandler(button_handler)
            ],
        },
        fallbacks=[CommandHandler("cancel", cancel)]
    )

    application.add_handler(conv_handler)

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("movie", movie_command))
    application.add_handler(CommandHandler("sentiment", sentiment_command))
    application.add_handler(CommandHandler("ask", ask_command))
    application.add_handler(CommandHandler("cancel", cancel))

    application.add_handler(CallbackQueryHandler(button_handler))

    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo))

    application.run_polling()

if __name__ == "__main__":
    main()