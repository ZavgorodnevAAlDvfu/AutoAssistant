import os
from dotenv import load_dotenv
import psutil
from elasticsearch import Elasticsearch
from langchain_elasticsearch import ElasticsearchStore
from utils import setup_logger
from .AutoAssistant import OpenAIApi, OpenAiEmbeddings, OpenAiElasticsearchDB, AutoAssistant
from .models import ActionType, ModelResponse, Question, QuestionType
import datetime

logger = setup_logger("carsFacade")

load_dotenv(override=True)

current_pid = os.getpid()
logger.info(f"Текущий процесс бота (PID: {current_pid})")

for proc in psutil.process_iter(['pid', 'name', 'create_time']):
    if proc.info['name'] == 'python' and proc.info['pid'] != current_pid:
        create_time = datetime.fromtimestamp(proc.info['create_time']).strftime('%Y-%m-%d %H:%M:%S')
        logger.warning(f"Найден другой процесс бота (PID: {proc.info['pid']}, запущен: {create_time})")
        try:
            process = psutil.Process(proc.info['pid'])
            process.terminate()
            process.wait(timeout=3)
            logger.info(f"Процесс {proc.info['pid']} завершен")
        except psutil.TimeoutExpired:
            process.kill()
            logger.warning(f"Процесс {proc.info['pid']} принудительно завершен")


index_settings = {
            "settings": {
                "analysis": {
                    "analyzer": {
                        "keyword_lowercase": {
                            "type": "custom",
                            "tokenizer": "keyword",
                            "filter": ["lowercase"]
                        }
                    }
                }
            }
        }

def create_db():
    api = OpenAIApi(os.getenv("PROXY_LOGIN"), os.getenv("PROXY_PASSWORD"))

    es_client = Elasticsearch([os.getenv("ELASTICSEARCH_URL")])

    es_client.indices.close(index="langchain_index")

    es_client.indices.put_settings(index="langchain_index", body=index_settings)

    es_client.indices.open(index="langchain_index")
    db = OpenAiElasticsearchDB(api)

    return db


def createAutoAssistantInstance():
    db = create_db()
    assistant = AutoAssistant(db)
    return assistant

class CarsFacade:
    def __init__(self, api, index_name="langchain_index", max_query=3, promt=None):
        self.db = OpenAiElasticsearchDB(api, index_name, max_query, promt)
        logger.debug("Инициализирован CarsFacade")

    def reset(self):
        """Сбрасывает состояние диалога"""
        self.db.reset()
        logger.info("Состояние диалога сброшено")

    def process_query(self, query):
        """Обрабатывает запрос пользователя и возвращает структурированный ответ"""
        try:
            # Получаем ответ от модели
            response = self.db.post_query(query)
            
            # Если модель решила показать машины
            if response.action == ActionType.SHOW_CARS:
                # Получаем фильтр
                filter = self.db.get_filter()
                
                # Ищем подходящие машины
                results = self.db.search(query, filter)
                
                # Формируем сообщение с результатами
                if results:
                    message = "🚗 Вот подходящие варианты:\n\n"
                    for i, car in enumerate(results, 1):
                        message += f"{i}. {car.metadata['brand']} {car.metadata['model']}\n"
                        message += f"   📅 Год: {car.metadata['start_year']}-{car.metadata['end_year']}\n"
                        message += f"   💰 Цена: {car.metadata['price']:,} ₽\n"
                        message += f"   🛠 Двигатель: {car.metadata['engine_type']}, {car.metadata['horsepower']} л.с.\n"
                        message += f"   ⚙️ Коробка: {car.metadata['transmission']}\n"
                        message += f"   🚘 Привод: {car.metadata['drive']}\n"
                        message += f"   ⛽️ Расход: {car.metadata['fuel_consumption']} л/100км\n"
                        message += f"   📏 Клиренс: {car.metadata['clearance']} мм\n"
                        message += f"   💺 Мест: {car.metadata['seats']}\n"
                        message += f"   🏎 Кузов: {car.metadata['body_type']}\n\n"
                else:
                    message = "😕 К сожалению, не удалось найти подходящие варианты. Давайте уточним ваши предпочтения."
                    response.action = ActionType.ASK_QUESTION
                    response.question = Question(
                        type=QuestionType.PRIORITY,
                        text="Какие характеристики для вас наиболее важны?",
                        options=[
                            "Бюджет",
                            "Марка",
                            "Тип кузова",
                            "Тип коробки передач",
                            "Тип привода",
                            "Тип топлива",
                            "Год выпуска",
                            "Мощность двигателя",
                            "Количество мест",
                            "Клиренс"
                        ]
                    )
            
            return response
            
        except Exception as e:
            logger.error(f"Ошибка при обработке запроса: {e}", exc_info=True)
            return ModelResponse(
                action=ActionType.CLARIFY,
                message="😔 Извините, произошла ошибка. Пожалуйста, повторите ваш запрос.",
                confidence=0.0
            )

    def add_documents(self, documents):
        """Добавляет документы в базу данных"""
        try:
            self.db.add_documents(documents)
            logger.info("Документы успешно добавлены в базу данных")
        except Exception as e:
            logger.error(f"Ошибка при добавлении документов: {e}", exc_info=True)
            raise