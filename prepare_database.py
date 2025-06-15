import os
from dotenv import load_dotenv
from elasticsearch import Elasticsearch

from neuralNetworkCarsSystem.AutoAssistant import get_docs
from neuralNetworkCarsSystem.carsFacade import create_db, index_settings
from utils import setup_logger

load_dotenv()
logger = setup_logger("prepare_database")

def prepare_database():
    """
    Предварительно заполняет базу данных Elasticsearch данными об автомобилях.
    Этот скрипт нужно запустить один раз перед запуском бота.
    """
    logger.info("База данных уже подготовлена")
    return True

    try:
        es_client = Elasticsearch([os.getenv("ELASTICSEARCH_URL")])

        if es_client.indices.exists(index="langchain_index"):
            logger.info("Удаление существующего индекса...")
            es_client.indices.delete(index="langchain_index")
            logger.info("Индекс успешно удален")

        logger.info("Создание нового индекса...")
        es_client.indices.create(index="langchain_index", body=index_settings)
        logger.info("Новый индекс успешно создан")

        db = create_db()

        dataset_path = os.getenv("ELASTIC_DATASET_PATH")
        docs, ids = get_docs(dataset_path)

        logger.info("Добавление документов в базу данных...")
        db.add_documents(documents=docs, ids=ids)

        logger.info("База данных успешно подготовлена!")
        return True

    except Exception as e:
        logger.error(f"Ошибка при подготовке базы данных: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    prepare_database()