import os
from dotenv import load_dotenv
import psutil
from elasticsearch import Elasticsearch
from langchain_elasticsearch import ElasticsearchStore
from utils import setup_logger
from .AutoAssistant import OpenAIApi, OpenAiEmbeddings, OpenAiElasticsearchDB, AutoAssistant
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