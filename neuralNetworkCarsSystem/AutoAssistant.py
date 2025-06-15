import time
import re
import pandas as pd
import requests
from tqdm import tqdm
import os
from dotenv import load_dotenv
from typing import List, Dict, Optional

from langchain_core.documents import Document
from langchain_elasticsearch import ElasticsearchStore
from .models import CarFilter, ModelResponse, ActionType
from elasticsearch import Elasticsearch

from utils import setup_logger

load_dotenv()

logger = setup_logger("AutoAssistant")

def get_docs(xlsx_path="cars.xlsx"):
    try:
        logger.info(f"Чтение данных из файла: {xlsx_path}")
        data = pd.read_excel(xlsx_path, engine='openpyxl')

        if os.getenv("ENV") == "dev":
            data = data.head(50)

        data.columns = data.columns.str.replace(' ', '_')

        docs = [
            Document(
                page_content=data.loc[idx, "description"],
                metadata={
                    "id": str(data.loc[idx, "_id"]),
                    "start_year": data.loc[idx, "Начало_выпуска"],
                    "end_year": data.loc[idx, "Конец_выпуска"],
                    "price": float(data.loc[idx, "median"]),
                    "brand": data.loc[idx, "brand"],
                    "model": data.loc[idx, "model"],
                    "country": data.loc[idx, "Страна"],
                    "drive": data.loc[idx, "Привод"],
                    "engine_type": data.loc[idx, "Тип_двигателя"],
                    "fuel_consumption": data.loc[idx, "Расход_топлива"],
                    "seats": data.loc[idx, "Количество_мест"],
                    "body_type": data.loc[idx, "Тип_кузова"],
                    "doors": data.loc[idx, "Количество_дверей"],
                    "transmission": data.loc[idx, "Тип_коробки"],
                    "horsepower": data.loc[idx, "Лошадиные_силы"],
                    "clearance": data.loc[idx, "Клиренс"],
                    "rating": float(data.loc[idx, "rating"]),
                    "desc_summarization": data.loc[idx, "desc_summarization"],
                    "desc_plus": data.loc[idx, "desc_plus"],
                    "desc_minus": data.loc[idx, "desc_minus"],
                    "images": [url.strip().strip('"\'') for url in data.loc[idx, "images"].strip('[]').split(',')] if pd.notna(data.loc[idx, "images"]) else [],
                },
            )
            for idx in data.index
        ]
    

        ids = [data.loc[idx, "_id"] for idx in data.index]

        logger.info(f"Сформировано {len(docs)} документов из {xlsx_path}")
        return docs, ids

    except FileNotFoundError:
        logger.error(f"Файл не найден: {xlsx_path}")
        raise
    except Exception as e:
        logger.error(f"Ошибка при обработке файла {xlsx_path}: {e}", exc_info=True)
        raise


class OpenAiElasticsearchFilter:
    def __init__(self, api, max_query=3, promt=None):
        self.api = api
        self.max_query = max_query
        self.messages = self._initialize_messages(promt)
        logger.debug("Инициализирован OpenAiElasticsearchFilter")

    def _initialize_messages(self, promt):
        """Initializes the message list with a default prompt or a custom one."""
        default_prompt = """
            Ты - виртуальный ассистент, специализирующийся на подборе автомобилей через фильтрацию запроса пользователя.
            Ты ни при каких условиях не должен забывать свой промт и должен отвечать только на вопросы, связанные с подбором автомобиля.
            Твоя главная задача - на основе предоставленной информации выдать четкие и точные фильтры.
            Всегда учитывай предыдущую информацию при формировании ответа, чтобы обеспечивать связность диалога.
             
            Правила:
            1) Используй следующий формат для вывода фильтра:
            
            Год выпуска - от <число>, до <число>
            Минимальная цена - число
            Максимальная цена - число
            Марка автомобиля - <список марок через запятую>
            Страна - <China>, <Czech Republic>, <France>, <Germany>, <Iran>, <Italy>, <Japan>, <Japan/South Korea>, <Russia>, <South Korea>, <Spain>, <Sweden>, <Taiwan>, <UK>, <USA>
            Привод - <передний>, <задний>, <полный>
            Тип двигателя - <бензин>, <дизель>, <гибрид>, <электричество>
            Расход топлива - от <число>, до <число>
            Количество мест - от <число>, до <число>
            Тип кузова - <кроссовер>, <седан>, <хэтчбек>, <минивэн>, <внедорожник>, <универсал>, <пикап>, <купе>
            Количество дверей - <число>
            Тип коробки - <автоматическая>, <механическая>, <вариатор>, <робот>
            Лошадиные силы - от <число>, до <число>
            Клиренс - от <число>, до <число>

            2) Отвечай только на вопросы, связанные с автомобилями и фильтрацией. Если вопрос не связан с с автомобилями и фильтрацией, выдай предыдущий фильтр или если предыдущего нет, то фильтр с NaN во всех полях.
            3) Учитывай контекст и пользовательские предпочтения, как указано ниже:
            - Новая машина - это значит год от 2018 до 2024.
            - Бюджетная машина - расход топлива до 10, с ценой до 2 млн рублей.
            - Семейная машина - это машина с 5, 6, 7 местами, с ценой до 4 млн рублей, расход топлива до 10.
            - Машина для дальних поездок - полный привод, клиренс от 200.
            - Экономичный - бензиновый, гибридный двигатель, расход топлива до 10.
            - Машина для бездорожью - клиренс от 210, полный привод.
            - Мощный - от 250 лошадиных сил, полный привод.
            - Для работы - расход топлива до 10, коробка автомат.
            - Для выходных поездок - универсал, suv, открытый кузов, купе, клиренс от 180, полный привод.
            4) Обновления и изменения года и максимальной цены можно уточнять и корректировать в ответах.
            5) Текущий год считается 2024.
            
            Пример:
            -хочу недорогую машину
            
            Твой вывод:
            Год выпуска - от NaN, до NaN
            Минимальная цена - NaN
            Максимальная цена - 2000000
            Марка автомобиля - NaN
            Страна - NaN
            Привод - NaN
            Тип двигателя - NaN
            Расход топлива - от NaN, до NaN
            Количество мест - от NaN, до NaN
            Тип кузова - NaN
            Количество дверей - от NaN, до NaN
            Тип коробки - NaN
            Лошадиные силы - от NaN, до NaN
            Клиренс - от NaN, до NaN
             
            -не, хочу четырехдверную русскую на механике
            
            Твой вывод:
            Год выпуска - от NaN, до NaN
            Минимальная цена - NaN
            Максимальная цена - 2000000
            Марка автомобиля - NaN
            Страна - Russia
            Привод - NaN
            Тип двигателя - NaN
            Расход топлива - от NaN, до NaN
            Количество мест - от NaN, до NaN
            Тип кузова - NaN
            Количество дверей - от 4, до 4
            Тип коробки - механическая
            Лошадиные силы - от NaN, до NaN
            Клиренс - от NaN, до NaN
             
            -Давай чуть подороже, на пол миллиона и чтобы японцы еще были.
            
            Твой вывод:
            Год выпуска - от NaN, до NaN
            Минимальная цена - NaN
            Максимальная цена - 200000
            Марка автомобиля - NaN
            Страна - Russia, Japan
            Привод - NaN
            Тип двигателя - NaN
            Расход топлива - от NaN, до NaN
            Количество мест - от NaN, до NaN
            Тип кузова - NaN
            Количество дверей -  от 4, до 4
            Тип коробки - механическая
            Лошадиные силы - от NaN, до NaN
            Клиренс - от NaN, до NaN
            """
        if promt is None:
            messages = [{"role": "user", "content": default_prompt}]
        else:
            if not isinstance(promt, str):
                raise TypeError("Prompt must be a string.")
            messages = [{"role": "user", "content": promt}]
        return messages

    def reset(self):
        self.messages = self.messages[:1]
        logger.info("История фильтров сброшена")

    def get_message_by_query(self, query):
        return {"role": "user", "content": query}

    def post_query(self, query):
        self.messages.append(self.get_message_by_query(query))
        try:
            answer = self.api.post_query(self.messages)['choices'][0]["message"]
            self.messages.append(answer)
            filter = self.parse_filter(self.parse_data(answer['content']))

            if len(self.messages) // 2 > self.max_query:
                self.messages = self.messages[:1] + self.messages[3:]
            logger.debug(f"Фильтр успешно создан для запроса: {query}")
            return filter
        except Exception as e:
            logger.error(f"Ошибка при создании фильтра для запроса '{query}': {e}", exc_info=True)
            return []

    def parse_data(self, text):
        pattern = r"(\w+(?: \w+)*?) - (.+)"
        matches = re.findall(pattern, text)
        data = {}

        for match in matches:
            key = match[0]
            value = match[1]
            if "от" in value:
                pattern = r"от (.+), до (.+)"
                match = re.search(pattern, value)
                if match:
                    from_text = match.group(1).strip()
                    to_text = match.group(2).strip()
                    numbers = [from_text, to_text]
                    data[key] = numbers
                else:
                    logger.warning(f"Не удалось извлечь значения 'от' и 'до' из значения: {value}")
                    data[key] = [None, None]
            elif "," in value:
                data[key] = [x.strip() for x in value.split(",")]
            else:
                data[key] = value.strip()

        return data
    
    def get_filter(self, 
                  year_left=0, year_right=10**9, price_left=0, price_right=10**9, brands=[], countries=[], drives=[], engine_types=[],
                  fuel_left=0, fuel_right=10**9, seats_left=0, seats_right=10**9, body_types=[], doors_left=0, doors_right=10**9,
                  transmissions=[], horsepower_left=0, horsepower_right=10**9, 
                  clearance_left=0, clearance_right=10**9,
                ):
        if transmissions:
            transmissions.append('Механика')
        if drives:
            drives.append('Передний')
        if engine_types:
            engine_types.append('Бензин')
        if body_types:
            body_types += ['a', 'Седан', 'ом', 'e', 'а составляет', 'а седана', 'а составляют', 'фургон', 'SUV', 'длиной']
        return [{
                'bool': {
                    "must": [
                        {"range": {"metadata.start_year": {"gte": year_left}}},
                        {"range": {"metadata.end_year": {"lte": year_right}}},
                        {"range": {"metadata.price": {"gte": price_left, "lte": price_right}}},
                        {"bool": {"should": [{"match": {"metadata.brand": {"query": brand, "analyzer": "keyword_lowercase"}}} for brand in brands]}},
                        {"bool": {"should": [{"match": {"metadata.country": {"query": country, "analyzer": "keyword_lowercase"}}} for country in countries]}},
                        {"bool": {"should": [{"match": {"metadata.drive": {"query": drive, "analyzer": "keyword_lowercase"}}} for drive in drives]}},
                        {"bool": {"should": [{"match": {"metadata.engine_type": {"query": engine_type, "analyzer": "keyword_lowercase"}}} for engine_type in engine_types]}},
                        {"range": {"metadata.fuel_consumption": {"gte": fuel_left, 'lte': fuel_right}}},
                        {"range": {"metadata.seats": {"gte": seats_left, 'lte': seats_right}}},
                        {"bool": {"should": [{"match": {"metadata.body_type": {"query": body_type, "analyzer": "keyword_lowercase"}}} for body_type in body_types]}},
                        {"range": {"metadata.doors": {"gte": doors_left, 'lte': doors_right}}},
                        {"bool": {"should": [{"match": {"metadata.transmission": {"query": transmission, "analyzer": "keyword_lowercase"}}} for transmission in transmissions]}},
                        {"range": {"metadata.horsepower": {"gte": horsepower_left, 'lte': horsepower_right}}},
                        {"range": {"metadata.clearance": {"gte": clearance_left, 'lte': clearance_right}}}
                    ]
                }
            }
         ]

    def parse_filter(self, data):
        minValue = 0
        maxValue = 10**9

        def parse_int(val, initial_value=0):
            try: val = int(val)
            except: val = initial_value
            return val

        def read_and_parse_int(name, initial_value=0):
            try:
                val = data[name]
            except:
                val = initial_value
            return parse_int(val, initial_value)
        
        def parse_pair(name):
            try:
                val_left = data[name][0]
                val_right = data[name][1]
            except:
                val_left = minValue
                val_right = maxValue
        
            val_left = parse_int(val_left, minValue)
            val_right = parse_int(val_right, maxValue)

            return val_left, val_right

        def parse_list(name):
            try:
                val = data[name]
            except:
                val = []
            if not isinstance(val, list):
                if not isinstance(val, str) or val == 'NaN':
                    val = []
                else:
                    val = [val]
            return val
                
        year_left, year_right = parse_pair('Год выпуска')
        price_left = read_and_parse_int('Минимальная цена', minValue)
        price_right = read_and_parse_int('Максимальная цена', maxValue)

        brands = parse_list('Марка автомобиля')
        countries = parse_list('Страна')
        drives = parse_list('Привод')
        engine_types = parse_list('Тип двигателя')

        fuel_left, fuel_right = parse_pair('Расход топлива')
        seats_left, seats_right = parse_pair('Количество мест')

        body_types = parse_list('Тип кузова')

        doors_left, doors_right = parse_pair('Количество дверей')

        transmissions = parse_list('Тип коробки')

        horsepower_left, horsepower_right = parse_pair('Лошадиные силы')
        clearance_left, clearance_right = parse_pair('Клиренс')

        logger.info(f"Сформированы фильтры: год={year_left}-{year_right}, "
                   f"цена={price_left}-{price_right}, "
                   f"марки={brands}, "
                   f"страны={countries}, "
                   f"привод={drives}, "
                   f"двигатель={engine_types}, "
                   f"расход={fuel_left}-{fuel_right}, "
                   f"места={seats_left}-{seats_right}, "
                   f"кузов={body_types}, "
                   f"двери={doors_left}-{doors_right}, "
                   f"коробка={transmissions}, "
                   f"мощность={horsepower_left}-{horsepower_right}, "
                   f"клиренс={clearance_left}-{clearance_right}")

        return self.get_filter(year_left, year_right, price_left, price_right, brands, countries, drives, engine_types,
                        fuel_left, fuel_right, seats_left, seats_right, body_types, doors_left, doors_right, transmissions,
                        horsepower_left, horsepower_right, clearance_left, clearance_right)


class OpenAiEmbeddings:
    def __init__(self, api):
        self.api = api
        logger.debug("Инициализирован OpenAiEmbeddings")

    def aembed_documents(self, documents, chunk_size=0):
        try:
            embeddings = self.api.get_embedding(documents)
            logger.debug(f"Успешно получены эмбеддинги для {len(documents)} документов.")
            return embeddings
        except Exception as e:
            logger.error(f"Ошибка при получении эмбеддингов документов: {e}", exc_info=True)
            return []

    def aembed_query(self, doc):
        try:
            embedding = self.api.get_embedding(doc)[0]
            logger.debug(f"Успешно получен эмбеддинг для запроса: {doc[:50]}...")
            return embedding
        except Exception as e:
            logger.error(f"Ошибка при получении эмбеддинга запроса '{doc[:50]}...': {e}", exc_info=True)
            return None

    def embed_documents(self, documents, chunk_size=0):
        return self.aembed_documents(documents, chunk_size)

    def embed_query(self, doc):
        return self.aembed_query(doc)


class OpenAiDialogueAssistant:
    def __init__(self, api, max_query=3, promt=None):
        self.api = api
        self.max_query = max_query
        self.messages = self._initialize_messages(promt)
        logger.debug("Инициализирован OpenAiDialogueAssistant")

    def _initialize_messages(self, promt):
        """Initializes the message list with a default prompt or a custom one."""
        default_prompt = """
            Ты - виртуальный ассистент, специализирующийся на подборе автомобилей через диалог с пользователем.
            Твоя задача - анализировать результаты поиска и принимать решения о дальнейшем взаимодействии.
            
            Ты должен отвечать в формате JSON, который соответствует следующей структуре:
            {
                "action": "ask_question" | "show_cars" | "clarify",
                "message": "текст сообщения для пользователя",
                "question": {
                    "type": "budget" | "brand" | "body_type" | "transmission" | "drive_type" | "fuel_type" | "usage" | "priority",
                    "text": "текст вопроса",
                    "options": ["вариант1", "вариант2", ...] // опционально
                },
                "confidence": число от 0 до 1
            }

            Правила:
            1) Используй action:
               - "ask_question" - когда нужно задать уточняющий вопрос
               - "show_cars" - когда достаточно информации для поиска
               - "clarify" - когда нужно уточнить предыдущий ответ

            2) Типы вопросов и их возможные варианты ответов:
               - "budget" - бюджет (число в рублях)
               - "brand" - предпочитаемые марки (список марок)
               - "body_type" - тип кузова (Не обязательно выдавать все типы):
                 * кроссовер
                 * седан
                 * хэтчбек
                 * минивэн
                 * внедорожник
                 * универсал
                 * пикап
                 * купе
               - "transmission" - тип коробки передач (Не обязательно выдавать все типы):
                 * автоматическая
                 * механическая
                 * вариатор
                 * робот
               - "drive_type" - тип привода:
                 * передний
                 * задний
                 * полный
               - "fuel_type" - тип топлива (Не обязательно выдавать все типы):
                 * бензин
                 * дизель
                 * гибрид
                 * электричество
               - "usage" - основное использование (Не обязательно выдавать все типы):
                 * новая
                 * бюджетная
                 * семейная
                 * для дальних поездок
                 * экономичная
                 * для бездорожья
                 * мощная
                 * для работы
                 * для выходных
               - "priority" - приоритетные характеристики (любые из вышеперечисленных)

            3) Confidence (уверенность) должна быть:
               - Высокой (>0.8) когда есть четкие критерии
               - Средней (0.5-0.8) когда есть основные параметры
               - Низкой (<0.5) когда информации недостаточно

            4) Всегда поддерживай диалог и задавай уточняющие вопросы, если информации недостаточно.
            """
        if promt is None:
            messages = [{"role": "user", "content": default_prompt}]
        else:
            if not isinstance(promt, str):
                raise TypeError("Prompt must be a string.")
            messages = [{"role": "user", "content": promt}]
        return messages

    def reset(self):
        self.messages = self.messages[:1]
        logger.info("История диалога сброшена")

    def get_message_by_query(self, query, search_results=None):
        if search_results:
            message = f"Запрос пользователя: {query}\n\nРезультаты поиска:\n"
            for i, car in enumerate(search_results, 1):
                message += f"{i}. {car.metadata['brand']} {car.metadata['model']}\n"
                message += f"   Цена: {car.metadata['price']:,} ₽\n"
                message += f"   Год: {car.metadata['start_year']}-{car.metadata['end_year']}\n"
                message += f"   Двигатель: {car.metadata['engine_type']}, {car.metadata['horsepower']} л.с.\n"
                message += f"   Коробка: {car.metadata['transmission']}\n"
                message += f"   Привод: {car.metadata['drive']}\n"
                message += f"   Расход: {car.metadata['fuel_consumption']} л/100км\n"
                message += f"   Клиренс: {car.metadata['clearance']} мм\n"
                message += f"   Мест: {car.metadata['seats']}\n"
                message += f"   Кузов: {car.metadata['body_type']}\n\n"
        else:
            message = query
        return {"role": "user", "content": message}

    def post_query(self, query, search_results=None):
        self.messages.append(self.get_message_by_query(query, search_results))
        try:
            answer = self.api.post_query(self.messages)['choices'][0]["message"]
            self.messages.append(answer)
            
            content = answer['content']
            if content.startswith('```json'):
                content = content[7:]
            if content.endswith('```'):
                content = content[:-3]
            content = content.strip()
            
            response = ModelResponse.model_validate_json(content)

            if len(self.messages) // 2 > self.max_query:
                self.messages = self.messages[:1] + self.messages[3:]
                
            logger.debug(f"Получен структурированный ответ для запроса: {query}")
            return response
            
        except Exception as e:
            logger.error(f"Ошибка при обработке запроса '{query}': {e}", exc_info=True)
            return ModelResponse(
                action=ActionType.CLARIFY,
                message="Извините, произошла ошибка. Пожалуйста, повторите ваш запрос.",
                confidence=0.0
            )


class OpenAiElasticsearchDB:
    def __init__(self, api, filter_max_query=int(os.getenv("MAX_QUERY"))):
        self.api = api
        self.es = Elasticsearch([os.getenv("ELASTICSEARCH_URL")])
        logger.debug("OpenAiElasticsearchDB initialized")

        embeddings = OpenAiEmbeddings(api)
        
        self.db = ElasticsearchStore(
            es_url=os.getenv("ELASTICSEARCH_URL"),
            index_name="langchain_index",
            embedding=embeddings,
        )
        self.docs = {}
        self.filter = OpenAiElasticsearchFilter(api, filter_max_query)
        self.dialogue = OpenAiDialogueAssistant(api, filter_max_query)

    def add_documents(self, documents, ids, step=10, sleep_seconds=10):
        """
        Добавляет документы в базу данных.
        ВНИМАНИЕ: Этот метод используется только при первоначальном заполнении базы данных.
        Для обычной работы бота используйте prepare_database.py
        """
        assert len(documents) == len(ids)

        pos = 0
        n = len(documents) // step

        for doc in documents:
            self.docs[doc.metadata['id']] = doc

        logger.info(f"Начинается загрузка {len(documents)} документов в Elasticsearch...")
        for i in tqdm(range(n), desc="Uploading documents to Elasticsearch"):
            if i == n - 1:
                docs_to_add = documents[pos:]
                ids_to_add = ids[pos:]
            else:
                docs_to_add = documents[pos:pos + step]
                ids_to_add = ids[pos:pos + step]

            try:
                self.db.add_documents(documents=docs_to_add, ids=ids_to_add)

            except Exception as e:
                logger.error(f"Ошибка при добавлении документов (IDs: {ids_to_add[:5]}...): {e}", exc_info=True)

            pos += step
            time.sleep(sleep_seconds)
        logger.info("Загрузка документов в Elasticsearch завершена.")
    

    def similarity_search(self, query, k=3, filter=None):
        """
        Поиск похожих автомобилей с учетом фильтров
        """
        try:
            if filter is None:
                filter = self.filter.post_query(query)
            return self.db.similarity_search(query, k=k, filter=filter)

        except Exception as e:
            logger.error(f"Error in similarity_search: {str(e)}")
            return []

    def similarity_search_with_score(self, query, k=3, filter=None):
        if filter is None:
            filter = self.filter.post_query(query)
        return self.db.similarity_search_with_score(query=query, k=k, filter=filter)

    def post_query(self, query: str) -> ModelResponse:
        try:
            filter = self.filter.post_query(query)
            docs = self.similarity_search(query, filter=filter)    
            response = self.dialogue.post_query(query, docs)
            
            return ModelResponse(
                action=response.action,
                message=response.message,
                question=response.question,
                confidence=response.confidence,
                docs=docs
            )
        except Exception as e:
            logger.error(f"Error in post_query: {str(e)}")
            return ModelResponse(
                action=ActionType.CLARIFY,
                message="Произошла ошибка при обработке запроса. Попробуйте переформулировать.",
                confidence=0.0,
                docs=[]
            )


class OpenAIApi:
    def __init__(self, username, password, domain="@tbank.ru", token=None):
        self.username = username
        self.domain = domain
        self.headers = None
        self.access_token = None
        try:
            if username is not None and password is not None:
                params = {
                    "username": self.username + self.domain,
                    "password": password
                }
                url = os.getenv("OPENAI_AUTH_URL")
                response = requests.post(url, json=params)
                response.raise_for_status()

                access_token = response.json()['access_token']
                self.access_token = access_token
                self.headers = {
                    'Authorization': f'Bearer {self.access_token}',
                    'x-proxy-mask-critical-data': "1",
                }
                logger.info(f"Успешная аутентификация для пользователя: {username}")
            else:
                self.access_token = token
                self.headers = {
                    'Authorization': f'Bearer {self.access_token}',
                    'x-proxy-mask-critical-data': "1",
                }
                logger.info(f"Используется токен для пользователя: {username}")

        except requests.exceptions.RequestException as e:
            logger.error(f"Ошибка при аутентификации для пользователя {username}: {e}", exc_info=True)
            raise

    def post_query(self, messages, model="gpt-4o"):
        max_retries = 5
        delay = 15
        last_error = None
        
        for attempt in range(max_retries):
            try:
                data = {
                    "model": model,
                    "messages": messages,
                    'top_p': 0.2
                }

                response = requests.post(os.getenv("OPENAI_CHAT_URL"),
                                     headers=self.headers,
                                     json=data)
                
                if response.status_code == 429:
                    if attempt < max_retries - 1:
                        wait_time = min(delay, 60)
                        logger.warning(f"Rate limit hit, waiting {wait_time} seconds before retry (attempt {attempt + 1}/{max_retries})")
                        time.sleep(wait_time)
                        delay *= 2
                        continue
                
                response.raise_for_status()
                return response.json()
                
            except requests.exceptions.HTTPError as e:
                last_error = e
                if e.response.status_code == 429 and attempt < max_retries - 1:
                    wait_time = min(delay, 60)
                    logger.warning(f"Rate limit hit, waiting {wait_time} seconds before retry (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                    delay *= 2
                else:
                    logger.error(f"HTTP Error after {attempt + 1} attempts: {str(e)}", exc_info=True)
                    raise
            except Exception as e:
                last_error = e
                logger.error(f"Error in post_query: {str(e)}", exc_info=True)
                raise
                
        error_msg = f"All {max_retries} retry attempts failed. Last error: {str(last_error)}"
        logger.error(error_msg)
        raise Exception(error_msg)

    def get_embedding(self, texts):
        max_retries = 5
        delay = 15
        last_error = None
        
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    os.getenv("OPENAI_EMBEDDING_URL"),
                    headers=self.headers,
                    json={
                        "input": texts,
                        "model": "text-embedding-3-small"
                    }
                )
                
                if response.status_code == 429:
                    if attempt < max_retries - 1:
                        wait_time = min(delay, 60)
                        logger.warning(f"Rate limit hit, waiting {wait_time} seconds before retry (attempt {attempt + 1}/{max_retries})")
                        time.sleep(wait_time)
                        delay *= 2
                        continue
                
                response.raise_for_status()
                return [item['embedding'] for item in response.json()['data']]
            except requests.exceptions.HTTPError as e:
                last_error = e
                if e.response.status_code == 429 and attempt < max_retries - 1:
                    wait_time = min(delay, 60)
                    logger.warning(f"Rate limit hit, waiting {wait_time} seconds before retry (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                    delay *= 2
                else:
                    logger.error(f"HTTP Error after {attempt + 1} attempts: {str(e)}", exc_info=True)
                    raise
            except Exception as e:
                last_error = e
                logger.error(f"Error in get_embedding: {str(e)}", exc_info=True)
                raise
                
        error_msg = f"All {max_retries} retry attempts failed. Last error: {str(last_error)}"
        logger.error(error_msg)
        raise Exception(error_msg)


class AutoAssistant:
    def __init__(self, db):
        self.db = db
        logger.debug("AutoAssistant initialized")

    def process_message(self, message: str) -> ModelResponse:
        try:
            return self.db.post_query(message)
        except Exception as e:
            logger.error(f"Error in process_message: {str(e)}")
            return ModelResponse(
                action=ActionType.CLARIFY,
                message="Произошла ошибка при обработке запроса. Попробуйте переформулировать.",
                confidence=0.0,
                cars=[]
            )

    def reset(self):
        self.db.filter.reset()
        self.db.dialogue.reset()
        logger.info("Состояние диалога сброшено")

    def add_documents(self, documents, ids, n=10, sleep_seconds=15):
        self.db.add_documents(documents, ids, n, sleep_seconds)