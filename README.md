### Активируем окружение и установим зависимости

```
source venv/bin/activate
pip install -r requirements.txt
```
---
### Поднимаем Elasticsearch

1. Запустите docker
2. Запустите образ 
```
docker run -p 9200:9200 -e "discovery.type=single-node" -e "xpack.security.enabled=false" -e "xpack.security.http.ssl.enabled=false" docker.elastic.co/elasticsearch/elasticsearch:8.12.1
```

---
Заполняем базу данных

```
python prepare_database.py
```

---
Запускаем бота
```
python tg_bot.py
```
