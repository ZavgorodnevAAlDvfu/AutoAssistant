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
3. Создать индекс:

`curl -X PUT "http://localhost:9200/langchain_index"`

А если надо перезапустить базу, то можно удалить индекс и создать снова:

`curl -X DELETE 'http://localhost:9200/langchain_index' `

---

