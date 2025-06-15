import pandas as pd
import json
import re


def prepare_oil(text):
    if 'дизельное' in text.lower():
        return "дизель"
    elif "электричество" in text.lower():
        return "электричество"
    else:
        return 'бензин'
    
def prepare_transmission(text):
    if 'АКПП' in text:
        return "автоматическая"
    elif "РКПП" in text:
        return "робот"
    elif "МКПП" in text:
        return "механическая"
    elif "Вариатор" in text:
        return "вариатор"
    else:
        return "автоматическая"
    

def prepare_year(text):
    expressions = text.split("-")
    numbers = []
    for expression in expressions:
        pattern = r'\d+'
        match = re.search(pattern, expression)
        if match:
            numbers.append(match.group())
        else:
            numbers.append(2024)
    return numbers


def summarization_description(query_model, api):
    messages = [
    {"role": "user", "content": f"""
        Ты - виртуальный ассистент, специализирующийся на подборе автомобилей.
        Твоя задача суммаризировать информацию о машине.
        Информация о машине: {query_model}
        Описание:
            <краткое описание модели в 3-4 предложениях>
        Плюсы:
            <краткое опиши плюсы модели в 3-4 предложениях>
        Минусы:
            <краткое опиши минусы модели в 3-4 предложениях>
                """
        }]

    answer = api.post_query(messages)
    answer_text = answer['choices'][0]['message']['content']
    return answer_text


def prepare_description(text):
    import re
    pattern = r"Описание:\s*(.*?)\nПлюсы:\s*(.*?)\nМинусы:\s*(.*)"
    matches = re.search(pattern, text, re.DOTALL)
    dict = {}
    if matches:
        description = matches.group(1).strip()
        dict['Описание'] = description
        plus = matches.group(2).strip()
        dict['Плюсы'] = plus
        minus = matches.group(3).strip()
        dict['Минусы'] = minus
    else:
        print("Текст не соответствует ожидаемому формату.")
    return dict


def prepare_date(path):
    data = pd.read_excel(path, index_col=0)
    required_fields = ['Страна сборки', 'Расход топлива в смешанном цикле, л/100 км', 'Клиренс']
    excluded_field = 'Редуктор'
    mask = data['trim_mode_datamodel'].apply(lambda x: all(field in x for field in required_fields) and excluded_field not in x)
    data = data[mask]
    data['rating'].fillna(9.0, inplace=True)
    
    def extract_json_data(x):
        json_data = json.loads(x)
        return pd.Series({
            'Количество мест': int(json_data['Число мест']),
            'Привод': json_data['Тип привода'],
            'Страна': json_data['Страна сборки'],
            'Количество дверей': int(json_data['Число дверей']),
            'Тип кузова': json_data['Тип кузова'],
            'Тип двигателя': prepare_oil(json_data['Используемое топливо']),
            'Расход топлива': float(json_data['Расход топлива в смешанном цикле, л/100 км'].replace(",", ".")),
            'Клиренс': int(json_data['Клиренс (высота дорожного просвета), мм']),
            'Лошадиные силы': int(json_data['Максимальная мощность, л.с. (кВт) при об./мин.'].split()[0]),
            'Тип коробки': prepare_transmission(json_data['Тип трансмиссии']),
            'Начало выпуска': prepare_year(json_data['Период выпуска'])[0],
            'Конец выпуска': prepare_year(json_data['Период выпуска'])[1]
        })
    
    extracted_data = data['trim_mode_datamodel'].apply(extract_json_data)
    data['Привод'] = data['Привод'].apply(lambda x: "Полный" if "Полный" in x else x)
    data = pd.concat([data, extracted_data], axis=1)
    description = []
    for desc in data['description']:
        answer_text = summarization_description(desc)
        description.append(prepare_description(answer_text))
    
    data['desc_summarization'] = [desc['Описание'] for desc in description]
    data['desc_plus'] = [desc['Плюсы'] for desc in description]
    data['desc_minus'] = [desc['Минусы'] for desc in description]

    drop_columns = ["high", "average", "low", "trim_mode_datamodel"]
    data.drop(columns=drop_columns, inplace=True)
    
    return data

