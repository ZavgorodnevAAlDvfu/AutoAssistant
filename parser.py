import json
import os
import random
from collections import Counter, defaultdict

import numpy as np
import requests
from dotenv import load_dotenv
from lxml import html
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from utils import setup_logger
from bs4 import BeautifulSoup
import time

logger = setup_logger(__name__)

load_dotenv()

CAR_BRANDS = [
    "Abarth", "Acura", "AITO", "Aiways", "Alfa Romeo", "Alpina", "Arcfox", "Aston Martin", "Audi", "Avatr",
    "BAIC", "Bajaj", "Barkas", "BAW", "Bedford", "Belgee", "Bentley", "BMW", "Brilliance", "Bugatti", "Buick", "BYD",
    "Cadillac", "Changan", "Changhe", "Chery", "Chevrolet", "Chrysler", "Ciimo", "Citroen", "Dacia", "Dadi", "Daewoo",
    "Daihatsu", "Datsun", "Dayun", "DeLorean", "Denza", "Dodge", "Dongfeng", "Dorcen", "DW Hower", "EXEED", "FAW",
    "Ferrari", "Fiat", "Fisker", "Ford", "Forthing", "Foton", "GAC", "Geely", "Genesis", "Geo", "GMC", "Great Wall",
    "Hafei", "Haima", "Haval", "Hawtai", "HiPhi", "Holden", "Honda", "Hongqi", "Hozon", "Huanghai", "Hummer", "Hyundai",
    "iCAR", "IM Motors", "Infiniti", "Iran Khodro", "Isuzu", "JAC", "Jaecoo", "Jaguar", "Jeep", "Jetour", "Jetta",
    "Jinbei", "JMC", "JMEV", "Kaiyi", "KG Mobility", "Kia", "Knewstar", "Kuayue", "Lamborghini", "Lancia", "Land Rover",
    "Landwind", "Leapmotor", "Lexus", "Li", "Lifan", "Lincoln", "Livan", "Lotus", "Lucid", "Luxeed", "Luxgen", "Lynk & Co",
    "M-Hero", "Marussia", "Maserati", "Maxus", "Maybach", "Mazda", "McLaren", "Mercedes-Benz", "Mercury", "MG", "MINI",
    "Mitsubishi", "Mitsuoka", "Nio", "Nissan", "Oldsmobile", "OMODA", "Opel", "ORA", "Oshan", "Oting", "Pagani", "Perodua",
    "Peugeot", "Piaggio", "Plymouth", "Polar Stone", "Polestar", "Pontiac", "Porsche", "Qingling", "Radar", "RAM", "Ravon",
    "Renault", "Renault Samsung", "Rimac", "Rising Auto", "Rivian", "Roewe", "Rolls-Royce", "Rover", "Saab", "SAIPA",
    "Saturn", "Scion", "SEAT", "Seres", "Shineray", "Shuanghuan", "Skoda", "Skywell", "Smart", "Solaris", "Soueast",
    "SsangYong", "Subaru", "Suzuki", "SWM", "Tank", "Tesla", "Tianma", "Tianye", "Toyota", "Venucia", "VGV", "Volkswagen",
    "Volvo", "Vortex", "Voyah", "Wartburg", "Weltmeister", "WEY", "Wuling", "Xcite", "Xiaomi", "Xpeng", "Zeekr", "Zotye",
    "ZX", "Амберавто", "Амбертрак", "Аурус", "Богдан", "Волга", "ГАЗ", "Донинвест", "ЗАЗ", "ЗИЛ", "ЗиС", "ИЖ", "Лада",
    "ЛуАЗ", "Москвич", "Соллерс", "ТагАЗ", "УАЗ", "Эволют"
]

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"

def save_to_excel(car_data, filename="cars_pred.xlsx"):
    """Сохраняет данные в Excel файл"""
    if os.path.exists(filename):
        df = pd.read_excel(filename)
    else:
        df = pd.DataFrame(columns=['number', '_id', 'brand', 'description', 'images', 'median', 'model', 'rating'])
    
    # Преобразуем данные в нужный формат
    excel_data = {
        'number': len(df),
        '_id': f"{car_data['brand']}_{car_data['model']}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
        'brand': car_data['brand'],
        'description': car_data['description'],
        'images': ','.join(car_data['images']),
        'median': car_data.get('median', 0),
        'model': car_data['model'],
        'rating': car_data.get('rating', 0)
    }
    
    new_df = pd.DataFrame([excel_data])
    df = pd.concat([df, new_df], ignore_index=True)
    df.to_excel(filename, index=False)
    logger.info(f"Автомобиль {car_data['brand']} {car_data['model']} успешно сохранен в Excel.")

def get_page_content(url):
    """Получает HTML-дерево страницы по URL с обработкой ошибок."""
    try:
        headers = {"User-Agent": USER_AGENT}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        tree = html.fromstring(response.content)
        return tree
    except requests.exceptions.RequestException as e:
        logger.error(f"Ошибка при запросе страницы {url}: {e}", exc_info=True)
        return None

def extract_car_links(tree):
    """Извлекает информацию об автомобилях из HTML-дерева."""
    try:
        car_links = tree.xpath(
            "//a[@class='b-link' and @href and normalize-space(text()) and count(@*) = 2]"
        )
        info_blocks = tree.xpath("//div[@class='b-info-block__image' and @style='min-width: 120px;']")

        cars = []
        for index, car_link in enumerate(car_links):
            text = car_link.text_content().strip()

            brand = next((b for b in CAR_BRANDS if text.startswith(b)), None)
            model = text[len(brand):].strip() if brand else text

            try:
                a_element = info_blocks[index].xpath(".//a[@href]")[0]
                href = a_element.get("href").strip()
            except IndexError:
                logger.warning(f"Не удалось извлечь ссылку для автомобиля {text} на позиции {index}")
                continue

            car = {"brand": brand, "model": model, "link": href}
            if car not in cars:
                cars.append(car)

        logger.debug(f"Найдено {len(cars)} ссылок на автомобили.")
        return cars

    except Exception as e:
        logger.error(f"Ошибка при извлечении ссылок на автомобили: {e}", exc_info=True)
        return []

def get_car_details(car):
    """Получает детали автомобиля с его страницы."""
    car_page_url = f"https://www.drom.ru{car['link']}"
    logger.info(f"Получаем детали автомобиля {car['brand']} {car['model']} со страницы {car_page_url}")
    car_tree = get_page_content(car_page_url)

    if car_tree is None:
        logger.warning(f"Не удалось получить HTML для {car_page_url}, пропуск.")
        return None

    try:
        images = car_tree.xpath(
            "//div[contains(@class, 'b-flex') and contains(@class, 'b-flex_align_left') and contains(@class, 'b-random-group') and contains(@class, 'b-random-group_margin_r-size-s') and contains(@class, "
            "'b-media-cont') and contains(@class, 'b-media-cont_margin_huge')]//a/@href"
        )
        description = car_tree.xpath("//div[@data-dropdown-container='description-text-expand']//text()")
        description_text = " ".join(description).strip().replace("Читать полностью", "")

        trim_levels = car_tree.xpath("//a[@data-name and @href]")
        trim_info = []
        for trim in trim_levels:
            trim_name = trim.text_content().strip()
            trim_link = trim.get("href").strip()
            full_trim_link = f"https://www.drom.ru{trim_link}"
            trim_info.append({"trim_name": trim_name, "trim_link": full_trim_link})

        rating_elements = car_tree.xpath(
            "//div[@class='b-sticker b-sticker_theme_rating b-sticker_type_high']/text()[normalize-space()]"
        )
        rating = float(rating_elements[0].strip()) if rating_elements else None

        sales_url_elements = car_tree.xpath(
            "//a[@class='g6gv8w4 g6gv8w8' and @data-ga-stats-name='sidebar_model_sales' and @data-ga-stats-track-click='true' and @data-ftid='component_brand-model_related-link']/@href"
        )
        sales_url = sales_url_elements[0] if sales_url_elements else None

        car_details = {
            "description": description_text,
            "images": images,
            "trim_info": trim_info,
            "rating": rating,
            "salesUrl": sales_url,
        }

        logger.debug(f"Детали автомобиля {car['brand']} {car['model']} успешно извлечены.")
        return car_details

    except Exception as e:
        logger.error(f"Ошибка при извлечении деталей автомобиля {car['brand']} {car['model']}: {e}", exc_info=True)
        return None

def get_trim_details(trim_info):
    """Получает детали комплектаций автомобиля."""
    all_keys_values = defaultdict(list)

    sample_size = max(1, int(len(trim_info) * 0.1)) if len(trim_info) > 4 else len(trim_info)
    sampled_trim_info = random.sample(trim_info, sample_size)

    logger.info(f"Получаем детали {sample_size} комплектаций.")
    for trim in sampled_trim_info:
        try:
            trim_response = requests.get(trim['trim_link'])
            trim_response.raise_for_status()
            trim_tree = html.fromstring(trim_response.content)
        except requests.exceptions.RequestException as e:
            logger.error(f"Ошибка при запросе страницы комплектации {trim['trim_link']}: {e}", exc_info=True)
            continue

        keys = trim_tree.xpath(
            "//tr[contains(@class, 'b-table__row') and contains(@class, 'b-table__row_padding_l-r-size-xs') and contains(@class, 'b-table__row_cols_2') and contains(@class, 'b-table__row_border_bottom') and contains(@class, 'b-table__row_border_light') and contains(@class, 'b-table__row_padding_t-b-size-s') and contains(@class, 'b-table_align_top')]/td[1]/text()")
        value_cells = trim_tree.xpath(
            "//tr[contains(@class, 'b-table__row') and contains(@class, 'b-table__row_padding_l-r-size-xs') and contains(@class, 'b-table__row_cols_2') and contains(@class, 'b-table__row_border_bottom') and contains(@class, 'b-table__row_border_light') and contains(@class, 'b-table__row_padding_t-b-size-s') and contains(@class, 'b-table_align_top')]/td[2]")

        for key, value_cell in zip(keys, value_cells):
            key = key.strip()
            value = "".join(value_cell.xpath(".//text()")).strip()

            if not value:
                value = "Присутствует" if value_cell.xpath(".//svg") else "Отсутствует"
            elif value == "—":
                value = "Отсутствует"

            if key and value:
                all_keys_values[key].append(value)

    trim_modes = {}
    for key, values in all_keys_values.items():
        try:
            most_common_value = Counter(values).most_common(1)[0][0]
            trim_modes[key] = most_common_value
        except IndexError:
            logger.warning(f"Нет данных для ключа {key} при обработке комплектаций.")
            trim_modes[key] = None

    return trim_modes

def get_prices_from_offer(url):
    """Извлекает информацию о ценах автомобиля из JSON-LD разметки."""
    if not url:
        logger.warning("URL для получения цен отсутствует.")
        return None

    try:
        response = requests.get(url)
        response.raise_for_status()
        tree = html.fromstring(response.content)
        json_script = tree.xpath(
            "//script[@type='application/ld+json' and contains(text(), '\"@type\":\"AggregateOffer\"')]/text()"
        )

        if not json_script:
            logger.warning("JSON-LD разметка не найдена на странице.")
            return None

        json_data = json.loads(json_script[0])
        prices = [offer["price"] for offer in json_data.get("offers", {}).get("offers", []) if offer.get("price")]

        if not prices:
            logger.warning("Информация о ценах не найдена в JSON-LD разметке.")
            return None

        low = min(prices)
        high = max(prices)
        average = sum(prices) / len(prices)
        median = np.median(prices)

        price_data = {"low": low, "high": high, "average": round(average, 2), "median": round(median, 2)}
        logger.info(f"Информация о ценах успешно извлечена: {price_data}")
        return price_data

    except requests.exceptions.RequestException as e:
        logger.error(f"Ошибка при запросе страницы {url} для получения цен: {e}", exc_info=True)
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Ошибка при разборе JSON-LD на странице {url}: {e}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Непредвиденная ошибка при получении информации о ценах с URL {url}: {e}", exc_info=True)
        return None

def process_car(car):
    try:
        logger.info(f"Начинаем обработку автомобиля: {car['brand']} {car['model']}")
        car_details = get_car_details(car)

        if not car_details or not car_details['description'].strip() or len(car_details['description'].strip()) < 10:
            logger.warning(f"Пропуск автомобиля {car['brand']} {car['model']} из-за отсутствия или короткого описания.")
            return

        sales_url = car_details.get('salesUrl')
        prices = get_prices_from_offer(sales_url) if sales_url else None
        if prices is None:
            logger.warning(
                f"Не удалось получить информацию о ценах для {car['brand']} {car['model']} (sales_url: {sales_url}).")

        trim_mode_data = get_trim_details(car_details['trim_info'])

        car_document = {
            "brand": car['brand'].strip(),
            "model": car['model'].strip(),
            "description": car_details['description'],
            "rating": car_details['rating'],
            "images": car_details['images'],
            "trim_mode_data": trim_mode_data,
            "sales_url": sales_url,
        }

        if prices:
            car_document.update(prices)

        save_to_excel(car_document)

        logger.info(f"Автомобиль {car['brand']} {car['model']} успешно обработан и сохранен.")

    except Exception as e:
        logger.error(f"Ошибка при обработке автомобиля {car['brand']} {car['model']}: {e}", exc_info=True)


def main(url='https://www.drom.ru/catalog/~search/?y_start=2010&wheel=0&page=1'):
    """Основная функция для выполнения всех шагов."""
    logger.info(f"Начинаем обработку страницы: {url}")
    tree = get_page_content(url)

    if not tree:
        logger.error(f"Не удалось получить HTML-дерево для URL: {url}")
        return

    cars = extract_car_links(tree)
    logger.info(f"Найдено {len(cars)} автомобилей на странице {url}.")

    for car in tqdm(cars, desc="Обработка машин"):
        process_car(car)

if __name__ == "__main__":
    try:
        num_pages = int(os.environ.get("NUM_PAGES", 1))
        num_pages = 195
        for i in range(1, num_pages + 1):
            url = f'https://www.drom.ru/catalog/~search/?&wheel=0&page={i}'
            main(url)
        logger.info("Обработка завершена.")
    except Exception as e:
        logger.critical(f"Непредвиденная ошибка во время выполнения: {e}", exc_info=True)
