import pandas as pd
import json
from create_dataset.utils import (
    prepare_oil, 
    prepare_transmission, 
    prepare_year, 
    summarization_description, 
    prepare_description
)
import re
from getpass import getpass
import neuralNetworkCarsSystem.AutoAssistant as aa
import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import Optional, Literal
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

load_dotenv()

class CarCharacteristics(BaseModel):
    """Pydantic model for car characteristics"""
    model_config = ConfigDict(validate_by_name=True, extra='ignore')

    Количество_мест: Optional[int] = Field(None, ge=2, le=9)
    Привод: Optional[str] = None
    Страна: Optional[str] = None
    Количество_дверей: Optional[int] = Field(None, ge=2, le=7)
    Тип_кузова: Optional[str] = None
    Тип_двигателя: Optional[str] = None
    Расход_топлива: Optional[float] = Field(None, ge=3.0, le=30.0)
    Клиренс: Optional[int] = Field(None, ge=100, le=400)
    Лошадиные_силы: Optional[int] = Field(None, ge=50, le=2000)
    Тип_коробки: Optional[str] = None
    Начало_выпуска: Optional[int] = Field(None, ge=1990, le=2025)
    Конец_выпуска: Optional[int] = Field(None, ge=1990, le=2025)

    @field_validator('Привод')
    @classmethod
    def validate_drive(cls, v: Optional[str]) -> Optional[str]:
        if not v:
            return None
        v = v.lower().strip()
        if any(word in v for word in ['задний', 'заднеприводный']):
            return 'задний'
        elif any(word in v for word in ['передний', 'переднеприводный']):
            return 'передний'
        elif any(word in v for word in ['полный', '4wd', 'awd']):
            return 'полный'
        return None

    @field_validator('Тип_двигателя')
    @classmethod
    def validate_engine(cls, v: Optional[str]) -> Optional[str]:
        if not v:
            return None
        v = v.lower().strip()
        if 'дизель' in v:
            return 'дизель'
        elif 'электрич' in v:
            return 'электричество'
        elif 'гибрид' in v:
            return 'гибрид'
        elif 'газ' in v:
            return 'газ'
        elif 'бензин' in v:
            return 'бензин'
        return None

    @field_validator('Тип_коробки')
    @classmethod
    def validate_transmission(cls, v: Optional[str]) -> Optional[str]:
        if not v:
            return None
        v = v.lower().strip()
        if 'механич' in v or 'мкпп' in v:
            return 'механическая'
        elif 'автомат' in v or 'акпп' in v:
            return 'автоматическая'
        elif 'робот' in v or 'ркпп' in v:
            return 'робот'
        elif 'вариатор' in v:
            return 'вариатор'
        return None

    @field_validator('Конец_выпуска')
    @classmethod
    def validate_years(cls, v: Optional[int], info) -> Optional[int]:
        if v is None:
            return None
        start_year = info.data.get('Начало_выпуска')
        if start_year and v < start_year:
            return None
        return v

def get_car_characteristics_from_model(description, api):
    """Get car characteristics using the model"""
    messages = [
        {"role": "user", "content": f"""
            Ты - виртуальный ассистент, специализирующийся на подборе автомобилей.
            Твоя задача - извлечь характеристики автомобиля из описания.
            Описание автомобиля: {description}
            
            Верни ТОЛЬКО JSON со следующими полями:
            {{
                "Количество_мест": число (от 2 до 9),
                "Привод": строка (передний/задний/полный),
                "Страна": строка,
                "Количество_дверей": число (от 2 до 5),
                "Тип_кузова": строка,
                "Тип_двигателя": строка (бензин/дизель/электричество/гибрид/газ),
                "Расход_топлива": число (от 3.0 до 30.0 л/100км),
                "Клиренс": число (от 100 до 400 мм),
                "Лошадиные_силы": число (от 50 до 2000),
                "Тип_коробки": строка (Механика/механическая/автоматическая/робот/вариатор),
                "Начало_выпуска": число (от 1990 до 2025),
                "Конец_выпуска": число (от 1990 до 2025)
            }}
            
            Если какое-то значение не удалось определить, верни null для этого поля.
            Убедись, что числовые значения действительно являются числами и находятся в указанных диапазонах.
            Не добавляй никаких дополнительных пояснений или текста, только JSON.
            """}
    ]

    try:
        answer = api.post_query(messages)
        response_text = answer['choices'][0]['message']['content'].strip()
        
        # Try to find JSON in the response
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        
        if json_start >= 0 and json_end > json_start:
            json_str = response_text[json_start:json_end]
            raw_characteristics = json.loads(json_str)
            
            # Convert to Pydantic model
            try:
                characteristics = CarCharacteristics(**raw_characteristics)
                return characteristics.model_dump(by_alias=True)
            except Exception as e:
                print(f"Error validating characteristics: {str(e)}")
                return None
        else:
            print(f"Could not find JSON in response: {response_text}")
            return None
            
    except Exception as e:
        print(f"Error getting characteristics from model: {str(e)}")
        print(f"Response: {answer if 'answer' in locals() else 'No response'}")
        return None

def extract_car_characteristics(description, api):
    """Extract car characteristics using regex patterns and model as fallback"""
    characteristics = {
        'Количество_мест': None,
        'Привод': None,
        'Страна': None,
        'Количество_дверей': None,
        'Тип_кузова': None,
        'Тип_двигателя': None,
        'Расход_топлива': None,
        'Клиренс': None,
        'Лошадиные_силы': None,
        'Тип_коробки': None,
        'Начало_выпуска': None,
        'Конец_выпуска': None
    }
    
    # Patterns for extracting information
    patterns = {
        'Количество_мест': r'(\d+)\s*(?:мест|местный)',
        'Привод': r'(?:привод|приводом)\s*[–-]?\s*([а-яА-Я\s]+)',
        'Страна': r'(?:сборка|производство)\s*[–-]?\s*([а-яА-Я\s]+)',
        'Количество_дверей': r'(\d+)\s*(?:дверн|двери)',
        'Тип_кузова': r'(?:кузов|тип кузова)\s*[–-]?\s*([а-яА-Я\s]+)',
        'Тип_двигателя': r'(?:двигатель|мотор)\s*[–-]?\s*([а-яА-Я\s]+)',
        'Расход_топлива': r'(\d+[.,]?\d*)\s*(?:л/100|л\/100)',
        'Клиренс': r'(\d+)\s*(?:мм|миллиметров)\s*(?:клиренс|дорожный просвет)',
        'Лошадиные_силы': r'(\d+)\s*(?:л\.с\.|лошадиных сил)',
        'Тип_коробки': r'(?:коробка|трансмиссия)\s*[–-]?\s*([а-яА-Я\s]+)',
        'Начало_выпуска': r'(?:начало|с)\s*(\d{4})',
        'Конец_выпуска': r'(?:до|по)\s*(\d{4})'
    }
    
    # Extract information using patterns
    for key, pattern in patterns.items():
        match = re.search(pattern, description, re.IGNORECASE)
        if match:
            value = match.group(1).strip()
            if key in ['Количество_мест', 'Количество_дверей', 'Клиренс', 'Лошадиные_силы', 'Начало_выпуска', 'Конец_выпуска']:
                try:
                    value = int(value)
                except ValueError:
                    continue
            elif key == 'Расход_топлива':
                try:
                    value = float(value.replace(',', '.'))
                except ValueError:
                    continue
            characteristics[key] = value
    
    # Post-process some fields
    if characteristics['Тип_двигателя']:
        characteristics['Тип_двигателя'] = prepare_oil(characteristics['Тип_двигателя'])
    if characteristics['Тип_коробки']:
        characteristics['Тип_коробки'] = prepare_transmission(characteristics['Тип_коробки'])
    if characteristics['Привод']:
        characteristics['Привод'] = "полный" if "Полный" in characteristics['Привод'] else characteristics['Привод'].lower()
    
    # Try to validate with Pydantic
    try:
        validated_characteristics = CarCharacteristics(**characteristics)
        characteristics = validated_characteristics.model_dump(by_alias=True)
    except Exception as e:
        print(f"Error validating regex-extracted characteristics: {str(e)}")
        # If validation fails, get characteristics from model
        model_characteristics = get_car_characteristics_from_model(description, api)
        if model_characteristics:
            characteristics = model_characteristics
    
    return characteristics

def process_single_car(row, api):
    """Process a single car entry"""
    try:
        # Extract characteristics
        characteristics = extract_car_characteristics(row['description'], api)
        
        # Generate summarization
        summary = summarization_description(row['description'], api)
        summary_dict = prepare_description(summary)
        
        # Update row with new data
        row_dict = row.to_dict()
        for key, value in characteristics.items():
            if value is not None:
                row_dict[key] = value
        
        row_dict['desc_summarization'] = summary_dict['Описание']
        row_dict['desc_plus'] = summary_dict['Плюсы']
        row_dict['desc_minus'] = summary_dict['Минусы']
        
        return row_dict
    except Exception as e:
        print(f"Error processing car: {str(e)}")
        return row.to_dict()

def process_cars(input_file='create_dataset/cars_deduplicated.xlsx', output_file='create_dataset/cars_processed.xlsx'):
    # Initialize API
    api = aa.OpenAIApi(os.getenv("PROXY_LOGIN"), os.getenv("PROXY_PASSWORD"))
    
    # Read deduplicated data
    print("Reading deduplicated data...")
    df = pd.read_excel(input_file)
    
    # Process cars in parallel
    print("Processing cars in parallel...")
    processed_rows = []
    
    with ProcessPoolExecutor(max_workers=5) as executor:
        # Create a partial function with the API instance
        process_func = partial(process_single_car, api=api)
        
        # Submit all tasks
        future_to_row = {executor.submit(process_func, row): idx for idx, row in df.iterrows()}
        
        # Process completed tasks
        for future in as_completed(future_to_row):
            idx = future_to_row[future]
            try:
                result = future.result()
                processed_rows.append(result)
                print(f"Completed processing car {idx + 1}/{len(df)}")
            except Exception as e:
                print(f"Error processing car {idx + 1}: {str(e)}")
                processed_rows.append(df.iloc[idx].to_dict())
    
    # Create new DataFrame from processed rows
    processed_df = pd.DataFrame(processed_rows)
    
    # Ensure all required columns exist
    required_columns = ['number', '_id', 'brand', 'description', 'images', 'median', 'model',
                       'rating', 'Количество_мест', 'Привод', 'Страна', 'Количество_дверей',
                       'Тип_кузова', 'Тип_двигателя', 'Расход_топлива', 'Клиренс',
                       'Лошадиные_силы', 'Тип_коробки', 'Начало_выпуска', 'Конец_выпуска',
                       'desc_summarization', 'desc_plus', 'desc_minus']
    
    for col in required_columns:
        if col not in processed_df.columns:
            processed_df[col] = None
    
    # Reorder columns
    processed_df = processed_df[required_columns]
    
    # Save processed data
    print("Saving processed data...")
    processed_df.to_excel(output_file, index=False)
    print("Processing complete!")

if __name__ == "__main__":
    process_cars() 