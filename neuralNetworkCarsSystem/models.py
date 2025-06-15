from enum import Enum
from typing import List, Optional, Dict
from pydantic import BaseModel, Field
from langchain_core.documents import Document

class ActionType(str, Enum):
    ASK_QUESTION = "ask_question"
    SHOW_CARS = "show_cars"
    CLARIFY = "clarify"

class QuestionType(str, Enum):
    BUDGET = "budget"
    BRAND = "brand"
    BODY_TYPE = "body_type"
    TRANSMISSION = "transmission"
    DRIVE_TYPE = "drive_type"
    FUEL_TYPE = "fuel_type"
    USAGE = "usage"
    PRIORITY = "priority"

class FuelType(str, Enum):
    PETROL = "бензин"
    DIESEL = "дизель"
    HYBRID = "гибрид"
    ELECTRIC = "электро"
    GAS = "газ"

class TransmissionType(str, Enum):
    AUTOMATIC = "автоматическая"
    MANUAL = "механическая"
    CVT = "вариатор"
    ROBOT = "робот"

class DriveType(str, Enum):
    FRONT = "передний"
    REAR = "задний"
    ALL_WHEEL = "полный"

class BodyType(str, Enum):
    SEDAN = "седан"
    HATCHBACK = "хэтчбек"
    LIFTBACK = "лифтбек"
    WAGON = "универсал"
    SUV = "suv"
    CROSSOVER = "кроссовер"
    MINIVAN = "минивэн"
    PICKUP = "пикап"
    COUPE = "купе"
    CONVERTIBLE = "кабриолет"
    ROADSTER = "родстер"

class UsageType(str, Enum):
    NEW = "новая"
    BUDGET = "бюджетная"
    FAMILY = "семейная"
    LONG_TRIP = "для дальних поездок"
    ECONOMIC = "экономичная"
    OFFROAD = "для бездорожья"
    POWERFUL = "мощная"
    WORK = "для работы"
    WEEKEND = "для выходных"

class Question(BaseModel):
    type: QuestionType
    text: str
    options: Optional[List[str]] = None

class CarFilter(BaseModel):
    min_price: Optional[int] = None
    max_price: Optional[int] = None
    brands: List[str] = Field(default_factory=list)
    body_types: List[BodyType] = Field(default_factory=list)
    transmissions: List[TransmissionType] = Field(default_factory=list)
    drive_types: List[DriveType] = Field(default_factory=list)
    fuel_types: List[FuelType] = Field(default_factory=list)
    min_year: Optional[int] = None
    max_year: Optional[int] = None
    min_horsepower: Optional[int] = None
    max_horsepower: Optional[int] = None
    min_seats: Optional[int] = None
    max_seats: Optional[int] = None
    min_clearance: Optional[int] = None
    max_clearance: Optional[int] = None

class ModelResponse(BaseModel):
    action: ActionType
    message: str
    question: Optional[Question] = None
    filter: Optional[CarFilter] = None
    confidence: float = Field(ge=0.0, le=1.0, description="Уверенность модели в том, что у нее достаточно информации для поиска")
    docs: List[Document] = [] 