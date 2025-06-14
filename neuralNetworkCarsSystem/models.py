from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field

class ActionType(str, Enum):
    ASK_QUESTION = "ask_question"
    SHOW_CARS = "show_cars"
    CLARIFY = "clarify"

class QuestionType(str, Enum):
    BUDGET = "budget"
    PREFERENCES = "preferences"
    USAGE = "usage"
    PRIORITIES = "priorities"

class Question(BaseModel):
    type: QuestionType
    text: str
    options: Optional[List[str]] = None

class CarFilter(BaseModel):
    year_from: Optional[int] = None
    year_to: Optional[int] = None
    price_from: Optional[float] = None
    price_to: Optional[float] = None
    brands: Optional[List[str]] = None
    countries: Optional[List[str]] = None
    drives: Optional[List[str]] = None
    engine_types: Optional[List[str]] = None
    fuel_consumption_from: Optional[float] = None
    fuel_consumption_to: Optional[float] = None
    seats_from: Optional[int] = None
    seats_to: Optional[int] = None
    body_types: Optional[List[str]] = None
    doors_from: Optional[int] = None
    doors_to: Optional[int] = None
    transmissions: Optional[List[str]] = None
    horsepower_from: Optional[int] = None
    horsepower_to: Optional[int] = None
    clearance_from: Optional[int] = None
    clearance_to: Optional[int] = None

class ConversationResponse(BaseModel):
    action: ActionType
    message: str
    question: Optional[Question] = None
    filter: Optional[CarFilter] = None
    confidence: float = Field(ge=0.0, le=1.0, description="Уверенность модели в том, что у нее достаточно информации для поиска") 