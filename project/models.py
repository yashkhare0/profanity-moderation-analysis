#models.py
from enum import Enum
from typing import List, Dict, Optional
from pydantic import BaseModel
class ClassificationObject(BaseModel):
    categories: Dict[str, bool]
    category_scores: Dict[str, float]
    category_applied_input_types: Optional[Dict[str, List[str]]] = None


class ClassificationResponse(BaseModel):
    id: str
    model: str
    flagged: Optional[bool] = None
    results: List[ClassificationObject]

    
class Providers(Enum):
    MISTRAL = 'mistral'
    OPENAI = 'openai'


MISTRAL_MODELS =[
    "mistral-large-latest",
    "mistral-small-latest",
    "open-mistral-nemo",
    "open-mistral-7b",
    "open-mixtral-8x7b",
    "open-mixtral-8x22b",
]

OPENAI_MODELS=[
    "gpt-4o",
    "gpt-4o-mini",
    "o1-preview",
    "o1-mini",
    "gpt-4-turbo",
    "gpt-3.5-turbo",
]

ANTHROPIC_MODELS=[
    "claude-3-5-sonnet-latest",
    "claude-3-5-haiku-latest",
    "claude-3-opus-latest",
]


MODERATION_MODELS=[
    "mistral-moderation-latest",
]