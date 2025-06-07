from pydantic import BaseModel
from typing import Dict, Any

class EmbeddingResponseModel(BaseModel):
    embedding: list[float]
