from pydantic import BaseModel
from typing import List


class ClassifierRequest(BaseModel):
    query: List[str]


class ClassifierResponse(BaseModel):
    org_content: List[str]
    topics: List[str]
