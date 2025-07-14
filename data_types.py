from pydantic import BaseModel
from typing import List, Literal 

class Function(BaseModel):
    repo_name: str  # NOT NULL
    func_path_in_repository: str  # NOT NULL
    func_name: str  # NOT NULL
    whole_func_string: str  # NOT NULL
    language: str  # NOT NULL
    func_documentation_string: str  # NOT NULL
    code_tokens: str  # NOT NULL

class Scenario(BaseModel):
    id: int
    question: str
    answer: str
    repo: str
    functions: List[str]
    how_realistic: float
    split: Literal["train", "test"]
