from pydantic import BaseModel
from typing import Literal, Optional

class Function(BaseModel):
    repo_name: str  # NOT NULL
    func_path_in_repository: str  # NOT NULL
    func_name: str  # NOT NULL
    whole_func_string: str  # NOT NULL
    language: str  # NOT NULL
    func_code_string: str  # NOT NULL
    func_documentation_string: str  # NOT NULL

class SyntheticQuery(BaseModel):
    id: int