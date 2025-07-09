from pydantic import BaseModel
from typing import Optional

class Issue(BaseModel):
    repo_name: str  # NOT NULL
    topic: Optional[str] = None
    issue_number: int  # NOT NULL
    title: str  # NOT NULL
    body: Optional[str] = None
    open: Optional[bool] = None
    created_at: Optional[str] = None  # ISO 8601 timestamp
    updated_at: Optional[str] = None  # ISO 8601 timestamp
    url: Optional[str] = None
    labels: Optional[str] = None  # JSON array stored as text
    user: Optional[str] = None
    comments: Optional[int] = 0  # Default 0


class SyntheticQuery(BaseModel):
    id: int