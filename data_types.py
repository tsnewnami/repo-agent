from pydantic import BaseModel

class Issue(BaseModel):
    issue_id: str
    
    
class SyntheticQuery(BaseModel):
    id: int