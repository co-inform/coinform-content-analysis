from pydantic import BaseModel

class StanceResponse(BaseModel):
    favor: float
    against: float
    na: float
