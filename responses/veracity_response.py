from pydantic import BaseModel


class VeracityResponse(BaseModel):
    stance_support: float
    stance_deny: float
    stance_query: float
    veracity_prediction: float
    veracity_label: str
