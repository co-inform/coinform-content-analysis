from typing import List, Optional, ForwardRef, Dict
from pydantic import BaseModel, UrlStr

ref = ForwardRef('Tweet')

class Tweet(BaseModel):
    id: str # serializing to string because otherwise the long id gets messed up
    text: str
    stance_comment:float
    stance_support:float
    stance_query:float
    stance_deny:float
    veracity_false: Optional[float]
    veracity_true: Optional[float]
    veracity_unknown: Optional[float]
    avg_stance_comment: Optional[float]
    avg_stance_support: Optional[float]
    avg_stance_query: Optional[float]
    avg_stance_deny: Optional[float]

class Conversation(BaseModel):
    source: Tweet
    replies: Dict[str,Tweet]

class Callback(BaseModel):
    callback_url: str

Tweet.update_forward_refs()
