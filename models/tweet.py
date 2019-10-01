from pydantic import BaseModel

class Tweet(BaseModel):
    tweetId: str