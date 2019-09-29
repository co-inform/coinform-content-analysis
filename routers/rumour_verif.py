from typing import List
from fastapi import APIRouter, HTTPException
from pydantic import UrlStr
from pydantic import BaseModel

class Tweet(BaseModel):
    tweetId: str

class StanceResponse(BaseModel):
    favor: float
    against: float
    na: float

router = APIRouter()

# Estimates stance of the tweet
# It returns stance estimation for given post
@router.post('/post/stance')
def estimate_stance(tweet:Tweet):
    print(tweet.tweetId)
    return StanceResponse(favor = 0, against = 0, na = 0)


#@router.post('/post/veracity')