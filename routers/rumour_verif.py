import os
import shutil

from typing import List
from fastapi import APIRouter, HTTPException
from pydantic import UrlStr
from pydantic import BaseModel

import service.twitter_service as twitter_service
from settings import get_twitter_credentials

class Tweet(BaseModel):
    tweetId: str

class StanceResponse(BaseModel):
    favor: float
    against: float
    na: float

class StanceModel:
    model = None

    def __init__(self, model):
        self.model = model

    def estimate_tweet_stance(self, tweet_id, conversation):
        model()
        p
        return p[tweet_id]

class ApiPool:
    def __init__(self, api_credentials):
        # TODO implement command pattern to avoid Twitter API's max retries limit exceeded'
        self.api_credentials = api_credentials
        self.pool = []
    
    def create(self):
        return twitter_service.create_api(self.api_credentials.pop()) if len(self.api_credentials) else None

    def get(self):
        # get first free API instance or create new one
        return self.pool.pop() if len(self.pool) else self.create()
    
    def release(self, api):
        self.pool.append(api)

router = APIRouter()
api_pool = ApiPool(get_twitter_credentials('config.ini'))

# Estimates stance of the tweet
# It returns stance estimation for given post
@router.post('/post/stance')
def estimate_stance(tweet: Tweet):
    outdir = 'data/' + tweet.tweetId
    # TODO uncomment outdir related path to disable caching
    #if os.path.isdir(outdir):
    #    raise HTTPException(status_code=503, detail="Tweet is currently being loaded")

    api = api_pool.get()
    try:
        if not api is None:
            conversation = twitter_service.load_conversation(api, tweet.tweetId, outdir)
            ps = model.estimate(tweet.tweetId, conversation)
            p = ps[tweet.tweetId]
            return StanceResponse(favor = p['favor'], against = p['against'], na = p['na'])
        else:
            raise HTTPException(status_code=503, detail="Pool is full")
    finally:
        if not api is None:
            api_pool.release(api)
        #if os.path.isdir(outdir):
        #    shutil.rmtree(outdir)

#@router.post('/post/veracity')