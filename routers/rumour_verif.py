import models.tweet as tweet
import estimators.stance as stance
import responses.stance_response as stance_response
import service.api_pool

from fastapi import APIRouter, HTTPException

import service.twitter_service as twitter_service
from settings import get_twitter_credentials



router = APIRouter()
api_pool = service.api_pool.ApiPool(get_twitter_credentials('config.ini'))

# Estimates stance of the tweet
# It returns stance estimation for given post
@router.post('/post/stance')
def estimate_stance(tweet: tweet.Tweet):
    outdir = 'data/' + tweet.tweetId
    # TODO uncomment outdir related path to disable caching
    #if os.path.isdir(outdir):
    #    raise HTTPException(status_code=503, detail="Tweet is currently being loaded")

    api = api_pool.get()
    try:
        if not api is None:
            conversation = twitter_service.load_conversation(api, tweet.tweetId, outdir)
            ps = stance.StanceModel.estimate(tweet.tweetId, conversation)
            p = ps[tweet.tweetId]
            return stance_response.StanceResponse(favor = p['favor'], against = p['against'], na = p['na'])
        else:
            raise HTTPException(status_code=503, detail="Pool is full")
    finally:
        if not api is None:
            api_pool.release(api)
        #if os.path.isdir(outdir):
        #    shutil.rmtree(outdir)

#@router.post('/post/veracity')