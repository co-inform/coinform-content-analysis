import models.tweet as tweet
import settings
import responses.stance_response as stance_response
import service.api_pool
import service.twitter_service as twitter_service

from fastapi import APIRouter, HTTPException
from settings import get_twitter_credentials
from estimators import clearumor
from pathlib import Path





router = APIRouter()
api_pool = service.api_pool.ApiPool(get_twitter_credentials())
stance_model = clearumor.ClearRumorModel(clearumor.utils.load_model_from_file(settings.get_stance_path()))
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
            # conversation = twitter_service.load_conversation(api, tweet.tweetId, outdir)
            root_dir = Path.cwd().parent
            outdir = root_dir / 'data/'
            conversation = twitter_service.load_conversation_from_folder(tweet.tweetId, outdir)
            ps = stance_model.estimate_stance(tweet.tweetId, conversation)
            # p = ps[tweet.tweetId]
            return None
            # return stance_response.StanceResponse(favor = p['favor'], against = p['against'], na = p['na'])
        else:
            raise HTTPException(status_code=503, detail="Pool is full")
    finally:
        if not api is None:
            api_pool.release(api)
        #if os.path.isdir(outdir):
        #    shutil.rmtree(outdir)

#@router.post('/post/veracity')