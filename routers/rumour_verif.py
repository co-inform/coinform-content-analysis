import models.tweet as tweet
import settings
import service.api_pool
import service.twitter_service as twitter_service
import logging

from fastapi import APIRouter, HTTPException
from settings import get_twitter_credentials
from estimators import clearumor
from pathlib import Path

from responses import veracity_response

logger = logging.getLogger('server')

router = APIRouter()
api_pool = service.api_pool.ApiPool(get_twitter_credentials())
clearumor_model = clearumor.ClearRumorModel(clearumor.utils.load_model_from_file(settings.get_stance_path()),
                                            clearumor.utils.load_model_from_file(settings.get_verif_path()))


# Estimates stance of the tweet
# It returns stance estimation for given post

@router.get('/post/veracity')
def estimate_veracity(tweet: tweet.Tweet):
    # outdir = 'data/' + tweet.tweetId
    # TODO uncomment outdir related path to disable caching
    # if os.path.isdir(outdir):
    #    raise HTTPException(status_code=503, detail="Tweet is currently being loaded")

    # first check whether tweet is

    api = api_pool.get()
    try:
        if not api is None:
            # conversation = twitter_service.load_conversation(api, tweet.tweetId, outdir)
            root_dir = Path.cwd()
            logger.info(root_dir)
            outdir = root_dir / 'data/'
            logger.info(outdir)
            conversation = twitter_service.load_conversation_from_folder(tweet.tweetId, outdir)
            results = clearumor_model.estimate_veracity(tweet.tweetId, conversation)
            logger.info(results)
            return veracity_response.VeracityResponse(
                stance_support=results['stance_support'],
                stance_deny=results['stance_deny'],
                stance_query=results['stance_query'],
                veracity_prediction=results['veracity_prediction'],
                veracity_label=results['veracity_label'])
        else:
            raise HTTPException(status_code=503, detail="Pool is full")
    finally:
        if not api is None:
            api_pool.release(api)
        # if os.path.isdir(outdir):
        #    shutil.rmtree(outdir)
