from app import settings
import logging

from fastapi import APIRouter, HTTPException, Path
from app.estimators import clearumor, utils
from app.service import twitter_service

from app.models import tweet

logger = logging.getLogger('server')

router = APIRouter()
clearumor_model = clearumor.ClearRumorModel(utils.load_model_from_file(settings.get_stance_path()),
                                            utils.load_model_from_file(settings.get_verif_path()))

connector = twitter_service.TwitterService()

# Estimates stance of the tweet
# It returns stance estimation for given post

@router.get('/post/veracity/{tweet_id}', response_model=tweet.Conversation)
def estimate_veracity(tweet_id: str = Path(..., title="The ID of the tweet to get")):
    logger.info('Get conversation of {}'.format(tweet_id))
    conversation = connector.get_conversation(tweet_id)
    if not conversation is None:
        results = clearumor_model.estimate_veracity(conversation)
    else:
        raise HTTPException(status_code = 503, detail = 'Cannot get conversations from twitter connector')
    if not results is None:
        return results
    else:
        raise HTTPException(status_code=503, detail='Cannot estimate veracity')



