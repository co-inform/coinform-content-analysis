from app import settings
import logging

from fastapi import APIRouter, HTTPException, Path
from app.estimators import baseline
from app.service import twitter_service
from app.threading import service_pool

from app.models import tweet

import json

log = logging.getLogger('server')
log.info('Starting rumour verification router.')

router = APIRouter()

log.info(settings.get_active_model()+' model has been selected.')

# if 'clearumor'== settings.get_active_model():
#     model = clearumor.ClearRumorModel(utils.load_model_from_file(settings.get_stance_path()),
#                                             utils.load_model_from_file(settings.get_verif_path()))
# else:
#

model = baseline.BaselineModel(settings.get_stance_path(), settings.get_verif_path())

connector = twitter_service.TwitterService()

pool = service_pool.ServicePool()

# Estimates stance of the tweet
# It returns stance estimation for given post

@router.get('/post/veracity/{tweet_id}', response_model=tweet.Conversation)
def estimate_veracity(tweet_id: str = Path(..., title="The ID of the tweet to get")):
    log.info('Get conversation of {}'.format(tweet_id))
    try:
        conversation = connector.get_conversation(tweet_id)
    except:
        raise HTTPException(status_code=503, detail='Cannot get conversations from twitter connector')
    if conversation is not None:
        results = model.estimate_veracity(conversation)
    else:
        raise HTTPException(status_code=503, detail='Cannot get conversations from twitter connector')
    if results is not None:
        return results

@router.post('/post/veracity_test/{tweet_id}')
def estimate_veracity_post_test(callback_url: tweet.Callback, tweet_id: str = Path(..., title="The ID of the tweet to get")):
    log.info('Get conversation of {}'.format(tweet_id))
    if pool.add(connector, tweet_id, model, callback_url):
        return None
    else:
        raise HTTPException(status_code=500, detail='Cannot estimate veracity')

@router.post('/post/veracity/{tweet_id}', response_model=tweet.Conversation)
def estimate_veracity_post_veracity_tweet_id_get(callback_url: tweet.Callback,tweet_id: str = Path(..., title="The ID of the tweet to get")):
    log.info('Get conversation of {}'.format(tweet_id))
    try:
        conversation = connector.get_conversation(tweet_id)
    except:
        raise HTTPException(status_code=503, detail='Cannot get conversations from twitter connector')
    if conversation is not None:
        results = model.estimate_veracity(conversation)
    else:
        raise HTTPException(status_code=503, detail='Cannot get conversations from twitter connector')
    if results is not None:
        return results
    else:
        raise HTTPException(status_code=503, detail='Cannot estimate veracity')



