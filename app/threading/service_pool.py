import logging
import requests
import threading
from fastapi import HTTPException

log = logging.getLogger('server')

queue = []

def compute_result(connector, tweet_id, model, callback_url):
    log.info('retrieving tweet in seperate thread')
    try:
        conversation = connector.get_conversation(tweet_id)
        if conversation is not None:
            results = model.estimate_veracity(conversation)
            cr = requests.post(url=callback_url, data=results, timeout=15)
            log.info('callback executed:')
            log.info(cr.json())
        else:
            raise HTTPException(status_code=503, detail='Cannot get conversations from twitter connector')
        if results is not None:
            return results
    except:
        raise HTTPException(status_code=503, detail='Cannot get conversations from twitter connector')
    finally:
        queue.remove(tweet_id)

class ServicePool:
    def __init__(self,num_threads):
        log.info('server pool created')
        self.num_threads = num_threads


    def add(self, connector, tweet_id, model, callback_url):
        log.info('using pool process to execute twitter call async')
        t = threading.Thread(target=compute_result, args=(connector, tweet_id, model, callback_url))
        if tweet_id in self.threads:
            raise HTTPException(status_code=400, detail='Tweet id is in already process,try later again.')
        if queue.count() <= self.num_threads:
            queue.append(tweet_id)
            t.start()
            log.info('thread started')
        else:
            raise HTTPException(status_code=401, detail='Thread pool is full.')
        return True
