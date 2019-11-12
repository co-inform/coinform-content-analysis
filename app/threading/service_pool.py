import logging
import requests
import threading

log = logging.getLogger('server')

def compute_result(connector, tweet_id, model, callback_url):
    log.info('retrieving tweet in seperate thread')
    conversation = connector.get_conversation(tweet_id)
    log.info('conversation loaded')
    results = model.estimate_veracity(conversation)
    log.info('veracity estimated')
    cr = requests.post(url=callback_url, data=results, timeout=15)
    log.info('callback executed:')
    log.info(cr.json())

class ServicePool:
    def __init__(self):
        log.info('server pool created')

    def add(self, connector, tweet_id, model, callback_url):
        log.info('using pool process to execute twitter call async')
        t = threading.Thread(target=compute_result, args=(connector, tweet_id, model, callback_url))
        t.start()
        log.info('thread started')
        return True
