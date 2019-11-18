import logging
import requests
import concurrent.futures.thread
from fastapi import HTTPException
import queue
import threading

log = logging.getLogger('server')

# structure for keeping what tweets the threads are actually working on. dont know if it needs locking or not in concurrency
tweet_set = set({})
set_lock = threading.Lock()
# the different queues for working on tweets, conversations etc
received_tweet_queue = queue.Queue(maxsize=0)
content_analysis_queue = queue.Queue(maxsize=0)
callback_queue = queue.Queue(maxsize=0)


def tweet_queue_consumer():
    while True:
        d = received_tweet_queue.get(block=True)
        log.info(str(d['callback_url']))

        conversation = d['connector'].get_conversation(d['tweet_id'])
        if conversation is None:
            with set_lock:
                tweet_set.discard(d['tweet_id'])
            raise IOError('Unable to get Twitter conversation')

        content_analysis_queue.put({"tweet_id": d['tweet_id'],
                                    "conversation": conversation,
                                    "model": d['model'],
                                    "callback_url": d['callback_url']})


def content_queue_consumer():
    while True:
        d = content_analysis_queue.get(block=True)

        results = d['model'].estimate_veracity(conversation=d['conversation'])
        if results is None:
            with set_lock:
                tweet_set.discard(d['tweet_id'])
            raise Exception('Unable to compute results from conversation')

        callback_queue.put({"tweet_id": d['tweet_id'],
                            "results": results,
                            "callback_url": d['callback_url']})


def callback_queue_consumer():
    while True:
        d = callback_queue.get(block=True)

        try:
            result = requests.post(url=d['callback_url'],
                                   json=d['results'])
            log.info(result.json())
        except requests.exceptions.RequestException as exc:
            log.error('Request error: {}'.format(exc))
        finally:
            with set_lock:
                tweet_set.discard(d['tweet_id'])


class ThreadPool:
    def __init__(self, num_threads=8):
        log.info('Thread pool created')

        with concurrent.futures.thread.ThreadPoolExecutor(max_workers=3 * num_threads) as executor:
            for n in range(start=1, stop=num_threads):
                executor.submit(tweet_queue_consumer)
                executor.submit(content_queue_consumer)
                executor.submit(callback_queue_consumer)

    def add(self, connector, tweet_id, model, callback_url):
        log.info('using pool process to execute twitter call async')

        with set_lock:
            if tweet_id in tweet_set:
                raise HTTPException(status_code=400,
                                    detail='Tweet id is already in process')

            tweet_set.add(tweet_id)
        received_tweet_queue.put({"tweet_id": tweet_id,
                                  "connector": connector,
                                  "model": model,
                                  "callback_url": callback_url})
        return True
