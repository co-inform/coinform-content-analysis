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
        log.info("tweet queue consumer {}".format(d['tweet_id']))
        log.info(str(d['callback_url']))
        conversation = None
        try:
            conversation = d['connector'].get_conversation(d['tweet_id'])
        except IOError as exc:
            log.error("ioerror when fetching twitter conversation {}".format(exc))
            with set_lock:
                tweet_set.discard(d['tweet_id'])
            raise IOError('Unable to get Twitter conversation')

        if conversation is None:
            log.debug('twitter conversation empty')
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
        log.info("content queue consumer {}".format(d['tweet_id']))

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
        log.info("callback queue consumer {}".format(d['tweet_id']))

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
        log.info('Thread pool created with {} threads / pool'.format(num_threads))

#        with concurrent.futures.thread.ThreadPoolExecutor(max_workers=3 * num_threads) as executor:
#            for n in range(num_threads):
#                executor.submit(tweet_queue_consumer)
#                executor.submit(content_queue_consumer)
#                executor.submit(callback_queue_consumer)

        self.fetchtweet_worker_pool = concurrent.futures.thread.ThreadPoolExecutor(max_workers=num_threads,
                                                                                   thread_name_prefix="FetchTweetPool-")
        self.content_worker_pool = concurrent.futures.thread.ThreadPoolExecutor(max_workers=num_threads,
                                                                                thread_name_prefix="ContentWorkerPool-")
        self.callback_worker_pool = concurrent.futures.thread.ThreadPoolExecutor(max_workers=num_threads,
                                                                                 thread_name_prefix="CallbackWorkerPool-")

        for n in range(num_threads):
            self.fetchtweet_worker_pool.submit(tweet_queue_consumer)
            self.content_worker_pool.submit(content_queue_consumer)
            self.callback_worker_pool.submit(callback_queue_consumer)

    def __del__(self):
        self.fetchtweet_worker_pool.shutdown()
        self.content_worker_pool.shutdown()
        self.callback_worker_pool.shutdown()

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
