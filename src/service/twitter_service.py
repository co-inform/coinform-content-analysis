#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
import requests
import json
import settings as settings
from typing import Dict

logger = logging.getLogger('server')


class TwitterService:
    def __init__(self):
        self.conn = None

    def get_conversation(self, tweetId: str) -> Dict:
        twitter_conn = settings.get_twitter_connector()
        url_conversation = "http://" + twitter_conn['host'] + ":" + twitter_conn['port'] + "/tweets/conversation/" + tweetId
        response = requests.get(url_conversation)
        if response.status_code is not 200:
            try:
                return {'source': self.get_tweet(tweetId),
                        'replies': []}
            except:
              raise requests.HTTPError
        else:
            return json.loads(response.content)

    def get_tweet(self,tweetId:str) -> Dict:
        twitter_conn = settings.get_twitter_connector()
        url_tweet = "http://" + twitter_conn['host'] + ":" + twitter_conn['port'] + "/tweets/" + tweetId
        response = requests.get(url_tweet)
        if response.status_code is not 200:
            raise requests.HTTPError
        else:
            return json.loads(response.content)