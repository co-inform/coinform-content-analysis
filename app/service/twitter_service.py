#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
import requests
import json
import app.settings as settings
from typing import Dict

logger = logging.getLogger('server')


class TwitterService:
    def __init__(self):
        self.conn = None

    def get_conversation(self, tweetId: str) -> Dict:
        twitter_conn = settings.get_twitter_connector()
        url = "http://" + twitter_conn['host'] + ":" + twitter_conn['port'] + "/tweets/conversation/" + tweetId
        response = requests.get(url)
        if not response.status_code is 200:
            raise requests.HTTPError
        else:
            return json.loads(response.content)
