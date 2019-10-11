#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
import requests
import json
from typing import Dict

logger = logging.getLogger('server')

CONVERSATION_URL = 'http://localhost:8001/tweets/conversation/'

class TwitterService:
  def __init__(self):
      self.conn = None

  def get_conversation(self,tweetId:str)->Dict:
      url = CONVERSATION_URL + tweetId
      response = requests.get(url)
      if not response.status_code is 200:
          raise requests.HTTPError
      else:
        return json.loads(response.content)