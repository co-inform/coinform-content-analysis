#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import logging
import os
import settings
from pathlib import Path

import tweepy

logger = logging.getLogger('server')

def create_api(api_credentials):
  logger.info(api_credentials)
  my_auth = get_auth(api_credentials)
  if my_auth is None:
    logger.warn('Failed to get authentification!')
  return None if my_auth is None else tweepy.API(my_auth)

def get_auth(api_credentials):
  """
  Gets twitter authentication.
  """
  try:
    auth = tweepy.OAuthHandler(api_credentials['CONSUMER_KEY'], api_credentials['CONSUMER_SECRET'])
    auth.set_access_token(api_credentials['ACCESS_TOKEN'], api_credentials['ACCESS_TOKEN_SECRET'])
    return auth
  except KeyError:
    logger.error('Set valid twitter variables!')

def get_all_replies(tweet, api,outdir, depth=10, Verbose=False, crr_depth=1):
  """ Gets all replies to one tweet (also replies-to-replies with a recursive call).
      Saves replies to the file fout in .jsonl format.
      Parameters: tweet : Status() object
                  api : a valid twitter API
                  fout: name of the output file
                  depth: max length of the conversation to stop at
                  Verbose: print some updated during the query
  """
  if depth < 1:
    if Verbose:
      logger.info('Max depth reached')
    return
  user = tweet.user.screen_name
  tweet_id = tweet.id
  search_query = '@' + user

  # filter out retweets
  retweet_filter = '-filter:retweets'

  query = search_query + retweet_filter
  rep = []
  try:
    myCursor = tweepy.Cursor(api.search, q=query,
      since_id=tweet_id,
      max_id=None,
      wait_on_rate_limit=True,
      wait_on_rate_limit_notify=True,
      tweet_mode='extended',
      full_text=True).items()
    rep = [reply for reply in myCursor if reply.in_reply_to_status_id == tweet_id]
  except tweepy.TweepError as e:
    logger.error(('Error get_all_replies: {}\n').format(e))

  if crr_depth == 0:
      logger.info("output path")
    # logger.info("Output path: %s" % tweet.)
  if len(rep) != 0:
    if Verbose:
      if hasattr(tweet, 'id'):
        logger.info('Saving replies to: %s' % tweet.id)

      for reply in rep:
        fout = '%s/%s.json' % (outdir,reply.id)
        # save to file
        with open(fout, 'w') as (f):
          # add current depth to tweet
          jo = dict(reply._json)
          jo['depth'] = 2 if crr_depth > 1 else crr_depth
          logger.info(jo)
          # write tweet to JSON dump
          data_to_file = json.dumps(jo)
          f.write(data_to_file + '\n')

      # recursive call
      get_all_replies(reply, api, outdir,depth=depth - 1, Verbose=False, crr_depth=crr_depth + 1)


def find_source(api, tweet, max_height=10):
  """
  If a tweet is a reply, find the origin of the conversation.
  Otherwise, returns the tweet itself
  """
  if tweet.in_reply_to_status_id != None:
    try:
      original_tw = api.get_status(tweet.in_reply_to_status_id,
        tweet_mode='extended',
        full_text=True)      
      if original_tw.in_reply_to_status_id == None:
        tweet = original_tw
      else:
        # recursive call
        tweet = find_source(api, original_tw, max_height=max_height - 1)
    except tweepy.error.TweepError:
      logger.warn("Original tweet not available. Looking only for replies to current tweet.")
      pass
  return tweet



def follow_conversations(tweet_id, api, outdir):
  original_tweet = api.get_status(tweet_id)
  logger.info('Tweet id: %s' %tweet_id)
  tweet = find_source(api, original_tweet)

  # save original tweet
  fout = '%s/%s.json' % (outdir,tweet.id)
  with open(fout, 'w') as f:
    jo = dict(tweet._json)
    jo['depth'] = 0
    f.write(json.dumps(jo) + '\n')

  # fetch and save full conversation
  get_all_replies(tweet, api,outdir, depth = 5, Verbose=True)

def load_conversation(api, tweet_id, outdir):
  logger.info('Loading conversation for tweet #' + str(tweet_id))
  # retrieve conversation, store to folder
  if not os.path.exists(outdir):
    os.makedirs(outdir)
  follow_conversations(tweet_id, api, outdir)


# function for reading conversation from folder
def load_conversation_from_folder(tweet_id, outdir):
  # load conversation from folder
  conversation_dir = outdir/tweet_id
  conversation = []
  for post in conversation_dir.glob('./*.json'):
    with open(post) as f:
      conversation.append(json.load(f))
  return conversation


# def load_conversation_from_folder(tweet_id, outdir):
#   # load conversation from folder
#   file_path = os.listdir(outdir)[0]
#   with open(outdir + '/' + file_path, 'r') as f:
#     for line in f:
#       logger.debug(json.loads(line))
#   with open(outdir + '/' + file_path, 'r') as f:
#     return [{
#       'id': tweet['id'],
#       'data': tweet,
#       'type': 2 if tweet['depth'] > 1 else tweet['depth']
#     } for tweet in [json.loads(line) for line in f]]
