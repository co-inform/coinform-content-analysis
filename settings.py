# settings.py
import os
import configparser

# OR, explicitly providing path to '.env'
from pathlib import Path  # python3 only

env_path = Path('.') / 'config.ini'
config = configparser.ConfigParser()
config.read(env_path)

def convert_str(tup):
    str =  ''.join(tup)
    return str

# oauth_keys = [
#     ["consumer_key_1", "consumer_secret_1", "access_token_1", "access_token_secret_1"],
#     ["consumer_key_2", "consumer_secret_2", "access_token_2", "access_token_secret_2"]
#     ]
#
# auths = []
# for consumer_key, consumer_secret, access_key, access_secret in oauth_keys:
#     auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
#     auth.set_access_token(access_key, access_secret)
#     auths.append(auth)

def get_twitter_credentials():
    sections = [s for s in config.sections() if s.startswith('Twitter API ')]
    return [{
        k.upper(): config[s][k] for k in config[s]
    } for s in sections]

def get_stance_path():
    return config.get('Stance Models', 'model')

def get_verif_path():
    return config.get('Verification Models', 'model')

def get_elmo_files():
    root = Path('.')
    weights = root/config.get('ELMo Embedding Settings','weights')
    options = root/config.get('ELMo Embedding Settings','options')
    return [weights, options]


def get_num_replies():
    return config.getint('Other Settings','NUM_REPLIES')
