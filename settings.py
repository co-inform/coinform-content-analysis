# settings.py
import os
import configparser
from dotenv import load_dotenv
from dotenv import parser
# load_dotenv()

# OR, the same with increased verbosity:
# load_dotenv(verbose=True)


# OR, explicitly providing path to '.env'
from pathlib import Path  # python3 only
env_path = Path('.') / 'config.ini'
config = configparser.ConfigParser()
config.read(env_path)
# load_dotenv(dotenv_path=env_path)

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


def get_twitter_apis():
    auths = []
    auth = []
    num_api = 0
    for key, value in config.items('Twitter API Settings'):
        print(key)
        print("twitter_{}_".format(num_api))
        if "twitter_{}_".format(num_api) in key:
            print("yes")
            auth.append(value)
        else:
            auths.append(auth)
            num_api = num_api + 1
    return auths

apis = get_twitter_apis()
for api in apis:
    print(api)

def get_consumer_key():
    return os.environ['CONSUMERKEY']

def get_consumer_secret():
    return os.environ['CONSUMERSECRET']


def get_access_token():
    return os.environ['ACCESSTOKEN']


def get_acess_tokensec():
    return os.environ['ACCESSTOKENSEC']

def get_num_replies():
    return config.getint('Other Settings','NUM_REPLIES')

def preserve_case():
    return os.environ['preserve_case']

def preserve_handles():
    return os.environ['preserve_handles']

def preserve_hashes():
    return os.environ['preserve_hashes']

def preserve_len():
    return os.environ['preserve_len']

def preserve_url():
    return os.environ['preserve_url']