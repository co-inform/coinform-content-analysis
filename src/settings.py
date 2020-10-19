# settings.py
import os
import configparser

# OR, explicitly providing path to '.env'
from pathlib import Path  # python3 only
import logging

project_path = Path('.')
env_path = project_path / 'config.ini'
logging.info("config file: {}".format(env_path))
config = configparser.ConfigParser()
config.read(env_path)



def convert_str(tup):
    s = ''.join(tup)
    return s

def get_stance_path():
    model_name = get_active_model() + '-Stance Model'
    return project_path/config.get(model_name, 'model')

def get_verif_path():
    model_name = get_active_model() + '-Verification Model'
    return project_path/config.get(model_name, 'model')

def get_elmo_files():
    weights = project_path/config.get('ELMo Embedding Settings','weights')
    options = project_path/config.get('ELMo Embedding Settings','options')
    return [weights, options]

def get_active_model():
    return config.get('Other Settings','active')

def get_num_replies():
    return config.getint('Other Settings','NUM_REPLIES')

def get_preprocessing():
    return config.get('Other Settings', 'preprocessing')

def get_twitter_connector():
    return {
        'host': config.get('Twitter Connector','host'),
        'port': config.get('Twitter Connector','port')
    }

def get_word2vec():
    return project_path/ config.get('Other Settings','word2vec')

def get_badwords():
    return project_path/config['Other Settings']['badwords']

def get_negative_smileys():
    return project_path/config['Other Settings']['negative_smileys']

def get_positive_smileys():
    return project_path/config['Other Settings']['positive_smileys']


def get_lang_model():
    return [project_path/config['Other Settings']['langmodel'], project_path/config['Other Settings']['langmodel_train']]
