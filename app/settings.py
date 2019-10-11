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
