import os
import json
import torch
import twitter
from pathlib import Path  # python3 only

env_path = Path('.') / 'data/models/sdqc_model_9.pt'
folder = Path('.') / 'data/models/Replies'

arr = os.listdir(folder)
print(arr)
# replace with redis
sqdc_model = torch.load(env_path)


#read replies




#predict stance





# predict veracity

