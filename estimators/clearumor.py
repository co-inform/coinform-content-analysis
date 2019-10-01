import settings
import estimators.utils as utils
import src.dataset as dataset
import src.util as util
import src.sdqc as sdqc
import service.twitter_service as twitter_service
from torch import nn
from torch import from_numpy
from torch import argmax
import torch.nn.functional as F

from pathlib import Path


class ClearRumorModel:
    model = None

    def __init__(self, model:nn.Module):
        self.model = model
        self.device = utils.get_device()
        #todo read model parameters from the model
        self.max_sentence_length = 32
        self.batch_size = 512
        self.scalar_mix_parameters = [0, 0, 0]


    def estimate_stance(self, src_tweet_id, conversation):
        posts = {}
        for post in conversation:
            print(post)
            print(post['id_str'])
            processed_post = dataset.Post(
                id = post['id_str'],
                text=post['full_text'] if 'full_text' in post else post['text'],
                depth = post['depth'],
                platform=dataset.Post.Platform.twitter,
                has_media='media' in post['entities'],
                source_id=src_tweet_id,
                user_verified=post['user']['verified'],
                followers_count=post['user']['followers_count'],
                friends_count=post['user']['friends_count'])
            posts[processed_post.id] = processed_post

        # calculate ELMO post embeddings
        post_embeddings = util.calculate_post_elmo_embeddings(
            posts,
            max_sentence_length=self.max_sentence_length,
            batch_size=self.batch_size,
            scalar_mix_parameters=self.scalar_mix_parameters,
            device=self.device,
            elmo_weights_file = settings.get_elmo_files()[0],
            elmo_options_file = settings.get_elmo_files()[1]
            )

        # load model
        self.model.eval()

        estimations = {}

        for key, post in posts.items():
            post_features = sdqc.Sdqc.Dataset.calc_features(post, post_embeddings)
            logits = self.model(post_embeddings[key].unsqueeze(0), from_numpy(post_features).unsqueeze(0))
            probs = F.softmax(logits, dim=1).squeeze().tolist()
            prediction = argmax(logits, dim=1)

            estimations[key] = \
                        (sdqc.SdqcInstance.Label(prediction.item()),
                         dict(zip(sdqc.SdqcInstance.Label, probs)))

        return estimations[src_tweet_id], estimations



# model = torch.load(env_path, map_location='cpu')
# model.eval()
#
# results = {}
# for key, post in replies.items():
#     post_features = sdqc.Sdqc.Dataset.calc_features(post, post_embeddings)
#     logits = model(post_embeddings[key].unsqueeze(0), torch.from_numpy(post_features).unsqueeze(0))
#     probs = F.softmax(logits, dim=1).squeeze().tolist()
#     prediction = torch.argmax(logits, dim=1)
#
#     results[key] = \
#                 (sdqc.SdqcInstance.Label(prediction.item()),
#                  dict(zip(sdqc.SdqcInstance.Label, probs)))
