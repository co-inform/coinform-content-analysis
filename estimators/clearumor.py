import settings
import estimators.utils as utils
import src.dataset as dataset
import src.util as util
import src.sdqc as sdqc
import numpy as np
from torch import nn
from torch import from_numpy
from torch import argmax
import torch.nn.functional as F

from pathlib import Path


class ClearRumorModel:
    stance_model = None

    def __init__(self, stance_model:nn.Module, verification_model:nn.Module):
        self.stance_model = stance_model
        self.verification_model = verification_model
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

        return estimations[src_tweet_id], estimations, posts, post_embeddings


    def estimate_veracity(self, src_tweet_id, conversation):
        # todo check if we have already stance estimations, if not run stance, because the model depend on stances
        _,estimations, posts, post_embeddings =self.estimate_stance(src_tweet_id, conversation)
        self.verification_model.eval()
        post_features = self.verification_mode.Verif.Dataset.calc_features(estimations[src_tweet_id], posts, post_embeddings,
                                                          estimations)

        logits = self.verification_model(from_numpy(post_features).unsqueeze(0))
        prediction_probs, batch_predictions =F.softmax(logits, dim=1).max(dim=1)
        predictions = batch_predictions.data.cpu().numpy()
        prediction_probs = \
            prediction_probs.data.cpu().numpy()

        batch_confidences = np.maximum(0.5, prediction_probs)
        batch_confidences[batch_predictions
                          == self.verification_model.VerifInstance.Label.unverified.value] = 0
        batch_predictions[batch_predictions
                          == self.verification_model.VerifInstance.Label.unverified.value] = \
            self.verification_model.VerifInstance.Label.true.value
        return logits, batch_confidences, batch_predictions
