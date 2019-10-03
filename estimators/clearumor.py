import settings
import logging
import estimators.utils as utils
import src.dataset as dataset
import src.util as util
import src.sdqc as sdqc
import src.verif as verif
import numpy as np
from torch import nn
from torch import from_numpy
from torch import argmax
import torch.nn.functional as F

logger = logging.getLogger('server')


class ClearRumorModel:
    stance_model = None

    def __init__(self, stance_model: nn.Module, verification_model: nn.Module):
        self.stance_model = stance_model
        self.verification_model = verification_model
        self.device = utils.get_device()
        # todo read model parameters from the model
        self.max_sentence_length = 32
        self.batch_size = 512
        self.scalar_mix_parameters = [0, 0, 0]

    def estimate_veracity(self, src_tweet_id, conversation):
        posts = {}
        for post in conversation:
            processed_post = dataset.Post(
                id=post['id_str'],
                text=post['full_text'] if 'full_text' in post else post['text'],
                depth=post['depth'],
                platform=dataset.Post.Platform.twitter,
                has_media='media' in post['entities'],
                source_id=src_tweet_id,
                user_verified=post['user']['verified'],
                followers_count=post['user']['followers_count'],
                friends_count=post['user']['friends_count'])
            posts[processed_post.id] = processed_post
            logger.info(processed_post.id)
        logger.info(posts.keys())

        logger.info("Post embeddings are computing...")
        embeddings = util.calculate_post_elmo_embeddings(
            posts,
            max_sentence_length=self.max_sentence_length,
            batch_size=self.batch_size,
            scalar_mix_parameters=self.scalar_mix_parameters,
            device=self.device,
            elmo_weights_file=settings.get_elmo_files()[0],
            elmo_options_file=settings.get_elmo_files()[1]
        )

        logger.info("Post embeddings are computed.")
        self.stance_model.eval()
        logger.info("Stance detection is starting...")

        sdqc_results= {}
        for key, post in posts.items():
            post_features = sdqc.Sdqc.Dataset.calc_features(post, embeddings)
            logits = self.stance_model(embeddings[key].unsqueeze(0), from_numpy(post_features).unsqueeze(0))
            probs = F.softmax(logits, dim=1).squeeze()
            prediction = argmax(logits, dim=1)
            sdqc_results[key] = \
                (sdqc.SdqcInstance.Label(prediction.item()),
                 dict(zip(sdqc.SdqcInstance.Label, probs.tolist())))

        logger.info('Stances are detected.')
        logger.info('Source post %s features are computing..'%src_tweet_id)
        post_features = verif.Verif.Dataset.calc_features(posts[src_tweet_id], posts, embeddings,
                                                                           sdqc_results)
        logger.info('Source post %s features have been computed.'%src_tweet_id)
        self.verification_model.eval()
        logger.info("Veracity prediction is starting...")
        logits = self.verification_model(from_numpy(post_features).unsqueeze(0))
        prediction_probs, batch_predictions = F.softmax(logits, dim=1).max(dim=1)
        prediction_probs = \
            prediction_probs.data.cpu().numpy()
        batch_confidences = np.maximum(0.5, prediction_probs)
        batch_confidences[batch_predictions
                          == verif.VerifInstance.Label.unverified.value] = 0
        batch_predictions[batch_predictions
                          == verif.VerifInstance.Label.unverified.value] = \
            verif.VerifInstance.Label.true.value

        return {"stance_support": post_features[13],
                "stance_deny": post_features[14],
                "stance_query": post_features[15],
                "veracity_prediction": prediction_probs.item(),
                "veracity_label": verif.VerifInstance.Label(prediction_probs.item()).name}
