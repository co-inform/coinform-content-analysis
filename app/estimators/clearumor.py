import settings
import logging
import app.estimators.utils as utils
import src.dataset as dataset
import src.util as util
import src.sdqc as sdqc
import src.verif as verif
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

    def estimate_veracity(self,conversation):
        posts = {}
        # add source tweet
        source_response = {}
        source_post = dataset.Post(
                id=conversation['source']['id'],
                text=conversation['source']['text'],
                depth=0,
                platform=dataset.Post.Platform.twitter,
                has_media=conversation['source']['media'],
                source_id=None,
                user_verified=conversation['source']['is_verified'],
                followers_count=conversation['source']['followers_count'],
                friends_count=conversation['source']['friends_count'])
        posts[source_post.id] = source_post
        source_response['id'] = conversation['source']['id']
        source_response['text'] = conversation['source']['text']

        replies_response = {}
        for post in conversation['replies']:
            reply = {}
            processed_post = dataset.Post(
                id=post['id'],
                text=post['text'],
                depth=1,
                platform=dataset.Post.Platform.twitter,
                has_media=post['media'],
                source_id=source_post.id,
                user_verified=post['is_verified'] == 0,
                followers_count=post['followers_count'],
                friends_count=post['friends_count'])
            posts[processed_post.id] = processed_post
            reply['id'] = post['id']
            reply['text'] = post['text']
            replies_response[post['id']] = reply

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
            stance_response = {}
            probabilities = probs.tolist()
            if key == source_post.id:
                source_response['stance_support'] = probabilities[0]
                source_response['stance_deny'] = probabilities[1]
                source_response['stance_comment'] = probabilities[2]
                source_response['stance_query'] = probabilities[3]
            else:
                replies_response[key]['stance_support'] = probabilities[0]
                replies_response[key]['stance_deny'] = probabilities[1]
                replies_response[key]['stance_comment'] = probabilities[2]
                replies_response[key]['stance_query'] = probabilities[3]
            prediction = argmax(logits, dim=1)
            sdqc_results[key] = \
                (sdqc.SdqcInstance.Label(prediction.item()),
                 dict(zip(sdqc.SdqcInstance.Label, probabilities)))

        logger.info('Stances are detected.')
        logger.info('Source post %s features are computing..'%source_post.id)
        post_features = verif.Verif.Dataset.calc_features(posts[source_post.id], posts, embeddings,
                                                                           sdqc_results)
        logger.info('Source post %s features have been computed.'%source_post.id)
        self.verification_model.eval()
        logger.info("Veracity prediction is starting...")
        logits = self.verification_model(from_numpy(post_features).unsqueeze(0))
        prediction_probs = F.softmax(logits, dim=1).squeeze()
        prediction_probs = prediction_probs.tolist()
        source_response['veracity_false'] = prediction_probs[0]
        source_response['veracity_true'] = prediction_probs[1]
        source_response['veracity_unknown'] = prediction_probs[2]

        response = {}
        response['source']= source_response
        response['replies'] = replies_response
        return response
