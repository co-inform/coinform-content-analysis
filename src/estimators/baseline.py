import joblib
import numpy as np
from estimators import language_model, feature_extractor
import logging
import settings as settings

log = logging.getLogger('server')

class BaselineModel:
    def __init__(self, stance_model_path:str,verification_model_path:str):
        self.stance_model = None
        self.verification_model = None
        with open(stance_model_path, 'rb') as f:
            self.stance_model = joblib.load(f)
        with open(verification_model_path, 'rb') as f:
            self.verification_model = joblib.load(f)
        # with open(verification_model_path, 'rb') as f:
        #     self.verification_model = pickle.loads(f)
        self.feat_extractor = feature_extractor.FeatureExtractor()
        # with open(settings.get_lang_model(), 'rb') as f:
        #     self.bow_trainings = np.load(f)
        # self.bow_generator = language_model.BagOfWordsGenerator(settings.get_lang_model()[0],settings.get_lang_model()[1])
        self.aux_feats = ['post_role', 'sentiment_analyzer','similarity','num_url','num_hashtag','num_mention','badwords','hasnegation','whwords',
                            'qmark','excmark','tripdot','smiley','named_entities']

    def estimate_veracity(self,conversation):
        log.info("stance prediction is starting...")
        source = conversation['source']
        replies = conversation['replies']

        source_response = {}
        source_response['id'] = source['id']
        source_response['text'] = source['text']
        bow_source = [self.feat_extractor.sentence_embeddings(source_response['text'])]

        replies_response = {}
        bow_replies = []
        for reply in replies:
            reply['id'] = reply['id']
            reply['text'] = reply['text']
            replies_response[reply['id']] = reply
            bow_replies.append(self.feat_extractor.sentence_embeddings(reply['text']))

        bow_all = bow_replies + bow_source
        replies_id = [item['id'] for item in replies]

        id_all = replies_id + [source['id']]

        embeddings = dict(zip(id_all, bow_all))
        feats_source = self.feat_extractor.extract_aux_feats(source, self.aux_feats, source_id=source['id'], prev_id=None,
                                                             embeddings=embeddings, post_type=1)

        feats_replies = []
        for i in range(len(replies)):
            if i == 0 :
                feats_replies.append(self.feat_extractor.extract_aux_feats(replies[i], self.aux_feats,source_id=source['id'],prev_id = None,embeddings=embeddings,post_type = 0))
            else:
                feats_replies.append(
                    self.feat_extractor.extract_aux_feats(replies[i], self.aux_feats, source_id=source['id'], prev_id=replies[i-1]['id'],
                                                          embeddings=embeddings,post_type = 0))
        if len(feats_replies) > 0:
            feats_replies = np.concatenate([bow_replies, feats_replies], axis=1)

            feats_replies = np.where(np.isnan(feats_replies), 0, feats_replies)
            log.info(self.stance_model.predict_proba(feats_replies))
            log.info(self.stance_model.predict(feats_replies))
            replies_stance_dict = dict(zip(replies_id,self.stance_model.predict_proba(feats_replies)))
            for id, value in replies_stance_dict.items():
                replies_response[id]['stance_comment'] = value[0]
                replies_response[id]['stance_deny'] =  value[1]
                replies_response[id]['stance_query']=value[2]
                replies_response[id]['stance_support']= value[3]

        log.info(np.asarray(bow_source).shape)
        log.info(feats_source.shape)
        feats_source = np.concatenate([np.asarray(bow_source), feats_source.reshape(1,-1)], axis=1)


        probs= self.stance_model.predict_proba(feats_source)[0]
        log.info(probs)
        source_response['stance_comment'] = probs[0]
        source_response['stance_deny'] = probs[1]
        source_response['stance_query'] = probs[2]
        source_response['stance_support'] = probs[3]


        if len(feats_replies) > 0:
            source_response['avg_stance_comment'] = float(sum(value[0] for value in replies_stance_dict.values())/replies_stance_dict.values().__len__())
            source_response['avg_stance_deny'] = float(sum(value[1] for value in replies_stance_dict.values())/replies_stance_dict.values().__len__())
            source_response['avg_stance_query'] = float(sum(value[2] for value in replies_stance_dict.values())/replies_stance_dict.values().__len__())
            source_response['avg_stance_support'] = float(sum(value[3] for value in replies_stance_dict.values())/replies_stance_dict.__len__())
        else:
            source_response['avg_stance_comment'] = 0.0
            source_response['avg_stance_deny'] = 0.0
            source_response['avg_stance_query'] = 0.0
            source_response['avg_stance_support'] = 0.0
        avg_stances = [source_response['avg_stance_comment'], source_response['avg_stance_deny'],
                       source_response['avg_stance_query'], source_response['avg_stance_support']]
        avg_stances = np.asarray(avg_stances).reshape(1,-1)
        log.info(avg_stances.shape)
        log.info(feats_source.shape)
        verif_feats = np.concatenate([feats_source, avg_stances], axis=1)
        log.info(verif_feats.shape)
        verif_probs = self.verification_model.predict_proba(verif_feats)[0]
        source_response['veracity_false'] = verif_probs[0]
        source_response['veracity_true'] = verif_probs[1]
        source_response['veracity_unknown'] = verif_probs[2]

        response = {}
        response['response']= source_response
        response['replies'] = replies_response

        ## compute credibility and confidence of the tweet
        veracity_true = source_response['veracity_true']
        veracity_false = source_response['veracity_false']
        veracity_unknown = source_response['veracity_unknown']

        cred_sum = veracity_true + veracity_false
        winner = veracity_true if veracity_true > veracity_false else veracity_false
        polarity = 1 if veracity_true > veracity_false else -1
        cred = polarity * winner / cred_sum
        conf = 1 - veracity_unknown

        response['credibility'] = cred
        response['confidence'] =  conf

        return response
