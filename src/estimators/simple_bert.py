import logging

from typing import Union

import torch
import torch.optim as optim
import numpy as np
from transformers import BertForSequenceClassification, BertTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from sklearn.metrics import precision_recall_fscore_support
from scipy.special import softmax

import pandas as pd
import re
import joblib

# from estimators.baseline import load_replies
from estimators import feature_extractor

from tqdm import tqdm

log = logging.getLogger('server')

tokens_replies = ["stcmm_0", "stdny_0", "stqry_0", "stspp_0",
                  "stcmm_1", "stdny_1", "stqry_1", "stspp_1",
                  "stcmm_2", "stdny_2", "stqry_2", "stspp_2",
                  "stcmm_3", "stdny_3", "stqry_3", "stspp_3"]


class SimpleBERT:
    def __init__(self, model_path: str, stance_model_path: str):
        model_state_dict = torch.load(model_path)
        model = BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                              num_labels=3)
        self.model = model
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.tokenizer.add_tokens(tokens_replies)
        self.model.resize_token_embeddings(len(self.tokenizer))

        model.load_state_dict(model_state_dict)

        # stance stuff
        with open(stance_model_path, 'rb') as f:
            self.stance_model = joblib.load(f)
        self.feat_extractor = feature_extractor.FeatureExtractor()
        self.aux_feats = ['post_role', 'sentiment_analyzer', 'similarity', 'num_url', 'num_hashtag', 'num_mention',
                          'badwords', 'hasnegation', 'whwords',
                          'qmark', 'excmark', 'tripdot', 'smiley', 'named_entities']

    def estimate_veracity(self, conversation):
        source = conversation['source']

        add, replies_response = self.count_replies((source['id'], source["text"]), conversation["replies"])
        text = source["text"] + " " + add

        trainer = Trainer(self.model, self.tokenizer)
        credibility, conf = trainer.predict(text)

        source_response = {'id': source['id'], 'text': source['text']}

        log.info('credibility {}, confidence {}'.format(credibility, conf))
        source_response['credibility'] = credibility
        source_response['confidence'] = conf

        response = {'response': source_response, 'replies': replies_response}

        return response

    def count_replies(self, id_text, replies):
        """
        Creates a string that contains four token types with counts of each type of reply
        the source tweet has.

        stcmm - number of comments
        stdny - number of denying replies
        stqry - number of queries
        stspp - number of supporting replies

        :param id_text: source tweet text and id tuple
        :param replies: replies to tweet
        :return: a string of four tokens with count of different types of replies
        """
        if len(replies) == 0:
            return "stcmm_0 stdny_0 stqry_0 stspp_0"

        id, text = id_text
        bow_source = [self.feat_extractor.sentence_embeddings(text)]

        bow_replies = []
        replies_response = {}
        for reply in replies:
            replies_response[reply['id']] = reply
            bow_replies.append(self.feat_extractor.sentence_embeddings(reply['text']))

        bow_all = bow_replies + bow_source
        replies_id = [item['id'] for item in replies]

        id_all = replies_id + [id]

        embeddings = dict(zip(id_all, bow_all))

        feats_replies = []
        for i in range(len(replies)):
            if i == 0:
                feats_replies.append(
                    self.feat_extractor.extract_aux_feats(replies[i], self.aux_feats, source_id=id,
                                                          prev_id=None,
                                                          embeddings=embeddings, post_type=0))
            else:
                feats_replies.append(
                    self.feat_extractor.extract_aux_feats(replies[i], self.aux_feats, source_id=id,
                                                          prev_id=replies[i - 1]['id'],
                                                          embeddings=embeddings, post_type=0))
        if len(feats_replies) > 0:
            feats_replies = np.concatenate([bow_replies, feats_replies], axis=1)

            feats_replies = np.where(np.isnan(feats_replies), 0, feats_replies)

            replies_stance_dict = dict(zip(replies_id, self.stance_model.predict_proba(feats_replies)))
            s_c = s_d = s_q = s_s = 0
            for id, value in replies_stance_dict.items():
                replies_response[id]['stance_comment'] = value[0]
                replies_response[id]['stance_deny'] = value[1]
                replies_response[id]['stance_query'] = value[2]
                replies_response[id]['stance_support'] = value[3]

                a = value.tolist()

                idx = a.index(max(a))
                if idx == 0:
                    s_c += 1
                elif idx == 1:
                    s_d += 1
                elif idx == 2:
                    s_q += 1
                elif idx == 3:
                    s_s += 1

            return "stcmm_" + str(map_to_range(s_c)) + " stdny_" + str(map_to_range(s_d)) + \
                   " stqry_" + str(map_to_range(s_q)) + " stspp_" + str(map_to_range(s_s)), replies_response


def map_to_range(n):
    if n == 0:
        return 0
    if 10 > n >= 1:
        return 0
    if 50 > n >= 10:
        return 3
    if n >= 50:
        return 4


class Trainer:
    def __init__(
            self,
            model: BertForSequenceClassification,
            tokenizer: BertTokenizer,
            max_seq_length: int = 315,
            device: Union[torch.device, str] = 'cpu'
    ):
        self.max_seq_length = max_seq_length
        self.model = model
        self.device = device
        self.tokenizer = tokenizer
        self.tokenizer.add_tokens(tokens_replies)
        self.model.resize_token_embeddings(len(tokenizer))

        self.label2idx = {"true": 1, "false": 0, "unverified": 2}
        self.idx2label = {v: k for k, v in self.label2idx.items()}

    def train(self, train_set, valid_set, batch_size, num_epochs):
        train_text, train_labels = train_set['text'], train_set['label']
        train_text = [tiny_preprocess(str(t)) for t in train_text]

        tokens_train = self.tokenizer.batch_encode_plus(
            train_text,
            max_length=self.max_seq_length,
            pad_to_max_length=True,
            truncation=True
        )

        enc_labels = [self.label2idx[x] for x in train_labels.values]

        train_seq = torch.tensor(tokens_train['input_ids'])
        train_mask = torch.tensor(tokens_train['attention_mask'])
        train_y = torch.tensor(enc_labels)

        train_data = TensorDataset(train_seq, train_mask, train_y)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

        valid_text, valid_labels = valid_set['text'], valid_set['label']
        valid_text = [tiny_preprocess(str(t)) for t in valid_text]

        tokens_valid = self.tokenizer.batch_encode_plus(
            valid_text,
            max_length=self.max_seq_length,
            pad_to_max_length=True,
            truncation=True
        )

        enc_labels_val = [self.label2idx[x] for x in valid_labels.values]

        val_seq = torch.tensor(tokens_valid['input_ids'])
        val_mask = torch.tensor(tokens_valid['attention_mask'])
        val_y = torch.tensor(enc_labels_val)

        val_data = TensorDataset(val_seq, val_mask, val_y)
        val_sampler = SequentialSampler(val_data)
        val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)

        model = self.model
        optimizer = optim.Adam(model.parameters(), lr=5e-5)

        for epoch in range(num_epochs):
            batch_iterator = tqdm(train_dataloader, desc="Iteration", disable=False)
            for step, batch in enumerate(batch_iterator):
                model.train()
                model.zero_grad()

                x, m, y = batch
                loss, data = model(input_ids=x, attention_mask=m, labels=y)

                loss.backward()
                optimizer.step()

            print("\nEpoch: " + str(epoch) + "__________________________\n")

            results = self.evaluate(val_dataloader, enc_labels_val)
            for key in sorted(results.keys()):
                print(key, str(results[key]))

            torch.save(model.state_dict(), "model/new_fine_tune_replies/bert_" + str(epoch) + ".pt")

    def predict(self, text):
        tokens = self.tokenizer.encode(
            text,
            max_length=self.max_seq_length,
            pad_to_max_length=True,
            truncation=True,
            return_tensors='pt'
        )

        with torch.no_grad():
            self.model.eval()

            data = self.model(input_ids=tokens)

            sm = softmax(data[0].detach().numpy())

            sm = sm.flatten()

            veracity_false = sm[0]
            veracity_true = sm[1]
            veracity_unknown = sm[2]

            # copied from baseline.py
            cred_sum = veracity_true + veracity_false
            winner = veracity_true if veracity_true > veracity_false else veracity_false
            polarity = 1 if veracity_true > veracity_false else -1
            cred = polarity * winner / cred_sum
            conf = 1 - veracity_unknown
            #

            return cred, conf

    def evaluate(self, items, labels):
        pred_list = []
        with torch.no_grad():
            self.model.eval()

            for step, batch in enumerate(items):
                x, m, y = batch
                data = self.model(input_ids=x, attention_mask=m)

                preds = [np.argmax(xd) for xd in data[0].detach().numpy()]
                pred_list += preds

        precision, recall, f_score, _ = precision_recall_fscore_support(labels, pred_list, average='macro')

        results = {
            "precision": precision,
            "recall": recall,
            "f1": f_score
        }

        return results


# def train():
#     bb = SimpleBERT("", "data/models/baseline.pkl")
#
#     model_f_t = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)
#     model_f_t.load_state_dict(torch.load("model/fine_tune/bert_BEST.pt"))
#
#     tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#     tokenizer.add_tokens(tokens_replies)
#
#     model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)
#     model.bert.load_state_dict(model_f_t.bert.state_dict())
#     model.resize_token_embeddings(len(tokenizer))
#
#     trainer = Trainer(model, tokenizer)
#
#     df = pd.read_csv("coinform4550_split/coinform4550_train_merged.tsv", sep='\t')
#     df = df.dropna(subset=['text', 'id', 'label'])
#     df['id'] = df['id'].astype(int)
#
#     replies = load_replies()
#     for i, row in df.iterrows():
#         id = str(row["id"])
#         if id in replies:
#             replies_ = replies[id]
#         else:
#             replies_ = []
#         add = bb.count_replies((id, row["text"]), replies_)
#         df.at[i, 'text'] = row['text'] + " " + add
#
#     df_dev = df.sample(frac=0.1)  # use as the development set
#     df_dev.to_csv('coinform_4550_train_dev.csv', index=False)
#
#     df = df.loc[~df.index.isin(df_dev.index)]
#
#     # df_dev = pd.concat([df_dev], axis=0, sort=False, join='inner')
#     # df = pd.concat([df], axis=0, sort=False, join='inner')
#
#     trainer.train(df, df_dev, 8, 10)


def tiny_preprocess(text):
    text = re.sub("http://\S+|https://\S+", 'URL', text, flags=re.MULTILINE)
    text = text.replace("\n", "")
    text = text.lower()
    text = re.sub(r'@\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r"(.)\1+", r"\1\1", text)
    text = text.replace('  ', ' ')

    return text


# def predict_all():
#     m = SimpleBERT("model/new_fine_tune_replies/bert_veracity.pt", "data/models/baseline.pkl")
#
#     tweet_id_replies = load_replies()
#
#     with open("coinform4550_split/coinform4550_test.tsv") as f, open("predict_BEST_new_fine_tune_replies_2.tsv",
#                                                                      "w") as fo:
#         fo.write("id\tcred\tconf\n")
#         n = 0
#         for l in f:
#             if n == 0:
#                 n += 1
#                 continue
#             parts = l.split('\t')
#             if len(parts[0]) == 0:
#                 continue
#
#             text = tiny_preprocess(parts[0])
#             id = parts[2]
#
#             replies = []
#             if id in tweet_id_replies:
#                 replies = tweet_id_replies[id]
#
#             rsp = m.estimate_veracity({'source': {
#                 "id": id,
#                 "text": text,
#                 "replies": replies}})
#             fo.write(parts[2] + '\t' + str(rsp['response']['credibility']) + '\t' + str(
#                 rsp['response']['confidence']) + '\n')

# if __name__ == '__main__':
#     # train()
#     # predict_all()
#     s = SimpleBERT("model/new_fine_tune_replies/bert_veracity.pt", "data/models/baseline.pkl")
#     c = {"source": {"id": 123, "text": "I hate apples"} ,
#     "replies": [{"id": 123, "text": "Me too"}]
#     }
#     rrr = s.estimate_veracity(c)
#     print()
