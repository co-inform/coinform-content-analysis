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

from tqdm import tqdm

log = logging.getLogger('server')


class SimpleBERT:
    def __init__(self, model_path: str):
        model_state_dict = torch.load(model_path)
        model = BertForSequenceClassification.from_pretrained("bert-base-uncased", state_dict=model_state_dict,
                                                              num_labels=3)

        self.model = model
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def estimate_veracity(self, conversation):
        source = conversation['source']

        trainer = Trainer(self.model, self.tokenizer)
        credibility, conf = trainer.predict(source['text'])

        source_response = {'id': source['id'], 'text': source['text']}

        log.info('credibility {}, confidence {}'.format(credibility, conf))
        source_response['credibility'] = credibility
        source_response['confidence'] = conf

        return {'response': source_response}


class Trainer:
    def __init__(
            self,
            model: BertForSequenceClassification,
            tokenizer: BertTokenizer,
            max_seq_length: int = 280,
            device: Union[torch.device, str] = 'cpu'
    ):
        self.max_seq_length = max_seq_length
        self.model = model
        self.device = device
        self.tokenizer = tokenizer
        self.label2idx = {"true": 1, "false": 0, "unverified": 2}
        self.idx2label = {v: k for k, v in self.label2idx.items()}

    def train(self, train_set, valid_set, batch_size, num_epochs):
        train_text, train_labels = train_set['text'], train_set['label']

        tokens_train = self.tokenizer.batch_encode_plus(
            train_text.values,
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

            torch.save(model.state_dict(), "model/fine_tune/bert_1_best_" + str(epoch) + ".pt")

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


def train():
    df = pd.read_csv("coinform4550_split/coinform4550_train_merged.tsv", sep='\t')
    # df_re = pd.read_csv("rumeval.tsv", sep='\t')
    # df_1516 = pd.read_csv("tweeter1516_non_rum_to_unverified.tsv", sep='\t')
    # df_rest = pd.read_csv("coinform4550_split/coinform4550_rest_merged_clean.tsv", sep='\t', error_bad_lines=False)

    # df = pd.concat([df, df_1516], axis=0, sort=False, join='inner')

    df_dev = df.sample(frac=0.1)  # use as the development set
    # df_dev.to_csv('all_and_rest_dev.csv', index=False)
    # df_dev_re = pd.read_csv("rumeval_dev.tsv", sep='\t')

    df = df.loc[~df.index.isin(df_dev.index)]

    df_dev = pd.concat([df_dev], axis=0, sort=False, join='inner')

    # df_test = df.sample(frac=0.1)
    # df_test.to_csv('tweeter1516_test_04_09_fine_tune.csv', index=False)

    # df_test = pd.read_csv("coinform4550_split/coinform4550_train.tsv", sep='\t')

    # df = df.loc[~df.index.isin(df_test.index)]

    df = pd.concat([df], axis=0, sort=False, join='inner')

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)

    trainer = Trainer(model, tokenizer)
    # trainer.train(df, df_dev, 8, 5)

    # test
    rumeval_test = pd.read_csv("rumeval_test.tsv", sep='\t')
    df_dev_re = pd.concat([rumeval_test, df_test], axis=0, sort=False, join='inner') # 10 percent of twitter 15-16 + rumeval dev
    model.load_state_dict(torch.load("model/bert_1_best.pt"))

    text, labels = df_dev_re['text'], df_dev_re['label']

    tokens = tokenizer.batch_encode_plus(
        text.values,
        max_length=280,
        pad_to_max_length=True,
        truncation=True
    )

    enc_labels = [trainer.label2idx[x] for x in labels.values]

    seq = torch.tensor(tokens['input_ids'])
    mask = torch.tensor(tokens['attention_mask'])
    y = torch.tensor(enc_labels)

    data = TensorDataset(seq, mask, y)
    sampler = SequentialSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=8)

    results = trainer.evaluate(dataloader, enc_labels)

    for key in sorted(results.keys()):
        print(key, str(results[key]))

    # rumeval test + 10 percent of twitter15/16
    # f1 0.8285199834435829
    # precision 0.8320027864855449
    # recall 0.8404404523045578


# if __name__ == '__main__':
    # train()
    # m = SimpleBERT("model/fine_tune/bert_1_best_1.pt")
    # rsp = m.estimate_veracity({'source': {
    #     "id": 1,
    #     "text": 'Very tense situation in Ottawa this morning.  Multiple gun shots fired outside of our caucus room.  I am safe and in lockdown. Unbelievable.'}})
    # print()