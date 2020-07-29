import logging

from typing import Union, List

import torch
import torch.optim as optim
import numpy as np
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from sklearn.metrics import precision_recall_fscore_support
from scipy.special import softmax

import pandas as pd

from tqdm import tqdm


log = logging.getLogger('server')


class SimpleBERT:
    def __init__(self, model_path: str):
        config = BertConfig(num_labels=3)
        model = BertForSequenceClassification(config)
        torch.load(model, model_path)

        self.model = model
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    def estimate_veracity(self, conversation):
        source = conversation['source']
        labels = ['true', 'false', 'unverified']

        trainer = Trainer(self.model, self.tokenizer, labels)
        credibility, conf = trainer.predict(None, source)

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
            labels: List[str],
            max_seq_length: int = 280,
            device: Union[torch.device, str] = 'cpu'
    ):
        self.max_seq_length = max_seq_length
        self.model = model
        self.device = device
        self.tokenizer = tokenizer
        self.label2idx = {label: idx for (idx, label) in enumerate(labels)}
        self.idx2label = {idx: label for (idx, label) in enumerate(labels)}

    def train(self, train_set, valid_set, batch_size, num_epochs):
        train_text, train_labels = train_set['text'], train_set['label']

        tokens_train = self.tokenizer.batch_encode_plus(
            train_text.tolist(),
            max_length=self.max_seq_length,
            pad_to_max_length=True,
            truncation=True
        )

        enc_labels = [self.label2idx[x] for x in train_labels.tolist()]

        train_seq = torch.tensor(tokens_train['input_ids'])
        train_mask = torch.tensor(tokens_train['attention_mask'])
        train_y = torch.tensor(enc_labels)

        train_data = TensorDataset(train_seq, train_mask, train_y)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

        valid_text, valid_labels = valid_set['text'], valid_set['label']

        tokens_valid = self.tokenizer.batch_encode_plus(
            valid_text.tolist(),
            max_length=self.max_seq_length,
            pad_to_max_length=True,
            truncation=True
        )

        enc_labels_val = [self.label2idx[x] for x in valid_labels.tolist()]

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

            torch.save(model.state_dict(), "model/bert_" + str(epoch) + ".pt")

    def predict(self, dataloader, text):
        if text is not None: # todo: refactor :/
            tokens = self.tokenizer.batch_encode_plus(
                [text],
                max_length=self.max_seq_length,
                pad_to_max_length=True,
                truncation=True
            )

            seq = torch.tensor(tokens['input_ids'])
            mask = torch.tensor(tokens['attention_mask'])

            with torch.no_grad():
                self.model.eval()

                data = self.model(input_ids=seq, attention_mask=mask)

                sm = softmax(data)

                preds = [np.argmax(xd) for xd in data[0].detach().numpy()]

                return self.idx2label[preds[0]], sm[preds[0]]

        pred_list = []
        with torch.no_grad():
            self.model.eval()

            for step, batch in enumerate(dataloader):
                x, m, y = batch
                data = self.model(input_ids=x, attention_mask=m)

                preds = [np.argmax(xd) for xd in data[0].detach().numpy()]
                pred_list += preds

        return pred_list

    def evaluate(self, items, labels):
        preds_list = self.predict(items, None)

        precision, recall, f_score, _ = precision_recall_fscore_support(labels, preds_list, average='macro')

        results = {
            "precision": precision,
            "recall": recall,
            "f1": f_score
        }

        return results


def main():
    labels = ['true', 'false', 'unverified']

    df = pd.read_csv("rumeval.tsv", sep='\t')
    df_dev = pd.read_csv("rumeval_dev.tsv", sep='\t')

    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    config = BertConfig(num_labels=3)
    model = BertForSequenceClassification(config)

    trainer = Trainer(model, tokenizer, labels)
    trainer.train(df, df_dev, 8, 10)


if __name__ == '__main__':
    main()