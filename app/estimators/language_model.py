from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import sys
import app.estimators.feature_extractor as feature_extractor
import numpy as np


class BagOfWordsGenerator:
    def __init__(self, clf_path,train_data_path):
        # inject feature extractor if symlink failed
        sys.modules['src.feature_extractor'] = feature_extractor
        self.clf = None
        with open(clf_path, 'rb') as f:
            self.clf = pickle.load(f)

        train_data = np.load(train_data_path)

        self.clf.fit_transform(train_data)

        # self.clf.fit_transform(train_data)
        # self.vector = TfidfVectorizer(
        #     tokenizer=feat_extractor.tokenize,
        #     preprocessor=feat_extractor.preprocess,
        #     ngram_range=(1, 3),
        #     stop_words=None,  # We do better when we keep stopwords
        #     strip_accents=None,
        #     use_idf=True,
        #     smooth_idf=False,
        #     norm=None,  # Applies l2 norm smoothing
        #     decode_error='replace',
        #     max_features=10000,
        #     min_df=5,
        #     max_df=0.501)
        # self.vector.fit_transform(train_data)
        # self.train = self.vector.transform(train_data).toarray()

    def transform(self, test_data):
        return self.clf.transform(test_data).toarray()

class BERTEmbeddings:
    def __init__(self):
        pass
