import spacy
import regex as re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import numpy as np
from nltk.stem.porter import *
from scipy.spatial import distance
import app.settings as settings


class FeatureExtractor:
    def __init__(self):
        nltk.download('vader_lexicon')
        nltk.download('punkt')
        self.tool = spacy.load("en_core_web_lg")
        self.sent_analyzer = SentimentIntensityAnalyzer()
        self.stemmer = PorterStemmer()
        self.badwords = []
        self.negative_smileys = []
        self.positive_smileys = []
        with open(settings.get_badwords(), 'r') as f:
            for line in f:
                self.badwords.append(line.strip().lower())

        with open(settings.get_negative_smileys(), 'r') as f:
            for line in f:
                self.negative_smileys.append(line.strip())

        with open(settings.get_positive_smileys(), 'r') as f:
            for line in f:
                self.positive_smileys.append(line.strip())

        self.negationwords = ['not', 'no', 'nobody', 'nothing', 'none', 'never',
                              'neither', 'nor', 'nowhere', 'hardly', 'scarcely',
                              'barely', 'don', 'isn', 'wasn', 'shouldn', 'wouldn',
                              'couldn', 'doesn']

        self.whwords = ['what', 'when', 'where', 'which', 'who', 'whom', 'whose', 'why',
                        'how']
        self.entity_annotations = ['__PERSON__', '__NORP__', '__FAC__', '__ORG__', '__GPE__', '__LOC__', '__PRODUCT__',
                                   '__EVENT__',
                                   '__WORK_OF_ART__', '__LAW__', '__LANGUAGE__', '__DATE__', '__TIME__', '__PERCENT__',
                                   '__MONEY__', '__QUANTITY__', '__ORDINAL__',
                                   '__CARDINAL__']

    def extract_lemma(self):
        pass

    def extract_ne(self, text: str) -> str:
        '''
        This function extracts the name entities, and then replace them with the ne labels.
        :param text:
        :type text:
        :return:
        :rtype:
        '''
        doc = self.tool(text)
        for ent in doc.ents:
            text = text.replace(ent.text, ' __' + ent.label_ + '__ ')
        return text

    def sentence_embeddings(self, text):
        text = self.preprocess(text)
        vector = self.tool(text).vector
        return vector

    def emoji(self, text: str) -> str:
        pass

    '@todo'

    def filter_hashtag(self, text: str) -> str:
        '''
        This function extracts hashtags and then replace with <hashtag>
        :param text:
        :type text:
        :return:
        :rtype:
        '''
        text = re.sub(r"^#.*", '__hashtag__', text)
        return [text, text.count('__hashtag__')]

    '@todo'

    def extract_mention(self, text: str) -> str:
        '''
        This function extracts mentions and the replace with <mention>
        :param text:
        :type text:
        :return:
        :rtype:
        '''
        return re.sub(r"^@.*", ' __usermention__ ', text)

    def post_role(self, post):
        '''
        if it is source return 1, if it is reply return 0
        :return:
        :rtype:
        '''
        return post.has_source_depth == 0

    '@todo make callback functions as input'

    def preprocess(self, text: str) -> str:
        space_pattern = '\s+'
        text = re.sub(space_pattern, ' ', text)
        text = self.filter_hashtag(text)[0]
        text = self.filter_url(text)[0]
        text = self.filter_mention(text)[0]
        text = re.sub('\'', ' ', text)
        text = re.sub('____', '__ __', text)
        text = self.extract_ne(text)
        return str(text)

    def tokenize(self, text):
        """Removes punctuation & excess whitespace, sets to lowercase,
        and stems tweets. Returns a list of stemmed tokens."""
        tweet = " ".join(re.split("[^a-zA-Z]*", text.lower())).strip()
        tokens = [self.stemmer.stem(t) for t in tweet.split()]
        return tokens

    def similarity(self, post_1_id, post_2_id, embeddings):
        dist = distance.cosine(embeddings[post_1_id], embeddings[post_2_id])
        return dist

    def filter_url(self, text):
        giant_url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
                           '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        text = re.sub(giant_url_regex, '__url__', text)
        return [text, text.count('__url__')]

    def filter_mention(self, text):
        mention_regex = '@[\w\-]+'
        text = re.sub(mention_regex, '__usermention__', text)
        return [text, text.count('__usermention__')]

    def has_badword(self, tokens):
        bad_words = 0
        for token in tokens:
            if token in self.badwords:
                bad_words += 1
        return bad_words / len(self.badwords)

    def has_negation(self, tokens):
        negation_words = 0
        for negationword in self.negationwords:
            if negationword in tokens:
                negation_words += 1
        return negation_words / len(self.negationwords)

    def has_smileys(self, text):
        if len(text) == 0:
            return [0, 0]
        positive_smileys = 0
        negative_smileys = 0
        for smiley in self.positive_smileys:
            if smiley in text:
                positive_smileys += text.count(smiley)
        for smiley in self.negative_smileys:
            if smiley in text:
                negative_smileys += text.count(smiley)
        return [positive_smileys / len(text), negative_smileys / len(text)]

    def has_whwords(self, tokens):
        wh_words = 0
        for token in tokens:
            if token in self.whwords:
                wh_words += 1
        return wh_words / len(self.whwords)

    def other_tokenizer(self, text):
        return nltk.word_tokenize(re.sub(r'([^\s\w]|_)+', '',
                                         text.lower()))

    def check_entities(self, text):
        entity_feats = []
        for annotation in self.entity_annotations:
            entity_feats.append(1) if annotation in text else entity_feats.append(0)
        return entity_feats

    def extract_aux_feats(self, item, args, source_id, prev_id, embeddings, post_type):
        aux_feats = []
        text = self.preprocess(item['text'])
        if 'post_role' in args:
            aux_feats.append(post_type)
        if 'sentiment_analyzer' in args:
            aux_feats.append(self.sent_analyzer.polarity_scores(text)['pos'])
            aux_feats.append(self.sent_analyzer.polarity_scores(text)['neu'])
            aux_feats.append(self.sent_analyzer.polarity_scores(text)['neg'])
            aux_feats.append(self.sent_analyzer.polarity_scores(text)['compound'])
        if 'similarity' in args:
            if item['id'] == source_id:
                aux_feats.append(1)
            else:
                aux_feats.append(self.similarity(item['id'], source_id, embeddings))
        if 'num_url' in args:
            aux_feats.append(self.filter_url(item['text'])[1])
        if 'num_mention' in args:
            aux_feats.append(self.filter_mention(item['text'])[1])
        if 'num_hashtag' in args:
            aux_feats.append(self.filter_hashtag(item['text'])[1])

        tokens = self.other_tokenizer(item['text'])
        if 'badwords' in args:
            aux_feats.append(self.has_badword(tokens))
        if 'hasnegation' in args:
            aux_feats.append(self.has_negation(tokens))
        if 'whwords' in args:
            aux_feats.append(self.has_whwords(tokens))
        if 'qmark' in args:
            aux_feats.append(1 if len(text) and '?' in text else 0)
        if 'excmark' in args:
            aux_feats.append(1 if len(text) and '!' in text else 0)
        if 'tripdot' in args:
            aux_feats.append(1 if len(text) and '...' in text else 0)
        if 'capital' in args:
            aux_feats.append(float(sum(1 for c in text if c.isupper())) / len(text)) if len(
                text) > 0 else aux_feats.append(0)
        if 'smileys' in args:
            aux_feats.append(self.has_smileys(text))
        if 'named_entities' in args:
            for score in self.check_entities(text):
                aux_feats.append(score)
        return np.asarray(aux_feats)

    def sentence_embeddings(self, text):
        text = self.preprocess(text)
        return self.tool(text).vector
