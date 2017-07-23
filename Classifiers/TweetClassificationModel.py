import re

import datetime
from sklearn.externals import joblib
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.svm import LinearSVC


class TweetClassificationModel():
    def __init__(self):
        self.classifier=None
        self.vectorizer = DictVectorizer()
        self.ch2=None

    def train(self, tweets):
        '''
        Trains a tweet classification model based on the defined tweet classes and list of annotated tweet object data
        :param tweets: list of Tweet objects
        :return: True if model trained successfully, False otherwise
        '''

        dict_features= self._vectorize(tweets)
        x_train =self.vectorizer.fit_transform(dict_features)
        y_train = self._get_labels(tweets)

        # Refine feature set
        x_train = self._chi2_training_features(x_train,y_train, k=400, vectorizer=self.vectorizer)

        self.classifier = LinearSVC()
        self.classifier.fit(x_train, y_train)

    def predict(self, tweets):
        if self.classifier:

            dict_features = self._vectorize(tweets)

            x_test = self.vectorizer.transform(dict_features)

            # refine feature set
            x_test = self._chi2_testing_features(x_test)

            return self.classifier.predict(x_test)

        else:
            raise Exception("Cannot predict without a model. Load a stored model using: .load_model(model_dir)")


    def load_model(self, model_dir):
        '''
        Loads dumped model from desired filepath
        :param model_dir: string
        :return: a sklearn classifier object
        '''
        self.classifier = joblib.load(model_dir)

    def dump_model(self, out_dir):
        '''
        Writes TweetClassification model to disk for persistance
        :return: True if write todisk success, false otherwise
        '''
        joblib.dump(self.classifier, out_dir)

    def _vectorize(self, tweets):
        '''
        Transform raw tweet objects to vector representations
        :param tweets: Tweet objects
        :return: list of feature vectors x
        '''
        feature_dicts = list()
        for tweet in tweets:
            feature_dict = {
                'handle=' + tweet.handle : True,
                'bio_contains_link=' + self._contains_link(tweet.text) :True,
                'num_followers=' + self._num_followers_bucketed(tweet.followers): True,
                'bio_first_pers_pron=' + self._contains_fpp(tweet.desc):True,
                'handle_ctns_coin_lex=' + self._has_coin_lex(tweet.desc):True,
                'num_hashtags=' + self._count_hashtags(tweet.text):True,
                'years_ago_acct_created=' + self._get_years_ago(tweet.user_created):True,
                'has_vowels_in_handle=' + self._has_vowels(tweet.handle):True
            }
            feature_dict.update(self._get_ngram_feats(tweet.text))
            feature_dict.update(self._get_ngram_feats(tweet.text, n=2))
            feature_dicts.append(feature_dict)
        return feature_dicts

    def _get_ngram_feats(self, text, n=1):
        text = re.sub('[^\s]+\.[^\s]+(\/)+[^\s]+', "URL", text)
        text=re.sub('\d', '0',text) # replace all numbers with 0s
        text=re.sub('[.,?\':;\"#@]', "", text)
        toks= text.split()
        ngrams= zip(*[toks[i:] for i in range(n)])
        new_data = ['_'.join(w).strip('.;\'\"=,') for w in ngrams]
        return dict.fromkeys(new_data, True)

    def _get_labels(self, tweets):
        return [t.gold_label for t in tweets]

    def _chi2_training_features(self,X_train, y_train, k, vectorizer=None):
        self.ch2 = SelectKBest(chi2, k=k)
        X_train = self.ch2.fit_transform(X_train, y_train)
        if vectorizer:
            feature_names = vectorizer.get_feature_names()
            # keep selected feature names
            feature_names = [feature_names[i] for i in self.ch2.get_support(indices=True)]
        return X_train

    def _chi2_testing_features(self, x_test):
        X_test = self.ch2.transform(x_test)
        return X_test

    def dump_vectorizer(self, out_dir):
        joblib.dump(self.vectorizer, out_dir)
        pass

    def load_vectorizer(self, vectorizer_path):
        self.vectorizer = joblib.load(vectorizer_path)

    def dump_ch2(self, out_dir):
        joblib.dump(self.ch2, out_dir)
        pass

    def load_ch2(self, ch2_path):
        self.ch2 = joblib.load(ch2_path)

    def _contains_link(self, text):
        if re.search("(http(s)?(://)?)?(www.)?[a-zA-Z]+.(com|net|io|org).*", text):
            return "True"
        return "False"

    def _num_followers_bucketed(self, followers):
        if followers > 1000:
            return ">1000"
        elif followers > 100 and followers<1000:
            return "100<x<1000"
        else:
            return "<100"


    def _contains_fpp(self, desc):
        if desc:
            if re.search("(i\s+|me|my)", desc.lower()):
                return "True"
        return "False"


    def _has_coin_lex(self, desc):
        if desc:
            if re.search("(coin|btc|crypto)", desc.lower()):
                return "True"
        return "False"

    def _count_hashtags(self, text):
        result = re.findall("#", text)
        num = len(result)
        if num ==1:
            return "1"
        elif num == 2:
            return"2"
        elif num ==0:
            return "0"
        else:
            return ">2"

    def _get_years_ago(self, user_created):
        now = datetime.datetime.now()
        now_year = now.year
        year = int(str(user_created).split("-")[0])
        return str(now_year-year)

    def _has_vowels(self, handle):
        if re.search("[aeiou]", handle):
            return "True"
        return "False"


