import pickle

from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC

from svm_module.CleanTextTransformer import CleanTextTransformer, tokenizeText, STOPLIST


class train:

    def __init__(self, train_df, test_df):
        self.train = train_df
        self.test = test_df
        self.fig = None
        self.vectorizer = CountVectorizer(tokenizer=tokenizeText, ngram_range=(1, 1))
        # CountVectorizer(tokenizer=tokenizeText, ngram_range=(1, 1))
        self.clf = LinearSVC()

    def printNMostInformative(slef, vectorizer, clf, N):
        feature_names = vectorizer.get_feature_names()
        coefs_with_fns = sorted(zip(clf.coef_[0], feature_names))
        topClass1 = coefs_with_fns[:N]
        topClass2 = coefs_with_fns[:-(N + 1):-1]
        print("Class 1 best: ")
        for feat in topClass1:
            print(feat)
        print("Class 2 best: ")
        for feat in topClass2:
            print(feat)

    def train_model(self):

        pipe = Pipeline([('cleanText', CleanTextTransformer()),
                         ('vectorizer', self.vectorizer), ('clf', self.clf)])
        # data
        train1 = self.train['text'].tolist()
        labelsTrain1 = self.train['label'].tolist()
        test1 = self.test['text'].tolist()
        labelsTest1 = self.test['label'].tolist()
        # train
        pipe.fit(train1, labelsTrain1)
        # store params
        joblib.dump(self.clf, 'clf.pkl', compress=1)
        joblib.dump(self.vectorizer,
                    'vectorizer.pkl', compress=1)
        # test
        preds = pipe.predict(test1)
        print("accuracy:", accuracy_score(labelsTest1, preds))
        print("Top 10 features used to predict: ")

        self.printNMostInformative(self.vectorizer, self.clf, 10)
        pipe = Pipeline([('cleanText', CleanTextTransformer()),
                         ('vectorizer', self.vectorizer)])
        transform = pipe.fit_transform(train1, labelsTrain1)

        vocab = self.vectorizer.get_feature_names()

        for i in range(len(train1)):
            s = ""
            indexIntoVocab = transform.indices[transform.indptr[i]:transform.indptr[i + 1]]
            numOccurences = transform.data[transform.indptr[i]:transform.indptr[i + 1]]
            for idx, num in zip(indexIntoVocab, numOccurences):
                s += str((vocab[idx], num))

    def load_and_test(self, my_df):

        # data
        test1 = my_df['text'].tolist()

        # pipline
        clf = joblib.load('clf.pkl')
        vectorizer = joblib.load('vectorizer.pkl')
        pipe = Pipeline([('cleanText', CleanTextTransformer()),
                         ('vectorizer', vectorizer), ('clf', clf)])
        # test
        preds = pipe.predict(test1)
        print(preds)
        print("Top 10 features used to predict: ")
