import pandas as pd
import string

from collections import Counter
from nltk.corpus import stopwords
import spacy

from svm_module.CleanTextTransformer import punctuations
from svm_module.visualize import visualize

stopwords = stopwords.words('english')


class procces:
    def __init__(self, train_df, test_df):
        self.train_df = train_df
        self.test_df = test_df

        self.nlp = spacy.load('en_core_web_sm')
        self.punctuations = string.punctuation

    def cleanup_text(self, docs, logging=False):
        texts = []
        counter = 1
        for doc in docs:
            if counter % 1000 == 0 and logging:
                print("Processed %d out of %d documents." % (counter, len(docs)))
            counter += 1
            doc = self.nlp(doc, disable=['parser', 'ner'])
            tokens = [tok.lemma_.lower().strip() for tok in doc if tok.lemma_ != '-PRON-']
            tokens = [tok for tok in tokens if tok not in stopwords and tok not in punctuations]
            tokens = ' '.join(tokens)
            texts.append(tokens)
        return pd.Series(texts)

    def ready_visulize(self):

        Action_text = [text for text in self.train_df[self.train_df['label']
                                                      == 'Action']['text']]
        Comedy_text = [text for text in self.train_df[self.train_df['label']
                                                      == 'Comedy']['text']]
        Action_clean = self.cleanup_text(Action_text)
        Action_clean = ' '.join(Action_clean).split()
        Comedy_clean = self.cleanup_text(Comedy_text)
        Comedy_clean = ' '.join(Comedy_clean).split()

        Action_counts = Counter(Action_clean)
        Comedy_counts = Counter(Comedy_clean)
        vis = visualize(train_df=self.train_df,
                        test_df=self.test_df)
        vis.visualize_chart()

        IS_common_words = [word[0] for word in Comedy_counts.most_common(20)]
        IS_common_counts = [word[1] for word in Comedy_counts.most_common(20)]

        vis.plot_words(IS_common_words, IS_common_counts,'Comedy')

        IS_common_words = [word[0] for word in Action_counts.most_common(20)]
        IS_common_counts = [word[1] for word in Action_counts.most_common(20)]

        vis.plot_words(IS_common_words, IS_common_counts,'Action')
