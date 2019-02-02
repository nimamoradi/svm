import ntpath

import pandas as pd

from glob import glob
from nltk.corpus import stopwords

from svm_module.srt_interface import srt


class data_transform:
    stopwords = stopwords.words('english')

    def __init__(self):
        self.train_df = None
        self.test_df = None

    def list_folders(self, path):
        folders = glob(path)
        print(folders)
        return folders

    def make_text(self, path, my_df):
        list_files = glob(path + "*.txt")

        for item in list_files:
            with open(item, 'r', encoding="utf-8") as file:
                my_df.loc[len(my_df)] = [file.read()]

    def make_row(self, path, my_df, label):
        list_files = glob(path + "*.txt")

        for item in list_files:
            with open(item, 'r', encoding="utf-8") as file:
                my_df.loc[len(my_df)] = [label, file.read()]

    def path_leaf(self, path):
        head, tail = ntpath.split(path)
        return tail or ntpath.basename(head)

    def fill_pandas(self, path):
        folders = self.list_folders(path)

        my_df = pd.DataFrame(columns=["label", "text"])

        for index in range(len(folders)):
            print("enter " + self.path_leaf(folders[index]))
            srt(folders[index])
            self.make_row(label=self.path_leaf(folders[index]), path=folders[index],
                          my_df=my_df)
        print(my_df.head())
        print(
            my_df.isnull().sum())
        return my_df

    def data(self, path):
        my_df = pd.DataFrame(columns=["text"])

        print("enter " + self.path_leaf(path))
        srt(path)
        self.make_text(path=path,
                       my_df=my_df)
        print(my_df.head())
        print(
            my_df.isnull().sum())
        return my_df

    def splitData(self, my_df):
        from sklearn.model_selection import train_test_split
        self.train_df, self.test_df = train_test_split(my_df, test_size=0.1, random_state=42)

        print('label sample:', self.train_df['label'].iloc[0])
        print('text of movie :', self.train_df['text'].iloc[0])
        print('Training Data Shape:', self.train_df.shape)
        print('Testing Data Shape:', self.test_df.shape)
