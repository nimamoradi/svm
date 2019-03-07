import ntpath
import os

import pandas as pd

from glob import glob
from nltk.corpus import stopwords
import re, sys


def is_time_stamp(l):
    if l[:2].isnumeric() and l[2] == ":":
        return True
    return False


def has_letters(line):
    if re.search("[a-zA-Z]", line):
        return True
    return False


def has_no_text(line):
    l = line.strip()
    if not len(l):
        return True
    if l.isnumeric():
        return True
    if is_time_stamp(l):
        return True
    if l[0] == "(" and l[-1] == ")":
        return True
    if not has_letters(line):
        return True
    return False


def is_lowercase_letter_or_comma(letter):
    if letter.isalpha() and letter.lower() == letter:
        return True
    if letter == ",":
        return True
    return False


def clean_up(lines):
    """
  Get rid of all non-text lines and
  try to combine text broken into multiple lines
  """
    new_lines = []
    for line in lines[1:]:
        if has_no_text(line):
            continue
        elif len(new_lines) and is_lowercase_letter_or_comma(line[0]):
            # combine with previous line
            new_lines[-1] = new_lines[-1].strip() + " " + line
        else:
            # append line
            new_lines.append(line)
    return new_lines


def main(file_name, ):
    """
    args[1]: file name
    args[2]: encoding. Default: utf-8.
      - If you get a lot of [?]s replacing characters,
      - you probably need to change file_encoding to 'cp1252'
  """

    file_encoding = "utf-8"
    with open(file_name, encoding=file_encoding, errors="replace") as f:
        lines = f.readlines()
        new_lines = clean_up(lines)
    new_file_name = file_name[:-4] + ".txt"
    with open(new_file_name, "w", encoding=file_encoding) as f:
        for line in new_lines:
            f.write(line)


def srt(path):
    import glob

    list_files = glob.glob(path + "/*.srt")
    print(len(list_files))

    for target_list in list_files:
        main(target_list)

    list_files = glob.glob(path + "/*.txt")
    print(len(list_files))


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
                my_df.loc[len(my_df)] = [os.path.basename(item), label, file.read()]

    def path_leaf(self, path):
        head, tail = ntpath.split(path)
        return tail or ntpath.basename(head)

    def fill_pandas(self, path):
        folders = self.list_folders(path)

        my_df = pd.DataFrame(columns=["movie_name", "label", "text"])

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
        self.train_df, self.test_df = train_test_split(my_df, test_size=0.1,
                                                       random_state=42)

        print('label sample:', self.train_df['label'].iloc[0])
        print('text of movie :', self.train_df['text'].iloc[0])
        print('Training Data Shape:', self.train_df.shape)
        print('Testing Data Shape:', self.test_df.shape)
