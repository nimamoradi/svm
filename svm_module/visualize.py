import seaborn as sns
import matplotlib.pyplot as plt


class visualize:
    def __init__(self, train_df, test_df):
        self.train_df = train_df
        self.test_df = test_df
        self.fig = None

    def visualize_chart(self):
        self.fig = plt.figure(figsize=(8, 4))
        sns.barplot(x=self.train_df['label'].unique(), y=self.train_df['label'].value_counts())
        plt.show()

    def plot_words(self, IS_common_words, IS_common_counts,label):
        self.fig = plt.figure(figsize=(18, 6))

        sns.barplot(x=IS_common_words, y=IS_common_counts)
        plt.title('Most Common Words used in '+label)
        plt.show()

