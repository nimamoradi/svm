import sys

from svm_module.data_transfrom import data_transform
from svm_module.train import train

if __name__ == '__main__':
    tra = train(train_df=None, test_df=None)
    data_transformed = data_transform()
    my_df = data_transformed.data(sys.argv[1])
    tra.load_and_test(my_df)
