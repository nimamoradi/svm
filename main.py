import pickle
import sys

from svm_module.data_transfrom import data_transform
from svm_module.procces import procces
from svm_module.train import train

if __name__ == '__main__':
    data_transformed = data_transform()
    my_df = data_transformed.fill_pandas(sys.argv[1] + "/*/")
    data_transformed.splitData(my_df)

    train_df = data_transformed.train_df
    test_df = data_transformed.test_df
    with open('train_df.pkl', 'wb') as f:
        pickle.dump(train_df, f, pickle.HIGHEST_PROTOCOL)

    with open('test_df.pkl', 'wb') as f:
        pickle.dump(test_df, f, pickle.HIGHEST_PROTOCOL)
    del data_transformed

    print(train_df.head())
    print(test_df.head())

    proc = procces(train_df=train_df, test_df=test_df)

    proc.ready_visulize()

    tra = train(train_df=train_df, test_df=test_df)

    tra.train_model()
