import pandas as pd 
import pickle

TRAINDATA = 'all_jets_train_4D_100_percent.pkl'
TESTDATA = 'all_jets_test_4D_100_percent.pkl'

# To be thread-safe on windows, using main()
def main():
    with open(TRAINDATA, 'rb') as f:
        print('-'*64)
        traindata = pickle.load(f)
        print(traindata.info())
        print(traindata.describe())

    with open(TESTDATA, 'rb') as f:
        print('-'*64)
        testdata = pickle.load(f)
        print(testdata.info())
        print(testdata.describe())


if __name__ == '__main__':
    main()