from torch.utils.data import random_split, Dataset
import pandas as pd
from sklearn.preprocessing import LabelEncoder


# Creates a dataset based on data stored in a CSV
class CSVDataset(Dataset):
    def __init__(self, path):
        # Read in the csv using pandas library
        dframe = pd.read_csv(path, header=None)
        # All rows, all but the final column will be the inputs
        self.X = dframe.values[:, :-1]
        # The final column of each row are the targets
        self.y = dframe.values[:, -1]
        # Inputs set to float32 for putting in the net
        self.X = self.X.astype('float32')
        # Encode labels to values 0 through number of classes - 1
        self.y = LabelEncoder().fit_transform(self.y)
        self.y = self.y.astype('int64')

    def __len__(self):
        return len(self.X)

    def __getitem__(self, item):
        return [self.X[item], self.y[item]]

    def get_split_sets(self, n_test=0.33):
        test_size = round(n_test * len(self.X))
        train_size = len(self.X) - test_size
        return random_split(self, [train_size, test_size])
