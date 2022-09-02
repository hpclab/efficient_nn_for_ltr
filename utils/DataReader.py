import torch

import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from rankeval.dataset import Dataset as DatasetRankEval



class MSNDataReader(Dataset):
    def __init__(self, X, original_model, scaler, transform=None):
        self.X = scaler.transform(X)

        empty_array = np.empty(shape=len(X))
        dataset_temp = DatasetRankEval(X, empty_array, empty_array)
        self.y = original_model.score(dataset_temp)

        self.length = len(X)
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class MSNBatchCreator(object):
    '''
    input: (X, y), where X is already scaled
    '''

    def __init__(self, midpoints, original_model, scaler, n_artificial_samples):
        self.midpoints = midpoints
        self.original_model = original_model
        self.scaler = scaler
        self.n_artificial_samples = n_artificial_samples
        print("ARTIFICIAL SAMPLES: ", self.n_artificial_samples)

    def __call__(self, batch):
        if self.n_artificial_samples == 0:
            X_total = np.array([x[0] for x in batch])
            y_total = np.array([x[1] for x in batch])
            return torch.from_numpy(X_total).type(torch.FloatTensor), torch.from_numpy(y_total).type(
                torch.FloatTensor) # for weighted loss compatibility

        # original_batch = np.stack(batch)
        X = np.array([x[0] for x in batch])
        y = np.array([x[1] for x in batch])
        n_features = X.shape[1]
        X_art = np.ndarray([self.n_artificial_samples, n_features], dtype=X[0].dtype)
        empty_array = np.empty(shape=self.n_artificial_samples)
        self.generate_multiple_random_feature_vector(X_art, self.n_artificial_samples)
        dataset_batch = DatasetRankEval(X_art, empty_array, empty_array)
        X_art_scaled = self.scaler.transform(X_art)

        y_art = self.original_model.score(dataset_batch)

        X_total = np.concatenate((X, X_art_scaled), axis=0)
        y_total = np.concatenate((y, y_art), axis=0)

        return torch.from_numpy(X_total).type(torch.FloatTensor), torch.from_numpy(y_total).type(torch.FloatTensor)

    def generate_multiple_random_feature_vector(self, new_batch, size):
        for f, m in enumerate(self.midpoints):
            if m.shape[0]:
                new_batch[:, f] = np.random.choice(m, size=size, replace=True)
            else:
                new_batch[:, f] = np.zeros(size)


class IstellaDataReader(Dataset):
    def __init__(self, X, original_model, scaler, imputer,transform=None):

        self.X = imputer.transform(X) #for the validation set
        self.X = scaler.transform(self.X)

        empty_array = np.empty(shape=len(X))
        dataset_temp = DatasetRankEval(X, empty_array, empty_array)
        self.y = original_model.score(dataset_temp)

        self.length = len(X)
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class BatchCreatorIstella(object):
    def __init__(self, midpoints, original_model, scaler ,  n_artificial_samples):
        #I midpoint sono calcolati dopo imputer su train_dataset
        self.midpoints = midpoints
        self.original_model = original_model
        #self.imputer = imputer
        self.scaler = scaler
        self.n_artificial_samples = n_artificial_samples
        print("ARTIFICIAL SAMPLES: ", self.n_artificial_samples)

    def __call__(self, batch):
        if self.n_artificial_samples == 0:
            X_total = np.array([x[0] for x in batch])
            y_total = np.array([x[1] for x in batch])
            return torch.from_numpy(X_total).type(torch.FloatTensor), torch.from_numpy(y_total).type(
                torch.FloatTensor)
        # original_batch = np.stack(batch)
        X = np.array([x[0] for x in batch])
        y = np.array([x[1] for x in batch])
        n_features = X.shape[1]
        X_art = np.ndarray([self.n_artificial_samples, n_features], dtype=X[0].dtype)
        empty_array = np.empty(shape=self.n_artificial_samples)
        self.generate_multiple_random_feature_vector(X_art, self.n_artificial_samples)
        dataset_batch = DatasetRankEval(X_art, empty_array, empty_array)
        y_art = self.original_model.score(dataset_batch)

        X_art_scaled = self.scaler.transform(X_art)



        X_total = np.concatenate((X, X_art_scaled), axis=0)
        y_total = np.concatenate((y, y_art), axis=0)

        return torch.from_numpy(X_total).type(torch.FloatTensor), torch.from_numpy(y_total).type(torch.FloatTensor)

    def generate_multiple_random_feature_vector(self, new_batch, size):
        for f, m in enumerate(self.midpoints):
            if m.shape[0]:
                new_batch[:, f] = np.random.choice(m, size=size, replace=True)
            else:
                new_batch[:, f] = np.zeros(size)


