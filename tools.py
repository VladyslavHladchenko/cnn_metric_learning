import numpy as np
import torch

# query if we have GPU
dev = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    """ For a more elaborate solution take a look at the EasyDict package https://pypi.org/project/easydict/ """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __getstate__(self): return self.__dict__

    def __setstate__(self, d): self.__dict__.update(d)


class DataXY(torch.utils.data.Dataset):
    def __init__(self, X: torch.Tensor, Y: torch.Tensor):
        self.X = X.to(dev)
        self.Y = Y.to(dev)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return self.X.size(0)

    def split(dataset, valid_size=0.1, random_seed=0):
        """
        creates a random split into train and validation
        """
        assert ((valid_size >= 0) and (valid_size <= 1)), "[!] valid_size should be in the range [0, 1]."
        num_train = len(dataset)
        indices = list(range(num_train))
        split = int(np.ceil(valid_size * num_train))
        np.random.seed(random_seed)
        np.random.shuffle(indices)

        train_idx, val_idx = indices[split:], indices[:split]

        train_set = DataXY(dataset.X[train_idx], dataset.Y[train_idx])
        val_set = DataXY(dataset.X[val_idx], dataset.Y[val_idx])

        return train_set, val_set

    def fraction(self, fraction=0.1, random_seed=0):
        """
        Take a random fraction of the dataset containing a given fractino of all class examples
        """
        np.random.seed(random_seed)
        #
        K = self.Y.max().item() + 1
        X = []
        Y = []
        for k in range(K):
            mask = self.Y == k
            kX = self.X[mask]
            kY = self.Y[mask]
            num = kY.size(0)
            indices = list(range(num))
            split = int(np.ceil(fraction * num))
            np.random.shuffle(indices)
            idx = indices[:split]
            X += [kX[idx]]
            Y += [kY[idx]]
        return DataXY(torch.cat(X), torch.cat(Y))


def Dataset_to_XY(dataset):
    # pump through loader to get the transforms on the dataset applied
    loader = torch.utils.data.DataLoader(dataset, batch_size=1024, shuffle=False, num_workers=0)
    X = []
    Y = []
    for data, target in loader:
        X += [data]
        Y += [target]
    X = torch.cat(X)
    Y = torch.cat(Y)
    return DataXY(X, Y)



