import numpy as np
from torch.utils.data import Dataset


class MnistDataset(Dataset):
    def __init__(self, length, partition, path):
        self.partition = partition
        imgs = np.load(path + '.npy')
        labels = np.load(path + '_label.npy')
        
        train_size = int(imgs.shape[0] * 0.8)
        
        if self.partition == 'train':
            self.imgs = imgs[:train_size]
            self.labels = labels[:train_size]
        else:
            self.imgs = imgs[train_size:]
            self.labels = labels[train_size:]
            
        self.points = self._calc_point(self.labels)
        
        self.length = length
        self.full_length = self.imgs.shape[1]
        
        self.imgs = self.imgs.reshape(-1, self.length, 1, 32, 32)
        self.labels = self.labels.reshape(-1, self.length, 1, 1)
        self.points = self.points.reshape(-1, self.length, 1, 1)
        
    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, index):
        imgs = self.imgs[index].astype(np.float32)
        labels = self.labels[index].astype(np.float32)
        points = self.points[index].astype(np.float32)
        data = {'img': imgs, 'label': labels, 'point': points}
        return data

    def _calc_point(self, arr):
        arr_ = np.insert(arr, len(arr), 10)
        arr = np.insert(arr, 0, 10)
        res = np.where(arr - arr_ != 0, 1, 0)
        res[0] = 0
        res = res[:-1]
        return res