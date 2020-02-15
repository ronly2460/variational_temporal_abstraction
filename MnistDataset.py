import numpy as np
from torch.utils.data import Dataset

class MnistDataset(Dataset):
    def __init__(self, length, partition, path):
        self.partition = partition
        dataset = np.load(path)
        num_seqs = int(dataset.shape[0] * 0.8)
        if self.partition == 'train':
            self.state = dataset[:num_seqs]
        else:
            self.state = dataset[num_seqs:]
        self.state = self.state.reshape(-1, 100, 1, 32, 32)

        self.length = length
        self.full_length = self.state.shape[1]

    def __len__(self):
        return self.state.shape[0]

    def __getitem__(self, index):
        idx0 = np.random.randint(0, self.full_length - self.length)
        idx1 = idx0 + self.length

        state = self.state[index, idx0:idx1].astype(np.float32)
        return state