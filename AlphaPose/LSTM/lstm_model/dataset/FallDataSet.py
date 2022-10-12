import torch
from torch.utils.data import Dataset
import pandas as pd


class FallDataset(Dataset):
    def __init__(self, data_dir):
        df = pd.read_csv(data_dir)
        self.inp = df.iloc[:, 0:34]
        self.outp = df.iloc[:, 34:]

    def __getitem__(self, index):
        inp = torch.FloatTensor(self.inp.loc[index])
        outp = torch.LongTensor(self.outp.loc[index])  # target : LongTensor (index)
        return inp, outp

    def __len__(self):
        return len(self.inp)
        # return self.x_data.shape[0]