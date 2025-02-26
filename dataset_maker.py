# Data loader: https://www.youtube.com/watch?v=iYisBtT6zvs

from enum import Enum
import pandas as pd
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader


def read_image(path):
    return torchvision.io.read_image(path) / 255


# Again create a Dataset but this time, do the split in train test val
class CustomDataset(Dataset):
    def __init__(self, config, train=True, transform=None):
        self.dataframe = pd.read_csv(config["train"] if train else pd.read_csv(config["val"]))
        self.transform = transform
        #
        # # load data and shuffle, befor splitting
        # self.df = pd.read_csv("Stars.csv").sample(frac=1, random_state=27)
        # train_split = 0.6
        # val_split = 0.8
        # self.df_labels = df[['Type']]
        # # drop non numeric columns, in real life do categorical encoding
        # self.df = df.drop(columns=['Type', 'Color', 'Spectral_Class'])
        # # split pointf.df
        # self.train, self.val, self.test = np.split(self.df, [int(train_split * len(self.df)), int(val_split * len(self.df))])
        # self.train_labels, self.val_labels, self.test_labels = np.split(self.df_labels, [int(train_split * len(self.df)), int(val_split * len(self.df))])
        # # do the feature scaling only on the train set!
        # self.scaler = preprocessing.StandardScaler().fit(self.train)
        # for data_split in [self.train, self.val, self.test]:
        #     data_split = self.scaler.transform(data_split)
        # # convet labels to 1 hot
        # return self.dataset[idx], self.labels[idx]

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {
            "image": read_image(self.dataframe.iloc[idx]["path"]),
            # "label": self.dataframe.iloc[idx]["label"]
            # "label": list(map(int, self.dataframe.iloc[idx]["label"].split(",")))
            "label": torch.tensor(list(map(int, self.dataframe.iloc[idx]["label"].split(","))), dtype=torch.long)
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

    # def set_fold(self, set_type):
    #     # Make sure to call this befor using the dataset
    #     if set_type == DatasetType.TRAIN:
    #         self.dataset, self.labels = self.train, self.train_labels
    #     if set_type == DatasetType.TEST:
    #         self.dataset, self.labels = self.test, self.test_labels
    #     if set_type == DatasetType.VAL:
    #         self.dataset, self.labels = self.val, self.val_labels
    #     # Convert the datasets and the labels to pytorch format
    #     # Also use the StdScaler on the training set
    #     self.dataset = torch.tensor(self.scaler.transform(self.dataset)).float()
    #     self.labels = torch.tensor(self.labels.to_numpy().reshape(-1)).long()
    #
    #     return self


if __name__ == '__main__':
    full_path = r"D:/src/YOLO_v3_tutorial_from_scratch/data/feb12/"
    config = {
        "test": f"{full_path}test.csv",
        "train": f"{full_path}train.csv",
        "val": f"{full_path}val.csv"
    }

    train_dataset = CustomDataset(config=config, train=True)
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
