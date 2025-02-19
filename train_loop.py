# Data loader: https://www.youtube.com/watch?v=iYisBtT6zvs
# Training loop: https://www.youtube.com/watch?v=NVxCKdp0NhQ


import pandas as pd
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
from torch import nn
from torch.nn import functional as F
from torchmetrics import Accuracy
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn import preprocessing
from enum import Enum
import pandas as pd
import copy
from pytorch_lightning import LightningModule, Trainer
from torch import nn
import torch
from torch.nn import functional as F
from torchmetrics import Accuracy

import torchvision
from torch.utils.data import Dataset, DataLoader
import os
from darknet import Darknet
from dataset_maker import CustomDataset
import argparse
from pytorch_lightning import LightningModule, Trainer
from torch import nn
import torch
from torch.nn import functional as F
from torchmetrics import Accuracy

# TODO cleanup imports


def arg_parse():
    parser = argparse.ArgumentParser(description='YOLO v3 Modified Implementation')
    parser.add_argument("--train", default=True, action='store_true', help="Enable training mode")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")  # 50
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--bs", type=int, default=16, help="Batch size")
    # parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    parser.add_argument("--cfg", default="cfg/yolov3.cfg", type=str, help="YOLO config file")
    parser.add_argument("--weights", default=None, type=str, help="Path to weights (for resuming training)")
    parser.add_argument("--data", default="data/coco.data", type=str, help="Path to dataset config")  # TODO modify this to replace config below
    return parser.parse_args()


if __name__ == '__main__':
    # Parse arguments
    args = arg_parse()

    # Load dataset
    full_path = r"D:/src/YOLO_v3_tutorial_from_scratch/data/feb12/"
    config = {
        "test": f"{full_path}test.csv",
        "train": f"{full_path}train.csv",
        "val": f"{full_path}val.csv"
    }
    train_dataset = CustomDataset(config=config, train=args.train)
    train_dataloader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True, num_workers=2)

    # train = copy.deepcopy(dataset).set_fold(DatasetType.TRAIN)
    # test = copy.deepcopy(dataset).set_fold(DatasetType.TEST)
    # val = copy.deepcopy(dataset).set_fold(DatasetType.VAL)

    # Start the Trainer
    trainer = Trainer(
        max_epochs=args.epochs,
        enable_progress_bar=True,
    )

    # Load model
    print("Loading network...")
    model = Darknet(args.cfg)  # TODO send to GPU device
    if args.weights:
        model.load_weights(args.weights)
    print("Network successfully loaded")
    # model = SimpleModel(train, test, val)
    # Train the Model
    trainer.fit(model)
    # Test on the Test SET, it will print validation
    trainer.test()

print("Done")