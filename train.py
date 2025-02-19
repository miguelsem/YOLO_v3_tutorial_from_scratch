import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import argparse
import os
from darknet import Darknet
from util import *

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import numpy as np


def arg_parse():
    parser = argparse.ArgumentParser(description='YOLO v3 Training & Detection Module')
    parser.add_argument("--train", default=True, action='store_true', help="Enable training mode")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--bs", type=int, default=16, help="Batch size")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    parser.add_argument("--cfg", default="cfg/yolov3.cfg", type=str, help="YOLO config file")
    parser.add_argument("--weights", default=None, type=str, help="Path to weights (for resuming training)")
    parser.add_argument("--data", default="data/coco.data", type=str, help="Path to dataset config")
    return parser.parse_args()


class LightPointDataset(Dataset):
    def __init__(self, img_dir, label_dir, img_size=416):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_size = img_size
        self.img_files = sorted(os.listdir(img_dir))
        self.label_files = sorted(os.listdir(label_dir))

        self.transform = transforms.Compose([
            transforms.ToTensor(),  # Converts to tensor and scales [0,255] -> [0,1]
        ])

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        label_path = os.path.join(self.label_dir, self.label_files[idx])

        # Load and preprocess the image (grayscale)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (self.img_size, self.img_size))  # Resize to YOLO input size
        img = self.transform(img)  # Convert to tensor

        # Load labels
        labels = []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    x_center, y_center, width, height = map(float, line.strip().split())
                    labels.append([0, x_center, y_center, width, height])  # Class 0 (single class)

        labels = torch.tensor(labels, dtype=torch.float32)
        return img, labels


def get_dataloader(img_dir, label_dir, batch_size=16, img_size=416):
    dataset = LightPointDataset(img_dir, label_dir, img_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)
    return dataloader


def collate_fn(batch):
    imgs, targets = zip(*batch)
    imgs = torch.stack(imgs)
    return imgs, targets


if __name__ == '__main__':
    # Parse arguments
    args = arg_parse()

    # Load model
    print("Loading network...")
    model = Darknet(args.cfg)
    if args.weights:
        model.load_weights(args.weights)
    print("Network successfully loaded")

    # If there's a GPU, use it
    CUDA = torch.cuda.is_available()
    if CUDA:
        model.cuda()

    if args.train:
        # Training setup
        model.train()

        # Define loss function (YOLO loss would be implemented separately)
        criterion = nn.MSELoss()  # Placeholder, YOLO has a custom loss function

        # Optimizer
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=5e-4)

        # Load dataset (assuming a DataLoader is implemented)
        img_dir = r"D:\src\YOLO_v3_tutorial_from_scratch\data\feb12\images\train"
        label_dir = r"D:\src\YOLO_v3_tutorial_from_scratch\data\feb12\labels\train"
        train_loader = get_dataloader(img_dir, label_dir, batch_size=args.bs)  # Implement a function to load dataset

        print("Starting training...")
        for epoch in range(args.epochs):
            total_loss = 0
            for i, (images, targets) in enumerate(train_loader):
                if CUDA:
                    images, targets = images.cuda(), targets.cuda()

                optimizer.zero_grad()
                predictions = model(images)
                loss = criterion(predictions, targets)  # Custom YOLO loss should be used
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            print(f"Epoch [{epoch + 1}/{args.epochs}], Loss: {total_loss / len(train_loader)}")

            # Save model periodically
            if (epoch + 1) % 10 == 0:
                model.save_weights(f"yolov3_epoch{epoch + 1}.weights")

        print("Training finished. Model saved.")
    else:
        # Run inference as in the original script
        model.eval()
        print("Running inference...")
