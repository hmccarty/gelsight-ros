#!/usr/bin/env python3

"""
Trains a depth reconstruction model using the dataset created by
the 'label_data.py' script.
"""

import csv
from csv import writer
import cv2
import gelsight_ros as gsr
import math
import numpy as np
import os
import rospy
from datetime import datetime as dt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Default parameters
DEFAULT_BATCH_SIZE = 64
DEFAULT_TRAIN_SPLIT = 0.8
DEFAULT_LR = 1e-3
DEFAULT_EPOCHS = 10

# Global parameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
    test_loss /= num_batches
    print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")

def get_param_or_err(name: str):
    if not rospy.has_param(f"~{name}"):
        rospy.signal_shutdown(f"Required parameter missing: {name}")
    return rospy.get_param(f"~{name}")

if __name__ == "__main__":
    rospy.init_node("train")

    # Retrieve path to dataset
    input_path = get_param_or_err("input_path")

    # Retrieve path where dataset will be saved
    output_path = get_param_or_err("output_path")
    if output_path[-1] == "/":
        output_path = output_path[:len(output_path)-1]

    if not os.path.exists(output_path):
        rospy.logwarn("Output folder doesn't exist, will create it.")
        os.makedirs(output_path)
        
        if not os.path.exists(output_path):
            rospy.signal_shutdown(f"Failed to create output folder: {output_path}")
    output_file = output_path + f"/model-{dt.now().strftime("%H-%M-%S")}.pth"

    # Create dataset
    dataset = gsr.GelsightDepthDataset(input_path)

    train_size = int(len(dataset) * DEFAULT_TRAIN_SPLIT)
    test_size = len(dataset) - train_size
    trainset, testset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_dataloader = DataLoader(trainset, batch_size=DEFAULT_BATCH_SIZE)
    test_dataloader = DataLoader(testset, batch_size=DEFAULT_BATCH_SIZE)

    # Initiate model and optimizer
    model = gsr.RGB2Grad().to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=DEFAULT_LR)

    # Train model
    epochs = DEFAULT_EPOCHS
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)
    print("Done!")

    torch.save(model.state_dict(), output_file)