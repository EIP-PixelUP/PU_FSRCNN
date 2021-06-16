#!/usr/bin/env python3

import torch
from torch import nn
from torch.utils.data import DataLoader
import h5py
from tqdm import tqdm
from pathlib import Path

from datasets import TrainDataset, TestDataset
from model import FSRCNN
from config import model_settings, batch_size, learning_rate, epochs

device = "cuda" if torch.cuda.is_available() else "cpu"

with h5py.File("datasets/General-100.h5") as f:

    outdir = Path("out")
    outdir.mkdir(exist_ok=True)

    # Create data loaders.
    train_dataloader = DataLoader(
        TrainDataset(f["train"]), batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(TestDataset(f["test"]), batch_size=batch_size)

    # Create the model
    model = FSRCNN(**model_settings).to(device)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    def train(dataloader, model, loss_fn, optimizer):
        size = len(dataloader.dataset)
        for batch, (X, y) in enumerate(tqdm(dataloader, total=size // batch_size)):
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

    def test(dataloader, model):
        size = len(dataloader.dataset)
        model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= size
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        torch.save(model.state_dict(), outdir / f"epoch_{t}.pth")
        test(test_dataloader, model)
    torch.save(model.state_dict(), outdir / "result.pth")
    print("Done!")
