import numpy as np
import torch


def train(model, train_loader, optimizer, criterion, device):
    losses = []
    correct, total = 0, 0

    for img, target in train_loader:
        img = img.to(device)
        target = target.to(device)

        y = model(img)

        _, predicted = torch.max(y.data, 1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        loss = criterion(y, target)
        losses.append(float(loss.data))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    accuracy = (correct / total) * 100
    mean_loss = np.mean(losses)
    return mean_loss, accuracy


def evaluate(model, test_loader, criterion, device):
    losses = []
    correct, total = 0, 0

    for img, target in test_loader:
        img = img.to(device)
        target = target.to(device)

        y = model(img)

        _, predicted = torch.max(y.data, 1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        loss = criterion(y, target)
        losses.append(float(loss.data))

    accuracy = (correct / total) * 100
    mean_loss = np.mean(losses)
    return mean_loss, accuracy
