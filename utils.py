import numpy as np


def split_dataset(train_dataset):
    dataset_size = train_dataset.__len__()
    idx = list(range(dataset_size))
    np.random.shuffle(idx)
    split = int(np.floor(dataset_size * 0.2))

    return idx[split:], idx[:split]


def sec2h(end, start):
    return ((end - start) / 60.0) / 60.0
