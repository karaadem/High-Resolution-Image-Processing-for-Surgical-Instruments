from torch.utils.data import DataLoader
from train import train, evaluate
from torchvision import models
from dataset import OP
import torch.nn as nn
from utils import *
import argparse
import torch
import json
import time
import os


def instruments_classification(args) -> None:
    data_path = args.data_path
    model_path = args.model_path
    logs_path = args.logs_path

    model_name = args.model_name
    image_size = args.image_size

    device = args.device
    num_classes = args.num_classes
    epoch = args.epochs
    learning_rate = args.lr
    batch_size = args.batch_size
    num_workers = args.num_workers

    # initialize model
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).to(device)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    if args.multi_gpu and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    train_dataset = OP(train=True, path=data_path, image_size=image_size)
    test_dataset = OP(train=False, path=data_path, image_size=image_size)

    train_idx, test_idx = split_dataset(train_dataset=train_dataset)

    train_dataloader = DataLoader(dataset=train_dataset,
                                  sampler=train_idx,
                                  batch_size=batch_size,
                                  num_workers=num_workers)
    test_dataloader = DataLoader(dataset=test_dataset,
                                 sampler=test_idx,
                                 batch_size=batch_size,
                                 num_workers=num_workers)

    print('Model: {}'.format(model_name))
    print('Length of the training data set: {}.'.format(len(train_idx)))
    print('Length of the test data set: {}.'.format(len(test_idx)))

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    logs = {}
    train_losses = []
    valid_losses = []
    train_accuracies = []
    valid_accuracies = []
    best_acc = 0

    print('start training')
    for e in range(epoch):
        print('__________________________________________________')
        print('Epoch {}/{}'.format(e, epoch - 1))

        model.train()
        train_loss, train_accuracy = train(model, train_dataloader, optimizer, criterion, device)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        print('train loss: {}'.format(train_losses[-1]))
        print('train accuracy: {} %'.format(train_accuracy))

        print("evaluate")
        model.eval()
        with torch.no_grad():
            valid_loss, valid_accuracy = evaluate(model, test_dataloader, criterion, device)
            valid_losses.append(valid_loss)
            valid_accuracies.append(valid_accuracy)

            print('valid loss: {}'.format(valid_losses[-1]))
            print('valid accuracy: {} %'.format(valid_accuracy))

        logs['train_losses'] = train_losses
        logs['valid_losses'] = valid_losses
        logs['train_accuracies'] = train_accuracies
        logs['valid_accuracies'] = valid_accuracies

        with open(logs_path, 'w') as outfile:
            json.dump(logs, outfile)

        if valid_accuracy > best_acc:
            best_acc = valid_accuracy
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optim_state_dict': optimizer.state_dict()
            }
            torch.save(checkpoint, model_path)
            print(f"{e}: Epoch completed and model successfully saved!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--multi_gpu', type=bool, default=True)
    parser.add_argument('--image_size', type=tuple, default=(2560, 2048))
    parser.add_argument('--num_classes', type=int, default=76)

    parser.add_argument('--artifact', type=str, default='with')
    parser.add_argument('--path', type=str, default='/home/karaadem/git/op-instruments-classifier/')
    parser.add_argument('--data_path', type=str,
                        default='/home/karaadem/git/dataset/{}_artifacts/'.format(
                            parser.get_default('artifact')))
    parser.add_argument('--model_name', type=str,
                        default='classifier_resnet18_{}_{}_artifacts'.format(
                            'x'.join(str(x) for x in parser.get_default('image_size')), parser.get_default('artifact')))
    parser.add_argument('--model_path', type=str,
                        default=os.path.join(parser.get_default('path'), 'model',
                                             parser.get_default('model_name') + '.ckpt'))
    parser.add_argument('--logs_path', type=str,
                        default=os.path.join(parser.get_default('path'), 'logs',
                                             'logs_' + parser.get_default('model_name') + '.json'))

    args = parser.parse_args()
    args.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(''.join(f'{k}={v}\n' for k, v in vars(args).items()))

    start = time.time()
    instruments_classification(args=args)
    end = time.time()

    print('Time estimate: {} h.'.format(sec2h(end, start)))
