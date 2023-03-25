import argparse
from pathlib import Path
import json
import os
from datetime import datetime
import pickle
from torch.utils.tensorboard import SummaryWriter


class Configs:

    def __init__(self):
        args = self.get_args()
        self.data = args.data
        self.model = args.model
        self.input_size = args.input_size
        self.epoch = args.epoch
        self.k = args.k
        self.batch = args.batch
        self.lr = args.lr
        self.augment = args.augment
        self.save_interval = args.save_interval

    def get_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--data', type=str, default='./Flowers/Train/', help='directry path of the dataset ... default=./Flowers/Train/')
        parser.add_argument('--model', type=str, default='net8', help='available models: net8, net11, vgg11, vgg11trained, vgg16, vgg16trained, resnet18, resnet18trained ... default=net8')
        parser.add_argument('--input_size', type=int, default=128, help='possible sizes: 128, 224(recommended for vgg, resnet), 256 ... default=128')
        parser.add_argument('--epoch', type=int, default=50, help='number of epoch ... default=50')
        parser.add_argument('--k', type=int, default=5, help='number of k-fold split ... default=5')
        parser.add_argument('--batch', type=int, default=32, help='batch size ... default=32')
        parser.add_argument('--lr', type=float, default=1e-4, help='learning rate ... default=1e-4')
        parser.add_argument('--augment', type=bool, default=False, help='data augmentation ... default=False')
        parser.add_argument('--save_interval', type=int, default=10, help='interval for saving model ... default=10')
        return parser.parse_args()

    def save_configs(self, dir):
        path = Path(dir, 'configs.json')
        with open(path, mode='w') as f:
            json.dump(vars(self), f, indent=4)


class Logger:

    def __init__(self):
        datetime_str = datetime.utcnow().strftime('%Y%m%d%H%M%S')
        self.__summarywriter__ = SummaryWriter(log_dir=f'./.logs/{datetime_str}/')
    
    def log(self, train_loss, val_loss, train_acc, val_acc, epoch):
        self.__summarywriter__.add_scalars('loss', {'train': train_loss, 'val': val_loss}, epoch)
        self.__summarywriter__.add_scalars('acc', {'train': train_acc, 'val': val_acc}, epoch)


def mkdir_checkpoint():
    datetime_str = datetime.utcnow().strftime('%Y%m%d%H%M%S')
    dir = f'./checkpoints/{datetime_str}/'
    os.makedirs(dir, exist_ok=True)
    return dir


def save_model(model, epoch, dir):
    path = Path(dir, f'{epoch:03d}.pkl')
    with open(path, mode='wb') as f:
        pickle.dump(model, f)