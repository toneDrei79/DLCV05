from pathlib import Path
import json
import os
from datetime import datetime
import pickle
from torch.utils.tensorboard import SummaryWriter


class Logger:

    def __init__(self):
        datetime_str = datetime.utcnow().strftime('%Y%m%d%H%M%S')
        self.__summarywriter__ = SummaryWriter(log_dir=f'./.logs/{datetime_str}/')
    
    def log(self, train_loss, val_loss, train_acc, val_acc, epoch):
        self.__summarywriter__.add_scalars('loss', {'train': train_loss, 'val': val_loss}, epoch)
        self.__summarywriter__.add_scalars('acc', {'train': train_acc, 'val': val_acc}, epoch)


def mkdir_checkpoint(dir):
    checkpoints = './checkpoints/'
    if dir == None:
        datetime_str = datetime.utcnow().strftime('%Y%m%d%H%M%S')
        dir = f'{datetime_str}/'
    os.makedirs(Path(checkpoints, dir), exist_ok=True)
    return dir


def save_configs(configs, dir):
    path = Path(dir, 'configs.json')
    with open(path, mode='w') as f:
        json.dump(vars(configs), f, indent=4)


def save_model(model, epoch, dir):
    path = Path(dir, f'{epoch:03d}.pkl')
    with open(path, mode='wb') as f:
        pickle.dump(model, f)