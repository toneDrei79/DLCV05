
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from sklearn.model_selection import KFold
from torchvision import datasets
import torchvision.transforms as transforms
import numpy as np
from collections import OrderedDict
from tqdm import tqdm
from utils import *
from models import *


def accuracy(preds, labals):
    _preds = torch.argmax(preds.cpu(), dim=1)
    return np.count_nonzero(_preds==labals) / _preds.shape[0]


def train(model, dataloader, criterion, optimizer):
    model.train()

    sum_loss, sum_acc = 0, 0
    with tqdm(dataloader, total=len(dataloader)) as progress:
        progress.set_description(f'train {e:3d}')
        for i, (images, labels) in enumerate(progress):
            optimizer.zero_grad()
            preds = model(images.to(device))
            loss = criterion(preds, labels.to(device))
            loss.backward()
            optimizer.step()

            sum_loss += loss.item()
            acc = accuracy(preds, labels)
            sum_acc += acc
            progress.set_postfix(OrderedDict(loss=f'{loss.item():5.3f}', acc=f'{acc:5.3f}'))
        
    return sum_loss/(i+1), sum_acc/(i+1)


def val(model, dataloader, criterion):
    model.eval()

    sum_loss, sum_acc = 0, 0
    with torch.no_grad():
        with tqdm(dataloader, total=len(dataloader)) as progress:
            progress.set_description(f'val   {e:3d}')
            for i, (images, labels) in enumerate(progress):
                preds = model(images.to(device))
                loss = criterion(preds, labels.to(device))

                sum_loss += loss.item()
                acc = accuracy(preds, labels)
                sum_acc += acc
                progress.set_postfix(OrderedDict(loss=f'{loss.item():5.3f}', acc=f'{acc:5.3f}'))
            
    return sum_loss/(i+1), sum_acc/(i+1)


if __name__ == '__main__':
    configs = Configs()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.CrossEntropyLoss()
    image_size = configs.input_size
    transform_aug = transforms.Compose([transforms.Resize((int(image_size*1.2),int(image_size*1.2))),
                                        transforms.RandomRotation(degrees=15),
                                        transforms.RandomCrop((image_size,image_size)),
                                        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
                                        transforms.ToTensor()])
    transform_basic = transforms.Compose([transforms.Resize((image_size,image_size)),
                                          transforms.ToTensor()])
    if configs.augment:
        _train_dataset = datasets.ImageFolder(root=configs.data, transform=transform_aug)
    else:
        _train_dataset = datasets.ImageFolder(root=configs.data, transform=transform_basic)
    _val_dataset = datasets.ImageFolder(root=configs.data, transform=transform_basic)

    checkpoint_dir = mkdir_checkpoint()
    configs.save_configs(checkpoint_dir)
    logger = Logger()

    kf = KFold(n_splits=configs.k, shuffle=True, random_state=0)
    sum_train_loss, sum_train_acc, sum_val_loss, sum_val_acc = np.zeros(configs.epoch), np.zeros(configs.epoch), np.zeros(configs.epoch), np.zeros(configs.epoch)
    for k, (train_idxes, val_idxes) in enumerate(kf.split(_train_dataset)):
        print(f'cross-validation {k:2d}:')
        model = select_model(configs.model).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), weight_decay=0.01, lr=configs.lr)
        train_dataset = Subset(_train_dataset, train_idxes)
        val_dataset = Subset(_val_dataset, val_idxes)
        train_dataloader = DataLoader(train_dataset, batch_size=configs.batch, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=configs.batch, shuffle=True)

        for e in range(configs.epoch):
            train_loss, train_acc = train(model, train_dataloader, criterion, optimizer)
            val_loss, val_acc = val(model, val_dataloader, criterion)

            sum_train_loss[e] += train_loss
            sum_train_acc[e] += train_acc
            sum_val_loss[e] += val_loss
            sum_val_acc[e] += val_acc
            logger.log(sum_train_loss[e]/(k+1), sum_val_loss[e]/(k+1), sum_train_acc[e]/(k+1), sum_val_acc[e]/(k+1), e)

            if k == 0 and (e+1) % configs.save_interval == 0:
                save_model(model, e, checkpoint_dir)