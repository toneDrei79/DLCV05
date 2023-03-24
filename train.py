
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from sklearn.model_selection import KFold
from torchvision import datasets
import torchvision.transforms as transforms
import numpy as np
import argparse
from collections import OrderedDict
import random
from tqdm import tqdm
from datetime import datetime
from models import *

from torch.utils.tensorboard import SummaryWriter


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='./Flowers/Train', help='directry path of the dataset')
    parser.add_argument('--model', type=str, default='net8', help='available models: net8, net11, vgg11, vgg11trained, vgg16, vgg16trained')
    parser.add_argument('--epoch', type=int, default=10, help='number of epoch')
    parser.add_argument('--k', type=int, default=5, help='number of k-fold split')
    parser.add_argument('--batch', type=int, default=32, help='batch size')
    parser.add_argument('--lr', type=int, default=1e-4, help='learning rate')
    return parser.parse_args()


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
    args = get_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.model == 'net8':
        model = Net8(n1=8,n2=8,n3=16,n4=32).to(device)
        image_size = 256 # available: 224, 256, 512
    elif args.model == 'net11':
        model = Net11().to(device)
        image_size = 256 # available: 224, 256, 512
    elif args.model == 'vgg11':
        model = Vgg11(n_class=10, pretrained=False).to(device)
        image_size = 224 # available: 224
    elif args.model == 'vgg11trained':
        model = Vgg11(n_class=10, pretrained=True).to(device)
        image_size = 224 # available: 224
    elif args.model == 'vgg16':
        model = Vgg16(n_class=10, pretrained=False).to(device)
        image_size = 224 # available: 224
    elif args.model == 'vgg16trained':
        model = Vgg16(n_class=10, pretrained=True).to(device)
        image_size = 224 # available: 224
    else:
        print('Error: No such model.')

    train_transform = transforms.Compose([transforms.Resize((int(image_size*1.2),int(image_size*1.2))),
                                          transforms.RandomRotation(degrees=15),
                                          transforms.RandomCrop((image_size,image_size)),
                                          transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
                                          transforms.ToTensor()])
    val_transform = transforms.Compose([transforms.Resize((image_size,image_size)),
                                        transforms.ToTensor()])
    _train_dataset = datasets.ImageFolder(root=args.data, transform=train_transform)
    _val_dataset = datasets.ImageFolder(root=args.data, transform=val_transform)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), weight_decay=0.01, lr=args.lr)

    log = SummaryWriter(log_dir='.logs/{}/'.format(datetime.utcnow().strftime('%Y%m%d%H%M%S')))
    kf = KFold(n_splits=args.k, shuffle=True, random_state=0)
    for i, (train_idxes, val_idxes) in enumerate(kf.split(_train_dataset)):
        print(f'cross-validation {i:2d}')
        train_dataset = Subset(_train_dataset, train_idxes)
        val_dataset = Subset(_val_dataset, val_idxes)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch, shuffle=True)

        sum_train_loss, sum_train_acc, sum_val_loss, sum_val_acc = 0, 0, 0, 0
        for e in range(args.epoch):
            train_loss, train_acc = train(model, train_dataloader, criterion, optimizer)
            val_loss, val_acc = val(model, val_dataloader, criterion)
            sum_train_loss += train_loss
            sum_train_acc += train_acc
            sum_val_loss += val_loss
            sum_val_acc += val_acc
            log.add_scalars('loss', {'train': sum_train_loss/(e+1), 'val': sum_val_loss/(e+1)}, e)
            log.add_scalars('acc', {'train': sum_train_acc/(e+1), 'val': sum_val_acc/(e+1)}, e)