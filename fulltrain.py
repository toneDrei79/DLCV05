from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms
import numpy as np
from collections import OrderedDict
from tqdm import tqdm
import argparse
from utils import *
from models import *


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--traindata', type=str, default='./Flowers/Train/', help='directry path of the train dataset ... default=./Flowers/Train/')
    parser.add_argument('--valdata', type=str, default='./Flowers/Test/', help='directry path of the val dataset ... default=./Flowers/Test/')
    parser.add_argument('--model', type=str, default='net6', help='available models: net3, net5, net6, net7, net11, vgg11, vgg16, resnet18 ... default=net6')
    parser.add_argument('--dropout', action='store_true', help='whether do dropout')
    parser.add_argument('--batchnorm', action='store_true', help='whether do batchnorm')
    parser.add_argument('--pretrained', action='store_true', help='use pretrained model (vgg, resnet)')
    parser.add_argument('--input_size', type=int, default=128, help='possible sizes: 128, 224(recommended for vgg, resnet), 256 ... default=128')
    parser.add_argument('--epoch', type=int, default=50, help='number of epoch ... default=50')
    parser.add_argument('--batch', type=int, default=32, help='batch size ... default=32')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate ... default=1e-4')
    parser.add_argument('--lr_scheduler', action='store_true', help='use exponential lr scheduler')
    parser.add_argument('--lr_gamma', type=float, default=0.95, help='gamma of exponential lr scheduler ... default=0.95')
    parser.add_argument('--augment', action='store_true', help='do data augmentation')
    parser.add_argument('--aug_rotate', type=int, default=15, help='rotation degrees of data augmentation (0~45) ... default=15')
    parser.add_argument('--aug_color', type=float, default=0.1, help='color changing range of data augmentation ... default=0.1')
    parser.add_argument('--save_interval', type=int, default=10, help='interval for saving model ... default=10')
    parser.add_argument('--save', type=str, default=None, help='save dir path for trained models ... default=None')
    return parser.parse_args()


def accuracy(preds, labals):
    _preds = torch.argmax(preds.cpu(), dim=1)
    return np.count_nonzero(_preds==labals) / _preds.shape[0]


def train(model, dataloader, criterion, optimizer, epoch, device):
    model.train()

    sum_loss, sum_acc = 0, 0
    with tqdm(dataloader, total=len(dataloader)) as progress:
        progress.set_description(f'train {epoch:3d}')
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


def val(model, dataloader, criterion, epoch, device):
    model.eval()

    sum_loss, sum_acc = 0, 0
    with torch.no_grad():
        with tqdm(dataloader, total=len(dataloader)) as progress:
            progress.set_description(f'val   {epoch:3d}')
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
    criterion = nn.CrossEntropyLoss()

    image_size = args.input_size
    margin_size = int(image_size * (np.sin(np.deg2rad(args.aug_rotate)) + np.cos(np.deg2rad(args.aug_rotate))))
    transform_aug = transforms.Compose([transforms.Resize((margin_size,margin_size)),
                                        transforms.RandomRotation(args.aug_rotate),
                                        transforms.RandomHorizontalFlip(p=0.5),
                                        transforms.CenterCrop((image_size,image_size)),
                                        transforms.ColorJitter(brightness=args.aug_color, contrast=args.aug_color, saturation=args.aug_color, hue=args.aug_color*0.5),
                                        transforms.ToTensor()])
    transform_basic = transforms.Compose([transforms.Resize((image_size,image_size)),
                                          transforms.ToTensor()])
    if args.augment:
        train_dataset = datasets.ImageFolder(root=args.traindata, transform=transform_aug)
    else:
        train_dataset = datasets.ImageFolder(root=args.traindata, transform=transform_basic)
    val_dataset = datasets.ImageFolder(root=args.valdata, transform=transform_basic)

    checkpoint_dir = mkdir_checkpoint(args.save)
    save_configs(args, checkpoint_dir)
    logger = Logger(checkpoint_dir)

    model = select_model(args.model, args.dropout, args.batchnorm, args.pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), weight_decay=0.01, lr=args.lr)
    if args.lr_scheduler:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_gamma)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch, shuffle=True)
    for e in range(args.epoch):
        train_loss, train_acc = train(model, train_dataloader, criterion, optimizer, e, device)
        val_loss, val_acc = val(model, val_dataloader, criterion, e, device)
        if args.lr_scheduler:
            scheduler.step()
        logger.log(train_loss, val_loss, train_acc, val_acc, e)
        if (e+1) % args.save_interval == 0:
            save_model(model, e, None, checkpoint_dir)