
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import numpy as np
import argparse
from collections import OrderedDict
from tqdm import tqdm
from datetime import datetime
from models import *
# from dataset import ImageDataset

from torch.utils.tensorboard import SummaryWriter


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='./Flowers/Train', help='directry path of the dataset')
    parser.add_argument('--model', type=str, default='net8', help='available models: net8, net11, vgg11, vgg11trained, vgg16, vgg16trained')
    parser.add_argument('--epoch', type=int, default=10, help='number of epoch')
    parser.add_argument('--batch', type=int, default=32, help='batch size')
    parser.add_argument('--lr', type=int, default=10, help='learning rate')
    return parser.parse_args()


def accuracy(preds, labals):
    _preds = torch.argmax(preds, dim=1)
    return np.count_nonzero(_preds.cpu()==labals.cpu()) / _preds.shape[0]


def train(model, dataloader, criterion, optimizer):
    model.train()

    total_loss = 0
    total_acc = 0
    with tqdm(dataloader, total=len(dataloader)) as progress:
        progress.set_description(f'train {epoch:3d}')
        for i, (images, labels) in enumerate(progress):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            preds = model(images)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            acc = accuracy(preds, labels)
            total_acc += acc
            progress.set_postfix(OrderedDict(loss=f'{loss.item():5.3f}', acc=f'{acc:5.3f}'))
        
        # print(f'train {epoch:3d}: loss={total_loss/(i+1):7.5f} acc={total_acc/(i+1):7.5f}')
    return total_loss/(i+1), total_acc/(i+1)


def val(model, dataloader, criterion):
    model.eval()

    total_loss = 0
    total_acc = 0
    with torch.no_grad():
        with tqdm(dataloader, total=len(dataloader)) as progress:
            progress.set_description(f'val   {epoch:3d}')
            for i, (images, labels) in enumerate(progress):
                images, labels = images.to(device), labels.to(device)
                preds = model(images)
                loss = criterion(preds, labels)

                total_loss += loss.item()
                acc = accuracy(preds, labels)
                total_acc += acc
                progress.set_postfix(OrderedDict(loss=f'{loss.item():5.3f}', acc=f'{acc:5.3f}'))
            
            # print(f'val   {epoch:3d}: loss={total_loss/(i+1):7.5f} acc={total_acc/(i+1):7.5f}')
    return total_loss/(i+1), total_acc/(i+1)


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

    transform = transforms.Compose([transforms.Resize((int(image_size*1.2),int(image_size*1.2))),
                                    transforms.RandomRotation(degrees=45),
                                    transforms.RandomCrop((image_size,image_size)),
                                    transforms.ColorJitter(brightness=0.3, contrast=0.5, saturation=0.2, hue=0.1),
                                    transforms.ToTensor()])
    # dataset = ImageDataset(path=args.data, transform=transform)
    dataset = datasets.ImageFolder(root=args.data, transform=transform)
    train_dataset, val_dataset = train_test_split(dataset, test_size=0.2, shuffle=True)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), weight_decay=0.01, lr=1e-4)

    log = SummaryWriter(log_dir='.logs/{}/'.format(datetime.utcnow().strftime('%Y%m%d%H%M%S')))
    for epoch in range(args.epoch):
        train_loss, train_acc = train(model, train_dataloader, criterion, optimizer)
        val_loss, val_acc = val(model, val_dataloader, criterion)
        log.add_scalars('loss', {'train': train_loss, 'val': val_loss}, epoch)
        log.add_scalars('acc', {'train': train_acc, 'val': val_acc}, epoch)