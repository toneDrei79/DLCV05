
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
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
    configs = Configs()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.CrossEntropyLoss()

    image_size = configs.input_size
    rotate_degree = 15
    margin_size2 = int(image_size * 1.2)
    margin_size1 = int(margin_size2 * (np.sin(np.deg2rad(rotate_degree)) + np.cos(np.deg2rad(rotate_degree))))
    transform_aug = transforms.Compose([transforms.Resize((margin_size1,margin_size1)),
                                        transforms.RandomRotation(rotate_degree),
                                        transforms.CenterCrop((margin_size2,margin_size2)),
                                        transforms.RandomCrop((image_size,image_size)),
                                        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
                                        transforms.ToTensor()])
    transform_basic = transforms.Compose([transforms.Resize((image_size,image_size)),
                                          transforms.ToTensor()])
    if configs.augment:
        train_dataset = datasets.ImageFolder(root=configs.data, transform=transform_aug)
    else:
        train_dataset = datasets.ImageFolder(root=configs.data, transform=transform_basic)
    val_dataset = datasets.ImageFolder(root=Path(Path(configs.data).parent, 'Test'), transform=transform_basic)

    checkpoint_dir = mkdir_checkpoint()
    configs.save_configs(checkpoint_dir)
    logger = Logger()

    model = select_model(configs.model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), weight_decay=0.01, lr=configs.lr)
    train_dataloader = DataLoader(train_dataset, batch_size=configs.batch, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=configs.batch, shuffle=True)
    for e in range(configs.epoch):
        train_loss, train_acc = train(model, train_dataloader, criterion, optimizer, e, device)
        val_loss, val_acc = val(model, val_dataloader, criterion, e, device)
        logger.log(train_loss, val_loss, train_acc, val_acc, e)
        if (e+1) % configs.save_interval == 0:
            save_model(model, e, checkpoint_dir)