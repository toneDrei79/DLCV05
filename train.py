
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import argparse
from collections import OrderedDict
from tqdm import tqdm
from models import *
from dataset import ImageDataset


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='./Flowers/Train', help='directry path of the dataset')
    parser.add_argument('--batch', type=int, default=32, help='batch size')
    parser.add_argument('--epoch', type=int, default=10, help='number of epoch')
    return parser.parse_args()


def train(model, dataloader, criterion, optimizer):
    model.train()
    with tqdm(dataloader, total=len(dataloader)) as progress:
        progress.set_description(f'train {epoch:3d}')
        for images, labels in progress:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            preds = model(images)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()

            progress.set_postfix(OrderedDict(loss=f'{loss.item():6.4f}'))
    return 

def val(model, dataloader, criterion):
    model.eval()
    with torch.no_grad():
        with tqdm(dataloader, total=len(dataloader)) as progress:
            progress.set_description(f'val   {epoch:3d}')
            for images, labels in progress:
                images, labels = images.to(device), labels.to(device)
                preds = model(images)
                loss = criterion(preds, labels)

                progress.set_postfix(OrderedDict(loss=f'{loss.item():6.4f}'))
    return


if __name__ == '__main__':
    args = get_args()

    # model = Net(n1=8, n2=8, n3=16, n4=16, n_class=10, size_image=256)
    model = Net()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # transform = transforms.Compose([transforms.Resize((256,256)),
    #                                 transforms.ToTensor()])
    transform = transforms.Compose([transforms.Resize((512,512)),
                                    transforms.ToTensor()])
    dataset = ImageDataset(path=args.data, transform=transform)
    train_dataset, val_dataset = train_test_split(dataset, test_size=0.2, shuffle=True)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), weight_decay=0.01, lr=1e-4)

    loss_total = 0.0
    acc_total = 0.0
    for epoch in range(args.epoch):
        train(model, train_dataloader, criterion, optimizer)
        val(model, val_dataloader, criterion)