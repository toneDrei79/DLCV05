from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms
import numpy as np
from collections import OrderedDict
from tqdm import tqdm
import json
import pickle

from models import *


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='./Flowers/Test/', help='directry path of the dataset ... default=./Flowers/Test/')
    parser.add_argument('--model', type=str, help='path of the trained model (.pkl)')
    parser.add_argument('--configs', type=str, help='path of the configs.json')
    return parser.parse_args()


def accuracy(preds, labals):
    _preds = torch.argmax(preds.cpu(), dim=1)
    return np.count_nonzero(_preds==labals) / _preds.shape[0]


def test(model, dataloader, criterion, device):
    model.eval()

    sum_loss, sum_acc = 0, 0
    with torch.no_grad():
        with tqdm(dataloader, total=len(dataloader)) as progress:
            progress.set_description(f'test')
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
    with open(args.model, mode='rb') as f:
        model = pickle.load(f).to(device)
    
    with open(args.configs, mode='r') as f:
        jsondata = json.load(f)
        image_size = jsondata['input_size']
        batch = jsondata['batch']
    transform = transforms.Compose([transforms.Resize((image_size,image_size)),
                                    transforms.ToTensor()])
    dataset = datasets.ImageFolder(root=args.data, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch, shuffle=True)

    loss, acc = test(model, dataloader, criterion, device)
    print(f'loss: {loss:7.5f}')
    print(f'acc : {acc:7.5f}')