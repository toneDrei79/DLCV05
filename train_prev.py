
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
from models import * 

from sklearn.model_selection import train_test_split


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='./Flower/', help='directry path of the dataset')
    parser.add_argument('--batch', type=int, default=256, help='batch size')
    parser.add_argument('--epoch', type=int, default=10, help='number of epoch')
    return parser.parse_args()



def accuracy(preds, labals):
    _preds = torch.argmax(preds, dim=1)
    return np.count_nonzero(_preds.cpu()==labals.cpu()) / _preds.shape[0]



def train(model, loader, criterion):
    model.train()

    loss_total = 0.0
    acc_total = 0.0
    n_train = 0
    for i, (images, labels) in enumerate(tqdm(loader, desc='training')):
        images, labels = images.to(model.device), labels.to(model.device)
        model.optimizer.zero_grad()
        preds = model(images)
        loss = criterion(preds, labels)
        acc = accuracy(preds, labels)
        loss.backward()
        model.optimizer.step()

        loss_total += loss.item()
        acc_total += acc
        n_train += len(labels)

    return loss_total/n_train, acc_total/(i+1)



def test(model, loader, criterion):
    model.eval()

    loss_total = 0.0
    acc_total = 0.0
    n_test = 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(tqdm(loader, desc='testing')):
            images, labels = images.to(model.device), labels.to(model.device)
            preds = model(images)
            loss = criterion(preds, labels)
            acc = accuracy(preds, labels)
            
            loss_total += loss.item()
            acc_total += acc
            n_test += len(labels)
    
    return loss_total/n_test, acc_total/(i+1)



if __name__ == '__main__':
    args = get_args()
    path_dataset = args.data
    batch = args.batch
    epoch = args.epoch

    model = Net(n1=32, n2=128, n3=512)
    model.to(model.device)

    data_train = torchvision.datasets.MNIST(root=path_dataset,
                                            train=True,
                                            transform=model.transforms,
                                            download = False)
    data_test = torchvision.datasets.MNIST(root=path_dataset,
                                           train=False,
                                           transform=model.transforms,
                                           download = False)

    loader_train = torch.utils.data.DataLoader(dataset=data_train,
                                               batch_size=batch,
                                               shuffle=True)
    loader_test = torch.utils.data.DataLoader(dataset=data_test,
                                              batch_size=batch,
                                              shuffle=True)

    criterion = nn.CrossEntropyLoss()
    for e in range(epoch):
        loss_train, acc_train = train(model, loader_train, criterion)
        print('[epoch %2d] loss: %.6f acc: %.6f' %(e, loss_train, acc_train))
    loss_test, acc_test = test(model, loader_test, criterion)
    print('[test] loss : %.6f acc: %.6f' %(loss_test, acc_test))