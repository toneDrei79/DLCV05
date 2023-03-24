import torch
import torch.nn as nn


class Net8(nn.Module):

    def __init__(self, n1=16, n2=32, n3=64, n4=128, n_class=10, size_image=256):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=n1, kernel_size=3, stride=1, padding=1), nn.ReLU(),
                                      nn.Conv2d(in_channels=n1, out_channels=n1, kernel_size=3, stride=1, padding=1), nn.ReLU(),
                                      nn.BatchNorm2d(n1),
                                      nn.MaxPool2d(kernel_size=4, stride=4),
                                      nn.Conv2d(in_channels=n1, out_channels=n2, kernel_size=3, stride=1, padding=1), nn.ReLU(),
                                      nn.Conv2d(in_channels=n2, out_channels=n2, kernel_size=3, stride=1, padding=1), nn.ReLU(),
                                      nn.BatchNorm2d(n2),
                                      nn.MaxPool2d(kernel_size=4, stride=4),
                                      nn.Conv2d(in_channels=n2, out_channels=n3, kernel_size=3, stride=1, padding=1), nn.ReLU(),
                                      nn.Conv2d(in_channels=n3, out_channels=n3, kernel_size=3, stride=1, padding=1), nn.ReLU(),
                                      nn.BatchNorm2d(n3),
                                      nn.MaxPool2d(kernel_size=2, stride=2))
        self.classifier = nn.Sequential(nn.Linear(in_features=n3*int(size_image/32*size_image/32), out_features=n4), nn.ReLU(),
                                        nn.Dropout(0.5),
                                        nn.Linear(in_features=n4, out_features=n_class))

    def forward(self, x):
        z = self.features(x)
        z = torch.flatten(z, start_dim=1)
        y = self.classifier(z)
        return y


class Net11(nn.Module):

    def __init__(self, n1=16, n2=32, n3=64, n4=128, n5=1024, n_class=10, size_image=256):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=n1, kernel_size=3, stride=1, padding=1), nn.ReLU(),
                                      nn.Conv2d(in_channels=n1, out_channels=n1, kernel_size=3, stride=1, padding=1), nn.ReLU(),
                                      nn.MaxPool2d(kernel_size=4, stride=4),
                                      nn.Conv2d(in_channels=n1, out_channels=n2, kernel_size=3, stride=1, padding=1), nn.ReLU(),
                                      nn.Conv2d(in_channels=n2, out_channels=n2, kernel_size=3, stride=1, padding=1), nn.ReLU(),
                                      nn.BatchNorm2d(n2),
                                      nn.MaxPool2d(kernel_size=2, stride=2),
                                      nn.Conv2d(in_channels=n2, out_channels=n3, kernel_size=3, stride=1, padding=1), nn.ReLU(),
                                      nn.Conv2d(in_channels=n3, out_channels=n3, kernel_size=3, stride=1, padding=1), nn.ReLU(),
                                      nn.MaxPool2d(kernel_size=2, stride=2),
                                      nn.Conv2d(in_channels=n3, out_channels=n4, kernel_size=3, stride=1, padding=1), nn.ReLU(),
                                      nn.Conv2d(in_channels=n4, out_channels=n4, kernel_size=3, stride=1, padding=1), nn.ReLU(),
                                      nn.BatchNorm2d(n4),
                                      nn.MaxPool2d(kernel_size=2, stride=2))
        self.classifier = nn.Sequential(nn.Linear(in_features=n4*int(size_image/32*size_image/32), out_features=n5), nn.ReLU(),
                                        nn.Dropout(0.5),
                                        nn.Linear(in_features=n5, out_features=n4), nn.ReLU(),
                                        nn.Linear(in_features=n4, out_features=n_class))

    def forward(self, x):
        z = self.features(x)
        z = torch.flatten(z, start_dim=1)
        y = self.classifier(z)
        return y


class Vgg11(nn.Module):
    
    def __init__(self, n_class=10, size_image=224, pretrained=False):
        super().__init__()
        
        from torchvision.models import vgg11
        if pretrained:
            from torchvision.models import VGG11_Weights
            _vgg11 = vgg11(weights=VGG11_Weights.IMAGENET1K_V1)
        else:
            _vgg11 = vgg11(weights=None)
        self.features = _vgg11.features
        self.classifier = nn.Sequential(nn.Linear(in_features=512*int(size_image/32*size_image/32), out_features=1024), nn.ReLU(),
                                        nn.Dropout(0.5),
                                        nn.Linear(in_features=1024, out_features=n_class))
    
    def forward(self, x):
        z = self.features(x)
        z = torch.flatten(z, start_dim=1)
        y = self.classifier(z)
        return y


class Vgg16(nn.Module):
    
    def __init__(self, n_class=10, size_image=224, pretrained=False):
        super().__init__()
        
        from torchvision.models import vgg16
        if pretrained:
            from torchvision.models import VGG16_Weights
            _vgg16 = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        else:
            _vgg16 = vgg16(weights=None)
        self.features = _vgg16.features
        self.classifier = nn.Sequential(nn.Linear(in_features=512*int(size_image/32*size_image/32), out_features=1024), nn.ReLU(),
                                        nn.Dropout(0.5),
                                        nn.Linear(in_features=1024, out_features=n_class))
    
    def forward(self, x):
        z = self.features(x)
        z = torch.flatten(z, start_dim=1)
        y = self.classifier(z)
        return y



import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='net8', help='available models: net8, net11, vgg11, vgg16')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    if args.model == 'net8':
        model = Net8(n1=8,n2=8,n3=16,n4=32)
    elif args.model == 'net11':
        model = Net11()
    elif args.model == 'vgg11':
        model = Vgg11(n_class=10, pretrained=False)
    elif args.model == 'vgg11trained':
        model = Vgg11(n_class=10, pretrained=True)
    elif args.model == 'vgg16':
        model = Vgg16(n_class=10, pretrained=False)
    elif args.model == 'vgg16trained':
        model = Vgg16(n_class=10, pretrained=True)
    else:
        print('Error: No such model.')

    print(model)