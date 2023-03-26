import torch
import torch.nn as nn


class Net5(nn.Module):

    def __init__(self, n1=16, n2=32, n3=512, n_class=10, image_size=128, batchnorm=True, dropout=True):
        super().__init__()
        if batchnorm:
            self.features = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=n1, kernel_size=3, stride=1, padding=1),
                                          nn.BatchNorm2d(n1),
                                          nn.ReLU(),
                                          nn.MaxPool2d(kernel_size=8, stride=8),
                                          nn.Conv2d(in_channels=n1, out_channels=n2, kernel_size=3, stride=1, padding=1),
                                          nn.BatchNorm2d(n2),
                                          nn.ReLU(),
                                          nn.MaxPool2d(kernel_size=4, stride=4))
        else:
            self.features = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=n1, kernel_size=3, stride=1, padding=1),
                                          nn.BatchNorm2d(n1),
                                          nn.ReLU(),
                                          nn.MaxPool2d(kernel_size=8, stride=8),
                                          nn.Conv2d(in_channels=n1, out_channels=n2, kernel_size=3, stride=1, padding=1),
                                          nn.BatchNorm2d(n2),
                                          nn.ReLU(),
                                          nn.MaxPool2d(kernel_size=4, stride=4))
        if dropout:
            self.classifier = nn.Sequential(nn.Linear(in_features=n2*int(image_size/32*image_size/32), out_features=n3),
                                            nn.Dropout(0.5),
                                            nn.ReLU(),
                                            nn.Linear(in_features=n3, out_features=n3),
                                            nn.Dropout(0.5),
                                            nn.ReLU(),
                                            nn.Linear(in_features=n3, out_features=n_class))
        else:
            self.classifier = nn.Sequential(nn.Linear(in_features=n2*int(image_size/32*image_size/32), out_features=n3),
                                            nn.ReLU(),
                                            nn.Linear(in_features=n3, out_features=n3),
                                            nn.ReLU(),
                                            nn.Linear(in_features=n3, out_features=n_class))

    def forward(self, x):
        z = self.features(x)
        z = torch.flatten(z, start_dim=1)
        y = self.classifier(z)
        return y


class Net7(nn.Module):

    def __init__(self, n1=16, n2=32, n3=64, n4=512, n_class=10, image_size=128, batchnorm=True, dropout=True):
        super().__init__()
        if batchnorm:
            self.features = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=n1, kernel_size=3, stride=1, padding=1),
                                          nn.ReLU(),
                                          nn.Conv2d(in_channels=n1, out_channels=n1, kernel_size=3, stride=1, padding=1),
                                          nn.BatchNorm2d(n1),
                                          nn.ReLU(),
                                          nn.MaxPool2d(kernel_size=4, stride=4),
                                          nn.Conv2d(in_channels=n1, out_channels=n2, kernel_size=3, stride=1, padding=1),
                                          nn.ReLU(),
                                          nn.MaxPool2d(kernel_size=4, stride=4),
                                          nn.Conv2d(in_channels=n2, out_channels=n3, kernel_size=3, stride=1, padding=1),
                                          nn.BatchNorm2d(n3),
                                          nn.ReLU(),
                                          nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            self.features = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=n1, kernel_size=3, stride=1, padding=1),
                                          nn.ReLU(),
                                          nn.Conv2d(in_channels=n1, out_channels=n1, kernel_size=3, stride=1, padding=1),
                                          nn.ReLU(),
                                          nn.MaxPool2d(kernel_size=4, stride=4),
                                          nn.Conv2d(in_channels=n1, out_channels=n2, kernel_size=3, stride=1, padding=1),
                                          nn.ReLU(),
                                          nn.MaxPool2d(kernel_size=4, stride=4),
                                          nn.Conv2d(in_channels=n2, out_channels=n3, kernel_size=3, stride=1, padding=1),
                                          nn.ReLU(),
                                          nn.MaxPool2d(kernel_size=2, stride=2))
        if dropout:
            self.classifier = nn.Sequential(nn.Linear(in_features=n3*int(image_size/32*image_size/32), out_features=n4),
                                            nn.Dropout(0.5),
                                            nn.ReLU(),
                                            nn.Linear(in_features=n4, out_features=n4),
                                            nn.Dropout(0.5),
                                            nn.ReLU(),
                                            nn.Linear(in_features=n4, out_features=n_class))
        else:
            self.classifier = nn.Sequential(nn.Linear(in_features=n3*int(image_size/32*image_size/32), out_features=n4),
                                            nn.ReLU(),
                                            nn.Linear(in_features=n4, out_features=n4),
                                            nn.ReLU(),
                                            nn.Linear(in_features=n4, out_features=n_class))

    def forward(self, x):
        z = self.features(x)
        z = torch.flatten(z, start_dim=1)
        y = self.classifier(z)
        return y


class Net11(nn.Module):

    def __init__(self, n1=16, n2=32, n3=64, n4=128, n5=512, n_class=10, image_size=128, batchnorm=True, dropout=True):
        super().__init__()
        if batchnorm:
            self.features = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=n1, kernel_size=3, stride=1, padding=1),
                                          nn.ReLU(),
                                          nn.Conv2d(in_channels=n1, out_channels=n1, kernel_size=3, stride=1, padding=1),
                                          nn.ReLU(),
                                          nn.MaxPool2d(kernel_size=4, stride=4),
                                          nn.Conv2d(in_channels=n1, out_channels=n2, kernel_size=3, stride=1, padding=1),
                                          nn.ReLU(),
                                          nn.Conv2d(in_channels=n2, out_channels=n2, kernel_size=3, stride=1, padding=1),
                                          nn.BatchNorm2d(n2),
                                          nn.ReLU(),
                                          nn.MaxPool2d(kernel_size=2, stride=2),
                                          nn.Conv2d(in_channels=n2, out_channels=n3, kernel_size=3, stride=1, padding=1),
                                          nn.ReLU(),
                                          nn.Conv2d(in_channels=n3, out_channels=n3, kernel_size=3, stride=1, padding=1),
                                          nn.ReLU(),
                                          nn.MaxPool2d(kernel_size=2, stride=2),
                                          nn.Conv2d(in_channels=n3, out_channels=n4, kernel_size=3, stride=1, padding=1),
                                          nn.ReLU(),
                                          nn.Conv2d(in_channels=n4, out_channels=n4, kernel_size=3, stride=1, padding=1),
                                          nn.BatchNorm2d(n4),
                                          nn.ReLU(),
                                          nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            self.features = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=n1, kernel_size=3, stride=1, padding=1),
                                          nn.ReLU(),
                                          nn.Conv2d(in_channels=n1, out_channels=n1, kernel_size=3, stride=1, padding=1),
                                          nn.ReLU(),
                                          nn.MaxPool2d(kernel_size=4, stride=4),
                                          nn.Conv2d(in_channels=n1, out_channels=n2, kernel_size=3, stride=1, padding=1),
                                          nn.ReLU(),
                                          nn.Conv2d(in_channels=n2, out_channels=n2, kernel_size=3, stride=1, padding=1),
                                          nn.ReLU(),
                                          nn.MaxPool2d(kernel_size=2, stride=2),
                                          nn.Conv2d(in_channels=n2, out_channels=n3, kernel_size=3, stride=1, padding=1),
                                          nn.ReLU(),
                                          nn.Conv2d(in_channels=n3, out_channels=n3, kernel_size=3, stride=1, padding=1),
                                          nn.ReLU(),
                                          nn.MaxPool2d(kernel_size=2, stride=2),
                                          nn.Conv2d(in_channels=n3, out_channels=n4, kernel_size=3, stride=1, padding=1),
                                          nn.ReLU(),
                                          nn.Conv2d(in_channels=n4, out_channels=n4, kernel_size=3, stride=1, padding=1),
                                          nn.ReLU(),
                                          nn.MaxPool2d(kernel_size=2, stride=2))
        if dropout:
            self.classifier = nn.Sequential(nn.Linear(in_features=n4*int(image_size/32*image_size/32), out_features=n5),
                                            nn.Dropout(0.5),
                                            nn.ReLU(),
                                            nn.Linear(in_features=n5, out_features=n5),
                                            nn.Dropout(0.5),
                                            nn.ReLU(),
                                            nn.Linear(in_features=n5, out_features=n_class))
        else:
            self.classifier = nn.Sequential(nn.Linear(in_features=n4*int(image_size/32*image_size/32), out_features=n5),
                                            nn.ReLU(),
                                            nn.Linear(in_features=n5, out_features=n5),
                                            nn.ReLU(),
                                            nn.Linear(in_features=n5, out_features=n_class))

    def forward(self, x):
        z = self.features(x)
        z = torch.flatten(z, start_dim=1)
        y = self.classifier(z)
        return y


class Vgg11(nn.Module):
    
    def __init__(self, n_class=10, image_size=128, pretrained=False, batchnorm=False, dropout=True):
        super().__init__()
        
        from torchvision.models import vgg11
        if pretrained:
            from torchvision.models import VGG11_Weights
            _vgg11 = vgg11(weights=VGG11_Weights.IMAGENET1K_V1)
        else:
            _vgg11 = vgg11(weights=None)
        if batchnorm:
            self.features = _vgg11.features[:4]\
                                .append(nn.BatchNorm2d(128))\
                                .extend(_vgg11.features[4:9])\
                                .append(nn.BatchNorm2d(256))\
                                .extend(_vgg11.features[9:15])\
                                .append(nn.BatchNorm2d(512))\
                                .extend(_vgg11.features[15:19])\
                                .append(nn.BatchNorm2d(512))\
                                .extend(_vgg11.features[19:])
        else:
            self.features = _vgg11.features
        if dropout:
            self.classifier = nn.Sequential(nn.Linear(in_features=512*int(image_size/32*image_size/32), out_features=512),
                                            nn.Dropout(0.5),
                                            nn.ReLU(),
                                            nn.Linear(in_features=512, out_features=512),
                                            nn.Dropout(0.5),
                                            nn.ReLU(),
                                            nn.Linear(in_features=512, out_features=n_class))
        else:
            self.classifier = nn.Sequential(nn.Linear(in_features=512*int(image_size/32*image_size/32), out_features=512),
                                            nn.ReLU(),
                                            nn.Linear(in_features=512, out_features=512),
                                            nn.ReLU(),
                                            nn.Linear(in_features=512, out_features=n_class))
    
    def forward(self, x):
        z = self.features(x)
        z = torch.flatten(z, start_dim=1)
        y = self.classifier(z)
        return y


class Vgg16(nn.Module):
    
    def __init__(self, n_class=10, image_size=128, pretrained=False, dropout=True):
        super().__init__()
        
        from torchvision.models import vgg16
        if pretrained:
            from torchvision.models import VGG16_Weights
            _vgg16 = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        else:
            _vgg16 = vgg16(weights=None)
        self.features = _vgg16.features
        if dropout:
            self.classifier = nn.Sequential(nn.Linear(in_features=512*int(image_size/32*image_size/32), out_features=512),
                                            nn.Dropout(0.5),
                                            nn.ReLU(),
                                            nn.Linear(in_features=512, out_features=512),
                                            nn.Dropout(0.5),
                                            nn.ReLU(),
                                            nn.Linear(in_features=512, out_features=n_class))
        else:
            self.classifier = nn.Sequential(nn.Linear(in_features=512*int(image_size/32*image_size/32), out_features=1024),
                                            nn.ReLU(),
                                            nn.Linear(in_features=1024, out_features=512),
                                            nn.ReLU(),
                                            nn.Linear(in_features=512, out_features=n_class))
    
    def forward(self, x):
        z = self.features(x)
        z = torch.flatten(z, start_dim=1)
        y = self.classifier(z)
        return y


class ResNet18(nn.Module):
    
    def __init__(self, n_class=10, pretrained=False, dropout=True):
        super().__init__()
        
        from torchvision.models import resnet18
        if pretrained:
            from torchvision.models import ResNet18_Weights
            _resnet18 = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            _resnet18 = resnet18(weights=None)
        self.conv1 = _resnet18.conv1
        self.bn1 = _resnet18.bn1
        self.relu = _resnet18.relu
        self.maxpool = _resnet18.maxpool
        self.layer1 = _resnet18.layer1
        self.layer2 = _resnet18.layer2
        self.layer3 = _resnet18.layer3
        self.layer4 = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(512)) # modify layer4 to keep feature size
        self.avgpool = _resnet18.avgpool
        if dropout:
            self.classifier = nn.Sequential(nn.Linear(in_features=512, out_features=512),
                                            nn.Dropout(0.5),
                                            nn.ReLU(),
                                            nn.Linear(in_features=512, out_features=512),
                                            nn.Dropout(0.5),
                                            nn.ReLU(),
                                            nn.Linear(in_features=512, out_features=n_class))
        else:
            self.classifier = nn.Sequential(nn.Linear(in_features=512, out_features=512),
                                            nn.ReLU(),
                                            nn.Linear(in_features=512, out_features=512),
                                            nn.ReLU(),
                                            nn.Linear(in_features=512, out_features=n_class))
    
    def forward(self, x):
        z = self.conv1(x)
        z = self.bn1(z)
        z = self.relu(z)
        z = self.maxpool(z)
        z = self.layer1(z)
        z = self.layer2(z)
        z = self.layer3(z)
        z = self.layer4(z)
        z = self.avgpool(z)
        z = torch.flatten(z, start_dim=1)
        y = self.classifier(z)
        return y


def select_model(key, dropout=True, batchnorm=True, pretrained=False):
    if key == 'net5':
        return Net5(batchnorm=batchnorm, dropout=dropout)
    if key == 'net7':
        return Net7(batchnorm=batchnorm, dropout=dropout)
    elif key == 'net11':
        return Net11(batchnorm=batchnorm, dropout=dropout)
    elif key == 'vgg11':
        return Vgg11(n_class=10, pretrained=pretrained, batchnorm=batchnorm, dropout=dropout)
    elif key == 'vgg16':
        return Vgg16(n_class=10, pretrained=pretrained, dropout=dropout)
    elif key == 'resnet18':
        return ResNet18(n_class=10, pretrained=pretrained, dropout=dropout)
    else:
        print('Error: No such model.')
        return None




import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='net7', help='available models: net5, net7, net11, vgg11, vgg16, resnet18 ... default=net7')
    parser.add_argument('--dropout', action='store_true', help='whether do dropout ... default=True')
    parser.add_argument('--batchnorm', action='store_true', help='whether do batchnorm ... default=True')
    parser.add_argument('--pretrained', action='store_true', help='use pretrained model (vgg, resnet) ... default=False')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    model = select_model(args.model, args.dropout, args.batchnorm, args.pretrained)
    print(model)