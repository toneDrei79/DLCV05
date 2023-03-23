import torch
import torch.nn as nn


class Net(nn.Module):

    def __init__(self, n1=16, n2=32, n3=64, n4=128, n5=1024, n_class=10, size_image=512):
        super().__init__()
        self.featrues = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=n1, kernel_size=3, stride=1, padding=1), nn.ReLU(),
                                      nn.Conv2d(in_channels=n1, out_channels=n1, kernel_size=3, stride=1, padding=1), nn.ReLU(),
                                      nn.MaxPool2d(kernel_size=2, stride=2),
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
        self.classifier = nn.Sequential(nn.Linear(in_features=n4*int(size_image/16*size_image/16), out_features=n5), nn.ReLU(),
                                        nn.Dropout(0.5),
                                        nn.Linear(in_features=n5, out_features=n4), nn.ReLU(),
                                        nn.Linear(in_features=n4, out_features=n_class))

    def forward(self, x):
        z = self.featrues(x)
        z = torch.flatten(z, start_dim=1)
        y = self.classifier(z)
        return y


if __name__ == '__main__':
    model = Net()
    print(model)