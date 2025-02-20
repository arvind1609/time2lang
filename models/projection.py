import torch
from torch import nn

class ProjectionModel(torch.nn.Module):
    def __init__(self, n_classes):
        super(ProjectionModel, self).__init__()

        self.projection = nn.Sequential(nn.Linear(2048, 512),
                                       nn.BatchNorm1d(512),
                                       nn.ReLU(512),
                                       nn.Linear(512, 256),
                                       nn.BatchNorm1d(256),
                                       nn.ReLU(256),
                                       nn.Linear(256, n_classes))
    def forward(self, x):
        return self.projection(x)

class ProjectionModelRes(torch.nn.Module):
    def __init__(self, n_classes):
        super(ProjectionModelRes, self).__init__()

        self.linear_1 =  nn.Sequential(nn.Linear(2048, 768),
                                       nn.BatchNorm1d(768),
                                       nn.ReLU(768))
        self.linear_2 =  nn.Sequential(nn.Linear(768, 256),
                                       nn.BatchNorm1d(256),
                                       nn.ReLU(256))
        self.output = nn.Linear(256, n_classes)

    def forward(self, x, z):
        return self.output(self.linear_2(self.linear_1(x) + z))

class ProjectionModelv2(torch.nn.Module):
    def __init__(self):
        super(ProjectionModelv2, self).__init__()

        self.projection = nn.Sequential(nn.Linear(2048, 512),
                                       nn.BatchNorm1d(512),
                                       nn.ReLU(512))
    def forward(self, x):
        return self.projection(x)

class HeadModel(torch.nn.Module):
    def __init__(self, n_classes):
        super(HeadModel, self).__init__()

        self.head = nn.Sequential(nn.Linear(512, 256),
                                       nn.BatchNorm1d(256),
                                       nn.ReLU(256),
                                  nn.Linear(256, n_classes)
                                     )
    def forward(self, x):
        return self.head(x)