import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, objective: str = 'binary'):
        super(CNN, self).__init__()
        # Lout = (Lin - k + 2p)/s + 1
        
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=10, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(10),
            nn.ReLU())
       
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=10, out_channels=10, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(10),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1))
        
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=10, out_channels=10, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(10),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=2, padding=1))
        
        self.conv4 = nn.Sequential(
            nn.Conv1d(in_channels=10, out_channels=10, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(10),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=5, stride=2, padding=1))
        
        obj_dict = {'binary': 2, 'ternary': 3, 'multiclass': 5}
        self.linear1 = nn.Linear(310, obj_dict[objective])
        self.activation = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = x.view(x.shape[0], 1, x.shape[1])
        # (B, 1, 7750) -> (B, 10, 7750)
        x = self.conv1(x)
        # (B, 10, 7750) -> (B, 10, 3875)
        x = self.conv3(x)
        # (B, 10, 3875) -> (B, 10, 1938)
        x = self.conv2(x)
        # (B, 10, 1938) -> (B, 10, 969)
        x = self.conv3(x)
        # (B, 10, 969) -> (B, 10, 485)
        x = self.conv2(x)
        # (B, 10, 485) -> (B, 10, 243)
        x = self.conv2(x)
        # (B, 10, 243) -> (B, 10, 122)
        x = self.conv2(x)
        # (B, 10, 122) -> (B, 10, 61)
        x = self.conv3(x)
        # (B, 10, 61) -> (B, 10, 31)
        x = self.conv2(x)
        # (B, 10, 31) -> (B, 310)
        x = x.view(x.shape[0], -1)
        return self.activation(self.linear1(x))
