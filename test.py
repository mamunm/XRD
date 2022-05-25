from xrd_analyzer.data.data_loader import get_data_loader
from xrd_analyzer.models.cnn_classification import CNN 
from xrd_analyzer.training.train import Trainer
import torch  
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_dataloader = get_data_loader(data='train', batch_size=128)
batch = next(iter(train_dataloader))
X = batch[0]
X = X.view(X.shape[0], 1, X.shape[1])
print("Initial")
print(X.shape)
X = X.to(device, dtype=torch.float)

conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=10, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(10),
            nn.ReLU())
       
conv2 = nn.Sequential(
            nn.Conv1d(in_channels=10, out_channels=10, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(10),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1))
        
conv3 = nn.Sequential(
            nn.Conv1d(in_channels=10, out_channels=10, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(10),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=2, padding=1))
        
conv4 = nn.Sequential(
            nn.Conv1d(in_channels=10, out_channels=10, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(10),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=5, stride=2, padding=1))

conv1 = conv1.to(device)
conv2 = conv2.to(device)
conv3 = conv3.to(device)
conv4 = conv4.to(device)

print("1st pass")
X = conv1(X)
print(X.shape)

print("2nd pass")
X = conv3(X)
print(X.shape)

print("3rd pass")
X = conv2(X)
print(X.shape)

print("4th pass")
X = conv3(X)
print(X.shape)

print("5th pass")
X = conv2(X)
print(X.shape)

print("6th pass")
X = conv2(X)
print(X.shape)

print("7th pass")
X = conv2(X)
print(X.shape)

print("8th pass")
X = conv3(X)
print(X.shape)

print("9th pass")
X = conv2(X)
print(X.shape)

