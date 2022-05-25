'''
cnn_org: 1D CNN with original data 
(i.e., no synthetic data)
'''
import torch 

from xrd_analyzer.data.data_loader_10K import get_data_loader
from xrd_analyzer.models.cnn_classification import CNN 
from xrd_analyzer.training.train import Trainer

batch_size = 128
objective = 'ternary'
train_dataloader = get_data_loader(data='train', batch_size=batch_size, 
                                   objective=objective)
val_dataloader = get_data_loader(data='valid', batch_size=batch_size,
                                 objective=objective)
test_dataloader = get_data_loader(data='test', batch_size=batch_size,
                                  objective=objective)
model = CNN(objective=objective)
loss = torch.nn.CrossEntropyLoss()
#optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
trainer = Trainer(model, optimizer, loss, device, objective=objective)

trainer.train(train_dataloader, val_dataloader, 
              test_dataloader, epochs=10)