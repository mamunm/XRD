'''
lstm: LSTM with original data 
data: 30K
objective: binary
include mid point: True
'''
import torch 

from xrd_analyzer.data.data_loader_30K import get_data_loader
from xrd_analyzer.models.lstm_classification import LSTM
from xrd_analyzer.training.train import Trainer

from pathlib import Path
import json

identifier = 'lstm_30K_binary_0.8train'
objective = 'binary'
save_path = Path(__file__).resolve().parent / "outputs" / identifier
if not save_path.exists():
    save_path.mkdir()
arg_dict = {'dataloader': {
                'data_ratio': [0.80, 0.10, 0.10],
                'batch_size': 256,
                'objective': objective,
                'include_mid_point': True,
                'save_path': save_path,
                'random_state': 42,
                'num_workers': 42},
            'model': {
                'input_size': 250,
                'hidden_size': 128,
                'num_layers': 2,
                'objective': objective},
            'train': {
                'objective': objective,
                'save_model': True,
                'save_path': save_path,
                'model_id': identifier}}
    
# dataloader 
train_data_loader, val_data_loader, test_data_loader = get_data_loader(
    **arg_dict['dataloader'])
# model + loss + optimizer
model = LSTM(**arg_dict['model'])
loss = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
# training
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
trainer = Trainer(model, optimizer, loss, device, **arg_dict['train'])
trainer.train(train_data_loader, val_data_loader, 
              test_data_loader, epochs=50)

for k in arg_dict:
    if 'save_path' in arg_dict[k]:
        arg_dict[k]['save_path'] = str(save_path)
with open(save_path / "args_dict.json", 'w+') as f:
    json.dump(arg_dict, f)