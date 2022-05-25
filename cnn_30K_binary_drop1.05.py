'''
cnn_org: 1D CNN with original data 
data: 30K
objective: binary
include mid point: False
'''
import torch 

from xrd_analyzer.data.data_loader_30K import get_data_loader
from xrd_analyzer.models.cnn_classification import CNN 
from xrd_analyzer.training.train import Trainer

from pathlib import Path
import json

identifier = 'cnn_30K_binary_false'
objective = 'binary'
save_path = Path(__file__).resolve().parent / "outputs" / identifier
if not save_path.exists():
    save_path.mkdir()
arg_dict = {'dataloader': {
                'data_ratio': [0.75, 0.15, 0.10],
                'batch_size': 256,
                'objective': objective,
                'include_mid_point': False,
                'save_path': save_path,
                'random_state': 42},
            'model': {
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
model = CNN(**arg_dict['model'])
loss = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
# training
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
trainer = Trainer(model, optimizer, loss, device, **arg_dict['train'])
trainer.train(train_data_loader, val_data_loader, 
              test_data_loader, epochs=25)

for k in arg_dict:
    if 'save_path' in arg_dict[k]:
        arg_dict[k]['save_path'] = str(save_path)
with open(save_path / "args_dict.json", 'w+') as f:
    json.dump(arg_dict, f)