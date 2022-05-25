'''
cnn_org: 1D CNN with original data 
data: 30K
objective: ternary
include mid point: True
temp scaling: True
'''
import sys
sys.path.append('/home/osman/src/temperature_scaling')

import torch 
from xrd_analyzer.data.data_loader_30K import get_data_loader
from xrd_analyzer.models.cnn_classification import CNN 
from xrd_analyzer.utils.utils import validate_model

from pathlib import Path
from temperature_scaling import ModelWithTemperature

identifier = 'cnn_30K_ternary_true'
objective = 'ternary'
save_path = Path(__file__).resolve().parent / "outputs" / identifier
if not save_path.exists():
    save_path.mkdir()
arg_dict = {'dataloader': {
                'data_ratio': [0.75, 0.15, 0.10],
                'batch_size': 256,
                'objective': objective,
                'include_mid_point': True,
                'save_path': save_path,
                'random_state': 42,
                'num_workers': 42},
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
epochs = 50
state_dict = torch.load(Path(__file__).resolve().parent / 'outputs' / identifier / f'model_{identifier}_{epochs-1}.pth')['state_dict']
model.load_state_dict(state_dict)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Tune temperature 
model = model.to(device)
model = ModelWithTemperature(model)
model.set_temperature(val_data_loader)
test_loss, test_acc = validate_model(model, test_data_loader, device, loss, objective)
print(test_loss, test_acc)
