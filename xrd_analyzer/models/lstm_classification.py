import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_size: int = 50, 
                 hidden_size: int = 128, 
                 num_layers: int = 2, objective: str = 'binary'):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, 
                          num_layers, batch_first=True)
        obj_dict = {'binary': 2, 'ternary': 3, 'multiclass': 5}
        self.linear = nn.Linear(hidden_size, obj_dict[objective])
        self.activation = nn.Softmax(dim=1)
    
    def forward(self, x):
        # x: (batch_size, seq_len, input_size)
        x = x.view(x.size(0), -1, self.input_size)
        # out: (batch_size, seq_len, hidden_size)
        out, _ = self.lstm(x, None)
        # out: (batch_size, hidden_size)
        out = out[:, -1, :]
        out = self.linear(out)
        return self.activation(out)
        
        