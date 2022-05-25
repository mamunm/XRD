from xrd_analyzer.data.data_loader import get_data_loader
from xrd_analyzer.models.cnn_classification import CNN 
from xrd_analyzer.training.train import Trainer
import torch  
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_dataloader = get_data_loader(data='train', batch_size=128)
test_dataloader = get_data_loader(data='test', batch_size=128)

model = CNN()
model = model.to(device)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

model.train()
train_loss = 0
for batch_idx, (data, target, _) in enumerate(train_dataloader):
    data = data.to(device, dtype=torch.float)
    target = target.to(device, dtype=torch.long)
    optimizer.zero_grad()
    output = model(data)
    loss = loss_fn(output, target)
    loss.backward()
    optimizer.step()
    train_loss += loss.item()
    #print(f"Batch Loss [{batch_idx}/{len(train_dataloader)}]: {loss.item():0.3f}")

train_loss = train_loss / len(train_dataloader)
print(train_loss)

model.eval()
valid_loss = 0
valid_accuracy = 0
        
for i, (data, target, _) in enumerate(test_dataloader):
    if i == 0:
        batch_size = len(target)
        print(batch_size)
    data = data.to(device, dtype=torch.float)
    target = target.to(device, dtype=torch.long)
    output = model(data)
    print(output, target)
    print(torch.argmax(output, dim=1))
    print(torch.argmax(output, dim=1) == target)
    print(torch.sum(torch.argmax(output, dim=1) == target).item())
    loss = loss_fn(output, target)
    valid_loss += loss.item()
    valid_accuracy += torch.sum(torch.argmax(output, dim=1) == target).item()

valid_loss = valid_loss / len(test_dataloader)
valid_accuracy = valid_accuracy / len(test_dataloader) / batch_size
print(valid_loss, valid_accuracy)
            