from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(rc={"lines.linewidth": 2}, style="whitegrid", 
        context="paper", palette="Set1")

files = Path(__file__).resolve().parent.glob('*/out.txt')

for f in files:
    with open(f, 'r') as file:
        lines = file.readlines()
        epochs = [int(line.split()[1].split('/')[0]) 
                  for line in lines if line.startswith('Epoch')]
        train_loss = [float(line.split()[4]) 
                  for line in lines if line.startswith('Epoch')]
        valid_loss = [float(line.split()[7]) 
                  for line in lines if line.startswith('Epoch')]
        
        sns.lineplot(x=epochs, y=train_loss, label='Train')
        sns.lineplot(x=epochs, y=valid_loss, label='Valid')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f.parent / 'loss.png')
        plt.show()