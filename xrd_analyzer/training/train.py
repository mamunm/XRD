import torch
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score

class Trainer:
    def __init__(self, model, optimizer, loss_fn, device, 
                 save_model=False, save_path=None, model_id=None, 
                 verbose=False, objective='binary'):
        """Trainer class for training and validating a model.

        Args:
            model : Model to be trained
            optimizer : Optimizer to be used
            loss_fn : Loss function to be used
            device : device to be used
            save_model : Whether to save the model
            save_path : Path to save the model
            model_id : ID of the model
            verbose : Whether to print training information
            objective : binary or ternary or multiclass
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.save_model = save_model
        self.save_path = save_path
        self.model_id = model_id
        self.verbose = verbose
        self.objective = objective
        self.save_out_file = open(str(save_path) + "/out.txt", "w")
        
    def train(self, train_loader, valid_loader, 
              test_loader, epochs):
        """Trains the model for a given number of epochs.

        Args:
            train_loader : DataLoader for training data
            valid_loader : DataLoader for validation data
            test_loader : DataLoader for test data
            epochs : Number of epochs to train for
        """
        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            valid_loss, valid_accuracy = self.validate(valid_loader)
            print(f"Epoch: {epoch}/{epochs}", 
                  f"Training Loss: {train_loss:0.3f}",
                  f"Validation Loss: {valid_loss:0.3f}",
                  f"Validation Acc: {valid_accuracy:0.3f}", sep="\t")
            if self.save_model:
                self.save_model_state(epoch, valid_accuracy)
                print(f"Epoch: {epoch}/{epochs}", 
                  f"Training Loss: {train_loss:0.3f}",
                  f"Validation Loss: {valid_loss:0.3f}",
                  f"Validation Acc: {valid_accuracy:0.3f}", sep="\t",
                  file=self.save_out_file)
        test_loss, test_accuracy = self.validate(test_loader, show_report=True)
        print(f"Test Loss: {test_loss:0.3f} | Test Acc: {test_accuracy:0.3f}")
        if self.save_model:
            print(f"Test Loss: {test_loss:0.3f} | Test Acc: {test_accuracy:0.3f}",
                file=self.save_out_file)
            self.save_out_file.close()
    
    def train_epoch(self, train_loader):
        self.model.train()
        train_loss = 0
        for batch_idx, (data, target, _) in enumerate(train_loader):
            data = data.to(self.device, dtype=torch.float)
            target = target.to(self.device, dtype=torch.long)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss_fn(output, target)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            if self.verbose:
                print(f"Batch Loss [{batch_idx}/{len(train_loader)}]: {loss.item():0.3f}")
        return train_loss / len(train_loader)
    
    def validate(self, valid_loader, show_report=False):
        self.model.eval()
        valid_loss = 0
        valid_accuracy = 0
        y_pred = []
        y_pred_proba = []
        y_target = []
        for batch_idx, (data, target, _) in enumerate(valid_loader):
            if batch_idx == 0:
                batch_size = len(target)
            y_target.extend(target.tolist())
            data = data.to(self.device, dtype=torch.float)
            target = target.to(self.device, dtype=torch.long)
            output = self.model(data)
            loss = self.loss_fn(output, target)
            valid_loss += loss.item()
            valid_accuracy += torch.sum(torch.argmax(output, dim=1) == target).item()
            y_pred.extend(torch.argmax(output, dim=1).tolist())
            y_pred_proba.extend(output.tolist())
        if show_report:
            self.print_report(y_target, y_pred, y_pred_proba)
        if show_report and self.save_model:
            np.save(self.save_path / 'test_report.npy', {'label': y_target,
                'pred': y_pred, 'pred_proba': y_pred_proba})
        return valid_loss / len(valid_loader), 100 * valid_accuracy / len(valid_loader) / batch_size
    
    def print_report(self, y_target, y_pred, y_pred_proba):
        print('Classification report:\n',
              classification_report(y_target, y_pred))
        if self.save_model:
            print('Classification report:\n',
              classification_report(y_target, y_pred), 
              file=self.save_out_file)
        if self.objective == 'binary':
            y_pred_proba = np.array(y_pred_proba)[:, 1]
            print('ROC AUC score:', roc_auc_score(y_target, y_pred_proba))
            if self.save_model:
                print('ROC AUC score:', roc_auc_score(y_target, y_pred_proba),
                      file=self.save_out_file)
        else:
            print(f"""ROC AUC score (one vs one): {roc_auc_score(y_target, 
                y_pred_proba, multi_class='ovo', average='macro')}""")
            print(f"""ROC AUC score (one vs rest): {roc_auc_score(y_target, 
                y_pred_proba, multi_class='ovr', average='macro')}""")
            if self.save_model:
                print(f"""ROC AUC score (one vs one): {roc_auc_score(y_target, 
                    y_pred_proba, multi_class='ovo', average='macro')}""",
                    file=self.save_out_file)
                print(f"""ROC AUC score (one vs rest): {roc_auc_score(y_target, 
                    y_pred_proba, multi_class='ovr', average='macro')}""",
                    file=self.save_out_file)
    
    def save_model_state(self, epoch, accuracy):
        torch.save({'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'epoch': epoch,
                    'accuracy': accuracy},
                   self.save_path / f"model_{self.model_id}_{epoch}.pth")
            
            
            