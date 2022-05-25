import torch
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score

def validate_model(model, valid_loader, device, loss_fn, objective, show_report=True):
    model.eval()
    valid_loss = 0
    valid_accuracy = 0
    y_pred = []
    y_pred_proba = []
    y_target = []
    for batch_idx, (data, target, _) in enumerate(valid_loader):
        if batch_idx == 0:
            batch_size = len(target)
        y_target.extend(target.tolist())
        data = data.to(device, dtype=torch.float)
        target = target.to(device, dtype=torch.long)
        output = torch.softmax(model(data), dim=1)
        loss = loss_fn(output, target)
        valid_loss += loss.item()
        valid_accuracy += torch.sum(torch.argmax(output, dim=1) == target).item()
        y_pred.extend(torch.argmax(output, dim=1).tolist())
        y_pred_proba.extend(output.tolist())
    if show_report:
        print_report(y_target, y_pred, y_pred_proba, objective)

    return valid_loss / len(valid_loader), 100 * valid_accuracy / len(valid_loader) / batch_size

def print_report(y_target, y_pred, y_pred_proba, objective):
    print('Classification report:\n',
          classification_report(y_target, y_pred))
    # if save_model:
    #     print('Classification report:\n',
    #       classification_report(y_target, y_pred), 
    #       file=self.save_out_file)
    if objective == 'binary':
        y_pred_proba = np.array(y_pred_proba)[:, 1]
        print('ROC AUC score:', roc_auc_score(y_target, y_pred_proba))
        # if self.save_model:
        #     print('ROC AUC score:', roc_auc_score(y_target, y_pred_proba),
        #           file=self.save_out_file)
    else:
        print(f"""ROC AUC score (one vs one): {roc_auc_score(y_target, 
            y_pred_proba, multi_class='ovo', average='macro')}""")
        print(f"""ROC AUC score (one vs rest): {roc_auc_score(y_target, 
            y_pred_proba, multi_class='ovr', average='macro')}""")
        # if self.save_model:
        #     print(f"""ROC AUC score (one vs one): {roc_auc_score(y_target, 
        #         y_pred_proba, multi_class='ovo', average='macro')}""",
        #         file=self.save_out_file)
        #     print(f"""ROC AUC score (one vs rest): {roc_auc_score(y_target, 
        #         y_pred_proba, multi_class='ovr', average='macro')}""",
        #         file=self.save_out_file)