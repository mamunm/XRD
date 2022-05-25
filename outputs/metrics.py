from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc
import pandas as pd
sns.set(rc={"lines.linewidth": 2}, style="whitegrid", 
        context="paper", palette="Set1", font_scale=2)

def plot_roc_auc_binary(y_true, y_pred, save_path=None):
    """
    Plot ROC curve for binary and save it to the specified path.
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:0.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    if save_path:
        plt.savefig(save_path)
    return plt

def plot_multiclass_roc(y_true, y_pred, save_path=None):

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    n_classes = len(set(y_true))

    y_true_dummies = pd.get_dummies(y_true, drop_first=False).values
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_dummies[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_dummies.ravel(), 
                                              y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # roc for each class
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver operating characteristic example')
    for i in range(n_classes):
        ax.plot(fpr[i], tpr[i], label=f'ROC curve (area = {roc_auc[i]:0.2f}) for label {i}')
    ax.plot(fpr["micro"], tpr["micro"],
         label=f"micro-average ROC curve (area = {roc_auc['micro']:0.2f})",
         color='deeppink', linestyle=':', linewidth=4)

    ax.plot(fpr["macro"], tpr["macro"],
         label=f"macro-average ROC curve (area = {roc_auc['micro']:0.2f})",
         color='navy', linestyle=':', linewidth=4)
    ax.legend(loc="best")
    ax.grid(alpha=.4)
    sns.despine()
    if save_path:
        plt.savefig(save_path)
    return plt

files = Path(__file__).resolve().parent.glob('*/test_report.npy')

for f in files:
    data = np.load(f, allow_pickle=True).item()
    cm = confusion_matrix(data['label'], data['pred'])
    df_cm = pd.DataFrame(cm)
    plt.figure(figsize=(10,7))
    sns.heatmap(df_cm, annot=True, 
                annot_kws={"size": 16}, fmt='d', cmap="Blues")
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.savefig(f.parent / 'confusion_matrix.png')
    plt.show()
    if len(set(data['label'])) == 2:
        plt = plot_roc_auc_binary(data['label'], 
                                  np.array(data['pred_proba'])[:, 1])
        plt.savefig(f.parent / 'roc_curve.png')
        plt.show()
    else:
        plt = plot_multiclass_roc(data['label'], 
                                  np.array(data['pred_proba']))
        plt.savefig(f.parent / 'roc_curve.png')
        plt.show()
    
        