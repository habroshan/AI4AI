import os
import csv
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix,
    precision_score, recall_score, f1_score
)


def plot_confusion_matrix(y_true, y_pred, labels, output_path, title=None):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots()
    im = ax.imshow(cm, cmap='Blues')
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha='center', va='center')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    if title:
        ax.set_title(title)
    fig.colorbar(im)
    fig.savefig(output_path)
    plt.close(fig)


def plot_multi_roc(roc_data, output_path, title=None):
    fig, ax = plt.subplots()
    for label, (fpr, tpr, auc_val) in roc_data.items():
        ax.plot(fpr, tpr, label=f'{label} (AUC={auc_val:.2f})')
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    if title:
        ax.set_title(title)
    ax.legend(loc='lower right')
    fig.savefig(output_path)
    plt.close(fig)


def save_metrics_list(metrics_list, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if not metrics_list:
        return
    keys = metrics_list[0].keys()
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(metrics_list)
