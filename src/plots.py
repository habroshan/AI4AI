import os
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# === ROC plotting function ===
def plot_integrated_roc(y_true, scores_dict, title, outfile):
    plt.figure(figsize=(7, 6))
    for name, scores in scores_dict.items():
        fpr, tpr, _ = roc_curve(y_true, scores)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], color="grey", lw=1.5, ls="--")
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(title); plt.legend(loc="lower right"); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(outfile, dpi=300); plt.close()

# === Confusion Matrix plotting function ===
def plot_confusion_matrix(y_true, y_pred, classes, title, outfile, normalize=False):
    cm = confusion_matrix(y_true, y_pred)
    if normalize: cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title, ylabel="True label", xlabel="Predicted label")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fmt = ".2f" if normalize else "d"; thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout(); plt.savefig(outfile, dpi=300, bbox_inches="tight"); plt.close()

# === Main driver ===
datasets = ["mnist", "cifar10", "chestxray14"]
detectors = ["xgb", "mlp", "if", "maha"]
classes = ["Clean", "Poisoned"]

for dataset in datasets:
    print(f"Processing {dataset}...")
    # Load ground truth
    y_true = np.load(f"results/{dataset}/y_true.npy")

    # Collect ROC scores
    scores_dict = {}
    for det in detectors:
        scores = np.load(f"results/{dataset}/{det}_scores.npy")
        scores_dict[det.upper()] = scores

        # Also confusion matrix
        y_pred = np.load(f"results/{dataset}/{det}_preds.npy")
        plot_confusion_matrix(
            y_true, y_pred, classes,
            title=f"Confusion Matrix for {dataset.upper()} ({det.upper()})",
            outfile=f"figures/CM_{dataset}_{det}.png"
        )

    # Integrated ROC
    plot_integrated_roc(
        y_true, scores_dict,
        title=f"ROC Curves for {dataset.upper()} (all detectors)",
        outfile=f"figures/ROC_{dataset}.png"
    )
