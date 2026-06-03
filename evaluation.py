import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from torch_geometric.loader import DataLoader as GeoDataLoader


def find_optimal_threshold(probabilities, labels, metric="f1", num_thresholds=101, default_threshold=0.5):
    best_threshold = default_threshold
    best_metric_val = -np.inf

    for threshold in np.linspace(0.01, 0.99, num_thresholds):
        preds = (probabilities >= threshold).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average="binary", zero_division=1
        )
        if metric == "precision":
            metric_val = precision
        elif metric == "recall":
            metric_val = recall
        elif metric == "balanced_accuracy":
            metric_val = (precision + recall) / 2
        else:
            metric_val = f1

        if metric_val > best_metric_val:
            best_metric_val = metric_val
            best_threshold = threshold

    return best_threshold, best_metric_val


def evaluate_model(discriminator, test_dataset, device, threshold=0.5, compute_optimal_threshold=False):
    test_loader = GeoDataLoader(test_dataset, batch_size=32, shuffle=False)
    discriminator.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for data, labels in test_loader:
            data = data.to(device)
            labels = labels.to(device)
            _, pu_out = discriminator(data, get_only_pu=True)
            probs = torch.sigmoid(pu_out).cpu().numpy().flatten()
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())

    all_probs = np.array(all_probs).flatten()
    all_labels = np.array(all_labels).flatten()
    all_preds = (all_probs >= threshold).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary", zero_division=1
    )
    auc = roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else float("nan")
    accuracy = accuracy_score(all_labels, all_preds)

    metrics = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "accuracy": accuracy,
        "selected_threshold": threshold,
        "probs": all_probs,
        "labels": all_labels,
        "preds": all_preds,
    }

    if compute_optimal_threshold and len(np.unique(all_labels)) > 1:
        optimal_threshold, best_f1 = find_optimal_threshold(all_probs, all_labels, default_threshold=threshold)
        metrics["optimal_threshold"] = optimal_threshold
        metrics["best_f1"] = best_f1

    return metrics


def evaluate_ensemble(fold_models, test_dataset, device, threshold=0.5):
    test_loader = GeoDataLoader(test_dataset, batch_size=32, shuffle=False)
    all_fold_probs = []
    all_labels = []

    for _, discriminator in fold_models:
        discriminator.eval()
        fold_probs = []
        collect_labels = len(all_labels) == 0

        with torch.no_grad():
            for data, labels in test_loader:
                data = data.to(device)
                labels = labels.to(device)
                _, pu_out = discriminator(data, get_only_pu=True)
                probs = torch.sigmoid(pu_out).cpu().numpy().flatten()
                fold_probs.extend(probs)
                if collect_labels:
                    all_labels.extend(labels.cpu().numpy())

        all_fold_probs.append(fold_probs)

    ensemble_probs = np.mean(np.array(all_fold_probs), axis=0)
    all_labels = np.array(all_labels).flatten()
    ensemble_preds = (ensemble_probs >= threshold).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, ensemble_preds, average="binary", zero_division=1
    )
    auc = roc_auc_score(all_labels, ensemble_probs) if len(np.unique(all_labels)) > 1 else float("nan")
    accuracy = accuracy_score(all_labels, ensemble_preds)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "accuracy": accuracy,
        "selected_threshold": threshold,
        "probs": ensemble_probs,
        "labels": all_labels,
        "preds": ensemble_preds,
    }


def plot_metrics(history, test_results):
    if not history.get("val_total_loss"):
        return

    epoch_range = range(1, len(history["val_total_loss"]) + 1)
    plt.figure(figsize=(18, 5))

    plt.subplot(1, 3, 1)
    plt.plot(epoch_range, history["val_adv_loss"], label="Val Adv Loss")
    plt.plot(epoch_range, history["val_pu_loss"], label="Val PU Loss")
    plt.plot(epoch_range, history["val_total_loss"], label="Val Total Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Validation Losses")
    plt.legend()

    if len(np.unique(test_results["labels"])) > 1:
        fpr, tpr, _ = roc_curve(test_results["labels"], test_results["probs"])
        auc = roc_auc_score(test_results["labels"], test_results["probs"])
        plt.subplot(1, 3, 2)
        plt.plot(fpr, tpr, marker=".", linestyle="-")
        plt.title(f"ROC Curve (AUC: {auc:.4f})")
        plt.xlabel("FPR")
        plt.ylabel("TPR")

    cm = confusion_matrix(test_results["labels"], np.array(test_results["preds"]))
    plt.subplot(1, 3, 3)
    disp = ConfusionMatrixDisplay(cm, display_labels=[0, 1])
    disp.plot(ax=plt.gca())
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()
