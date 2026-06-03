import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch_geometric.data import Batch

from config import AA_TO_IDX, ProjectConfig
from data_processing import peptide_to_graph
from self_training import generate_and_filter_samples


def save_predicted_positives(
    discriminator,
    unlabeled_peptides,
    device,
    confidence_threshold=0.9,
    output_file="predicted_positive_peptides.csv",
):
    discriminator.eval()
    high_confidence = []
    batch_size = 64
    idx_to_aa = {idx: aa for aa, idx in AA_TO_IDX.items()}

    with torch.no_grad():
        for start in range(0, len(unlabeled_peptides), batch_size):
            batch_peptides = unlabeled_peptides[start:start + batch_size]
            batch_graphs = [peptide_to_graph(peptide) for peptide in batch_peptides]
            batch_data = Batch.from_data_list(batch_graphs).to(device)
            _, pu_out = discriminator(batch_data, get_only_pu=True)
            probs = torch.sigmoid(pu_out).cpu().numpy().flatten()

            for peptide, score in zip(batch_peptides, probs):
                if score > confidence_threshold:
                    high_confidence.append({
                        "Peptide": peptide,
                        "Length": len(peptide),
                        "Confidence": float(score),
                    })

    if high_confidence:
        result_df = pd.DataFrame(high_confidence).sort_values("Confidence", ascending=False)
        result_df.to_csv(output_file, index=False)
        print(f"Saved {len(high_confidence)} peptides with confidence > {confidence_threshold} to {output_file}")
    else:
        print("No peptides met the confidence threshold.")


def analyze_self_training_samples(
    discriminator,
    generator,
    config: ProjectConfig,
    device,
    num_samples=100,
    min_confidence=None,
):
    if min_confidence is None:
        min_confidence = config.min_confidence_threshold

    filtered_latents, confidence_scores = generate_and_filter_samples(
        generator,
        discriminator,
        num_samples=num_samples * 5,
        noise_dim=config.noise_dim,
        latent_dim=config.latent_dim,
        min_confidence=min_confidence,
        device=device,
    )

    if len(filtered_latents) == 0:
        print("No high-confidence generated samples found for analysis")
        return None

    with torch.no_grad():
        features = discriminator.get_adv_features(filtered_latents)
        features_np = features.cpu().numpy()
        latents_np = filtered_latents.cpu().numpy()
        confidence_np = confidence_scores.cpu().numpy()

    results = {
        "confidence": confidence_np,
        "latent_norm": np.linalg.norm(latents_np, axis=1),
        "feature_norm": np.linalg.norm(features_np, axis=1),
    }
    for i in range(min(10, features_np.shape[1])):
        results[f"feature_{i}"] = features_np[:, i]

    df = pd.DataFrame(results)
    print(f"Self-training samples: {len(df)}")
    print(f"Confidence mean={df['confidence'].mean():.4f}, std={df['confidence'].std():.4f}")
    print(f"Latent norm mean={df['latent_norm'].mean():.4f}, std={df['latent_norm'].std():.4f}")

    plt.figure(figsize=(10, 6))
    plt.hist(df["confidence"], bins=20)
    plt.xlabel("Confidence Score")
    plt.ylabel("Count")
    plt.title("Confidence Scores in Generated Samples")
    plt.show()

    return df


def analyze_model_predictions(discriminator, test_loader, device, threshold=0.5):
    discriminator.eval()
    results = []
    idx_to_aa = {idx: aa for aa, idx in AA_TO_IDX.items()}

    with torch.no_grad():
        for data, labels in test_loader:
            data = data.to(device)
            labels = labels.to(device)
            _, pu_out = discriminator(data, get_only_pu=True)
            probs = torch.sigmoid(pu_out).cpu().numpy().flatten()

            for graph, label, score in zip(data.to_data_list(), labels.cpu().numpy(), probs):
                indices = graph.x.cpu().numpy().flatten()
                peptide = "".join(idx_to_aa[int(idx)] for idx in indices)
                results.append({
                    "Peptide": peptide,
                    "Length": len(peptide),
                    "Predicted_Score": float(score),
                    "True_Label": int(label),
                })

    results_df = pd.DataFrame(results)
    if results_df.empty:
        return results_df

    results_df["Hydrophobic_Ratio"] = results_df["Peptide"].apply(
        lambda seq: sum(aa in "AVILMFYW" for aa in seq) / len(seq)
    )
    results_df["Charged_Ratio"] = results_df["Peptide"].apply(
        lambda seq: sum(aa in "DEKR" for aa in seq) / len(seq)
    )
    results_df["Error"] = ((results_df["Predicted_Score"] >= threshold).astype(int) != results_df["True_Label"]).astype(int)

    corr_matrix = results_df[["Length", "Hydrophobic_Ratio", "Charged_Ratio", "Predicted_Score", "True_Label"]].corr()
    print("Correlation with prediction score:")
    print(corr_matrix["Predicted_Score"].sort_values(ascending=False))

    error_df = results_df[results_df["Error"] == 1]
    if not error_df.empty:
        false_positives = sum((error_df["True_Label"] == 0) & (error_df["Predicted_Score"] >= threshold))
        false_negatives = sum((error_df["True_Label"] == 1) & (error_df["Predicted_Score"] < threshold))
        print(f"False positives: {false_positives}")
        print(f"False negatives: {false_negatives}")

    return results_df


def plot_feature_importance(prediction_analysis, threshold=0.5):
    if prediction_analysis is None or len(prediction_analysis) == 0:
        print("No prediction analysis data available")
        return

    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    plt.scatter(prediction_analysis["Length"], prediction_analysis["Predicted_Score"], c=prediction_analysis["True_Label"], alpha=0.6)
    plt.colorbar(label="True Label")
    plt.xlabel("Peptide Length")
    plt.ylabel("Predicted Score")
    plt.title("Prediction Score vs Peptide Length")

    plt.subplot(2, 2, 2)
    plt.scatter(prediction_analysis["Hydrophobic_Ratio"], prediction_analysis["Predicted_Score"], c=prediction_analysis["True_Label"], alpha=0.6)
    plt.colorbar(label="True Label")
    plt.xlabel("Hydrophobic Ratio")
    plt.ylabel("Predicted Score")
    plt.title("Prediction Score vs Hydrophobic Ratio")

    plt.subplot(2, 2, 3)
    plt.scatter(prediction_analysis["Charged_Ratio"], prediction_analysis["Predicted_Score"], c=prediction_analysis["True_Label"], alpha=0.6)
    plt.colorbar(label="True Label")
    plt.xlabel("Charged Ratio")
    plt.ylabel("Predicted Score")
    plt.title("Prediction Score vs Charged Ratio")

    plt.subplot(2, 2, 4)
    positives = prediction_analysis[prediction_analysis["True_Label"] == 1]["Predicted_Score"]
    negatives = prediction_analysis[prediction_analysis["True_Label"] == 0]["Predicted_Score"]
    plt.hist(positives, bins=20, alpha=0.6, label="Positive")
    plt.hist(negatives, bins=20, alpha=0.6, label="Negative benchmark")
    plt.axvline(threshold, linestyle="--", label=f"Threshold={threshold:.2f}")
    plt.xlabel("Predicted Score")
    plt.ylabel("Count")
    plt.title("Prediction Score Distribution")
    plt.legend()

    plt.tight_layout()
    plt.show()
