import argparse
import os

import numpy as np
import torch
from sklearn.model_selection import KFold
from torch_geometric.loader import DataLoader as GeoDataLoader

from analysis_tools import analyze_model_predictions, plot_feature_importance, save_predicted_positives
from benchmark import build_reliable_negative_benchmark, estimate_class_prior_elkan_noto
from config import AA_LIST, ProjectConfig
from data_processing import GraphPeptideDataset, load_project_data, peptide_to_graph
from evaluation import evaluate_ensemble, evaluate_model, plot_metrics
from training import default_model_config, train_model
from utils import ensure_dir, get_device, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train a leakage-safe GAN-PU peptide classifier.")
    parser.add_argument("--positive-data", default=None, help="CSV/XLSX file with known positive peptide sequences.")
    parser.add_argument("--unlabeled-data", default=None, help="CSV/XLSX file with unlabeled peptide sequences.")
    parser.add_argument("--negative-pool", default=None, help="CSV/XLSX file used only for reliable negative benchmark mining.")
    parser.add_argument("--output-dir", default=None, help="Directory for model checkpoints.")
    parser.add_argument("--no-plots", action="store_true", help="Disable matplotlib plots at the end.")
    return parser.parse_args()


def build_dataset(positive_sequences, negative_sequences=None, unlabeled_graphs=None):
    graphs = [peptide_to_graph(p) for p in positive_sequences]
    labels = [1.0] * len(positive_sequences)

    if negative_sequences is not None:
        graphs.extend(peptide_to_graph(p) for p in negative_sequences)
        labels.extend([0.0] * len(negative_sequences))

    if unlabeled_graphs is not None:
        graphs.extend(unlabeled_graphs)
        labels.extend([0.0] * len(unlabeled_graphs))

    return GraphPeptideDataset(graphs, np.array(labels, dtype=np.float32))


def main():
    args = parse_args()
    config = ProjectConfig()

    if args.positive_data:
        config.positive_data_path = args.positive_data
    if args.unlabeled_data:
        config.unlabeled_train_path = args.unlabeled_data
    if args.negative_pool:
        config.external_negative_pool_path = args.negative_pool
    if args.output_dir:
        config.output_dir = args.output_dir

    set_seed(config.random_state)
    device = get_device()
    ensure_dir(config.output_dir)

    print("Starting PU-GAN training with a leakage-safe reliable negative benchmark")
    print(f"Using device: {device}")

    data = load_project_data(config)
    positive_cv = data["positive_cv"]
    positive_calibration = data["positive_calibration"]
    positive_train_all = data["positive_train_all"]
    positive_test = data["positive_test"]
    unlabeled_peptides = data["unlabeled_peptides"]
    unlabeled_graphs = data["unlabeled_graphs"]
    benchmark_val_pool = data["benchmark_val_pool"]
    benchmark_test_pool = data["benchmark_test_pool"]

    print(f"Positives total: {len(data['positive_peptides'])}")
    print(f"Training unlabeled total: {len(unlabeled_peptides)}")
    print(f"External benchmark candidates after overlap removal: {len(data['external_negative_pool'])}")
    print(f"CV positives: {len(positive_cv)} | Calibration positives: {len(positive_calibration)} | Test positives: {len(positive_test)}")

    all_fold_models = []
    all_fold_histories = []
    validation_thresholds = []
    model_config = default_model_config(config, full_training=False)
    full_model_config = default_model_config(config, full_training=True)

    kf = KFold(n_splits=config.regular_folds, shuffle=True, random_state=config.random_state)
    cv_indices = np.arange(len(positive_cv))

    print("\n=== Starting folds 1-5 on positive-only CV splits ===")
    for fold, (train_idx, val_idx) in enumerate(kf.split(cv_indices), start=1):
        print(f"\n=== Running Fold {fold}/{config.regular_folds + 1} ===")
        fold_train_positive = [positive_cv[i] for i in train_idx]
        fold_val_positive = [positive_cv[i] for i in val_idx]

        fold_prior_info = estimate_class_prior_elkan_noto(
            fold_train_positive,
            unlabeled_peptides,
            AA_LIST,
            n_splits=config.class_prior_estimation_splits,
            random_state=config.random_state + fold,
            min_prior=config.class_prior_min,
            max_prior=config.class_prior_max,
        )
        fold_val_negative = build_reliable_negative_benchmark(
            fold_train_positive,
            fold_val_positive,
            benchmark_val_pool,
            config,
            AA_LIST,
        )

        fold_train_dataset = build_dataset(fold_train_positive, unlabeled_graphs=unlabeled_graphs)
        fold_val_dataset = build_dataset(fold_val_positive, negative_sequences=fold_val_negative)

        generator, discriminator, history = train_model(
            config=config,
            train_dataset=fold_train_dataset,
            val_dataset=fold_val_dataset,
            output_dir=config.output_dir,
            fold=fold,
            class_prior_estimate=fold_prior_info["pi_p"],
            class_prior_details=fold_prior_info,
            model_config=model_config,
            device=device,
        )
        all_fold_models.append((generator, discriminator))
        all_fold_histories.append(history)

        val_metrics = evaluate_model(
            discriminator,
            fold_val_dataset,
            device,
            threshold=config.classification_threshold,
            compute_optimal_threshold=True,
        )
        validation_thresholds.append(val_metrics["optimal_threshold"])
        print(f"Fold {fold} validation F1: {val_metrics['f1']:.4f} | AUC: {val_metrics['auc']:.4f}")
        print(f"Validation-selected threshold: {val_metrics['optimal_threshold']:.4f}")

    print("\n=== Running Fold 6 with external calibration benchmark ===")
    calibration_negative = build_reliable_negative_benchmark(
        positive_cv,
        positive_calibration,
        benchmark_val_pool,
        config,
        AA_LIST,
    )
    fold6_prior_info = estimate_class_prior_elkan_noto(
        positive_cv,
        unlabeled_peptides,
        AA_LIST,
        n_splits=config.class_prior_estimation_splits,
        random_state=config.random_state + config.regular_folds + 1,
        min_prior=config.class_prior_min,
        max_prior=config.class_prior_max,
    )

    fold6_train_dataset = build_dataset(positive_cv, unlabeled_graphs=unlabeled_graphs)
    fold6_val_dataset = build_dataset(positive_calibration, negative_sequences=calibration_negative)

    generator, discriminator, history = train_model(
        config=config,
        train_dataset=fold6_train_dataset,
        val_dataset=fold6_val_dataset,
        output_dir=config.output_dir,
        fold=config.regular_folds + 1,
        class_prior_estimate=fold6_prior_info["pi_p"],
        class_prior_details=fold6_prior_info,
        model_config=full_model_config,
        device=device,
    )
    all_fold_models.append((generator, discriminator))
    all_fold_histories.append(history)

    fold6_val_metrics = evaluate_model(
        discriminator,
        fold6_val_dataset,
        device,
        threshold=config.classification_threshold,
        compute_optimal_threshold=True,
    )
    validation_thresholds.append(fold6_val_metrics["optimal_threshold"])
    ensemble_threshold = float(np.median(validation_thresholds)) if validation_thresholds else config.classification_threshold
    print(f"Median validation threshold across folds: {ensemble_threshold:.4f}")

    torch.save(
        {"generator": generator.state_dict(), "discriminator": discriminator.state_dict()},
        os.path.join(config.output_dir, "final_model.pth"),
    )

    print("\n=== Building final held-out benchmark ===")
    heldout_negative = build_reliable_negative_benchmark(
        positive_train_all,
        positive_test,
        benchmark_test_pool,
        config,
        AA_LIST,
    )
    if len(heldout_negative) != len(positive_test):
        raise ValueError("Held-out benchmark must be balanced.")

    test_dataset = build_dataset(positive_test, negative_sequences=heldout_negative)
    print(f"Held-out positives: {len(positive_test)} | Held-out benchmark negatives: {len(heldout_negative)}")

    print("\n=== Final Test Set Evaluation ===")
    test_metrics = evaluate_ensemble(all_fold_models, test_dataset, device, threshold=ensemble_threshold)
    print("Final Test Set Results (Ensemble):")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall: {test_metrics['recall']:.4f}")
    print(f"F1 Score: {test_metrics['f1']:.4f}")
    print(f"ROC AUC: {test_metrics['auc']:.4f}")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Fixed threshold from validation: {test_metrics['selected_threshold']:.4f}")

    fold6_metrics = evaluate_model(discriminator, test_dataset, device, threshold=ensemble_threshold)
    print("\nFold 6 Model Test Results:")
    print(f"Precision: {fold6_metrics['precision']:.4f}")
    print(f"Recall: {fold6_metrics['recall']:.4f}")
    print(f"F1 Score: {fold6_metrics['f1']:.4f}")
    print(f"ROC AUC: {fold6_metrics['auc']:.4f}")
    print(f"Accuracy: {fold6_metrics['accuracy']:.4f}")

    if not args.no_plots:
        plot_metrics(history, {"labels": fold6_metrics["labels"], "probs": fold6_metrics["probs"], "preds": fold6_metrics["preds"]})
        test_loader = GeoDataLoader(test_dataset, batch_size=32, shuffle=False)
        prediction_analysis = analyze_model_predictions(discriminator, test_loader, device, threshold=ensemble_threshold)
        plot_feature_importance(prediction_analysis, threshold=ensemble_threshold)

    save_predicted_positives(
        discriminator,
        unlabeled_peptides,
        device,
        confidence_threshold=ensemble_threshold,
        output_file=config.predicted_positive_output,
    )

    print(f"\nCheck the '{config.output_dir}' directory for saved models.")
    print(f"Check '{config.predicted_positive_output}' for predicted positive peptides.")
    return test_metrics, fold6_metrics


if __name__ == "__main__":
    main()
