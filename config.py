from dataclasses import dataclass
from typing import Tuple


NATURAL_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")
AA_LIST = sorted(NATURAL_AMINO_ACIDS)
AA_TO_IDX = {aa: idx for idx, aa in enumerate(AA_LIST)}
NUM_NODE_TOKENS = len(AA_TO_IDX)


@dataclass
class ProjectConfig:
    positive_data_path: str = "data/train.csv"
    unlabeled_train_path: str = "data/unlabeled.xlsx"
    external_negative_pool_path: str = "data/external_negative_pool.xlsx"
    output_dir: str = "models"
    predicted_positive_output: str = "predicted_positives_validation_threshold.csv"

    random_state: int = 42
    outer_test_size: float = 0.20
    calibration_positive_fraction: float = 0.10
    negative_to_positive_ratio: float = 1.0

    sequence_similarity_threshold: float = 0.40
    sequence_filter_top_k: int = 3
    sequence_gray_zone: float = 0.10
    sequence_filter_ngram_range: Tuple[int, int] = (2, 3)
    strict_sequence_filter: bool = False

    pca_variance_to_keep: float = 0.95
    max_pca_components: int = 32
    distance_quantiles: Tuple[float, ...] = (0.99, 0.975, 0.95, 0.90)
    require_exact_benchmark_balance: bool = True

    class_prior_min: float = 0.01
    class_prior_max: float = 0.95
    class_prior_estimation_splits: int = 5

    latent_dim: int = 100
    noise_dim: int = 50
    node_emb_dim: int = 50
    hidden_dim: int = 100
    num_transformer_layers: int = 1
    classification_threshold: float = 0.5

    feature_matching_weight: float = 1.0
    generator_pu_feedback_weight: float = 0.5
    pu_loss_weight: float = 1.0

    min_confidence_threshold: float = 0.75
    max_self_train_samples: int = 5000
    self_train_update_freq: int = 1
    diversity_threshold: float = 0.15

    regular_folds: int = 5
    batch_size: int = 32
    cv_epochs: int = 60
    full_epochs: int = 80
    num_generator_updates: int = 2
    curriculum_epochs: int = 15
    gan_warmup_epochs: int = 8
    adversarial_loss_type: str = "hinge"
