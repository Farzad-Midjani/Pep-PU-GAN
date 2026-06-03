import math
from typing import List, Sequence

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import AA_LIST, ProjectConfig
from data_processing import build_feature_matrix, clean_and_filter_sequences


def estimate_class_prior_elkan_noto(
    positive_peptides: Sequence[str],
    unlabeled_peptides: Sequence[str],
    aa_list: Sequence[str] = AA_LIST,
    n_splits: int = 5,
    random_state: int = 42,
    min_prior: float = 0.01,
    max_prior: float = 0.95,
) -> dict:
    positive_peptides = clean_and_filter_sequences(positive_peptides)
    unlabeled_peptides = clean_and_filter_sequences(unlabeled_peptides)
    n_pos = len(positive_peptides)
    n_unl = len(unlabeled_peptides)
    label_frequency = float(n_pos) / max(float(n_pos + n_unl), 1.0)

    if n_pos < 2 or n_unl < 2:
        fallback = float(np.clip(label_frequency, min_prior, max_prior))
        return {
            "pi_p": fallback,
            "c_hat": fallback,
            "label_frequency": label_frequency,
            "method": "observed_label_frequency",
            "n_pos": n_pos,
            "n_unl": n_unl,
            "n_splits": 1,
        }

    X_pos = build_feature_matrix(positive_peptides, aa_list)
    X_unl = build_feature_matrix(unlabeled_peptides, aa_list)
    X = np.vstack([X_pos, X_unl]).astype(np.float32)
    s = np.concatenate([np.ones(n_pos, dtype=np.int32), np.zeros(n_unl, dtype=np.int32)])

    valid_splits = int(min(n_splits, n_pos, n_unl))
    if valid_splits < 2:
        fallback = float(np.clip(label_frequency, min_prior, max_prior))
        return {
            "pi_p": fallback,
            "c_hat": fallback,
            "label_frequency": label_frequency,
            "method": "observed_label_frequency",
            "n_pos": n_pos,
            "n_unl": n_unl,
            "n_splits": 1,
        }

    skf = StratifiedKFold(n_splits=valid_splits, shuffle=True, random_state=random_state)
    oof_probs = np.zeros(len(s), dtype=np.float64)

    for fold_idx, (train_idx, valid_idx) in enumerate(skf.split(X, s), start=1):
        selector = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, solver="lbfgs", random_state=random_state + fold_idx)),
        ])
        selector.fit(X[train_idx], s[train_idx])
        oof_probs[valid_idx] = selector.predict_proba(X[valid_idx])[:, 1]

    c_hat = float(np.clip(oof_probs[:n_pos].mean(), 1e-4, 1.0))
    pi_p = float(np.clip(label_frequency / c_hat, min_prior, max_prior))

    return {
        "pi_p": pi_p,
        "c_hat": c_hat,
        "label_frequency": label_frequency,
        "method": "elkan_noto_oof_logistic",
        "n_pos": n_pos,
        "n_unl": n_unl,
        "n_splits": valid_splits,
    }


def fit_positive_manifold(
    positive_peptides: Sequence[str],
    aa_list: Sequence[str],
    variance_to_keep: float,
    max_components: int,
    random_state: int,
) -> dict:
    X_pos = build_feature_matrix(positive_peptides, aa_list)
    n_components = min(max_components, X_pos.shape[0] - 1, X_pos.shape[1])
    if n_components < 2:
        raise ValueError("Not enough positive peptides to fit the positive manifold.")

    pca = PCA(n_components=n_components, random_state=random_state)
    X_pca = pca.fit_transform(X_pos)
    explained = np.cumsum(pca.explained_variance_ratio_)
    keep = np.searchsorted(explained, variance_to_keep) + 1
    keep = max(2, min(keep, X_pca.shape[1]))
    X_reduced = X_pca[:, :keep]
    covariance_model = LedoitWolf().fit(X_reduced)

    return {
        "pca": pca,
        "keep": keep,
        "covariance_model": covariance_model,
        "positive_distances": covariance_model.mahalanobis(X_reduced),
    }


def transform_mahalanobis(peptides: Sequence[str], manifold_model: dict, aa_list: Sequence[str]) -> np.ndarray:
    X = build_feature_matrix(peptides, aa_list)
    X_pca = manifold_model["pca"].transform(X)[:, :manifold_model["keep"]]
    return manifold_model["covariance_model"].mahalanobis(X_pca)


def levenshtein_distance(seq_a: str, seq_b: str) -> int:
    if seq_a == seq_b:
        return 0
    if len(seq_a) < len(seq_b):
        seq_a, seq_b = seq_b, seq_a

    previous = list(range(len(seq_b) + 1))
    for i, aa in enumerate(seq_a, start=1):
        current = [i]
        for j, bb in enumerate(seq_b, start=1):
            insertion = current[j - 1] + 1
            deletion = previous[j] + 1
            substitution = previous[j - 1] + (aa != bb)
            current.append(min(insertion, deletion, substitution))
        previous = current
    return previous[-1]


def normalized_edit_similarity(seq_a: str, seq_b: str) -> float:
    denom = max(len(seq_a), len(seq_b), 1)
    return 1.0 - levenshtein_distance(seq_a, seq_b) / denom


def fit_sequence_similarity_index(reference_sequences: Sequence[str], ngram_range=(2, 3)) -> dict:
    if not reference_sequences:
        raise ValueError("reference_sequences cannot be empty.")

    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=ngram_range,
        lowercase=False,
        norm="l2",
        dtype=np.float32,
    )
    reference_matrix = vectorizer.fit_transform(reference_sequences)
    return {
        "vectorizer": vectorizer,
        "reference_matrix": reference_matrix,
        "reference_sequences": list(reference_sequences),
    }


def run_sequence_filter(
    reference_sequences: Sequence[str],
    candidate_sequences: Sequence[str],
    similarity_threshold: float,
    top_k: int,
    ngram_range=(2, 3),
    gray_zone: float = 0.10,
    chunk_size: int = 2048,
    strict: bool = False,
    return_metadata: bool = False,
):
    if not reference_sequences:
        raise ValueError("reference_sequences cannot be empty.")
    if not candidate_sequences:
        return []

    similarity_index = fit_sequence_similarity_index(reference_sequences, ngram_range=ngram_range)
    vectorizer = similarity_index["vectorizer"]
    reference_matrix = similarity_index["reference_matrix"]
    reference_sequences = similarity_index["reference_sequences"]
    candidate_matrix = vectorizer.transform(candidate_sequences)

    n_neighbors = min(max(1, int(top_k)), reference_matrix.shape[0])
    nn_index = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine", algorithm="brute")
    nn_index.fit(reference_matrix)

    accepted = []
    gray_low = max(0.0, similarity_threshold - max(0.0, gray_zone))
    gray_high = min(1.0, similarity_threshold + max(0.0, gray_zone))
    accepted_fast = rejected_fast = gray_checked = 0

    for start in range(0, candidate_matrix.shape[0], chunk_size):
        stop = min(start + chunk_size, candidate_matrix.shape[0])
        dist_chunk, idx_chunk = nn_index.kneighbors(candidate_matrix[start:stop], return_distance=True)

        for offset, (dist_row, idx_row) in enumerate(zip(dist_chunk, idx_chunk)):
            peptide = candidate_sequences[start + offset]
            nearest_cosine = 1.0 - float(dist_row[0])

            if nearest_cosine < gray_low:
                accepted_fast += 1
                record = {
                    "sequence": peptide,
                    "max_sequence_similarity": float(nearest_cosine),
                    "screen_similarity": float(nearest_cosine),
                    "edit_similarity": None,
                }
                accepted.append(record if return_metadata else peptide)
                continue

            if nearest_cosine > gray_high:
                rejected_fast += 1
                continue

            gray_checked += 1
            max_edit_similarity = max(
                normalized_edit_similarity(peptide, reference_sequences[int(ref_idx)])
                for ref_idx in idx_row
            )
            conservative_similarity = max(float(nearest_cosine), float(max_edit_similarity))
            if conservative_similarity < similarity_threshold:
                record = {
                    "sequence": peptide,
                    "max_sequence_similarity": conservative_similarity,
                    "screen_similarity": float(nearest_cosine),
                    "edit_similarity": float(max_edit_similarity),
                }
                accepted.append(record if return_metadata else peptide)

    if strict and not accepted:
        raise ValueError("No candidates remained after sequence filtering.")

    print(
        f"Sequence filter kept {len(accepted)} / {len(candidate_sequences)} candidates "
        f"(fast accepted={accepted_fast}, fast rejected={rejected_fast}, "
        f"gray-zone checked={gray_checked}, threshold={similarity_threshold:.2f})."
    )
    return accepted


def percentile_rank(values, higher_is_better=True):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return np.array([], dtype=np.float64)
    if values.size == 1:
        return np.ones(1, dtype=np.float64)

    order = np.argsort(values)
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.arange(values.size, dtype=np.float64)
    percentiles = ranks / float(values.size - 1)
    return percentiles if higher_is_better else 1.0 - percentiles


def add_reliability_scores(candidate_records: List[dict]) -> List[dict]:
    if not candidate_records:
        return []

    distances = np.array([r["distance_from_positive_manifold"] for r in candidate_records], dtype=np.float64)
    similarities = np.array([r["max_sequence_similarity"] for r in candidate_records], dtype=np.float64)
    distance_rank = percentile_rank(distances, higher_is_better=True)
    dissimilarity_rank = percentile_rank(similarities, higher_is_better=False)

    for idx, record in enumerate(candidate_records):
        record["distance_rank"] = float(distance_rank[idx])
        record["dissimilarity_rank"] = float(dissimilarity_rank[idx])
        record["reliability_score"] = float(0.6 * distance_rank[idx] + 0.4 * dissimilarity_rank[idx])
    return candidate_records


def select_top_ranked_length_matched_negatives(
    positive_peptides: Sequence[str],
    candidate_records: Sequence[dict],
    ratio: float,
) -> List[str]:
    target_n = int(math.ceil(len(positive_peptides) * ratio))
    if target_n == 0:
        return []
    if len(candidate_records) < target_n:
        raise ValueError(f"Only {len(candidate_records)} candidate negatives are available; {target_n} are required.")

    desired_lengths = [len(seq) for seq in positive_peptides]
    if len(desired_lengths) < target_n and desired_lengths:
        reps = int(math.ceil(target_n / len(desired_lengths)))
        desired_lengths = (desired_lengths * reps)[:target_n]
    else:
        desired_lengths = desired_lengths[:target_n]

    by_length = {}
    for record in candidate_records:
        by_length.setdefault(len(record["sequence"]), []).append(record)

    for length_value in by_length:
        by_length[length_value].sort(
            key=lambda rec: (
                -rec["reliability_score"],
                rec["max_sequence_similarity"],
                -rec["distance_from_positive_manifold"],
                rec["sequence"],
            )
        )

    selected = []
    for target_len in desired_lengths:
        available_lengths = [length for length, pool in by_length.items() if pool]
        if not available_lengths:
            break
        best_length = min(
            available_lengths,
            key=lambda length: (
                abs(length - target_len),
                -by_length[length][0]["reliability_score"],
                by_length[length][0]["max_sequence_similarity"],
                -by_length[length][0]["distance_from_positive_manifold"],
                length,
            ),
        )
        selected.append(by_length[best_length].pop(0))

    selected_sequences = clean_and_filter_sequences([record["sequence"] for record in selected])
    if len(selected_sequences) != target_n:
        raise ValueError(f"Expected {target_n} negatives, obtained {len(selected_sequences)}.")
    return selected_sequences


def build_reliable_negative_benchmark(
    reference_positive_peptides: Sequence[str],
    target_positive_peptides: Sequence[str],
    candidate_pool: Sequence[str],
    config: ProjectConfig,
    aa_list: Sequence[str] = AA_LIST,
) -> List[str]:
    target_n = int(math.ceil(len(target_positive_peptides) * config.negative_to_positive_ratio))
    if target_n == 0:
        return []

    candidate_pool = clean_and_filter_sequences(candidate_pool)
    reference_positive_peptides = clean_and_filter_sequences(reference_positive_peptides)
    target_positive_peptides = clean_and_filter_sequences(target_positive_peptides)

    seq_filtered_records = run_sequence_filter(
        reference_positive_peptides,
        candidate_pool,
        similarity_threshold=config.sequence_similarity_threshold,
        top_k=config.sequence_filter_top_k,
        ngram_range=config.sequence_filter_ngram_range,
        gray_zone=config.sequence_gray_zone,
        strict=config.strict_sequence_filter,
        return_metadata=True,
    )
    if not seq_filtered_records:
        raise ValueError("No candidates remained after sequence filtering.")

    manifold = fit_positive_manifold(
        reference_positive_peptides,
        aa_list,
        variance_to_keep=config.pca_variance_to_keep,
        max_components=config.max_pca_components,
        random_state=config.random_state,
    )
    positive_distances = manifold["positive_distances"]
    seq_filtered_sequences = [record["sequence"] for record in seq_filtered_records]
    candidate_distances = transform_mahalanobis(seq_filtered_sequences, manifold, aa_list)

    candidate_records = []
    for record, distance in zip(seq_filtered_records, candidate_distances):
        merged = dict(record)
        merged["distance_from_positive_manifold"] = float(distance)
        candidate_records.append(merged)

    candidate_records = add_reliability_scores(candidate_records)
    chosen_records = []

    for q in config.distance_quantiles:
        cutoff = np.quantile(positive_distances, q)
        eligible_records = [r for r in candidate_records if r["distance_from_positive_manifold"] > cutoff]
        if len(eligible_records) >= target_n:
            chosen_records = eligible_records
            print(f"Distance cutoff succeeded at q={q:.3f} with {len(eligible_records)} eligible negatives.")
            break

    if len(chosen_records) < target_n:
        chosen_records = sorted(
            candidate_records,
            key=lambda rec: (
                -rec["reliability_score"],
                rec["max_sequence_similarity"],
                -rec["distance_from_positive_manifold"],
                rec["sequence"],
            ),
        )
        print("WARNING: distance cutoffs were too sparse; using reliability ranking.")

    selected_negatives = select_top_ranked_length_matched_negatives(
        positive_peptides=target_positive_peptides,
        candidate_records=chosen_records,
        ratio=config.negative_to_positive_ratio,
    )

    if config.require_exact_benchmark_balance and len(selected_negatives) != target_n:
        raise ValueError(f"Balanced benchmark requires exactly {target_n} negatives.")

    print(f"Selected {len(selected_negatives)} reliable negatives for {len(target_positive_peptides)} positives.")
    return selected_negatives
