from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torch_geometric.data import Data

from config import AA_LIST, AA_TO_IDX, NATURAL_AMINO_ACIDS, ProjectConfig


def is_natural(peptide: str) -> bool:
    return set(peptide).issubset(NATURAL_AMINO_ACIDS)


def clean_and_filter_sequences(sequences: Iterable[str]) -> List[str]:
    cleaned = []
    seen = set()
    for peptide in sequences:
        peptide = str(peptide).strip().upper()
        if not peptide or peptide == "NAN":
            continue
        if not (5 < len(peptide) < 51):
            continue
        if not is_natural(peptide):
            continue
        if peptide not in seen:
            cleaned.append(peptide)
            seen.add(peptide)
    return cleaned


def load_sequences_from_table(path: str) -> List[str]:
    path_obj = Path(path)
    if path_obj.suffix.lower() == ".csv":
        df = pd.read_csv(path_obj)
    else:
        df = pd.read_excel(path_obj)

    preferred_cols = [
        c for c in df.columns
        if str(c).strip().lower() in {"sequence", "peptide", "seq"}
    ]
    if preferred_cols:
        raw = df[preferred_cols[0]].dropna().astype(str).tolist()
    else:
        raw = [str(value) for value in df.values.flatten() if isinstance(value, str) and value.strip()]
    return clean_and_filter_sequences(raw)


def peptide_to_graph(peptide: str, aa_to_idx: dict = AA_TO_IDX) -> Data:
    node_indices = [aa_to_idx[aa] for aa in peptide]
    x = torch.tensor(node_indices, dtype=torch.long).view(-1, 1)

    if len(node_indices) > 1:
        edges = []
        for i in range(len(node_indices) - 1):
            edges.append([i, i + 1])
            edges.append([i + 1, i])
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    return Data(x=x, edge_index=edge_index)


def build_feature_matrix(peptides: Sequence[str], aa_list: Sequence[str] = AA_LIST) -> np.ndarray:
    aa_index = {aa: i for i, aa in enumerate(aa_list)}
    dipeptides = [a + b for a in aa_list for b in aa_list]
    di_index = {di: i for i, di in enumerate(dipeptides)}
    X = np.zeros((len(peptides), 1 + len(aa_list) + len(dipeptides)), dtype=np.float32)

    for row, peptide in enumerate(peptides):
        length = len(peptide)
        X[row, 0] = float(length)

        for aa in peptide:
            X[row, 1 + aa_index[aa]] += 1.0
        X[row, 1:1 + len(aa_list)] /= max(length, 1)

        if length > 1:
            for i in range(length - 1):
                di = peptide[i:i + 2]
                X[row, 1 + len(aa_list) + di_index[di]] += 1.0
            X[row, 1 + len(aa_list):] /= (length - 1)

    return X


class GraphPeptideDataset(Dataset):
    def __init__(self, graphs, labels):
        self.graphs = list(graphs)
        self.labels = np.asarray(labels, dtype=np.float32)

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx], torch.tensor(self.labels[idx], dtype=torch.float32)


def load_project_data(config: ProjectConfig) -> dict:
    positive_peptides = load_sequences_from_table(config.positive_data_path)
    unlabeled_peptides = load_sequences_from_table(config.unlabeled_train_path)
    external_negative_pool = load_sequences_from_table(config.external_negative_pool_path)

    seen_nonnegative = set(positive_peptides) | set(unlabeled_peptides)
    external_negative_pool = [seq for seq in external_negative_pool if seq not in seen_nonnegative]

    positive_train_all, positive_test = train_test_split(
        positive_peptides,
        test_size=config.outer_test_size,
        random_state=config.random_state,
        shuffle=True,
    )
    positive_cv, positive_calibration = train_test_split(
        positive_train_all,
        test_size=config.calibration_positive_fraction,
        random_state=config.random_state,
        shuffle=True,
    )
    benchmark_val_pool, benchmark_test_pool = train_test_split(
        external_negative_pool,
        test_size=0.5,
        random_state=config.random_state,
        shuffle=True,
    )

    return {
        "positive_peptides": positive_peptides,
        "unlabeled_peptides": unlabeled_peptides,
        "external_negative_pool": external_negative_pool,
        "positive_train_all": positive_train_all,
        "positive_cv": positive_cv,
        "positive_calibration": positive_calibration,
        "positive_test": positive_test,
        "benchmark_val_pool": benchmark_val_pool,
        "benchmark_test_pool": benchmark_test_pool,
        "unlabeled_graphs": [peptide_to_graph(p) for p in unlabeled_peptides],
    }
