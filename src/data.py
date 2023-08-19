import torch
from torch.utils.data import Dataset
from typing import List
from pathlib import Path

from tqdm import tqdm


class FluorescenceDataset(Dataset):
    def __init__(self, embeddings: List[torch.Tensor], fluorescence_values: List[float]):
        self.embeddings = embeddings
        self.fluorescence_values = fluorescence_values

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx: int):
        return self.embeddings[idx], self.fluorescence_values[idx]


def train_validation_test(p: Path, layer: int):
    embeddings = []
    fluorescence_list = []
    for i in tqdm(p.glob("*.pt")):
        t = torch.load(i)
        # get fluorescence (you can check what split does with ChatGPT
        idx, fluorescence = t["label"].split("|")
        fluorescence_list.append(float(fluorescence))
        emb = t["mean_representations"][layer]
        embeddings.append(emb)
    return (embeddings, fluorescence_list)
