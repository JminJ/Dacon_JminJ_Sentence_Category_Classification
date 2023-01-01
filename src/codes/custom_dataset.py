import pandas as pd
import torch
from torch.utils.data import Dataset
# from transformers import AutoTokenizer
from typing import Dict

class SentenceCategoryDataset(Dataset):
    def __init__(self, dataset:pd.DataFrame):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index)->pd.Series:
        temp_index_data = self.dataset.loc[index, ["문장", "유형", "극성", "시제", "확실성"]]

        return temp_index_data