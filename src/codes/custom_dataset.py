import pandas as pd
import torch
from torch.utils.data import Dataset
# from transformers import AutoTokenizer
from typing import Dict

class SentenceCategoryDataset(Dataset):
    def __init__(self, dataset:pd.DataFrame, device:str):
        self.dataset = dataset
        self.device = device

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index)->pd.Series:
        temp_index_data = self.dataset.iloc[index]
        temp_index_data = temp_index_data.loc[:, ["문장", "유형", "극성", "시제", "확실성", "label"]]

        return temp_index_data

if __name__ == "__main__":
    target_dataset = pd.read_csv("Dacon_JminJ_Sentence_Category_Classification/dataset/preprocessed_dataset/convert_label_ver_train.csv")

    sentence_dataset = SentenceCategoryDataset(target_dataset, base_ckpt="monologg/koelectra-small-v3-discriminator", device = "cpu")

    print(sentence_dataset.__getitem__(index = 0))