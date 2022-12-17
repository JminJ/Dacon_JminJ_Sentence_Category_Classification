import torch
import torch.nn as nn
from transformers import AutoModel
from typing import Dict, Tuple

class SentenceCategoryClassifier(nn.Module):
    def __init__(self, base_ckpt:str, drop_rate:float):
        super(SentenceCategoryClassifier, self).__init__()
        self.base_model = AutoModel.from_pretrained(base_ckpt)

        self.category_layer = nn.Linear(256, 4)
        self.sentiment_layer = nn.Linear(256, 3)
        self.tense_layer = nn.Linear(256, 3)
        self.certainty_layer = nn.Linear(256, 2)

        self.dropout = nn.Dropout(drop_rate)
        self.gelu = nn.GELU()

        self.softmax = nn.Softmax(dim = 1)

    def forward(self, input:Dict)->Tuple:
        base_model_output = self.base_model(input["input_ids"], attention_mask = input["attention_mask"])
        last_hidden_state = base_model_output["last_hidden_state"][:, 0, :]
        
        pooler = self.gelu(last_hidden_state)
        pooler = self.dropout(pooler)

        category_result = self.softmax(self.category_layer(pooler))
        sentiment_result = self.softmax(self.sentiment_layer(pooler))
        tense_result = self.softmax(self.tense_layer(pooler))
        certainty_result = self.softmax(self.certainty_layer(pooler))

        return category_result, sentiment_result, tense_result, certainty_result
        

if __name__ == "__main__":
    sentence_classifier = SentenceCategoryClassifier(base_ckpt="monologg/koelectra-small-v3-discriminator")