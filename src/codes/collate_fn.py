import torch
from transformers import AutoTokenizer
from typing import Dict, List

class MyCustomCollateFN:
    def __init__(self, base_ckpt:str, device:str):
        self.tokenizer = AutoTokenizer.from_pretrained(base_ckpt)
        self.device = device

    def __call__(self, batch:List)->Dict:
        text_list = []
        category_list = []
        sentiment_list = []
        tense_list = []
        certainty_list = []

        for b in batch:
            text_list.append(b.loc["문장"])
            category_list.append(b.loc["유형"])
            sentiment_list.append(b.loc["극성"])
            tense_list.append(b.loc["시제"])
            certainty_list.append(b.loc["확실성"])

        toked_result = self.tokenizer(text_list, return_tensors='pt', padding = 'longest')
        category_batch_tensor = torch.tensor(category_list).long()
        sentiment_batch_tensor = torch.tensor(sentiment_list).long()
        tense_batch_tensor = torch.tensor(tense_list).long()
        certainty_batch_tensor = torch.tensor(certainty_list).long()

        return_dict = {
            "input_ids" : toked_result["input_ids"].to(self.device),
            "attention_mask" : toked_result["attention_mask"].to(self.device),
            "category" : category_batch_tensor.to(self.device),
            "sentiment" : sentiment_batch_tensor.to(self.device),
            "tense" : tense_batch_tensor.to(self.device),
            "certainty" : certainty_batch_tensor.to(self.device)
        }

        return return_dict