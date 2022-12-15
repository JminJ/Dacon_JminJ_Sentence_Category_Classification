import argparse
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from electra_classifier import SentenceCategoryClassifier
from typing import Dict, Tuple


class ClassifierOperation:
    def __init__(self, classifier:SentenceCategoryClassifier, loss_functions:Dict, mode:str, args:argparse.Namespace):
        self.mode = mode
        self.args = args
        self.classifier = classifier
        
        self.loss_functions = loss_functions

    def _calc_loss(self, target_output:torch.Tensor, target_label:torch.Tensor, label_type:str)->torch.Tensor:
        loss_function = self.loss_functions[label_type]

        loss_val = loss_function(target_output, target_label)

        return loss_val
    
    def _calc_correct_cnt(self, target_output:torch.Tensor, target_label:torch.Tensor, mode:str):
        max_index_each_data = torch.argmax(target_output, dim = 1)
        max_index_each_data = max_index_each_data.detach().cpu().numpy()

        target_label = target_label.detach().cpu().numpy()

        correct_cnt = 0
        correct_index = []
        false_index = []
        for l in range(len(target_label)):
            if target_label[l] == max_index_each_data[l]:
                correct_cnt += 1
                correct_index.append(l)
            else:
                false_index.append(l)

        if mode == "train":
            return correct_cnt, max_index_each_data
        else:
            return correct_cnt, max_index_each_data, correct_index, false_index 

    def _calc_f1_score(self, max_index_each_data:np.ndarray, target_label:torch.Tensor):
        target_label = target_label.detach().cpu().numpy()
        weighted_f1_score_value = f1_score(target_label, max_index_each_data, average="weighted")

        return weighted_f1_score_value

    def _define_all_return_argumetns(self)->Tuple[Dict, Dict, Dict, Dict]:
        correct_cnt_each_label = {
            "category" : 0,
            "sentiment" : 0,
            "tense" : 0,
            "certainty" : 0
        }

        f1_score_each_label = {
            "category" : 0.0,
            "sentiment" : 0.0,
            "tense" : 0.0,
            "certainty" : 0.0
        }

        correct_each_labels = {
            "category" : [],
            "sentiment" : [],
            "tense" : [],
            "certainty" : []
        }
        false_each_labels = {
            "category" : [],
            "sentiment" : [],
            "tense" : [],
            "certainty" : []
        }

        return correct_cnt_each_label, f1_score_each_label, correct_each_labels, false_each_labels

    def forward(self, input_batch:Dict)->Tuple:
        output_list = list(self.classifier(input_batch))
        loss_fn_keys = list(self.loss_functions.keys())

        all_loss_result = []

        correct_cnt_each_label, f1_score_each_label, correct_each_labels, false_each_labels = self._define_all_return_argumetns()

        for i in range(len(output_list)):
            temp_loss_key_name = loss_fn_keys[i]

            temp_label_loss_result = self._calc_loss(target_output=output_list[i], target_label=input_batch[temp_loss_key_name], label_type=temp_loss_key_name)
            all_loss_result.append(temp_label_loss_result)
            # print(f"{temp_loss_key_name} loss result : {temp_label_loss_result}")

            if self.mode == "train":
                temp_label_correct_cnt, temp_label_max_index_each_data = self._calc_correct_cnt(target_output=output_list[i], target_label=input_batch[temp_loss_key_name], mode=self.mode)
            else:
                temp_label_correct_cnt, temp_label_max_index_each_data, temp_label_correct_index, temp_label_false_index = self._calc_correct_cnt(target_output=output_list[i], target_label=input_batch[temp_loss_key_name], mode=self.mode)
                correct_each_labels[temp_loss_key_name].extend(temp_label_correct_index)
                false_each_labels[temp_loss_key_name].extend(temp_label_false_index)
            correct_cnt_each_label[temp_loss_key_name] += temp_label_correct_cnt
            # print(f"temp_label_max_index_each_data : {temp_label_max_index_each_data}")
            # print(f"temp_label_correct_cnt : {temp_label_correct_cnt}")

            temp_label_f1_score = self._calc_f1_score(max_index_each_data=temp_label_max_index_each_data, target_label=input_batch[temp_loss_key_name])
            f1_score_each_label[temp_loss_key_name] += temp_label_f1_score
            # print(f"temp_label_f1_score : {temp_label_f1_score}")

        final_loss_result = sum(all_loss_result) / len(loss_fn_keys)
        # print(f"\nfinal_loss_result : {final_loss_result}\n\n")

        if self.mode == "train":
            return final_loss_result, correct_cnt_each_label, f1_score_each_label
        else:
            return final_loss_result, correct_cnt_each_label, f1_score_each_label, correct_each_labels, false_each_labels

