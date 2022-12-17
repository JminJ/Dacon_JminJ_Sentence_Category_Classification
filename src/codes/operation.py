import torch
from sklearn.metrics import f1_score
from electra_classifier import SentenceCategoryClassifier
from typing import Dict, List, Tuple
from itertools import product


class ClassifierOperation:
    def __init__(self, classifier:SentenceCategoryClassifier, loss_functions:Dict, mode:str):
        self.mode = mode
        self.classifier = classifier
        
        self.loss_functions = loss_functions

        self.category_label_str = ["사실형", "추론형", "대화형", "예측형"]
        self.sentiment_label_str = ["긍정", "부정", "미정"]
        self.tense_label_str = ["과거", "현재", "미래"]
        self.certainty_label_str = ["확실", "불확실"]
        self.full_label_str = list(product(*[self.category_label_str, self.sentiment_label_str, self.tense_label_str, self.certainty_label_str]))
        self.full_label_str = ["-".join(list(t)) for t in self.full_label_str]

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

    def _calc_f1_score(self, max_index_each_data:List, target_label:torch.Tensor):
        target_label = target_label.detach().cpu().numpy()
        print(max_index_each_data)
        print(target_label)
        weighted_f1_score_value = f1_score(target_label, max_index_each_data, average="weighted")

        return weighted_f1_score_value

    ### full_label을 생성
    def _make_full_label(self, each_label_max_index_datas:Dict, batch_size:int):
        full_label_list = []

        for i in range(batch_size):
            temp_category_label = int(each_label_max_index_datas["category"][i])
            temp_sentiment_label = int(each_label_max_index_datas["sentiment"][i])
            temp_tense_label = int(each_label_max_index_datas["tense"][i])
            temp_certainty_label = int(each_label_max_index_datas["certainty"][i])

            temp_full_label = f"{self.category_label_str[temp_category_label]}-{self.sentiment_label_str[temp_sentiment_label]}-{self.tense_label_str[temp_tense_label]}-{self.certainty_label_str[temp_certainty_label]}"

            # print(f"full-label : {temp_full_label}")
            full_label_list.append(temp_full_label)

        return full_label_list
    
    def _define_all_return_argumetns(self)->Tuple[Dict, Dict, Dict, Dict, Dict]:
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

        # full-label 생성용
        each_label_max_index_datas = {
            "category" : [],
            "sentiment" : [],
            "tense" : [],
            "certainty" : []
        }

        return correct_cnt_each_label, f1_score_each_label, correct_each_labels, false_each_labels, each_label_max_index_datas

    def forward(self, input_batch:Dict)->Tuple:
        output_list = list(self.classifier(input_batch))
        loss_fn_keys = list(self.loss_functions.keys())
        batch_size = len(input_batch["label"])

        all_loss_result = []
        correct_cnt_each_label, f1_score_each_label, correct_each_labels, false_each_labels, each_label_max_index_datas = self._define_all_return_argumetns()

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

            each_label_max_index_datas[temp_loss_key_name].extend(temp_label_max_index_each_data)
            correct_cnt_each_label[temp_loss_key_name] += temp_label_correct_cnt
            # print(f"temp_label_max_index_each_data : {temp_label_max_index_each_data}")
            # print(f"temp_label_correct_cnt : {temp_label_correct_cnt}")

            temp_label_f1_score = self._calc_f1_score(max_index_each_data=temp_label_max_index_each_data, target_label=input_batch[temp_loss_key_name])
            f1_score_each_label[temp_loss_key_name] += temp_label_f1_score
            # print(f"temp_label_f1_score : {temp_label_f1_score}")

        full_label_list = self._make_full_label(each_label_max_index_datas=each_label_max_index_datas, batch_size = batch_size)
        # print(f"\n\nfull_label_list : {full_label_list}")
        converted_full_label_list = [self.full_label_str.index(f) for f in full_label_list]
        # print(converted_full_label_list)
        converted_full_label_f1_score = self._calc_f1_score(max_index_each_data=converted_full_label_list, target_label=input_batch["label"])
        # print(f"full_label_f1_score : {converted_full_label_f1_score}")

        final_loss_result = sum(all_loss_result) / len(loss_fn_keys)
        # print(f"\nfinal_loss_result : {final_loss_result}\n\n")

        if self.mode == "train":
            return final_loss_result, correct_cnt_each_label, f1_score_each_label, converted_full_label_f1_score
        else:
            return final_loss_result, correct_cnt_each_label, f1_score_each_label, converted_full_label_f1_score, correct_each_labels, false_each_labels