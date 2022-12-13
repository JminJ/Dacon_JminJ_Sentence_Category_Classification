import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from typing import Dict, List
import matplotlib.pyplot as plt

class DatasetChecker:
    def __init__(self, data_path:str):
        self.data = pd.read_csv(data_path)
        self.tokenizer = AutoTokenizer.from_pretrained("beomi/KcELECTRA-base-v2022")

    def check_data_length(self)->int:
        return len(self.data)

    def check_data_each_label_unique_count(self)->Dict:
        data_label_columns = self.data.columns[2:6]
        data_label_check_dict = {
            "unique" : [],
            "counts" : []
        }

        for c in data_label_columns:
            label_unique, label_count = np.unique(self.data.loc[:, c], return_counts=True)

            data_label_check_dict["unique"].append(label_unique)
            data_label_check_dict["counts"].append(label_count)

        return data_label_check_dict

    def check_data_label_unique_count(self)->Dict:
        data_label_check_dict = {
            "unique" : [],
            "counts" : []
        }

        label_unique, label_counts = np.unique(self.data.loc["Label"], return_counts=True)

        data_label_check_dict["unique"] = label_unique
        data_label_check_dict["counts"] = label_counts

        return data_label_check_dict

    def check_data_token_length(self)->List:
        token_length_list = []

        for i in range(len(self.data)):
            temp_text = str(self.data.loc[i, "문장"])

            toked_result = self.tokenizer(temp_text)
            # print(toked_result)
            # print()

            token_length_list.append(len(toked_result["input_ids"]))

        return token_length_list
    
    def check_data_text_length(self)->List:
        text_length_list = []
        processed_length_list = [] # 띄어쓰기 제거본

        for i in range(len(self.data)):
            temp_text = str(self.data.loc[i, "문장"])

            text_length_list.append(len(temp_text.strip()))
            processed_length_list.append(len(temp_text.strip().replace(" ", "")))

        return text_length_list, processed_length_list


def plot_token_lengths(token_lengths:list):
    plt.hist(token_lengths, label="bins=10", bins=10)

    plt.legend()
    plt.show()

def plot_text_lengths(text_lenghts:List, processed_lengths:List):
    plt.hist(text_lenghts, bins=10, label="original text len")
    plt.hist(processed_lengths, bins=10, label="processed text len")

    plt.legend()
    plt.show()

if __name__ == "__main__":
    DATA_PATH = "/Users/jeongminju/Documents/GITHUB/Dacon_JminJ_Sentence_Category_Classification/open_dataset/train.csv"

    dataset_checker = DatasetChecker(data_path=DATA_PATH)

    ## data length
    data_length = dataset_checker.check_data_length()
    print(f"===== data_length =====")
    print(f"data_length : {data_length}\n")

    ## data each label_unique, label_count
    each_label_inform_dict = dataset_checker.check_data_each_label_unique_count()
    print(f"===== each label inform")
    print(each_label_inform_dict, "\n")

    ## data label_unique, label_count -> print only lengths
    # label_inform_dict = dataset_checker.check_data_label_unique_count()
    # print(f"===== label inform =====")
    # print(label_inform_dict)

    ## token len list
    token_len_list = dataset_checker.check_data_token_length()
    print(f"===== token lengths =====")
    print(f"max_token length : {max(token_len_list)}")
    print(f"min_token length : {min(token_len_list)}")
    print(f"mean_token length : {np.mean(token_len_list)}")
    # plot_token_lengths(token_lengths=token_len_list)

    ## text len list
    origin_text_len_list, processed_text_len_list = dataset_checker.check_data_text_length()
    print(f"===== text lengths =====")
    print(f"\tOriginal text")
    print(f"\tmax_len : {max(origin_text_len_list)}, index : {origin_text_len_list.index(max(origin_text_len_list))}")
    print(f"\tmin_len : {min(origin_text_len_list)}, index : {origin_text_len_list.index(min(origin_text_len_list))}")
    print(f"\tmean_len : {np.mean(origin_text_len_list)}\n")

    print(f"\tProcessed text")
    print(f"\tmax_len : {max(processed_text_len_list)}, index : {processed_text_len_list.index(max(processed_text_len_list))}")
    print(f"\tmin_len : {min(processed_text_len_list)}, index : {processed_text_len_list.index(min(processed_text_len_list))}")
    print(f"\tmean_len : {np.mean(processed_text_len_list)}")

    plot_text_lengths(text_lenghts=origin_text_len_list, processed_lengths=processed_text_len_list)

