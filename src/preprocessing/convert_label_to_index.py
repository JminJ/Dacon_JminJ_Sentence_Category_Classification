import pandas as pd
import numpy as np

def convert_label_to_tensor(target_dataframe:pd.DataFrame):
    category_label_dict = {"사실형" : 0, "추론형" : 1, "대화형" : 2, "예측형" : 3}
    sentiment_label_dict = {"긍정" : 0, "부정" : 1, "미정" : 2}
    tense_label_dict = {"과거" : 0, "현재" : 1, "미래" : 2}
    certainty_label_dict = {"확실" : 0, "불확실" : 1}

    full_label_unique = list(np.unique(target_dataframe.loc[:, "label"]))

    for i in range(len(target_dataframe)):
        temp_category = target_dataframe.loc[i, "유형"]
        temp_sentiment = target_dataframe.loc[i, "극성"]
        temp_tense = target_dataframe.loc[i, "시제"]
        temp_certainty = target_dataframe.loc[i, "확실성"]

        target_dataframe.loc[i, "유형"] = category_label_dict[temp_category]
        target_dataframe.loc[i, "극성"] = sentiment_label_dict[temp_sentiment]
        target_dataframe.loc[i, "시제"] = tense_label_dict[temp_tense]
        target_dataframe.loc[i, "확실성"] = certainty_label_dict[temp_certainty]

        target_dataframe.loc[i, "label"] = full_label_unique.index(target_dataframe.loc[i, "label"])

        print(target_dataframe.iloc[i])

    return target_dataframe


if __name__ == "__main__":
    TARGET_DATA_PATH = "Dacon_JminJ_Sentence_Category_Classification/dataset/train.csv"
    dataset = pd.read_csv(TARGET_DATA_PATH)

    converted_dataset = convert_label_to_tensor(dataset)
    converted_dataset.to_csv("Dacon_JminJ_Sentence_Category_Classification/dataset/preprocessed_dataset/convert_label_ver_train.csv", index=False)

