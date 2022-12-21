import pandas as pd

# convert to original label(str)
def convert_to_original_label(original_dataset:pd.DataFrame, target_dataset:pd.DataFrame)->pd.DataFrame:
    for i in range(len(target_dataset)):
        temp_id = target_dataset.loc[i, "ID"]
        temp_id_original_index = original_dataset[original_dataset["ID"] == temp_id].index
        
        original_type = original_trainset.loc[temp_id_original_index ,"유형"].values
        original_polarity = original_trainset.loc[temp_id_original_index ,"극성"].values
        original_tense = original_trainset.loc[temp_id_original_index ,"시제"].values
        original_certainty = original_trainset.loc[temp_id_original_index ,"확실성"].values
        original_label = original_trainset.loc[temp_id_original_index ,"label"].values

        try:
            target_dataset.loc[i, "유형"] = original_type[0]
            target_dataset.loc[i, "극성"] = original_polarity[0]
            target_dataset.loc[i, "시제"] = original_tense[0]
            target_dataset.loc[i, "확실성"] = original_certainty[0]
            target_dataset.loc[i, "label"] = original_label[0]
        except:
            print(target_dataset.iloc[i])
            print()
            print(f"orignal type : {original_type}")
            print(f"orignal polarity : {original_polarity}")
            print(f"orignal tense : {original_tense}")
            print(f"orignal certainty : {original_certainty}")
            print(f"orignal label : {original_label}")

            raise KeyboardInterrupt

    return target_dataset

if __name__ == "__main__":
    original_trainset = pd.read_csv("dataset/train.csv")
    my_trainset = pd.read_csv("dataset/preprocessed_dataset/train_valid/train_preprocessed_dataset_13232.csv")
    back_translation_set = pd.read_csv("dataset/translate_results/train_translated_version_en.csv")

    # convert each data's labels 
    converted_my_trainset = convert_to_original_label(original_dataset=original_trainset, target_dataset=my_trainset)
    converted_back_translation_set = convert_to_original_label(original_dataset=original_trainset, target_dataset=back_translation_set)
    converted_my_trainset.to_csv("dataset/preprocessed_dataset/train_valid/converted_split_trainset.csv", index=False)
    converted_back_translation_set.to_csv("dataset/translate_results/converted_back_translation_en.csv", index=False)

    # concat my trainset with back translation data
    concat_my_train_back_translation = pd.concat([converted_my_trainset, converted_back_translation_set])
    concat_my_train_back_translation.to_csv("dataset/preprocessed_dataset/train_valid/concat_with_my_split_trainset_back_translation.csv", index = False)