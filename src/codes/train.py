import os
import pandas as pd
import numpy as np
import argparse
import pprint
from typing import Dict, Tuple, List

import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from electra_classifier import SentenceCategoryClassifier
from custom_dataset import SentenceCategoryDataset
from collate_fn import MyCustomCollateFN
from operation import ClassifierOperation

class Trainer:
    def __init__(self, base_ckpt:str, device:str, save_dir_path:str, args:argparse.Namespace):
        self.args = args
        self.save_dir_path = save_dir_path
        self._make_save_dir(save_dir_path = self.save_dir_path)

        self.train_pd_dataset = pd.read_csv(args.trainset_path)
        self.valid_pd_dataset = pd.read_csv(args.validset_path)

        train_Dataset = SentenceCategoryDataset(self.train_pd_dataset)
        valid_Dataset = SentenceCategoryDataset(self.valid_pd_dataset)
        custom_collate_fn = MyCustomCollateFN(base_ckpt=base_ckpt, device=device)     
        
        self.train_dataloader = DataLoader(train_Dataset, batch_size=args.train_batch_size, collate_fn=custom_collate_fn)
        self.valid_dataloader = DataLoader(valid_Dataset, batch_size=args.valid_batch_size, collate_fn=custom_collate_fn)

        classifier = SentenceCategoryClassifier(base_ckpt=base_ckpt, drop_rate=0.2, device=device) 
        loss_fn_dict = self._define_each_loss_functions()  

        self.operation_cls = ClassifierOperation(classifier = classifier, loss_functions=loss_fn_dict, mode="train")

        self.optimizer = torch.optim.RAdam(self.operation_cls.classifier.parameters(), lr = args.learning_rate, weight_decay=0.15) # weight_decay는 고정 값으로 사용한다
        
        num_total_steps = len(self.train_dataloader) * args.epochs
        num_warmup_steps = self._calc_num_warmup_step(warmup_rate=args.warmup_rate, num_total_steps=num_total_steps)
        self.learning_rate_schedular = get_cosine_schedule_with_warmup(optimizer=self.optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_total_steps)

    def _make_save_dir(self, save_dir_path:str):
        if not os.path.exists(save_dir_path):
            os.makedirs(save_dir_path)
            print(f"MAKE : {save_dir_path}")
        else:
            print(f"\'{save_dir_path}\' is already existing.")

    def _wandb_init(self):
        wandb.init(project="Dacon_Sentence_Category_classification",
            config={
                'epochs' : self.args.epochs,
                'batch_size' : self.args.train_batch_size,
                'learning_rate' : self.args.learning_rate
            }
        )
        wandb.watch(self.operation_cls.classifier)

    def _calc_num_warmup_step(self, warmup_rate:float, num_total_steps:int)->int:
        num_warmup_steps = int(num_total_steps * warmup_rate)

        return num_warmup_steps

    def __calc_loss_weight(self)->Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        category_unique, category_cnt=np.unique(self.train_pd_dataset.loc[:, "유형"], return_counts=True)
        sentiment_unique, sentiment_cnt=np.unique(self.train_pd_dataset.loc[:, "극성"], return_counts=True)
        tense_unique, tense_cnt=np.unique(self.train_pd_dataset.loc[:, "시제"], return_counts=True)
        certainty_unique, certainty_cnt=np.unique(self.train_pd_dataset.loc[:, "확실성"], return_counts=True)

        category_weight = torch.tensor([max(category_cnt)/category_cnt[i] for i in range(len(category_cnt))]).float()
        sentiment_weight = torch.tensor([max(sentiment_cnt)/sentiment_cnt[i] for i in range(len(sentiment_cnt))]).float()
        tense_weight = torch.tensor([max(tense_cnt)/tense_cnt[i] for i in range(len(tense_cnt))]).float()
        certainty_weight = torch.tensor([max(certainty_cnt)/certainty_cnt[i] for i in range(len(certainty_cnt))]).float()

        return category_weight, sentiment_weight, tense_weight, certainty_weight

    def _define_each_loss_functions(self)->Dict:
    
        if self.args.use_weighted_loss:
            category_weight, sentiment_weight, tense_weight, certainty_weight = self.__calc_loss_weight()

            category_loss_fn = nn.CrossEntropyLoss(weight=category_weight)
            sentiment_loss_fn = nn.CrossEntropyLoss(weight=sentiment_weight)
            tense_loss_fn = nn.CrossEntropyLoss(weight=tense_weight)
            certainty_loss_fn = nn.CrossEntropyLoss(weight=certainty_weight)

            weight_loss_fn_dict = {
                "category" : category_loss_fn,
                "sentiment" : sentiment_loss_fn,
                "tense" : tense_loss_fn,
                "certainty" : certainty_loss_fn
            }

            return weight_loss_fn_dict
        else:
            category_loss_fn = nn.CrossEntropyLoss()
            sentiment_loss_fn = nn.CrossEntropyLoss()
            tense_loss_fn = nn.CrossEntropyLoss()
            certainty_loss_fn = nn.CrossEntropyLoss()

            loss_fn_dict = {
                "category" : category_loss_fn,
                "sentiment" : sentiment_loss_fn,
                "tense" : tense_loss_fn,
                "certainty" : certainty_loss_fn
            }

            return loss_fn_dict

    def _define_to_save_each_data(self)->Tuple[float, Dict, Dict, float]:
        loss_save_value = 0
        correct_cnt_save_dict = {
            "category" : 0,
            "sentiment" : 0,
            "tense" : 0,
            "certainty" : 0
        }
        f1_score_save_dict = {
            "category" : 0.0,
            "sentiment" : 0.0,
            "tense" : 0.0,
            "certainty" : 0.0
        }
        full_label_f1_score_save_value = 0.0

        return loss_save_value, correct_cnt_save_dict, f1_score_save_dict, full_label_f1_score_save_value

    def _train(self):
        self.operation_cls.classifier.train()

        train_loss, train_each_corrects, train_each_f1_scores, train_full_label_f1_score = self._define_to_save_each_data()
        train_steps, train_examples = 0, 0

        for _, batch in enumerate(self.train_dataloader, 0):
            temp_step_loss, temp_step_each_correct_cnts, temp_step_each_f1_scores, temp_step_full_label_f1_score = self.operation_cls.forward(input_batch=batch)

            train_loss += temp_step_loss.item()
            train_each_corrects = [train_each_corrects[k] + temp_step_each_correct_cnts[k] for k in train_each_corrects.keys()]
            train_each_f1_scores = [train_each_f1_scores[k] + temp_step_each_f1_scores[k] for k in train_each_f1_scores.keys()]
            train_full_label_f1_score += temp_step_full_label_f1_score

            train_steps += 1
            train_examples += len(batch["label"])

            wandb.log({
                "train_loss" : train_loss / train_steps,
                "train_category_f1_score" : train_each_f1_scores["category"] / train_examples,
                "train_sentiment_f1_score" : train_each_f1_scores["sentiment"] / train_examples,
                "train_tense_f1_score" : train_each_f1_scores["tense"] / train_examples,
                "train_certainty_f1_score" : train_each_f1_scores["certainty"] / train_examples,
                "train_full_label_f1_score" : train_full_label_f1_score / train_examples
            })

            ## update
            self.optimizer.zero_grad()
            temp_step_loss.backward()
            self.optimizer.step()
            self.learning_rate_schedular.step()


    '''
        Return
            - valid_examples(int) : forward에서 corrects 결과를 출력하기 위한 인자
            - valid_each_corrects(Dict) : 각 label의 correct 결과
            - valid_results(Dict) : valid loss와 각 label, full-label의 f1-score들
    '''
    def _valid(self)->Tuple[int, int, Dict, Dict]:
        self.operation_cls.classifier.eval()

        valid_loss, valid_each_corrects, valid_each_f1_scores, valid_full_label_f1_score = self._define_to_save_each_data()
        valid_steps = 0
        valid_examples = 0

        with torch.no_grad():
            for _, batch in self.valid_dataloader:
                temp_step_loss, temp_step_each_correct_cnts, temp_step_each_f1_scores, temp_step_full_label_f1_score = self.operation_cls.forward(input_batch=batch)

                valid_loss += temp_step_loss.item()
                valid_each_corrects = [valid_each_corrects[k] + temp_step_each_correct_cnts[k] for k in valid_each_corrects.keys()]
                valid_each_f1_scores = [valid_each_f1_scores[k] + temp_step_each_f1_scores[k] for k in valid_each_f1_scores.keys()]
                valid_full_label_f1_score += temp_step_full_label_f1_score

                valid_steps += 1
                valid_examples += len(batch["label"])
            
            valid_results = {
                "valid_loss" : valid_loss / valid_steps,
                "valid_category_f1_score" : valid_each_f1_scores["category"] / valid_examples,
                "valid_sentiment_f1_score" : valid_each_f1_scores["sentiment"] / valid_examples,
                "valid_tense_f1_score" : valid_each_f1_scores["tense"] / valid_examples,
                "valid_certainty_f1_score" : valid_each_f1_scores["certainty"] / valid_examples,
                "valid_full_label_f1_score" : valid_full_label_f1_score / valid_examples
            }
            wandb.log(valid_results)

        return valid_examples, valid_each_corrects, valid_results

    def _model_ckpt_save(self, epoch:int):
        torch.save({
            "epoch" : epoch,
            "model_state_dict" : self.operation_cls.classifier.state_dict(),
            "optimizer_state_dict" : self.optimizer.state_dict()
        }, os.path.join(self.save_dir_path, f"koelectra_sentence_category_cls_epoch_{epoch}.pt"))

    def forward(self):
        for epoch in range(self.args.epochs):
            self._train()
            self._model_ckpt_save(epoch=epoch)
            valid_examples, valid_each_corrects, valid_result = self._valid()

            print(f"===== {epoch} epoch valid result =====")
            pprint.pprint(valid_result)
            for k in valid_each_corrects.keys():
                valid_each_corrects[k] = valid_each_corrects[k] / valid_examples
            pprint.pprint('\n', valid_each_corrects, '\n')

        print(f"\n\n ----------------------\nTrain Operation Is Done..")
        