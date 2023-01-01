# Dacon_JminJ_Sentence_Category_Classification
## Intro
성균관 대학교에서 주최하신 문장 유형 분류 AI 경진대회를 진행하며 관련 내용을 욜려 둔 repository입니다.
## 성적
* 상위 22%(private 0.73654)
## 디렉토리 설명
* src/codes : 개인적으로 작성한 코드를 모아놓은 공간입니다. 
* src/base_code_variant : 데이콘에서 제공한 base_code를 개선해 학습을 진행시킨 코드입니다. 대회 제출 및 최고 모델은 해당 코드로 학습시켰습니다.
* data_check : 데이터셋의 EDA를 확인하는데 사용한 코드가 담긴 공간입니다.
## 최고 모델 학습 파라미터
* 사용 코드 : dacon_base_code_plus_my.ipynb
* base model : KoElecta Base size
* learning rate : 1e-05
* loss weight : True
* loss : cross entropy
* back translation : False
## 대회 링크
[문장 유형 분류 AI 경진대회](https://dacon.io/competitions/official/236037/overview/description)