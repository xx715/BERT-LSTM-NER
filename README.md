# BERT-LSTM-NER
**Named-entity Recoginition with transfer learning**

Course project for NYU CSCI-GA 3033-091 Introduction to Deep Learning Systems

Group Members: Yuchuan Fu(yf2127), Xiaohan Xue(xx715)


# Dataset
CoNLL-2003 newswire articles: https://www.clips.uantwerpen.be/conll2003/ner/

GloVe vector representation from Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 2014. See https://nlp.stanford.edu/projects/glove/

# Models
- We implemented four models here: Bi-LSTM + CNN, Bert-base, RoBERTa, RoBERTa_news.
- RoBERTa_news was further pretrained on a news corpus data, which was proposed in this paper: Don't Stop Pretraining: Adapt Language Models to Domains and Tasks(https://arxiv.org/abs/2004.10964). This model can be accessed from https://huggingface.co/allenai/dsp_roberta_base_dapt_news_tapt_ag_115K


# Requirements
-  `python3`
- `pip3 install -r requirements_lstm.txt`
- `pip3 install -r requirements_bert.txt`
- Download and store GloVe embeddings into `embeddings` for LSTM models

Tips:
You may want to create different virtual envs to avoid conflicts.

# Code Structure
## Bi-LSTM + CNN
`nn_CoNLL.ipynb` contains codes to run Bi-LSTM + CNN models, we ran our experiments on Google Colab.

## BERT
- `run_ner_new.py` contains all codes needed to run BERT models. Specifically, we finished the classes `NerRoBERTa` and `NerNews`.
- `run_ner_records.py` has the same structure as `run_ner_new.py`. Moreover, we added evaluation of test and valid set for each epoch, and print the corresponding `valid_f1_scores` and `test_f1_scores` for visualization of training process.

# Run
`python run_ner.py --data_dir=data/ --bert_model=bert-base-cased --task_name=ner --output_dir=out_base --max_seq_length=128 --do_train --num_train_epochs 5 --do_eval --eval_on test --warmup_proportion=0.1`

- For 'bert-base-cased': `--bert_model=bert-base-cased`
- For 'roberta': `--bert_model=roberta`
- For 'bert-news': `--bert_model=bert-base-cased`


# Result 
## Bi-LSTM + CNN
The implementation achieves a valid F1 score of 0.9311 and a test F1 score of 0.8910 with 100 epochs.

### BERT-base on Test Data
```
             precision    recall  f1-score   support

        ORG     0.8772    0.9163    0.8963      1661
        LOC     0.9273    0.9329    0.9301      1668
        PER     0.9645    0.9567    0.9606      1617
       MISC     0.7843    0.8134    0.7986       702

avg / total     0.9054    0.9200    0.9125      5648
```

# Reference
BERT Code adapted from: https://github.com/kamalkraj/BERT-NER

LSTM Code adapted from: https://github.com/kamalkraj/Named-Entity-Recognition-with-Bidirectional-LSTM-CNNs

Other Githubs used for reference:
1. https://github.com/lonePatient/BERT-NER-Pytorch
2. https://github.com/liuyukid/transformers-ner
3. https://github.com/mohammadKhalifa/xlm-roberta-ner

