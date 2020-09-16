
#! /bin/sh

mkdir -p yahoo_model_save
mkdir -p yahoo_Lexicon
mkdir -p result
mkdir -p model_save
mkdir -p temp_data
mkdir -p data


python3.6 model/classifier_yahoo.py \
    --task mrpc \
    --mode train \
    --train_cfg train_mrpc.json \
    --data_train_file total_data/yahootrain.tsv \
    --data_test_file total_data/yahoo_test.tsv \
    --max_len 100

