
#! /bin/sh

mkdir -p AGNews_model_save
mkdir -p AGNews_Lexicon
mkdir -p result
mkdir -p model_save
mkdir -p temp_data
mkdir -p data

python3.6 model/classifier_AGNews.py \
    --task mrpc \
    --mode train \
    --train_cfg train_mrpc.json \
    --data_train_file total_data/agtrain.tsv \
    --data_test_file total_data/ag_test.tsv \
    --max_len 150

