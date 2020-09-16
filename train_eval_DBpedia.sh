
#! /bin/sh

mkdir -p DBpedia_model_save
mkdir -p DBpedia_Lexicon
mkdir -p result
mkdir -p model_save
mkdir -p temp_data
mkdir -p data


python3.6 model/classifier_DBpedia.py \
    --task mrpc \
    --mode train \
    --train_cfg train_mrpc.json \
    --data_train_file total_data/dbtrain.tsv \
    --data_test_file total_data/db_test.tsv \
    --max_len 200

