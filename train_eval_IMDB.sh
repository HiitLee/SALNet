
#! /bin/sh
mkdir -p IMDB_model_save
mkdir -p IMDB_Lexicon
mkdir -p result
mkdir -p model_save
mkdir -p temp_data
mkdir -p data


python3.6 model/classifier_imdb.py \
    --task mrpc \
    --mode train \
    --train_cfg train_mrpc.json \
    --data_train_file total_data/imdbtrain.tsv \
    --data_test_file total_data/IMDB_test.tsv \
    --max_len 300

