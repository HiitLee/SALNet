### SALNet: Semi-Supervised Few-Shot Text Classification with Attention-based Lexicon Construction

We propose a semi-supervised bootstrap learning framework for few-shot text classification. From a small number of the initial data, our framework obtains a larger set of reliable training data by using the attention weights from an LSTM-based trained classifier. We first train an LSTM-based text classifier from a given labeled dataset using the attention mechanism. Then, we collect a set of words for each class called a lexicon, which is supposed to be a representative set of words for each class based on the attention weights calculated for the classification task. We bootstrap the classifier using the new data that are labeled by the combination of the classifier and the constructed lexicons to improve the prediction accuracy. As a result, our approach outperforms the previous state-of-the-art methods including semi-supervised learning algorithms and pretraining algorithms for  few-shot text classification task on four publicly available benchmark datasets. Moreover, we empirically confirm that the constructed lexicons are reliable enough and substantially improve the performance of the original classifier.

### Requirements

    python >= 3.6
    torch, nltk, numpy, gensim, scikit-learn, scipy, sklearn, tqdm
    fire, tqdm, tensorboardx, tensorflow (for loading checkpoint file of BERT)

    pip install -r requirements.txt

### Download Glove

Please download pre-trained model `glove.42B.300d.zip` on https://nlp.stanford.edu/projects/glove/. 

### Download BERT

Please download pre-trained model **[`BERT-Base, Uncased`](https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip)** on https://github.com/google-research/bert#pre-trained-models.

### Datasets

We use four benchmark datasets. (IMDB review, AG News, Yahoo! answer, DBpedia). We take only 1% of the original training data as our labeled data with random sampling. In the new labeled dataset, we use its 85% data as a training set and 15% data as a development set. We remove the labels of the remaining 99% data. All data has balanced class distribution.

### Training and evaluation of the SALNet (attention-based LSTM + TextCNN)

Before running this example you must download the `glove.42B.300d.zip` and unzip it to some directory `$SALNet`. 

    ./train_eval_IMDB.sh (IMDB)
    ./train_eval_AGNews.sh (AGNews)
    ./train_eval_yahoo.sh (Yahoo! answer)
    ./train_eval_DBpedia.sh (DBpedia)

### Training and evaluation of the SALNet (attention-based LSTM + BERT)

Before running this example you must download the **[`BERT-Base, Uncased`](https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip)** and unzip it to some directory `$SALNet/model `.

    ./train_eval_BERT_IMDB.sh (IMDB)
    ./train_eval_BERT_AGNews.sh (AGNews)
    ./train_eval_BERT_yahoo.sh (Yahoo! answer)
    ./train_eval_BERT_DBpedia.sh (DBpedia)

We used RTX 2070 GPU, Quadro RTX 6000/8000 GPU for model training.

During training, we get the accuracy of test set and  size of training set (1% labeled data + pseudo-labeled data) at each epoch in the  `result/result_(IMDB/AGNews/yahoo/DBpedia).txt, pseudo_train_set_(IMDB/AGNews/yahoo/DBpedia).txt`.

During training, The lexicons for each class is created with '`(IMDB/AGNews/yahoo/DBpedia)_Lexicon/(imdb/ag/db/yahoo).txt` at each epoch.

Train until  no more training set (pseudo-labeled data) are added.

