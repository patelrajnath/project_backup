import logging
import time

import torch
from scipy.stats import pearsonr, spearmanr
from torch import from_numpy
import pandas as pd

from models.encoders import SbertEncoderClient, LaserEncoderClient, CombinedEncoderClient

from models.classifier import MultilingualSTS
from models.model_utils import load_model_state

glog = logging.getLogger(__name__)

if __name__ == '__main__':
    def pearson_corr(preds, labels):
        return pearsonr(preds, labels)[0]

    def spearman_corr(preds, labels):
        return spearmanr(preds, labels)[0]

    class hparamset():
        def __init__(self):
            self.batchsize = 32
            self.balance_data = False
            self.output_size = None
            self.activation = 'relu'
            self.hidden_layer_size = 512
            self.num_hidden_layers = 1
            self.batch_size = 32
            self.dropout = 0.70
            self.optimizer = 'sgd'
            self.learning_rate = 0.7
            self.lr_decay_pow = 1
            self.epochs = 10
            self.eval_each_epoch = True

    sbert_model = 'distiluse-base-multilingual-cased'
    sbert_model2 = 'xlm-r-100langs-bert-base-nli-stsb-mean-tokens'
    sbert_model3 = 'distilbert-multilingual-nli-stsb-quora-ranking'
    sbert_encoder = SbertEncoderClient(sbert_model)
    sbert_encoder2 = SbertEncoderClient(sbert_model2)
    sbert_encoder3 = SbertEncoderClient(sbert_model3)
    laser_encoder = LaserEncoderClient()
    encoder_client = CombinedEncoderClient([laser_encoder, sbert_encoder, sbert_encoder2, sbert_encoder3])
    # encoder_client = CombinedEncoderClient([laser_encoder])

    train_df = pd.read_csv('sample-data/STS-B/train.tsv', sep='\t', error_bad_lines=False)
    eval_df = pd.read_csv('sample-data/STS-B/dev.tsv', sep='\t', error_bad_lines=False)

    wallet_train_df = pd.read_csv('sample-data/200410_train_stratshuf_english_with_sts_synthesis.csv')
    wallet_eval_df = pd.read_csv('sample-data/200410_test_stratshuf_chinese_200410_english_with_sts_synthesis.csv')

    train_df = train_df.rename(columns={'sentence1': 'text_a', 'sentence2': 'text_b', 'score': 'labels'}).dropna()
    eval_df = eval_df.rename(columns={'sentence1': 'text_a', 'sentence2': 'text_b', 'score': 'labels'}).dropna()

    wallet_train_df = wallet_train_df.rename(
        columns={'text': 'text_a', 'intent': 'text_b', 'scores': 'labels'}).dropna()
    wallet_eval_df = wallet_eval_df.rename(columns={'text': 'text_a', 'intent': 'text_b', 'scores': 'labels'}).dropna()

    # train_df = pd.concat([wallet_train_df, train_df])
    # train_df = pd.concat([wallet_train_df])
    # train_df = wallet_train_df
    test_df = eval_df

    # eval_df = pd.concat([wallet_eval_df, eval_df])
    num_samples = 50
    text_a = test_df.text_a.tolist()[:num_samples]
    text_b = test_df.text_b.tolist()[:num_samples]
    test_scores = torch.FloatTensor(test_df.labels.tolist()[:num_samples])
    start_time = time.time()
    text_enc_a = encoder_client.encode_sentences(text_a)
    text_enc_b = encoder_client.encode_sentences(text_b)
    hparams = hparamset()
    hparams.input_size = text_enc_b.shape[1]
    model = MultilingualSTS(hparams)
    load_model_state("model.pt", model)

    with torch.no_grad():
        model.eval()
        test_predict = model(from_numpy(text_enc_a), from_numpy(text_enc_b))
        print(pearson_corr(test_predict.view(-1), test_scores))
        print(spearman_corr(test_predict.view(-1), test_scores))

    print('Decoding time:{}'.format(time.time() - start_time))
