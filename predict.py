import logging
import time

import torch
from torch import from_numpy
import pandas as pd

from models.encoders import SbertEncoderClient, LaserEncoderClient, CombinedEncoderClient

from models.classifier import MultilingualSTS
from models.eval import pearson_corr, spearman_corr
from models.model_utils import load_model_state, hparamset

glog = logging.getLogger(__name__)

if __name__ == '__main__':
    sbert_model = 'distiluse-base-multilingual-cased'
    sbert_model2 = 'xlm-r-100langs-bert-base-nli-stsb-mean-tokens'
    sbert_model3 = 'distilbert-multilingual-nli-stsb-quora-ranking'
    sbert_encoder = SbertEncoderClient(sbert_model)
    sbert_encoder2 = SbertEncoderClient(sbert_model2)
    sbert_encoder3 = SbertEncoderClient(sbert_model3)
    laser_encoder = LaserEncoderClient()
    encoder_client = CombinedEncoderClient([laser_encoder, sbert_encoder, sbert_encoder2, sbert_encoder3])

    train_df = pd.read_csv('datasets/STS-B/train.tsv', sep='\t', error_bad_lines=False)
    test_df = pd.test_df('datasets/STS-B/dev.tsv', sep='\t', error_bad_lines=False)

    train_df = train_df.rename(columns={'sentence1': 'text_a', 'sentence2': 'text_b', 'score': 'labels'}).dropna()
    test_df = test_df.rename(columns={'sentence1': 'text_a', 'sentence2': 'text_b', 'score': 'labels'}).dropna()

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
