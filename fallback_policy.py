import logging
import time

import torch
from torch import from_numpy
import pandas as pd

from models.encoders import SbertEncoderClient, LaserEncoderClient, CombinedEncoderClient, encode_multiple_text_list

from models.classifier import MultilingualSTS
from models.eval import pearson_corr, spearman_corr
from models.model_utils import load_model_state, hparamset

glog = logging.getLogger(__name__)

if __name__ == '__main__':
    test_df = pd.read_csv('datasets/stack-exchange/test-stackexchange_with_sts_synthetic.csv')
    test_df = test_df[['text', 'intent', 'scores']]
    test_df = test_df.rename(columns={'text': 'text_a', 'intent': 'text_b',
                                      'scores': 'labels'}).dropna()
    num_samples = 5000
    test_a = test_df.text_a.tolist()[:num_samples]
    test_b = test_df.text_b.tolist()[:num_samples]
    test_scores = test_df.labels.tolist()[:num_samples]
    start_time = time.time()

    test_a_encoded, test_b_encoded = encode_multiple_text_list([test_a, test_b])

    hparams = hparamset()
    hparams.input_size = test_a_encoded.shape[1]
    model = MultilingualSTS(hparams)
    load_model_state("model-sts_stack-exchange_test.pt", model)

    with torch.no_grad():
        model.eval()
        test_predict = model(from_numpy(test_a_encoded), from_numpy(test_b_encoded))
        print(pearson_corr(test_predict.view(-1), test_scores))
        print(spearman_corr(test_predict.view(-1), test_scores))

    print('Decoding time:{}'.format(time.time() - start_time))
