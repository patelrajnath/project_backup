import logging
import os
import time

import torch
from torch import nn, from_numpy
import pandas as pd

from data.batcher import SamplingBatcherSTS
import numpy as np

from models.classifier import MultilingualSTS
from models.encoders import SbertEncoderClient, LaserEncoderClient, CombinedEncoderClient
from models.eval import pearson_corr, spearman_corr
from models.model_utils import save_state, hparamset, set_seed

glog = logging.getLogger(__name__)

if __name__ == '__main__':
    train_df = pd.read_csv('datasets/STS-B/train.tsv', sep='\t', error_bad_lines=False)
    test_df = pd.read_csv('datasets/STS-B/dev.tsv', sep='\t', error_bad_lines=False)

    train_df = train_df.rename(columns={'sentence1': 'text_a',
                                        'sentence2': 'text_b', 'score': 'labels'}).dropna()
    test_df = test_df.rename(columns={'sentence1': 'text_a',
                                      'sentence2': 'text_b', 'score': 'labels'}).dropna()
    num_samples = 50000
    file_suffix = 'datasets/STS-B'
    if not os.path.isfile('train_a_encoded_{}.txt'.format(file_suffix)):
        start_time = time.time()
        sbert_model = 'distiluse-base-multilingual-cased'
        sbert_model2 = 'xlm-r-100langs-bert-base-nli-stsb-mean-tokens'
        sbert_model3 = 'distilbert-multilingual-nli-stsb-quora-ranking'
        sbert_encoder = SbertEncoderClient(sbert_model)
        sbert_encoder2 = SbertEncoderClient(sbert_model2)
        sbert_encoder3 = SbertEncoderClient(sbert_model3)
        laser_encoder = LaserEncoderClient()
        encoder_client = CombinedEncoderClient([laser_encoder, sbert_encoder,
                                                sbert_encoder2, sbert_encoder3])
        train_a = train_df.text_a.tolist()[:num_samples]
        train_b = train_df.text_b.tolist()[:num_samples]
        text_a_encoded = encoder_client.encode_sentences(train_a)
        text_b_encoded = encoder_client.encode_sentences(train_b)
        text_a = test_df.text_a.tolist()[:num_samples]
        text_b = test_df.text_b.tolist()[:num_samples]
        text_enc_a = encoder_client.encode_sentences(text_a)
        text_enc_b = encoder_client.encode_sentences(text_b)
        np.savetxt('train_a_encoded_{}.txt'.format(file_suffix), text_a_encoded, fmt="%.8g")
        np.savetxt('train_b_encoded_{}.txt'.format(file_suffix), text_b_encoded, fmt="%.8g")
        np.savetxt('test_a_encoded_{}.txt'.format(file_suffix), text_enc_a, fmt="%.8g")
        np.savetxt('test_b_encoded_{}.txt'.format(file_suffix), text_enc_b, fmt="%.8g")
        print('Encoding time:{}'.format(time.time() - start_time))

    hparams = hparamset()
    test_scores = test_df.labels.tolist()[:num_samples]
    train_scores = train_df.labels.tolist()[:num_samples]
    text_a_encoded = np.loadtxt('train_a_encoded_{}.txt'.format(file_suffix))
    text_b_encoded = np.loadtxt('train_b_encoded_{}.txt'.format(file_suffix))
    text_enc_a = np.loadtxt('test_a_encoded_{}.txt'.format(file_suffix))
    text_enc_b = np.loadtxt('test_b_encoded_{}.txt'.format(file_suffix))

    text_a_encoded = text_a_encoded.astype(np.float32)
    text_b_encoded = text_b_encoded.astype(np.float32)
    text_enc_a = text_enc_a.astype(np.float32)
    text_enc_b = text_enc_b.astype(np.float32)

    train_scores = np.asarray(train_scores, dtype=np.float32)
    test_scores = np.asarray(test_scores, dtype=np.float32)
    n_samples = text_a_encoded.shape[0]
    hparams.input_size = text_a_encoded.shape[1]

    def train_model():
        set_seed(hparams.seed)

        start_time = time.time()
        batcher = SamplingBatcherSTS(text_a_encoded, text_b_encoded, train_scores,
                                     batch_size=hparams.batch_size)

        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")

        model = MultilingualSTS(hparams)
        model = model.to(device)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5)
        # optimizer = NoamOpt(hparams.input_size, 1, 200,
        #                     torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-9))
        optimizer = torch.optim.Adam(model.parameters(),
                                     betas=(0.9, 0.98), eps=1e-9)

        criterion = nn.MSELoss()
        # optimizer = torch.optim.SGD(model.parameters(), lr=hparams.learning_rate)

        for epoch in range(hparams.epochs):
            total_loss = 0
            model.train()
            for batch in batcher:
                optimizer.zero_grad()
                a, b, score = batch
                score_tensor = from_numpy(score)
                score_tensor = score_tensor.to(device)
                a_tensor = from_numpy(a).to(device)
                b_tensor = from_numpy(b).to(device)

                logits = model(a_tensor, b_tensor)
                loss = criterion(logits, score_tensor)

                loss.backward()
                optimizer.step()
                total_loss += loss.data

            # scheduler.step(epoch)
            print("Loss:{}".format(total_loss))
            with torch.no_grad():
                model.eval()
                score_tensor = from_numpy(test_scores).to(device)
                test_a_tensor = from_numpy(text_enc_a).to(device)
                test_b_tensor = from_numpy(text_enc_b).to(device)
                test_predict = model(test_a_tensor, test_b_tensor)
                valid_acc_pearson = pearson_corr(test_predict.cpu().data, score_tensor.cpu().data)
                valid_acc_spearman = spearman_corr(test_predict.cpu().data, score_tensor.cpu().data)
                print("Valid Accuracy:{0}, {1}".format(valid_acc_pearson, valid_acc_spearman))

        save_state("model-sts_{}.pt".format(file_suffix), model, criterion, optimizer, num_updates=0)

        with torch.no_grad():
            model.eval()
            test_a_tensor = from_numpy(text_enc_a).to(device)
            test_b_tensor = from_numpy(text_enc_b).to(device)
            test_predict = model(test_a_tensor, test_b_tensor)
            score_tensor = from_numpy(test_scores).to(device)
            print(pearson_corr(test_predict.cpu().data, score_tensor.cpu().data))
            print(spearman_corr(test_predict.cpu().data, score_tensor.cpu().data))
        print('Training time:{}'.format(time.time() - start_time))
    train_model()
