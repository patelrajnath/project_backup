import logging

import torch
import wandb
from sklearn.metrics import accuracy_score
from torch import nn, from_numpy
import pandas as pd

from data.batcher import SamplingBatcher
import numpy as np

from models.classifier import MultilingualClassifier
from models.model_utils import save_state, hparamset, set_seed

glog = logging.getLogger(__name__)

if __name__ == '__main__':
    train_df = pd.read_csv('sample-data/HuaweiWallet/200909_train_huawei_wallet_60strat_min3_manyanswers.csv')
    test_df = pd.read_csv('sample-data/HuaweiWallet/200909_test_huawei_wallet_30strat_min3_manyanswers.csv')
    eval_df = pd.read_csv('sample-data/HuaweiWallet/200909_val_huawei_wallet_10strat_min3_manyanswers.csv')

    train_df = pd.concat([train_df, eval_df])

    num_samples = 50000
    # start_time = time.time()
    # sbert_model = 'distiluse-base-multilingual-cased'
    # sbert_model2 = 'xlm-r-100langs-bert-base-nli-stsb-mean-tokens'
    # sbert_model3 = 'distilbert-multilingual-nli-stsb-quora-ranking'
    # sbert_encoder = SbertEncoderClient(sbert_model)
    # sbert_encoder2 = SbertEncoderClient(sbert_model2)
    # sbert_encoder3 = SbertEncoderClient(sbert_model3)
    # laser_encoder = LaserEncoderClient()
    # encoder_client = CombinedEncoderClient([laser_encoder, sbert_encoder,
    #                                         sbert_encoder2, sbert_encoder3])
    # encoder_client = CombinedEncoderClient([laser_encoder])
    # train_text = train_df.text.tolist()[:num_samples]
    # train_text_encoded = encoder_client.encode_sentences(train_text)
    # test_text = test_df.text.tolist()[:num_samples]
    # test_text_encoded = encoder_client.encode_sentences(test_text)
    # np.savetxt('train-text_encoded.txt', train_text_encoded, fmt="%.8g")
    # np.savetxt('test-text_encoded.txt', test_text_encoded, fmt="%.8g")
    # print('Encoding time:{}'.format(time.time() - start_time))
    # exit()

    hparams = hparamset()
    test_labels = test_df.labels.tolist()[:num_samples]
    train_labels = train_df.labels.tolist()[:num_samples]
    train_text_encoded = np.loadtxt('train-text_encoded.txt')
    test_text_encoded = np.loadtxt('test-text_encoded.txt')
    train_text_encoded = train_text_encoded.astype(np.float32)
    test_text_encoded = test_text_encoded.astype(np.float32)

    train_labels = np.asarray(train_labels)
    test_labels = np.asarray(test_labels)
    n_samples = train_text_encoded.shape[0]
    hparams.input_size = train_text_encoded.shape[1]

    categories = [i for i in set(train_labels)]

    distribution = None if not hparams.balance_data else {
        x: 1. / len(categories) for x in range(len(categories))}

    def train_model():
        # Set seed
        set_seed(hparams.seed)

        batcher = SamplingBatcher(
            train_text_encoded, train_labels, hparams.batch_size, distribution)

        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")

        model = MultilingualClassifier(hparams, len(categories))
        model = model.to(device)
        # optimizer = torch.optim.SGD(model.parameters(), lr=hparams.learning_rate)
        # optimizer = NoamOpt(hparams.input_size, 1, 200,
        #                     torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-9))
        optimizer = torch.optim.Adam(model.parameters(),
                                     betas=(0.9, 0.98), eps=1e-9)

        criterion = nn.NLLLoss()
        total_loss = 0
        updates = 1
        max_steps = 15000
        patience = 500

        for batch in batcher:
            optimizer.zero_grad()
            text_encoded, labels = batch
            labels = torch.tensor(labels, dtype=torch.long).to(device)
            text_encoded = from_numpy(text_encoded).to(device)
            logits, prediction = model(text_encoded)
            loss = criterion(prediction, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.data
            updates += 1
            if updates % patience == 0:
                print("Loss:{0}, updates:{1}".format(total_loss / updates, updates))
                with torch.no_grad():
                    model.eval()
                    score_tensor = test_labels
                    test_text_tensor = from_numpy(test_text_encoded).to(device)
                    _, test_predictions = model(test_text_tensor)
                    predictions = test_predictions.argmax(dim=1)
                    valid_acc = accuracy_score(predictions.cpu().data.numpy(), score_tensor)
                    print("Valid Accuracy:{}".format(valid_acc))
                # Reset the loss accumulation
                total_loss = 0
                # Change the model to training mode
                model.train()

            if updates % max_steps == 0:
                break

        save_state("model_classifier.pt", model, criterion, optimizer, num_updates=updates)