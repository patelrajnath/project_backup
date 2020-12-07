import logging
import os
import time

import torch
from sklearn.metrics import accuracy_score
from torch import nn, from_numpy
import pandas as pd

from data.batcher import SamplingBatcher
import numpy as np

from models.classifier import MultilingualClassifier
from models.encoders import encode_multiple_text_list
from models.model_utils import save_state, hparamset, set_seed

glog = logging.getLogger(__name__)

if __name__ == '__main__':
    # train_df = pd.read_csv('datasets/NLU/64intent_11k_sample_split_MAIN/200924v2_train_custom_nlu_60_64intents.csv')
    # test_df = pd.read_csv('datasets/NLU/64intent_11k_sample_split_MAIN/200924v2_test_custom_nlu_30_64intents.csv')
    # eval_df = pd.read_csv('datasets/NLU/64intent_11k_sample_split_MAIN/200924v2_val_custom_nlu_10_64intents.csv')
    # train_df = pd.concat([train_df, eval_df])

    train_df = pd.read_csv('datasets/stack-exchange/train-stackexchange.csv')
    test_df = pd.read_csv('datasets/stack-exchange/test-stackexchange.csv')
    train_df = train_df.rename(columns={'intentId': 'labels'}).dropna()
    test_df = test_df.rename(columns={'intentId': 'labels'}).dropna()

    num_samples = 50000
    # file_suffix = 'stack-exchange-classifier'
    file_suffix = 'stack'
    start_time = time.time()
    train_text = train_df.text.tolist()[:num_samples]
    train_text = [str(t) for t in train_text]

    test_text = test_df.text.tolist()[:num_samples]
    test_text = [str(t) for t in test_text]

    # The encoding method returns list of text in the same order as given in the input
    train_text_encoded, test_text_encoded = encode_multiple_text_list([train_text, test_text])
    print('Encoding time:{}'.format(time.time() - start_time))

    hparams = hparamset()
    test_labels = test_df.labels.tolist()[:num_samples]
    train_labels = train_df.labels.tolist()[:num_samples]

    train_labels = np.asarray(train_labels)
    test_labels = np.asarray(test_labels)
    n_samples = train_text_encoded.shape[0]
    hparams.input_size = train_text_encoded.shape[1]

    categories = [int(i) for i in set(train_labels)]
    distribution = None if not hparams.balance_data else {
        x: 1. / len(categories) for x in range(len(categories))}

    def train_model():
        # Set seed
        set_seed(hparams.seed)

        batcher = SamplingBatcher(
            train_text_encoded, train_labels, hparams.batch_size, distribution)

        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")
        # +1 in num_labels to accommodate in case the labeling is started from 1 instead of 0
        model = MultilingualClassifier(hparams, num_labels=len(categories) + 1)
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
        best_accuracy = 0
        for batch in batcher:
            optimizer.zero_grad()
            text_encoded, labels = batch
            # labels = torch.tensor(labels, dtype=torch.long).to(device)
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
                    if best_accuracy< valid_acc:
                        save_state("model_best_classifier_{}.pt".format(file_suffix), model,
                                   criterion, optimizer, num_updates=updates)
                        best_accuracy = valid_acc
                # Reset the loss accumulation
                total_loss = 0
                # Change the model to training mode
                model.train()

            if updates % max_steps == 0:
                break
        save_state("model_classifier_{}.pt".format(file_suffix), model, criterion, optimizer, num_updates=updates)

    start_time = time.time()
    train_model()
    print('Training time:{}'.format(time.time() - start_time))

