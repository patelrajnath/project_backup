import logging
import os
import time

import torch
import wandb
from sklearn.metrics import accuracy_score
from torch import nn, from_numpy
import pandas as pd

from data.batcher import SamplingBatcher
import numpy as np

from models.classifier import MultilingualClassifier
from models.encoders import SbertEncoderClient, LaserEncoderClient, CombinedEncoderClient
from models.model_utils import save_state, hparamset, set_seed

glog = logging.getLogger(__name__)

if __name__ == '__main__':
    train_df = pd.read_csv('NLU/64intent_11k_sample_split_MAIN/200924v2_train_custom_nlu_60_64intents.csv')
    test_df = pd.read_csv('NLU/64intent_11k_sample_split_MAIN/200924v2_test_custom_nlu_30_64intents.csv')
    eval_df = pd.read_csv('NLU/64intent_11k_sample_split_MAIN/200924v2_val_custom_nlu_10_64intents.csv')
    train_df = pd.concat([train_df, eval_df])

    num_samples = 50000
    file_suffix = 'nlu'
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
        train_text = train_df.text.tolist()[:num_samples]
        train_text_encoded = encoder_client.encode_sentences(train_text)
        test_text = test_df.text.tolist()[:num_samples]
        test_text_encoded = encoder_client.encode_sentences(test_text)
        np.savetxt('train-text_encoded_{}.txt'.format(file_suffix), train_text_encoded, fmt="%.8g")
        np.savetxt('test-text_encoded_{}.txt'.format(file_suffix), test_text_encoded, fmt="%.8g")
        print('Encoding time:{}'.format(time.time() - start_time))

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
        # Initialize WandB
        wandb.init()
        print('This wandb config', wandb.config)
        print("HyperParam lr=>>{}".format(wandb.config.learning_rate))
        print("HyperParam dropouts=>>{}".format(wandb.config.dropouts))
        print("HyperParam batch size=>>{}".format(wandb.config.batch_size))
        print("HyperParam num hidden layers=>>{}".format(wandb.config.num_hidden_layers))
        print("HyperParam hidden_layer_size=>>{}".format(wandb.config.hidden_layer_size))

        # update dropout
        hparams.learning_rate = wandb.config.learning_rate

        # update dropout
        hparams.dropout = wandb.config.dropouts

        # Update batch-size
        hparams.batch_size = wandb.config.batch_size

        # update num_hidden_layers
        hparams.num_hidden_layers = wandb.config.num_hidden_layers

        # update num_hidden_size
        hparams.hidden_layer_size = wandb.config.hidden_layer_size

        # Set seed
        set_seed(hparams.seed)

        batcher = SamplingBatcher(
            train_text_encoded, train_labels, hparams.batch_size, distribution)

        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")

        model = MultilingualClassifier(hparams, len(categories))
        model = model.to(device)

        wandb.watch(model)

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

                # Log the loss and accuracy values at the end of each epoch
                wandb.log({
                    "Updates": updates,
                    "Train Loss": total_loss,
                    "valid_acc": valid_acc,
                    "lr": wandb.config.learning_rate,
                    "batch_size": wandb.config.batch_size,
                    "dropouts": wandb.config.dropouts,
                    "num_hidden_layers": wandb.config.num_hidden_layers,
                    "hidden_layer_size": wandb.config.hidden_layer_size
                })
                # Reset the loss accumulation
                total_loss = 0
                # Change the model to training mode
                model.train()

            if updates % max_steps == 0:
                break
        save_state("model_classifier.pt", model, criterion, optimizer, num_updates=updates)

    # WandB Configurations (optional)
    sweep_config = {
        'method': 'random',  # grid, random
        'metric': {
            'name': 'valid_acc_pearson',
            'goal': 'maximize'
        },
        'parameters': {
            'learning_rate': {
                'values': [0.7, 0.8, 0.9, 0.95]
            },
            'dropouts': {
                'values': [0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
            },
            'batch_size': {
                'values': [64, 32, 16, 8]
            },
            'num_hidden_layers': {
                'values': [1, 2, 4]
            },
            'hidden_layer_size': {
                'values': [128, 256, 512, 1024, 2048, 3072]
            },
        }
    }

    sweep_id = wandb.sweep(sweep_config, project="multilingual_classifier_adam")
    # Call the wandb agent
    wandb.agent(sweep_id, function= lambda: train_model())
