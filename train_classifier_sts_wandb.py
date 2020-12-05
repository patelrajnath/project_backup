import logging
import time

import torch
import wandb
from torch import nn, from_numpy
import pandas as pd

from data.batcher import SamplingBatcherSTSClassification
import numpy as np

from models.classifier import MultilingualSTS
from models.eval import pearson_corr, spearman_corr
from models.model_utils import save_state, hparamset, set_seed

glog = logging.getLogger(__name__)


if __name__ == '__main__':
    # train_df = pd.read_csv('sample-data/STS-B/train.tsv', sep='\t', error_bad_lines=False)
    # eval_df = pd.read_csv('sample-data/STS-B/dev.tsv', sep='\t', error_bad_lines=False)
    # train_df = train_df.rename(columns={'sentence1': 'text_a',
    #                                     'sentence2': 'text_b', 'score': 'labels'}).dropna()
    # eval_df = eval_df.rename(columns={'sentence1': 'text_a',
    #                                   'sentence2': 'text_b', 'score': 'labels'}).dropna()

    wallet_train_df = pd.read_csv('sample-data/200410_train_stratshuf_english_with_sts_synthesis.csv')
    wallet_eval_df = pd.read_csv('sample-data/200410_test_stratshuf_chinese_200410_'
                                 'english_with_sts_synthesis.csv')

    wallet_train_df = wallet_train_df.rename(
        columns={'text': 'text_a', 'intent': 'text_b', 'scores': 'labels'}).dropna()
    wallet_eval_df = wallet_eval_df.rename(columns={'text': 'text_a',
                                                    'intent': 'text_b', 'scores': 'labels'}).dropna()

    # train_df = pd.concat([wallet_train_df, train_df])
    # train_df = pd.concat([wallet_train_df])
    # train_df = wallet_train_df
    test_df = wallet_eval_df
    train_df = wallet_train_df

    # eval_df = pd.concat([wallet_eval_df, eval_df])
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
    # train_a = train_df.text_a.tolist()[:num_samples]
    # train_b = train_df.text_b.tolist()[:num_samples]
    # text_a_encoded = encoder_client.encode_sentences(train_a)
    # text_b_encoded = encoder_client.encode_sentences(train_b)
    # text_a = test_df.text_a.tolist()[:num_samples]
    # text_b = test_df.text_b.tolist()[:num_samples]
    # text_enc_a = encoder_client.encode_sentences(text_a)
    # text_enc_b = encoder_client.encode_sentences(text_b)
    # np.savetxt('train_a_encoded_wallet.txt', text_a_encoded, fmt="%.8g")
    # np.savetxt('train_b_encoded_wallet.txt', text_b_encoded, fmt="%.8g")
    # np.savetxt('test_a_encoded_wallet.txt', text_enc_a, fmt="%.8g")
    # np.savetxt('test_b_encoded_wallet.txt', text_enc_b, fmt="%.8g")
    # print('Encoding time:{}'.format(time.time() - start_time))
    # exit()

    hparams = hparamset()
    test_scores = test_df.labels.tolist()[:num_samples]
    train_scores = train_df.labels.tolist()[:num_samples]
    text_a_encoded = np.loadtxt('train_a_encoded_wallet.txt')
    text_b_encoded = np.loadtxt('train_b_encoded_wallet.txt')
    text_enc_a = np.loadtxt('test_a_encoded_wallet.txt')
    text_enc_b = np.loadtxt('test_b_encoded_wallet.txt')

    text_a_encoded = text_a_encoded.astype(np.float32)
    text_b_encoded = text_b_encoded.astype(np.float32)
    text_enc_a = text_enc_a.astype(np.float32)
    text_enc_b = text_enc_b.astype(np.float32)

    train_scores = np.asarray(train_scores, dtype=np.float32)
    test_scores = np.asarray(test_scores, dtype=np.float32)
    n_samples = text_a_encoded.shape[0]
    hparams.input_size = text_a_encoded.shape[1]

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

        # update num_hidden_size
        hparams.seed = wandb.config.seed

        set_seed(hparams.seed)

        start_time = time.time()
        batcher = SamplingBatcherSTSClassification(text_a_encoded, text_b_encoded, train_scores,
                                     batch_size=hparams.batch_size)

        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")

        model = MultilingualSTS(hparams)
        model = model.to(device)

        wandb.watch(model)

        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5)
        # optimizer = NoamOpt(hparams.input_size, 1, 200,
        #                     torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-9))
        optimizer = torch.optim.Adam(model.parameters(), lr=hparams.learning_rate,
                                     betas=(0.9, 0.98), eps=1e-9)
        criterion = nn.MSELoss()
        # optimizer = torch.optim.SGD(model.parameters(), lr=hparams.learning_rate)

        # for epoch in range(hparams.epochs):
        total_loss = 0
        updates = 1
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
            updates += 1
            if updates % hparams.patience == 0:
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

                # Log the loss and accuracy values at the end of each epoch
                wandb.log({
                    "Updates": updates,
                    "Train Loss": total_loss,
                    "valid_acc_pearson": valid_acc_pearson,
                    "valid_acc_spearman": valid_acc_spearman,
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

            if updates % hparams.max_steps == 0:
                break
        save_state("model-sts-wallet.pt", model, criterion, optimizer, num_updates=updates)
        with torch.no_grad():
            model.eval()
            test_a_tensor = from_numpy(text_enc_a).to(device)
            test_b_tensor = from_numpy(text_enc_b).to(device)
            test_predict = model(test_a_tensor, test_b_tensor)
            score_tensor = from_numpy(test_scores).to(device)
            print(pearson_corr(test_predict.cpu().data, score_tensor.cpu().data))
            print(spearman_corr(test_predict.cpu().data, score_tensor.cpu().data))
        print('Training time:{}'.format(time.time() - start_time))

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
                'values': [1, 2]
            },
            'hidden_layer_size': {
                'values': [128, 256, 512, 1024]
            },
            'seed': {
                'values': [i for i in range(0, 10)]
            },
        }
    }

    sweep_id = wandb.sweep(sweep_config, project="multilingual_sts_adam_with_seed-test")
    # Call the wandb agent
    wandb.agent(sweep_id, function=lambda: train_model())
    # train_model()
