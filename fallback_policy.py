import logging
import time

import torch
import numpy as np

from torch.nn import functional as F
from sklearn.metrics import accuracy_score
from torch import from_numpy
import pandas as pd

from models.encoders import SbertEncoderClient, LaserEncoderClient, CombinedEncoderClient, encode_multiple_text_list, \
    CachingEncoderClient

from models.classifier import MultilingualSTS, MultilingualClassifier
from models.eval import pearson_corr, spearman_corr
from models.model_utils import load_model_state, hparamset

glog = logging.getLogger(__name__)

if __name__ == '__main__':
    test_df_sts = pd.read_csv('datasets/stack-exchange/test-stackexchange_with_sts_synthetic.csv')
    test_df = pd.read_csv('datasets/stack-exchange/test-stackexchange.csv')
    train_df = pd.read_csv('datasets/stack-exchange/train-stackexchange.csv')

    intent_map_unique = dict()
    for id, row in train_df.iterrows():
        label = row['intentId']
        intent = row['intent']
        if label not in intent_map_unique:
            intent_map_unique[label] = intent
    print(len(intent_map_unique))

    test = test_df.text.tolist()
    test_labels = test_df.intentId.tolist()
    train_labels = train_df.intentId.tolist()
    categories = [i for i in set(train_labels)]

    num_samples = 5000
    test_a = test_df_sts.text.tolist()[:num_samples]
    test_b = test_df_sts.intent.tolist()[:num_samples]
    test_scores = test_df_sts.scores.tolist()[:num_samples]

    test_a_encoded, test_b_encoded, test_encoded = encode_multiple_text_list([test_a, test_b, test])

    sbert_model = 'distiluse-base-multilingual-cased'
    sbert_model2 = 'xlm-r-100langs-bert-base-nli-stsb-mean-tokens'
    sbert_model3 = 'distilbert-multilingual-nli-stsb-quora-ranking'
    sbert_encoder = SbertEncoderClient(sbert_model)
    sbert_encoder2 = SbertEncoderClient(sbert_model2)
    sbert_encoder3 = SbertEncoderClient(sbert_model3)
    laser_encoder = LaserEncoderClient()
    encoder_client = CombinedEncoderClient([laser_encoder, sbert_encoder,
                                            sbert_encoder2, sbert_encoder3])

    cached_encoder = CachingEncoderClient(encoder_client, cache_dir='cache-encoding',
                                          encoder_id='combined_encoders')

    hparams = hparamset()
    hparams.input_size = test_a_encoded.shape[1]
    model = MultilingualSTS(hparams)
    load_model_state("model_best-sts_stack.pt", model)

    # +1 in num_labels to accommodate in case the labeling is started from 1 instead of 0
    model_classifier = MultilingualClassifier(hparams, num_labels=len(categories) + 1)
    load_model_state("model_best_classifier_stack.pt", model_classifier)

    # with torch.no_grad():
    #     model.eval()
    #     test_predict = model(from_numpy(test_a_encoded), from_numpy(test_b_encoded))
    #     print(pearson_corr(test_predict.view(-1), test_scores))
    #     print(spearman_corr(test_predict.view(-1), test_scores))

    with torch.no_grad():
        model_classifier.eval()
        score_tensor = test_labels
        test_text_tensor = from_numpy(test_encoded)
        _, test_predictions = model_classifier(test_text_tensor)
        predictions = test_predictions.argmax(dim=1)
        valid_acc = accuracy_score(predictions.cpu().data.numpy(), score_tensor)
        print("Valid Accuracy:{}".format(valid_acc))

    start_time = time.time()
    with torch.no_grad():
        model_classifier.eval()
        model.eval()
        predictions = []
        processed = 0
        rasa_def = True
        for id, row in test_df.iterrows():
            text = [row.text]
            label = row.intentId
            text_encoding = cached_encoder.encode_sentences(text)
            logits, test_predictions = model_classifier(from_numpy(text_encoding))
            logits_prob = F.softmax(logits, dim=1)
            prob, topk = torch.topk(logits_prob, k=5)
            topk_list = topk.data.tolist()[0]
            prob_top = prob.data.tolist()[0][0]
            predict = topk_list[0]
            top_2 = topk_list[:2]
            threshold = 0.85

            if rasa_def:
                print(top_2, predict, label)
                if label in top_2:
                    predictions.append(label)
                else:
                    predictions.append(predict)
            else:
                if label == predict:
                    predictions.append(predict)

                elif prob_top < threshold:
                    processed += 1
                    topk_list_exclude_first = topk_list[1:]
                    first = topk_list[0]
                    topk_text_a_encoded = []
                    topk_text_b = []
                    topk_text_a = []
                    for key in topk_list_exclude_first:
                        topk_text_a_encoded.append(text_encoding[0])
                        # topk_text_a.append(intent_map_unique.get(first, "This is default text"))
                        topk_text_b.append(intent_map_unique.get(key))

                    topk_text_a_encoded = np.asarray(topk_text_a_encoded)
                    # topk_text_a_encoded = cached_encoder.encode_sentences(topk_text_a)

                    topk_text_b_encoded = cached_encoder.encode_sentences(topk_text_b)

                    semantic_similarity_score = model(from_numpy(topk_text_a_encoded),
                                                      from_numpy(topk_text_b_encoded))
                    indice = np.argmax(semantic_similarity_score)
                    predictions.append(topk_list_exclude_first[indice])
                else:
                    predictions.append(predict)
        print("Processed with sts:{}".format(processed))
        print(accuracy_score(predictions, test_labels))
    print('Decoding time:{}'.format(time.time() - start_time))
