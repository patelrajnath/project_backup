import torch
from torch import nn, cosine_similarity, from_numpy
from torch.nn import functional as F


class MultilingualSTS(nn.Module):
    def __init__(self, hparams):
        """
        Implementation of Regression model for Semantic Textual Similarity
        :param hparams:
        """
        super().__init__()
        self.hparams = hparams
        self.max_sts_score = hparams.max_sts_score
        self.classifier = nn.Sequential()

        input_size = hparams.input_size

        for i in range(hparams.num_hidden_layers):
            self.classifier.add_module('dropout{}'.format(i), nn.Dropout(hparams.dropout))
            self.classifier.add_module('ff{}'.format(i), nn.Linear(input_size, hparams.hidden_layer_size))
            self.classifier.add_module('activation{}'.format(i), nn.ReLU())
            input_size = hparams.hidden_layer_size
        # self.classifier.add_module('ff_output', nn.Linear(hparams.hidden_layer_size, 1))

    def forward(self, text_a, text_b):
        """
        The forward class of the Semantic Textual Similarity of text_a and text_b
        :param text_a:
        :param text_b:
        :return:
        """
        # concatenate = torch.cat([text_a, text_b], dim=1)
        # print(concatenate.size())
        logits_a = self.classifier(text_a)
        logits_b = self.classifier(text_b)
        similarity = F.cosine_similarity(logits_a, logits_b, dim=1)
        # print(similarity.size())
        sts_scores = similarity * self.max_sts_score
        # print(sts_scores)
        return sts_scores


class MultilingualClassifier(nn.Module):
    def __init__(self, hparams, num_labels):
        """
        Implementation of classification model for Text classification
        :param hparams:
        """
        super().__init__()

        self.hparams = hparams
        self.classifier = nn.Sequential()

        input_size = hparams.input_size
        for i in range(hparams.num_hidden_layers):
            self.classifier.add_module('dropout{}'.format(i), nn.Dropout(hparams.dropout))
            self.classifier.add_module('ff{}'.format(i), nn.Linear(input_size, hparams.hidden_layer_size))
            self.classifier.add_module('activation{}'.format(i), nn.ReLU())
            input_size = hparams.hidden_layer_size

        self.classifier.add_module('ff_output', nn.Linear(hparams.hidden_layer_size, num_labels))

    def forward(self, text):
        """
        The forward class of the Semantic Textual Similarity of text_a and text_b
        :param text:
        :return:
        """
        logits = self.classifier(text)
        prediction = F.log_softmax(logits, dim=1)
        return logits, prediction
