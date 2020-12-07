import abc
import logging
import os
import pickle
import time

import numpy as np
from laserembeddings import Laser
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


batch_size_super = 128
glog = logging.getLogger(__name__)


class ClassificationEncoderClient(object):
    """A model that maps from text to dense vectors."""
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def encode_sentences(self, sentences):
        """Encodes a list of sentences

        Args:
            sentences: a list of strings

        Returns:
            an (N, d) numpy matrix of sentence encodings.
        """
        return NotImplementedError


class CombinedEncoderClient(ClassificationEncoderClient):
    """concatenates the encodings of several ClassificationEncoderClients

    Args:
        encoders: A list of ClassificationEncoderClients
    """

    def __init__(self, encoders: list):
        """constructor"""
        self._encoders = encoders

    def encode_sentences(self, sentences):
        """Encode a list of sentences

        Args:
            sentences: the list of sentences

        Returns:
            an array with shape (len(sentences), ENCODING_SIZE)
        """
        encodings = np.hstack([encoder.encode_sentences(sentences)
                               for encoder in self._encoders])
        print('DEBUG combined size:', encodings.shape)
        return encodings


class LaserEncoderClient(ClassificationEncoderClient):
    """A wrapper around ClassificationEncoderClient to normalise the output"""

    def __init__(self, batch_size=batch_size_super):
        """Create a new ConvertEncoderClient object

        Args:
            uri: The uri to the tensorflow_hub module
            batch_size: maximum number of sentences to encode at once
        """
        self._batch_size = batch_size
        self._encoder_client = Laser()

    def encode_sentences(self, sentences):
        """Encode a list of sentences

        Args:
            sentences: the list of sentences

        Returns:
            an (N, d) numpy matrix of sentence encodings.
        """
        encodings = []
        #         glog.setLevel("ERROR")
        for i in tqdm(range(0, len(sentences), self._batch_size),
                      "encoding sentence batches"):
            encodings.append(
                self._encoder_client.embed_sentences(
                    sentences[i:i + self._batch_size], lang='en'))
        #         glog.setLevel("INFO")
        print('DEBUG LASER SIZE:', np.vstack(encodings).shape)
        return l2_normalize(np.vstack(encodings))


class SbertEncoderClient(ClassificationEncoderClient):
    """A wrapper around ClassificationEncoderClient to normalise the output"""

    def __init__(self, sbert_model, batch_size=batch_size_super):
        """Create a new ConvertEncoderClient object

        Args:
            uri: The uri to the tensorflow_hub module
            batch_size: maximum number of sentences to encode at once
        """
        self._batch_size = batch_size
        self._encoder_client = SentenceTransformer(sbert_model)

    def encode_sentences(self, sentences):
        """Encode a list of sentences

        Args:
            sentences: the list of sentences

        Returns:
            an (N, d) numpy matrix of sentence encodings.
        """
        encodings = []
        #         glog.setLevel("ERROR")
        for i in tqdm(range(0, len(sentences), self._batch_size),
                      "encoding sentence batches"):
            encodings.append(
                self._encoder_client.encode(
                    sentences[i:i + self._batch_size]))
        #         glog.setLevel("INFO")
        return l2_normalize(np.vstack(encodings))


def l2_normalize(encodings):
    """L2 normalizes the given matrix of encodings."""
    norms = np.linalg.norm(encodings, ord=2, axis=-1, keepdims=True)
    return encodings / norms


def encoding_cache_read():
    with open('encoding_cache.json', '') as cache_file:
        pass


def encoding_cache_write():
    with open('encoding_cache.json', '') as cache_file:
        pass


def encode_multiple_text_list(text_list):
    sbert_model = 'distiluse-base-multilingual-cased'
    sbert_model2 = 'xlm-r-100langs-bert-base-nli-stsb-mean-tokens'
    sbert_model3 = 'distilbert-multilingual-nli-stsb-quora-ranking'
    sbert_encoder = SbertEncoderClient(sbert_model)
    sbert_encoder2 = SbertEncoderClient(sbert_model2)
    sbert_encoder3 = SbertEncoderClient(sbert_model3)
    laser_encoder = LaserEncoderClient()
    encoder_client = CombinedEncoderClient([laser_encoder, sbert_encoder,
                                            sbert_encoder2, sbert_encoder3])

    class CachingEncoderClient(object):
        """Wrapper around an encoder to cache the encodings on disk"""
        def __init__(self, encoder_client, encoder_id, cache_dir):
            """Create a new CachingEncoderClient object
            Args:
                encoder_client: An EncoderClient
                encoder_id: An unique ID for the encoder
                cache_dir: The directory where the encodings will be cached
            """
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
            self._encodings_dict_path = os.path.join(
                cache_dir, encoder_id)
            self._encoder_client = encoder_client
            self._encodings_dict = self._load_or_create_encodings_dict()

        def _load_or_create_encodings_dict(self):
            if os.path.exists(self._encodings_dict_path):
                with open(self._encodings_dict_path, "rb") as f:
                    encodings_dict = pickle.load(f)
            else:
                encodings_dict = {}
            return encodings_dict

        def _save_encodings_dict(self):
            with open(self._encodings_dict_path, "wb") as f:
                pickle.dump(self._encodings_dict, f)

        def encode_sentences(self, sentences):
            """Encode a list of sentences

            Args:
                sentences: the list of sentences

            Returns:
                an (N, d) numpy matrix of sentence encodings.
            """
            missing_sentences = [
                sentence for sentence in sentences
                if sentence not in self._encodings_dict]
            if len(sentences) != len(missing_sentences):
                glog.info(f"{len(sentences) - len(missing_sentences)} cached "
                          f"sentences will not be encoded")
            if missing_sentences:
                missing_encodings = self._encoder_client.encode_sentences(
                    missing_sentences)
                for sentence, encoding in zip(missing_sentences,
                                              missing_encodings):
                    self._encodings_dict[sentence] = encoding
                self._save_encodings_dict()

            encodings = np.array(
                [self._encodings_dict[sentence] for sentence in sentences])
            return encodings

    text_all = list()
    for t in text_list:
        text_all += t
    text_all_unique = list(set(text_all))
    cached_encoder = CachingEncoderClient(encoder_client, cache_dir='cache-encoding',
                                          encoder_id='combined_encoders')
    text_all_unique_encoded = cached_encoder.encode_sentences(text_all_unique)

    text_encoding_map = dict()
    for text, encoding in zip(text_all_unique, text_all_unique_encoded):
        text_encoding_map[text] = encoding

    text_list_encoded = []
    for t in text_list:
        t_encoded = list()
        for a in t:
            t_encoded.append(text_encoding_map[a])
        text_list_encoded.append(np.asarray(t_encoded))

    return text_list_encoded
