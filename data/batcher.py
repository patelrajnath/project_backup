import collections
from typing import Optional, Dict

import numpy as np


class SamplingBatcherSTSClassification(collections.abc.Iterator):
    """Batcher that samples according to a given distribution.

    It defaults to sampling from the data distribution.

    WARNING: this class is not deterministic. if you want deterministic
    behaviour, just freeze the numpy seed.
    """

    def __init__(
            self,
            examples_a: np.ndarray,
            examples_b: np.ndarray,
            labels: np.ndarray,
            batch_size: int,
            sample_distribution: Optional[Dict[int, float]] = None,
    ):
        """Create a new BalancedBatcher.

        Args:
            examples_a: np.ndarray containing examples
            examples_b: np.ndarray containing examples
            labels: np.ndarray containing labels
            batch_size: int size of a single batch
        """
        _validate_labels_examples(examples_a, labels)
        self._examples_a = examples_a
        self._examples_b = examples_b
        self._labels = labels
        self._label_classes = np.unique(labels)
        self._class_to_indices = {
            label: np.argwhere(labels == label).flatten()
            for label in self._label_classes
        }
        if sample_distribution is None:
            # Default to the data distribution
            sample_distribution = {
                label: float(indices.size)
                for label, indices in self._class_to_indices.items()
            }
        self._label_choices, self._label_probs = (
            self._get_label_choices_and_probs(sample_distribution))
        self._batch_size = batch_size

    def _get_label_choices_and_probs(self, sample_distribution):
        label_choices = []
        label_probs = []
        weight_sum = sum(sample_distribution.values())
        for label, weight in sample_distribution.items():
            if label not in self._labels:
                raise ValueError(
                    f"label {label} in sample distribution does not exist")
            if weight < 0.0:
                raise ValueError(
                    f"weight {weight} for label {label} is negative")
            label_choices.append(label)
            label_probs.append(weight / weight_sum)

        return np.array(label_choices), np.array(label_probs)

    def __next__(self):
        """Generates the next batch.

        Returns:
            (batch_of_examples, batch_of_labels) - a tuple of ndarrays
        """
        class_choices = np.random.choice(
            self._label_choices, size=self._batch_size, p=self._label_probs)

        batch_indices = []
        for class_choice in class_choices:
            indices = self._class_to_indices[class_choice]
            batch_indices.append(np.random.choice(indices))

            return self._examples_a[batch_indices], \
                   self._examples_b[batch_indices], self._labels[batch_indices]

    def __iter__(self):
        """Gets an iterator for this iterable

        Returns:
            self because the class is an iterator itself
        """
        return self


class SamplingBatcherSTS(collections.abc.Iterator):
    """Batcher that samples according to a given distribution.

    It defaults to sampling from the data distribution.

    WARNING: this class is not deterministic. if you want deterministic
    behaviour, just freeze the numpy seed.
    """

    def __init__(
            self,
            examples_a: np.ndarray,
            examples_b: np.ndarray,
            labels: np.ndarray,
            batch_size: int,
    ):
        """Create a new BalancedBatcher.

        Args:
            examples_a: np.ndarray containing examples
            examples_b: np.ndarray containing examples
            labels: np.ndarray containing labels
            batch_size: int size of a single batch
        """
        self._num_items = labels.size
        self._examples_a = examples_a
        self._examples_b = examples_b
        self._labels = labels
        self._batch_size = batch_size
        self._indices = np.arange(self._num_items)
        self.rnd = np.random.RandomState(0)
        self.ptr = 0

    def __next__(self):
        """Generates the next batch.

        Returns:
            (batch_of_examples, batch_of_labels) - a tuple of ndarrays
        """
        if self.ptr + self._batch_size > self._num_items:
            self.rnd.shuffle(self._indices)
            self.ptr = 0
            raise StopIteration  # ugly Python
        else:
            batch_indices = \
                self._indices[self.ptr:self.ptr + self._batch_size]
            self.ptr += self._batch_size
            return self._examples_a[batch_indices], \
                   self._examples_b[batch_indices], self._labels[batch_indices]

    def __iter__(self):
        """Gets an iterator for this iterable

        Returns:
            self because the class is an iterator itself
        """
        return self


def _validate_labels_examples(examples, labels):
    if not isinstance(examples, np.ndarray):
        raise ValueError("examples should be an ndarray")

    if not isinstance(labels, np.ndarray):
        raise ValueError("labels should be ndarray")

    if not labels.size == examples.shape[0]:
        raise ValueError("number of labels != number of examples")


class SamplingBatcher(collections.abc.Iterator):
    """Batcher that samples according to a given distribution.

    It defaults to sampling from the data distribution.

    WARNING: this class is not deterministic. if you want deterministic
    behaviour, just freeze the numpy seed.
    """

    def __init__(
            self,
            examples: np.ndarray,
            labels: np.ndarray,
            batch_size: int,
            sample_distribution: Optional[Dict[int, float]] = None,
    ):
        """Create a new BalancedBatcher.

        Args:
            examples: np.ndarray containing examples
            labels: np.ndarray containing labels
            batch_size: int size of a single batch
            sample_distribution: optional distribution over label
                classes for sampling. This is normalized to sum to 1. Defines
                the target distribution that batches will be sampled with.
                Defaults to the data distribution.
        """
        _validate_labels_examples(examples, labels)
        self._examples = examples
        self._labels = labels
        self._label_classes = np.unique(labels)
        self._class_to_indices = {
            label: np.argwhere(labels == label).flatten()
            for label in self._label_classes
        }
        if sample_distribution is None:
            # Default to the data distribution
            sample_distribution = {
                label: float(indices.size)
                for label, indices in self._class_to_indices.items()
            }
        self._label_choices, self._label_probs = (
            self._get_label_choices_and_probs(sample_distribution))
        self._batch_size = batch_size

    def _get_label_choices_and_probs(self, sample_distribution):
        label_choices = []
        label_probs = []
        weight_sum = sum(sample_distribution.values())
        for label, weight in sample_distribution.items():
            if label not in self._labels:
                raise ValueError(
                    f"label {label} in sample distribution does not exist")
            if weight < 0.0:
                raise ValueError(
                    f"weight {weight} for label {label} is negative")
            label_choices.append(label)
            label_probs.append(weight / weight_sum)

        return np.array(label_choices), np.array(label_probs)

    def __next__(self):
        """Generates the next batch.

        Returns:
            (batch_of_examples, batch_of_labels) - a tuple of ndarrays
        """
        class_choices = np.random.choice(
            self._label_choices, size=self._batch_size, p=self._label_probs)

        batch_indices = []
        for class_choice in class_choices:
            indices = self._class_to_indices[class_choice]
            batch_indices.append(np.random.choice(indices))

        return self._examples[batch_indices], self._labels[batch_indices]

    def __iter__(self):
        """Gets an iterator for this iterable

        Returns:
            self because the class is an iterator itself
        """
        return self


if __name__ == '__main__':
    examples_1 = np.arange(240).reshape(12, 20)
    examples_2 = np.arange(240).reshape(12, 20)
    scores = np.arange(12).reshape(12)
    batcher = SamplingBatcherSTS(examples_1, examples_2, scores, batch_size=8)
    for e in range(10):
        print("epoch:{}".format(e))
        for batch in batcher:
            print(batch)
