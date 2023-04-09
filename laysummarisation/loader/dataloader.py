from typing import Sized

import torch
from torch.utils.data import DataLoader, Dataset


class LSDataLoader(DataLoader):
    """
    Data loader class
    """

    def __init__(self, *args, **kwargs):
        super(LSDataLoader, self).__init__(*args, **kwargs)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def __repr__(self):
        s = f"{self.__class__.__name__} (\n"
        s += f"\tdevice: {self.device}\n"
        s += ")"
        return s

    def _to_device(self, batch_sequence):
        device_batch = dict()
        # for field, batch in zip(self.dataset.fields, batch_sequence):
        #     device_batch[field.name] = field.batchfy(device=self.device, batch=batch)

        return device_batch

    def __iter__(self):
        for raw_batch in super(LSDataLoader, self).__iter__():
            batch, indices = raw_batch
            yield {
                "data_indices": indices,
                # "origin": [self.dataset.corpus[i] for i in indices],
                "batch": self._to_device(batch),
            }


class LSDataset(Dataset):
    """
    Dataset class
    """

    def __init__(self, corpus, **kwargs):
        super(LSDataset, self).__init__()

        self.corpus = corpus

    def __getitem__(self, index):
        # Returns numericalized sample and data index
        return self.corpus[index]

    def __len__(self):
        return len(self.corpus)

    @staticmethod
    def collate_fn(batch):
        # Returns (i) batched fields and (ii) data indices
        return [field for field in zip(*[b[0] for b in batch])], [
            b[1] for b in batch
        ]  # Indices


def construct_loader(
    corpus: Sized,
    batch_size: int,
    shuffle: bool = False,
    **kwargs,
) -> LSDataLoader:
    """
    Build data loader from the corpus and numericalizer

    Parameters
    ----------
    corpus : Sized
        The corpus instance
    numericalizer : AMNumericalizer
        The numericalizer instance
    batch_size : int
        The bach size
    shuffle : bool
        Whether to shuffle the batch samples

    Returns
    ----------
    data_loader : AMDataLoader
        The data loader instance
    """
    dataset = LSDataset(corpus=corpus, **kwargs)
    return LSDataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=dataset.collate_fn,
        sampler=None,
    )
