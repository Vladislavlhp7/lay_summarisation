import os
from dataclasses import dataclass, field

import torch
from transformers import BertForSequenceClassification, BertTokenizerFast
from transformers import HfArgumentParser

from laysummarisation.utils import get_binary_sentence_dataset, set_seed


@dataclass
class Arguments:
    """
    Arguments
    """

    fname: str = field(
        metadata={"help": "The input binary label dataset file path"},
    )
    seed: int = field(
        default=42,
        metadata={"help": "Random seed."},
    )


def main(conf: Arguments):
    print("CUDA available:" + str(torch.cuda.is_available()))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.empty_cache()

    # Load files
    print("Loading files...")

    set_seed(conf.seed)

    os.environ["WANDB_PROJECT"] = "laysummarisation"
    os.environ["WANDB_LOG_MODEL"] = "true"

    model = BertForSequenceClassification.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    model.to(device)
    tokenizer = BertTokenizerFast.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

    # Get the dataset with the binary labels for each sentence in the article
    dataset = get_binary_sentence_dataset(conf.fname)

    # Split the dataset into stratified train, validation and test (80%, 10%, 10%)
    dataset_split = dataset.train_test_split(test_size=0.2, seed=42, shuffle=True, stratify_by_column='label')
    train_dataset = dataset_split['train']
    test_dataset = dataset_split['test'].train_test_split(test_size=0.5, seed=42, stratify_by_column='label')['train']
    val_dataset = dataset_split['test'].train_test_split(test_size=0.5, seed=42, stratify_by_column='label')['test']


if __name__ == "__main__":
    parser = HfArgumentParser(Arguments)
    conf = parser.parse_args_into_dataclasses()[0]
    main(conf)
