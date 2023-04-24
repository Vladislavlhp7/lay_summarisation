import os
from dataclasses import field, dataclass
from typing import Optional, List, Union

import pandas as pd
from tqdm import tqdm
from transformers import HfArgumentParser
import glob

from laysummarisation.utils import load_jsonl_pandas
from laysummarisation.utils import sentence_tokenize, preprocess


def get_binary_labels(df: pd.DataFrame, check_consistency=False):
    """
    Get the article sentences and their binary labels (1 if the sentence is in the pseudo summary, 0 otherwise).

    :param df: The dataframe with the pseudo summary and the whole article
    :param check_consistency: Check if the pseudo summary is in the article (i.e., if preprocessing works correctly)
    :return: The dataset with the binary labels
    """
    # Store all article's sentences and their binary labels
    # (1 if the sentence is in the pseudo summary, 0 otherwise)
    article_sents = []
    article_sents_binary = []
    for i, row in tqdm(df.iterrows(), desc="Binary sentence extraction"):
        id_ = row['id']
        pseudo_summary = row['article']
        article_whole = row['article_whole']

        # Prepare for sentence tokenization
        pseudo_summary = preprocess(pseudo_summary)
        article_whole = preprocess(article_whole)

        # Tokenize sentences
        pseudo_summary_sents = sentence_tokenize(pseudo_summary)
        article_whole_sents = sentence_tokenize(article_whole)

        # Report sentence sizes
        print(f'ID: {id_}', end=' | ')
        print(f'Pseudo summary: {len(pseudo_summary_sents)}', end=' | ')
        print(f'Whole article: {len(article_whole_sents)}')

        if check_consistency:
            for sent in pseudo_summary_sents:
                if sent in article_whole_sents:
                    article_sents_binary.append(1)
                else:
                    article_sents_binary.append(0)
                    print("Sentence not found in article:", sent)
            print("--------")

        # Find the sentences that are in the pseudo summary
        for sent in article_whole_sents:
            article_sents.append(sent)
            if sent in pseudo_summary_sents:
                article_sents_binary.append(1)
            else:
                article_sents_binary.append(0)
    return article_sents, article_sents_binary


@dataclass
class Arguments:
    """
    Arguments
    """

    data_dir: str = field(
        metadata={"help": "The input data directory."},
    )
    orig_dir: str = field(
        metadata={"help": "The original data directory."},
    )
    output_dir: str = field(
        metadata={"help": "The output data directory path"},
    )
    corpus: str = field(
        metadata={"help": "The corpus to use."},
    )
    narticles: Optional[int] = field(
        default=None,
        metadata={"help": "The number of entries to process. (0 for all)"},
    )
    all: Optional[bool] = field(
        default=False,
        metadata={"help": "Process all the articles."},
    )
    balance: Optional[bool] = field(
        default=False,
        metadata={"help": "Balance the dataset."},
    )
    seed: Optional[int] = field(
        default=42,
        metadata={"help": "The random seed."}
    )

def load_all(paths: List[str]):
    # Load all datasets provided
    df_from_each_file = (load_jsonl_pandas(f, nrows=conf.narticles) for f in paths)
    return pd.concat(df_from_each_file, ignore_index=True)
    

def main(conf: Arguments):
    print("Loading files...")

    # Load Pseudo-summarising dataset
    if conf.all:
        assert conf.corpus == "all"
        summary_files = glob.glob(os.path.join(conf.data_dir, "*.jsonl"))     
        article_files = glob.glob(os.path.join(conf.orig_dir, "*.jsonl"))     
        pseudo_summary_dataset = load_all(summary_files)
        article_dataset = load_all(article_files)
    else:
        pseudo_summary_dataset = load_jsonl_pandas(os.path.join(conf.data_dir, f"{conf.corpus}_train.jsonl"), nrows=conf.narticles)

        # Load original dataset
        article_dataset = load_jsonl_pandas(os.path.join(conf.orig_dir, f"{conf.corpus}_train.jsonl"), nrows=conf.narticles)

    # Merge the pseudo-summary (based on ROUGE) and the original article
    df = pseudo_summary_dataset.merge(article_dataset[['article', 'id']], on='id', how='inner', suffixes=('', '_whole'))

    # Get the dataset with the binary labels for each sentence in the article
    article_sents, article_sents_binary = get_binary_labels(df, check_consistency=False)

    # Export the sentences and their labels to a csv file
    df = pd.DataFrame({'sentence': article_sents, 'label': article_sents_binary})

    if conf.balance:
        df[df["label"] == 0].sample(n=df[df["label"] == 1], seed=conf.seed, replace=False)

    os.makedirs(os.path.dirname(conf.output_dir), exist_ok=True)
    df.to_csv(os.path.join(conf.output_dir,f"{conf.corpus}_train.csv"), index=False)
    return

if __name__ == "__main__":
    parser = HfArgumentParser(Arguments)
    conf = parser.parse_args_into_dataclasses()[0]
    if conf.narticles == 0:
        conf.narticles = None
    main(conf)
