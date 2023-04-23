import os
from dataclasses import field, dataclass
from typing import Optional

import pandas as pd
from tqdm import tqdm
from transformers import HfArgumentParser

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

    summary_fname: str = field(
        metadata={"help": "The input jsonl file path for the pseudo summary."},
    )
    article_fname: str = field(
        metadata={"help": "The input jsonl file path for the original article."},
    )
    output: str = field(
        metadata={"help": "The output mrp file path"},
    )
    narticles: Optional[int] = field(
        default=None,
        metadata={"help": "The number of entries to process. (0 for all)"},
    )


def main(conf: Arguments):
    print("Loading files...")

    # Load Pseudo-summarising dataset
    pseudo_summary_dataset = load_jsonl_pandas(conf.summary_fname, nrows=conf.narticles)

    # Load original dataset
    article_dataset = load_jsonl_pandas(conf.article_fname, nrows=conf.narticles)

    # Merge the pseudo-summary (based on ROUGE) and the original article
    df = pseudo_summary_dataset.merge(article_dataset[['article', 'id']], on='id', how='inner', suffixes=('', '_whole'))

    # Get the dataset with the binary labels for each sentence in the article
    article_sents, article_sents_binary = get_binary_labels(df, check_consistency=False)

    # Export the sentences and their labels to a csv file
    df = pd.DataFrame({'sentence': article_sents, 'label': article_sents_binary})
    os.makedirs(os.path.dirname(conf.output), exist_ok=True)
    df.to_csv(conf.output, index=False)
    return


if __name__ == "__main__":
    parser = HfArgumentParser(Arguments)
    conf = parser.parse_args_into_dataclasses()[0]
    main(conf)
