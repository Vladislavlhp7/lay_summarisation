import glob
import os
from dataclasses import dataclass, field
from typing import List, Optional, Union

import nltk
import numpy as np
import pandas as pd
from pandarallel import pandarallel
from rouge_score import rouge_scorer
from transformers import HfArgumentParser

from laysummarisation.utils import (load_jsonl_pandas, load_multiple_df,
                                    preprocess, sentence_tokenize, set_seed)


@dataclass
class Arguments:
    """
    Arguments
    """

    data_dir: str = field(
        metadata={"help": "The input data directory."},
    )
    output_dir: str = field(
        metadata={"help": "The output data directory path"},
    )
    corpus: str = field(
        metadata={"help": "The corpus to use."},
    )
    nsent: int = field(
        default=10,
        metadata={
            "help": "The number of sentences to extract from the article."
        },
    )
    seed: Optional[int] = field(
        default=None, metadata={"help": "The random seed."}
    )
    workers: Union[int, None] = field(
        default=None,
        metadata={"help": "The number of workers to use."},
    )
    nrows: Optional[int] = field(
        default=None,
        metadata={"help": "The number of entries to process. (0 for all)"},
    )
    all: Optional[bool] = field(
        default=False,
        metadata={"help": "Process all the articles."},
    )


def rouge_maximise(corpus, sentence):
    """
    Compute the rouge f1 metric for each sentence in the corpus and return the array
    of scores.
    """
    scores = []
    rouge = rouge_scorer.RougeScorer(["rougeL"])
    for c in corpus:
        try:
            rouge_score = rouge.score(c, sentence)["rougeL"].fmeasure
            scores.append(rouge_score)
        except ValueError:
            scores.append(0.0)
    return np.array(scores)


def rouge_lay_sent(corpus: List[str], lay: List[str]):
    """
    Compute the rouge metric for each sentence in the lay, for each sentence in the summary
    """
    scores = []
    for sent in lay:
        scores.append(rouge_maximise(corpus, sent))
    return scores


def remove_abstract(article: str):
    """
    Remove the abstract from an article (anything before first newline).
    """
    return "\n".join(article.split("\n")[1:])


def process_entry(entry: pd.Series, nsent: int):
    """
    Process a single entry from the dataset.
    """
    entry.article = preprocess(entry.article)
    entry.lay_summary = preprocess(entry.lay_summary)
    art_sent = sentence_tokenize(entry.article)
    lay_sent = sentence_tokenize(entry.lay_summary)

    rl = rouge_maximise(art_sent, entry.lay_summary)
    rl2 = rouge_lay_sent(art_sent, lay_sent)

    # Sort the sentences by rouge score
    rl_sort = sorted(enumerate(rl), reverse=True, key=lambda x: x[1])
    rl2_sort = [
        sorted(enumerate(r), reverse=True, key=lambda x: x[1])[0] for r in rl2
    ]

    # Get the top n sentences
    rl_i = sorted([i for i, _ in rl_sort[:nsent]])
    rl2_i = sorted([i for i, _ in rl2_sort[:nsent]])

    # Ensure that the sentences from both lists are unique
    merged_list = set(rl_i + [x for x in rl2_i if x not in rl_i])

    return " ".join([art_sent[x] for x in sorted(merged_list)])


def main(conf: Arguments):
    if conf.workers is not None:
        pandarallel.initialize(nb_workers=conf.workers, progress_bar=True)
    if conf.seed is not None:
        set_seed(conf.seed)

    # Load files
    print("Loading files...")

    # Load dataset
    if conf.all:
        assert conf.corpus == "all"
        all_files = glob.glob(os.path.join(conf.data_dir, "*.jsonl"))
        data = load_multiple_df(all_files)
    else:
        assert conf.corpus != "all"
        data = load_jsonl_pandas(
            os.path.join(conf.data_dir, f"{conf.corpus}_val.jsonl"),
            nrows=conf.nrows,
        )

    # Set the mode, either 'split' for split abstract and append it
    # or 'include' to include the abstract when summarising.
    # if conf.mode == "split":
    #     data["article"] = data.parallel_apply(
    #         lambda x: remove_abstract(x.article), axis=1
    #     )
    # elif conf.mode == "include":
    #     pass
    # else:
    #     raise ValueError("Invalid mode")

    if conf.workers is None:
        data["article"] = data.apply(
            lambda x: process_entry(x, conf.nsent), axis=1
        )
    else:
        data["article"] = data.parallel_apply(
            lambda x: process_entry(x, conf.nsent), axis=1
        )

    # Save the data
    print("Saving data...")
    data.to_json(
        os.path.join(conf.output_dir, f"{conf.corpus}_val.jsonl"),
        orient="records",
        lines=True,
    )
    return


if __name__ == "__main__":
    nltk.download("punkt")
    parser = HfArgumentParser(Arguments)
    conf = parser.parse_args_into_dataclasses()[0]

    main(conf)
