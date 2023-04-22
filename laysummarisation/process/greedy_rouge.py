from dataclasses import dataclass, field
from typing import List, Optional

import nltk
import numpy as np
import pandas as pd
from pandarallel import pandarallel
from transformers import HfArgumentParser

from laysummarisation.utils import load_jsonl_pandas, preprocess, sentence_tokenize
from rouge import Rouge


@dataclass
class Arguments:
    """
    Arguments
    """

    fname: str = field(
        metadata={"help": "The input jsonl file path."},
    )
    output: str = field(
        metadata={"help": "The output mrp file path"},
    )
    nsent: Optional[int] = field(
        default=10,
        metadata={"help": "The number of sentences to extract from the article."},
    )
    nrows: Optional[int] = field(
        default=None,
        metadata={"help": "The number of entries to process. (0 for all)"},
    )
    mode: Optional[str] = field(
        default="split",
        metadata={"help": "The mode to run the script in."},
    )
    workers: Optional[int] = field(
        default=1,
        metadata={"help": "The number of workers to use."},
    )


def rouge_maximise(corpus, sentence):
    """
    Compute the rouge f1 metric for each sentence in the corpus and return the array
    of scores.
    """
    scores = []
    rouge = Rouge()
    for c in corpus:
        scores.append(dict(rouge.get_scores(c, sentence, avg=True))["rouge-l"]["f"])
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


def process_entry(entry: pd.Series, conf: Arguments):
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
    rl2_sort = [sorted(enumerate(r), reverse=True, key=lambda x: x[1])[0] for r in rl2]

    # Get the top n sentences
    rl_i = sorted([i for i, _ in rl_sort[: conf.nsent]])
    rl2_i = sorted([i for i, _ in rl2_sort[: conf.nsent]])

    # Ensure that the sentences from both lists are unique
    merged_list = set(rl_i + [x for x in rl2_i if x not in rl_i])

    return " ".join([art_sent[x] for x in sorted(merged_list)])


def main(conf: Arguments):
    pandarallel.initialize(conf.workers)

    # Load files
    print("Loading files...")
    data = load_jsonl_pandas(conf.fname, nrows=conf.nrows)

    # Set the mode, either 'split' for split abstract and append it
    # or 'include' to include the abstract when summarising.
    if conf.mode == "split":
        data["article"] = data.parallel_apply(
            lambda x: remove_abstract(x.article), axis=1
        )
    elif conf.mode == "include":
        pass
    else:
        raise ValueError("Invalid mode")

    data["article"] = data.parallel_apply(lambda x: process_entry(x, conf), axis=1)

    # Save the data
    print("Saving data...")
    data.to_json(conf.output, orient="records", lines=True)
    return


if __name__ == "__main__":
    nltk.download("punkt")
    parser = HfArgumentParser(Arguments)
    conf = parser.parse_args_into_dataclasses()[0]
    if conf.nrows == 0:
        conf.nrows = None

    main(conf)
