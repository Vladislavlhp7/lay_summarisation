from dataclasses import dataclass, field
from typing import Optional, List

from transformers import HfArgumentParser
import pandas as pd

from laysummarisation.utils import lexrank_summarize, load_jsonl_pandas


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
    rouge_sent: Optional[int] = field(
        default=25,
        metadata={"help": "The number of sentences to extract from the article."},
    )
    nrows: Optional[int] = field(
        default=None,
        metadata={"help": "The number of entries to process."},
    )


def rouge_maximise(corpus, sentence):
    """
    Compute the rouge f1 metric for each sentence in the corpus and return the array
    of scores.
    """
    scores = []
    rouge = Rouge()
    for i, c in enumerate(corpus):
        scores.append(rouge.get_scores(c, sentence, avg=True)["rouge-l"]["f"])
    return np.array(scores)


def rouge_lay_sent(corpus: List[str], lay: List[str]):
    """
    Compute the rouge metric for each sentence in the lay, for each sentence in the summary
    """
    scores = []
    for l in lay:
        scores.append(rouge_maximise(corpus, l))
    return scores


def main(conf: Arguments):
    # raise NotImplementedError("Rouge maximisation preprocessing not implemented yet")
    # Load files
    print("Loading files...")
    data = load_jsonl_pandas(conf.fname, nrows=conf.nrows)

    # LexRank summarise the articles
    print("Summarising articles...")
    data["article"] = data["article"].apply(rouge_summarize, args=(conf.rouge_sent,))

    with open("tmp.txt", "r") as f:
        text = f.read()

    data = pd.read_json("data/orig/train/eLife_train.jsonl", lines=True, nrows=1)
    point = data.iloc[0]
    r = Rouge()

    rl = r.get_scores(text, point.lay_summary, avg=True)
    print(rl)
    # Save the data
    print("Saving data...")
    data.to_json(conf.output, orient="records", lines=True)
    return


if __name__ == "__main__":
    parser = HfArgumentParser(Arguments)
    conf = parser.parse_args_into_dataclasses()[0]
    main(conf)
