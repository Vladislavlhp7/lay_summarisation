from dataclasses import dataclass, field
from typing import Optional

from transformers import HfArgumentParser
from pandarallel import pandarallel

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
    lex_sent: Optional[int] = field(
        default=25,
        metadata={"help": "The number of sentences to extract from the article."},
    )
    nrows: Optional[int] = field(
        default=None,
        metadata={"help": "The number of entries to process."},
    )
    workers: Optional[int] = field(
        default=1,
        metadata={"help": "The number of workers to use."},
    )


def main(conf: Arguments):
    pandarallel.initialize(conf.workers)

    # Load files
    print("Loading files...")
    data = load_jsonl_pandas(conf.fname, nrows=conf.nrows)

    # LexRank summarise the articles
    print("Summarising articles...")
    data["article"] = data["article"].parallel_apply(lexrank_summarize)

    # Save the data
    print("Saving data...")
    data.to_json(conf.output, orient="records", lines=True)
    return


if __name__ == "__main__":
    parser = HfArgumentParser(Arguments)
    conf = parser.parse_args_into_dataclasses()[0]
    if conf.nrows == 0:
        conf.nrows = None

    main(conf)
