import glob
import os
from dataclasses import dataclass, field
from typing import Optional

from pandarallel import pandarallel
from transformers import HfArgumentParser

from laysummarisation.utils import (lexrank_summarize, load_jsonl_pandas,
                                    load_multiple_df, set_seed)


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
    nrows: Optional[int] = field(
        default=None,
        metadata={"help": "The number of entries to process. (0 for all)"},
    )
    nsent: Optional[int] = field(
        default=25,
        metadata={
            "help": "The number of sentences to extract from the article."
        },
    )
    all: Optional[bool] = field(
        default=False,
        metadata={"help": "Process all the articles."},
    )
    seed: Optional[int] = field(
        default=None, metadata={"help": "The random seed."}
    )
    workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of workers to use."},
    )


def main(conf: Arguments):
    if conf.workers is None:
        conf.workers = 1
    if conf.nrows == 0:
        conf.nrows = None
    if conf.workers is not None:
        pandarallel.initialize(nb_workers=conf.workers, progress_bar=True)

    # Load files
    print("Loading files...")

    if conf.seed is not None:
        set_seed(conf.seed)

    # Load dataset
    if conf.all:
        assert conf.corpus == "all"
        all_files = glob.glob(os.path.join(conf.data_dir, "*.jsonl"))
        data = load_multiple_df(all_files)
    else:
        assert conf.corpus != "all"
        data = load_jsonl_pandas(
            os.path.join(conf.data_dir, f"{conf.corpus}_train.jsonl"),
            nrows=conf.nrows,
        )

    # LexRank summarise the articles
    print("Summarising articles...")
    if conf.workers is None:
        data["article"] = data["article"].apply(lexrank_summarize)
    else:
        data["article"] = data["article"].parallel_apply(lexrank_summarize)

    # Save the data
    print("Saving data...")
    data.to_json(
        os.path.join(conf.output_dir, f"{conf.corpus}_train.jsonl"),
        orient="records",
        lines=True,
    )
    return


if __name__ == "__main__":
    parser = HfArgumentParser(Arguments)
    conf = parser.parse_args_into_dataclasses()[0]

    main(conf)
