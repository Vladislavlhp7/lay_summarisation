import glob
import logging
import os
from dataclasses import dataclass, field

from transformers import AutoTokenizer, HfArgumentParser


@dataclass
class Arguments:
    """
    Arguments
    """

    dir_aasd: str = field(
        default="",
        metadata={"help": "The input directory path which contains .conll files"},
    )
    prefix: str = field(
        default="elife_",
        metadata={"help": "The prefix for the output"},
    )
    output: str = field(
        default="",
        metadata={"help": "The output file path"},
    )


def main(conf: Arguments):
    # Setup logger
    logging.info(conf)
    # Load files
    conll_files = glob.glob(os.path.join(conf.dir_aasd, "*.conll"), recursive=True)
    # Sort the files
    conll_files = sorted(conll_files)
    logging.info(conll_files)

    return


def tokenize_function(data):
    """
    Tokenize the data.
    :param data: data to be tokenized
    :return: tokenized data
    """
    tokenizer = AutoTokenizer.from_pretrained("yikuan8/Clinical-Longformer")
    return tokenizer(data)


if __name__ == "__main__":
    parser = HfArgumentParser(Arguments)
    conf = parser.parse_args_into_dataclasses()[0]
    main(conf)
