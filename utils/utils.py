import pandas as pd
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.lex_rank import LexRankSummarizer


def read_jsonl_data(path):
    """
    Load data from the given path.
    :param path: path to the data
    :return: data
    """
    return pd.read_json(path, lines=True)


def lexrank_data(data, max_length=130):
    """
    Repare the data.
    :param data: data to be repared
    :param max_len: max length of the summary
    :return: repared data
    """
    summarizer = LexRankSummarizer()
    summaries = []
    for d in data:
        parser = PlaintextParser.from_string(d, Tokenizer("english"))
        summary = summarizer(parser.document, max_length)
        summaries.append(summary)
    return summaries


if __name__ == "__main__":
    data = read_jsonl_data("data/task1_development/train/eLife_train.jsonl")
    summaries = lexrank_data(data.article.to_list()[:5])
    [print(s) for s in summaries[1]]
