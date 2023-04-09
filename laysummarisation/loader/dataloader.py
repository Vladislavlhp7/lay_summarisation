import nltk
import pandas as pd
import sumy
from lexrank import LexRank
from lexrank.mappings.stopwords import STOPWORDS
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.lex_rank import LexRankSummarizer

nltk.download("punkt")


def load_data(path):
    """
    Load data from the given path.
    :param path: path to the data
    :return: data
    """
    return pd.read_json(path, lines=True)


def repare_data(data):
    """
    Repare the data.
    :param data: data to be repared
    :return: repared data
    """
    summarizer = LexRankSummarizer()
    summaries = []
    for d in data:
        parser = PlaintextParser.from_string(d, Tokenizer("english"))
        summary = summarizer(parser.document, 130)
        summaries.append(summary)
    return summaries


if __name__ == "__main__":
    data = load_data("data/task1_development/train/eLife_train.jsonl")
    summaries = repare_data(data.article.to_list()[:5])
    [print(s) for s in summaries[1]]
