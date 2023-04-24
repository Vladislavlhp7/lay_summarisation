from time import time

import evaluate
import pandas as pd
from rouge import Rouge
from rouge_score import rouge_scorer

from laysummarisation.utils import sentence_tokenize


def main():
    df = pd.read_json("data/orig/train/eLife_train.jsonl", lines=True, nrows=1)

    rouge = Rouge()
    rouge1 = rouge_scorer.RougeScorer(["rougeL"])
    rouge2 = evaluate.load("rouge")

    art_list = df["article"].tolist()
    lay_list = df["lay_summary"].tolist()

    art_tok = sentence_tokenize(art_list[0])

    # start = time()
    # for i in range(10):
    #     print(i)
    #     res = rouge.get_scores(art_list[0], art_list[0], avg=True)
    # print("Current Rouge:", time() - start)
    # print(res)
    # print("\n\n")

    start = time()
    for i in range(10):
        print(i)
        for c in art_tok:
            res = rouge1.score(c, lay_list[0])

    print("Rouge1:", time() - start)
    print(res)
    print("\n\n")

    start = time()
    for i in range(10):
        print(i)
        res = rouge2.compute(
            predictions=art_tok,
            references=lay_list * len(art_tok),
            rouge_types=["rougeL"],
            num_process=6,
        )
    print("Rouge2:", time() - start)
    print(res)
    print("\n\n")


if __name__ == "__main__":
    main()
