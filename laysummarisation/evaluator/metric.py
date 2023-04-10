from rouge import Rouge


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    rouge = Rouge()
    scores = dict(rouge.get_scores(predictions, labels, avg=True))
    return {
        "rouge1_f": scores["rouge-1"]["f"],
        "rouge2_f": scores["rouge-2"]["f"],
        "rougeL_f": scores["rouge-l"]["f"],
    }


if __name__ == "__main__":
    predictions = ["The cat was found under the bed.", "It was a small cat."]
    labels = ["The cat was under the bed.", "It was a small cat."]
    print(compute_metrics((predictions, labels)))
