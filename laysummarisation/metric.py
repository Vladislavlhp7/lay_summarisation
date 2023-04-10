from rouge import Rouge


def compute_metrics(eval_pred) -> dict:
    """
    Compute the ROUGE scores for a given set of predictions and labels.

    Args:
    eval_pred (tuple): A tuple containing two lists of predictions and labels.

    Returns:
    dict: A dictionary containing the ROUGE scores for ROUGE-1, ROUGE-2, and ROUGE-L.
    """

    # Unpack the tuple into separate lists of predictions and labels
    predictions, labels = eval_pred

    # Compute the ROUGE scores for the predictions and labels using the Rouge package
    rouge = Rouge()
    scores = dict(rouge.get_scores(predictions, labels, avg=True))

    # Return the ROUGE scores as a dictionary with keys for each metric
    return {
        "rouge1_f": scores["rouge-1"]["f"],
        "rouge2_f": scores["rouge-2"]["f"],
        "rougeL_f": scores["rouge-l"]["f"],
    }


if __name__ == "__main__":
    predictions = ["The cat was found under the bed.", "It was a small cat."]
    labels = ["The cat was under the bed.", "It was a small cat."]
    print(compute_metrics((predictions, labels)))
