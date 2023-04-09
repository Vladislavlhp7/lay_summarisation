from rouge import Rouge


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    rouge = Rouge()
    scores = rouge.get_scores(predictions, labels, avg=True)
    return {
        "rouge1_f": scores["rouge-1"]["f"],
        "rouge2_f": scores["rouge-2"]["f"],
        "rougeL_f": scores["rouge-l"]["f"],
    }
