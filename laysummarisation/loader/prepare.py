from transformers import AutoTokenizer


def tokenize_function(data):
    """
    Tokenize the data.
    :param data: data to be tokenized
    :return: tokenized data
    """
    tokenizer = AutoTokenizer.from_pretrained("yikuan8/Clinical-Longformer")
    return tokenizer(data)
