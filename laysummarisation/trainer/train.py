from transformers import AutoModelForMaskedLM


def train(dataset):
    model = AutoModelForMaskedLM.from_pretrained("yikuan8/Clinical-Longformer")
