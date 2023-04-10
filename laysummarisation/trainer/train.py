from transformers import AutoModelForMaskedLM

from laysummarisation.trainer.base_trainer import BaseTrainer


class LaySumTrainer(BaseTrainer):
    def __init__(self, config) -> None:
        super().__init__(config)


# def train(dataset):
#     model = AutoModelForMaskedLM.from_pretrained("yikuan8/Clinical-Longformer")
