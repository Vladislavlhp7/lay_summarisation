import datetime
from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class LFParserConfig:
    """
    Arguments for model training
    """

    ftrain: str = field(
        default=None,
        metadata={"help": "Train file (.jsonl)"},
    )
    fval: str = field(
        default=None,
        metadata={"help": "Validation file (.jsonl)"},
    )
    seed: Optional[int] = field(
        default=42,
        metadata={"help": "The random seed for model and training initialization"},
    )
    attention_window: Optional[int] = field(
        default=512,
        metadata={"help": "The attention window-size for Longformer"},
    )
    max_tokens: Optional[int] = field(
        default=4096,
        metadata={"help": "The max token length for the encoder"},
    )
    max_encode: Optional[int] = field(
        default=4096,
        metadata={"help": "The max token length for the encoder"},
    )
    model_name_or_path: Optional[str] = field(
        default="roberta-base",
        metadata={
            "help": "The model path or name."
            " If there exists a corresponding local path, the local model will be loaded."
            " Otherwise, the pre-trained model from Huggingface will be used."
        },
    )
    model_checkpoint: Optional[str] = field(
        default="yikuan8/Clinical-Longformer",
        metadata={"help": "The model checkpoint path or name."},
    )
    device: Optional[str] = field(
        default="cuda" if torch.cuda.is_available() else "cpu",
        metadata={"help": "The device to use for training"},
    )
    lr: Optional[float] = field(
        default=3e-5,
        metadata={"help": "The learning rate"},
    )
    batch_size: Optional[int] = field(
        default=4,
        metadata={"help": "The batch size for training"},
    )

    # Optimizers
    beta1: Optional[float] = field(
        default=0.9,
        metadata={"help": "Adam beta1"},
    )
    beta2: Optional[float] = field(
        default=0.998,
        metadata={"help": "Adam beta2"},
    )
    warmup_ratio: Optional[float] = field(
        default=0.1,
        metadata={"help": "The linear warmup ratio"},
    )
    clip: Optional[float] = field(
        default=5.0,
        metadata={"help": "The value of gradient clipping"},
    )
    batch_size: Optional[int] = field(
        default=4,
        metadata={"help": "The batch size for training"},
    )
    eval_batch_size: Optional[int] = field(
        default=4,
        metadata={"help": "The batch size for prediction and evaluation"},
    )
    epochs: Optional[int] = field(
        default=20,
        metadata={"help": "The number of epochs"},
    )
    terminate_epochs: Optional[int] = field(
        default=20,
        metadata={
            "help": "The number of terminate epochs. "
            "Training will be forced to terminate by this epoch number."
        },
    )
    evaluate_epochs: Optional[int] = field(
        default=2,
        metadata={
            "help": "The number of epochs for evaluation. "
            "Validation will be conducted every this epochs."
        },
    )
