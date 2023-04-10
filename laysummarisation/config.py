from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class LFParserConfig:
    """
    Arguments for model training
    """

    ftrain: str = field(
        metadata={"help": "Train file (.jsonl)"},
    )
    fvalid: str = field(
        metadata={"help": "Validation file (.jsonl)"},
    )
    seed: Optional[int] = field(
        default=42,
        metadata={"help": "The random seed for model and training initialization"},
    )
    model_checkpoint: Optional[str] = field(
        default="yikuan8/Clinical-Longformer",
        metadata={"help": "The model checkpoint path or name."},
    )
    device: Optional[str] = field(
        default="cuda" if torch.cuda.is_available() else "cpu",
        metadata={"help": "The device to use for training"},
    )
    # attention_window: Optional[int] = field(
    #     default=512,
    #     metadata={"help": "The attention window-size for Longformer"},
    # )
    max_encode: Optional[int] = field(
        default=4096,
        metadata={"help": "The max token length for the encoder"},
    )
    max_decode: Optional[int] = field(
        default=1024,
        metadata={"help": "The max token length for the decoder"},
    )
    lr: Optional[float] = field(
        default=3e-5,
        metadata={"help": "The learning rate"},
    )
    batch_size: Optional[int] = field(
        default=1,
        metadata={"help": "The batch size for training, (=1)"},
    )
    epochs: Optional[int] = field(
        default=20,
        metadata={"help": "The number of epochs"},
    )
    save_steps: Optional[int] = field(
        default=1000,
        metadata={"help": "The number of steps to save the model"},
    )
    eval_steps: Optional[int] = field(
        default=1000,
        metadata={"help": "The number of steps to evaluate the model"},
    )
    weight_decay: Optional[float] = field(
        default=0.01,
        metadata={"help": "The weight decay for training"},
    )
    metric: Optional[str] = field(
        default="rouge2_f",
        metadata={"help": "The metric to use for model selection"},
    )
