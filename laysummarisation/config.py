from dataclasses import dataclass, field

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
    save_dir: str = field(
        metadata={"help": "The directory to save the model"},
    )
    seed: int = field(
        default=42,
        metadata={"help": "The random seed for model and training initialization"},
    )
    model_checkpoint: str = field(
        default="yikuan8/Clinical-Longformer",
        metadata={"help": "The model checkpoint path or name."},
    )
    device: str = field(
        default="cuda" if torch.cuda.is_available() else "cpu",
        metadata={"help": "The device to use for training"},
    )
    attention_window: int = field(
        default=512,
        metadata={"help": "The attention window-size for Longformer"},
    )
    max_encode: int = field(
        default=512,
        metadata={"help": "The max token length for the encoder"},
    )
    min_encode: int = field(
        default=128,
        metadata={"help": "The min token length for the encoder"},
    )
    max_decode: int = field(
        default=1024,
        metadata={"help": "The max token length for the decoder"},
    )
    nbeams: int = field(
        default=4,
        metadata={"help": "The number of beams for beam search"},
    )
    length_penalty: float = field(
        default=2.0,
        metadata={"help": "The length penalty for beam search"},
    )
    early_stopping: bool = field(
        default=True,
        metadata={"help": "Whether to use early stopping for beam search"},
    )
    lr: float = field(
        default=3e-5,
        metadata={"help": "The learning rate"},
    )
    batch_size: int = field(
        default=1,
        metadata={"help": "The batch size for training, (=1)"},
    )
    epochs: int = field(
        default=20,
        metadata={"help": "The number of epochs"},
    )
    weight_decay: float = field(
        default=0.01,
        metadata={"help": "The weight decay for training"},
    )
    save_steps: int = field(
        default=1000,
        metadata={"help": "The number of steps to save the model"},
    )
    eval_steps: int = field(
        default=1000,
        metadata={"help": "The number of steps to evaluate the model"},
    )
    metric: str = field(
        default="rouge2_f",
        metadata={"help": "The metric to use for model selection"},
    )
    logging_steps: int = field(
        default=250,
        metadata={"help": "The number of steps to log the training"},
    )
    warmup_steps: int = field(
        default=1500,
        metadata={"help": "The number of steps for the warmup"},
    )
    gradient_accum_steps: int = field(
        default=4,
        metadata={"help": "The number of steps for gradient accumulation"},
    )
