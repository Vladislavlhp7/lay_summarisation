from dataclasses import dataclass, field
from typing import Optional


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
    corpus: str = field(
        metadata={"help": "The corpus name"},
    )
    save_dir: str = field(
        metadata={"help": "The directory to save the model"},
    )
    seed: int = field(
        default=42,
        metadata={
            "help": "The random seed for model and training initialization"
        },
    )
    model: str = field(
        default="yikuan8/Clinical-Longformer",
        metadata={"help": "The model checkpoint path or name."},
    )
    checkpoint: Optional[str] = field(
        default=None,
        metadata={"help": "The model checkpoint path or name."},
    )
    device: str = field(
        default="cpu",
        metadata={"help": "The device to use for training"},
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
    attention_window: int = field(
        default=512,
        metadata={"help": "The attention window-size for Longformer"},
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
        default=5e-5,
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
    metric: str = field(
        default="rouge2_f",
        metadata={"help": "The metric to use for model selection"},
    )
    warmup_steps: int = field(
        default=1500,
        metadata={"help": "The number of steps for the warmup"},
    )
    gradient_accum_steps: int = field(
        default=4,
        metadata={"help": "The number of steps for gradient accumulation"},
    )
    save_strategy: str = field(
        default="steps",
        metadata={"help": "The strategy to save the model"},
    )
    logging_strategy: str = field(
        default="steps",
        metadata={"help": "The strategy to log the training"},
    )
    eval_strategy: str = field(
        default="steps",
        metadata={"help": "The strategy to evaluate the model"},
    )
    save_steps: int = field(
        default=1000,
        metadata={"help": "The number of steps to save the model"},
    )
    eval_steps: int = field(
        default=1000,
        metadata={"help": "The number of steps to evaluate the model"},
    )
    logging_steps: int = field(
        default=250,
        metadata={"help": "The number of steps to log the training"},
    )
    load_best_model_at_end: bool = field(
        default=True,
        metadata={"help": "Whether to load the best model at the end"},
    )
    fp16: bool = field(
        default=False,
        metadata={"help": "Whether to use fp16 for training"},
    )
    fp16_full_eval: bool = field(
        default=False,
        metadata={"help": "Whether to use fp16 for evaluation"},
    )
