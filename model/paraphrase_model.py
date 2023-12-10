from typing import List, Tuple

import comet_ml
import pytorch_lightning as pl
import torch
from more_itertools import flatten
from polars import DataFrame
from segmentation_models_pytorch.losses import LovaszLoss
from torch import nn
from torch.nn.functional import cross_entropy, kl_div, log_softmax
from transformers import Trainer, get_linear_schedule_with_warmup
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.models.bart.modeling_bart import shift_tokens_right

class ParaphraseTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        output = model(
            **{
                k: v
                for k, v in inputs.items()
                if k in ("input_ids", "attention_mask", "labels")
            }
        )

        logits = output.logits
        labels = inputs["labels"]
        distribution_loss = None
        if "distribution" in inputs:
            distribution_loss = kl_div(
                log_softmax(logits, dim=-1).reshape(-1, model.config.vocab_size),
                inputs["distribution"].reshape(-1, model.config.vocab_size),
                reduction="batchmean",
            )
        ce = cross_entropy(
            logits.reshape(-1, model.config.vocab_size), labels.reshape(-1)
        )
        loss = ce
        if distribution_loss is not None:
            loss = 0.9 * loss + 0.1 * distribution_loss  # + 0.1 * paraphrase_loss
            comet_ml.config.get_global_experiment().log_parameter(
                "use_distribution_loss", True
            )
        return (loss, output) if return_outputs else loss
