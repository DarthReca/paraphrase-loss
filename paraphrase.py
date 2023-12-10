import logging
import os
from datetime import datetime
from typing import Optional

import comet_ml
import evaluate
import hydra
import pytorch_lightning as pl
import torch
from datamodule import SummarizationDatamodule
from datasets import disable_caching
from model import ParaphraseTrainer
from omegaconf import DictConfig
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CometLogger
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.integrations import CometCallback


class CometCallBackWithName(CometCallback):
    def __init__(self, experiment_name: str):
        self._initialized = False
        self._log_assets = False
        self.experiment_name = experiment_name

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        super().on_train_begin(args, state, control, model, **kwargs)
        comet_ml.config.get_global_experiment().set_name(self.experiment_name)


@hydra.main(config_path="configs", config_name="paraphrase", version_base=None)
def main(args: DictConfig):
    logging.basicConfig(level=logging.INFO)
    set_seed(42)
    if args.log_comet:
        with open(".comet", "r") as f:
            os.environ["COMET_API_KEY"] = f.readline().strip()
    dm = SummarizationDatamodule(**args.dataset, cache_dir="cache")
    dm.setup()
    comet_ml.init(project_name="", workspace="")
    name = f"{args.model.model_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"

    model = AutoModelForSeq2SeqLM.from_pretrained(args.model.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model.model_name)

    training_arguments = TrainingArguments(
        output_dir=f"checkpoints/{name}", remove_unused_columns=False, **args.trainer
    )

    rouge = evaluate.load("rouge")

    def compute_metrics(pred):
        labels_ids = pred.label_ids
        pred_ids = pred.predictions[0]

        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        labels_ids[labels_ids == -100] = tokenizer.pad_token_id
        label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

        rouge_output = rouge.compute(
            predictions=pred_str,
            references=label_str,
            rouge_types=["rouge1", "rouge2", "rougeL", "rougeLsum"],
        )

        return {
            "R1": round(rouge_output["rouge1"], 4),
            "R2": round(rouge_output["rouge2"], 4),
            "RL": round(rouge_output["rougeL"], 4),
            "RLsum": round(rouge_output["rougeLsum"], 4),
        }

    def preprocess_logits_for_metrics(logits, labels):
        """
        Original Trainer may have a memory leak.
        This is a workaround to avoid storing too many tensors that are not needed.
        """
        pred_ids = torch.argmax(logits[0], dim=-1)
        return pred_ids, labels

    trainer = ParaphraseTrainer(
        model=model,
        args=training_arguments,
        train_dataset=dm.train_set,
        eval_dataset=dm.val_set.select(range(100 * dm.batch_size)),
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=[CometCallBackWithName(name)] if args.log_comet else None,
        data_collator=dm._distribution_collator,
    )
    trainer.train()


if __name__ == "__main__":
    main()
