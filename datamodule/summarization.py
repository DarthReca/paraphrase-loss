import logging
import multiprocessing
import pathlib
from base64 import b64encode
from collections import ChainMap, Counter
from hashlib import md5
from itertools import chain
from multiprocessing import Pool
from string import punctuation
from typing import Optional

import h5py
import numpy as np
import polars as pl
import torch
from datasets import Dataset, load_dataset
from hydra.utils import log
from more_itertools import chunked, flatten
from pytorch_lightning import LightningDataModule
from rich.progress import track
from scipy import sparse
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BartForConditionalGeneration,
    DataCollatorForSeq2Seq,
    DataCollatorForTokenClassification,
    default_data_collator,
)
class SummarizationDatamodule(LightningDataModule):
    def __init__(
        self,
        train_set_file: str,
        val_set_file: str,
        test_set_file: str,
        num_workers: int,
        batch_size: int,
        model_name: str,
        cache_dir: str = "cache",
    ) -> None:
        super().__init__()
        self.train_set_file = train_set_file
        self.val_set_file = val_set_file
        self.test_set_file = test_set_file

        self.train_set = None
        self.val_set = None
        self.test_set = None

        self.num_workers = num_workers
        self.batch_size = batch_size
        self.cache_dir = cache_dir

        self.mode = "conditional"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.model)

    def prepare_data(self) -> None:
        # If the files does not exists, raise an error
        if not pathlib.Path(self.train_set_file).exists():
            raise FileNotFoundError(f"File {self.train_set_file} does not exists")
        if not pathlib.Path(self.val_set_file).exists():
            raise FileNotFoundError(f"File {self.val_set_file} does not exists")
        if not pathlib.Path(self.test_set_file).exists():
            raise FileNotFoundError(f"File {self.test_set_file} does not exists")

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in ("fit", None):
            self.train_set = (
                load_dataset(
                    "parquet",
                    data_files=self.train_set_file,
                    split="train",
                    cache_dir=self.cache_dir,
                ).select_columns(
                    ["input_ids", "labels", "attention_mask", "paraphrases_ids"]
                )
                # .map(self._compute_per_token_weight)
                # .remove_columns(["paraphrases_ids"])
            )

        if stage in ("fit", "validate", None):
            self.val_set = load_dataset(
                "parquet",
                data_files=self.val_set_file,
                split="train",
                cache_dir=self.cache_dir,
            ).select_columns(["input_ids", "labels", "attention_mask"])
            logging.info(f"Val set size: {len(self.val_set)}")
        if stage in ("test", None):
            self.test_set = load_dataset(
                "parquet",
                data_files=self.test_set_file,
                split="train",
                cache_dir=self.cache_dir,
            ).select_columns(["input_ids", "labels", "attention_mask"])

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self._distribution_collator,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collator,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collator,
        )

    def _tokenize(self, example: dict):
        model_inputs = self.tokenizer(
            example["article"], truncation=True, max_length=1024
        )
        model_inputs["labels"] = self.tokenizer(
            example["highlights"], truncation=True, max_length=256
        )["input_ids"]

        if "paraphrases" in example:
            paraphrases_ids = self.tokenizer(
                list(chain(*example["paraphrases"])),
                truncation=True,
                max_length=256,
            )["input_ids"]
            paraphrases_ids = list(chunked(paraphrases_ids, 5))
            model_inputs["paraphrases_ids"] = paraphrases_ids
        return model_inputs

    def _distribution_collator(self, examples):
        if "distribution" not in examples[0]:
            return self.collator(examples)
        distributions = torch.empty(
            (len(examples), len(examples[0]["labels"]), self.model.config.vocab_size)
        )
        cleaned = []
        for i, example in enumerate(examples):
            distributions[i] = torch.sparse_coo_tensor(
                torch.tensor(
                    [example["distribution"]["row"], example["distribution"]["col"]]
                ),
                torch.tensor(example["distribution"]["data"]),
                (len(examples[0]["labels"]), self.model.config.vocab_size),
            ).to_dense()
            example.pop("distribution")
            cleaned.append(example)
        batch = self.collator(cleaned)
        batch["distribution"] = distributions
        return batch
      
    def _compute_per_token_weight(self, example):
        pid, lid = example["paraphrases_ids"], example["labels"]
        frequency = []
        for elements in zip(lid, *pid):
            count = Counter(elements)
            token_freq = sparse.coo_array(
                (
                    ([v / sum(count.values()) for v in count.values()]),
                    ([0] * len(count), list(count.keys())),
                ),
                shape=(1, self.model.config.vocab_size),
            )
            frequency.append(token_freq)
        frequency = sparse.vstack(frequency)
        example["distribution"] = {
            "row": frequency.row,
            "col": frequency.col,
            "data": frequency.data,
        }
        return example
