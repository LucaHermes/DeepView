import os
import shutil

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import tensorflow_text as text
# from official.nlp import optimization  # to create AdamW optimizer
from typing import List, Optional
import matplotlib.pyplot as plt
from transformers import TFBertModel
from keras import utils
import pandas as pd
import datasets
from dataclasses import dataclass, field
from datasets import load_dataset
import random
import evaluate
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
import logging
logger = logging.getLogger(__name__)
@dataclass
class Task:
    id: int
    name: str
    type: str
    num_labels: int


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    do_regression: bool = field(
        default=None,
        metadata={
            "help": "Whether to do regression instead of classification. If None, will be inferred from the dataset."
        },
    )
    text_column_names: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The name of the text column in the input dataset or a CSV/JSON file. "
                'If not specified, will use the "sentence" column for single/multi-label classifcation task.'
            )
        },
    )
    text_column_delimiter: Optional[str] = field(
        default=" ", metadata={"help": "THe delimiter to use to join text columns into a single sentence."}
    )
    train_split_name: Optional[str] = field(
        default=None,
        metadata={
            "help": 'The name of the train split in the input dataset. If not specified, will use the "train" split when do_train is enabled'
        },
    )
    validation_split_name: Optional[str] = field(
        default=None,
        metadata={
            "help": 'The name of the validation split in the input dataset. If not specified, will use the "validation" split when do_eval is enabled'
        },
    )
    test_split_name: Optional[str] = field(
        default=None,
        metadata={
            "help": 'The name of the test split in the input dataset. If not specified, will use the "test" split when do_predict is enabled'
        },
    )
    remove_splits: Optional[str] = field(
        default=None,
        metadata={"help": "The splits to remove from the dataset. Multiple splits should be separated by commas."},
    )
    remove_columns: Optional[str] = field(
        default=None,
        metadata={"help": "The columns to remove from the dataset. Multiple columns should be separated by commas."},
    )
    label_column_name: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The name of the label column in the input dataset or a CSV/JSON file. "
                'If not specified, will use the "label" column for single/multi-label classifcation task'
            )
        },
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )
    shuffle_train_dataset: bool = field(
        default=False, metadata={"help": "Whether to shuffle the train dataset or not."}
    )
    shuffle_seed: int = field(
        default=42, metadata={"help": "Random seed that will be used to shuffle the train dataset."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    metric_name: Optional[str] = field(default=None, metadata={"help": "The metric to use for evaluation."})
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})

    # def __post_init__(self):
    #     if self.dataset_name is None:
    #         if self.train_file is None or self.validation_file is None:
    #             raise ValueError(" training/validation file or a dataset name.")
    #
    #         train_extension = self.train_file.split(".")[-1]
    #         assert train_extension in ["csv", "json"], "`train_file` should be a csv or a json file."
    #         validation_extension = self.validation_file.split(".")[-1]
    #         assert (
    #             validation_extension == train_extension
    #         ), "`validation_file` should have the same extension (csv or json) as `train_file`."

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    use_auth_token: bool = field(
        default=None,
        metadata={
            "help": "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token` instead."
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option"
                "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
                "execute code present on the Hub on your local machine."
            )
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )



def tokenize_token_classification_dataset(
    raw_datasets,
    tokenizer,
    task_id,
    label_list,
    text_column_name,
    label_column_name,
    data_args,
    training_args,
):

    label_to_id = {i: i for i in range(len(label_list))}

    # Map that sends B-Xxx label to its I-Xxx counterpart
    b_to_i_label = []
    for idx, label in enumerate(label_list):
        if label.startswith("B-") and label.replace("B-", "I-") in label_list:
            b_to_i_label.append(label_list.index(label.replace("B-", "I-")))
        else:
            b_to_i_label.append(idx)

    # Padding strategy
    padding = "max_length" if data_args.pad_to_max_length else False

    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples[text_column_name],
            padding=padding,
            truncation=True,
            max_length=data_args.max_seq_length,
            # We use this argument because the texts in our dataset are lists of words (with a label for each word).
            is_split_into_words=True,
        )
        labels = []
        for i, label in enumerate(examples[label_column_name]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label_to_id[label[word_idx]])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                # else:
                #     if data_args.label_all_tokens:
                #         label_ids.append(b_to_i_label[label_to_id[label[word_idx]]])
                #     else:
                #         label_ids.append(-100)
                previous_word_idx = word_idx

            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        tokenized_inputs["task_ids"] = [task_id] * len(tokenized_inputs["labels"])
        return tokenized_inputs

    with training_args.main_process_first(desc="dataset map pre-processing"):
        col_to_remove = ["chunk_tags", "id", "ner_tags", "pos_tags", "tokens"]

        tokenized_datasets = raw_datasets.map(
            tokenize_and_align_labels,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            remove_columns=col_to_remove,
        )

    return tokenized_datasets


def load_token_classification_dataset(task_id, tokenizer, data_args, training_args):

    dataset_name = "conll2003"
    raw_datasets = load_dataset(dataset_name)

    text_column_name = "tokens"
    label_column_name = "ner_tags"

    label_list = raw_datasets["train"].features[label_column_name].feature.names
    num_labels = len(label_list)

    tokenized_datasets = tokenize_token_classification_dataset(
        raw_datasets,
        tokenizer,
        task_id,
        label_list,
        text_column_name,
        label_column_name,
        data_args,
        training_args,
    )

    task_info = Task(
        id=task_id,
        name=dataset_name,
        num_labels=num_labels,
        type="token_classification",
    )

    return (
        tokenized_datasets["train"],
        tokenized_datasets["validation"],
        task_info,
    )

def load_datasets(tokenizer, data_args, training_args):
    (
        sim_para_classification_train_dataset,
        sim_para_classification_validation_dataset,
        sim_para_classification_task,
    ) = load_sim_para_classification_dataset(0, tokenizer, data_args, training_args)
    (
        single_sent_classification_train_dataset,
        single_sent_classification_validation_dataset,
        single_sent_classification_task,
    ) = load_single_sent_classification_dataset(1, tokenizer, data_args, training_args)
    (
        inference_classification_train_dataset,
        inference_classification_validation_dataset,
        inference_classification_task,
    ) = load_inference_classification_dataset(2, tokenizer, data_args, training_args)

    # Merge train datasets
    train_dataset_df = pd.concat([sim_para_classification_train_dataset.to_pandas(),
                                  single_sent_classification_train_dataset.to_pandas(),
                                  inference_classification_train_dataset.to_pandas()])

    train_dataset = datasets.Dataset.from_pandas(train_dataset_df)
    train_dataset.shuffle(seed=123)

    # Append validation datasets
    validation_dataset = [
        sim_para_classification_validation_dataset,
        single_sent_classification_validation_dataset,
        inference_classification_validation_dataset
    ]

    dataset = datasets.DatasetDict(
        {"train": train_dataset, "validation": validation_dataset}
    )
    tasks = [sim_para_classification_task, single_sent_classification_task, inference_classification_task]
    # tasks = [sim_para_classification_task,  inference_classification_task]
    #
    return tasks, dataset

def tokenize_glue_datasets(
    tokenizer, raw_datasets, task_id, task_name, data_args, training_args
):
    task_to_keys = {
        "cola": ("sentence", None),
        "mnli": ("premise", "hypothesis"),
        "mnli-mm": ("premise", "hypothesis"),
        "mrpc": ("sentence1", "sentence2"),
        "qnli": ("question", "sentence"),
        "qqp": ("question1", "question2"),
        "rte": ("sentence1", "sentence2"),
        "sst2": ("sentence", None),
        "stsb": ("sentence1", "sentence2"),
        "wnli": ("sentence1", "sentence2"),
    }
    sentence1_key, sentence2_key = task_to_keys[task_name]

    validation_key = (
        "validation_mismatched"
        if task_name == "mnli-mm"
        else "validation_matched"
        if task_name == "mnli"
        else "validation"
    )
    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def tokenize_text(examples):
        args = (
            (examples[sentence1_key],)
            if sentence2_key is None
            else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(
            *args, padding=padding, max_length=max_seq_length, truncation=True
        )
        examples["labels"] = examples.pop("label")
        result["task_ids"] = [task_id] * len(examples["labels"])
        return result

    def tokenize_and_pad_text(examples):
        result = tokenize_text(examples)

        examples["labels"] = [
            [l] + [-100] * (max_seq_length - 1) for l in examples["labels"]
        ]
        return result

    with training_args.main_process_first(desc="dataset map pre-processing"):
        if sentence2_key is None:
            col_to_remove = ["idx", sentence1_key]
        else:
            col_to_remove = ["idx", sentence1_key, sentence2_key]
        train_dataset = raw_datasets["train"].map(
            tokenize_and_pad_text,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            remove_columns=col_to_remove,
            desc="Running tokenizer on dataset",
        )
        validation_dataset = raw_datasets[validation_key].map(
            tokenize_text,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            remove_columns=col_to_remove,
            desc="Running tokenizer on dataset",
        )

    return train_dataset, validation_dataset

def load_single_sent_classification_dataset(task_id, tokenizer, data_args, training_args):
    # task_list = ['cola', 'sst2']
    # randint = np.random.randint(0,2)
    # task_name = task_list[randint]
    task_name = "cola"
    raw_datasets = load_dataset("glue", task_name)

    num_labels = len(raw_datasets["train"].features["label"].names)

    task_info = Task(
        id=task_id, name=task_name, num_labels=num_labels, type="single_sent_classification"
    )

    train_dataset, validation_dataset = tokenize_glue_datasets(
        tokenizer,
        raw_datasets,
        task_id,
        task_info.name,
        data_args,
        training_args,
    )
    return train_dataset, validation_dataset, task_info

def load_inference_classification_dataset(task_id, tokenizer, data_args, training_args):
    # task_list = ['mnli', 'qnli', 'rte']
    # randint = np.random.randint(0,4)
    # task_name = task_list[randint]
    task_name = "rte"
    raw_datasets = load_dataset("glue", task_name)

    num_labels = len(raw_datasets["train"].features["label"].names)

    task_info = Task(
        id=task_id, name=task_name, num_labels=num_labels, type="inference_classification"
    )

    train_dataset, validation_dataset = tokenize_glue_datasets(
        tokenizer,
        raw_datasets,
        task_id,
        task_info.name,
        data_args,
        training_args,
    )
    return train_dataset, validation_dataset, task_info

def load_sim_para_classification_dataset(task_id, tokenizer, data_args, training_args):
    # task_list = ['mrpc', 'stsb', 'qqp']
    # randint = np.random.randint(0,3)
    # task_name = task_list[randint]
    task_name = "mrpc"
    raw_datasets = load_dataset("glue", task_name)

    num_labels = len(raw_datasets["train"].features["label"].names)

    task_info = Task(
        id=task_id, name=task_name, num_labels=num_labels, type="sim_para_classification"
    )

    train_dataset, validation_dataset = tokenize_glue_datasets(
        tokenizer,
        raw_datasets,
        task_id,
        task_info.name,
        data_args,
        training_args,
    )
    return train_dataset, validation_dataset, task_info

