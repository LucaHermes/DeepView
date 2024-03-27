import os
import shutil

import evaluate
import torch
from torch import nn
# import tensorflow as tf
# import tensorflow_hub as hub
import numpy as np
# import tensorflow_text as text
# from official.nlp import optimization  # to create AdamW optimizer
import matplotlib.pyplot as plt
from transformers import TFBertModel
import transformers
import random
from keras import utils
import pandas as pd
from .Multi_Task_Data_load import *
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoModel,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification,
    default_data_collator,
    set_seed,
)
import evaluate
import sys
import datasets
import logging
logger = logging.getLogger(__name__)

class MultiTaskModel(nn.Module):
    def __init__(self, model_name_or_path, tasks: []):
        super().__init__()

        # if state == "new":
        self.encoder = AutoModel.from_pretrained(model_name_or_path)

        self.output_heads = nn.ModuleDict()
        for task in tasks:
            decoder = self._create_output_head(self.encoder.config.hidden_size, task)
            # ModuleDict requires keys to be strings
            self.output_heads[str(task.id)] = decoder
        # else:


    @staticmethod
    def _create_output_head(encoder_hidden_size: int, task):
        if task.type == "glue":
            return SequenceClassificationHead(encoder_hidden_size, task.num_labels)
        elif task.type == "sim_para_classification":
            return SequenceClassificationHead(encoder_hidden_size, task.num_labels)
        elif task.type == "single_sent_classification":
            return SequenceClassificationHead(encoder_hidden_size, task.num_labels)
        elif task.type == "inference_classification":
            return SequenceClassificationHead(encoder_hidden_size, task.num_labels)
        # elif task.type == "token_classification":
        #     return TokenClassificationHead(encoder_hidden_size, task.num_labels)
        else:
            raise NotImplementedError()

    def embed(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        task_ids=None,
        **kwargs,
    ):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output, pooled_output = outputs[:2]
        return pooled_output

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        task_ids=None,
        **kwargs,
    ):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output, pooled_output = outputs[:2]

        unique_task_ids_list = torch.unique(task_ids).tolist()

        loss_list = []
        logits = None
        for unique_task_id in unique_task_ids_list:

            task_id_filter = task_ids == unique_task_id
            logits, task_loss = self.output_heads[str(unique_task_id)].forward(
                # sequence_output[task_id_filter], #only need for token classification
                pooled_output[task_id_filter],
                labels=None if labels is None else labels[task_id_filter],
                attention_mask=attention_mask[task_id_filter],
            )

            if labels is not None:
                loss_list.append(task_loss)

        # logits are only used for eval. and in case of eval the batch is not multi task
        # For training only the loss is used
        outputs = (logits, outputs[2:])

        if loss_list:
            loss = torch.stack(loss_list)
            outputs = (loss.mean(),) + outputs

        return outputs


# class TokenClassificationHead(nn.Module):
#     def __init__(self, hidden_size, num_labels, dropout_p=0.1):
#         super().__init__()
#         self.dropout = nn.Dropout(dropout_p)
#         self.classifier = nn.Linear(hidden_size, num_labels)
#         self.num_labels = num_labels
#
#         self._init_weights()
#
#     def _init_weights(self):
#         self.classifier.weight.data.normal_(mean=0.0, std=0.02)
#         if self.classifier.bias is not None:
#             self.classifier.bias.data.zero_()
#
#     def forward(
#         self, sequence_output, pooled_output, labels=None, attention_mask=None, **kwargs
#     ):
#         sequence_output_dropout = self.dropout(sequence_output)
#         logits = self.classifier(sequence_output_dropout)
#
#         loss = None
#         if labels is not None:
#             loss_fct = torch.nn.CrossEntropyLoss()
#
#             labels = labels.long()
#
#             # Only keep active parts of the loss
#             if attention_mask is not None:
#                 active_loss = attention_mask.view(-1) == 1
#                 active_logits = logits.view(-1, self.num_labels)
#                 active_labels = torch.where(
#                     active_loss,
#                         labels.view(-1),
#                     torch.tensor(loss_fct.ignore_index).type_as(labels),
#                 )
#                 loss = loss_fct(active_logits, active_labels)
#             else:
#                 loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
#
#         return logits, loss


class SequenceClassificationHead(nn.Module):
    def __init__(self, hidden_size, num_labels, dropout_p=0.1):
        super().__init__()
        self.num_labels = num_labels
        self.dropout = nn.Dropout(dropout_p)
        self.classifier = nn.Linear(hidden_size, num_labels)

        self._init_weights()

    def forward(self,  pooled_output, labels=None, **kwargs):
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if labels.dim() != 1:
                # Remove padding
                labels = labels[:, 0]

            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                logits.view(-1, self.num_labels), labels.long().view(-1)
            )

        return logits, loss

    def _init_weights(self):
        self.classifier.weight.data.normal_(mean=0.0, std=0.02)
        if self.classifier.bias is not None:
            self.classifier.bias.data.zero_()

def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions

    if preds.ndim == 2:
        # Token classification
        preds = np.argmax(preds, axis=1)
        def f1_score(preds, labels):
            # Calculate True Positives, False Positives, False Negatives
            TP = np.sum((preds == 1) & (labels == 1))
            FP = np.sum((preds == 1) & (labels == 0))
            FN = np.sum((preds == 0) & (labels == 1))

            # Calculate Precision and Recall
            precision = TP / (TP + FP) if TP + FP > 0 else 0
            recall = TP / (TP + FN) if TP + FN > 0 else 0

            # Calculate F1 Score
            f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

            return f1.astype(np.float32)

        def matthews_corr_coef(preds, labels):
            TP = np.sum((preds == 1) & (labels == 1))
            TN = np.sum((preds == 0) & (labels == 0))
            FP = np.sum((preds == 1) & (labels == 0))
            FN = np.sum((preds == 0) & (labels == 1))

            numerator = TP * TN - FP * FN
            denominator = np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))

            mcc = numerator / denominator if denominator != 0 else 0

            return mcc.astype(np.float32)

        return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item(),
                "F1_score": f1_score(preds, p.label_ids).item(),
                "Matthew's Corr": matthews_corr_coef(preds,p.label_ids).item() }

    if preds.ndim == 3:
        # Sequence classification
        metric = evaluate.load("seqeval")

        predictions = np.argmax(preds, axis=2)

        true_predictions = [
            [f"tag-idx-{p}" for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, p.label_ids)
        ]
        true_labels = [
            [f"tag-idx-{l}" for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, p.label_ids)
        ]

        # Remove ignored index (special tokens)
        results = metric.compute(
            predictions=true_predictions, references=true_labels
        )
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }
    else:
        raise NotImplementedError()


def main(model_args, data_args, training_args):
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if (
            os.path.isdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif (
                last_checkpoint is not None and training_args.resume_from_checkpoint is None
        ):
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    set_seed(training_args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    tasks, raw_datasets = load_datasets(tokenizer, data_args, training_args)

    model = MultiTaskModel(model_args.model_name_or_path, tasks)

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))

    if training_args.do_eval:
        if (
                "validation" not in raw_datasets
                and "validation_matched" not in raw_datasets
        ):
            raise ValueError("--do_eval requires a validation dataset")
        eval_datasets = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            new_ds = []
            for ds in eval_datasets:
                new_ds.append(ds.select(range(data_args.max_eval_samples)))

            eval_datasets = new_ds

    # Log a few random samples from the training set:
    if training_args.do_train:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    data_collator = DataCollatorForTokenClassification(
        tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None
    )
    # print(data_collator)
    # print(tasks)
    # print(raw_datasets)
    #
    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples
            if data_args.max_train_samples is not None
            else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:

        for eval_dataset, task in zip(eval_datasets, tasks):
            logger.info(f"*** Evaluate {task} ***")
            data_collator = None
            if task.type == "token_classification":
                data_collator = DataCollatorForTokenClassification(
                    tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None
                )
            else:
                if data_args.pad_to_max_length:
                    data_collator = default_data_collator
                elif training_args.fp16:
                    data_collator = DataCollatorWithPadding(
                        tokenizer, pad_to_multiple_of=8
                    )
                else:
                    data_collator = None

            trainer.data_collator = data_collator
            metrics = trainer.evaluate(eval_dataset=eval_dataset)

            max_eval_samples = (
                data_args.max_eval_samples
                if data_args.max_eval_samples is not None
                else len(eval_datasets)
            )
            metrics["eval_samples"] = min(max_eval_samples, len(eval_datasets))

            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    model_args = ModelArguments(model_name_or_path="bert-base-cased")
    training_args = TrainingArguments(
        do_train=True,
        do_eval=True,
        output_dir="/tmp/test",
        learning_rate=2e-5,
        num_train_epochs=3,
        overwrite_output_dir=True,
        remove_unused_columns=False,
    )
    data_args = DataTrainingArguments(max_seq_length=128)
    main(model_args, data_args, training_args)

