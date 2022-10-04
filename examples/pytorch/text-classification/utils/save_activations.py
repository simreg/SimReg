#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE."""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.

import logging
import os

import torch.utils.data

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from run_glue import setup_args, load_datasets, load_labels, load_models
from transformers.utils import check_min_version
from transformers.utils.versions import require_version


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.12.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

logger = logging.getLogger(__name__)


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    data_args, last_checkpoint, model_args, training_args, *other = setup_args()

    # Set seed before initializing model.
    set_seed(training_args.seed)
    raw_datasets = load_datasets(data_args, model_args, training_args)
    is_regression, label_list, num_labels = load_labels(data_args, raw_datasets)

    config, model, tokenizer, _ = load_models(data_args=data_args, model_args=model_args, num_labels=num_labels, training_args=training_args)
    # TODO: continue refactoring to run_glue functions
    # Preprocessing the raw_datasets
    if data_args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[data_args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in raw_datasets["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and data_args.task_name is not None
        and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            label_to_id = {i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)}
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )
    elif data_args.task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in config.label2id.items()}
    elif data_args.task_name is not None and not is_regression:
        model.config.label2id = {l: i for i, l in enumerate(label_list)}
        model.config.id2label = {id: label for label, id in config.label2id.items()}

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)

        # Map labels to IDs (not necessary for GLUE tasks)
        if label_to_id is not None and "label" in examples:
            result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]
        return result

    with training_args.main_process_first(desc="dataset map pre-processing"):
        raw_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )


    # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    training_args: TrainingArguments = training_args
    if training_args.do_train:
        ds_name = 'train'
    elif training_args.do_eval:
        ds_name = 'validation_matched'
    else:
        raise Exception('Please select dataset (eval/train)')

    predict_dataset = raw_datasets[ds_name]
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=None,
        eval_dataset=None,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )


    predict_dataloader = trainer.get_test_dataloader(predict_dataset)
    activations = dict()
    model.eval()
    from tqdm import tqdm
    sz = 0
    chunk_num = 0
    # currently compatible with Bert based models
    if trainer.is_world_process_zero():
        with torch.no_grad():
            # need to save the data in chunks
            for batch in tqdm(predict_dataloader):
                batch = trainer._prepare_inputs(batch)
                out = model(**batch, output_hidden_states=True)
                # skip the embeddings layer TODO: need to change this to add support for non-bert models
                for layer_num in range(1, len(out.hidden_states)):
                    if layer_num not in activations:
                        activations[layer_num] = []
                    cur_activation = out.hidden_states[layer_num][:, 0, :].clone()
                    activations[layer_num].append(cur_activation)
                    sz += cur_activation.numel()

                # (2 ** 30) / 4 : 1 Gb of data
                if sz > 2 ** 28:
                    save_chunk(activations, chunk_num, ds_name, training_args)
                    sz = 0
                    chunk_num += 1

    if sz > 0:
        save_chunk(activations, chunk_num, ds_name, training_args)
        chunk_num += 1

    for layer_num in activations.keys():
        if chunk_num > 0:
            chunks = []
            for i in range(chunk_num):
                chunk_path = os.path.join(training_args.output_dir,
                                                      f'{ds_name}_{layer_num}_CLS_activations_chunk_{i}.bin')
                chunks += torch.load(chunk_path)
                os.remove(chunk_path)
            torch.save(torch.cat(chunks, dim=0), os.path.join(training_args.output_dir, f'{ds_name}_{layer_num}_CLS_activations.bin'))


def save_chunk(activations, chunk_num, ds_name, training_args):
    logging.info("saving a chunk")
    for layer_num in activations.keys():
        torch.save(activations[layer_num], os.path.join(training_args.output_dir,
                                                        f'{ds_name}_{layer_num}_CLS_activations_chunk_{chunk_num}.bin'))
        activations[layer_num] = []


if __name__ == "__main__":
    main()
