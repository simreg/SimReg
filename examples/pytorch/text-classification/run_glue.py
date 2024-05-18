#!/usr/bin/env python
import json
import logging
import os
import random
import sys
from typing import Tuple, Any, Optional, List

import torch
import datasets
import numpy as np
from datasets import load_dataset, DatasetDict, Dataset, Value, concatenate_datasets, load_from_disk

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    HfArgumentParser,
    PretrainedConfig,
    TrainingArguments,
    default_data_collator,
    set_seed,
    PreTrainedTokenizer
)

from poe_trainer import PoETrainer
from conf_reg_trainer import ConfRegTrainer
# from grad_reg_trainer import BaseGradRegTrainer
# from loss_sampling_trainer import MisclassificationTrainer, TopKLossTrainer
from models.lexical_bias_bert import ClarkLexicalBiasModel, ClarkLexicalBiasConfig
from models.models_weak import BaselineTokenizer, BaselineConfig, BaselineModel
from my_trainer import BaseTrainer, SimRegTrainer
from models import BertWithLexicalBiasModel, BertWithLexicalBiasConfig
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from os import path

from arguments_classes import DataTrainingArguments, ModelArguments, task_to_keys, WandbArguments
from metrics import get_metrics_function
from utils.misc import EvaluationSet, extract_split_name, is_partial_input_model

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.12.0.dev0")
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")
logger = logging.getLogger(__name__)
from utils.misc import DATA_DIR, setup_wandb


def main(outside_usage=False):
    data_args, last_checkpoint, model_args, training_args, *other_args = setup_args((WandbArguments,))
    wandb_args: WandbArguments = other_args[0]

    # Set seed before initializing model.
    set_seed(training_args.seed)

    setup_wandb(data_args, wandb_args, training_args)

    raw_datasets = load_datasets(data_args, model_args, training_args)

    # datasets module MNLI labels order: entailment, neutral, contradiction.
    # POE project mnli labels order: contradiction, entailment, neutral

    is_regression, label_list, num_labels = load_labels(data_args, raw_datasets)

    config, model, tokenizer, weak_models, weak_tokenizers, lexical_bias_settings = load_models(
        data_args=data_args,
        model_args=model_args,
        training_args=training_args,
        num_labels=num_labels
    )

    training_args.separate_weak_tokenization = any(
        wt is not None and wt.get_vocab() != tokenizer.get_vocab() for wt in weak_tokenizers)
    logger.info(f"separate_weak_tokenization: {training_args.separate_weak_tokenization}")

    if training_args.separate_weak_tokenization and not data_args.pad_to_max_length:
        data_args.pad_to_max_length = True
        logger.warning('setting pad_to_max_length to True!')

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
            # EDIT, ignore the re-labeling done.
            # label_to_id = {i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)}
            pass
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

    if tokenizer is not None and data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length if tokenizer is not None else 0)
    assert not (lexical_bias_settings and data_args.synthetic_bias_prevalence > 0)

    preprocessing_func = get_preprocess_function(
        tokenizer=tokenizer,
        sentence1_key=sentence1_key,
        sentence2_key=sentence2_key,
        max_seq_length=max_seq_length,
        padding=padding,
        label_to_id=label_to_id,
        weak_tokenizers=weak_tokenizers if training_args.separate_weak_tokenization else None
    ) if tokenizer is not None else None
    synthetic_bias_indices, raw_datasets = preprocess_raw_datasets(
        data_args=data_args,
        training_args=training_args,
        raw_datasets=raw_datasets,
        context_manager=training_args,
        sentence1_key=sentence1_key,
        lexical_bias_settings=lexical_bias_settings,
        preprocessing_func=preprocessing_func
    )
    if data_args.old_labels_order:
        # datasets module MNLI labels order: entailment, neutral, contradiction.
        # POE project mnli labels order: contradiction, entailment, neutral
        # ['entailment', 'non-entailment']
        # datasets module HANS labels order: entailment, non-entailment
        # ["non-entailment", "entailment"]
        # POE project HANS labels order: non-entailment, entailment

        # FEVER:
        # my code: ["SUPPORTS", "NOT ENOUGH INFO", "REFUTES"]
        # old : ["REFUTES", "SUPPORTS", "NOT ENOUGH INFO"]

        old_labels_mapper = {0: 1, 1: 2, 2: 0}
        hans_labels_map = {0: 1, 1: 0}
        hans_ds = None
        if 'hans' in raw_datasets:
            hans_ds = raw_datasets.pop('hans')

        def labels_mapper(samples):
            samples['label'] = ((torch.tensor(samples['label']) + 1) % 3).tolist()
            # samples['label'] = [(old_labels_mapper[l] if l != -1 else -1) for l in samples["label"]]
            return samples

        def hans_labels_mapper(samples):
            samples['label'] = [(hans_labels_map[l] if l != -1 else -1) for l in samples["label"]]
            return samples

        raw_datasets = raw_datasets.map(labels_mapper, batched=True, load_from_cache_file=False)

        if hans_ds is not None:
            raw_datasets['hans'] = hans_ds.map(hans_labels_mapper, batched=True, load_from_cache_file=False)
        del hans_ds

    if data_args.task_name == 'qqp':
        raw_datasets = raw_datasets.remove_columns(['question1', 'question2'])

    # this is the indices of biases/anti-biased samples in validation set, relevant in synthetic bias settings.
    metrics_bias_indices = synthetic_bias_indices['validation_matched'] if synthetic_bias_indices is not None else None
    biased_indices = None

    # here we extract the indices of biases samples in the training set.
    if data_args.synthetic_bias_prevalence > 0:
        biased_indices = synthetic_bias_indices['train']['aligned_indices']
        logger.info('Using synthetic bias indices for selecting biased samples')
        logger.info(f"Num of biased samples in train: {biased_indices.shape}")
    elif training_args.bias_indices is not None:
        # load bias indices from file provided by args.
        biased_indices = []
        for indices_collection_path in training_args.bias_indices:
            biased_indices.append(torch.load(indices_collection_path).squeeze())
        biased_indices = torch.cat(biased_indices).unique()
        logger.info(f"Using the concatenation of {training_args.bias_indices} for selecting biased samples")
        logger.info(f"Num of biased samples in train: {biased_indices.shape}")

    eval_dataset, predict_dataset, train_dataset = get_datasets(
        data_args=data_args,
        raw_datasets=raw_datasets,
        training_args=training_args,
        tokenizer=tokenizer,
        biased_samples=biased_indices
    )

    # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    evaluation_sets = extract_evaluation_sets(data_args, eval_dataset, raw_datasets, wandb_args)

    trainer = build_trainer(
        bias_indices=biased_indices,
        data_args=data_args,
        data_collator=data_collator,
        eval_dataset=eval_dataset,
        is_regression=is_regression,
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        training_args=training_args,
        weak_models=weak_models,
        metrics_bias_indices=metrics_bias_indices,
        evaluation_sets=evaluation_sets
    )

    if outside_usage:
        return trainer, training_args, data_args, model_args

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
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        do_eval(data_args=data_args, eval_dataset=eval_dataset, raw_datasets=raw_datasets, trainer=trainer)

    if training_args.do_predict:
        do_predict(data_args, is_regression, label_list, predict_dataset, raw_datasets, trainer, training_args)

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-classification"}
    if data_args.task_name is not None:
        kwargs["language"] = "en"
        kwargs["dataset_tags"] = "glue"
        kwargs["dataset_args"] = data_args.task_name
        kwargs["dataset"] = f"GLUE {data_args.task_name.upper()}"

    if training_args.create_model_card:
        try:
            trainer.create_model_card(**kwargs)
        except:
            pass

    # Clear optimizer state files (optimizer.pt) to save disk space.
    from glob import glob
    for f in glob(f'{training_args.output_dir}/checkpoint-*/optimizer.pt'):
        logger.info(f"removing {f}")
        os.remove(f)


def extract_evaluation_sets(data_args, eval_dataset, raw_datasets, wandb_args):
    evaluation_sets: List[EvaluationSet] = []
    if 'hans' in raw_datasets:
        from metrics import hans_compute_metrics
        hans_ds = raw_datasets['hans']
        hans_mask = np.array(hans_ds['label']) == (0 if not data_args.old_labels_order else 1)
        hans_ent = hans_ds.select(np.nonzero(hans_mask)[0])
        from functools import partial
        hans_metric = partial(hans_compute_metrics, old_labels_order=data_args.old_labels_order)
        evaluation_sets.append(EvaluationSet(set=hans_ent, logging_mode='eval_HANS_ent', metrics_func=hans_metric))
        hans_non_ent = hans_ds.select(np.nonzero(~hans_mask)[0])
        evaluation_sets.append(
            EvaluationSet(set=hans_non_ent, logging_mode='eval_HANS_non_ent', metrics_func=hans_metric))
    # adding validation matched subsets
    if data_args.indices_dir is not None:
        for split_dir in data_args.indices_dir:
            splits = os.listdir(split_dir)
            for split_file_name in splits:
                if split_file_name.split(".")[-1] != "bin":
                    continue
                split_indices = torch.load(os.path.join(split_dir, split_file_name))
                split_ds = eval_dataset.select(split_indices)
                split_name = extract_split_name(split_dir, split_file_name)
                evaluation_sets.append(EvaluationSet(set=split_ds, logging_mode=f'eval_{split_name}'))

    if data_args.task_name == "mnli" and wandb_args.tags is not None and (
            'hypothesis_only_bias' in wandb_args.tags or 'unknown_bias' in wandb_args.tags):
        if data_args.mismatched_indices_dir is None:
            data_args.mismatched_indices_dir = []
        if 'data/hypothesis_bias_splits/validation_mismatched' not in data_args.mismatched_indices_dir:
            data_args.mismatched_indices_dir.append("data/hypothesis_bias_splits/validation_mismatched")

    # adding validation-mismatched subsets
    if data_args.mismatched_indices_dir is not None:
        for split_dir in data_args.mismatched_indices_dir:
            splits = os.listdir(split_dir)
            for split_file_name in splits:
                if split_file_name.split(".")[-1] != "bin":
                    continue
                split_indices = torch.load(os.path.join(split_dir, split_file_name))
                split_ds = raw_datasets['validation_mismatched'].select(split_indices)
                split_name = extract_split_name(split_dir, split_file_name)
                evaluation_sets.append(EvaluationSet(set=split_ds, logging_mode=f'eval_{split_name}'))
    if data_args.synthetic_bias_prevalence > 0:
        evaluation_sets.append(
            EvaluationSet(set=raw_datasets['unbiased_validation_matched'], logging_mode='eval_unbiased',
                          metrics_func=get_metrics_function(
                              False, data_args, None)))
    if data_args.task_name == 'fever':
        evaluation_sets.append(EvaluationSet(set=raw_datasets['fever_symmetric'], logging_mode='eval_FEVER_symmetric'))
        evaluation_sets.append(
            EvaluationSet(set=raw_datasets['fever_symmetric_v2'], logging_mode='eval_FEVER_symmetricV2'))
    if data_args.task_name == 'qqp':
        evaluation_sets.append(EvaluationSet(set=raw_datasets['paws_test'], logging_mode='eval_PAWS'))
    return evaluation_sets


def load_datasets(data_args, model_args, training_args):
    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.

    if data_args.task_name is not None:
        # Downloading and loading a dataset from the hub.
        if data_args.task_name == 'fever':
            raw_datasets = load_dataset(path.join(DATA_DIR, 'fever/fever_nli.py'), cache_dir=model_args.cache_dir,
                                        trust_remote_code=True)
            sym = load_dataset(path.join(DATA_DIR, 'fever/fever_symmetric.py'), cache_dir=model_args.cache_dir,
                               trust_remote_code=True)
            raw_datasets['fever_symmetric'] = sym['test'].cast_column('id', Value(dtype='int64', id=None))
            raw_datasets['fever_symmetric_v2'] = sym['test_v2'].cast_column('id', Value(dtype='int64', id=None))
        else:
            raw_datasets = datasets.load_from_disk(path.join(DATA_DIR, "cached_glue", data_args.task_name))
            # raw_datasets = load_dataset("glue", data_args.task_name, cache_dir=model_args.cache_dir)
            if data_args.task_name == "qqp":
                paws = load_dataset('csv', data_files={'test': path.join(DATA_DIR, 'PAWS_QQP/dev_and_test.tsv')},
                                    delimiter='\t')['test']
                paws = paws.rename_columns({'sentence1': 'question1', 'sentence2': 'question2'})
                raw_datasets['paws_test'] = paws
            elif data_args.task_name == 'mnli' and (training_args.do_train or training_args.evaluate_on_hans):
                # beware of adding this in synthetic bias
                raw_datasets['hans'] = datasets.load_from_disk(path.join(DATA_DIR, "cached_glue", "hans"))['validation']

    elif data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name, data_args.dataset_config_name, cache_dir=model_args.cache_dir
        )
    else:
        # Loading a dataset from your local files.
        # CSV/JSON training and evaluation files are needed.
        data_files = {"train": data_args.train_file, "validation": data_args.validation_file}

        # Get the test dataset: you can provide your own CSV/JSON test file (see below)
        # when you use `do_predict` without specifying a GLUE benchmark task.
        if training_args.do_predict:
            if data_args.test_file is not None:
                train_extension = data_args.train_file.split(".")[-1]
                test_extension = data_args.test_file.split(".")[-1]
                assert (
                        test_extension == train_extension
                ), "`test_file` should have the same extension (csv or json) as `train_file`."
                data_files["test"] = data_args.test_file
            else:
                raise ValueError("Need either a GLUE task or a test file for `do_predict`.")

        for key in data_files.keys():
            logger.info(f"load a local file for {key}: {data_files[key]}")

        if data_args.train_file.endswith(".csv"):
            # Loading a dataset from local csv files
            raw_datasets = load_dataset("csv", data_files=data_files, cache_dir=model_args.cache_dir)
        else:
            # Loading a dataset from local json files
            raw_datasets = load_dataset("json", data_files=data_files, cache_dir=model_args.cache_dir)

    return raw_datasets


def build_trainer(bias_indices, data_args, data_collator, eval_dataset, is_regression, model, tokenizer,
                  train_dataset, training_args, weak_models, metrics_bias_indices=None,
                  evaluation_sets: List[EvaluationSet] = None) -> BaseTrainer:
    trainer_cls = BaseTrainer
    kwargs = {}
    special_trainer = {
        # 'missclassification_only': MisclassificationTrainer,
        # 'topk_loss_trainer': TopKLossTrainer,
        'POE': PoETrainer,
        'conf_reg': ConfRegTrainer,
    }
    if training_args.regularization_method is not None:
        if training_args.regularization_method in special_trainer.keys():
            trainer_cls = special_trainer[training_args.regularization_method]
        else:
            if training_args.regularize_grads:
                raise Exception("Error: Deprecated")
                # from grad_reg_trainer import BaseGradRegTrainer
                # trainer_cls = BaseGradRegTrainer
            else:
                trainer_cls = SimRegTrainer
    if weak_models is not None:
        kwargs.update({'weak_models': weak_models})

    if training_args.regularization_method in ['conf_reg', 'POE']:
        kwargs.update({'columns_to_keep': ['idx', 'id']})

    if training_args.regularize_only_biased:
        kwargs.update({'columns_to_keep': ['idx', 'id'], 'bias_indices': bias_indices})

    if training_args.separate_weak_tokenization:
        if 'columns_to_keep' not in kwargs:
            kwargs['columns_to_keep'] = []
        for fw_idx in range(len(weak_models)):
            kwargs['columns_to_keep'].append(f'weak_{fw_idx}_input_ids')
            kwargs['columns_to_keep'].append(f'weak_{fw_idx}_attention_mask')
            kwargs['columns_to_keep'].append(f'weak_{fw_idx}_token_type_ids')

    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=get_metrics_function(is_regression, data_args, synthetic_bias_indices=metrics_bias_indices),
        tokenizer=tokenizer,
        data_collator=data_collator,
        evaluation_sets=evaluation_sets,
        **kwargs
    )
    trainer.synthetic_bias_settings = data_args.synthetic_bias_prevalence > 0

    return trainer


def setup_args(additional_args: Tuple[Any] = ()) -> Tuple[
    DataTrainingArguments, Any, ModelArguments, TrainingArguments, Any]:
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    args_set = (ModelArguments, DataTrainingArguments, TrainingArguments)
    if len(additional_args) > 0:
        args_set = args_set + additional_args
    parser = HfArgumentParser(args_set)
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, *other_args = parser.parse_json_file(json_file=path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, *other_args = parser.parse_args_into_dataclasses()

    setup_logging(training_args)
    # Detecting last checkpoint.
    last_checkpoint = None
    if path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    from transformers.modelcard import _TRAINING_ARGS_KEYS
    training_args: TrainingArguments
    _TRAINING_ARGS_KEYS.append('regularize_grads')
    _TRAINING_ARGS_KEYS.append('enforce_similarity')
    _TRAINING_ARGS_KEYS.append('regularization_delay')

    if training_args.bias_indices is not None:
        _TRAINING_ARGS_KEYS.append('bias_indices')

    return data_args, last_checkpoint, model_args, training_args, *other_args


# using wrapper to save the context
def get_preprocess_function(tokenizer, sentence1_key, sentence2_key, max_seq_length, padding, label_to_id=None,
                            weak_tokenizers=None):
    def preprocess_function(examples):
        if sentence1_key is None:
            args = ((examples[sentence2_key],))
        elif sentence2_key is None:
            args = ((examples[sentence1_key],))
        else:
            args = ((examples[sentence1_key], examples[sentence2_key]))

        result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)
        if weak_tokenizers is not None and len(weak_tokenizers) > 0:
            for model_idx, wt in enumerate(weak_tokenizers):
                weak_encoding = wt(*args, padding='max_length', max_length=max_seq_length, truncation=True)
                for k in weak_encoding:
                    result[f'weak_{model_idx}_{k}'] = weak_encoding[k]

        # Map labels to IDs (not necessary for GLUE tasks)
        if label_to_id is not None and "label" in examples:
            result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]
        return result

    return preprocess_function


def preprocess_raw_datasets(data_args: DataTrainingArguments, raw_datasets: DatasetDict, context_manager,
                            sentence1_key: str,
                            training_args: TrainingArguments, num_labels: int = 3, lexical_bias_settings: bool = False,
                            preprocessing_func=None) \
        -> Tuple[Optional[torch.Tensor], DatasetDict]:
    """
        Tokenize / Synthesize bias / load processed cached dataset
    """
    if lexical_bias_settings:
        # Load cached pre-processed dataset
        # pre-processing is done earlier for lexical features.
        if training_args.separate_weak_tokenization and data_args.task_name == 'mnli':
            # DeBERTa case
            lex_dataset = load_from_disk(path.join(DATA_DIR, 'padded_mnli_with_hans_lexical_features'))
            old_columns = lex_dataset['train'].column_names
            columns_mapping = {}
            for old_column in old_columns:
                if old_column == 'id':
                    continue
                columns_mapping[old_column] = "weak_" + old_column
            lex_dataset = DatasetDict(
                {k: dataset.rename_columns(column_mapping=columns_mapping) for k, dataset in lex_dataset.items()})
            for split in raw_datasets.keys():
                if split in lex_dataset:
                    raw_datasets[split] = concatenate_datasets([raw_datasets[split], lex_dataset[split]], axis=1)
        else:
            mapper = {
                'mnli': 'MNLI_HANS_Clark',
                'qqp': 'QQP_clark_lex'
            }
            if data_args.old_lexical_settings:
                mapper['mnli'] = 'padded_mnli_with_hans_lexical_features'

            lex_ds = datasets.load_from_disk(path.join(DATA_DIR, mapper[data_args.task_name]))
            if preprocessing_func is None:
                return None, lex_ds
            lex_ds = lex_ds.remove_columns('label')
            for split in raw_datasets.keys():
                if split in lex_ds:
                    raw_datasets[split] = concatenate_datasets([raw_datasets[split], lex_ds[split]], axis=1)

    def synthesize_bias(ds_name, ds_object):
        logging.info(f'synthesizing bias for {ds_name}')
        # indices of samples to be injected with bias tokens
        biased_indices = np.random.choice(len(ds_object['label']),
                                          int(data_args.synthetic_bias_prevalence * len(ds_object['label'])),
                                          replace=False)

        aligned_samples_num = int(data_args.bias_correlation_prob * len(biased_indices))
        aligned_indices = biased_indices[:aligned_samples_num]
        misaligned_indices = biased_indices[aligned_samples_num:]
        labels_offset = np.random.randint(1, num_labels, len(misaligned_indices))
        torch.save({'aligned_indices': aligned_indices, 'misaligned_indices': misaligned_indices},
                   path.join(data_cache_dir, f"{ds_name}_biased_indices"))

        def label_injector_preprocessing(examples):
            indices = np.array(examples['idx'])

            _, indices_in_batch, _ = np.intersect1d(indices, aligned_indices, return_indices=True)
            # inject bias
            for i in indices_in_batch:
                examples[sentence1_key][i] = str(examples['label'][i]) + " " + examples[sentence1_key][i]

            _, anti_biased_batch_idx, anti_biased_overall_idx = np.intersect1d(indices, misaligned_indices,
                                                                               return_indices=True)
            for batch_i, misaligned_j in zip(anti_biased_batch_idx, anti_biased_overall_idx):
                new_label = (examples['label'][batch_i] + labels_offset[misaligned_j]) % num_labels
                examples[sentence1_key][batch_i] = str(new_label) + " " + examples[sentence1_key][batch_i]

            # tokenize
            return preprocessing_func(examples)

        return label_injector_preprocessing

    with context_manager.main_process_first(desc="dataset map pre-processing"):
        if data_args.synthetic_bias_prevalence > 0:
            unbiased_validation_matched = raw_datasets['validation_matched'].map(preprocessing_func,
                                                                                 batched=True,
                                                                                 load_from_cache_file=True,
                                                                                 desc='tokenizing unbiased validation set')

        data_cache_dir = f"ds_cache/mnli_{data_args.synthetic_bias_prevalence}_{data_args.bias_correlation_prob}_bias"
        if data_args.synthetic_bias_prevalence > 0 and not path.isdir(data_cache_dir):
            os.makedirs(data_cache_dir)
            raw_datasets = DatasetDict({
                k: dataset.map(
                    synthesize_bias(k, dataset),
                    batched=True,
                    load_from_cache_file=True,
                    cache_file_name=path.join(data_cache_dir, k),
                    desc=f"Running tokenizer on synthetic dataset {k}",
                ) for k, dataset in raw_datasets.items()
            })
        else:
            raw_datasets = raw_datasets.map(
                preprocessing_func,
                batched=True,
                load_from_cache_file=not data_args.overwrite_cache,
                cache_file_names={
                    k: path.join(data_cache_dir, k) if data_args.synthetic_bias_prevalence > 0 else None
                    for k in raw_datasets},
                desc="Running tokenizer on datasets",
            )
        # load synthetic bias indices
        if data_args.synthetic_bias_prevalence > 0:
            synthetic_bias_indices = {}
            for k in raw_datasets.keys():
                synthetic_bias_indices[k] = torch.load(path.join(data_cache_dir, f"{k}_biased_indices"))

            raw_datasets['unbiased_validation_matched'] = unbiased_validation_matched
        else:
            synthetic_bias_indices = None

    return synthetic_bias_indices, raw_datasets


def get_datasets(data_args, raw_datasets, training_args, tokenizer: PreTrainedTokenizer = None, biased_samples=None):
    predict_dataset = None
    eval_dataset = None
    train_dataset = None

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]

        if data_args.remove_biased_samples_from_train:
            train_dataset = train_dataset.select(np.setdiff1d(np.arange(len(train_dataset)), biased_samples))
            logger.info(f"Train len: {len(train_dataset)}")
        elif data_args.select_only_biased_samples:
            train_dataset = train_dataset.select(biased_samples)
            logger.info(
                f"Restricting train to biased samples provided by bias_indices argument, Train len: {len(train_dataset)}")

        if data_args.max_train_samples is not None:
            logger.info(f"Subseting training dataset to len({data_args.max_train_samples})")
            train_dataset = train_dataset.select(torch.randperm(len(train_dataset))[:data_args.max_train_samples])

    if training_args.do_eval:
        if "validation" not in raw_datasets and "validation_matched" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation_matched" if data_args.task_name == "mnli" else "validation"]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))
    # currently, FEVER does not have a test set.
    if training_args.do_predict or (
            data_args.task_name is not None and data_args.task_name not in [
        'fever']) or data_args.test_file is not None:
        if "test" not in raw_datasets and "test_matched" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test_matched" if data_args.task_name == "mnli" else "test"]
        if data_args.max_predict_samples is not None:
            predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))
    # Log a few random samples from the training set:
    if training_args.do_train:
        for index in random.sample(range(len(train_dataset)), 3):
            sample = train_dataset[index]
            logger.info(f"Sample {index} of the training set: {sample}.")
            if tokenizer is not None:
                sample_inputs = torch.tensor(sample['input_ids'])
                sample_attention_mask = torch.tensor(sample['attention_mask'])
                logger.info(f"decoded input: {tokenizer.decode(sample_inputs[sample_attention_mask == 1])}")
    if training_args.do_eval:
        for index in random.sample(range(len(eval_dataset)), 3):
            sample = eval_dataset[index]
            logger.info(f"Sample {index} of the evaluation set: {sample}.")
            if tokenizer is not None:
                sample_inputs = torch.tensor(sample['input_ids'])
                sample_attention_mask = torch.tensor(sample['attention_mask'])
                logger.info(f"decoded input: {tokenizer.decode(sample_inputs[sample_attention_mask == 1])}")
    return eval_dataset, predict_dataset, train_dataset


def load_labels(data_args, raw_datasets):
    # Labels
    if data_args.task_name is not None:
        is_regression = data_args.task_name == "stsb"
        if not is_regression:
            label_list = raw_datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = raw_datasets["train"].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = raw_datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)
    return is_regression, label_list, num_labels


def setup_logging(training_args):
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s | %(filename)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    logging.root.setLevel(log_level)
    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")


def load_models(data_args: DataTrainingArguments, model_args: ModelArguments, training_args: TrainingArguments,
                num_labels: int = 3) -> Tuple[
    AutoConfig, AutoModelForSequenceClassification, PreTrainedTokenizer, List[AutoModelForSequenceClassification],
    Optional[
        List[PreTrainedTokenizer]], bool]:
    # Load pretrained model and tokenizer
    if data_args.lexical_bias_model:
        config_cls = BertWithLexicalBiasConfig if data_args.old_lexical_settings else ClarkLexicalBiasConfig
    else:
        config_cls = AutoConfig
    if path.isfile(path.join(model_args.model_name_or_path, "config.json")):
        with open(path.join(model_args.model_name_or_path, "config.json"), 'r') as fp:
            config = json.load(fp)
            config_mapper = {
                'ClarkLexicalBiasModel': ClarkLexicalBiasConfig,
                'BOW': BaselineConfig
            }
            if 'architectures' in config and config['architectures'][0] in config_mapper.keys():
                config_cls = config_mapper[config['architectures'][0]]

    if model_args.model_name_or_path.lower() == 'bow' or os.path.exists(
            os.path.join(model_args.model_name_or_path, 'class.txt')):
        config_cls = BaselineConfig

    config = config_cls.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None
    )
    if not data_args.lexical_bias_model and config_cls != ClarkLexicalBiasConfig:
        tokenizer_args = {}
        tokenizer_cls = AutoTokenizer
        if config_cls == BaselineConfig:
            tokenizer_args.update({'vocab_file': config.vocab_file, 'do_lower_case': False})
            tokenizer_cls = BaselineTokenizer

        tokenizer = tokenizer_cls.from_pretrained(
            model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            **tokenizer_args
        )
    else:
        tokenizer = None

    lexical_bias_settings = False
    weak_models = []
    weak_tokenizers = []
    for fw_path in training_args.weak_models_path:
        if os.path.exists(os.path.join(fw_path, 'class.txt')):
            config_cls = BaselineConfig
        else:
            config_cls = AutoConfig

        weak_model_config = config_cls.from_pretrained(fw_path, num_labels=num_labels,
                                                       finetuning_task=data_args.task_name)

        if is_partial_input_model(weak_model_config):
            from models import PartialInputBert
            model_cls = PartialInputBert
        elif weak_model_config.architectures is not None:
            if 'BertWithLexicalBiasModel' in weak_model_config.architectures:
                model_cls = BertWithLexicalBiasModel
                lexical_bias_settings = True
            elif 'ClarkLexicalBiasModel' in weak_model_config.architectures:
                model_cls = ClarkLexicalBiasModel
                lexical_bias_settings = True
            elif 'BOW' in weak_model_config.architectures:
                model_cls = BaselineModel
            else:
                model_cls = AutoModelForSequenceClassification
        else:
            model_cls = AutoModelForSequenceClassification

        weak_models.append(model_cls.from_pretrained(fw_path, config=weak_model_config).to(training_args.device))
        tokenizer_args = {}
        tokenizer_cls = AutoTokenizer
        if model_cls == BaselineModel:
            tokenizer_args.update(
                {'vocab_file': weak_model_config.vocab_file, 'do_lower_case': weak_model_config.do_lower_case})
            tokenizer_cls = BaselineTokenizer

        weak_tokenizers.append(tokenizer_cls.from_pretrained(fw_path) if model_cls != ClarkLexicalBiasModel else None)

    if data_args.hypothesis_only:
        config.hypothesis_only = True
    if data_args.claim_only:
        config.claim_only = True

    if is_partial_input_model(config):
        from models import PartialInputBert
        model_cls = PartialInputBert
    elif config.architectures is not None and 'BertWithLexicalBiasModel' in config.architectures:
        model_cls = BertWithLexicalBiasModel
    elif config.architectures is not None and 'ClarkLexicalBiasModel' in config.architectures:
        model_cls = ClarkLexicalBiasModel
    elif data_args.lexical_bias_model and data_args.old_lexical_settings:
        model_cls = BertWithLexicalBiasModel
    elif data_args.lexical_bias_model and not data_args.old_lexical_settings:
        model_cls = ClarkLexicalBiasModel
    elif config_cls == BaselineConfig:
        model_cls = BaselineModel
    else:
        model_cls = AutoModelForSequenceClassification

    if model_cls in (ClarkLexicalBiasModel, BertWithLexicalBiasModel):
        lexical_bias_settings = True

    model, loading_info = model_cls.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        output_loading_info=True
    )

    logger.info(loading_info)

    if lexical_bias_settings:
        assert not training_args.remove_unused_columns

    if model_args.randomly_initialize_model:
        logger.info("Initializing weights of main model randomly!")
        model.init_weights()
    if model_args.freeze_embeddings:
        logger.info("Freezing embeddings of the model")
        model.bert.embeddings.requires_grad_(False)
    if model_args.freeze_encoder:
        logger.info("Freezing encoder!")
        model.bert.requires_grad_(False)
        if model_args.new_clf_head:
            logger.info("Initializing classifier")
            model.classifier.apply(model._init_weights)

    return config, model, tokenizer, weak_models if len(
        weak_models) > 0 else None, weak_tokenizers, lexical_bias_settings


def do_eval(
        data_args: DataTrainingArguments,
        eval_dataset: Dataset,
        raw_datasets,
        trainer: BaseTrainer,
):
    logger.info("*** Evaluate ***")
    evaluation_sets = [EvaluationSet(set=eval_dataset, logging_mode='eval')]
    if data_args.task_name == "mnli":
        evaluation_sets.append(EvaluationSet(raw_datasets["validation_mismatched"], logging_mode="eval_mnli-mm"))

    evaluation_sets = evaluation_sets + trainer.evaluation_sets

    for es in evaluation_sets:
        prev_metrics_func = trainer.compute_metrics
        if es.metrics_func is not None:
            trainer.compute_metrics = es.metrics_func
        metrics = trainer.evaluate(eval_dataset=es.set, metric_key_prefix=es.logging_mode, compute_speed_metrics=False)
        trainer.compute_metrics = prev_metrics_func
        trainer.log_metrics(es.logging_mode, metrics)
        trainer.save_metrics(es.logging_mode, metrics, combined=False)
        max_eval_samples = (
            data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        )
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))


def do_predict(data_args, is_regression, label_list, predict_dataset, raw_datasets, trainer, training_args):
    logger.info("*** Predict ***")
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    tasks = [data_args.task_name]
    predict_datasets = [predict_dataset]
    if data_args.task_name == "mnli":
        tasks.append("mnli-mm")
        predict_datasets.append(raw_datasets["test_mismatched"])
    for predict_dataset, task in zip(predict_datasets, tasks):
        # Removing the `label` columns because it contains -1 and Trainer won't like that.
        predict_dataset = predict_dataset.remove_columns("label")
        predictions = trainer.predict(predict_dataset, metric_key_prefix="predict").predictions
        predictions = np.squeeze(predictions) if is_regression else np.argmax(predictions, axis=1)

        output_predict_file = path.join(training_args.output_dir, f"predict_results_{task}.txt")
        if trainer.is_world_process_zero():
            with open(output_predict_file, "w") as writer:
                logger.info(f"***** Predict results {task} *****")
                writer.write("index\tprediction\n")
                for index, item in enumerate(predictions):
                    if is_regression:
                        writer.write(f"{index}\t{item:3.3f}\n")
                    else:
                        item = label_list[item]
                        writer.write(f"{index}\t{item}\n")


if __name__ == "__main__":
    main()
