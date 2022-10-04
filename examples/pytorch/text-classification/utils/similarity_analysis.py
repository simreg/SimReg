# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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
""" Finetuning a 🤗 Transformers model for sequence classification on GLUE."""
import argparse
import json
import logging
import os
from typing import List

import datasets
import numpy as np
import torch
from torch.nn import Linear
from torch.utils.data import DataLoader
from datasets import load_dataset, load_metric, DatasetDict
from tqdm.auto import tqdm

import transformers
from accelerate import Accelerator
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    set_seed,
    HfArgumentParser,
    DataCollatorWithPadding,
)
from transformers.utils.versions import require_version
import matplotlib.pyplot as plt
import seaborn as sns
from arguments_classes import DataTrainingArguments
from misc import is_partial_input_model, lexical_bias_cache_mapper, DATA_DIR
from models import PartialInputBert, BertWithLexicalBiasModel
from models.lexical_bias_bert import ClarkLexicalBiasModel
from run_glue import task_to_keys, load_datasets, load_labels
from my_trainer import LinCKA, SimRegTrainer, AggregationStrategy, CorSimReg, get_activation, get_model_signature

sns.set_theme()

logger = logging.getLogger(__name__)

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

# Similarity measures
sim_measures = {
    # 'linear_cka': LinCKA,
    'abs_cos_cor': CorSimReg
}


def parse_args():
    parser = HfArgumentParser((DataTrainingArguments,), description="Finetune a transformers model on a text classification task")

    parser.add_argument(
        "--per_device_batch_size",
        type=int,
        default=64,
        help="Batch size (per device) for the dataloader.",
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the results.")
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible run.")

    parser.add_argument("--weak_model_path", required=True, type=str, help="Path of the weak model")
    parser.add_argument("--weak_name", default=None, type=str, help="name of the weak model in the plots")
    parser.add_argument("--main_model_path", required=True, type=str, help="Path of the main model")
    parser.add_argument("--main_name", default=None, type=str, help="name of the main model in the plots")
    parser.add_argument('--randomly_init_models', action='store_true', help='re-initialize weights of models randomly')

    parser.add_argument("--use_saved_activations", action='store_true',
                        help="Use saved activations of the model (should be saved in activations dir under model dir)")
    parser.add_argument("--do_train", action='store_true', help="Whether to run on training dataset.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run on evaluation dataset.")

    parser.add_argument('--regularized_tokens', type=str, default='all', help='Compute similarity on CLS/all tokens --valid values: CLS/all')
    parser.add_argument('--aggregation_strategy', type=str, default='mean', help='analysis on sum/seperation/mean of tokens')
    parser.add_argument('--grads_sim', action='store_true', help='Whether to compute similarity on gradients or activations.')

    parser.add_argument('--evaluate_on_hans', action='store_true')

    parser.add_argument('--bias_indices', action='append', help='list of path to biased subsets (the indices will be merged to one list)')

    args = parser.parse_args()

    # Sanity checks
    if args.task_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a task name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."

    if args.regularized_tokens not in ('CLS', 'all'):
        raise Exception('Invalid regularized_tokens argument')

    # hack to reuse more code (to use load_datasets safely)
    args.cache_dir = None

    return args


def main():
    args = parse_args()
    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.

    accelerator = Accelerator()
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    raw_datasets = load_datasets(data_args=args, model_args=args, training_args=args)

    _, _, num_labels = load_labels(args, raw_datasets)

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    weak_tokenizer = AutoTokenizer.from_pretrained(args.weak_model_path)
    main_tokenizer = AutoTokenizer.from_pretrained(args.main_model_path)
    weak_config = AutoConfig.from_pretrained(args.weak_model_path)
    main_config = AutoConfig.from_pretrained(args.main_model_path)
    weak_config.num_labels = num_labels
    main_config.num_labels = num_labels
    main_model = AutoModelForSequenceClassification.from_pretrained(args.main_model_path, config=main_config)

    lexical_bias_settings = False
    main_signature = None
    if is_partial_input_model(weak_config):
        model_cls = PartialInputBert
    elif weak_config.architectures is not None and ('BertWithLexicalBiasModel' in weak_config.architectures or 'ClarkLexicalBiasModel' in weak_config.architectures):
        if args.task_name == 'qqp':
            model_cls = ClarkLexicalBiasModel
        else:
            model_cls = BertWithLexicalBiasModel
        lexical_bias_settings = True
        main_signature = set(get_model_signature(main_model))
    else:
        model_cls = AutoModelForSequenceClassification

    weak_model = model_cls.from_pretrained(args.weak_model_path, config=weak_config)


    if args.randomly_init_models:
        assert not lexical_bias_settings
        main_model.init_weights()
        weak_model.init_weights()

    # Preprocessing the datasets
    if args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[args.task_name]
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

    if args.regularized_tokens == 'CLS' or args.aggregation_strategy != 'seperate':
        # dropping pwcca, because it requires size of batch > num of features, which is not ensured in CLS/aggregation
        if 'pwcca' in sim_measures:
            sim_measures.pop('pwcca')

    bias_indices = None
    if args.do_train:
        ds_name = 'train'
    elif args.do_eval:
        ds_name = 'validation_matched' if args.task_name == 'mnli' else 'validation'
    elif args.evaluate_on_hans:
        ds_name = 'hans'
    else:
        raise Exception('Please select dataset (eval/train)')

    # not so elegant, but I don't have time.
    raw_datasets = DatasetDict({ds_name: raw_datasets[ds_name]})

    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (
                examples[sentence1_key], examples[sentence2_key])
        )
        result = weak_tokenizer(*args, padding=False, max_length=128, truncation=True)
        return result
    if not lexical_bias_settings:
        if args.synthetic_bias_prevalence > 0:
            bias_indices = {}
            data_cache_dir = f"ds_cache/mnli_{args.synthetic_bias_prevalence}_{args.bias_correlation_prob}_bias"
            if not os.path.isdir(data_cache_dir):
                raise Exception(f"Dataset not found! prevalence: {args.synthetic_bias_prevalence}, correlation: {args.bias_correlation_prob}")
            raw_datasets = raw_datasets.map(
                lambda x: 0 / 0,  # should not call this func, raising error incase called.
                batched=True,
                load_from_cache_file=True,
                cache_file_names={k: os.path.join(data_cache_dir, k) for k in raw_datasets},
                desc="loading DS from cache",
            )
            for k in raw_datasets.keys():
                bias_indices[k] = torch.load(os.path.join(data_cache_dir, f"{k}_biased_indices"))
        else:
            raw_datasets = raw_datasets.map(
                preprocess_function,
                batched=True,
                load_from_cache_file=True,
                desc="Tokenizing dataset",
            )
        unused_columns = {
            'qqp': ['question1', 'question2'],
            'mnli': ['premise', 'hypothesis'],
            'fever': ['claim', 'evidence']
        }

        # raw_datasets
        raw_datasets = raw_datasets.remove_columns(unused_columns[args.task_name])
    else:
        # Load cached pre-processed dataset
        # pre-processing is done earlier for lexical features.
        if args.task_name == 'qqp':
            qqp_lex = datasets.load_from_disk(os.path.join(DATA_DIR, 'QQP_clark_lex'))
            from datasets import concatenate_datasets
            qqp_lex = qqp_lex.remove_columns('label')
            for split in raw_datasets.keys():
                if split in qqp_lex:
                    raw_datasets[split] = concatenate_datasets([raw_datasets[split], qqp_lex[split]], axis=1)
            raw_datasets = raw_datasets.map(
                preprocess_function,
                batched=True,
                desc="Tokenizing dataset",
            )
            raw_datasets = raw_datasets.remove_columns(['question1', 'question2'])
        else:
            raw_datasets = raw_datasets.map(None, cache_file_names=lexical_bias_cache_mapper[args.task_name])

    # Prepare everything with our `accelerator`.
    # , train_dataloader, eval_dataloader
    weak_model, main_model = accelerator.prepare(weak_model, main_model)
    weak_model.eval()
    main_model.eval()

    # Note -> the training dataloader needs to be prepared before we grab his length below (cause its length will be
    # shorter in multiprocess)

    # Train!
    logger.info(f"  Instantaneous batch size per device = {args.per_device_batch_size}")
    bs = args.per_device_batch_size
    sim_hist = dict()

    train_ds = raw_datasets[ds_name]
    if bias_indices is not None:
        bias_indices = bias_indices[ds_name]
        raise NotImplementedError('Not fully supported, needs different implementation for anti-biased vs unbiased')

    if args.bias_indices is not None:
        biased_indices = []
        for indices_collection_path in args.bias_indices:
            biased_indices.append(torch.load(indices_collection_path).squeeze())
        bias_indices = torch.cat(biased_indices).unique()

    ds_len, subset_indices, train_ds = get_ds(args, train_ds, subset_indices=bias_indices)
    dl = DataLoader(train_ds, bs, shuffle=False, collate_fn=DataCollatorWithPadding(weak_tokenizer))

    main_num_layers = getattr(main_model.config, 'num_hidden_layers') + 1  # embeddings
    weak_num_layers = getattr(weak_model.config, 'num_hidden_layers') + 1
    if lexical_bias_settings:
        weak_num_layers = 1
        if args.task_name == 'qqp':
            weak_num_layers = 4
            lexical_weak_activations = [None] * weak_num_layers
            for i in range(weak_num_layers):
                weak_model.classifier[i * 2].register_forward_hook(get_activation(lexical_weak_activations, i))
        else:
            for i in range(len(weak_model.classifier.classifier)):
                if type(weak_model.classifier.classifier[i]) is Linear:
                    weak_num_layers += 1
            lexical_weak_activations = [None] * weak_num_layers
            # save classifier's input
            weak_model.classifier.classifier_input.register_forward_hook(get_activation(lexical_weak_activations, 0))
            j = 1
            for i in range(len(weak_model.classifier.classifier)):
                if type(weak_model.classifier.classifier[i]) is Linear:
                    weak_model.classifier.classifier[i].register_forward_hook(get_activation(lexical_weak_activations, j))
                    j += 1
            del j

    # weak_models_activation = [None] * weak_num_layers
    # main_models_activation = [None] * main_num_layers

    # from my_trainer import get_activation
    # for i in range(main_num_layers):
    #     main_model.get_submodule(f"bert.encoder.layer.{i}").register_forward_hook(get_activation(main_models_activation, i))
    # for i in range(weak_num_layers):
    #     weak_model.get_submodule(f"bert.encoder.layer.{i}").register_forward_hook(get_activation(weak_models_activation, i))

    if args.aggregation_strategy == 'mean':
        aggregation_strategy = AggregationStrategy.MEAN
    if args.aggregation_strategy == 'sum':
        aggregation_strategy = AggregationStrategy.SUM
    if args.aggregation_strategy == 'seperate':
        aggregation_strategy = AggregationStrategy.NO_AGGREGATION

    if args.select_only_biased_samples:
        sim_key = 'biased'
    elif args.remove_biased_samples_from_train:
        sim_key = 'unbiased'
    else:
        sim_key = 'overall'

    if os.path.isfile(os.path.join(args.output_dir, "sim_hist.bin")):
        print('found sim_hist, using it!')
        exit(0)
        sim_hist = torch.load(os.path.join(args.output_dir, "sim_hist.bin"))
    else:
        # if args.use_saved_activations:
        #     compute_similarity_cached_activations(args, bs, ds_len, ds_name, weak_model, main_model, sim_hist, subset_indices)
        # else:
        with torch.no_grad():
            # range(0, ds_len, bs)
            for batch in tqdm(dl):
                batch_labels = batch.pop('labels')
                # [1] returns the indices of intersection elements in `batch` tensor.
                # not elegant work around to removing unused columns
                if 'idx' in batch:
                    del batch['idx']
                if 'id' in batch:
                    del batch['id']
                weak_encodings = batch
                main_encodings = batch

                if torch.cuda.is_available():
                    main_encodings = main_encodings.to('cuda')
                    weak_encodings = weak_encodings.to('cuda')
                    batch_labels = batch_labels.to('cuda')

                # TODO: this code works only for BertModels (and models using BertTokenizer)
                grads_gate = torch.enable_grad if args.grads_sim else torch.no_grad
                with grads_gate():
                    if lexical_bias_settings:
                        out1 = weak_model(**weak_encodings, labels=batch_labels)
                        weak_hidden_states = lexical_weak_activations
                        main_inputs = {k: main_encodings[k] for k in main_signature.intersection(main_encodings)}
                        out2 = main_model(**main_inputs, labels=batch_labels, output_hidden_states=True)
                        main_hidden_states = out2.hidden_states
                    else:
                        out1 = weak_model(**weak_encodings, labels=batch_labels, output_hidden_states=True)
                        weak_hidden_states = out1.hidden_states
                        out2 = main_model(**main_encodings, labels=batch_labels, output_hidden_states=True)
                        main_hidden_states = out2.hidden_states

                    if args.grads_sim:
                        main_hidden_states = torch.autograd.grad(
                            out2.loss,
                            inputs=main_hidden_states
                        )
                        weak_hidden_states = torch.autograd.grad(
                            out1.loss,
                            inputs=weak_hidden_states
                        )
                    del out1
                    del out2
                for i in range(main_num_layers):
                    main_activations, _ = SimRegTrainer.extract_relevant_activations(
                        main_encodings['attention_mask'],
                        main_hidden_states[i].clone(),
                        regularized_tokens=args.regularized_tokens,
                        aggregation_strategy=aggregation_strategy
                    )
                    for j in range(weak_num_layers):
                        if not lexical_bias_settings:
                            weak_activations, _ = SimRegTrainer.extract_relevant_activations(
                                weak_encodings['attention_mask'],
                                weak_hidden_states[j].clone(),
                                regularized_tokens=args.regularized_tokens,
                                aggregation_strategy=aggregation_strategy
                            )
                        else:
                            weak_activations = lexical_weak_activations[j].clone()
                        add_entries(main_activations, weak_activations, i, j, sim_hist, key=sim_key)
                del weak_activations
                del main_activations
        accelerator.wait_for_everyone()
        # torch.save(sim_hist, os.path.join(args.output_dir, "sim_hist.bin"))

    heatmap_scale = {
        'linear_cka': {'vmax': 1, 'vmin': 0},
        'abs_cos_cor': {'vmax': 1, 'vmin': 0},
        'cos_cor': {'vmax': 1, 'vmin': -1},
        'cosine': {'vmax': 0, 'vmin': -1},
        'opd': {'vmax': 1, 'vmin': 0},
        'pwcca': {'vmax': 1, 'vmin': 0}
    }
    grads_title_arg = '$\\nabla$' if args.grads_sim else ''
    if args.main_name == 'reg':
        file_name='sim_hypothesis_regularized_vs_hypothesis_fg.pdf'
        args.main_name = '$SimReg\\uparrow$'
    else:
        file_name='sim_bert_base_vs_hypothesis_fg.pdf'
        args.main_name = 'BERT-base'

    args.weak_name = '$f_g$'
    x_title = args.weak_name if args.weak_name is not None else weak_model.config._name_or_path
    y_title = args.main_name if args.main_name is not None else main_model.config._name_or_path

    for reg_method in sim_measures.keys():
        avg_mats = dict()
        for i in range(main_num_layers):
            for j in range(weak_num_layers):
                current_key = f"{i}-{j}"
                for combination_k in sim_hist[current_key][reg_method].keys():
                    if combination_k not in avg_mats:
                        avg_mats[combination_k] = torch.zeros((main_num_layers, weak_num_layers))
                    avg_mat = avg_mats[combination_k]
                    avg_mat[i, j] = torch.mean(torch.tensor(sim_hist[current_key][reg_method][combination_k]))
        torch.save(avg_mat, os.path.join(args.output_dir, f"sim_results_{reg_method}.bin"))
        for combination_k in avg_mats.keys():
            plt.rcParams.update({
                'figure.dpi': 200,
                'legend.fontsize': 24,
                'axes.labelsize': 24,
                'xtick.labelsize': 24,
                'ytick.labelsize': 24,
            })
            # plt.clf()
            f, ax = plt.subplots(figsize=(20, 10))
            v_limits = {}
            if reg_method in heatmap_scale:
                v_limits['vmax'] = heatmap_scale[reg_method]['vmax']
                v_limits['vmin'] = heatmap_scale[reg_method]['vmin']
            sns.set(font_scale=1.3)
            sns.heatmap(avg_mats[combination_k], annot=True, ax=ax, **v_limits)

            # ax.set_title(f'{grads_title_arg} {reg_method} - {combination_k}')
            # ax.set_title(f'CosCor - validation matched')

            ax.set_xlabel(x_title)
            ax.set_ylabel(y_title)
            plt.tight_layout(pad=0.0)
            # os.makedirs(os.path.join(args.output_dir, reg_method), exist_ok=True)
            # f.savefig(os.path.join(args.output_dir, f"{ds_name}_{combination_k}_{reg_method}_mat.pdf"))
            f.savefig(os.path.join(args.output_dir, file_name))
    json.dump(args.__dict__, open(os.path.join(args.output_dir, 'run_args.json'), 'a'))

#
# def compute_similarity_cached_activations(args, bs, ds_len, ds_name, weak_model, main_model, sim_hist, subset_indices):
#     print("Not fully supported")
#     exit(1)
#     print('Using saved activations')
#     activations1_path = os.path.join(args.weak_model_path, 'activations')
#     activations2_path = os.path.join(args.main_model_path, 'activations')
#     for i in range(1, weak_model.config.num_hidden_layers + 1):
#         layer_data_path = f'{ds_name}_{i}_CLS_activations.bin'
#
#         weak_model_activations = torch.load(os.path.join(activations1_path, layer_data_path),
#                                             map_location=torch.device('cpu') if not torch.cuda.is_available() else None)
#         if subset_indices is not None:
#             weak_model_activations = weak_model_activations[subset_indices]
#
#         for j in range(1, main_model.config.num_hidden_layers + 1):
#             layer_data_path = f'{ds_name}_{j}_CLS_activations.bin'
#             main_model_activations = torch.load(os.path.join(activations2_path, layer_data_path),
#                                                 map_location=torch.device('cpu') if not torch.cuda.is_available() else None)
#             if subset_indices is not None:
#                 main_model_activations = main_model_activations[subset_indices]
#             for start_i in tqdm(range(0, ds_len, bs)):
#                 z, h = weak_model_activations[start_i: min(start_i + bs, ds_len)], \
#                        main_model_activations[start_i: min(start_i + bs, ds_len)]
#                 add_entries(z, h, i, j, sim_hist)


def add_entries(z, h, i, j, sim_hist, bias_in_batch=None, key='overall'):
    current_key = f"{i}-{j}"
    if current_key not in sim_hist:
        sim_hist[current_key] = {k: dict() for k in sim_measures.keys()}
    for k, v in sim_measures.items():
        kwargs = {}
        if k == 'abs_cos_cor':
            kwargs = {'abs_cor': True}
        if bias_in_batch is not None:
            sim_res = v.sim_measure_combinations(z=z, h=h, bias_indices=bias_in_batch)
            for combination_k, combination_v in sim_res.items():
                if combination_k not in sim_hist[current_key][k]:
                    sim_hist[current_key][k][combination_k] = []
                sim_hist[current_key][k][combination_k].append(combination_v.item())
            # sim_hist[current_key][k].append(sim_res['overall'].item())
            # print(f'{i}-{j}-{v}: {sim_res}')
            # 1. sim betweeen biased an rest
            # 2. sim between biased and biased
            # 3. sim between rest and rest
            # 4. overall sim
        else:
            if key not in sim_hist[current_key][k]:
                sim_hist[current_key][k][key] = []
            sim_hist[current_key][k][key].append(v.sim_measure(z, h, **kwargs).item())


def get_ds(args, train_ds, subset_indices=None):
    if subset_indices is not None:
        assert args.remove_biased_samples_from_train or args.select_only_biased_samples
        if args.remove_biased_samples_from_train:
            train_ds = train_ds.select(np.setdiff1d(np.arange(len(train_ds)), subset_indices))
            logger.info(f"removing biased samples, Train len: {len(train_ds)}")
        elif args.select_only_biased_samples:
            train_ds = train_ds.select(subset_indices)
            logger.info(f"Restricting train to biased samples provided by bias_indices argument, Train len: {len(train_ds)}")
    return len(train_ds), subset_indices, train_ds


if __name__ == "__main__":
    main()
