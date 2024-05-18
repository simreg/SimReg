import os
import random
import time
from functools import partial

from transformers import HfArgumentParser
from transformers.file_utils import ExplicitEnum


class BiasType(ExplicitEnum):
    LEXICAL = "lexical_bias"
    UNKNOWN = "unknown_bias"


def get_indices_arg(bias_type):
    evaluations_subsets = {
        BiasType.LEXICAL: [
            # "data/qqp_lexical_bias/0.7_confidence/validation",
            # "data/qqp_unknown_bias_splits/5_folds_tinybert_0.9_confidence_splits/validation",
        ]
    }
    if bias_type == BiasType.UNKNOWN:
        evals = []
        # merge all evaluations sets and add unknown bias set
        for k in evaluations_subsets:
            evals += evaluations_subsets[k]
        evaluations_subsets = evals
    else:
        evaluations_subsets = evaluations_subsets[bias_type]
    if len(evaluations_subsets) > 0:
        indices_arg = f"--indices_dir  {' '.join(evaluations_subsets)}"
    else:
        indices_arg = ''
    return indices_arg


def sim_args(bias_type, biased_models, enforce_sim, unbiased_models):
    if enforce_sim:
        enforce_sim_arg = "--enforce_similarity"
        sim_name_arg = "_sim"
        weak_model_dir, fw_name = unbiased_models[bias_type]
    else:
        sim_name_arg = ""
        enforce_sim_arg = ""
        weak_model_dir, fw_name = biased_models[bias_type]
    return enforce_sim_arg, fw_name, sim_name_arg, weak_model_dir

def layer_to_arg(l, model_name):
    layer_maper = {
        'P': 'pooler' if model_name == 'deberta' else 'bert.pooler',
        'e': f'{model_name}.embeddings'
    }
    return layer_maper[l] if l in layer_maper else f"{model_name}.encoder.layer.{l}"


def layers_args(regularized_layers_list, enforce_sim, model_name):
    if type(regularized_layers_list) is tuple:
        main_model_layers = regularized_layers_list[0]
        weak_model_layers = regularized_layers_list[1]
    else:
        main_model_layers = regularized_layers_list
        weak_model_layers = regularized_layers_list

    if type(main_model_layers) is not list:
        main_model_layers = [main_model_layers]
    if type(weak_model_layers) is not list:
        weak_model_layers = [weak_model_layers]

    weak_model_layers = list(map(str, weak_model_layers))
    main_model_layers = list(map(str, main_model_layers))

    layers_suffix = "-".join(weak_model_layers)
    layers_prefix = "-".join(map(lambda x: f"{x}M", main_model_layers))
    layers_arg = f"{layers_prefix}_{layers_suffix}"

    main_regularized_layers = " ".join(map(partial(layer_to_arg, model_name=model_name), main_model_layers))
    weak_regularized_layers = " ".join(map(partial(layer_to_arg, model_name=model_name if enforce_sim else 'bert'), weak_model_layers))
    return layers_arg, main_regularized_layers, weak_regularized_layers


def only_bias_arg(reg_only_biased, bias_type: BiasType):
    bias_dir = {
        BiasType.LEXICAL: 'lexical_bias',
        BiasType.UNKNOWN: 'unknown_bias'
    }[bias_type]

    bias_indices = {
        BiasType.LEXICAL: "data/qqp_lexical_bias/0.6_confidence/train/train_biased_correct_indices.bin",
        BiasType.UNKNOWN: "data/qqp_unknown_bias_splits/5_folds_tinybert_0.8_confidence_splits/train/train_biased_correct_indices.bin",
    }[bias_type]

    reg_only_biased_arg = ""
    reg_only_biased_identifier = ""
    if reg_only_biased:
        reg_only_biased_arg = f'--regularize_only_biased --bias_indices {bias_indices}'
        reg_only_biased_identifier = 'STOB_'
        bias_dir = {
            BiasType.LEXICAL: 'lexical_bias/reg_OB/0.6_confidence',
            BiasType.UNKNOWN: 'unknown_bias/reg_OB/0.8_confidence'
        }[bias_type]
    return bias_dir, reg_only_biased_arg, reg_only_biased_identifier


def grads_arg(node_name, regularize_grads):
    grads_suffix = "_GReg" if regularize_grads else ""
    grads_arguments = "--regularize_grads" if regularize_grads else ""
    eval_bs = 64 if regularize_grads or node_name in ['plato1', 'plato2', 'nlp-2080-1', 'nlp-2080-2'] else 256
    return eval_bs, grads_arguments, grads_suffix


def main():
    parser = HfArgumentParser([])

    parser.add_argument('--regularize_only_grads', action='store_true', help='re-initialize weights of models randomly')
    parser.add_argument('--enforce_sim', action='store_true', help='re-initialize weights of models randomly')
    args = parser.parse_args()
    # main configs:
    regularize_only_grads = args.regularize_only_grads
    bias_type = BiasType.UNKNOWN
    enforce_sim = args.enforce_sim
    regularize_only_biased = True
    debertav3 = False
    node_name = "nlp-a40-1"  # nlp-a40-1, galileo2, galileo1
    if debertav3:
        unbiased_models = {
            BiasType.LEXICAL: ([
                'qqp_runs/debertav3/extrensic_debiasing/lexical_bias/debertav3_0.6_lexical_extrensic_debiasing_s401',
                'qqp_runs/debertav3/extrensic_debiasing/lexical_bias/debertav3_0.6_lexical_extrensic_debiasing_s420',
                'qqp_runs/debertav3/extrensic_debiasing/lexical_bias/debertav3_0.6_lexical_extrensic_debiasing_s638',
            ], 'EXDBXE'),
            BiasType.UNKNOWN: ('qqp_runs/debertav3/extrensic_debiasing/unknown_bias/debertav3_64_0.8_unknown_extrensic_debiasing/checkpoint-5248', 'EDBXE')
        }
    else:
        # "qqp_runs/extrensic_debiasing/lexical_bias/bert_64_0.6_lexical_extrensic_debiasing/checkpoint-6800"
        qqp_lexical_models = [
            'qqp_runs/extrensic_debiasing/lexical_bias/bert_64_0.8_s1052/checkpoint-12819',
            'qqp_runs/extrensic_debiasing/lexical_bias/bert_64_0.8_s95/checkpoint-12819',
            'qqp_runs/extrensic_debiasing/lexical_bias/bert_64_0.8_s682/checkpoint-12819',
        ]
        unbiased_models = {
            BiasType.LEXICAL: (qqp_lexical_models, 'EDBXE_0.8'),
            BiasType.UNKNOWN: ('qqp_runs/extrensic_debiasing/unknown_bias/bert_64_0.8_unknown_extrensic_debiasing/checkpoint-7872', 'EDBXE')
        }
    biased_models = {
        BiasType.LEXICAL: ('qqp_runs/lexicalModel', 'LexicalModel'),
        BiasType.UNKNOWN: ([
           'qqp_runs/5_folds_unknown_bias/tinybert_0',
           'qqp_runs/5_folds_unknown_bias/tinybert_1',
           'qqp_runs/5_folds_unknown_bias/tinybert_2',
           'qqp_runs/5_folds_unknown_bias/tinybert_3',
           'qqp_runs/5_folds_unknown_bias/tinybert_4'
        ], 'TB'),
    }
    batch_size = 32
    epochs = 3
    eval_bs, grads_arguments, grads_suffix = grads_arg(node_name, regularize_grads=regularize_only_grads)

    enforce_sim_arg, fw_name, sim_name_arg, weak_model_dir = sim_args(bias_type, biased_models, enforce_sim, unbiased_models)

    run_name_identifier = {
        BiasType.LEXICAL: 'lexicalDeBias',
        BiasType.UNKNOWN: 'UNB'
    }[bias_type]

    main_task_lambda = 1
    indices_arg = get_indices_arg(bias_type)
    tags_arg = f'--tags {bias_type.value}'
    bias_dir, reg_only_biased_arg, reg_only_biased_identifier = only_bias_arg(reg_only_biased=regularize_only_biased, bias_type=bias_type)
    model_name = 'debertav3' if debertav3 else 'bert'
    model_arg = 'microsoft/deberta-v3-base' if debertav3 else 'bert-base-uncased'
    tokenization_arg = '--separate_weak_tokenization' if bias_type in [BiasType.UNKNOWN] and not enforce_sim else ''
    deberta_dir = 'debertav3/' if debertav3 else ''
    counter = 0
    for regularization_lambda in [10]:
        for regularization_method in ["abs_cos_cor"]:
            # (['e', 10, 11], ['e', 1, 1]), ['e', 0, 6, 10, 11], [9, 10, 11], ([6, 10, 11], [0, 1, 1])
            for l in [(['e'], [2])]:
                layers_arg, main_regularized_layers, weak_regularized_layers = layers_args(regularized_layers_list=l, enforce_sim=enforce_sim, model_name='deberta' if debertav3 else 'bert')
                model_name = f"{model_name}_{fw_name}_{reg_only_biased_identifier}{run_name_identifier}_{regularization_method}_{layers_arg}_lambda_" \
                             f"{main_task_lambda}_{regularization_lambda}_{batch_size}{sim_name_arg}{grads_suffix}"

                remove_unused_cols_arg = '--no_remove_unused_columns' if (not enforce_sim) and bias_type == BiasType.LEXICAL else ''
                padding_arg = '--no_pad_to_max_length' if enforce_sim else ''
                num_seeds = 3

                for i in range(num_seeds):
                    seed = random.randint(0, 10000)
                    output_dir = f"qqp_runs/{deberta_dir}{bias_dir}/{regularization_method}/{model_name}_s{seed}"
                    if type(weak_model_dir) is list:
                        current_seed_weak_model = weak_model_dir[i % len(weak_model_dir)]
                    else:
                        # --learning_rate 2e-5
                        current_seed_weak_model = weak_model_dir
                    os.system(f"""
                        echo nlp_sbatch {node_name} run_glue.py --report_to wandb --model_name_or_path {model_arg} \
                            --output_dir {output_dir} --num_train_epochs {epochs}  --per_device_eval_batch_size {eval_bs} \
                             --task_name qqp --do_train --max_seq_length 128 --per_device_train_batch_size {batch_size} \
                             --weak_model_layer {weak_regularized_layers} --weak_model_path {current_seed_weak_model} --regularized_layers {main_regularized_layers} \
                             --regularization_method {regularization_method} --evaluation_strategy epoch --save_strategy no \
                             --regularization_lambda {regularization_lambda} --run_name {model_name} --weight_decay 0.1 --warmup_ratio 0.1 \
                             --logging_first_step --regularized_tokens all --token_aggregation_strategy mean {enforce_sim_arg} \
                             {grads_arguments} {indices_arg} {tags_arg} {reg_only_biased_arg} --main_task_lambda {main_task_lambda} \
                             --bias_sampling_strategy stochastic --seed {seed} --wandb_group {remove_unused_cols_arg} {tokenization_arg} {padding_arg}
                    """)

                    print()
                    counter += 1
    print(f"dispatched {counter} jobs")


if __name__ == '__main__':
    main()
