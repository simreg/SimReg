import os
import time
from functools import partial
import random
from transformers import HfArgumentParser
from transformers.file_utils import ExplicitEnum


class BiasType(ExplicitEnum):
    LEXICAL = "lexical_bias"
    HYPOTHESIS = "hypothesis_only_bias"
    UNKNOWN = "unknown_bias"


def get_indices_arg(bias_type):
    evaluations_subsets = {
        BiasType.LEXICAL: [],
        BiasType.HYPOTHESIS: [
            # 'data/mnli_hypothesis_only_hard'
        ]
    }
    if bias_type == BiasType.UNKNOWN:
        evals = []
        # merge all evaluations sets and add unknown bias set
        for k in evaluations_subsets:
            evals += evaluations_subsets[k]
        # evals += ['data/tinybert_splits/5_folds_0.8_confidence/validation_matched']
        evaluations_subsets = evals
    else:
        evaluations_subsets = evaluations_subsets[bias_type]
    indices_arg = '--evaluate_on_hans' if bias_type in [BiasType.LEXICAL, BiasType.UNKNOWN] else ''
    if len(evaluations_subsets) > 0:
        indices_arg = indices_arg + f" --indices_dir  {' '.join(evaluations_subsets)}"

    if bias_type in [BiasType.UNKNOWN, BiasType.HYPOTHESIS]:
        indices_arg = f"{indices_arg} --mismatched_indices_dir data/hypothesis_bias_splits/validation_mismatched"
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


def layer_to_arg(l, model_name='deberta'):
    layer_maper = {
        'P': 'pooler' if model_name == 'deberta' else 'bert.pooler',
        'e': f'{model_name}.embeddings'
    }
    return layer_maper[l] if l in layer_maper else f"{model_name}.encoder.layer.{l}"


def layers_args(regularized_layers_list, incraese_sim: bool):
    if type(regularized_layers_list) is tuple:
        main_model_layers = regularized_layers_list[0]
        weak_model_layers  = regularized_layers_list[1]
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

    main_regularized_layers = " ".join(map(layer_to_arg, main_model_layers))
    if incraese_sim:
        weak_regularized_layers = " ".join(map(layer_to_arg, weak_model_layers))
    else:
        weak_regularized_layers = " ".join(map(partial(layer_to_arg, model_name='bert'), weak_model_layers))
    return layers_arg, main_regularized_layers, weak_regularized_layers


def only_bias_arg(reg_only_biased, bias_type: BiasType):
    bias_indices = {
        BiasType.LEXICAL: "data/lexical_bias_splits/train/confidence_based_0.65/train_biased_correct_indices.bin",
        BiasType.UNKNOWN: "data/tinybert_splits/5_folds_0.8_confidence/train/train_biased_correct_indices.bin",
        BiasType.HYPOTHESIS: "data/hypothesis_bias_splits/5_folds_0.6_confidence/train/train_biased_correct_indices.bin"
    }[bias_type]

    reg_only_biased_arg = ""
    reg_only_biased_identifier = ""
    if reg_only_biased:
        reg_only_biased_arg = f'--regularize_only_biased --bias_indices {bias_indices}'
        reg_only_biased_identifier = 'STOB_'
    return reg_only_biased_arg, reg_only_biased_identifier


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
    bias_type = BiasType.LEXICAL
    enforce_sim = args.enforce_sim

    # if regularize_only_grads:
    #     node_name = "nlp-a40-1"  # galileo2, newton2
    # else:
    #     node_name = "galileo2"  # plato1, plato2
    node_name = "nlp-a40-1"
    debertav3 = True
    if debertav3:
        unbiased_models = {
            BiasType.LEXICAL: ('runs/debertav3/extrensic_debiasing/lexical_bias/debertav3_64_lexical_extrensic_debiasing/checkpoint-11678', 'EDXE'),
            BiasType.UNKNOWN: ('runs/debertav3/extrensic_debiasing/unknown_bias/debertav3_64_unknown_extrensic_debiasing/checkpoint-9020', 'EDXE'),
            BiasType.HYPOTHESIS: ('runs/debertav3/extrensic_debiasing/hypohtesis_only_bias/debertav3_64_hypothesis_extrensic_debiasing/checkpoint-7044', 'EDXE')
        }
    else:
        unbiased_models = {
            BiasType.LEXICAL: ('runs/deberta/extrensic_debiasing/lexical_bias/deberta_64_lexical_extrensic_debiasing/checkpoint-11678', 'EDXE'),
            BiasType.UNKNOWN: ('runs/deberta/extrensic_debiasing/unknown_bias/deberta_64_unknown_extrensic_debiasing/checkpoint-9020', 'EDXE'),
            BiasType.HYPOTHESIS: ('runs/deberta/extrensic_debiasing/hypothesis_bias/deberta_64_hypothesis_extrensic_debiasing/checkpoint-7044', 'EDXE')
        }
    biased_models = {
        BiasType.LEXICAL: ('runs/bert_lexical_bias', 'LexicalModel'),
        BiasType.HYPOTHESIS: ([f'runs/5_fold_hypothesis_only/bert_{i}_hypothesis_only/checkpoint-14724' for i in range(4)], 'HOBert'),
        BiasType.UNKNOWN: ([f'runs/5_folds_unknown_bias/tinybert_{i}' for i in range(5)], 'TB')
    }
    # BiasType.LEXICAL: ('runs/deberta/extrensic_debiasing/lexical_bias/deberta_64_lexical_extrensic_debiasing', 'ED'),


    # BiasType.LEXICAL: ('runs/bert_lexical_bias', 'LexicalModel'),
    # BiasType.LEXICAL: ('runs/deberta/UBD/lexical_bias/UB_deberta', 'UBD'),

    batch_size = 64
    epochs = 10
    eval_bs, grads_arguments, grads_suffix = grads_arg(node_name, regularize_grads=regularize_only_grads)

    enforce_sim_arg, fw_name, sim_name_arg, weak_model_dir = sim_args(bias_type, biased_models, enforce_sim, unbiased_models)

    run_name_identifier = {
        BiasType.LEXICAL: 'lexicalDeBias',
        BiasType.HYPOTHESIS: 'HypoDeBias',
        BiasType.UNKNOWN: 'UNB'
    }[bias_type]

    main_task_lambda = 1
    indices_arg = get_indices_arg(bias_type)
    tags_arg = f'--tags {bias_type.value}'
    bias_dir = {
        BiasType.LEXICAL: 'lexical_bias_debiasing',
        BiasType.HYPOTHESIS: 'hypothesis_bias_debiasing',
        BiasType.UNKNOWN: 'unknown_bias/0.8_confidence'
    }[bias_type]
    bias_dir = bias_dir + ("/similarity" if enforce_sim else "/dissimilarity")

    reg_only_biased_arg, reg_only_biased_identifier = only_bias_arg(reg_only_biased=True, bias_type=bias_type)
    model_arg = 'microsoft/deberta-base' if not debertav3 else 'microsoft/deberta-v3-base'
    base_dir = 'deberta' if not debertav3 else 'debertav3'
    counter = 0
    for regularization_lambda in [100]:
        for regularization_method in ["linear_cka"]:
            # (['e', 10, 11], ['e', 1, 1]), ['e', 0, 6, 10, 11], ['e', 0, 6, 10, 11, 'P']
            for l in [[9, 10, 11]]:
                layers_arg, main_regularized_layers, weak_regularized_layers = layers_args(regularized_layers_list=l, incraese_sim=enforce_sim)
                reg_tokens_arg = "all" if 'P' not in l else " ".join(map(lambda x: "none" if x == 'P' else "all", l))
                model_name = f"{base_dir}_{fw_name}_{reg_only_biased_identifier}{run_name_identifier}_{regularization_method}_{layers_arg}_lambda_" \
                             f"{main_task_lambda}_{regularization_lambda}_{batch_size}{sim_name_arg}{grads_suffix}"
                seeds = [random.randint(0, 10000), random.randint(0, 1000)]
                for seed in seeds:
                    if type(weak_model_dir) is list:
                        current_seed_weak_model = weak_model_dir[random.randint(0, len(weak_model_dir) - 1)]
                    else:
                        current_seed_weak_model = weak_model_dir
                    output_dir = f"runs/{base_dir}/{bias_dir}/{regularization_method}/{model_name}_s{seed}"
                    tokenization_arg = '--separate_weak_tokenization' if bias_type in [BiasType.HYPOTHESIS, BiasType.UNKNOWN] and not enforce_sim else ''
                    padding_arg = '' #'--no_pad_to_max_length' if enforce_sim else ''
                    os.system(f"""
                        echo nlp_sbatch {node_name} run_glue.py --report_to wandb --model_name_or_path {model_arg} \
                            --output_dir {output_dir} --num_train_epochs {epochs} --learning_rate 2e-5 --per_device_eval_batch_size {eval_bs} \
                             --task_name mnli --do_train --do_eval --max_seq_length 128 --per_device_train_batch_size {batch_size} \
                             --weak_model_layer {weak_regularized_layers} --weak_model_path {current_seed_weak_model} --regularized_layers {main_regularized_layers} \
                             --regularization_method {regularization_method} --evaluation_strategy half_epoch --save_strategy no \
                             --regularization_lambda {regularization_lambda} --run_name {model_name} --weight_decay 0.1 --warmup_ratio 0.1 \
                             --logging_first_step --regularized_tokens {reg_tokens_arg} --token_aggregation_strategy mean {enforce_sim_arg} \
                             {grads_arguments} {indices_arg} {tags_arg} {reg_only_biased_arg} --main_task_lambda {main_task_lambda} \
                             --bias_sampling_strategy stochastic --seed {seed} --wandb_group {tokenization_arg} {padding_arg}
                    """)
                    print()
                    counter += 1

    print(f"dispatched {counter} jobs")


if __name__ == '__main__':
    main()
