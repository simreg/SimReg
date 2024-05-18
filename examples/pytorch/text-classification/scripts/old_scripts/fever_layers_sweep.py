import os
import random
import time
from functools import partial

from transformers import HfArgumentParser
from transformers.file_utils import ExplicitEnum


class BiasType(ExplicitEnum):
    CLAIM = "claim_bias"
    UNKNOWN = "unknown_bias"


def get_indices_arg(bias_type):
    evaluations_subsets = {
        BiasType.CLAIM: [
            'data/fever_claim_bias_splits/validation_0.8_confidence'
        ]
    }
    if bias_type == BiasType.UNKNOWN:
        evals = []
        # merge all evaluations sets and add unknown bias set
        for k in evaluations_subsets:
            evals += evaluations_subsets[k]
        evals.append('data/fever_unknown_bias_splits/validation')
        evaluations_subsets = evals
    else:
        evaluations_subsets = evaluations_subsets[bias_type]
    indices_arg = f"--indices_dir  {' '.join(evaluations_subsets)}"
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


def layer_to_arg(l, model_name='bert'):
    layer_maper = {
        'P': 'pooler' if model_name == 'deberta' else 'bert.pooler',
        'e': f'{model_name}.embeddings'
    }
    return layer_maper[l] if l in layer_maper else f"{model_name}.encoder.layer.{l}"

def layers_args(regularized_layers_list, model_name_arg):
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
    main_regularized_layers = " ".join(map(partial(layer_to_arg, model_name=model_name_arg), main_model_layers))
    weak_regularized_layers = " ".join(map(layer_to_arg, weak_model_layers))
    return layers_arg, main_regularized_layers, weak_regularized_layers


def only_bias_arg(reg_only_biased, bias_type: BiasType, enforce_sim: bool):

    bias_dir = {
        BiasType.CLAIM: 'claim_bias_debiasing',
        BiasType.UNKNOWN: 'unknown_bias_debiasing'
    }[bias_type]

    bias_indices = {
        BiasType.UNKNOWN: "data/fever_unknown_bias_splits/5_folds_0.8_confidence/train/train_biased_correct_indices.bin",
        BiasType.CLAIM: "data/fever_claim_bias_splits/5_folds_0.8_confidence/train_biased_correct_indices.bin"
    }[bias_type]

    reg_only_biased_arg = ""
    reg_only_biased_identifier = ""
    if reg_only_biased:
        reg_only_biased_arg = f'--regularize_only_biased --bias_indices {bias_indices}'
        reg_only_biased_identifier = 'STOB_'
        bias_dir = {
            BiasType.UNKNOWN: 'unknown_bias_debiasing/reg_OB',
            BiasType.CLAIM: 'claim_bias_debiasing/reg_OB'
        }[bias_type]
    bias_dir = bias_dir + ("/similarity" if enforce_sim else "/dissimilarity")
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
    debertav3 = False
    node_name = "nlp-a40-1"  # nlp-a40-1, galileo2, plato1, plato2
    threshold = 0.9
    claim_guidance_models = {
        0.7: ('fever_runs/extrensic_debiasing/claim_bias/bert_64_0.7_claim_extrensic_debiasing/checkpoint-1352', 'EXDBE_0.7'),
        0.8: ('fever_runs/extrensic_debiasing/claim_bias/bert_64_0.8_claim_extrensic_debiasing/checkpoint-3360', 'EXDBE_0.8'),
        0.9: ('fever_runs/extrensic_debiasing/claim_bias/bert_64_0.9_claim_extrensic_debiasing/checkpoint-4372', 'EXDBE_0.9'),
    }
    # ['fever_runs/bert_64_base_s420/checkpoint-3796', 'fever_runs/bert_64_base_s638/checkpoint-3796', 'fever_runs/bert_64_base/checkpoint-3796']

    if debertav3:
        unbiased_models = {
            BiasType.CLAIM: ([
                'fever_runs/debertav3/extrensic_debiasing/claim_bias/debertav3_0.8_claim_extrensic_debiasing_s735',
                '',
            ], 'EXDB_0.8')
        }
        biased_models = {
            BiasType.UNKNOWN: ('fever_runs/tinybert', 'TB'),
            BiasType.CLAIM: ('fever_runs/bert_claim_only/checkpoint-7592', 'COBert')
        }
    else:

        unbiased_models = {
            # BiasType.CLAIM: claim_guidance_models[threshold],
            BiasType.UNKNOWN: ('fever_runs/extrensic_debiasing/unknown_bias/bert_64_0.8_unknown_extrensic_debiasing/checkpoint-2808', 'EDBXE')
        }
        biased_models = {
            BiasType.UNKNOWN: ('fever_runs/tinybert', 'TB'),
            BiasType.CLAIM: ('fever_runs/bert_claim_only/checkpoint-7592', 'COBert')
        }

    batch_size = 64
    epochs = 10

    eval_bs, grads_arguments, grads_suffix = grads_arg(node_name, regularize_grads=regularize_only_grads)

    enforce_sim_arg, fw_name, sim_name_arg, weak_model_dir = sim_args(bias_type, biased_models, enforce_sim, unbiased_models)

    run_name_identifier = {
        BiasType.CLAIM: 'ClaimDeBias',
        BiasType.UNKNOWN: '0.8_UNB'
    }[bias_type]

    main_task_lambda = 1
    indices_arg = get_indices_arg(bias_type)
    tags_arg = f'--tags {bias_type.value}'
    bias_dir, reg_only_biased_arg, reg_only_biased_identifier = only_bias_arg(reg_only_biased=True, bias_type=bias_type, enforce_sim=enforce_sim)

    model_name_arg = 'debertav3' if debertav3 else 'bert'
    deberta_dir = 'debertav3/' if debertav3 else ''
    model_arg = 'microsoft/deberta-v3-base' if debertav3 else 'bert-base-uncased'
    tokenization_arg = '--separate_weak_tokenization' if debertav3 and not enforce_sim else ''
    counter = 0
    for regularization_lambda in [100]:
        for regularization_method in ["linear_cka"]:
            # (['e', 10, 11], ['e', 1, 1]) , ['e', 0, 6, 10, 11]
            for l in [[9, 10, 11]]:
                layers_arg, main_regularized_layers, weak_regularized_layers = layers_args(regularized_layers_list=l, model_name_arg='deberta' if debertav3 else 'bert')
                model_name = f"{model_name_arg}_{fw_name}_{reg_only_biased_identifier}{run_name_identifier}_{regularization_method}_{layers_arg}_lambda_" \
                             f"{main_task_lambda}_{regularization_lambda}_{batch_size}{sim_name_arg}{grads_suffix}"
                num_seeds = 3
                for i in range(num_seeds):
                    seed = random.randint(0, 10000)
                    if type(weak_model_dir) is list:
                        current_seed_weak_model = weak_model_dir[i % len(weak_model_dir)]
                    else:
                        current_seed_weak_model = weak_model_dir
                    output_dir = f"fever_runs/{deberta_dir}{bias_dir}/{regularization_method}/{model_name}_s{seed}"
                    os.system(f"""
                        echo nlp_sbatch {node_name} run_glue.py --report_to wandb --model_name_or_path {model_arg} \
                            --output_dir {output_dir} --num_train_epochs {epochs} --learning_rate 2e-5 --per_device_eval_batch_size {eval_bs} \
                             --task_name fever --do_train --do_eval --max_seq_length 128 --per_device_train_batch_size {batch_size} \
                             --weak_model_layer {weak_regularized_layers} --weak_model_path {current_seed_weak_model} --regularized_layers {main_regularized_layers} \
                             --regularization_method {regularization_method} --evaluation_strategy half_epoch --save_strategy no \
                             --regularization_lambda {regularization_lambda} --run_name {model_name} --weight_decay 0.1 --warmup_ratio 0.1 --no_pad_to_max_length \
                             --logging_first_step --regularized_tokens all --token_aggregation_strategy mean {enforce_sim_arg} \
                             {grads_arguments} {indices_arg} {tags_arg} {reg_only_biased_arg} --main_task_lambda {main_task_lambda} \
                             --bias_sampling_strategy stochastic --seed {seed} --wandb_group {tokenization_arg}
                    """)

                    print()
                    counter += 1

    print(f"dispatched {counter} jobs")


if __name__ == '__main__':
    main()
