import os
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
        BiasType.HYPOTHESIS: []
        # BiasType.LEXICAL: [
        #     "data/lexical_bias_splits/validation_matched",
        #     "data/lexical_bias_splits/validation_matched/confidence_based_0.65",
        # ],
        # BiasType.HYPOTHESIS: [
        #     'data/mnli_hypothesis_only_hard',
        #     'data/hypothesis_bias_splits/5_folds_0.8_confidence/validation_matched'
        # ]
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


def layer_to_arg(l):
    layer_maper = {
        'P': 'bert.pooler',
        'e': 'bert.embeddings'
    }
    return layer_maper[l] if l in layer_maper else f"bert.encoder.layer.{l}"


def layers_args(regularized_layers_list):
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
    main_regularized_layers = " ".join(map(layer_to_arg, main_model_layers))
    weak_regularized_layers = " ".join(map(layer_to_arg, weak_model_layers))
    return layers_arg, main_regularized_layers, weak_regularized_layers


def only_bias_arg(reg_only_biased, bias_type: BiasType, enforce_sim: bool):
    clark_model = False

    bias_dir = {
        BiasType.LEXICAL: 'lexical_bias_debiasing/clark_model' if clark_model else 'lexical_bias_debiasing',
        BiasType.HYPOTHESIS: 'hypothesis_bias_debiasing/0.8_confidence',
        BiasType.UNKNOWN: 'unknown_bias'
    }[bias_type]
    if not enforce_sim and bias_type == BiasType.HYPOTHESIS:
        bias_dir = 'hypothesis_bias_debiasing/dissimilarity'

    miki_lexical_split = "data/lexical_bias_splits/train/confidence_based_0.65/train_biased_correct_indices.bin"
    clark_lexical_split = "data/clark_lexical_bias_splits/train_biased_correct_indices.bin"

    Hypothesis_bias_low_threshold = "data/hypothesis_bias_splits/5_folds_0.6_confidence/train/train_biased_correct_indices.bin"

    bias_indices = {
        BiasType.LEXICAL: clark_lexical_split if clark_model else miki_lexical_split,
        BiasType.UNKNOWN: "data/tinybert_splits/5_folds_0.8_confidence/train/train_biased_correct_indices.bin",
        # BiasType.UNKNOWN: "bow_logits/train_biased_correct_indices.bin",
        BiasType.HYPOTHESIS: "data/hypothesis_bias_splits/5_folds_0.8_confidence/train/train_biased_correct_indices.bin"
    }[bias_type]

    reg_only_biased_arg = ""
    reg_only_biased_identifier = ""
    if reg_only_biased:
        reg_only_biased_arg = f'--regularize_only_biased --bias_sampling_strategy stochastic --bias_indices {bias_indices}'
        reg_only_biased_identifier = 'STOB_'

        bias_dir = {
            BiasType.LEXICAL: 'lexical_bias_debiasing/clark_model/reg_OB' if clark_model else 'lexical_bias_debiasing/reg_OB',
            BiasType.UNKNOWN: 'unknown_bias/reg_OB/0.8_confidence',
            BiasType.HYPOTHESIS: 'hypothesis_bias_debiasing/reg_OB/0.8_confidence'
        }[bias_type]
        if not enforce_sim and bias_type == BiasType.HYPOTHESIS:
            bias_dir = 'hypothesis_bias_debiasing/reg_OB/dissimilarity'
    return bias_dir, reg_only_biased_arg, reg_only_biased_identifier


def grads_arg(node_name, regularize_grads):
    grads_suffix = "_GReg" if regularize_grads else ""
    grads_arguments = "--regularize_grads" if regularize_grads else ""
    if regularize_grads:
        eval_bs = 64
    else:
        eval_bs = 128 if node_name in ['plato1', 'plato2', 'nlp-2080-1', 'nlp-2080-2'] else 256
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

    if regularize_only_grads:
        node_name = "nlp-a40-1"  # galileo2, newton2, nlp-a40-1, galileo1
    else:
        node_name = "nlp-a40-1"  # plato1, plato2
    #
    # ERM_models = [
    #     'runs/bert_64_base_s401/checkpoint-12272',
    #     'runs/bert_64_base_s254/checkpoint-12272',
    #     'runs/bert_64_base_s420/checkpoint-12272',
    #     'runs/bert_64_base_s638/checkpoint-12272'
    # ]

    clark_lexical_models = [
        'runs/extrensic_debiasing/lexical_bias/clark_bert_64_remove_confident_0.8_s249',
        'runs/extrensic_debiasing/lexical_bias/clark_bert_64_remove_confident_0.8_s501',
        'runs/extrensic_debiasing/lexical_bias/clark_bert_64_remove_confident_0.8_s694'
    ]
    Miki_lexical_model = 'runs/extrensic_debiasing/lexical_bias/bert_64_remove_confident_correct/checkpoint-11678'
    Hypothesis_low_threshold = 'runs/extrensic_debiasing/hypothesis_bias/bert_64_0.6_hypothesis_extrensic_debiasing/checkpoint-7044'

    unbiased_models = {
        BiasType.LEXICAL: (clark_lexical_models, 'CEXDBE'),
        BiasType.HYPOTHESIS: ("runs/extrensic_debiasing/hypothesis_bias/bert_64_0.8_hypothesis_extrensic_debiasing/checkpoint-14355", 'EDBXE_0.8'),
        # BiasType.UNKNOWN: ('runs/extrensic_debiasing/unknown_bias/bert_64_0.8_unknown_extrensic_debiasing/checkpoint-9020', 'EDBXE')
        BiasType.UNKNOWN: ('runs/extrensic_debiasing/unknown_bias/bert_bow_32_0.8_unknown_extrensic_debiasing', 'BowEDBXE')
    }

    biased_models = {
        BiasType.LEXICAL: ('runs/utama_biased_bert_64_2', 'UB2'),
        BiasType.UNKNOWN: (['runs/tiny_bert_mnli'], 'TB'),
        BiasType.HYPOTHESIS: ('runs/5_fold_hypothesis_only/bert_1_hypothesis_only/checkpoint-4908', 'HOBert')
    }
    # ('runs/bert_hypothesis_only', 'HOBert')
    # ('runs/bert_lexical_bias', 'LexicalModel')
    # ('runs/utama_biased_bert_64_2', 'UB2')
    # BiasType.UNKNOWN: ([
    #     'runs/5_folds_unknown_bias/tinybert_0',
    #     'runs/5_folds_unknown_bias/tinybert_1',
    #     'runs/5_folds_unknown_bias/tinybert_2',
    #     'runs/5_folds_unknown_bias/tinybert_3',
    #     'runs/5_folds_unknown_bias/tinybert_4',
    # ], 'TB'),

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
    bias_dir, reg_only_biased_arg, reg_only_biased_identifier = only_bias_arg(reg_only_biased=regularize_only_biased, bias_type=bias_type,
                                                                              enforce_sim=enforce_sim)

    counter = 0
    for regularization_lambda in [100]:
        for regularization_method in ["linear_cka"]:  # abs_cos_cor
            # (['e', 10, 11], ['e', 1, 1]), ['e', 0, 6, 10, 11], ['e', 0, 6, 10, 11, 'P'], [9, 10, 11], (['e', 0, 6, 10, 11], ['e', 0, 0, 0, 1])
            for l in [[9, 10, 11]]:
                layers_arg, main_regularized_layers, weak_regularized_layers = layers_args(regularized_layers_list=l)
                reg_tokens_arg = "all" if 'P' not in l else " ".join(map(lambda x: "none" if x == 'P' else "all", l))
                model_name = f"bert_{fw_name}_{reg_only_biased_identifier}{run_name_identifier}_{regularization_method}_{layers_arg}_lambda_" \
                             f"{main_task_lambda}_{regularization_lambda}_{batch_size}{sim_name_arg}{grads_suffix}"

                num_seeds = 3
                for i in range(num_seeds):
                    seed = random.randint(0, 10000)
                    if type(weak_model_dir) is list:
                        current_seed_weak_model = weak_model_dir[i % len(weak_model_dir)]  # random.randint(0, len(weak_model_dir) - 1)
                    else:
                        current_seed_weak_model = weak_model_dir
                    output_dir = f"runs/{bias_dir}/{regularization_method}/{model_name}_s{seed}"
                    os.system(f"""
                        echo nlp_sbatch {node_name} run_glue.py --report_to wandb --model_name_or_path bert-base-uncased \
                            --output_dir {output_dir} --num_train_epochs {epochs} --learning_rate 2e-5 --per_device_eval_batch_size {eval_bs} \
                             --task_name mnli --do_train --do_eval --max_seq_length 128 --per_device_train_batch_size {batch_size} \
                             --weak_model_layer {weak_regularized_layers} --weak_model_path {current_seed_weak_model} --regularized_layers {main_regularized_layers} \
                             --regularization_method {regularization_method} --evaluation_strategy epoch --save_strategy no \
                             --regularization_lambda {regularization_lambda} --run_name {model_name} --weight_decay 0.1 --warmup_ratio 0.1 --no_pad_to_max_length \
                             --logging_first_step --regularized_tokens {reg_tokens_arg} --token_aggregation_strategy mean {enforce_sim_arg} \
                             {grads_arguments} {indices_arg} {tags_arg} {reg_only_biased_arg} --main_task_lambda {main_task_lambda} \
                             --seed {seed} --wandb_group
                        echo
                    """)
                    counter += 1

    print(f"dispatched {counter} jobs")


if __name__ == '__main__':
    main()
