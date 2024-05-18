import os
import random


def layer_to_arg(l):
    layer_maper = {
        'P': 'bert.pooler',
        'e': 'bert.embeddings'
    }
    return layer_maper[l] if l in layer_maper else f"bert.encoder.layer.{l}"


def layers_args(regularized_layers_list):
    regularized_layers_list = list(map(str, regularized_layers_list))
    layers_suffix = "-".join(regularized_layers_list)
    layers_prefix = "-".join(map(lambda x: f"{x}M", regularized_layers_list))
    layers_arg = f"{layers_prefix}_{layers_suffix}"
    regularized_layers = " ".join(map(layer_to_arg, regularized_layers_list))
    return layers_arg, regularized_layers


def only_bias_arg(bias_dir, reg_only_biased):
    reg_only_biased_arg = ""
    reg_only_biased_identifier = ""
    if reg_only_biased:
        reg_only_biased_arg = f'--regularize_only_biased'
        reg_only_biased_identifier = 'STOB_'
        bias_dir = f'{bias_dir}/reg_OB'
    return bias_dir, reg_only_biased_arg, reg_only_biased_identifier


def grads_arg(node_name, regularize_grads):
    grads_suffix = "_GReg" if regularize_grads else ""
    grads_arguments = "--regularize_grads" if regularize_grads else ""
    eval_bs = 64 if regularize_grads or node_name in ['plato1', 'plato2', 'nlp-2080-1', 'nlp-2080-2'] else 256
    return eval_bs, grads_arguments, grads_suffix


def main():
    # ["nlp-a40-1", "plato2", "plato1", "nlp-2080-1", "galileo2"]
    node_name = "galileo2"
    enforce_sim = True
    regularize_grads = True
    # '1_0.95': 'runs/synthetic_bias_1_0.95/bert_64_base',
    bias_models = {
        '1_0.95': 'runs/synthetic_bias_1_0.95/bert_64_biased',
    }
    unbiased_models = {
        '1_0.95': [
            'runs/bert_64_base',
            'runs/bert_64_base_s254',
            'runs/bert_64_base_s401',
            'runs/bert_64_base_s420',
            'runs/bert_64_base_s638',
        ],
    }
    batch_size = 64
    epochs = 10
    bias_type = '1_0.95'
    if enforce_sim:
        enforce_sim_arg = "--enforce_similarity"
        sim_name_arg = "_sim"
        weak_model_dir = unbiased_models[bias_type]
    else:
        sim_name_arg = ""
        enforce_sim_arg = ""
        weak_model_dir = bias_models[bias_type]

    eval_bs, grads_arguments, grads_suffix = grads_arg(node_name, regularize_grads=regularize_grads)


    run_name_identifier = '1_0.95'
    bias_dir = 'synthetic_bias_1_0.95'
    main_task_lambda = 1
    tags_arg = f'--tags 0.95'
    limits = [('galileo2', 5), ('galileo1', 5), ('nlp-a40-1', 8)]
    current_i = 0
    current_limit = 5
    jobs_counter = 0
    bias_dir, reg_only_biased_arg, reg_only_biased_identifier = only_bias_arg(bias_dir, reg_only_biased=True)
    layers_list = [['e']] + [[i] for i in range(1,12,2)]
    for l in layers_list:
        # layers_arg, regularized_layers = layers_args(regularized_layers_list=['e', 0, 6, 10, 11])
        layers_arg, regularized_layers = layers_args(regularized_layers_list=l)

        for regularization_method in ["abs_cos_cor"]:
            for regularization_lambda in [10]:
                model_name = f"bert_{reg_only_biased_identifier}{run_name_identifier}_{regularization_method}_{layers_arg}_lambda" \
                             f"_{main_task_lambda}_{regularization_lambda}_{batch_size}{sim_name_arg}{grads_suffix}"
                for _ in range(2):
                    seed = random.randint(0, 10000)
                    if type(weak_model_dir) is list:
                        current_seed_weak_model = weak_model_dir[random.randint(0, len(weak_model_dir) - 1)]
                    else:
                        current_seed_weak_model = weak_model_dir
                    output_dir = f"runs/{bias_dir}/{regularization_method}/{model_name}_s{seed}"
                    os.system(f"""
                        echo nlp_sbatch {node_name} run_glue.py --report_to wandb --model_name_or_path bert-base-uncased \
                            --output_dir {output_dir} --num_train_epochs {epochs} --learning_rate 2e-5 --per_device_eval_batch_size {eval_bs} \
                             --task_name mnli --do_train --do_eval --max_seq_length 128 --per_device_train_batch_size {batch_size} \
                             --weak_model_layer {regularized_layers} --weak_model_path {current_seed_weak_model} --regularized_layers {regularized_layers} \
                             --regularization_method {regularization_method} --evaluation_strategy epoch --save_strategy no \
                             --regularization_lambda {regularization_lambda} --run_name {model_name} --weight_decay 0.1 --warmup_ratio 0.1 --no_pad_to_max_length \
                             --logging_first_step --regularized_tokens all --token_aggregation_strategy mean \
                             {grads_arguments} {tags_arg} {reg_only_biased_arg} --main_task_lambda {main_task_lambda} {enforce_sim_arg} \
                             --bias_sampling_strategy stochastic --seed {seed} --wandb_group --synthetic_bias_prevalence 1 --bias_correlation_prob 0.95
                    """)
                    print("")
                    current_limit -= 1
                    if current_limit == 0:
                        current_i += 1
                        node_name = limits[current_i][0]
                        current_limit = limits[current_i][1]
                    # if jobs_counter == 15:
                    #     node_name = "plato2"


if __name__ == '__main__':
    main()
