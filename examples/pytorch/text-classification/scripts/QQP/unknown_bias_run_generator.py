import os
from glob import glob

import torch

weak_models = {
    # 'bow': {
    #     '0_epochs': ('bert-base-uncased', 'Uncased'),
    #     '5_epochs': ('batch_32/{task}/unknown_bias/extrensic_debiasing/bow_1_epoch/5_epochs/bert', '5BOWExtrensic')
    # },
    'bow': {
        '5_epochs': ('batch_32/qqp/unknown_bias/extrensic_debiasing/BOW_5Epoch_2e5_0.8_threshold/5_epochs/bert', '5BOWExtrensic')
    },
    'tiny_bert': {
        '0_epochs': ('bert-base-uncased', 'Uncased'),
        '5_epochs': ("batch_32/qqp/unknown_bias/extrensic_debiasing/TB_0.8_threshold/5_epochs/bert", '5TBExtrensic')
    },
    "bow_balanced": {
        "5_epochs": ("batch_32/qqp/unknown_bias/balanced_debiasing/BOW/5_epochs/bert", "5BowBalanced_Correct")
    },
    "tinybert_balanced": {
        "5_epochs": ("batch_32/qqp/unknown_bias/balanced_debiasing/TinyBERT/5_epochs/bert", "5TBBalanced_Correct")
    }
}

bias_indices_choices = {
    'tiny_bert': "data/qqp_unknown_bias_splits/5_folds_tinybert_0.8_confidence_splits/train/train_biased_correct_indices.bin",
    'tinybert_balanced': "data/qqp_unknown_bias_splits/5_folds_tinybert_0.8_confidence_splits/train/all_confident.bin",
    # "bow": "data/qqp_unknown_bias_splits/BOW/1Epoch/0.8_threshold/train_biased_correct_indices.bin"
    'bow': 'data/qqp_unknown_bias_splits/BOW/5Epoch_2e5/0.8_threshold/train_biased_correct_indices.bin'
}

bias_type = "unknown_bias"
# ----------------------------------------------------------------------------------------------------------------------
# RUN PARAMETERS
bias_set_choice = "bow"
bias_indices = bias_indices_choices[bias_set_choice]
weak_models = weak_models[bias_set_choice]
batch_size = 64
reg_lambda = 100
epochs = 5
weak_model_epochs = 5
regularized_layers = [11, 10, 9]
regularized_tokens = ['M', 'M', 'M']
lr = '5e-5'
task = 'qqp'
# ----------------------------------------------------------------------------------------------------------------------
regularized_token_map = {'M': 'all', 'C': 'CLS', 'N': 'none'}
regularized_tokens_arg = " ".join(map(lambda x: regularized_token_map[x], regularized_tokens))
dir_name = "-".join(
    map(lambda x: str(x[0]) + x[1] if x[0] != 'P' else x[0], zip(regularized_layers, regularized_tokens)))

regularized_layers = " ".join(["bert.pooler" if l == 'P' else f"bert.encoder.layer.{l}" for l in regularized_layers])

weak_model_path, weak_model_name = weak_models[f'{weak_model_epochs}_epochs']
weak_model_paths = glob(f"{weak_model_path}*")


lr_arg = "_2e5" if lr == '2e-5' else ""

for i in range(len(weak_model_paths)):  # 491, 9620
    seed = torch.randint(1000, (1,)).item()
    run_name = f"{dir_name}_{reg_lambda}_{epochs}Epochs_0.8Threshold_{weak_model_name}_{batch_size}Batch{lr_arg}_s{seed}"
    output_dir = f"batch_{batch_size}/{task}/{bias_type}/{dir_name}/{run_name}"
    assert not os.path.exists(output_dir), f"Output directory already exists ({output_dir})"
    # weak_model_path_arg = f"{weak_model_path}_s{seed}" if weak_model_epochs > 0 else weak_model_path
    weak_model_path_arg = weak_model_paths[i]

    os.system(f"""
    nlp_sbatch run_glue.py --report_to wandb --model_name_or_path bert-base-uncased --output_dir {output_dir} \
        --task_name {task} --do_train --do_eval --tags {bias_type} --regularization_lambda {reg_lambda} \
        --save_strategy no --no_pad_to_max_length --logging_first_step --evaluation_strategy half_epoch --seed {seed} \
        --run_name {run_name} --num_train_epochs {epochs} --learning_rate {lr} --weight_decay 0.1 --warmup_steps 3000 \
        --max_seq_length 128 --per_device_train_batch_size {batch_size} --per_device_eval_batch_size {batch_size} --regularize_only_biased \
        --bias_indices {bias_indices} --token_aggregation_strategy mean --regularization_method linear_cka \
        --bias_sampling_strategy stochastic --enforce_similarity --regularized_tokens {regularized_tokens_arg}  \
        --regularized_layers {regularized_layers} --weak_models_layers {regularized_layers} --weak_models_path {weak_model_path_arg}
    """)
