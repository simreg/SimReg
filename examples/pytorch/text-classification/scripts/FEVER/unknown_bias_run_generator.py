import os
from glob import glob

weak_models = {
    'bow': {
        '5_epochs': ('batch_32/fever/unknown_bias/extrensic_debiasing/bow_1_epoch/5_epochs/bert', '5BOWExtrensic')
    },
    'bow_balanced': {
        '5_epochs': ('batch_32/fever/unknown_bias/balanced_debiasing/BOW/5_epochs/bert', '5BowBalanced_correct')
    },
    'tiny_bert': {
        '5_epochs': ('batch_32/fever/unknown_bias/extrensic_debiasing/tiny_bert/5_epochs/bert', '5TinyBertExtrensic')
    },
}

bias_indices_choices = {
    # 'bow': "data/fever_unknown_bias_splits/bow_1_epoch/train_biased_correct_indices.bin",
    'bow': "data/fever_unknown_bias_splits/bow/0.8_threshold/train_biased_correct_indices.bin",
    'tiny_bert': "data/fever_unknown_bias_splits/5_folds_0.8_confidence/train/train_biased_correct_indices.bin",
    'bow_balanced': "data/fever_unknown_bias_splits/bow/0.8_threshold/all_confident.bin"
}

bias_type = "unknown_bias"
# ----------------------------------------------------------------------------------------------------------------------
# RUN PARAMETERS
bias_set_choice = "bow_balanced"
bias_indices = bias_indices_choices['bow']
weak_models = weak_models[bias_set_choice]
batch_size = 64
reg_lambda = 100
epochs = 5
weak_model_epochs = 5
regularized_layers = [11, 10, 9]
regularized_tokens = ['M', 'M', 'M']
lr = '2e-5'
# ----------------------------------------------------------------------------------------------------------------------

regularized_token_map = {'M': 'all', 'C': 'CLS', 'N': 'none'}
regularized_tokens_arg = " ".join(map(lambda x: regularized_token_map[x], regularized_tokens))
dir_name = "-".join(
    map(lambda x: str(x[0]) + x[1] if x[0] != 'P' else x[0], zip(regularized_layers, regularized_tokens)))

regularized_layers = " ".join(["bert.pooler" if l == 'P' else f"bert.encoder.layer.{l}" for l in regularized_layers])

weak_model_path, weak_model_name = weak_models[f'{weak_model_epochs}_epochs']

lr_arg = "_2e5" if lr == '2e-5' else ""


for seed in [936, 833, 742]:
    run_name = f"{dir_name}_{reg_lambda}_{epochs}Epochs_0.8Threshold_{weak_model_name}_{batch_size}Batch{lr_arg}_s{seed}"
    output_dir = f"batch_{batch_size}/fever/{bias_type}/{dir_name}/{run_name}"
    weak_model_path_arg = f"{weak_model_path}_s{seed}"
    assert not os.path.exists(output_dir), "Output directory already exists"
    assert os.path.exists(weak_model_path_arg)

    os.system(f"""
    nlp_sbatch run_glue.py --report_to wandb --model_name_or_path bert-base-uncased --output_dir {output_dir} \
        --task_name fever --do_train --do_eval --tags {bias_type} --regularization_lambda {reg_lambda} \
        --save_strategy no --no_pad_to_max_length --logging_first_step --evaluation_strategy half_epoch --seed {seed} \
        --run_name {run_name} --num_train_epochs {epochs} --learning_rate {lr} --weight_decay 0.1 --warmup_steps 3000 \
        --max_seq_length 128 --per_device_train_batch_size {batch_size} --per_device_eval_batch_size {batch_size} --regularize_only_biased \
        --bias_indices {bias_indices} --token_aggregation_strategy mean --regularization_method linear_cka \
        --bias_sampling_strategy stochastic --enforce_similarity --regularized_tokens {regularized_tokens_arg}  \
        --regularized_layers {regularized_layers} --weak_models_layers {regularized_layers} --weak_models_path {weak_model_path_arg}
    """)
