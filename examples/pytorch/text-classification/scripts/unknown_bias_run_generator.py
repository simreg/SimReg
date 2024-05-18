import os

os.chdir("/home/redaigbaria/SimReg")

weak_models = {
    'bow': {
        '0_epochs': ('bert-base-uncased', 'Uncased'),
        '5_epochs': ('batch_32/mnli/unknown_bias/extrensic_debiasing/bow/5_epochs/bert', '5BOWExtrensic')
    },
    'tiny_bert': {
        '0_epochs': ('bert-base-uncased', 'Uncased'),
        '5_epochs': ('batch_32/mnli/unknown_bias/extrensic_debiasing/tiny_bert/5_epochs/bert', '5TinyBertExtrensic')
    },
    'balanced_bow': {
        '5_epochs': (
            "batch_32/mnli/unknown_bias/balanced_debiasing/bow/5_epochs/bert", '5BalancedBOW_AllConfident')
    },
    'balanced_tinybert': {
        '5_epochs': ("batch_32/mnli/unknown_bias/balanced_debiasing/TinyBERT/5_epochs/bert", '5BalancedTB_AllConfident')
    }
}

bias_indices_choices = {
    'bow': "bow_logits/train_biased_correct_indices.bin",
    'balanced_bow': "bow_logits/all_confident_indices.bin",
    'tiny_bert': "data/tinybert_splits/5_folds_0.8_confidence/train/train_biased_correct_indices.bin",
    'balanced_tinybert': "data/tinybert_splits/5_folds_0.8_confidence/train/all_confident.bin"
}

bias_type = "unknown_bias"
# ----------------------------------------------------------------------------------------------------------------------
# RUN PARAMETERS
bias_set_choice = "tiny_bert"
bias_indices = bias_indices_choices[bias_set_choice]
weak_models = weak_models[bias_set_choice]
batch_size = 64
reg_lambda = 100
epochs = 5
weak_model_epochs = 0
regularized_layers = [11, 10, 9]
regularized_tokens = ['M', 'M', 'M']

# ----------------------------------------------------------------------------------------------------------------------
regularized_token_map = {'M': 'all', 'C': 'CLS', 'N': 'none'}
regularized_tokens_arg = " ".join(map(lambda x: regularized_token_map[x], regularized_tokens))
dir_name = "-".join(
    map(lambda x: str(x[0]) + x[1] if x[0] != 'P' else x[0], zip(regularized_layers, regularized_tokens)))

regularized_layers = " ".join(["bert.pooler" if l == 'P' else f"bert.encoder.layer.{l}" for l in regularized_layers])

weak_model_path, weak_model_name = weak_models[f'{weak_model_epochs}_epochs']

# 9620, 491, 163
for seed in []:
    run_name = f"{dir_name}_{reg_lambda}_{epochs}Epochs_{weak_model_name}_{batch_size}Batch_s{seed}"
    output_dir = f"batch_{batch_size}/mnli/{bias_type}/{dir_name}/{run_name}"
    assert not os.path.exists(output_dir), "Output directory already exists"
    if weak_model_epochs > 0:
        weak_model_path_arg = f"{weak_model_path}_s{seed}"
        assert os.path.exists(weak_model_path_arg)
    else:
        weak_model_path_arg = weak_model_path

    os.system(f"""
    echo nlp_sbatch run_glue.py --report_to wandb --model_name_or_path bert-base-uncased --output_dir {output_dir} \
        --task_name mnli --do_train --do_eval --tags {bias_type} --regularization_lambda {reg_lambda} \
        --save_strategy no --no_pad_to_max_length --logging_first_step --evaluation_strategy half_epoch \
        --run_name {run_name} --num_train_epochs {epochs} --learning_rate 5e-5 --weight_decay 0.1  \
        --max_seq_length 128 --seed {seed} --per_device_train_batch_size {batch_size} --per_device_eval_batch_size {batch_size} \
        --bias_indices {bias_indices} --regularize_only_biased --warmup_steps 3000 --regularization_method linear_cka \
        --bias_sampling_strategy stochastic --enforce_similarity --regularized_tokens {regularized_tokens_arg} --token_aggregation_strategy mean  \
        --regularized_layers {regularized_layers} --weak_models_layers {regularized_layers} --weak_models_path {weak_model_path_arg}
    """)
