import os
import random
from glob import glob

os.chdir("/home/redaigbaria/SimReg")

weak_models = {
    'hypothesis_bert': {
        '5_epochs': (
            'batch_32/mnli/hypothesis_only_bias/extrensic_debiasing/HypothesisBERT/5_epochs/bert', '5HypoExtrensic')
    }
}

bias_indices_choices = {
    'hypothesis_bert': "data/hypothesis_bias_splits/5_folds_0.6_confidence/train/train_biased_correct_indices.bin",
}

bias_type = "hypothesis_only_bias"
# ----------------------------------------------------------------------------------------------------------------------
# RUN PARAMETERS
bias_set_choice = "hypothesis_bert"
bias_indices = bias_indices_choices[bias_set_choice]
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
weak_model_paths = glob(f"{weak_model_path}*")
lr_arg = "_2e5" if lr == '2e-5' else ""

# 9620, 491, 163
for weak_model_path_arg in weak_model_paths:
    seed = random.randint(0, 1000)
    run_name = f"{dir_name}_{reg_lambda}_{epochs}Epochs_{weak_model_name}_{batch_size}Batch{lr_arg}_s{seed}"
    output_dir = f"batch_{batch_size}/mnli/{bias_type}/{dir_name}/{run_name}"
    assert not os.path.exists(output_dir), "Output directory already exists"

    os.system(f"""
    nlp_sbatch run_glue.py --report_to wandb --model_name_or_path bert-base-uncased --output_dir {output_dir} \
        --task_name mnli --do_train --do_eval --tags {bias_type} --regularization_lambda {reg_lambda} \
        --save_strategy no --no_pad_to_max_length --logging_first_step --evaluation_strategy half_epoch \
        --run_name {run_name} --num_train_epochs {epochs} --learning_rate 5e-5 --weight_decay 0.1  \
        --max_seq_length 128 --seed {seed} --per_device_train_batch_size {batch_size} --per_device_eval_batch_size {batch_size} \
        --bias_indices {bias_indices} --regularize_only_biased --warmup_steps 3000 --regularization_method linear_cka \
        --bias_sampling_strategy stochastic --enforce_similarity --regularized_tokens {regularized_tokens_arg} --token_aggregation_strategy mean  \
        --regularized_layers {regularized_layers} --weak_models_layers {regularized_layers} --weak_models_path {weak_model_path_arg}
        """)
