import os

weak_models = {
    'bias_correct': {
        '0_epochs': ('bert-base-uncased', 'Uncased'),
        '3_epochs': ('batch_32/mnli/lexical_bias/extrensic_debiasing/bert', '3Extrensic'),
        '5_epochs': ('batch_32/mnli/lexical_bias/extrensic_debiasing/5_epochs/bert', '5Extrensic')
    },
    'all_confident': {
        '0_epochs': ('bert-base-uncased', 'Uncased'),
        '3_epochs': ('batch_32/mdl_sanity/lexical_extrensic/lexical_extrensic_all_confident', '3AllConfidentExtrensic'),
        '5_epochs': ('batch_32/mdl_sanity/lexical_extrensic/5Epochs/lexical_extrensic_all_confident_5Epochs', '5AllConfidentExtrensic')
    }
}

bias_indices_choices = {
    'all_confident': "data/lexical_bias_splits/train/confidence_based_0.65/all_confident_indices.bin",
    'bias_correct': "data/lexical_bias_splits/train/confidence_based_0.65/train_biased_correct_indices.bin"
}

# ----------------------------------------------------------------------------------------------------------------------
# RUN PARAMETERS
bias_set_choice = "bias_correct"  # pretrained, all_confident or bias_correct
bias_indices = bias_indices_choices[bias_set_choice]
weak_models = weak_models[bias_set_choice]
batch_size = 64
reg_lambda = 100
epochs = 5
weak_model_epochs = 5
regularized_layers = ['P', 11, 10]
regularized_tokens = ['N', 'M', 'M']

# ----------------------------------------------------------------------------------------------------------------------
regularized_token_map = {'M': 'all', 'C': 'CLS', 'N': 'none'}
regularized_tokens_arg = " ".join(map(lambda x: regularized_token_map[x], regularized_tokens))
dir_name = "-".join(map(lambda x: str(x[0]) + x[1] if x[0] != 'P' else x[0], zip(regularized_layers, regularized_tokens)))

regularized_layers = " ".join(["bert.pooler" if l == 'P' else f"bert.encoder.layer.{l}" for l in regularized_layers])

weak_model_path, weak_model_name = weak_models[f'{weak_model_epochs}_epochs']
#
for seed in [9620, 491, 163]:
    run_name = f"{dir_name}_{reg_lambda}_{epochs}Epochs_{weak_model_name}_{batch_size}Batch_s{seed}"
    output_dir = f"batch_{batch_size}/mnli/lexical_bias/{dir_name}/{run_name}"
    assert not os.path.exists(output_dir), "Output directory already exists"
    weak_model_path_arg = f"{weak_model_path}_s{seed}" if weak_model_epochs > 0 else weak_model_path

    os.system(f"""
    nlp_sbatch run_glue.py --report_to wandb --model_name_or_path bert-base-uncased --output_dir {output_dir} \
        --task_name mnli --do_train --do_eval --tags lexical_bias --regularization_lambda {reg_lambda} \
        --save_strategy epoch --no_pad_to_max_length --logging_first_step --evaluation_strategy half_epoch \
        --evaluate_on_hans --run_name {run_name} --num_train_epochs {epochs} --learning_rate 5e-5 --weight_decay 0.1 --warmup_steps 3000 \
        --max_seq_length 128 --seed {seed} --per_device_train_batch_size {batch_size} --per_device_eval_batch_size {batch_size} --regularize_only_biased \
        --bias_indices {bias_indices} \
        --bias_sampling_strategy stochastic --enforce_similarity --regularized_tokens {regularized_tokens_arg} --token_aggregation_strategy mean --regularization_method linear_cka \
        --regularized_layers {regularized_layers} --weak_models_layers {regularized_layers} --weak_models_path {weak_model_path_arg}
    """)
