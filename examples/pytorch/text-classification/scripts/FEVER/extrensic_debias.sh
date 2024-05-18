#!/bin/bash

# Claim debiasing
seeds=(97)
epochs=5
for seed in ${seeds[@]}; do
	nlp_sbatch run_glue.py --report_to wandb --model_name_or_path bert-base-uncased --output_dir batch_32/fever/claim_bias/extrensic_debiasing/claim_bert/${epochs}_epochs/2e5_bert_s${seed} \
		--task_name fever --do_train --do_eval --tags claim_bias --save_strategy no --no_pad_to_max_length --weight_decay 0.1 --warmup_steps 3000 \
		--logging_first_step --evaluation_strategy half_epoch --remove_biased_samples_from_train \
		--run_name claim_debiasing_ClaimBERT0.8_${epochs}Epochs_2e5_s${seed} --num_train_epochs ${epochs} --learning_rate 2e-5 \
		--max_seq_length 128 --seed ${seed} --per_device_train_batch_size 32 --per_device_eval_batch_size 32 \
		--bias_indices data/fever_claim_bias_splits/5_folds_0.8_confidence/train_biased_correct_indices.bin
done
