#!/bin/bash

seeds=(742 936)
epochs=5
for seed in ${seeds[@]}; do
	nlp_sbatch run_glue.py --report_to wandb --model_name_or_path bert-base-uncased --output_dir batch_32/qqp/unknown_bias/balanced_debiasing/BOW/${epochs}_epochs/bert_s${seed} \
		--task_name qqp --do_train --do_eval --tags unknown_bias balanced_debiasing \
		--save_strategy no --no_pad_to_max_length --logging_first_step --evaluation_strategy half_epoch \
		--run_name unknown_bias_BalancedDebiasing_BOW_${epochs}Epochs_s${seed} --num_train_epochs ${epochs} --learning_rate 5e-5 --weight_decay 0.1 --warmup_steps 3000 \
		--max_seq_length 128 --seed ${seed} --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --select_only_biased_samples \
		--bias_indices data/qqp_unknown_bias_splits/BOW/1Epoch/0.8_threshold/qqp_balanced_indices.bin
done

# data/qqp_unknown_bias_splits/5_folds_tinybert_0.8_confidence_splits/train/qqp_balanced_indices.bin
