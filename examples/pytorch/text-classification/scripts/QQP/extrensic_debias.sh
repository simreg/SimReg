#!/bin/bash

seeds=(9620 163 197) # 491
epochs=5
for seed in ${seeds[@]}; do
	nlp_sbatch run_glue.py --report_to wandb --model_name_or_path bert-base-uncased --output_dir batch_32/qqp/lexical_bias/extrensic_debiasing/new_0.65_threshold/${epochs}_epochs/bert_s${seed} \
		--task_name qqp --do_train --do_eval --tags lexical_bias \
		--save_strategy no --no_pad_to_max_length --logging_first_step --evaluation_strategy half_epoch \
		--run_name lexical_extrensic_debiasing_0.65NewClark_${epochs}Epochs_s${seed} --num_train_epochs ${epochs} --learning_rate 5e-5 --weight_decay 0.1 --warmup_steps 3000 \
		--max_seq_length 128 --seed ${seed} --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --remove_biased_samples_from_train \
		--bias_indices data/new_qqp_lexical_bias/0.65_threshold/train_biased_correct_indices.bin
done

# Unknown bias:
#seeds=(9620 163 491)
#epochs=5
#for seed in ${seeds[@]}; do
#nlp_sbatch run_glue.py --report_to wandb --model_name_or_path bert-base-uncased --output_dir batch_32/qqp/unknown_bias/extrensic_debiasing/BOW_1Epoch_0.8_threshold/${epochs}_epochs/bert_s${seed} \
#  --task_name qqp --do_train --do_eval --tags unknown_bias \
#  --save_strategy no --no_pad_to_max_length --logging_first_step --evaluation_strategy half_epoch \
#  --run_name unknown_bias_extrensic_debiasing_0.8BOW_${epochs}Epochs_s${seed} --num_train_epochs ${epochs} --learning_rate 5e-5 --weight_decay 0.1 --warmup_steps 3000 \
#  --max_seq_length 128 --seed ${seed} --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --remove_biased_samples_from_train \
#  --bias_indices data/qqp_unknown_bias_splits/BOW/1Epoch/0.8_threshold/train_biased_correct_indices.bin
#done

#seeds=(9620 163 491)
#epochs=5
#for seed in ${seeds[@]}; do
#	nlp_sbatch run_glue.py --report_to wandb --model_name_or_path bert-base-uncased --output_dir batch_32/qqp/unknown_bias/extrensic_debiasing/BOW_5Epoch_2e5_0.8_threshold/${epochs}_epochs/bert_s${seed} \
#		--task_name qqp --do_train --do_eval --tags unknown_bias \
#		--save_strategy no --no_pad_to_max_length --logging_first_step --evaluation_strategy half_epoch \
#		--run_name unknown_bias_extrensic_debiasing_0.8BOW5Epoch_${epochs}Epochs_s${seed} --num_train_epochs ${epochs} --learning_rate 5e-5 --weight_decay 0.1 --warmup_steps 3000 \
#		--max_seq_length 128 --seed ${seed} --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --remove_biased_samples_from_train \
#		--bias_indices data/qqp_unknown_bias_splits/BOW/5Epoch_2e5/0.8_threshold/train_biased_correct_indices.bin
#done

#seeds=(9620 163 491)
#epochs=5
#for seed in ${seeds[@]}; do
#	nlp_sbatch run_glue.py --report_to wandb --model_name_or_path bert-base-uncased --output_dir batch_32/qqp/unknown_bias/extrensic_debiasing/BOW_5Epoch_2e5_0.9_threshold/${epochs}_epochs/bert_s${seed} \
#		--task_name qqp --do_train --do_eval --tags unknown_bias \
#		--save_strategy no --no_pad_to_max_length --logging_first_step --evaluation_strategy half_epoch \
#		--run_name unknown_bias_extrensic_debiasing_0.9BOW5Epoch_${epochs}Epochs_s${seed} --num_train_epochs ${epochs} --learning_rate 5e-5 --weight_decay 0.1 --warmup_steps 3000 \
#		--max_seq_length 128 --seed ${seed} --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --remove_biased_samples_from_train \
#		--bias_indices data/qqp_unknown_bias_splits/BOW/5Epoch_2e5/0.9_threshold/train_biased_correct_indices.bin
#done

#
#seeds=(9620 163 491)
#epochs=5
#for seed in ${seeds[@]}; do
#nlp_sbatch run_glue.py --report_to wandb --model_name_or_path bert-base-uncased --output_dir batch_32/qqp/unknown_bias/extrensic_debiasing/TB_0.8_threshold/${epochs}_epochs/bert_s${seed} \
#  --task_name qqp --do_train --do_eval --tags unknown_bias \
#  --save_strategy no --no_pad_to_max_length --logging_first_step --evaluation_strategy half_epoch \
#  --run_name unknown_bias_extrensic_debiasing_0.8TB_${epochs}Epochs_s${seed} --num_train_epochs ${epochs} --learning_rate 5e-5 --weight_decay 0.1 --warmup_steps 3000 \
#  --max_seq_length 128 --seed ${seed} --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --remove_biased_samples_from_train \
#  --bias_indices data/qqp_unknown_bias_splits/5_folds_tinybert_0.8_confidence_splits/train/train_biased_correct_indices.bin
#done

#seeds=(9620 163 491)
#epochs=5
#for seed in ${seeds[@]}; do
#nlp_sbatch run_glue.py --report_to wandb --model_name_or_path bert-base-uncased --output_dir batch_32/qqp/unknown_bias/extrensic_debiasing/TB_0.9_threshold/${epochs}_epochs/bert_s${seed} \
#  --task_name qqp --do_train --do_eval --tags unknown_bias \
#  --save_strategy no --no_pad_to_max_length --logging_first_step --evaluation_strategy half_epoch \
#  --run_name unknown_bias_extrensic_debiasing_0.9TB_${epochs}Epochs_s${seed} --num_train_epochs ${epochs} --learning_rate 5e-5 --weight_decay 0.1 --warmup_steps 3000 \
#  --max_seq_length 128 --seed ${seed} --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --remove_biased_samples_from_train \
#  --bias_indices data/qqp_unknown_bias_splits/5_folds_tinybert_0.9_confidence_splits/train/train_biased_correct_indices.bin
#done
