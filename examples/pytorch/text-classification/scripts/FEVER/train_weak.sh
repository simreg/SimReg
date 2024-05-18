#!/bin/bash


nlp_sbatch run_glue.py --model_name_or_path google/bert_uncased_L-2_H-128_A-2 --output_dir fever_runs/tiny_bert_1_epoch --task_name fever --do_train --do_eval --max_seq_length 128 --per_device_train_batch_size 32 --weight_decay 0.1 --warmup_ratio 0.1 --learning_rate 5e-5 --num_train_epochs 1 --report_to wandb --run_name TinyBERT --evaluation_strategy half_epoch --logging_first_step --no_pad_to_max_length --save_strategy no

# Train Utama style BERT:
nlp_sbatch run_glue.py --model_name_or_path bert-base-uncased --output_dir fever_runs/bert_utama_1_epoch --task_name fever --do_train --do_eval --max_seq_length 128 --per_device_train_batch_size 32 --weight_decay 0.1 --warmup_ratio 0.1 --learning_rate 5e-5 --num_train_epochs 1 --report_to wandb --run_name utama_bert --evaluation_strategy half_epoch --logging_first_step --no_pad_to_max_length --save_strategy no --max_train_samples 2016

# Train BOW:
nlp_sbatch run_glue.py --model_name_or_path bow --output_dir fever_runs/bow_1_epoch_s401 --task_name fever --do_train --do_eval --max_seq_length 128 --per_device_train_batch_size 256 --learning_rate 1e-3 --num_train_epochs 1 --report_to wandb --evaluation_strategy half_epoch --config_name /home/redaigbaria/hans-forgetting/config/lstmatt_small_config_abs.json --seed 401