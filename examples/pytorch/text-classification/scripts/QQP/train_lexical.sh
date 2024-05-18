#!/bin/bash


nlp_sbatch run_glue.py --model_name_or_path google/bert_uncased_L-2_H-128_A-2 --output_dir qqp_runs/5_folds_lexicalModel/fold_{i} --task_name qqp --do_train --do_eval --max_seq_length 128 --per_device_train_batch_size 32 --weight_decay 0.1 --warmup_ratio 0.1 --learning_rate 2e-5 --num_train_epochs 5 --report_to wandb --run_name 5_fold_lexicalModel --per_device_eval_batch_size 256 --evaluation_strategy half_epoch --logging_first_step --no_pad_to_max_length --save_strategy no --tags lexical_bias --seed {seed} --lexical_bias_model --no_remove_unused_columns --bias_indices

