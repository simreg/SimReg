#!/bin/bash
# head back to source code dir
cd ../

batch_size=64
epochs=10

function run_model {
    model_name=$1
    run_name=${2}_lexical_bias
    ./scripts/py-sbatch.sh run_glue.py --report_to wandb --model_name_or_path ${model_name} \
         --output_dir runs/${run_name} --task_name mnli --do_train --do_eval --max_seq_length 128 \
         --per_device_train_batch_size ${batch_size} --per_device_eval_batch_size 256 \
         --learning_rate 2e-5 --num_train_epochs ${epochs} --evaluation_strategy half_epoch --save_strategy no \
         --run_name ${run_name} --weight_decay 0.1 --warmup_ratio 0.1 --logging_first_step \
          --lexical_bias_model --no_pad_to_max_length --evaluate_on_hans --no_remove_unused_columns
}

run_model bert-base-uncased bert
#run_model google/bert_uncased_L-2_H-128_A-2 tiny_bert
#run_model google/bert_uncased_L-4_H-256_A-4 mini_bert_64



