#!/bin/bash
cd ../
base_dir=runs/toy_dataset_runs
export WANDB_PROJECT=mlp_reg_test
# weak_model
#python toy_example.py --model_name_or_path MLP --seed 10 --output_dir ${base_dir}/weak_model \
#                    --train_file ../../../data/toy_dataset/train.csv --validation_file ../../../data/toy_dataset/validation.csv \
#                    --do_train --do_eval --max_seq_length 128 --per_device_train_batch_size 32 --per_device_eval_batch_size 32 \
#                    --learning_rate 3e-3 --num_train_epochs 30 --no_pad_to_max_length --overwrite_output_dir \
#                    --evaluation_strategy half_epoch --regularized_tokens MLP --logging_steps 8 --logging_first_step \
#                    --lr_scheduler_type constant_with_warmup


weak_layers=('cat_res')
main_layers=('cat_res' 'classifier.0')
methods=( 'rbf_cka')
lambdas=(1 3 10 50)
i=0
for lambda in ${lambdas[@]}; do
for wl in ${weak_layers[@]}; do
    for ml in ${main_layers[@]}; do
        for method in ${methods[@]}; do
            out_dir=${base_dir}/${method}_${ml}_${wl}_${lambda}_1.0t_l2_10s
            if [ -d ${out_dir} ]; then
                echo directory already exists
                exit
            fi

            python toy_example.py --model_name_or_path MLP --seed 10 --output_dir ${out_dir} --regularization_lambda ${lambda} \
                    --train_file ../../../data/toy_dataset/train.csv --validation_file ../../../data/toy_dataset/validation.csv \
                    --do_train --do_eval --max_seq_length 128 --per_device_train_batch_size 32 --per_device_eval_batch_size 32 \
                    --learning_rate 3e-3 --num_train_epochs 30 --no_pad_to_max_length --overwrite_output_dir \
                    --evaluation_strategy half_epoch --weak_model_layer ${wl} --weak_model_path ${base_dir}/weak_model \
                    --regularized_layers ${ml} --regularization_method ${method} --regularized_tokens MLP \
                    --logging_first_step --logging_steps 8  --lr_scheduler_type constant_with_warmup --warmup_ratio 0.1 \
                    --report_to wandb --rbf_threshold 1.0 --rbf_l2_core > /dev/null &

            let i=i+1
            if [ ${i} -gt 8 ]; then
                echo stopping at ${i}
                let i=0
                wait
            fi
            echo resuming
        done
    done
done
done
wait
