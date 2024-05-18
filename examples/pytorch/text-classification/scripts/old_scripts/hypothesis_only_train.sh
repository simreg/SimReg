#!/bin/bash
# head back to source code dir
cd ../

batch_size=64
epochs=10

function run_model {
    model_name=$1
    run_name=${2}_hypothesis_only
    i=$3
    ./scripts/py-sbatch.sh run_glue.py --report_to wandb --model_name_or_path ${model_name} \
         --output_dir runs/5_fold_hypothesis_only/${run_name} --task_name mnli --do_train --do_eval --max_seq_length 128 \
         --per_device_train_batch_size ${batch_size} --per_device_eval_batch_size 64 \
         --learning_rate 2e-5 --num_train_epochs ${epochs} --evaluation_strategy half_epoch --save_strategy half_epoch \
         --run_name ${run_name} --weight_decay 0.1 --warmup_ratio 0.1667 --logging_first_step --hypothesis_only \
         --indices_dir data/mnli_hypothesis_only_hard  data/hypothesis_bias_splits/hypothesis_bias_0.9_confidence/validation_matched \
         --bias_indices ../../../data/MNLI_train_5_folds/fold_${i}.bin --remove_biased_samples_from_train
}

for i in {0..4} ; do
    run_model bert-base-uncased bert_${i} $i
done

#run_model bert-base-uncased bert
#run_model google/bert_uncased_L-2_H-128_A-2 tiny_bert
#run_model google/bert_uncased_L-4_H-256_A-4 mini_bert
