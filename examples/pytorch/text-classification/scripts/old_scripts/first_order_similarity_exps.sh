#!/bin/bash

cd ~/research_project/examples/pytorch/text-classification/

function reg_bert() {
	if [[ $# < 5 ]]; then
	    echo "missing arguments"
	exit
	fi
	ml=$1
	wl=$2
	regularization_method=$3
	if [[ ${regularization_method} == "none" ]]; then
		regularization_method=""
	fi
	regularization_lambda=$4
	batch_size=$5
	if [[ $# > 5 ]]; then
		epochs=$6
	else
		epochs=12
	fi;
	echo $ml-${regularization_method} batch_size: ${batch_size}, regularization_lambda: ${regularization_lambda};

	model_name=bert_UB2_${regularization_method}_${ml}_lambda_${regularization_lambda}_batch_${batch_size}_CLS
    ./scripts/py-sbatch.sh run_glue.py --report_to wandb --model_name_or_path bert-base-uncased \
    --output_dir runs/self_regularization/${model_name} --task_name mnli --do_train --do_eval --max_seq_length 128 \
     --per_device_train_batch_size ${batch_size} --per_device_eval_batch_size 128 \
    --learning_rate 2e-5 --num_train_epochs ${epochs} --regularized_layers bert.encoder.layer.${ml} \
    --regularization_method ${regularization_method} --evaluation_strategy half_epoch --save_strategy no \
    --regularization_lambda ${regularization_lambda} --run_name ${model_name} \
    --weight_decay 0.1 --warmup_ratio 0.1667 --logging_first_step --evaluate_on_hans \
    --regularized_tokens CLS --no_pad_to_max_length \
    --regularize_only_biased --bias_indices data/tiny_bert_mnli/train_biased_correct_indices.bin \
    --indices_dir data/tinybert_validation_matched

}

#reg_bert ml wl reg_method reg_lambda batch_size epochs aggregation_strategy
lambdas=(7 30)
methods=('cosine_self')
mls=(11)
for ml in ${mls[@]}; do
    for method in ${methods[@]}; do
        for lambda in ${lambdas[@]}; do
            reg_bert ${ml} -1 ${method} ${lambda} 256 6
            exit
        done
    done
done



