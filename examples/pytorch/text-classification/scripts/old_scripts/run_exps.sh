#!/bin/bash

weak_model_dir=runs/tiny_bert_mnli
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
	echo $ml-$wl-${regularization_method} batch_size: ${batch_size}, regularization_lambda: ${regularization_lambda};
	model_name=FTBert_OBTB_${regularization_method}_0-6-10-11_0-0-1-1_lambda_0.1_${regularization_lambda}_batch_${batch_size}_GReg

#    if [[ -d runs/reg_only_biased/${model_name} ]]; then
#        ./scripts/py-sbatch_nlp.sh
    ./scripts/py-sbatch.sh run_glue.py --report_to wandb --model_name_or_path runs/bert_64_base --output_dir runs/gradient_regularization/fine_tuned_main_model/${model_name} \
     --task_name mnli --do_train --do_eval --max_seq_length 128 --per_device_train_batch_size ${batch_size} --per_device_eval_batch_size ${batch_size} \
     --learning_rate 2e-5 --num_train_epochs ${epochs} --weak_model_layer bert.embeddings bert.encoder.layer.0 bert.encoder.layer.0 \
     bert.encoder.layer.1 bert.encoder.layer.1 \
     --weak_model_path ${weak_model_dir} --regularized_layers bert.embeddings bert.encoder.layer.0 bert.encoder.layer.6 \
     bert.encoder.layer.10 bert.encoder.layer.11 \
     --regularization_method ${regularization_method} --evaluation_strategy half_epoch --save_strategy no \
     --regularization_lambda ${regularization_lambda} --run_name ${model_name} --weight_decay 0.1 --warmup_ratio 0.1 \
     --logging_first_step --evaluate_on_hans \
     --regularized_tokens all all all all CLS --token_aggregation_strategy mean \
     --indices_dir data/tinybert_validation_matched data/validation_matched_group_dro_splits \
        data/lexical_bias_splits/validation_matched data/mnli_hypothesis_only_hard \
     --regularize_only_biased --bias_indices data/tinybert_splits/train/train_biased_correct_indices.bin  \
     data/tinybert_splits/train/train_anti_biased_indices.bin \
     --regularize_grads --main_task_lambda 0.1

#    else
#        echo ${model_name} does not exist
#    fi
}

#reg_bert ml wl reg_method reg_lambda batch_size epochs aggregation_strategy
#reg_bert 11 3 linear_cka 1 128
#reg_bert 11 3 linear_cka 7 128
#reg_bert 11 3 linear_cka 12 128
#reg_bert 11 3 linear_cka 1 64 6
#reg_bert 11 3 linear_cka 0.1 128 12
#reg_bert 11 3 linear_cka 50 128 12
lambdas=(1)
methods=('cos_cor')
#agg_strategies=('mean')
#for agg_strategy in ${agg_strategies[@]}; do
wls=(1)
mls=(11)
exit
for ml in ${mls[@]}; do
    for cwl in ${wls[@]}; do
        for method in ${methods[@]}; do
            for lambda in ${lambdas[@]}; do
                reg_bert ${ml} ${cwl} ${method} ${lambda} 128 10
            done
        done
    done
done
