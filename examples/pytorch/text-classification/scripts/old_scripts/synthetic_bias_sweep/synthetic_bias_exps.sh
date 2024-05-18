#!/bin/bash

weak_model_dir=runs/bert_64_base

cd ~/research_project/examples/pytorch/text-classification/
function reg_bert() {
	ml=$1
	wl=$2
	regularization_method=$3
	regularization_lambda=$4
	batch_size=$5
    epochs=$6
    eval_bs=256

	regularize_grads=true
	if [ "${regularize_grads}" == "true" ]; then
	    grads_suffix="_GReg"
	    grads_arguments="--regularize_grads"
	    eval_bs=${batch_size}
	else
	    grads_suffix=""
	    grads_arguments=""
	fi
	increase_similarity="false"
    if [ "${increase_similarity}" == "true" ]; then
        sim_arg="--enforce_similarity"
        sim_dir="force_similarity"
        sim_name_arg="_sim"
    else
        sim_arg=""
        sim_dir="dissimilarity"
        sim_name_arg=""
    fi
	echo ${regularization_method} batch_size: ${batch_size}, regularization_lambda: ${regularization_lambda} regularize_grads:${regularize_grads};

    layers_arg="eM-0M-6M-10M-11M_e-0-6-10-11"
    regularized_layers="bert.embeddings bert.encoder.layer.0 bert.encoder.layer.6 bert.encoder.layer.10 bert.encoder.layer.11"
	model_name=bert_1_0.95_${regularization_method}_${layers_arg}_lambda_1_${regularization_lambda}_batch_${batch_size}${sim_name_arg}${grads_suffix}
	output_dir="runs/synthetic_bias_1_0.95/${sim_dir}/${regularization_method}/${model_name}"
    seeds=(401 638 42)
    for seed in ${seeds[@]}; do
        echo nlp_sbatch nlp-a40-1 run_glue.py --report_to wandb --model_name_or_path bert-base-uncased \
         --task_name mnli --do_train --do_eval --max_seq_length 128 --per_device_train_batch_size ${batch_size} \
         --per_device_eval_batch_size ${eval_bs} --learning_rate 2e-5 --num_train_epochs ${epochs} --output_dir "${output_dir}_s${seed}" \
         --weak_model_path ${weak_model_dir} --weak_model_layer ${regularized_layers} --regularized_layers ${regularized_layers} \
         --regularization_method ${regularization_method} --evaluation_strategy half_epoch --save_strategy no \
         --regularization_lambda ${regularization_lambda} --run_name ${model_name} --weight_decay 0.1 --warmup_ratio 0.1 \
         --logging_first_step --regularized_tokens all --token_aggregation_strategy mean ${sim_arg} ${grads_arguments} \
         --synthetic_bias_prevalence 1 --bias_correlation_prob 0.95 --no_pad_to_max_length --wandb_group --seed ${seed} --tags 0.95
         echo
    done
}

lambdas=(1 10 100)
methods=('linear_cka')
for method in ${methods[@]}; do
    for lambda in ${lambdas[@]}; do
        reg_bert -1 -1 ${method} ${lambda} 64 10
    done
done
