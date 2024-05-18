#!/bin/bash

#weak_model_dir=runs/extrensic_debiasing/unknown_bias/bert_64_0.8_unknown_extrensic_debiasing/checkpoint-42845
weak_model_dir=runs/extrensic_debiasing/unknown_bias/bert_64_0.8_unknown_extrensic_debiasing/checkpoint-9020


cd ~/research_project/examples/pytorch/text-classification/
function reg_bert() {
	ml=$1
	wl=$2
	regularization_method=$3
	regularization_lambda=$4
	batch_size=$5
    epochs=$6

	regularize_grads=false
	if [ "${regularize_grads}" == "true" ]; then
	    grads_suffix="_GReg"
	    grads_arguments="--regularize_grads"
	else
	    grads_suffix=""
	    grads_arguments=""
	fi

	reg_only_biased=true
	reg_only_biased_identifier=''
    run_name_identifier='UNB'
    bias_dir='unknown_bias'
    tags_arg='--tags unknown_bias'
    indices_arg="--evaluate_on_hans --indices_dir data/lexical_bias_splits/validation_matched
        data/lexical_bias_splits/validation_matched/confidence_based_0.65 data/mnli_hypothesis_only_hard
        data/hypothesis_bias_splits/5_folds_0.8_confidence/validation_matched data/tinybert_splits/5_folds_0.8_confidence/validation_matched
        --mismatched_indices_dir data/hypothesis_bias_splits/validation_mismatched"

    if [ ${reg_only_biased} == "true" ]; then
        reg_only_biased_arg='--regularize_only_biased --bias_indices data/tinybert_splits/5_folds_0.8_confidence/train/train_biased_correct_indices.bin'
        reg_only_biased_identifier='STOB_'
        bias_dir="${bias_dir}/reg_OB/0.8_confidence"
    fi

	echo $ml-$wl-${regularization_method} batch_size: ${batch_size}, regularization_lambda: ${regularization_lambda}, regularize_grads: ${regularize_grads};
	# 638 420 42
    seeds=(638 420 42)
    layers_arg="bert.encoder.layer.9 bert.encoder.layer.10 bert.encoder.layer.11"
    regularized_layers="9M-10M-11M_9-10-11"
	main_task_lambda=1
    for seed in ${seeds[@]}; do
	    model_name=bert_EDBXE_${reg_only_biased_identifier}${run_name_identifier}_${regularization_method}_${regularized_layers}_lambda_${main_task_lambda}_${regularization_lambda}_${batch_size}_sim${grads_suffix}
        nlp_sbatch plato1 run_glue.py --report_to wandb --model_name_or_path bert-base-uncased \
        --output_dir runs/${bias_dir}/${regularization_method}/${model_name}_s${seed} --save_strategy epoch \
         --task_name mnli --do_train --do_eval --max_seq_length 128 --per_device_train_batch_size ${batch_size} --per_device_eval_batch_size ${batch_size} \
         --learning_rate 2e-5 --num_train_epochs ${epochs} --weak_model_layer ${layers_arg} --weak_model_path ${weak_model_dir} \
         --regularized_layers ${layers_arg} --regularization_method ${regularization_method} --evaluation_strategy half_epoch \
         --regularization_lambda ${regularization_lambda} --run_name ${model_name} --weight_decay 0.1 --warmup_ratio 0.1 --no_pad_to_max_length \
         --logging_first_step --regularized_tokens all --token_aggregation_strategy mean --enforce_similarity \
         ${grads_arguments} ${indices_arg} ${tags_arg} ${reg_only_biased_arg} --main_task_lambda ${main_task_lambda} \
         --bias_sampling_strategy stochastic --seed ${seed} --wandb_group
         echo "---------"
    done
}

lambdas=(1 100 10)
methods=('linear_cka')
for method in ${methods[@]}; do
    for lambda in ${lambdas[@]}; do
        reg_bert -1 -1 ${method} ${lambda} 64 10
    done
done
