#!/bin/bash


#weak_model_dir=fever_runs/extrensic_debiasing/claim_bias/bert_64_0.7_claim_extrensic_debiasing
# This checkpoint has lowest validation loss.
#weak_model_dir=fever_runs/extrensic_debiasing/claim_bias/bert_64_0.8_claim_extrensic_debiasing/checkpoint-3360

# Lowest validation loss
#weak_model_dir=fever_runs/extrensic_debiasing/unknown_bias/bert_64_0.9_unknown_extrensic_debiasing/checkpoint-5976
#weak_model_dir=fever_runs/extrensic_debiasing/unknown_bias/bert_64_0.8_unknown_extrensic_debiasing/checkpoint-2808


cd ~/research_project/examples/pytorch/text-classification/
function reg_bert() {
	ml=$1
	wl=$2
	regularization_method=$3
	regularization_lambda=$4
	batch_size=$5
    epochs=$6

	regularize_grads=true

	if [ "${regularize_grads}" == "true" ]; then
	    grads_suffix="_GReg"
	    grads_arguments="--regularize_grads"
	    eval_bs=64
	else
	    grads_suffix=""
	    grads_arguments=""
	    eval_bs=256
	fi
	
	bias_type='unknown'
	reg_only_biased=true
	reg_only_biased_arg=''
	reg_only_biased_identifier=''
	if [ ${bias_type} == 'unknown' ]; then
	    run_name_identifier='0.8_UNB'
	    weak_model_dir="fever_runs/extrensic_debiasing/unknown_bias/bert_64_0.8_unknown_extrensic_debiasing/checkpoint-2808"
        bias_dir='unknown_bias_debiasing'
        indices_arg="--indices_dir data/fever_claim_bias_splits/validation_0.8_confidence data/fever_unknown_bias_splits/validation"
        tags_arg='--tags unknown_bias'
        if [ ${reg_only_biased} == "true" ]; then
            reg_only_biased_arg='--regularize_only_biased --bias_indices data/fever_unknown_bias_splits/5_folds_0.8_confidence/train/train_biased_correct_indices.bin'
            reg_only_biased_identifier='STOB_'
            bias_dir='unknown_bias_debiasing/reg_OB'
        fi
	elif [ ${bias_type} == 'claim' ]; then
	    weak_model_dir="fever_runs/extrensic_debiasing/claim_bias/bert_64_0.8_claim_extrensic_debiasing/checkpoint-3360"
        run_name_identifier='ClaimDeBias'
        bias_dir='claim_bias_debiasing'
        indices_arg="--indices_dir data/fever_claim_bias_splits/validation_0.8_confidence"
        tags_arg='--tags claim_bias'
        if [ ${reg_only_biased} == "true" ]; then
            reg_only_biased_arg='--regularize_only_biased --bias_indices data/fever_claim_bias_splits/5_folds_0.8_confidence/train_biased_correct_indices.bin'
            reg_only_biased_identifier='STOB_'
            bias_dir='claim_bias_debiasing/reg_OB'
        fi
	fi

	echo $ml-$wl-${regularization_method} batch_size: ${batch_size}, regularization_lambda: ${regularization_lambda}, regularize_grads: ${regularize_grads};
	#seeds=(254 420 638 401 42)
	seeds=(420 638)
#	layers_arg="9M-10M-11M_9-10-11"
#	regularized_layers="bert.encoder.layer.9 bert.encoder.layer.10 bert.encoder.layer.11"
	layers_arg="eM-0M-6M-10M-11M_e-0-6-10-11"
	regularized_layers="bert.embeddings bert.encoder.layer.0 bert.encoder.layer.6 bert.encoder.layer.10 bert.encoder.layer.11"
	main_task_lambda=1
    for seed in ${seeds[@]}; do
	    model_name=bert_${reg_only_biased_identifier}${run_name_identifier}_${regularization_method}_${layers_arg}_lambda_${main_task_lambda}_${regularization_lambda}_${batch_size}_sim${grads_suffix}
        echo nlp_sbatch nlp-a40-1 run_glue.py --report_to wandb --model_name_or_path bert-base-uncased \
        --output_dir fever_runs/${bias_dir}/${regularization_method}/${model_name}_s${seed} \
         --task_name fever --do_train --do_eval --max_seq_length 128 --per_device_train_batch_size ${batch_size} --per_device_eval_batch_size ${eval_bs} \
         --learning_rate 2e-5 --num_train_epochs ${epochs} \
         --weak_model_layer ${regularized_layers} --weak_model_path ${weak_model_dir} --regularized_layers ${regularized_layers} \
         --regularization_method ${regularization_method} --evaluation_strategy half_epoch --save_strategy epoch \
         --regularization_lambda ${regularization_lambda} --run_name ${model_name} --weight_decay 0.1 --warmup_ratio 0.1 --no_pad_to_max_length \
         --logging_first_step --regularized_tokens all --token_aggregation_strategy mean --enforce_similarity \
         ${grads_arguments} ${indices_arg} ${tags_arg} ${reg_only_biased_arg} --main_task_lambda ${main_task_lambda} \
         --bias_sampling_strategy stochastic --seed ${seed} --wandb_group
         echo
    done
    echo "---------"
}

lambdas=(10 100)
methods=('linear_cka' 'abs_cos_cor')
for method in ${methods[@]}; do
    for lambda in ${lambdas[@]}; do
        reg_bert -1 -1 ${method} ${lambda} 64 10
    done
done

