#!/bin/bash


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
	else
	    grads_suffix=""
	    grads_arguments=""
	fi

	reg_only_biased=true
	reg_only_biased_arg=""
	reg_only_biased_identifier=''
    bias_type="unknown"

	if [ ${bias_type} == 'lexical' ]; then
	    run_name_identifier='lexicalDeBias'
        bias_dir='lexical_bias'
        tags_arg='--tags lexical_bias'
        weak_model_dir="qqp_runs/extrensic_debiasing/lexical_bias/bert_64_0.6_lexical_extrensic_debiasing/checkpoint-6800"
        indices_arg=" --indices_dir data/qqp_lexical_bias/0.7_confidence/validation data/qqp_unknown_bias_splits/5_folds_tinybert_0.9_confidence_splits/validation"
        if [ ${reg_only_biased} == "true" ]; then
            reg_only_biased_arg='--regularize_only_biased --bias_indices data/qqp_lexical_bias/0.6_confidence/train/train_biased_correct_indices.bin'
            reg_only_biased_identifier='STOB_'
            bias_dir="${bias_dir}/reg_OB/0.6_confidence"
        fi
    elif [ ${bias_type} == 'unknown' ]; then
        run_name_identifier='UNB'
        bias_dir='unknown_bias'
        weak_model_dir="qqp_runs/extrensic_debiasing/unknown_bias/bert_64_0.8_unknown_extrensic_debiasing/checkpoint-7872"
        tags_arg='--tags unknown_bias'
        indices_arg=" --indices_dir data/qqp_unknown_bias_splits/5_folds_tinybert_0.9_confidence_splits/validation"
        if [ ${reg_only_biased} == "true" ]; then
            reg_only_biased_arg='--regularize_only_biased --bias_indices data/qqp_unknown_bias_splits/5_folds_tinybert_0.8_confidence_splits/train/train_biased_correct_indices.bin'
            reg_only_biased_identifier='STOB_'
            bias_dir="${bias_dir}/reg_OB/0.8_confidence"
        fi
	fi



	echo $ml-$wl-${regularization_method} batch_size: ${batch_size}, regularization_lambda: ${regularization_lambda}, regularize_grads: ${regularize_grads};
	seeds=(420 638 42)
#    layers_arg="9M-10M-11M_9-10-11"
#    regularized_layers="bert.encoder.layer.9 bert.encoder.layer.10 bert.encoder.layer.11"
#    layers_arg="11M_11"
#    regularized_layers="bert.encoder.layer.11"
    layers_arg="eM-0M-6M-10M-11M_e-0-6-10-11"
    regularized_layers="bert.embeddings bert.encoder.layer.0 bert.encoder.layer.6 bert.encoder.layer.10 bert.encoder.layer.11"
	main_task_lambda=1
    for seed in ${seeds[@]}; do
	    model_name=bert_${reg_only_biased_identifier}${run_name_identifier}_${regularization_method}_${layers_arg}_lambda_${main_task_lambda}_${regularization_lambda}_${batch_size}_sim${grads_suffix}
        echo nlp_sbatch nlp-a40-1 run_glue.py --report_to wandb --model_name_or_path bert-base-uncased \
        --output_dir qqp_runs/${bias_dir}/${regularization_method}/${model_name}_s${seed} \
         --task_name qqp --do_train --do_eval --max_seq_length 128 --per_device_train_batch_size ${batch_size} --per_device_eval_batch_size ${batch_size} \
         --learning_rate 2e-5 --num_train_epochs ${epochs} \
         --weak_model_layer ${regularized_layers} --weak_model_path ${weak_model_dir} --regularized_layers ${regularized_layers} \
         --regularization_method ${regularization_method} --evaluation_strategy half_epoch --save_strategy epoch \
         --regularization_lambda ${regularization_lambda} --run_name ${model_name} --weight_decay 0.1 --warmup_ratio 0.1 --no_pad_to_max_length \
         --logging_first_step --regularized_tokens all --token_aggregation_strategy mean --enforce_similarity \
         ${grads_arguments} ${indices_arg} ${tags_arg} ${reg_only_biased_arg} --main_task_lambda ${main_task_lambda} \
         --bias_sampling_strategy stochastic --seed ${seed} --wandb_group
         echo "---------"
    done
}


lambdas=(10)
methods=('abs_cos_cor')
for method in ${methods[@]}; do
    for lambda in ${lambdas[@]}; do
        reg_bert -1 -1 ${method} ${lambda} 64 10
    done
done
