#!/bin/bash


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
	    eval_bs=64
	else
	    grads_suffix=""
	    grads_arguments=""
	    eval_bs=128
	fi
	bias_type='lexical'
	reg_only_biased=true
	reg_only_biased_arg=''
	reg_only_biased_identifier=''
	if [ ${bias_type} == 'lexical' ]; then
	    #This checkpoint is best on HANS
        #weak_model_dir=runs/extrensic_debiasing/lexical_bias/bert_64_remove_confident_correct/checkpoint-52551
        # This checkpoint has lowest validation loss.
        weak_model_dir="runs/extrensic_debiasing/lexical_bias/bert_64_remove_confident_correct/checkpoint-11678"
        run_name_identifier='lexicalDeBias'
        bias_dir='lexical_bias_debiasing'
        indices_arg=" --evaluate_on_hans --indices_dir data/lexical_bias_splits/validation_matched
        data/lexical_bias_splits/validation_matched/confidence_based_0.65"
        tags_arg='--tags lexical_bias'
        if [ ${reg_only_biased} == "true" ]; then
            reg_only_biased_arg='--regularize_only_biased --bias_indices data/lexical_bias_splits/train/confidence_based_0.65/train_biased_correct_indices.bin'
            reg_only_biased_identifier='STOB_'
            bias_dir='lexical_bias_debiasing/reg_OB'
        fi
    elif [ ${bias_type} == 'hypothesis' ]; then
        # This checkpoint has lowest validation loss.
        weak_model_dir="runs/extrensic_debiasing/hypothesis_bias/bert_64_0.6_hypothesis_extrensic_debiasing/checkpoint-7044"
        #This checkpoint is best on when evaluating the difference of accuracy between biased and anti-biased subsets of MNLI validation matched.
        #weak_model_dir=runs/extrensic_debiasing/hypothesis_bias/bert_64_0.6_hypothesis_extrensic_debiasing/checkpoint-21132
        run_name_identifier='HypoDeBias'
        bias_dir='hypothesis_bias_debiasing/0.6_confidence'
        indices_arg="--indices_dir data/mnli_hypothesis_only_hard \
        data/hypothesis_bias_splits/5_folds_0.8_confidence/validation_matched --mismatched_indices_dir data/hypothesis_bias_splits/validation_mismatched"
        tags_arg='--tags hypothesis_only_bias'
        if [ ${reg_only_biased} == "true" ]; then
            reg_only_biased_arg='--regularize_only_biased --bias_indices data/hypothesis_bias_splits/5_folds_0.6_confidence/train/train_biased_correct_indices.bin'
            reg_only_biased_identifier='STOB_'
            bias_dir='hypothesis_bias_debiasing/reg_OB/0.6_confidence'
        fi
	fi
	echo $ml-$wl-${regularization_method} batch_size: ${batch_size}, regularization_lambda: ${regularization_lambda}, regularize_grads: ${regularize_grads};
    #seeds=(254 420 638 401 42)
    #seeds=(638 420 42)
    seeds=(420 42 638)
    layers_arg="9M-10M-11M_9-10-11"
    regularized_layers="bert.encoder.layer.9 bert.encoder.layer.10 bert.encoder.layer.11"
#     layers_arg="eM-1M-2M_e-1-2"
#    regularized_layers="bert.embeddings bert.encoder.layer.1 bert.encoder.layer.2"
	main_task_lambda=1

    for seed in ${seeds[@]}; do
	    model_name=bert_EDBXE_${reg_only_biased_identifier}${run_name_identifier}_${regularization_method}_${layers_arg}_lambda_${main_task_lambda}_${regularization_lambda}_${batch_size}_sim${grads_suffix}
        nlp_sbatch plato1 run_glue.py --report_to wandb --model_name_or_path bert-base-uncased \
        --output_dir runs/${bias_dir}/${regularization_method}/${model_name}_s${seed} --num_train_epochs ${epochs} --learning_rate 2e-5 \
         --task_name mnli --do_train --do_eval --max_seq_length 128 --per_device_train_batch_size ${batch_size} --per_device_eval_batch_size ${eval_bs} \
         --weak_model_layer ${regularized_layers} --weak_model_path ${weak_model_dir} --regularized_layers ${regularized_layers} \
         --regularization_method ${regularization_method} --evaluation_strategy half_epoch --save_strategy epoch \
         --regularization_lambda ${regularization_lambda} --run_name ${model_name} --weight_decay 0.1 --warmup_ratio 0.1 --no_pad_to_max_length \
         --logging_first_step --regularized_tokens all --token_aggregation_strategy mean --enforce_similarity \
         ${grads_arguments} ${indices_arg} ${tags_arg} ${reg_only_biased_arg} --main_task_lambda ${main_task_lambda} \
         --bias_sampling_strategy stochastic --seed ${seed} --wandb_group
         echo "---------"
    done
}

#exit
lambdas=(100)
methods=('abs_cos_cor')
for method in ${methods[@]}; do
    for lambda in ${lambdas[@]}; do
        reg_bert -1 -1 ${method} ${lambda} 64 10
    done
done
