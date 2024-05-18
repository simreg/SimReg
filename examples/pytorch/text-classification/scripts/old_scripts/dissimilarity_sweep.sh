#!/bin/bash

# Dissimilarity for representations

seeds=(42 420 638)
run_name="bert_LexicalModel_STOB_lexicalDeBias_abs_cos_cor_9M-10M-11M_2-4-4_lambda_1_10_64"
weak_model_layer="classifier.classifier.2 classifier.classifier.4 classifier.classifier.4"
weak_model_path="runs/bert_lexical_bias"
main_model_layers="bert.encoder.layer.9 bert.encoder.layer.10 bert.encoder.layer.11"
for seed in ${seeds[@]}; do
    nlp_sbatch nlp-a40-1 run_glue.py --report_to wandb --model_name_or_path bert-base-uncased \
    --output_dir runs/lexical_bias_debiasing/reg_OB/abs_cos_cor/${run_name}_s${seed} \
    --num_train_epochs 10 --learning_rate 2e-5 --per_device_eval_batch_size 256 --task_name mnli --do_train --do_eval --max_seq_length 128 \
    --per_device_train_batch_size 64 --weak_model_layer ${weak_model_layer} --weak_model_path ${weak_model_path} \
    --regularized_layers ${main_model_layers} --regularization_method abs_cos_cor --evaluation_strategy half_epoch --save_strategy epoch \
    --regularization_lambda 10 --run_name ${run_name} --weight_decay 0.1 --evaluate_on_hans \
    --warmup_ratio 0.1 --no_pad_to_max_length --logging_first_step --regularized_tokens all --token_aggregation_strategy mean \
    --indices_dir data/lexical_bias_splits/validation_matched data/lexical_bias_splits/validation_matched/confidence_based_0.65 --tags lexical_bias \
     --regularize_only_biased --bias_indices data/lexical_bias_splits/train/confidence_based_0.65/train_biased_correct_indices.bin \
     --main_task_lambda 1 --bias_sampling_strategy stochastic --seed ${seed} --wandb_group --no_remove_unused_columns
     echo
done

seeds=(420 1997)
for seed in ${seeds[@]}; do
    run_name="bert_LexicalModel_lexicalDeBias_abs_cos_cor_eM-0M-6M-11M_2-2-2-2_lambda_1_10_64"
    nlp_sbatch plato1 run_glue.py --report_to wandb --model_name_or_path bert-base-uncased --output_dir runs/lexical_bias_debiasing/linear_cka/${run_name}_s${seed} --num_train_epochs 10 --learning_rate 2e-5 --per_device_eval_batch_size 128 --task_name mnli --do_train --do_eval --max_seq_length 128 --per_device_train_batch_size 64 --weak_model_layer classifier.classifier.2 classifier.classifier.2 classifier.classifier.2 classifier.classifier.2 --weak_model_path runs/bert_lexical_bias --regularized_layers bert.embeddings bert.encoder.layer.0 bert.encoder.layer.6 bert.encoder.layer.11 --regularization_method abs_cos_cor --evaluation_strategy half_epoch --save_strategy no --regularization_lambda 10 --run_name ${run_name} --weight_decay 0.1 --warmup_ratio 0.1 --no_pad_to_max_length --logging_first_step --regularized_tokens all --token_aggregation_strategy mean --evaluate_on_hans --tags lexical_bias --main_task_lambda 1 --seed ${seed} --wandb_group --no_remove_unused_columns
done


seeds=(420 638 42)
run_name="bert_UB2_STOB_lexicalDeBias_abs_cos_cor_eM-0M-6M-11M_e-0-6-10-11_lambda_1_100_64"
weak_model_path="runs/utama_biased_bert_64_2"
regularized_layers="bert.embeddings bert.encoder.layer.0 bert.encoder.layer.6 bert.encoder.layer.10 bert.encoder.layer.11"
for seed in ${seeds[@]}; do
    nlp_sbatch plato1 run_glue.py --report_to wandb --model_name_or_path bert-base-uncased \
    --output_dir runs/lexical_bias_debiasing/reg_OB/abs_cos_cor/${run_name}_s${seed} \
    --num_train_epochs 10 --learning_rate 2e-5 --per_device_eval_batch_size 64 --task_name mnli --do_train --do_eval --max_seq_length 128 \
    --per_device_train_batch_size 64 --weak_model_layer ${regularized_layers} --weak_model_path ${weak_model_path} \
    --regularized_layers ${regularized_layers} --regularization_method abs_cos_cor --evaluation_strategy half_epoch --save_strategy epoch \
    --regularization_lambda 100 --run_name ${run_name} --weight_decay 0.1 --evaluate_on_hans \
    --warmup_ratio 0.1 --no_pad_to_max_length --logging_first_step --regularized_tokens all --token_aggregation_strategy mean \
    --indices_dir data/lexical_bias_splits/validation_matched data/lexical_bias_splits/validation_matched/confidence_based_0.65 --tags lexical_bias \
     --regularize_only_biased --bias_indices data/lexical_bias_splits/train/confidence_based_0.65/train_biased_correct_indices.bin \
     --main_task_lambda 1 --bias_sampling_strategy stochastic --seed ${seed} --wandb_group
done




# For gradients

seeds=(420 638 42)
run_name="bert_LexicalModel_STOB_lexicalDeBias_linear_cka_eM_2_lambda_1_10_64_GReg"
weak_model_layer="classifier.classifier.2"
weak_model_path="runs/bert_lexical_bias"
main_model_layers="bert.embeddings"
for seed in ${seeds[@]}; do
    nlp_sbatch nlp-a40-1 run_glue.py --report_to wandb --model_name_or_path bert-base-uncased \
    --output_dir runs/lexical_bias_debiasing/reg_OB/linear_cka/${run_name}_s${seed} \
    --num_train_epochs 10 --learning_rate 2e-5 --per_device_eval_batch_size 64 --task_name mnli --do_train --do_eval --max_seq_length 128 \
    --per_device_train_batch_size 64 --weak_model_layer ${weak_model_layer} --weak_model_path ${weak_model_path} \
    --regularized_layers ${main_model_layers} --regularization_method linear_cka --evaluation_strategy half_epoch --save_strategy epoch \
    --regularization_lambda 10 --run_name ${run_name} --weight_decay 0.1 --evaluate_on_hans \
    --warmup_ratio 0.1 --no_pad_to_max_length --logging_first_step --regularized_tokens all --token_aggregation_strategy mean \
    --regularize_only_biased --bias_indices data/lexical_bias_splits/train/confidence_based_0.65/train_biased_correct_indices.bin \
    --bias_sampling_strategy stochastic --seed ${seed} --wandb_group --no_remove_unused_columns --regularize_grads --tags lexical_bias
done


seeds=(420 638 42)
run_name="bert_TB_STOB_lexicalDeBias_abs_cos_cor_eM_e_lambda_1_10_64_GReg"
weak_model_layer="bert.embeddings"
weak_model_path="runs/tiny_bert_mnli"
main_model_layers="bert.embeddings"
for seed in ${seeds[@]}; do
    nlp_sbatch nlp-a40-1 run_glue.py --report_to wandb --model_name_or_path bert-base-uncased \
    --output_dir runs/lexical_bias_debiasing/reg_OB/abs_cos_cor/${run_name}_s${seed} \
    --num_train_epochs 10 --learning_rate 2e-5 --per_device_eval_batch_size 64 --task_name mnli --do_train --do_eval --max_seq_length 128 \
    --per_device_train_batch_size 64 --weak_model_layer ${weak_model_layer} --weak_model_path ${weak_model_path} \
    --regularized_layers ${main_model_layers} --regularization_method abs_cos_cor --evaluation_strategy half_epoch --save_strategy epoch \
    --regularization_lambda 10 --run_name ${run_name} --weight_decay 0.1 --evaluate_on_hans \
    --warmup_ratio 0.1 --no_pad_to_max_length --logging_first_step --regularized_tokens all --token_aggregation_strategy mean \
    --indices_dir data/lexical_bias_splits/validation_matched data/lexical_bias_splits/validation_matched/confidence_based_0.65 --tags lexical_bias \
     --regularize_only_biased --bias_indices data/lexical_bias_splits/train/confidence_based_0.65/train_biased_correct_indices.bin \
     --main_task_lambda 1 --bias_sampling_strategy stochastic --seed ${seed} --wandb_group --regularize_grads
done


seeds=(420 638 42)
for seed in ${seeds[@]}; do
    run_name="debertav3_LexicalModel_STOB_lexicalDeBias_abs_cos_cor_1M_2_lambda_1_5_64_GReg"
    output_dir="runs/debertav3/lexical_bias_debiasing/dissimilarity/abs_cos_cor/${run_name}_s${seed}"
    weak_model_path="runs/bert_lexical_bias"
    nlp_sbatch nlp-a40-1 run_glue.py --report_to wandb --model_name_or_path microsoft/deberta-v3-base \
    --output_dir ${output_dir} --num_train_epochs 10 --learning_rate 2e-5 --per_device_eval_batch_size 64 --task_name mnli --do_train --do_eval \
    --max_seq_length 128 --per_device_train_batch_size 64 --weak_model_layer classifier.classifier.2 --weak_model_path runs/bert_lexical_bias \
    --regularized_layers deberta.encoder.layer.1 --regularization_method abs_cos_cor --evaluation_strategy half_epoch --save_strategy no \
    --regularization_lambda 5 --run_name ${run_name} --weight_decay 0.1 --warmup_ratio 0.1 --no_pad_to_max_length --logging_first_step \
    --regularized_tokens all --token_aggregation_strategy mean --regularize_grads --evaluate_on_hans --tags lexical_bias --regularize_only_biased \
    --bias_indices data/lexical_bias_splits/train/confidence_based_0.65/train_biased_correct_indices.bin --bias_sampling_strategy stochastic \
    --seed ${seed} --wandb_group --no_remove_unused_columns
    echo
done



seeds=(420 638 42)
for seed in ${seeds[@]}; do
    run_name="debertav3_LexicalModel_STOB_lexicalDeBias_abs_cos_cor_10M-11M_2-2_lambda_1_5_64"
    output_dir="runs/debertav3/lexical_bias_debiasing/dissimilarity/abs_cos_cor/${run_name}_s${seed}"
    weak_model_path="runs/bert_lexical_bias"
    nlp_sbatch nlp-a40-1 run_glue.py --report_to wandb --model_name_or_path microsoft/deberta-v3-base \
    --output_dir ${output_dir} --num_train_epochs 10 --learning_rate 2e-5 --per_device_eval_batch_size 64 --task_name mnli --do_train --do_eval \
    --max_seq_length 128 --per_device_train_batch_size 64 --weak_model_layer classifier.classifier.2 classifier.classifier.2 \
    --weak_model_path ${weak_model_path} --regularized_layers deberta.encoder.layer.10 deberta.encoder.layer.11 --regularization_method abs_cos_cor \
    --evaluation_strategy half_epoch --save_strategy no \
    --regularization_lambda 5 --run_name ${run_name} --weight_decay 0.1 --warmup_ratio 0.1 --no_pad_to_max_length --logging_first_step \
    --regularized_tokens all --token_aggregation_strategy mean --regularize_grads --evaluate_on_hans --tags lexical_bias --regularize_only_biased \
    --bias_indices data/lexical_bias_splits/train/confidence_based_0.65/train_biased_correct_indices.bin --bias_sampling_strategy stochastic \
    --seed ${seed} --wandb_group --no_remove_unused_columns
    echo
done



    nlp_sbatch nlp-a40-1 run_glue.py --report_to wandb --model_name_or_path microsoft/deberta-base --output_dir ${output_dir} --num_train_epochs 10 \
    --learning_rate 2e-5 --per_device_eval_batch_size 64 --task_name mnli --do_train --do_eval --max_seq_length 128 --per_device_train_batch_size 64 \
    --weak_model_layer classifier.classifier.2 --weak_model_path ${weak_model_path} --regularized_layers deberta.embeddings \
    --regularization_method abs_cos_cor --evaluation_strategy half_epoch --save_strategy epoch --regularization_lambda 10 --run_name ${run_name} \
     --weight_decay 0.1 --warmup_ratio 0.1 --no_pad_to_max_length --logging_first_step --regularized_tokens all --token_aggregation_strategy mean \
     --enforce_similarity --evaluate_on_hans --tags lexical_bias --bias_sampling_strategy stochastic --wandb_group --regularize_grads \
     --regularize_only_biased --bias_indices  data/lexical_bias_splits/train/confidence_based_0.65/train_biased_correct_indices.bin  --seed ${seed} \
     --no_remove_unused_columns



#    DeBERTa V1 --------------------------------------------------------------------------------------------------------------------------------------

seed=54041
run_name="deberta_LexicalModel_STOB_lexicalDeBias_abs_cos_cor_eM_2_lambda_1_10_64_GReg"
nlp_sbatch galileo1 run_glue.py --report_to wandb --model_name_or_path microsoft/deberta-base --output_dir runs/deberta/lexical_bias_debiasing/dissimilarity/abs_cos_cor/${run_name}_s${seed} --num_train_epochs 10 --learning_rate 2e-5 --per_device_eval_batch_size 64 --task_name mnli --do_train --do_eval --max_seq_length 128 --per_device_train_batch_size 64 --weak_model_layer classifier.classifier.2 --weak_model_path runs/bert_lexical_bias --regularized_layers deberta.embeddings --regularization_method abs_cos_cor --evaluation_strategy half_epoch --save_strategy no --regularization_lambda 10 --run_name ${run_name} --weight_decay 0.1 --warmup_ratio 0.1 --logging_first_step --regularized_tokens all --token_aggregation_strategy mean --regularize_grads --evaluate_on_hans --tags lexical_bias --regularize_only_biased --bias_indices data/lexical_bias_splits/train/confidence_based_0.65/train_biased_correct_indices.bin --main_task_lambda 1 --bias_sampling_strategy stochastic --seed ${seed} --wandb_group --no_remove_unused_columns --separate_weak_tokenization



seed=1352
run_name="deberta_LexicalModel_STOB_lexicalDeBias_abs_cos_cor_eM-10M-11M_2-2-2_lambda_1_10_64"
nlp_sbatch galileo2 run_glue.py --report_to wandb --model_name_or_path microsoft/deberta-base --output_dir runs/deberta/lexical_bias_debiasing/dissimilarity/abs_cos_cor/${run_name}_s${seed} --num_train_epochs 10 --learning_rate 2e-5 --per_device_eval_batch_size 256 --task_name mnli --do_train --do_eval --max_seq_length 128 --per_device_train_batch_size 64 --weak_model_layer bert.encoder.layer.2 bert.encoder.layer.2 bert.encoder.layer.2 --weak_model_path runs/bert_lexical_bias --regularized_layers deberta.embeddings deberta.encoder.layer.10 deberta.encoder.layer.11 --regularization_method abs_cos_cor --evaluation_strategy half_epoch --save_strategy no --regularization_lambda 10 --run_name deberta_LexicalModel_STOB_lexicalDeBias_abs_cos_cor_eM-10M-11M_2-2-2_lambda_1_10_64 --weight_decay 0.1 --warmup_ratio 0.1 --logging_first_step --regularized_tokens all --token_aggregation_strategy mean --evaluate_on_hans --tags lexical_bias --regularize_only_biased --bias_indices data/lexical_bias_splits/train/confidence_based_0.65/train_biased_correct_indices.bin --main_task_lambda 1 --bias_sampling_strategy stochastic --seed ${seed} --wandb_group --separate_weak_tokenization --no_remove_unused_columns







#    DeBERTa V3 --------------------------------------------------------------------------------------------------------------------------------------
        seed=501
        run_name="debertav3_LexicalModel_STOB_lexicalDeBias_abs_cos_cor_eM_2_lambda_1_10_64_GReg"
#        output_dir="${run_name}_s"
        nlp_sbatch nlp-a40-1 run_glue.py --report_to wandb --model_name_or_path microsoft/deberta-v3-base \
         --output_dir runs/debertav3/lexical_bias_debiasing/dissimilarity/abs_cos_cor/${run_name}_s${seed} --num_train_epochs 10 --learning_rate 2e-5 \
        --per_device_eval_batch_size 64 --task_name mnli --do_train --do_eval --max_seq_length 128 --per_device_train_batch_size 64 \
        --separate_weak_tokenization \
        --weak_model_layer classifier.classifier.2 --weak_model_path runs/bert_lexical_bias --regularized_layers deberta.embeddings \
        --regularization_method abs_cos_cor --evaluation_strategy half_epoch --save_strategy no --regularization_lambda 10 --run_name ${run_name} \
          --weight_decay 0.1 --warmup_ratio 0.1 --logging_first_step --regularized_tokens all --token_aggregation_strategy mean --regularize_grads \
          --evaluate_on_hans --tags lexical_bias --regularize_only_biased --bias_indices data/lexical_bias_splits/train/confidence_based_0.65/train_biased_correct_indices.bin --main_task_lambda 1 --bias_sampling_strategy stochastic --seed ${seed} --wandb_group --no_remove_unused_columns



       seed=420
        run_name="debertav3_LexicalModel_STOB_lexicalDeBias_abs_cos_cor_eM-10M-11M_2-2-2_lambda_1_10_64"
#        output_dir="${run_name}_s"
        nlp_sbatch nlp-a40-1 run_glue.py --report_to wandb --model_name_or_path microsoft/deberta-v3-base \
         --output_dir runs/debertav3/lexical_bias_debiasing/dissimilarity/abs_cos_cor/${run_name}_s${seed} --num_train_epochs 10 --learning_rate 2e-5 \
        --per_device_eval_batch_size 64 --task_name mnli --do_train --do_eval --max_seq_length 128 --per_device_train_batch_size 64 \
        --separate_weak_tokenization \
        --weak_model_layer classifier.classifier.2 classifier.classifier.2 classifier.classifier.2 --weak_model_path runs/bert_lexical_bias \
        --regularized_layers deberta.embeddings deberta.encoder.layer.10 deberta.encoder.layer.11 \
        --regularization_method abs_cos_cor --evaluation_strategy half_epoch --save_strategy no --regularization_lambda 10 --run_name ${run_name} \
          --weight_decay 0.1 --warmup_ratio 0.1 --logging_first_step --regularized_tokens all --token_aggregation_strategy mean \
          --evaluate_on_hans --tags lexical_bias --regularize_only_biased --bias_indices data/lexical_bias_splits/train/confidence_based_0.65/train_biased_correct_indices.bin --main_task_lambda 1 --bias_sampling_strategy stochastic --seed ${seed} --wandb_group --no_remove_unused_columns


# QQP ------------------------------------------------------------------------------------------------------------------------------------------------

seeds=(9354 4365 1344)
regularization_method="abs_cos_cor"
run_name="bert_LexicalModel_STOB_lexicalDeBias_${regularization_method}_6M-7M-9M-10M-11M_2-2-2-4-4_lambda_1_10_64"
for seed in ${seeds[@]}; do
    nlp_sbatch nlp-a40-1 run_glue.py --report_to wandb --model_name_or_path microsoft/deberta-v3-base --output_dir qqp_runs/debertav3/lexical_bias/reg_OB/0.6_confidence/${regularization_method}/${run_name}_s${seed} --num_train_epochs 10 --learning_rate 2e-5 --per_device_eval_batch_size 128 --task_name qqp --do_train --do_eval --max_seq_length 128 --per_device_train_batch_size 64 --weak_model_layer classifier.2 classifier.2 classifier.2 classifier.4 classifier.4 --weak_model_path qqp_runs/lexicalModel --regularized_layers deberta.encoder.layer.6 deberta.encoder.layer.7 deberta.encoder.layer.9 deberta.encoder.layer.10 deberta.encoder.layer.11 --regularization_method ${regularization_method} --evaluation_strategy half_epoch --save_strategy no --regularization_lambda 10 --run_name ${run_name} --weight_decay 0.1 --warmup_ratio 0.1 --no_pad_to_max_length --logging_first_step --regularized_tokens all --token_aggregation_strategy mean --tags lexical_bias --regularize_only_biased --bias_indices data/qqp_lexical_bias/0.6_confidence/train/train_biased_correct_indices.bin --main_task_lambda 1 --bias_sampling_strategy stochastic --seed ${seed} --wandb_group --no_remove_unused_columns
done

seeds=(9354 4365 1344)
regularization_method="linear_cka"
run_name="bert_LexicalModel_STOB_lexicalDeBias_${regularization_method}_eM-0M-0-2_lambda_1_10_64_GReg"
for seed in ${seeds[@]}; do
    nlp_sbatch galileo2 run_glue.py --report_to wandb --model_name_or_path bert-base-uncased --output_dir qqp_runs/lexical_bias/reg_OB/0.6_confidence/${regularization_method}/${run_name}_s${seed} --num_train_epochs 10 --learning_rate 2e-5 --per_device_eval_batch_size 64 --task_name qqp --do_train --do_eval --max_seq_length 128 --per_device_train_batch_size 64 --weak_model_layer classifier.0 classifier.2 --weak_model_path qqp_runs/lexicalModel --regularized_layers bert.embeddings bert.encoder.layer.0 --regularization_method ${regularization_method} --evaluation_strategy half_epoch --save_strategy no --regularization_lambda 10 --run_name ${run_name} --weight_decay 0.1 --warmup_ratio 0.1 --no_pad_to_max_length --logging_first_step --regularized_tokens all --token_aggregation_strategy mean --tags lexical_bias --regularize_only_biased --bias_indices data/qqp_lexical_bias/0.6_confidence/train/train_biased_correct_indices.bin --main_task_lambda 1 --bias_sampling_strategy stochastic --seed ${seed} --wandb_group --no_remove_unused_columns --regularize_grads
done