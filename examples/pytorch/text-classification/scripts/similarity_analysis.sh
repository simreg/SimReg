#!/bin/bash

cd ~/research_project/examples/pytorch/text-classification/

mnli_trained="runs/bert_32_base_s638"
mnli2_trained="runs/bert_32_base_s420"
mnli3_trained="runs/bert_32_base_s254"
mnli4_trained="runs/bert_64_base_s254"
lexical_extrensicly_debiased="runs/extrensic_debiasing/lexical_bias/bert_64_remove_confident_correct"
unknwon_32_extrensicly_debiased="runs/extrensic_debiasing/unknown_bias/bert_32_0.8_unknown_extrensic_debiasing"
unknown_extrensicly_debiased="runs/extrensic_debiasing/unknown_bias/bert_64_0.8_unknown_extrensic_debiasing"
tinybert="runs/tiny_bert_mnli"
lexical_model="runs/bert_lexical_bias"
tb_bias_indices="data/tinybert_splits/5_folds_0.8_confidence/validation_matched/validation_matched_biased_correct_indices.bin"
hard_subset="/home/redaigbaria/research_project/examples/pytorch/text-classification/data/mnli_hypothesis_only_hard/utamas_paper_hypothesis_hard.bin"
hypothesis_extrensic="runs/extrensic_debiasing/hypothesis_bias/bert_64_0.6_hypothesis_extrensic_debiasing/checkpoint-7044"
regularized_model="runs/hypothesis_bias_debiasing/reg_OB/0.6_confidence/linear_cka/bert_EDBXE_STOB_HypoDeBias_linear_cka_9M-10M-11M_9-10-11_lambda_1_100_64_sim_s420"
dissimilarity_regularized_model="runs/hypothesis_bias_debiasing/reg_OB/dissimilarity/abs_cos_cor/bert_HOBert_STOB_HypoDeBias_abs_cos_cor_eM-0M-6M-10M-11M_e-0-6-10-11_lambda_1_10_64_s420"

nlp_sbatch nlp-a40-1 utils/similarity_analysis.py --main_model_path ${mnli4_trained} \
            --weak_model_path ${hypothesis_extrensic} \
             --output_dir mnli_similarity_analysis/validation_matched/mnli4_trained-hypothesis_extrensic --task_name mnli --max_seq_length 128 \
             --do_eval --per_device_batch_size 2048  --regularized_tokens all --aggregation_strategy mean \
             --weak_name 'f_g' --main_name "base"
nlp_sbatch nlp-a40-1 utils/similarity_analysis.py --main_model_path ${regularized_model} \
            --weak_model_path ${hypothesis_extrensic} \
             --output_dir mnli_similarity_analysis/validation_matched/hypothesis_regularized-hypothesis_extrensic --task_name mnli --max_seq_length 128 \
             --do_eval --per_device_batch_size 2048  --regularized_tokens all --aggregation_strategy mean \
             --weak_name 'f_g' --main_name 'reg'


#nlp_sbatch galileo1 utils/similarity_analysis.py --main_model_path ${dissimilarity_regularized_model} \
#            --weak_model_path ${mnli4_trained} \
#             --output_dir mnli_similarity_analysis/validation_matched/dissimilarity_regularized_model-mnli4_trained --task_name mnli --max_seq_length 128 \
#             --do_eval --per_device_batch_size 2048  --regularized_tokens all --aggregation_strategy mean
#
#nlp_sbatch galileo1 utils/similarity_analysis.py --main_model_path ${dissimilarity_regularized_model} \
#            --weak_model_path ${hypothesis_extrensic} \
#             --output_dir mnli_similarity_analysis/validation_matched/dissimilarity_regularized_model-hypothesis_extrensic --task_name mnli --max_seq_length 128 \
#             --do_eval --per_device_batch_size 2048  --regularized_tokens all --aggregation_strategy mean

#nlp_sbatch nlp-a40-1 utils/similarity_analysis.py --main_model_path ${mnli3_trained} \
#            --weak_model_path ${mnli2_trained} \
#             --output_dir mnli_similarity_analysis/validation_matched/mnli3_trained-mnli2_trained --task_name mnli --max_seq_length 128 \
#             --do_eval --per_device_batch_size 2048  --regularized_tokens all --aggregation_strategy mean

#nlp_sbatch nlp-a40-1 utils/similarity_analysis.py --main_model_path ${unknwon_32_extrensicly_debiased} \
#            --weak_model_path ${mnli_trained} \
#             --output_dir mnli_similarity_analysis/grads/validation_matched/UNB_32_debiased-mnli_trained --task_name mnli --max_seq_length 128 \
#             --do_eval --per_device_batch_size 128  --regularized_tokens all --aggregation_strategy mean --grads_sim

#
#nlp_sbatch nlp-a40-1 utils/similarity_analysis.py --main_model_path ${mnli_trained} \
#            --weak_model_path ${tinybert} \
#             --output_dir mnli_similarity_analysis/validation_matched/mnli_trained-tinybert --task_name mnli --max_seq_length 128 \
#             --do_eval --per_device_batch_size 2048  --regularized_tokens all --aggregation_strategy mean
#
#nlp_sbatch nlp-a40-1 utils/similarity_analysis.py --main_model_path ${mnli_trained} \
#            --weak_model_path ${tinybert} \
#             --output_dir mnli_similarity_analysis/grads/validation_matched/mnli_trained-tinybert --task_name mnli --max_seq_length 128 \
#             --do_eval --per_device_batch_size 128  --regularized_tokens all --aggregation_strategy mean --grads_sim

exit

#
#another_mnli="runs/bert_64_base_s401/checkpoint-12272"
#lexical_bias_indices="data/lexical_bias_splits/validation_matched/confidence_based_0.65/validation_matched_biased_correct_indices.bin"
#python utils/similarity_analysis.py --main_model_path ${another_mnli} \
#            --weak_model_path ${lexical_model} \
#             --output_dir grads_similarity_analysis/biased/mnli2-lexical_model --task_name mnli --max_seq_length 128 \
#             --do_eval --per_device_batch_size 256  --regularized_tokens all --aggregation_strategy mean --grads_sim \
#             --select_only_biased --bias_indices ${lexical_bias_indices}
#
#
#
#
##nlp_sbatch nlp-a40-1 utils/similarity_analysis.py --main_model_path ${mnli_trained} \
##            --weak_model_path ${tinybert} \
##             --output_dir grads_similarity_analysis/overall/mnli_trained-tinybert --task_name mnli --max_seq_length 128 \
##             --do_eval --per_device_batch_size 128  --regularized_tokens all --aggregation_strategy mean --grads_sim
#
#nlp_sbatch nlp-a40-1 utils/similarity_analysis.py --main_model_path ${extrensicly_debiased}/checkpoint-9020 \
#            --weak_model_path ${tinybert} \
#             --output_dir grads_similarity_analysis/overall/tb_extrensic_ckp_9020-tinybert --task_name mnli --max_seq_length 128 \
#             --do_eval --per_device_batch_size 128  --regularized_tokens all --aggregation_strategy mean --grads_sim
#

#exit
# ------------------
# Biased subset
#nlp_sbatch nlp-a40-1 utils/similarity_analysis.py --main_model_path ${mnli_trained} \
#            --weak_model_path ${extrensicly_debiased} \
#             --output_dir grads_similarity_analysis/biased/mnli_trained-tb_extrensic --task_name mnli --max_seq_length 128 \
#             --do_eval --per_device_batch_size 128  --regularized_tokens all --aggregation_strategy mean --grads_sim \
#             --select_only_biased  --bias_indices ${tb_bias_indices}
#
##nlp_sbatch nlp-a40-1 utils/similarity_analysis.py --main_model_path ${mnli_trained} \
##            --weak_model_path ${tinybert} \
##             --output_dir grads_similarity_analysis/biased/mnli_trained-tinybert --task_name mnli --max_seq_length 128 \
##             --do_eval --per_device_batch_size 128  --regularized_tokens all --aggregation_strategy mean --grads_sim \
##             --select_only_biased  --bias_indices ${tb_bias_indices}
#
#nlp_sbatch nlp-a40-1 utils/similarity_analysis.py --main_model_path ${extrensicly_debiased} \
#            --weak_model_path ${tinybert} \
#             --output_dir grads_similarity_analysis/biased/tb_extrensic-tinybert --task_name mnli --max_seq_length 128 \
#             --do_eval --per_device_batch_size 128  --regularized_tokens all --aggregation_strategy mean --grads_sim \
#             --select_only_biased  --bias_indices ${tb_bias_indices}

# ------------------
#
#python utils/similarity_analysis.py --main_model_path bert-base-uncased \
#            --weak_model_path ${extrensicly_debiased}/checkpoint-11678 \
#             --output_dir similarity_analysis/extrensic-bert_base_uncased --task_name mnli --max_seq_length 128 --do_eval --per_device_batch_size 2048  \
#            --regularized_tokens all --aggregation_strategy mean
#
#nlp_sbatch nlp-a40-1 utils/similarity_analysis.py --main_model_path ${extrensicly_debiased} \
#            --weak_model_path bert-base-uncased \
#             --output_dir similarity_analysis/extrensic-bert_base_uncased --task_name mnli --max_seq_length 128 --do_eval --per_device_batch_size 2048  \
#            --regularized_tokens all --aggregation_strategy mean
#
#nlp_sbatch nlp-a40-1 utils/similarity_analysis.py --main_model_path ${mnli_trained} \
#            --weak_model_path bert-base-uncased \
#             --output_dir similarity_analysis/mnli_trained_base_uncased --task_name mnli --max_seq_length 128 --do_eval --per_device_batch_size 2048  \
#            --regularized_tokens all --aggregation_strategy mean
#
#
#
#python utils/similarity_analysis.py --main_model_path bert-base-uncased \
#            --weak_model_path ${mnli_trained}/checkpoint-12000 \
#             --output_dir similarity_analysis/base_uncased-mnli_trained_fixed_aggregation --task_name mnli --max_seq_length 128 --do_eval \
#             --per_device_batch_size 2048  --regularized_tokens all --aggregation_strategy mean
#
#
#python utils/similarity_analysis.py --main_model_path ${debiased_bert_path} \
#            --weak_model_path ${extrensicly_debiased}/checkpoint-11678 \
#             --output_dir similarity_analysis/extrensic_debiasing-poe_debiasing --task_name mnli --max_seq_length 128 --do_eval \
#             --per_device_batch_size 2048  --regularized_tokens all --aggregation_strategy mean
#
#for ckp in ${extrensicly_debiased}/checkpoint-*; do
#    nlp_sbatch nlp-a40-1 utils/similarity_analysis.py --main_model_path ${mnli_trained}/checkpoint-12000 --weak_model_path ${ckp} \
#             --output_dir similarity_analysis/overall/mnli_trained_12000-extrensic_debiased_`basename ${ckp}` --task_name mnli \
#             --max_seq_length 128 --do_eval --per_device_batch_size 2048  --regularized_tokens all --aggregation_strategy mean
##             --remove_biased_samples_from_train \
##             --bias_indices data/lexical_bias_splits/validation_matched/confidence_based_0.65/validation_matched_biased_correct_indices.bin
#done
#
#
#lexical_bias_indices="data/lexical_bias_splits/validation_matched/confidence_based_0.65/validation_matched_biased_correct_indices.bin"
#nlp_sbatch nlp-a40-1 utils/similarity_analysis.py --main_model_path ${mnli_trained} \
#            --weak_model_path ${extrensicly_debiased} \
#             --output_dir grads_similarity_analysis/overall/mnli_trained-extrensic_debiased --task_name mnli --max_seq_length 128 \
#             --do_eval --per_device_batch_size 128  --regularized_tokens all --aggregation_strategy mean --grads_sim --select_only_biased \
#             --bias_indices ${lexical_bias_indices}
#
#
#nlp_sbatch nlp-a40-1 utils/similarity_analysis.py --main_model_path bert-base-uncased \
#            --weak_model_path ${mnli_trained} \
#             --output_dir grads_similarity_analysis/overall/bert_base-mnli_trained --task_name mnli --max_seq_length 128 \
#             --do_eval --per_device_batch_size 128  --regularized_tokens all --aggregation_strategy mean --grads_sim
#
#nlp_sbatch nlp-a40-1 utils/similarity_analysis.py --main_model_path bert-base-uncased \
#            --weak_model_path ${mnli_trained}/checkpoint-12000 \
#             --output_dir grads_similarity_analysis/overall/bert_base-mnli_trained_12000 --task_name mnli --max_seq_length 128 \
#             --do_eval --per_device_batch_size 128  --regularized_tokens all --aggregation_strategy mean --grads_sim
#
#python utils/similarity_analysis.py --main_model_path ${mnli_trained} \
#            --weak_model_path ${sim_debiased} \
#             --output_dir similarity_analysis/mnli_trained-sim_debiased --task_name mnli --max_seq_length 128 --do_eval \
#             --per_device_batch_size 2048  --regularized_tokens all --aggregation_strategy mean
#
#for ckp in ${extrensicly_debiased}/checkpoint-*; do
#    python utils/similarity_analysis.py --main_model_path ${mnli_trained}/checkpoint-12000 --weak_model_path ${ckp} \
#             --output_dir similarity_analysis/unbiased_subset/mnli_trained_12000-extrensic_debiased_`basename ${ckp}` --task_name mnli \
#             --max_seq_length 128 --do_eval --per_device_batch_size 2048  --regularized_tokens all --aggregation_strategy mean --remove_biased_samples_from_train \
#             --bias_indices data/lexical_bias_splits/validation_matched/confidence_based_0.65/validation_matched_biased_correct_indices.bin
#done
#
#nlp_sbatch nlp-a40-1 utils/similarity_analysis.py --main_model_path /home/redaigbaria/research_project/examples/pytorch/text-classification/runs/bert_64_base_s420 \
#            --weak_model_path /home/redaigbaria/research_project/examples/pytorch/text-classification/runs/bert_64_base_s254 \
#             --output_dir similarity_analysis/overall/bert_mnli_s254-bert_mnli_s420 --task_name mnli --max_seq_length 128 --do_eval \
#             --per_device_batch_size 2048  --regularized_tokens all --aggregation_strategy mean

# ------------------------------------------------------------------------------------------------------------
# Deberta V1

#mnli_trained="runs/deberta/deberta_64_base_s420"
#mnli2_trained="runs/deberta/deberta_64_base_s638"
#deberta_base="microsoft/deberta-base"
#lexical_extrensicly_debiased="runs/deberta/extrensic_debiasing/lexical_bias/deberta_64_lexical_extrensic_debiasing"
#
#
#
#nlp_sbatch nlp-a40-1 utils/similarity_analysis.py --main_model_path ${mnli2_trained} \
#            --weak_model_path ${mnli_trained} \
#             --output_dir deberta_similarity_analysis/mnli2_trained_mnli_trained --task_name mnli --max_seq_length 128 --do_eval \
#             --per_device_batch_size 1024  --regularized_tokens all --aggregation_strategy mean
#
#nlp_sbatch nlp-a40-1 utils/similarity_analysis.py --main_model_path ${mnli2_trained} \
#            --weak_model_path ${mnli_trained} \
#             --output_dir deberta_similarity_analysis/grads/mnli2_trained_mnli_trained --task_name mnli --max_seq_length 128 --do_eval \
#             --per_device_batch_size 64  --regularized_tokens all --aggregation_strategy mean --grads_sim
#
#
#
#exit


# Deberta V3
#mnli_trained="runs/debertav3/debertav3_64_base_s420"
#deberta_base="microsoft/deberta-v3-base"
#extrensicly_debiased="runs/debertav3/extrensic_debiasing/lexical_bias/debertav3_64_lexical_extrensic_debiasing"
#nlp_sbatch nlp-a40-1 utils/similarity_analysis.py --main_model_path ${extrensicly_debiased} \
#            --weak_model_path ${mnli_trained} \
#             --output_dir debertav3_similarity_analysis/extrensic_mnli_trained --task_name mnli --max_seq_length 128 --do_eval \
#             --per_device_batch_size 1024  \
#            --regularized_tokens all --aggregation_strategy mean
#
#
#nlp_sbatch nlp-a40-1 utils/similarity_analysis.py --main_model_path ${extrensicly_debiased} \
#            --weak_model_path ${mnli_trained} \
#             --output_dir debertav3_similarity_analysis/lexical_bias/extrensic_mnli_trained --task_name mnli --max_seq_length 128 --do_eval \
#             --per_device_batch_size 1024 --regularized_tokens all --aggregation_strategy mean --select_only_biased  \
#             --bias_indices ${lexical_bias_indices}
#
#
#
#nlp_sbatch nlp-a40-1 utils/similarity_analysis.py --main_model_path ${extrensicly_debiased} \
#            --weak_model_path ${deberta_base} \
#             --output_dir debertav3_similarity_analysis/extrensic_deberta_base --task_name mnli --max_seq_length 128 --do_eval \
#             --per_device_batch_size 1024  \
#            --regularized_tokens all --aggregation_strategy mean
#
#nlp_sbatch nlp-a40-1 utils/similarity_analysis.py --main_model_path ${mnli_trained} \
#            --weak_model_path ${deberta_base} \
#             --output_dir debertav3_similarity_analysis/mnli_trained_deberta_base --task_name mnli --max_seq_length 128 --do_eval \
#             --per_device_batch_size 1024  \
#            --regularized_tokens all --aggregation_strategy mean
#
#nlp_sbatch nlp-a40-1 utils/similarity_analysis.py --main_model_path ${mnli_trained} --weak_model_path ${deberta_base} \
#             --output_dir debertav3_similarity_analysis/lexical_bias/mnli_trained_deberta_base --task_name mnli --max_seq_length 128 --do_eval \
#             --per_device_batch_size 1024  --regularized_tokens all --aggregation_strategy mean --select_only_biased \
#             --bias_indices data/lexical_bias_splits/train/confidence_based_0.65/train_biased_correct_indices.bin
#
#

#QQP section -----------------------------------------------------------------------------------------------------------------------------------------
#qqp_trained="qqp_runs/bert_64_base"
#extrensicly_debiased="qqp_runs/extrensic_debiasing/unknown_bias/bert_64_0.8_unknown_extrensic_debiasing"
#tinybert="qqp_runs/5_folds_unknown_bias/tinybert_0"
#utama_biased="qqp_runs/UBBert_lexical_bias_64"
#python utils/similarity_analysis.py --main_model_path bert-base-uncased \
#            --weak_model_path ${extrensicly_debiased} \
#             --output_dir qqp_similarity_analysis/overall/extrensic-bert_base_uncased --task_name qqp --max_seq_length 128 --do_eval  \
#             --per_device_batch_size 2048 --regularized_tokens all --aggregation_strategy mean
#
#python utils/similarity_analysis.py --main_model_path ${qqp_trained} \
#            --weak_model_path ${tinybert} \
#             --output_dir qqp_similarity_analysis/overall/tinybert-qqp_trained --task_name qqp --max_seq_length 128 --do_eval  \
#             --per_device_batch_size 2048 --regularized_tokens all --aggregation_strategy mean
#
#
#python utils/similarity_analysis.py --main_model_path ${qqp_trained} \
#            --weak_model_path ${extrensicly_debiased} \
#             --output_dir qqp_similarity_analysis/overall/extrensic-qqp_trained --task_name qqp --max_seq_length 128 --do_eval  \
#             --per_device_batch_size 256 --regularized_tokens all --aggregation_strategy mean
#
#
#python utils/similarity_analysis.py --main_model_path ${qqp_trained} \
#            --weak_model_path ${utama_biased} \
#             --output_dir qqp_similarity_analysis/overall/utama_biased-qqp_trained --task_name qqp --max_seq_length 128 --do_eval  \
#             --per_device_batch_size 128 --regularized_tokens all --aggregation_strategy mean
#
#
#python utils/similarity_analysis.py --main_model_path ${extrensic_debiased} \
#            --weak_model_path ${tinybert} \
#             --output_dir qqp_similarity_analysis/overall/tinybert-extrensic --task_name qqp --max_seq_length 128 --do_eval  \
#             --per_device_batch_size 256 --regularized_tokens all --aggregation_strategy mean
#
#python utils/similarity_analysis.py --main_model_path ${utama_biased} \
#            --weak_model_path ${tinybert} \
#             --output_dir qqp_similarity_analysis/overall/tinybert-utama_biased --task_name qqp --max_seq_length 128 --do_eval  \
#             --per_device_batch_size 256 --regularized_tokens all --aggregation_strategy mean
#
#
#
#
#
#
#for ckp in ${qqp_trained}/checkpoint-*; do
#    python utils/similarity_analysis.py --main_model_path ${ckp} \
#            --weak_model_path bert-base-uncased \
#             --output_dir qqp_similarity_analysis/overall/base_uncased-qqp_trained_`basename ${ckp}` --task_name qqp --max_seq_length 128 --do_eval  \
#             --per_device_batch_size 2048 --regularized_tokens all --aggregation_strategy mean
#done
#
#for ckp in ${extrensicly_debiased}/checkpoint-*; do
#    python utils/similarity_analysis.py --main_model_path ${ckp} \
#            --weak_model_path ${qqp_trained}/checkpoint-11372 \
#             --output_dir qqp_similarity_analysis/overall/qqp_trained_11372-extrensicly_debiased_`basename ${ckp}` --task_name qqp \
#             --max_seq_length 128 --do_eval  --per_device_batch_size 2048 --regularized_tokens all --aggregation_strategy mean
#done
##Synthetic Bias section ------------------------------------------------------------------------------------------------------------------------------
#
#
#nlp_sbatch nlp-a40-1 utils/similarity_analysis.py --main_model_path runs/bert_64_base \
#            --weak_model_path runs/synthetic_bias_1_0.95/bert_64_base \
#             --output_dir synthetic_similarity_analysis/grads_sim/overall/bert_unbiased-bert_1_0.95_64_base --task_name mnli --max_seq_length 128 \
#             --do_eval --per_device_batch_size 64  --regularized_tokens all --aggregation_strategy mean --grads_sim
#
#
#
#
#nlp_sbatch nlp-a40-1 utils/similarity_analysis.py --main_model_path runs/synthetic_bias_1_0.95/bert_64_biased \
#            --weak_model_path runs/bert_64_base \
#             --output_dir synthetic_similarity_analysis/overall/unbiased_bert-bert_1_0.95_biased --task_name mnli --max_seq_length 128 \
#             --do_eval --per_device_batch_size 2048  --regularized_tokens all --aggregation_strategy mean
#
## FEVER section --------------------------------------------------------------------------------------------------------------------------------------
#fever_trained="fever_runs/bert_64_base"
#extrensicly_debiased="fever_runs/extrensic_debiasing/claim_bias/bert_64_0.8_claim_extrensic_debiasing"
#
#nlp_sbatch nlp-a40-1 utils/similarity_analysis.py --main_model_path ${fever_trained} --weak_model_path ${extrensicly_debiased} \
#             --output_dir fever_similarity_analysis/grads/overall/fever_trained-extrensic_debiased --task_name fever --max_seq_length 128 \
#             --do_eval --per_device_batch_size 128  --regularized_tokens all --aggregation_strategy mean --grads_sim
#
#
#nlp_sbatch galileo2 utils/similarity_analysis.py --main_model_path ${fever_trained}/checkpoint-3796 --weak_model_path ${extrensicly_debiased}/checkpoint-3360 \
#             --output_dir fever_similarity_analysis/grads/overall/fever_trained_ckp_3796-extrensic_debiased_ckp_3360 --task_name fever \
#             --max_seq_length 128 --do_eval --per_device_batch_size 128  --regularized_tokens all --aggregation_strategy mean --grads_sim
