#!/bin/bash

cd ~/research_project/examples/pytorch/text-classification/

function save_model_activations() {
  local model_dir=${1}
  if [ $# -lt 2 ]; then
      echo "specify train/eval mode"
      exit 1
  fi
  local ds=${2}
  if [[ $ds != "eval" && $ds != "train" ]]; then
      echo "specify train/eval mode"
      exit 1
  fi
  echo "Saving ${ds} activations of ${model_dir}"
  ./scripts/py-sbatch.sh save_activations.py --model_name_or_path ${model_dir} --do_predict --do_${ds} \
  --task_name mnli --max_seq_length 128 --per_device_eval_batch_size 128 --output_dir ${model_dir}/activations
}


#save_model_activations runs/bert_reproduce

#save_model_activations runs/mini_bert


#save_model_activations runs/new_bert_linear_cka_10_3_lambda_1 train
#save_model_activations runs/new_bert_linear_cka_10_3_lambda_1 eval

#save_model_activations runs/new_bert_linear_cka_11_3 train
#save_model_activations runs/new_bert_linear_cka_11_3 eval

#save_model_activations runs/large_batch/large_batch_bert_linear_cka_10_3_lambda_1 train
#
#save_model_activations runs/large_batch/large_batch_bert_linear_cka_10_3_lambda_1 eval

#save_model_activations runs/new_bert_linear_cka_10_3_lambda_7 train
#save_model_activations runs/new_bert_linear_cka_10_3_lambda_7 eval


#save_model_activations runs/bert_linear_cka_11_3_lambda__batch_128 eval


#save_model_activations runs/bert_linear_cka_11_3_lambda__batch_128 eval

#save_model_activations runs/bert_linear_cka_11_3_lambda_7_batch_128 train
#save_model_activations runs/bert_linear_cka_11_3_lambda_1_batch_128 train
#save_model_activations runs/bert_linear_cka_11_3_lambda_12_batch_128 train
#save_model_activations runs/bert_linear_cka_11_3_lambda_1_batch_64 train

save_model_activations runs/bert_linear_cka_11_3_lambda_0.1_batch_128 eval
#
#save_model_activations bert-base-uncased eval
#save_model_activations google/bert_uncased_L-4_H-256_A-4 eval


#save_model_activations runs/tiny_bert_mnli


