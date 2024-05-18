#!/bin/bash

if [ $# -lt 1 ]; then
    echo "Missing model name"
    exit
fi

HANS_CODE_PATH="${HOME}/research_project/examples/research_projects/adversarial"
HANS_DATA_DIR="/home/redaigbaria/research_project/data/HANS"

cd ${HANS_CODE_PATH}

model_name=$1
model_path="/home/redaigbaria/research_project/examples/pytorch/text-classification/${model_name}"
#model_path=${1}
HANS_out="${model_path}/hans"
python run_hans.py --task_name hans --do_eval --data_dir ${HANS_DATA_DIR} --model_name_or_path ${model_path}\
       --max_seq_length 128  --output_dir ${HANS_out} --per_device_eval_batch_size 256

python evaluate_heur_output.py ${HANS_out}/hans_predictions.txt > ${HANS_out}/evaluate_heuristic_output.txt

