#!/bin/bash
export PYTHONPATH=~/research_project/examples/pytorch/text-classification/
cd ~/research_project/examples/pytorch/text-classification/

function save_model_logits() {
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

  echo "Saving ${ds} logits of ${model_dir}"

}


