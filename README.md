
# Installation
This work is built on HuggingFace's transformers library. To run the code you need to install this library to your local
environment using the following command (preferably in a new environment, since it will override current version of transformers).

```python setup.py install```


# Running experiments

The method runs in 3 stages
1. Train the bias model.
2. Extract D^B and train guidance model.
3. Train the main model.

Example of debiasing **Hypothesis-only bias in MNLI** dataset.

## Stage 1
Train the bias model:
```shell
cd examples/pytorch/text-classification
run_name=bert_hypothesis_only
out_dir=mnli_runs/${run_name}
python run_glue.py --model_name_or_path bert-base-uncased --output_dir ${out_dir}  --task_name mnli \
            --do_train --do_eval --per_device_train_batch_size 64 --weight_decay 0.1 --warmup_ratio 0.1 \
            --learning_rate 2e-5 --num_train_epochs 10 --run_name ${run_name} \
            --evaluation_strategy half_epoch --logging_first_step --no_pad_to_max_length --hypothesis_only --save_strategy half_epoch
```
## Stage 2

Save the logits of the model for biased samples extraction:

```shell

cd examples/pytorch/text-classification/utils
export PYTHONPATH=examples/pytorch/text-classification
## Pick the checkpoint with lowest cross-entropy loss on the validation set. 
model_dir="./mnli_runs/bert_hypothesis_only/checkpoint-"
python save_logits.py --model_name_or_path ${model_dir} --do_train \
  --task_name mnli --no_pad_to_max_length --per_device_eval_batch_size 256 \
  --output_dir ${model_dir}/logits
```

Plot the confidence of the bias model:

```shell
cd examples/pytorch/text-classification/utils
export PYTHONPATH=examples/pytorch/text-classification
model_dir="logits from previous section"
python logits_distribution.py --task_name mnli --do_eval \
--logits_path  ${model_dir}/logits --output_dir ${model_dir}/logits
```

Extract the biased samples based on the picked confidence threshold
--- you can do the same for validation set (to evaluate on) --- :
```shell
cd examples/pytorch/text-classification/utils
export PYTHONPATH=examples/pytorch/text-classification
SPLITS_DIR=../splits
mkdir "${SPLITS_DIR}/mnli_hypothesis_bias_splits"
confidence_threshold="pick one" # usually 0.8 is good enough
python extract_biased_samples.py --output_dir ${SPLITS_DIR}/mnli_hypothesis_bias_splits/train --bias_threshold ${confidence_threshold} \
--logits_path ./mnli_runs/hypothesis_only/logits/train_logits.bin --task_name mnli --do_train
```


Finally, for training with increasing similarity, train the "unbiased" guidance model:

```shell
cd examples/pytorch/text-classification
python run_glue.py --model_name_or_path bert-base-uncased --task_name mnli --logging_first_step \
            --output_dir mnli_runs/extrensic_debiasing/hypothesis_bias/bert_64_0.8_hypothesis_extrensic_debiasing --no_pad_to_max_length \
            --do_train --do_eval --per_device_train_batch_size 64 --per_device_eval_batch_size 256 \
             --weight_decay 0.1 --warmup_ratio 0.1 --save_strategy half_epoch --evaluation_strategy half_epoch \
            --learning_rate 2e-5 --num_train_epochs 10 --run_name bert_64_0.8_hypothesis_extrensic_debiasing \
            --remove_biased_samples_from_train --bias_indices ${SPLITS_DIR}/mnli_hypothesis_bias_splits/train/train_biased_correct_indices.bin
```
**_NOTE:_**  for decreasing similarity use the model from stage 1.

## Stage 3

```shell
cd examples/pytorch/text-classification/

guidance_model_dir=""
regularized_layers="bert.embeddings bert.encoder.layer.0 bert.encoder.layer.6 bert.encoder.layer.10 bert.encoder.layer.11"
batch_size=64
lambda=100
model_name=bert_EDBXE_linear_cka_eM-0M-6M-10M-11M_e-0-6-10-11_lambda_${lambda}_${batch_size}_sim
mnli_SPLIT_PATH="${SPLITS_DIR}/mnli_hypothesis_bias_splits/train_biased_correct_indices.bin"

python run_glue.py --model_name_or_path bert-base-uncased \
    --output_dir mnli_runs/hypothesis_debiasing_example/${model_name} \
     --task_name mnli --do_train --do_eval --per_device_train_batch_size ${batch_size} --per_device_eval_batch_size ${batch_size} \
     --learning_rate 2e-5 --num_train_epochs ${epochs} --weak_model_layer ${regularized_layers} \
     --weak_model_path ${guidance_model_dir} --regularized_layers ${regularized_layers} \
     --regularization_method abs_cos_cor --evaluation_strategy half_epoch \
     --regularization_lambda ${lambda} --run_name ${model_name} --weight_decay 0.1 --warmup_ratio 0.1 --no_pad_to_max_length \
     --logging_first_step --regularized_tokens all --token_aggregation_strategy mean --enforce_similarity \
      --regularize_only_biased --bias_indices ${mnli_SPLIT_PATH} --wandb_group
```


