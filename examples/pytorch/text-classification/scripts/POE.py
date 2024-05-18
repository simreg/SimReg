import os
import random


for i in range(3):
    seed = random.randint(0, 1000)
    run_name = "bert_BertLexicalModel_lexicalDeBias_POE_2e5_32"
    out_dir = f"runs/lexical_bias_debiasing/POE/{run_name}_s{seed}"
    assert not os.path.exists(out_dir)
    teacher_logits = "/home/redaigbaria/SimReg/runs/bert_lexical_bias/logits/train_logits.bin"
    # teacher_logits = "runs/5_folds_unknown_bias/combined_logits/train_logits.bin"
    os.system(
        f" nlp_sbatch run_glue.py --report_to wandb --model_name_or_path bert-base-uncased --output_dir {out_dir} --num_train_epochs 3 --learning_rate 2e-5 --per_device_eval_batch_size 128 --do_eval --max_seq_length 128 --per_device_train_batch_size 32 --teacher_logits {teacher_logits} --regularization_method POE --evaluation_strategy half_epoch --save_strategy no --run_name {run_name} --weight_decay 0.1 --warmup_ratio 0.1 --no_pad_to_max_length --logging_first_step --tags lexical_bias --seed {seed} --task_name mnli --do_train")



# for i in range(3):
#     seed = random.randint(0, 1000)
#     run_name = "bert_HypothesisBias_POE_2e5_32"
#     out_dir = f"runs/hypothesis_bias_debiasing/POE/{run_name}_s{seed}"
#     assert not os.path.exists(out_dir)
#     teacher_logits = "runs/5_fold_hypothesis_only/combined_logits/train_logits.bin"
#     os.system(f"nlp_sbatch run_glue.py --report_to wandb --model_name_or_path bert-base-uncased --output_dir {out_dir}\
#          --num_train_epochs 3 --learning_rate 2e-5 --per_device_eval_batch_size 128 --do_eval --max_seq_length 128 \
#          --per_device_train_batch_size 32 --teacher_logits {teacher_logits} --regularization_method POE \
#          --evaluation_strategy half_epoch --save_strategy no --run_name {run_name} --weight_decay 0.1 \
#          --warmup_ratio 0.1 --no_pad_to_max_length --logging_first_step --tags hypothesis_only_bias --seed {seed} \
#          --task_name mnli --do_train --mismatched_indices_dir data/hypothesis_bias_splits/validation_mismatched")
