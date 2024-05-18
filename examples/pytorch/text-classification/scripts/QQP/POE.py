import os
import random

unknown_bias = False
for i in range(3):
    seed = random.randint(0, 1000)
    if unknown_bias:
        run_name = "bert_TB_UNB_POE_2e5_32"
        out_dir = f"qqp_runs/unknown_bias/POE/{run_name}_s{seed}"
        teacher_logits = "qqp_runs/5_folds_unknown_bias/combined_logits/train_logits.bin"
        tag = "unknown_bias"
    else:
        run_name = "bert_new_ClarkLexical_POE_2e5_32"
        out_dir = f"qqp_runs/lexical_bias/POE/{run_name}_s{seed}"
        assert not os.path.exists(out_dir)
        # teacher_logits = "/home/redaigbaria/debias/debias/preprocessing/train_logits.bin"
        teacher_logits = "/home/redaigbaria/SimReg/qqp_runs/clark_lexical_bias/train_logits.bin"
        tag = "lexical_bias"
    assert not os.path.exists(out_dir)
    os.system(
        f"nlp_sbatch run_glue.py --report_to wandb --model_name_or_path bert-base-uncased --output_dir {out_dir} --num_train_epochs 3 --learning_rate 2e-5 --per_device_eval_batch_size 256 --task_name qqp --do_train --do_eval --max_seq_length 128 --per_device_train_batch_size 32 --teacher_logits {teacher_logits} --regularization_method POE --evaluation_strategy half_epoch --save_strategy no --run_name {run_name} --weight_decay 0.1 --warmup_ratio 0.1 --no_pad_to_max_length --logging_first_step --tags {tag} --seed {seed}")
