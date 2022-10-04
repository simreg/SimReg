#!/usr/bin/env python
# coding=utf-8

import logging
import os
import torch

from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from run_glue import main as rg_main

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.12.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

logger = logging.getLogger(__name__)


def main():
    trainer, training_args, data_args, model_args = rg_main(True)

    if training_args.do_train:
        ds_name = 'train'
        predict_dataset = trainer.train_dataset
    elif training_args.do_eval:
        ds_name = 'validation_matched' if data_args.task_name == 'mnli' else 'validation'
        predict_dataset = trainer.eval_dataset
    else:
        raise Exception('Please select dataset (eval/train)')

    # Removing the `label` columns because it contains -1 and Trainer won't like that.
    predict_dataset = predict_dataset.remove_columns("label")
    predictions = torch.from_numpy(trainer.predict(predict_dataset, metric_key_prefix="predict").predictions)
    torch.save(predictions, os.path.join(training_args.output_dir, f"{ds_name}_logits.bin"))
    logging.info(f'saved logits of {model_args.model_name_or_path} to {training_args.output_dir} successfully')


if __name__ == "__main__":
    main()

