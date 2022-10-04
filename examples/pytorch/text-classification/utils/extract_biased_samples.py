import json
import logging
import os
from dataclasses import dataclass, field
from typing import Optional
import argparse

import torch
from datasets import load_dataset

from transformers.utils.versions import require_version

logger = logging.getLogger(__name__)
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")
task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
    "fever": ("claim", "evidence")
}




def parse_args():
    parser = argparse.ArgumentParser(description="Compute indices of biased/anti-biased subset")
    parser.add_argument(
        "--task_name",
        type=str,
        default=None,
        help="The name of the glue task to train on.",
        choices=list(task_to_keys.keys()),
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--logits_path", required=True, type=str, help="Path of the saved logits")
    parser.add_argument("--do_train", action='store_true', help="Whether to run on training dataset.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run on evaluation dataset.")
    parser.add_argument("--bias_threshold", type=float, default=0.90)
    parser.add_argument("--unconfident_threshold", type=float, default=0.45)
    parser.add_argument(
        '--extract_correct_indices',
        action='store_true',
        help="Save indices of correctly classified labels (without regard to their confidence)"
    )
    parser.add_argument("--old_labels_order", action='store_true', help='Set to true if using old model logits (models from POE project/Utama\'s '
                                                                        'project')

    args = parser.parse_args()

    # Sanity checks
    if args.task_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a task name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."

    return args


def main():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO)

    args = parse_args()

    # if args.bias_threshold < 0.5:
    #     raise Exception('bias threshold should be greater than 0.5')

    if args.task_name is not None:
        # Downloading and loading a dataset from the hub.
        if args.task_name == 'fever':
            raw_datasets = load_dataset('../../../../data/fever/fever_nli.py')
        else:
            raw_datasets = load_dataset("glue", args.task_name)
    else:
        # Loading the dataset from local csv or json file.
        data_files = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        extension = (args.train_file if args.train_file is not None else args.valid_file).split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files)

    # raw_datasets = raw_datasets.map(
    #     None,
    #     batched=True,
    #     cache_file_names={
    #         k: os.path.join(DATA_DIR, f'mnli_bias_features/mnli_hans_{k}.cache') for k in raw_datasets.keys()
    #     }
    # )

    if args.do_train:
        ds_name = 'train'
    elif args.do_eval:
        ds_name = 'validation_matched' if args.task_name == 'mnli' else 'validation'
    else:
        raise Exception('Please select dataset (eval/train)')
    dataset = raw_datasets[ds_name]
    if args.old_labels_order:
        # datasets module MNLI labels order: entailment, neutral, contradiction.
        # POE project mnli labels order: contradiction, entailment, neutral
        mapper = {0: 1, 1: 2, 2: 0}
        labels = torch.tensor(list(map(lambda x: mapper[x], dataset['label']))).flatten()
    else:
        labels = torch.tensor(dataset['label']).flatten()

    os.makedirs(args.output_dir, exist_ok=True)
    model1_logits = torch.tensor(torch.load(args.logits_path))
    model1_probs = torch.softmax(model1_logits, dim=-1)
    model1_preds = torch.argmax(model1_probs, dim=-1)
    if args.extract_correct_indices:
        correct_indices = torch.nonzero(labels == model1_preds).squeeze()
        hard_indices = torch.nonzero(labels != model1_preds).squeeze()
        print(f"Correct indices shape: {correct_indices.shape}")
        print(f"Incorrect (hard) indices shape: {hard_indices.shape}")
        torch.save(hard_indices, os.path.join(args.output_dir, f"{ds_name}_hard_indices.bin"))
        torch.save(correct_indices, os.path.join(args.output_dir, f"{ds_name}_correct_indices.bin"))
    else:
        biased_indices = torch.nonzero(model1_probs.max(dim=-1)[0] >= args.bias_threshold).squeeze()
        unconfident_indices = torch.nonzero(model1_probs.max(dim=-1)[0] <= args.unconfident_threshold).squeeze()
        labels = labels[biased_indices]

        biased_correct_indices = biased_indices[model1_preds[biased_indices] == labels]
        anti_biased_indices = biased_indices[model1_preds[biased_indices] != labels]

        print(f"anti_biased_indices.shape: {anti_biased_indices.shape}")
        print(f"biased_correct_indices.shape: {biased_correct_indices.shape}")
        print(f"unconfident_indices.shape: {unconfident_indices.shape}")


        torch.save(biased_correct_indices, os.path.join(args.output_dir, f"{ds_name}_biased_correct_indices.bin"))
        torch.save(anti_biased_indices, os.path.join(args.output_dir, f"{ds_name}_anti_biased_indices.bin"))
        # torch.save(unconfident_indices, os.path.join(args.output_dir, f"{ds_name}_unconfident_indices.bin"))

    # hans_samples = torch.nonzero(model1_preds == labels)
    # anti_hans_samples = torch.nonzero(model1_preds != labels)
    # hans_confident_samples = torch.nonzero(model1_probs.max(dim=-1)[0] > 0.7)
    # hans_correct_confident_samples = torch.nonzero((model1_probs.max(dim=-1)[0] > 0.7) & (model1_preds == labels))
    # hans_confident_mistakes = torch.nonzero((model1_probs.max(dim=-1)[0] > 0.7) & (model1_preds != labels))
    json.dump(args.__dict__, open(os.path.join(args.output_dir, f'{ds_name}_data_extraction_config.json'), 'w'))


if __name__ == '__main__':
    main()
