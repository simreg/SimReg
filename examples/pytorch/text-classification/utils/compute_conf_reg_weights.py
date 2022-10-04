import argparse
from os import path

import datasets
import torch
from datasets import load_dataset
from torch.nn.functional import softmax

from misc import DATA_DIR


def main():
    parser = argparse.ArgumentParser(description="This script computes the scaled teacher probs for confidence-regularization method")
    parser.add_argument('--teacher_logits', type=str, default=None, required=True, help='Path to teacher logits on MNLI train')
    parser.add_argument('--biased_model_logits', type=str, default=None, required=True, help='Path to biased model logits on MNLI train')
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--task_name', type=str, required=True)
    args = parser.parse_args()

    if args.task_name == 'fever':
        train_ds = load_dataset(path.join(DATA_DIR, 'fever/fever_nli.py'))['train']
    else:
        train_ds = datasets.load_dataset('glue', args.task_name)['train']

    ds_labels = torch.tensor(train_ds['label'])
    weak_probs = softmax(torch.tensor(torch.load(args.biased_model_logits)), dim=-1)
    teacher_probs = softmax(torch.tensor(torch.load(args.teacher_logits)), dim=-1)
    weights = 1 - torch.gather(weak_probs, 1, ds_labels.unsqueeze(1))
    weights = weights.expand_as(teacher_probs)

    exp_teacher_probs = teacher_probs ** weights
    norm_teacher_probs = exp_teacher_probs / exp_teacher_probs.sum(1).unsqueeze(1).expand_as(teacher_probs)

    torch.save(norm_teacher_probs, args.output_path)


if __name__ == '__main__':
    main()
