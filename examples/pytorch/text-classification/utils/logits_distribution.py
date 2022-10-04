#!/usr/bin/env python
# coding=utf-8

import argparse
import logging
import os

import numpy as np
import torch
from datasets import load_dataset
from scipy.special import softmax
from utils.misc import DATA_DIR
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from tueplots import axes, bundles

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--do_eval', default=False, action='store_true')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--task_name', required=True, choices=['mnli', 'fever', 'qqp'])
    parser.add_argument('--logits_path', required=True)
    args = parser.parse_args()
    print(args)

    out_name = 'eval_logits_distribution.pdf' if args.do_eval else 'train_logits_distribution.pdf'
    ds_key = 'validation' if args.do_eval else 'train'
    if args.task_name == 'mnli' and args.do_eval:
        ds_key = 'validation_matched'

    if args.task_name == 'fever':
        ds = load_dataset(os.path.join(DATA_DIR, 'fever/fever_nli.py'))
    else:
        ds = load_dataset('glue', args.task_name)

    predict_dataset = ds[ds_key]

    labels_name = predict_dataset.features['label'].names
    labels = np.array(predict_dataset['label'])

    predictions = np.array(torch.load(args.logits_path).squeeze())

    plt.clf()
    with plt.rc_context(bundles.iclr2023()):
        plt.rcParams.update({
            'figure.dpi': 200,
            'axes.titlesize': 18,
            'legend.fontsize': 18,
            'axes.labelsize': 18,
            'xtick.labelsize': 18,
            'ytick.labelsize': 18,
        })

        fig, axs = plt.subplots(predictions.shape[1] + 1, 2, figsize=(20, 20))
        probs = softmax(predictions, axis=-1)
        kwargs = dict(alpha=0.3, bins=100, stacked=True)

        for i in range(predictions.shape[1]):
            axs[i, 0].hist(predictions[:, i], label="all", **kwargs)
            axs[i, 0].hist(predictions[np.nonzero(labels == i), i].squeeze(), label="correct", **kwargs)
            axs[i, 0].set_ylabel('Frequency')
            axs[i, 1].hist(probs[:, i], label="all", **kwargs)
            axs[i, 1].hist(probs[np.nonzero(labels == i), i].squeeze(), label="correct", **kwargs)
            axs[i, 1].set_ylabel('Frequency')
            axs[i, 0].yaxis.set_major_formatter(PercentFormatter(xmax=predictions.shape[0]))
            axs[i, 1].yaxis.set_major_formatter(PercentFormatter(xmax=predictions.shape[0]))
            axs[i, 0].set_title(f"Logits Distribution {labels_name[i]}")
            axs[i, 1].set_title(f"Probs Distribution {labels_name[i]}")
            axs[i, 1].set_xlim([0, 1])
            axs[i, 0].legend()
            axs[i, 1].legend()

        last_row_axs = axs[predictions.shape[1], 1]

        last_row_axs.hist(probs.max(axis=1, initial=0).squeeze(), label='all', **kwargs)
        last_row_axs.hist(probs[np.nonzero(probs.argmax(axis=1) == labels)].max(axis=1).squeeze(), label='correct', **kwargs)

        last_row_axs.set_ylabel('Frequency')
        last_row_axs.set_xlabel("$c_{f_b}()$")
        last_row_axs.set_title(f'{args.task_name.upper()} {ds_key}')
        last_row_axs.yaxis.set_major_formatter(PercentFormatter(xmax=predictions.shape[0]))
        # last_row_axs.set_xlim([0, 1])
        last_row_axs.set_xlim(right=1)
        # last_row_axs.axvline(x=0.8, color='r', label='$c_t$')
        last_row_axs.legend()
        last_row_axs.grid()
        plt.tight_layout(pad=0.0)
        # plt.show()
        plt.savefig(os.path.join(args.output_dir, out_name))


if __name__ == "__main__":
    main()
