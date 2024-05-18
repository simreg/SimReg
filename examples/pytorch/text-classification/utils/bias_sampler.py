import logging
from typing import List, Iterator
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Sampler
from transformers.trainer_utils import BiasSamplingStrategy


class BiasBatchSampler(Sampler[List[int]]):
    def __init__(self, bias_indices: Tensor, ds_len: int, batch_size: int, drop_last: bool,
                 sampling_strategy: BiasSamplingStrategy) -> None:
        # Since collections.abc.Iterable does not check for `__getitem__`, which
        # is one way for an object to be an iterable, we don't do an `isinstance`
        # check here.
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        logging.info("initializing BiasBatchSampler")
        self.batch_size = batch_size
        self.bias_indices = bias_indices
        self.other_indices = torch.tensor(np.setdiff1d(np.arange(ds_len), np.array(bias_indices)))
        self.num_bias_batches = (len(self.bias_indices) + self.batch_size - 1) // self.batch_size
        self.num_other_batches = (len(self.other_indices) + self.batch_size - 1) // self.batch_size
        self.ds_len = ds_len
        self.bias_turn = False
        self.sampling_strategy = sampling_strategy

    def __iter__(self) -> Iterator[List[int]]:
        bias_i = 0
        rest_i = 0
        self.bias_turn = False
        current_bias_perm = torch.randperm(len(self.bias_indices)).tolist()
        remaining_bias_batchess = self.num_bias_batches
        remaining_other_batches = self.num_other_batches
        current_rest_perm = torch.randperm(len(self.other_indices)).tolist()
        if self.sampling_strategy == BiasSamplingStrategy.STOCHASTIC:
            self.bias_turn = bool(torch.bernoulli(torch.zeros(1), remaining_bias_batchess / (
                        remaining_bias_batchess + remaining_other_batches)).item())

        for i in range(len(self)):
            if self.bias_turn:
                start = bias_i
                end = min(bias_i + self.batch_size, len(self.bias_indices))
                bias_i += end - start
                remaining_bias_batchess -= 1
                yield self.bias_indices[current_bias_perm[start: end]].tolist()
                if self.sampling_strategy == BiasSamplingStrategy.INORDER:
                    self.bias_turn = False
                elif self.sampling_strategy == BiasSamplingStrategy.STOCHASTIC:
                    if remaining_bias_batchess + remaining_other_batches == 0:
                        break
                    self.bias_turn = bool(
                        torch.bernoulli(torch.zeros(1), remaining_bias_batchess / (
                                    remaining_bias_batchess + remaining_other_batches)).item())
                elif self.sampling_strategy == BiasSamplingStrategy.DOWN_SAMPLING:
                    if remaining_bias_batchess == 0:
                        bias_i = 0
                        current_bias_perm = torch.randperm(len(self.bias_indices)).tolist()
                        remaining_bias_batchess = self.num_bias_batches
                    self.bias_turn = False
            else:
                start = rest_i
                end = min(rest_i + self.batch_size, len(self.other_indices))
                rest_i += end - start
                remaining_other_batches -= 1
                yield self.other_indices[current_rest_perm[start: end]].tolist()
                if self.sampling_strategy == BiasSamplingStrategy.INORDER:
                    self.bias_turn = bias_i < len(self.bias_indices)
                elif self.sampling_strategy == BiasSamplingStrategy.STOCHASTIC:
                    if remaining_bias_batchess + remaining_other_batches == 0:
                        break
                    self.bias_turn = bool(
                        torch.bernoulli(torch.zeros(1), remaining_bias_batchess / (
                                    remaining_bias_batchess + remaining_other_batches)).item())
                elif self.sampling_strategy == BiasSamplingStrategy.DOWN_SAMPLING:
                    if remaining_other_batches == 0:
                        rest_i = 0
                        current_rest_perm = torch.randperm(len(self.other_indices)).tolist()
                        remaining_other_batches = self.num_other_batches
                    self.bias_turn = True

    def __len__(self) -> int:
        if self.sampling_strategy == BiasSamplingStrategy.DOWN_SAMPLING:
            return min(self.num_bias_batches, self.num_other_batches) * 2
        if self.sampling_strategy == BiasSamplingStrategy.UP_SAMPLING:
            return max(self.num_bias_batches, self.num_other_batches) * 2
        return self.num_bias_batches + self.num_other_batches
