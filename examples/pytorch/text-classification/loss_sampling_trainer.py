import logging
import math

import torch
import numpy as np

from typing import Dict, Union, Any, Optional, Tuple, List

from torch.nn import CrossEntropyLoss
from my_trainer import BaseTrainer


class BaseLossSamplingTrainer(BaseTrainer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.loss_fct = CrossEntropyLoss(reduction='none')
        self.effective_batch_size = {}
        logging.info('Using Loss based Trainer!')

    def get_optimization_mask(self, batch_loss, labels, logits):
        raise NotImplementedError

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]


        logits = outputs["logits"] if isinstance(outputs, dict) else outputs[0]
        batch_loss = self.loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        relevant_samples_mask = self.get_optimization_mask(batch_loss, labels, logits)
        ebs = relevant_samples_mask.sum().item()

        if self.logging_mode not in self.effective_batch_size:
            self.effective_batch_size[self.logging_mode] = []
        self.effective_batch_size[self.logging_mode].append(ebs)

        self.log_loss(batch_loss.detach().mean())
        loss = batch_loss[relevant_samples_mask].mean()

        return (loss, outputs) if return_outputs else loss


# class LossSamplingTrainer(BaseLossSamplingTrainer):
#     def get_optimization_mask(self, batch_loss, labels, logits):
#         preds = logits.detach().argmax(-1)
#         return preds == labels.view(-1)


class TopKLossTrainer(BaseLossSamplingTrainer):
    def __init__(self, k=0.25, **kwargs):
        super(TopKLossTrainer, self).__init__(**kwargs)
        self.k = k

    def get_optimization_mask(self, batch_loss, labels, logits):
        if self.state.epoch < 1:
            # train on the whole dataset at the first epoch.
            return torch.ones_like(batch_loss).bool()
        mask = torch.zeros_like(batch_loss)
        mask[batch_loss.topk(math.ceil(self.k * batch_loss.shape[0]), sorted=False).indices] = 1
        return mask.bool()

class MisclassificationTrainer(BaseLossSamplingTrainer):
    def get_optimization_mask(self, batch_loss, labels, logits):
        preds = logits.detach().argmax(-1)
        return preds != labels.view(-1)


