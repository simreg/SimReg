import torch
from transformers import TrainingArguments
from torch import nn
from torch.nn.functional import log_softmax
from my_trainer import BaseTrainer, get_model_signature
from transformers.utils import logging

logger = logging.get_logger(__name__)

class PoETrainer(BaseTrainer):
    def __init__(self, model, weak_models=None, args: TrainingArguments = None, **kwargs):
        super().__init__(model=model, args=args, **kwargs)
        self.weak_model = weak_models[0] if weak_models is not None else None
        self.main_signature = set(get_model_signature(model))
        if args.teacher_logits is not None:
            logger.info(f"Using logits {args.teacher_logits} for POE")
            self.bias_logits = torch.load(args.teacher_logits, map_location=args.device)
        else:
            self.bias_logits = None
        self.loss_fn = nn.CrossEntropyLoss()

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop('labels')
        if 'idx' in inputs:
            indices = inputs.pop('idx')
        elif 'id' in inputs:
            indices = inputs.pop('id')
        main_inputs = {k: inputs[k] for k in self.main_signature.intersection(inputs.keys())}
        outputs = model(**main_inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if self.is_in_train and self.logging_mode == 'train':

            if self.bias_logits is None:
                with torch.no_grad():
                    weak_logits = self.weak_model(**inputs).logits
            else:
                weak_logits = self.bias_logits[indices]

            loss = self.loss_fn(log_softmax(weak_logits, dim=-1) + log_softmax(outputs.logits, dim=-1), target=labels)
        else:
            loss = self.loss_fn(outputs.logits, target=labels)

        self.log_loss(loss)
        return (loss, outputs) if return_outputs else loss
