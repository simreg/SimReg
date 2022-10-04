import torch
from transformers import TrainingArguments
from torch import nn
from torch.nn.functional import log_softmax
from my_trainer import BaseTrainer, get_model_signature


class PoETrainer(BaseTrainer):
    def __init__(self, model, weak_model, args: TrainingArguments = None, **kwargs):
        super().__init__(model=model, args=args, **kwargs)
        self.weak_model = weak_model
        self.main_signature = set(get_model_signature(model))
        self.loss_fn = nn.CrossEntropyLoss()

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop('labels')
        main_inputs = {k: inputs[k] for k in self.main_signature.intersection(inputs.keys())}
        outputs = model(**main_inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        with torch.no_grad():
            weak_logits = self.weak_model(**inputs).logits

        loss = self.loss_fn(log_softmax(weak_logits) + log_softmax(outputs.logits), target=labels)
        self.log_loss(loss)
        return (loss, outputs) if return_outputs else loss
