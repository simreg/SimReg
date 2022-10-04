import torch
from torch.nn import CrossEntropyLoss
from transformers import TrainingArguments
from torch.nn.functional import log_softmax
from my_trainer import BaseTrainer


class ConfRegTrainer(BaseTrainer):
    def __init__(self, model, args: TrainingArguments = None, **kwargs):
        super().__init__(model=model, args=args, **kwargs)
        assert args.teacher_logits is not None
        self.teacher_targets = torch.load(args.teacher_logits, map_location=args.device)
        self.ce_loss = CrossEntropyLoss()
        assert self.teacher_targets.shape[0] == len(self.train_dataset)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop('labels')
        if 'idx' in inputs:
            indices = inputs.pop('idx')
        elif 'id' in inputs:
            indices = inputs.pop('id')

        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if self.is_in_train and self.logging_mode == 'train':
            # use Distillation loss
            loss = -(self.teacher_targets[indices] * log_softmax(outputs.logits, -1)).sum(1)
            loss = loss.mean()
        else:
            # Regular cross entropy loss
            loss = self.ce_loss(outputs.logits.view(-1, self.model.config.num_labels), labels.view(-1))

        self.log_loss(loss)
        return (loss, outputs) if return_outputs else loss
