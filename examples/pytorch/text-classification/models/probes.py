from torch import nn
from transformers.modeling_outputs import SequenceClassifierOutput


class LinearProbe(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_layers, out_dim):
        super(LinearProbe, self).__init__()
        layers = []
        assert n_layers > 0
        # first layer receives concatenation of (premise, hypothesis)
        next_in = in_dim * 2
        for i in range(n_layers - 1):
            layers.append(nn.Linear(next_in, hidden_dim))
            layers.append(nn.ReLU())
            next_in = hidden_dim
        layers.append(nn.Linear(next_in, out_dim))
        self.classifier = nn.Sequential(*layers)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_embeddings=None, labels=None):
        logits = self.classifier(input_embeddings)
        loss = self.loss_fn(logits, labels)
        return SequenceClassifierOutput(loss=loss, logits=logits)

